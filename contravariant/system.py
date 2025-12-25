"""
The System class: unified interface to Lagrangian mechanics.

Novice path: construct with Lagrangian, call integrate(), get correct results.
Expert path: access all internals, swap integrators, custom loss functions.
"""

from .symbolic import (
    derive_equations_of_motion,
    extract_kinetic_potential,
    find_cyclic_coordinates,
    derive_hamiltonian,
    conserved_from_symmetry,
)
from .codegen import (
    make_lagrangian_dynamics_fn,
    make_grad_V_fn,
    make_energy_fn,
    make_conserved_quantity_fn,
)
from .integrators import (
    make_rk4_integrator,
    make_verlet_integrator,
)

import jax.numpy as jnp
from jax import vmap


class System:
    """
    A Lagrangian mechanical system.

    Construction: symbolic analysis (slow, once)
    Integration: compiled numerics (fast, many times)

    Example:
        >>> from sympy import symbols, Rational
        >>> q, q_dot = symbols('q q_dot')
        >>> m, k = symbols('m k', positive=True)
        >>> L = Rational(1,2)*m*q_dot**2 - Rational(1,2)*k*q**2
        >>> sys = System(L, [q], [q_dot])
        >>> traj = sys.integrate(state_0, n_steps, dt, params)
    """

    def __init__(self, L, q_vars, q_dot_vars):
        """
        Construct a System from a symbolic Lagrangian.

        Args:
            L: sympy expression for the Lagrangian
            q_vars: list of position symbols [q1, q2, ...]
            q_dot_vars: list of velocity symbols [q1_dot, q2_dot, ...]
        """
        # Store symbolic ingredients
        self.L = L
        self.q_vars = list(q_vars)
        self.q_dot_vars = list(q_dot_vars)
        self.n_dof = len(q_vars)

        # Derive everything symbolically (once, at construction)
        self._eom = derive_equations_of_motion(L, q_vars, q_dot_vars)
        self._energy_parts = extract_kinetic_potential(L, q_vars, q_dot_vars)
        self._H, self._momenta = derive_hamiltonian(L, q_vars, q_dot_vars)
        self._cyclic = find_cyclic_coordinates(L, q_vars, q_dot_vars)

        # Extract parameter symbols
        self.param_syms = self._eom["param_syms"]

        # Generate JAX functions (once, at construction)
        self._dynamics_fn = make_lagrangian_dynamics_fn(self._eom)
        self._energy_fn = make_energy_fn(self._eom, self._energy_parts)

        # Separable systems get Verlet capability
        self._is_separable = self._energy_parts["is_separable"]
        if self._is_separable:
            self._grad_V_fn = make_grad_V_fn(self._eom, self._energy_parts)
        else:
            self._grad_V_fn = None

        # Cache for integrators (created on demand)
        self._integrators = {}

    # -------------------------------------------------------------------------
    # Symbolic properties
    # -------------------------------------------------------------------------

    @property
    def lagrangian(self):
        """The symbolic Lagrangian."""
        return self.L

    @property
    def hamiltonian(self):
        """The symbolic Hamiltonian H = Σ pᵢq̇ᵢ - L."""
        return self._H

    @property
    def momenta(self):
        """List of (q_dot, p) pairs where p = ∂L/∂q̇."""
        return self._momenta

    @property
    def cyclic_coordinates(self):
        """List of (q, p) pairs where q is cyclic and p is conserved."""
        return self._cyclic

    @property
    def equations_of_motion(self):
        """Dict mapping q̈ symbols to their expressions."""
        return self._eom["solutions"]

    @property
    def is_separable(self):
        """True if H = T(p) + V(q), enabling symplectic integration."""
        return self._is_separable

    @property
    def kinetic_energy(self):
        """Symbolic kinetic energy T."""
        return self._energy_parts["T"]

    @property
    def potential_energy(self):
        """Symbolic potential energy V."""
        return self._energy_parts["V"]

    # -------------------------------------------------------------------------
    # Numerical functions
    # -------------------------------------------------------------------------

    def dynamics(self, state, params):
        """
        Compute d(state)/dt.

        Args:
            state: array [q1, ..., qn, q1_dot, ..., qn_dot]
            params: dict of parameter values

        Returns:
            array [q1_dot, ..., qn_dot, q1_ddot, ..., qn_ddot]
        """
        return self._dynamics_fn(state, params)

    def energy(self, state, params):
        """
        Compute total energy H = T + V.

        Args:
            state: array [q1, ..., qn, q1_dot, ..., qn_dot]
            params: dict of parameter values

        Returns:
            scalar energy
        """
        return self._energy_fn(state, params)

    def energy_along_trajectory(self, traj, params):
        """Compute energy at each point in a trajectory."""
        return vmap(lambda s: self._energy_fn(s, params))(traj)

    # -------------------------------------------------------------------------
    # Integration
    # -------------------------------------------------------------------------

    def integrate(self, state_0, n_steps, dt, params, method="auto", mass_matrix=None):
        """
        Integrate the system forward in time.

        Args:
            state_0: initial state [q1, ..., qn, q1_dot, ..., qn_dot]
            n_steps: number of integration steps
            dt: timestep
            params: dict of parameter values
            method: 'auto', 'verlet', or 'rk4'
                - 'auto': Verlet if separable, else RK4
                - 'verlet': Störmer-Verlet (requires separable system)
                - 'rk4': 4th-order Runge-Kutta
            mass_matrix: array of masses (required for Verlet, inferred if possible)

        Returns:
            trajectory array of shape (n_steps, 2*n_dof)
        """
        # Resolve method
        if method == "auto":
            method = "verlet" if self._is_separable else "rk4"

        # Validate method choice
        if method == "verlet" and not self._is_separable:
            raise ValueError(
                "Verlet integration requires separable Hamiltonian H = T(p) + V(q). "
                "This system is not separable. Use method='rk4' instead."
            )

        # Get or create integrator
        integrator = self._get_integrator(method)

        # Call appropriate integrator
        if method == "verlet":
            if mass_matrix is None:
                mass_matrix = self._infer_mass_matrix(params)
            return integrator(state_0, n_steps, dt, params, mass_matrix)
        else:
            return integrator(state_0, n_steps, dt, params)

    def _get_integrator(self, method):
        """Get or create cached integrator."""
        if method not in self._integrators:
            if method == "verlet":
                self._integrators[method] = make_verlet_integrator(
                    self._grad_V_fn, self.n_dof
                )
            elif method == "rk4":
                self._integrators[method] = make_rk4_integrator(self._dynamics_fn)
            else:
                raise ValueError(f"Unknown integration method: {method}")
        return self._integrators[method]

    def _infer_mass_matrix(self, params):
        """Try to infer mass matrix from params. Override if needed."""
        # Simple heuristic: look for 'm' or 'm1', 'm2', etc.
        if "m" in params:
            return jnp.array([params["m"]] * self.n_dof)

        masses = []
        for i in range(self.n_dof):
            key = f"m{i+1}" if self.n_dof > 1 else "m"
            if key in params:
                masses.append(params[key])
            else:
                raise ValueError(
                    f"Cannot infer mass matrix. Provide mass_matrix argument "
                    f"or ensure params contains 'm' or 'm1', 'm2', etc."
                )
        return jnp.array(masses)

    # -------------------------------------------------------------------------
    # Conservation analysis
    # -------------------------------------------------------------------------

    def conserved_quantity(self, xi):
        """
        Compute the conserved quantity for a symmetry.

        Args:
            xi: list of expressions [ξ1, ξ2, ...] defining the infinitesimal generator

        Returns:
            symbolic expression for the conserved quantity Q = Σ pᵢξᵢ
        """
        return conserved_from_symmetry(self.L, self.q_vars, self.q_dot_vars, xi)

    def make_conserved_fn(self, Q):
        """
        Generate a JAX function for a conserved quantity.

        Args:
            Q: symbolic expression for conserved quantity

        Returns:
            function (state, params) -> scalar
        """
        return make_conserved_quantity_fn(
            Q, self.q_vars, self.q_dot_vars, self.param_syms
        )

    def check_conservation(self, traj, params, quantities=None):
        """
        Check conservation of energy and any specified quantities.

        Args:
            traj: trajectory array
            params: dict of parameter values
            quantities: dict of {name: Q_expr} for additional conserved quantities

        Returns:
            dict of {name: (max_error, relative_error)}
        """
        results = {}

        # Always check energy
        energies = self.energy_along_trajectory(traj, params)
        E0 = energies[0]
        max_err = float(jnp.max(jnp.abs(energies - E0)))
        rel_err = max_err / float(jnp.abs(E0)) if E0 != 0 else max_err
        results["energy"] = (max_err, rel_err)

        # Check additional quantities
        if quantities:
            for name, Q in quantities.items():
                Q_fn = self.make_conserved_fn(Q)
                values = vmap(lambda s: Q_fn(s, params))(traj)
                Q0 = values[0]
                max_err = float(jnp.max(jnp.abs(values - Q0)))
                rel_err = max_err / float(jnp.abs(Q0)) if Q0 != 0 else max_err
                results[name] = (max_err, rel_err)

        return results

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __repr__(self):
        sep_str = "separable" if self._is_separable else "non-separable"
        cyclic_str = ", ".join(str(q) for q, p in self._cyclic) if self._cyclic else "none"
        param_str = ", ".join(str(p) for p in self.param_syms)

        return (
            f"System: {self.n_dof} DOF, {sep_str}\n"
            f"  L = {self.L}\n"
            f"  H = {self._H}\n"
            f"  Cyclic coordinates: {cyclic_str}\n"
            f"  Parameters: {param_str}"
        )
