"""
The LagrangianSystem class: unified interface to Lagrangian mechanics.

Novice path: construct with Lagrangian, call integrate(), get correct results.
Expert path: access all internals, swap integrators, custom loss functions.
"""

from .symbolic import (
    derive_equations_of_motion,
    extract_kinetic_potential,
    find_cyclic_coordinates,
    derive_hamiltonian,
    derive_conserved_quantity,
)
from .codegen import (
    compile_lagrangian_dynamics,
    compile_grad_V,
    compile_energy,
    compile_expression,
)
from .integrators import (
    make_rk4_integrator,
    make_verlet_integrator,
    make_yoshida_integrator,
)

import jax.numpy as jnp
from jax import vmap


class LagrangianSystem:
    """
    A Lagrangian mechanical system: from symbolic physics to compiled numerics.

    This class is the core abstraction of the framework. Given a symbolic
    Lagrangian L(q, q̇), it automatically:

    1. Derives the Euler-Lagrange equations of motion
    2. Computes the Hamiltonian via Legendre transform
    3. Identifies conserved quantities (cyclic coordinates → momenta)
    4. Detects separability (determines which integrators can be used)
    5. Compiles everything to efficient JAX functions

    The physics is specified once, symbolically. The numerics are generated
    automatically and run fast.

    Attributes:
        coordinates: The generalized coordinate symbols [q1, q2, ...]
        velocities: The generalized velocity symbols [q1_dot, q2_dot, ...]
        parameters: The parameter symbols [m, k, ...]
        dof: Number of degrees of freedom
        lagrangian: The symbolic Lagrangian L
        hamiltonian: The symbolic Hamiltonian H = Σ pᵢq̇ᵢ - L
        momenta: List of (q̇, p) pairs where p = ∂L/∂q̇
        cyclic_coordinates: List of (q, p) pairs where q is cyclic
        is_separable: True if H = T(p) + V(q), enabling symplectic methods
        kinetic_energy: Symbolic T
        potential_energy: Symbolic V

    Example:
        >>> from sympy import symbols, Rational, cos
        >>> from contravariant import LagrangianSystem

        # Define a simple pendulum
        >>> theta, theta_dot = symbols('theta theta_dot')
        >>> m, l, g = symbols('m l g', positive=True)
        >>> L = Rational(1,2)*m*l**2*theta_dot**2 - m*g*l*(1 - cos(theta))

        # Create the system (symbolic analysis happens here)
        >>> pendulum = LagrangianSystem(L, [theta], [theta_dot])
        >>> print(pendulum.hamiltonian)  # Automatically derived

        # Integrate (compiled numerics, fast)
        >>> import jax.numpy as jnp
        >>> state_0 = jnp.array([0.5, 0.0])  # θ=0.5 rad, θ̇=0
        >>> params = {'m': 1.0, 'l': 1.0, 'g': 9.8}
        >>> traj = pendulum.integrate(state_0, 10000, 0.01, params)

    See Also:
        - contravariant.catalog: Pre-built systems (harmonic_oscillator, double_pendulum, ...)
        - Composition via + operator: sys_2d = sys_x + sys_y
    """

    def __init__(self, L, q_vars, q_dot_vars):
        """
        Construct a LagrangianSystem from a symbolic Lagrangian.

        Args:
            L: sympy expression for the Lagrangian
            q_vars: list of position symbols [q1, q2, ...]
            q_dot_vars: list of velocity symbols [q1_dot, q2_dot, ...]

        Raises:
            ValueError: If the Lagrangian has explicit time dependence (not yet supported)
        """
        from sympy import symbols

        # Store symbolic ingredients
        self.L = L
        self.q_vars = list(q_vars)
        self.q_dot_vars = list(q_dot_vars)
        self.n_dof = len(q_vars)

        # Check for explicit time dependence
        # Convention: symbol named 't' is time
        t = symbols("t")
        self._time_symbol = t if t in L.free_symbols else None
        self._is_time_dependent = self._time_symbol is not None

        if self._is_time_dependent:
            raise ValueError(
                "Time-dependent Lagrangians (L contains explicit 't') are not yet supported.\n"
                "The Lagrangian contains the time symbol 't', which requires modified "
                "integrators.\n"
                "This feature is planned for a future release.\n\n"
                f"Your Lagrangian: L = {L}"
            )

        # Derive everything symbolically (once, at construction)
        self._eom = derive_equations_of_motion(L, q_vars, q_dot_vars)
        self._energy_parts = extract_kinetic_potential(L, q_vars, q_dot_vars)
        self._H, self._momenta = derive_hamiltonian(L, q_vars, q_dot_vars)
        self._cyclic = find_cyclic_coordinates(L, q_vars, q_dot_vars)

        # Extract parameter symbols
        self.param_syms = self._eom["param_syms"]

        # Generate JAX functions (once, at construction)
        self._dynamics_fn = compile_lagrangian_dynamics(self._eom)
        self._energy_fn = compile_energy(self._eom, self._energy_parts)

        # Separable systems get Verlet capability
        self._is_separable = self._energy_parts["is_separable"]
        if self._is_separable:
            self._grad_V_fn = compile_grad_V(self._eom, self._energy_parts)
        else:
            self._grad_V_fn = None

        # Cache for integrators (created on demand)
        self._integrators = {}

    # -------------------------------------------------------------------------
    # Symbolic properties
    # -------------------------------------------------------------------------

    @property
    def coordinates(self):
        """The generalized coordinate symbols [q1, q2, ...]."""
        return self.q_vars

    @property
    def velocities(self):
        """The generalized velocity symbols [q1_dot, q2_dot, ...]."""
        return self.q_dot_vars

    @property
    def parameters(self):
        """The parameter symbols [m, k, ...]."""
        return self.param_syms

    @property
    def dof(self):
        """Degrees of freedom."""
        return self.n_dof

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
    def is_time_dependent(self):
        """
        True if L depends explicitly on time t.

        For autonomous systems (time-independent), energy is conserved.
        For non-autonomous systems (time-dependent), energy is NOT conserved:
        dH/dt = -∂L/∂t
        """
        return self._is_time_dependent

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

    def evaluate_dynamics(self, state, params):
        """
        Compute d(state)/dt.

        Args:
            state: array [q1, ..., qn, q1_dot, ..., qn_dot]
            params: dict of parameter values

        Returns:
            array [q1_dot, ..., qn_dot, q1_ddot, ..., qn_ddot]
        """
        return self._dynamics_fn(state, params)

    def evaluate_energy(self, state, params):
        """
        Compute total energy H = T + V.

        Args:
            state: array [q1, ..., qn, q1_dot, ..., qn_dot]
            params: dict of parameter values

        Returns:
            scalar energy
        """
        return self._energy_fn(state, params)

    def evaluate_energy_along_trajectory(self, traj, params):
        """Compute energy at each point in a trajectory."""
        return vmap(lambda s: self._energy_fn(s, params))(traj)

    # -------------------------------------------------------------------------
    # Integration
    # -------------------------------------------------------------------------

    def integrate(self, state_0, n_steps, dt, params, method="auto", mass_matrix=None):
        """
        Integrate the system forward in time.

        This is the main simulation method. It evolves the initial state forward
        using the compiled equations of motion. The integrator is chosen
        automatically based on the system's structure, or can be specified manually.

        Integrator Selection:
            For separable Hamiltonians H = T(p) + V(q), symplectic integrators
            (Yoshida, Verlet) preserve phase space structure and maintain bounded
            energy error over arbitrarily long times. For non-separable systems,
            RK4 is used.

            - 'yoshida': 4th-order symplectic. Best for long simulations of
              separable systems. Same accuracy as RK4 per step, but energy
              stays bounded even in chaotic regimes.
            - 'verlet': 2nd-order symplectic. Faster per step than Yoshida,
              but lower accuracy.
            - 'rk4': 4th-order Runge-Kutta. Works for any system, but energy
              drifts over time (especially in chaotic systems).
            - 'auto': Yoshida if separable, else RK4 (recommended).

        Args:
            state_0: Initial state as JAX array [q1, ..., qn, q1_dot, ..., qn_dot].
                     Positions first, then velocities.
            n_steps: Number of integration steps (static, for JIT compilation)
            dt: Timestep size. Smaller = more accurate but slower.
            params: Dict of parameter values, e.g., {'m': 1.0, 'k': 2.0}.
                    Must include all parameters appearing in the Lagrangian.
            method: Integration method. One of 'auto', 'yoshida', 'verlet', 'rk4'.
            mass_matrix: Array of masses for symplectic integrators.
                         If None, inferred from params (looks for 'm' or 'm1', 'm2', ...).

        Returns:
            Trajectory array of shape (n_steps, 2*n_dof) containing the state
            at each timestep.

        Raises:
            ValueError: If params is missing required parameters, or if a
                        symplectic method is requested for a non-separable system.

        Example:
            >>> import jax.numpy as jnp
            >>> state_0 = jnp.array([1.0, 0.0])  # q=1, q̇=0
            >>> params = {'m': 1.0, 'k': 4.0}
            >>> traj = sys.integrate(state_0, 10000, 0.01, params)
            >>> traj.shape
            (10000, 2)
        """
        # Validate parameters
        self._validate_params(params)

        # Resolve method
        if method == "auto":
            method = "yoshida" if self._is_separable else "rk4"

        # Validate method choice
        if method in ("verlet", "yoshida") and not self._is_separable:
            raise ValueError(
                f"{method.capitalize()} integration requires a separable Hamiltonian "
                f"H = T(p) + V(q), where T depends only on momenta and V only on positions.\n\n"
                f"This system is NOT separable because the kinetic energy depends on positions:\n"
                f"  T = {self.kinetic_energy}\n\n"
                f"Use method='rk4' instead, or let method='auto' choose automatically."
            )

        # Get or create integrator
        integrator = self._get_integrator(method)

        # Call appropriate integrator
        if method in ("verlet", "yoshida"):
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
            elif method == "yoshida":
                self._integrators[method] = make_yoshida_integrator(
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

    def _validate_params(self, params):
        """
        Validate that all required parameters are provided.

        Raises:
            ValueError: If params is missing required symbols
        """
        required = {str(p) for p in self.param_syms}
        provided = set(params.keys())
        missing = required - provided

        if missing:
            raise ValueError(
                f"Missing required parameter(s): {', '.join(sorted(missing))}\n"
                f"Required: {{{', '.join(sorted(required))}}}\n"
                f"Provided: {{{', '.join(sorted(provided))}}}"
            )

        # Also warn about extra params (might be typos)
        extra = provided - required
        if extra:
            import warnings

            warnings.warn(
                f"Extra parameter(s) not used by system: {', '.join(sorted(extra))}",
                UserWarning,
            )

    # -------------------------------------------------------------------------
    # Conservation analysis
    # -------------------------------------------------------------------------

    def check_symmetry(self, xi):
        """
        Check if a transformation is a symmetry of the Lagrangian.

        Noether's theorem states: every continuous symmetry of L corresponds to
        a conserved quantity. This method checks if a given infinitesimal
        transformation qᵢ → qᵢ + ε·ξᵢ leaves L invariant.

        The variation of L under the transformation is:
            δL = Σᵢ (∂L/∂qᵢ) ξᵢ

        If δL = 0, the transformation is a symmetry and conserved_quantity(xi)
        gives the corresponding conserved charge.

        Common symmetries:
            - Translation: ξᵢ = 1 for all i → conserves total momentum
            - Rotation (2D): ξ = [-y, x] → conserves angular momentum
            - Time translation (implicit): → conserves energy (Hamiltonian)

        Args:
            xi: List of expressions [ξ₁, ξ₂, ...] defining the infinitesimal
                generator. Must have same length as number of coordinates.

        Returns:
            Symbolic expression for δL. If this simplifies to 0, the
            transformation is a symmetry.

        Example:
            >>> # Check translation symmetry for FPUT chain with periodic BC
            >>> xi = [1] * N  # Uniform translation
            >>> delta_L = sys.check_symmetry(xi)
            >>> print(delta_L)  # Should be 0 for periodic BC
            0

        See Also:
            conserved_quantity: Compute the conserved charge for a symmetry
        """
        from sympy import diff, simplify

        delta_L = sum(diff(self.L, q) * x for q, x in zip(self.q_vars, xi))
        return simplify(delta_L)

    def conserved_quantity(self, xi):
        """
        Compute the Noether charge for a symmetry transformation.

        Given an infinitesimal generator ξ that defines a symmetry (δL = 0),
        the corresponding conserved quantity is:
            Q = Σᵢ pᵢ ξᵢ = Σᵢ (∂L/∂q̇ᵢ) ξᵢ

        This is Noether's theorem in computational form.

        WARNING: This method computes Q regardless of whether ξ actually
        defines a symmetry. Use check_symmetry(xi) first to verify that
        δL = 0, otherwise Q will not be conserved.

        Args:
            xi: List of expressions [ξ₁, ξ₂, ...] defining the infinitesimal
                generator. Same length as number of coordinates.

        Returns:
            Symbolic expression for the conserved quantity Q.

        Example:
            >>> # Compute angular momentum for 2D isotropic oscillator
            >>> xi = [-y, x]  # Rotation generator
            >>> if sys.check_symmetry(xi) == 0:
            ...     L_z = sys.conserved_quantity(xi)
            ...     print(f"Angular momentum: {L_z}")
            Angular momentum: m*(x*y_dot - y*x_dot)

        See Also:
            check_symmetry: Verify that ξ defines a symmetry before computing Q
        """
        return derive_conserved_quantity(self.L, self.q_vars, self.q_dot_vars, xi)

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
        energies = self.evaluate_energy_along_trajectory(traj, params)
        E0 = energies[0]
        max_err = float(jnp.max(jnp.abs(energies - E0)))
        rel_err = max_err / float(jnp.abs(E0)) if E0 != 0 else max_err
        results["energy"] = (max_err, rel_err)

        # Check additional quantities
        if quantities:
            for name, Q in quantities.items():
                Q_fn = self.compile(Q)
                values = vmap(lambda s: Q_fn(s, params))(traj)
                Q0 = values[0]
                max_err = float(jnp.max(jnp.abs(values - Q0)))
                rel_err = max_err / float(jnp.abs(Q0)) if Q0 != 0 else max_err
                results[name] = (max_err, rel_err)

        return results

    # -------------------------------------------------------------------------
    # Advanced user functionality
    # -------------------------------------------------------------------------

    def compile(self, expr):
        """
        Compile a symbolic expression to a JAX function.

        The expert escape hatch: turn any sympy expression involving
        coordinates, velocities, and parameters into a callable.

        Args:
            expr: sympy expression

        Returns:
            function (state, params) -> value

        Example:
            >>> L_z = x*p_y - y*p_x
            >>> lz_fn = sys.compile(L_z)
            >>> lz_fn(state, params)
        """
        return compile_expression(expr, self.q_vars, self.q_dot_vars, self.param_syms)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __repr__(self):
        sep_str = "separable" if self._is_separable else "non-separable"
        cyclic_str = (
            ", ".join(str(q) for q, p in self._cyclic) if self._cyclic else "none"
        )
        param_str = ", ".join(str(p) for p in self.param_syms)

        return (
            f"LagrangianSystem: {self.n_dof} DOF, {sep_str}\n"
            f"  L = {self.L}\n"
            f"  H = {self._H}\n"
            f"  Cyclic coordinates: {cyclic_str}\n"
            f"  Parameters: {param_str}"
        )

    # -------------------------------------------------------------------------
    # Composition and Lagrangian Arithmetic
    # -------------------------------------------------------------------------

    def __add__(self, other):
        """
        Add another system (composition) or a sympy expression.

        Args:
            other: LagrangianSystem or sympy expression

        Returns:
            New LagrangianSystem

        Example:
            >>> sys_2d = sys_x + sys_y  # composition
            >>> sys = sys + T_extra     # add kinetic term
        """
        if isinstance(other, LagrangianSystem):
            L_combined = self.L + other.L
            q_combined = list(self.q_vars) + list(other.q_vars)
            q_dot_combined = list(self.q_dot_vars) + list(other.q_dot_vars)
            return LagrangianSystem(L_combined, q_combined, q_dot_combined)
        else:
            return LagrangianSystem(
                self.L + other, list(self.q_vars), list(self.q_dot_vars)
            )

    def __sub__(self, other):
        """
        Subtract a sympy expression or another system from the Lagrangian.

        Args:
            other: sympy expression or LagrangianSystem

        Returns:
            New LagrangianSystem

        Example:
            >>> sys = sys_x + sys_y - Rational(1,2) * k_c * (x - y)**2
        """
        if isinstance(other, LagrangianSystem):
            L_combined = self.L - other.L
            q_combined = list(self.q_vars) + list(other.q_vars)
            q_dot_combined = list(self.q_dot_vars) + list(other.q_dot_vars)
            return LagrangianSystem(L_combined, q_combined, q_dot_combined)
        else:
            return LagrangianSystem(
                self.L - other, list(self.q_vars), list(self.q_dot_vars)
            )

    # -------------------------------------------------------------------------
    # Comparison and Visualization
    # -------------------------------------------------------------------------

    def compare_integrators(
        self,
        state_0,
        n_steps,
        dt,
        params,
        methods=None,
        quantities=None,
        save_as=None,
        show=True,
    ):
        """
        Compare integration methods on conservation and accuracy.
        ...
        """
        from .plotting import plot_energy_errors, plot_configuration_space

        # Default methods: fair comparison at same order
        if methods is None:
            methods = ["rk4", "yoshida"] if self.is_separable else ["rk4"]

        # Filter symplectic methods if not separable
        if not self.is_separable:
            symplectic = {"verlet", "yoshida"}
            if symplectic & set(methods):
                print(f"Note: Symplectic methods removed (system not separable)")
                methods = [m for m in methods if m not in symplectic]

        # Integrate with each method
        trajectories = {}
        for method in methods:
            trajectories[method] = self.integrate(
                state_0, n_steps, dt, params, method=method
            )

        # Check conservation
        print(f"Integration: {n_steps} steps, dt={dt}, T={n_steps * dt}")
        print()

        all_quantities = ["energy"] + (list(quantities.keys()) if quantities else [])
        header = f"{'Method':<10}" + "".join(f"{q:>15}" for q in all_quantities)
        print(header)
        print("-" * len(header))

        for method in methods:
            cons = self.check_conservation(trajectories[method], params, quantities)
            row = f"{method:<10}"
            for q in all_quantities:
                max_err, _ = cons[q]
                row += f"{max_err:>15.2e}"
            print(row)
        print()

        # Plot energy comparison
        plot_energy_errors(
            trajectories,
            self._energy_fn,
            params,
            title=f"Energy Error ({n_steps} steps)",
            save_as=f"{save_as}_energy" if save_as else None,
            show=show,
        )

        # Plot configuration/phase space
        best_method = "yoshida" if "yoshida" in trajectories else methods[0]
        is_phase_space = self.n_dof == 1
        plot_configuration_space(
            trajectories[best_method],
            coord_indices=(0, 1),
            xlabel=str(self.q_vars[0]),
            ylabel=str(self.q_dot_vars[0]) if is_phase_space else str(self.q_vars[1]),
            title="Phase Space" if is_phase_space else "Configuration Space",
            save_as=(
                f"{save_as}_{'phase' if is_phase_space else 'config'}"
                if save_as
                else None
            ),
            show=show,
        )

        return trajectories

    # -------------------------------------------------------------------------
    # Parameter Learning
    # -------------------------------------------------------------------------

    def learn_parameters(
        self,
        traj_observed,
        state_0,
        n_steps,
        dt,
        params_fixed,
        params_init,
        loss_type="energy_statistic",
        learning_rate=0.1,
        max_iterations=100,
        tolerance=1e-8,
        verbose=True,
    ):
        """
        Learn unknown parameters from an observed trajectory.

        Args:
            traj_observed: observed trajectory array
            state_0: initial state
            n_steps: integration steps
            dt: timestep
            params_fixed: dict of fixed parameters {'m': 1.0}
            params_init: dict of initial guesses {'k': 0.5}
            loss_type: 'trajectory', 'energy_statistic', or callable
            learning_rate: optimizer learning rate
            max_iterations: max optimization steps
            tolerance: stop when loss < tolerance
            verbose: print progress

        Returns:
            dict of learned parameters
        """
        from .learning import learn_parameters as _learn_parameters

        return _learn_parameters(
            integrate_fn=self.integrate,
            traj_observed=traj_observed,
            state_0=state_0,
            n_steps=n_steps,
            dt=dt,
            n_dof=self.n_dof,
            params_fixed=params_fixed,
            params_init=params_init,
            loss_type=loss_type,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=verbose,
        )
