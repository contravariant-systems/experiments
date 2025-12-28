"""
Numerical integrators for dynamical systems.

This module provides time-stepping methods for solving the equations of motion.
The key distinction is between general-purpose integrators (RK4) and symplectic
integrators (Verlet, Yoshida) that preserve the geometric structure of
Hamiltonian mechanics.

Why Symplectic Integrators Matter
---------------------------------
For Hamiltonian systems, the flow preserves a geometric quantity called the
symplectic 2-form. Standard integrators (Euler, RK4) do not preserve this
structure, which causes:

1. Systematic energy drift over long times
2. Artificial damping or growth of phase space volume
3. Qualitatively wrong long-term behavior (especially in chaotic systems)

Symplectic integrators preserve the symplectic structure exactly (up to
floating-point precision), so:

1. Energy error stays bounded and oscillates, never drifts
2. Phase space volume is exactly conserved
3. Long-term statistics are correct even if individual trajectories diverge

The catch: symplectic integrators require a separable Hamiltonian H = T(p) + V(q).
This is why LagrangianSystem checks for separability and auto-selects methods.

Integrator Hierarchy
--------------------
- O(h¹): Euler — pedagogical only, systematically wrong
- O(h²): Störmer-Verlet — symplectic, robust, widely used in MD
- O(h⁴): RK4 — accurate per-step, but energy drifts long-term
- O(h⁴): Yoshida — same accuracy as RK4, but symplectic

For most purposes, use 'auto' method selection: Yoshida if separable, RK4 if not.

State Convention
----------------
All integrators use state = [q₁, ..., qₙ, v₁, ..., vₙ] where q are generalized
coordinates and v are velocities. Symplectic integrators convert to momenta
(p = m·v) internally.

Usage
-----
    # For arbitrary dynamics (e.g., Lagrangian systems)
    integrator = make_rk4_integrator(dynamics_fn)
    traj = integrator(state_0, n_steps, dt, params)

    # For separable Hamiltonians H = T(p) + V(q)
    integrator = make_verlet_integrator(grad_V_fn, n_dof)
    traj = integrator(state_0, n_steps, dt, params, mass_matrix)

    # For separable Hamiltonians, 4th-order accurate
    integrator = make_yoshida_integrator(grad_V_fn, n_dof)
    traj = integrator(state_0, n_steps, dt, params, mass_matrix)

References
----------
- Hairer, Lubich, Wanner: "Geometric Numerical Integration"
- Yoshida (1990): "Construction of higher order symplectic integrators"
"""

from functools import partial
import jax.numpy as jnp
from jax import jit
from jax.lax import scan


# =============================================================================
# Constants
# =============================================================================

# Yoshida 4th-order coefficients
# Solve: w₀ + 2w₁ = 1 (consistency) and w₀³ + 2w₁³ = 0 (4th order)
_CBRT2 = 2.0 ** (1 / 3)
_YOSHIDA_W1 = 1.0 / (2.0 - _CBRT2)  # ≈  1.3512
_YOSHIDA_W0 = -_CBRT2 * _YOSHIDA_W1  # ≈ -1.7024


# =============================================================================
# Single-Step Methods
# =============================================================================


def euler_step(state, params, dynamics, dt):
    """
    Forward Euler step. O(h) accuracy.

    Systematically adds energy to Hamiltonian systems.
    For pedagogy only—do not use in production.
    """
    return state + dt * dynamics(state, params)


def rk4_step(state, params, dynamics, dt):
    """
    Classic 4th-order Runge-Kutta step. O(h⁴) accuracy.

    Excellent accuracy but not symplectic—energy drifts over long times.
    """
    k1 = dynamics(state, params)
    k2 = dynamics(state + dt * k1 / 2.0, params)
    k3 = dynamics(state + dt * k2 / 2.0, params)
    k4 = dynamics(state + dt * k3, params)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def verlet_step(q, p, params, grad_V, mass_matrix, dt):
    """
    Störmer-Verlet step for separable Hamiltonian H = T(p) + V(q).
    O(h²) accuracy, symplectic.

    The leapfrog structure:
        1. Half kick: p ← p - (dt/2)·∇V(q)
        2. Full drift: q ← q + dt·(p/m)
        3. Half kick: p ← p - (dt/2)·∇V(q)

    Args:
        q: positions (n_dof,)
        p: momenta (n_dof,) — NOTE: momenta, not velocities
        params: physical parameters dict
        grad_V: function (q, params) → ∇V(q)
        mass_matrix: diagonal masses (n_dof,)
        dt: timestep

    Returns:
        (q_new, p_new)
    """
    # Half kick
    p_half = p - 0.5 * dt * grad_V(q, params)

    # Full drift
    q_new = q + dt * p_half / mass_matrix

    # Half kick
    p_new = p_half - 0.5 * dt * grad_V(q_new, params)

    return q_new, p_new


def yoshida_step(q, p, params, grad_V, mass_matrix, dt):
    """
    4th-order Yoshida step. O(h⁴) accuracy, symplectic.

    Composes three Verlet steps with coefficients that cancel O(h³) error.

    Reference:
        Yoshida, H. (1990). "Construction of higher order symplectic integrators."
        Physics Letters A, 150(5-7), 262-268.

    Args:
        q: positions (n_dof,)
        p: momenta (n_dof,)
        params: physical parameters dict
        grad_V: function (q, params) → ∇V(q)
        mass_matrix: diagonal masses (n_dof,)
        dt: timestep

    Returns:
        (q_new, p_new)
    """
    q, p = verlet_step(q, p, params, grad_V, mass_matrix, _YOSHIDA_W1 * dt)
    q, p = verlet_step(q, p, params, grad_V, mass_matrix, _YOSHIDA_W0 * dt)
    q, p = verlet_step(q, p, params, grad_V, mass_matrix, _YOSHIDA_W1 * dt)
    return q, p


# =============================================================================
# Factory Functions: General Dynamics
# =============================================================================


def make_rk4_integrator(dynamics):
    """
    Create RK4 integrator bound to a dynamics function.

    O(h⁴) accuracy, not symplectic.

    Args:
        dynamics: function (state, params) → d_state/dt

    Returns:
        integrate(state_0, n_steps, dt, params) → trajectory
    """

    @partial(jit, static_argnums=(1,))
    def integrate(state_0, n_steps, dt, params):
        def step_fn(state, _):
            new_state = rk4_step(state, params, dynamics, dt)
            return new_state, new_state

        _, trajectory = scan(step_fn, state_0, None, length=n_steps)
        return trajectory

    return integrate


# =============================================================================
# Factory Functions: Symplectic (Separable Hamiltonians)
# =============================================================================


def make_verlet_integrator(grad_V, n_dof):
    """
    Create Störmer-Verlet integrator bound to a grad_V function.

    O(h²) accuracy, symplectic.

    For separable Hamiltonians H = T(p) + V(q) where T = Σᵢ pᵢ²/(2mᵢ).

    Args:
        grad_V: function (q, params) → ∇V(q)
        n_dof: degrees of freedom

    Returns:
        integrate(state_0, n_steps, dt, params, mass_matrix) → trajectory

    Note:
        Input state = [q, v] uses velocities.
        Internally converts to momenta p = m·v for symplectic integration.
    """

    @partial(jit, static_argnums=(1,))
    def integrate(state_0, n_steps, dt, params, mass_matrix):
        # Unpack state: [q, v] with velocities
        q = state_0[:n_dof]
        v = state_0[n_dof:]

        # Convert velocity → momentum for symplectic integration
        p = v * mass_matrix

        def step_fn(carry, _):
            q, p = carry
            q_new, p_new = verlet_step(q, p, params, grad_V, mass_matrix, dt)

            # Convert momentum → velocity for output
            v_new = p_new / mass_matrix
            state = jnp.concatenate([q_new, v_new])

            return (q_new, p_new), state

        _, trajectory = scan(step_fn, (q, p), None, length=n_steps)
        return trajectory

    return integrate


def make_yoshida_integrator(grad_V, n_dof):
    """
    Create 4th-order Yoshida integrator bound to a grad_V function.

    O(h⁴) accuracy, symplectic.

    Same accuracy as RK4, but preserves symplectic structure.
    Composes three Verlet steps with coefficients that cancel O(h³) error.

    Args:
        grad_V: function (q, params) → ∇V(q)
        n_dof: degrees of freedom

    Returns:
        integrate(state_0, n_steps, dt, params, mass_matrix) → trajectory

    Note:
        Input state = [q, v] uses velocities.
        Internally converts to momenta p = m·v for symplectic integration.
    """

    @partial(jit, static_argnums=(1,))
    def integrate(state_0, n_steps, dt, params, mass_matrix):
        # Unpack state: [q, v] with velocities
        q = state_0[:n_dof]
        v = state_0[n_dof:]

        # Convert velocity → momentum for symplectic integration
        p = v * mass_matrix

        def step_fn(carry, _):
            q, p = carry
            q_new, p_new = yoshida_step(q, p, params, grad_V, mass_matrix, dt)

            # Convert momentum → velocity for output
            v_new = p_new / mass_matrix
            state = jnp.concatenate([q_new, v_new])

            return (q_new, p_new), state

        _, trajectory = scan(step_fn, (q, p), None, length=n_steps)
        return trajectory

    return integrate


# =============================================================================
# Standalone Integration Functions (for direct use without factory)
# =============================================================================


@partial(jit, static_argnums=(1, 4))
def integrate_rk4(state_0, n_steps, dt, params, dynamics):
    """
    Integrate using RK4.

    Args:
        state_0: initial state
        n_steps: number of steps (static)
        dt: timestep
        params: parameter dict
        dynamics: function (state, params) → d_state/dt (static)

    Returns:
        trajectory of shape (n_steps, state_dim)
    """

    def step_fn(state, _):
        new_state = rk4_step(state, params, dynamics, dt)
        return new_state, new_state

    _, trajectory = scan(step_fn, state_0, None, length=n_steps)
    return trajectory


@partial(jit, static_argnums=(1, 4))
def integrate_euler(state_0, n_steps, dt, params, dynamics):
    """
    Integrate using forward Euler.

    For pedagogy only—use RK4 or symplectic methods in practice.
    """

    def step_fn(state, _):
        new_state = euler_step(state, params, dynamics, dt)
        return new_state, new_state

    _, trajectory = scan(step_fn, state_0, None, length=n_steps)
    return trajectory
