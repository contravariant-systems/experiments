"""
Numerical integrators for dynamical systems.

Includes:
- Forward Euler (pedagogical, not recommended)
- RK4 (accurate, not symplectic)
- Störmer-Verlet (symplectic, for separable Hamiltonians)
"""

from functools import partial
import jax.numpy as jnp
from jax import jit
from jax.lax import scan


# ---------------------------------------------------------------------
# Single-step methods
# ---------------------------------------------------------------------


def euler_step(state, params, dynamics, dt):
    """
    Forward Euler: simplest integrator, O(h) accuracy.
    Systematically adds energy to Hamiltonian systems.
    """
    return state + dt * dynamics(state, params)


def rk4_step(state, params, dynamics, dt):
    """
    Classic 4th-order Runge-Kutta: O(h^4) accuracy.
    Excellent accuracy but not symplectic—energy drifts over long times.
    """
    k1 = dynamics(state, params)
    k2 = dynamics(state + dt * k1 / 2.0, params)
    k3 = dynamics(state + dt * k2 / 2.0, params)
    k4 = dynamics(state + dt * k3, params)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def verlet_step(q, p, params, grad_V, mass_matrix, dt):
    """
    Störmer-Verlet for separable Hamiltonian H = T(p) + V(q).

    Symplectic: preserves phase space volume, bounded energy error.

    The leapfrog structure:
        1. Half kick: p moves under V for dt/2
        2. Full drift: q moves under T for dt
        3. Half kick: p moves under V for dt/2

    Args:
        q: position array
        p: momentum array
        params: dict of physical parameters
        grad_V: function (q, params) -> ∇V(q)
        mass_matrix: array of masses (diagonal mass matrix)
        dt: timestep

    Returns:
        (q_new, p_new)
    """
    # Half kick
    p_half = p - 0.5 * dt * grad_V(q, params)

    # Full drift (assuming T = Σ p²/2m)
    q_new = q + dt * p_half / mass_matrix

    # Half kick
    p_new = p_half - 0.5 * dt * grad_V(q_new, params)

    return q_new, p_new


def verlet_step_scalar(state, params, grad_V, m, dt):
    """
    Störmer-Verlet for single degree of freedom with scalar mass.

    state = [q, p]

    Convenience wrapper for the common 1D case.
    """
    q, p = state[0], state[1]

    # Half kick
    p_half = p - 0.5 * dt * grad_V(jnp.array([q]), params)[0]

    # Full drift
    q_new = q + dt * p_half / m

    # Half kick
    p_new = p_half - 0.5 * dt * grad_V(jnp.array([q_new]), params)[0]

    return jnp.array([q_new, p_new])


# ---------------------------------------------------------------------
# Trajectory integration via scan
# ---------------------------------------------------------------------


@partial(jit, static_argnums=(1, 4))
def integrate_rk4(state_0, n_steps, dt, params, dynamics):
    """
    Integrate using RK4 for n_steps.

    Args:
        state_0: initial state array
        n_steps: number of integration steps (static)
        dt: timestep
        params: dict of parameters
        dynamics: function (state, params) -> d_state/dt (static)

    Returns:
        trajectory: array of shape (n_steps, state_dim)
    """

    def step_fn(carry, _):
        state, t = carry
        new_state = rk4_step(state, params, dynamics, dt)
        return (new_state, t + dt), new_state

    _, trajectory = scan(step_fn, (state_0, 0.0), None, length=n_steps)
    return trajectory


@partial(jit, static_argnums=(1, 4))
def integrate_euler(state_0, n_steps, dt, params, dynamics):
    """
    Integrate using forward Euler for n_steps.

    Not recommended for production—use for pedagogy only.
    """

    def step_fn(carry, _):
        state, t = carry
        new_state = euler_step(state, params, dynamics, dt)
        return (new_state, t + dt), new_state

    _, trajectory = scan(step_fn, (state_0, 0.0), None, length=n_steps)
    return trajectory


@partial(jit, static_argnums=(1, 5))
def integrate_verlet(state_0, n_steps, dt, params, mass_matrix, grad_V):
    """
    Integrate using Störmer-Verlet for n_steps.

    Symplectic integrator for separable Hamiltonians.

    Args:
        state_0: initial state [q1, ..., qn, p1, ..., pn]
        n_steps: number of integration steps (static)
        dt: timestep
        params: dict of parameters
        mass_matrix: array of masses
        grad_V: function (q, params) -> ∇V(q) (static)

    Returns:
        trajectory: array of shape (n_steps, state_dim)
    """
    n_dof = len(state_0) // 2
    q_0 = state_0[:n_dof]
    p_0 = state_0[n_dof:]

    def step_fn(carry, _):
        q, p, t = carry
        q_new, p_new = verlet_step(q, p, params, grad_V, mass_matrix, dt)
        state_new = jnp.concatenate([q_new, p_new])
        return (q_new, p_new, t + dt), state_new

    _, trajectory = scan(step_fn, (q_0, p_0, 0.0), None, length=n_steps)
    return trajectory


# ---------------------------------------------------------------------
# Factory functions for integrators
# ---------------------------------------------------------------------


def make_rk4_integrator(dynamics):
    """
    Create an RK4 integrator bound to a specific dynamics function.

    Returns:
        integrate(state_0, n_steps, dt, params) -> trajectory
    """

    @partial(jit, static_argnums=(1,))
    def integrate(state_0, n_steps, dt, params):
        def step_fn(carry, _):
            state, t = carry
            new_state = rk4_step(state, params, dynamics, dt)
            return (new_state, t + dt), new_state

        _, trajectory = scan(step_fn, (state_0, 0.0), None, length=n_steps)
        return trajectory

    return integrate


def make_verlet_integrator(grad_V, n_dof):
    """
    Create a Verlet integrator bound to a specific grad_V function.

    Args:
        grad_V: function (q, params) -> ∇V(q)
        n_dof: number of degrees of freedom

    Returns:
        integrate(state_0, n_steps, dt, params, mass_matrix) -> trajectory
    """

    @partial(jit, static_argnums=(1,))
    def integrate(state_0, n_steps, dt, params, mass_matrix):
        q_0 = state_0[:n_dof]
        p_0 = state_0[n_dof:]

        def step_fn(carry, _):
            q, p, t = carry
            q_new, p_new = verlet_step(q, p, params, grad_V, mass_matrix, dt)
            state_new = jnp.concatenate([q_new, p_new])
            return (q_new, p_new, t + dt), state_new

        _, trajectory = scan(step_fn, (q_0, p_0, 0.0), None, length=n_steps)
        return trajectory

    return integrate
