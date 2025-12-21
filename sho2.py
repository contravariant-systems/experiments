from time import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import scan
from jax import vmap, grad, jit
import matplotlib.pyplot as plt


def f(state, params):
    """
    Returns d/dt (q, p) for a given current state.
    """
    q, p = state
    m, k = params['m'], params['k']
    dqdt = p / m
    dpdt = -k * q
    return jnp.array([dqdt, dpdt])


def euler_step(state, params, dt=0.01):
    """
    Given a state at time t, this function finds the state at a
    time t + dt in the most naive way possible.

    It turns out this method systematically adds more energy to
    Hamiltionian systems as it progresses.
    """
    return state + dt * f(state, params)


def rk4_step(state, params, dt=0.01):
    """
    Given a state at time t, this function finds the state at a
    time t + dt using the Runge–Kutta (4) approximation.

    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """

    k_1 = f(state, params)
    k_2 = f(state + dt*k_1/2.0, params)
    k_3 = f(state + dt*k_2/2.0, params)
    k_4 = f(state + dt*k_3, params)

    return state + dt/6.0*(k_1 + 2.0*k_2 + 2.0*k_3 + k_4)


@partial(jit, static_argnums=(1,))
def integrate_euler(state_0, n_steps, dt, params):

    def step_fn(carry, _):
        state, t = carry
        new_state = euler_step(state, params, dt)
        return (new_state, t + dt), new_state

    _, trajectory = scan(step_fn, (state_0, 0), None, length=n_steps)
    return trajectory

@partial(jit, static_argnums=(1,))
def integrate_rk4(state_0, n_steps, dt, params):

    def step_fn(carry, _):
        state, t = carry
        new_state = rk4_step(state, params, dt)
        return (new_state, t + dt), new_state

    _, trajectory = scan(step_fn, (state_0, 0), None, length=n_steps)
    return trajectory

def total_energy(state, params):
    """H = p²/2m + kq²/2"""
    q, p = state
    m, k = params['m'], params['k']
    return p**2 / (2*m) + k * q**2 / 2

def final_energy(state_0, t_span, dt, params):
    traj = integrate_euler(state_0, t_span, dt, params)
    return total_energy(traj[-1], params)

params = {
    "m": 1.0,
    "k": 1.0,
}
state_0 = jnp.array([1.0, 0.0])
dt = 0.01
n_steps = 100000

# First, let us understand what even means to differentiate final
# energy relative to the initial conditions. As in, how do the
# internal representations look. For this, we start with something
# small and with known sizes to avoid any tracing warnings.

def integrate_euler_small(state_0, params):

    def step_fn(carry, _):
        state, t = carry
        new_state = euler_step(state, params, 0.01)
        return (new_state, t + 0.01), new_state

    _, trajectory = scan(step_fn, (state_0, 0.0), None, length=5)
    return trajectory

def final_energy_small(state_0, params):
    traj = integrate_euler_small(state_0, params)
    return total_energy(traj[-1], params)

# print(final_energy_small(state_0, params))
# grad_final_energy_small = grad(final_energy_small)
# print(jax.make_jaxpr(grad_final_energy_small)(state_0, params))

# Then, we move onto comparing different integrators. Let's compare
# the basic forward Euler with Runge-Kutta (RK 4).

traj_euler = integrate_euler(state_0, n_steps, dt, params)

plt.plot(traj_euler[:, 0])
plt.ylabel('Position over time (Forward Euler)')
plt.show()

traj_rk4 = integrate_rk4(state_0, n_steps, dt, params)

plt.plot(traj_rk4[:, 0])
plt.ylabel('Position over time (Runge-Kutta 4)')
plt.show()

energy_evolution_euler = vmap(total_energy, (0, None))(traj_euler, params)
plt.plot(energy_evolution_euler)
plt.ylabel('Energy over time (Forward Euler)')
plt.show()

energy_evolution_rk4 = vmap(total_energy, (0, None))(traj_rk4, params)
plt.plot(energy_evolution_rk4)
plt.ylabel('Energy over time (Runge-Kutta 4)')
plt.show()

