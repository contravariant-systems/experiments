from time import time
from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import scan
from jax import vmap, grad, jit
import matplotlib.pyplot as plt


def dynamics(state, params):
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
    """
    return state + dt * dynamics(state, params)

params = {
    "m": 1.0,
    "k": 1.0,
}

# state_0 = jnp.array([1.0, 0.0])
# print("Manual loop:")
# print(state_0)
# for j in range(10):
#     state_1 = euler_step(state_0, params, 0.01)
#     print(state_1)
#     state_0 = state_1

# print("-"*72)

def integrate_euler(state0, t_span, dt, params):
    n_steps = int((t_span[1] - t_span[0]) / dt)

    def step_fn(carry, _):
        state, t = carry
        new_state = euler_step(state, params, dt)
        return (new_state, t + dt), new_state

    _, trajectory = scan(step_fn, (state0, t_span[0]), None, length=n_steps)
    return trajectory

# print("jax.lax.scan version:")
state_0 = jnp.array([1.0, 0.0])
t_span = jnp.array([0.0, 100.0])
dt = 0.01
# print(state_0)

traj = integrate_euler(state_0, t_span, dt, params)

# You can see in this plot that it grows in an unphysical way.
plt.plot(traj[:, 0])
plt.ylabel('Position over time')
# plt.show()

def total_energy(state, params):
    """H = p²/2m + kq²/2"""
    q, p = state
    m, k = params['m'], params['k']
    return p**2 / (2*m) + k * q**2 / 2

energy_evolution = vmap(total_energy, (0, None))(traj, params)
plt.plot(energy_evolution)
plt.ylabel('Energy over time')
# plt.show()

# Let's try to make the final energy a function of the inputs, so that
# we can try to study it.
jnp.array([0.0, 1.0])

def final_energy(state_0, t_span, dt, params):
    traj = integrate_euler(state_0, t_span, dt, params)
    return total_energy(traj[-1], params)

print(final_energy(state_0, t_span, dt, params))
grad_final_energy = grad(final_energy, argnums=[0, 3])
print(jax.make_jaxpr(grad_final_energy)(state_0, t_span, dt, params))
# print(grad_final_energy(state_0, t_span, dt, params))

# The following allows us to quickly do 100 independent simulations,
# each 10,000 steps long, without needing to resort to Python looping.
# This will allow for parallelisation lower in the stack as it's just
# setting intent.
state_0_batch = jnp.stack([jnp.array([0.0 + 0.1*i, 0.0]) for i in range(1, 101)])
batched_final_energy = vmap(final_energy, in_axes=(0, None, None, None))(state_0_batch, t_span, dt, params)
# print(batched_final_energy)


# Finally, we try to understand the gains of JIT compilation. We first
# need to reorganise the code a little to be more JIT-friendly.

@partial(jit, static_argnums=(1,))
def integrate_euler_2(state_0, n_steps, dt, params):

    def step_fn(carry, _):
        state, t = carry
        new_state = euler_step(state, params, dt)
        return (new_state, t + dt), new_state

    _, trajectory = scan(step_fn, (state_0, 0), None, length=n_steps)
    return trajectory

n_steps = 100000

t_0 = time()
_ = integrate_euler_2(state_0, n_steps, dt, params)
t_f = time()
print(f"First call: {t_f - t_0:.4f} s")

t_0 = time()
_ = integrate_euler_2(state_0, n_steps, dt, params)
t_f = time()
print(f"Second call: {t_f - t_0:.4f} s")
