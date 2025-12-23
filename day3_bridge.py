from functools import partial

from sympy import symbols, Rational, diff, solve, lambdify
import jax
from jax import jit, grad
from jax.lax import scan
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from optax import transforms


def derive_equations_of_motion(L, q_vars, q_dot_vars):
    """
    Given a symbolic Lagrangian, derive symbolic accelerations.
    """

    q_ddot_vars = [symbols(f"{qd}_dot") for qd in q_dot_vars]
    accelerations = {}
    for q, q_dot, q_ddot in zip(q_vars, q_dot_vars, q_ddot_vars):
        dL_dq = diff(L, q)
        dL_dq_dot = diff(L, q_dot)
        d_dt_dL_dq_dot = sum(
            diff(dL_dq_dot, qv) * qv_dot for qv, qv_dot in zip(q_vars, q_dot_vars)
        ) + sum(
            diff(dL_dq_dot, qv_dot) * qv_ddot
            for qv_dot, qv_ddot in zip(q_dot_vars, q_ddot_vars)
        )

        euler_lagrange = d_dt_dL_dq_dot - dL_dq
        accelerations[q_ddot] = euler_lagrange

    solutions = solve(list(accelerations.values()), q_ddot_vars)
    return solutions, q_ddot_vars


def derive_equations_of_motion_2(L, q_vars, q_dot_vars):
    """
    Given a symbolic Lagrangian, derive symbolic accelerations.
    Returns everything needed to generate dynamics.
    """
    q_ddot_vars = [symbols(f"{qd}_dot") for qd in q_dot_vars]

    # Extract parameter symbols: everything in L that isn't q or q_dot
    all_symbols = L.free_symbols
    param_syms = list(all_symbols - set(q_vars) - set(q_dot_vars))

    accelerations = {}
    for q, q_dot, q_ddot in zip(q_vars, q_dot_vars, q_ddot_vars):
        dL_dq = diff(L, q)
        dL_dq_dot = diff(L, q_dot)
        d_dt_dL_dq_dot = sum(
            diff(dL_dq_dot, qv) * qv_dot for qv, qv_dot in zip(q_vars, q_dot_vars)
        ) + sum(
            diff(dL_dq_dot, qv_dot) * qv_ddot
            for qv_dot, qv_ddot in zip(q_dot_vars, q_ddot_vars)
        )
        euler_lagrange = d_dt_dL_dq_dot - dL_dq
        accelerations[q_ddot] = euler_lagrange

    solutions = solve(list(accelerations.values()), q_ddot_vars)

    return {
        "solutions": solutions,
        "q_vars": q_vars,
        "q_dot_vars": q_dot_vars,
        "q_ddot_vars": q_ddot_vars,
        "param_syms": param_syms,
    }


def make_dynamics_from_eom(eom):
    """
    Given result from derive_equations_of_motion, generate JAX dynamics.
    """
    solutions = eom["solutions"]
    q_vars = eom["q_vars"]
    q_dot_vars = eom["q_dot_vars"]
    q_ddot_vars = eom["q_ddot_vars"]
    param_syms = eom["param_syms"]

    n_dof = len(q_vars)
    q_ddot_exprs = [solutions[qdd] for qdd in q_ddot_vars]

    all_inputs = list(q_vars) + list(q_dot_vars) + list(param_syms)
    q_ddot_fn = lambdify(all_inputs, q_ddot_exprs, modules="jax")

    def dynamics(state, params):
        q_vals = [state[i] for i in range(n_dof)]
        q_dot_vals = [state[n_dof + i] for i in range(n_dof)]
        param_vals = [params[str(p)] for p in param_syms]
        q_ddot_vals = q_ddot_fn(*q_vals, *q_dot_vals, *param_vals)
        return jnp.array([*q_dot_vals, *q_ddot_vals])

    return dynamics


def make_dynamics_generated(q_ddot_expr, q_sym, q_dot_sym, param_syms):
    """
    Given a sympy expression for q_ddot, generate the dynamics.
    """

    q_ddot_fn = lambdify([q_sym, q_dot_sym] + param_syms, q_ddot_expr, modules="jax")

    def dynamics(state, params):
        q, q_dot = state
        params = [params[str(p)] for p in param_syms]
        q_ddot = q_ddot_fn(q, q_dot, *params)
        return jnp.array([q_dot, q_ddot])

    return dynamics


def dynamics_handwritten(state, params):
    """
    Simply write out the (Lagrangian) dynamocs for a simple
    harmonic oscillator by hand.
    """
    q, q_dot = state
    m, k = params["m"], params["k"]
    return jnp.array([q_dot, -k / m * q])


def rk4_step(state, params, dynamics, dt):
    """
    Given a state at time t, this function finds the state at a
    time t + dt using the Runge–Kutta (4) approximation.

    https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """

    k_1 = dynamics(state, params)
    k_2 = dynamics(state + dt * k_1 / 2.0, params)
    k_3 = dynamics(state + dt * k_2 / 2.0, params)
    k_4 = dynamics(state + dt * k_3, params)

    return state + dt / 6.0 * (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4)


# @partial(jit, static_argnums=(1,))
def integrate_rk4(state_0, n_steps, dt, params, dynamics):

    def step_fn(carry, _):
        state, t = carry
        new_state = rk4_step(state, params, dynamics, dt)
        return (new_state, t + dt), new_state

    _, trajectory = scan(step_fn, (state_0, 0), None, length=n_steps)
    return trajectory


def loss(traj_observed, state_0, n_steps, dt, params_guess, dynamics):
    traj_guess = integrate_rk4(state_0, n_steps, dt, params_guess, dynamics)
    return jnp.linalg.norm(traj_observed - traj_guess) ** 2


# Use sympy to work out the Euler-Lagrange equation from the
# Lagrangian, and then generate runnable dynamics from it.
q, q_dot, q_ddot = symbols("q q_dot q_ddot")
m, k = symbols("m k", positive=True)

L = Rational(1, 2) * m * q_dot**2 - Rational(1, 2) * k * q**2

dL_dq = diff(L, q)
dL_dq_dot = diff(L, q_dot)
d_dt_dL_dq_dot = diff(dL_dq_dot, q) * q_dot + diff(dL_dq_dot, q_dot) * q_ddot

euler_lagrange = d_dt_dL_dq_dot - dL_dq
q_ddot_solution = solve(euler_lagrange, q_ddot)[0]

dynamics_generated = make_dynamics_generated(q_ddot_solution, q, q_dot, [m, k])

state_0 = jnp.array([1.0, 0.0])
params = {"m": 1.0, "k": 2.0}
n_steps = 1000
dt = 0.01

traj_handwritten = integrate_rk4(state_0, n_steps, dt, params, dynamics_handwritten)
traj_generated = integrate_rk4(state_0, n_steps, dt, params, dynamics_generated)

# print("Max difference:", jnp.max(jnp.abs(traj_handwritten - traj_generated)))

# plt.plot(traj_handwritten[:, 0])
# plt.ylabel('Position over time (Runge-Kutta 4, Handwritten Dynamics)')
# plt.show()

# plt.plot(traj_generated[:, 0])
# plt.ylabel('Position over time (Runge-Kutta 4, Generated Dynamics)')
# plt.show()

params_true = {"m": 1.0, "k": 2.0}
traj_observed = integrate_rk4(state_0, n_steps, dt, params_true, dynamics_generated)

params_guess = {"m": 1.0, "k": 1.0}

grad_loss = grad(loss, argnums=4)
# print(grad_loss(traj_observed, state_0, n_steps, dt, params_guess, dynamics_generated))


eom = derive_equations_of_motion_2(L, [q], [q_dot])
dynamics_generated_2 = make_dynamics_from_eom(eom)
solution, q_dot_dot_vars = derive_equations_of_motion(L, [q], [q_dot])
dynamics_generated_2 = make_dynamics_generated(
    solution[q_dot_dot_vars[0]], q, q_dot, [m, k]
)
traj_generated_2 = integrate_rk4(state_0, n_steps, dt, params, dynamics_generated_2)
# plt.plot(traj_generated_2[:, 0])
# plt.ylabel('Position over time (Runge-Kutta 4, Generated Dynamics 2)')
# plt.show()

print("Max difference:", jnp.max(jnp.abs(traj_handwritten - traj_generated_2)))

q1, q1_dot, q2, q2_dot = symbols("q1 q1_dot q2 q2_dot")
m, k = symbols("m k", positive=True)

L_iso = Rational(1, 2) * m * (q1_dot**2 + q2_dot**2) - Rational(1, 2) * k * (
    q1**2 + q2**2
)
eom_iso = derive_equations_of_motion_2(L_iso, [q1, q2], [q1_dot, q2_dot])
dynamics_iso = make_dynamics_from_eom(eom_iso)

state_0 = jnp.array([1.0, 2.0, 3.0, 4.0])
params = {"m": 1.0, "k": 1.0}
n_steps = 1000
dt = 0.01
traj_iso = integrate_rk4(state_0, n_steps, dt, params, dynamics_iso)
# print(traj_iso.shape)

# plt.plot(traj_iso)
# plt.ylabel('2D Isotropic Oscillator (All DOFs)')
# plt.show()

params_guess = {"m": 2.0, "k": 1.0}
print(grad(loss, argnums=4)(traj_iso, state_0, n_steps, dt, params_guess, dynamics_iso))

q1, q1_dot, q2, q2_dot = symbols("q1 q1_dot q2 q2_dot")
m, k, k_c = symbols("m k k_c", positive=True)

L_coupled = (
    Rational(1, 2) * m * (q1_dot**2 + q2_dot**2)
    - Rational(1, 2) * k * (q1**2 + q2**2)
    - Rational(1, 2) * k_c * (q2 - q1) ** 2
)

eom_coupled = derive_equations_of_motion_2(L_coupled, [q1, q2], [q1_dot, q2_dot])
# print(eom_coupled)

dynamics_coupled = make_dynamics_from_eom(eom_coupled)

state_0 = jnp.array([1.0, 0.0, 0.0, 0.0])  # q1=1, q1_dot=0, q2=0, q2_dot=0
params = {"m": 1.0, "k": 1.0, "k_c": 0.5}
n_steps = 2000
dt = 0.01

traj_coupled = integrate_rk4(state_0, n_steps, dt, params, dynamics_coupled)

# plt.figure()
# plt.plot(traj_coupled[:, 0], label='q1')
# plt.plot(traj_coupled[:, 2], label='q2')
# plt.legend()
# plt.ylabel('Position')
# plt.xlabel('Time step')
# plt.show()

params_true = {"m": 1.0, "k": 1.0, "k_c": 0.5}
state_0 = jnp.array([1.0, 0.0, 0.0, 0.0])
n_steps = 20
dt = 0.01

traj_observed = integrate_rk4(state_0, n_steps, dt, params_true, dynamics_coupled)

key = jax.random.PRNGKey(42)
noise = 0.05 * jax.random.normal(key, traj_observed.shape)
traj_observed = traj_observed + noise

params_guess = {"m": 1.0, "k": 1.0, "k_c": 0.1}
print(
    grad(loss, argnums=4)(
        traj_observed, state_0, n_steps, dt, params_guess, dynamics_coupled
    )
)

eta = 1.0e-2
freeze = transforms.freeze
mask = {"m": True, "k": True, "k_c": False}

optimizer = optax.chain(optax.adam(eta), freeze(mask))
params_guess = {"m": 1.0, "k": 1.0, "k_c": 0.1}
opt_state = optimizer.init(params_guess)

# for i in range(1000):
#     grads = grad(loss, argnums=4)(traj_observed, state_0, n_steps, dt, params_guess, dynamics_coupled)
#     updates, opt_state = optimizer.update(grads, opt_state)
#     params_guess = optax.apply_updates(params_guess, updates)

#     if i % 10 == 0:
#         print(f"k_c={params_guess['k_c']:.6f}")

q, q_dot = symbols("q q_dot")
m, a, b = symbols("m a b", positive=True)

V = a * q**4 - b * q**2
L_dw = Rational(1, 2) * m * q_dot**2 - V

eom_dw = derive_equations_of_motion_2(L_dw, [q], [q_dot])
print(eom_dw["solutions"])

dynamics_dw = make_dynamics_from_eom(eom_dw)

params = {"m": 1.0, "a": 1.0, "b": 2.0}
n_steps = 2000
dt = 0.01

# Low energy: oscillates in one well
state_low = jnp.array([0.5, 0.0])

# High energy: crosses between wells
state_high = jnp.array([0.1, 2.0])

traj_low = integrate_rk4(state_low, n_steps, dt, params, dynamics_dw)
traj_high = integrate_rk4(state_high, n_steps, dt, params, dynamics_dw)

plt.figure()
plt.plot(traj_low[:, 0], label="low energy")
plt.plot(traj_high[:, 0], label="high energy")
plt.legend()
plt.ylabel("q")
plt.show()

params_true = {"m": 1.0, "a": 1.0, "b": 2.0}
state_0 = jnp.array([0.5, 0.0])
traj_observed = integrate_rk4(state_0, 500, dt, params_true, dynamics_dw)

# Can we learn b?
params_guess = {"m": 1.0, "a": 1.0, "b": 1.0}
grads = grad(loss, argnums=4)(
    traj_observed, state_0, 500, dt, params_guess, dynamics_dw
)
print(grads)
