from sympy import symbols, diff, Rational
from jax import grad
import jax.numpy as jnp

from contravariant import (
    find_cyclic_coordinates,
    derive_hamiltonian,
    derive_conserved_quantity,
    derive_equations_of_motion,
    extract_kinetic_potential,
    compile_grad_V,
    compile_energy,
    make_verlet_integrator,
    compile_expression,
    compile_lagrangian_dynamics,
    make_rk4_integrator,
    plot_trajectory,
    plot_energy_error,
)


# ---------------------------------------------------------------------
# Deduce cyclic coordinates automatically
# ---------------------------------------------------------------------

x, x_dot = symbols("x x_dot")
m = symbols("m", positive=True)

L_free = Rational(1, 2) * m * x_dot**2

print("Free particle cyclic coords:", find_cyclic_coordinates(L_free, [x], [x_dot]))

r, r_dot, theta, theta_dot = symbols("r r_dot theta theta_dot")
m = symbols("m", positive=True)

# V(r) is some function of r only
V_r = symbols("V_r")  # treat as symbol for now

L_central = Rational(1, 2) * m * (r_dot**2 + r**2 * theta_dot**2) - V_r

print(
    "Central force cyclic coords:",
    find_cyclic_coordinates(L_central, [r, theta], [r_dot, theta_dot]),
)


# ---------------------------------------------------------------------
# Derive the Hamiltonian automatically
# ---------------------------------------------------------------------

q, q_dot = symbols("q q_dot")
m, k = symbols("m k", positive=True)

L_sho = Rational(1, 2) * m * q_dot**2 - Rational(1, 2) * k * q**2

H, momenta = derive_hamiltonian(L_sho, [q], [q_dot])
print("H =", H)
print("momenta:", momenta)


# ---------------------------------------------------------------------
# Check which quantity is conserved by a symmetry, verify numerically
# ---------------------------------------------------------------------

x, x_dot, y, y_dot = symbols("x x_dot y y_dot")
m, k = symbols("m k", positive=True)

L_2d = Rational(1, 2) * m * (x_dot**2 + y_dot**2) - Rational(1, 2) * k * (x**2 + y**2)

# At point (x, y), the vector (−y,x) is perpendicular to the position
# vector and points counterclockwise. It's the direction of
# infinitesimal rotation.
xi_rotation = [-y, x]

eom_2d = derive_equations_of_motion(L_2d, [x, y], [x_dot, y_dot])
energy_parts_2d = extract_kinetic_potential(L_2d, [x, y], [x_dot, y_dot])
Q = derive_conserved_quantity(L_2d, [x, y], [x_dot, y_dot], xi_rotation)
print("Conserved quantity:", Q)

print("Separable?", energy_parts_2d["is_separable"])
print("∇V =", energy_parts_2d["grad_V"])

grad_V_2d = compile_grad_V(eom_2d, energy_parts_2d)
energy_fn_2d = compile_energy(eom_2d, energy_parts_2d)

integrate_verlet_2d = make_verlet_integrator(grad_V_2d, n_dof=2)

state_0_2d = jnp.array([1.0, 0.5, 0.0, 1.0])
params = {"m": 1.0, "k": 1.0}
mass_matrix_2d = jnp.array([params["m"], params["m"]])

n_steps = 10000000
dt = 0.01

traj_2d = integrate_verlet_2d(state_0_2d, n_steps, dt, params, mass_matrix_2d)
# plot_trajectory(traj_2d)

conserved_quantity_fn = compile_expression(Q, [x, y], [x_dot, y_dot], [m, k])
plot_energy_error(traj_2d, conserved_quantity_fn, params)

# ---------------------------------------------------------------------
# Energy conservation from time translation
# ---------------------------------------------------------------------

t = symbols("t")

print("∂L/∂t =", diff(L_sho, t))
print("Autonomous?", diff(L_sho, t) == 0)

H, momenta = derive_hamiltonian(L_sho, [q], [q_dot])
print("H =", H)

# Numerically, we have shown this in Day 4. The Hamiltonian is bounded
# over very many time steps.

# ---------------------------------------------------------------------
# More robust loss functions that are not based on trajectories
# ---------------------------------------------------------------------

eom = derive_equations_of_motion(L_sho, [q], [q_dot])
energy_parts = extract_kinetic_potential(L_sho, [q], [q_dot])
dynamics = compile_lagrangian_dynamics(eom)
integrate_rk4 = make_rk4_integrator(dynamics)

# Generate observed trajectory with k_true = 2.0
state_0 = jnp.array([1.0, 0.0])
params_true = {"m": 1.0, "k": 2.0}
n_steps = 1000
dt = 0.01

traj_observed = integrate_rk4(state_0, n_steps, dt, params_true)


# Trajectory matching loss (what we used before)
def trajectory_loss(params_guess):
    traj_pred = integrate_rk4(state_0, n_steps, dt, params_guess)
    return jnp.sum((traj_pred - traj_observed) ** 2)


# Energy-statistic loss (phase-invariant)
def energy_statistic_loss(params_guess):
    traj_pred = integrate_rk4(state_0, n_steps, dt, params_guess)
    mean_p2_pred = jnp.mean(traj_pred[:, 1] ** 2)
    mean_p2_obs = jnp.mean(traj_observed[:, 1] ** 2)
    return (mean_p2_pred - mean_p2_obs) ** 2


# Compare gradients starting far from true k
params_guess = {"m": 1.0, "k": 0.5}  # Far from k_true=2.0

print("Trajectory loss gradient:", grad(trajectory_loss)(params_guess))
print("Energy stat loss gradient:", grad(energy_statistic_loss)(params_guess))


# Gradient descent on trajectory loss
k_traj = 0.5
lr_traj = 1e-6  # small because gradient is huge
print("Trajectory matching:")
for i in range(24):
    params_guess = {"m": 1.0, "k": k_traj}
    g = grad(trajectory_loss)(params_guess)["k"]
    k_traj = k_traj - lr_traj * g
    if i % 2 == 0:
        print(f"  step {i}: k = {k_traj:.4f}")

# Gradient descent on energy statistic loss
k_energy = 0.5
lr_energy = 0.5  # can be much larger
print("\nEnergy statistic matching:")
for i in range(24):
    params_guess = {"m": 1.0, "k": k_energy}
    g = grad(energy_statistic_loss)(params_guess)["k"]
    k_energy = k_energy - lr_energy * g
    if i % 2 == 0:
        print(f"  step {i}: k = {k_energy:.4f}")
