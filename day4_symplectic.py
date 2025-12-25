"""
Day 4: Structure-Preserving Integration

Compare RK4 vs Störmer-Verlet on some example problems.
Demonstrate that symplectic integration preserves energy bounds.
"""

from sympy import cos
from jax import vmap, grad
import jax.numpy as jnp
from sympy import symbols, Rational

# Import from our clean package
from contravariant import (
    derive_equations_of_motion,
    extract_kinetic_potential,
    compile_lagrangian_dynamics,
    compile_grad_V,
    compile_energy,
    make_rk4_integrator,
    make_verlet_integrator,
    compare_energy_errors,
    plot_phase_space_cloud,
)


# ---------------------------------------------------------------------
# Define the physics symbolically
# ---------------------------------------------------------------------

q, q_dot = symbols("q q_dot")
m, k = symbols("m k", positive=True)

# Lagrangian for simple harmonic oscillator
L = Rational(1, 2) * m * q_dot**2 - Rational(1, 2) * k * q**2

# Derive equations of motion and extract structure
eom = derive_equations_of_motion(L, [q], [q_dot])
energy_parts = extract_kinetic_potential(L, [q], [q_dot])

print("Equations of motion:", eom["solutions"])
print("Separable?", energy_parts["is_separable"])
print("V =", energy_parts["V"])
print("∇V =", energy_parts["grad_V"])


# ---------------------------------------------------------------------
# Generate JAX functions from symbolic expressions
# ---------------------------------------------------------------------

dynamics = compile_lagrangian_dynamics(eom)
grad_V = compile_grad_V(eom, energy_parts)
energy_fn = compile_energy(eom, energy_parts)


# ---------------------------------------------------------------------
# Create integrators
# ---------------------------------------------------------------------

integrate_rk4 = make_rk4_integrator(dynamics)
integrate_verlet = make_verlet_integrator(grad_V, n_dof=1)


# ---------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------

# Physical parameters
params = {"m": 1.0, "k": 1.0}

# Initial state: [q, p] for Hamiltonian / [q, q_dot] for Lagrangian
# Note: for SHO with m=1, p = m*q_dot = q_dot, so they're the same
state_0 = jnp.array([1.0, 0.0])

# Integration parameters
n_steps = 10000
dt = 0.001

# Mass matrix for Verlet
mass_matrix = jnp.array([params["m"]])

print(f"\nIntegrating for {n_steps} steps with dt={dt}...")

# Run both integrators
traj_rk4 = integrate_rk4(state_0, n_steps, dt, params)
traj_verlet = integrate_verlet(state_0, n_steps, dt, params, mass_matrix)

print("RK4 trajectory shape:", traj_rk4.shape)
print("Verlet trajectory shape:", traj_verlet.shape)


# ---------------------------------------------------------------------
# Compare energy conservation
# ---------------------------------------------------------------------

print("\n--- Energy Conservation Comparison ---")

# Compute energies

energies_rk4 = vmap(lambda s: energy_fn(s, params))(traj_rk4)
energies_verlet = vmap(lambda s: energy_fn(s, params))(traj_verlet)

E0 = energy_fn(state_0, params)
print(f"Initial energy: {E0:.10f}")
print(f"RK4 final energy: {energies_rk4[-1]:.10f}")
print(f"Verlet final energy: {energies_verlet[-1]:.10f}")

print(f"\nRK4 energy drift: {energies_rk4[-1] - E0:.2e}")
print(f"Verlet energy drift: {energies_verlet[-1] - E0:.2e}")

print(f"\nRK4 max energy error: {jnp.max(jnp.abs(energies_rk4 - E0)):.2e}")
print(f"Verlet max energy error: {jnp.max(jnp.abs(energies_verlet - E0)):.2e}")


# ---------------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------------

compare_energy_errors(
    [traj_rk4, traj_verlet],
    energy_fn,
    params,
    ["RK4", "Verlet"],
    title="Energy Error: RK4 vs Störmer-Verlet (100,000 steps)",
)

# ---------------------------------------------------------------------
# Extend to multi-DOF: 2D Isotropic Oscillator
# ---------------------------------------------------------------------

q1, q1_dot, q2, q2_dot = symbols("q1 q1_dot q2 q2_dot")
m, k = symbols("m k", positive=True)

L_2d = Rational(1, 2) * m * (q1_dot**2 + q2_dot**2) - Rational(1, 2) * k * (
    q1**2 + q2**2
)

eom_2d = derive_equations_of_motion(L_2d, [q1, q2], [q1_dot, q2_dot])
energy_parts_2d = extract_kinetic_potential(L_2d, [q1, q2], [q1_dot, q2_dot])

print("Separable?", energy_parts_2d["is_separable"])
print("∇V =", energy_parts_2d["grad_V"])

dynamics_2d = compile_lagrangian_dynamics(eom_2d)
grad_V_2d = compile_grad_V(eom_2d, energy_parts_2d)
energy_fn_2d = compile_energy(eom_2d, energy_parts_2d)

integrate_rk4_2d = make_rk4_integrator(dynamics_2d)
integrate_verlet_2d = make_verlet_integrator(grad_V_2d, n_dof=2)

# Initial state: [q1, q2, q1_dot, q2_dot]
state_0_2d = jnp.array([1.0, 0.5, 0.0, 1.0])
params = {"m": 1.0, "k": 1.0}
mass_matrix_2d = jnp.array([params["m"], params["m"]])

n_steps = 100000
dt = 0.01

traj_rk4_2d = integrate_rk4_2d(state_0_2d, n_steps, dt, params)
traj_verlet_2d = integrate_verlet_2d(state_0_2d, n_steps, dt, params, mass_matrix_2d)

compare_energy_errors(
    [traj_rk4_2d, traj_verlet_2d],
    energy_fn_2d,
    params,
    ["RK4", "Verlet"],
    title="2D Oscillator: Energy Error",
)

# ---------------------------------------------------------------------
# Extend to multi-DOF: Double pendulum
# ---------------------------------------------------------------------

theta1, theta1_dot, theta2, theta2_dot = symbols("theta1 theta1_dot theta2 theta2_dot")
m1, m2, l1, l2, g = symbols("m1 m2 l1 l2 g", positive=True)

# Double pendulum Lagrangian (simplified, equal masses and lengths)
# T depends on theta1, theta2 (not separable!)
L_dp = (
    Rational(1, 2) * (m1 + m2) * l1**2 * theta1_dot**2
    + Rational(1, 2) * m2 * l2**2 * theta2_dot**2
    + m2 * l1 * l2 * theta1_dot * theta2_dot * cos(theta1 - theta2)
    - (m1 + m2) * g * l1 * (1 - cos(theta1))
    - m2 * g * l2 * (1 - cos(theta2))
)


eom_dp = derive_equations_of_motion(L_dp, [theta1, theta2], [theta1_dot, theta2_dot])
energy_parts_dp = extract_kinetic_potential(
    L_dp, [theta1, theta2], [theta1_dot, theta2_dot]
)

print("Separable?", energy_parts_dp["is_separable"])

# Because the above is not separable, the following is not supposed to
# work, but it still seems to. As in RK4 goes way off, but the Verlet
# still seems to hold in some oscillatory range.

print("T =", energy_parts_dp["T"])
print("V =", energy_parts_dp["V"])

grad_V_dp = compile_grad_V(eom_dp, energy_parts_dp)
energy_fn_dp = compile_energy(eom_dp, energy_parts_dp)

dynamics_dp = compile_lagrangian_dynamics(eom_dp)
integrate_rk4_dp = make_rk4_integrator(dynamics_dp)
integrate_verlet_dp = make_verlet_integrator(grad_V_dp, n_dof=2)

# Initial state: [q1, q2, q1_dot, q2_dot]
state_0_dp = jnp.array([2.5, 2.0, 0.0, 0.0])
params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.8}
mass_matrix_dp = jnp.array([params["m1"], params["m2"]])

n_steps = 100000
dt = 0.1

traj_rk4_dp = integrate_rk4_dp(state_0_dp, n_steps, dt, params)
traj_verlet_dp = integrate_verlet_dp(state_0_dp, n_steps, dt, params, mass_matrix_dp)

compare_energy_errors(
    [traj_rk4_dp, traj_verlet_dp],
    energy_fn_dp,
    params,
    ["RK4", "Verlet"],
    title="Double Pendulum: Energy Error",
)

# ---------------------------------------------------------------------
# Extend to multi-DOF: Coupled pendulum
# ---------------------------------------------------------------------

q1, q1_dot, q2, q2_dot = symbols("q1 q1_dot q2 q2_dot")
m, k, k_c = symbols("m k k_c", positive=True)

L_coupled = (
    Rational(1, 2) * m * (q1_dot**2 + q2_dot**2)
    - Rational(1, 2) * k * (q1**2 + q2**2)
    - Rational(1, 2) * k_c * (q2 - q1) ** 2
)

energy_parts_coupled = extract_kinetic_potential(L_coupled, [q1, q2], [q1_dot, q2_dot])

print("T =", energy_parts_coupled["T"])
print("V =", energy_parts_coupled["V"])
print("Separable?", energy_parts_coupled["is_separable"])

# ---------------------------------------------------------------------
# Look at phase space plots
# ---------------------------------------------------------------------

n_particles = 100
theta_angles = jnp.linspace(0, 2 * jnp.pi, n_particles, endpoint=False)
theta, theta_dot = symbols("theta theta_dot")
m, l, g = symbols("m l g", positive=True)

L_pendulum = Rational(1, 2) * m * l**2 * theta_dot**2 - m * g * l * (1 - cos(theta))

eom_pend = derive_equations_of_motion(L_pendulum, [theta], [theta_dot])
energy_parts_pend = extract_kinetic_potential(L_pendulum, [theta], [theta_dot])

print("Separable?", energy_parts_pend["is_separable"])

dynamics_pend = compile_lagrangian_dynamics(eom_pend)
grad_V_pend = compile_grad_V(eom_pend, energy_parts_pend)

integrate_rk4_pend = make_rk4_integrator(dynamics_pend)
integrate_verlet_pend = make_verlet_integrator(grad_V_pend, n_dof=1)

# Cloud near the unstable equilibrium (top) where nonlinearity is strong
initial_states_pend = jnp.stack(
    [
        2.5 + 0.2 * jnp.cos(theta_angles),  # Near top of swing
        0.0 + 0.2 * jnp.sin(theta_angles),
    ],
    axis=1,
)

params_pend = {"m": 1.0, "l": 1.0, "g": 9.8}
mass_matrix_pend = jnp.array([params_pend["m"] * params_pend["l"] ** 2])  # I = ml²

n_steps = 50000
dt = 0.01

final_rk4_pend = vmap(lambda s: integrate_rk4_pend(s, n_steps, dt, params_pend)[-1])(
    initial_states_pend
)
final_verlet_pend = vmap(
    lambda s: integrate_verlet_pend(s, n_steps, dt, params_pend, mass_matrix_pend)[-1]
)(initial_states_pend)

plot_phase_space_cloud(
    initial_states_pend,
    [final_rk4_pend, final_verlet_pend],
    ["RK4", "Verlet"],
    dof_index=0,
)

# ---------------------------------------------------------------------
# Does learning (grad) work across all this new work?
# ---------------------------------------------------------------------

params_true = {"m": 1.0, "l": 1.0, "g": 9.8}
params_guess = {"m": 1.0, "l": 1.0, "g": 8.0}

state_0 = jnp.array([0.5, 0.0])
mass_matrix = jnp.array([params_true["m"] * params_true["l"] ** 2])
n_steps = 500
dt = 0.01

traj_observed = integrate_verlet_pend(state_0, n_steps, dt, params_true, mass_matrix)


def loss_verlet(params):
    traj = integrate_verlet_pend(state_0, n_steps, dt, params, mass_matrix)
    return jnp.sum((traj - traj_observed) ** 2)


print("Gradient:", grad(loss_verlet)(params_guess))
