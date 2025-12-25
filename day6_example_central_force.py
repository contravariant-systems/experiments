"""
Day 6: Comprehensive API Test
Central Force Problem (Kepler-like)

Tests:
- LagrangianSystem construction
- Symbolic properties (L, H, T, V, cyclic coordinates)
- Auto integrator selection (separable → Verlet)
- Manual integrator override (RK4)
- Conservation checking (energy + angular momentum)
- Plotting (trajectory, phase space, energy error)
- Parameter learning with energy-statistic loss
"""

from sympy import symbols, Rational
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt

from contravariant.systems import LagrangianSystem
from contravariant.plotting import (
    plot_phase_space,
    compare_energy_errors,
)

# =============================================================================
# 1. DEFINE THE SYSTEM
# =============================================================================

# Coordinates: r (radial), theta (angular)
r, r_dot = symbols("r r_dot")
theta, theta_dot = symbols("theta theta_dot")
m, k = symbols("m k", positive=True)

# Central force Lagrangian: L = T - V
# T = ½m(ṙ² + r²θ̇²)  (kinetic energy in polar)
# V = -k/r            (gravitational-like attractive potential)
L_central = Rational(1, 2) * m * (r_dot**2 + r**2 * theta_dot**2) + k / r

# Create the system
sys = LagrangianSystem(L_central, [r, theta], [r_dot, theta_dot])

print("=" * 60)
print("SYSTEM ANALYSIS")
print("=" * 60)
print(sys)
print()

# =============================================================================
# 2. INSPECT SYMBOLIC PROPERTIES
# =============================================================================

print("Kinetic energy T =", sys.kinetic_energy)
print("Potential energy V =", sys.potential_energy)
print("Equations of motion:", sys.equations_of_motion)
print()

# Cyclic coordinates: theta doesn't appear in L
print("Cyclic coordinates:", sys.cyclic_coordinates)
# This should show (theta, m*r**2*theta_dot) - angular momentum!

# Derive angular momentum from rotation symmetry
# For polar coords, rotation is just translation in theta: ξ = [0, 1]
L_z = sys.conserved_quantity([0, 1])
print("Angular momentum L_z =", L_z)
print()

# =============================================================================
# 3. INTEGRATE WITH AUTO-SELECTED METHOD
# =============================================================================

print("=" * 60)
print("INTEGRATION (AUTO METHOD)")
print("=" * 60)

# Physical parameters
params = {"m": 1.0, "k": 1.0}

# Initial conditions: circular-ish orbit
# For circular orbit: v_theta = sqrt(k/mr) gives balance
r_0 = 1.0
theta_0 = 0.0
r_dot_0 = 0.1  # small radial velocity (slightly elliptical)
theta_dot_0 = 1.0  # angular velocity

state_0 = jnp.array([r_0, theta_0, r_dot_0, theta_dot_0])

n_steps = 10000
dt = 0.01

# Auto-selects Verlet (system is separable)
traj_auto = sys.integrate(state_0, n_steps, dt, params)
print(f"Integrated {n_steps} steps with dt={dt}")
print(f"Method auto-selected: {'Verlet' if sys.is_separable else 'RK4'}")
print(f"Final state: {traj_auto[-1]}")
print()

# =============================================================================
# 4. CHECK CONSERVATION
# =============================================================================

print("=" * 60)
print("CONSERVATION CHECK")
print("=" * 60)

conservation = sys.check_conservation(traj_auto, params, {"L_z": L_z})
for name, (max_err, rel_err) in conservation.items():
    print(f"  {name}: max_err = {max_err:.2e}, rel_err = {rel_err:.2e}")
print()

# =============================================================================
# 5. COMPARE INTEGRATORS (use RK4 with different dt)
# =============================================================================

print("=" * 60)
print("INTEGRATOR COMPARISON (RK4, varying dt)")
print("=" * 60)

traj_coarse = sys.integrate(state_0, 1000, 0.1, params, method='rk4')
traj_fine = sys.integrate(state_0, 10000, 0.01, params, method='rk4')

conservation_coarse = sys.check_conservation(traj_coarse, params, {'L_z': L_z})
conservation_fine = sys.check_conservation(traj_fine, params, {'L_z': L_z})

print("RK4 (dt=0.1, 1000 steps):")
for name, (max_err, rel_err) in conservation_coarse.items():
    print(f"  {name}: max_err = {max_err:.2e}")

print("RK4 (dt=0.01, 10000 steps):")
for name, (max_err, rel_err) in conservation_fine.items():
    print(f"  {name}: max_err = {max_err:.2e}")

# =============================================================================
# 6. PLOTTING
# =============================================================================

print("=" * 60)
print("PLOTTING")
print("=" * 60)


# Convert polar to Cartesian for visualization
def polar_to_cartesian(traj):
    r_vals = traj[:, 0]
    theta_vals = traj[:, 1]
    x = r_vals * jnp.cos(theta_vals)
    y = r_vals * jnp.sin(theta_vals)
    return x, y


x, y = polar_to_cartesian(traj_fine)

# Orbit plot
plt.figure(figsize=(6, 6))
plt.plot(x, y, "b-", linewidth=0.5)
plt.plot(0, 0, "ko", markersize=10)  # Central body
plt.xlabel("x")
plt.ylabel("y")
plt.title("Central Force Orbit")
plt.axis("equal")
plt.grid(True)
plt.savefig("central_force_orbit.png", dpi=150)
plt.show()

# Energy error comparison
energy_fn = sys._energy_fn  # Access for plotting
compare_energy_errors(
    [traj_coarse, traj_fine],
    energy_fn,
    params,
    ["Coarse", "Fine"],
    title="Central Force: Energy Error Comparison",
)

# Phase space for radial coordinate
plot_phase_space(traj_fine, dof_index=0, title="Radial Phase Space (r, ṙ)")

print("Plots saved.")
print()

# =============================================================================
# 7. PARAMETER LEARNING
# =============================================================================

print("=" * 60)
print("PARAMETER LEARNING")
print("=" * 60)

# Generate "observed" trajectory with true k
params_true = {"m": 1.0, "k": 2.0}
traj_observed = sys.integrate(state_0, 1000, 0.01, params_true)


# Try to recover k starting from a wrong guess
def energy_statistic_loss(params_guess):
    traj_pred = sys.integrate(state_0, 1000, 0.01, params_guess)
    # Match mean squared radial velocity (depends on k)
    mean_rdot2_pred = jnp.mean(traj_pred[:, 2] ** 2)
    mean_rdot2_obs = jnp.mean(traj_observed[:, 2] ** 2)
    return (mean_rdot2_pred - mean_rdot2_obs) ** 2


# Gradient descent
k_guess = 0.5
lr = 0.5
print(f"True k = {params_true['k']}")
print(f"Initial guess k = {k_guess}")
print()
print("Gradient descent:")

for i in range(20):
    params_guess = {"m": 1.0, "k": k_guess}
    g = grad(energy_statistic_loss)(params_guess)["k"]
    k_guess = k_guess - lr * g
    if i % 4 == 0:
        loss = energy_statistic_loss(params_guess)
        print(f"  step {i:2d}: k = {k_guess:.4f}, loss = {loss:.2e}")

print()
print(f"Final k = {k_guess:.4f} (true = {params_true['k']})")

# =============================================================================
# 8. COMPILE CUSTOM EXPRESSION
# =============================================================================

print()
print("=" * 60)
print("CUSTOM COMPILED EXPRESSION")
print("=" * 60)

# Compile the specific angular momentum expression
L_z_fn = sys.compile(L_z)

# Evaluate along trajectory
L_z_values = jnp.array([L_z_fn(state, params) for state in traj_fine[:10]])
print(f"L_z at first 10 states: {L_z_values}")
print(f"L_z variation: {jnp.std(L_z_values):.2e}")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
