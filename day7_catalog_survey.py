"""
Day 7: Full API Survey
Systematically exercise every capability on every catalog system.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from contravariant.catalog import (
    harmonic_oscillator,
    coupled_oscillators,
    anharmonic_oscillator,
    simple_pendulum,
    double_pendulum,
    spherical_pendulum,
)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# =============================================================================
# 1. SIMPLE HARMONIC OSCILLATOR
# =============================================================================
section("1. SIMPLE HARMONIC OSCILLATOR")

sys = harmonic_oscillator()
print(sys)
print()

params = {'m': 1.0, 'k': 1.0}
state_0 = jnp.array([1.0, 0.0])

# Compare integrators
print("Integrator comparison:")
trajs = sys.compare_integrators(
    state_0, n_steps=10000, dt=0.01, params=params,
    save_as='sho', show=False
)

# Learn k from trajectory with true k=2.0
print("Parameter learning:")
params_true = {'m': 1.0, 'k': 2.0}
traj_obs = sys.integrate(state_0, 1000, 0.01, params_true)
result = sys.learn_parameters(
    traj_obs, state_0, n_steps=1000, dt=0.01,
    params_fixed={'m': 1.0},
    params_init={'k': 0.5},
    max_iterations=50,
)
print(f"True k=2.0, Learned k={result['k']:.4f}")


# =============================================================================
# 2. COUPLED OSCILLATORS
# =============================================================================
section("2. COUPLED OSCILLATORS")

sys = coupled_oscillators()
print(sys)
print()

params = {'m': 1.0, 'k': 1.0, 'k_c': 0.5}
state_0 = jnp.array([1.0, 0.0, 0.0, 0.0])

# Compare integrators
print("Integrator comparison:")
trajs = sys.compare_integrators(
    state_0, n_steps=10000, dt=0.01, params=params,
    save_as='coupled', show=False
)

# Learn coupling constant k_c
print("Parameter learning (coupling constant):")
params_true = {'m': 1.0, 'k': 1.0, 'k_c': 0.8}
traj_obs = sys.integrate(state_0, 1000, 0.01, params_true)
result = sys.learn_parameters(
    traj_obs, state_0, n_steps=1000, dt=0.01,
    params_fixed={'m': 1.0, 'k': 1.0},
    params_init={'k_c': 0.2},
    max_iterations=50,
)
print(f"True k_c=0.8, Learned k_c={result['k_c']:.4f}")


# =============================================================================
# 3. ANHARMONIC OSCILLATOR
# =============================================================================
section("3. ANHARMONIC OSCILLATOR")

sys = anharmonic_oscillator(order=4)
print(sys)
print()

params = {'m': 1.0, 'k': 1.0, 'lambda': 0.1}
state_0 = jnp.array([1.0, 0.0])

# Compare integrators
print("Integrator comparison:")
trajs = sys.compare_integrators(
    state_0, n_steps=10000, dt=0.01, params=params,
    save_as='anharmonic', show=False
)

# Learn anharmonic coefficient
print("Parameter learning (anharmonic coefficient):")
params_true = {'m': 1.0, 'k': 1.0, 'lambda': 0.3}
traj_obs = sys.integrate(state_0, 1000, 0.01, params_true)
result = sys.learn_parameters(
    traj_obs, state_0, n_steps=1000, dt=0.01,
    params_fixed={'m': 1.0, 'k': 1.0},
    params_init={'lambda': 0.1},
    max_iterations=50,
)
print(f"True λ=0.3, Learned λ={result['lambda']:.4f}")


# =============================================================================
# 4. SIMPLE PENDULUM
# =============================================================================
section("4. SIMPLE PENDULUM")

sys = simple_pendulum()
print(sys)
print()

params = {'m': 1.0, 'l': 1.0, 'g': 9.81}
state_0 = jnp.array([0.5, 0.0])  # moderate angle

# Compare integrators
print("Integrator comparison:")
trajs = sys.compare_integrators(
    state_0, n_steps=10000, dt=0.01, params=params,
    save_as='pendulum', show=False
)

# Learn gravity
print("Parameter learning (gravity):")
params_true = {'m': 1.0, 'l': 1.0, 'g': 9.81}
params_guess = {'m': 1.0, 'l': 1.0, 'g': 5.0}
traj_obs = sys.integrate(state_0, 1000, 0.01, params_true)
result = sys.learn_parameters(
    traj_obs, state_0, n_steps=1000, dt=0.01,
    params_fixed={'m': 1.0, 'l': 1.0},
    params_init={'g': 5.0},
    max_iterations=50,
)
print(f"True g=9.81, Learned g={result['g']:.4f}")


# =============================================================================
# 5. DOUBLE PENDULUM (chaotic, non-separable)
# =============================================================================
section("5. DOUBLE PENDULUM")

sys = double_pendulum()
print(sys)
print()

params = {'m1': 1.0, 'm2': 1.0, 'l1': 1.0, 'l2': 1.0, 'g': 9.81}
state_0 = jnp.array([jnp.pi/2, jnp.pi/2, 0.0, 0.0])

# Only RK4 (non-separable)
print("Integration (RK4 only):")
traj = sys.integrate(state_0, 10000, 0.01, params)
cons = sys.check_conservation(traj, params)
print(f"Energy conservation: {cons['energy'][0]:.2e}")

# Visualize chaotic trajectory
print("Plotting chaotic trajectory...")
l1, l2 = params['l1'], params['l2']
x1 = l1 * jnp.sin(traj[:, 0])
y1 = -l1 * jnp.cos(traj[:, 0])
x2 = x1 + l2 * jnp.sin(traj[:, 1])
y2 = y1 - l2 * jnp.cos(traj[:, 1])

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x2, y2, 'b-', linewidth=0.2, alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Double Pendulum: Tip Trajectory (Chaotic)')
ax.axis('equal')
ax.grid(True, alpha=0.3)
plt.savefig('double_pendulum_chaos.png', dpi=150)
plt.close()

# Skip learning (chaotic = gradients unreliable)
print("Parameter learning: SKIPPED (chaotic system)")


# =============================================================================
# 6. SPHERICAL PENDULUM (non-separable but integrable)
# =============================================================================
section("6. SPHERICAL PENDULUM")

sys = spherical_pendulum()
print(sys)
print(f"Cyclic coordinates: {sys.cyclic_coordinates}")
print()

params = {'m': 1.0, 'l': 1.0, 'g': 9.81}
state_0 = jnp.array([0.5, 0.0, 0.0, 2.0])  # tilted, spinning

# Check both energy and angular momentum
theta, phi = sys.coordinates
p_phi_expr = sys.cyclic_coordinates[0][1]  # conserved momentum

print("Integration (RK4 only):")
traj = sys.integrate(state_0, 10000, 0.01, params)
cons = sys.check_conservation(traj, params, {'p_phi': p_phi_expr})
print(f"Energy conservation: {cons['energy'][0]:.2e}")
print(f"p_phi conservation: {cons['p_phi'][0]:.2e}")

# 3D visualization
print("Plotting 3D trajectory...")
theta_vals = traj[:, 0]
phi_vals = traj[:, 1]
l = params['l']
x = l * jnp.sin(theta_vals) * jnp.cos(phi_vals)
y = l * jnp.sin(theta_vals) * jnp.sin(phi_vals)
z = -l * jnp.cos(theta_vals)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b-', linewidth=0.3, alpha=0.7)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Spherical Pendulum')
plt.savefig('spherical_pendulum_3d.png', dpi=150)
plt.close()

# Learn gravity
print("Parameter learning (gravity):")
params_true = {'m': 1.0, 'l': 1.0, 'g': 9.81}
traj_obs = sys.integrate(state_0, 1000, 0.01, params_true)
result = sys.learn_parameters(
    traj_obs, state_0, n_steps=1000, dt=0.01,
    params_fixed={'m': 1.0, 'l': 1.0},
    params_init={'g': 5.0},
    max_iterations=50,
)
print(f"True g=9.81, Learned g={result['g']:.4f}")


# =============================================================================
# SUMMARY
# =============================================================================
section("SUMMARY")

print("""
| System              | Separable | Integrator | Energy Err | Learn | Extra Q |
|---------------------|-----------|------------|------------|-------|---------|
| SHO                 | ✓         | Both       | O(1e-5)    | ✓     | —       |
| Coupled Oscillators | ✓         | Both       | O(1e-5)    | ✓     | —       |
| Anharmonic          | ✓         | Both       | O(1e-5)    | ✓     | —       |
| Simple Pendulum     | ✓         | Both       | O(1e-4)    | ✓     | —       |
| Double Pendulum     | ✗         | RK4        | O(1e-3)    | skip  | —       |
| Spherical Pendulum  | ✗         | RK4        | O(1e-6)    | ✓     | p_φ     |
""")

print("All plots saved. Survey complete.")
