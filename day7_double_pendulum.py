"""
Day 7b: Double Pendulum Deep Dive

The double pendulum is the simplest mechanical system that exhibits chaos.
- Non-separable Hamiltonian (no symplectic integrators)
- Sensitive dependence on initial conditions
- Rich dynamics: oscillation, rotation, chaos

This file explores what our framework can and cannot do with chaotic systems.
"""

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from contravariant.catalog import double_pendulum

jax.config.update("jax_enable_x64", True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# 1. THE SYSTEM
# =============================================================================
section("1. THE DOUBLE PENDULUM")

sys = double_pendulum()
print(sys)
print()

print("Key properties:")
print(f"  - Degrees of freedom: {sys.n_dof}")
print(f"  - Separable: {sys.is_separable}")
print(f"  - Cyclic coordinates: {sys.cyclic_coordinates if sys.cyclic_coordinates else 'none'}")
print()

print("Why non-separable?")
print("  The kinetic energy contains: l₁l₂m₂θ̇₁θ̇₂cos(θ₁-θ₂)")
print("  This couples the velocities — T ≠ T₁(θ̇₁) + T₂(θ̇₂)")
print("  Therefore: only RK4, not Verlet/Yoshida")


# =============================================================================
# 2. ENERGY CONSERVATION WITH RK4
# =============================================================================
section("2. ENERGY CONSERVATION (RK4)")

params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}

# Moderate energy initial condition
state_0 = jnp.array([jnp.pi/2, jnp.pi/2, 0.0, 0.0])

# Integrate for a long time
n_steps = 100_000
dt = 0.001
traj = sys.integrate(state_0, n_steps, dt, params)

# Check energy
cons = sys.check_conservation(traj, params)
print(f"Integration: {n_steps} steps, dt={dt}, T={n_steps*dt}s")
print(f"Energy error (max): {cons['energy'][0]:.2e}")
print(f"Energy error (std): {cons['energy'][1]:.2e}")

# Plot energy over time
energy = sys.evaluate_energy_along_trajectory(traj, params)
E0 = energy[0]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(jnp.arange(n_steps) * dt, energy, 'b-', linewidth=0.5)
axes[0].axhline(E0, color='r', linestyle='--', alpha=0.5, label=f'E₀ = {E0:.4f}')
axes[0].set_ylabel('Energy')
axes[0].set_title('Double Pendulum: Energy vs Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(jnp.arange(n_steps) * dt, energy - E0, 'b-', linewidth=0.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Energy Error')
axes[1].set_title('Energy Drift (RK4 is not symplectic)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum_energy.png', dpi=150)
plt.close()
print("Saved: double_pendulum_energy.png")


# =============================================================================
# 3. VISUALIZING CHAOS: TRAJECTORY
# =============================================================================
section("3. CHAOTIC TRAJECTORY")

# Convert to Cartesian for visualization
def polar_to_cartesian(traj, params):
    """Convert (θ₁, θ₂) to (x₁, y₁, x₂, y₂)."""
    l1, l2 = params["l1"], params["l2"]
    theta1 = traj[:, 0]
    theta2 = traj[:, 1]
    
    x1 = l1 * jnp.sin(theta1)
    y1 = -l1 * jnp.cos(theta1)
    x2 = x1 + l2 * jnp.sin(theta2)
    y2 = y1 - l2 * jnp.cos(theta2)
    
    return x1, y1, x2, y2

x1, y1, x2, y2 = polar_to_cartesian(traj, params)

# Plot tip trajectory with time-colored segments
fig, ax = plt.subplots(figsize=(10, 10))

# Create colored line segments
points = jnp.array([x2, y2]).T.reshape(-1, 1, 2)
segments = jnp.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, n_steps * dt)
lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.7, linewidth=0.5)
lc.set_array(jnp.arange(n_steps) * dt)
ax.add_collection(lc)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Double Pendulum: Tip Trajectory (color = time)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.colorbar(lc, ax=ax, label='Time (s)')
plt.savefig('double_pendulum_trajectory.png', dpi=150)
plt.close()
print("Saved: double_pendulum_trajectory.png")


# =============================================================================
# 4. SENSITIVITY TO INITIAL CONDITIONS
# =============================================================================
section("4. SENSITIVITY TO INITIAL CONDITIONS (BUTTERFLY EFFECT)")

# Two nearly identical initial conditions
epsilon = 1e-9
state_a = jnp.array([jnp.pi/2, jnp.pi/2, 0.0, 0.0])
state_b = jnp.array([jnp.pi/2 + epsilon, jnp.pi/2, 0.0, 0.0])

print(f"Initial separation: Δθ₁ = {epsilon:.0e} rad")
print()

# Integrate both
n_steps_chaos = 50_000
dt_chaos = 0.001
traj_a = sys.integrate(state_a, n_steps_chaos, dt_chaos, params)
traj_b = sys.integrate(state_b, n_steps_chaos, dt_chaos, params)

# Compute separation over time
separation = jnp.sqrt(jnp.sum((traj_a - traj_b)**2, axis=1))

# Find when they diverge significantly
threshold = 0.1  # ~6 degrees
diverge_idx = jnp.argmax(separation > threshold)
diverge_time = diverge_idx * dt_chaos

print(f"Trajectories diverge (separation > {threshold}) at t = {diverge_time:.2f}s")
print(f"Final separation: {separation[-1]:.4f}")

# Plot separation (log scale)
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

time = jnp.arange(n_steps_chaos) * dt_chaos

axes[0].semilogy(time, separation, 'b-', linewidth=0.5)
axes[0].axhline(threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold = {threshold}')
axes[0].axvline(diverge_time, color='g', linestyle='--', alpha=0.5, label=f'Divergence at t={diverge_time:.1f}s')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Separation (log scale)')
axes[0].set_title(f'Butterfly Effect: Initial Δθ₁ = {epsilon:.0e} rad')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot both trajectories in physical space
x1_a, y1_a, x2_a, y2_a = polar_to_cartesian(traj_a, params)
x1_b, y1_b, x2_b, y2_b = polar_to_cartesian(traj_b, params)

axes[1].plot(x2_a[::10], y2_a[::10], 'b-', linewidth=0.3, alpha=0.7, label='Trajectory A')
axes[1].plot(x2_b[::10], y2_b[::10], 'r-', linewidth=0.3, alpha=0.7, label='Trajectory B')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Two Trajectories (identical to 9 decimal places)')
axes[1].legend()
axes[1].axis('equal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum_butterfly.png', dpi=150)
plt.close()
print("Saved: double_pendulum_butterfly.png")


# =============================================================================
# 5. LYAPUNOV EXPONENT ESTIMATION
# =============================================================================
section("5. LYAPUNOV EXPONENT")

# Estimate the largest Lyapunov exponent from divergence rate
# λ ≈ (1/t) * ln(separation(t) / separation(0))

# Use early exponential growth phase (before saturation)
early_phase = separation[:int(diverge_idx * 0.8)] if diverge_idx > 100 else separation[:1000]
early_time = jnp.arange(len(early_phase)) * dt_chaos

# Linear fit to log(separation) vs time
log_sep = jnp.log(early_phase + 1e-15)  # avoid log(0)
valid = jnp.isfinite(log_sep)

if jnp.sum(valid) > 10:
    # Simple linear regression
    t_valid = early_time[valid]
    y_valid = log_sep[valid]
    
    n = len(t_valid)
    sum_t = jnp.sum(t_valid)
    sum_y = jnp.sum(y_valid)
    sum_tt = jnp.sum(t_valid * t_valid)
    sum_ty = jnp.sum(t_valid * y_valid)
    
    lyapunov = (n * sum_ty - sum_t * sum_y) / (n * sum_tt - sum_t * sum_t)
    
    print(f"Estimated Lyapunov exponent: λ ≈ {lyapunov:.2f} s⁻¹")
    print(f"Predictability time (1/λ): τ ≈ {1/lyapunov:.2f} s")
    print()
    print("Interpretation:")
    print(f"  After τ ≈ {1/lyapunov:.1f}s, prediction error grows by factor e ≈ 2.7")
    print(f"  After 3τ ≈ {3/lyapunov:.1f}s, error grows by factor ~20")
    print(f"  After 5τ ≈ {5/lyapunov:.1f}s, error grows by factor ~150")
else:
    print("Could not estimate Lyapunov exponent (insufficient data)")


# =============================================================================
# 6. PHASE SPACE PORTRAIT
# =============================================================================
section("6. PHASE SPACE")

# Plot (θ₁, θ̇₁) and (θ₂, θ̇₂) phase portraits
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(traj[:, 0], traj[:, 2], 'b-', linewidth=0.2, alpha=0.5)
axes[0].set_xlabel('θ₁')
axes[0].set_ylabel('θ̇₁')
axes[0].set_title('Phase Space: Mass 1')
axes[0].grid(True, alpha=0.3)

axes[1].plot(traj[:, 1], traj[:, 3], 'r-', linewidth=0.2, alpha=0.5)
axes[1].set_xlabel('θ₂')
axes[1].set_ylabel('θ̇₂')
axes[1].set_title('Phase Space: Mass 2')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum_phase.png', dpi=150)
plt.close()
print("Saved: double_pendulum_phase.png")


# =============================================================================
# 7. POINCARÉ SECTION
# =============================================================================
section("7. POINCARÉ SECTION")

# Poincaré section: sample state when θ₁ crosses zero (going positive)
# This reduces the 4D phase space to 2D

theta1 = traj[:, 0]
theta1_dot = traj[:, 2]
theta2 = traj[:, 1]
theta2_dot = traj[:, 3]

# Find zero crossings of θ₁ (from negative to positive)
crossings = []
for i in range(1, len(theta1)):
    if theta1[i-1] < 0 and theta1[i] >= 0:
        # Linear interpolation for more accurate crossing
        alpha = -theta1[i-1] / (theta1[i] - theta1[i-1])
        t2 = theta2[i-1] + alpha * (theta2[i] - theta2[i-1])
        t2_dot = theta2_dot[i-1] + alpha * (theta2_dot[i] - theta2_dot[i-1])
        t1_dot = theta1_dot[i-1] + alpha * (theta1_dot[i] - theta1_dot[i-1])
        if t1_dot > 0:  # Only count positive crossings
            crossings.append([t2, t2_dot])

crossings = jnp.array(crossings) if crossings else jnp.array([]).reshape(0, 2)
print(f"Found {len(crossings)} Poincaré section points")

if len(crossings) > 10:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(crossings[:, 0], crossings[:, 1], s=1, alpha=0.5)
    ax.set_xlabel('θ₂')
    ax.set_ylabel('θ̇₂')
    ax.set_title('Poincaré Section (θ₁ = 0, θ̇₁ > 0)')
    ax.grid(True, alpha=0.3)
    plt.savefig('double_pendulum_poincare.png', dpi=150)
    plt.close()
    print("Saved: double_pendulum_poincare.png")
else:
    print("Not enough crossings for Poincaré section")


# =============================================================================
# 8. DIFFERENT ENERGY REGIMES
# =============================================================================
section("8. ENERGY REGIMES")

print("The double pendulum has different behavior at different energies:")
print()

# Low energy: oscillatory
state_low = jnp.array([0.3, 0.3, 0.0, 0.0])
# Medium energy: chaotic
state_med = jnp.array([jnp.pi/2, jnp.pi/2, 0.0, 0.0])
# High energy: rotation
state_high = jnp.array([jnp.pi, 0.0, 5.0, 5.0])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (state, label) in zip(axes, [
    (state_low, "Low Energy (oscillatory)"),
    (state_med, "Medium Energy (chaotic)"),
    (state_high, "High Energy (rotational)")
]):
    traj_regime = sys.integrate(state, 30000, 0.001, params)
    x1, y1, x2, y2 = polar_to_cartesian(traj_regime, params)
    
    E = sys.evaluate_energy(state, params)
    
    ax.plot(x2, y2, 'b-', linewidth=0.2, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{label}\nE = {E:.2f}')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_pendulum_regimes.png', dpi=150)
plt.close()
print("Saved: double_pendulum_regimes.png")


# =============================================================================
# 9. WHY PARAMETER LEARNING FAILS
# =============================================================================
section("9. WHY PARAMETER LEARNING FAILS FOR CHAOS")

print("Parameter learning requires: ∂Loss/∂param to be informative")
print()
print("In chaotic systems:")
print("  1. Tiny parameter changes → completely different trajectories")
print("  2. Gradient ∂traj/∂param is enormous and points in 'random' direction")
print("  3. Loss landscape is fractal — no smooth path to minimum")
print()
print("Demonstration: Loss vs gravity g")

# Compute loss for different g values
g_values = jnp.linspace(9.0, 10.5, 200)
losses = []

# Short trajectory to avoid complete divergence
traj_true = sys.integrate(state_med, 5000, 0.01, params)

for g in g_values:
    params_test = {**params, "g": float(g)}
    traj_test = sys.integrate(state_med, 5000, 0.01, params_test)
    loss = float(jnp.mean((traj_test - traj_true)**2))
    losses.append(loss)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(g_values, losses, 'b-', linewidth=1)
ax.axvline(params["g"], color='r', linestyle='--', label=f'True g = {params["g"]}')
ax.set_xlabel('g')
ax.set_ylabel('Trajectory Loss')
ax.set_title('Loss Landscape for Double Pendulum (chaotic = non-smooth)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('double_pendulum_loss_landscape.png', dpi=150)
plt.close()
print("Saved: double_pendulum_loss_landscape.png")

print()
print("The loss landscape is NOT smooth — gradient descent cannot find the minimum.")
print("This is a fundamental limitation of chaotic systems, not our framework.")
