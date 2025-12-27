"""
Day 8: FPUT Recurrence
"""

import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
from contravariant.catalog import fput_chain


def normal_mode_shape(N, k):
    """k-th normal mode shape (k=1 is lowest)."""
    j = jnp.arange(1, N + 1)
    shape = jnp.sin(jnp.pi * k * j / (N + 1))
    return shape / jnp.linalg.norm(shape)


def normal_mode_freq(N, k):
    """k-th normal mode frequency (units of sqrt(k/m))."""
    return 2 * jnp.sin(jnp.pi * k / (2 * (N + 1)))


def project_onto_modes(traj, N):
    """Project trajectory onto normal modes."""
    q = traj[:, :N]
    v = traj[:, N:]

    modes = jnp.stack([normal_mode_shape(N, k) for k in range(1, N + 1)])
    freqs = jnp.array([normal_mode_freq(N, k) for k in range(1, N + 1)])

    amps = jnp.einsum("kj,tj->tk", modes, q)
    vels = jnp.einsum("kj,tj->tk", modes, v)
    energies = 0.5 * (vels**2 + (freqs**2) * (amps**2))

    return amps, energies


# =============================================================================
# Main experiment
# =============================================================================

N = 32
t0 = time.time()
sys = fput_chain(N, beta=True)
t_symbolic = time.time() - t0
print(sys)
print(f"Separable: {sys.is_separable}")

params = {"m": 1.0, "k": 1.0, "beta": 6}

# Initialize in lowest mode
amplitude = 1.5
q0 = amplitude * normal_mode_shape(N, 1)
v0 = jnp.zeros(N)
state_0 = jnp.concatenate([q0, v0])

# Integrate
dt = 0.1
n_steps = 1_000_000
print(f"\nIntegrating for T = {n_steps * dt}...")
t0 = time.time()
traj = sys.integrate(state_0, n_steps, dt, params)
traj.block_until_ready()
t_first = time.time() - t0

# Check energy conservation
E = sys.evaluate_energy_along_trajectory(traj, params)
print(
    f"Energy conservation: max |ΔE/E₀| = {float(jnp.max(jnp.abs(E - E[0])) / E[0]):.2e}"
)

# Project onto modes
amps, mode_E = project_onto_modes(traj, N)
E_total = jnp.sum(mode_E, axis=1)
mode_frac = mode_E / E_total[:, None]

# Integrator comparison
t0 = time.time()
traj_yoshida = sys.integrate(state_0, n_steps, dt, params, method="yoshida")
traj_yoshida.block_until_ready()
t_second = time.time() - t0
traj_rk4 = sys.integrate(state_0, n_steps, dt, params, method="rk4")

E_yoshida = sys.evaluate_energy_along_trajectory(traj_yoshida, params)
E_rk4 = sys.evaluate_energy_along_trajectory(traj_rk4, params)

# Print timings for different stages
print(f"N={N}")
print(f"  Symbolic:  {t_symbolic:.2f}s")
print(f"  First run: {t_first:.2f}s (includes JIT)")
print(f"  Second run: {t_second:.2f}s (pure numerics)")

# Plot
times = jnp.arange(n_steps) * dt
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
for k in range(5):
    ax.plot(times, mode_frac[:, k] * 100, label=f"Mode {k+1}")
ax.set_xlabel("Time")
ax.set_ylabel("Energy fraction (%)")
ax.set_title("FPUT: Energy in Normal Modes")
ax.legend()
ax.set_ylim(0, 105)

ax = axes[1]
ax.plot(times, mode_frac[:, 0] * 100)
ax.set_xlabel("Time")
ax.set_ylabel("Mode 1 energy (%)")
ax.set_title("FPUT Recurrence")

plt.tight_layout()
plt.savefig("images/fput_recurrence.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    times, (E_yoshida - E_yoshida[0]) / E_yoshida[0], label="Yoshida", linewidth=0.5
)
ax.plot(times, (E_rk4 - E_rk4[0]) / E_rk4[0], label="RK4", linewidth=0.5)
ax.set_xlabel("Time")
ax.set_ylabel("Relative energy error")
ax.legend()
ax.set_title("Yoshida vs RK4")
plt.savefig("images/fput_yoshida_vs_rk4.png", dpi=150)
plt.show()
