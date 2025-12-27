"""
FPUT Animation: Chain motion + Mode energy evolution
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from contravariant.catalog import fput_chain


def normal_mode_shape(N, k):
    j = jnp.arange(1, N + 1)
    shape = jnp.sin(jnp.pi * k * j / (N + 1))
    return shape / jnp.linalg.norm(shape)


def normal_mode_freq(N, k):
    return 2 * jnp.sin(jnp.pi * k / (2 * (N + 1)))


def project_onto_modes(traj, N):
    q = traj[:, :N]
    v = traj[:, N:]
    modes = jnp.stack([normal_mode_shape(N, k) for k in range(1, N + 1)])
    freqs = jnp.array([normal_mode_freq(N, k) for k in range(1, N + 1)])
    amps = jnp.einsum("kj,tj->tk", modes, q)
    vels = jnp.einsum("kj,tj->tk", modes, v)
    energies = 0.5 * (vels**2 + (freqs**2) * (amps**2))
    return amps, energies


# =============================================================================
# Generate trajectory
# =============================================================================

N = 32
sys = fput_chain(N, beta=True)
params = {"m": 1.0, "k": 1.0, "beta": 8.0}

amplitude = 2.0
q0 = amplitude * np.array(normal_mode_shape(N, 1))
v0 = np.zeros(N)
state_0 = jnp.concatenate([jnp.array(q0), jnp.array(v0)])

dt = 0.1
n_steps = 100000
print("Integrating...")
traj = sys.integrate(state_0, n_steps, dt, params)
traj.block_until_ready()

# Convert to numpy for matplotlib
traj_np = np.array(traj)

# Project onto modes
_, mode_energies = project_onto_modes(traj, N)
mode_energies_np = np.array(mode_energies)
E_total = mode_energies_np.sum(axis=1, keepdims=True)
mode_frac = mode_energies_np / E_total

print("Setting up animation...")

# =============================================================================
# Animation setup
# =============================================================================

# Subsample for smooth animation (every 100th frame)
skip = 100
frames = range(0, n_steps, skip)
n_frames = len(frames)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1]})

# --- Top panel: Physical chain ---
ax1 = axes[0]
ax1.set_xlim(-1, N + 2)
ax1.set_ylim(-3, 3)
ax1.set_xlabel("Mass index")
ax1.set_ylabel("Displacement")
ax1.set_title("FPUT Chain Motion")
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

# Wall markers
ax1.plot([0, 0], [-0.5, 0.5], "k-", linewidth=4)
ax1.plot([N + 1, N + 1], [-0.5, 0.5], "k-", linewidth=4)

# Initialize chain visualization
x_positions = np.arange(1, N + 1)
(masses,) = ax1.plot([], [], "o", markersize=8, color="steelblue", zorder=3)
(springs,) = ax1.plot([], [], "-", color="gray", linewidth=1, zorder=1)
time_text = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, fontsize=12)

# --- Bottom panel: Mode energies ---
ax2 = axes[1]
n_modes_show = 8  # Only show first 8 modes
mode_indices = np.arange(1, n_modes_show + 1)
bars = ax2.bar(mode_indices, np.zeros(n_modes_show), color="steelblue", alpha=0.8)
ax2.set_xlim(0.5, n_modes_show + 0.5)
ax2.set_ylim(0, 105)
ax2.set_xlabel("Mode number")
ax2.set_ylabel("Energy fraction (%)")
ax2.set_title("Energy Distribution Across Modes")
ax2.set_xticks(mode_indices)

# Mode 1 indicator line
ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.3)


def init():
    """Initialize animation artists."""
    masses.set_data([], [])
    springs.set_data([], [])
    time_text.set_text("")
    for bar in bars:
        bar.set_height(0)
    return [masses, springs, time_text] + list(bars)


def update(frame_idx):
    """Update animation for frame i."""
    i = frames[frame_idx]
    t = i * dt

    # Get positions at this timestep
    q = traj_np[i, :N]

    # Update masses (x = index, y = displacement)
    masses.set_data(x_positions, q)

    # Update springs (connect wall - masses - wall)
    spring_x = [0] + list(x_positions) + [N + 1]
    spring_y = [0] + list(q) + [0]
    springs.set_data(spring_x, spring_y)

    # Update time display
    time_text.set_text(f"t = {t:.0f}")

    # Update mode energy bars
    for j, bar in enumerate(bars):
        bar.set_height(mode_frac[i, j] * 100)

    # Color mode 1 based on its energy (visual emphasis)
    bars[0].set_color("crimson" if mode_frac[i, 0] < 0.5 else "steelblue")

    return [masses, springs, time_text] + list(bars)


# Create animation
anim = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    init_func=init,
    blit=True,
    interval=30,  # milliseconds between frames
)

plt.tight_layout()

# Save as MP4 (requires ffmpeg) or GIF
print("Saving animation...")
anim.save("images/fput_animation.mp4", writer="ffmpeg", fps=30, dpi=250)
# Or for GIF: anim.save('images/fput_animation.gif', writer='pillow', fps=30, dpi=100)

print("Done! Saved to images/fput_animation.mp4")

# Or just show interactive:
plt.show()
