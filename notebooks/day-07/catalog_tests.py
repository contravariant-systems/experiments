import jax.numpy as jnp
import matplotlib.pyplot as plt
from contravariant.catalog import anharmonic_oscillator

sys = anharmonic_oscillator()
params = {"m": 1.0, "k": 1.0, "lambda": 0.1}
state_0 = jnp.array([1.0, 0.0])

n_steps = 1_000_000
dt = 0.01

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for method in ["rk4", "yoshida"]:
    traj = sys.integrate(state_0, n_steps, dt, params, method=method)
    energy = sys.evaluate_energy_along_trajectory(traj, params)
    error = energy - energy[0]

    # Plot every 1000th point
    axes[0].plot(error[::1000], label=method, alpha=0.7)

axes[0].set_ylabel("Energy Error")
axes[0].legend()
axes[0].set_title("Energy Error Over Time (anharmonic oscillator)")

# Zoom into last 10% to see if one is drifting
start = int(0.9 * n_steps) // 1000
for method in ["rk4", "yoshida"]:
    traj = sys.integrate(state_0, n_steps, dt, params, method=method)
    energy = sys.evaluate_energy_along_trajectory(traj, params)
    error = energy - energy[0]
    axes[1].plot(error[start * 1000 :: 1000], label=method, alpha=0.7)

axes[1].set_ylabel("Energy Error (last 10%)")
axes[1].set_xlabel("Time step (x1000)")
axes[1].legend()

plt.tight_layout()
plt.savefig("energy_drift_comparison.png", dpi=150)
plt.show()
