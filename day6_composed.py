from contravariant.catalog import harmonic_oscillator, harmonic_oscillator_2d
import jax.numpy as jnp

# Compose 2D isotropic oscillator
sys_x = harmonic_oscillator(coord="x")
sys_y = harmonic_oscillator(coord="y")
sys_2d = sys_x + sys_y

print("Composed system:")
print(sys_2d)
print()

# Verify it works
params = {"m": 1.0, "k": 1.0}
state_0 = jnp.array([1.0, 0.0, 0.0, 1.0])
traj = sys_2d.integrate(state_0, 1000, 0.01, params)
print(f"Final state: {traj[-1]}")
print()

# Standard 2D isotropic oscillator
sys_2d_manual = harmonic_oscillator_2d()

print("(Manual) 2D system:")
print(sys_2d_manual)
print()

# Verify it works
params = {"m": 1.0, "k": 1.0}
state_0 = jnp.array([1.0, 0.0, 0.0, 1.0])
traj = sys_2d_manual.integrate(state_0, 1000, 0.01, params)
print(f"Final state: {traj[-1]}")
print()

# Anisotropic version
sys_aniso = harmonic_oscillator(coord="x", spring="k_x") + harmonic_oscillator(
    coord="y", spring="k_y"
)
print("Anisotropic system:")
print(sys_aniso)
