import jax.numpy as jnp
from contravariant.catalog import harmonic_oscillator_2d

# One line to get a fully-analysed system
sys = harmonic_oscillator_2d()
print(sys)
print()

# Rotation symmetry → angular momentum
x, y = sys.coordinates
L_z = sys.conserved_quantity([-y, x])
print(f"Angular momentum: L_z = {L_z}")
print()

# Compare integrators
params = {'m': 1.0, 'k': 1.0}
state_0 = jnp.array([1.0, 0.0, 0.0, 1.0])

sys.compare_integrators(
    state_0, n_steps=100000, dt=0.01, params=params,
    quantities={'L_z': L_z},
    save_as='2d_oscillator'
)

# Learn parameters
params_true = {'m': 1.0, 'k': 2.0}
traj_observed = sys.integrate(state_0, 1000, 0.01, params_true)

result = sys.learn_parameters(
    traj_observed, state_0, n_steps=1000, dt=0.01,
    params_fixed={'m': 1.0},
    params_init={'k': 0.5},
)

print(f"True k = {params_true['k']}, Learned k = {result['k']:.4f}")
