"""
Day 6: Comprehensive API Test — 2D Isotropic Oscillator
"""

from sympy import symbols, Rational
import jax.numpy as jnp
from contravariant.systems import LagrangianSystem

# Define system
x, x_dot, y, y_dot = symbols("x x_dot y y_dot")
m, k = symbols("m k", positive=True)
L = Rational(1, 2) * m * (x_dot**2 + y_dot**2) - Rational(1, 2) * k * (x**2 + y**2)

sys = LagrangianSystem(L, [x, y], [x_dot, y_dot])
print(sys)
print()

# Conserved quantity from rotation symmetry
L_z = sys.conserved_quantity([-y, x])
print(f"Angular momentum: L_z = {L_z}")
print()

# Compare integrators
params = {"m": 1.0, "k": 1.0}
state_0 = jnp.array([1.0, 0.0, 0.0, 1.0])

sys.compare_integrators(
    state_0,
    n_steps=100000,
    dt=0.01,
    params=params,
    quantities={"L_z": L_z},
    save_as="oscillator",
)

# Learn parameters
params_true = {"m": 1.0, "k": 2.0}
traj_observed = sys.integrate(state_0, 1000, 0.01, params_true)

result = sys.learn_parameters(
    traj_observed=traj_observed,
    state_0=state_0,
    n_steps=1000,
    dt=0.01,
    params_fixed={"m": 1.0},
    params_init={"k": 0.5},
    loss_type="energy_statistic",
    learning_rate=0.05,
    max_iterations=100,
)

print()
print(f"True k = {params_true['k']}, Learned k = {result['k']:.4f}")
