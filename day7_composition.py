import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from sympy import symbols, Rational
from contravariant.catalog import harmonic_oscillator, simple_pendulum


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# 1. NON-INTERACTING COMPOSITION
# =============================================================================
section("1. NON-INTERACTING COMPOSITION")

sys_x = harmonic_oscillator(coord="x")
sys_y = harmonic_oscillator(coord="y")
sys_2d = sys_x + sys_y

print("sys_x + sys_y:")
print(f"  L = {sys_2d.L}")
print(f"  DOF: {sys_2d.n_dof}, Separable: {sys_2d.is_separable}")

params = {"m": 1.0, "k": 1.0}
state_0 = jnp.array([1.0, 0.5, 0.0, 0.3])
traj = sys_2d.integrate(state_0, 5000, 0.01, params)
cons = sys_2d.check_conservation(traj, params)
print(f"  Energy error: {cons['energy'][0]:.2e}")


# =============================================================================
# 2. ADDING COUPLING WITH __sub__
# =============================================================================
section("2. COUPLING VIA SUBTRACTION")

x, y = symbols("x y")
k_c = symbols("k_c", positive=True)
V_coupling = Rational(1, 2) * k_c * (x - y) ** 2

# The clean API: compose then subtract potential
sys_coupled = sys_x + sys_y - V_coupling

print("sys_x + sys_y - V_coupling:")
print(f"  L = {sys_coupled.L}")
print()

params_coupled = {"m": 1.0, "k": 1.0, "k_c": 0.5}
state_0 = jnp.array([1.0, 0.0, 0.0, 0.0])
traj = sys_coupled.integrate(state_0, 5000, 0.01, params_coupled)
cons = sys_coupled.check_conservation(traj, params_coupled)
print(f"Energy error: {cons['energy'][0]:.2e}")
print(f"Energy transfer: x=1,y=0 → x={traj[-1, 0]:.2f}, y={traj[-1, 1]:.2f}")


# =============================================================================
# 3. COUPLED PENDULUMS
# =============================================================================
section("3. COUPLED PENDULUMS")

theta1, theta2 = symbols("theta1 theta2")
l, k_c = symbols("l k_c", positive=True)
V_spring = Rational(1, 2) * k_c * l**2 * (theta1 - theta2) ** 2

sys_pend = simple_pendulum(coord="theta1") + simple_pendulum(coord="theta2") - V_spring

print(f"L = {sys_pend.L}")
print()

params = {"m": 1.0, "l": 1.0, "g": 9.81, "k_c": 2.0}
state_0 = jnp.array([0.3, -0.3, 0.0, 0.0])
traj = sys_pend.integrate(state_0, 10000, 0.001, params)
cons = sys_pend.check_conservation(traj, params)
print(f"Energy error: {cons['energy'][0]:.2e}")


# =============================================================================
# 4. OSCILLATOR CHAIN (3 masses)
# =============================================================================
section("4. OSCILLATOR CHAIN")

sys_1 = harmonic_oscillator(coord="q1")
sys_2 = harmonic_oscillator(coord="q2")
sys_3 = harmonic_oscillator(coord="q3")

q1, q2, q3 = symbols("q1 q2 q3")
k_c = symbols("k_c", positive=True)
V_neighbors = Rational(1, 2) * k_c * ((q2 - q1) ** 2 + (q3 - q2) ** 2)

sys_chain = sys_1 + sys_2 + sys_3 - V_neighbors

print(f"L = {sys_chain.L}")
print(f"DOF: {sys_chain.n_dof}, Separable: {sys_chain.is_separable}")
print()

params = {"m": 1.0, "k": 1.0, "k_c": 0.5}
state_0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
traj = sys_chain.integrate(state_0, 10000, 0.01, params)
cons = sys_chain.check_conservation(traj, params)
print(f"Energy error: {cons['energy'][0]:.2e}")

# Plot
fig, ax = plt.subplots(figsize=(12, 4))
t = jnp.arange(len(traj)) * 0.01
ax.plot(t, traj[:, 0], label="q₁", alpha=0.8)
ax.plot(t, traj[:, 1], label="q₂", alpha=0.8)
ax.plot(t, traj[:, 2], label="q₃", alpha=0.8)
ax.set_xlabel("Time")
ax.set_ylabel("Position")
ax.set_title("Energy Flow Through 3-Mass Chain")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("oscillator_chain.png", dpi=150)
plt.close()
print("Saved: oscillator_chain.png")


# =============================================================================
# 5. NORMAL MODE VERIFICATION
# =============================================================================
section("5. NORMAL MODE VERIFICATION")

# Theory: ω₊ = √(k/m), ω₋ = √((k+2k_c)/m)
k_val, m_val, k_c_val = 1.0, 1.0, 0.5
T_plus = 2 * np.pi / np.sqrt(k_val / m_val)
T_minus = 2 * np.pi / np.sqrt((k_val + 2 * k_c_val) / m_val)

print(f"Theory:  T₊ = {T_plus:.4f}s, T₋ = {T_minus:.4f}s")

# Measure from simulation
sys = sys_x + sys_y - V_coupling
params = {"m": 1.0, "k": 1.0, "k_c": 0.5}


def measure_period(initial_state):
    traj = sys.integrate(initial_state, 20000, 0.001, params)
    x = traj[:, 0]
    crossings = [i * 0.001 for i in range(1, len(x)) if x[i - 1] < 0 and x[i] >= 0]
    return crossings[1] - crossings[0] if len(crossings) >= 2 else None


T_plus_m = measure_period(jnp.array([1.0, 1.0, 0.0, 0.0]))
T_minus_m = measure_period(jnp.array([1.0, -1.0, 0.0, 0.0]))

print(f"Measured: T₊ = {T_plus_m:.4f}s, T₋ = {T_minus_m:.4f}s")
print(
    f"Error: {abs(T_plus - T_plus_m)/T_plus*100:.2f}%, {abs(T_minus - T_minus_m)/T_minus*100:.2f}%"
)
