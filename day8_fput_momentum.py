from contravariant.catalog import fput_chain
import jax.numpy as jnp
from sympy import diff, simplify


N = 16

# Periodic BC — should conserve momentum
sys_periodic = fput_chain(N, beta=True, boundary="periodic")

# Fixed BC — should NOT conserve momentum
sys_fixed = fput_chain(N, beta=True, boundary="fixed")

params = {"m": 1.0, "k": 1.0, "beta": 2.0}

# Initial condition: lowest mode with some net momentum
q0 = jnp.sin(jnp.pi * jnp.arange(1, N + 1) / (N + 1))
v0 = jnp.ones(N) * 0.5  # Net momentum = m * N * 0.5
state_0 = jnp.concatenate([q0, v0])

dt = 0.01
n_steps = 100000

# Integrate both
traj_periodic = sys_periodic.integrate(state_0, n_steps, dt, params)
traj_fixed = sys_fixed.integrate(state_0, n_steps, dt, params)


# Total momentum: P = m * sum(v)
def total_momentum(traj, m):
    v = traj[:, N:]
    return m * jnp.sum(v, axis=1)


P_periodic = total_momentum(traj_periodic, params["m"])
P_fixed = total_momentum(traj_fixed, params["m"])

print("Periodic BC:")
print(f"  P(0) = {P_periodic[0]:.6f}")
print(f"  P(T) = {P_periodic[-1]:.6f}")
print(f"  max |ΔP| = {jnp.max(jnp.abs(P_periodic - P_periodic[0])):.2e}")

print("\nFixed BC:")
print(f"  P(0) = {P_fixed[0]:.6f}")
print(f"  P(T) = {P_fixed[-1]:.6f}")
print(f"  max |ΔP| = {jnp.max(jnp.abs(P_fixed - P_fixed[0])):.2e}")

xi = [1] * N

if sys_periodic.check_symmetry(xi) == 0:
    P = sys_periodic.conserved_quantity(xi)
    print(f"Periodic BC symmetry confirmed. Conserved: {P}")
else:
    print(f"Periodic BC is not a symmetry: δL = {sys_periodic.check_symmetry(xi)}")


print()
if sys_fixed.check_symmetry(xi) == 0:
    P = sys_fixed.conserved_quantity(xi)
    print(f"Fixed BC symmetry confirmed. Conserved: {P}")
else:
    print(f"Fixed BC is not a symmetry: δL = {sys_fixed.check_symmetry(xi)}")
