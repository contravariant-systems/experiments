import jax.numpy as jnp
from jax.numpy import sin
from contravariant.integrators import make_rk4_integrator
import matplotlib.pyplot as plt


def thomas_dynamics(state, params):
    x, y, z = state
    b = params["b"]
    return jnp.array(
        [
            sin(y) - b * x,
            sin(z) - b * y,
            sin(x) - b * z,
        ]
    )


integrator = make_rk4_integrator(thomas_dynamics)

state_0 = jnp.array([0.1, 0.0, 0.0])
params = {"b": 0.208186}
traj = integrator(state_0, 100000, 0.01, params)

# 3D phase space
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.3, alpha=0.8)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Thomas Attractor")
plt.savefig("thomas_attractor.png", dpi=200)
plt.show()
