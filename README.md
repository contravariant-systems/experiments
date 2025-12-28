# Contravariant

Most computational physics discards the structure that makes equations
meaningful—conservation laws disappear into arrays and loops.
*Contravariant* takes the opposite path: specify variational
principles symbolically, and the framework derives, analyzes, and
compiles them to differentiable code. What's known is enforced
exactly; what's unknown can be learned.

This is v0.1—Lagrangian mechanics. The architecture extends to
Hamiltonians, continua, and fields.

If you find this interesting, you should [follow me on
Mastodon](https://hachyderm.io/@harish) to learn about the other
things I do.

## Installation

````
pip install contravariant
````

Requires Python ≥3.10, JAX, SymPy, and Optax.

## Usage

With Contravariant you can go from a Lagrangian to simulation in a few lines:

````python
from sympy import symbols, cos, Rational
import jax.numpy as jnp
from contravariant import LagrangianSystem, plot_phase_space, plot_energy_error

# Define a pendulum
theta, theta_dot = symbols("theta theta_dot")
m, l, g = symbols("m l g", positive=True)
L = Rational(1, 2) * m * l**2 * theta_dot**2 - m * g * l * (1 - cos(theta))

# Create the system (everything is derived automatically)
pendulum = LagrangianSystem(L, [theta], [theta_dot])

# Simulate
params = {"m": 1.0, "l": 1.0, "g": 9.8}
traj = pendulum.integrate(
    jnp.array([0.5, 0.0]), n_steps=10000, dt=0.01, params=params
)

# Plot phase space
plot_phase_space(traj, title="Pendulum")

# Verify energy conservation
plot_energy_error(traj, pendulum.evaluate_energy, params)
````

The framework automatically:

- Derives Euler-Lagrange equations of motion
- Computes the Hamiltonian via Legendre transform
- Detects cyclic coordinates and their conserved momenta
- Chooses symplectic integrators when applicable

### Catalog

Common systems are pre-built:
````python
from contravariant.catalog import (
    harmonic_oscillator, coupled_oscillators, anharmonic_oscillator,
    simple_pendulum, double_pendulum, spherical_pendulum,
    free_particle, central_force, kepler,
    fput_chain,
)

sys = double_pendulum()
````

### Composition

Compose simple systems into more complex ones:
````python
sys_2d = harmonic_oscillator(coord='x') + harmonic_oscillator(coord='y')
````

### Parameter learning

Infer parameters from observed trajectories:
````python
learned = sys.learn_parameters(
    traj_observed, state_0, n_steps, dt,
    params_fixed={'m': 1.0},
    params_init={'k': 0.5}
)
````

### Noether's theorem

Check symmetries and compute conserved quantities:
````python
xi = [-y, x]  # Rotation generator
delta_L = sys.check_symmetry(xi)  # 0 if symmetric
Lz = sys.conserved_quantity(xi)   # Angular momentum
````

## Known limitations of v0.1

- Symplectic integrators assume diagonal mass matrices
- Symbolic derivation may be slow for N > 5 degrees of freedom
- Time-dependent Lagrangians are not yet supported

## Roadmap

- v0.1 — Lagrangian Mechanics ✓

  Automatic conservation laws from variational principles. Symbolic
  Lagrangian → Euler-Lagrange equations → symplectic integration.
  Catalog of classical systems. Parameter learning from trajectories.

- v0.2 — Hamiltonian Mechanics

  Poisson brackets, canonical transformations, generating functions.
  Phase space visualization: Poincaré sections, KAM tori, homoclinic
  tangles. Action-angle variables for integrable systems.

  *"SICM in Python"*

- v0.3 — Continuum Mechanics

  Elastic rods, membranes, nonlinear shells. Lagrangian density
  formulation. FEM/FVM discretization that preserves variational
  structure. Structure-preserving spatial discretization.

- v0.4 — Multiphysics

  Coupled conservation laws: mass, momentum, energy. Non-equilibrium
  thermodynamics and dissipation. Constitutive learning: infer
  material laws from data while enforcing thermodynamic constraints.

- v∞ — Field Theory

  Relativistic mechanics, gauge symmetries, Noether currents for
  fields. The full vertical stack from symbolic principles to compiled
  differentiable numerics—across all physics.

## Authors and contributing

Contravariant is primarily written and maintained by [Harish
Narayanan](https://harishnarayanan.org).

If you're interested in contributing, please consider addressing some
of the [issues people have previously
reported](https://github.com/hnarayanan/contravariant/issues) and
[submitting a pull
request](https://help.github.com/articles/using-pull-requests/). Thank you!


## Copyright and license

Copyright (c) 2025 [Harish Narayanan](https://harishnarayanan.org).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
