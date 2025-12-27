"""
Contravariant: Automatic Conservation Laws from Variational Principles

A framework for executable, differentiable physics.

Symbolic specification → Structural analysis → Compiled numerics
"""

from .systems import LagrangianSystem

from .symbolic import (
    derive_equations_of_motion,
    extract_kinetic_potential,
    find_cyclic_coordinates,
    derive_hamiltonian,
    derive_conserved_quantity,
)

from .codegen import (
    compile_lagrangian_dynamics,
    compile_hamiltonian_dynamics,
    compile_grad_V,
    compile_energy,
    compile_expression,
)

from .integrators import (
    make_rk4_integrator,
    make_verlet_integrator,
    make_yoshida_integrator,
)

from .learning import (
    learn_parameters,
    trajectory_loss,
    energy_statistic_loss,
)

from .plotting import (
    plot_trajectory,
    plot_positions,
    plot_phase_space,
    plot_energy_evolution,
    plot_energy_error,
    compare_energy_errors,
    plot_phase_space_multi,
)

__version__ = "0.1.0"
