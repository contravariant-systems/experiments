"""
Catalog of classical mechanical systems.

A growing collection of well-known systems, ready to use.
"""

from .oscillators import (
    harmonic_oscillator,
    harmonic_oscillator_2d,
    coupled_oscillators,
    anharmonic_oscillator,
)

from .pendulums import (
    simple_pendulum,
    double_pendulum,
    spherical_pendulum,
)

from .particles import (
    free_particle_2d,
    central_force,
    kepler,
)

from .fput import (
    fput_chain,
)

__all__ = [
    # Oscillators
    "harmonic_oscillator",
    "harmonic_oscillator_2d",
    "coupled_oscillators",
    "anharmonic_oscillator",
    # Pendulums
    "simple_pendulum",
    "double_pendulum",
    "spherical_pendulum",
    # Particles
    "free_particle_2d",
    "central_force",
    "kepler",
    # FPUT,
    "fput_chain",
]
