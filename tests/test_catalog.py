import pytest
import jax.numpy as jnp

from contravariant.catalog import (
    # Oscillators
    harmonic_oscillator,
    harmonic_oscillator_2d,
    coupled_oscillators,
    anharmonic_oscillator,
    # Pendulums
    simple_pendulum,
    double_pendulum,
    spherical_pendulum,
    # Particles
    free_particle,
    free_particle_2d,
    central_force,
    kepler,
    # FPUT
    fput_chain,
)


class TestCatalogCoverage:
    """Verify that all catalog systems can be created and integrated."""

    def test_harmonic_oscillator(self):
        sys = harmonic_oscillator()
        assert sys.dof == 1
        assert sys.is_separable == True

        traj = sys.integrate(jnp.array([1.0, 0.0]), 100, 0.01, {"m": 1.0, "k": 1.0})
        assert traj.shape == (100, 2)

    def test_harmonic_oscillator_2d(self):
        sys = harmonic_oscillator_2d()
        assert sys.dof == 2
        assert sys.is_separable == True

        traj = sys.integrate(
            jnp.array([1.0, 0.0, 0.0, 1.0]), 100, 0.01, {"m": 1.0, "k": 1.0}
        )
        assert traj.shape == (100, 4)

    def test_coupled_oscillators(self):
        sys = coupled_oscillators()
        assert sys.dof == 2
        assert sys.is_separable == True

        traj = sys.integrate(
            jnp.array([1.0, 0.0, 0.0, 0.0]), 100, 0.01, {"m": 1.0, "k": 1.0, "k_c": 0.5}
        )
        assert traj.shape == (100, 4)

    def test_anharmonic_oscillator(self):
        sys = anharmonic_oscillator()
        assert sys.dof == 1
        assert sys.is_separable == True

        traj = sys.integrate(
            jnp.array([1.0, 0.0]), 100, 0.01, {"m": 1.0, "k": 1.0, "lambda": 0.1}
        )
        assert traj.shape == (100, 2)

    def test_simple_pendulum(self):
        sys = simple_pendulum()
        assert sys.dof == 1
        assert sys.is_separable == True

        traj = sys.integrate(
            jnp.array([0.5, 0.0]), 100, 0.01, {"m": 1.0, "l": 1.0, "g": 9.81}
        )
        assert traj.shape == (100, 2)

    def test_double_pendulum(self):
        sys = double_pendulum()
        assert sys.dof == 2
        assert sys.is_separable == False  # Non-separable!

        traj = sys.integrate(
            jnp.array([0.5, 0.5, 0.0, 0.0]),
            100,
            0.01,
            {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81},
        )
        assert traj.shape == (100, 4)

    def test_spherical_pendulum(self):
        sys = spherical_pendulum()
        assert sys.dof == 2
        assert sys.is_separable == False  # Non-separable!
        assert len(sys.cyclic_coordinates) == 1  # phi is cyclic

        traj = sys.integrate(
            jnp.array([0.5, 0.0, 0.0, 2.0]),
            100,
            0.01,
            {"m": 1.0, "l": 1.0, "g": 9.81},
        )
        assert traj.shape == (100, 4)

    def test_free_particle(self):
        sys = free_particle()
        assert sys.dof == 1
        assert sys.is_separable == True
        assert len(sys.cyclic_coordinates) == 1  # q is cyclic

        traj = sys.integrate(jnp.array([0.0, 1.0]), 100, 0.01, {"m": 1.0})
        assert traj.shape == (100, 2)

    def test_free_particle_2d(self):
        sys = free_particle_2d()
        assert sys.dof == 2
        assert sys.is_separable == True
        assert len(sys.cyclic_coordinates) == 2  # Both cyclic

        traj = sys.integrate(jnp.array([0.0, 0.0, 1.0, 0.5]), 100, 0.01, {"m": 1.0})
        assert traj.shape == (100, 4)

    def test_central_force(self):
        sys = central_force()
        assert sys.dof == 2
        # NOT separable: T = ½m(ṙ² + r²θ̇²) depends on r
        assert sys.is_separable == False
        assert len(sys.cyclic_coordinates) == 1  # theta is cyclic

        traj = sys.integrate(
            jnp.array([1.0, 0.0, 0.0, 1.0]), 100, 0.001, {"m": 1.0, "k": 1.0}
        )
        assert traj.shape == (100, 4)

    def test_kepler(self):
        sys = kepler()
        assert sys.dof == 2
        # NOT separable: T = ½m(ṙ² + r²θ̇²) depends on r
        assert sys.is_separable == False
        assert len(sys.cyclic_coordinates) == 1  # theta is cyclic

        traj = sys.integrate(
            jnp.array([1.0, 0.0, 0.0, 1.0]), 100, 0.001, {"m": 1.0, "k": 1.0}
        )
        assert traj.shape == (100, 4)

    def test_fput_chain_alpha(self):
        sys = fput_chain(N=4, alpha=0.25)
        assert sys.dof == 4
        assert sys.is_separable == True

        state_0 = jnp.zeros(8)
        state_0 = state_0.at[0].set(1.0)
        traj = sys.integrate(state_0, 100, 0.01, {"m": 1.0, "k": 1.0, "alpha": 0.25})
        assert traj.shape == (100, 8)

    def test_fput_chain_beta(self):
        sys = fput_chain(N=4, beta=0.25)
        assert sys.dof == 4
        assert sys.is_separable == True

        state_0 = jnp.zeros(8)
        state_0 = state_0.at[0].set(1.0)
        traj = sys.integrate(state_0, 100, 0.01, {"m": 1.0, "k": 1.0, "beta": 0.25})
        assert traj.shape == (100, 8)


class TestSymbolCustomization:
    """Test that catalog functions accept custom symbol names."""

    def test_harmonic_oscillator_custom_symbols(self):
        sys = harmonic_oscillator(coord="x", mass="M", spring="K")
        assert sys.is_separable == True
        # Symplectic works because mass extracted from ∂²T/∂q̇²
        traj = sys.integrate(
            jnp.array([1.0, 0.0]), 100, 0.01, {"M": 1.0, "K": 2.0}, method="yoshida"
        )
        assert traj.shape == (100, 2)

    def test_simple_pendulum_custom_symbols(self):
        """simple_pendulum should accept custom symbol names."""
        sys = simple_pendulum(coord="phi", mass="M", length="L", gravity="G")

        coord_names = [str(c) for c in sys.coordinates]
        param_names = [str(p) for p in sys.parameters]

        assert "phi" in coord_names
        assert "M" in param_names
        assert "L" in param_names
        assert "G" in param_names

        # Symplectic works: mass coefficient M*L² extracted from ∂²T/∂φ̇²
        traj = sys.integrate(
            jnp.array([0.5, 0.0]),
            100,
            0.01,
            {"M": 1.0, "L": 1.0, "G": 9.81},
            method="yoshida",
        )
        assert traj.shape == (100, 2)

    def test_double_pendulum_custom_symbols(self):
        """double_pendulum should accept custom symbol names."""
        sys = double_pendulum(
            coords=("a", "b"),
            masses=("M1", "M2"),
            lengths=("L1", "L2"),
            gravity="G",
        )

        coord_names = [str(c) for c in sys.coordinates]
        param_names = [str(p) for p in sys.parameters]

        assert "a" in coord_names
        assert "b" in coord_names
        assert "M1" in param_names
        assert "L2" in param_names
        assert "G" in param_names

        traj = sys.integrate(
            jnp.array([0.5, 0.5, 0.0, 0.0]),
            100,
            0.01,
            {"M1": 1.0, "M2": 1.0, "L1": 1.0, "L2": 1.0, "G": 9.81},
        )
        assert traj.shape == (100, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
