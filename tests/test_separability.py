import pytest
import jax.numpy as jnp

from contravariant.catalog import (
    harmonic_oscillator,
    coupled_oscillators,
    simple_pendulum,
    double_pendulum,
    spherical_pendulum,
    kepler,
)


class TestNonSeparableSystems:
    """
    Symplectic integrators (Verlet, Yoshida) require H = T(p) + V(q).
    Non-separable systems like the double pendulum cannot use them.
    """

    def test_double_pendulum_is_non_separable(self):
        """Double pendulum should be detected as non-separable."""
        sys = double_pendulum()

        assert sys.is_separable == False
        assert sys.dof == 2

    def test_yoshida_on_double_pendulum_raises(self):
        """Requesting Yoshida for double pendulum should raise ValueError."""
        sys = double_pendulum()

        # Correct params for double_pendulum: m1, m2, l1, l2, g
        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}
        state_0 = jnp.array([0.5, 0.5, 0.0, 0.0])

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params, method="yoshida")

        error_msg = str(excinfo.value).lower()
        assert "separable" in error_msg
        assert "rk4" in error_msg  # Should suggest RK4

    def test_verlet_on_double_pendulum_raises(self):
        """Requesting Verlet for double pendulum should raise ValueError."""
        sys = double_pendulum()

        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}
        state_0 = jnp.array([0.5, 0.5, 0.0, 0.0])

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params, method="verlet")

        assert "separable" in str(excinfo.value).lower()

    def test_spherical_pendulum_is_non_separable(self):
        """Spherical pendulum has sin²θ coupling, making it non-separable."""
        sys = spherical_pendulum()

        assert sys.is_separable == False
        assert sys.dof == 2
        # But phi is cyclic (angular momentum conserved)
        assert len(sys.cyclic_coordinates) == 1

    def test_yoshida_on_spherical_pendulum_raises(self):
        """Requesting Yoshida for spherical pendulum should raise ValueError."""
        sys = spherical_pendulum()

        # Correct params for spherical_pendulum: m, l, g
        params = {"m": 1.0, "l": 1.0, "g": 9.81}
        state_0 = jnp.array([0.5, 0.0, 0.0, 2.0])

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params, method="yoshida")

        assert "separable" in str(excinfo.value).lower()

    def test_rk4_on_double_pendulum_works(self):
        """RK4 should work fine for non-separable systems."""
        sys = double_pendulum()

        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}
        state_0 = jnp.array([0.5, 0.5, 0.0, 0.0])

        # Should not raise
        traj = sys.integrate(state_0, 100, 0.01, params, method="rk4")

        assert traj.shape == (100, 4)

    def test_auto_selects_rk4_for_non_separable(self):
        """method='auto' should select RK4 for non-separable systems."""
        sys = double_pendulum()

        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.81}
        state_0 = jnp.array([0.5, 0.5, 0.0, 0.0])

        # Should not raise (auto selects RK4)
        traj = sys.integrate(state_0, 100, 0.01, params, method="auto")

        assert traj.shape == (100, 4)

    def test_harmonic_oscillator_is_separable(self):
        """Harmonic oscillator should be separable and accept symplectic methods."""
        sys = harmonic_oscillator()

        assert sys.is_separable == True

        # Correct params for harmonic_oscillator: m, k
        params = {"m": 1.0, "k": 1.0}
        state_0 = jnp.array([1.0, 0.0])

        # Both should work
        traj_yoshida = sys.integrate(state_0, 100, 0.01, params, method="yoshida")
        traj_verlet = sys.integrate(state_0, 100, 0.01, params, method="verlet")

        assert traj_yoshida.shape == (100, 2)
        assert traj_verlet.shape == (100, 2)

    def test_simple_pendulum_is_separable(self):
        """Simple pendulum should be separable."""
        sys = simple_pendulum()

        assert sys.is_separable == True

        # Correct params for simple_pendulum: m, l, g
        params = {"m": 1.0, "l": 1.0, "g": 9.81}
        state_0 = jnp.array([0.5, 0.0])

        traj = sys.integrate(state_0, 100, 0.01, params, method="yoshida")
        assert traj.shape == (100, 2)

    def test_coupled_oscillators_is_separable(self):
        """Coupled oscillators should be separable."""
        sys = coupled_oscillators()

        assert sys.is_separable == True

        # Correct params for coupled_oscillators: m, k, k_c
        params = {"m": 1.0, "k": 1.0, "k_c": 0.5}
        state_0 = jnp.array([1.0, 0.0, 0.0, 0.0])

        traj = sys.integrate(state_0, 100, 0.01, params, method="yoshida")
        assert traj.shape == (100, 4)

    def test_kepler_is_non_separable(self):
        """Kepler problem is NOT separable because T = ½m(ṙ² + r²θ̇²) depends on r."""
        sys = kepler()

        # T contains r²θ̇² which couples position r with velocity θ̇
        assert sys.is_separable == False
        # theta is cyclic (angular momentum conserved)
        assert len(sys.cyclic_coordinates) == 1

        # Correct params for kepler: m, k
        params = {"m": 1.0, "k": 1.0}
        state_0 = jnp.array([1.0, 0.0, 0.0, 1.0])

        # Must use RK4 since non-separable
        traj = sys.integrate(state_0, 100, 0.001, params, method="rk4")
        assert traj.shape == (100, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
