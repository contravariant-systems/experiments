import pytest
import warnings
import jax.numpy as jnp
from sympy import symbols, Rational

from contravariant import LagrangianSystem
from contravariant.catalog import (
    harmonic_oscillator,
    coupled_oscillators,
    anharmonic_oscillator,
    simple_pendulum,
    double_pendulum,
    free_particle_2d,
    fput_chain,
)


class TestParameterValidation:
    """
    The integrate() method should validate that all required parameters
    are provided and warn about extra parameters.
    """

    def test_missing_single_parameter_harmonic_oscillator(self):
        """Missing 'k' for harmonic_oscillator should raise ValueError."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params_missing_k = {"m": 1.0}  # Missing 'k'

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params_missing_k)

        error_msg = str(excinfo.value)
        assert "k" in error_msg
        assert "Missing" in error_msg or "missing" in error_msg

    def test_missing_single_parameter_simple_pendulum(self):
        """Missing 'g' for simple_pendulum should raise ValueError."""
        sys = simple_pendulum()

        state_0 = jnp.array([0.5, 0.0])
        params_missing_g = {"m": 1.0, "l": 1.0}  # Missing 'g'

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params_missing_g)

        error_msg = str(excinfo.value)
        assert "g" in error_msg

    def test_missing_single_parameter_double_pendulum(self):
        """Missing 'l2' for double_pendulum should raise ValueError."""
        sys = double_pendulum()

        state_0 = jnp.array([0.5, 0.5, 0.0, 0.0])
        params_missing_l2 = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "g": 9.81}  # Missing 'l2'

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params_missing_l2)

        error_msg = str(excinfo.value)
        assert "l2" in error_msg

    def test_missing_multiple_parameters_raises(self):
        """Missing multiple parameters should list all of them."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params_empty = {}

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params_empty)

        error_msg = str(excinfo.value)
        assert "k" in error_msg
        assert "m" in error_msg

    def test_extra_parameter_warns_harmonic_oscillator(self):
        """Extra parameters should trigger a warning."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        # 'g' is not used by harmonic_oscillator (only m, k)
        params_extra = {"m": 1.0, "k": 1.0, "g": 9.81}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traj = sys.integrate(state_0, 100, 0.01, params_extra)

            # Should still work
            assert traj.shape == (100, 2)

            # But should warn about extra parameter
            assert len(w) == 1
            assert "g" in str(w[0].message)

    def test_extra_parameter_warns_coupled_oscillators(self):
        """Extra 'l' parameter for coupled_oscillators should warn."""
        sys = coupled_oscillators()

        state_0 = jnp.array([1.0, 0.0, 0.0, 0.0])
        # coupled_oscillators uses m, k, k_c — not 'l'
        params_extra = {"m": 1.0, "k": 1.0, "k_c": 0.5, "l": 1.0}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traj = sys.integrate(state_0, 100, 0.01, params_extra)

            assert traj.shape == (100, 4)
            assert len(w) == 1
            assert "l" in str(w[0].message)

    def test_typo_in_parameter_name_raises(self):
        """A typo like 'mass' instead of 'm' should be caught."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params_typo = {"mass": 1.0, "k": 1.0}  # 'mass' instead of 'm'

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params_typo)

        error_msg = str(excinfo.value)
        assert "m" in error_msg  # Missing 'm'

    def test_correct_parameters_harmonic_oscillator(self):
        """Correct parameters should work without warnings."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traj = sys.integrate(state_0, 100, 0.01, params)

            assert traj.shape == (100, 2)
            assert len(w) == 0  # No warnings

    def test_correct_parameters_anharmonic_oscillator(self):
        """anharmonic_oscillator uses m, k, lambda."""
        sys = anharmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0, "lambda": 0.1}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traj = sys.integrate(state_0, 100, 0.01, params)

            assert traj.shape == (100, 2)
            assert len(w) == 0

    def test_correct_parameters_free_particle_2d(self):
        """free_particle_2d uses only m."""
        sys = free_particle_2d()

        state_0 = jnp.array([0.0, 0.0, 1.0, 0.5])
        params = {"m": 1.0}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traj = sys.integrate(state_0, 100, 0.01, params)

            assert traj.shape == (100, 4)
            assert len(w) == 0

    def test_correct_parameters_fput_chain(self):
        """fput_chain uses m, k, and alpha or beta."""
        sys = fput_chain(N=4, alpha=0.25)

        state_0 = jnp.zeros(8)
        state_0 = state_0.at[0].set(1.0)  # Displace first mass
        params = {"m": 1.0, "k": 1.0, "alpha": 0.25}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            traj = sys.integrate(state_0, 100, 0.01, params)

            assert traj.shape == (100, 8)
            assert len(w) == 0


class TestIntegratorMethodValidation:
    """Test that invalid integrator methods are rejected."""

    def test_unknown_method_raises(self):
        """A truly unknown method name should raise ValueError."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        with pytest.raises(ValueError):
            sys.integrate(state_0, 100, 0.01, params, method="symplectic_euler")

    def test_euler_not_exposed(self):
        """
        Euler exists in integrators.py but is intentionally not exposed
        through LagrangianSystem.integrate() because it systematically
        adds energy to Hamiltonian systems.
        """
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        with pytest.raises(ValueError):
            sys.integrate(state_0, 100, 0.01, params, method="euler")

    def test_valid_methods_accepted(self):
        """All valid method names should work for separable systems."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        for method in ["auto", "rk4", "verlet", "yoshida"]:
            traj = sys.integrate(state_0, 100, 0.01, params, method=method)
            assert traj.shape == (100, 2)


class TestSympyFailures:
    """
    SymPy's solve() can fail for pathological Lagrangians.
    The framework should give helpful error messages.
    """

    def test_zero_kinetic_energy_fails(self):
        """A Lagrangian with no kinetic term has singular mass matrix."""
        q, q_dot = symbols("q q_dot")
        k = symbols("k", positive=True)

        # Pure potential, no kinetic energy
        L_no_kinetic = -Rational(1, 2) * k * q**2

        with pytest.raises(RuntimeError) as excinfo:
            LagrangianSystem(L_no_kinetic, [q], [q_dot])

        error_msg = str(excinfo.value).lower()
        assert "singular" in error_msg or "solve" in error_msg

    def test_valid_complex_lagrangian_works(self):
        """A complicated but valid Lagrangian should still work."""
        x, x_dot = symbols("x x_dot")
        m, k, alpha = symbols("m k alpha", positive=True)

        # Nonlinear kinetic energy (position-dependent mass)
        L_complex = (
            Rational(1, 2) * m * (1 + alpha * x**2) * x_dot**2
            - Rational(1, 2) * k * x**2
        )

        # Should work (non-separable but solvable)
        sys = LagrangianSystem(L_complex, [x], [x_dot])

        assert sys.dof == 1
        # Non-separable because T depends on x
        assert sys.is_separable == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
