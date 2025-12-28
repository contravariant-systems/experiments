import pytest
import warnings
import jax.numpy as jnp
from sympy import symbols, Rational

from contravariant import LagrangianSystem
from contravariant.catalog import (
    harmonic_oscillator,
    harmonic_oscillator_2d,
    coupled_oscillators,
    anharmonic_oscillator,
    simple_pendulum,
    double_pendulum,
    spherical_pendulum,
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


class TestCompareIntegrators:
    """
    Test the compare_integrators() convenience method.

    This is a user-facing method that should:
    - Run multiple integrators on the same system
    - Return trajectories for each method
    - Print conservation diagnostics
    - Filter invalid methods (symplectic on non-separable)
    """

    def test_returns_trajectories_dict(self):
        """compare_integrators should return dict of trajectories."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        trajectories = sys.compare_integrators(
            state_0, 100, 0.01, params, methods=["rk4", "yoshida"], show=False
        )

        assert isinstance(trajectories, dict)
        assert "rk4" in trajectories
        assert "yoshida" in trajectories
        assert trajectories["rk4"].shape == (100, 2)
        assert trajectories["yoshida"].shape == (100, 2)

    def test_default_methods_separable(self):
        """For separable systems, default should include yoshida and rk4."""
        sys = simple_pendulum()
        state_0 = jnp.array([0.5, 0.0])
        params = {"m": 1.0, "l": 1.0, "g": 9.8}

        trajectories = sys.compare_integrators(state_0, 100, 0.01, params, show=False)

        # Default for separable: rk4 and yoshida
        assert "rk4" in trajectories or "yoshida" in trajectories

    def test_default_methods_non_separable(self):
        """For non-separable systems, default should only include rk4."""
        sys = double_pendulum()
        state_0 = jnp.array([0.5, 0.3, 0.0, 0.0])
        params = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 9.8}

        trajectories = sys.compare_integrators(state_0, 100, 0.01, params, show=False)

        # Should only have rk4 (symplectic methods filtered out)
        assert "rk4" in trajectories
        assert "yoshida" not in trajectories
        assert "verlet" not in trajectories

    def test_filters_symplectic_for_non_separable(self, capsys):
        """Requesting symplectic methods on non-separable should filter them out."""
        sys = spherical_pendulum()
        state_0 = jnp.array([0.5, 0.0, 0.0, 1.0])
        params = {"m": 1.0, "l": 1.0, "g": 9.8}

        trajectories = sys.compare_integrators(
            state_0, 100, 0.01, params, methods=["rk4", "yoshida", "verlet"], show=False
        )

        # Symplectic methods should be filtered
        assert "rk4" in trajectories
        assert "yoshida" not in trajectories
        assert "verlet" not in trajectories

        # Should print a note about filtering
        captured = capsys.readouterr()
        assert (
            "not separable" in captured.out.lower() or "removed" in captured.out.lower()
        )

    def test_with_custom_quantities(self):
        """Can pass custom conserved quantities to check."""
        sys = harmonic_oscillator_2d()
        state_0 = jnp.array([1.0, 0.0, 0.0, 1.0])
        params = {"m": 1.0, "k": 1.0}

        # Define angular momentum
        q1, q2 = sys.q_vars
        v1, v2 = sys.q_dot_vars
        m = symbols("m", positive=True)
        Lz_expr = m * (q1 * v2 - q2 * v1)

        trajectories = sys.compare_integrators(
            state_0,
            100,
            0.01,
            params,
            methods=["yoshida"],
            quantities={"Lz": Lz_expr},
            show=False,
        )

        assert "yoshida" in trajectories

    def test_all_methods_separable(self):
        """All valid methods should work for separable system."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        trajectories = sys.compare_integrators(
            state_0, 100, 0.01, params, methods=["rk4", "verlet", "yoshida"], show=False
        )

        assert len(trajectories) == 3
        for method in ["rk4", "verlet", "yoshida"]:
            assert method in trajectories
            assert trajectories[method].shape == (100, 2)

    def test_trajectories_are_different(self):
        """Different integrators should give slightly different trajectories."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        trajectories = sys.compare_integrators(
            state_0, 1000, 0.01, params, methods=["rk4", "yoshida"], show=False
        )

        # Trajectories should differ (different numerical methods)
        diff = jnp.max(jnp.abs(trajectories["rk4"] - trajectories["yoshida"]))
        assert diff > 1e-10, "Trajectories should differ slightly"

        # But not too much (both should be accurate)
        assert diff < 0.1, "Trajectories should be similar"


class TestCheckConservation:
    """
    Test the check_conservation() convenience method.
    """

    def test_returns_dict_with_energy(self):
        """check_conservation should return dict with at least 'energy'."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        traj = sys.integrate(state_0, 100, 0.01, params)
        result = sys.check_conservation(traj, params)

        assert isinstance(result, dict)
        assert "energy" in result

        # Each entry should be (max_error, mean_error) or similar
        assert len(result["energy"]) == 2

    def test_energy_error_small_for_yoshida(self):
        """Yoshida should have small energy error."""
        sys = simple_pendulum()
        state_0 = jnp.array([0.5, 0.0])
        params = {"m": 1.0, "l": 1.0, "g": 9.8}

        traj = sys.integrate(state_0, 1000, 0.01, params, method="yoshida")
        result = sys.check_conservation(traj, params)

        max_error, mean_error = result["energy"]
        assert max_error < 1e-7, f"Yoshida energy error too large: {max_error}"

    def test_with_custom_quantities(self):
        """Can check custom conserved quantities."""
        sys = free_particle_2d()
        state_0 = jnp.array([0.0, 0.0, 3.0, 4.0])
        params = {"m": 2.0}

        # Define momenta
        v1, v2 = sys.q_dot_vars
        m = symbols("m", positive=True)

        p1 = m * v1
        p2 = m * v2

        traj = sys.integrate(state_0, 100, 0.01, params, method="yoshida")
        result = sys.check_conservation(
            traj, params, quantities={"p1": p1, "p2": p2}
        )

        assert "energy" in result
        assert "p1" in result
        assert "p2" in result

        # Momenta should be exactly conserved
        assert result["p1"][0] < 1e-10
        assert result["p2"][0] < 1e-10

    def test_non_conserved_quantity_detected(self):
        """A non-conserved quantity should show significant error."""
        sys = simple_pendulum()
        state_0 = jnp.array([1.0, 0.0])  # Large amplitude
        params = {"m": 1.0, "l": 1.0, "g": 9.8}

        # θ itself is not conserved (it oscillates!)
        theta = sys.q_vars[0]

        traj = sys.integrate(state_0, 1000, 0.01, params, method="yoshida")
        result = sys.check_conservation(traj, params, quantities={"theta": theta})

        # θ should vary significantly
        max_error, _ = result["theta"]
        assert max_error > 0.5, f"θ should vary, but max_error = {max_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
