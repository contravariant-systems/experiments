"""
Tests for edge cases and error handling in the contravariant package.

These tests verify that the framework gives clear, helpful error messages
when users encounter common problems:
1. Time-dependent Lagrangians (not yet supported)
2. Requesting symplectic integrators for non-separable systems
3. SymPy failing to solve the Euler-Lagrange equations
4. Missing or extra parameters

Run with: pytest tests/test_edge_cases.py -v
"""

import pytest
import warnings
import jax.numpy as jnp
from sympy import symbols, Rational, cos, sin

from contravariant import LagrangianSystem
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


# =============================================================================
# 1. TIME-DEPENDENT LAGRANGIANS
# =============================================================================


class TestTimeDependentLagrangians:
    """
    Time-dependent Lagrangians L(q, q̇, t) require modified integrators
    that we haven't implemented yet. The framework should reject them
    with a clear error message.
    """

    def test_driven_oscillator_rejected(self):
        """A driven harmonic oscillator has explicit time dependence."""
        q, q_dot = symbols("q q_dot")
        m, k, F_0, omega = symbols("m k F_0 omega", positive=True)
        t = symbols("t")

        # Driven harmonic oscillator: L = T - V + F(t)·q
        L_driven = (
            Rational(1, 2) * m * q_dot**2
            - Rational(1, 2) * k * q**2
            + F_0 * cos(omega * t) * q
        )

        with pytest.raises(ValueError) as excinfo:
            LagrangianSystem(L_driven, [q], [q_dot])

        # Check error message is helpful
        assert "time-dependent" in str(excinfo.value).lower()
        assert "t" in str(excinfo.value)

    def test_moving_potential_rejected(self):
        """A potential that moves in time is time-dependent."""
        q, q_dot = symbols("q q_dot")
        m, k, omega = symbols("m k omega", positive=True)
        t = symbols("t")

        # Oscillator with moving equilibrium point
        L_moving = (
            Rational(1, 2) * m * q_dot**2
            - Rational(1, 2) * k * (q - sin(omega * t)) ** 2
        )

        with pytest.raises(ValueError) as excinfo:
            LagrangianSystem(L_moving, [q], [q_dot])

        assert "time-dependent" in str(excinfo.value).lower()

    def test_time_independent_accepted(self):
        """Standard SHO has no explicit time dependence and should work."""
        q, q_dot = symbols("q q_dot")
        m, k = symbols("m k", positive=True)

        L_sho = Rational(1, 2) * m * q_dot**2 - Rational(1, 2) * k * q**2

        # Should not raise
        sys = LagrangianSystem(L_sho, [q], [q_dot])

        assert sys.is_time_dependent == False
        assert sys.is_separable == True

    def test_parameter_named_tau_accepted(self):
        """A parameter named 'tau' (not 't') should not trigger time-dependence."""
        theta, theta_dot = symbols("theta theta_dot")
        m, k, tau = symbols("m k tau", positive=True)

        # tau is a time constant parameter, not the time variable
        L_with_tau = (
            Rational(1, 2) * m * theta_dot**2 - Rational(1, 2) * k * theta**2 / tau
        )

        # Should not raise
        sys = LagrangianSystem(L_with_tau, [theta], [theta_dot])

        assert sys.is_time_dependent == False
        assert "tau" in [str(p) for p in sys.parameters]


# =============================================================================
# 2. NON-SEPARABLE SYSTEMS + SYMPLECTIC METHODS
# =============================================================================


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


# =============================================================================
# 3. SYMPY SOLVE FAILURES
# =============================================================================


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


# =============================================================================
# 4. PARAMETER VALIDATION
# =============================================================================


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


# =============================================================================
# 5. INTEGRATOR METHOD VALIDATION
# =============================================================================


class TestIntegratorMethodValidation:
    """Test that invalid integrator methods are rejected."""

    def test_unknown_method_raises(self):
        """An unknown method name should raise ValueError."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        with pytest.raises(ValueError) as excinfo:
            sys.integrate(state_0, 100, 0.01, params, method="euler")

        assert (
            "euler" in str(excinfo.value).lower()
            or "unknown" in str(excinfo.value).lower()
        )

    def test_valid_methods_accepted(self):
        """All valid method names should work for separable systems."""
        sys = harmonic_oscillator()

        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        for method in ["auto", "rk4", "verlet", "yoshida"]:
            traj = sys.integrate(state_0, 100, 0.01, params, method=method)
            assert traj.shape == (100, 2)


# =============================================================================
# 6. CATALOG COVERAGE - ALL SYSTEMS WORK
# =============================================================================


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


# =============================================================================
# 7. SYMBOL CUSTOMIZATION
# =============================================================================


class TestSymbolCustomization:
    """Test that catalog functions accept custom symbol names."""

    def test_harmonic_oscillator_custom_symbols(self):
        """harmonic_oscillator should accept custom coord/param names."""
        sys = harmonic_oscillator(coord="x", mass="M", spring="K")

        assert sys.dof == 1
        # Check the symbols were renamed
        coord_names = [str(c) for c in sys.coordinates]
        param_names = [str(p) for p in sys.parameters]

        assert "x" in coord_names
        assert "M" in param_names
        assert "K" in param_names

        # Use RK4 because _infer_mass_matrix looks for 'm', not 'M'
        traj = sys.integrate(
            jnp.array([1.0, 0.0]), 100, 0.01, {"M": 1.0, "K": 2.0}, method="rk4"
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

        # Use RK4 because _infer_mass_matrix looks for 'm', not 'M'
        traj = sys.integrate(
            jnp.array([0.5, 0.0]),
            100,
            0.01,
            {"M": 1.0, "L": 1.0, "G": 9.81},
            method="rk4",
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
