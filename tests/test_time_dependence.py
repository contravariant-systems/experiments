import pytest
from sympy import symbols, Rational, cos, sin

from contravariant import LagrangianSystem


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
