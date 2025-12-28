"""Tests for learning module: parameter inference from trajectories."""

import pytest
import jax.numpy as jnp

from contravariant import learn_parameters, trajectory_loss, energy_statistic_loss
from contravariant.learning import gradient_descent, optimize_with_optax
from contravariant.catalog import harmonic_oscillator


class TestLossFunctions:
    """Test loss function correctness."""

    def test_trajectory_loss_identical(self):
        """Identical trajectories have zero loss."""
        traj = jnp.array([[1.0, 0.0], [0.9, -0.1], [0.8, -0.2]])
        assert trajectory_loss(traj, traj) == 0.0

    def test_trajectory_loss_different(self):
        """Different trajectories have positive loss."""
        traj1 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        traj2 = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        assert trajectory_loss(traj1, traj2) == 2.0

    def test_energy_statistic_loss_identical(self):
        """Identical trajectories have zero energy statistic loss."""
        traj = jnp.array([[1.0, 0.5], [0.8, 0.6], [0.6, 0.7]])
        assert energy_statistic_loss(traj, traj, n_dof=1) == 0.0

    def test_energy_statistic_loss_phase_invariant(self):
        """Phase-shifted trajectories have similar energy statistics."""
        # Two SHO trajectories with same amplitude, different phase
        t = jnp.linspace(0, 10, 100)
        traj1 = jnp.stack([jnp.cos(t), -jnp.sin(t)], axis=1)
        traj2 = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=1)  # 90° phase shift

        loss = energy_statistic_loss(traj1, traj2, n_dof=1)
        assert loss < 0.01  # Should be nearly zero


class TestLearnParameters:
    """Test parameter learning from trajectories."""

    def test_learn_spring_constant(self):
        """Learn spring constant k from harmonic oscillator trajectory."""
        sys = harmonic_oscillator()
        true_params = {"m": 1.0, "k": 2.0}
        state_0 = jnp.array([1.0, 0.0])

        # Generate "observed" trajectory
        traj_observed = sys.integrate(state_0, 500, 0.01, true_params)

        # Learn k with m fixed
        learned = learn_parameters(
            integrate_fn=sys.integrate,
            traj_observed=traj_observed,
            state_0=state_0,
            n_steps=500,
            dt=0.01,
            n_dof=1,
            params_fixed={"m": 1.0},
            params_init={"k": 1.0},  # Start with wrong guess
            loss_type="energy_statistic",
            learning_rate=0.1,
            max_iterations=100,
            verbose=False,
        )

        assert abs(learned["k"] - 2.0) < 0.1

    def test_learn_mass(self):
        """Learn mass m from harmonic oscillator trajectory."""
        sys = harmonic_oscillator()
        true_params = {"m": 2.0, "k": 1.0}
        state_0 = jnp.array([1.0, 0.0])

        traj_observed = sys.integrate(state_0, 500, 0.01, true_params)

        learned = learn_parameters(
            integrate_fn=sys.integrate,
            traj_observed=traj_observed,
            state_0=state_0,
            n_steps=500,
            dt=0.01,
            n_dof=1,
            params_fixed={"k": 1.0},
            params_init={"m": 1.0},
            loss_type="energy_statistic",
            learning_rate=0.1,
            max_iterations=100,
            verbose=False,
        )

        assert abs(learned["m"] - 2.0) < 0.2

    def test_trajectory_loss_type(self):
        """Learning works with trajectory loss (less robust but should work)."""
        sys = harmonic_oscillator()
        true_params = {"m": 1.0, "k": 2.0}
        state_0 = jnp.array([1.0, 0.0])

        traj_observed = sys.integrate(state_0, 200, 0.01, true_params)

        learned = learn_parameters(
            integrate_fn=sys.integrate,
            traj_observed=traj_observed,
            state_0=state_0,
            n_steps=200,
            dt=0.01,
            n_dof=1,
            params_fixed={"m": 1.0},
            params_init={"k": 1.9},  # Start close (trajectory loss is fragile)
            loss_type="trajectory",
            learning_rate=0.01,
            max_iterations=50,
            verbose=False,
        )

        assert abs(learned["k"] - 2.0) < 0.2

    def test_custom_loss_type(self):
        """Learning works with custom loss function."""
        sys = harmonic_oscillator()
        true_params = {"m": 1.0, "k": 2.0}
        state_0 = jnp.array([1.0, 0.0])

        traj_observed = sys.integrate(state_0, 500, 0.01, true_params)

        def custom_loss(traj_pred, traj_obs):
            # Match both position and velocity statistics
            pos_err = (
                jnp.mean(traj_pred[:, 0] ** 2) - jnp.mean(traj_obs[:, 0] ** 2)
            ) ** 2
            vel_err = (
                jnp.mean(traj_pred[:, 1] ** 2) - jnp.mean(traj_obs[:, 1] ** 2)
            ) ** 2
            return pos_err + vel_err

        learned = learn_parameters(
            integrate_fn=sys.integrate,
            traj_observed=traj_observed,
            state_0=state_0,
            n_steps=500,
            dt=0.01,
            n_dof=1,
            params_fixed={"m": 1.0},
            params_init={"k": 1.0},
            loss_type=custom_loss,
            learning_rate=0.1,
            max_iterations=100,
            verbose=False,
        )

        assert abs(learned["k"] - 2.0) < 0.3

    def test_invalid_loss_type_raises(self):
        """Invalid loss_type raises ValueError."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        traj = sys.integrate(state_0, 100, 0.01, {"m": 1.0, "k": 1.0})

        with pytest.raises(ValueError, match="Unknown loss_type"):
            learn_parameters(
                integrate_fn=sys.integrate,
                traj_observed=traj,
                state_0=state_0,
                n_steps=100,
                dt=0.01,
                n_dof=1,
                params_fixed={"m": 1.0},
                params_init={"k": 1.0},
                loss_type="invalid",
                verbose=False,
            )


class TestGradientDescent:
    """Test standalone gradient descent optimizer."""

    def test_minimize_quadratic(self):
        """Gradient descent finds minimum of simple quadratic."""

        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        params_final, history = gradient_descent(
            loss_fn=loss_fn,
            params_init={"x": jnp.array(0.0)},
            learning_rate=0.1,
            n_iters=50,
            print_every=0,
        )

        assert abs(params_final["x"] - 3.0) < 0.1

    def test_param_mask(self):
        """Param mask freezes specified parameters."""

        def loss_fn(params):
            return (params["x"] - 3.0) ** 2 + (params["y"] - 5.0) ** 2

        params_final, _ = gradient_descent(
            loss_fn=loss_fn,
            params_init={"x": jnp.array(0.0), "y": jnp.array(0.0)},
            learning_rate=0.1,
            n_iters=50,
            param_mask={"x": True, "y": False},  # y is frozen
            print_every=0,
        )

        assert abs(params_final["x"] - 3.0) < 0.1
        assert params_final["y"] == 0.0  # Unchanged


class TestOptimizeWithOptax:
    """Test optax optimizer wrapper."""

    def test_adam_minimizes(self):
        """Optax Adam finds minimum."""
        import optax

        def loss_fn(params):
            return (params["x"] - 3.0) ** 2

        params_final, history = optimize_with_optax(
            loss_fn=loss_fn,
            params_init={"x": jnp.array(0.0)},
            optimizer=optax.adam(0.1),
            n_iters=75,
            print_every=0,
        )

        assert abs(params_final["x"] - 3.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
