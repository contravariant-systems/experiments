"""
Learning utilities: loss functions and optimization helpers.

For inverse problems: learning parameters from observed trajectories.
"""

import jax.numpy as jnp
from jax import grad


def trajectory_loss(traj_observed, traj_predicted):
    """
    Simple trajectory matching loss: sum of squared differences.

    Note: This can be non-convex for oscillatory systems due to
    phase shifts creating local minima.
    """
    return jnp.sum((traj_observed - traj_predicted) ** 2)


def energy_statistic_loss(traj_observed, traj_predicted, n_dof):
    """
    Energy statistic loss: match mean squared positions and velocities.

    More robust than trajectory_loss because it's phase-invariant.
    Two trajectories with the same energy but different phases will
    have similar statistics.

    Args:
        traj_observed: observed trajectory (n_steps, 2*n_dof)
        traj_predicted: predicted trajectory (n_steps, 2*n_dof)
        n_dof: degrees of freedom

    Returns:
        scalar loss
    """
    obs_mean_q2 = jnp.mean(jnp.sum(traj_observed[:, :n_dof] ** 2, axis=1))
    obs_mean_v2 = jnp.mean(jnp.sum(traj_observed[:, n_dof:] ** 2, axis=1))
    pred_mean_q2 = jnp.mean(jnp.sum(traj_predicted[:, :n_dof] ** 2, axis=1))
    pred_mean_v2 = jnp.mean(jnp.sum(traj_predicted[:, n_dof:] ** 2, axis=1))
    return (pred_mean_q2 - obs_mean_q2) ** 2 + (pred_mean_v2 - obs_mean_v2) ** 2


def make_loss_fn(integrate_fn, traj_observed, state_0, n_steps, dt):
    """
    Create a loss function for parameter learning.

    Args:
        integrate_fn: function (state_0, n_steps, dt, params) -> trajectory
        traj_observed: observed trajectory to match
        state_0: initial state
        n_steps: number of integration steps
        dt: timestep

    Returns:
        loss(params) -> scalar
    """

    def loss(params):
        traj_predicted = integrate_fn(state_0, n_steps, dt, params)
        return trajectory_loss(traj_observed, traj_predicted)

    return loss


def make_loss_fn_with_dynamics(
    integrate_fn, dynamics, traj_observed, state_0, n_steps, dt
):
    """
    Create a loss function for parameter learning when dynamics is a separate arg.

    Args:
        integrate_fn: function (state_0, n_steps, dt, params, dynamics) -> trajectory
        dynamics: the dynamics function
        traj_observed: observed trajectory to match
        state_0: initial state
        n_steps: number of integration steps
        dt: timestep

    Returns:
        loss(params) -> scalar
    """

    def loss(params):
        traj_predicted = integrate_fn(state_0, n_steps, dt, params, dynamics)
        return trajectory_loss(traj_observed, traj_predicted)

    return loss


def gradient_descent(
    loss_fn, params_init, learning_rate, n_iters, param_mask=None, print_every=10
):
    """
    Simple gradient descent optimization.

    Args:
        loss_fn: function params -> scalar
        params_init: initial parameter dict
        learning_rate: step size
        n_iters: number of iterations
        param_mask: dict of {param_name: bool} where True = learnable
                   If None, all params are learnable
        print_every: print progress every N iterations (0 to disable)

    Returns:
        params_final: optimized parameters
        history: list of (iteration, loss, params) tuples
    """
    params = params_init.copy()
    grad_loss = grad(loss_fn)
    history = []

    for i in range(n_iters):
        loss_val = loss_fn(params)
        grads = grad_loss(params)

        # Update parameters
        for key in params:
            if param_mask is None or param_mask.get(key, True):
                params[key] = params[key] - learning_rate * grads[key]

        if print_every > 0 and i % print_every == 0:
            print(f"Iter {i}: loss = {loss_val:.6f}, params = {params}")

        history.append((i, float(loss_val), params.copy()))

    return params, history


def optimize_with_optax(
    loss_fn, params_init, optimizer, n_iters, param_mask=None, print_every=10
):
    """
    Optimize using an optax optimizer.

    Args:
        loss_fn: function params -> scalar
        params_init: initial parameter dict
        optimizer: optax optimizer (e.g., optax.adam(1e-2))
        n_iters: number of iterations
        param_mask: dict of {param_name: bool} where False = frozen
        print_every: print progress every N iterations (0 to disable)

    Returns:
        params_final: optimized parameters
        history: list of (iteration, loss, params) tuples
    """
    import optax

    # Apply freezing if mask provided
    if param_mask is not None:
        optimizer = optax.chain(
            optimizer,
            optax.masked(
                optax.set_to_zero(), {k: not v for k, v in param_mask.items()}
            ),
        )

    params = params_init.copy()
    opt_state = optimizer.init(params)
    grad_loss = grad(loss_fn)
    history = []

    for i in range(n_iters):
        loss_val = loss_fn(params)
        grads = grad_loss(params)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if print_every > 0 and i % print_every == 0:
            print(f"Iter {i}: loss = {loss_val:.6f}, params = {params}")

        history.append((i, float(loss_val), params.copy()))

    return params, history


def learn_parameters(
    integrate_fn,
    traj_observed,
    state_0,
    n_steps,
    dt,
    n_dof,
    params_fixed,
    params_init,
    loss_type="energy_statistic",
    learning_rate=0.1,
    max_iterations=100,
    tolerance=1e-8,
    verbose=True,
):
    """
    Learn unknown parameters from an observed trajectory.

    Args:
        integrate_fn: function (state_0, n_steps, dt, params) -> trajectory
        traj_observed: observed trajectory array
        state_0: initial state
        n_steps: integration steps
        dt: timestep
        n_dof: degrees of freedom (for splitting state into q and q_dot)
        params_fixed: dict of fixed parameters {'m': 1.0}
        params_init: dict of initial guesses {'k': 0.5}
        loss_type: 'trajectory', 'energy_statistic', or callable
            - 'trajectory': match trajectories point-by-point (fragile)
            - 'energy_statistic': match energy distribution (robust)
            - callable: custom loss(traj_pred, traj_obs) -> scalar
        learning_rate: optimizer learning rate
        max_iterations: max optimization steps
        tolerance: stop when loss < tolerance
        verbose: print progress

    Returns:
        dict of learned parameters
    """
    import optax
    from jax import jit

    def make_full_params(params_learn):
        return {**params_fixed, **params_learn}

    # Build loss function based on type
    if loss_type == "trajectory":

        def loss_fn(params_learn):
            params = make_full_params(params_learn)
            traj = integrate_fn(state_0, n_steps, dt, params)
            return jnp.mean((traj - traj_observed) ** 2)

    elif loss_type == "energy_statistic":
        # Precompute observed statistics
        obs_mean_v2 = jnp.mean(jnp.sum(traj_observed[:, n_dof:] ** 2, axis=1))
        obs_mean_q2 = jnp.mean(jnp.sum(traj_observed[:, :n_dof] ** 2, axis=1))

        def loss_fn(params_learn):
            params = make_full_params(params_learn)
            traj = integrate_fn(state_0, n_steps, dt, params)
            pred_mean_v2 = jnp.mean(jnp.sum(traj[:, n_dof:] ** 2, axis=1))
            pred_mean_q2 = jnp.mean(jnp.sum(traj[:, :n_dof] ** 2, axis=1))
            return (pred_mean_v2 - obs_mean_v2) ** 2 + (pred_mean_q2 - obs_mean_q2) ** 2

    elif callable(loss_type):

        def loss_fn(params_learn):
            params = make_full_params(params_learn)
            traj = integrate_fn(state_0, n_steps, dt, params)
            return loss_type(traj, traj_observed)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    params_learn = {k: jnp.array(float(v)) for k, v in params_init.items()}
    opt_state = optimizer.init(params_learn)

    grad_fn = jit(grad(loss_fn))
    loss_fn_jit = jit(loss_fn)

    if verbose:
        print(f"Learning: {list(params_init.keys())}")
        print(f"Method: {loss_type if isinstance(loss_type, str) else 'custom'}")
        print()
        header = f"{'Step':>6} | {'Loss':>12} | " + " | ".join(
            f"{k:>10}" for k in params_init.keys()
        )
        print(header)
        print("-" * len(header))

    # Optimization loop
    for i in range(max_iterations):
        loss = float(loss_fn_jit(params_learn))

        if verbose and (
            i % max(1, max_iterations // 10) == 0 or i == max_iterations - 1
        ):
            values = " | ".join(
                f"{float(params_learn[k]):>10.4f}" for k in params_init.keys()
            )
            print(f"{i:>6} | {loss:>12.2e} | {values}")

        if loss < tolerance:
            if verbose:
                print(f"\nConverged at step {i}")
            break

        grads = grad_fn(params_learn)
        updates, opt_state = optimizer.update(grads, opt_state, params_learn)
        params_learn = optax.apply_updates(params_learn, updates)

    return {k: float(v) for k, v in params_learn.items()}
