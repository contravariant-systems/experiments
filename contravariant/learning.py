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
