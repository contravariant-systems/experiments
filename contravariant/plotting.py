"""
Visualization utilities for Contravariant.

Plotting trajectories, phase space, energy evolution, etc.
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap


def plot_trajectory(traj, labels=None, title=None, xlabel="Time step"):
    """
    Plot state variables over time.

    Args:
        traj: trajectory array of shape (n_steps, state_dim)
        labels: list of labels for each state variable
        title: plot title
        xlabel: x-axis label
    """
    n_vars = traj.shape[1]
    if labels is None:
        labels = [f"x_{i}" for i in range(n_vars)]

    plt.figure()
    for i in range(n_vars):
        plt.plot(traj[:, i], label=labels[i])
    plt.xlabel(xlabel)
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


def plot_positions(traj, n_dof=None, labels=None, title=None):
    """
    Plot only position variables (first half of state).

    Args:
        traj: trajectory array of shape (n_steps, 2*n_dof)
        n_dof: number of degrees of freedom (inferred if None)
        labels: list of labels for each position
        title: plot title
    """
    if n_dof is None:
        n_dof = traj.shape[1] // 2

    if labels is None:
        labels = [f"q_{i}" for i in range(n_dof)]

    plt.figure()
    for i in range(n_dof):
        plt.plot(traj[:, i], label=labels[i])
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


def plot_phase_space(traj, dof_index=0, title=None):
    """
    Plot phase space trajectory for a single degree of freedom.

    Args:
        traj: trajectory array of shape (n_steps, 2*n_dof)
        dof_index: which degree of freedom to plot (default: 0)
        title: plot title
    """
    n_dof = traj.shape[1] // 2
    q = traj[:, dof_index]
    p = traj[:, n_dof + dof_index]

    plt.figure()
    plt.plot(q, p)
    plt.xlabel(f"q_{dof_index}")
    plt.ylabel(f"p_{dof_index}")
    if title:
        plt.title(title)
    else:
        plt.title("Phase Space")
    plt.axis("equal")
    plt.show()


def plot_energy_evolution(traj, energy_fn, params, title=None):
    """
    Plot total energy over time.

    Args:
        traj: trajectory array
        energy_fn: function (state, params) -> energy
        params: parameter dict
        title: plot title
    """
    energies = vmap(lambda s: energy_fn(s, params))(traj)

    plt.figure()
    plt.plot(energies)
    plt.xlabel("Time step")
    plt.ylabel("Energy")
    if title:
        plt.title(title)
    plt.show()

    return energies


def plot_energy_error(traj, energy_fn, params, title=None, relative=False):
    """
    Plot energy error (deviation from initial energy) over time.

    Args:
        traj: trajectory array
        energy_fn: function (state, params) -> energy
        params: parameter dict
        title: plot title
        relative: if True, plot relative error (E - E0) / E0
    """
    energies = vmap(lambda s: energy_fn(s, params))(traj)
    E0 = energies[0]

    if relative:
        error = (energies - E0) / jnp.abs(E0)
        ylabel = "Relative Energy Error"
    else:
        error = energies - E0
        ylabel = "Energy Error"

    plt.figure()
    plt.plot(error)
    plt.xlabel("Time step")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.show()

    return energies, error


def plot_phase_space_multi(initial_states, final_states_list, labels, dof_index=0):
    """
    Plot evolution of a cloud of initial conditions in phase space.

    Useful for visualizing symplectic vs non-symplectic integrators.

    Args:
        initial_states: array of initial states (n_particles, state_dim)
        final_states_list: list of arrays of final states for each integrator
        labels: list of labels for each integrator
        dof_index: which degree of freedom to plot
    """
    n_dof = initial_states.shape[1] // 2

    plt.figure(figsize=(12, 4))

    # Initial cloud
    plt.subplot(1, len(final_states_list) + 1, 1)
    plt.scatter(
        initial_states[:, dof_index],
        initial_states[:, n_dof + dof_index],
        s=10,
        alpha=0.5,
    )
    plt.xlabel("q")
    plt.ylabel("p")
    plt.title("Initial")
    plt.axis("equal")

    # Final clouds for each integrator
    for i, (final_states, label) in enumerate(zip(final_states_list, labels)):
        plt.subplot(1, len(final_states_list) + 1, i + 2)
        plt.scatter(
            final_states[:, dof_index],
            final_states[:, n_dof + dof_index],
            s=10,
            alpha=0.5,
        )
        plt.xlabel("q")
        plt.ylabel("p")
        plt.title(label)
        plt.axis("equal")

    plt.tight_layout()
    plt.show()


def plot_energy_errors(
    trajectories,
    energy_fn,
    params,
    labels=None,
    title=None,
    save_as=None,
    show=True,
):
    """
    Compare energy errors across multiple trajectories.

    Args:
        trajectories: dict of {method_name: trajectory} or list of trajectories
        energy_fn: function (state, params) -> energy
        params: parameter dict
        labels: list of labels (required if trajectories is a list)
        title: plot title
        save_as: filename to save (without extension)
        show: whether to display the plot

    Returns:
        fig, ax: matplotlib figure and axes
    """
    # Normalize input to dict
    if isinstance(trajectories, list):
        if labels is None:
            labels = [f"Method {i}" for i in range(len(trajectories))]
        trajectories = dict(zip(labels, trajectories))

    fig, ax = plt.subplots(figsize=(10, 4))

    # Get E0 from first trajectory
    first_traj = list(trajectories.values())[0]
    E0 = energy_fn(first_traj[0], params)

    for label, traj in trajectories.items():
        energies = vmap(lambda s: energy_fn(s, params))(traj)
        ax.plot(energies - E0, label=label, alpha=0.8, linewidth=0.5)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Energy Error")
    ax.set_title(title or "Energy Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(f"{save_as}.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax


def plot_configuration_space(
    traj,
    coord_indices=(0, 1),
    xlabel=None,
    ylabel=None,
    title=None,
    save_as=None,
    show=True,
):
    """
    Plot trajectory in configuration space.

    Args:
        traj: trajectory array
        coord_indices: tuple of (x_index, y_index) into state vector
        xlabel: x-axis label
        ylabel: y-axis label
        title: plot title
        save_as: filename to save (without extension)
        show: whether to display the plot

    Returns:
        fig, ax: matplotlib figure and axes
    """
    i, j = coord_indices

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(traj[:, i], traj[:, j], "b-", linewidth=0.3, alpha=0.7)

    ax.set_xlabel(xlabel or f"$q_{i}$")
    ax.set_ylabel(ylabel or f"$q_{j}$")
    ax.set_title(title or "Configuration Space")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_as:
        plt.savefig(f"{save_as}.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax
