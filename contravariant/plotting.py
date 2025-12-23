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


def compare_energy_errors(trajs, energy_fns, params_list, labels, title=None):
    """
    Compare energy error across multiple integrators.

    Args:
        trajs: list of trajectories
        energy_fns: list of energy functions (or single function)
        params_list: list of param dicts (or single dict)
        labels: list of labels for each trajectory
        title: plot title
    """
    if not isinstance(energy_fns, list):
        energy_fns = [energy_fns] * len(trajs)
    if not isinstance(params_list, list):
        params_list = [params_list] * len(trajs)

    plt.figure()
    for traj, energy_fn, params, label in zip(trajs, energy_fns, params_list, labels):
        energies = vmap(lambda s: energy_fn(s, params))(traj)
        E0 = energies[0]
        error = energies - E0
        plt.plot(error, label=label)

    plt.xlabel("Time step")
    plt.ylabel("Energy Error")
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


def plot_phase_space_cloud(initial_states, final_states_list, labels, dof_index=0):
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
