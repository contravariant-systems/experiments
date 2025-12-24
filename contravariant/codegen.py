"""
Code generation: sympy expressions → JAX functions.

Bridges the symbolic and numerical worlds.
"""

from sympy import lambdify
import jax.numpy as jnp


def make_lagrangian_dynamics_fn(eom):
    """
    Given result from derive_equations_of_motion, generate JAX dynamics.

    For Lagrangian mechanics: state = [q1, ..., qn, q1_dot, ..., qn_dot]
    Returns function: f(state, params) -> [q1_dot, ..., qn_dot, q1_ddot, ..., qn_ddot]

    Args:
        eom: dict from derive_equations_of_motion

    Returns:
        JAX-compatible dynamics function
    """
    solutions = eom["solutions"]
    q_vars = eom["q_vars"]
    q_dot_vars = eom["q_dot_vars"]
    q_ddot_vars = eom["q_ddot_vars"]
    param_syms = eom["param_syms"]

    n_dof = len(q_vars)
    q_ddot_exprs = [solutions[qdd] for qdd in q_ddot_vars]

    # Build lambdified function
    all_inputs = list(q_vars) + list(q_dot_vars) + list(param_syms)
    q_ddot_fn = lambdify(all_inputs, q_ddot_exprs, modules="jax")

    def dynamics(state, params):
        q_vals = [state[i] for i in range(n_dof)]
        q_dot_vals = [state[n_dof + i] for i in range(n_dof)]
        param_vals = [params[str(p)] for p in param_syms]
        q_ddot_vals = q_ddot_fn(*q_vals, *q_dot_vals, *param_vals)
        return jnp.array([*q_dot_vals, *q_ddot_vals])

    return dynamics


def make_hamiltonian_dynamics_fn(eom, energy_parts):
    """
    Generate Hamiltonian dynamics for separable H = T(p) + V(q).

    state = [q1, ..., qn, p1, ..., pn]
    Returns function: f(state, params) -> d_state/dt

    Hamilton's equations:
        dq/dt = ∂H/∂p = ∂T/∂p
        dp/dt = -∂H/∂q = -∂V/∂q

    Args:
        eom: dict from derive_equations_of_motion
        energy_parts: dict from extract_kinetic_potential

    Returns:
        JAX-compatible dynamics function
    """
    from sympy import diff

    q_vars = eom["q_vars"]
    q_dot_vars = eom["q_dot_vars"]
    param_syms = eom["param_syms"]

    n_dof = len(q_vars)
    T = energy_parts["T"]
    V = energy_parts["V"]

    # For standard kinetic energy T = Σ p²/2m, we have p = m*q_dot
    # So ∂T/∂p = p/m = q_dot
    # For now, assume simple case: ∂T/∂q_dot = m*q_dot, so ∂T/∂p = q_dot
    grad_T_exprs = [diff(T, qd) for qd in q_dot_vars]  # These are momenta-like
    grad_V_exprs = [diff(V, q) for q in q_vars]

    all_inputs = list(q_vars) + list(q_dot_vars) + list(param_syms)
    grad_V_fn = lambdify(all_inputs, grad_V_exprs, modules="jax")

    def dynamics(state, params):
        q_vals = [state[i] for i in range(n_dof)]
        q_dot_vals = [state[n_dof + i] for i in range(n_dof)]
        param_vals = [params[str(p)] for p in param_syms]

        # dq/dt = q_dot (trivial for Lagrangian form)
        dq_dt = q_dot_vals

        # For Lagrangian form with standard T = ½m*q_dot²:
        # The EOM gives us q_ddot directly
        # This function is really for when you want grad_V separately
        grad_V_vals = grad_V_fn(*q_vals, *q_dot_vals, *param_vals)

        return jnp.array([*dq_dt, *[-g for g in grad_V_vals]])

    return dynamics


def make_grad_V_fn(eom, energy_parts):
    """
    Generate a function that computes ∇V(q) for use in symplectic integrators.

    Args:
        eom: dict from derive_equations_of_motion
        energy_parts: dict from extract_kinetic_potential

    Returns:
        Function: grad_V(q, params) -> array of shape (n_dof,)
    """
    q_vars = eom["q_vars"]
    param_syms = eom["param_syms"]
    grad_V_exprs = energy_parts["grad_V"]

    n_dof = len(q_vars)

    # For grad_V, we only need q (not q_dot)
    all_inputs = list(q_vars) + list(param_syms)

    # For single DOF, lambdify returns scalar; for multi-DOF, returns list
    # We handle both cases uniformly
    if n_dof == 1:
        grad_V_lambdified = lambdify(all_inputs, grad_V_exprs[0], modules="jax")

        def grad_V(q, params):
            q_val = q[0]
            param_vals = [params[str(p)] for p in param_syms]
            result = grad_V_lambdified(q_val, *param_vals)
            return jnp.array([result])

    else:
        grad_V_lambdified = lambdify(all_inputs, grad_V_exprs, modules="jax")

        def grad_V(q, params):
            q_vals = [q[i] for i in range(n_dof)]
            param_vals = [params[str(p)] for p in param_syms]
            result = grad_V_lambdified(*q_vals, *param_vals)
            return jnp.array(result)

    return grad_V


def make_energy_fn(eom, energy_parts):
    """
    Generate a function that computes total energy H = T + V.

    Args:
        eom: dict from derive_equations_of_motion
        energy_parts: dict from extract_kinetic_potential

    Returns:
        Function: energy(state, params) -> scalar
    """
    q_vars = eom["q_vars"]
    q_dot_vars = eom["q_dot_vars"]
    param_syms = eom["param_syms"]
    T = energy_parts["T"]
    V = energy_parts["V"]
    H = T + V

    n_dof = len(q_vars)
    all_inputs = list(q_vars) + list(q_dot_vars) + list(param_syms)
    H_fn = lambdify(all_inputs, H, modules="jax")

    def energy(state, params):
        q_vals = [state[i] for i in range(n_dof)]
        q_dot_vals = [state[n_dof + i] for i in range(n_dof)]
        param_vals = [params[str(p)] for p in param_syms]
        return H_fn(*q_vals, *q_dot_vals, *param_vals)

    return energy


def make_conserved_quantity_fn(expr, q_vars, q_dot_vars, param_syms):
    """Generate JAX function for a conserved quantity."""
    n_dof = len(q_vars)
    all_inputs = list(q_vars) + list(q_dot_vars) + list(param_syms)
    fn = lambdify(all_inputs, expr, modules='jax')

    def conserved_qty(state, params):
        q_vals = [state[i] for i in range(n_dof)]
        q_dot_vals = [state[n_dof + i] for i in range(n_dof)]
        param_vals = [params[str(p)] for p in param_syms]
        return fn(*q_vals, *q_dot_vals, *param_vals)

    return conserved_qty
