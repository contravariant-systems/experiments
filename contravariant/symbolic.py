"""
Symbolic mechanics utilities using SymPy.

This module implements the mathematical backbone of the framework: deriving
equations of motion from Lagrangians via the Euler-Lagrange equations.

The Euler-Lagrange Equation
---------------------------
For a Lagrangian L(q, q̇, t), the equations of motion are:

    d/dt(∂L/∂q̇) - ∂L/∂q = 0

For autonomous systems (no explicit time dependence), we expand d/dt using
the chain rule and solve for the accelerations q̈:

    d/dt(∂L/∂q̇) = Σⱼ (∂²L/∂q̇∂qⱼ) q̇ⱼ + Σⱼ (∂²L/∂q̇∂q̇ⱼ) q̈ⱼ

This gives a system of linear equations in q̈, which SymPy solves symbolically.

Key Functions
-------------
derive_equations_of_motion:
    The main workhorse. Takes L, returns symbolic expressions for q̈.

derive_hamiltonian:
    Legendre transform: H = Σᵢ pᵢq̇ᵢ - L where pᵢ = ∂L/∂q̇ᵢ

find_cyclic_coordinates:
    A coordinate q is cyclic if ∂L/∂q = 0. Its conjugate momentum p = ∂L/∂q̇
    is then conserved (Noether's theorem for translation symmetry).

extract_kinetic_potential:
    Attempt to write L = T - V where T depends only on velocities.
    If successful, the system is "separable" and symplectic integrators apply.

derive_conserved_quantity:
    Noether's theorem: given a symmetry generator ξ, compute Q = Σᵢ pᵢξᵢ.

Notes
-----
Richardson's theorem proves that symbolic equality is undecidable in general.
We use multiple simplification strategies (simplify, expand, trigsimp) to
handle most practical cases, but pathological expressions may fail.
"""

from sympy import symbols, diff, solve, simplify, expand, trigsimp


def is_zero(expr):
    """Try multiple strategies to determine if expression is zero."""
    if expr == 0:
        return True
    if simplify(expr) == 0:
        return True
    if expand(expr) == 0:
        return True
    if trigsimp(expr) == 0:
        return True
    # Last resort: equals() tries harder
    if expr.equals(0):
        return True
    return False


def find_cyclic_coordinates(L, q_vars, q_dot_vars):
    """
    Find coordinates that don't appear in L.
    For each cyclic coordinate, the conjugate momentum is conserved.

    The problem is that, we know from Richardson's theorem that it's
    undecidable whether two symbolic expressions are equal. So we just
    try a bunch of things and hope that they suffice in aggregate.

    Unsatisfactory, I know.

    Returns:
        list of (q, p_expr) tuples where p = ∂L/∂q̇ is conserved

    """
    cyclic = []
    for q, q_dot in zip(q_vars, q_dot_vars):
        dL_dq = diff(L, q)
        if is_zero(dL_dq):
            # q is cyclic; conjugate momentum is conserved
            p = diff(L, q_dot)
            cyclic.append((q, p))
    return cyclic


def derive_hamiltonian(L, q_vars, q_dot_vars):
    """
    Compute H = Σ pᵢq̇ᵢ - L via Legendre transform.

    Returns:
        H: sympy expression for the Hamiltonian
        momenta: list of (q_dot, p) pairs
    """
    H = -L
    momenta = []
    for q_dot in q_dot_vars:
        p = diff(L, q_dot)
        H = H + p * q_dot
        momenta.append((q_dot, p))
    return simplify(H), momenta


def derive_equations_of_motion(L, q_vars, q_dot_vars):
    """
    Given a symbolic Lagrangian, derive symbolic accelerations.

    The Euler-Lagrange equation:
        d/dt(∂L/∂q̇) - ∂L/∂q = 0

    For autonomous systems, we expand d/dt via chain rule and solve for q̈.

    Args:
        L: sympy expression for the Lagrangian
        q_vars: list of position symbols [q1, q2, ...]
        q_dot_vars: list of velocity symbols [q1_dot, q2_dot, ...]

    Returns:
        dict with keys:
            'solutions': dict mapping q_ddot symbols to expressions
            'q_vars': input position symbols
            'q_dot_vars': input velocity symbols
            'q_ddot_vars': generated acceleration symbols
            'param_syms': extracted parameter symbols
            'lagrangian': the input Lagrangian
    """
    # Generate acceleration symbols
    q_ddot_vars = [symbols(f"{qd}_dot") for qd in q_dot_vars]

    # Extract parameter symbols: everything in L that isn't q or q_dot
    all_symbols = L.free_symbols
    param_syms = list(all_symbols - set(q_vars) - set(q_dot_vars))

    # Apply Euler-Lagrange to each coordinate
    accelerations = {}
    for q, q_dot, q_ddot in zip(q_vars, q_dot_vars, q_ddot_vars):
        dL_dq = diff(L, q)
        dL_dq_dot = diff(L, q_dot)

        # d/dt(∂L/∂q̇) via chain rule:
        # = Σ (∂²L/∂q̇∂qᵢ) q̇ᵢ + Σ (∂²L/∂q̇∂q̇ᵢ) q̈ᵢ
        d_dt_dL_dq_dot = sum(
            diff(dL_dq_dot, qv) * qv_dot for qv, qv_dot in zip(q_vars, q_dot_vars)
        ) + sum(
            diff(dL_dq_dot, qv_dot) * qv_ddot
            for qv_dot, qv_ddot in zip(q_dot_vars, q_ddot_vars)
        )

        euler_lagrange = d_dt_dL_dq_dot - dL_dq
        accelerations[q_ddot] = euler_lagrange

    # Solve the system for accelerations
    try:
        solutions = solve(list(accelerations.values()), q_ddot_vars)
    except Exception as e:
        raise RuntimeError(
            f"SymPy could not solve the Euler-Lagrange equations.\n\n"
            f"This can happen for highly nonlinear or pathological Lagrangians.\n"
            f"Original error: {type(e).__name__}: {e}\n\n"
            f"Lagrangian: L = {L}"
        ) from e

    # Check for empty solutions (solve() returns {} if it can't find a solution)
    if not solutions:
        raise RuntimeError(
            f"SymPy could not solve the Euler-Lagrange equations for accelerations.\n\n"
            f"The system of equations may be:\n"
            f"  - Singular (determinant of mass matrix is zero)\n"
            f"  - Under/over-determined\n"
            f"  - Too complex for symbolic solution\n\n"
            f"Euler-Lagrange equations:\n"
            + "\n".join(f"  {qdd} : {eq} = 0" for qdd, eq in accelerations.items())
            + f"\n\nLagrangian: L = {L}"
        )

    # Handle case where solve() returns a list of multiple solutions
    if isinstance(solutions, list):
        if len(solutions) == 1:
            solutions = solutions[0]
        else:
            raise RuntimeError(
                f"SymPy found multiple solutions for the accelerations.\n\n"
                f"This is unexpected for a well-posed mechanical system.\n"
                f"Number of solution branches: {len(solutions)}\n\n"
                f"Lagrangian: L = {L}"
            )

    cyclic = find_cyclic_coordinates(L, q_vars, q_dot_vars)
    H, momenta = derive_hamiltonian(L, q_vars, q_dot_vars)

    return {
        "solutions": solutions,
        "q_vars": q_vars,
        "q_dot_vars": q_dot_vars,
        "q_ddot_vars": q_ddot_vars,
        "param_syms": param_syms,
        "lagrangian": L,
        "cyclic_coords": cyclic,
        "hamiltonian": H,
        "momenta": momenta,
    }


def extract_kinetic_potential(L, q_vars, q_dot_vars):
    """
    Attempt to separate L = T - V where T depends only on q_dot
    and V depends only on q.

    Args:
        L: sympy expression for the Lagrangian
        q_vars: list of position symbols
        q_dot_vars: list of velocity symbols

    Returns:
        dict with keys:
            'T': kinetic energy expression
            'V': potential energy expression
            'is_separable': bool indicating if separation succeeded
            'grad_V': list of ∂V/∂qᵢ for each coordinate
    """
    # V is the part of L that doesn't depend on velocities (negated)
    # Set all q_dots to zero
    V_candidate = -L
    for qd in q_dot_vars:
        V_candidate = V_candidate.subs(qd, 0)
    V = simplify(V_candidate)

    # T is L + V
    T = simplify(L + V)

    # Check separability: T should not depend on q_vars
    T_depends_on_q = any(T.has(q) for q in q_vars)

    # Compute gradient of V
    grad_V = [diff(V, q) for q in q_vars]

    return {
        "T": T,
        "V": V,
        "is_separable": not T_depends_on_q,
        "grad_V": grad_V,
    }


def derive_conserved_quantity(L, q_vars, q_dot_vars, xi):
    """
    Given infinitesimal generator ξ, compute Noether charge Q = Σ pᵢξᵢ.

    Args:
        L: Lagrangian
        q_vars: [q1, q2, ...]
        q_dot_vars: [q1_dot, q2_dot, ...]
        xi: [ξ1, ξ2, ...] generator of the symmetry

    Returns:
        Q: conserved quantity
    """
    Q = 0
    for q_dot, xi_i in zip(q_dot_vars, xi):
        p_i = diff(L, q_dot)
        Q = Q + p_i * xi_i
    return simplify(Q)
