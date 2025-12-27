from sympy import symbols, Rational
from ..systems import LagrangianSystem


def fput_chain(N, alpha=None, beta=None, boundary="fixed"):
    """
    Create FPUT lattice with N masses.

    V(Δ) = ½k Δ² + α Δ³ + β Δ⁴

    Args:
        N: number of masses
        alpha: cubic coefficient (α-FPUT)
        beta: quartic coefficient (β-FPUT)
        boundary: "fixed" or "periodic"
    """
    if alpha is None and beta is None:
        raise ValueError("Specify alpha or beta")

    q_vars = [symbols(f"q{i}") for i in range(N)]
    q_dot_vars = [symbols(f"q{i}_dot") for i in range(N)]

    m, k = symbols("m k", positive=True)
    alpha_sym = symbols("alpha", real=True) if alpha is not None else None
    beta_sym = symbols("beta", real=True) if beta is not None else None

    # Kinetic energy
    T = Rational(1, 2) * m * sum(qd**2 for qd in q_dot_vars)

    def spring_V(delta):
        V = Rational(1, 2) * k * delta**2
        if alpha_sym is not None:
            V += alpha_sym * delta**3
        if beta_sym is not None:
            V += beta_sym * delta**4
        return V

    V = 0
    if boundary == "fixed":
        # Wall — mass 0
        V += spring_V(q_vars[0])
        # Interior springs
        for i in range(N - 1):
            V += spring_V(q_vars[i + 1] - q_vars[i])
        # Mass N-1 — wall
        V += spring_V(-q_vars[N - 1])
    elif boundary == "periodic":
        for i in range(N):
            V += spring_V(q_vars[(i + 1) % N] - q_vars[i])

    return LagrangianSystem(T - V, q_vars, q_dot_vars)
