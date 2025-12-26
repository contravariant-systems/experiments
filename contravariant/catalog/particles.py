"""
Free particles and central force problems.
"""

from sympy import symbols, Rational
from ..systems import LagrangianSystem


def free_particle_2d():
    """
    Free particle in 2D.

    L = ½m(ẋ² + ẏ²)

    Conserved quantities:
        - Energy (time translation symmetry)
        - p_x (x translation symmetry, x cyclic)
        - p_y (y translation symmetry, y cyclic)
    """
    x, x_dot, y, y_dot = symbols("x x_dot y y_dot")
    m = symbols("m", positive=True)

    L = Rational(1, 2) * m * (x_dot**2 + y_dot**2)

    return LagrangianSystem(L, [x, y], [x_dot, y_dot])


def central_force(V=None):
    """
    Central force problem in 2D polar coordinates.

    L = ½m(ṙ² + r²θ̇²) - V(r)

    Default V(r) = -k/r (Kepler problem).

    Conserved quantities:
        - Energy (time translation symmetry)
        - p_θ (θ cyclic, rotation symmetry → angular momentum)

    Args:
        V: potential as function of r (sympy expression), or None for Kepler
    """
    r, r_dot, theta, theta_dot = symbols("r r_dot theta theta_dot")
    m = symbols("m", positive=True)

    T = Rational(1, 2) * m * (r_dot**2 + r**2 * theta_dot**2)

    if V is None:
        # Kepler problem: V = -k/r
        k = symbols("k", positive=True)
        V_expr = -k / r
    else:
        V_expr = V

    L = T - V_expr

    return LagrangianSystem(L, [r, theta], [r_dot, theta_dot])


def kepler():
    """Alias for central_force() with default Kepler potential."""
    return central_force(V=None)
