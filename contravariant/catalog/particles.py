"""
Free particles and central force problems.
"""

from sympy import symbols, Rational
from ..systems import LagrangianSystem


def free_particle(coord="q", mass="m"):
    """
    Free particle in 1D.

    L = ½mq̇²

    The coordinate is cyclic → momentum is conserved.

    Args:
        coord: coordinate name (default 'q')
        mass: mass parameter name (default 'm')

    Example:
        >>> sys = free_particle()
        >>> sys_2d = free_particle(coord='x') + free_particle(coord='y')
    """
    q = symbols(coord)
    q_dot = symbols(f"{coord}_dot")
    m_sym = symbols(mass, positive=True)

    L = Rational(1, 2) * m_sym * q_dot**2
    return LagrangianSystem(L, [q], [q_dot])


def free_particle_2d(coords=("q1", "q2"), mass="m"):
    """
    Free particle in 2D.

    L = ½m(q̇₁² + q̇₂²)

    Both coordinates cyclic → both momenta conserved.

    Args:
        coords: coordinate names (default ('q1', 'q2'))
        mass: mass parameter name (default 'm')

    Example:
        >>> sys = free_particle_2d()
        >>> sys_xy = free_particle_2d(coords=('x', 'y'))
    """
    c1, c2 = coords
    return free_particle(coord=c1, mass=mass) + free_particle(coord=c2, mass=mass)


def central_force(coords=("r", "theta"), mass="m", V=None, coupling="k"):
    """
    Central force problem in 2D polar coordinates.

    L = ½m(ṙ² + r²θ̇²) - V(r)

    Default V(r) = -k/r (Kepler problem).

    θ is cyclic → angular momentum conserved.

    Args:
        coords: coordinate names (default ('r', 'theta'))
        mass: mass parameter name (default 'm')
        V: potential as sympy expression in r, or None for Kepler
        coupling: coupling constant name for default Kepler potential (default 'k')

    Example:
        >>> sys = central_force()  # Kepler
        >>> r = symbols('r')
        >>> sys_ho = central_force(V=r**2 / 2)  # 2D harmonic oscillator
    """
    c1, c2 = coords
    r, theta = symbols(c1), symbols(c2)
    r_dot, theta_dot = symbols(f"{c1}_dot"), symbols(f"{c2}_dot")
    m_sym = symbols(mass, positive=True)

    T = Rational(1, 2) * m_sym * (r_dot**2 + r**2 * theta_dot**2)

    if V is None:
        k_sym = symbols(coupling, positive=True)
        V_expr = -k_sym / r
    else:
        V_expr = V

    L = T - V_expr
    return LagrangianSystem(L, [r, theta], [r_dot, theta_dot])


def kepler(coords=("r", "theta"), mass="m", coupling="k"):
    """
    Kepler problem (inverse-square central force).

    L = ½m(ṙ² + r²θ̇²) + k/r

    Args:
        coords: coordinate names (default ('r', 'theta'))
        mass: mass parameter name (default 'm')
        coupling: coupling constant name (default 'k')
    """
    return central_force(coords=coords, mass=mass, V=None, coupling=coupling)
