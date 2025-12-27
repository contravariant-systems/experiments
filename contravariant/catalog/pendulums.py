"""
Pendulum systems: simple, double, spherical.
"""

from sympy import symbols, Rational, cos, sin
from ..systems import LagrangianSystem


def simple_pendulum(coord="theta", mass="m", length="l", gravity="g"):
    """
    Simple pendulum.

    L = ½ml²θ̇² - mgl(1 - cosθ)

    Separable: yes (T depends only on θ̇, V depends only on θ).

    Args:
        coord: coordinate name (default 'theta')
        mass: mass parameter name (default 'm')
        length: length parameter name (default 'l')
        gravity: gravity parameter name (default 'g')

    Example:
        >>> sys = simple_pendulum()
        >>> sys_double = simple_pendulum(coord='theta1') + simple_pendulum(coord='theta2')
    """
    theta = symbols(coord)
    theta_dot = symbols(f"{coord}_dot")
    m_sym = symbols(mass, positive=True)
    l_sym = symbols(length, positive=True)
    g_sym = symbols(gravity, positive=True)

    L = Rational(1, 2) * m_sym * l_sym**2 * theta_dot**2 - m_sym * g_sym * l_sym * (
        1 - cos(theta)
    )
    return LagrangianSystem(L, [theta], [theta_dot])


def double_pendulum(
    coords=("theta1", "theta2"), masses=("m1", "m2"), lengths=("l1", "l2"), gravity="g"
):
    """
    Double pendulum.

    Two masses on rigid rods, free to swing. Classic chaotic system.

    NOT separable: T contains cos(θ₁ - θ₂) coupling positions and velocities.
    System will auto-select RK4.

    Args:
        coords: coordinate names (default ('theta1', 'theta2'))
        masses: mass parameter names (default ('m1', 'm2'))
        lengths: length parameter names (default ('l1', 'l2'))
        gravity: gravity parameter name (default 'g')

    Example:
        >>> sys = double_pendulum()
        >>> sys_ab = double_pendulum(coords=('a', 'b'))
    """
    c1, c2 = coords
    theta1, theta2 = symbols(c1), symbols(c2)
    theta1_dot, theta2_dot = symbols(f"{c1}_dot"), symbols(f"{c2}_dot")
    m1_sym, m2_sym = symbols(masses[0], positive=True), symbols(
        masses[1], positive=True
    )
    l1_sym, l2_sym = symbols(lengths[0], positive=True), symbols(
        lengths[1], positive=True
    )
    g_sym = symbols(gravity, positive=True)

    L = (
        Rational(1, 2) * (m1_sym + m2_sym) * l1_sym**2 * theta1_dot**2
        + Rational(1, 2) * m2_sym * l2_sym**2 * theta2_dot**2
        + m2_sym * l1_sym * l2_sym * theta1_dot * theta2_dot * cos(theta1 - theta2)
        - (m1_sym + m2_sym) * g_sym * l1_sym * (1 - cos(theta1))
        - m2_sym * g_sym * l2_sym * (1 - cos(theta2))
    )
    return LagrangianSystem(L, [theta1, theta2], [theta1_dot, theta2_dot])


def spherical_pendulum(coords=("theta", "phi"), mass="m", length="l", gravity="g"):
    """
    Spherical pendulum (pendulum free to swing in 3D).

    L = ½ml²(θ̇² + sin²θ φ̇²) - mgl(1 - cosθ)

    φ is cyclic → angular momentum about vertical axis is conserved.
    NOT separable: T contains sin²θ coupling position and velocity.

    Args:
        coords: coordinate names (default ('theta', 'phi'))
        mass: mass parameter name (default 'm')
        length: length parameter name (default 'l')
        gravity: gravity parameter name (default 'g')

    Example:
        >>> sys = spherical_pendulum()
        >>> sys_alt = spherical_pendulum(coords=('polar', 'azimuth'))
    """
    c1, c2 = coords
    theta, phi = symbols(c1), symbols(c2)
    theta_dot, phi_dot = symbols(f"{c1}_dot"), symbols(f"{c2}_dot")
    m_sym = symbols(mass, positive=True)
    l_sym = symbols(length, positive=True)
    g_sym = symbols(gravity, positive=True)

    L = Rational(1, 2) * m_sym * l_sym**2 * (
        theta_dot**2 + sin(theta) ** 2 * phi_dot**2
    ) - m_sym * g_sym * l_sym * (1 - cos(theta))
    return LagrangianSystem(L, [theta, phi], [theta_dot, phi_dot])
