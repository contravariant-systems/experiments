"""
Pendulum systems: simple, double, spherical.
"""

from sympy import symbols, Rational, cos, sin
from ..systems import LagrangianSystem


def simple_pendulum(coord="theta", mass="m", length="l", gravity="g"):
    """
    Simple pendulum with configurable symbols.

    L = ½ml²θ̇² - mgl(1 - cosθ)

    Separable: yes (T depends only on θ̇, V depends only on θ).

    Args:
        coord: coordinate name (default 'theta')
        mass: mass parameter name (default 'm')
        length: length parameter name (default 'l')
        gravity: gravity parameter name (default 'g')

    Returns:
        LagrangianSystem

    Example:
        >>> sys = simple_pendulum()
        >>> sys1 = simple_pendulum(coord='theta1')
        >>> sys2 = simple_pendulum(coord='theta2')
        >>> sys_two = sys1 + sys2  # two non-interacting pendulums
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


def double_pendulum():
    """
    Double pendulum.

    Two masses on rigid rods, free to swing. Classic chaotic system.

    NOT separable: T contains cos(θ₁ - θ₂) coupling positions and velocities.
    System will auto-select RK4.

    Parameters: m1, m2 (masses), l1, l2 (lengths), g (gravity)

    Returns:
        LagrangianSystem
    """
    theta1, theta1_dot = symbols("theta1 theta1_dot")
    theta2, theta2_dot = symbols("theta2 theta2_dot")
    m1, m2, l1, l2, g = symbols("m1 m2 l1 l2 g", positive=True)

    L = (
        Rational(1, 2) * (m1 + m2) * l1**2 * theta1_dot**2
        + Rational(1, 2) * m2 * l2**2 * theta2_dot**2
        + m2 * l1 * l2 * theta1_dot * theta2_dot * cos(theta1 - theta2)
        - (m1 + m2) * g * l1 * (1 - cos(theta1))
        - m2 * g * l2 * (1 - cos(theta2))
    )

    return LagrangianSystem(L, [theta1, theta2], [theta1_dot, theta2_dot])


def spherical_pendulum():
    """
    Spherical pendulum (pendulum free to swing in 3D).

    L = ½ml²(θ̇² + sin²θ φ̇²) - mgl(1 - cosθ)

    φ is cyclic → angular momentum about vertical axis is conserved.
    NOT separable: T contains sin²θ coupling position and velocity.

    Parameters: m (mass), l (length), g (gravity)

    Returns:
        LagrangianSystem
    """
    theta, theta_dot = symbols("theta theta_dot")
    phi, phi_dot = symbols("phi phi_dot")
    m, l, g = symbols("m l g", positive=True)

    L = Rational(1, 2) * m * l**2 * (
        theta_dot**2 + sin(theta) ** 2 * phi_dot**2
    ) - m * g * l * (1 - cos(theta))

    return LagrangianSystem(L, [theta, phi], [theta_dot, phi_dot])
