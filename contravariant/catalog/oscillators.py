"""
Oscillator systems: harmonic, coupled, anharmonic.
"""

from sympy import symbols, Rational
from ..systems import LagrangianSystem


def harmonic_oscillator(coord="q", mass="m", spring="k"):
    """
    Simple harmonic oscillator with configurable symbols.

    L = ½m q̇² - ½k q²

    Args:
        coord: coordinate name (default 'q')
        mass: mass parameter name (default 'm')
        spring: spring constant name (default 'k')

    Returns:
        LagrangianSystem

    Example:
        >>> sys = harmonic_oscillator()
        >>> sys_x = harmonic_oscillator(coord='x')
        >>> sys_2d = harmonic_oscillator(coord='x') + harmonic_oscillator(coord='y')
    """
    q = symbols(coord)
    q_dot = symbols(f"{coord}_dot")
    m_sym = symbols(mass, positive=True)
    k_sym = symbols(spring, positive=True)

    L = Rational(1, 2) * m_sym * q_dot**2 - Rational(1, 2) * k_sym * q**2

    return LagrangianSystem(L, [q], [q_dot])


def harmonic_oscillator_2d():
    """
    2D isotropic harmonic oscillator.

    L = ½m(ẋ² + ẏ²) - ½k(x² + y²)

    Has rotation symmetry → conserves angular momentum L_z = m(xẏ - ẋy).

    Parameters: m (mass), k (spring constant)

    Returns:
        LagrangianSystem
    """
    x, x_dot, y, y_dot = symbols("x x_dot y y_dot")
    m, k = symbols("m k", positive=True)

    L = Rational(1, 2) * m * (x_dot**2 + y_dot**2) - Rational(1, 2) * k * (x**2 + y**2)

    return LagrangianSystem(L, [x, y], [x_dot, y_dot])


def coupled_oscillators():
    """
    Two coupled harmonic oscillators.

    L = ½m(q̇₁² + q̇₂²) - ½k(q₁² + q₂²) - ½kc(q₂ - q₁)²

    Two masses connected by springs, with a coupling spring between them.

    Parameters: m (mass), k (spring constant), k_c (coupling constant)

    Returns:
        LagrangianSystem
    """
    q1, q1_dot, q2, q2_dot = symbols("q1 q1_dot q2 q2_dot")
    m, k, k_c = symbols("m k k_c", positive=True)

    L = (
        Rational(1, 2) * m * (q1_dot**2 + q2_dot**2)
        - Rational(1, 2) * k * (q1**2 + q2**2)
        - Rational(1, 2) * k_c * (q2 - q1) ** 2
    )

    return LagrangianSystem(L, [q1, q2], [q1_dot, q2_dot])


def anharmonic_oscillator(order=4):
    """
    Anharmonic oscillator with polynomial potential.

    L = ½mq̇² - ½kq² - λqⁿ

    Parameters: m (mass), k (spring constant), lam (anharmonic coefficient)

    Args:
        order: power of anharmonic term (default 4, i.e., quartic)

    Returns:
        LagrangianSystem
    """
    q, q_dot = symbols("q q_dot")
    m, k, lam = symbols("m k lambda", positive=True)

    L = Rational(1, 2) * m * q_dot**2 - Rational(1, 2) * k * q**2 - lam * q**order

    return LagrangianSystem(L, [q], [q_dot])
