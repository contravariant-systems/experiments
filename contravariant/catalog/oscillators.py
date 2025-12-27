"""
Oscillator systems: harmonic, coupled, anharmonic.
"""

from sympy import symbols, Rational
from ..systems import LagrangianSystem


def harmonic_oscillator(coord="q", mass="m", spring="k"):
    """
    Simple harmonic oscillator.

    L = ½m q̇² - ½k q²

    Args:
        coord: coordinate name (default 'q')
        mass: mass parameter name (default 'm')
        spring: spring constant name (default 'k')

    Example:
        >>> sys = harmonic_oscillator()
        >>> sys_2d = harmonic_oscillator(coord='x') + harmonic_oscillator(coord='y')
    """
    q = symbols(coord)
    q_dot = symbols(f"{coord}_dot")
    m_sym = symbols(mass, positive=True)
    k_sym = symbols(spring, positive=True)

    L = Rational(1, 2) * m_sym * q_dot**2 - Rational(1, 2) * k_sym * q**2
    return LagrangianSystem(L, [q], [q_dot])


def harmonic_oscillator_2d(coords=("q1", "q2"), mass="m", spring="k"):
    """
    2D isotropic harmonic oscillator.

    L = ½m(q̇₁² + q̇₂²) - ½k(q₁² + q₂²)

    Has rotation symmetry → conserves angular momentum.

    Args:
        coords: coordinate names (default ('q1', 'q2'))
        mass: mass parameter name (default 'm')
        spring: spring constant name (default 'k')
    """
    c1, c2 = coords
    return harmonic_oscillator(
        coord=c1, mass=mass, spring=spring
    ) + harmonic_oscillator(coord=c2, mass=mass, spring=spring)


def coupled_oscillators(coords=("q1", "q2"), mass="m", spring="k", coupling="k_c"):
    """
    Two coupled harmonic oscillators.

    L = ½m(q̇₁² + q̇₂²) - ½k(q₁² + q₂²) - ½kc(q₂ - q₁)²

    Two masses connected to walls by springs, with a coupling spring between them.

    Args:
        coords: coordinate names (default ('q1', 'q2'))
        mass: mass parameter name (default 'm')
        spring: wall spring constant name (default 'k')
        coupling: coupling spring constant name (default 'k_c')

    Example:
        >>> sys = coupled_oscillators()
        >>> sys_ab = coupled_oscillators(coords=('a', 'b'), coupling='J')
    """
    c1, c2 = coords
    q1, q2 = symbols(c1), symbols(c2)
    q1_dot, q2_dot = symbols(f"{c1}_dot"), symbols(f"{c2}_dot")
    m_sym = symbols(mass, positive=True)
    k_sym = symbols(spring, positive=True)
    k_c_sym = symbols(coupling, positive=True)

    L = (
        Rational(1, 2) * m_sym * (q1_dot**2 + q2_dot**2)
        - Rational(1, 2) * k_sym * (q1**2 + q2**2)
        - Rational(1, 2) * k_c_sym * (q2 - q1) ** 2
    )
    return LagrangianSystem(L, [q1, q2], [q1_dot, q2_dot])


def anharmonic_oscillator(
    coord="q", mass="m", spring="k", anharmonic="lambda", order=4
):
    """
    Anharmonic oscillator with polynomial potential.

    L = ½mq̇² - ½kq² - λqⁿ

    Args:
        coord: coordinate name (default 'q')
        mass: mass parameter name (default 'm')
        spring: spring constant name (default 'k')
        anharmonic: anharmonic coefficient name (default 'lambda')
        order: power of anharmonic term (default 4, i.e., quartic)

    Example:
        >>> sys = anharmonic_oscillator()
        >>> sys_cubic = anharmonic_oscillator(anharmonic='alpha', order=3)
    """
    q = symbols(coord)
    q_dot = symbols(f"{coord}_dot")
    m_sym = symbols(mass, positive=True)
    k_sym = symbols(spring, positive=True)
    lam_sym = symbols(anharmonic, positive=True)

    L = (
        Rational(1, 2) * m_sym * q_dot**2
        - Rational(1, 2) * k_sym * q**2
        - lam_sym * q**order
    )
    return LagrangianSystem(L, [q], [q_dot])
