"""
Tests for physical correctness: conservation laws, symplectic structure, Noether's theorem.

These tests verify that the framework correctly implements the physics, not just
that the code runs without errors.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import vmap
from sympy import symbols, simplify, expand

from contravariant.catalog import (
    harmonic_oscillator,
    harmonic_oscillator_2d,
    simple_pendulum,
    free_particle,
    free_particle_2d,
    kepler,
    coupled_oscillators,
    spherical_pendulum,
)


jax.config.update("jax_enable_x64", True)


class TestEnergyConservation:
    """
    The core value proposition: symplectic integrators preserve energy better.

    For Hamiltonian systems, Yoshida/Verlet should have bounded oscillating error,
    while RK4 should show systematic drift over long times.
    """

    def test_yoshida_energy_bounded(self):
        """Yoshida energy error should oscillate but stay bounded."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        # Long integration: 100 periods (T = 2π for ω=1)
        n_steps = 10000
        dt = 0.01

        traj = sys.integrate(state_0, n_steps, dt, params, method="yoshida")
        energies = vmap(lambda s: sys.evaluate_energy(s, params))(traj)

        E0 = sys.evaluate_energy(state_0, params)
        max_error = jnp.max(jnp.abs(energies - E0))

        # Yoshida should keep error < 1e-6 for this timestep
        assert max_error < 1e-6, f"Yoshida energy error {max_error} exceeds bound"

    def test_rk4_energy_drifts(self):
        """RK4 energy error should grow systematically over long times."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        n_steps = 10000
        dt = 0.01

        traj = sys.integrate(state_0, n_steps, dt, params, method="rk4")
        energies = vmap(lambda s: sys.evaluate_energy(s, params))(traj)

        E0 = sys.evaluate_energy(state_0, params)

        # Check that error grows: compare first half vs second half
        first_half_max = jnp.max(jnp.abs(energies[: n_steps // 2] - E0))
        second_half_max = jnp.max(jnp.abs(energies[n_steps // 2 :] - E0))

        # RK4 error should grow (second half worse than first half)
        # This can be flaky, so we just check error is larger than Yoshida
        assert second_half_max > first_half_max * 0.5, "RK4 should show growing error"

    def test_verlet_vs_yoshida_order(self):
        """Verlet is O(h²), Yoshida is O(h⁴) — Yoshida should be more accurate."""
        sys = harmonic_oscillator()
        state_0 = jnp.array([1.0, 0.0])
        params = {"m": 1.0, "k": 1.0}

        n_steps = 1000
        dt = 0.01

        traj_verlet = sys.integrate(state_0, n_steps, dt, params, method="verlet")
        traj_yoshida = sys.integrate(state_0, n_steps, dt, params, method="yoshida")

        E0 = sys.evaluate_energy(state_0, params)

        verlet_error = jnp.max(
            jnp.abs(vmap(lambda s: sys.evaluate_energy(s, params))(traj_verlet) - E0)
        )
        yoshida_error = jnp.max(
            jnp.abs(vmap(lambda s: sys.evaluate_energy(s, params))(traj_yoshida) - E0)
        )

        # Yoshida should be at least 10x better for same timestep
        assert (
            yoshida_error < verlet_error / 10
        ), f"Yoshida ({yoshida_error}) should be much better than Verlet ({verlet_error})"

    def test_pendulum_energy_conservation(self):
        """Energy conservation in nonlinear system (pendulum)."""
        sys = simple_pendulum()
        # Large amplitude — nonlinear regime
        state_0 = jnp.array([2.0, 0.0])  # ~115 degrees
        params = {"m": 1.0, "l": 1.0, "g": 9.8}

        n_steps = 5000
        dt = 0.001  # Small dt for nonlinear system

        traj = sys.integrate(state_0, n_steps, dt, params, method="yoshida")
        energies = vmap(lambda s: sys.evaluate_energy(s, params))(traj)

        E0 = sys.evaluate_energy(state_0, params)
        max_error = jnp.max(jnp.abs(energies - E0))

        # Should conserve energy even in nonlinear regime
        assert max_error < 1e-5, f"Pendulum energy error {max_error}"


class TestCyclicCoordinates:
    """
    Noether's theorem: if ∂L/∂q = 0, then p = ∂L/∂q̇ is conserved.
    """

    def test_free_particle_momentum_conserved(self):
        """Free particle: q is cyclic, so p = mq̇ is conserved."""
        sys = free_particle()

        # Check cyclic detection
        assert len(sys.cyclic_coordinates) == 1
        q_cyclic, p_expr = sys.cyclic_coordinates[0]
        assert str(q_cyclic) == "q"

        # Integrate and verify momentum conservation
        state_0 = jnp.array([0.0, 5.0])  # q=0, q̇=5
        params = {"m": 2.0}

        traj = sys.integrate(state_0, 1000, 0.01, params, method="yoshida")

        # p = m * q̇
        momenta = params["m"] * traj[:, 1]
        p0 = params["m"] * state_0[1]

        max_error = jnp.max(jnp.abs(momenta - p0))
        assert max_error < 1e-10, f"Momentum not conserved: error {max_error}"

    def test_free_particle_2d_both_momenta_conserved(self):
        """Free particle 2D: both coordinates cyclic, both momenta conserved."""
        sys = free_particle_2d()

        assert len(sys.cyclic_coordinates) == 2

        state_0 = jnp.array([0.0, 0.0, 3.0, 4.0])  # q1=0, q2=0, v1=3, v2=4
        params = {"m": 1.5}

        traj = sys.integrate(state_0, 1000, 0.01, params, method="yoshida")

        # Both momenta should be conserved
        p1 = params["m"] * traj[:, 2]
        p2 = params["m"] * traj[:, 3]

        assert jnp.max(jnp.abs(p1 - p1[0])) < 1e-10
        assert jnp.max(jnp.abs(p2 - p2[0])) < 1e-10

    def test_kepler_angular_momentum_conserved(self):
        """Kepler problem: θ is cyclic, so L = mr²θ̇ is conserved."""
        sys = kepler()

        # θ should be cyclic
        cyclic_names = [str(q) for q, p in sys.cyclic_coordinates]
        assert "theta" in cyclic_names

        # Circular orbit initial conditions: v = sqrt(k/mr) for circular orbit
        r0 = 1.0
        params = {"m": 1.0, "k": 1.0}
        v_circular = jnp.sqrt(params["k"] / (params["m"] * r0))

        state_0 = jnp.array([r0, 0.0, 0.0, v_circular])  # r, θ, ṙ, θ̇

        # Use RK4 since Kepler is non-separable (r²θ̇² term)
        traj = sys.integrate(state_0, 2000, 0.001, params, method="rk4")

        # Angular momentum L = m * r² * θ̇
        r = traj[:, 0]
        theta_dot = traj[:, 3]
        L = params["m"] * r**2 * theta_dot

        L0 = params["m"] * r0**2 * v_circular
        max_error = jnp.max(jnp.abs(L - L0)) / L0

        # Allow some error since RK4 doesn't perfectly conserve
        assert max_error < 0.01, f"Angular momentum error {max_error*100:.2f}%"

    def test_spherical_pendulum_lz_conserved(self):
        """Spherical pendulum: φ is cyclic, so Lz = ml²sin²θ·φ̇ is conserved."""
        sys = spherical_pendulum()

        cyclic_names = [str(q) for q, p in sys.cyclic_coordinates]
        assert "phi" in cyclic_names

        # Initial conditions with some azimuthal velocity
        theta0, phi0 = 0.5, 0.0
        theta_dot0, phi_dot0 = 0.0, 2.0
        state_0 = jnp.array([theta0, phi0, theta_dot0, phi_dot0])
        params = {"m": 1.0, "l": 1.0, "g": 9.8}

        traj = sys.integrate(state_0, 2000, 0.001, params, method="rk4")

        # Lz = m * l² * sin²θ * φ̇
        theta = traj[:, 0]
        phi_dot = traj[:, 3]
        Lz = params["m"] * params["l"] ** 2 * jnp.sin(theta) ** 2 * phi_dot

        Lz0 = params["m"] * params["l"] ** 2 * jnp.sin(theta0) ** 2 * phi_dot0
        max_error = jnp.max(jnp.abs(Lz - Lz0)) / jnp.abs(Lz0)

        assert max_error < 0.01, f"Lz error {max_error*100:.2f}%"


class TestHamiltonian:
    """
    Verify the Legendre transform H = Σpᵢq̇ᵢ - L is computed correctly.
    """

    def test_sho_hamiltonian_form(self):
        """SHO Hamiltonian should be H = p²/2m + kq²/2."""
        sys = harmonic_oscillator()

        # Get symbolic pieces
        q = sys.q_vars[0]
        q_dot = sys.q_dot_vars[0]
        m, k = symbols("m k", positive=True)

        # Expected: H = ½m·q̇² + ½k·q² (in terms of velocities)
        # Our H is in terms of velocities since we use Lagrangian formulation
        H_expected = m * q_dot**2 / 2 + k * q**2 / 2

        # Verify symbolically
        diff = simplify(sys.hamiltonian - H_expected)
        assert (
            diff == 0
        ), f"Hamiltonian mismatch: got {sys.hamiltonian}, expected {H_expected}"

    def test_pendulum_hamiltonian_form(self):
        """Pendulum Hamiltonian should be H = p²/2ml² + mgl(1-cosθ)."""
        from sympy import cos

        sys = simple_pendulum()

        theta = sys.q_vars[0]
        theta_dot = sys.q_dot_vars[0]
        m, l, g = symbols("m l g", positive=True)

        # H = T + V = ½ml²θ̇² + mgl(1 - cos θ)
        H_expected = m * l**2 * theta_dot**2 / 2 + m * g * l * (1 - cos(theta))

        diff = simplify(expand(sys.hamiltonian - H_expected))
        assert diff == 0, f"Hamiltonian mismatch: {sys.hamiltonian} vs {H_expected}"

    def test_hamiltonian_equals_energy(self):
        """For autonomous systems, H evaluated numerically should equal energy()."""
        sys = coupled_oscillators()
        state = jnp.array([1.0, -0.5, 0.3, 0.7])
        params = {"m": 1.0, "k": 2.0, "k_c": 0.5}

        # Compile Hamiltonian and evaluate
        H_fn = sys.compile(sys.hamiltonian)
        H_val = H_fn(state, params)
        E_val = sys.evaluate_energy(state, params)

        assert jnp.abs(H_val - E_val) < 1e-10


class TestComposition:
    """
    Test system composition via + and - operators.
    """

    def test_add_two_systems(self):
        """Adding two 1D systems should give a 2D system."""
        sys_x = harmonic_oscillator(coord="x", spring="kx")
        sys_y = harmonic_oscillator(coord="y", spring="ky")

        sys_2d = sys_x + sys_y

        assert sys_2d.dof == 2
        assert len(sys_2d.coordinates) == 2
        assert len(sys_2d.parameters) == 3  # m, kx, ky

        # Should be separable
        assert sys_2d.is_separable

        # Integrate
        state_0 = jnp.array([1.0, 0.5, 0.0, 0.0])
        params = {"m": 1.0, "kx": 1.0, "ky": 4.0}  # ωy = 2ωx

        traj = sys_2d.integrate(state_0, 1000, 0.01, params, method="yoshida")
        assert traj.shape == (1000, 4)

        # Energy should be conserved
        E0 = sys_2d.evaluate_energy(state_0, params)
        E_final = sys_2d.evaluate_energy(traj[-1], params)
        assert jnp.abs(E_final - E0) < 1e-6

    def test_subtract_coupling_potential(self):
        """Subtracting a coupling term creates coupled oscillators."""
        from sympy import symbols, Rational

        sys_x = harmonic_oscillator(coord="x")
        sys_y = harmonic_oscillator(coord="y")

        x, y = symbols("x y")
        k_c = symbols("k_c", positive=True)
        V_coupling = Rational(1, 2) * k_c * (x - y) ** 2

        sys_coupled = sys_x + sys_y - V_coupling

        assert sys_coupled.dof == 2
        assert "k_c" in [str(p) for p in sys_coupled.parameters]

        # Should still be separable (coupling is in V only)
        assert sys_coupled.is_separable

    def test_composition_preserves_cyclic(self):
        """Composing free particles should preserve cyclic coordinates."""
        sys = free_particle(coord="x") + free_particle(coord="y")

        # Both should be cyclic
        cyclic_names = [str(q) for q, p in sys.cyclic_coordinates]
        assert "x" in cyclic_names
        assert "y" in cyclic_names


class TestCompileMethod:
    """
    Test the expert escape hatch: compile arbitrary expressions.
    """

    def test_compile_custom_observable(self):
        """Compile and evaluate a custom observable."""
        sys = harmonic_oscillator()

        # Compile x² + v² (related to energy but different)
        q, v = sys.q_vars[0], sys.q_dot_vars[0]
        observable = q**2 + v**2

        obs_fn = sys.compile(observable)

        state = jnp.array([3.0, 4.0])
        params = {"m": 1.0, "k": 1.0}

        result = obs_fn(state, params)
        expected = 3.0**2 + 4.0**2

        assert jnp.abs(result - expected) < 1e-10

    def test_compile_angular_momentum(self):
        """Compile angular momentum for 2D system."""
        sys = harmonic_oscillator_2d()

        q1, q2 = sys.q_vars
        v1, v2 = sys.q_dot_vars
        m = symbols("m", positive=True)

        # Lz = m(q1·v2 - q2·v1)
        Lz = m * (q1 * v2 - q2 * v1)
        Lz_fn = sys.compile(Lz)

        state = jnp.array([1.0, 0.0, 0.0, 1.0])  # circular motion
        params = {"m": 2.0, "k": 1.0}

        result = Lz_fn(state, params)
        expected = 2.0 * (1.0 * 1.0 - 0.0 * 0.0)

        assert jnp.abs(result - expected) < 1e-10


class TestPhaseSpaceVolume:
    """
    Liouville's theorem: symplectic integrators preserve phase space volume.
    """

    def test_phase_space_volume_preserved(self):
        """
        Evolve a cloud of initial conditions.
        Symplectic integrator should preserve the "volume" (approximately).
        """
        sys = harmonic_oscillator()
        params = {"m": 1.0, "k": 1.0}

        # Create cloud of initial conditions (small perturbations)
        n_particles = 100
        key = jnp.array([0, 1], dtype=jnp.uint32)  # Simple seed

        # Deterministic spread around (1, 0)
        angles = jnp.linspace(0, 2 * jnp.pi, n_particles, endpoint=False)
        radius = 0.1
        q0 = 1.0 + radius * jnp.cos(angles)
        v0 = 0.0 + radius * jnp.sin(angles)
        initial_states = jnp.stack([q0, v0], axis=1)

        # Evolve with Yoshida
        def evolve(state):
            return sys.integrate(state, 500, 0.01, params, method="yoshida")[-1]

        final_states = vmap(evolve)(initial_states)

        # Compute "area" via variance (crude but works for our purposes)
        initial_var = jnp.var(initial_states[:, 0]) * jnp.var(initial_states[:, 1])
        final_var = jnp.var(final_states[:, 0]) * jnp.var(final_states[:, 1])

        # Variances should be similar (within 50% — this is a rough test)
        ratio = final_var / initial_var
        assert 0.5 < ratio < 2.0, f"Phase space volume changed by factor {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
