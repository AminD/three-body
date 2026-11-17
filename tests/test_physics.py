import numpy as np
import pytest

from threebody import figure_eight
from threebody.physics import (
    accelerations,
    angular_momentum,
    center_of_mass,
    kinetic_energy,
    momentum,
    potential_energy,
    to_com_frame,
    total_energy,
)


@pytest.fixture
def random_system():
    rng = np.random.default_rng(42)
    return rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), rng.uniform(0.5, 2.0, size=3)


def test_two_body_acceleration():
    # Two bodies on the x axis, 2 apart: a = G m_other / d^2, towards the other.
    r = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    m = np.array([3.0, 5.0])
    a = accelerations(r, m)
    np.testing.assert_allclose(a[0], [5.0 / 4.0, 0.0, 0.0])
    np.testing.assert_allclose(a[1], [-3.0 / 4.0, 0.0, 0.0])


def test_newton_third_law(random_system):
    # Internal forces cancel: sum_i m_i a_i = 0.
    r, _, m = random_system
    a = accelerations(r, m)
    np.testing.assert_allclose(m @ a, 0.0, atol=1e-12)


def test_acceleration_is_minus_gradient_of_potential(random_system):
    # m_i a_i = - dU/dr_i, checked with a centred finite difference.
    r, _, m = random_system
    eps, h = 0.1, 1e-6
    a = accelerations(r, m, eps)
    for i in range(3):
        for k in range(3):
            shift = np.zeros_like(r)
            shift[i, k] = h
            du = potential_energy(r + shift, m, eps) - potential_energy(r - shift, m, eps)
            assert m[i] * a[i, k] == pytest.approx(-du / (2 * h), rel=1e-6)


def test_softening_keeps_force_finite():
    r = np.zeros((2, 3))  # two bodies at the same point
    a = accelerations(r, np.ones(2), softening=0.1)
    assert np.all(np.isfinite(a))


def test_energy_of_a_simple_configuration():
    r = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    v = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    m = np.array([1.0, 3.0])
    assert kinetic_energy(v, m) == pytest.approx(2.0)
    assert potential_energy(r, m) == pytest.approx(-1.5)
    assert total_energy(r, v, m) == pytest.approx(0.5)


def test_com_frame(random_system):
    r, v, m = random_system
    r2, v2 = to_com_frame(r, v, m)
    np.testing.assert_allclose(center_of_mass(r2, m), 0.0, atol=1e-12)
    np.testing.assert_allclose(momentum(v2, m), 0.0, atol=1e-12)
    # Relative positions, hence forces, are unchanged.
    np.testing.assert_allclose(accelerations(r2, m), accelerations(r, m))


def test_angular_momentum_of_a_circular_orbit():
    r = np.array([[1.0, 0.0, 0.0]])
    v = np.array([[0.0, 2.0, 0.0]])
    np.testing.assert_allclose(angular_momentum(r, v, np.array([3.0])), [0.0, 0.0, 6.0])


def test_batched_inputs_match_single_calls():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(5, 3, 3))
    v = rng.normal(size=(5, 3, 3))
    m = np.array([1.0, 2.0, 3.0])
    a = accelerations(r, m)
    e = total_energy(r, v, m)
    for k in range(5):
        np.testing.assert_allclose(a[k], accelerations(r[k], m))
        assert e[k] == pytest.approx(total_energy(r[k], v[k], m))


def test_figure_eight_starts_at_rest_in_com_frame():
    ic = figure_eight()
    np.testing.assert_allclose(momentum(ic.v, ic.m), 0.0, atol=1e-12)
    np.testing.assert_allclose(center_of_mass(ic.r, ic.m), 0.0, atol=1e-12)
    # The figure-eight orbit has zero angular momentum.
    np.testing.assert_allclose(angular_momentum(ic.r, ic.v, ic.m), 0.0, atol=1e-12)
