import numpy as np
import pytest

from threebody import InitialConditions, figure_eight, integrate, integrate_adaptive, lagrange
from threebody.analysis import convergence_study, observed_order
from threebody.integrators import ORDERS
from threebody.physics import (
    angular_momentum,
    momentum,
    relative_energy_error,
)
from threebody.scenarios import FIGURE_EIGHT_PERIOD


@pytest.fixture
def circular_binary():
    """Two unit masses on a circular orbit of diameter 1 (exact solution known)."""
    omega = np.sqrt(2.0)  # omega^2 = G (m1 + m2) / d^3
    r = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    v = np.array([[0.0, -0.5 * omega, 0.0], [0.0, 0.5 * omega, 0.0]])
    return InitialConditions("binary", r, v, np.ones(2), t_end=2 * np.pi / omega)


@pytest.mark.parametrize("method, tol", [("rk4", 1e-8), ("verlet", 1e-3)])
def test_circular_binary_closes_after_one_period(circular_binary, method, tol):
    traj = integrate(circular_binary, dt=circular_binary.t_end / 2000, method=method)
    np.testing.assert_allclose(traj.r[-1], circular_binary.r, atol=tol)


def test_adaptive_solver_on_circular_binary(circular_binary):
    traj = integrate_adaptive(circular_binary, n_out=50)
    assert traj.completed
    np.testing.assert_allclose(traj.r[-1], circular_binary.r, atol=1e-8)
    # The separation stays equal to 1 all along the orbit.
    sep = np.linalg.norm(traj.r[:, 0] - traj.r[:, 1], axis=-1)
    np.testing.assert_allclose(sep, 1.0, atol=1e-8)


@pytest.mark.parametrize("method", ["euler", "rk4", "verlet"])
def test_order_of_convergence(method):
    dts = 0.02 / 2 ** np.arange(4)
    errors = convergence_study(figure_eight(), 1.0, dts, methods=(method,))
    assert observed_order(dts, errors[method]) == pytest.approx(ORDERS[method], abs=0.25)


def test_figure_eight_is_periodic():
    ic = figure_eight()
    traj = integrate_adaptive(ic, FIGURE_EIGHT_PERIOD, n_out=2, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(traj.r[-1], ic.r, atol=1e-6)
    np.testing.assert_allclose(traj.v[-1], ic.v, atol=1e-6)


def test_lagrange_triangle_stays_equilateral_for_one_turn():
    ic = lagrange()
    traj = integrate_adaptive(ic, ic.t_end / 9, n_out=100)
    sides = np.stack(
        [np.linalg.norm(traj.r[:, i] - traj.r[:, j], axis=-1) for i, j in ((0, 1), (0, 2), (1, 2))]
    )
    np.testing.assert_allclose(sides, np.sqrt(3.0), atol=1e-6)
    np.testing.assert_allclose(traj.r[-1], ic.r, atol=1e-6)


@pytest.mark.parametrize("method", ["euler", "rk4", "verlet"])
def test_momentum_is_conserved(method):
    traj = integrate(figure_eight(), t_end=5.0, dt=0.01, method=method)
    np.testing.assert_allclose(momentum(traj.v, traj.m), 0.0, atol=1e-11)


def test_verlet_conserves_angular_momentum():
    # Use a start with non-zero angular momentum.
    ic = lagrange()
    traj = integrate(ic, t_end=10.0, dt=0.01, method="verlet")
    L = angular_momentum(traj.r, traj.v, traj.m)
    np.testing.assert_allclose(L, np.broadcast_to(L[0], L.shape), atol=1e-11)


def test_verlet_energy_error_stays_bounded_while_euler_drifts():
    ic = figure_eight()
    t_end, dt = 10 * FIGURE_EIGHT_PERIOD, 0.01
    verlet = relative_energy_error(integrate(ic, t_end, dt, "verlet"))
    euler = relative_energy_error(integrate(ic, t_end, dt, "euler"))
    half = len(verlet) // 2
    # Verlet: no growth between the first and the second half of the run.
    assert verlet[half:].max() < 1.5 * verlet[:half].max()
    assert verlet.max() < 1e-3
    # Euler: orders of magnitude worse.
    assert euler.max() > 100 * verlet.max()


def test_verlet_is_time_reversible():
    ic = figure_eight()
    forward = integrate(ic, t_end=3.0, dt=0.01, method="verlet")
    # Flip the velocities and integrate back: we must return to the start.
    back_ic = InitialConditions("back", forward.r[-1], -forward.v[-1], ic.m)
    backward = integrate(back_ic, t_end=3.0, dt=0.01, method="verlet")
    np.testing.assert_allclose(backward.r[-1], ic.r, atol=1e-10)
    np.testing.assert_allclose(-backward.v[-1], ic.v, atol=1e-10)


def test_save_every_and_output_shapes():
    traj = integrate(figure_eight(), t_end=1.0, dt=0.01, save_every=10)
    assert traj.t.shape == (11,)
    assert traj.r.shape == traj.v.shape == (11, 3, 3)
    assert traj.t[-1] == pytest.approx(1.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        integrate(figure_eight(), t_end=1.0, method="magic")


def test_budget_stops_the_adaptive_solver_early():
    traj = integrate_adaptive(figure_eight(), 50.0, n_out=500, max_nfev=200)
    assert not traj.completed
    assert 0 < len(traj.t) < 500
    assert len(traj.t) == len(traj.r) == len(traj.v)
