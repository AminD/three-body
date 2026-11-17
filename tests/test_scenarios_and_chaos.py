from pathlib import Path

import numpy as np
import pytest

from threebody import SCENARIOS, chaotic, figure_eight, load_config
from threebody.chaos import _pair_forces, divergence, escape_time_map, lyapunov_estimate
from threebody.physics import accelerations, momentum, total_energy
from threebody.scenarios import free_fall, random_bound, read_config, save_config


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenarios_are_bound_and_do_not_drift(name):
    ic = SCENARIOS[name]()
    assert ic.r.shape == ic.v.shape == (3, 3)
    np.testing.assert_allclose(ic.m, 1.0)  # three identical bodies
    np.testing.assert_allclose(momentum(ic.v, ic.m), 0.0, atol=1e-12)
    assert total_energy(ic.r, ic.v, ic.m) < 0.0


def test_random_bound_is_reproducible():
    a, b = random_bound(seed=7), random_bound(seed=7)
    np.testing.assert_array_equal(a.r, b.r)
    np.testing.assert_array_equal(a.v, b.v)
    assert not np.allclose(a.r, random_bound(seed=8).r)


def test_config_round_trip(tmp_path):
    path = tmp_path / "ic.toml"
    r = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.25], [-1.0, 0.125, 0.75]])
    v = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [-0.3, 0.2, -0.1]])
    m = np.ones(3)
    save_config(path, r, v, m, t_end=12.5, softening=0.01, name="demo")

    raw = read_config(path)
    np.testing.assert_allclose(raw["r"], r)
    np.testing.assert_allclose(raw["v"], v)
    assert raw["t_end"] == 12.5 and raw["softening"] == 0.01 and raw["name"] == "demo"

    ic = load_config(path)  # recentred on the centre of mass
    np.testing.assert_allclose(momentum(ic.v, ic.m), 0.0, atol=1e-12)
    # Recentring does not change relative positions.
    np.testing.assert_allclose(ic.r[1] - ic.r[0], r[1] - r[0])


def test_default_config_loads():
    ic = load_config(Path(__file__).resolve().parents[1] / "configs" / "custom.toml")
    assert total_energy(ic.r, ic.v, ic.m) < 0.0


def test_chaotic_start_diverges_much_faster_than_figure_eight():
    _, d_chaos = divergence(chaotic(), delta=1e-8, t_end=40.0, n_out=200)
    _, d_eight = divergence(figure_eight(), delta=1e-8, t_end=40.0, n_out=200)
    assert d_chaos[0] == pytest.approx(1e-8)
    assert d_chaos[-1] > 1e4 * d_chaos[0]
    assert d_chaos[-1] > 100 * d_eight[-1]


def test_lyapunov_exponent_is_larger_for_the_chaotic_start():
    _, lam_chaos = lyapunov_estimate(chaotic(), t_end=40.0)
    _, lam_eight = lyapunov_estimate(figure_eight(), t_end=40.0)
    assert lam_chaos[-1] > 0.2
    assert lam_chaos[-1] > 2 * lam_eight[-1]


def test_pair_forces_match_general_accelerations():
    r = np.random.default_rng(1).normal(size=(20, 3, 3))
    a, d2_min = _pair_forces(r, 0.01)
    np.testing.assert_allclose(a, accelerations(r, np.ones(3), 0.01))
    assert d2_min.shape == (20,)


def test_escape_time_map_small_grid():
    xs = np.array([0.1, 0.4, 0.9])
    ys = np.array([0.3, 0.7])
    t_esc = escape_time_map(xs, ys, t_max=20.0)
    assert t_esc.shape == (2, 3)
    finite = t_esc[~np.isnan(t_esc)]
    assert finite.size > 0
    assert np.all((finite > 0.0) & (finite < 21.0))


def test_free_fall_starts_at_rest():
    ic = free_fall(0.2, 0.6)
    np.testing.assert_allclose(ic.v, 0.0)
    np.testing.assert_allclose(ic.r[1] - ic.r[0], [1.0, 0.0, 0.0])
