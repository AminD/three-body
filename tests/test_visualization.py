import matplotlib

matplotlib.use("Agg")  # no window during the tests

import matplotlib.pyplot as plt
import numpy as np

from threebody import figure_eight, integrate_adaptive
from threebody.analysis import energy_comparison
from threebody.plots import dark_style, plot_energy_error, plot_escape_map
from threebody.visualization import animate_trajectory, plot_orbits, save_gif, view_radius


def test_animation_can_be_saved(tmp_path):
    traj = integrate_adaptive(figure_eight(), 2.0, n_out=60)
    anim = animate_trajectory(traj, duration=0.2, fps=10)
    path = tmp_path / "test.gif"
    save_gif(anim, path, fps=10, dpi=40)
    assert path.stat().st_size > 0
    plt.close("all")


def test_view_radius_ignores_an_ejected_body():
    r = np.zeros((1000, 3, 3))
    r[:, 0, 0] = 1.0
    r[-10:, 1, 0] = 500.0  # body 2 flies away at the very end
    assert view_radius(r) < 5.0
    # Without ejection the cube contains everything.
    assert view_radius(r[:900]) >= 1.0


def test_static_plots_render():
    traj = integrate_adaptive(figure_eight(), 2.0, n_out=60)
    plot_orbits(traj)
    plot_energy_error(energy_comparison(figure_eight(), 1.0, 0.01, n_out=50))
    xs = np.linspace(0.1, 1.0, 4)
    t_esc = np.full((4, 4), 3.0)
    t_esc[0, 0] = np.nan
    fig = plot_escape_map(xs, xs, t_esc, t_max=40.0)
    assert fig.axes[0].images[0].get_array().shape == (8, 8)  # mirrored quadrant
    plt.close("all")


def test_dark_style_restores_only_what_it_changed():
    # Regression test: restoring every setting (as plt.rc_context does) made
    # Jupyter close the figures of the following notebook cells.
    before = dict(plt.rcParams)
    with dark_style():
        assert plt.rcParams["axes.facecolor"] != before["axes.facecolor"]
    after = dict(plt.rcParams)
    assert after == before
