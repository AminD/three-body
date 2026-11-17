"""Diagnostic figures (dark theme, consistent with the animation)."""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .analysis import observed_order
from .physics import relative_energy_error

SURFACE = "#1a1a19"
TEXT_PRIMARY = "#f0efec"
TEXT_SECONDARY = "#c3c2b7"
GRID = "#383835"

# One fixed colour per method / scenario: the colour follows the entity.
SERIES_COLORS = {
    "euler": "#3987e5",
    "rk4": "#d95926",
    "verlet": "#199e70",
    "dop853": "#c98500",
    "chaotic": "#3987e5",
    "figure_eight": "#d95926",
    "lagrange": "#199e70",
}
SERIES_LABELS = {
    "euler": "Euler",
    "rk4": "RK4",
    "verlet": "Verlet",
    "dop853": "DOP853 (adaptive)",
    "chaotic": "Chaotic",
    "figure_eight": "Figure-eight",
    "lagrange": "Lagrange",
}
_FALLBACK = ["#3987e5", "#d95926", "#199e70", "#c98500"]

# Single-hue ramp for magnitudes: dark blue (low) to pale blue (high).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#0d366b", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"]
)


@contextmanager
def dark_style():
    """Temporarily apply the dark chart style."""
    style = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_SECONDARY,
        "font.size": 10,
    }
    # Only the keys above are saved and restored. plt.rc_context would restore
    # every setting, including the backend, which makes Jupyter's inline
    # backend close (and never display) the figures of the following cells.
    saved = {key: plt.rcParams[key] for key in style}
    plt.rcParams.update(style)
    try:
        yield
    finally:
        plt.rcParams.update(saved)


def _color(key: str, i: int) -> str:
    return SERIES_COLORS.get(key, _FALLBACK[i % len(_FALLBACK)])


def _label(key: str) -> str:
    return SERIES_LABELS.get(key, key)


def plot_energy_error(trajs: dict, title: str = "Relative energy error"):
    """|E(t) - E(0)| / |E(0)| against time, one line per method."""
    with dark_style():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (key, traj) in enumerate(trajs.items()):
            err = np.maximum(relative_energy_error(traj), 1e-17)
            ax.plot(traj.t[1:], err[1:], color=_color(key, i), label=_label(key))
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("|E(t) − E(0)| / |E(0)|")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=4)
        fig.tight_layout()
    return fig


def plot_convergence(dts, errors: dict, title: str = "Convergence of the integrators"):
    """Position error against the time step on a log-log plot."""
    with dark_style():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (key, err) in enumerate(errors.items()):
            order = observed_order(np.asarray(dts), err)
            ax.plot(
                dts,
                err,
                "o-",
                color=_color(key, i),
                mec=SURFACE,
                mew=2,
                label=f"{_label(key)} (slope {order:.2f})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time step dt")
        ax.set_ylabel("Position error at the final time")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
        fig.tight_layout()
    return fig


def plot_divergence(curves: dict, title: str = "Divergence of two nearby trajectories"):
    """Phase-space distance between twin simulations, one line per scenario.

    ``curves`` maps a scenario name to a pair (t, distance).
    """
    with dark_style():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (key, (t, d)) in enumerate(curves.items()):
            ax.plot(t, d, color=_color(key, i), label=_label(key))
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance in phase space")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
        fig.tight_layout()
    return fig


def plot_lyapunov(curves: dict, title: str = "Largest Lyapunov exponent (running estimate)"):
    """Running Benettin estimate lambda(t), one line per scenario."""
    with dark_style():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (key, (t, lam)) in enumerate(curves.items()):
            ax.plot(t, lam, color=_color(key, i), label=_label(key))
            ax.annotate(
                f"{lam[-1]:.2f}",
                (t[-1], lam[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=TEXT_PRIMARY,
            )
        ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("λ(t)")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
        fig.tight_layout()
    return fig


def plot_escape_map(xs, ys, t_escape, t_max: float, mirror: bool = True):
    """Escape-time map of the free-fall three-body problem.

    With ``mirror=True`` the grid is assumed to cover one quadrant
    (x >= 0, y >= 0) and the picture is completed by symmetry.
    """
    # Systems where nobody escaped get the largest value: they lived longest.
    z = np.log10(np.where(np.isnan(t_escape), t_max, t_escape))
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    if mirror:
        z = np.block([[z[::-1, ::-1], z[::-1, :]], [z[:, ::-1], z]])
        extent = [-xs[-1], xs[-1], -ys[-1], ys[-1]]
    with dark_style():
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        image = ax.imshow(
            z, origin="lower", extent=extent, cmap=SEQUENTIAL, interpolation="nearest"
        )
        ax.plot([-0.5, 0.5], [0, 0], "o", ms=8, color=TEXT_PRIMARY, mec=SURFACE, mew=2)
        ax.grid(False)
        ax.set_xlabel("Initial x of body 3")
        ax.set_ylabel("Initial y of body 3")
        ax.set_title(
            "Time before a body is ejected (three bodies released at rest)\n"
            "White dots: bodies 1 and 2",
        )
        bar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.03)
        bar.set_label(f"log10 of the escape time (brightest: none before t = {t_max:g})")
        bar.outline.set_edgecolor(GRID)
        fig.tight_layout()
    return fig
