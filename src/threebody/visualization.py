"""3D animation of a trajectory: black background, glowing fading trails."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .state import Trajectory

BACKGROUND = "#000000"
BODY_COLORS = ("#3987e5", "#d95926", "#199e70")  # blue, orange, aqua
TEXT_COLOR = "#c3c2b7"


def view_radius(r: np.ndarray) -> float:
    """Half-width of the cube shown on screen.

    Normally the cube contains the whole trajectory. If a body is ejected far
    away, the view stays on the region where most of the action happens, so
    that the rest is not shrunk to a dot.
    """
    size = np.abs(r)
    return 1.05 * float(min(size.max(), 2.0 * np.quantile(size, 0.95))) + 1e-9


def make_scene(
    radius: float,
    figsize: float = 6.0,
    stars: bool = True,
    seed: int = 1,
    fig=None,
    rect=(0.0, 0.0, 1.0, 1.0),
):
    """Create the scene: a star field behind a borderless 3D axes.

    By default a new square figure is created. Pass ``fig`` and ``rect``
    (left, bottom, width, height) to place the scene inside an existing figure.
    """
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize), facecolor=BACKGROUND)
    if stars:
        rng = np.random.default_rng(seed)
        sky = fig.add_axes(rect, facecolor=BACKGROUND)
        n_stars = 220
        sky.scatter(
            rng.random(n_stars),
            rng.random(n_stars),
            s=rng.uniform(0.1, 2.2, n_stars),
            c="white",
            alpha=rng.uniform(0.15, 0.7, n_stars),
            linewidths=0,
        )
        sky.set_xlim(0, 1)
        sky.set_ylim(0, 1)
        sky.set_axis_off()
    ax = fig.add_axes(rect, projection="3d", facecolor="none")
    ax.set_axis_off()
    set_view_radius(ax, radius)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    return fig, ax


def set_view_radius(ax, radius: float) -> None:
    """Show the cube [-radius, radius]^3."""
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)


def _fading_colors(color: str, n: int, max_alpha: float) -> np.ndarray:
    """RGBA colours for n segments, transparent (old) to opaque (recent)."""
    rgba = np.empty((n, 4))
    rgba[:, :3] = to_rgb(color)
    rgba[:, 3] = max_alpha * np.linspace(0.0, 1.0, n) ** 1.5
    return rgba


def animate_trajectory(
    traj: Trajectory,
    duration: float = 12.0,
    fps: int = 25,
    trail: float = 0.15,
    rotation: float = 120.0,
    elevation: float = 25.0,
    stars: bool = True,
    show_time: bool = True,
    figsize: float = 6.0,
) -> FuncAnimation:
    """Build the animation of a trajectory.

    Parameters
    ----------
    duration : length of the animation in seconds.
    fps : frames per second.
    trail : length of the trails, as a fraction of the whole simulation.
    rotation : total rotation of the camera around the vertical axis (degrees).
    elevation : height of the camera above the x-y plane (degrees).
    """
    r = traj.r
    n_samples, n_bodies, _ = r.shape
    n_frames = int(duration * fps)
    frame_index = np.linspace(0, n_samples - 1, n_frames).astype(int)
    trail_len = max(2, int(trail * n_samples))
    colors = [BODY_COLORS[b % len(BODY_COLORS)] for b in range(n_bodies)]

    fig, ax = make_scene(view_radius(r), figsize=figsize, stars=stars)

    # Each trail is drawn twice: a wide faint line (the glow) under a thin one.
    glows, cores, halos, dots, sparks = [], [], [], [], []
    empty = np.zeros((1, 2, 3))  # placeholder segment, replaced at the first frame
    for color in colors:
        glow = Line3DCollection(empty, linewidths=5.0, capstyle="round")
        core = Line3DCollection(empty, linewidths=1.4, capstyle="round")
        ax.add_collection3d(glow)
        ax.add_collection3d(core)
        glows.append(glow)
        cores.append(core)
        # Same marker sizes for every body: three identical points.
        (halo,) = ax.plot([], [], [], "o", ms=17, color=color, alpha=0.22, mew=0)
        (dot,) = ax.plot([], [], [], "o", ms=8, color=color, mew=0)
        (spark,) = ax.plot([], [], [], "o", ms=3, color="white", mew=0)
        halos.append(halo)
        dots.append(dot)
        sparks.append(spark)

    set_view_radius(ax, view_radius(r))  # adding artists may have moved the limits

    label = fig.text(
        0.03, 0.03, "", color=TEXT_COLOR, fontsize=9, family="monospace", alpha=0.8
    )

    def update(frame: int):
        i = frame_index[frame]
        j0 = max(0, i - trail_len)
        for b in range(n_bodies):
            pts = r[j0 : i + 1, b]
            segments = np.stack([pts[:-1], pts[1:]], axis=1)
            k = len(segments)
            glows[b].set_segments(segments)
            cores[b].set_segments(segments)
            if k:
                glows[b].set_color(_fading_colors(colors[b], k, 0.18))
                cores[b].set_color(_fading_colors(colors[b], k, 0.95))
            x, y, z = r[i, b]
            for marker in (halos[b], dots[b], sparks[b]):
                marker.set_data_3d([x], [y], [z])
        ax.view_init(elev=elevation, azim=-60.0 + rotation * frame / n_frames)
        if show_time:
            label.set_text(f"t = {traj.t[i]:6.2f}")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000.0 / fps, blit=False)
    return anim


def save_gif(anim: FuncAnimation, path: str | Path, fps: int = 25, dpi: int = 80) -> None:
    """Save an animation as a GIF (needs nothing beyond matplotlib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BACKGROUND})


def plot_orbits(traj: Trajectory, ax=None, radius: float | None = None):
    """Static picture of the full orbits (used by the interactive explorer)."""
    r = traj.r
    if ax is None:
        _, ax = make_scene(view_radius(r) if radius is None else radius)
    n = len(r)
    for b in range(r.shape[1]):
        color = BODY_COLORS[b % len(BODY_COLORS)]
        segments = np.stack([r[:-1, b], r[1:, b]], axis=1)
        line = Line3DCollection(segments, linewidths=1.2, capstyle="round")
        rgba = _fading_colors(color, n - 1, 0.8)
        rgba[:, 3] += 0.2  # keep the oldest part of the orbit faintly visible
        line.set_color(rgba)
        ax.add_collection3d(line)
        # Hollow marker = start, filled marker = end.
        ax.plot(*r[0, b, :, None], "o", ms=7, mfc="none", mec=color, mew=1.5)
        ax.plot(*r[-1, b, :, None], "o", ms=8, color=color, mew=0)
    return ax
