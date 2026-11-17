"""Interactive explorer: move the sliders, watch the orbits change.

    python scripts/interactive.py
    python scripts/interactive.py --config configs/custom.toml

Body 1 keeps the initial conditions written in the configuration file.
The sliders set the initial position and velocity of bodies 2 and 3.
"Save config" writes the current values back to the file, so that

    python scripts/animate.py --config configs/custom.toml

animates exactly what is on screen.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from threebody import InitialConditions, integrate_adaptive
from threebody.physics import to_com_frame, total_energy
from threebody.scenarios import read_config, save_config
from threebody.visualization import (
    BACKGROUND,
    BODY_COLORS,
    TEXT_COLOR,
    make_scene,
    plot_orbits,
    set_view_radius,
    view_radius,
)

POSITION_RANGE = 2.0
VELOCITY_RANGE = 1.5


class Explorer:
    """Figure with a 3D view on the left and twelve sliders on the right."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.cfg = read_config(config_path)
        self.fig = plt.figure(figsize=(12, 7), facecolor=BACKGROUND)
        _, self.ax = make_scene(1.0, fig=self.fig, rect=(0.0, 0.0, 0.6, 1.0))
        self.status = self.fig.text(
            0.02, 0.03, "", color=TEXT_COLOR, fontsize=9, family="monospace"
        )
        self.fig.text(
            0.66,
            0.115,
            "Body 1: fixed initial conditions (from the config file)",
            color=BODY_COLORS[0],
            fontsize=9,
        )
        self.sliders: dict[tuple[int, str, int], Slider] = {}
        self._updating = False
        self._build_sliders()
        self._build_buttons()
        self.update()

    def _build_sliders(self) -> None:
        top, row = 0.93, 0.055
        for block, body in enumerate((1, 2)):  # array rows of bodies 2 and 3
            y0 = top - block * 7.3 * row
            self.fig.text(
                0.66,
                y0,
                f"Body {body + 1}",
                color=BODY_COLORS[body],
                fontsize=11,
                fontweight="bold",
            )
            for k, (kind, axis) in enumerate(
                [(kind, axis) for kind in ("r", "v") for axis in range(3)]
            ):
                limit = POSITION_RANGE if kind == "r" else VELOCITY_RANGE
                label = ("" if kind == "r" else "v") + "xyz"[axis]
                ax = self.fig.add_axes([0.70, y0 - (k + 1) * row, 0.22, 0.025])
                ax.set_facecolor("#1a1a19")
                slider = Slider(
                    ax,
                    label,
                    -limit,
                    limit,
                    valinit=float(self.cfg[kind][body, axis]),
                    color=BODY_COLORS[body],
                    track_color="#383835",
                )
                slider.label.set_color(TEXT_COLOR)
                slider.valtext.set_color(TEXT_COLOR)
                slider.on_changed(self.update)
                self.sliders[(body, kind, axis)] = slider

    def _build_buttons(self) -> None:
        self.buttons = []
        for x, text, callback in (
            (0.70, "Reset", self.reset),
            (0.82, "Save config", self.save),
        ):
            button = Button(
                self.fig.add_axes([x, 0.04, 0.10, 0.05]),
                text,
                color="#1a1a19",
                hovercolor="#383835",
            )
            button.label.set_color(TEXT_COLOR)
            button.on_clicked(callback)
            self.buttons.append(button)

    def current_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions and velocities as written in the file / set by the sliders."""
        r, v = self.cfg["r"].copy(), self.cfg["v"].copy()
        for (body, kind, axis), slider in self.sliders.items():
            (r if kind == "r" else v)[body, axis] = slider.val
        return r, v

    def update(self, _event=None) -> None:
        if self._updating:
            return
        r, v = self.current_state()
        m = self.cfg["m"]
        if self.cfg["recenter"]:
            r, v = to_com_frame(r, v, m)
        ic = InitialConditions(
            "explorer", r, v, m, t_end=self.cfg["t_end"], softening=self.cfg["softening"]
        )
        # Loose tolerance and a budget of force evaluations keep the view responsive.
        traj = integrate_adaptive(ic, n_out=1000, rtol=1e-8, atol=1e-10, max_nfev=40_000)

        for artist in list(self.ax.collections) + list(self.ax.lines):
            artist.remove()
        plot_orbits(traj, ax=self.ax)
        set_view_radius(self.ax, view_radius(traj.r))

        energy = total_energy(r, v, m, ic.softening)
        text = f"E = {energy:+.3f} ({'bound' if energy < 0 else 'unbound'})   t = {traj.t[-1]:.1f}"
        if not traj.completed:
            text += "   stopped early: very close encounter"
        self.status.set_text(text)
        self.fig.canvas.draw_idle()

    def reset(self, _event=None) -> None:
        self._updating = True  # redraw once at the end, not twelve times
        for slider in self.sliders.values():
            slider.reset()
        self._updating = False
        self.update()

    def save(self, _event=None) -> None:
        r, v = self.current_state()
        save_config(
            self.config_path,
            r,
            v,
            self.cfg["m"],
            self.cfg["t_end"],
            self.cfg["softening"],
            self.cfg["name"],
        )
        self.status.set_text(f"saved to {self.config_path}")
        self.fig.canvas.draw_idle()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/custom.toml")
    args = parser.parse_args()
    explorer = Explorer(args.config)  # keep a reference so the widgets stay alive
    plt.show()
    del explorer


if __name__ == "__main__":
    main()
