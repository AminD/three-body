"""Compare Euler, RK4 and Verlet on the figure-eight orbit.

    python scripts/compare_integrators.py

Writes figures/energy_error.png and figures/convergence.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from threebody import figure_eight
from threebody.analysis import convergence_study, energy_comparison, observed_order
from threebody.physics import relative_energy_error
from threebody.plots import plot_convergence, plot_energy_error
from threebody.scenarios import FIGURE_EIGHT_PERIOD

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ic = figure_eight()

    # 1) Energy conservation over 30 periods, same time step for everyone.
    t_end, dt = 30 * FIGURE_EIGHT_PERIOD, 0.01
    trajs = energy_comparison(ic, t_end, dt)
    print(f"Energy error after {t_end:.0f} time units (dt = {dt}):")
    for name, traj in trajs.items():
        print(f"  {name:8s} max = {relative_energy_error(traj).max():.2e}")
    fig = plot_energy_error(
        trajs, title=f"Energy error on the figure-eight orbit (fixed dt = {dt})"
    )
    fig.savefig(FIGURES / "energy_error.png", dpi=150)

    # 2) Order of convergence: error after one time unit against dt.
    dts = 0.02 / 2 ** np.arange(6)
    errors = convergence_study(ic, 1.0, dts)
    print("Observed order of convergence:")
    for name, err in errors.items():
        print(f"  {name:8s} {observed_order(dts, err):.2f}")
    fig = plot_convergence(dts, errors)
    fig.savefig(FIGURES / "convergence.png", dpi=150)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
