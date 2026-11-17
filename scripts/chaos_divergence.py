"""Sensitivity to initial conditions: chaotic start versus figure-eight orbit.

    python scripts/chaos_divergence.py

Two simulations start 1e-8 apart; we follow the distance between them and
estimate the largest Lyapunov exponent.
Writes figures/divergence.png and figures/lyapunov.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from threebody import chaotic, figure_eight
from threebody.chaos import divergence, lyapunov_estimate
from threebody.plots import plot_divergence, plot_lyapunov

FIGURES = Path(__file__).resolve().parents[1] / "figures"
T_END = 60.0
DELTA = 1e-8


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    scenarios = {"chaotic": chaotic(), "figure_eight": figure_eight()}

    curves = {name: divergence(ic, DELTA, T_END) for name, ic in scenarios.items()}
    for name, (_, d) in curves.items():
        print(f"{name:13s} distance grows from {d[0]:.1e} to {d[-1]:.1e}")
    fig = plot_divergence(
        curves, title=f"Two simulations started {DELTA:g} apart: distance over time"
    )
    fig.savefig(FIGURES / "divergence.png", dpi=150)

    lyap = {name: lyapunov_estimate(ic, T_END) for name, ic in scenarios.items()}
    for name, (_, lam) in lyap.items():
        print(f"{name:13s} Lyapunov exponent ~ {lam[-1]:.3f}")
    fig = plot_lyapunov(lyap)
    fig.savefig(FIGURES / "lyapunov.png", dpi=150)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
