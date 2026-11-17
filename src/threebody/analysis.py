"""Numerical experiments comparing the integrators."""

from __future__ import annotations

import numpy as np

from .integrators import STEPPERS, integrate, integrate_adaptive
from .state import InitialConditions, Trajectory


def energy_comparison(
    ic: InitialConditions,
    t_end: float,
    dt: float,
    methods: tuple[str, ...] = ("euler", "rk4", "verlet"),
    n_out: int = 2000,
) -> dict[str, Trajectory]:
    """Run the same problem with every fixed-step method (same dt) and with
    the adaptive DOP853 reference."""
    n_steps = int(round(t_end / dt))
    save_every = max(1, n_steps // n_out)
    out = {
        name: integrate(ic, t_end, dt, method=name, save_every=save_every)
        for name in methods
    }
    out["dop853"] = integrate_adaptive(ic, t_end, n_out=n_out)
    return out


def convergence_study(
    ic: InitialConditions,
    t_end: float,
    dts: np.ndarray,
    methods: tuple[str, ...] = ("euler", "rk4", "verlet"),
) -> dict[str, np.ndarray]:
    """Position error at ``t_end`` as a function of the time step.

    The error is measured against a very accurate DOP853 solution. For a
    method of order p the error behaves like C * dt^p: a straight line of
    slope p on a log-log plot.
    """
    ref = integrate_adaptive(ic, t_end, n_out=2, rtol=1e-13, atol=1e-14)
    errors = {}
    for name in methods:
        if name not in STEPPERS:
            raise ValueError(f"unknown method {name!r}")
        errs = []
        for dt in dts:
            n_steps = int(round(t_end / dt))
            traj = integrate(ic, t_end, dt, method=name, save_every=n_steps)
            errs.append(np.linalg.norm(traj.r[-1] - ref.r[-1]))
        errors[name] = np.array(errs)
    return errors


def observed_order(dts: np.ndarray, errors: np.ndarray) -> float:
    """Slope of log(error) against log(dt) (least-squares fit)."""
    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    return float(slope)
