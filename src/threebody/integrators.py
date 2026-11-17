"""Time integrators for the N-body equations of motion.

Three fixed-step schemes written by hand:

* ``euler``  - explicit Euler, order 1. Simple, but the energy drifts.
* ``rk4``    - classical Runge-Kutta, order 4. Accurate on short times.
* ``verlet`` - velocity Verlet (leapfrog), order 2. Symplectic and
  time-reversible: the energy error stays bounded over long times.

And one adaptive reference: ``integrate_adaptive`` wraps SciPy's DOP853
(Runge-Kutta of order 8 with automatic step-size control).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .physics import accelerations
from .state import InitialConditions, Trajectory

# Each stepper takes (r, v, a, m, dt, eps), where a is the acceleration at r,
# and returns (r_new, v_new, a_new). Passing the acceleration around avoids
# computing it twice (Verlet then needs a single force evaluation per step).


def euler_step(r, v, a, m, dt, eps):
    """Explicit Euler: use the slope at the start of the step."""
    r_new = r + dt * v
    v_new = v + dt * a
    return r_new, v_new, accelerations(r_new, m, eps)


def rk4_step(r, v, a, m, dt, eps):
    """Classical 4th-order Runge-Kutta on the system (r, v)' = (v, a(r))."""
    k1r, k1v = v, a
    k2r = v + 0.5 * dt * k1v
    k2v = accelerations(r + 0.5 * dt * k1r, m, eps)
    k3r = v + 0.5 * dt * k2v
    k3v = accelerations(r + 0.5 * dt * k2r, m, eps)
    k4r = v + dt * k3v
    k4v = accelerations(r + dt * k3r, m, eps)
    r_new = r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
    v_new = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return r_new, v_new, accelerations(r_new, m, eps)


def verlet_step(r, v, a, m, dt, eps):
    """Velocity Verlet: half kick, full drift, half kick."""
    v_half = v + 0.5 * dt * a
    r_new = r + dt * v_half
    a_new = accelerations(r_new, m, eps)
    v_new = v_half + 0.5 * dt * a_new
    return r_new, v_new, a_new


STEPPERS = {"euler": euler_step, "rk4": rk4_step, "verlet": verlet_step}

# Theoretical order of accuracy of each fixed-step scheme.
ORDERS = {"euler": 1, "rk4": 4, "verlet": 2}


def integrate(
    ic: InitialConditions,
    t_end: float | None = None,
    dt: float = 1e-3,
    method: str = "verlet",
    softening: float | None = None,
    save_every: int = 1,
) -> Trajectory:
    """Integrate with a fixed time step.

    Parameters
    ----------
    ic : initial conditions.
    t_end : final time (defaults to ``ic.t_end``).
    dt : time step.
    method : "euler", "rk4" or "verlet".
    softening : softening length (defaults to ``ic.softening``).
    save_every : store one state every ``save_every`` steps.
    """
    if method not in STEPPERS:
        raise ValueError(f"unknown method {method!r}, choose from {sorted(STEPPERS)}")
    step = STEPPERS[method]
    t_end = ic.t_end if t_end is None else t_end
    eps = ic.softening if softening is None else softening

    n_steps = int(round(t_end / dt))
    n_saved = n_steps // save_every + 1
    n = ic.m.shape[0]
    r_out = np.empty((n_saved, n, 3))
    v_out = np.empty((n_saved, n, 3))
    t_out = np.empty(n_saved)

    r, v, m = ic.r.copy(), ic.v.copy(), ic.m
    a = accelerations(r, m, eps)
    r_out[0], v_out[0], t_out[0] = r, v, 0.0
    k = 1
    for i in range(1, n_steps + 1):
        r, v, a = step(r, v, a, m, dt, eps)
        if i % save_every == 0:
            r_out[k], v_out[k], t_out[k] = r, v, i * dt
            k += 1
    return Trajectory(t_out, r_out, v_out, m, eps, method, ic.name)


class _BudgetExceeded(Exception):
    """Raised internally when the force-evaluation budget is used up."""


def integrate_adaptive(
    ic: InitialConditions,
    t_end: float | None = None,
    n_out: int = 2000,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    softening: float | None = None,
    max_nfev: int | None = None,
    method: str = "DOP853",
) -> Trajectory:
    """Integrate with SciPy's adaptive solver (DOP853 by default).

    The solver shrinks its step automatically during close encounters, which
    is what makes it a good reference solution.

    Parameters
    ----------
    n_out : number of equally spaced output times.
    rtol, atol : relative and absolute error tolerances.
    max_nfev : optional budget of force evaluations. If a very close
        encounter makes the solver crawl, the integration stops early and the
        returned trajectory has ``completed = False``.
    """
    t_end = ic.t_end if t_end is None else t_end
    eps = ic.softening if softening is None else softening
    m = ic.m
    n = m.shape[0]
    nfev = 0

    def rhs(_t, y):
        nonlocal nfev
        nfev += 1
        if max_nfev is not None and nfev > max_nfev:
            raise _BudgetExceeded
        y = y.reshape(2, n, 3)
        return np.concatenate([y[1].ravel(), accelerations(y[0], m, eps).ravel()])

    t_eval = np.linspace(0.0, t_end, n_out)
    # With a budget we integrate chunk by chunk, so that whatever was computed
    # before the budget ran out can still be returned.
    n_chunks = 1 if max_nfev is None else 25
    edges = np.unique(np.linspace(0, n_out - 1, n_chunks + 1).astype(int))

    y = np.concatenate([ic.r.ravel(), ic.v.ravel()])
    states = [y]
    completed = True
    for i0, i1 in zip(edges[:-1], edges[1:]):
        try:
            sol = solve_ivp(
                rhs,
                (t_eval[i0], t_eval[i1]),
                y,
                method=method,
                t_eval=t_eval[i0 + 1 : i1 + 1],
                rtol=rtol,
                atol=atol,
            )
        except _BudgetExceeded:
            completed = False
            break
        if not sol.success:
            completed = False
            break
        states.extend(sol.y.T)
        y = sol.y[:, -1]

    states = np.array(states).reshape(-1, 2, n, 3)
    return Trajectory(
        t_eval[: len(states)],
        states[:, 0],
        states[:, 1],
        m,
        eps,
        method.lower(),
        ic.name,
        completed,
    )
