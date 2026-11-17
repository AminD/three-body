"""Tools to study the sensitivity to initial conditions."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .physics import G, accelerations, to_com_frame
from .state import InitialConditions


def _twin_rhs(m: np.ndarray, eps: float):
    """Right-hand side for two copies of the system integrated together.

    The state has shape (2 copies, 2 [r or v], N, 3), flattened. Integrating
    both copies in a single call makes them share the same time steps.
    """
    n = m.shape[0]

    def rhs(_t, y):
        y = y.reshape(2, 2, n, 3)
        out = np.empty_like(y)
        out[:, 0] = y[:, 1]
        out[:, 1] = accelerations(y[:, 0], m, eps)
        return out.ravel()

    return rhs


def _twin_state(ic: InitialConditions, delta: float) -> np.ndarray:
    """Reference state and a copy with body 3 shifted by ``delta`` along x."""
    y = np.empty((2, 2, ic.m.shape[0], 3))
    y[:, 0] = ic.r
    y[:, 1] = ic.v
    y[1, 0, -1, 0] += delta
    return y


def divergence(
    ic: InitialConditions,
    delta: float = 1e-8,
    t_end: float | None = None,
    n_out: int = 1000,
    rtol: float = 1e-12,
    atol: float = 1e-13,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance between two simulations that start ``delta`` apart.

    Returns the times and the phase-space distance (positions and velocities
    together). For a chaotic system the distance grows roughly like
    ``delta * exp(lambda * t)``: a straight line on a log scale.
    """
    t_end = ic.t_end if t_end is None else t_end
    y0 = _twin_state(ic, delta)
    t_eval = np.linspace(0.0, t_end, n_out)
    sol = solve_ivp(
        _twin_rhs(ic.m, ic.softening),
        (0.0, t_end),
        y0.ravel(),
        method="DOP853",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    y = sol.y.T.reshape(-1, 2, 2 * ic.m.shape[0] * 3)
    return sol.t, np.linalg.norm(y[:, 1] - y[:, 0], axis=-1)


def lyapunov_estimate(
    ic: InitialConditions,
    t_end: float | None = None,
    tau: float = 0.5,
    d0: float = 1e-8,
    rtol: float = 1e-11,
    atol: float = 1e-13,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest Lyapunov exponent, estimated with Benettin's method.

    Two neighbouring trajectories are followed for a short time ``tau``. The
    growth of their separation is recorded, then the neighbour is pulled back
    to a distance ``d0`` (keeping the direction) and the process repeats:

        lambda(t) = (1 / t) * sum_k ln(d_k / d0)

    Returns the times and the running estimate lambda(t). This is a
    finite-time estimate: if a body gets ejected the motion stops being
    chaotic and lambda(t) then slowly decays.
    """
    t_end = ic.t_end if t_end is None else t_end
    rhs = _twin_rhs(ic.m, ic.softening)
    y = _twin_state(ic, d0)
    n_intervals = int(round(t_end / tau))
    times = np.empty(n_intervals)
    lam = np.empty(n_intervals)
    log_sum = 0.0
    for k in range(n_intervals):
        sol = solve_ivp(rhs, (0.0, tau), y.ravel(), method="DOP853", rtol=rtol, atol=atol)
        y = sol.y[:, -1].reshape(y.shape)
        diff = y[1] - y[0]
        d = np.linalg.norm(diff)
        log_sum += np.log(d / d0)
        y[1] = y[0] + diff * (d0 / d)  # renormalise the separation
        times[k] = (k + 1) * tau
        lam[k] = log_sum / times[k]
    return times, lam


def _escaped(r: np.ndarray, v: np.ndarray, m: np.ndarray, d_escape: float) -> np.ndarray:
    """True for every system of the batch in which one body is escaping.

    Body i escapes when it is far from the other two, moving away from them,
    and too fast to come back (positive two-body energy with respect to the
    centre of mass of the other pair).
    """
    out = np.zeros(r.shape[0], dtype=bool)
    for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        m_pair = m[j] + m[k]
        rel_r = r[:, i] - (m[j] * r[:, j] + m[k] * r[:, k]) / m_pair
        rel_v = v[:, i] - (m[j] * v[:, j] + m[k] * v[:, k]) / m_pair
        dist = np.linalg.norm(rel_r, axis=-1)
        energy = 0.5 * np.sum(rel_v**2, axis=-1) - G * (m[i] + m_pair) / dist
        moving_away = np.sum(rel_r * rel_v, axis=-1) > 0.0
        out |= (dist > d_escape) & moving_away & (energy > 0.0)
    return out


def _pair_forces(r: np.ndarray, softening: float) -> tuple[np.ndarray, np.ndarray]:
    """Accelerations for a batch of three unit masses, and the squared
    distance of the closest pair in each system.

    Same physics as ``physics.accelerations``, written pair by pair because
    this loop is the bottleneck of the escape-time map.
    """
    a = np.zeros_like(r)
    d2_min = np.full(r.shape[0], np.inf)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        d = r[:, j] - r[:, i]
        d2 = np.einsum("ij,ij->i", d, d)
        f = d * (G * (d2 + softening**2) ** -1.5)[:, None]
        a[:, i] += f
        a[:, j] -= f
        np.minimum(d2_min, d2, out=d2_min)
    return a, d2_min


def escape_time_map(
    xs: np.ndarray,
    ys: np.ndarray,
    t_max: float = 40.0,
    softening: float = 1e-2,
    eta: float = 0.05,
    d_escape: float = 5.0,
    max_steps: int = 40_000,
    progress: bool = False,
) -> np.ndarray:
    """Time before one body is ejected, on a grid of initial conditions.

    Free-fall three-body problem: three equal masses start at rest, bodies 1
    and 2 at (-0.5, 0) and (0.5, 0), body 3 at (x, y) for every point of the
    grid. All the systems are integrated at once (vectorised leapfrog; each
    system has its own time step, which shrinks during close encounters).

    Returns an array of shape (len(ys), len(xs)) holding the escape time, or
    NaN where nobody escaped before ``t_max`` (or before ``max_steps`` steps).

    The integration is deliberately rough (softened force, moderate step):
    individual pixels in the chaotic regions are not reliable, but the
    overall structure of the map is.
    """
    gx, gy = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    n_sys = gx.size
    m = np.ones(3)
    r = np.zeros((n_sys, 3, 3))
    r[:, 0, 0], r[:, 1, 0] = -0.5, 0.5
    r[:, 2, 0], r[:, 2, 1] = gx.ravel(), gy.ravel()
    v = np.zeros_like(r)
    r, v = to_com_frame(r, v, m)

    t = np.zeros(n_sys)
    t_escape = np.full(n_sys, np.nan)
    index = np.arange(n_sys)  # which grid point each active system belongs to
    a, d2_min = _pair_forces(r, softening)

    for step in range(max_steps):
        if index.size == 0:
            break
        # Time step: a fraction of the free-fall time of the closest pair.
        dt = eta * np.sqrt((d2_min + softening**2) ** 1.5 / (2.0 * G))
        dt3 = dt[:, None, None]
        # Leapfrog (kick - drift - kick).
        v += 0.5 * dt3 * a
        r += dt3 * v
        a, d2_min = _pair_forces(r, softening)
        v += 0.5 * dt3 * a
        t += dt

        if step % 10 == 0:
            esc = _escaped(r, v, m, d_escape)
            t_escape[index[esc]] = t[esc]
            keep = ~(esc | (t >= t_max))
            if not keep.all():
                r, v, a = r[keep], v[keep], a[keep]
                t, d2_min, index = t[keep], d2_min[keep], index[keep]
            if progress and step % 5000 == 0:
                print(f"step {step:6d}: {index.size:7d} systems still running", flush=True)

    return t_escape.reshape(gx.shape)
