"""Newtonian gravity for N point masses.

Units are dimensionless: G = 1, masses and lengths of order 1.

Every function accepts arrays with extra leading dimensions, e.g. positions of
shape (T, N, 3) for a whole trajectory or (B, N, 3) for a batch of B systems.
"""

from __future__ import annotations

import numpy as np

G = 1.0


def accelerations(r: np.ndarray, m: np.ndarray, softening: float = 0.0) -> np.ndarray:
    """Gravitational acceleration of every body.

    a_i = G * sum_{j != i} m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)

    Parameters
    ----------
    r : positions, shape (..., N, 3).
    m : masses, shape (N,).
    softening : Plummer softening length eps. With eps > 0 the force stays
        finite when two bodies get very close.
    """
    r = np.asarray(r, dtype=float)
    m = np.asarray(m, dtype=float)
    n = r.shape[-2]
    # d[..., i, j, :] = r_j - r_i
    d = r[..., None, :, :] - r[..., :, None, :]
    dist2 = np.sum(d * d, axis=-1) + softening**2
    diag = np.arange(n)
    dist2[..., diag, diag] = 1.0  # placeholder to avoid dividing by zero
    inv_d3 = dist2**-1.5
    inv_d3[..., diag, diag] = 0.0  # a body does not attract itself
    return G * np.einsum("...ij,...ijk,j->...ik", inv_d3, d, m)


def kinetic_energy(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """K = 1/2 sum_i m_i |v_i|^2."""
    return 0.5 * np.sum(m * np.sum(np.square(v), axis=-1), axis=-1)


def potential_energy(r: np.ndarray, m: np.ndarray, softening: float = 0.0) -> np.ndarray:
    """U = - G sum_{i<j} m_i m_j / sqrt(|r_i - r_j|^2 + eps^2)."""
    r = np.asarray(r, dtype=float)
    n = r.shape[-2]
    i, j = np.triu_indices(n, k=1)
    d = r[..., i, :] - r[..., j, :]
    dist = np.sqrt(np.sum(d * d, axis=-1) + softening**2)
    return -G * np.sum(m[i] * m[j] / dist, axis=-1)


def total_energy(
    r: np.ndarray, v: np.ndarray, m: np.ndarray, softening: float = 0.0
) -> np.ndarray:
    """E = K + U. Conserved by the exact dynamics."""
    return kinetic_energy(v, m) + potential_energy(r, m, softening)


def momentum(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """P = sum_i m_i v_i. Conserved by the exact dynamics."""
    return np.einsum("j,...jk->...k", m, v)


def angular_momentum(r: np.ndarray, v: np.ndarray, m: np.ndarray) -> np.ndarray:
    """L = sum_i m_i (r_i x v_i). Conserved by the exact dynamics."""
    return np.einsum("j,...jk->...k", m, np.cross(r, v))


def center_of_mass(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Mass-weighted mean position (also works for velocities)."""
    return np.einsum("j,...jk->...k", m, r) / np.sum(m)


def to_com_frame(
    r: np.ndarray, v: np.ndarray, m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shift positions and velocities to the centre-of-mass frame.

    In that frame the centre of mass sits at the origin and the total
    momentum is zero, so the system does not drift away. The physics is
    unchanged (Galilean invariance).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_com = center_of_mass(r, m)
    v_com = center_of_mass(v, m)
    return r - r_com[..., None, :], v - v_com[..., None, :]


def relative_energy_error(traj) -> np.ndarray:
    """|E(t) - E(0)| / |E(0)| along a trajectory."""
    e = total_energy(traj.r, traj.v, traj.m, traj.softening)
    return np.abs((e - e[0]) / e[0])
