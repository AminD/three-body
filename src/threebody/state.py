"""Small containers shared by the whole package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InitialConditions:
    """Starting point of a simulation.

    Attributes
    ----------
    name : label used in plots and file names.
    r : positions, shape (N, 3).
    v : velocities, shape (N, 3).
    m : masses, shape (N,).
    t_end : suggested simulation time (in units where G = 1).
    softening : suggested softening length (0 = pure Newtonian gravity).
    """

    name: str
    r: np.ndarray
    v: np.ndarray
    m: np.ndarray
    t_end: float = 20.0
    softening: float = 0.0

    def __post_init__(self) -> None:
        # Store clean float copies so callers cannot modify them by accident.
        object.__setattr__(self, "r", np.array(self.r, dtype=float))
        object.__setattr__(self, "v", np.array(self.v, dtype=float))
        object.__setattr__(self, "m", np.array(self.m, dtype=float))
        n = self.m.shape[0]
        if self.r.shape != (n, 3) or self.v.shape != (n, 3):
            raise ValueError(
                f"expected r and v of shape ({n}, 3), got {self.r.shape} and {self.v.shape}"
            )


@dataclass(frozen=True)
class Trajectory:
    """Result of a simulation.

    Attributes
    ----------
    t : times, shape (T,).
    r : positions, shape (T, N, 3).
    v : velocities, shape (T, N, 3).
    m : masses, shape (N,).
    softening : softening length used during the integration.
    method : name of the integrator.
    completed : False if the integration stopped before the requested end time.
    """

    t: np.ndarray
    r: np.ndarray
    v: np.ndarray
    m: np.ndarray
    softening: float = 0.0
    method: str = ""
    name: str = ""
    completed: bool = True
