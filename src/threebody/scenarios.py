"""Ready-made initial conditions, plus loading and saving of custom ones."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from .physics import G, to_com_frame, total_energy
from .state import InitialConditions

# Period of the figure-eight orbit (Chenciner & Montgomery, 2000).
FIGURE_EIGHT_PERIOD = 6.32591398


def figure_eight() -> InitialConditions:
    """Three equal masses chasing each other along a figure-eight curve.

    A periodic and (remarkably) stable solution of the three-body problem.
    """
    r1 = np.array([0.97000436, -0.24308753, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    r = np.array([r1, -r1, np.zeros(3)])
    v = np.array([-v3 / 2.0, -v3 / 2.0, v3])
    return InitialConditions("figure_eight", r, v, np.ones(3), t_end=3 * FIGURE_EIGHT_PERIOD)


def lagrange(radius: float = 1.0) -> InitialConditions:
    """Three equal masses on a rotating equilateral triangle (Lagrange, 1772).

    An exact solution, but an unstable one for equal masses: the tiniest
    numerical error grows until the triangle breaks apart (here after about
    seven turns).
    """
    m = np.ones(3)
    angles = np.array([0.0, 2.0, 4.0]) * np.pi / 3.0
    # Balance between gravity and centripetal force: omega^2 = G m / (sqrt(3) R^3).
    omega = np.sqrt(G * m[0] / (np.sqrt(3.0) * radius**3))
    r = radius * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(3)])
    v = radius * omega * np.column_stack([-np.sin(angles), np.cos(angles), np.zeros(3)])
    period = 2.0 * np.pi / omega
    return InitialConditions("lagrange", r, v, m, t_end=9 * period)


def random_bound(seed: int = 0, speed: float = 0.35) -> InitialConditions:
    """Three equal masses with random 3D positions and velocities.

    The system is placed in the centre-of-mass frame and has negative total
    energy (the bodies cannot all fly apart). This is the generic, chaotic case.
    """
    rng = np.random.default_rng(seed)
    m = np.ones(3)
    while True:
        r = rng.uniform(-1.0, 1.0, size=(3, 3))
        v = speed * rng.normal(size=(3, 3))
        r, v = to_com_frame(r, v, m)
        # Reject starts where two bodies are almost on top of each other.
        gaps = [np.linalg.norm(r[i] - r[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
        if min(gaps) > 0.6 and total_energy(r, v, m) < 0.0:
            return InitialConditions(f"random_seed{seed}", r, v, m, t_end=40.0)


def chaotic() -> InitialConditions:
    """A hand-picked chaotic 3D start (three equal masses) that stays bound
    for a long time, which makes it a good showcase."""
    ic = random_bound(seed=CHAOTIC_SEED)
    return InitialConditions("chaotic", ic.r, ic.v, ic.m, t_end=CHAOTIC_T_END)


CHAOTIC_SEED = 41
CHAOTIC_T_END = 40.0


def free_fall(x: float, y: float) -> InitialConditions:
    """Three equal masses released at rest (Agekyan-Anosova problem).

    Bodies 1 and 2 start at (-0.5, 0) and (0.5, 0); body 3 starts at (x, y).
    """
    r = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [x, y, 0.0]])
    v = np.zeros((3, 3))
    m = np.ones(3)
    r, v = to_com_frame(r, v, m)
    return InitialConditions(f"free_fall_{x:g}_{y:g}", r, v, m, t_end=30.0)


SCENARIOS = {
    "figure_eight": figure_eight,
    "lagrange": lagrange,
    "chaotic": chaotic,
}


def read_config(path: str | Path) -> dict:
    """Read a TOML configuration file and return its raw content:
    name, r, v, m (exactly as written in the file), t_end, softening, recenter."""
    path = Path(path)
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    bodies = [cfg[f"body{i}"] for i in (1, 2, 3)]
    return {
        "name": cfg.get("name", path.stem),
        "m": np.array([b.get("mass", 1.0) for b in bodies], dtype=float),
        "r": np.array([b["position"] for b in bodies], dtype=float),
        "v": np.array([b["velocity"] for b in bodies], dtype=float),
        "t_end": float(cfg.get("t_end", 30.0)),
        "softening": float(cfg.get("softening", 0.0)),
        "recenter": bool(cfg.get("recenter", True)),
    }


def load_config(path: str | Path) -> InitialConditions:
    """Initial conditions from a TOML file (see configs/custom.toml)."""
    cfg = read_config(path)
    r, v = cfg["r"], cfg["v"]
    if cfg["recenter"]:
        r, v = to_com_frame(r, v, cfg["m"])
    return InitialConditions(
        cfg["name"], r, v, cfg["m"], t_end=cfg["t_end"], softening=cfg["softening"]
    )


def save_config(
    path: str | Path,
    r: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    t_end: float,
    softening: float = 0.0,
    name: str = "custom",
) -> None:
    """Write initial conditions to a TOML file readable by ``load_config``."""

    def vec(x) -> str:
        return "[" + ", ".join(f"{c:.6g}" for c in x) + "]"

    lines = [
        "# Initial conditions for the three-body simulation (units: G = 1).",
        "# Body 1 is the reference body: keep its initial conditions fixed and",
        "# play with bodies 2 and 3. All three bodies then move freely.",
        "",
        f'name = "{name}"',
        f"t_end = {t_end:.6g}        # simulated time",
        f"softening = {softening:.6g}   # 0 = pure Newtonian gravity",
        "recenter = true     # shift to the centre-of-mass frame (same physics, no drift)",
    ]
    notes = ["fixed initial conditions", "free: edit me", "free: edit me"]
    for i in range(3):
        lines += [
            "",
            f"[body{i + 1}]  # {notes[i]}",
            f"mass = {m[i]:.6g}",
            f"position = {vec(r[i])}",
            f"velocity = {vec(v[i])}",
        ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
