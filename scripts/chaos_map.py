"""Map of the chaos: escape time on a grid of initial conditions.

    python scripts/chaos_map.py            # 150 x 150 grid, about a minute
    python scripts/chaos_map.py --n 400    # sharper picture, several minutes

Three equal masses are released at rest. Bodies 1 and 2 start at (-0.5, 0)
and (0.5, 0); the initial position of body 3 is the pixel. The colour is the
time before one of the bodies is ejected. Writes figures/chaos_map.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from threebody.chaos import escape_time_map
from threebody.plots import plot_escape_map

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=150, help="grid points per side (one quadrant)")
    parser.add_argument("--size", type=float, default=1.5, help="the map covers [-size, size]^2")
    parser.add_argument("--t-max", type=float, default=40.0, help="longest simulated time")
    parser.add_argument("--output", default=str(FIGURES / "chaos_map.png"))
    args = parser.parse_args()

    # Only one quadrant is computed: the others follow by mirror symmetry.
    # Pixel centres are used so that no grid point sits exactly on the axes.
    step = args.size / args.n
    xs = (np.arange(args.n) + 0.5) * step
    ys = (np.arange(args.n) + 0.5) * step

    start = time.time()
    t_escape = escape_time_map(xs, ys, t_max=args.t_max, progress=True)
    print(f"{args.n**2} simulations in {time.time() - start:.0f} s")
    print(f"ejection before t = {args.t_max:g} in {np.mean(~np.isnan(t_escape)):.0%} of them")

    fig = plot_escape_map(xs, ys, t_escape, args.t_max)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
