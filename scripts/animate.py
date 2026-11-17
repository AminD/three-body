"""Animate a three-body scenario in 3D.

Run from the project folder, for example:

    python scripts/animate.py --scenario chaotic
    python scripts/animate.py --scenario figure_eight --save output/figure_eight.gif
    python scripts/animate.py --config configs/custom.toml

Without --save a window opens (drag with the mouse to move the camera).
"""

from __future__ import annotations

import argparse

import matplotlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="chaotic", help="figure_eight, lagrange or chaotic")
    parser.add_argument("--config", help="TOML file with custom initial conditions")
    parser.add_argument("--t-end", type=float, help="simulated time (default: scenario value)")
    parser.add_argument("--duration", type=float, default=12.0, help="animation length in seconds")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--trail", type=float, default=0.15, help="trail length (fraction of the run)")
    parser.add_argument("--save", help="write a GIF to this path instead of opening a window")
    args = parser.parse_args()

    if args.save:
        matplotlib.use("Agg")  # no window needed when saving to a file
    import matplotlib.pyplot as plt

    from threebody import SCENARIOS, integrate_adaptive, load_config
    from threebody.physics import relative_energy_error
    from threebody.visualization import animate_trajectory, save_gif

    if args.config:
        ic = load_config(args.config)
    elif args.scenario in SCENARIOS:
        ic = SCENARIOS[args.scenario]()
    else:
        parser.error(f"unknown scenario {args.scenario!r}, choose from {sorted(SCENARIOS)}")

    # Several simulation samples per frame keep the trails smooth.
    n_out = int(args.duration * args.fps) * 8
    traj = integrate_adaptive(ic, args.t_end, n_out=n_out, max_nfev=2_000_000)
    print(f"scenario: {ic.name}, simulated time: {traj.t[-1]:.2f}")
    print(f"largest relative energy error: {relative_energy_error(traj).max():.1e}")
    if not traj.completed:
        print("warning: stopped early (extremely close encounter); try a small softening")

    anim = animate_trajectory(traj, duration=args.duration, fps=args.fps, trail=args.trail)
    if args.save:
        save_gif(anim, args.save, fps=args.fps)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
