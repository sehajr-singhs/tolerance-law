"""Local debug: sweep clearance with the expert to check the physics.

Usage: PYTHONPATH=src python scripts/debug_insertion.py [--quick]
"""

from __future__ import annotations

import argparse

import numpy as np

from tolerance.envs.planar_insertion import (
    PlanarInsertion, SEAT_X, PEG_L, SLOT_X0)
from tolerance.policies.expert import DitherExpert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--clearances", type=str,
                    default="0.0005,0.001,0.002,0.004,0.008,0.016")
    ap.add_argument("--episodes", type=int, default=40)
    args = ap.parse_args()
    cs = [float(x) for x in args.clearances.split(",")]
    n_ep = args.episodes if not args.quick else 15
    print(f"clearances: {cs}  episodes: {n_ep}")
    for c in cs:
        env = PlanarInsertion(clearance=c, seed=0)
        expert = DitherExpert()
        ok = 0
        dists = []
        steps = []
        fails = []
        for ep in range(n_ep):
            o = env.reset()
            expert.reset(o[4], env.y_noise)
            done = False
            while not done:
                a = expert.act(o)
                o = env.step(a)
                done = env.done
            ok += int(env.success)
            dists.append(env.final_dist_to_seat)
            steps.append(env.t)
            if not env.success:
                fails.append((round(env.peg_x, 4), round(env.peg_y - env.y_channel, 4)))
        print(f"c={c:>7.4f}  success {ok}/{n_ep} ({ok/n_ep:5.1%})  "
              f"dist {np.mean(dists):.4f}  steps {np.mean(steps):.0f}  "
              f"fails(y-offset) {fails[:3]}")


if __name__ == "__main__":
    main()
