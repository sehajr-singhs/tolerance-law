"""Local shape-check: does the learned boundary c* move with budget/capacity?

Runs a small grid (clearance x budget x capacity), prints every cell, and
summarizes where the >=0.5 success boundary sits.  This is the falsifiable
claim of the Tolerance Law: c*(N, width) should move DOWN as N grows and
should be U-shaped in width at fixed N (capacity needs data to pay off).

    PYTHONPATH=src python scripts/boundary_check.py
"""

from __future__ import annotations

import time

import numpy as np

from tolerance.envs.planar_insertion import PlanarInsertion
from tolerance.policies.expert import DitherExpert
from tolerance.train.bc import collect_demos, eval_policy, train_bc

GRID = [
    (0.001, 15, 32, 2), (0.001, 15, 128, 3),
    (0.001, 60, 32, 2), (0.001, 60, 128, 3),
    (0.004, 15, 32, 2), (0.004, 15, 128, 3),
    (0.004, 60, 32, 2), (0.004, 60, 128, 3),
    (0.008, 15, 32, 2), (0.008, 15, 128, 3),
    (0.008, 60, 32, 2), (0.008, 60, 128, 3),
]


def main() -> None:
    results = {}
    for (c, n, w, d) in GRID:
        t0 = time.time()
        env = PlanarInsertion(clearance=c, seed=0)
        exp = DitherExpert(rng=np.random.default_rng(0))
        ds = collect_demos(env, exp, n_demos=n, seed=0)
        model, _ = train_bc(ds, width=w, depth=d, epochs=15, batch_size=512,
                            seed=0)
        res = eval_policy(env, model, n_episodes=12, seed=1)
        results[(c, n, w)] = res["success"]
        print(f"c={c:.3f} N={n:>2d} w={w:<3d} -> succ={res['success']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("\nboundary check (learned if succ >= 0.5):")
    for (c, n, w) in results:
        s = results[(c, n, w)]
        print(f"  c={c:.3f} N={n:>2d} w={w:<3d}: {s:.2f} "
              f"{'LEARNED' if s >= 0.5 else 'below'}", flush=True)


if __name__ == "__main__":
    main()
