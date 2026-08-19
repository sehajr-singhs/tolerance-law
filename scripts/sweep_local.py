"""Local driver for the Tolerance Law experiments.

Usage:
    PYTHONPATH=src python scripts/sweep_local.py --quick     # smoke test
    PYTHONPATH=src python scripts/sweep_local.py --epochs 40 # full grid
    PYTHONPATH=src python scripts/sweep_local.py --vision --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tolerance.envs.planar_insertion import PlanarInsertion
from tolerance.policies.expert import DitherExpert
from tolerance.train.bc import (
    collect_demos, collect_demos_vision, eval_policy, eval_policy_vision,
    train_bc, train_bc_cnn)


def run_vision_quick(device: str, epochs: int, n_eval: int) -> None:
    out = Path("results_tolerance/vision/quick.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in [0.001, 0.008]:
        env = PlanarInsertion(clearance=c, seed=0)
        expert = DitherExpert(rng=np.random.default_rng(0))
        ds = collect_demos_vision(env, expert, n_demos=10, seed=0)
        if ds["episodes"] == 0:
            rows.append({"clearance": c, "success": 0.0, "status": "no-demos"})
            continue
        model, _ = train_bc_cnn(ds, channels=8, depth=2, epochs=epochs,
                                seed=0, device=device)
        res = eval_policy_vision(env, model, n_episodes=n_eval, seed=1,
                                 device=device)
        rows.append({"clearance": c, "success": round(res["success"], 4),
                     "episodes": ds["episodes"],
                     "attempts": ds["attempts"], "status": "ok"})
        print(f"vision c={c}: success={res['success']:.2f}", flush=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=40)
    args = ap.parse_args()

    if args.vision:
        run_vision_quick(args.device, 8 if args.quick else args.epochs,
                         args.n_eval)
        return

    if args.quick:
        from tolerance.experiments.sweep import run_adaptive_budget, run_adaptive_capacity, run_grid
        run_grid(clearances=[0.001, 0.008], capacities=[(32, 2), (128, 3)],
                 budgets=[20, 40], seeds=[0], device=args.device, epochs=8,
                 n_eval=10, out=Path("results_tolerance/state/sweep_quick.json"))
        run_adaptive_budget(clearances=[0.001, 0.008], capacity=(128, 3),
                            n_min=20, n_max=80, seeds=[0], device=args.device,
                            epochs=8, n_eval=10,
                            out=Path("results_tolerance/state/adaptive_quick.json"))
        run_adaptive_capacity(clearances=[0.001, 0.008], n_demos=40,
                              capacities=[(32, 2), (128, 3)], seeds=[0],
                              device=args.device, epochs=8, n_eval=10,
                              out=Path("results_tolerance/state/capacity_quick.json"))
    elif args.smoke:
        from tolerance.experiments.sweep import run_grid
        run_grid(clearances=[0.008], capacities=[(64, 2)], budgets=[15],
                 seeds=[0], device=args.device, epochs=5, n_eval=6,
                 out=Path("results_tolerance/state/smoke.json"))
    else:
        from tolerance.experiments.sweep import run_adaptive_budget, run_adaptive_capacity, run_grid
        run_grid(device=args.device, epochs=args.epochs, n_eval=args.n_eval,
                 out=Path("results_tolerance/state/sweep.json"))
        run_adaptive_budget(device=args.device, epochs=args.epochs,
                            n_eval=args.n_eval,
                            out=Path("results_tolerance/state/adaptive.json"))
        run_adaptive_capacity(device=args.device, epochs=args.epochs,
                              n_eval=args.n_eval,
                              out=Path("results_tolerance/state/capacity.json"))


if __name__ == "__main__":
    main()
