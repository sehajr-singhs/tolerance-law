"""Tolerance Law experiments: phase-diagram sweep + adaptive-budget controller.

The core claim: the learnability of an insertion skill is gated by the
engineering tolerance (clearance c), and the boundary c*(N, capacity) moves
predictably with the data budget N and model capacity.  This module runs:

  1. FIXED GRID  -- success(c, capacity, N) for every cell.  The phase
     diagram.  The boundary c* is the smallest clearance at which a
     (capacity, N) policy clears SUCCESS_THRESHOLD.

  2. ADAPTIVE BUDGET -- for each clearance c, start at a small budget and
     double it until the policy learns (or the max budget is exhausted).
     The result is the *minimal budget* N*(c): the Tolerance Law measured
     online by a controller that does not know the law.

  3. ADAPTIVE CAPACITY -- fixed budget; walk capacity upward until the
     policy learns.  Mirrors the "capacity must be bought with data" side
     of the law and is the cheap online probe a factory could actually run.

Everything writes JSON rows; the analysis script turns them into figures,
fits the power law c* ~ N^-alpha, and emits LaTeX macros.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..envs.planar_insertion import PlanarInsertion
from ..policies.expert import DitherExpert
from ..train.bc import collect_demos, eval_policy, train_bc

SUCCESS_THRESHOLD = 0.60
DEFAULT_CLEARANCES = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
DEFAULT_CAPACITIES = [(32, 2), (128, 3), (256, 4)]
DEFAULT_BUDGETS = [20, 40, 80, 160]


def run_cell(
    clearance: float,
    width: int,
    depth: int,
    n_demos: int,
    seed: int,
    device: str = "cpu",
    epochs: int = 40,
    n_eval: int = 40,
    progress: bool = False,
) -> dict:
    """One phase-diagram cell: collect demos, train BC, evaluate."""
    env = PlanarInsertion(clearance=clearance, seed=seed)
    expert = DitherExpert(rng=np.random.default_rng(seed))
    t0 = time.time()
    dataset = collect_demos(env, expert, n_demos=n_demos, seed=seed,
                            progress=progress)
    t_demo = time.time() - t0
    if dataset["episodes"] < max(4, n_demos // 4):
        return {
            "clearance": clearance, "width": width, "depth": depth,
            "n_demos": n_demos, "seed": seed,
            "success": 0.0, "expert_rate": dataset["success_rate"],
            "episodes": dataset["episodes"], "attempts": dataset["attempts"],
            "n_samples": int(dataset["obs"].shape[0]),
            "t_demo": round(t_demo, 1), "t_train": 0.0, "status": "no-demos",
        }
    t0 = time.time()
    model, hist = train_bc(dataset, width=width, depth=depth, epochs=epochs,
                           seed=seed, device=device, progress=progress)
    t_train = time.time() - t0
    res = eval_policy(env, model, n_episodes=n_eval, seed=seed + 1,
                      device=device)
    return {
        "clearance": clearance, "width": width, "depth": depth,
        "n_demos": n_demos, "seed": seed,
        "success": round(float(res["success"]), 4),
        "final_dist": round(float(res["final_dist"]), 4),
        "steps": round(float(res["steps"]), 1),
        "expert_rate": round(float(dataset["success_rate"]), 4),
        "episodes": dataset["episodes"], "attempts": dataset["attempts"],
        "n_samples": int(dataset["obs"].shape[0]),
        "train_loss_last": round(hist["train_loss"][-1], 5),
        "t_demo": round(t_demo, 1), "t_train": round(t_train, 1),
        "status": "ok",
    }


def _save(rows: list[dict], out: Optional[Path]) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))


def run_grid(
    clearances: list[float] = DEFAULT_CLEARANCES,
    capacities: list[tuple[int, int]] = DEFAULT_CAPACITIES,
    budgets: list[int] = DEFAULT_BUDGETS,
    seeds: list[int] = (0, 1),
    device: str = "cpu",
    epochs: int = 40,
    n_eval: int = 40,
    progress: bool = True,
    out: Optional[Path] = None,
) -> list[dict]:
    rows: list[dict] = []
    for c in clearances:
        for (w, d) in capacities:
            for n in budgets:
                for seed in seeds:
                    row = run_cell(c, w, d, n, seed, device=device,
                                   epochs=epochs, n_eval=n_eval,
                                   progress=progress)
                    rows.append(row)
                    _save(rows, out)   # incremental: partial runs survive
                    if progress:
                        tag = (f"c={c:>7.4f} w={w:<3d} N={n:<3d} seed={seed} "
                               f"-> {row['success']:.2f} [{row['status']}]")
                        print(tag, flush=True)
        if progress:
            print(f"--- clearance {c} done ({sum(1 for r in rows if r['clearance'] == c)} rows) ---",
                  flush=True)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)
    return rows


def _budget_doubling(
    clearance: float,
    width: int,
    depth: int,
    n_min: int,
    n_max: int,
    seed: int,
    device: str,
    epochs: int,
    n_eval: int,
    progress: bool,
) -> dict:
    """Train at doubling budgets until success >= threshold (or n_max)."""
    env = PlanarInsertion(clearance=clearance, seed=seed)
    expert = DitherExpert(rng=np.random.default_rng(seed))
    n = n_min
    steps: list[dict] = []
    while n <= n_max:
        dataset = collect_demos(env, expert, n_demos=n, seed=seed,
                                progress=False)
        model, _ = train_bc(dataset, width=width, depth=depth, epochs=epochs,
                            seed=seed, device=device, progress=False)
        res = eval_policy(env, model, n_episodes=n_eval, seed=seed + 1,
                          device=device)
        s = float(res["success"])
        steps.append({"n": n, "success": round(s, 4),
                      "episodes": dataset["episodes"],
                      "attempts": dataset["attempts"]})
        if progress:
            print(f"    adaptive c={clearance:.4f} N={n}: {s:.2f}", flush=True)
        if s >= SUCCESS_THRESHOLD:
            return {"clearance": clearance, "width": width, "depth": depth,
                    "min_budget": n, "learned": True, "steps": steps,
                    "n_eval": n_eval}
        n *= 2
    return {"clearance": clearance, "width": width, "depth": depth,
            "min_budget": None, "learned": False, "steps": steps,
            "n_eval": n_eval}


def run_adaptive_budget(
    clearances: list[float] = DEFAULT_CLEARANCES,
    capacity: tuple[int, int] = (128, 3),
    n_min: int = 20,
    n_max: int = 320,
    seeds: list[int] = (0,),
    device: str = "cpu",
    epochs: int = 40,
    n_eval: int = 40,
    progress: bool = True,
    out: Optional[Path] = None,
) -> list[dict]:
    """The Tolerance Law measured online: minimal budget N*(c) per clearance."""
    rows = []
    for c in clearances:
        for seed in seeds:
            r = _budget_doubling(c, capacity[0], capacity[1], n_min, n_max,
                                 seed, device, epochs, n_eval, progress)
            r["seed"] = seed
            rows.append(r)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {out} ({len(rows)} rows)", flush=True)
    return rows


def _capacity_walk(
    clearance: float,
    n_demos: int,
    capacities: list[tuple[int, int]],
    seed: int,
    device: str,
    epochs: int,
    n_eval: int,
    progress: bool,
) -> dict:
    env = PlanarInsertion(clearance=clearance, seed=seed)
    expert = DitherExpert(rng=np.random.default_rng(seed))
    dataset = collect_demos(env, expert, n_demos=n_demos, seed=seed,
                            progress=False)
    steps: list[dict] = []
    for (w, d) in capacities:
        model, _ = train_bc(dataset, width=w, depth=d, epochs=epochs,
                            seed=seed, device=device, progress=False)
        res = eval_policy(env, model, n_episodes=n_eval, seed=seed + 1,
                          device=device)
        s = float(res["success"])
        steps.append({"width": w, "depth": d, "success": round(s, 4)})
        if progress:
            print(f"    capacity c={clearance:.4f} w={w}: {s:.2f}", flush=True)
        if s >= SUCCESS_THRESHOLD:
            return {"clearance": clearance, "n_demos": n_demos,
                    "min_capacity": (w, d), "learned": True, "steps": steps}
    return {"clearance": clearance, "n_demos": n_demos,
            "min_capacity": None, "learned": False, "steps": steps}


def run_adaptive_capacity(
    clearances: list[float] = DEFAULT_CLEARANCES,
    n_demos: int = 80,
    capacities: list[tuple[int, int]] = DEFAULT_CAPACITIES,
    seeds: list[int] = (0,),
    device: str = "cpu",
    epochs: int = 40,
    n_eval: int = 40,
    progress: bool = True,
    out: Optional[Path] = None,
) -> list[dict]:
    rows = []
    for c in clearances:
        for seed in seeds:
            r = _capacity_walk(c, n_demos, capacities, seed, device, epochs,
                               n_eval, progress)
            r["seed"] = seed
            rows.append(r)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {out} ({len(rows)} rows)", flush=True)
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--quick", action="store_true",
                    help="tiny grid for local smoke tests")
    args = ap.parse_args()

    if args.quick:
        out = Path("results_tolerance/state/sweep_quick.json")
        run_grid(clearances=[0.001, 0.008],
                 capacities=[(32, 2), (128, 3)],
                 budgets=[20, 40],
                 seeds=[0],
                 device=args.device, epochs=8, n_eval=10,
                 out=out)
        run_adaptive_budget(clearances=[0.001, 0.008], capacity=(128, 3),
                            n_min=20, n_max=80, seeds=[0], device=args.device,
                            epochs=8, n_eval=10,
                            out=Path("results_tolerance/state/adaptive_quick.json"))
        run_adaptive_capacity(clearances=[0.001, 0.008], n_demos=40,
                              capacities=[(32, 2), (128, 3)], seeds=[0],
                              device=args.device, epochs=8, n_eval=10,
                              out=Path("results_tolerance/state/capacity_quick.json"))
    else:
        out = Path("results_tolerance/state/sweep.json")
        run_grid(device=args.device, epochs=args.epochs, n_eval=args.n_eval,
                 out=out)
        run_adaptive_budget(device=args.device, epochs=args.epochs,
                            n_eval=args.n_eval,
                            out=Path("results_tolerance/state/adaptive.json"))
        run_adaptive_capacity(device=args.device, epochs=args.epochs,
                              n_eval=args.n_eval,
                              out=Path("results_tolerance/state/capacity.json"))
