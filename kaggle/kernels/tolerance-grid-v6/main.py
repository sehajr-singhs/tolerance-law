#!/usr/bin/env python
"""Tolerance Law: full statistical grid (10 seeds × 4c × 3w × 3N = 360 cells).

Runs on Kaggle GPU. Saves results incrementally to survive disconnection.
"""
import os, sys, json, time, glob

os.environ["MUJOCO_GL"] = "egl"

# ── install bundled wheels ────────────────────────────────────────────
import subprocess
PKG_ROOT = "/kaggle/input/datasets/sehajrsingh/tolerance-pkg"
WHEELS_DIR = os.path.join(PKG_ROOT, "wheels")
if os.path.isdir(WHEELS_DIR):
    whls = glob.glob(os.path.join(WHEELS_DIR, "*.whl"))
    print(f"[wheels] installing {len(whls)} bundled wheels")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps"] + whls)
else:
    print("[wheels] no wheels dir found, pip install")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "mujoco"])

# ── find package ──────────────────────────────────────────────────────
SRC_DIR = os.path.join(PKG_ROOT, "src")
if os.path.isdir(SRC_DIR):
    sys.path.insert(0, SRC_DIR)
    print(f"[pkg] added {SRC_DIR}")
else:
    raise RuntimeError(f"source not found at {SRC_DIR}")

import numpy as np
import torch
from tolerance.envs.planar_insertion import PlanarInsertion
from tolerance.policies.expert import DitherExpert
from tolerance.train.bc import collect_demos, train_bc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[dev] device={DEVICE}  torch={torch.__version__}")

# ── grid ──────────────────────────────────────────────────────────────
CLEARANCES = [0.0005, 0.001, 0.002, 0.004]
WIDTHS     = [32, 128, 256]
BUDGETS    = [20, 60, 160]
N_SEEDS    = 10
N_EVAL     = 40
OUT_DIR    = "/kaggle/working/results_tolerance_v6"
os.makedirs(OUT_DIR, exist_ok=True)

SWEEP_FILE = os.path.join(OUT_DIR, "sweep.json")

# load existing results for resume
existing = []
if os.path.exists(SWEEP_FILE):
    with open(SWEEP_FILE) as f:
        existing = json.load(f)
done_keys = {(c["clearance"], c["width"], c["n_demos"], c["seed"]) for c in existing}
print(f"[grid] {len(existing)} cells already done, resuming...")

total = len(CLEARANCES) * len(WIDTHS) * len(BUDGETS) * N_SEEDS
count = 0
t_grid = time.time()

for clearance in CLEARANCES:
    for width in WIDTHS:
        for n_demos in BUDGETS:
            for seed in range(N_SEEDS):
                count += 1
                if (clearance, width, n_demos, seed) in done_keys:
                    continue

                t0 = time.time()
                rng = np.random.default_rng(seed)
                env = PlanarInsertion(clearance=clearance, seed=seed, rng=rng)
                expert = DitherExpert(rng=np.random.default_rng(seed))

                # collect demos
                dataset = collect_demos(env, expert, n_demos=n_demos, seed=seed)
                teacher = dataset["success_rate"]
                n_actual = dataset["episodes"]

                # train BC
                policy, hist = train_bc(
                    dataset, width=width, depth=3, epochs=40,
                    lr=1e-3, batch_size=256, seed=seed, device=DEVICE,
                )

                # eval
                wins = 0
                for s in range(N_EVAL):
                    env2 = PlanarInsertion(
                        clearance=clearance, seed=10_000 + s,
                        rng=np.random.default_rng(10_000 + s),
                    )
                    obs = env2.reset()
                    for step in range(1200):
                        with torch.no_grad():
                            a = policy(
                                torch.as_tensor(obs, dtype=torch.float32).to(DEVICE)
                            ).cpu().numpy()
                        obs = env2.step(np.clip(a, -1.0, 1.0))
                        if env2.done:
                            break
                    if env2.success:
                        wins += 1

                test_rate = wins / N_EVAL
                elapsed = time.time() - t0

                cell = {
                    "clearance": clearance,
                    "width": width,
                    "depth": 3,
                    "n_demos": n_demos,
                    "seed": seed,
                    "success": test_rate,
                    "expert_rate": teacher,
                    "n_samples": len(dataset["obs"]),
                    "episodes": n_actual,
                    "train_loss_last": hist["train_loss"][-1] if hist["train_loss"] else -1,
                    "t_train": elapsed,
                    "status": "ok",
                }
                existing.append(cell)

                # save incrementally
                with open(SWEEP_FILE, "w") as f:
                    json.dump(existing, f)

                print(
                    f"[{count}/{total}] c={clearance*1000:.1f}mm w={width:3d} "
                    f"N={n_demos:3d} s={seed} | teacher={teacher:.2f} "
                    f"test={test_rate:.2f} ({elapsed:.0f}s) "
                    f"[total={time.time()-t_grid:.0f}s]"
                )

# final summary
print("\n" + "=" * 60)
print("FULL GRID COMPLETE")
print(f"Total time: {time.time()-t_grid:.0f}s ({(time.time()-t_grid)/60:.1f}min)")
print(f"Cells: {len(existing)}")
print("=" * 60)
