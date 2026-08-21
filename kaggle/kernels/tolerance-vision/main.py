"""Tolerance Law, vision modality — Kaggle kernel.

The same clearance phase diagram measured with image observations: a CNN
behavior-clones the same expert demos, but the observation is a 64x64
top-view RGB render instead of the 12-dim state vector.  This tests whether
the law survives the perception bottleneck — a strict replication of the
state result under weaker information.

Grid (lean): 4 clearances x 2 CNN capacities x 2 budgets x 2 seeds.
Results -> /kaggle/working/results_tolerance/vision/
"""

import glob
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
t0 = time.time()
OUT = Path("/kaggle/working")

pkg = Path("/kaggle/working/pkg")
for tar in glob.glob("/kaggle/input/**/*.tar", recursive=True):
    print("extracting", tar, flush=True)
    with tarfile.open(tar) as t:
        t.extractall(pkg)
whls = sorted(glob.glob(str(pkg / "**" / "wheels" / "*.whl"), recursive=True) or
              glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True))
for w in whls:
    print("installing", w, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", w],
                   check=True)


def find_pkg_root():
    for base in ("/kaggle/input", str(pkg)):
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base, followlinks=True):
            if "tolerance" in dirnames:
                sub = os.path.join(dirpath, "tolerance")
                if any(f.endswith(".py") for f in os.listdir(sub)):
                    return dirpath
    return None


root = find_pkg_root()
assert root is not None, "tolerance package not mounted"
sys.path.insert(0, root)
print("package root:", root, flush=True)

import torch  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} device={DEVICE} cpus={os.cpu_count()}", flush=True)

from tolerance.envs.planar_insertion import PlanarInsertion  # noqa: E402
from tolerance.policies.expert import DitherExpert  # noqa: E402
from tolerance.train.bc import (  # noqa: E402
    collect_demos_vision, eval_policy_vision, train_bc_cnn)

FAST = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "fast"
IMG_SIZE = 64
if FAST:
    CLEARANCES = [0.001, 0.004]
    CHANNELS = [8, 16]
    BUDGETS = [20, 60]
    SEEDS = [0]
    EPOCHS = 10
    N_EVAL = 10
else:
    CLEARANCES = [0.0005, 0.001, 0.002, 0.004]
    CHANNELS = [8, 16]
    BUDGETS = [20, 60]
    SEEDS = [0, 1]
    EPOCHS = 25
    N_EVAL = 20

out = OUT / "results_tolerance" / "vision"
out.mkdir(parents=True, exist_ok=True)


def run_cell(c, ch, n, seed):
    env = PlanarInsertion(clearance=c, seed=seed)
    exp = DitherExpert(rng=np.random.default_rng(seed))
    t1 = time.time()
    ds = collect_demos_vision(env, exp, n_demos=n, img_size=IMG_SIZE, seed=seed)
    t_demo = time.time() - t1
    if ds["episodes"] == 0:
        return {"clearance": c, "channels": ch, "n_demos": n, "seed": seed,
                "success": 0.0, "episodes": 0, "attempts": ds["attempts"],
                "status": "no-demos", "t_demo": round(t_demo, 1)}
    t1 = time.time()
    model, _ = train_bc_cnn(ds, channels=ch, depth=3, epochs=EPOCHS,
                            batch_size=128, seed=seed, device=DEVICE)
    t_train = time.time() - t1
    res = eval_policy_vision(env, model, n_episodes=N_EVAL, seed=seed + 1,
                             img_size=IMG_SIZE, device=DEVICE)
    return {"clearance": c, "channels": ch, "n_demos": n, "seed": seed,
            "success": round(float(res["success"]), 4),
            "final_dist": round(float(res["final_dist"]), 4),
            "episodes": ds["episodes"], "attempts": ds["attempts"],
            "t_demo": round(t_demo, 1), "t_train": round(t_train, 1),
            "status": "ok"}


rows = []
for c in CLEARANCES:
    for ch in CHANNELS:
        for n in BUDGETS:
            for seed in SEEDS:
                row = run_cell(c, ch, n, seed)
                rows.append(row)
                (out / "sweep.json").write_text(json.dumps(rows, indent=1))
                print(f"c={c:.4f} ch={ch} N={n} s{seed} -> {row['success']:.2f} "
                      f"[{row['status']}]", flush=True)

meta = {"modality": "vision", "env": "planar_insertion", "fast": FAST,
        "clearances": CLEARANCES, "channels": CHANNELS, "budgets": BUDGETS,
        "seeds": SEEDS, "epochs": EPOCHS, "n_eval": N_EVAL, "img_size": IMG_SIZE,
        "wall_s": round(time.time() - t0)}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"ALL DONE in {time.time() - t0:.0f}s — {out}", flush=True)
