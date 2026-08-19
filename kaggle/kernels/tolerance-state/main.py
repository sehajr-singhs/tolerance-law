"""Tolerance Law, state modality — Kaggle GPU kernel.

Measures the phase diagram of insertion-skill learnability:

  success(clearance c, capacity (width, depth), data budget N)

over 6 clearances x 3 capacities x 4 budgets x 2 seeds (144 cells), plus
the two online controllers:

  - ADAPTIVE BUDGET:  minimal demos N*(c) needed to learn at each clearance
                      (the law measured online, without knowing it).
  - ADAPTIVE CAPACITY: minimal capacity that learns at each clearance under
                      a fixed budget (the "buy capacity with data" side).

and the teacher curve (expert success per clearance) for the paper.

The mujoco + glfw wheels ship in the dataset and install with --no-deps,
so the kernel is fully offline (Kaggle egress killed r7's pip before).

Results -> /kaggle/working/results_tolerance/state/
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
    """Locate the tolerance package under /kaggle/input or the extraction dir.

    Kaggle's dataset mount can be a symlink, so plain glob('**') can miss it;
    os.walk with followlinks covers every layout.
    """
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

from tolerance.envs.planar_insertion import PlanarInsertion  # noqa: E402
from tolerance.policies.expert import DitherExpert  # noqa: E402
from tolerance.experiments.sweep import (  # noqa: E402
    DEFAULT_BUDGETS, DEFAULT_CAPACITIES, DEFAULT_CLEARANCES, SUCCESS_THRESHOLD,
    run_adaptive_budget, run_adaptive_capacity, run_grid)
from tolerance.train.bc import eval_policy  # noqa: E402

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} device={DEVICE} "
      f"cuda_available={torch.cuda.is_available()} cpus={os.cpu_count()}",
      flush=True)

FAST = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "fast"
EPOCHS = 10 if FAST else 40
N_EVAL = 12 if FAST else 40
SEEDS = [0] if FAST else [0, 1]
if FAST:
    CLEARANCES = [0.001, 0.004]
    CAPACITIES = [(32, 2), (128, 3)]
    BUDGETS = [20, 60]
else:
    # the boundary lives in the tight end; 0.008/0.016 are the
    # "everything works" region and only cost compute
    CLEARANCES = [0.0005, 0.001, 0.002, 0.004]
    CAPACITIES = [(32, 2), (128, 3), (256, 4)]
    BUDGETS = [20, 60, 160]

out = OUT / "results_tolerance" / "state"
out.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- #
# 0. teacher curve: expert success per clearance (honest teacher)        #
# --------------------------------------------------------------------- #
print("=== 0: TEACHER CURVE ===", flush=True)
teacher = []
for c in CLEARANCES:
    env = PlanarInsertion(clearance=c, seed=0)
    exp = DitherExpert(rng=np.random.default_rng(0))
    ok = 0
    n = 12 if FAST else 30
    for _ in range(n):
        o = env.reset()
        exp.reset(o[4], env.y_noise)
        done = False
        while not done:
            a = exp.act(o)
            o = env.step(a)
            done = env.done
        ok += int(env.success)
    teacher.append({"clearance": c, "expert_success": ok / n, "n": n})
    print(f"  c={c}: expert {ok}/{n}", flush=True)
(out / "teacher.json").write_text(json.dumps(teacher, indent=1))

# --------------------------------------------------------------------- #
# 1. fixed grid: success(c, capacity, budget)                            #
# --------------------------------------------------------------------- #
print("=== 1: FIXED GRID ===", flush=True)
run_grid(clearances=CLEARANCES, capacities=CAPACITIES, budgets=BUDGETS,
         seeds=SEEDS, device=DEVICE, epochs=EPOCHS, n_eval=N_EVAL,
         out=out / "sweep.json")
print(f"grid done in {time.time() - t0:.0f}s", flush=True)

# --------------------------------------------------------------------- #
# 2. adaptive budget: N*(c) -- the law measured online                  #
# --------------------------------------------------------------------- #
print("=== 2: ADAPTIVE BUDGET ===", flush=True)
run_adaptive_budget(clearances=CLEARANCES, capacity=CAPACITIES[len(CAPACITIES) // 2],
                    n_min=15, n_max=120, seeds=SEEDS[:1], device=DEVICE,
                    epochs=EPOCHS, n_eval=N_EVAL,
                    out=out / "adaptive.json")
print(f"adaptive budget done in {time.time() - t0:.0f}s", flush=True)

# --------------------------------------------------------------------- #
# 3. adaptive capacity: minimal width that learns at each c              #
# --------------------------------------------------------------------- #
print("=== 3: ADAPTIVE CAPACITY ===", flush=True)
run_adaptive_capacity(clearances=CLEARANCES, n_demos=BUDGETS[len(BUDGETS) // 2],
                      capacities=CAPACITIES, seeds=SEEDS[:1], device=DEVICE,
                      epochs=EPOCHS, n_eval=N_EVAL,
                      out=out / "capacity.json")
print(f"adaptive capacity done in {time.time() - t0:.0f}s", flush=True)

meta = {"modality": "state", "env": "planar_insertion", "fast": FAST,
        "clearances": CLEARANCES, "capacities": CAPACITIES, "budgets": BUDGETS,
        "seeds": SEEDS, "epochs": EPOCHS, "n_eval": N_EVAL,
        "success_threshold": SUCCESS_THRESHOLD, "wall_s": round(time.time() - t0)}
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"ALL DONE in {time.time() - t0:.0f}s — {out}", flush=True)
