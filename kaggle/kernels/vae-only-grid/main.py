#!/usr/bin/env python
"""VAE-only grid to fill the gap left by the VAE eval bug in v1."""
import os, sys, json, time, glob, traceback

os.environ["MUJOCO_GL"] = "egl"

import subprocess
PKG_ROOT = "/kaggle/input/datasets/sehajrsingh/tolerance-pkg"
WHEELS_DIR = os.path.join(PKG_ROOT, "wheels")
if os.path.isdir(WHEELS_DIR):
    whls = glob.glob(os.path.join(WHEELS_DIR, "*.whl"))
    print(f"[wheels] installing {len(whls)} bundled wheels")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps"] + whls)
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "mujoco"])

SRC_DIR = os.path.join(PKG_ROOT, "src")
sys.path.insert(0, SRC_DIR)
print(f"[pkg] added {SRC_DIR}")

import numpy as np
import torch

from tolerance.envs.planar_insertion import PlanarInsertion
from tolerance.envs.cable_routing import CableRouting
from tolerance.policies.vae_policy import VAEPolicy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[dev] device={DEVICE}")

CLEARANCES = [0.0005, 0.001, 0.002, 0.004]
WIDTHS = [32, 128, 256]
BUDGETS = [20, 60, 160]
N_SEEDS = 5
N_EVAL = 20


def collect_insertion_demos(clearance, n_demos, seed):
    from tolerance.policies.expert import DitherExpert
    rng = np.random.default_rng(seed)
    env = PlanarInsertion(clearance=clearance, seed=seed)
    expert = DitherExpert(rng=rng)
    obs_list, act_list = [], []
    attempts, successes = 0, 0
    max_attempts = max(4 * n_demos, 40)
    while successes < n_demos and attempts < max_attempts:
        attempts += 1
        o = env.reset()
        expert.reset(o[4], env.y_noise)
        done = False
        while not done:
            a = expert.act(o)
            o = env.step(a)
            done = env.done
        if env.success:
            successes += 1
            traj = env.trajectory()
            obs_list.append(traj["obs"])
            act_list.append(traj["actions"])
    if not obs_list:
        return np.zeros((0, 8), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(obs_list).astype(np.float32), np.concatenate(act_list).astype(np.float32)


def collect_cable_demos(n_demos, seed):
    class CableExpert:
        def act(self, obs):
            tip, target = obs[6:9], obs[9:12]
            err = target - tip
            a = np.zeros(5, dtype=np.float64)
            a[0] = np.clip(err[0] * 5, -1, 1)
            a[1] = np.clip(err[1] * 5, -1, 1)
            a[2] = 0.3
            return a
    env = CableRouting(seed=seed)
    expert = CableExpert()
    obs_list, act_list = [], []
    for _ in range(n_demos):
        obs = env.reset()
        for t in range(800):
            a = expert.act(obs)
            obs_list.append(obs[:18])
            act_list.append(a[:5])
            obs, r, done, info = env.step(a)
            if done:
                break
    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def train_vae(X, Y, width, seed, n_epochs=200):
    obs_dim, act_dim = X.shape[1], Y.shape[1]
    model = VAEPolicy(
        obs_dim=obs_dim, action_dim=act_dim, latent_dim=min(width, 32),
        hidden_dim=width, action_horizon=1, obs_horizon=1,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.as_tensor(X, device=DEVICE).unsqueeze(1)
    Y_t = torch.as_tensor(Y, device=DEVICE).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            tol = torch.zeros(xb.shape[0], 2, device=DEVICE)
            _, _, total_loss = model(xb, yb, tol)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
    return model


def eval_insertion_vae(policy, clearance, seed, n_episodes=N_EVAL):
    env = PlanarInsertion(clearance=clearance, seed=seed + 1000)
    successes = 0
    for _ in range(n_episodes):
        o = env.reset()
        for t in range(1200):
            obs_np = o[:8].astype(np.float32)
            tol_np = np.array([1.0], dtype=np.float32)
            a = policy.act(obs_np, tol_np)[:2]
            o = env.step(np.clip(a, -1.0, 1.0))
            if env.done:
                if env.success:
                    successes += 1
                break
    return successes / n_episodes


def eval_cable_vae(policy, seed, n_episodes=N_EVAL):
    env = CableRouting(seed=seed + 1000)
    successes = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for t in range(800):
            obs_np = obs[:18].astype(np.float32)
            tol_np = np.array([1.0], dtype=np.float32)
            a = policy.act(obs_np, tol_np)[:5]
            obs, r, done, info = env.step(np.clip(a, -1.0, 1.0))
            if done:
                if info.get("success", False):
                    successes += 1
                break
    return successes / n_episodes


# ── main grid ─────────────────────────────────────────────────────────
RESULTS_FILE = "/kaggle/working/results_vae.json"

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

results = load_results()
done_cells = {(r["task"], r["clearance"], r["width"], r["budget"], r["seed"]) for r in results}

grid = []
for c in CLEARANCES:
    for w in WIDTHS:
        for n in BUDGETS:
            for s in range(N_SEEDS):
                grid.append(("planar_insertion", c, w, n, s))
for w in WIDTHS:
    for n in BUDGETS:
        for s in range(N_SEEDS):
            grid.append(("cable_routing", 0.0, w, n, s))

total = len(grid)
done_count = len(results)
print(f"[grid] {done_count}/{total} cells done")

t0 = time.time()
for task_key, clearance, width, budget, seed in grid:
    cell = (task_key, clearance, width, budget, seed)
    if cell in done_cells:
        continue

    print(f"[cell] {task_key} c={clearance} w={width} N={budget} s={seed} ", end="", flush=True)
    try:
        t_start = time.time()
        if task_key == "planar_insertion":
            X, Y = collect_insertion_demos(clearance, budget, seed)
        else:
            X, Y = collect_cable_demos(budget, seed)

        if len(X) == 0:
            print("no data, skip")
            continue

        policy = train_vae(X, Y, width, seed)

        if task_key == "planar_insertion":
            sr = eval_insertion_vae(policy, clearance, seed)
        else:
            sr = eval_cable_vae(policy, seed)

        elapsed = time.time() - t_start
        result = {
            "task": task_key, "method": "vae",
            "clearance": clearance, "width": width,
            "budget": budget, "seed": seed,
            "success_rate": sr, "n_samples": len(X),
            "elapsed": round(elapsed, 1),
        }
        results.append(result)
        save_results(results)
        done_count += 1
        print(f"sr={sr:.3f} ({elapsed:.1f}s) [{done_count}/{total}]")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        results.append({
            "task": task_key, "method": "vae",
            "clearance": clearance, "width": width,
            "budget": budget, "seed": seed,
            "success_rate": -1, "error": str(e)[:200],
        })
        save_results(results)
        done_count += 1

print(f"\n[done] {done_count}/{total} in {time.time()-t0:.0f}s")

print("\n=== VAE SUMMARY ===")
valid = [r for r in results if r.get("success_rate", -1) >= 0]
for task in ["planar_insertion", "cable_routing"]:
    print(f"\n--- {task} ---")
    clearances = CLEARANCES if task == "planar_insertion" else [0.0]
    for c in clearances:
        for w in WIDTHS:
            for n in BUDGETS:
                vals = [r["success_rate"] for r in valid
                        if r["task"] == task and r["clearance"] == c
                        and r["width"] == w and r["budget"] == n]
                if vals:
                    print(f"  c={c} w={w} N={n}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
