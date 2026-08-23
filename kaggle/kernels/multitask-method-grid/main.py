#!/usr/bin/env python
"""Multi-Task Multi-Method Tolerance Law: Full Evidence Grid.

2 tasks (PlanarInsertion, CableRouting) × 4 methods (BC-MLP, Diffusion, VAE, TCP)
× 4 clearances × 3 budgets × 5 seeds.

Runs on Kaggle GPU. Saves results incrementally to survive disconnection.
"""
import os, sys, json, time, glob, traceback

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
    print("[wheels] no wheels dir, pip install mujoco")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "mujoco"])

SRC_DIR = os.path.join(PKG_ROOT, "src")
sys.path.insert(0, SRC_DIR)
print(f"[pkg] added {SRC_DIR}")

import numpy as np
import torch

from tolerance.envs.planar_insertion import PlanarInsertion
from tolerance.envs.cable_routing import CableRouting
from tolerance.policies.expert import DitherExpert
from tolerance.policies.mlp import BCMLP
from tolerance.policies.diffusion import DiffusionPolicy
from tolerance.policies.vae_policy import VAEPolicy
from tolerance.policies.tcp import ToleranceConditionedMLP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[dev] device={DEVICE}  torch={torch.__version__}")

# ══════════════════════════════════════════════════════════════════════
# Cable routing expert (proportional controller)
# ══════════════════════════════════════════════════════════════════════
class CableExpert:
    def __init__(self, env):
        pass
    def act(self, obs):
        tip = obs[6:9]
        target = obs[9:12]
        err = target - tip
        a = np.zeros(5, dtype=np.float64)
        a[0] = np.clip(err[0] * 5, -1, 1)
        a[1] = np.clip(err[1] * 5, -1, 1)
        a[2] = 0.3
        return a

# ══════════════════════════════════════════════════════════════════════
# Task configs
# ══════════════════════════════════════════════════════════════════════
TASKS = {
    "planar_insertion": {
        "env_cls": PlanarInsertion,
        "obs_dim": 8,
        "act_dim": 2,
        "horizon": 1200,
        "expert_cls": DitherExpert,
    },
    "cable_routing": {
        "env_cls": CableRouting,
        "obs_dim": 18,
        "act_dim": 5,
        "horizon": 800,
        "expert_cls": CableExpert,
    },
}

CLEARANCES_PI = [0.0005, 0.001, 0.002, 0.004]  # for planar_insertion
WIDTHS = [32, 128, 256]
BUDGETS = [20, 60, 160]
N_SEEDS = 5
N_EVAL = 20  # evaluation episodes

# ══════════════════════════════════════════════════════════════════════
# Data collection
# ══════════════════════════════════════════════════════════════════════
def collect_insertion_demos(clearance, n_demos, seed):
    """Collect expert demos for planar insertion at a given clearance."""
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

    X = np.concatenate(obs_list).astype(np.float32)
    Y = np.concatenate(act_list).astype(np.float32)
    return X, Y


def collect_cable_demos(n_demos, seed):
    """Collect cable routing expert demos."""
    env = CableRouting(seed=seed)
    expert = CableExpert(env)

    obs_list, act_list = [], []
    horizon = TASKS["cable_routing"]["horizon"]

    for _ in range(n_demos):
        obs = env.reset()
        for t in range(horizon):
            a = expert.act(obs)
            obs_list.append(obs[:TASKS["cable_routing"]["obs_dim"]])
            act_list.append(a[:TASKS["cable_routing"]["act_dim"]])
            obs, r, done, info = env.step(a)
            if done:
                break

    X = np.array(obs_list, dtype=np.float32)
    Y = np.array(act_list, dtype=np.float32)
    return X, Y


# ══════════════════════════════════════════════════════════════════════
# Training functions
# ══════════════════════════════════════════════════════════════════════
def train_bc_mlp(X, Y, width, seed, n_epochs=200):
    obs_dim, act_dim = X.shape[1], Y.shape[1]
    model = BCMLP(obs_dim=obs_dim, act_dim=act_dim, width=width, depth=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.as_tensor(X, device=DEVICE)
    Y_t = torch.as_tensor(Y, device=DEVICE)
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_diffusion(X, Y, width, seed, n_epochs=200):
    obs_dim, act_dim = X.shape[1], Y.shape[1]
    model = DiffusionPolicy(
        obs_dim=obs_dim, action_dim=act_dim, hidden_dim=width,
        n_diffusion_steps=20, action_horizon=1,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.as_tensor(X, device=DEVICE)
    Y_t = torch.as_tensor(Y, device=DEVICE).reshape(len(Y), -1)
    dataset = torch.utils.data.TensorDataset(X_t, Y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            loss = model.compute_loss(yb, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


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


def train_tcp(X, Y, width, seed, n_epochs=200):
    obs_dim, act_dim = X.shape[1], Y.shape[1]
    model = ToleranceConditionedMLP(
        obs_dim=obs_dim, action_dim=act_dim, tolerance_dim=1, hidden_dim=width,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.as_tensor(X, device=DEVICE)
    Y_t = torch.as_tensor(Y, device=DEVICE)
    tol = torch.ones(len(X), 1, device=DEVICE)
    dataset = torch.utils.data.TensorDataset(X_t, Y_t, tol)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for xb, yb, tb in loader:
            pred = model(xb, tb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════
def eval_insertion(policy, clearance, seed, method, n_episodes=N_EVAL):
    """Evaluate a policy on planar insertion."""
    env = PlanarInsertion(clearance=clearance, seed=seed + 1000)
    obs_dim = TASKS["planar_insertion"]["obs_dim"]
    act_dim = TASKS["planar_insertion"]["act_dim"]
    horizon = TASKS["planar_insertion"]["horizon"]

    successes = 0
    for _ in range(n_episodes):
        o = env.reset()
        for t in range(horizon):
            obs_np = o[:obs_dim].astype(np.float32)
            ob_t = torch.as_tensor(obs_np, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if method == "diffusion":
                    a = policy.generate(ob_t, n_steps=5)
                    a = a[0, :, 0].cpu().numpy()[:act_dim]
                elif method == "vae":
                    a = policy.act(obs_np, np.array([1.0], dtype=np.float32))
                    a = a[:act_dim]
                elif method == "tcp":
                    a = policy.act(obs_np, np.array([1.0], dtype=np.float32))
                    a = a[:act_dim]
                else:
                    a = policy(ob_t)[0].cpu().numpy()[:act_dim]
            o = env.step(np.clip(a, -1.0, 1.0))
            if env.done:
                if env.success:
                    successes += 1
                break
    return successes / n_episodes


def eval_cable(policy, seed, method, n_episodes=N_EVAL):
    """Evaluate a policy on cable routing."""
    env = CableRouting(seed=seed + 1000)
    obs_dim = TASKS["cable_routing"]["obs_dim"]
    act_dim = TASKS["cable_routing"]["act_dim"]
    horizon = TASKS["cable_routing"]["horizon"]

    successes = 0
    for _ in range(n_episodes):
        obs = env.reset()
        for t in range(horizon):
            obs_np = obs[:obs_dim].astype(np.float32)
            ob_t = torch.as_tensor(obs_np, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                if method == "diffusion":
                    a = policy.generate(ob_t, n_steps=5)
                    a = a[0, :, 0].cpu().numpy()[:act_dim]
                elif method == "vae":
                    a = policy.act(obs_np, np.array([1.0], dtype=np.float32))
                    a = a[:act_dim]
                elif method == "tcp":
                    a = policy.act(obs_np, np.array([1.0], dtype=np.float32))
                    a = a[:act_dim]
                else:
                    a = policy(ob_t)[0].cpu().numpy()[:act_dim]
            obs, r, done, info = env.step(np.clip(a, -1.0, 1.0))
            if done:
                if info.get("success", False):
                    successes += 1
                break
    return successes / n_episodes


# ══════════════════════════════════════════════════════════════════════
# Main grid
# ══════════════════════════════════════════════════════════════════════
RESULTS_FILE = "/kaggle/working/results_multitask.json"

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

results = load_results()
done_cells = {(r["task"], r["method"], r["clearance"], r["width"], r["budget"], r["seed"])
              for r in results}

# Build the full grid
grid = []
for task_key in TASKS:
    clearances = CLEARANCES_PI if task_key == "planar_insertion" else [0.0]  # cable has no clearance param
    for clearance in clearances:
        for width in WIDTHS:
            for budget in BUDGETS:
                for seed in range(N_SEEDS):
                    grid.append((task_key, clearance, width, budget, seed))

total = len(grid)
done_count = len(results)
print(f"[grid] {done_count}/{total} cells done, resuming...")

t0 = time.time()
TRAIN_FUNCS = {
    "bc_mlp": train_bc_mlp,
    "diffusion": train_diffusion,
    "vae": train_vae,
    "tcp": train_tcp,
}

for task_key, clearance, width, budget, seed in grid:
    # Collect data (once per task/clearance/budget/seed — shared across methods)
    data_key = (task_key, clearance, budget, seed)
    if task_key == "planar_insertion":
        X, Y = collect_insertion_demos(clearance, budget, seed)
    else:
        X, Y = collect_cable_demos(budget, seed)

    n_samples = len(X)
    if n_samples == 0:
        print(f"[skip] {task_key} c={clearance} N={budget} s={seed}: no data")
        continue

    for method, train_fn in TRAIN_FUNCS.items():
        cell = (task_key, method, clearance, width, budget, seed)
        if cell in done_cells:
            continue

        print(f"[cell] {task_key}/{method} c={clearance} w={width} N={budget} s={seed} ",
              end="", flush=True)

        try:
            t_start = time.time()

            # Train
            policy = train_fn(X, Y, width, seed)

            # Evaluate
            if task_key == "planar_insertion":
                sr = eval_insertion(policy, clearance, seed, method)
            else:
                sr = eval_cable(policy, seed, method)

            elapsed = time.time() - t_start
            result = {
                "task": task_key, "method": method,
                "clearance": clearance, "width": width,
                "budget": budget, "seed": seed,
                "success_rate": sr, "n_samples": n_samples,
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
                "task": task_key, "method": method,
                "clearance": clearance, "width": width,
                "budget": budget, "seed": seed,
                "success_rate": -1, "error": str(e)[:200],
            })
            save_results(results)
            done_count += 1

elapsed_total = time.time() - t0
print(f"\n[done] {done_count}/{total} cells in {elapsed_total:.0f}s")

# ── summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for task_key in TASKS:
    clearances = CLEARANCES_PI if task_key == "planar_insertion" else [0.0]
    print(f"\n--- {task_key} ---")
    valid = [r for r in results if r["task"] == task_key and r.get("success_rate", -1) >= 0]
    for method in ["bc_mlp", "diffusion", "vae", "tcp"]:
        print(f"\n  {method}:")
        print(f"  {'c':>8s} {'w':>4s} {'N':>4s} {'sr':>6s} {'std':>6s}")
        for c in clearances:
            for w in WIDTHS:
                for n in BUDGETS:
                    vals = [r["success_rate"] for r in valid
                            if r["method"] == method and r["clearance"] == c
                            and r["width"] == w and r["budget"] == n]
                    if vals:
                        print(f"  {c:8.4f} {w:4d} {n:4d} {np.mean(vals):6.3f} {np.std(vals):6.3f}")
                    else:
                        print(f"  {c:8.4f} {w:4d} {n:4d} {'---':>6s}")
