"""Behavior cloning: dataset assembly, training, and evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..envs.planar_insertion import PlanarInsertion
from ..policies.cnn import BCCNN
from ..policies.mlp import BCMLP
from ..utils.vision import render_top


def collect_demos(env: PlanarInsertion, expert, n_demos: int,
                  max_attempts: int | None = None, seed: int = 0,
                  progress: bool = False) -> dict:
    """Run the expert and return {'obs': (N,8), 'actions': (N,2), 'success_rate'}.

    Collects up to n_demos successful episodes; stops early after max_attempts
    attempts (data-starved regime). Returns the raw dataset.
    """
    rng = np.random.default_rng(seed)
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    attempts = 0
    successes = 0
    max_attempts = max_attempts or max(4 * n_demos, 40)
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
            obs_all.append(traj["obs"])
            act_all.append(traj["actions"])
        if progress and attempts % 20 == 0:
            print(f"    expert {successes}/{n_demos} (attempts {attempts})", flush=True)
    if not obs_all:
        return {"obs": np.zeros((0, 8)), "actions": np.zeros((0, 2)),
                "success_rate": 0.0, "attempts": attempts, "episodes": 0}
    obs = np.concatenate(obs_all, axis=0)
    acts = np.concatenate(act_all, axis=0)
    return {"obs": obs, "actions": acts,
            "success_rate": successes / attempts, "attempts": attempts,
            "episodes": successes}


def train_bc(dataset: dict, width: int = 64, depth: int = 2, epochs: int = 30,
             batch_size: int = 256, lr: float = 1e-3, seed: int = 0,
             device: str = "cpu", eval_fn=None, progress: bool = False) -> tuple[BCMLP, dict]:
    """Train a BC policy. eval_fn(policy, epoch) -> float is called every 5 epochs."""
    torch.manual_seed(seed)
    obs = torch.as_tensor(dataset["obs"], dtype=torch.float32)
    acts = torch.as_tensor(dataset["actions"], dtype=torch.float32)
    n = obs.shape[0]
    model = BCMLP(obs_dim=obs.shape[1], act_dim=acts.shape[1],
                  width=width, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    history: dict[str, list] = {"train_loss": [], "eval": []}
    n_batches = max(1, n // batch_size)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        nb = 0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if idx.numel() == 0:
                continue
            ob, ac = obs[idx].to(device), acts[idx].to(device)
            pred = model(ob)
            loss = lossf(pred, ac)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        history["train_loss"].append(total / max(nb, 1))
        if eval_fn is not None and (ep + 1) % 5 == 0:
            score = eval_fn(model)
            history["eval"].append((ep + 1, score))
            if progress:
                print(f"    epoch {ep+1}/{epochs} loss={history['train_loss'][-1]:.4f} "
                      f"eval={score:.3f}", flush=True)
    return model, history


def collect_demos_vision(env: PlanarInsertion, expert, n_demos: int,
                         img_size: int = 64, max_attempts: int | None = None,
                         seed: int = 0, progress: bool = False) -> dict:
    """Expert demos as (image, action) pairs (vision modality)."""
    rng = np.random.default_rng(seed)
    imgs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    attempts = 0
    successes = 0
    max_attempts = max_attempts or max(4 * n_demos, 40)
    while successes < n_demos and attempts < max_attempts:
        attempts += 1
        o = env.reset()
        expert.reset(o[4], env.y_noise)
        done = False
        while not done:
            a = expert.act(o)
            img = render_top(env, size=img_size)
            imgs_all.append(img)
            act_all.append(a.copy())
            o = env.step(a)
            done = env.done
        if env.success:
            successes += 1
        if progress and attempts % 10 == 0:
            print(f"    expert-vision {successes}/{n_demos} (attempts {attempts})",
                  flush=True)
    if not imgs_all:
        return {"imgs": np.zeros((0, img_size, img_size, 3)),
                "actions": np.zeros((0, 2)), "success_rate": 0.0,
                "attempts": attempts, "episodes": 0}
    imgs = np.stack(imgs_all)
    acts = np.stack(act_all)
    return {"imgs": imgs, "actions": acts,
            "success_rate": successes / attempts, "attempts": attempts,
            "episodes": successes}


def train_bc_cnn(dataset: dict, channels: int = 16, depth: int = 3,
                 epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
                 seed: int = 0, device: str = "cpu", eval_fn=None,
                 progress: bool = False) -> tuple[BCCNN, dict]:
    """Train a vision BC policy. dataset has 'imgs' (N,H,W,3) + 'actions'."""
    torch.manual_seed(seed)
    imgs = torch.as_tensor(dataset["imgs"], dtype=torch.float32)
    # N,H,W,3 -> N,3,H,W
    imgs = imgs.permute(0, 3, 1, 2)
    acts = torch.as_tensor(dataset["actions"], dtype=torch.float32)
    n = imgs.shape[0]
    model = BCCNN(act_dim=acts.shape[1], channels=channels, depth=depth,
                  img_size=imgs.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    history: dict[str, list] = {"train_loss": [], "eval": []}
    n_batches = max(1, n // batch_size)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        nb = 0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if idx.numel() == 0:
                continue
            im, ac = imgs[idx].to(device), acts[idx].to(device)
            pred = model(im)
            loss = lossf(pred, ac)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        history["train_loss"].append(total / max(nb, 1))
        if eval_fn is not None and (ep + 1) % 5 == 0:
            score = eval_fn(model)
            history["eval"].append((ep + 1, score))
            if progress:
                print(f"    epoch {ep+1}/{epochs} loss={history['train_loss'][-1]:.4f} "
                      f"eval={score:.3f}", flush=True)
    return model, history


def eval_policy_vision(env: PlanarInsertion, policy, n_episodes: int = 30,
                       seed: int = 0, img_size: int = 64,
                       device: str = "cpu") -> dict:
    """Evaluate a vision policy (images in, actions out)."""
    rng = np.random.default_rng(seed)
    s = 0
    dists = []
    steps = []
    with torch.no_grad():
        for _ in range(n_episodes):
            o = env.reset()
            done = False
            while not done:
                img = render_top(env, size=img_size)
                im = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1)
                im = im.unsqueeze(0).to(device)
                a = policy(im).squeeze(0).cpu().numpy()
                o = env.step(a)
                done = env.done
            s += int(env.success)
            dists.append(env.final_dist_to_seat)
            steps.append(env.t)
    return {"success": s / n_episodes, "n": n_episodes,
            "final_dist": float(np.mean(dists)), "steps": float(np.mean(steps))}


def eval_policy(env: PlanarInsertion, policy, n_episodes: int = 30, seed: int = 0,
                expert=None, device: str = "cpu") -> dict:
    """Evaluate a policy. Returns success rate, mean final dist, mean steps."""
    rng = np.random.default_rng(seed)
    s = 0
    dists = []
    steps = []
    with torch.no_grad():
        for _ in range(n_episodes):
            o = env.reset()
            done = False
            while not done:
                ob = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device)
                a = policy(ob).squeeze(0).cpu().numpy()
                o = env.step(a)
                done = env.done
            s += int(env.success)
            dists.append(env.final_dist_to_seat)
            steps.append(env.t)
    return {"success": s / n_episodes, "n": n_episodes, "final_dist": float(np.mean(dists)),
            "steps": float(np.mean(steps))}
