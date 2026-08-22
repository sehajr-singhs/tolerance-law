#!/usr/bin/env python
"""Quick boundary check: does BC show the clearance boundary?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, time, torch
from tolerance.envs.planar_insertion import PlanarInsertion
from tolerance.policies.expert import DitherExpert
from tolerance.train.bc import collect_demos, train_bc

print("=" * 70)
print("QUICK BOUNDARY CHECK — corrected expert, reduced noise")
print("=" * 70)

# 4 clearances × 3 widths, N=60, 3 seeds
for c in [0.0005, 0.001, 0.002, 0.004]:
    print(f"\n--- c = {c*1000:.1f} mm ---")
    for w in [32, 128, 256]:
        scores = []
        for seed in range(3):
            t0 = time.time()
            rng = np.random.default_rng(seed)
            env = PlanarInsertion(clearance=c, seed=seed, rng=rng)
            expert = DitherExpert(rng=np.random.default_rng(seed))
            
            # collect demos
            dataset = collect_demos(env, expert, n_demos=60, seed=seed)
            teacher = dataset["success_rate"]
            
            # train
            policy, hist = train_bc(dataset, width=w, depth=3, epochs=40, 
                                    lr=1e-3, batch_size=256, seed=seed)
            
            # eval
            wins = 0
            for s in range(40):
                rng2 = np.random.default_rng(1000 + s)
                env2 = PlanarInsertion(clearance=c, seed=1000+s, rng=rng2)
                obs = env2.reset()
                for step in range(1200):
                    with torch.no_grad():
                        a = policy(torch.as_tensor(obs, dtype=torch.float32)).numpy()
                    a = np.clip(a, -1.0, 1.0)
                    obs = env2.step(a)
                    if env2.done:
                        break
                if env2.success:
                    wins += 1
            
            elapsed = time.time() - t0
            score = wins / 40
            scores.append(score)
            print(f"  w={w:3d} seed={seed} | teacher={teacher:.2f} test={score:.2f} ({elapsed:.1f}s)")
        
        avg = np.mean(scores)
        std = np.std(scores)
        print(f"  w={w:3d} AVG: {avg:.2f} +/- {std:.2f}")
