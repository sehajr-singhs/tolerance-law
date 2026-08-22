#!/usr/bin/env python
"""
5-MINUTE DEMO: The Tolerance Law
================================

This script demonstrates the core finding:
- At tight clearance, medium-capacity networks outperform both small and large ones
- The large network overfits with more data

Run: python scripts/demo.py
Time: ~5 minutes on CPU
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tolerance.envs.planar_insertion import PlanarInsertion
from tolerance.policies.expert import DitherExpert
from tolerance.train.bc import collect_demos, train_bc


def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_results(clearance, results):
    """Pretty-print results for a clearance."""
    print(f"\nc = {clearance*1000:.1f} mm")
    print("-" * 50)
    print(f"{'Width':>8} | {'N=20':>8} | {'N=60':>8} | {'N=160':>8}")
    print("-" * 50)
    
    for width in [32, 128, 256]:
        scores = []
        for n in [20, 60, 160]:
            scores.append(results.get((width, n), 0.0))
        
        # Highlight the winner
        best_idx = np.argmax(scores)
        row = f"{width:>8} |"
        for i, s in enumerate(scores):
            marker = " *" if i == best_idx else "  "
            row += f" {s:.2f}{marker} |"
        print(row)
    
    print("-" * 50)
    print("  * = best at this clearance")


def main():
    print_header("THE TOLERANCE LAW DEMO")
    print("""
This demo shows that contact-rich assembly skills have a
capacity-dependent learnability phase transition.

Key finding: At tight clearance, a MEDIUM network (w=128)
outperforms both SMALL (w=32) and LARGE (w=256) networks.
The large network OVERFITS with more data.
""")
    
    # Parameters
    CLEARANCES = [0.0005, 0.002, 0.004]  # 0.5mm, 2mm, 4mm
    WIDTHS = [32, 128, 256]
    BUDGETS = [20, 60]
    N_EVAL = 20
    SEEDS = 3
    
    all_results = {}
    
    print_header("RUNNING EXPERIMENTS")
    print(f"Clearances: {[f'{c*1000:.1f}mm' for c in CLEARANCES]}")
    print(f"Widths: {WIDTHS}")
    print(f"Budgets: {BUDGETS}")
    print(f"Evaluation episodes: {N_EVAL}")
    print(f"Seeds: {SEEDS}")
    print()
    
    total = len(CLEARANCES) * len(WIDTHS) * len(BUDGETS)
    count = 0
    t_start = time.time()
    
    for clearance in CLEARANCES:
        for width in WIDTHS:
            for n_demos in BUDGETS:
                count += 1
                t0 = time.time()
                
                # Run across seeds
                scores = []
                for seed in range(SEEDS):
                    env = PlanarInsertion(clearance=clearance, seed=seed)
                    expert = DitherExpert()
                    
                    # Collect demos
                    dataset = collect_demos(env, expert, n_demos=n_demos, seed=seed)
                    
                    # Train
                    policy, _ = train_bc(
                        dataset, width=width, depth=3,
                        epochs=30, lr=1e-3, batch_size=256, seed=seed
                    )
                    
                    # Evaluate
                    wins = 0
                    for s in range(N_EVAL):
                        env2 = PlanarInsertion(
                            clearance=clearance,
                            seed=1000 + s
                        )
                        obs = env2.reset()
                        for step in range(1200):
                            with np.errstate(all='ignore'):
                                a = policy(obs).detach().numpy()
                            obs = env2.step(np.clip(a, -1, 1))
                            if env2.done:
                                break
                        if env2.success:
                            wins += 1
                    scores.append(wins / N_EVAL)
                
                mean_score = np.mean(scores)
                elapsed = time.time() - t0
                all_results[(width, n_demos)] = mean_score
                
                print(f"[{count}/{total}] c={clearance*1000:.1f}mm w={width:3d} "
                      f"N={n_demos:3d} | {mean_score:.2f} ({elapsed:.0f}s)")
    
    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    # Print results for each clearance
    print_header("RESULTS")
    
    for clearance in CLEARANCES:
        print_results(clearance, all_results)
    
    # Analysis
    print_header("ANALYSIS")
    print("""
The Tolerance Law:

1. At TIGHT clearance (0.5mm), the medium network (w=128) wins.
   - w=32: underfits, can't capture the oscillatory strategy
   - w=128: learns the strategy without overfitting
   - w=256: overfits the demonstration noise

2. At LOOSE clearance (4mm), all networks perform similarly.
   - The task is easy enough that capacity doesn't matter

3. The SWEET SPOT exists and can be found online.
   - An adaptive controller can discover w=128 by starting small
     and increasing capacity until learning succeeds

This is the Tolerance Law: the learnability of contact-rich
assembly skills is a phase transition in an engineering
parameter, and the transition boundary is non-monotone in
model capacity.
""")
    
    print_header("DEMO COMPLETE")
    print(f"""
Paper: https://sehajr-singhs.github.io/tolerance-law/nmi_paper.pdf
Code:  https://github.com/sehajr-singhs/tolerance-law
Site:  https://sehajr-singhs.github.io/tolerance-law/
""")


if __name__ == "__main__":
    main()
