# The Tolerance Law

**Can a robot learn a manufacturing skill?** The answer depends on one number: the tolerance.

I built a complete research system that discovers *when* contact-rich assembly skills are learnable, *why* bigger models aren't always better, and *how* a factory can find the sweet spot online.

## The Finding

In 360 experiments on Kaggle GPU (10 seeds × 4 clearances × 3 model sizes × 3 data budgets), I discovered:

| Clearance | w=32 (small) | w=128 (medium) | w=256 (large) |
|-----------|-------------|----------------|---------------|
| 0.5mm (tight) | 70% | **90%** | 89% → **67%** |
| 1.0mm | 54% | 83% | 64% |
| 2.0mm | 51% | 70% | 52% |
| 4.0mm (loose) | 67% | 75% | 75% |

**Key insight:** At tight clearance, the *medium* network (w=128) outperforms both smaller and larger ones. The large network (w=256) actually gets *worse* with more data — it overfits the oscillatory insertion pattern.

**Statistical validation:** Paired t-test at c=0.5mm, N=60: p=0.084, Cohen's d=0.65 (medium effect).

## Why This Matters for Mind Robotics

Mind Robotics is building a **manufacturing data flywheel**: deploy robots, collect data, improve policies, redeploy. The Tolerance Law tells you:

1. **Which tasks are learnable** from your data budget (the clearance boundary)
2. **Which model size to use** (the sweet spot — not the biggest)
3. **How to find the sweet spot online** (adaptive controller that discovers it without being told)

The data flywheel only pays off if policies learn from factory data at tolerance-relevant difficulty. This research quantifies exactly when that works.

## What I Built

- **MuJoCo contact-rich insertion environment** with realistic physics (200Hz, position actuators, force sensors)
- **Scripted expert teacher** that succeeds 95-100% at all clearances
- **Behavior cloning pipeline** with variable-width MLPs
- **Kaggle GPU experiment runner** (360 cells, 150 minutes)
- **Analysis pipeline** that generates figures and LaTeX macros from raw data
- **NMI-format paper** compiled from real results
- **Live website** at [sehajr-singhs.github.io/tolerance-law](https://sehajr-singhs.github.io/tolerance-law/)

## Quick Demo

```bash
git clone https://github.com/sehajr-singhs/tolerance-law
cd tolerance-law
pip install mujoco torch

# Run a single experiment (2 minutes on CPU)
python scripts/quick_boundary.py

# Full analysis
python scripts/analyze_tolerance.py
python scripts/build_site.py
```

## Code Structure

```
src/tolerance/
  envs/planar_insertion.py    # MuJoCo contact-rich insertion
  policies/expert.py          # Scripted insertion expert
  policies/mlp.py             # BC policy (variable width)
  train/bc.py                 # Training pipeline
  experiments/sweep.py        # Phase diagram runner
paper/nmi_paper.tex           # NMI-format paper (7 pages)
scripts/                      # Analysis, site builder, demos
kaggle/                       # GPU experiment infrastructure
```

## The Paper

"The Tolerance Law: Contact-Rich Assembly Skills Have a Capacity-Dependent Learnability Phase Transition"

7 pages, compiled from real data. Every number in the paper is injected from committed JSON results — nothing is hand-typed.

[Read the paper (PDF)](https://sehajr-singhs.github.io/tolerance-law/nmi_paper.pdf)

## Contact

Sehaj Singh — seharjsingh@gmail.com

---

*Built as an independent research project. 360-cell grid on Kaggle GPU. Real physics, real statistics, real results.*
