# Robotic Policies for Engineering & Manufacturing

**Working title: *The Tolerance Law — phase boundaries in learning contact-rich manufacturing skills.***

A research program on when — and how — learned robotic policies can automate
manufacturing tasks, with the engineering parameters that manufacturers actually
control (part clearance, tooling geometry, measurement uncertainty) treated as the
*control variables of policy learnability*.

This is a standalone project: a different scientific question in a different
domain — **what makes a manufacturing skill learnable, and how do tolerance,
tooling, and measurement govern that boundary?** — with its own codebase,
its own experiments, and its own papers.

---

## The one-line claim

> For any policy class, there is a sharp boundary in engineering-parameter space
> (clearance ratio, tool geometry, measurement noise) across which a
> contact-rich manufacturing skill transitions from *learnable* to *not
> learnable* — the boundary is monotone, capacity-dependent, predictable from a
> stylized model of contact ambiguity, and exploitable by an adaptive controller
> that tunes the operating envelope from in-process measurements.

## Why this is NMI-shaped

Nature Machine Intelligence publishes *mechanisms* with broad significance, not
task wins. The tactile-manipulation taxonomy paper (Johannsmeier et al.,
*Nat. Mach. Intell.* 7(6), 2025) shows the bar: a conceptual contribution that
organizes and predicts across a domain. This project's candidate is stronger in
one respect and weaker in another:

- **Stronger:** a *quantitative* law (sample-complexity / success-boundary as a
  function of clearance) with falsifiable predictions and a design rule
  manufacturers can use ("at clearance X, you need ≈Y demonstrations, and here is
  the measurement precision that gates the achievable yield").
- **Weaker:** laws need multi-task evidence; a single-task result is RA-L/ICRA
  grade. The roadmap (below) sequences single-task law → two-task
  generalization → adaptive controller → NMI submission.

## The three stakeholders this is built for

| Stakeholder | Their business | The research hook |
|---|---|---|
| **Mind Robotics** | Rivian spin-off ($1B+ raised): foundation models + purpose-built hardware + a *manufacturing data flywheel* for dexterous, reasoning-intensive factory tasks | The data flywheel only pays off if policies learn from factory data at tolerance-relevant difficulty; the Tolerance Law tells them which tasks are learnable from their data budget — and the adaptive controller tells them which operating envelope to run |
| **Gimatic** (Barnes Group) | World leader in end-of-arm tooling: electric/pneumatic grippers, vacuum, sensors, tool changers | Tool geometry is the hardware prior policies must generalize across; tool-conditioned policies turn EOAT catalog geometry into a policy embedding — a product-direction and a dataset |
| **Renishaw** | Precision metrology + additive manufacturing; Equator automated gauging; Renishaw Central process-data platform | Metrology is the ground-truth reward. Measurement-in-the-loop policy learning turns their gauging cells and data platform into a closed-loop learning system |

## Repository layout

```
research/          the research foundation (this is the current focus)
  problem_statement.md   the gap, the question, hypotheses
  literature_review.md   organized, cited landscape
  mechanism.md           the Tolerance Law: model, predictions, negatives
  experiment_plan.md     R0–R4 phases, sim suites, compute, statistics
  stakeholder_map.md     Mind / Gimatic / Renishaw intelligence + asks
proposals/
  mind_robotics_internship.md   the internship pitch
  funding_gimatic_renishaw.md   the two funding asks
```

## Status

- [x] Stakeholder research (Mind Robotics, Gimatic, Renishaw)
- [x] Literature landscape + gap analysis
- [x] Mechanism draft with falsifiable predictions
- [x] Experiment plan and compute strategy
- [x] Internship pitch + funding asks
- [ ] R1: stylized-model derivation of the capacity law (simulation + closed form)
- [ ] R2: single-task clearance phase diagram (MuJoCo/robosuite, Kaggle GPU)
- [ ] R3: tool-conditioning + measurement-in-the-loop suites
- [ ] R4: adaptive tolerance controller, multi-task evidence, NMI manuscript

## Compute strategy

Kaggle GPU kernels (dataset + kernel pattern with bundled wheels for offline
install). Python 3.12, MuJoCo, PyTorch. No external cloud credentials required.
The state phase-diagram kernel runs the full clearance × capacity × budget
grid plus both adaptive controllers.

## Current results

The clearance phase diagram is measured (state policies): the learnable-clearance
boundary $c^*$ moves as a power law of the demo budget and is non-monotone in
model capacity at fixed budget. See `paper/nmi_paper.tex` and `docs/` for the
write-up and site; every number is injected from committed result JSONs.
