# Experiment plan (R0–R4)

## Phase map

| Phase | Deliverable | Venue target | Compute |
|---|---|---|---|
| R0 | Research foundation (this repo) | — | none |
| R1 | Stylized-model derivation + sim calibration of the capacity law | internal | CPU |
| R2 | Single-task clearance phase diagram (insertion) + law fit | RA-L / ICRA | Kaggle GPU |
| R3 | Tool-conditioning (Gimatic axis) + measurement-in-the-loop (Renishaw axis) | ICRA / RSS | Kaggle GPU |
| R4 | Adaptive tolerance controller + multi-task evidence + NMI manuscript | NMI | Kaggle GPU + optional hardware |

## R1 — The stylized model and calibration

- Derive the capacity law from the quasi-static insertion model in
  `mechanism.md` (closed-form ĉ* ~ N^{−α} with an explicit α from the geometry).
- Calibrate against a *single well-understood* insertion task in MuJoCo to
  check the functional form and exponent before committing GPU hours.
- Success criteria: the simulated boundary is monotone and the fitted exponent
  has a stable value across two seeds of initialization.

## R2 — The clearance phase diagram (the core experiment)

**Task.** Precision insertion with swept clearance: peg-in-hole where the
clearance ratio ĉ is swept over ~2 orders of magnitude (e.g., 0.5% → 50% of
diameter, log-spaced, ~10 levels). Primary: rigid peg-in-hole. Secondary (to
make the law a law, not a curve): a snap-fit / press-fit variant.

**Policy classes.** (a) BC-MLP on state+force; (b) BC-CNN on images;
(c) tactile-augmented BC (contact-force features); (d) RL (SAC with the
standard double-Q/tuning discipline); (e) a diffusion policy (VLA-lite) as the
foundation-model proxy. Each at 2–3 capacities.

**Data.** Expert demonstrations from a scripted/teleop expert with a
controllable sweep policy; the data axis is swept directly (budget N, and
label noise) as controlled variables.

**Grid.** clearance × policy-class × capacity × {1, 3, 5} seeds. Each cell
reports: final success, train/val divergence (memorization proxy), oracle-query
count (if DAgger-style), and the fitted boundary location.

**Output.** The **success-vs-clearance phase diagram** (P1), the **capacity
law fit** ĉ* ~ N^{−α} (P2), and negative control 1 (fixed-N, epoch-swept) to
establish the wall is structural.

## R3 — Two generalization axes

### R3a. Tool-conditioning (Gimatic axis)
Same insertion/grasp task with EOAT geometry swept along catalog-plausible
dimensions: parallel-jaw width, fingertip compliance (soft vs rigid), vacuum
cup, 2-finger vs 3-finger. Train tool-conditioned policies (geometry embedding
as input) and measure **transfer cost vs. geometry distance** (P3). Deliverable
a *tool-conditioned policy zoo* + the transfer law — the publishable core of
the Gimatic partnership.

### R3b. Measurement-in-the-loop (Renishaw axis)
A gauging loop: post-insertion measurements (simulated CMM/Equator with
realistic noise σ swept over an order of magnitude) define the *reward* and the
*yield metric* (simulated Cp/Cpk). Learn without hand-crafted shaping; measure
the **noise floor** (P4) — yield saturates as σ↑ regardless of policy budget.
Deliverable: the σ-boundary curve + a quantitative "gauge precision buys
process window" statement.

## R4 — The adaptive tolerance controller + the paper

- **Adaptive tolerance controller**: closes the loop on in-process yield;
  tightens the operating envelope when yield is high, backs off when it drops,
  converges near the measured boundary without knowing it (P5). The same
  closed-loop discipline as the adaptive-budget probe in the R2 grid, with the
  controlled variable being the engineering parameter itself.
- **Multi-task evidence**: insertion, tool-conditioned placement, and
  gauged-process-control each show the boundary; the law is fit within each
  family. This is the NMI-grade package.
- **Manuscript**: Nature Machine Intelligence format (the repo ships a
  measurement-to-macro pipeline: every number in the paper is injected from
  committed result JSONs).
- **Stretch**: hardware validation on an instrumented cell (funding-dependent;
  see proposals).

## Simulation and compute

- **Simulator:** MuJoCo (industry-standard, credible with reviewers) +
  robosuite for standard assembly task baselines; a tolerance-parameterized
  custom env keeps the policy stack swappable across tasks.
- **Compute:** Kaggle GPU kernels (Python 3.12, bundled wheels for offline
  install so no internet egress is required). Dataset-versioned code so every
  result is reproducible from kernel → JSON → figure → paper.
- **Budget estimate:** R2 ≈ 6–10 kernels; R3 ≈ 6–8; R4 ≈ 4–6. Runs 1–4 h each.
  Fit within Kaggle's free weekly quota if spread across weeks; paid quota if a
  funding ask lands.

## Statistical protocol

- ≥3 seeds per cell; report median + spread.
- Boundary location estimated by logistic fit on the success-vs-clearance
  curve; exponent α by log-log regression with confidence intervals.
- Pre-register P1–P5 in `research/mechanism.md` before running R2 (done —
  that document is the pre-registration; freeze before kernel launch).
