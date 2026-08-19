# Problem statement

## The gap

Industrial automation today fails in one of two ways, and both are visible in
the 2025–2026 landscape:

1. **Classic industrial robotics** (the Gimatic/Renishaw world): hand-programmed
   cells, brittle to variation, one program per part per tolerance. Quality is
   enforced by *measurement after the fact* (gauging cells, CMM) and by
   tightening process windows — not by learning.
2. **Learned manipulation** (the Mind Robotics / foundation-model world):
   contact-rich policy learning (RL, imitation, VLA models) has proven that
   dexterous skills can be *learned* — but the literature treats the
   engineering parameters that define manufacturability (clearance, tolerance,
   friction, tool geometry, measurement noise) as **incidental constants**:
   fixed in the task, swept ad hoc if at all, never treated as the **control
   variables** they are in a factory.

Meanwhile, the frontier is a race to put learned policies into real factories:
Mind Robotics (Rivian spin-off, $1B+ raised) is running a *manufacturing data
flywheel*; Google DeepMind's Gemini Robotics models are being deployed on
industrial platforms (Agile Robots, March 2026); the EOAT market is projected to
grow from ~$2.2B (2025) to ~$5.8B (2035) and tactile sensing is the most-cited
frontier in grasping research.

**The missing piece is a theory that connects the two worlds**: what is the
relationship between *engineering parameters* (what manufacturers control and
measure) and *policy learnability* (what ML research optimizes)? Nobody has
measured this relationship systematically, and nobody has built the adaptive
controller that exploits it.

## The question

For a contact-rich manufacturing skill (insertion, assembly, finishing,
gauging-driven process control) learned by a policy of class **C** (architecture,
sensor suite, data budget **N**):

> **Q1 (boundary).** Is there a sharp phase transition in the engineering
> parameters — clearance ratio *c*, tool geometry *g*, measurement noise *σ* —
> across which the skill goes from learnable to unlearnable, at fixed budget N?
>
> **Q2 (law).** Is the boundary *predictable* — can a stylized model of contact
> ambiguity derive its functional form (e.g., critical clearance c* ~ N^{−α})?
>
> **Q3 (control).** Can an adaptive controller that tunes the operating envelope
> from in-process measurements *find and exploit* the boundary, the way a human
> process engineer tightens tolerance until yield drops and backs off?

## Hypotheses

- **H1 (monotone boundary).** Final success under fixed policy class and data
  budget is a monotone function of clearance ratio *c* (and of measurement noise
  *σ*), with a knee separating a "solved" regime from a "crash" regime — and the
  crash is *structural* (capacity/ambiguity), not a training failure.
- **H2 (capacity law).** The critical clearance scales as a power law in
  effective capacity/data: c* ~ N^{−α} for some α > 0. The exponent α is
  predicted by the stylized model and measurable.
- **H3 (tool-conditioned transfer).** A policy conditioned on a tool-geometry
  embedding transfers to a *new* tool with a data cost proportional to a
  geometry distance — i.e., EOAT catalog geometry is a structured prior, not a
  nuisance.
- **H4 (measurement floor).** Measurement noise *σ* sets a floor on achievable
  tolerance yield regardless of policy quality; the yield boundary in *σ* is
  measurable and gives metrology vendors (Renishaw) a quantitative ROI story
  ("your gauge precision buys this much learnable process window").

## What "done" looks like

A **Tolerance Law** — measured phase diagrams across three manufacturing-task
families (precision insertion, tooling-dependent grasping/placement, gauged
process control) with a fitted capacity law, plus a working **adaptive tolerance
controller** that locates the boundary without knowing it a priori. That package
(mechanism + law + controller + multi-task evidence) is the NMI submission; the
single-task law alone is an RA-L/ICRA paper.
