# The Tolerance Law: mechanism, model, predictions

## 1. The mechanism in one paragraph

A contact-rich manufacturing skill is a policy that must resolve an
**ambiguous contact state** (where are the surfaces touching? what forces are
being transmitted?) from a partial observation (vision, proprioception, force,
tactile). As the engineering parameters tighten — clearance ratio *c* → 0,
measurement noise *σ* ↑, tool geometry *g* departing from what the policy was
trained on — the ambiguity the policy must resolve *grows*, and the data budget
*N* that buys that resolution *does not*. The mechanism is therefore a
**learnability phase transition**: for fixed (architecture, sensor suite, N),
there is a critical parameter value beyond which the skill is structurally
unlearnable — not under-trained, *under-determined*. The boundary is
**(i) monotone, (ii) capacity-dependent, (iii) predictable from a stylized
model, and (iv) controllable**: an adaptive controller that tunes the operating
envelope from in-process measurements can *find* the boundary and run the
process at its learnable edge.

## 2. The stylized model (R1 deliverable)

We derive the capacity law from a minimal, physically-grounded model of
quasi-static insertion. The ingredients:

1. **Geometry.** Peg of diameter *d*, hole of diameter *d + 2c*, so clearance
   ratio *ĉ = c/d*. Contact geometry defines an ambiguity manifold: a pose
   offset *δ* maps to a measurable force *f* through a contact Jacobian whose
   conditioning degrades as *ĉ → 0* (the classic result that small-clearance
   insertion is "blind": force signals saturate before pose error is
   resolvable).
2. **Observation.** The policy sees a noisy version of the force/tactile
   signal, noise level *σ* (measurement axis) — including actuator/encoder
   uncertainty.
3. **Information budget.** A policy trained on *N* demonstrations/transitions
   can resolve an observation-space ambiguity whose *mutual information*
   is ≤ some monotone function of N (a data-efficiency law, e.g.,
   I_resolvable(N) ~ N^{γ} for γ < 1, from standard statistical learning).
4. **Learnability condition.** The skill is learnable iff the information
   required to disambiguate the contact state at tolerance *ĉ* is ≤ the
   resolvable information at budget N.

This yields the **capacity law** as a falsifiable prediction:

> **ĉ*(N; σ, g) ~ (N / N₀)^{−α} · h(σ) · k(g)**  — the critical clearance
> ratio below which learning fails scales as a power law in data/capacity, is
> monotonically *decreasing* in measurement noise (worse gauges ⇒ smaller
> learnable window), and is modulated by tool geometry through a factor k(g)
> that is a function of a geometry-distance metric.

The *exact* exponent α and the functions h, k are the empirical deliverables;
the *existence and functional form* (power-law boundary, monotone in ĉ and σ)
is the mechanism claim.

## 3. Falsifiable predictions (pre-registered)

- **P1 — Monotone boundary.** For fixed policy class and N, final success is a
  monotone function of clearance ratio *ĉ*, with a sharp knee separating a
  "solved" regime (success ≳ 0.9) from a "crashed" regime (success ≲ 0.2). The
  knee is reproducible across seeds and does not vanish with more training
  epochs at fixed N (structural, not optimization failure).
- **P2 — Power law.** The critical clearance ĉ* (boundary location) shifts
  with capacity N as ĉ* ∝ N^{−α}, α > 0 measurable. Two independent capacity
  axes: (a) model capacity (width/depth), (b) data budget N. Both must obey the
  law with *consistent* α.
- **P3 — Tool-conditioned transfer law.** Transfer cost (demonstrations needed
  on a new tool to recover performance) scales with a tool-geometry distance
  D(g₁, g₂) that is *predictable* from catalog geometry (e.g., fingertip
  curvature/compliance for Gimatic-style fingers), and tool-conditioned
  policies reduce transfer cost by a constant factor versus re-training from
  scratch.
- **P4 — Measurement floor.** Achievable tolerance yield saturates as a
  function of measurement noise *σ*; the saturation level is a *lower* bound on
  yield that no amount of policy data can cross. (This is the Renishaw ROI
  statement: gauge precision buys learnable process window.)
- **P5 — The controller.** An adaptive tolerance controller that tightens the
  operating envelope until yield drops and backs off converges near the
  measured boundary **without knowing ĉ* a priori** (closed-loop, in-process
  signal only).

## 4. Negative controls and honest risks

- **Negative control 1 (the naive baseline):** "more data fixes everything."
  We will show the boundary does *not* vanish with unlimited training epochs at
  fixed N, and only shifts as P2 predicts with N — i.e., there is a real
  structural wall, not a training artifact.
- **Negative control 2 (the free-lunch baseline):** a fixed tolerance policy
  trained at one clearance, evaluated at others, collapses per P1 — showing
  that boundary-agnostic training is insufficient and the adaptive controller
  is not decoration.
- **Risk — the boundary may be soft, not sharp.** If success decays
  continuously with no knee, P1 fails in its strong form; the fallback is a
  weaker claim (monotone decay with fitted functional form), which still
  supports the design-rule framing but is not a "phase transition."
- **Risk — exponent instability across tasks.** If α differs wildly between
  insertion and finishing, the "law" reduces to per-task curves; the mitigation
  is reporting the law per task family and claiming *within-family*
  universality (which is still a usable engineering statement).
- **Risk — sim-only critique.** Contact-rich sim (MuJoCo/robosuite) is
  credible but not hardware; the mitigation is a hardware-validation phase
  (R4) framed as a stretch goal, with the paper's core claim clearly scoped to
  simulated contact with stated physics parameters.

## 5. Why this is a mechanism, not a benchmark

Benchmarks report *who wins at a fixed difficulty*. This project reports *how
learnability depends on the difficulty knob manufacturers actually turn* —
and builds the controller that turns it. That is the difference between "our
method beats yours on peg-in-hole" (a task paper) and "here is the law
governing when any method can learn an insertion skill, and here is the
closed-loop controller that exploits it" (a mechanism paper). NMI's tactile
taxonomy (2025) establishes that process-physics organization is NMI-grade;
a *quantitative, falsifiable, controllable law of the same domain* is the
natural next rung.
