# Funding asks — Gimatic and Renishaw

Two research partnerships, each anchored to a measured, publishable result with
a direct product story for the partner. Both build on the same machinery: the
Tolerance Law research program (see `research/mechanism.md`).

---

## A. Gimatic (Barnes Group) — tool-conditioned policies for end-of-arm tooling

### The problem we solve for you
Your catalog is thousands of gripper, finger, vacuum, and sensor geometries —
each a *hardware prior* that a learned policy must either exploit or fight. The
robotics literature treats EOAT geometry as a fixed nuisance; nobody has
measured **how policy transfer cost scales with tool-geometry distance**. That
missing law is the difference between "our grippers need re-programming per
tool" and "our grippers are the prior that makes policies transfer."

### The research (R3a in `research/experiment_plan.md`)
- Same contact-rich task, EOAT geometry swept along catalog-plausible
  dimensions (jaw width, fingertip compliance, vacuum vs. parallel-jaw,
  2- vs 3-finger).
- Train **tool-conditioned policies** (geometry embedding as input) and measure
  the **transfer law**: demonstrations needed on a new tool vs. a
  geometry-distance metric D(g₁, g₂) derived from *your* catalog specs.
- Publishable prediction (P3): transfer cost scales predictably with D, and
  tool-conditioning cuts it by a constant factor vs. re-training.

### What we ask
1. **Catalog geometry data**: finger/gripper dimension and compliance specs to
   define the distance metric on real products (not made-up geometry).
2. **Funding** for the GPU runs and a stipend (~$15–25K) to execute R3a and
   write it up.
3. **A joint publication** (ICRA/RSS-grade; NMI-format if the law holds across
   task families) + a **tool-conditioned policy zoo** you can demo with your
   hardware — the "smart EOAT" story your sensors and instrumented grippers
   have been heading toward.

### What you get back
- The first quantitative **EOAT-transfer law** in the literature, built on
  Gimatic products.
- A policy zoo conditioned on Gimatic geometry — a demo that turns your
  gripper catalog into a machine-learning asset.
- Reproducible, committed code (Kaggle-GPU pipeline, MuJoCo) you can run in
  your own lab.

---

## B. Renishaw — measurement-in-the-loop: metrology as the reward

### The problem we solve for you
Renishaw Central already collects and *actions* process + metrology data; the
Equator/Equator-X cells already run robot-loaded, in-process gauging. What's
missing is the **learning layer**: today's learned policies are trained on
hand-crafted rewards with offline inspection. We propose the first systematic
treatment of **measurement as the reward** — the policy optimizes measured
tolerance yield (Cp/Cpk-style), and the metrology noise σ is treated as a
**control variable** with a measurable floor.

### The research (R3b in `research/experiment_plan.md`)
- A gauging loop in simulation: post-process measurements with realistic noise
  σ (calibrated to your Equator accuracy specs) define reward and yield.
- Learn without reward shaping; measure the **σ-boundary** (P4): achievable
  yield saturates as σ worsens, *no matter the policy budget*.
- The headline result is an ROI law for your own sales story: **gauge
  precision buys learnable process window** — a quantitative curve a process
  engineer can read.

### What we ask
1. **Accuracy/uncertainty specs** (or measurement-trace samples) for the
   Equator family, to calibrate the simulated gauge noise — and optionally
   access to Renishaw Central's data model as the training substrate.
2. **Funding** for GPU runs + stipend (~$15–25K) to execute R3b and write up.
3. **A joint study** published at a top robotics venue + a
   measurement-in-the-loop demo on the closed loop you already sell.

### What you get back
- The first **measurement-as-reward** result in the manipulation literature,
  calibrated to Renishaw instruments.
- A quantitative link between your gauging hardware and achievable learned
  process control — an engineering argument that upgrades your "measure more,
  control better" pitch with data.
- Reproducible code committed under a public research license (or dual
  license, as you prefer).

---

## Common terms

- **Both asks are phase-aligned**: R3a and R3b run in parallel on the same
  infrastructure; funding either or both.
- **Ownership**: the research and the code are public (CC-BY + permissive
  code); the partner gets joint authorship on publications and a right of
  first refusal on the demo/zoo as a product.
- **Timeline**: 8–12 weeks per suite from funding; publications ~1 quarter
  later; the NMI submission if the multi-task law lands (R4).
- **Track record**: the measurement discipline (public repos, papers, and
  committed result JSONs behind every figure) is the risk-reducer; this
  program extends it into manufacturing process control.
