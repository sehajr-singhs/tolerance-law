# Literature review

Organized by research cluster, with the citations that anchor the gap. All
sources verified via web search, June–August 2026.

## 1. Contact-rich manipulation learning (the method frontier)

- **Elguea-Aguinaco et al. (2023), "A review on reinforcement learning for
  contact-rich robotic manipulation tasks", *Robotics and Computer-Integrated
  Manufacturing*, 236+ citations.** The standard survey; documents that RL for
  contact-rich tasks is viable but sample-hungry and task-specific. Notably:
  no systematic treatment of *tolerance/clearance* as a variable — clearance is
  a fixed property of each surveyed task.
- **Imitation learning for contact-rich tasks, survey (arXiv 2506.13498,
  2025).** Maps IL approaches (behavior cloning, DAgger-style, diffusion) for
  contact-rich manipulation; confirms the field's emphasis on *methods* over
  *task-parameter dependence*.
- **Johannsmeier et al. (2025), "A process-centric manipulation taxonomy for
  the organization, classification and synthesis of tactile robot skills",
  *Nature Machine Intelligence* 7(6).** The existence proof that a
  *conceptual/organizational* contribution about manufacturing-relevant
  manipulation is NMI-grade. Their taxonomy organizes *skills by process
  physics*; it does not quantify *learnability as a function of parameters* —
  that quantitative law is the open space we target.
- **Jin et al. (2023), "Vision-force-fused curriculum learning for robotic
  contact-rich assembly", 38+ citations.** Curriculum over task difficulty for
  contact-rich assembly; adjacent to our boundary idea but treats difficulty as
  a curriculum schedule, not as a *measured control law*.

## 2. Assembly / insertion (the canonical manufacturing skill)

- **"Advances in Robotic Peg-in-Hole Assembly: A Comprehensive Review"
  (*Chinese J. Mech. Eng.*, 2025).** Single-peg tasks are considered solved at
  fixed geometry; the open problems are multi-peg, high-precision (µm), and
  *generalization across hole poses and clearances*.
- **Ensemble RL framework for high-precision peg-in-hole from human
  demonstrations (*Robotics and Computer-Integrated Manufacturing*, 2026).**
  Representative of the current best practice: RL + demos, fixed clearance,
  single geometry.
- **"A General Peg-in-Hole Assembly Policy Based on Domain Randomization"
  (arXiv 2504.04148, 2025).** Domain-randomized policy generalizing across 6-DOF
  hole poses — the closest existing work to "parameter-swept" assembly, but
  randomization is used as a *robustness trick*, not as a *measurement of the
  learnability boundary* (the DR distribution is fixed; we sweep and map it).
- **Gap confirmed:** no work in this cluster reports the *success-vs-clearance
  phase diagram* or fits a law to it.

## 3. Tactile sensing, grasping, and end-of-arm tooling (the Gimatic axis)

- **Li et al. (2024), "A comprehensive review of robot intelligent grasping
  based on tactile perception", *RCIM*, 108+ citations.** Tactile sensors are
  the fastest-moving grasping modality; finger/EOAT geometry is treated as a
  fixed design choice in nearly all surveyed methods.
- **Ahmed et al. (2026), "A review of adaptive intelligence in tactile sensing
  robotic systems" (Springer), synthesizing 115 studies.** Confirms embodiment
  (gripper/finger shape) as a recognized but *under-modeled* factor in adaptive
  tactile control.
- **PP-Tac (RSS 2025): paper picking with tactile feedback.** Data-driven
  grasp synthesis with a purpose-built hand; again, hand geometry is fixed and
  the *transfer across tool geometries* is not quantified.
- **EOAT market data:** robotic end-of-arm tooling ~US$2.2B (2025) →
  ~US$5.8B (2035, 10.1% CAGR); soft gripper control crossing a "near-human
  accuracy" threshold per 2026 industry analysis.
- **Gap confirmed:** no one has measured *how policy transfer cost scales with
  tool-geometry distance* — the exact question a tooling manufacturer would fund.

## 4. Foundation models and manufacturing deployment (the Mind Robotics axis)

- **Gemini Robotics (arXiv 2503.20020, 2025)** and **Gemini Robotics 2 /
  Robotics-ER 2 (July 2026).** VLA + embodied-reasoning models now target
  industrial environments explicitly (DeepMind's blog: "complex manipulation
  tasks in real-world industrial environments").
- **Agile Robots × DeepMind (March 2026):** first announced deployment of Gemini
  Robotics foundation models on industrial platforms.
- **Mind Robotics (Rivian spin-off, Nov 2025; $115M seed, $500M Series A, $400M
  round led by Kleiner Perkins, May 2026; >$1B total).** Explicit strategy: a
  "full-stack platform of foundation models, purpose-built robotics, and
  deployment infrastructure," with Rivian's factory as a live high-volume
  training environment feeding a **manufacturing data flywheel**. Their stated
  goal is generalization *across core tasks and manufacturing domains* — not
  single-task machines.
- **Physical Intelligence (π0) and RT-2 lineage:** VLA policies trained on
  heterogeneous robot data; the data-flywheel paradigm is now the industry
  default.
- **Gap confirmed:** foundation models are evaluated on *task breadth*, never
  on *tolerance difficulty*; no benchmark reports success vs. clearance for a
  VLA. The Tolerance Law would be the first *engineering-parameter axis* for
  comparing foundation models.

## 5. Finishing processes (deburring / polishing / grinding)

- **Düzgün (2026), "Learning from Demonstration for Robotic Deburring and
  Polishing: a systematic mapping" (MDPI JMMS).** LfD in finishing is an
  emerging but fragmented field; compliance/force control is the recognized key
  (Chinese J. Mech. Eng. 2021 review, 31+ cites).
- **Q-learning trajectory generation for grinding/polishing (2020).** Early RL
  attempts; low-tier venue. **The finishing cluster is publication-sparse at
  top venues** — a high-rigor treatment here would be novel regardless of
  method.

## 6. Manufacturing measurement and process data (the Renishaw axis)

- **Renishaw Central**: a smart-manufacturing data platform collecting and
  *actioning* process + metrology data — a closed-loop process-control substrate
  with no published learned-policy layer on top.
- **Renishaw Equator / Equator-X gauging**: robot-loaded, automated gauging
  cells (delta-robot gauges, up to 250–500 mm/s scan rates); Equator is
  explicitly designed for robot loading and shop-floor process control.
- **Renishaw robot-automation product line (Automate 2023) and "robot
  efficiency solutions" (Automate 2026):** improving robot accuracy, reducing
  downtime, enhancing process control — i.e., Renishaw is already selling the
  *instrumentation* a measurement-in-the-loop learning system would consume.
- **Gap confirmed:** "measurement as reward for policy learning" (metrology
  yield as the RL objective) appears nowhere in the surveyed robotics
  literature; the standard is hand-crafted rewards with offline inspection.

## Synthesis: the open space

| Axis | Existing work | What is missing (this project) |
|---|---|---|
| Methods | RL/IL/VLA for contact-rich tasks | *Why* some tasks are learnable — parameter dependence |
| Assembly | Fixed-clearance insertion, DR robustness | The measured success-vs-clearance phase diagram + law |
| Tooling | Fixed-geometry grasping, tactile | Transfer cost vs. tool-geometry distance |
| Measurement | Offline gauging, process-data platforms | Metrology-as-reward, noise floor on yield |
| Foundation models | Task-breadth benchmarks | Tolerance-difficulty axis for comparing VLAs |
