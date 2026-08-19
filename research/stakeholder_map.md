# Stakeholder map

Verified research, June–August 2026. Sources: company sites, The Robot Report,
LinkedIn, industry press.

## 1. Mind Robotics — the internship target

**What they are.** American robotics company, spun out of Rivian in November
2025, HQ Palo Alto. Total funding >$1B: $115M seed (late 2025), $500M Series A
(March 2026), $400M round led by Kleiner Perkins (May 2026) with Accel, a16z,
Bain Capital Ventures, Greenoaks, Redpoint, Meritech, SV Angel among investors.
CEO: RJ Scaringe (Rivian founder).

**What they build.** A "full-stack platform of foundation models, purpose-built
robotics, and deployment infrastructure to automate industrial and
manufacturing tasks at scale." Their own words: "We are not building
single-task machines... a platform that generalizes across core tasks and
scales across manufacturing domains." Rivian is the first customer and a
shareholder, providing a **live, high-volume manufacturing environment for
model training and deployment** — production data feeding their **"data
flywheel"** for rapid iteration. They are early-stage: the website has no robot
images yet; the team is growing fast across AI, robotics, and industrial
manufacturing.

**Why this project is their vocabulary.** Their three pillars map 1:1 onto
this research:
- *Foundation models* → the tolerance-difficulty axis for VLAs (P2) — nobody
  benchmarks foundation models on clearance; we would.
- *Purpose-built robotics / dexterous factory tasks* → the contact-rich phase
  diagram (R2) and tool-conditioned policies (R3a) on exactly the task class
  their factory data flywheel is generating.
- *Data flywheel* → the Tolerance Law's core question is the one *their* data
  flywheel needs answered: when is a factory task learnable from a given data
  budget, and which operating envelope (tolerance) is learnable at all.

**The internship angle (see `proposals/mind_robotics_internship.md`):** a
candidate who has *built and measured a robotics data flywheel* (their literal
infrastructure), ported it to contact-rich MuJoCo, and is now mapping the
learnability boundaries of manufacturing skills — i.e., the person who knows
both the data side and the contact side of what Mind is trying to do.

## 2. Gimatic — the tooling partner / funding target

**What they are.** Gimatic Srl (Italy), owned by Barnes Group since 2018 —
"cornerstone of Barnes' automation segment." Global leader in End-of-Arm
Tooling (EOAT): electric and pneumatic grippers, vacuum components, gripping
fingers, sensors, rotary units, tool changers. Market: EOAT ~US$2.2B (2025) →
~US$5.8B (2035), 10.1% CAGR.

**Why they'd fund.** Their catalog *is* a structured prior: thousands of
gripper/finger/vacuum geometries, each a plausible policy condition. The
tool-conditioning result (P3, R3a) would (a) produce a **tool-geometry-distance
metric** derived from their own catalog, (b) show that policies can transfer
across their EOAT with predictable data cost, and (c) position them as *the*
EOAT company with a learned-policy layer — a product story for their sensors
and instrumented grippers (tactile-sensing reviews, 100+ citations, mark this
as the frontier).

**The ask:** research collaboration + funding for the tool-conditioning suite:
access to catalog geometry data (finger shapes, compliance specs) as the
distance metric input, and a joint publication + a tool-conditioned policy zoo
they can demo. See `proposals/funding_gimatic_renishaw.md`.

## 3. Renishaw — the metrology partner / funding target

**What they are.** UK precision-engineering giant: coordinate measurement
machines, encoders, the **Equator / Equator-X automated gauging systems**
(robot-loaded, shop-floor, in-process gauging — up to 250–500 mm/s scan), the
**Renishaw Central smart-manufacturing data platform** (collects, presents, and
*actions* process + metrology data), and metal additive manufacturing. Since
2023 they have an explicit **robotic-automation product line** and are
marketing "robot efficiency solutions... enhance process control across
automated manufacturing" (Automate 2026).

**Why they'd fund.** The measurement-in-the-loop result (P4, R3b) is a direct
ROI statement for their instrumentation: *gauge precision buys learnable
process window* — a quantitative relationship between measurement noise σ and
achievable tolerance yield under learned policies. Their Equator cells are
literally the closed loop (robot loads part → gauge measures → process adjusts);
Renishaw Central is the data substrate a measurement-rewarded policy would
train on. They already sell the instrumentation; we supply the learning layer
and the law that justifies buying better gauges.

**The ask:** funding + data/access partnership: Equator measurement traces (or
simulated equivalents calibrated to their accuracy specs) as the reward
substrate, and a joint study: "measurement precision as a control variable in
learned process control." See `proposals/funding_gimatic_renishaw.md`.

## Positioning summary

| | Mind Robotics | Gimatic | Renishaw |
|---|---|---|---|
| Kind of ask | Internship | Funding + collaboration | Funding + collaboration |
| Hook | Data flywheel + contact-rich policy learning, their exact vocabulary | Catalog geometry as a policy prior | Metrology as reward; gauge precision buys process window |
| Deliverable they see | An engineer who can run their flywheel and tune its learnability | A tool-conditioned policy zoo + transfer law on their EOAT | The σ-boundary law + closed-loop gauged learning demo |
| Evidence we bring | Committed flywheel mechanism + MuJoCo contact-rich port (live repo) | The R3a suite | The R3b suite + the capacity-law machinery |
