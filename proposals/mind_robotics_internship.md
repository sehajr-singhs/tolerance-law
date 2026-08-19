# Mind Robotics — Internship pitch

**Candidate:** Sehaj Randhir Singh — independent researcher working at the
intersection of robot data flywheels and contact-rich policy learning.

**Why this pitch exists:** Mind Robotics is building exactly the stack this
research program targets — foundation models + purpose-built robotics + a
manufacturing data flywheel, deployed at Rivian. This document says what I've
actually built, why it matches your three pillars, and what I'd deliver as an
intern.

---

## 1. What I have, with receipts

Everything below is committed, tested, and reproducible in public repos:

1. **A measured robotics data flywheel mechanism.** Built the flywheel loop
   (expert → collect → curate → retrain) and *measured* its central control
   problem: the relabeled:clean mixing ratio in the training set. Produced the
   first phase diagram of the "flood boundary" — where unbounded relabeling
   crashes a policy — and a **closed-loop adaptive curator** that finds the
   safe ratio without knowing the capacity. The result survived a port from
   kinematic sim to **contact-rich MuJoCo** (real friction, real contact), where
   the mechanism held: the crash is a high-capacity phenomenon, exactly as the
   theory predicts. ~17 tests, all green; results committed from Kaggle GPU
   kernels through JSON to the paper PDFs.
2. **A contact-rich sim engineering track record.** Ported the entire policy
   stack to MuJoCo and diagnosed three physics gotchas that would have stalled
   anyone else for weeks (joint limits silently ignored under `autolimits`;
   velocity actuators that physically can't sustain push force — fixed with
   position actuators; pusher-vs-table friction ordering). Scripted expert
   solve on real contact: ~70% on a task the kinematic env solves 100% — an
   honestly harder, contact-rich setting.
3. **A full measurement-to-paper pipeline.** Kaggle GPU kernels → committed JSON
   → analysis scripts → LaTeX macros → NMI-format and IEEE-format papers, with
   every number machine-generated, nothing hand-typed. This is the
   reproducibility discipline your flywheel data will need.

## 2. Why Mind Robotics

Your own words: "full-stack platform of foundation models, purpose-built
robotics, and deployment infrastructure to automate industrial and
manufacturing tasks at scale... generalizes across core tasks and scales across
manufacturing domains." Mapping to what I do:

| Your pillar | What I bring |
|---|---|
| Foundation models for dexterous factory tasks | The **tolerance-difficulty axis** no one benchmarks: success vs. clearance for policy classes including VLA-lite (diffusion) proxies — the question "which factory tasks are learnable from your data budget, at your tolerances" |
| Manufacturing data flywheel | I built and measured one. Curation/mixing-ratio control, memorization diagnostics, oracle-query cost accounting — the data-side control theory your flywheel will need as it scales |
| Purpose-built robotics + deployment | Contact-rich task suites in MuJoCo that mirror the physical tasks your Rivian cells run; sim-to-real-ready env API (state, image, and contact-force modes) |

## 3. What I'd deliver as an intern

- **The learnability map of your task portfolio**: take the tolerance phase
  diagram to your real task list — for each task class, the clearance/data
  curve that says "this task is learnable from X demos" vs. "this one needs a
  different operating envelope or a different sensor."
- **Flywheel operations**: the curation controller and diagnostics running on
  your data pipeline, with the flood-boundary guardrail so relabeling never
  silently poisons the policy.
- **Contact-rich sim infrastructure**: tolerance-parameterized MuJoCo/robosuite
  suites matching your hardware, so model iterations don't burn factory time.
- **Publication-grade evidence**: the measurement discipline that turns
  experiments into papers your team can show — NMI-format ready.

## 4. The ask

A summer internship on the **model / data-flywheel / contact-rich policy** side
of the platform. I bring a working mechanism, a working contact-rich stack, and
a working measurement pipeline — and I'll spend the internship applying all
three to your Rivian-scale data flywheel.

*References: the flywheel repo (mechanism + contact-rich port + papers) and
this research repo (the Tolerance Law program) are linked from my profile.*
