"""Tolerance Law: whether a robot can learn a manufacturing skill is gated by
the engineering tolerance, and the learnability boundary moves predictably
with data budget and model capacity.

Modules:
    envs.planar_insertion  -- clearance-parameterized peg-in-hole (MuJoCo)
    policies.expert        -- force-blind sweeping insertion teacher
    policies.mlp, cnn      -- behavior-cloning policies (state / vision)
    train.bc               -- dataset assembly, training, evaluation
    experiments.sweep      -- phase-diagram grid + adaptive controllers
"""

__version__ = "0.1.0"
