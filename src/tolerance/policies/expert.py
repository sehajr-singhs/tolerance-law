"""Scripted insertion expert: align to the (noisy) measured slot center, then
insert along a deterministic *time-based* lateral sweep with shift-search
retries.

Design for learnability (this is the Tolerance Law's teacher): every element
of the expert's policy is a function of *observables*.

  - The lateral dither is a sine of the episode time:  lateral = A*sin(2*pi*f*t)
    with f = 1.9 Hz.  Time is in the observation (obs[7] = t/horizon), so a
    behavior-cloned policy can reproduce the sweep; its fitted amplitude is
    what grows with the data budget and model capacity -- the mechanism
    behind the law's power-law boundary.  Because the target keeps
    oscillating even while the peg is jammed against the mouth edge, the
    peg ratchets laterally into the channel (the classic vibratory-search
    trick), which keeps the teacher honest at tight clearance.

  - Retry search is a deterministic shift of the sweep bias: try shift = 0,
    then -A, then +A, then back.  The shift is visible in the trajectory
    (the peg tracks y_meas + shift at the mouth), so the search is learnable.

The expert is deliberately force-blind: it never reads the contact forces.
This makes it the natural *teacher* whose data quality is gated by
measurement noise, and it gives learned policies something to exceed
(force-aware closed-loop alignment).

Behavior: approach to just before the mouth at y = y_meas; insert with a
time-based lateral sweep; if forward progress stalls for STALL_WINDOW steps
(peg jammed against the mouth edge), retract and re-approach with the next
shift, up to MAX_TRIES attempts.
"""

from __future__ import annotations

import numpy as np

from ..envs.planar_insertion import SLOT_X0, PEG_L, SEAT_X, MAX_VEL

APPROACH_X = SLOT_X0 - PEG_L - 0.04   # peg front 4 cm before the mouth
DITHER_FREQ = 30.0                    # rad/s (4.8 Hz lateral sweep — faster ratchet)
SHIFT_CYCLE = [0.0, -1.0, 1.0, 0.5, -0.5]  # retry biases, in units of amplitude


class DitherExpert:
    """Position-based sweeping insertion expert (force-blind)."""

    def __init__(
        self,
        insert_speed: float = 0.4,
        approach_speed: float = 0.5,
        dither_amp: float = 0.012,
        dither_freq: float = DITHER_FREQ,
        stall_window: int = 80,
        stall_eps: float = 0.0001,
        max_tries: int = 16,
        rng: np.random.Generator | None = None,
    ):
        self.insert_speed = insert_speed
        self.approach_speed = approach_speed
        self.dither_amp = dither_amp
        self.dither_freq = dither_freq
        self.stall_window = stall_window
        self.stall_eps = stall_eps
        self.max_tries = max_tries
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._amp = 0.0
        self._shift = 0.0
        self._try = 0
        self._mode = "approach"   # approach | insert | retract
        self._stall_t = 0
        self._last_x = 0.0
        self._approach_px = 0.0   # peg center x at the start of this push

    def reset(self, y_meas: float, y_noise: float) -> None:
        # sweep amplitude must be able to catch the channel given the noise
        self._amp = max(self.dither_amp, 2.0 * y_noise + 0.001)
        self._shift = 0.0
        self._try = 0
        self._mode = "approach"
        self._stall_t = 0
        self._last_x = -1.0
        self._t0 = 0.0

    def _bias(self) -> float:
        """Shift for the current attempt, in units of the sweep amplitude."""
        return SHIFT_CYCLE[self._try % len(SHIFT_CYCLE)] * self._amp

    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: [peg_x, peg_y, peg_vx, peg_vy, y_meas, Fx, Fy, t]"""
        px, py, _, _, y_meas, _, _, t_norm = obs
        dt = 0.02
        approach_px = APPROACH_X + PEG_L / 2.0

        if self._mode == "approach":
            target_x = approach_px
            ax = np.clip((target_x - px) * 8.0 / MAX_VEL, -self.approach_speed,
                         self.approach_speed)
            ay = np.clip((y_meas + self._bias() - py) * 8.0 / MAX_VEL,
                         -1.0, 1.0)
            if abs(target_x - px) < 0.004 and abs(y_meas + self._bias() - py) < 0.004:
                self._mode = "insert"
                self._stall_t = 0
                self._last_x = px
                self._t0 = t_norm * 900.0   # insert start time (steps)
            return np.array([ax, ay])

        if self._mode == "retract":
            target_x = approach_px
            ax = np.clip((target_x - px) * 6.0 / MAX_VEL, -self.approach_speed,
                         self.approach_speed)
            ay = np.clip((y_meas + self._bias() - py) * 6.0 / MAX_VEL,
                         -1.0, 1.0)
            if abs(target_x - px) < 0.005 and abs(y_meas + self._bias() - py) < 0.005:
                self._mode = "approach"
                self._try += 1
            return np.array([ax, ay])

        # insert mode: deterministic time-based lateral sweep + attempt bias.
        # The phase advances with sim time and resets at each insert start;
        # t and px (both in the observation) locate it, so the sweep stays
        # learnable while the target keeps oscillating during jams (the
        # ratchet that slides the peg laterally into the channel).
        self._t0 += dt
        phase = self.dither_freq * self._t0
        lateral = self._amp * np.sin(phase) + self._bias()
        target_y = y_meas + lateral
        err_y = target_y - py
        ay = np.clip(err_y * 8.0 / MAX_VEL, -1.0, 1.0)
        ay = np.clip(ay * 0.7 + 0.3 * np.clip(lateral / max(self._amp, 1e-9),
                                              -1.0, 1.0), -1.0, 1.0)

        if px < SEAT_X - 0.004:
            ax = self.insert_speed
        elif px < SEAT_X + 0.001:
            ax = 0.2
        else:
            ax = 0.0

        # stall detection: no forward progress -> retract and retry
        if self._try < self.max_tries and px < SEAT_X - 0.008:
            if px - self._last_x < self.stall_eps:
                self._stall_t += 1
            else:
                self._stall_t = 0
            if self._stall_t >= self.stall_window:
                self._mode = "retract"
                self._stall_t = 0
        self._last_x = px
        return np.array([ax, ay])
