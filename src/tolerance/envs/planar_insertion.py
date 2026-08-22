"""PlanarInsertion: a clearance-parameterized peg-in-hole task in MuJoCo.

A gantry/SCARA-style two-axis stage (prismatic X and Y) carries a rigid peg
and must insert it into a slot whose channel width is parameterized by the
clearance c. The task is the canonical contact-rich insertion: the peg meets
chamfered mouth edges, slides into the channel, and the only way to resolve
lateral misalignment at tight clearance is the contact-force field — which
becomes ambiguous as c -> 0. That ambiguity is the mechanism the Tolerance Law
is about.

Observation (what the robot actually sees):
    [peg_x, peg_y, peg_vx, peg_vy, y_meas, F_x, F_y, t]
      peg_x/y      : peg center position (world)
      peg_vx/vy    : peg velocity
      y_meas       : noisy measurement of the slot center y (simulated sensor)
      F_x/F_y      : total contact force on the peg (world, filtered)
      t            : time since reset (normalized) — lets the policy track
                     progress without explicit phase bookkeeping

Action: 2-dim velocity command in [-1, 1]^2, integrated by the stage into
position targets (position actuators hold sustained insertion force).

Success: peg front face reaches the slot back wall (fully seated) and the peg
stays laterally centered, held for SUCCESS_HOLD steps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DT = 0.02          # env step (s)
SIM_DT = 0.005     # mujoco substep (s)
SUBSTEPS = max(1, int(round(DT / SIM_DT)))
HORIZON = 1200     # max env steps per episode (room for expert retries at tight clearance)
SUCCESS_HOLD = 10
PEG_W = 0.05       # peg width (y extent)
PEG_L = 0.10       # peg length (x extent)
SLOT_X0 = 0.30     # slot mouth x position
SLOT_DEPTH = 0.12  # slot depth (x)
BACK_WALL_X = SLOT_X0 + SLOT_DEPTH
SEAT_X = BACK_WALL_X - PEG_L / 2.0   # peg center x when fully seated
MAX_VEL = 0.08     # max stage speed (m/s) per action unit
Y_MEAS_NOISE = 0.0008  # sigma of the slot-center sensor (m) — sub-clearance
FORCE_NOISE = 0.15     # sigma of the force sensor (N)
CLEARANCE_DEFAULT = 0.006  # default half-gap between peg and wall (m)


def _build(clearance: float) -> tuple:
    """Build a MuJoCo model for the given clearance (half-gap, meters)."""
    import mujoco

    # channel half-width = peg half-width + clearance
    hw = PEG_W / 2.0 + clearance
    wall_t = 0.02       # wall thickness (y)
    wall_h = 0.025      # wall half-height (z); walls span z in [0, 0.05]
    wall_cx = SLOT_X0 + SLOT_DEPTH / 2.0
    wall_cz = 0.025

    def wall(side: str) -> str:
        sgn = 1.0 if side == "top" else -1.0
        cy = sgn * (hw + wall_t / 2.0)
        return (
            f'<geom name="wall_{side}" type="box" pos="{wall_cx:.6f} {cy:.6f} {wall_cz}" '
            f'size="{SLOT_DEPTH/2:.6f} {wall_t/2:.6f} {wall_h}" '
            f'rgba="0.45 0.45 0.55 1" friction="0.12 0.005 0.0001"/>')

    # chamfer wedges at the mouth: 45-deg rotated boxes whose inner corner
    # forms a lead-in funnel (center sits at the channel corner, rotated so
    # the diagonal face faces the approaching peg)
    # NOTE: disabled in v1 — the rotated-box corner poked into the approach
    # path and blocked before the mouth; the expert's dither + retract search
    # provides the loose-clearance ease instead.
    def chamfer(side: str) -> str:
        return ""

    mjcf = f"""
<mujoco model="stage_insertion">
  <compiler angle="radian"/>
  <option timestep="{SIM_DT}" iterations="50" tolerance="1e-9" gravity="0 0 -9.81"/>
  <worldbody>
    <light name="sun" pos="0.4 0 2.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" pos="0 0 0" size="1.2 1.2 0.01"
          friction="0.5 0.005 0.0001"/>
    <camera name="top" pos="0.4 0 2.2" xyaxes="1 0 0 0 1 0" fovy="36"/>
    <body name="gantry" pos="0 0 0.055">
      <joint name="jx" type="slide" axis="1 0 0" limited="true" range="-0.10 0.45"/>
      <!-- rail sits above the slot walls (wall top z=0.05; rail bottom z=0.053)
           so the wide beam never rams the wall fronts while traveling -->
      <geom name="gantry_rail" type="box" pos="0 0 0.010" size="0.015 0.13 0.012" mass="0.5"
            friction="0.4"/>
      <body name="carriage" pos="0 0 -0.02">
        <joint name="jy" type="slide" axis="0 1 0" limited="true" range="-0.18 0.18"/>
        <geom name="carriage_block" type="box" size="0.012 0.014 0.010" mass="0.3"
              friction="0.4"/>
        <body name="peg" pos="0.05 0 -0.023">
          <geom name="peg_geom" type="box" size="{PEG_L/2:.6f} {PEG_W/2:.6f} 0.010"
                mass="0.06" rgba="0.9 0.55 0.2 1" friction="0.25 0.005 0.0001"/>
        </body>
      </body>
    </body>
    <body name="slot" pos="0 0 0">
      <!-- slot is rigid in sim (frictionloss locks it at y_channel); qpos is
           still settable at reset, so the task can place the channel anywhere -->
      <joint name="slot_y" type="slide" axis="0 1 0" limited="true" range="-0.12 0.12"
             frictionloss="500"/>
      {wall("top")}
      {wall("bottom")}
      <geom name="wall_back" type="box" pos="{BACK_WALL_X + wall_t/2:.6f} 0 {wall_cz}"
            size="{wall_t/2:.6f} {hw + wall_t:.6f} {wall_h}"
            rgba="0.35 0.35 0.45 1" friction="0.12 0.005 0.0001"/>
    </body>
    <body name="marker" pos="0 0 0.004">
      <joint name="mx" type="slide" axis="1 0 0" limited="false"/>
      <joint name="my" type="slide" axis="0 1 0" limited="false"/>
      <geom name="marker_geom" type="cylinder" size="0.03 0.002"
            rgba="0.13 0.72 0.34 0.4" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act_x" joint="jx" kp="150" kv="10"
              ctrlrange="-0.10 0.45" forcerange="-80 80"/>
    <!-- stiffer y: the lateral sweep must generate enough force to slide a
         jammed peg along the wall face (ratchet into the channel) -->
    <position name="act_y" joint="jy" kp="400" kv="25"
              ctrlrange="-0.18 0.18" forcerange="-80 80"/>
  </actuator>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    return mujoco, model, data


class PlanarInsertion:
    """Clearance-parameterized planar peg-in-hole on a two-axis stage."""

    def __init__(
        self,
        clearance: float = CLEARANCE_DEFAULT,
        seed: int = 0,
        horizon: int = HORIZON,
        dt: float = DT,
        y_noise: float = Y_MEAS_NOISE,
        force_noise: float = FORCE_NOISE,
        rng: Optional[np.random.Generator] = None,
    ):
        assert clearance > 0, "clearance must be positive"
        self.clearance = float(clearance)
        self.horizon = horizon
        self.dt = dt
        self.y_noise = y_noise
        self.force_noise = force_noise
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.mujoco, self.model, self.data = _build(self.clearance)
        self._peg_gid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_GEOM, "peg_geom")
        self._peg_bid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, "peg")
        self._qadr = {j: self.model.jnt_qposadr[self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, j)] for j in
            ("jx", "jy", "mx", "my", "slot_y")}
        self._vadr = {j: self.model.jnt_dofadr[self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ("jx", "jy")}
        self.t = 0
        self.y_channel = 0.0     # true slot center (world)
        self.y_meas = 0.0        # noisy measurement the robot acts on
        self._x_cmd = 0.0
        self._y_cmd = 0.0
        self._in_goal = 0
        self.obs: Optional[np.ndarray] = None
        self._traj_obs: list[np.ndarray] = []
        self._traj_act: list[np.ndarray] = []
        self._force_filt = np.zeros(2)

    # ------------------------------------------------------------------ #
    # internals                                                           #
    # ------------------------------------------------------------------ #
    def _read_peg(self) -> tuple[float, float, float, float]:
        d = self.data
        px = d.qpos[self._qadr["jx"]] + 0.05
        py = d.qpos[self._qadr["jy"]]
        vx = d.qvel[self._vadr["jx"]]
        vy = d.qvel[self._vadr["jy"]]
        return px, py, vx, vy

    def _contact_force(self) -> np.ndarray:
        """Sum of contact forces on the peg geom (world xy)."""
        mj, model, data = self.mujoco, self.model, self.data
        total = np.zeros(2)
        if data.ncon == 0:
            return total
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 != self._peg_gid and c.geom2 != self._peg_gid:
                continue
            # contact frame: c.frame columns are (normal, tangent1, tangent2)
            n = np.array(c.frame[0:3])
            t1 = np.array(c.frame[3:6])
            t2 = np.array(c.frame[6:9])
            force = np.zeros(6)
            mj.mj_contactForce(model, data, i, force)
            f_world = n * force[0] + t1 * force[1] + t2 * force[2]
            total += f_world[:2]
        return total

    def _make_obs(self) -> np.ndarray:
        px, py, vx, vy = self._read_peg()
        f = self._force_filt
        return np.array([px, py, vx, vy, self.y_meas, f[0], f[1],
                         self.t / self.horizon], dtype=np.float64)

    def _set_stage(self, x: float, y: float) -> None:
        d = self.data
        d.qpos[self._qadr["jx"]] = float(x)
        d.qpos[self._qadr["jy"]] = float(y)
        d.qvel[self._vadr["jx"]] = 0.0
        d.qvel[self._vadr["jy"]] = 0.0
        self._x_cmd = float(x)
        self._y_cmd = float(y)
        # move the slot (walls + back wall) to the task's channel position
        d.qpos[self._qadr["slot_y"]] = float(self.y_channel)
        # marker for visualization
        d.qpos[self._qadr["mx"]] = SLOT_X0
        d.qpos[self._qadr["my"]] = float(self.y_channel)
        self.mujoco.mj_forward(self.model, d)

    # ------------------------------------------------------------------ #
    # API                                                                 #
    # ------------------------------------------------------------------ #
    def sample_start(self) -> None:
        """Sample a task: slot y, peg start (laterally offset, retracted)."""
        self.y_channel = self.rng.uniform(-0.10, 0.10)
        self.y_meas = self.y_channel + self.rng.normal(0.0, self.y_noise)
        start_x = SLOT_X0 - PEG_L - 0.06          # peg front 6 cm before mouth
        start_y = self.y_channel + self.rng.uniform(-0.025, 0.025)
        self._set_stage(start_x, start_y)

    def reset(self, task: Optional[tuple[float, float]] = None) -> np.ndarray:
        """Reset. task = (y_channel, y_meas) to replay a specific instance."""
        if task is not None:
            self.y_channel, self.y_meas = float(task[0]), float(task[1])
            start_x = SLOT_X0 - PEG_L - 0.06
            start_y = self.y_channel + self.rng.uniform(-0.025, 0.025)
            self._set_stage(start_x, start_y)
        else:
            self.sample_start()
        self.t = 0
        self._in_goal = 0
        self._force_filt = np.zeros(2)
        self.obs = self._make_obs()
        self._traj_obs = [self.obs.copy()]
        self._traj_act = []
        return self.obs

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        vx, vy = a * MAX_VEL
        self._x_cmd = np.clip(self._x_cmd + vx * self.dt, -0.10, 0.45)
        self._y_cmd = np.clip(self._y_cmd + vy * self.dt, -0.18, 0.18)
        d = self.data
        mj = self.mujoco
        d.ctrl[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_x")] = self._x_cmd
        d.ctrl[mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "act_y")] = self._y_cmd
        for _ in range(SUBSTEPS):
            mj.mj_step(self.model, d)
        # low-pass the contact force, then add sensor noise
        raw = self._contact_force()
        self._force_filt = 0.6 * self._force_filt + 0.4 * raw
        self.t += 1
        self.obs = self._make_obs()
        self.obs[5] += self.rng.normal(0.0, self.force_noise)
        self.obs[6] += self.rng.normal(0.0, self.force_noise)

        px, py, _, _ = self._read_peg()
        seated = px >= SEAT_X - 0.004 and abs(py - self.y_channel) < 0.02
        self._in_goal = self._in_goal + 1 if seated else 0
        self._traj_obs.append(self.obs.copy())
        self._traj_act.append(a.copy())
        return self.obs

    @property
    def success(self) -> bool:
        return self._in_goal >= SUCCESS_HOLD

    @property
    def done(self) -> bool:
        return self.success or self.t >= self.horizon

    @property
    def peg_x(self) -> float:
        return float(self._read_peg()[0])

    @property
    def peg_y(self) -> float:
        return float(self._read_peg()[1])

    @property
    def final_dist_to_seat(self) -> float:
        px, py, _, _ = self._read_peg()
        return float(abs(px - SEAT_X))

    def trajectory(self) -> dict:
        obs = np.stack(self._traj_obs)
        act = np.stack(self._traj_act) if self._traj_act else np.zeros((0, 2))
        if len(act) and len(obs) == len(act) + 1:
            obs = obs[:-1]
        return {"obs": obs, "actions": act, "success": bool(self.success),
                "steps": self.t, "y_channel": float(self.y_channel),
                "clearance": float(self.clearance)}
