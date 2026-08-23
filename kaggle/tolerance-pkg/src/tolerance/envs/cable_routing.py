"""Cable routing task — multi-contact, sequential constraint satisfaction.

A robot must push a flexible cable (modeled as a kinematic chain of linked
segments) through a series of guide clips. This task tests:
1. Multi-contact reasoning (cable touches clips, table, gripper)
2. Sequential constraints (must pass through clips in order)
3. Deformable object manipulation (cable shape changes under contact)

Tolerance parameter: clip spacing — tighter spacing = harder routing.
This is fundamentally harder than insertion because the cable has internal
degrees of freedom and contacts are distributed along its length.

Observation: 18D [gripper_pos(3), gripper_vel(3), cable_tip(3),
            clip_target(3), cable_shape(6)] — captures both the task
            geometry and the cable's current configuration.
Action: 5D [gx, gy, gz, finger_left, finger_right]
"""
import numpy as np
import mujoco
from typing import Tuple, Dict


# 3-segment cable chain: base → seg1 → seg2 → tip
# Each segment is a hinged body connected to the previous
_XML = """\
<mujoco model="cable_routing">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <default>
    <geom condim="4" friction="0.4 0.005 0.0001"/>
    <joint armature="0.001" damping="0.05"/>
  </default>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" pos="0 0 0" size="0.4 0.4 0.01"
          rgba="0.7 0.7 0.7 1"/>

    <!-- Gripper (3 DOF position + 2 DOF fingers) -->
    <body name="gripper" pos="0.0 0.0 0.08">
      <joint name="gx" type="slide" axis="1 0 0" range="-0.15 0.15"/>
      <joint name="gy" type="slide" axis="0 1 0" range="-0.15 0.15"/>
      <joint name="gz" type="slide" axis="0 0 1" range="0 0.12"/>
      <geom name="grip_body" type="box" size="0.015 0.015 0.008" mass="0.05"/>
      <body name="finger_l" pos="0 0.008 -0.02">
        <joint name="fl" type="slide" axis="0 1 0" range="0 0.015"/>
        <geom name="fl_geom" type="box" size="0.004 0.004 0.025" mass="0.01"/>
      </body>
      <body name="finger_r" pos="0 -0.008 -0.02">
        <joint name="fr" type="slide" axis="0 -1 0" range="0 0.015"/>
        <geom name="fr_geom" type="box" size="0.004 0.004 0.025" mass="0.01"/>
      </body>
    </body>

    <!-- Cable: chain of 4 segments linked by hinge joints -->
    <body name="cable_base" pos="0.05 0.0 0.015">
      <geom name="cb" type="cylinder" size="0.004 0.015" mass="0.005"/>
      <body name="cable_seg1" pos="0.03 0.0 0.0">
        <joint name="c1_rz" type="hinge" axis="0 0 1" range="-1.2 1.2"/>
        <joint name="c1_ry" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom name="cs1" type="cylinder" size="0.004 0.015" mass="0.005"/>
        <body name="cable_seg2" pos="0.03 0.0 0.0">
          <joint name="c2_rz" type="hinge" axis="0 0 1" range="-1.2 1.2"/>
          <joint name="c2_ry" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
          <geom name="cs2" type="cylinder" size="0.004 0.015" mass="0.005"/>
          <body name="cable_tip" pos="0.03 0.0 0.0">
            <joint name="c3_rz" type="hinge" axis="0 0 1" range="-1.2 1.2"/>
            <joint name="c3_ry" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
            <geom name="cst" type="sphere" size="0.005" mass="0.005"
                  rgba="0.2 0.6 1 1"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Clip obstacles (guide posts) -->
    <body name="clip0" pos="0.14 0.0 0.02">
      <geom name="cl0" type="cylinder" size="0.003 0.025"
            rgba="1.0 0.2 0.2 1" mass="10"/>
    </body>
    <body name="clip1" pos="0.21 0.04 0.02">
      <geom name="cl1" type="cylinder" size="0.003 0.025"
            rgba="0.2 1.0 0.2 1" mass="10"/>
    </body>
    <body name="clip2" pos="0.28 -0.03 0.02">
      <geom name="cl2" type="cylinder" size="0.003 0.025"
            rgba="0.2 0.2 1.0 1" mass="10"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a_gx" joint="gx" kp="80" ctrlrange="-0.15 0.15"/>
    <position name="a_gy" joint="gy" kp="80" ctrlrange="-0.15 0.15"/>
    <position name="a_gz" joint="gz" kp="80" ctrlrange="0 0.12"/>
    <position name="a_fl" joint="fl" kp="40" ctrlrange="0 0.015"/>
    <position name="a_fr" joint="fr" kp="40" ctrlrange="0 0.015"/>
  </actuator>
</mujoco>
"""

# Clip positions (the tolerance parameter adjusts these via clip_spacing)
CLIP_POSITIONS = np.array([
    [0.14, 0.0, 0.02],
    [0.21, 0.04, 0.02],
    [0.28, -0.03, 0.02],
])

HORIZON = 800  # steps
N_CLIPS = 3


class CableRouting:
    """Cable routing through guide clips.

    Observation (18D):
      [gripper_pos(3), gripper_vel(3), cable_tip_pos(3),
       target_clip(3), cable_shape(6)]
      where cable_shape = [midpoint1(3), midpoint2(3)] — captures curvature.

    Action (5D): [gx, gy, gz, finger_left, finger_right]
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.data = mujoco.MjData(self.model)

        # Body IDs
        self._gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cable_tip")
        self._seg1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cable_seg1")
        self._seg2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cable_seg2")
        self._clip_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"clip{i}")
            for i in range(N_CLIPS)
        ]

        self._step_count = 0
        self._current_clip = 0

    @property
    def observation_size(self) -> int:
        return 18

    @property
    def action_size(self) -> int:
        return 5

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._current_clip = 0

        # Randomize initial cable shape slightly
        for jnt_name in ["c1_rz", "c1_ry", "c2_rz", "c2_ry", "c3_rz", "c3_ry"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            qpos_adr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_adr] = self.rng.uniform(-0.2, 0.2)

        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        a = np.clip(action, -1.0, 1.0)
        self.data.ctrl[0] = a[0] * 0.15  # gx
        self.data.ctrl[1] = a[1] * 0.15  # gy
        self.data.ctrl[2] = max(a[2], 0.0) * 0.12  # gz (up only)
        self.data.ctrl[3] = max(a[3], 0.0) * 0.015  # finger left
        self.data.ctrl[4] = max(a[4], 0.0) * 0.015  # finger right

        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._obs()

        # Reward: progress toward current clip
        tip_pos = self.data.xpos[self._tip_id]
        target = CLIP_POSITIONS[min(self._current_clip, N_CLIPS - 1)]
        dist = np.linalg.norm(tip_pos[:2] - target[:2])

        reward = -dist * 5.0

        # Check if tip reached current clip
        if dist < 0.025:
            self._current_clip += 1
            reward += 20.0

        success = self._current_clip >= N_CLIPS
        done = success or self._step_count >= HORIZON

        return obs, reward, done, {
            "success": success,
            "clip_progress": self._current_clip / N_CLIPS,
            "distance_to_target": dist,
            "steps": self._step_count,
        }

    def _obs(self) -> np.ndarray:
        d = self.data
        gpos = d.xpos[self._gripper_id].copy()
        gvel = d.cvel[self._gripper_id, :3].copy()
        tip = d.xpos[self._tip_id].copy()
        target = CLIP_POSITIONS[min(self._current_clip, N_CLIPS - 1)].copy()
        mid1 = d.xpos[self._seg1_id].copy()
        mid2 = d.xpos[self._seg2_id].copy()

        return np.concatenate([gpos, gvel, tip, target, mid1, mid2]).astype(np.float64)
