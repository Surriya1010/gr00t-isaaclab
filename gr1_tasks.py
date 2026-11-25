# Ported from isaac-sim/IsaacLabEvalTasks (Apache-2.0, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES)
# nutpour_gr1t2_closedloop_env_cfg.py / exhaustpipe_gr1t2_closedloop_env_cfg.py.
# Changes: adapted to Isaac Lab 3.0 (xyzw camera quaternion taken from the 3.0 base config,
# preserve_order on the action term), plus an optional laptop rendering profile.
"""GR1-T2 closed-loop tasks for GR00T N1 in Isaac Lab 3.0 (joint-position control).

Registers:
    Isaac-NutPour-GR1T2-ClosedLoop-v0
    Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0
"""

import gymnasium as gym

import isaaclab.envs.mdp as mdp
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_base_env_cfg import (
    ExhaustPipeGR1T2BaseEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place.nutpour_gr1t2_base_env_cfg import NutPourGR1T2BaseEnvCfg

# Action vector layout (36 joints), identical to IsaacLabEvalTasks' action_joint_space.yaml.
# preserve_order=True makes Isaac Lab keep exactly this order.
ACTION_JOINTS = [
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint",
    "L_ring_proximal_joint", "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint", "R_middle_proximal_joint", "R_pinky_proximal_joint",
    "R_ring_proximal_joint", "R_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint", "L_middle_intermediate_joint", "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint", "L_thumb_proximal_pitch_joint",
    "R_index_intermediate_joint", "R_middle_intermediate_joint", "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint", "R_thumb_proximal_pitch_joint",
    "L_thumb_distal_joint", "R_thumb_distal_joint",
]

# GR00T N1 "gr1_arms_only" state/action groups (IsaacLabEvalTasks gr00t_joint_space.yaml).
GR00T_GROUPS = {
    "left_arm": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                 "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
                 "left_wrist_pitch_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_roll_joint",
                  "right_wrist_pitch_joint"],
    "left_hand": ["L_pinky_proximal_joint", "L_ring_proximal_joint", "L_middle_proximal_joint",
                  "L_index_proximal_joint", "L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint"],
    "right_hand": ["R_pinky_proximal_joint", "R_ring_proximal_joint", "R_middle_proximal_joint",
                   "R_index_proximal_joint", "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint"],
}

INSTRUCTIONS = {
    "Isaac-NutPour-GR1T2-ClosedLoop-v0": (
        "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on"
        " the metallic measuring scale."
    ),
    "Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0": "Pick up the blue pipe and place it into the blue bin.",
}


def _closed_loop(cfg):
    # Stiffer arm tracking used by IsaacLabEvalTasks for GR00T closed-loop evaluation.
    for name in ("left-arm", "right-arm"):
        cfg.scene.robot.actuators[name].stiffness = 3000
        cfg.scene.robot.actuators[name].damping = 100

    # Same head camera as the base task (pose already in Isaac Lab 3.0 xyzw order), as a tiled camera.
    cam = cfg.scene.robot_pov_cam
    cfg.scene.robot_pov_cam = TiledCameraCfg(
        prim_path=cam.prim_path,
        update_period=0.0,
        height=160,
        width=256,
        data_types=["rgb"],
        spawn=cam.spawn,
        offset=TiledCameraCfg.OffsetCfg(pos=cam.offset.pos, rot=cam.offset.rot, convention=cam.offset.convention),
    )
    cfg.actions.gr1_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ACTION_JOINTS, scale=1.0, use_default_offset=False, preserve_order=True
    )
    cfg.viewer.eye = (0.0, 1.8, 1.5)
    cfg.viewer.lookat = (0.0, 0.0, 1.0)

    cfg.episode_length_s = 30.0
    cfg.sim.dt = 1 / 100
    cfg.decimation = 5
    # One render per control step (base task renders 2.5x per step).
    cfg.sim.render_interval = 5

    # Laptop rendering profile: FXAA instead of DLAA keeps the 256x160 policy image at native
    # resolution (DLSS would upscale from a lower internal resolution) at a fraction of DLAA's cost.
    cfg.sim.render.antialiasing_mode = "FXAA"
    cfg.num_rerenders_on_reset = 1
    cfg.recorders = {}


@configclass
class NutPourGR1T2ClosedLoopEnvCfg(NutPourGR1T2BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _closed_loop(self)


@configclass
class ExhaustPipeGR1T2ClosedLoopEnvCfg(ExhaustPipeGR1T2BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _closed_loop(self)


for _id, _cls in (
    ("Isaac-NutPour-GR1T2-ClosedLoop-v0", "NutPourGR1T2ClosedLoopEnvCfg"),
    ("Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0", "ExhaustPipeGR1T2ClosedLoopEnvCfg"),
):
    if _id not in gym.registry:
        gym.register(
            id=_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}:{_cls}"},
            disable_env_checker=True,
        )
