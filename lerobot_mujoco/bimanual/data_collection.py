from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from quest3.quest3_teleop import Quest3Teleop
from mink_ik.bimanual_mink_ik import (
    pick_two_ee_sites,
    site_pose,
    check_reached,
    initialize_model,
    build_ctrl_map_for_joints,
    apply_configuration,
)

from mink_ik.quest3_utils import (
    Controller,
    T_from_pos_quat_xyzw,
    set_mocap_from_T,
    T_from_mocap,
    CTRL2EE_LEFT, CTRL2EE_RIGHT
)

_HERE = Path(__file__).parent

SOLVER = "daqp"

# IK
POSTURE_COST = 1e-3
MAX_ITERS_PER_CYCLE = 20
DAMPING = 5e-4

# Convergence thresholds
POS_THRESHOLD = 1e-3
ORI_THRESHOLD = 1e-2

# Viewer loop rate
RATE_HZ = 100.0

# Data collection rate 
REC_HZ = 20.0
REC_DT = 1.0 / REC_HZ

# Lerobot Config
REPO_ID = "dual_arm_teleop"  
ROOT = str((_HERE / "demo_data").resolve()) 
FPS = int(REC_HZ)  # 20

def make_or_load_dataset(model: mujoco.MjModel):
    root_path = Path(ROOT)
    create_new = not (root_path / "meta").exists()

    if create_new:
        features={
                "observation.image": {
                    "dtype": "image",                 
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channels"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["state"],
                },
                "observation.target": {
                    "dtype": "float32",
                    "shape": (14,),
                    "names": ["target"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (model.nq,),
                    "names": ["qpos"],
                },
            }
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            root=ROOT,
            robot_type="mujoco",
            fps=FPS,
            features=features
        )
    else:
        dataset = LeRobotDataset(REPO_ID, root=ROOT)

    return dataset

def get_obs_state(model: mujoco.MjModel, data: mujoco.MjData, site_left_id: int, site_right_id: int) -> np.ndarray:
    lpos, lquat = site_pose(model, data, site_left_id) # 3 4
    rpos, rquat = site_pose(model, data, site_right_id) # 3 4 
    return np.concatenate([lpos, lquat, rpos, rquat], axis=0)

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def main():
    # dataset setup
    TASK_NAME = "dual_arm_teleop"
    NUM_DEMO = 50

    model, data, configuration = initialize_model()
    dataset = make_or_load_dataset(model)

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    ee_left, ee_right = pick_two_ee_sites(model)
    print(f"[INFO] EE sites: {ee_left}, {ee_right}")

    site_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_left)
    site_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_right)

    # tasks
    left_task = mink.FrameTask(
        frame_name=ee_left, frame_type="site",
        position_cost=1.0, orientation_cost=1.0,
        lm_damping=1.0,
    )
    right_task = mink.FrameTask(
        frame_name=ee_right, frame_type="site",
        position_cost=1.0, orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=POSTURE_COST)
    tasks = [left_task, right_task, posture_task]
    posture_task.set_target_from_configuration(configuration)

    joint2act = build_ctrl_map_for_joints(model)

    # init mocap targets to current EE
    mink.move_mocap_to_frame(model, data, "target_left", ee_left, "site")
    mink.move_mocap_to_frame(model, data, "target_right", ee_right, "site")
    mujoco.mj_forward(model, data)

    # timing
    rate = RateLimiter(frequency=RATE_HZ, warn=False)
    rec_accum = 0.0

    # Quest3 input
    teleop = Quest3Teleop()

    follow_left  = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
    follow_right = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # episode logic
    episode_id = 0
    record_flag = False

    #renderer
    renderer = mujoco.Renderer(model, height=256, width=256)
    camera_name = "my_camera"  
    
    # reset
    def hard_reset():
        nonlocal record_flag, follow_left, follow_right
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        mink.move_mocap_to_frame(model, data, "target_left", ee_left, "site")
        mink.move_mocap_to_frame(model, data, "target_right", ee_right, "site")
        mujoco.mj_forward(model, data)
        dataset.clear_episode_buffer()
        record_flag = False
        follow_left  = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
        follow_right = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))
        print("[RESET] env + episode buffer cleared")
    
    prev_reset = False
    prev_done = False

    # loop
    try:
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            while viewer.is_running() and episode_id < NUM_DEMO:
                frame_dt = rate.dt # 0.01s
                ik_dt = frame_dt / MAX_ITERS_PER_CYCLE # 0.0005s

                # controller input
                frame = teleop.read()

                # controller 4x4 homogeneous transform
                T_ctrl_L = T_from_pos_quat_xyzw(frame.left_pose.pos, frame.left_pose.quat)
                T_ctrl_R = T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)

                # left controller axis compensation
                T_ctrl_L = T_ctrl_L @ CTRL2EE_LEFT
                T_ctrl_R = T_ctrl_R @ CTRL2EE_RIGHT

                # mocap
                mocap_l = model.body("target_left").mocapid
                mocap_r = model.body("target_right").mocapid

                # mocap 4x4 homogeneous transform
                T_moc_L_now = T_from_mocap(model, data, mocap_l)
                T_moc_R_now = T_from_mocap(model, data, mocap_r)

                # controller delta -> mocap
                left_ok, T_L_des = follow_left.update(frame.left_state.squeeze, T_ctrl_L, T_moc_L_now)
                right_ok, T_R_des = follow_right.update(frame.right_state.squeeze, T_ctrl_R, T_moc_R_now)

                if left_ok and T_L_des is not None:
                    set_mocap_from_T(data, mocap_l, T_L_des)
                if right_ok and T_R_des is not None:
                    set_mocap_from_T(data, mocap_r, T_R_des)

                # record start condition: mocap being updated
                if (not record_flag) and (left_ok or right_ok):
                    record_flag = True
                    print("[DATASET] Start recording")

                # reset: right controller button B
                reset_now = bool(frame.right_state.button1)
                reset = reset_now and (not prev_reset)
                prev_reset = reset_now
                if reset:
                    hard_reset()
                    viewer.sync()
                    rate.sleep()
                    continue

                # done: right controller button A 
                done_now = bool(frame.right_state.button0)
                done = done_now and (not prev_done)
                prev_done = done_now

                if done:
                    print(f"[DONE] button A pressed. ")
                    if record_flag:
                        dataset.save_episode()
                        episode_id += 1
                        print(f"[DATASET] Episode done. episode_id={episode_id}/{NUM_DEMO}")
                        hard_reset()
                    continue

                # set mocap to IK task target
                left_task.set_target(mink.SE3.from_mocap_name(model, data, "target_left"))
                right_task.set_target(mink.SE3.from_mocap_name(model, data, "target_right"))

                # copy current targets (to check reached)
                left_target_pos = data.mocap_pos[mocap_l].copy()
                right_target_pos = data.mocap_pos[mocap_r].copy()
                left_target_quat = data.mocap_quat[mocap_l].copy()
                right_target_quat = data.mocap_quat[mocap_r].copy()

                # IK sub-iterations
                reached = False
                for _ in range(MAX_ITERS_PER_CYCLE):
                    vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                    configuration.integrate_inplace(vel, ik_dt)

                    apply_configuration(model, data, configuration, joint2act=joint2act)
                    mujoco.mj_step(model, data)

                    reached = check_reached(
                        model, data,
                        site_left_id, site_right_id,
                        left_target_pos, left_target_quat,
                        right_target_pos, right_target_quat,
                        POS_THRESHOLD, ORI_THRESHOLD,
                    )
                    if reached:
                        break

                # --------------- data collection ---------------
                rec_accum += frame_dt
                if rec_accum >= REC_DT:
                    rec_accum -= REC_DT

                    if camera_name is None:
                        renderer.update_scene(data)          # default free camera
                    else:
                        renderer.update_scene(data, camera=camera_name)
                    rgb = renderer.render()                  # (256,256,3) uint8

                    obs_state = get_obs_state(model, data, site_left_id, site_right_id)
                    action_qpos = data.qpos.copy()

                    left_target_pos  = _vec1(data.mocap_pos[mocap_l])[:3]
                    right_target_pos = _vec1(data.mocap_pos[mocap_r])[:3]
                    left_target_quat  = _vec1(data.mocap_quat[mocap_l])[:4]
                    right_target_quat = _vec1(data.mocap_quat[mocap_r])[:4]

                    target_state = np.concatenate(
                        [left_target_pos, left_target_quat, right_target_pos, right_target_quat],
                        axis=0
                    )

                    if record_flag:
                        dataset.add_frame(
                            {
                                "observation.image": rgb,
                                "observation.state": obs_state.astype(np.float32),
                                "observation.target": target_state.astype(np.float32),
                                "action": action_qpos.astype(np.float32),
                                "task": TASK_NAME,  
                            }
                        )
                # ----------------------------------------------------

                viewer.sync()
                rate.sleep()
    finally:
        dataset.finalize()
        print("[DATASET] finalize() done")

if __name__ == "__main__":
    main()