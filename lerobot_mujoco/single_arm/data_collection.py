from __future__ import annotations

from pathlib import Path
from typing import Dict
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
import mink

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from quest3.quest3_teleop import Quest3Teleop

from camera_utils.camera_renderer import draw_rgb_panels_on_viewer

from mink_ik.single_arm_mink_ik import (
    pick_ee_site,
    site_pose,
    check_reached_single,
    initialize_model,
    build_ctrl_map_for_joints,
    apply_configuration,
)

from mink_ik.quest3_utils import (
    Controller,
    T_from_pos_quat_xyzw,
    set_mocap_from_T,
    T_from_mocap,
)

_HERE = Path(__file__).parent

SOLVER = "daqp"

# IK
POSTURE_COST = 1e-3
MAX_ITERS_PER_CYCLE = 20
DAMPING = 1e-3

# Convergence thresholds
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4

# Viewer loop rate
RATE_HZ = 50.0

# Data collection rate
REC_HZ = 20.0
REC_DT = 1.0 / REC_HZ

# roll <-> yaw
R_SWAP_XZ = np.array([
    [0.0, 0.0, 1.0],
    [0.0,-1.0, 0.0],
    [1.0, 0.0, 0.0],
], dtype=np.float64)

# sign fix
R_FLIP_RP = np.array([
    [-1.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0],
    [ 0.0,  0.0,  1.0],
], dtype=np.float64)

REPO_ID = "piper_single_arm_teleop"
DATASET_HOME = (_HERE / "demo_data").resolve()     
FPS = int(REC_HZ)

IMG_H, IMG_W = 256, 256
AGENT_CAM = "agentview"
WRIST_CAM = "wrist_cam"

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def make_or_load_dataset(*, model: mujoco.MjModel) -> LeRobotDataset:
    dataset_root = DATASET_HOME / REPO_ID
    create_new = not (dataset_root / "meta").exists()

    if create_new:
        features = {
            "observation.front_image": { # front cam
                "dtype": "image",                 
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": { # wrist cam
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),  # pos(3)+quat(4)
                "names": ["state"],
            },
            "observation.target": {
                "dtype": "float32",
                "shape": (7,),  # target pos(3)+quat(4)
                "names": ["target"],
            },
            "action": {
                "dtype": "float32",
                "shape": (model.nq,),  # 1~6 for arm 7, 8 for gripper
                "names": ["qpos"],
            }
        }
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            root=str(DATASET_HOME),     
            robot_type="mujoco",
            fps=FPS,
            features=features,
        )
        print(f"[DATASET] created at: {dataset_root}")
    else:
        dataset = LeRobotDataset(REPO_ID, root=str(DATASET_HOME))
        print(f"[DATASET] loaded from: {dataset_root}")

    return dataset


def get_obs_state(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    pos, quat = site_pose(model, data, site_id)
    return np.concatenate([pos, quat], axis=0)  # (7,)

def gripper_step(
    *, model: mujoco.MjModel, data: mujoco.MjData, cmd: float, state: Dict,
    init: bool = False, act_name: str = "gripper"
) -> None:
    gripper_value = float(np.clip(cmd, 0.0, 1.0))
    gripper_value = 1.0 - gripper_value  # invert

    if init or ("act_id" not in state):
        state["act_id"] = model.actuator(act_name).id
        low, high = model.actuator_ctrlrange[state["act_id"]]
        state["low"] = float(low)
        state["high"] = float(high)

    act_id = int(state["act_id"])
    low = float(state["low"])
    high = float(state["high"])
    data.ctrl[act_id] = low + (high - low) * gripper_value


def main():
    TASK_NAME = REPO_ID
    NUM_DEMO = 50

    # Quest3
    teleop = Quest3Teleop()

    # MuJoCo model
    model, data, configuration = initialize_model()

    # dataset
    dataset = make_or_load_dataset(model=model)

    # camera
    def require_camera(name: str):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id == -1:
            raise RuntimeError(f"Camera '{name}' not found in model. Check XML include/name.")
        return cam_id

    require_camera(AGENT_CAM)
    require_camera(WRIST_CAM)

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)

    # EE site
    ee_site = pick_ee_site(model)
    print(f"[INFO] EE site: {ee_site}")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    if site_id < 0:
        raise RuntimeError(f"EE site not found in model: {ee_site}")

    # IK tasks
    ee_task = mink.FrameTask(
        frame_name=ee_site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=POSTURE_COST)
    tasks = [ee_task, posture_task]
    posture_task.set_target_from_configuration(configuration)

    # map joints to <actuator> for dynamics in simulator
    joint2act = build_ctrl_map_for_joints(model)

    # gripper init
    grip_state: Dict = {}
    gripper_step(model=model, data=data, cmd=0.0, state=grip_state, init=True)

    # init mocap target
    mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=RATE_HZ, warn=False)
    rec_accum = 0.0

    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # episode logic
    episode_id = 0
    record_flag = False
    prev_reset = False
    prev_done = False

    def hard_reset():
        nonlocal record_flag, follow, grip_state

        follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

        ## move to key frame
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        ## 

        posture_task.set_target_from_configuration(configuration)
        gripper_step(model=model, data=data, cmd=0.0, state=grip_state, init=False)

        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)

        dataset.clear_episode_buffer()
        record_flag = False
        print("[RESET] env + episode buffer cleared")

    try:
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) as viewer:
            ## camera ##
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
            last_front_rgb = None
            last_wrist_rgb = None
            ##

            ## teleop ##
            latest = {"frame": None}
            stop_event = threading.Event()
            def teleop_thread():
                while not stop_event.is_set():
                    latest["frame"] = teleop.read()
                    time.sleep(0.01)  
            th = threading.Thread(target=teleop_thread, daemon=True)
            th.start()
            ##

            while viewer.is_running() and episode_id < NUM_DEMO:
                frame = latest["frame"] # teleop frame
                if frame is None:
                    viewer.sync()
                    rate.sleep()
                    continue
                
                frame_dt = rate.dt
                ik_dt = frame_dt / MAX_ITERS_PER_CYCLE

                # gripper
                grip_cmd = float(np.clip(float(frame.right_state.trigger), 0.0, 1.0))

                # controller pose
                T_ctrl = T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)
                T_ctrl[:3, :3] = T_ctrl[:3, :3] @ (R_FLIP_RP @ R_SWAP_XZ)

                mocap_id = model.body("target").mocapid
                T_moc_now = T_from_mocap(model, data, mocap_id)

                ok, T_des = follow.update(frame.right_state.squeeze, T_ctrl, T_moc_now)
                if ok and T_des is not None:
                    set_mocap_from_T(data, mocap_id, T_des)

                if (not record_flag) and ok:
                    record_flag = True
                    print("[DATASET] Start recording")

                # reset (B)
                reset_now = bool(frame.right_state.button1)
                reset = reset_now and (not prev_reset)
                prev_reset = reset_now
                if reset:
                    hard_reset()
                    viewer.sync()
                    rate.sleep()
                    continue

                # done (A)
                done_now = bool(frame.right_state.button0)
                done = done_now and (not prev_done)
                prev_done = done_now
                if done:
                    print(f"[DONE] button A pressed. record_flag={record_flag}")
                    if record_flag:
                        dataset.save_episode()
                        episode_id += 1
                        print(f"[DATASET] Episode done. episode_id={episode_id}/{NUM_DEMO}")
                    hard_reset()
                    continue

                # IK target
                ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))
                target_pos = data.mocap_pos[mocap_id].copy()
                target_quat = data.mocap_quat[mocap_id].copy()

                for _ in range(MAX_ITERS_PER_CYCLE):
                    vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                    configuration.integrate_inplace(vel, ik_dt)

                    # physics
                    apply_configuration(model, data, configuration, joint2act=joint2act)
                    gripper_step(model=model, data=data, cmd=grip_cmd, state=grip_state, init=False)
                    mujoco.mj_step(model, data) # mj_step is for physics

                    reached = check_reached_single(
                        model, data, site_id,
                        target_pos, target_quat,
                        POS_THRESHOLD, ORI_THRESHOLD,
                    )
                    if reached:
                        break

                # -------- data collection --------
                rec_accum += frame_dt
                if rec_accum >= REC_DT:
                    rec_accum -= REC_DT

                    # camera agent view
                    renderer.update_scene(data, camera=AGENT_CAM)
                    front_rgb = np.ascontiguousarray(renderer.render().copy())

                    # camera wrist view
                    renderer.update_scene(data, camera=WRIST_CAM)
                    wrist_rgb = np.ascontiguousarray(renderer.render().copy())

                    last_front_rgb = front_rgb
                    last_wrist_rgb = wrist_rgb
                    
                    if record_flag:
                        obs_state = get_obs_state(model, data, site_id).astype(np.float32)
                        action_qpos = data.qpos.copy().astype(np.float32)

                        tpos = _vec1(data.mocap_pos[mocap_id])[:3]
                        tquat = _vec1(data.mocap_quat[mocap_id])[:4]
                        target_state = np.concatenate([tpos, tquat], axis=0).astype(np.float32)

                        frame_dict = {
                            "observation.state": obs_state,
                            "observation.target": target_state,
                            "action": action_qpos,
                            "task": TASK_NAME,   
                            "observation.front_image": front_rgb,
                            "observation.wrist_image": wrist_rgb,
                        }

                        dataset.add_frame(frame_dict)
                    draw_rgb_panels_on_viewer(viewer, last_front_rgb, last_wrist_rgb) # camera
                # ---------------------------------------
                viewer.sync()
                rate.sleep()

    finally:
        stop_event.set()

        try:
            th.join(timeout=1.0)
        except Exception:
            pass

        dataset.finalize()
        print("[DATASET] finalize() done")

if __name__ == "__main__":
    main()