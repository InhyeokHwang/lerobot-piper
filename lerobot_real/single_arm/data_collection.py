from __future__ import annotations

from pathlib import Path
from typing import Dict
import threading
import time
from typing import Dict
import numpy as np
import mujoco
import mujoco.viewer

import mink
from loop_rate_limiters import RateLimiter

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from quest3.quest3_teleop import Quest3Teleop

from mink_ik.single_arm_mink_ik import (
    pick_ee_site,
    site_pose,
    check_reached_single,
    initialize_model,
)

from mink_ik.quest3_utils import (
    Controller,
    T_from_pos_quat_xyzw,
    set_mocap_from_T,
    T_from_mocap,
)

from lerobot_real.hardware_config.piper.piper_config import PiperRobotConfig
from lerobot_real.hardware_config.piper.piper_follower import PiperFollower

_HERE = Path(__file__).parent
SOLVER = "daqp"

# IK
POSTURE_COST = 1e-3
MAX_ITERS_PER_CYCLE = 20
DAMPING = 1e-3

POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4

# Control loop rate
RATE_HZ = 50.0

# Data collection rate
REC_HZ = 20.0
REC_DT = 1.0 / REC_HZ

# Safety clamp
MAX_DQ_PER_STEP_DEG = 3.5

HOME_REACH_TOL_DEG = 1.5
HOME_TIMEOUT_SEC = 12.0
HOME_SETTLE_SEC = 0.3

# roll <-> yaw
R_SWAP_XZ = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# sign fix
R_FLIP_RP = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

REPO_ID = "piper_single_arm_teleop"
DATASET_HOME = (_HERE / "demo_data").resolve()
FPS = int(REC_HZ)

IMG_H, IMG_W = 256, 256  # dataset image shape
DUMMY_FRONT = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
DUMMY_WRIST = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def make_or_load_dataset(*, model: mujoco.MjModel) -> LeRobotDataset:
    dataset_root = DATASET_HOME / REPO_ID
    create_new = not (dataset_root / "meta").exists()

    features = {
        "observation.front_image": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "observation.target": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["target"],
        },
        "action": {
            "dtype": "float32",
            "shape": (model.nq,),
            "names": ["qpos"],
        },
        "action.gripper": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_norm"],
        },
    }

    if create_new:
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            root=str(DATASET_HOME),
            robot_type="real",
            fps=FPS,
            features=features,
        )
        print(f"[DATASET] created at: {dataset_root}")
    else:
        dataset = LeRobotDataset(REPO_ID, root=str(DATASET_HOME))
        print(f"[DATASET] loaded from: {dataset_root}")
    return dataset

def get_obs_state(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    pos, quat = site_pose(model, data, site_id)  # quat is wxyz
    return np.concatenate([pos, quat], axis=0).astype(np.float32)

def clamp_step_deg(q_meas_deg: np.ndarray, q_cmd_deg: np.ndarray, max_dq_deg: float) -> np.ndarray:
    dq = np.clip(q_cmd_deg - q_meas_deg, -max_dq_deg, max_dq_deg)
    return q_meas_deg + dq

def extract_qpos_deg_from_obs(obs: Dict) -> np.ndarray:
    return np.array([obs[f"joint_{i}.pos"] for i in range(1, 7)], dtype=np.float64)


def get_home_qpos_deg(model: mujoco.MjModel, key_id: int) -> np.ndarray:
    kq = np.asarray(model.key_qpos)
    if kq.ndim == 2:
        q_home = kq[key_id]
    else:
        q_home = kq[key_id * model.nq : (key_id + 1) * model.nq]
    q_home = np.asarray(q_home, dtype=np.float64).reshape(-1)
    return np.rad2deg(q_home[:6]).reshape(6,)

def move_to_keyframe_home(
    *,
    model: mujoco.MjModel,
    key_id: int,
    robot: PiperFollower,
    rate: RateLimiter,
    tol_deg: float,
    timeout_sec: float,
) -> bool:
    q_goal_deg = get_home_qpos_deg(model, key_id)
    print(f"[HOME] goal_deg = {np.round(q_goal_deg, 2)}")
    robot.send_action(q_goal_deg)

    t_start = time.perf_counter()
    while True:
        obs = robot.get_observation()
        q_meas_deg = extract_qpos_deg_from_obs(obs)
        err = np.abs(q_goal_deg - q_meas_deg)

        if np.all(err <= tol_deg):
            print(f"[HOME] reached (max_err={float(err.max()):.2f} deg).")
            return True

        if (time.perf_counter() - t_start) > timeout_sec:
            print(f"[HOME] timeout after {timeout_sec:.1f}s (max_err={float(err.max()):.2f} deg). Holding.")
            robot.send_action(q_meas_deg)
            return False

        rate.sleep()

def main():
    TASK_NAME = "piper_single_arm_quest3_real"
    NUM_DEMO = 50

    # Quest3
    teleop = Quest3Teleop()

    # MuJoCo model for IK/FK internal
    model, data, configuration = initialize_model()

    # dataset
    dataset = make_or_load_dataset(model=model)

    # keyframe home
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    # EE site
    ee_site = pick_ee_site(model)
    print(f"[INFO] EE site: {ee_site}")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    if site_id < 0:
        raise RuntimeError(f"EE site not found: {ee_site}")

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

    mocap_id = model.body("target").mocapid
    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # Connect hardware
    piper_cfg = PiperRobotConfig(port="can0")
    robot = PiperFollower(piper_cfg)
    robot.connect(calibrate=False)

    # ---------------- timing ----------------
    rate = RateLimiter(frequency=RATE_HZ, warn=False)
    rec_accum = 0.0

    # episode logic
    episode_id = 0
    record_flag = False
    episode_frames = 0  

    prev_reset = False
    prev_done = False

    latest = {"frame": None}
    stop_event = threading.Event()

    def teleop_thread():
        while not stop_event.is_set():
            latest["frame"] = teleop.read()
            time.sleep(0.01)

    th = threading.Thread(target=teleop_thread, daemon=True)
    th.start()

    # hw -> mujoco sync
    def sync_mujoco_from_measured():
        obs = robot.get_observation()
        q_meas_deg = extract_qpos_deg_from_obs(obs)
        data.qpos[:6] = np.deg2rad(q_meas_deg)
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)
        return q_meas_deg

    def hard_reset():
        nonlocal record_flag, follow, rec_accum, episode_frames

        dataset.clear_episode_buffer()
        episode_frames = 0
        record_flag = False
        rec_accum = 0.0

        follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=follow.R_fix)

        if key_id != -1:
            print("[RESET] moving to keyframe 'home'...")
            move_to_keyframe_home(
                model=model,
                key_id=key_id,
                robot=robot,
                rate=rate,
                tol_deg=HOME_REACH_TOL_DEG,
                timeout_sec=HOME_TIMEOUT_SEC,
            )
            time.sleep(HOME_SETTLE_SEC)

        try:
            robot.send_action({"gripper.pos": 1.0})
        except Exception as e:
            print(f"[WARN] gripper open failed: {e}")

        sync_mujoco_from_measured()
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)
        posture_task.set_target_from_configuration(configuration)

        print("[RESET] env + episode buffer cleared")

    try:
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) as viewer:

            hard_reset()

            while viewer.is_running() and episode_id < NUM_DEMO:
                frame = latest["frame"]
                if frame is None:
                    viewer.sync()
                    rate.sleep()
                    continue

                grip_norm = float(np.clip(float(frame.right_state.trigger), 0.0, 1.0))

                frame_dt = rate.dt
                ik_dt = frame_dt / float(max(1, MAX_ITERS_PER_CYCLE))

                reset_now = bool(frame.right_state.button1)  # B
                reset = reset_now and (not prev_reset)
                prev_reset = reset_now
                if reset:
                    print("[ABORT] button B pressed. discard episode + reset")
                    hard_reset()
                    rate.sleep()
                    continue

                done_now = bool(frame.right_state.button0)  # A
                done = done_now and (not prev_done)
                prev_done = done_now
                if done:
                    print(f"[DONE] button A pressed. record_flag={record_flag} frames={episode_frames}")
                    if record_flag:
                        dataset.save_episode()
                        episode_id += 1
                        print(f"[DATASET] Episode saved. episode_id={episode_id}/{NUM_DEMO}")
                    hard_reset()
                    continue

                q_meas_deg = sync_mujoco_from_measured()

                T_ctrl = T_from_pos_quat_xyzw(frame.right_pose.pos, frame.right_pose.quat)
                T_ctrl[:3, :3] = T_ctrl[:3, :3] @ (R_FLIP_RP @ R_SWAP_XZ)

                T_moc_now = T_from_mocap(model, data, mocap_id)
                ok, T_des = follow.update(frame.right_state.squeeze, T_ctrl, T_moc_now)
                if ok and T_des is not None:
                    set_mocap_from_T(data, mocap_id, T_des)

                if (not record_flag) and ok:
                    record_flag = True
                    print("[DATASET] Start recording")

                ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))

                target_pos = data.mocap_pos[mocap_id].copy()
                target_quat = data.mocap_quat[mocap_id].copy()

                for _ in range(MAX_ITERS_PER_CYCLE):
                    vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                    configuration.integrate_inplace(vel, ik_dt)

                    data.qpos[:] = configuration.q
                    mujoco.mj_forward(model, data)

                    reached = check_reached_single(
                        model,
                        data,
                        site_id,
                        target_pos,
                        target_quat,
                        POS_THRESHOLD,
                        ORI_THRESHOLD,
                    )
                    if reached:
                        break

                q_cmd_deg = np.rad2deg(np.array(configuration.q, dtype=np.float64)[:6].copy())
                q_cmd_deg = clamp_step_deg(q_meas_deg, q_cmd_deg, MAX_DQ_PER_STEP_DEG)

                robot.send_action(q_cmd_deg, grip_norm)

                # -------- data collection --------
                rec_accum += frame_dt
                if rec_accum >= REC_DT:
                    rec_accum -= REC_DT

                    obs_state = get_obs_state(model, data, site_id)
                    tpos = _vec1(data.mocap_pos[mocap_id])[:3]
                    tquat = _vec1(data.mocap_quat[mocap_id])[:4]
                    target_state = np.concatenate([tpos, tquat], axis=0).astype(np.float32)

                    action_qpos = np.deg2rad(q_cmd_deg).astype(np.float32)

                    if record_flag:
                        frame_dict = {
                            "observation.state": obs_state,
                            "observation.target": target_state,
                            "action": action_qpos,
                            "action.gripper": np.array([grip_norm], dtype=np.float32),
                            "task": TASK_NAME,
                        }
                        frame_dict["observation.front_image"] = DUMMY_FRONT
                        frame_dict["observation.wrist_image"] = DUMMY_WRIST
                        
                        dataset.add_frame(frame_dict)
                        episode_frames += 1
                # ---------------------------------------
                viewer.sync()
                rate.sleep()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        # stop threads first
        stop_event.set()

        try:
            th.join(timeout=1.0)
        except Exception:
            pass

        # safe hold
        try:
            obs = robot.get_observation()
            q_hold_deg = extract_qpos_deg_from_obs(obs)
            robot.send_action(q_hold_deg)
            print("[SAFE] sent hold action (measured posture).")
        except Exception as e:
            print(f"[WARN] failed to send hold action: {e}")

        # save partial if any
        try:
            if record_flag and episode_frames > 0:
                print("[INFO] Saving partial episode before exit.")
                dataset.save_episode()
        except Exception as e:
            print(f"[WARN] failed to save partial episode: {e}")

        dataset.finalize()

        try:
            robot.disconnect()
        except Exception as e:
            print(f"[WARN] robot.disconnect failed: {e}")

        print("[DATASET] finalize() done")


if __name__ == "__main__":
    main()