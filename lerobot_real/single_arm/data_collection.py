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

from quest3.quest3_thread import start_teleop_thread
from lerobot_real.hardware_config.piper.piper_thread import start_piper_thread, publish_piper_cmd, clear_piper_cmd
from lerobot_real.hardware_config.camera.camera_thread import start_cam_thread
from queue import Queue, Full
from lerobot_real.hardware_config.camera.dataset_writer_thread import start_dataset_writer_thread
from lerobot_real.hardware_config.piper.obs_thread import start_obs_thread, get_latest_obs_copy
from lerobot_real.hardware_config.piper.piper_config import PiperRobotConfig
from lerobot_real.hardware_config.piper.piper_follower import PiperFollower

_HERE = Path(__file__).parent
SOLVER = "daqp"

# IK
POSTURE_COST = 1e-3
MAX_ITERS_PER_CYCLE = 20
DAMPING = 1e-3

POS_THRESHOLD = 1e-2   # 1 cm
ORI_THRESHOLD = 2e-2   # 1.1 deg

# Control loop rate
RATE_HZ = 100.0

# Data collection rate
REC_HZ = 15.0
REC_DT = 1.0 / REC_HZ

HOME_REACH_TOL_DEG = 1.5
HOME_TIMEOUT_SEC = 12.0
HOME_SETTLE_SEC = 2.0

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

REPO_ID = "piper_single_arm_teleop_real"
DATASET_HOME = (_HERE / "demo_data").resolve()
FPS = int(REC_HZ)

IMG_W = 640 
IMG_H = 480

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def make_or_load_dataset(*, model: mujoco.MjModel) -> LeRobotDataset:
    dataset_root = DATASET_HOME / REPO_ID
    create_new = not (dataset_root / "meta").exists()

    features = {
        "observation.front_image": { # front cam
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": { # wrist cam
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
            "shape": (model.nq,), # 1~6 for arm 7, 8 for gripper
            "names": ["qpos"],
        }
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

# extract arm joint 1~6 from obs
def extract_qpos_deg_from_obs(obs: Dict) -> np.ndarray:
    return np.array([obs[f"joint_{i}.pos"] for i in range(1, 7)], dtype=np.float64)

# get keyframe
def get_home_qpos(model: mujoco.MjModel, key_id: int) -> np.ndarray:
    kq = np.asarray(model.key_qpos)
    if kq.ndim == 2:
        q_home = kq[key_id]
    else:
        q_home = kq[key_id * model.nq : (key_id + 1) * model.nq]
    q_home = np.asarray(q_home, dtype=np.float64).reshape(-1)
    q_arm_deg = np.rad2deg(q_home[:6]) # arm joint (deg)
    q_gripper = q_home[6:8] # gripper 0-0.035 (m)
    return np.concatenate([q_arm_deg, q_gripper])

def move_to_keyframe_home(
    *,
    model: mujoco.MjModel,
    key_id: int,
    robot: PiperFollower,
    rate: RateLimiter,
    tol_arm_deg: float,
    tol_gripper: float,   
    timeout_sec: float,
    latest_cmd: dict,
    cmd_lock: threading.Lock,
) -> bool:
    q_goal = get_home_qpos(model, key_id)  # arm(deg) + gripper(m,m)
    arm_goal_deg = np.asarray(q_goal[:6], dtype=np.float64)

    # (0~0.035)
    grip_goal_m = float(q_goal[6])
    grip_goal_norm = float(np.clip(grip_goal_m / 0.035, 0.0, 1.0))

    print(f"[HOME] arm_goal_deg = {np.round(arm_goal_deg, 2)} | grip_goal_norm = {grip_goal_norm:.3f}")

    # send initial command 
    publish_piper_cmd(
        latest_cmd=latest_cmd,
        cmd_lock=cmd_lock,
        q_cmd_deg=arm_goal_deg,
        grip_cmd_m=grip_goal_m,
    )

    print("Home settle waiting")
    time.sleep(HOME_SETTLE_SEC)
    print("Home settle finished")

    t_start = time.perf_counter()
    while True:
        obs = robot.get_observation()
        arm_meas_deg = extract_qpos_deg_from_obs(obs)

        arm_err = np.abs(arm_goal_deg - arm_meas_deg)
        arm_ok = bool(np.all(arm_err <= tol_arm_deg))
        
        grip_meas_raw = obs.get("gripper.pos", None)
        if grip_meas_raw is None:
            grip_meas_norm = None
            grip_ok = True
            grip_err = float("nan")
        else:
            grip_meas_norm = float(np.clip(float(grip_meas_raw), 0.0, 1.0))
            grip_err = abs(grip_goal_norm - grip_meas_norm)
            grip_ok = bool(grip_err <= tol_gripper)

        if arm_ok and grip_ok:
            max_arm_err = float(arm_err.max())
            print(f"[HOME] reached (max_arm_err={max_arm_err:.2f} deg, grip_err={grip_err}).")
            return True

        if (time.perf_counter() - t_start) > timeout_sec:
            max_arm_err = float(arm_err.max())
            print(f"[HOME] timeout after {timeout_sec:.1f}s (max_arm_err={max_arm_err:.2f} deg). Holding measured.")

            grip_hold_m = 0.0 if grip_meas_norm is None else (grip_meas_norm * 0.035)
            publish_piper_cmd(
                latest_cmd=latest_cmd,
                cmd_lock=cmd_lock,
                q_cmd_deg=np.asarray(arm_meas_deg, dtype=np.float64),
                grip_cmd_m=grip_hold_m,
            )
            return False
        # retry
        publish_piper_cmd(
            latest_cmd=latest_cmd,
            cmd_lock=cmd_lock,
            q_cmd_deg=arm_goal_deg,
            grip_cmd_m=grip_goal_m,
        )
        rate.sleep()

def main():
    TASK_NAME = REPO_ID
    NUM_DEMO = 50

    # Quest3
    teleop = Quest3Teleop()

    # stop event
    stop_event = threading.Event()

    # MuJoCo model for IK/FK internal
    model, data, configuration = initialize_model()

    # dataset
    dataset = make_or_load_dataset(model=model)

    ### threads ###
    # quest3 thread
    latest_quest3 = {"frame": None}
    # piper thread
    latest_piper = {"payload": None, "seq": 0}
    cmd_lock = threading.Lock()
    piper_th = None
    # camera thread
    latest_cams = {"front_cam": None, "wrist_cam": None, "stamp": 0.0, "seq": 0}
    cam_lock = threading.Lock()
    cam_th = None
    # writer thread
    write_queue = Queue(maxsize=128)
    writer_th = None
    # obs thread
    latest_obs = {"obs": None, "stamp": 0.0, "seq": 0}
    obs_lock = threading.Lock()
    obs_th = None

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
    posture_task.set_target_from_configuration(configuration)

    # Connect hardware
    piper_cfg = PiperRobotConfig(port="can0")
    robot = PiperFollower(piper_cfg)
    robot.connect(calibrate=False)

    print("robot connected state:", getattr(robot, "is_connected", "NO_ATTR"))
    print("bus connected state:", getattr(robot.bus, "is_connected", "NO_ATTR"))    

    rate = RateLimiter(frequency=RATE_HZ, warn=False)
    rec_accum = 0.0

    follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=np.eye(3))

    # hw -> mujoco sync
    def sync_mujoco_from_measured(obs: Dict):
        q_meas_deg = extract_qpos_deg_from_obs(obs)
        data.qpos[:6] = np.deg2rad(q_meas_deg)
        grip_meas_norm = obs.get("gripper.pos", None)
        grip_m = 0.0 if grip_meas_norm is None else float(np.clip(grip_meas_norm, 0.0, 1.0)) * 0.035
        data.qpos[6] = grip_m
        data.qpos[7] = -grip_m
        mujoco.mj_forward(model, data)
        configuration.update(data.qpos)

    # episode logic
    episode_id = 0 # number of episodes
    record_flag = False
    prev_reset = False
    prev_done = False

    def hard_reset():
        nonlocal record_flag, follow, rec_accum
        record_flag = False
        rec_accum = 0.0

        write_queue.join() # flush queue

        follow = Controller(use_rotation=True, pos_scale=1.0, R_fix=follow.R_fix)

        clear_piper_cmd(latest_cmd=latest_piper, cmd_lock=cmd_lock)
        ## move to key frame
        if key_id != -1:
            print("[RESET] moving to keyframe 'home'...")
            move_to_keyframe_home(
                model=model,
                key_id=key_id,
                robot=robot,
                rate=rate,
                tol_arm_deg=HOME_REACH_TOL_DEG,
                tol_gripper=0.003/0.035,  # 3mm -> norm
                timeout_sec=HOME_TIMEOUT_SEC,
                latest_cmd= latest_piper,
                cmd_lock= cmd_lock,
            )
            print("Home Settle Done")
        ## 

        obs = get_latest_obs_copy(latest_obs, obs_lock)

        if obs is not None:
            sync_mujoco_from_measured(obs)

        posture_task.set_target_from_configuration(configuration)
        
        mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
        mujoco.mj_forward(model, data)

        dataset.clear_episode_buffer()
        print("[RESET] env + episode buffer cleared")

    try:
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=True) as viewer:
            ## quest3 thread ##
            quest3_th = start_teleop_thread(teleop=teleop, latest=latest_quest3, stop_event=stop_event, hz=100.0)
            ## piper thread ##
            piper_th = start_piper_thread(robot=robot, latest_cmd=latest_piper, cmd_lock=cmd_lock, stop_event=stop_event, hz=RATE_HZ)
            ## camera thread ##
            cam_th = start_cam_thread(robot=robot, latest_cams=latest_cams, cam_lock=cam_lock, stop_event=stop_event, hz=15.0)
            ## writer thread ##
            writer_th = start_dataset_writer_thread(dataset=dataset, write_queue=write_queue, stop_event=stop_event)
            ## obs thread ##
            obs_th = start_obs_thread(robot=robot, latest_obs=latest_obs, obs_lock=obs_lock, stop_event=stop_event, hz=RATE_HZ)
            
            # hard reset for sending robot to keyframe pos
            hard_reset()

            while viewer.is_running() and episode_id < NUM_DEMO:
                frame = latest_quest3["frame"] # teleop frame
                if frame is None:
                    viewer.sync()
                    rate.sleep()
                    continue

                frame_dt = rate.dt
                ik_dt = frame_dt / float(max(1, MAX_ITERS_PER_CYCLE))

                # obs -> mujoco
                obs = get_latest_obs_copy(latest_obs, obs_lock)# camera should not block control loop -> separate
                if obs is not None:
                    sync_mujoco_from_measured(obs)

                # gripper
                grip_cmd_norm = 1.0 - float(np.clip(float(frame.right_state.trigger), 0.0, 1.0))
                grip_cmd_m = grip_cmd_norm * 0.035

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
                    print("[ABORT] button B pressed. discard episode + reset")
                    hard_reset()
                    rate.sleep()
                    continue
                
                # done (A)
                done_now = bool(frame.right_state.button0) 
                done = done_now and (not prev_done)
                prev_done = done_now
                if done:
                    print(f"[DONE] button A pressed. record_flag={record_flag}")
                    if record_flag:
                        write_queue.join() # flush queue
                        dataset.save_episode()
                        episode_id += 1
                        print(f"[DATASET] Episode saved. episode_id={episode_id}/{NUM_DEMO}")
                    hard_reset()
                    continue

                # IK target
                ee_task.set_target(mink.SE3.from_mocap_name(model, data, "target"))
                target_pos = data.mocap_pos[mocap_id].copy()
                target_quat = data.mocap_quat[mocap_id].copy()

                for _ in range(MAX_ITERS_PER_CYCLE):
                    vel = mink.solve_ik(configuration, tasks, ik_dt, SOLVER, DAMPING)
                    configuration.integrate_inplace(vel, ik_dt)

                    data.qpos[:] = configuration.q # IK calculated value -> data
                    mujoco.mj_forward(model, data) # FK -> EE pos update -> data

                    reached = check_reached_single(
                        model, data, site_id,
                        target_pos, target_quat,
                        POS_THRESHOLD, ORI_THRESHOLD,
                    )
                    if reached:
                        break
                
                # IK calculated deg
                q_cmd_deg = np.rad2deg(np.array(configuration.q, dtype=np.float64)[:6].copy())
                publish_piper_cmd(latest_cmd=latest_piper, cmd_lock=cmd_lock, q_cmd_deg=q_cmd_deg, grip_cmd_m=grip_cmd_m)

                # -------- data collection --------
                rec_accum += frame_dt
                if rec_accum >= REC_DT:
                    rec_accum -= REC_DT

                    obs_state = get_obs_state(model, data, site_id)
                    tpos = _vec1(data.mocap_pos[mocap_id])[:3]
                    tquat = _vec1(data.mocap_quat[mocap_id])[:4]
                    target_state = np.concatenate([tpos, tquat], axis=0).astype(np.float32)

                    # arm 6, grip 2 
                    arm_qpos = np.deg2rad(q_cmd_deg).astype(np.float32)
                    grip_qpos = np.array([grip_cmd_m, -grip_cmd_m], dtype=np.float32)
                    action_qpos = np.concatenate([arm_qpos, grip_qpos], axis=0)
                                        
                    if record_flag:
                        front_img = None
                        wrist_img = None
                        with cam_lock:
                            if latest_cams["front_cam"] is not None:
                                front_img = latest_cams["front_cam"].copy()
                            if latest_cams["wrist_cam"] is not None:
                                wrist_img = latest_cams["wrist_cam"].copy()

                        if front_img is not None and wrist_img is not None:
                            frame_dict = {
                                "observation.state": obs_state,
                                "observation.target": target_state,
                                "action": action_qpos,
                                "task": TASK_NAME,
                                "observation.front_image": front_img,
                                "observation.wrist_image": wrist_img,
                            }
                            # put into queue 
                            try:
                                write_queue.put_nowait(frame_dict)
                            except Full:
                                print("[WRITE][DROP] queue full, dropping frame")
                # ---------------------------------------
                viewer.sync()
                rate.sleep()

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt")
    finally:
        # stop threads
        stop_event.set()

        # thread
        try:
            quest3_th.join(timeout=1.0)
            piper_th.join(timeout=1.0)
            cam_th.join(timeout=1.0)
            write_queue.join()
            writer_th.join(timeout=1.0)
            obs_th.join(timeout=1.0)
        except Exception:
            pass

        # all zero
        try:
            time.sleep(1.0)#wait
            q_zero_deg = np.zeros(6, dtype=np.float64)
            grip_zero_norm = 0.0

            robot.send_action(q_zero_deg, grip_zero_norm)
            print("[SAFE] sent zero-pose command.")
            time.sleep(3.0)# wait
        except Exception as e:
            print(f"[WARN] failed to send zero-pose command: {e}")

        dataset.finalize()

        try:
            robot.disconnect()
        except Exception as e:
            print(f"[WARN] robot.disconnect failed: {e}")

        print("[DATASET] finalize() done")


if __name__ == "__main__":
    main()