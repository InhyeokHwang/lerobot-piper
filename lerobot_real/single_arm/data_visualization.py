from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import mink
import numpy as np
import cv2
from loop_rate_limiters import RateLimiter

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mink_ik.single_arm_mink_ik import pick_ee_site, initialize_model

def _vec1(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)

def set_mocap_target_from_target_state(
    model: mujoco.MjModel, data: mujoco.MjData, target_state: np.ndarray
) -> None:
    ts = _vec1(target_state)

    pos = ts[0:3]
    quat = ts[3:7]  # wxyz expected

    mocap_id = model.body("target").mocapid
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = quat

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str((Path(__file__).resolve().parent / "demo_data").resolve()))
    p.add_argument("--repo_id", type=str, default="piper_single_arm_teleop_real")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()

    root = Path(args.root).resolve()

    dataset_dir = root  
    if not (dataset_dir / "meta").exists() or not (dataset_dir / "data").exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. Expected {dataset_dir}/meta and {dataset_dir}/data."
        )
    
    ds = LeRobotDataset(args.repo_id, root=str(root), download_videos=False)

    if not hasattr(ds, "hf_dataset"):
        raise RuntimeError("This LeRobotDataset version has no hf_dataset attribute; tell me what attributes it has.")

    hf = ds.hf_dataset  # datasets.Dataset
    ep_idx = np.asarray(hf["episode_index"])
    indices = np.nonzero(ep_idx == args.episode)[0]
    if indices.size == 0:
        raise ValueError(f"No frames found for episode_index={args.episode}. Available episodes: {np.unique(ep_idx)[:20]}...")

    # timestamp 
    ts = np.asarray(hf["timestamp"])[indices]
    order = np.argsort(ts)
    indices = indices[order]

    print(f"[VIS] Loaded episode {args.episode}: frames={len(indices)} from dataset={dataset_dir}")

    # MuJoCo init
    model, data, configuration = initialize_model()

    # initial pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # mocap target init
    ee_site = pick_ee_site(model)
    mink.move_mocap_to_frame(model, data, "target", ee_site, "site")
    mujoco.mj_forward(model, data)

    rate = RateLimiter(frequency=float(args.hz), warn=False)

    cv2.namedWindow("front_cam")
    cv2.namedWindow("wrist_cam")

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        i = 0
        while viewer.is_running():
            row = ds[int(indices[i])]  # dict-like

            if "action" in row and row["action"] is not None:
                qpos = _vec1(row["action"])
                if qpos.size != model.nq:
                    raise ValueError(f"action size mismatch: got {qpos.size}, expected {model.nq}")
                data.qpos[:] = qpos
                if data.qvel.size > 0:
                    data.qvel[:] = 0.0

            if "observation.target" in row and row["observation.target"] is not None:
                try:
                    set_mocap_target_from_target_state(model, data, row["observation.target"])
                except Exception as e:
                    print(f"[VIS][WARN] target update failed at i={i}: {type(e).__name__}: {e}")

            # ---------- camera display ----------
            if "observation.front_image" in row:
                front = row["observation.front_image"]
                if front is not None:
                    if hasattr(front, "cpu"):  
                        front = front.cpu().numpy()
                    else:
                        front = np.asarray(front)

                    if front.ndim == 3 and front.shape[0] == 3:
                        front = np.transpose(front, (1, 2, 0))

                    if np.issubdtype(front.dtype, np.floating):
                        if front.max() <= 1.0:
                            front = (front * 255.0).clip(0, 255)
                        else:
                            front = front.clip(0, 255)

                    front = np.ascontiguousarray(front).astype(np.uint8)
                    cv2.imshow("front_cam", cv2.cvtColor(front, cv2.COLOR_RGB2BGR))

            if "observation.wrist_image" in row:
                wrist = row["observation.wrist_image"]
                if wrist is not None:
                    if hasattr(wrist, "cpu"):
                        wrist = wrist.cpu().numpy()
                    else:
                        wrist = np.asarray(wrist)

                    if wrist.ndim == 3 and wrist.shape[0] == 3:
                        wrist = np.transpose(wrist, (1, 2, 0))

                    if np.issubdtype(wrist.dtype, np.floating):
                        if wrist.max() <= 1.0:
                            wrist = (wrist * 255.0).clip(0, 255)
                        else:
                            wrist = wrist.clip(0, 255)

                    wrist = np.ascontiguousarray(wrist).astype(np.uint8)
                    cv2.imshow("wrist_cam", cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            # ------------------------------------

            mujoco.mj_forward(model, data)
            viewer.sync()
            rate.sleep()

            i += 1
            if i >= len(indices):
                if args.loop:
                    i = 0
                else:
                    print("[VIS] Done.")
                    break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()