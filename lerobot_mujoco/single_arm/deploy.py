from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import torch
import torchvision
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata 
from lerobot.datasets.utils import dataset_to_policy_features 
from lerobot.datasets.factory import resolve_delta_timestamps 
from lerobot.configs.types import FeatureType  
from lerobot.policies.act.configuration_act import ACTConfig  
from lerobot.policies.act.modeling_act import ACTPolicy  

XML = str(
    (Path(__file__).resolve().parent.parent.parent / "description" / "agilex_piper" / "scene.xml")
)

class SingleArmMujocoEnv:
    def __init__(
        self,
        xml_path: str,
        *,
        ee_site: str,
        agent_cam: str = "agentview",
        wrist_cam: str = "wrist_cam",
        img_h: int = 256,
        img_w: int = 256,
        keyframe: str = "home",
    ):
        self.xml_path = str(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.ee_site_name = ee_site
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        if self.ee_site_id < 0:
            raise RuntimeError(f"EE site not found: {self.ee_site_name}")

        self.agent_cam = agent_cam
        self.wrist_cam = wrist_cam
        self.agent_cam_id = self._require_camera(agent_cam)
        self.wrist_cam_id = self._require_camera(wrist_cam)

        self.key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        self.renderer = mujoco.Renderer(self.model, height=img_h, width=img_w)

        self.mocap_id = None
        try:
            self.mocap_id = self.model.body("target").mocapid
        except Exception:
            self.mocap_id = None

        self.img_h = img_h
        self.img_w = img_w

    def _require_camera(self, name: str) -> int:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            raise RuntimeError(f"Camera not found: {name}")
        return cam_id

    def reset(self, seed: int = 0):
        if self.key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_ee_state(self) -> np.ndarray:
        pos = self.data.site_xpos[self.ee_site_id].copy()  # (3,)
        xmat = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, xmat.reshape(-1))
        # quat is wxyz
        return np.concatenate([pos, quat], axis=0).astype(np.float32)  # (7,)

    def grab_images(self) -> tuple[np.ndarray, np.ndarray]:
        # agent
        self.renderer.update_scene(self.data, camera=self.agent_cam_id)
        agent = np.ascontiguousarray(self.renderer.render())

        # wrist
        self.renderer.update_scene(self.data, camera=self.wrist_cam_id)
        wrist = np.ascontiguousarray(self.renderer.render())

        return agent, wrist

    def step_qpos(self, qpos: np.ndarray):
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self.model.nq:
            raise ValueError(f"qpos size mismatch: got {qpos.size}, expected {self.model.nq}")

        self.data.qpos[:] = qpos
        if self.data.qvel.size > 0:
            self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", type=str, default=XML, help="MuJoCo xml path")
    p.add_argument("--ee_site", type=str, default="gripper", help="EE site name")
    p.add_argument("--agent_cam", type=str, default="agentview")
    p.add_argument("--wrist_cam", type=str, default="wrist_cam")
    p.add_argument("--img_hw", type=int, default=256)

    p.add_argument("--repo_id", type=str, default="piper_single_arm_teleop")
    p.add_argument("--data_root", type=str, default="./demo_data", help="LeRobot dataset root")
    p.add_argument("--ckpt", type=str, required=True, help="policy checkpoint dir (from_pretrained path)")
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_wrist", action="store_true", help="drop wrist image input")
    p.add_argument("--task", type=str, default="single_arm_task")
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        device = "cpu"
        print("[WARN] CUDA not available -> using CPU")

    # ---- Load dataset metadata for normalization & feature spec ----
    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=args.data_root)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    if args.no_wrist and "observation.wrist_image" in input_features:
        input_features.pop("observation.wrist_image")

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=10,
        n_action_steps=1,
        temporal_ensemble_coeff=0.9,
    )
    _ = resolve_delta_timestamps(cfg, dataset_metadata)

    # policy
    policy = ACTPolicy.from_pretrained(args.ckpt, config=cfg, dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()
    policy.reset()

    # env
    env = SingleArmMujocoEnv(
        args.xml,
        ee_site=args.ee_site,
        agent_cam=args.agent_cam,
        wrist_cam=args.wrist_cam,
        img_h=args.img_hw,
        img_w=args.img_hw,
        keyframe="home",
    )
    env.reset(seed=0)

    to_tensor = torchvision.transforms.ToTensor()
    rate = RateLimiter(frequency=float(args.hz), warn=False)

    step = 0

    with mujoco.viewer.launch_passive(model=env.model, data=env.data, show_left_ui=False, show_right_ui=True) as viewer:
        mujoco.mjv_defaultFreeCamera(env.model, viewer.cam)

        while viewer.is_running():
            # obs
            state = env.get_ee_state()  # (7,)

            agent_rgb, wrist_rgb = env.grab_images()
            agent_img = to_tensor(Image.fromarray(agent_rgb).resize((args.img_hw, args.img_hw)))
            data_dict = {
                "observation.state": torch.tensor([state], device=device),  # (1,7)
                "observation.image": agent_img.unsqueeze(0).to(device),    # (1,3,H,W)
                "task": [args.task],
                "timestamp": torch.tensor([step / args.hz], device=device),
            }

            if (not args.no_wrist):
                wrist_img = to_tensor(Image.fromarray(wrist_rgb).resize((args.img_hw, args.img_hw)))
                data_dict["observation.wrist_image"] = wrist_img.unsqueeze(0).to(device)

            # action
            with torch.no_grad():
                action = policy.select_action(data_dict)  # shape: (1, nq) or similar
            action = action[0].detach().cpu().numpy()

            # step
            env.step_qpos(action)

            viewer.sync()
            rate.sleep()
            step += 1


if __name__ == "__main__":
    main()