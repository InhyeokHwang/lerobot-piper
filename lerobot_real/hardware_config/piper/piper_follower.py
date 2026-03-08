# piper_robot.py
from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Optional, Union, Mapping, Any
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from .piper_config import PiperRobotConfig
from .piper_bus import PiperMotorsBus

logger = logging.getLogger(__name__)

class PiperFollower(Robot):
    config_class = PiperRobotConfig
    name = "piper_follower"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config

        motors: dict[str, Motor] = {
            "joint_1": Motor(1, "piper_joint", MotorNormMode.DEGREES),
            "joint_2": Motor(2, "piper_joint", MotorNormMode.DEGREES),
            "joint_3": Motor(3, "piper_joint", MotorNormMode.DEGREES),
            "joint_4": Motor(4, "piper_joint", MotorNormMode.DEGREES),
            "joint_5": Motor(5, "piper_joint", MotorNormMode.DEGREES),
            "joint_6": Motor(6, "piper_joint", MotorNormMode.DEGREES),
        }

        self.bus = PiperMotorsBus(port=self.config.port, motors=motors)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float

            if getattr(self.bus, "supports_velocity", False): # TODO: 이거 확인하는거 검증
                features[f"{motor}.vel"] = float

            if getattr(self.bus, "supports_torque", False): # TODO: 이거 확인하는거 검증
                features[f"{motor}.torque"] = float
        return features
    
    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}
    
    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        self._calibrated = True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        self.bus.enable_torque()
        try:
            self.bus._sdk.set_motion_mode(ctrl_mode=0x01, move_mode=0x01, speed=50, is_mit_mode=0x00)
            logger.info("[PIPER] MotionCtrl_2 re-applied after enable_torque()")
        except Exception as e:
            logger.warning(f"[PIPER] MotionCtrl_2 re-apply failed: {e}")
        if calibrate:
            self.calibrate()

        logger.info(f"{self} connected.")

    def configure(self) -> None:
        self.bus.configure_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation: # only for robot joints and gripper
        start = time.perf_counter()

        states = self.bus.sync_read_all_states()
        obs: dict[str, Any] = {}

        # arm joints (deg)
        for m in self.bus.motors.keys():
            st = states.get(m, {})
            obs[f"{m}.pos"] = float(st.get("position", 0.0))
            obs[f"{m}.vel"] = float(st.get("velocity", 0.0))
            obs[f"{m}.torque"] = float(st.get("torque", 0.0))

        # gripper (mm + norm)
        try:
            mm = float(self.bus._sdk.read_gripper_position_mm())
            obs["gripper.mm"] = mm

            max_mm = float(getattr(self.config, "gripper_max_mm", 35.0))
            obs["gripper.pos"] = float(np.clip(mm / max_mm, 0.0, 1.0))
        except Exception:
            pass

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation: {dt_ms:.1f}ms")
        return obs
    
    @check_if_not_connected
    def get_camera_frames(self) -> dict[str, Any]: # camera obs
        obs: dict[str, Any] = {}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs
    
    def _gripper_norm_to_mm(self, g: float) -> float:
        g = max(0.0, min(1.0, float(g)))

        close_mm, open_mm = self.config.gripper_range_mm

        # open=1, close=0
        return close_mm + g * (open_mm - close_mm)


    @check_if_not_connected
    def send_action(
        self,
        action: Union[Mapping[str, Any], np.ndarray],
        gripper_norm: Optional[float] = None,
    ) -> RobotAction:
        """
        Accepts either:
        1) action dict: {"joint_1.pos": ..., "gripper.pos": ...}
        2) q_deg ndarray: (6,) degrees, with optional gripper_norm

        This lets main code always call robot.send_action(...)
        """
        if isinstance(action, np.ndarray):
            q_deg = np.asarray(action, dtype=np.float64).reshape(-1)
            if q_deg.shape[0] < 6:
                raise ValueError(f"q_deg must have at least 6 elements, got {q_deg.shape}")

            act: RobotAction = {f"joint_{i}.pos": float(q_deg[i - 1]) for i in range(1, 7)}
            if gripper_norm is not None:
                act["gripper.pos"] = float(np.clip(gripper_norm, 0.0, 1.0))
            action = act  

        action = dict(action) 

        goal_pos = {
            k.removesuffix(".pos"): float(v)
            for k, v in action.items()
            if isinstance(k, str) and k.endswith(".pos")
        }

        joint_limits = getattr(self.config, "joint_limits", None)
        if joint_limits:
            for m, val in list(goal_pos.items()):
                if m in joint_limits:
                    lo, hi = joint_limits[m]
                    goal_pos[m] = max(float(lo), min(float(hi), float(val)))

        gripper_norm2 = goal_pos.pop("gripper", None)

        # joints 전송
        if goal_pos:
            self.bus.sync_write("Goal_Position", goal_pos)

        # gripper 전송 (0~1 -> mm)
        if gripper_norm2 is not None:
            gripper_norm2 = float(np.clip(gripper_norm2, 0.0, 1.0))
            target_mm = self._gripper_norm_to_mm(gripper_norm2)
            self.bus._sdk.send_gripper_mm(
                target_mm,
                effort_n=self.config.gripper_effort_n,
                enable=True,
            )
        out = {f"{m}.pos": v for m, v in goal_pos.items()}
        if gripper_norm2 is not None:
            out["gripper.pos"] = float(gripper_norm2)
        return out
    

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(getattr(self.config, "disable_torque_on_disconnect", False))
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
