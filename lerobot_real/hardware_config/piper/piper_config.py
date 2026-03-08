from dataclasses import dataclass, field
from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Backends
from typing import Tuple

@RobotConfig.register_subclass("piper") # CLI
@dataclass
class PiperRobotConfig(RobotConfig):
    port: str    
    interface: str = "can"
    can_channel: str = "can0"
    gripper_range_mm: Tuple[float, float] = (0.0, 70.0)  # (close_mm, open_mm) 각각 0.035씩임
    gripper_effort_n: float = 2.0  
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front_cam": OpenCVCameraConfig(
                index_or_path="/dev/video4",
                fps=15,
                width=640,
                height=480,
                backend=Cv2Backends.V4L2,
            ),
            "wrist_cam": OpenCVCameraConfig(
                index_or_path="/dev/video10",
                fps=15,
                width=640,
                height=480,
                backend=Cv2Backends.V4L2,
            ),
        }
    )