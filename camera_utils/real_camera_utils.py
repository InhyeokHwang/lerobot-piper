# io/camera.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import time

import numpy as np
import cv2


@dataclass
class CameraStreamerConfig:
    camera: Union[int, str] = 0      # int index OR "/dev/videoX"
    rgb: bool = True                 # convert BGR->RGB
    duplicate_stereo: bool = True    # left=right=frame
    warn_every_n: int = 60           # warn print period on read failure

    # capture preferences (optional)
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    fourcc: Optional[str] = None     # e.g. "YUYV", "MJPG"


class OpenCVCameraStreamer:
    def __init__(self, dst_stereo_array: np.ndarray, config: CameraStreamerConfig | None = None):
        self.cfg = config if config is not None else CameraStreamerConfig()

        self.dst = dst_stereo_array
        if self.dst.ndim != 3 or self.dst.shape[2] != 3:
            raise ValueError(f"dst_stereo_array must be (H, 2W, 3), got {self.dst.shape}")

        self.img_h = int(self.dst.shape[0])
        self.img_w = int(self.dst.shape[1])
        if self.img_w % 2 != 0:
            raise ValueError(f"dst width must be even (2W). got W={self.img_w}")
        self.single_w = self.img_w // 2

        self.cap: Optional[cv2.VideoCapture] = None
        self._fail_count = 0
        self._tick = 0
        self._last_open_try = 0.0

    def _open_target(self):
        cam = self.cfg.camera

        # "/dev/videoX" 같은 디바이스 path면: path 그대로 먼저 open (CAP_ANY)
        if isinstance(cam, str) and cam.startswith("/dev/video"):
            cap = cv2.VideoCapture(cam)  # <- backend 강제하지 말기
            if cap.isOpened():
                return cap
            cap.release()

            return None

        cap = cv2.VideoCapture(cam)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def open(self) -> bool:
        if self.cap is not None:
            return True

        # 너무 자주 open 재시도 방지(USB 순간 끊김 대비)
        now = time.time()
        if now - self._last_open_try < 0.5:
            return False
        self._last_open_try = now

        cap = self._open_target()
        if cap is None:
            print(f"[Camera] Failed to open camera={self.cfg.camera}.")
            self.cap = None
            return False

        # apply settings
        target_w = self.cfg.width if self.cfg.width is not None else self.single_w
        target_h = self.cfg.height if self.cfg.height is not None else self.img_h

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(target_w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(target_h))

        if self.cfg.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(self.cfg.fps))

        if self.cfg.fourcc is not None and len(self.cfg.fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))

        self.cap = cap
        print(f"[Camera] Opened camera={self.cfg.camera}. Target: {target_h} x {target_w}")
        return True

    def close(self):
        if self.cap is None:
            return
        try:
            self.cap.release()
            print("[Camera] Released camera.")
        finally:
            self.cap = None

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] != self.img_h or frame.shape[1] != self.single_w:
            frame = cv2.resize(frame, (self.single_w, self.img_h), interpolation=cv2.INTER_AREA)
        return frame

    def step(self) -> bool:
        self._tick += 1

        if self.cap is None:
            if not self.open():
                return False

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._fail_count += 1
            if self.cfg.warn_every_n > 0 and (self._tick % self.cfg.warn_every_n == 0):
                print("[Camera] Failed to read frame from camera.")
            return False

        if frame.ndim != 3 or frame.shape[2] != 3:
            self._fail_count += 1
            if self.cfg.warn_every_n > 0 and (self._tick % self.cfg.warn_every_n == 0):
                print(f"[Camera] Unexpected frame shape: {getattr(frame,'shape',None)}")
            return False

        if self.cfg.rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = self._resize_if_needed(frame)

        # stereo buffer write
        if self.cfg.duplicate_stereo:
            stereo = np.hstack((frame, frame))
        else:
            stereo = np.hstack((frame, frame))

        np.copyto(self.dst, stereo)
        return True