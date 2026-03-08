from __future__ import annotations

import threading
import time
from typing import Dict, Any

import numpy as np

def start_cam_thread(
    *,
    robot,
    latest_cams: Dict[str, Any],
    cam_lock: threading.Lock,
    stop_event: threading.Event,
    hz: float = 15.0,
):
    period = 1.0 / float(hz)

    def _worker():
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                cam_obs = robot.get_camera_frames()

                front = cam_obs.get("front_cam", None)
                wrist = cam_obs.get("wrist_cam", None)

                if front is not None:
                    front = np.ascontiguousarray(front).copy()
                if wrist is not None:
                    wrist = np.ascontiguousarray(wrist).copy()

                with cam_lock:
                    latest_cams["front_cam"] = front
                    latest_cams["wrist_cam"] = wrist
                    latest_cams["stamp"] = time.perf_counter()
                    latest_cams["seq"] = int(latest_cams.get("seq", 0)) + 1

            except Exception as e:
                # camera thread should never crash the whole control loop
                print(f"[CAM][WARN] failed to read camera frames: {type(e).__name__}: {e}")

            dt = time.perf_counter() - t0
            sleep_t = period - dt
            if sleep_t > 0:
                time.sleep(sleep_t)

    th = threading.Thread(target=_worker, daemon=True, name="cam_thread")
    th.start()
    return th