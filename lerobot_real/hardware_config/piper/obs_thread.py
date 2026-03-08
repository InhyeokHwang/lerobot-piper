from __future__ import annotations

import threading
import time
from typing import Any, Dict

def start_obs_thread(
    *,
    robot,
    latest_obs: Dict[str, Any],
    obs_lock: threading.Lock,
    stop_event: threading.Event,
    hz: float = 100.0,
):
    period = 1.0 / float(hz)

    def _worker():
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                obs = robot.get_observation()

                # shallow copy 정도면 충분
                obs_copy = dict(obs) if obs is not None else None

                with obs_lock:
                    latest_obs["obs"] = obs_copy
                    latest_obs["stamp"] = time.perf_counter()
                    latest_obs["seq"] = int(latest_obs.get("seq", 0)) + 1

            except Exception as e:
                print(f"[OBS][WARN] failed to get observation: {type(e).__name__}: {e}")

            dt = time.perf_counter() - t0
            sleep_t = period - dt
            if sleep_t > 0:
                time.sleep(sleep_t)

    th = threading.Thread(target=_worker, daemon=True, name="obs_thread")
    th.start()
    return th

def get_latest_obs_copy(
    latest_obs: Dict[str, Any],
    obs_lock: threading.Lock,
):
    with obs_lock:
        obs = latest_obs.get("obs", None)
        if obs is None:
            return None
        return dict(obs)