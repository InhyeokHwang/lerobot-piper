# lerobot_real/single_arm/piper_thread.py
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from lerobot_real.hardware_config.piper.piper_follower import PiperFollower

Payload = Tuple[np.ndarray, float]  # (q_cmd_deg(6,), grip_cmd(0~1))

def _piper_send_loop(
    *,
    robot: PiperFollower,
    latest_cmd: Dict[str, Any],
    cmd_lock: threading.Lock,
    stop_event: threading.Event,
    hz: float = 50.0,
    warn_every_sec: float = 2.0,
) -> None:
    
    sleep_dt = 1.0 / float(hz)
    last_warn_t = 0.0

    # local cache of the last known command so we can keep sending even if publisher pauses
    cached: Optional[Payload] = None

    while not stop_event.is_set():
        with cmd_lock:
            payload = latest_cmd.get("payload", None)

        if payload is not None:
            # Expect (q_cmd_deg, grip_cmd) where q_cmd_deg is array-like and grip_cmd is float
            try:
                q_cmd_deg, grip_cmd = payload
                q_cmd_deg = np.asarray(q_cmd_deg, dtype=np.float64).reshape(6,)
                grip_cmd = float(grip_cmd)
                cached = (q_cmd_deg, grip_cmd)
            except Exception as e:
                # Bad payload format; ignore and keep last cached
                now = time.perf_counter()
                if now - last_warn_t > warn_every_sec:
                    print(f"[PIPER SEND WARN] bad payload ignored: {e}")
                    last_warn_t = now

        if cached is None:
            time.sleep(sleep_dt)
            continue

        # Always send the latest cached command (even if it hasn't changed)
        try:
            q_cmd_deg, grip_cmd_m = cached
            # convert meters (0~0.035) -> normalized (0~1)
            grip_norm = float(np.clip(grip_cmd_m / 0.035, 0.0, 1.0))
            robot.send_action(q_cmd_deg, grip_norm)

        except Exception as e:
            now = time.perf_counter()
            if now - last_warn_t > warn_every_sec:
                print(f"[PIPER SEND WARN] {e}")
                last_warn_t = now

        time.sleep(sleep_dt)


def start_piper_thread(
    *,
    robot: Any,
    latest_cmd: Dict[str, Any],
    cmd_lock: threading.Lock,
    stop_event: threading.Event,
    hz: float = 50.0,
) -> threading.Thread:
    th = threading.Thread(
        target=_piper_send_loop,
        kwargs={
            "robot": robot,
            "latest_cmd": latest_cmd,
            "cmd_lock": cmd_lock,
            "stop_event": stop_event,
            "hz": hz,
        },
        daemon=True,
    )
    th.start()
    return th


def publish_piper_cmd(*, latest_cmd, cmd_lock, q_cmd_deg, grip_cmd_m: float) -> None:
    q_cmd_deg = np.asarray(q_cmd_deg, dtype=np.float64).reshape(6,)
    grip_cmd_m = float(grip_cmd_m)

    with cmd_lock:
        latest_cmd["payload"] = (q_cmd_deg, grip_cmd_m)
        latest_cmd["seq"] = int(latest_cmd.get("seq", 0)) + 1


def clear_piper_cmd(*, latest_cmd: Dict[str, Any], cmd_lock: threading.Lock) -> None:
    with cmd_lock:
        latest_cmd["payload"] = None