import threading
import time
from typing import Dict
from quest3.quest3_teleop import Quest3Teleop

def _teleop_loop(*, teleop: Quest3Teleop, latest: Dict, stop_event: threading.Event, hz: float = 100.0) -> None:
    sleep_dt = 1.0 / float(hz)
    while not stop_event.is_set():
        try:
            latest["frame"] = teleop.read()
        except Exception:
            # keep last frame if read fails
            pass
        time.sleep(sleep_dt)

def start_teleop_thread(*, teleop: Quest3Teleop, latest: Dict, stop_event: threading.Event, hz: float = 100.0) -> threading.Thread:
    th = threading.Thread(
        target=_teleop_loop,
        kwargs={"teleop": teleop, "latest": latest, "stop_event": stop_event, "hz": hz},
        daemon=True,
    )
    th.start()
    return th