from __future__ import annotations

import threading
from queue import Empty, Queue


def start_dataset_writer_thread(*, dataset, write_queue: Queue, stop_event: threading.Event):
    def _worker():
        while not stop_event.is_set() or not write_queue.empty():
            try:
                item = write_queue.get(timeout=0.1)
            except Empty:
                continue

            if item is None:
                write_queue.task_done()
                break

            try:
                dataset.add_frame(item)
            except Exception as e:
                print(f"[WRITE][WARN] add_frame failed: {type(e).__name__}: {e}")
            finally:
                write_queue.task_done()

    th = threading.Thread(target=_worker, daemon=True, name="dataset_writer_thread")
    th.start()
    return th