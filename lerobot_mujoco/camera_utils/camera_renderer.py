from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import mujoco
import numpy as np
import mujoco

# Mujoco에 overlay할 위치 지정
@dataclass(frozen=True)
class OverlaySlot:
    width: int
    height: int
    location: str  # "top_right" | "bottom_right" | "top_left" | "bottom_left"
    margin: int = 8


def _compute_rect(full: mujoco.MjrRect, spec: OverlaySlot) -> mujoco.MjrRect:
    L, B = int(full.left), int(full.bottom)
    W, H = int(full.width), int(full.height)

    w, h = int(spec.width), int(spec.height)
    m = int(spec.margin)

    if w > W or h > H:
        w = min(w, W - 2 * m)
        h = min(h, H - 2 * m)
        if w <= 0 or h <= 0:
            return mujoco.MjrRect(0, 0, 0, 0)

    if spec.location == "top_right":
        x, y = L + (W - w - m), B + (H - h - m)
    elif spec.location == "bottom_right":
        x, y = L + (W - w - m), B + m
    elif spec.location == "top_left":
        x, y = L + m, B + (H - h - m)
    elif spec.location == "bottom_left":
        x, y = L + m, B + m
    else:
        raise ValueError(f"Unknown loc: {spec.location}")

    return mujoco.MjrRect(int(x), int(y), int(w), int(h))

# image resize
def _nearest_resize_rgb(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3), got {img.shape}")
    in_h, in_w = img.shape[:2]
    if (in_h, in_w) == (out_h, out_w):
        return np.ascontiguousarray(img)

    ys = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
    xs = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
    out = img[ys[:, None], xs[None, :], :]
    return np.ascontiguousarray(out)

# overlay
def draw_rgb_panels_on_viewer(
    handle,
    agent_rgb: Optional[np.ndarray],
    wrist_rgb: Optional[np.ndarray],
    *,
    agent_panel: OverlaySlot = OverlaySlot(width=320, height=240, location="top_right", margin=8),
    wrist_panel: OverlaySlot = OverlaySlot(width=320, height=240, location="bottom_right", margin=8),
) -> None:
    if agent_rgb is None and wrist_rgb is None:
        with handle.lock():
            handle.clear_images()
        return

    full = handle.viewport
    if full is None:
        return

    items: List[Tuple[mujoco.MjrRect, np.ndarray]] = []

    if agent_rgb is not None:
        rect = _compute_rect(full, agent_panel)
        if rect.width > 0 and rect.height > 0:
            img = _nearest_resize_rgb(agent_rgb, rect.height, rect.width)
            items.append((rect, img))

    if wrist_rgb is not None:
        rect = _compute_rect(full, wrist_panel)
        if rect.width > 0 and rect.height > 0:
            img = _nearest_resize_rgb(wrist_rgb, rect.height, rect.width)
            items.append((rect, img))

    with handle.lock():
        handle.set_images(items)