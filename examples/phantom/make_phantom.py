"""Generate a synthetic ultrasound frame set with known reference poses.

The phantom is an analytic three-dimensional medium: skin and fascial layers, a
spherical target and a fixed speckle texture. Frames are sampled exactly as
``model.render`` samples them, so the set can be used to check the whole
train -> render -> evaluate cycle without any clinical data.
"""
import json
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 128, 96
SW, SH = 0.25, 0.5
NEAR, FAR = 0.0, 48.0
FRAMES = 16
SPACING = 1.0
GROUP_EDGES = (6, 11)
# File order deliberately differs from scan order: the frames of the last group
# start with its anchor, which sits at the far end of the sweep.
ORDER = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 11, 12, 13, 14)

ROOT = Path(__file__).resolve().parent


def texture(x, y, z):
    acc = np.zeros_like(x)
    for fx, fy, fz, phase, amp in (
        (2.1, 3.3, 1.7, 0.0, 1.0),
        (3.7, 1.9, 2.9, 1.1, 0.8),
        (5.3, 4.1, 3.1, 2.3, 0.6),
        (1.3, 5.7, 4.3, 0.7, 0.5),
        (7.1, 2.7, 5.9, 3.1, 0.4),
    ):
        acc += amp * np.sin(fx * x + fy * y + fz * z + phase)
    return acc / 3.4


def medium(x, y, z):
    value = 0.16
    value += 0.45 * np.exp(-(((y - 3.0) / 0.9) ** 2))
    value += 0.30 * np.exp(-(((y - 11.0) / 0.8) ** 2))
    value += 0.08 * np.exp(-(((y - 26.0) / 6.0) ** 2))
    radius = np.sqrt((x - 2.0) ** 2 + (y - 30.0) ** 2 + (z - 7.5) ** 2)
    value += 0.10 * (radius < 6.0)
    value += 0.32 * np.exp(-(((radius - 6.0) / 0.7) ** 2))
    value += 0.11 * texture(x, y, z)
    value *= np.exp(-0.010 * y)
    value *= 1.0 + 0.35 * y / FAR
    return value


def render_frame(z_offset):
    xs = (np.arange(-WIDTH / 2, WIDTH / 2) * SW)[:, None]
    depths = (NEAR * (1 - np.linspace(0.0, 1.0, HEIGHT)) + FAR * np.linspace(0.0, 1.0, HEIGHT))[None, :]
    x = np.broadcast_to(xs, (WIDTH, HEIGHT))
    y = np.broadcast_to(depths, (WIDTH, HEIGHT))
    z = np.full((WIDTH, HEIGHT), z_offset)
    frame = medium(x, y, z)
    return np.clip(frame, 0.0, 1.0).T


def main():
    images = ROOT / "data" / "images"
    images.mkdir(parents=True, exist_ok=True)
    poses = np.repeat(np.eye(4)[None], FRAMES, axis=0)
    positions = np.array(ORDER, dtype=np.float64) * SPACING
    for index in range(FRAMES):
        frame = render_frame(positions[index])
        poses[index, :3, 3] = (0.0, 0.0, positions[index])
        cv2.imwrite(str(images / f"{index}.png"), np.rint(frame * 255).astype(np.uint8))

    labels = np.zeros(FRAMES, dtype=np.int64)
    labels[GROUP_EDGES[0]:GROUP_EDGES[1]] = 1
    labels[GROUP_EDGES[1]:] = 2

    distance = positions[:, None] - positions[None, :]
    similarity = np.exp(-((distance / 5.0) ** 2))

    np.save(ROOT / "data" / "poses.npy", poses[:, :3, :])
    np.save(ROOT / "data" / "class.npy", labels)
    np.save(ROOT / "data" / "similarity.npy", similarity)

    config = {
        "near": NEAR,
        "far": FAR,
        "sw": SW,
        "sh": SH,
        "seed": 3407,
        "steps_init": 100,
        "steps_stage": 100,
        "partial_views": 4,
        "global_every": 8,
        "lr_field": 0.001,
        "lr_pose": 0.0001,
        "field_step": 20,
        "pose_step": 100,
        "field_decay": 0.9954,
        "pose_decay": 0.9,
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {FRAMES} frames to {images}")
    print(f"Scan range: {(FRAMES - 1) * SPACING:g} mm")


if __name__ == "__main__":
    main()
