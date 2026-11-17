from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np

from .media import check_cancel, run_ffmpeg


def demo_frame(index: int, width=960, height=540, fps=30):
    """Deterministic synthetic shot; returns exact ground-truth ROI for tests."""
    scale = width / 960
    yy, xx = np.mgrid[:height, :width]
    t = index / fps
    bg = 18 + 16 * (xx / width) + 8 * np.cos(yy / height * np.pi)
    frame = np.stack([bg + 6, bg + 3, bg], axis=-1).clip(0, 255).astype(np.uint8)
    # Abstract studio panel and independently moving reflections.
    for lane in range(8):
        left = int((lane * 170 + 40 * t) * scale) % (width + 180) - 90
        cv2.line(frame, (left, 0), (left - int(180 * scale), height), (45, 43, 40), max(1, int(2 * scale)))
    cv2.putText(frame, "LOCKFRAME  /  SYNTHETIC TEST SHOT", (int(32 * scale), int(40 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (120, 126, 133), 1, cv2.LINE_AA)
    cx = width * 0.5 + scale * (115 * np.sin(t * 1.35) + 10 * np.sin(index * 0.67))
    cy = height * 0.5 + scale * (55 * np.sin(t * 1.65) + 7 * np.cos(index * 0.83))
    size = max(40, round(142 * scale))
    left, top = round(cx - size / 2), round(cy - size / 2)
    patch = np.full((size, size, 3), (42, 43, 45), np.uint8)
    rng = np.random.default_rng(20261017)
    noise = rng.integers(0, 20, (size, size, 1), dtype=np.uint8)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    center = size // 2
    cv2.circle(patch, (center, center), round(size * 0.40), (181, 184, 188), max(1, size // 35), cv2.LINE_AA)
    cv2.circle(patch, (center, center), round(size * 0.33), (78, 85, 90), max(1, size // 60), cv2.LINE_AA)
    points = np.array([[center, round(size * 0.22)], [round(size * 0.26), round(size * 0.73)],
                       [round(size * 0.75), round(size * 0.73)]], np.int32)
    cv2.polylines(patch, [points], True, (124, 178, 226), max(2, size // 26), cv2.LINE_AA)
    cv2.putText(patch, "LF", (round(size * 0.36), round(size * 0.61)), cv2.FONT_HERSHEY_SIMPLEX,
                size / 180, (235, 239, 242), max(1, size // 90), cv2.LINE_AA)
    frame[top:top + size, left:left + size] = patch
    cv2.putText(frame, f"FRAME {index + 1:04d}", (int(32 * scale), height - int(28 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (130, 136, 144), 1, cv2.LINE_AA)
    return frame, (left / width, top / height, size / width, size / height)


def create_demo(path: str, frames: int = 180, width=960, height=540, fps=30,
                cancel: threading.Event | None = None):
    cancel = cancel or threading.Event()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    silent = destination.with_name(destination.stem + ".silent.mp4")
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建演示视频")
    try:
        for i in range(frames):
            check_cancel(cancel)
            frame, _ = demo_frame(i, width, height, fps)
            writer.write(frame)
    finally:
        writer.release()
    try:
        run_ffmpeg([
            "-i", str(silent), "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", str(frames / fps), "-c:v", "libx264", "-crf", "18", "-c:a", "aac",
            "-af", "volume=0.06", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ], cancel)
    finally:
        silent.unlink(missing_ok=True)
    return destination
