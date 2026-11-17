"""Subject centering and trajectory smoothing, informed by StableFlow Pro.

The Apple-only pipeline is replaced with OpenCV/FFmpeg. Scale is solved from
frame geometry instead of upstream lockOn's image-content/black-pixel heuristic.
"""
from __future__ import annotations

import cv2
import numpy as np

from .models import RenderPlan, Settings, TrackResult, VideoInfo
from .stableflow_filters import smooth_transform_series
from .quality import render_filtered_positions


def output_size(info: VideoInfo, settings: Settings) -> tuple[int, int]:
    ratio = info.width / info.height
    if settings.aspect != "original":
        w, h = map(float, settings.aspect.split(":"))
        ratio = w / h
    # Do not upscale: output resolution follows the usable crop of the source.
    if ratio >= info.width / info.height:
        width, height = info.width, info.width / ratio
    else:
        width, height = info.height * ratio, info.height
    limit = settings.resolution or max(width, height)
    factor = min(1.0, limit / max(width, height))
    return max(2, int(width * factor) // 2 * 2), max(2, int(height * factor) // 2 * 2)


def smooth_trajectory(points: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    if len(points) < 3:
        return points.copy()
    radius = min(len(points) - 1, max(1, round(fps * seconds)))
    return smooth_transform_series(points, radius)


def build_plan(info: VideoInfo, track: TrackResult, settings: Settings) -> RenderPlan:
    if not track.performance.get("complete", True):
        raise ValueError("跟踪已暂停，不能将未测量的轨迹用于成片")
    settings.validate()
    track.validate(info.frames)
    width, height = output_size(info, settings)
    source_size = np.array([info.width, info.height], dtype=np.float64)
    dest_size = np.array([width, height], dtype=np.float64)
    points = track.positions * source_size
    strength = settings.strength
    if settings.mode == "lock":
        a = np.broadcast_to((1 - strength) * dest_size / 2 + strength * dest_size *
                            np.array([settings.target_x, settings.target_y]), points.shape).copy()
        b = (1 - strength) * source_size / 2 + strength * points
    else:
        sections = track.performance.get('tracking_sections', [])
        if sections:
            from .section_workflow import active_sections
            smoothed = points.copy()
            for section in active_sections(sections, track.start, track.end):
                first, last = section['start']-track.start, section['end']-track.start+1
                clean = render_filtered_positions(track.positions[first:last]) * source_size
                smoothed[first:last] = smooth_trajectory(clean, info.fps, settings.smoothing)
        else:
            clean = render_filtered_positions(track.positions) * source_size
            smoothed = smooth_trajectory(clean, info.fps, settings.smoothing)
        a = np.broadcast_to(dest_size / 2, points.shape).copy()
        b = source_size / 2 - strength * (smoothed - points)
    base_scale = max(width / info.width, height / info.height)
    # t = a - scale*b; source coverage requires dest-size*scale <= t <= 0.
    valid_geometry = (b > 1e-6).all() and (b < source_size - 1e-6).all()
    if valid_geometry:
        required_scale = max(base_scale, float(np.max(a / b)),
                             float(np.max((dest_size - a) / (source_size - b))))
    else:
        required_scale = base_scale * 100
    scale = min(required_scale * 1.001, base_scale * settings.max_zoom)
    scale = max(base_scale, scale)
    desired_translation = a - scale * b
    minimum_translation = dest_size - source_size * scale
    translation = np.clip(desired_translation, minimum_translation, 0)
    limited = int(np.count_nonzero(np.max(np.abs(translation - desired_translation), axis=1) > 0.5))
    matrices = np.zeros((track.count, 2, 3), dtype=np.float64)
    matrices[:, 0, 0] = scale
    matrices[:, 1, 1] = scale
    matrices[:, :, 2] = translation
    return RenderPlan(width, height, matrices, scale / base_scale,
                      required_scale / base_scale, limited)


def preview_frame(frame: np.ndarray, info: VideoInfo, plan: RenderPlan, index: int,
                  max_side: int | None = 1280) -> np.ndarray:
    factor = min(1.0, max_side / max(plan.width, plan.height)) if max_side else 1.0
    width, height = max(2, round(plan.width * factor)), max(2, round(plan.height * factor))
    sx, sy = width / plan.width, height / plan.height
    matrix = plan.matrices[index].copy()
    matrix[:, 0] *= info.width / frame.shape[1]
    matrix[:, 1] *= info.height / frame.shape[0]
    matrix[0] *= sx
    matrix[1] *= sy
    return cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
