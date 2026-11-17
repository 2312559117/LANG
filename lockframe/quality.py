"""Measured regional evidence and trajectory review, separate from measured poses."""
import numpy as np
from .stableflow_filters import suppress_trajectory_outliers

STAGES = ("分析", "计算", "检查", "缓存")
BOUNDARIES = (0., .12, .85, .95, 1.)


def phase_progress(callback, stage, fraction, detail=""):
    if callback:
        index = STAGES.index(stage)
        fraction = max(0., min(1., float(fraction)))
        overall = BOUNDARIES[index] + (BOUNDARIES[index + 1] - BOUNDARIES[index]) * fraction
        callback(overall, f"[{stage}:{fraction:.3f}] {detail}")


def parse_phase(text):
    if not text.startswith("[") or "]" not in text:
        return None
    prefix, detail = text[1:].split("]", 1)
    try:
        stage, fraction = prefix.split(":")
        if stage not in STAGES:
            return None
        return stage, max(0., min(1., float(fraction))), detail.strip()
    except (ValueError, TypeError):
        return None


def confidence_grid(reference, gray, matrix, lost=False):
    """4x4 masked NCC, with -1 for cells lacking enough texture/support."""
    aligned = reference.warp(gray, matrix)
    output = np.full((4, 4), -1., np.float32)
    ys = np.linspace(0, reference.height, 5).astype(int)
    xs = np.linspace(0, reference.width, 5).astype(int)
    for row in range(4):
        for col in range(4):
            region = np.s_[ys[row]:ys[row + 1], xs[col]:xs[col + 1]]
            mask = reference.mask[region] > 0
            if mask.sum() < 12:
                continue
            a = reference.image[region][mask].astype(np.float32)
            b = aligned[region][mask].astype(np.float32)
            a -= a.mean()
            b -= b.mean()
            energy = float(np.linalg.norm(a) * np.linalg.norm(b))
            if energy < 25:
                continue
            output[row, col] = 0. if lost else np.clip(float(a @ b / energy), 0., 1.)
    return output


def review_track(track):
    """Flag questionable evidence without replacing true subject movement."""
    low = [track.start + i for i, (state, value) in enumerate(zip(track.states, track.confidence))
           if state not in {"lost", "manual"} and value < .70]
    clean = suppress_trajectory_outliers(track.positions)
    spikes = [track.start + int(i) for i in np.flatnonzero(np.linalg.norm(clean - track.positions, axis=1) > 1e-6)
              if track.states[i] not in {"lost", "manual"}]
    return {"low_confidence": low, "motion_review": spikes, "lost_count": len(track.lost_frames),
            "raw_positions_preserved": True}


def render_filtered_positions(points):
    """Existing render-only jitter suppression; never used as recovered tracking."""
    return suppress_trajectory_outliers(points)
