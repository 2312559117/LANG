"""Python ports of StableFlow Pro trajectory helpers.

Copyright (c) 2026 StableFlow Pro contributors (MIT).
Source: Sources/ImprovedStabilizer.swift, suppressTrajectoryOutliers and
smoothTransformSeries, commit ab54646964b637222e44274e4e745b29a869e269.
Ported 2026-10-17: CGPoint tuples become NumPy vectors; smoothing is vectorized
but retains the upstream clipped-window mean and outlier thresholds.
The original MIT license is included in third_party/StableFlowPro/LICENSE.
"""
from __future__ import annotations

import numpy as np


def suppress_trajectory_outliers(points: np.ndarray) -> np.ndarray:
    result = points.copy()
    if len(points) <= 4:
        return result
    for index in range(1, len(points) - 1):
        previous, current, following = result[index - 1], points[index], points[index + 1]
        expected = (previous + following) * 0.5
        deviation = float(np.linalg.norm(current - expected))
        neighbor_span = float(np.linalg.norm(following - previous))
        previous_jump = float(np.linalg.norm(current - previous))
        next_jump = float(np.linalg.norm(following - current))
        direction_flip = float(np.dot(current - previous, following - current)) < 0
        threshold = max(0.010, neighbor_span * 1.8 + 0.004)
        isolated = deviation > threshold and previous_jump > threshold * 0.85 and next_jump > threshold * 0.85
        if isolated and (direction_flip or deviation > threshold * 1.5):
            result[index] = expected
    return result


def smooth_transform_series(points: np.ndarray, radius: int) -> np.ndarray:
    if not len(points) or radius <= 0:
        return points.copy()
    indices = np.arange(len(points))
    lower = np.maximum(0, indices - radius)
    upper = np.minimum(len(points), indices + radius + 1)
    cumulative = np.vstack([np.zeros((1, 2)), np.cumsum(points, axis=0)])
    return (cumulative[upper] - cumulative[lower]) / (upper - lower)[:, None]
