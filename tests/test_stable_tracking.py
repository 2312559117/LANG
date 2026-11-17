"""TrackStable robust motion-estimation contracts.

These tests use known synthetic correspondences. They verify the replacement
kernel itself and do not claim accuracy on every real shot.
"""
from __future__ import annotations

import cv2
import numpy as np

from lockframe.reference_tracking import (
    adaptive_fb_limit,
    evaluate_motion_quality,
    fit_similarity,
    guard_similarity_scale,
    transform_points,
)
from lockframe.tracking import FeatureTracker


def _textured_frame():
    rng = np.random.default_rng(2401)
    patch = rng.integers(25, 225, (72, 72), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), .45)
    cv2.circle(patch, (36, 36), 22, 245, 2, cv2.LINE_AA)
    frame = np.full((180, 260), 18, np.uint8)
    frame[54:126, 84:156] = patch
    return frame


def test_adaptive_fb_threshold_keeps_coherent_blur_but_caps_outlier():
    errors = np.array([1.45, 1.55, 1.62, 1.68, 1.74, 1.81, 24.0])
    limit = adaptive_fb_limit(errors, base=1.0, absolute_cap=6.0, multiplier=3.0)
    assert 1.81 <= limit < 3.0
    assert errors[-1] > limit


def test_two_stage_similarity_rejects_reflection_like_outliers():
    rng = np.random.default_rng(83)
    source = rng.uniform([0, 0], [160, 90], (80, 2)).astype(np.float32)
    angle, scale = np.deg2rad(7.5), 1.035
    linear = scale * np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
    expected = np.column_stack([linear, [13.5, -6.25]])
    destination = transform_points(source, expected)
    destination += rng.normal(0, .16, destination.shape)
    bad = rng.choice(len(source), 24, replace=False)
    destination[bad] = rng.uniform([-40, -35], [230, 150], (len(bad), 2))

    matrix, inliers = fit_similarity(source, destination)
    clean = np.ones(len(source), dtype=bool)
    clean[bad] = False
    error = np.linalg.norm(transform_points(source[clean], matrix) -
                           transform_points(source[clean], expected), axis=1)
    assert np.median(error) < .2
    assert np.count_nonzero(inliers[bad]) <= 1
    assert np.count_nonzero(inliers[clean]) >= 52


def test_quality_gate_requires_distributed_support():
    distributed = np.array([[5, 5], [45, 5], [85, 5], [5, 45], [45, 45],
                            [85, 45], [5, 75], [45, 75], [85, 75]], np.float32)
    matrix = np.array([[1., 0., 4.], [0., 1., -3.]])
    destination = transform_points(distributed, matrix)
    good = evaluate_motion_quality(distributed, destination, matrix,
                                   np.ones(len(distributed), bool), (90, 80))
    assert good.quality_ok
    assert good.confidence > .9

    line = distributed.copy()
    line[:, 1] = 40
    line_destination = transform_points(line, matrix)
    degenerate = evaluate_motion_quality(line, line_destination, matrix,
                                         np.ones(len(line), bool), (90, 80))
    assert not degenerate.quality_ok
    assert degenerate.point_spread == 0


def test_scale_guard_freezes_untrusted_zoom_without_moving_center():
    center = np.array([40., 30.])
    previous = np.array([[1., 0., 80.], [0., 1., 50.]])
    candidate = np.array([[1.22, 0., 72.], [0., 1.22, 43.]])
    quality = evaluate_motion_quality(
        np.array([[0, 0], [80, 0], [0, 60], [80, 60]], np.float32),
        np.array([[72, 43], [169.6, 43], [72, 116.2], [169.6, 116.2]], np.float32),
        candidate, np.ones(4, bool), (80, 60))
    guarded = guard_similarity_scale(candidate, previous, center, quality)
    assert np.isclose(np.hypot(guarded[0, 0], guarded[1, 0]), 1.)
    np.testing.assert_allclose(transform_points([center], guarded),
                               transform_points([center], candidate), atol=1e-8)


def test_verified_pose_replenishes_fixed_reference_feature_ids():
    frame = _textured_frame()
    tracker = FeatureTracker(frame, [84, 54, 72, 72])
    assert len(tracker.reference.points) >= 12
    surviving = np.arange(4)
    tracker._bind_points(surviving, tracker.matrix, .9)
    expected = min(len(tracker.reference.points),
                   max(8, int(np.ceil(len(tracker.reference.points) * .65))))
    assert len(tracker.point_ids) == expected
    assert tracker.replenishment_events == 1
    assert set(surviving) <= set(tracker.point_ids)

