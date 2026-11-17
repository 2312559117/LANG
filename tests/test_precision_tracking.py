"""Measured precision-mode improvements and conservative failure behavior.

These synthetic scenes specify a known target transform independently of the
tracker. They are regression evidence, not a claim about every real car shot.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from lockframe.media import Cancelled
from lockframe.precision_tracking import PrecisionFeatureTracker
from lockframe.tracking import FeatureTracker


def detail(size=64, seed=704):
    rng = np.random.default_rng(seed)
    image = rng.integers(35, 150, (size, size), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .5)
    cv2.circle(image, (size // 2, size // 2), size // 3, 230, 2, cv2.LINE_AA)
    cv2.putText(image, "L7", (size // 5, size * 2 // 3), cv2.FONT_HERSHEY_SIMPLEX,
                size / 85, 245, 2, cv2.LINE_AA)
    return image


def place(background, patch, center, angle=0., scale=1.):
    height, width = patch.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += np.asarray(center) - [width / 2, height / 2]
    rendered = cv2.warpAffine(patch, matrix, background.shape[::-1])
    mask = cv2.warpAffine(np.full_like(patch, 255), matrix, background.shape[::-1])
    frame = background.copy()
    frame[mask > 250] = rendered[mask > 250]
    return frame


def center(observation):
    return observation.box[:2] + observation.box[2:] / 2


@pytest.mark.parametrize("case", ["exposure", "rotation_and_jump"])
def test_precision_keeps_difficult_frame_accuracy_after_ms_v2_upgrade(case):
    background = np.full((240, 400), 24, np.uint8)
    patch = detail()
    initial_center = np.array([150., 120.])
    first = place(background, patch, initial_center)
    ordinary = FeatureTracker(first, [118, 88, 64, 64])
    precise = PrecisionFeatureTracker(first, [118, 88, 64, 64])
    counts = [0, 0]
    errors = []
    for index in range(1, 9):
        expected = initial_center + [index * 1.2, index * .35]
        angle, scale = index * .4, 1 + index * .003
        if case == "rotation_and_jump":
            expected += [55, 0]
            angle += 30
            scale += .22
        frame = place(background, patch, expected, angle, scale)
        if case == "exposure":
            frame = np.clip(frame.astype(float) * 1.12 + 40, 0, 255).astype(np.uint8)
        for mode, tracker in enumerate((ordinary, precise)):
            observation = tracker.step(frame)
            if observation.state != "lost":
                counts[mode] += 1
                if mode == 1:
                    errors.append(np.linalg.norm(center(observation) - expected))
    assert counts[1] == 8
    if case == "exposure":
        assert counts[0] == 8
    else:
        assert counts[1] >= counts[0]
    assert max(errors) < .25


def test_precision_preserves_easy_motion_accuracy_and_measured_jumps():
    background = np.full((240, 400), 24, np.uint8)
    patch = detail()
    first = place(background, patch, [170, 120])
    trackers = [cls(first, [138, 88, 64, 64])
                for cls in (FeatureTracker, PrecisionFeatureTracker)]
    for index in range(1, 15):
        expected = np.array([170 + index * 1.2, 120 + index * .35])
        if index >= 9:
            expected += [-21, 9]
        frame = place(background, patch, expected, index * .4, 1 + index * .003)
        for tracker in trackers:
            observation = tracker.step(frame)
            assert observation.state != "lost"
            # A real discontinuity remains a discontinuity; no lag or smoothed
            # prediction is allowed to masquerade as a precise measurement.
            assert np.linalg.norm(center(observation) - expected) < .25


@pytest.mark.parametrize("size", [24, 32, 48, 64])
def test_tiny_detail_under_mild_blur_exposure_and_rotation(size):
    background = np.full((240, 400), 24, np.uint8)
    patch = detail(size)
    initial_center = np.array([150., 120.])
    tracker = PrecisionFeatureTracker(place(background, patch, initial_center),
                                      [*(initial_center - size / 2), size, size])
    for index in range(1, 9):
        expected = initial_center + [index * 1.2, index * .5]
        frame = place(background, patch, expected, index * .7, 1 + index * .004)
        frame = cv2.GaussianBlur(frame, (3, 3), .75)
        frame = np.clip(frame.astype(float) * 1.08 + 32, 0, 255).astype(np.uint8)
        observation = tracker.step(frame)
        assert observation.state != "lost", f"size {size}, frame {index}"
        assert np.linalg.norm(center(observation) - expected) < .35


def test_busy_independently_moving_background_cannot_outvote_selected_center():
    background = np.random.default_rng(102).integers(15, 235, (240, 400), dtype=np.uint8)
    patch = detail(62)
    tracker = PrecisionFeatureTracker(place(background, patch, [140, 115]), [95, 70, 90, 90])
    for index in range(1, 13):
        expected = np.array([140 + index * 1.6, 115 + index * .35])
        frame = place(np.roll(background, -index * 2, axis=1), patch, expected)
        observation = tracker.step(frame)
        assert observation.state != "lost"
        assert np.linalg.norm(center(observation) - expected) < .6


def test_occluded_center_is_lost_then_recovered_only_with_real_reference_evidence():
    background = np.full((240, 400), 24, np.uint8)
    patch = detail()
    tracker = PrecisionFeatureTracker(place(background, patch, [160, 120]), [128, 88, 64, 64])
    hidden = place(background, patch, [164, 120])
    hidden[91:150, 140:187] = 90
    observation = tracker.step(hidden)
    assert observation.state == "lost"
    assert observation.confidence == 0
    np.testing.assert_allclose(center(observation), [160, 120])
    resumed = tracker.step(place(background, patch, [170, 120]))
    assert resumed.state == "recovered"
    assert np.linalg.norm(center(resumed) - [170, 120]) < .25


def test_repeated_identical_details_after_losing_continuity_stay_ambiguous():
    background = np.full((240, 400), 24, np.uint8)
    patch = detail(48)
    tracker = PrecisionFeatureTracker(place(background, patch, [180, 120]), [156, 96, 48, 48])
    assert tracker.step(background).state == "lost"
    duplicates = place(place(background, patch, [144, 120]), patch, [216, 120])
    observation = tracker.step(duplicates)
    assert observation.state == "lost"
    assert observation.confidence == 0
    np.testing.assert_allclose(center(observation), [180, 120])


def test_cancel_before_construction_and_between_expensive_hypotheses(monkeypatch):
    background = np.full((180, 300), 24, np.uint8)
    frame = place(background, detail(), [140, 90])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        PrecisionFeatureTracker(frame, [108, 58, 64, 64], cancel=cancel)
    cancel.clear()
    tracker = PrecisionFeatureTracker(frame, [108, 58, 64, 64], cancel=cancel)
    original = tracker._descriptor_candidates

    def descriptors_then_cancel(*args):
        result = original(*args)
        cancel.set()
        return result

    monkeypatch.setattr(tracker, "_descriptor_candidates", descriptors_then_cancel)
    tracker.age = 7  # The eighth frame performs a full audit, including candidates.
    with pytest.raises(Cancelled):
        tracker.step(place(background, detail(), [144, 90]))
