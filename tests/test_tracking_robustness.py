"""Measured target locations, independent motion, and honest failure handling."""
import cv2
import numpy as np
import pytest

from lockframe.tracking import FeatureTracker


def _detail(size=64, seed=704):
    rng = np.random.default_rng(seed)
    image = rng.integers(35, 150, (size, size), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .5)
    cv2.circle(image, (size // 2, size // 2), size // 3, 230, 2, cv2.LINE_AA)
    cv2.putText(image, "L7", (size // 5, size * 2 // 3), cv2.FONT_HERSHEY_SIMPLEX,
                size / 85, 245, 2, cv2.LINE_AA)
    return image


def _place(background, patch, center, angle=0., scale=1.):
    size = patch.shape[0]
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, scale)
    matrix[:, 2] += np.asarray(center) - size / 2
    frame = background.copy()
    rendered = cv2.warpAffine(patch, matrix, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpAffine(np.full_like(patch, 255), matrix,
                          (frame.shape[1], frame.shape[0]))
    frame[mask > 250] = rendered[mask > 250]
    return frame


def _center(observation):
    return observation.box[:2] + observation.box[2:] / 2


def test_center_detail_wins_over_independently_moving_background():
    rng = np.random.default_rng(102)
    background = rng.integers(15, 235, (240, 400), dtype=np.uint8)
    patch = _detail(62)
    center = np.array([140., 115.])
    # The selection includes a busy ring of background around the actual logo.
    first = _place(background, patch, center)
    tracker = FeatureTracker(first, [center[0] - 45, center[1] - 45, 90, 90])
    errors = []
    for index in range(1, 36):
        target = center + [index * 1.6, index * .35]
        moving_background = np.roll(background, -index * 2, axis=1)
        observation = tracker.step(_place(moving_background, patch, target))
        assert observation.state != "lost", f"lost at {index}"
        errors.append(np.linalg.norm(_center(observation) - target))
    assert max(errors) < 1.5


def test_ms_v2_center_motion_handles_rotation_and_scale_without_drifting():
    rng = np.random.default_rng(29)
    background = rng.integers(15, 45, (260, 440), dtype=np.uint8)
    patch = _detail(72)
    center = np.array([145., 125.])
    tracker = FeatureTracker(_place(background, patch, center), [109, 89, 72, 72])
    errors = []
    for index in range(1, 61):
        target = center + [index * 1.7, np.sin(index / 8) * 12]
        observation = tracker.step(_place(background, patch, target,
                                           angle=index * .7, scale=1 + index * .004))
        assert observation.state != "lost", f"lost at {index}"
        errors.append(np.linalg.norm(_center(observation) - target))
        # Recovered MS v2 transforms the ROI centre while retaining the user's
        # selected dimensions during continuous flow.
        assert np.allclose(observation.box[2:], [72, 72])
    assert max(errors) < .2


def test_duplicate_reference_is_ambiguous_after_occlusion():
    background = np.full((240, 360), 24, np.uint8)
    patch = _detail(48)
    first = _place(background, patch, [180, 120])
    tracker = FeatureTracker(first, [156, 96, 48, 48])
    assert tracker.step(background).state == "lost"
    duplicate = _place(_place(background, patch, [144, 120]), patch, [216, 120])
    observation = tracker.step(duplicate)
    assert observation.state == "lost"
    assert observation.confidence == 0
    # Do not replace the target with either repeated grille-like detail.
    np.testing.assert_allclose(_center(observation), [180, 120])


def test_heavy_center_occlusion_is_rejected_then_unique_detail_is_recovered():
    background = np.full((240, 400), 24, np.uint8)
    patch = _detail(64)
    first = _place(background, patch, [160, 120])
    tracker = FeatureTracker(first, [128, 88, 64, 64])
    hidden = _place(background, patch, [164, 120])
    hidden[91:150, 140:187] = 90
    observation = tracker.step(hidden)
    assert observation.state == "lost"
    assert observation.confidence == 0
    resumed = tracker.step(_place(background, patch, [170, 120]))
    assert resumed.state == "recovered"
    # v2 template reacquisition is integer-positioned at this analysis size.
    assert np.linalg.norm(_center(resumed) - [170, 120]) <= np.sqrt(2)


@pytest.mark.parametrize("jump", [(-21, 9), (25, -14)])
def test_real_sudden_movement_is_measured_without_trajectory_smoothing(jump):
    background = np.full((240, 400), 24, np.uint8)
    patch = _detail(64)
    tracker = FeatureTracker(_place(background, patch, [180, 120]), [148, 88, 64, 64])
    target = np.array([180, 120]) + jump
    observation = tracker.step(_place(background, patch, target))
    assert observation.state != "lost"
    assert np.linalg.norm(_center(observation) - target) < 1


def test_partial_occlusion_never_receives_full_match_confidence():
    background = np.full((240, 400), 24, np.uint8)
    patch = _detail(72)
    first = _place(background, patch, [170, 120])
    tracker = FeatureTracker(first, [134, 84, 72, 72])
    clean = tracker.step(_place(background, patch, [172, 120]))
    occluded = _place(background, patch, [174, 120])
    occluded[84:156, 190:210] = 24
    uncertain = tracker.step(occluded)
    assert uncertain.confidence < clean.confidence - .08
    if uncertain.state != "lost":
        assert np.linalg.norm(_center(uncertain) - [174, 120]) < 1.5


@pytest.mark.parametrize("size", [24, 32, 48, 64])
def test_small_clear_detail_and_moderate_exposure_change(size):
    background = np.full((180, 300), 24, np.uint8)
    patch = _detail(size)
    center = np.array([110., 80.])
    tracker = FeatureTracker(_place(background, patch, center),
                              [110 - size / 2, 80 - size / 2, size, size])
    for index in range(1, 16):
        target = center + [index * 1.2, index * .5]
        frame = _place(background, patch, target, angle=index * .35, scale=1 + index * .003)
        # Normalized reference matching should tolerate a real exposure ramp.
        frame = np.clip(frame.astype(float) * (1 + index * .008) + index * .4, 0, 255).astype(np.uint8)
        observation = tracker.step(frame)
        assert observation.state != "lost", f"size {size}, frame {index}"
        assert np.linalg.norm(_center(observation) - target) < 1


def test_predicted_pose_is_verified_when_flow_cannot_run():
    background = np.full((240, 400), 24, np.uint8)
    patch = _detail(64)
    center = np.array([160., 120.])
    tracker = FeatureTracker(_place(background, patch, center), [128, 88, 64, 64])
    jump = np.array([9., 4.])
    tracker.velocity = jump
    verified = tracker._verify_predicted(_place(background, patch, center + jump))
    assert verified is not None
    _, box, _, _ = verified
    assert np.linalg.norm((box[:2] + box[2:] / 2) - (center + jump)) < 1.5


def test_flow_retries_with_velocity_seed(monkeypatch):
    background = np.full((200, 320), 24, np.uint8)
    patch = _detail(48)
    tracker = FeatureTracker(_place(background, patch, [120., 100.]), [96, 76, 48, 48])
    calls = []

    def fake_once(gray, initial):
        calls.append(initial)
        if initial is None:
            return None
        return tracker.matrix.copy(), tracker.box.copy(), 0.8, \
            np.arange(len(tracker.reference.points))

    monkeypatch.setattr(tracker, "_flow_once", fake_once)
    tracker.velocity = np.array([3., 0.])
    result = tracker._flow(np.zeros_like(background))
    assert result is not None
    assert len(calls) == 2 and calls[0] is None and calls[1] is not None


def test_mild_blur_is_tolerated_by_normal_tracking():
    background = np.full((240, 400), 24, np.uint8)
    patch = _detail(64)
    center = np.array([150., 120.])
    tracker = FeatureTracker(_place(background, patch, center), [118, 88, 64, 64])
    for index in range(1, 16):
        target = center + [index * 1.3, index * .4]
        frame = cv2.GaussianBlur(_place(background, patch, target), (0, 0), .9)
        observation = tracker.step(frame)
        assert observation.state != "lost", f"lost at {index}"
        assert np.linalg.norm(_center(observation) - target) < 1.5
