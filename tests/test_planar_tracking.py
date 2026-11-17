"""Regional evidence and bounded appearance learning, using measured target motion."""
import threading
from dataclasses import asdict

import cv2
import numpy as np
import pytest

from lockframe.models import Settings, read_project, save_project
from lockframe.reference_tracking import ReferencePatch
from lockframe.tracking import FeatureTracker, track_video
from lockframe.precision_tracking import PrecisionFeatureTracker, _normalise


def textures(size=80):
    rng = np.random.default_rng(421)
    values = []
    for _ in range(2):
        image = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (5, 5), .9)
        values.append(np.clip(image / image.std() * 34 + 127, 12, 242).astype(np.uint8))
    return values


def scene(patch, index=0):
    center = np.array([160. + index * .6, 120. + index * .15])
    h, w = patch.shape
    transform = np.array([[1., 0., center[0] - w / 2], [0., 1., center[1] - h / 2]])
    image = np.full((240, 400), 24, np.uint8)
    rendered = cv2.warpAffine(patch, transform, (400, 240))
    mask = cv2.warpAffine(np.full_like(patch, 255), transform, (400, 240))
    image[mask > 250] = rendered[mask > 250]
    return image, center


@pytest.mark.parametrize("shape", ["rectangle", "ellipse", "circle"])
def test_region_correlation_distinguishes_identical_unrelated_and_uniform(shape):
    first, other = textures()
    reference = ReferencePatch(first, [0, 0, 80, 80], shape)
    assert reference.regional_correlation(first) == pytest.approx(1., abs=1e-6)
    assert reference.regional_correlation(other) < .3
    assert reference.regional_correlation(np.full_like(first, 100)) == -1.


def test_ellipse_excludes_pixels_outside_selected_shape():
    first, _ = textures()
    reference = ReferencePatch(first, [0, 0, 80, 80], "ellipse")
    changed = first.copy()
    changed[reference.mask == 0] = 255 - changed[reference.mask == 0]
    assert reference.regional_correlation(changed) == pytest.approx(1., abs=1e-6)


@pytest.mark.parametrize("confidence", [.1, .84, float("nan")])
def test_low_confidence_cannot_mutate_working_or_original_reference(confidence):
    first, other = textures()
    reference = ReferencePatch(first, [0, 0, 80, 80])
    image, points = reference.image.copy(), reference.points.copy()
    assert not reference.adapt(other, reference.matrix, confidence)
    np.testing.assert_array_equal(reference.image, image)
    np.testing.assert_array_equal(reference.original_image, image)
    np.testing.assert_array_equal(reference.points, points)


def test_redundant_or_unrelated_appearance_is_not_learned():
    first, other = textures()
    reference = ReferencePatch(first, [0, 0, 80, 80])
    assert not reference.adapt(first, reference.matrix, .99)
    assert not reference.adapt(other, reference.matrix, .99)
    np.testing.assert_array_equal(reference.image, first)


def test_learning_blends_slowly_and_preserves_the_original_anchor():
    first, other = textures()
    reference = ReferencePatch(first, [0, 0, 80, 80])
    changed = cv2.addWeighted(first, .75, other, .25, 0)
    assert reference.adapt(changed, reference.matrix, .99)
    np.testing.assert_array_equal(reference.original_image, first)
    np.testing.assert_array_equal(reference.image, cv2.addWeighted(first, .8, changed, .2, 0))


def test_no_cumulative_drift_can_bypass_the_original_anchor():
    first, other = textures()
    reference = ReferencePatch(first, [0, 0, 80, 80])
    successes = 0
    for alpha in np.linspace(.05, 1., 160):
        appearance = cv2.addWeighted(first, 1 - alpha, other, alpha, 0)
        changed = reference.adapt(appearance, reference.matrix, .99)
        if changed:
            successes += 1
            assert reference.regional_correlation(appearance, reference=first) >= .55
    assert successes > 3
    assert not reference.adapt(other, reference.matrix, .99)
    np.testing.assert_array_equal(reference.original_image, first)


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
def test_learning_is_disabled_by_default(cls, monkeypatch):
    first, _ = textures()
    image, _ = scene(first)
    tracker = cls(image, [120, 80, 80, 80])
    monkeypatch.setattr(tracker.reference, "adapt", lambda *a, **k: pytest.fail("learning was disabled"))
    for index in range(1, 8):
        image, expected = scene(first, index)
        value = tracker.step(image)
        assert value.state != "lost"
        assert np.linalg.norm(value.box[:2] + value.box[2:] / 2 - expected) < .35


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
def test_gradual_appearance_is_measured_without_feature_or_cache_mismatch(cls):
    first, other = textures()
    image, _ = scene(first)
    tracker = cls(image, [120, 80, 80, 80], adaptive_reference=True)
    fixed = cls(image, [120, 80, 80, 80])
    learned_valid = fixed_valid = 0
    errors = []
    for index, alpha in enumerate(np.linspace(0, .59, 66)[1:], 1):
        appearance = cv2.addWeighted(first, 1 - alpha, other, alpha, 0)
        image, expected = scene(appearance, index)
        value = tracker.step(image)
        fixed_value = fixed.step(image)
        fixed_valid += fixed_value.state != "lost"
        if value.state != "lost":
            learned_valid += 1
            errors.append(np.linalg.norm(value.box[:2] + value.box[2:] / 2 - expected))
        assert len(tracker.point_ids) == (len(tracker.points) if tracker.points is not None else len(tracker.point_ids))
        assert all(i < len(tracker.reference.points) for i in tracker.point_ids)
        if cls is PrecisionFeatureTracker:
            np.testing.assert_array_equal(tracker.normalized_reference, _normalise(tracker.reference.image))
    assert tracker.adaptation_updates >= 3
    # The persistent adjacent-frame kernel can now carry the fixed-reference
    # tracker through this entire gradual blend.  Adaptation remains useful,
    # but it no longer has to beat a deliberately weaker fixed baseline.
    if cls is FeatureTracker:
        assert fixed_valid == 65
        assert learned_valid >= 60, (learned_valid, fixed_valid)
    else:
        assert learned_valid >= fixed_valid + 3, (learned_valid, fixed_valid)
    assert learned_valid >= 60
    assert max(errors) < .8
    np.testing.assert_array_equal(tracker.reference.original_image, first)


def test_region_gate_rejects_verified_points_when_area_evidence_disagrees(monkeypatch):
    first, _ = textures()
    image, _ = scene(first)
    tracker = FeatureTracker(image, [120, 80, 80, 80], planar_region=True)
    assert tracker._verified(image, tracker.matrix) is not None
    monkeypatch.setattr(tracker.reference, "regional_evidence", lambda *a, **k: -.5)
    assert tracker._verified(image, tracker.matrix) is None


@pytest.mark.parametrize("field,value", [("adaptive_reference", True), ("perspective", True), ("smoothness", 5)])
def test_reuse_is_invalidated_by_tracking_settings(field, value, monkeypatch):
    from lockframe.demo import demo_frame
    from lockframe.media import import_video, resource_root
    cancel = threading.Event()
    info = import_video(str(resource_root() / "assets/demo.mp4"), cancel)
    anchors = {5: demo_frame(5)[1]}
    original = track_video(info, anchors, 0, 9, cancel)
    steps = []
    step = FeatureTracker.step
    def counted(self, gray):
        steps.append(1)
        return step(self, gray)
    monkeypatch.setattr(FeatureTracker, "step", counted)
    result = track_video(info, anchors, 0, 9, cancel, previous=original, **{field: value})
    assert len(steps) == 9
    assert result.performance[field] == value


def test_setting_validation_and_old_project_defaults():
    assert Settings().adaptive_reference is False
    assert Settings(**{"tracking_mode": "precision"}).adaptive_reference is False
    with pytest.raises(ValueError):
        Settings(adaptive_reference="true").validate()


def test_project_roundtrip_preserves_learning_and_distinct_perspective(tmp_path):
    from lockframe.demo import demo_frame
    from lockframe.media import import_video, resource_root
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    settings = Settings(adaptive_reference=True, perspective=False)
    target = tmp_path / "外观学习.lockframe"
    save_project(str(target), info, settings, {5: demo_frame(5)[1]}, 0, 9, None)
    restored = Settings(**read_project(str(target))["settings"])
    assert restored.adaptive_reference and not restored.perspective


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
def test_perspective_with_adaptation_tracks_measured_positions(cls):
    first, _ = textures()
    image, _ = scene(first)
    tracker = cls(image, [120, 80, 80, 80], perspective=True, adaptive_reference=True)
    for index in range(1, 13):
        image, expected = scene(first, index)
        result = tracker.step(image)
        assert result.state != "lost"
        assert np.linalg.norm(result.box[:2] + result.box[2:] / 2 - expected) < .6


def test_gui_persists_learning_and_invalidates_old_tracks(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from lockframe import app as app_module
    from lockframe.demo import demo_frame
    from lockframe.media import import_video, resource_root
    application = QApplication.instance() or QApplication([])
    preference_path = str(tmp_path / "isolated-settings.ini")
    monkeypatch.setattr(app_module, "QSettings", lambda *args: QSettings(preference_path, QSettings.Format.IniFormat))
    window = app_module.MainWindow()
    try:
        info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
        window._import_done(info)
        window.anchors = {5: demo_frame(5)[1]}
        window.track = track_video(info, window.anchors, 0, 9, threading.Event())
        window.adaptive_reference_check.setChecked(True)
        assert window.track is None
        assert window.settings().adaptive_reference is True
        assert window.settings().perspective is False
        assert window.preferences.value("adaptive_reference", type=bool) is True
        window._ready = False
        window._restore_settings(Settings(adaptive_reference=False, perspective=True))
        window._ready = True
        assert not window.settings().adaptive_reference and window.settings().perspective
    finally:
        window.dirty = False
        window.close()
        application.processEvents()
