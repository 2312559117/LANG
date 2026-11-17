"""Full surface geometry, masked pixel fitting, and the default motion kernel."""
import threading

import cv2
import numpy as np
import pytest

from lockframe.models import Settings, read_project, save_project
from lockframe.planar import UNIT_CORNERS, as_homography, project, valid_surface, template_pose
from lockframe.reference_tracking import ReferencePatch
from lockframe.tracking import FeatureTracker, track_video
from lockframe.precision_tracking import PrecisionFeatureTracker


def projective_scene(index, shape="rectangle", flicker=False):
    rng = np.random.default_rng(702)
    patch = rng.integers(25, 230, (100, 120), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), .55)
    cv2.putText(patch, "PLANE", (6, 55), cv2.FONT_HERSHEY_SIMPLEX, .65, 245, 2)
    mask = np.full_like(patch, 255)
    if shape != "rectangle":
        cv2.ellipse(mask := np.zeros_like(patch), (60, 50), (59, 49), 0, 0, 360, 255, -1)
        if flicker:
            patch[mask == 0] = np.random.default_rng(index).integers(0, 255, np.count_nonzero(mask == 0), dtype=np.uint8)
    t = index / 24
    corners = np.array([[110 + 20*t, 70 + 5*t], [230 - 8*t, 70 + 14*t],
                        [230 + 12*t, 170 - 7*t], [110 - 10*t, 170 + 9*t]], np.float32)
    matrix = cv2.getPerspectiveTransform((UNIT_CORNERS * [120, 100]).astype(np.float32), corners)
    image = np.full((260, 400), 20, np.uint8)
    rendered = cv2.warpPerspective(patch, matrix, (400, 260))
    support = cv2.warpPerspective(np.full_like(patch, 255), matrix, (400, 260))
    image[support > 250] = rendered[support > 250]
    return image, matrix, corners


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
@pytest.mark.parametrize("shape", ["rectangle", "ellipse"])
def test_real_projective_motion_keeps_four_corners_and_target_center(cls, shape):
    first, _, _ = projective_scene(0, shape)
    tracker = cls(first, [110, 70, 120, 100], perspective=True, planar_region=True,
                  adaptive_reference=True, shape=shape)
    errors = []
    for index in range(1, 25):
        image, truth, corners = projective_scene(index, shape)
        result = tracker.step(image)
        assert result.state != "lost", (cls.__name__, shape, index)
        measured = project(UNIT_CORNERS, result.homography)
        errors.extend(np.linalg.norm(measured - corners, axis=1))
        assert np.linalg.norm(result.center - project([[60, 50]], truth)[0]) < .8
        assert abs(result.homography[2, 0]) + abs(result.homography[2, 1]) > 1e-4
    assert max(errors) < 1.5, max(errors)


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
def test_shape_mask_excludes_flickering_outside_pixels(cls):
    first, _, _ = projective_scene(0, "ellipse", True)
    tracker = cls(first, [110, 70, 120, 100], perspective=True, planar_region=True, shape="ellipse")
    for index in range(1, 12):
        frame, _, corners = projective_scene(index, "ellipse", True)
        result = tracker.step(frame)
        assert result.state != "lost"
        assert np.max(np.linalg.norm(project(UNIT_CORNERS, result.homography) - corners, axis=1)) < 1.5


@pytest.mark.parametrize("bad", [np.array([[-1,0,120],[0,1,70],[0,0,1]]),
                                 np.array([[1,0,110],[0,1,70],[-.02,0,1]]),
                                 np.array([[1,0,110],[0,0,70],[0,0,1]])])
def test_reflection_horizon_crossing_and_collapse_are_rejected(bad):
    assert not valid_surface(bad, [120, 100])


def test_recovery_template_keeps_projective_terms():
    _, matrix, _ = projective_scene(20)
    render, low, size, candidate = template_pose(matrix, [120,100], 1.04, 5.)
    assert render.shape == (3,3) and size.min() > 50
    assert abs(candidate[2,0]) + abs(candidate[2,1]) > 1e-5
    np.testing.assert_allclose(project(UNIT_CORNERS * [120,100], render) + low,
                               project(UNIT_CORNERS * [120,100], candidate), atol=1e-10)


def test_planar_gate_and_dense_fitting_do_not_run_when_disabled(monkeypatch):
    image, _, _ = projective_scene(0)
    tracker = FeatureTracker(image, [110,70,120,100])
    monkeypatch.setattr(tracker.reference, "regional_evidence", lambda *a,**k: pytest.fail("region was disabled"))
    from lockframe import tracking
    monkeypatch.setattr(tracking, "dense_refine", lambda *a,**k: pytest.fail("dense region was disabled"))
    assert tracker.step(image).state != "lost"


@pytest.mark.parametrize("name", ["FeatureTracker", "PrecisionFeatureTracker"])
def test_default_motion_kernel_measures_subpixel_translation(name):
    """The v0.7.24 default intentionally replaces the v0.6.4 pixel path."""
    cls = FeatureTracker if name == "FeatureTracker" else PrecisionFeatureTracker
    image, _, _ = projective_scene(0)
    current = cls(image, [110, 70, 120, 100])
    for index in range(1, 8):
        moved = cv2.warpAffine(image, np.array([[1.,0.,index*1.25],[0.,1.,index*.25]]), (400,260))
        cv2.setRNGSeed(701)
        actual = current.step(moved)
        assert actual.state != "lost"
        np.testing.assert_allclose(
            actual.box, [110 + index * 1.25, 70 + index * .25, 120, 100],
            atol=.15, rtol=0)
        assert actual.confidence > 0


def test_full_matrices_survive_pipeline_project_roundtrip_and_incremental_reuse(tmp_path):
    from lockframe.demo import demo_frame
    from lockframe.media import import_video, resource_root
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    anchors = {5: demo_frame(5)[1]}
    result = track_video(info, anchors, 0, 12, threading.Event(), planar_region=True, perspective=True)
    assert result.homographies.shape == (13,3,3)
    assert not result.lost_frames
    again = track_video(info, anchors, 0,12,threading.Event(), planar_region=True,perspective=True,previous=result)
    np.testing.assert_array_equal(again.homographies,result.homographies)
    file = tmp_path / "完整单应.lockframe"
    save_project(str(file),info,Settings(planar_region=True,perspective=True),anchors,0,12,result)
    np.testing.assert_array_equal(np.asarray(read_project(str(file))["track"]["homographies"]),result.homographies)


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
def test_projective_recovery_after_occlusion_uses_the_same_surface(cls):
    first, _, _ = projective_scene(0)
    tracker = cls(first, [110,70,120,100], perspective=True,planar_region=True)
    for index in range(1, 14):
        assert tracker.step(projective_scene(index)[0]).state != "lost"
    assert tracker.step(np.full_like(first,20)).state == "lost"
    image, _, corners = projective_scene(15)
    result = tracker.step(image)
    assert result.state == "recovered"
    assert np.max(np.linalg.norm(project(UNIT_CORNERS,result.homography)-corners,axis=1)) < 1.5


def test_projected_mask_overlay_uses_homography_across_zoom():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from lockframe.widgets import PreviewCanvas
    application = QApplication.instance() or QApplication([])
    frame, matrix, corners = projective_scene(20)
    normalized = np.diag([1/400,1/260,1.]) @ matrix @ np.diag([120,100,1.])
    canvas = PreviewCanvas()
    canvas.resize(1000,680)
    canvas.mode = "source"
    canvas.set_frames(cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR),box=(.25,.25,.4,.4),homography=normalized)
    canvas.show()
    application.processEvents()
    for zoom in (1,3):
        canvas.set_zoom(zoom)
        polygon = canvas._surface_polygon()
        values = np.array([[(p.x()-canvas.source_rect.x())/canvas.source_rect.width(),
                            (p.y()-canvas.source_rect.y())/canvas.source_rect.height()] for p in polygon])
        np.testing.assert_allclose(values,corners/[400,260],atol=1e-6)
    canvas.selection_shape="ellipse"
    assert len(canvas._surface_polygon()) == 64
    canvas.close()


def test_planar_preset_and_independent_flags_restore(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from lockframe import app as app_module
    application = QApplication.instance() or QApplication([])
    preferences = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(app_module, "QSettings", lambda *args: preferences)
    window = app_module.MainWindow()
    try:
        assert not window.settings().planar_region
        window._apply_planar_preset()
        settings = window.settings()
        assert settings.planar_region and settings.perspective and settings.adaptive_reference
        assert settings.tracking_mode == "precision"
        assert preferences.value("planar_region", type=bool)
        window._restore_settings(Settings(planar_region=False, perspective=True, adaptive_reference=False))
        settings = window.settings()
        assert settings.perspective and not settings.planar_region and not settings.adaptive_reference
    finally:
        window.dirty = False
        window.close()
