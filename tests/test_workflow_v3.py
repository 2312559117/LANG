"""Real v3 mode/preview/project/export integration, including Qt job cleanup."""
from dataclasses import asdict, replace
import copy
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from lockframe.demo import create_demo, demo_frame
from lockframe.export import export_video
from lockframe.media import check_cancel, import_video, probe
from lockframe.models import Settings, read_project, save_project
from lockframe.stabilization import build_plan
from lockframe.tracking import track_video


@pytest.fixture(scope="module")
def qt_app_v3():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def clip_v3(tmp_path_factory):
    source = tmp_path_factory.mktemp("workflow-v3") / "精准跟踪原片.mp4"
    create_demo(str(source), frames=30, width=640, height=360)
    return import_video(str(source), threading.Event())


@pytest.fixture(scope="module")
def anchors_v3():
    return {frame: demo_frame(frame, 640, 360)[1] for frame in (7, 19)}


@pytest.fixture(scope="module")
def tracks_v3(clip_v3, anchors_v3):
    results = {}
    for mode in ("normal", "precision"):
        results[mode] = track_video(clip_v3, anchors_v3, 3, 26,
                                    threading.Event(), mode=mode)
    return results


def finish_jobs(window, application, timeout=40):
    """Pump queued callbacks, including any preview job started by import finish."""
    deadline = time.monotonic() + timeout
    while window.busy and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.005)
    application.processEvents()
    assert not window.busy, "后台线程未在期限内完成"
    assert window.worker is None


@pytest.fixture
def window_v3(qt_app_v3, clip_v3):
    from lockframe.app import MainWindow
    window = MainWindow()
    window.resize(1560, 960)
    window._import_done(clip_v3)
    window._sync_ui()
    window.show()
    qt_app_v3.processEvents()
    errors = []
    window._error = errors.append
    window._test_errors = errors
    yield window
    if window.worker is not None:
        window.worker.cancel()
        finish_jobs(window, qt_app_v3)
    window.dirty = False
    window.close()
    qt_app_v3.processEvents()


def test_implicit_normal_and_explicit_normal_have_identical_real_trajectories(clip_v3, anchors_v3):
    # Seed OpenCV's RANSAC identically: the API default must select exactly the
    # original normal-mode computation, not a weakened or altered path.
    cv2.setRNGSeed(72)
    implicit = track_video(clip_v3, anchors_v3, 3, 26, threading.Event())
    cv2.setRNGSeed(72)
    explicit = track_video(clip_v3, anchors_v3, 3, 26, threading.Event(), mode="normal")
    for field in ("positions", "boxes", "confidence"):
        np.testing.assert_allclose(getattr(implicit, field), getattr(explicit, field),
                                   atol=1e-10, rtol=0)
    assert implicit.states == explicit.states
    assert implicit.tracking_mode == explicit.tracking_mode == "normal"
    assert implicit.analysis_size == explicit.analysis_size == (640, 360)


def test_precision_bidirectional_multi_anchor_trim_metadata_and_actual_audio_export(
        clip_v3, anchors_v3, tracks_v3, tmp_path):
    track = tracks_v3["precision"]
    assert (track.start, track.end, track.count) == (3, 26, 24)
    assert track.tracking_mode == "precision"
    assert track.analysis_size == (640, 360)
    assert not track.lost_frames
    assert {3 + index for index, state in enumerate(track.states) if state == "manual"} == {7, 19}
    # Frames before the first anchor exercise reverse decoding/tracking;
    # frames between anchors and after the second exercise separate forward jobs.
    for index, position in enumerate(track.positions):
        box = np.asarray(demo_frame(track.start + index, 640, 360)[1])
        expected = box[:2] + box[2:] / 2
        assert np.linalg.norm((position - expected) * [640, 360]) < 1.0
    target = tmp_path / "精准裁段含音频.mp4"
    settings = Settings(tracking_mode="precision")
    export_video(clip_v3, track, settings, str(target), threading.Event())
    result = probe(str(target))
    assert result.has_audio
    assert abs(result.duration - 24 / clip_v3.fps) < .06
    capture = cv2.VideoCapture(str(target))
    source_capture = cv2.VideoCapture(clip_v3.path)
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 24
        ok, actual = capture.read()
        source_capture.set(cv2.CAP_PROP_POS_FRAMES, 3)
        source_ok, original = source_capture.read()
    finally:
        capture.release()
        source_capture.release()
    assert ok and source_ok
    plan = build_plan(clip_v3, track, settings)
    expected = cv2.warpAffine(original, plan.matrices[0], (plan.width, plan.height),
                              borderMode=cv2.BORDER_REPLICATE)
    assert np.mean(np.abs(actual.astype(float) - expected.astype(float))) < 4


def test_ui_mode_change_keeps_anchors_clears_track_and_prepares_source_detail_async(
        window_v3, qt_app_v3, clip_v3, anchors_v3, tracks_v3, tmp_path):
    # A real lower-detail normal preview makes accidental reuse/upscaling of its
    # pixels observable even with a short, inexpensive 640px source fixture.
    low_proxy = tmp_path / "低清普通代理.mp4"
    writer = cv2.VideoWriter(str(low_proxy), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    assert writer.isOpened()
    try:
        for frame in range(30):
            writer.write(cv2.resize(demo_frame(frame, 640, 360)[0], (320, 180)))
    finally:
        writer.release()
    window_v3._import_done(replace(clip_v3, proxy=str(low_proxy)))
    window_v3._apply_trim(3, 26)
    window_v3.anchors = dict(anchors_v3)
    window_v3.track = tracks_v3["normal"]
    window_v3._rebuild_plan()
    window_v3._seek(12)
    window_v3._set_view("source")
    window_v3.canvas.set_zoom(2.)
    assert window_v3.frame.shape[1::-1] == (320, 180)
    window_v3.tracking_combo.setCurrentIndex(window_v3.tracking_combo.findData("precision"))
    assert window_v3.busy and window_v3.worker is not None
    assert window_v3.anchors == anchors_v3
    assert window_v3.track is None and window_v3.plan is None
    assert window_v3.frame_index == 12
    finish_jobs(window_v3, qt_app_v3)
    assert not window_v3._test_errors
    assert window_v3.precision_info is not None
    assert window_v3.precision_info.path == clip_v3.path
    assert window_v3.frame.shape[1::-1] == (640, 360)
    assert window_v3.canvas.zoom_factor == 2.
    assert window_v3.anchors == anchors_v3
    assert window_v3.settings().tracking_mode == "precision"
    window_v3.tracking_combo.setCurrentIndex(window_v3.tracking_combo.findData("normal"))
    assert not window_v3.busy
    assert window_v3.frame.shape[1::-1] == (320, 180)
    assert window_v3.anchors == anchors_v3


@pytest.mark.parametrize("outcome", ["cancel", "failure"])
def test_mode_change_cancel_or_failure_cannot_leave_stale_processed_preview(
        outcome, window_v3, qt_app_v3, anchors_v3, tracks_v3, monkeypatch):
    window_v3._apply_trim(3, 26)
    window_v3.anchors = dict(anchors_v3)
    window_v3.track = tracks_v3["normal"]
    window_v3._rebuild_plan()
    window_v3._seek(7)
    assert window_v3.canvas.processed is not None
    assert window_v3.canvas.processed_image is not None
    entered = threading.Event()

    def preparation_with_failure_or_cancel(info, cancel, progress=None):
        entered.set()
        if outcome == "failure":
            raise RuntimeError("受控的高清预览准备失败")
        cancel.wait(5)
        check_cancel(cancel)
        raise AssertionError("测试应在预览准备阶段取消")

    monkeypatch.setattr("lockframe.app.prepare_precision_video", preparation_with_failure_or_cancel)
    window_v3.tracking_combo.setCurrentIndex(window_v3.tracking_combo.findData("precision"))
    assert window_v3.track is None and window_v3.plan is None
    assert window_v3.canvas.processed is None
    assert window_v3.canvas.processed_image is None
    assert entered.wait(2)
    if outcome == "cancel":
        window_v3.worker.cancel()
    finish_jobs(window_v3, qt_app_v3)
    assert window_v3.anchors == anchors_v3
    assert window_v3.track is None and window_v3.plan is None
    assert window_v3.canvas.processed is None
    assert window_v3.canvas.processed_image is None
    assert bool(window_v3._test_errors) == (outcome == "failure")


@pytest.mark.parametrize("mode", ["precision", "normal"])
def test_saved_precision_and_revision2_normal_projects_restore_real_tracks(
        mode, window_v3, qt_app_v3, clip_v3, anchors_v3, tracks_v3, tmp_path):
    project = tmp_path / f"{mode}.lockframe"
    save_project(str(project), clip_v3, Settings(tracking_mode=mode), anchors_v3,
                 3, 26, tracks_v3[mode])
    payload = read_project(str(project))
    if mode == "normal":
        # Revision 2 predates mode/analysis metadata; it remains normal mode.
        payload["tracker_revision"] = 2
        payload["settings"].pop("tracking_mode")
        payload["track"].pop("tracking_mode")
        payload["track"].pop("analysis_size")
    window_v3._pending_project = payload
    window_v3._job(import_video, (clip_v3.path,), window_v3._import_done, "测试工程恢复")
    finish_jobs(window_v3, qt_app_v3)
    assert not window_v3._test_errors
    assert window_v3.settings().tracking_mode == mode
    assert window_v3.anchors == anchors_v3
    assert window_v3.track is not None
    assert window_v3.track.tracking_mode == mode
    np.testing.assert_allclose(window_v3.track.positions, tracks_v3[mode].positions,
                               atol=1e-10, rtol=0)
    assert window_v3.track.states == tracks_v3[mode].states
    assert (window_v3.start_spin.value(), window_v3.end_spin.value()) == (4, 27)
    assert not window_v3.dirty
    if mode == "precision":
        assert window_v3.precision_info is not None
        assert window_v3.track.analysis_size == [640, 360]


def test_source_zoom_does_not_dirty_project_change_settings_or_change_export_plan(
        window_v3, qt_app_v3, anchors_v3, tracks_v3):
    window_v3.anchors = dict(anchors_v3)
    window_v3.track = tracks_v3["normal"]
    window_v3._apply_trim(3, 26)
    # Applying a changed range invalidates the track by design; put back the
    # previously measured track for exactly that range before testing view state.
    window_v3.track = tracks_v3["normal"]
    window_v3._rebuild_plan()
    window_v3._seek(7)
    window_v3._set_view("source")
    window_v3.dirty = False
    original_settings = asdict(window_v3.settings())
    original_anchors = copy.deepcopy(window_v3.anchors)
    original_positions = window_v3.track.positions.copy()
    original_plan = window_v3.plan.matrices.copy()
    original_zoom = window_v3.plan.zoom
    window_v3.preview_zoom.setValue(3.)
    qt_app_v3.processEvents()
    assert window_v3.canvas.zoom_factor == 3.
    assert not window_v3.dirty
    assert asdict(window_v3.settings()) == original_settings
    assert window_v3.anchors == original_anchors
    np.testing.assert_array_equal(window_v3.track.positions, original_positions)
    np.testing.assert_array_equal(window_v3.plan.matrices, original_plan)
    assert window_v3.plan.zoom == original_zoom
    window_v3.preview_fit.click()
    assert window_v3.canvas.zoom_factor == 1.
    assert not window_v3.dirty


def test_zoomed_point_and_rectangle_selection_stay_in_original_image_coordinates(window_v3, qt_app_v3):
    window_v3._set_view("source")
    window_v3.pick_mode.setCurrentIndex(window_v3.pick_mode.findData("point"))
    window_v3.point_size.setValue(32)
    window_v3.preview_zoom.setValue(3.)
    qt_app_v3.processEvents()
    normalized_center = np.array([.51, .48])
    rect = window_v3.canvas.source_rect
    point = QPoint(round(rect.left() + normalized_center[0] * rect.width()),
                   round(rect.top() + normalized_center[1] * rect.height()))
    QTest.mouseClick(window_v3.canvas, Qt.MouseButton.LeftButton, pos=point)
    selected = np.asarray(window_v3.anchors[0])
    np.testing.assert_allclose(selected[2:] * [640, 360], [32, 32], atol=.05)
    np.testing.assert_allclose(selected[:2] + selected[2:] / 2, normalized_center, atol=.001)
    window_v3.pick_mode.setCurrentIndex(window_v3.pick_mode.findData("box"))
    low, high = np.array([.48, .45]), np.array([.55, .53])
    start = QPoint(round(rect.left() + low[0] * rect.width()),
                   round(rect.top() + low[1] * rect.height()))
    end = QPoint(round(rect.left() + high[0] * rect.width()),
                 round(rect.top() + high[1] * rect.height()))
    QTest.mousePress(window_v3.canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(window_v3.canvas, end)
    QTest.mouseRelease(window_v3.canvas, Qt.MouseButton.LeftButton, pos=end)
    selected = np.asarray(window_v3.anchors[0])
    # Native pointer positions are whole screen pixels. The narrower dual
    # monitor can quantize two endpoints by >.001; compare the actual clicks.
    clicked_low = np.array([(start.x()-rect.left())/rect.width(),
                            (start.y()-rect.top())/rect.height()])
    clicked_high = np.array([(end.x()-rect.left())/rect.width(),
                             (end.y()-rect.top())/rect.height()])
    np.testing.assert_allclose(selected, [*clicked_low, *(clicked_high-clicked_low)], atol=1e-7)
