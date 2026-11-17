from dataclasses import replace
from pathlib import Path
import json
import subprocess
import threading

import cv2
import numpy as np
import pytest

from lockframe.demo import create_demo, demo_frame
from lockframe.export import export_video
from lockframe.media import Cancelled, NO_WINDOW, FrameReader, find_tool, import_video, probe, run_ffmpeg
from lockframe.models import Settings, TrackResult, VideoInfo, read_project, save_project
from lockframe.stabilization import build_plan, preview_frame
from lockframe.stableflow_filters import smooth_transform_series, suppress_trajectory_outliers
from lockframe.tracking import FeatureTracker, track_video


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    folder = tmp_path_factory.mktemp("汽车视频测试")
    path = folder / "含声音测试.mp4"
    create_demo(str(path), frames=90, width=640, height=360)
    return import_video(str(path), threading.Event())


@pytest.fixture(scope="module")
def tracked(clip):
    _, box = demo_frame(35, 640, 360)
    return track_video(clip, {35: box}, 0, 89, threading.Event())


def test_bidirectional_tracking_matches_ground_truth(clip, tracked):
    expected = []
    for i in range(90):
        _, box = demo_frame(i, 640, 360)
        expected.append([box[0] + box[2] / 2, box[1] + box[3] / 2])
    error_px = np.linalg.norm((tracked.positions - expected) * [640, 360], axis=1)
    assert not tracked.lost_frames
    assert error_px.max() < 2.5, f"max error: {error_px.max()} px"
    assert tracked.states[35] == "manual"


def test_blank_target_is_rejected():
    with pytest.raises(ValueError, match="纹理"):
        FeatureTracker(np.full((200, 300), 90, np.uint8), [40, 40, 50, 50])


def test_occlusion_is_not_reported_as_valid_tracking():
    frame, normalized = demo_frame(0, 640, 360)
    box = np.array(normalized) * [640, 360, 640, 360]
    tracker = FeatureTracker(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), box)
    observation = tracker.step(np.full((360, 640), 15, np.uint8))
    assert observation.state == "lost"
    assert observation.confidence == 0
    resumed, _ = demo_frame(1, 640, 360)
    observation = tracker.step(cv2.cvtColor(resumed, cv2.COLOR_BGR2GRAY))
    assert observation.state == "recovered"


def test_manual_anchors_restart_each_segment(clip):
    anchors = {index: demo_frame(index, 640, 360)[1] for index in (20, 65)}
    result = track_video(clip, anchors, 8, 80, threading.Event())
    assert result.count == 73
    for index, box in anchors.items():
        assert result.states[index - 8] == "manual"
        np.testing.assert_allclose(result.boxes[index - 8], box)
    assert not result.lost_frames


@pytest.mark.parametrize("aspect", ["original", "9:16", "16:9", "1:1"])
@pytest.mark.parametrize("mode", ["lock", "follow"])
def test_plan_keeps_full_frame_coverage_and_preview_geometry(clip, tracked, aspect, mode):
    plan = build_plan(clip, tracked, Settings(mode=mode, aspect=aspect, max_zoom=1.3))
    for matrix in plan.matrices:
        sx, sy = matrix[0, 0], matrix[1, 1]
        tx, ty = matrix[:, 2]
        assert tx <= 1e-5 and ty <= 1e-5
        assert tx + sx * clip.width >= plan.width - 1e-5
        assert ty + sy * clip.height >= plan.height - 1e-5
    reader = FrameReader(clip.proxy)
    try:
        image = preview_frame(reader.read(20), clip, plan, 20)
        assert abs(image.shape[1] / image.shape[0] - plan.width / plan.height) < 0.01
    finally:
        reader.close()


def test_lock_centers_subject_when_crop_budget_is_sufficient(clip, tracked):
    settings = Settings(max_zoom=3)
    plan = build_plan(clip, tracked, settings)
    assert plan.limited_frames == 0
    points = tracked.positions * [clip.width, clip.height]
    rendered = np.einsum("nij,nj->ni", plan.matrices[:, :, :2], points) + plan.matrices[:, :, 2]
    np.testing.assert_allclose(rendered, np.tile([plan.width / 2, plan.height / 2], (tracked.count, 1)), atol=1e-5)


def test_crop_limit_is_reported_rather_than_exposing_borders(clip, tracked):
    plan = build_plan(clip, tracked, Settings(max_zoom=1))
    assert plan.limited_frames > 0
    assert plan.zoom == 1


def test_sdr_export_uses_original_frames_and_keeps_audio(clip, tracked, tmp_path):
    first, last = 12, 74
    selection = slice(first, last + 1)
    trimmed = TrackResult(first, last, tracked.positions[selection], tracked.boxes[selection],
                          tracked.confidence[selection], tracked.states[selection])
    output = tmp_path / "稳定后_带声音.mp4"
    result = export_video(clip, trimmed, Settings(aspect="9:16", max_zoom=2.5), str(output), threading.Event())
    exported = probe(str(output))
    assert exported.has_audio
    assert exported.width == result["width"] and exported.height == result["height"]
    assert abs(exported.fps - clip.fps) < 0.01
    assert abs(exported.duration - trimmed.count / clip.fps) < 0.05
    cap = cv2.VideoCapture(str(output))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == trimmed.count
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.mean() > 10


def test_cancel_export_preserves_existing_destination(clip, tracked, tmp_path):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing file must remain")
    event = threading.Event()

    def progress(value, message):
        if value > 0.08:
            event.set()

    with pytest.raises(Cancelled):
        export_video(clip, tracked, Settings(), str(output), event, progress)
    assert output.read_bytes() == b"existing file must remain"
    assert not list(tmp_path.glob(".lockframe-export-*"))


def test_export_blocks_lost_track_and_source_overwrite(clip, tracked, tmp_path):
    states = tracked.states.copy()
    states[5] = "lost"
    bad = replace(tracked, states=states)
    with pytest.raises(ValueError, match="跟丢"):
        export_video(clip, bad, Settings(), str(tmp_path / "invalid.mp4"), threading.Event())
    with pytest.raises(ValueError, match="覆盖"):
        export_video(clip, tracked, Settings(), clip.path, threading.Event())


def test_project_roundtrip_unicode_paths(clip, tracked, tmp_path):
    project = tmp_path / "我的汽车镜头.lockframe"
    anchor = demo_frame(35, 640, 360)[1]
    save_project(str(project), clip, Settings(), {35: anchor}, 0, 89, tracked)
    payload = read_project(str(project))
    assert payload["source"] == clip.path
    assert payload["fingerprint"] == clip.fingerprint
    np.testing.assert_allclose(payload["track"]["positions"], tracked.positions)


def test_import_handles_phone_rotation_metadata(clip, tmp_path):
    rotated = tmp_path / "手机竖向.mp4"
    run_ffmpeg(["-display_rotation:v:0", "90", "-i", clip.path, "-c", "copy", str(rotated)], threading.Event())
    info = import_video(str(rotated), threading.Event())
    assert (info.width, info.height) == (360, 640)
    reader = FrameReader(info.proxy)
    try:
        frame = reader.read(0)
        assert (frame.shape[1], frame.shape[0]) == (360, 640)
    finally:
        reader.close()


def test_stableflow_ports_preserve_upstream_window_and_spike_behavior():
    points = np.column_stack([np.arange(8), np.arange(8)]).astype(float) / 100
    averaged = smooth_transform_series(points, 2)
    np.testing.assert_allclose(averaged[0], points[:3].mean(axis=0))
    np.testing.assert_allclose(averaged[3], points[1:6].mean(axis=0))
    points[4] = [0.8, 0.8]
    clean = suppress_trajectory_outliers(points)
    assert np.linalg.norm(clean[4] - [0.04, 0.04]) < 0.01


def test_desktop_roi_mapping_tracking_and_project_restore(clip, tracked, tmp_path):
    import os
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from lockframe.app import MainWindow

    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    errors = []
    window._error = errors.append
    window.resize(1620, 960)
    window.show()
    application.processEvents()
    try:
        window._import_done(clip)
        window._sync_ui()
        window._set_view("source")
        application.processEvents()
        rect = window.canvas.source_rect
        expected = demo_frame(0, 640, 360)[1]
        x, y, w, h = expected
        start = QPoint(round(rect.left() + rect.width() * x), round(rect.top() + rect.height() * y))
        end = QPoint(round(rect.left() + rect.width() * (x + w)), round(rect.top() + rect.height() * (y + h)))
        QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(window.canvas, end)
        QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=end)
        assert 0 in window.anchors
        np.testing.assert_allclose(window.anchors[0], expected, atol=0.004)
        assert window.track_button.isEnabled()
        window._start_tracking()
        deadline = time.monotonic() + 20
        while window.busy and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.005)
        assert not window.busy and not errors
        assert window.track is not None and not window.track.lost_frames
        assert window.export_button.isEnabled()
        window.mode_combo.setCurrentIndex(1)
        assert window.smoothing_slider.isEnabled() and not window.target_combo.isEnabled()
        window.mode_combo.setCurrentIndex(0)
        window.zoom_slider.setValue(240)
        window._set_view("compare")
        window._seek(40)
        application.processEvents()
        artifact_dir = Path(__file__).resolve().parents[1] / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        assert window.grab().save(str(artifact_dir / "desktop-preview.png"))
        project = tmp_path / "继续工作.lockframe"
        window.project_path = str(project)
        assert window._save_project()
        payload = read_project(str(project))
        window._pending_project = payload
        window._import_done(clip)
        window._sync_ui()
        assert window.track is not None and window.export_button.isEnabled()
        # Manual correction must invalidate old trajectories, never export stale data.
        source_track = window.track
        window._seek(45)
        window._select_roi(demo_frame(45, 640, 360)[1])
        assert window.track is source_track and window._retrack_previous is source_track
        assert window._region_stale() and not window.export_button.isEnabled()
    finally:
        window.dirty = False
        window.close()
        application.processEvents()
