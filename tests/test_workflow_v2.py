"""Regression tests for the user-visible trim -> select -> stabilize workflow."""
from dataclasses import replace
from pathlib import Path
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from lockframe.demo import create_demo, demo_frame
from lockframe.export import export_video
from lockframe.media import import_video, probe
from lockframe.models import Settings, file_fingerprint, read_project, save_project
from lockframe.stabilization import build_plan


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(scope="module")
def clip_v2(tmp_path_factory):
    source = tmp_path_factory.mktemp("trim-workflow") / "片段.mp4"
    create_demo(str(source), frames=60, width=640, height=360)
    return import_video(str(source), threading.Event())


@pytest.fixture
def window(app, clip_v2):
    from lockframe.app import MainWindow
    window = MainWindow()
    window.resize(1560, 960)
    window._import_done(clip_v2)
    window._sync_ui()
    window.show()
    app.processEvents()
    yield window
    window.dirty = False
    if window.busy:
        window.worker.cancel()
        deadline = time.monotonic() + 10
        while window.busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
    window.close()
    app.processEvents()


def test_precise_pick_uses_image_pixels_and_keeps_clicked_center(window, app):
    window._set_view("source")
    window.pick_mode.setCurrentIndex(window.pick_mode.findData("point"))
    window.point_size.setValue(48)
    app.processEvents()
    rect = window.canvas.source_rect
    box = demo_frame(0, 640, 360)[1]
    center = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2])
    point = QPoint(round(rect.x() + center[0] * rect.width()), round(rect.y() + center[1] * rect.height()))
    QTest.mouseMove(window.canvas, point)
    QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=point)
    selected = np.array(window.anchors[0])
    np.testing.assert_allclose(selected[2:] * [640, 360], [48, 48], atol=0.1)
    np.testing.assert_allclose(selected[:2] + selected[2:] / 2, center, atol=0.002)
    artifact = Path(__file__).resolve().parents[1] / "artifacts" / "v2"
    artifact.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(artifact / "precise-picking.png"))


def test_precise_pick_near_edge_is_rejected_without_shifting_target(window, app):
    window._set_view("source")
    window.pick_mode.setCurrentIndex(1)
    app.processEvents()
    rect = window.canvas.source_rect
    QTest.mouseClick(window.canvas, Qt.MouseButton.LeftButton,
                     pos=QPoint(round(rect.left() + 2), round(rect.center().y())))
    assert not window.anchors
    assert "边缘" in window.status_label.text()


def test_trim_cancellation_keeps_range_and_target(window, monkeypatch):
    window.anchors[0] = demo_frame(0, 640, 360)[1]

    class CancelledDialog:
        def __init__(self, *args, **kwargs):
            self.start_frame, self.end_frame = 12, 35
        def exec(self):
            return QDialog.DialogCode.Rejected
        def deleteLater(self):
            pass

    monkeypatch.setattr("lockframe.app.TrimDialog", CancelledDialog)
    window._edit_trim()
    assert (window.start_spin.value(), window.end_spin.value()) == (1, 60)
    assert 0 in window.anchors


def test_apply_trim_limits_playback_and_requires_an_in_range_target(window, app):
    window.anchors[0] = demo_frame(0, 640, 360)[1]
    window._apply_trim(12, 35)
    assert (window.start_spin.value(), window.end_spin.value()) == (13, 36)
    assert window.frame_index == 12
    assert "0.800 秒" in window.trim_summary.text()
    assert not window.track_button.isEnabled()
    window._seek(20)
    window._select_roi(demo_frame(20, 640, 360)[1])
    assert window.track_button.isEnabled()
    window._seek(0)
    window._toggle_play()
    assert window.frame_index == 12
    window._play_started -= 10
    window._play_tick()
    assert window.frame_index == 35 and not window.timer.isActive()


def test_trim_to_tracking_to_actual_export_and_old_project_invalidation(window, app, clip_v2, tmp_path):
    errors = []
    window._error = errors.append
    window._apply_trim(12, 47)
    window._seek(25)
    window._select_roi(demo_frame(25, 640, 360)[1])
    window._start_tracking()
    deadline = time.monotonic() + 30
    while window.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert not window.busy and not errors
    assert window.track is not None and not window.track.lost_frames
    assert (window.track.start, window.track.end, window.track.count) == (12, 47, 36)
    path = tmp_path / "仅保留选段.mp4"
    export_video(clip_v2, window.track, Settings(), str(path), threading.Event())
    exported = probe(str(path))
    assert abs(exported.duration - 1.2) < 0.05 and exported.has_audio
    capture = cv2.VideoCapture(str(path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 36
    ok, actual = capture.read()
    capture.release()
    source_capture = cv2.VideoCapture(clip_v2.path)
    source_capture.set(cv2.CAP_PROP_POS_FRAMES, 12)
    ok_source, source = source_capture.read()
    source_capture.release()
    assert ok and ok_source
    plan = build_plan(clip_v2, window.track, Settings())
    expected = cv2.warpAffine(source, plan.matrices[0], (plan.width, plan.height), borderMode=cv2.BORDER_REPLICATE)
    assert np.mean(np.abs(expected.astype(float) - actual.astype(float))) < 4
    project = tmp_path / "旧工程.lockframe"
    save_project(str(project), clip_v2, Settings(), window.anchors, 12, 47, window.track)
    payload = read_project(str(project))
    payload.pop("tracker_revision")
    window._pending_project = payload
    window._import_done(clip_v2)
    window._sync_ui()
    assert window.track is None and window.anchors
    assert "重新" in window.status_label.text()


def test_crossed_inline_range_is_clamped_to_a_valid_frame(window):
    window.start_spin.setValue(50)
    window.end_spin.setValue(12)
    assert window.start_spin.value() <= window.end_spin.value()
    assert window.end_spin.value() - window.start_spin.value() + 1 >= 1


def test_changed_source_cannot_stamp_old_anchors_as_a_fresh_project(clip_v2, tmp_path):
    import shutil
    source = tmp_path / "外部会改动的原片.mp4"
    shutil.copy2(clip_v2.path, source)
    original = replace(clip_v2, path=str(source), fingerprint=file_fingerprint(str(source)))
    project = tmp_path / "已有工程.lockframe"
    save_project(str(project), original, Settings(), {25: demo_frame(25, 640, 360)[1]}, 12, 47, None)
    saved = project.read_bytes()
    with source.open("ab") as stream:
        stream.write(b"changed externally")
    with pytest.raises(ValueError, match="原视频已经被修改"):
        save_project(str(project), original, Settings(), {}, 0, 59, None)
    assert project.read_bytes() == saved
