"""Frame-accurate excerpt selection and preview lifecycle checks."""
from dataclasses import asdict
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from lockframe.models import VideoInfo
from lockframe import trim


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
    yield app


@pytest.fixture
def dialog_factory(qapp, monkeypatch):
    readers = []
    dialogs = []

    class Reader:
        def __init__(self, path):
            self.path = path
            self.reads = []
            self.closed = 0
            self.fail = set()
            readers.append(self)

        def read(self, frame):
            assert not self.closed
            self.reads.append(frame)
            if frame in self.fail:
                raise ValueError("bad frame")
            return np.full((90, 160, 3), frame % 256, np.uint8)

        def close(self):
            self.closed += 1

    monkeypatch.setattr(trim, "FrameReader", Reader)

    def create(frames=300, fps=30000 / 1001, start=0, end=None, current=0):
        info = VideoInfo("untouched-source.mp4", 1920, 1080, fps, frames / fps,
                         True, frames, "preview-only.mp4")
        dialog = trim.TrimDialog(info, start, frames - 1 if end is None else end, current)
        dialogs.append(dialog)
        return dialog, readers[-1]

    yield create
    for dialog in dialogs:
        dialog.reject()
        dialog.deleteLater()
    qapp.processEvents()


def test_cancel_keeps_source_metadata_and_closes_preview(dialog_factory, qapp):
    dialog, reader = dialog_factory(start=12, end=100, current=35)
    before = asdict(dialog.info)
    dialog.show()
    qapp.processEvents()
    dialog.start_input.setValue(1.2)
    dialog.play()
    assert dialog.timer.isActive()
    QTest.mouseClick(dialog.cancel_button, Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert asdict(dialog.info) == before
    assert reader.path == "preview-only.mp4"
    assert reader.closed == 1 and dialog.reader is None
    assert not dialog.timer.isActive()
    dialog.close()
    assert reader.closed == 1


def test_fractional_fps_time_entry_is_frame_accurate_and_exclusive(dialog_factory):
    dialog, reader = dialog_factory(start=0, end=299)
    dialog.start_input.setValue(1.2)
    dialog.end_input.setValue(2.5)
    assert (dialog.start_frame, dialog.end_frame) == (36, 74)
    assert dialog.current_frame == 74
    assert reader.reads[-2:] == [36, 74]
    assert dialog.start_input.value() == pytest.approx(36 / dialog.info.fps, abs=.00051)
    assert dialog.end_input.value() == pytest.approx(75 / dialog.info.fps, abs=.00051)
    assert "39 帧" in dialog.duration_label.text()
    assert "00:01.301" in dialog.duration_label.text()


@pytest.mark.parametrize("fps", [29.97, 59.94, 120, 240])
def test_time_arrows_keep_advancing_exactly_one_frame(dialog_factory, fps):
    dialog, _ = dialog_factory(frames=80, fps=fps)
    for frame in range(1, 30):
        dialog.start_input.stepUp()
        assert dialog.start_frame == frame
    for frame in range(79, 49, -1):
        dialog.end_input.stepDown()
        assert dialog.end_frame == frame - 1


def test_time_edits_cannot_invert_range_and_keep_one_frame(dialog_factory):
    dialog, _ = dialog_factory(frames=2, fps=29.97)
    dialog.start_input.setValue(999)
    assert (dialog.start_frame, dialog.end_frame) == (1, 1)
    dialog.end_input.setValue(-999)
    assert (dialog.start_frame, dialog.end_frame) == (1, 1)
    assert dialog.end_input.value() > dialog.start_input.value()
    assert "1 帧" in dialog.duration_label.text()
    dialog._reset()
    assert (dialog.start_frame, dialog.end_frame) == (0, 1)
    dialog.end_input.setValue(0)
    assert (dialog.start_frame, dialog.end_frame) == (0, 0)
    assert dialog.end_input.value() == pytest.approx(1 / 29.97, abs=.00051)


@pytest.mark.parametrize("start,end,current,expected", [
    (-5, 999, -99, (0, 1, 0)),
    (999, -1, 999, (1, 1, 1)),
    (1, 0, 0, (1, 1, 0)),
])
def test_initial_intervals_are_clamped_for_two_frame_video(
        dialog_factory, start, end, current, expected):
    dialog, _ = dialog_factory(frames=2, start=start, end=end, current=current)
    assert (dialog.start_frame, dialog.end_frame, dialog.current_frame) == expected


def test_mouse_drags_handles_and_scrubs_independently(dialog_factory, qapp):
    dialog, _ = dialog_factory(frames=100, fps=25)
    dialog.show()
    qapp.processEvents()
    bar = dialog.timeline
    QTest.mousePress(bar, Qt.MouseButton.LeftButton,
                     pos=QPoint(round(bar.boundary_x(0)), bar.HANDLE_Y))
    QTest.mouseMove(bar, QPoint(round(bar.boundary_x(20)), bar.HANDLE_Y))
    QTest.mouseRelease(bar, Qt.MouseButton.LeftButton,
                       pos=QPoint(round(bar.boundary_x(20)), bar.HANDLE_Y))
    assert dialog.start_frame == 20 and dialog.current_frame == 20
    QTest.mousePress(bar, Qt.MouseButton.LeftButton,
                     pos=QPoint(round(bar.boundary_x(100)), bar.HANDLE_Y))
    QTest.mouseMove(bar, QPoint(round(bar.boundary_x(70)), bar.HANDLE_Y))
    QTest.mouseRelease(bar, Qt.MouseButton.LeftButton,
                       pos=QPoint(round(bar.boundary_x(70)), bar.HANDLE_Y))
    assert dialog.end_frame == 69 and dialog.current_frame == 69
    # The lower rail can seek past an in/out handle without moving the selection.
    QTest.mouseClick(bar, Qt.MouseButton.LeftButton,
                     pos=QPoint(round(bar.frame_x(90)), bar.RAIL_Y))
    assert dialog.current_frame == 90
    assert (dialog.start_frame, dialog.end_frame) == (20, 69)
    QTest.keyClick(bar, Qt.Key.Key_Left)
    assert dialog.current_frame == 89
    QTest.mouseClick(dialog.set_end_button, Qt.MouseButton.LeftButton)
    assert (dialog.start_frame, dialog.end_frame) == (20, 89)


def test_manual_point_buttons_use_the_visible_frame_and_reset(dialog_factory, qapp):
    dialog, _ = dialog_factory(frames=100, fps=25, start=20, end=70)
    dialog.show()
    qapp.processEvents()
    dialog.seek(90)
    QTest.mouseClick(dialog.set_start_button, Qt.MouseButton.LeftButton)
    assert (dialog.start_frame, dialog.end_frame) == (90, 90)
    dialog.seek(10)
    QTest.mouseClick(dialog.set_end_button, Qt.MouseButton.LeftButton)
    assert (dialog.start_frame, dialog.end_frame) == (10, 10)
    QTest.mouseClick(dialog.reset_button, Qt.MouseButton.LeftButton)
    assert (dialog.start_frame, dialog.end_frame) == (0, 99)


def test_playback_loops_only_inside_selection_then_stops(dialog_factory, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(trim, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    dialog, reader = dialog_factory(frames=100, fps=25, start=20, end=24, current=90)
    reader.reads.clear()
    dialog.play()
    assert dialog.current_frame == 20
    clock[0] += 4.1 / 25
    dialog._play_tick()
    assert dialog.current_frame == 24
    clock[0] += 1 / 25
    dialog._play_tick()
    assert dialog.current_frame == 20
    clock[0] += 17 / 25
    dialog._play_tick()
    assert 20 <= dialog.current_frame <= 24
    assert reader.reads and all(20 <= frame <= 24 for frame in reader.reads)
    dialog.loop_check.setChecked(False)
    clock[0] += 20 / 25
    dialog._play_tick()
    assert dialog.current_frame == 24
    assert not dialog.timer.isActive()


def test_playback_handles_single_frame_and_range_change(dialog_factory, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(trim, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    dialog, reader = dialog_factory(frames=2, fps=29.97, start=1, end=1)
    dialog.play()
    clock[0] += 1
    dialog._play_tick()
    assert dialog.current_frame == 1 and dialog.timer.isActive()
    dialog._reset()
    assert not dialog.timer.isActive()
    assert (dialog.start_frame, dialog.end_frame) == (0, 1)
    dialog.play()
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not dialog.timer.isActive() and reader.closed == 1


def test_accept_commits_typed_time_and_releases_reader(dialog_factory, qapp):
    dialog, reader = dialog_factory(frames=100, fps=25)
    dialog.show()
    qapp.processEvents()
    edit = dialog.end_input.lineEdit()
    edit.setFocus()
    QTest.keyClick(edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClicks(edit, "2.000")
    QTest.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert (dialog.start_frame, dialog.end_frame) == (0, 49)
    assert reader.closed == 1 and not dialog.timer.isActive()


def test_preview_read_failure_does_not_block_selection_or_next_seek(dialog_factory):
    dialog, reader = dialog_factory(frames=100)
    reader.fail.add(12)
    dialog.play()
    dialog.seek(12)
    assert not dialog.timer.isActive()
    assert dialog.preview.image is None
    dialog._range_changed(10, 30)
    dialog.seek(13)
    assert dialog.preview.image is not None
    assert dialog.current_frame == 13


def test_failed_playback_start_leaves_timer_stopped(dialog_factory):
    dialog, reader = dialog_factory(frames=100, start=10, end=40, current=80)
    reader.fail.add(10)
    dialog.play()
    assert not dialog.timer.isActive()
    assert dialog.current_frame == 10


@pytest.mark.parametrize("fps,frames", [(0, 2), (float("nan"), 2), (30, 0)])
def test_invalid_video_timing_is_rejected(qapp, fps, frames):
    info = VideoInfo("source.mp4", 1920, 1080, fps, 1, frames=frames)
    with pytest.raises(ValueError, match="帧率或帧数"):
        trim.TrimDialog(info, 0, 1)
