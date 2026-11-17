"""Parallel completion order must not mix the followed clip's UI identity."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QSlider

from lockframe import live_preview_ui
from lockframe.live_preview_ui import LivePreviewMixin
from lockframe.models import VideoInfo, source_identity
from lockframe.progress_ui import TrackingProgressPanel
from lockframe.sequence import Clip
from lockframe.sequence_ui import SequenceMixin
from lockframe.workspace_ui import AssetStatusBar


class _Reader:
    def __init__(self, proxy):
        self.proxy = proxy
        self.closed = False

    def read(self, index):
        # A different pixel value makes a foreign source observable on canvas.
        return np.full((2, 2, 3), ord(Path(self.proxy).stem), dtype=np.uint8)

    def close(self):
        self.closed = True


class _Canvas:
    mode = "source"
    frame = None

    def set_frames(self, frame, **kwargs):
        self.frame = frame


class _QueueView(SequenceMixin, LivePreviewMixin):
    def _refresh_sequence(self, *args):
        pass

    def _tracking_done(self, track):
        self.track = track

    def _update_strip(self):
        pass


@pytest.fixture
def queue_view(monkeypatch):
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(live_preview_ui, "FrameReader", _Reader)
    view = _QueueView()
    view.clips = [Clip(VideoInfo(f"{name}.mp4", 640 + index, 360, 25., 4.,
                                frames=100, proxy=f"{name}.proxy"), id=name)
                  for index, name in enumerate("ABC")]
    view.active_clip_id = "A"
    view.info, view.reader = view.clips[0].info, _Reader("A.proxy")
    view._live_queue_follow, view._live_follow_clip_id = True, "A"
    view._live_completed_ids = set()
    view._paused_by_clip = {}
    view._queue_order = ["A", "B", "C"]
    view._queue_running_ids, view._queue_progress_by_id = set(), {}
    view._queue_panel_clip_id = "A"
    view.tracking_progress_panel = TrackingProgressPanel()
    view.tracking_progress_panel.begin("A.mp4", 1, 3, batch=True, follow_task_phases=True)
    view.asset_status_bar = AssetStatusBar()
    view.canvas, view.timeline = _Canvas(), QSlider()
    view.live_preview_check = QCheckBox()
    view.live_preview_check.setChecked(True)
    view.preview_note, view.time_label = QLabel(), QLabel()
    yield view
    view._close_foreign_preview()
    for widget in (view.tracking_progress_panel, view.asset_status_bar, view.timeline,
                   view.live_preview_check, view.preview_note, view.time_label):
        widget.close()
    application.processEvents()


def _frame(view, clip_id, index):
    view._on_queue_tracking_frame(clip_id, {
        "frame": index, "source_identity": source_identity(view._clip(clip_id).info),
        "box": (.2, .2, .3, .3), "confidence": .9, "state": "tracked",
        "completed": index + 1, "total": 100,
    })


def _assert_displayed_clip(view, clip_id, calculation_fraction):
    clip = view._clip(clip_id)
    assert view._queue_panel_clip_id == view._live_follow_clip_id == clip_id
    assert view.tracking_progress_panel.filename == Path(clip.info.path).name
    assert view.tracking_progress_panel.task_index == "ABC".index(clip_id) + 1
    assert view.tracking_progress_panel.phases["计算"].fraction == calculation_fraction
    assert view._display_info is clip.info
    assert view.asset_status_bar.source_path == clip.info.path
    assert f"{clip.info.width} × {clip.info.height}" in view.asset_status_bar.metadata_label.text()
    assert np.all(view.canvas.frame == ord(clip_id))
    assert Path(clip.info.path).name in view.preview_note.text()


def test_parallel_replacement_cannot_steal_followed_clip_identity(queue_view):
    view = queue_view
    # A and B start together. B has measured progress before A completes.
    for clip_id in "AB":
        view._queue_item_done(clip_id, None, "正在跟踪")
    view._queue_item_progress("A", .7, "[计算:0.800] A 的进度")
    view._queue_item_progress("B", .3, "[计算:0.250] B 的进度")
    _frame(view, "A", 10)
    _assert_displayed_clip(view, "A", .8)

    view._queue_item_done("A", object(), "")
    assert view._queue_panel_clip_id == view._live_follow_clip_id == "B"
    assert view.tracking_progress_panel.filename == "B.mp4"
    assert view.tracking_progress_panel.phases["计算"].fraction == .25

    # The freed worker starts C; its progress and frame arrive before B's next
    # event. Those legitimate C observations must not change the followed B UI.
    view._queue_item_done("C", None, "正在跟踪")
    view._queue_item_progress("C", .6, "[计算:0.650] C 的进度")
    _frame(view, "C", 30)
    assert view._queue_panel_clip_id == view._live_follow_clip_id == "B"
    assert view.tracking_progress_panel.filename == "B.mp4"
    assert view.tracking_progress_panel.phases["计算"].fraction == .25
    # Until B produces a frame, retain the last measured image and its own
    # metadata instead of labelling an A image as B or painting C out of turn.
    assert view.asset_status_bar.source_path == "A.mp4"
    assert np.all(view.canvas.frame == ord("A"))
    _frame(view, "B", 20)
    _assert_displayed_clip(view, "B", .25)
    view._queue_item_progress("B", .5, "[计算:0.500] B 的下一帧")
    _frame(view, "B", 21)
    _assert_displayed_clip(view, "B", .5)

    view._queue_item_done("B", object(), "")
    _frame(view, "C", 31)
    _assert_displayed_clip(view, "C", .65)
    # Queued late observations from completed B cannot restore its old identity.
    view._queue_item_progress("B", .9, "[计算:0.900] B 迟到的进度")
    _frame(view, "B", 22)
    _assert_displayed_clip(view, "C", .65)
    assert view.active_clip_id == "A" and view.info is view.clips[0].info


def test_first_live_frame_respects_existing_phase_panel_identity(queue_view):
    view = queue_view
    view._show_queue_task("B")
    view._queue_item_progress("B", .3, "[计算:0.250] B 的进度")
    # No live frame has been selected yet, while phase events already chose B.
    view._live_follow_clip_id = None
    _frame(view, "C", 30)
    assert view._live_follow_clip_id == "B"
    assert view.canvas.frame is None
    assert view.tracking_progress_panel.filename == "B.mp4"
    assert view.asset_status_bar.source_path is None
    _frame(view, "B", 20)
    _assert_displayed_clip(view, "B", .25)
