import os
import threading
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lockframe.demo import demo_frame
from lockframe.media import Cancelled, import_video, resource_root
from lockframe.models import Settings
from lockframe.sequence import Clip, run_queue
from lockframe.tracking import FeatureTracker, track_video
from lockframe.workers import Job, QueueJob, _FrameMailbox


@pytest.fixture
def clip():
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    return Clip(info, Settings(acceleration="cpu", compute_threads=2), {5: demo_frame(5)[1]}, 0, 20)


def test_observations_are_real_bidirectional_measurements_and_immutable(clip):
    observations = []
    result = track_video(clip.info, clip.anchors, 0, 20, threading.Event(),
                         confidence_mesh=True, on_frame=observations.append)
    assert [value["frame"] for value in observations] == [5, 4, 3, 2, 1, 0, *range(6, 21)]
    for value in observations:
        index = value["frame"]
        np.testing.assert_array_equal(value["box"], result.boxes[index])
        np.testing.assert_array_equal(value["position"], result.positions[index])
        np.testing.assert_array_equal(value["homography"], result.homographies[index])
        np.testing.assert_allclose(value["grid"], result.confidence_grids[index], atol=1e-6)
        assert value["state"] == result.states[index]
        assert value["confidence"] == result.confidence[index]
        assert value["source_time"] == index / clip.info.fps
    old_box = observations[-1]["box"]
    result.boxes[-1] = 0
    assert observations[-1]["box"] == old_box
    assert observations[-1]["completed"] == observations[-1]["total"] == 21


def test_lost_frame_stays_lost_in_live_observation(clip, monkeypatch):
    original = FeatureTracker.step
    def lost(self, gray):
        value = original(self, gray)
        value.state, value.confidence = "lost", 0.
        return value
    monkeypatch.setattr(FeatureTracker, "step", lost)
    observations = []
    track_video(clip.info, clip.anchors, 0, 6, threading.Event(), on_frame=observations.append)
    assert all(value["state"] == "lost" and value["confidence"] == 0
               for value in observations if value["frame"] != 5)


def test_live_callback_can_cancel_without_another_observation(clip):
    cancel, observations = threading.Event(), []
    def stop(payload):
        observations.append(payload)
        cancel.set()
    with pytest.raises(Cancelled):
        track_video(clip.info, clip.anchors, 0, 20, cancel, on_frame=stop)
    assert [value["frame"] for value in observations] == [5]


@pytest.mark.parametrize("workers", [1, 2])
def test_queue_routes_real_observations_to_each_clip_even_short_clips(clip, workers):
    other = Clip(clip.info, replace(clip.settings), {5: demo_frame(5)[1]}, 3, 8)
    observations = []
    outcomes = run_queue([clip, other], workers, threading.Event(),
                         on_frame=lambda clip_id, payload: observations.append((clip_id, payload)))
    assert {clip_id for clip_id, _ in observations} == {clip.id, other.id}
    for clip_id, value in observations:
        result = outcomes[clip_id]["track"]
        index = value["frame"] - result.start
        np.testing.assert_array_equal(value["box"], result.boxes[index])
        assert value["state"] == result.states[index]
    assert all(any(clip_id == candidate.id and value["completed"] == value["total"]
                   for clip_id, value in observations) for candidate in (clip, other))


def _app():
    return QApplication.instance() or QApplication([])


def test_mailbox_coalesces_per_clip_and_delivers_only_latest_on_gui_thread():
    app, received = _app(), []
    from PySide6.QtCore import QObject, QThread
    parent = QObject()
    mailbox = _FrameMailbox(parent, lambda key, value: received.append((key, value, QThread.currentThread())))
    def push():
        for index in range(500):
            mailbox.put({"frame": index}, "first")
            mailbox.put({"frame": index + 1000}, "second")
    thread = threading.Thread(target=push)
    thread.start()
    thread.join()
    assert received == []
    mailbox.flush()
    assert [(key, value["frame"]) for key, value, _ in received] == [("first", 499), ("second", 1499)]
    assert all(thread == app.thread() for _, _, thread in received)
    mailbox.flush()
    assert len(received) == 2


def test_job_coalesces_until_gui_can_receive_and_keeps_result_order():
    app, delivered = _app(), []
    def work(cancel, progress, on_frame):
        for index in range(500):
            on_frame({"frame": index})
        return "done"
    job = Job(work, live_preview=True)
    job.tracking_frame.connect(lambda value: delivered.append(value["frame"]))
    job.result.connect(delivered.append)
    job.start()
    assert job.wait(5000)
    app.processEvents()
    assert delivered == [499, "done"]
    assert not job.frames.timer.isActive()


def test_disabled_job_never_injects_preview_into_unrelated_function():
    app, results = _app(), []
    def work(cancel, progress):
        return 7
    job = Job(work)
    assert job.frames is None
    job.result.connect(results.append)
    job.start()
    assert job.wait(5000)
    app.processEvents()
    assert results == [7]


def test_ui_preview_reads_exact_frame_keeps_anchors_and_track_and_ignores_other_clip(clip):
    app = _app()
    from types import SimpleNamespace
    from PySide6.QtWidgets import QCheckBox, QLabel, QSlider
    from lockframe.live_preview_ui import LivePreviewMixin
    from lockframe.models import source_identity
    observed = []
    class Harness(LivePreviewMixin):
        def _update_strip(self):
            pass
    window = Harness()
    window.info = clip.info
    window.anchors = dict(clip.anchors)
    window.track = object()
    saved_track = window.track
    window.active_clip_id = clip.id
    window.reader = SimpleNamespace(read=lambda index: observed.append(index) or demo_frame(index)[0])
    window.canvas = SimpleNamespace(mode="output", set_frames=lambda *args, **kwargs: observed.append(kwargs))
    window.timeline = QSlider()
    window.timeline.setRange(0, clip.info.frames - 1)
    window.preview_note, window.time_label = QLabel(), QLabel()
    window.mesh_overlay_check, window.live_preview_check = QCheckBox(), QCheckBox()
    window.live_preview_check.setChecked(True)
    payload = {"frame": 8, "source_identity": source_identity(clip.info), "box": demo_frame(8)[1],
               "confidence": 0., "state": "lost", "completed": 7, "total": 21}
    window._on_queue_tracking_frame("different-clip", payload)
    assert observed == []
    window._on_queue_tracking_frame(clip.id, payload)
    assert observed[0] == 8
    assert observed[1]["state"] == "lost" and observed[1]["confidence"] == 0.
    assert window.frame_index == window.timeline.value() == 8
    assert window.anchors == clip.anchors and window.track is saved_track
    assert "跟丢" in window.preview_note.text()
    window._on_tracking_frame(dict(payload, source_identity={"different": "source"}))
    assert len(observed) == 2
    window.live_preview_check.setChecked(False)
    window._on_tracking_frame(payload)
    assert len(observed) == 2


def test_queue_job_signal_coalesces_each_clip_and_preserves_ids(monkeypatch):
    app, delivered = _app(), []
    from lockframe import sequence
    def fake_queue(clips, workers, cancel, progress, item_done, directory, on_frame=None,
                   on_item_progress=None, pause_on_lost=False, pause_event=None):
        for index in range(500):
            on_frame("clip-a", {"frame": index})
            on_frame("clip-b", {"frame": index + 1000})
        return {}
    monkeypatch.setattr(sequence, "run_queue", fake_queue)
    job = QueueJob([], 2, None, live_preview=True)
    job.frame_event.connect(lambda clip_id, value: delivered.append((clip_id, value["frame"])))
    job.start()
    assert job.wait(5000)
    app.processEvents()
    assert delivered == [("clip-a", 499), ("clip-b", 1499)]
    assert not job.frames.timer.isActive()


def test_queue_item_stage_signal_is_independent_of_live_preview(monkeypatch):
    application, delivered = _app(), []
    from lockframe import sequence
    def fake_queue(clips, workers, cancel, progress, item_done, directory,
                   on_frame=None, on_item_progress=None, pause_on_lost=False, pause_event=None):
        assert on_frame is None
        on_item_progress("clip-b", .47, "[计算:0.479] 实际匹配")
        return {}
    monkeypatch.setattr(sequence, "run_queue", fake_queue)
    job = QueueJob([], 2, None, live_preview=False)
    job.item_progress.connect(lambda *values: delivered.append(values))
    job.start()
    assert job.wait(5000)
    application.processEvents()
    assert delivered == [("clip-b", .47, "[计算:0.479] 实际匹配")]
    assert job.frames is None


def test_opt_in_queue_follow_previews_next_clip_without_editing_active_workspace(clip, monkeypatch):
    app = _app()
    from types import SimpleNamespace
    from PySide6.QtWidgets import QCheckBox, QLabel, QSlider
    from lockframe import live_preview_ui
    from lockframe.models import source_identity
    opened, displayed, restored = [], [], []
    class Reader:
        def __init__(self, path):
            self.path, self.indices, self.closed = path, [], False
            opened.append(self)
        def read(self, index):
            self.indices.append(index)
            return np.full((12, 20, 3), 99, dtype=np.uint8)
        def close(self):
            self.closed = True
    monkeypatch.setattr(live_preview_ui, "FrameReader", Reader)
    class Harness(live_preview_ui.LivePreviewMixin):
        def _update_strip(self):
            raise AssertionError("A foreign frame must not change the active clip's strip")
    other = Clip(replace(clip.info, path="next-clip.mp4", proxy="next-proxy.mp4"))
    window = Harness()
    window.info, window.frame_index = clip.info, 3
    window.frame = np.zeros((12, 20, 3), dtype=np.uint8)
    original_frame = window.frame
    window.reader = object()
    window.track, window.anchors = object(), dict(clip.anchors)
    original_track = window.track
    window.active_clip_id = clip.id
    window._clip = lambda clip_id: other if clip_id == other.id else clip if clip_id == clip.id else None
    window._seek = restored.append
    window.timeline = QSlider()
    window.timeline.setRange(0, clip.info.frames - 1)
    window.timeline.setValue(3)
    window.canvas = SimpleNamespace(mode="compare", set_frames=lambda *args, **kwargs: displayed.append((args, kwargs)))
    window.preview_note, window.time_label, window.live_preview_status = QLabel(), QLabel(), QLabel()
    window.mesh_overlay_check, window.live_preview_check = QCheckBox(), QCheckBox()
    window.live_preview_check.setChecked(True)
    window._begin_live_preview()
    window._live_queue_follow, window._live_follow_clip_id = True, None
    window._live_completed_ids = {clip.id}
    payload = {"frame": 12, "source_identity": source_identity(other.info), "box": demo_frame(12)[1],
               "state": "tracked", "confidence": .9, "completed": 4, "total": 21}
    window._on_queue_tracking_frame(other.id, payload)
    assert opened[0].path == other.info.proxy and opened[0].indices == [12]
    assert np.all(displayed[0][0][0] == 99)
    assert window.info is clip.info and window.frame is original_frame and window.frame_index == 3
    assert window.timeline.value() == 3 and window.track is original_track and window.anchors == clip.anchors
    assert "next-clip.mp4" in window.live_preview_status.text()
    window._on_queue_tracking_frame(clip.id, payload)
    assert len(displayed) == 1
    window._end_live_preview()
    assert opened[0].closed and window._live_foreign_reader is None
    assert restored == [3] and window.canvas.mode == "compare"
    assert not window._live_queue_follow
