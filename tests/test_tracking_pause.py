import pickle
import threading

import cv2
import numpy as np
import pytest

from lockframe.models import VideoInfo, file_fingerprint
from lockframe.tracking import FeatureTracker, track_video
from lockframe.tracking_pause import TrackingPaused, computed_mask, is_complete


@pytest.fixture
def scene(tmp_path, monkeypatch):
    """Real trackers on measured synthetic pixels, with deterministic frame IDs."""
    from lockframe import media, performance
    rng = np.random.default_rng(712)
    texture = cv2.GaussianBlur(rng.integers(15, 240, (64, 72), np.uint8), (3, 3), .6)
    images = []
    for index in range(25):
        image = np.full((160, 240), 22, np.uint8)
        matrix = np.array([[1., 0., 60 + index * .65], [0., 1., 48 + index * .15]])
        warped = cv2.warpAffine(texture, matrix, (240, 160))
        support = cv2.warpAffine(np.full_like(texture, 255), matrix, (240, 160))
        image[support > 254] = warped[support > 254]
        image[0, 0] = index
        images.append(image)
    source = tmp_path / "原始素材.bin"
    source.write_bytes(b"synthetic tracking identity")
    info = VideoInfo(str(source), 240, 160, 30., 25/30, frames=25,
                     proxy=str(source), fingerprint=file_fingerprint(str(source)))
    caches = []
    class MemoryCache:
        def __init__(self, path, options, cancel):
            self.closed = False
            self.peak_bytes = self.hits = 0
            caches.append(self)
        def read(self, index):
            return images[index]
        def iter_frames(self, indices):
            for index in indices:
                yield index, images[index]
        def close(self):
            self.closed = True
    monkeypatch.setattr(performance, "GrayFrameCache", MemoryCache)
    monkeypatch.setattr(media, "prepare_precision_video", lambda value, *args: value)
    monkeypatch.setenv("LOCKFRAME_CACHE_DIR", str(tmp_path / "cache"))
    def box(frame):
        return ((60 + frame * .65) / 240, (48 + frame * .15) / 160, .3, .4)
    return info, box, caches


def _record_steps(monkeypatch, mode, lost_frame=None):
    from lockframe.precision_tracking import PrecisionFeatureTracker
    tracker_type = PrecisionFeatureTracker if mode == "precision" else FeatureTracker
    original = tracker_type.step
    calls = []
    def step(self, gray):
        calls.append(int(gray[0, 0]))
        observation = original(self, gray)
        if int(gray[0, 0]) == lost_frame:
            observation.state, observation.confidence = "lost", 0.
        return observation
    monkeypatch.setattr(tracker_type, "step", step)
    return calls


@pytest.mark.parametrize("mode", ["normal", "precision"])
def test_lost_stops_computation_immediately_and_exception_is_pickle_safe(scene, monkeypatch, mode):
    info, box, caches = scene
    calls = _record_steps(monkeypatch, mode, lost_frame=2)
    observations = []
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, {5: box(5)}, 0, 24, threading.Event(), mode=mode,
                    perspective=True, confidence_mesh=True, pause_on_lost=True,
                    on_frame=observations.append)
    paused = raised.value
    assert calls == [4, 3, 2]  # No subsequent matching and no forward walk.
    assert paused.frame == 2 and paused.direction == -1 and paused.reason == "lost"
    assert paused.done_count == 4 and not is_complete(paused.result)
    assert np.flatnonzero(computed_mask(paused.result)).tolist() == [2, 3, 4, 5]
    assert observations[-1]["frame"] == 2 and observations[-1]["state"] == "lost"
    assert paused.result.states[2] == "lost" and paused.result.confidence[2] == 0
    paused.result.validate(info.frames)
    assert all(cache.closed for cache in caches)
    restored = pickle.loads(pickle.dumps(paused))
    assert restored.metadata == paused.metadata and restored.done_count == paused.done_count
    np.testing.assert_array_equal(restored.resume["arrays"]["reference_image"],
                                  paused.resume["arrays"]["reference_image"])


def test_default_tracking_retains_legacy_full_range_behavior(scene, monkeypatch):
    info, box, _ = scene
    calls = _record_steps(monkeypatch, "normal", lost_frame=2)
    result = track_video(info, {5: box(5)}, 0, 24, threading.Event())
    assert len(calls) == 24 and is_complete(result)
    assert 2 in result.lost_frames


@pytest.mark.parametrize("mode", ["normal", "precision"])
def test_manual_pause_resume_reuses_numeric_tracker_without_replaying_prefix(scene, monkeypatch, mode):
    info, box, caches = scene
    anchors = {0: box(0)}
    options = dict(mode=mode, perspective=True, adaptive_reference=True, confidence_mesh=True)
    expected = track_video(info, anchors, 0, 24, threading.Event(), **options)
    calls = _record_steps(monkeypatch, mode)
    pause = threading.Event()
    def observe(value):
        if value["frame"] == 7:
            pause.set()
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, anchors, 0, 24, threading.Event(), pause_event=pause,
                    on_frame=observe, **options)
    suspended = raised.value
    assert calls == list(range(1, 8)) and suspended.reason == "manual"
    assert suspended.frame == 7 and suspended.done_count == 8
    calls.clear()
    actual = track_video(info, anchors, 0, 24, threading.Event(), previous=suspended.result,
                         resume=suspended.resume, pause_event=threading.Event(), **options)
    assert calls == list(range(8, 25))
    assert is_complete(actual) and actual.performance["reused_frames"] == 7
    assert actual.states == expected.states
    np.testing.assert_allclose(actual.positions, expected.positions, atol=1e-9)
    np.testing.assert_allclose(actual.homographies, expected.homographies, atol=1e-9)
    np.testing.assert_allclose(actual.confidence_grids, expected.confidence_grids, atol=1e-6)
    assert all(cache.closed for cache in caches)


def test_lost_requires_correction_and_new_anchor_reuses_only_unchanged_dependency(scene, monkeypatch):
    info, box, _ = scene
    calls = _record_steps(monkeypatch, "normal", lost_frame=7)
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, {0: box(0)}, 0, 24, threading.Event(), pause_on_lost=True)
    suspended = raised.value
    calls.clear()
    with pytest.raises(TrackingPaused) as retry:
        track_video(info, {0: box(0)}, 0, 24, threading.Event(), pause_on_lost=True,
                    previous=suspended.result, resume=suspended.resume)
    assert calls == [] and retry.value.frame == 7
    actual = track_video(info, {0: box(0), 7: box(7)}, 0, 24, threading.Event(),
                         previous=suspended.result, resume=suspended.resume, pause_on_lost=True)
    assert calls == list(range(8, 25)) and is_complete(actual)
    assert actual.states[7] == "manual" and not actual.lost_frames
    np.testing.assert_array_equal(actual.positions[1:7], suspended.result.positions[1:7])


def test_backward_correction_recomputes_changed_reference_and_finishes_other_walks(scene, monkeypatch):
    info, box, _ = scene
    calls = _record_steps(monkeypatch, "normal", lost_frame=8)
    anchors = {12: box(12), 20: box(20)}
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, anchors, 0, 24, threading.Event(), pause_on_lost=True)
    suspended = raised.value
    assert calls == [11, 10, 9, 8]
    calls.clear()
    anchors[8] = box(8)
    result = track_video(info, anchors, 0, 24, threading.Event(), pause_on_lost=True,
                         previous=suspended.result, resume=suspended.resume)
    assert calls == [7, 6, 5, 4, 3, 2, 1, 0, 9, 10, 11, *range(13, 20), *range(21, 25)]
    assert result.states[8] == "manual" and is_complete(result) and not result.lost_frames


def test_segmented_checkpoint_persists_pause_and_resumes_exact_manual_boundary(scene, monkeypatch):
    info, box, _ = scene
    calls = _record_steps(monkeypatch, "normal")
    pause = threading.Event()
    def observe(value):
        if value["frame"] == 11:
            pause.set()
    with pytest.raises(TrackingPaused):
        track_video(info, {0: box(0)}, 0, 24, threading.Event(), pause_event=pause,
                    on_frame=observe, segmented_tracking=True, segment_frames=10)
    calls.clear()
    result = track_video(info, {0: box(0)}, 0, 24, threading.Event(),
                         segmented_tracking=True, segment_frames=10)
    assert calls == list(range(12, 25)) and result.performance["resumed_frames"] == 12
    assert is_complete(result)


def test_lost_checkpoint_cannot_silently_skip_failed_frame(scene, monkeypatch):
    info, box, _ = scene
    calls = _record_steps(monkeypatch, "normal", lost_frame=11)
    options = dict(segmented_tracking=True, segment_frames=10, pause_on_lost=True)
    with pytest.raises(TrackingPaused):
        track_video(info, {0: box(0)}, 0, 24, threading.Event(), **options)
    calls.clear()
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, {0: box(0)}, 0, 24, threading.Event(), **options)
    assert calls == [] and raised.value.frame == 11


def test_pause_during_precision_preparation_stops_subprocess_before_matching(scene, monkeypatch):
    from lockframe import media
    from lockframe.quality import parse_phase
    info, box, caches = scene
    pause = threading.Event()
    def preparation(value, cancel, progress):
        assert not cancel.is_set()
        progress(.36)
        pause.set()
        assert cancel.is_set() and cancel.wait(0)
        raise media.Cancelled()
    monkeypatch.setattr(media, "prepare_precision_video", preparation)
    calls = _record_steps(monkeypatch, "precision")
    phases = []
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, {0: box(0)}, 0, 24, threading.Event(), mode="precision", pause_event=pause,
                    progress=lambda value, detail: phases.append(parse_phase(detail)))
    assert calls == [] and raised.value.reason == "manual" and raised.value.done_count == 1
    assert raised.value.metadata["stage"] == "分析"
    assert raised.value.metadata["phase_fraction"] == pytest.approx(.324)
    assert all(stage == "分析" and fraction <= .324 for stage, fraction, _ in phases)
    assert all(cache.closed for cache in caches)
