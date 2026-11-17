import threading

import cv2
import numpy as np
import pytest

from lockframe.models import VideoInfo, file_fingerprint
from lockframe.tracking import FeatureTracker, track_video
from lockframe.tracking_pause import TrackingPaused, is_complete
from lockframe.tracking_sections import normalize_sections


SECTIONS = [{"start": 0, "end": 11}, {"start": 12, "end": 23}]


@pytest.fixture
def section_scene(tmp_path, monkeypatch):
    from lockframe import media, performance
    rng = np.random.default_rng(712)
    textures = [cv2.GaussianBlur(rng.integers(15, 240, (48, 54), np.uint8), (3, 3), .6)
                for _ in range(2)]
    images = []
    def box(frame):
        local = frame % 12
        x, y = (25 + local * .65, 28 + local * .15) if frame < 12 else (142 - local * .55, 92 - local * .2)
        return (x / 240, y / 160, 54 / 240, 48 / 160)
    for frame in range(24):
        image = np.full((160, 240), 22, np.uint8)
        x, y, _, _ = np.array(box(frame)) * [240, 160, 240, 160]
        matrix = np.array([[1., 0., x], [0., 1., y]])
        texture = textures[frame // 12]
        warped = cv2.warpAffine(texture, matrix, (240, 160))
        support = cv2.warpAffine(np.full_like(texture, 255), matrix, (240, 160))
        image[support > 254] = warped[support > 254]
        image[0, 0] = frame
        images.append(image)
    source = tmp_path / "独立目标.bin"
    source.write_bytes(b"independent target identity")
    info = VideoInfo(str(source), 240, 160, 30., 24 / 30, frames=24,
                     proxy=str(source), fingerprint=file_fingerprint(str(source)))
    class MemoryCache:
        def __init__(self, path, options, cancel):
            self.peak_bytes = self.hits = 0
        def read(self, index):
            return images[index]
        def iter_frames(self, indices):
            for index in indices:
                yield index, images[index]
        def close(self):
            pass
    monkeypatch.setattr(performance, "GrayFrameCache", MemoryCache)
    monkeypatch.setattr(media, "prepare_precision_video", lambda value, *args: value)
    monkeypatch.setenv("LOCKFRAME_CACHE_DIR", str(tmp_path / "cache"))
    return info, box


def record_steps(monkeypatch, lost_frame=None):
    original = FeatureTracker.step
    calls = []
    def step(self, gray):
        frame = int(gray[0, 0])
        calls.append(frame)
        observation = original(self, gray)
        if frame == lost_frame:
            observation.state, observation.confidence = "lost", 0.
        return observation
    monkeypatch.setattr(FeatureTracker, "step", step)
    return calls


@pytest.mark.parametrize("perspective", [False, True])
def test_distinct_targets_initialize_independently_and_backward_stays_in_section(section_scene, monkeypatch, perspective):
    info, box = section_scene
    calls = record_steps(monkeypatch)
    events = []
    result = track_video(info, {4: box(4), 17: box(17)}, 0, 23, threading.Event(),
                         tracking_sections=list(reversed(SECTIONS)), on_frame=events.append,
                         perspective=perspective)
    assert calls == [3, 2, 1, 0, *range(5, 12), 16, 15, 14, 13, 12, *range(18, 24)]
    assert is_complete(result) and not result.lost_frames
    assert result.performance["tracking_sections"] == SECTIONS
    for frame in range(24):
        expected = np.array(box(frame))[:2] + np.array(box(frame))[2:] / 2
        assert np.linalg.norm((result.positions[frame] - expected) * [240, 160]) < .8
    assert len(events) == 24 and events[-1]["completed"] == 24
    assert not ({11, 12} & set(result.performance["quality"]["motion_review"]))


@pytest.mark.parametrize("mode", ["normal", "precision"])
def test_joint_result_is_exact_concatenation_of_independent_section_runs(section_scene, mode):
    info, box = section_scene
    anchors = {4: box(4), 17: box(17), 21: box(21)}
    kwargs = dict(mode=mode, perspective=True, adaptive_reference=True, confidence_mesh=True)
    combined = track_video(info, anchors, 0, 23, threading.Event(), tracking_sections=SECTIONS, **kwargs)
    for section in SECTIONS:
        separate = track_video(info, anchors, section["start"], section["end"], threading.Event(), **kwargs)
        interval = slice(section["start"], section["end"] + 1)
        assert combined.states[interval] == separate.states
        for name in ("positions", "boxes", "homographies", "confidence", "confidence_grids"):
            np.testing.assert_array_equal(getattr(combined, name)[interval], getattr(separate, name))


@pytest.mark.parametrize("sections, message", [
    ([{"start": 0, "end": 10}, {"start": 12, "end": 23}], "未覆盖"),
    ([{"start": 0, "end": 12}, {"start": 12, "end": 23}], "重叠"),
    ([{"start": 0, "end": 24}], "超出"),
    ([{"start": 1, "end": 23}], "首尾"),
    ([{"start": True, "end": 23}], "无效"),
    ([{"start": 0, "end": 23.0}], "无效"),
])
def test_invalid_topology_fails_before_reading(section_scene, sections, message):
    info, box = section_scene
    with pytest.raises(ValueError, match=message):
        track_video(info, {0: box(0)}, 0, 23, threading.Event(), tracking_sections=sections)


def test_draft_gaps_allowed_but_section_requires_own_anchor(section_scene):
    assert normalize_sections([SECTIONS[1]], 0, 23, require_coverage=False) == [SECTIONS[1]]
    info, box = section_scene
    with pytest.raises(ValueError, match="第 2 段.*尚未设置跟踪目标"):
        track_video(info, {0: box(0)}, 0, 23, threading.Event(), tracking_sections=SECTIONS)


def test_correction_reuses_other_section_and_unchanged_walk(section_scene, monkeypatch):
    info, box = section_scene
    anchors = {0: box(0), 12: box(12), 18: box(18)}
    original = track_video(info, anchors, 0, 23, threading.Event(), tracking_sections=SECTIONS)
    calls = record_steps(monkeypatch)
    anchors[18] = (*np.array(box(18))[:2] + [.001, 0], *box(18)[2:])
    actual = track_video(info, anchors, 0, 23, threading.Event(), tracking_sections=SECTIONS, previous=original)
    assert calls == list(range(19, 24))
    np.testing.assert_array_equal(actual.positions[:18], original.positions[:18])
    assert actual.performance["reused_frames"] == 16


def test_changed_boundaries_invalidate_old_topology(section_scene, monkeypatch):
    info, box = section_scene
    anchors = {0: box(0), 12: box(12)}
    original = track_video(info, anchors, 0, 23, threading.Event(), tracking_sections=SECTIONS)
    calls = record_steps(monkeypatch)
    actual = track_video(info, anchors, 0, 23, threading.Event(), previous=original,
                         tracking_sections=[{"start": 0, "end": 10}, {"start": 11, "end": 23}])
    assert actual.performance["reused_frames"] == 0 and len(calls) == 22


def test_lost_pauses_then_manual_correction_continues_only_own_remaining_walk(section_scene, monkeypatch):
    info, box = section_scene
    calls = record_steps(monkeypatch, lost_frame=15)
    anchors = {0: box(0), 12: box(12)}
    kwargs = dict(tracking_sections=SECTIONS, pause_on_lost=True)
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, anchors, 0, 23, threading.Event(), **kwargs)
    paused = raised.value
    assert calls == [*range(1, 12), 13, 14, 15] and paused.frame == 15
    assert paused.resume["signature"]["tracking_sections"] == SECTIONS
    calls.clear()
    with pytest.raises(TrackingPaused):
        track_video(info, anchors, 0, 23, threading.Event(), previous=paused.result, resume=paused.resume, **kwargs)
    assert not calls
    anchors[15] = box(15)
    result = track_video(info, anchors, 0, 23, threading.Event(), previous=paused.result, resume=paused.resume, **kwargs)
    assert calls == list(range(16, 24)) and is_complete(result) and not result.lost_frames
    np.testing.assert_array_equal(result.positions[:15], paused.result.positions[:15])


def test_checkpoint_resumes_section_and_topology_has_distinct_cache_key(section_scene, monkeypatch):
    info, box = section_scene
    calls = record_steps(monkeypatch)
    anchors = {0: box(0), 12: box(12)}
    pause = threading.Event()
    kwargs = dict(tracking_sections=SECTIONS, segmented_tracking=True, segment_frames=10)
    def observe(event):
        if event["frame"] == 15:
            pause.set()
    with pytest.raises(TrackingPaused):
        track_video(info, anchors, 0, 23, threading.Event(), pause_event=pause, on_frame=observe, **kwargs)
    import json
    from lockframe.media import cache_root
    records = list((cache_root() / "tracking-checkpoints").glob("*.locksequence"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["clips"][0]["section_tracking"] is True
    assert record["clips"][0]["tracking_sections"] == SECTIONS
    calls.clear()
    result = track_video(info, anchors, 0, 23, threading.Event(), **kwargs)
    assert calls == list(range(16, 24)) and result.performance["resumed_frames"] == 16
    calls.clear()
    result = track_video(info, anchors, 0, 23, threading.Event(), segmented_tracking=True, segment_frames=10,
                         tracking_sections=[{"start": 0, "end": 10}, {"start": 11, "end": 23}])
    assert result.performance["resumed_frames"] == 0 and len(calls) == 22


def test_manual_backward_pause_resumes_numeric_state_without_crossing_boundary(section_scene, monkeypatch):
    info, box = section_scene
    anchors = {0: box(0), 17: box(17)}
    kwargs = dict(tracking_sections=SECTIONS, perspective=True, confidence_mesh=True)
    expected = track_video(info, anchors, 0, 23, threading.Event(), **kwargs)
    calls = record_steps(monkeypatch)
    pause = threading.Event()
    def observe(event):
        if event["frame"] == 14:
            pause.set()
    with pytest.raises(TrackingPaused) as raised:
        track_video(info, anchors, 0, 23, threading.Event(), pause_event=pause, on_frame=observe, **kwargs)
    paused = raised.value
    assert calls == [*range(1, 12), 16, 15, 14]
    calls.clear()
    result = track_video(info, anchors, 0, 23, threading.Event(), previous=paused.result,
                         resume=paused.resume, **kwargs)
    assert calls == [13, 12, *range(18, 24)]
    np.testing.assert_array_equal(result.positions, expected.positions)
    np.testing.assert_array_equal(result.homographies, expected.homographies)
    assert is_complete(result)


def test_no_sections_matches_legacy_and_single_frame_sections_need_no_steps(section_scene, monkeypatch):
    info, box = section_scene
    original = track_video(info, {0: box(0)}, 0, 11, threading.Event())
    actual = track_video(info, {0: box(0)}, 0, 11, threading.Event(), tracking_sections=[])
    np.testing.assert_array_equal(actual.positions, original.positions)
    calls = record_steps(monkeypatch)
    sections = [{"start": frame, "end": frame} for frame in range(24)]
    result = track_video(info, {frame: box(frame) for frame in range(24)}, 0, 23,
                         threading.Event(), tracking_sections=sections)
    assert not calls and result.states == ["manual"] * 24
