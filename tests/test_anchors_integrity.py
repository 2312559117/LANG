"""Regression tests for incremental re-tracking and TrackResult.anchors integrity.

Covers the P1-1/P1-2 gaps from the code review:
- save_project now serializes track.anchors (tracker_revision>=4)
- TrackResult.validate rejects malformed anchors
- re-tracking with a NEW anchor inside a previously-clean interval
  recomputes the affected interval instead of silently reusing stale results
"""
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from lockframe.demo import create_demo, demo_frame
from lockframe.media import import_video
from lockframe.models import Settings, TrackResult, VideoInfo, read_project, save_project
from lockframe.tracking import FeatureTracker, track_video


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    source = tmp_path_factory.mktemp("anchors") / "anchors.mp4"
    create_demo(str(source), frames=40, width=480, height=270)
    return import_video(str(source), threading.Event())


def _anchor(info, frame):
    return demo_frame(min(frame, info.frames - 1), info.width, info.height)[1]


def test_track_result_rejects_anchor_outside_interval():
    n = 10
    result = TrackResult(
        start=0, end=n - 1,
        positions=np.zeros((n, 2)), boxes=np.full((n, 4), 0.5),
        confidence=np.zeros(n), states=["tracked"] * n,
        anchors={15: (0.4, 0.4, 0.2, 0.2)},  # frame 15 outside [0, 9]
    )
    with pytest.raises(ValueError, match="不在跟踪区间内"):
        result.validate(n)


def test_track_result_rejects_malformed_anchor_box():
    n = 5
    result = TrackResult(
        start=0, end=n - 1,
        positions=np.zeros((n, 2)), boxes=np.full((n, 4), 0.5),
        confidence=np.zeros(n), states=["tracked"] * n,
        anchors={2: (0.4, 0.4, 0.2)},  # 3-tuple, not 4
    )
    with pytest.raises(ValueError, match="格式无效"):
        result.validate(n)


def test_track_result_rejects_anchor_out_of_bounds_box():
    n = 5
    result = TrackResult(
        start=0, end=n - 1,
        positions=np.zeros((n, 2)), boxes=np.full((n, 4), 0.5),
        confidence=np.zeros(n), states=["tracked"] * n,
        anchors={2: (0.9, 0.9, 0.5, 0.5)},  # x+w > 1.001
    )
    with pytest.raises(ValueError, match="超出画面"):
        result.validate(n)


def test_track_result_accepts_valid_anchors():
    n = 5
    result = TrackResult(
        start=0, end=n - 1,
        positions=np.zeros((n, 2)), boxes=np.full((n, 4), 0.5),
        confidence=np.zeros(n), states=["tracked"] * n,
        anchors={0: (0.4, 0.4, 0.2, 0.2), 4: (0.3, 0.3, 0.15, 0.15)},
    )
    result.validate(n)  # no exception


def test_save_project_serializes_track_anchors(tmp_path, clip):
    """P1-1: save_project must write track.anchors into the project payload."""
    cancel = threading.Event()
    anchor = _anchor(clip, 5)
    track = track_video(clip, {5: anchor}, 0, min(clip.frames - 1, 20), cancel)
    assert track.anchors is not None and 5 in track.anchors

    project_path = tmp_path / "test.lockframe"
    save_project(str(project_path), clip, Settings(), {5: list(anchor)},
                 0, min(clip.frames - 1, 20), track)

    payload = json.loads(project_path.read_text(encoding="utf-8"))
    assert payload["tracker_revision"] == 4
    assert payload["track"]["anchors"] is not None
    assert "5" in payload["track"]["anchors"]


def test_read_project_round_trips_track_anchors(tmp_path, clip):
    """Full round-trip: save → read → TrackResult.anchors equals the saved value."""
    cancel = threading.Event()
    anchor = _anchor(clip, 5)
    end = min(clip.frames - 1, 20)
    track = track_video(clip, {5: anchor}, 0, end, cancel)

    project_path = tmp_path / "round.lockframe"
    save_project(str(project_path), clip, Settings(), {5: list(anchor)}, 0, end, track)
    payload = read_project(str(project_path))
    assert payload["track"]["anchors"] is not None
    restored_anchors = {int(k): tuple(v) for k, v in payload["track"]["anchors"].items()}
    assert restored_anchors[5] == pytest.approx(list(anchor), abs=1e-6)


def test_retrack_with_new_anchor_recomputes_affected_interval(clip, monkeypatch):
    """P1-2 gap: adding a NEW anchor in a previously-clean interval must recompute.

    Previously (pre-P1-1) the reuse check compared self.anchors against
    previous.anchors. With track.anchors serialized, the check stays correct
    when anchors diverge — adding anchor@frame 12 in [0..30] (anchor@5 only)
    forces the interval around frame 12 to be recomputed.
    """
    cancel = threading.Event()
    anchor5 = _anchor(clip, 5)
    end = min(clip.frames - 1, 30)
    original = track_video(clip, {5: anchor5}, 0, end, cancel)
    assert not original.lost_frames

    original_step = FeatureTracker.step
    calls = []

    def counting_step(self, gray):
        calls.append(gray)
        return original_step(self, gray)

    monkeypatch.setattr(FeatureTracker, "step", counting_step)
    anchor12 = _anchor(clip, 12)
    result = track_video(clip, {5: anchor5, 12: anchor12}, 0, end, cancel,
                           previous=original)
    assert not result.lost_frames
    # Without the new anchor, all frames would be reused (0 step calls).
    # With a new anchor at 12, intervals 5..11 and 12..30 must be recomputed.
    # Interval 5..11 = 7 frames; 12..30 = 18 frames; total = 25 steps.
    # (Plus backward interval 0..4 stays reused because anchor 5 didn't change.)
    assert len(calls) > 0, "新增关键帧未触发重算（增量复用判定失效）"
    assert len(calls) <= 30, "新增关键帧导致全量重算（复用失效）"


def test_retrack_with_identical_anchors_reuses_everything(clip, monkeypatch):
    """Sanity: re-tracking with unchanged anchors and no lost frames must reuse all."""
    cancel = threading.Event()
    anchor5 = _anchor(clip, 5)
    end = min(clip.frames - 1, 25)
    original = track_video(clip, {5: anchor5}, 0, end, cancel)
    assert not original.lost_frames

    calls = []
    original_step = FeatureTracker.step

    def counting_step(self, gray):
        calls.append(gray)
        return original_step(self, gray)

    monkeypatch.setattr(FeatureTracker, "step", counting_step)
    result = track_video(clip, {5: anchor5}, 0, end, cancel, previous=original)
    assert not result.lost_frames
    assert len(calls) == 0, "未变化但仍重算（复用失效）"
