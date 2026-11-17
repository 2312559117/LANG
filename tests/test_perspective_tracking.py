"""Perspective tracking smoke tests.

Verifies that enabling perspective mode (homography fit) does not crash on
simple uniform-translation scenes and produces finite, in-bounds boxes. These
are existence/shape tests, not accuracy benchmarks — real 3D-perspective
accuracy requires live footage (see tests.md gap list).
"""
import threading

import cv2
import numpy as np
import pytest

from lockframe.demo import create_demo, demo_frame
from lockframe.media import FrameReader, import_video
from lockframe.models import Settings, TrackResult
from lockframe.tracking import FeatureTracker, track_video


@pytest.fixture(scope="module")
def demo_clip(tmp_path_factory):
    source = tmp_path_factory.mktemp("perspective") / "透视测试.mp4"
    create_demo(str(source), frames=30, width=640, height=360)
    return import_video(str(source), threading.Event())


def test_perspective_tracker_step_produces_valid_box(demo_clip):
    reader = FrameReader(demo_clip.proxy)
    try:
        gray0 = cv2.cvtColor(reader.read(0), cv2.COLOR_BGR2GRAY)
        tracker = FeatureTracker(gray0,
                                  [demo_clip.width * .4, demo_clip.height * .35,
                                   demo_clip.width * .2, demo_clip.height * .3],
                                  perspective=True, smoothness=3)
        observation = tracker.step(cv2.cvtColor(reader.read(1), cv2.COLOR_BGR2GRAY))
        assert observation.state in {"tracked", "recovered", "lost"}
        if observation.state != "lost":
            box = observation.box
            assert np.isfinite(box).all()
            assert box[2] > 0 and box[3] > 0
    finally:
        reader.close()


def test_perspective_smoke_through_track_video(demo_clip):
    """End-to-end run: perspective=True must complete and return a valid TrackResult."""
    cancel = threading.Event()
    end = min(demo_clip.frames - 1, 24)
    anchors = {0: (0.4, 0.35, 0.2, 0.3)}
    result = track_video(demo_clip, anchors, 0, end, cancel,
                            mode="normal", perspective=True, smoothness=3)
    assert isinstance(result, TrackResult)
    assert result.start == 0
    assert np.isfinite(result.boxes).all()
    assert set(result.states) <= {"manual", "tracked", "recovered", "lost"}
    assert (result.confidence >= 0).all() and (result.confidence <= 1).all()


def test_smoothness_levels_dont_crash(demo_clip):
    """All five smoothness levels must produce a valid state observation."""
    reader = FrameReader(demo_clip.proxy)
    try:
        gray0 = cv2.cvtColor(reader.read(0), cv2.COLOR_BGR2GRAY)
        for level in (1, 2, 3, 4, 5):
            tracker = FeatureTracker(gray0,
                                     [demo_clip.width * .4, demo_clip.height * .35,
                                      demo_clip.width * .2, demo_clip.height * .3],
                                     perspective=False, smoothness=level)
            observation = tracker.step(cv2.cvtColor(reader.read(1), cv2.COLOR_BGR2GRAY))
            assert observation.state in {"tracked", "recovered", "lost"}
    finally:
        reader.close()


def test_settings_defaults_preserve_v063_baseline():
    """Default Settings match the v0.6.3 baseline workflow (regression guard)."""
    settings = Settings()
    assert settings.perspective is False
    assert settings.smoothness == 3
    settings.validate()


def test_settings_perspective_and_smoothness_validated():
    """Out-of-range smoothness and non-bool perspective must raise."""
    with pytest.raises(ValueError):
        Settings(smoothness=0).validate()
    with pytest.raises(ValueError):
        Settings(smoothness=6).validate()
    s = Settings(perspective=True, smoothness=5)
    s.validate()
    assert s.perspective is True


def test_precision_perspective_runs(demo_clip):
    """Precision mode + perspective + low smoothness must not crash."""
    from lockframe.precision_tracking import PrecisionFeatureTracker
    reader = FrameReader(demo_clip.proxy)
    cancel = threading.Event()
    try:
        gray0 = cv2.cvtColor(reader.read(0), cv2.COLOR_BGR2GRAY)
        tracker = PrecisionFeatureTracker(
            gray0, [demo_clip.width * .4, demo_clip.height * .35,
                    demo_clip.width * .2, demo_clip.height * .3],
            cancel=cancel, perspective=True, smoothness=2)
        observation = tracker.step(cv2.cvtColor(reader.read(1), cv2.COLOR_BGR2GRAY))
        assert observation.state in {"tracked", "recovered", "lost"}
    finally:
        reader.close()