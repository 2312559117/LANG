import threading

from lockframe.demo import demo_frame
from lockframe.media import import_video, resource_root
from lockframe.tracking import FeatureTracker, track_video


def test_retrack_only_recomputes_the_lost_interval(monkeypatch):
    cancel = threading.Event()
    info = import_video(str(resource_root() / "assets" / "demo.mp4"), cancel)
    anchor = demo_frame(min(55, info.frames - 1))[1]
    original = track_video(info, {55: anchor}, 0, info.frames - 1, cancel)
    for frame in (80, 81, 82):
        original.states[frame] = "lost"
        original.confidence[frame] = 0

    original_step = FeatureTracker.step
    calls = []

    def counting_step(self, gray):
        calls.append(gray)
        return original_step(self, gray)

    monkeypatch.setattr(FeatureTracker, "step", counting_step)
    result = track_video(info, {55: anchor}, 0, info.frames - 1, cancel,
                         previous=original)
    assert not result.lost_frames
    # Only the forward interval containing the lost frames is re-tracked:
    # frames 56..179 = 124 steps, while the backward interval 0..54 is reused.
    assert len(calls) == 124
