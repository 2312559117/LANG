from copy import deepcopy

import pytest

from lockframe.models import Settings, VideoInfo
from lockframe.sequence import Clip
from lockframe.tracking_pause import reset_clip_tracking


def make_clip(path):
    return Clip(VideoInfo(str(path), 100, 80, 25, 4, frames=100),
                Settings(smoothing=.8), {5: (.1, .1, .2, .2)}, 5, 90,
                frame=35, project_path="saved.locksequence")


@pytest.mark.parametrize("paused", [False, True])
def test_reset_discards_all_reusable_state_only_for_requested_clip(tmp_path, monkeypatch, paused):
    from lockframe import checkpoints
    monkeypatch.setattr(checkpoints, "cache_root", lambda: tmp_path / "cache")
    clip = make_clip(tmp_path / "one.mp4")
    other = make_clip(tmp_path / "two.mp4")
    clip.track = object()
    clip.retrack_previous = object()
    clip.tracking_resume = {"result": object()} if paused else None
    clip.status = "已暂停" if paused else "已完成"
    preserved = (clip.info, deepcopy(clip.settings), clip.start, clip.end,
                 clip.frame, clip.project_path, clip.id)
    path, _ = checkpoints.checkpoint_path(clip.info, clip.anchors, 5, 90, {})
    other_path, _ = checkpoints.checkpoint_path(other.info, other.anchors, 5, 90, {})
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old result")
    path.with_suffix(".locksequence").write_bytes(b"old targets")
    other_path.write_bytes(b"other clip")
    old_anchors = clip.anchors
    assert reset_clip_tracking(clip) == 1
    assert clip.anchors == {} and old_anchors
    assert clip.track is clip.retrack_previous is clip.tracking_resume is None
    assert clip.dirty and clip.status == "" and not clip.ready
    assert preserved == (clip.info, clip.settings, clip.start, clip.end,
                         clip.frame, clip.project_path, clip.id)
    assert not path.exists() and not path.with_suffix(".locksequence").exists()
    assert other_path.read_bytes() == b"other clip" and other.anchors and not other.dirty
    assert reset_clip_tracking(clip) == 0
    clip.anchors[35] = (.2, .2, .3, .3)
    assert clip.ready and clip.track is None and clip.retrack_previous is None


def test_reset_reports_cache_failure_without_clearing_clip(tmp_path, monkeypatch):
    def fail(info):
        raise OSError("cache unavailable")
    monkeypatch.setattr("lockframe.checkpoints.clear_checkpoints", fail)
    clip = make_clip(tmp_path / "one.mp4")
    with pytest.raises(OSError, match="cache unavailable"):
        reset_clip_tracking(clip)
    assert clip.anchors and not clip.dirty
