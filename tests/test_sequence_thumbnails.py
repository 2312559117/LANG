import cv2
import numpy as np
import pytest

from lockframe.models import VideoInfo
from lockframe.sequence import Clip, clip_thumbnail


def test_thumbnail_uses_trim_start_and_preserves_aspect(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 40))
    assert writer.isOpened()
    for color in [(0, 0, 240), (0, 240, 0), (240, 0, 0)]:
        writer.write(np.full((40, 80, 3), color, dtype=np.uint8))
    writer.release()
    clip = Clip(VideoInfo(str(path), 80, 40, 10, .3, frames=3), start=1)
    result = cv2.imdecode(np.frombuffer(clip_thumbnail(clip), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert result.shape == (80, 160, 3)
    assert result[:, :, 1].mean() > 220
    assert result[:, :, 0].mean() < 20
    assert clip.start == 1 and clip.frame == 0


def test_thumbnail_releases_reader_on_decode_failure(monkeypatch):
    closed = []
    class Reader:
        def __init__(self, path):
            assert path == "proxy.mp4"
        def read(self, frame):
            raise ValueError("decode failure")
        def close(self):
            closed.append(True)
    monkeypatch.setattr("lockframe.media.FrameReader", Reader)
    clip = Clip(VideoInfo("source.mp4", 80, 40, 10, 1, frames=10, proxy="proxy.mp4"))
    with pytest.raises(ValueError, match="decode failure"):
        clip_thumbnail(clip)
    assert closed == [True]


@pytest.mark.parametrize("width,height", [(0, 90), (160, 0), (1025, 90), (True, 90)])
def test_thumbnail_rejects_unbounded_sizes(width, height):
    clip = Clip(VideoInfo("missing.mp4", 80, 40, 10, 1, frames=10))
    with pytest.raises(ValueError, match="尺寸"):
        clip_thumbnail(clip, width, height)
