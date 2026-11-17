import os
import threading
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
import numpy as np
import pytest

from lockframe import performance as perf
from lockframe.encoding import HardwareEncodingError
from lockframe.media import Cancelled
from lockframe.models import Settings, TrackResult
from lockframe.reference_tracking import ReferencePatch


def test_memory_budget_reserves_working_space_and_never_fills_machine():
    gib = 1024**3
    stats = {"total": 16*gib, "used": 10*gib, "available": 6*gib}
    assert perf.cache_budget(80, stats) == int(2.8*gib - 384*1024**2)
    stats.update(used=14*gib, available=2*gib)
    assert perf.cache_budget(80, stats) == 0
    with pytest.raises(ValueError):
        Settings(memory_percent=81).validate()


def test_prefetch_order_cache_reuse_pressure_and_cancel(monkeypatch):
    reader = Mock()
    reader.read.side_effect = lambda i: np.full((40, 50, 3), i, np.uint8)
    monkeypatch.setattr(perf, "FrameReader", lambda path: reader)
    memory = {"total": 2*1024**3, "used": 0, "available": 2*1024**3}
    monkeypatch.setattr(perf, "memory_status", lambda: memory)
    cancel = threading.Event()
    cache = perf.GrayFrameCache("unused", perf.ComputeOptions(), cancel)
    assert [int(frame[0, 0]) for _, frame in cache.iter_frames(range(5, -1, -1))] == list(range(5, -1, -1))
    cache.read(3)
    assert cache.hits == 1 and reader.read.call_count == 6
    memory.update(used=2*1024**3, available=0)
    cache.last_pressure_check = 0
    cache.read(2)
    assert cache.bytes == 0 and not cache.items
    cancel.set()
    with pytest.raises(Cancelled):
        list(cache.iter_frames(range(5)))
    assert not any(t.name == "tomato-frame-prefetch" for t in threading.enumerate())
    cache.close()
    reader.close.assert_called_once()


def test_gpu_masked_matching_agrees_with_cpu_and_rejects_uniform_windows():
    rng = np.random.default_rng(712)
    search = rng.integers(25, 240, (390, 410), dtype=np.uint8)
    template = search[125:189, 231:295].copy()
    mask = np.zeros_like(template)
    cv2.ellipse(mask, (32, 32), (25, 30), 20, 0, 360, 255, -1)
    matcher = perf.TemplateMatcher("gpu")
    if not matcher.available:
        pytest.skip("No compatible OpenCL GPU on this machine")
    scores = matcher.scores(search, template, mask)
    expected = perf.TemplateMatcher("cpu").scores(search, template, mask)
    assert matcher.gpu_calls == 1, matcher.reason
    assert cv2.minMaxLoc(scores)[3] == (231, 125)
    np.testing.assert_allclose(scores, expected, atol=1e-4)
    blank = matcher.scores(np.full_like(search, 52), template, mask)
    assert np.max(blank) < .1


def test_gpu_failure_retries_with_cpu(monkeypatch):
    image = np.random.default_rng(19).integers(0, 255, (80, 80), dtype=np.uint8)
    matcher = perf.TemplateMatcher("gpu")
    matcher.available = True
    def fail(*args):
        raise cv2.error("Unsupported OpenCL driver")
    monkeypatch.setattr(cv2, "UMat", fail)
    scores = matcher.scores(image, image[20:40, 30:50], np.full((20, 20), 255, np.uint8))
    assert cv2.minMaxLoc(scores)[3] == (30, 20)
    assert matcher.cpu_calls == 1 and not matcher.available


@pytest.mark.parametrize("shape", ["circle", "ellipse"])
def test_feature_detection_obeys_shape_mask(shape):
    image = np.random.default_rng(93).integers(0, 255, (140, 160), dtype=np.uint8)
    reference = ReferencePatch(image, [30, 30, 80, 70], shape=shape)
    assert reference.mask[0, 0] == 0 and reference.mask[35, 40] == 255
    xy = np.round(reference.points.reshape(-1, 2)).astype(int)
    assert len(xy) > 10
    assert np.all(reference.mask[xy[:, 1], xy[:, 0]] > 0)


@pytest.mark.parametrize("overwrite", [True, False])
def test_hardware_encoder_fallback_keeps_user_destination(monkeypatch, overwrite):
    from lockframe import export as module
    monkeypatch.setattr(module, "select_encoder", lambda *args: "h264_nvenc")
    calls = []
    def render(*args, **kwargs):
        assert kwargs["overwrite"] == overwrite
        calls.append(args[-1])
        if args[-1] != "libx264":
            raise HardwareEncodingError("driver lost")
        return {"path": args[3], "encoder": args[-1]}
    monkeypatch.setattr(module, "_export_video", render)
    track = TrackResult(0, 0, np.array([[.5, .5]]), np.array([[.4, .4, .2, .2]]),
                        np.array([1.]), ["manual"])
    result = module.export_video(None, track, Settings(), "same-path.mp4", threading.Event(), overwrite=overwrite)
    assert calls == ["h264_nvenc", "libx264"]
    assert result["path"] == "same-path.mp4" and result["hardware_fallback"]


def test_encoder_probe_requires_real_success_and_cpu_never_probes(monkeypatch):
    from lockframe import encoding
    encoding._verified.clear()
    monkeypatch.setattr(encoding, "find_tool", lambda name: "probe-test-ffmpeg")
    run = Mock(side_effect=[Mock(returncode=1), Mock(returncode=1), Mock(returncode=0)])
    monkeypatch.setattr(encoding.subprocess, "run", run)
    assert encoding.select_encoder("cpu", threading.Event()) == "libx264"
    run.assert_not_called()
    assert encoding.select_encoder("auto", threading.Event()) == "h264_qsv"
    assert run.call_count == 3
    encoding._verified.clear()
