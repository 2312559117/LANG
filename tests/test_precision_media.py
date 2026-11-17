from dataclasses import asdict, replace
from pathlib import Path
import shutil
import threading

import pytest

from lockframe import media
from lockframe.media import Cancelled, FrameReader, import_video, prepare_precision_video, run_ffmpeg


@pytest.fixture(scope="module")
def source_clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("precision-source")
    result = {}
    for width, height in ((320, 180), (1920, 1080), (3072, 1728)):
        source = folder / f"实际原片-{width}.mp4"
        run_ffmpeg([
            "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=12",
            "-frames:v", "6", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "12", "-pix_fmt", "yuv420p", str(source),
        ], threading.Event())
        result[width] = import_video(str(source), threading.Event())
    return result


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    cache = tmp_path / "分析缓存"
    cache.mkdir()
    monkeypatch.setattr(media, "cache_root", lambda: cache)
    return cache


@pytest.mark.parametrize("source_width,expected_size", [
    (320, (320, 180)), (1920, (1920, 1080)), (3072, (2560, 1440)),
])
def test_precision_uses_source_detail_without_upscaling(source_clips, isolated_cache,
                                                       source_width, expected_size):
    original = source_clips[source_width]
    saved = asdict(original)
    progress = []
    precise = prepare_precision_video(original, threading.Event(), progress.append)
    reader = FrameReader(precise.proxy)
    normal = FrameReader(original.proxy)
    try:
        first = reader.read(0)
        last = reader.read(5)
        ordinary = normal.read(0)
        assert (first.shape[1], first.shape[0]) == expected_size
        assert first.shape == last.shape
        if source_width > 1280:
            assert first.shape[1] > ordinary.shape[1] == 1280
    finally:
        reader.close()
        normal.close()
    assert precise is not original and asdict(original) == saved
    assert {key: value for key, value in asdict(precise).items() if key != "proxy"} == {
        key: value for key, value in saved.items() if key != "proxy"
    }
    assert precise.proxy != original.proxy
    assert progress[-1] == 1 and all(a <= b for a, b in zip(progress, progress[1:]))
    assert not list(isolated_cache.glob("*.partial.mp4"))


def test_precision_cache_hit_skips_transcoding_and_obeys_cancellation(source_clips,
                                                                     isolated_cache,
                                                                     monkeypatch):
    info = source_clips[320]
    first = prepare_precision_video(info, threading.Event())
    stamp = Path(first.proxy).stat().st_mtime_ns

    def unexpected(*args, **kwargs):
        pytest.fail("a valid precision cache must be reused")

    monkeypatch.setattr(media, "run_ffmpeg", unexpected)
    second = prepare_precision_video(info, threading.Event())
    assert second.proxy == first.proxy and Path(second.proxy).stat().st_mtime_ns == stamp
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(Cancelled):
        prepare_precision_video(info, cancelled)
    assert Path(first.proxy).is_file()
    during_hit = threading.Event()
    with pytest.raises(Cancelled):
        prepare_precision_video(info, during_hit, lambda _: during_hit.set())
    assert Path(first.proxy).is_file()


def test_cancelled_transcode_removes_its_unique_partial_only(source_clips, isolated_cache,
                                                           monkeypatch):
    other_job = isolated_cache / "precision-other.000.partial.mp4"
    other_job.write_bytes(b"another app instance")
    attempts = []

    def interrupted(args, cancel, *unused):
        path = Path(args[-1])
        attempts.append(path)
        path.write_bytes(b"unfinished")
        cancel.set()
        raise Cancelled()

    monkeypatch.setattr(media, "run_ffmpeg", interrupted)
    for _ in range(2):
        with pytest.raises(Cancelled):
            prepare_precision_video(source_clips[320], threading.Event())
    assert attempts[0] != attempts[1]
    assert not any(path.exists() for path in attempts)
    assert list(isolated_cache.iterdir()) == [other_job]


def test_wrong_frame_count_is_rejected_before_cache_is_published(source_clips, isolated_cache,
                                                               monkeypatch):
    def wrong_length(args, *unused):
        shutil.copy2(source_clips[320].proxy, args[-1])

    monkeypatch.setattr(media, "run_ffmpeg", wrong_length)
    with pytest.raises(ValueError, match="帧数"):
        prepare_precision_video(replace(source_clips[320], frames=7), threading.Event())
    assert not list(isolated_cache.iterdir())


def test_precision_rejects_changed_source_and_invalid_hdr_flag(source_clips, isolated_cache, tmp_path):
    info = source_clips[320]
    with pytest.raises(ValueError, match="HDR"):
        prepare_precision_video(replace(info, hdr=True), threading.Event())
    changed = tmp_path / "被修改.mp4"
    shutil.copy2(info.path, changed)
    with changed.open("ab") as stream:
        stream.write(b"source changed after import")
    with pytest.raises(ValueError, match="原视频已经被修改"):
        prepare_precision_video(replace(info, path=str(changed)), threading.Event())
    assert not list(isolated_cache.iterdir())


def test_invalid_existing_precision_cache_is_rebuilt(source_clips, isolated_cache):
    info = source_clips[320]
    first = prepare_precision_video(info, threading.Event())
    cached = Path(first.proxy)
    cached.write_bytes(b"incomplete cache")
    rebuilt = prepare_precision_video(info, threading.Event())
    assert rebuilt.proxy == first.proxy
    reader = FrameReader(rebuilt.proxy)
    try:
        assert reader.read(5).shape[:2] == (180, 320)
    finally:
        reader.close()
    assert not list(isolated_cache.glob("*.partial.mp4"))


def test_source_change_during_transcode_is_rejected(source_clips, isolated_cache,
                                                   tmp_path, monkeypatch):
    from lockframe.models import file_fingerprint

    source = tmp_path / "转码期间被改动.mp4"
    shutil.copy2(source_clips[320].path, source)
    info = replace(source_clips[320], path=str(source), fingerprint=file_fingerprint(str(source)))

    def source_changed(args, *unused):
        shutil.copy2(source_clips[320].proxy, args[-1])
        with source.open("ab") as stream:
            stream.write(b"changed during analysis")

    monkeypatch.setattr(media, "run_ffmpeg", source_changed)
    with pytest.raises(ValueError, match="原视频已经被修改"):
        prepare_precision_video(info, threading.Event())
    assert not list(isolated_cache.iterdir())
