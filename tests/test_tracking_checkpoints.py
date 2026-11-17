import threading
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from lockframe.media import import_video, resource_root, Cancelled
from lockframe.demo import demo_frame
from lockframe.tracking import track_video
from lockframe.quality import parse_phase, confidence_grid, review_track


@pytest.mark.parametrize("mode", ["normal", "precision"])
def test_checkpoint_resume_preserves_full_pose_grid_and_bidirectional_tracking(mode, tmp_path, monkeypatch):
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    monkeypatch.setenv("LOCKFRAME_CACHE_DIR", str(tmp_path))
    anchors = {55: demo_frame(55)[1], 85: demo_frame(85)[1]}
    options = dict(mode=mode, perspective=True, planar_region=True, adaptive_reference=True, confidence_mesh=True)
    expected = track_video(info, anchors, 0, 100, threading.Event(), **options)
    cancel = threading.Event()
    from lockframe import performance
    original = performance.GrayFrameCache.iter_frames
    def interrupted(cache, indices):
        for frame, pixels in original(cache, indices):
            if frame == 19:
                cancel.set()
            yield frame, pixels
    monkeypatch.setattr(performance.GrayFrameCache, "iter_frames", interrupted)
    with pytest.raises(Cancelled):
        track_video(info, anchors, 0, 100, cancel, segmented_tracking=True, segment_frames=10, **options)
    files = list((tmp_path / "tracking-checkpoints").glob("*.npz"))
    assert len(files) == 1
    from lockframe.sequence import read_sequence
    task = read_sequence(files[0].with_suffix(".locksequence"))[0]
    assert task.anchors == anchors and task.settings.segmented_tracking
    assert task.settings.confidence_mesh and task.settings.perspective
    assert (task.start, task.end) == (0, 100)
    monkeypatch.setattr(performance.GrayFrameCache, "iter_frames", original)
    phases = []
    result = track_video(info, anchors, 0, 100, threading.Event(),
                         progress=lambda v, text: phases.append(parse_phase(text)),
                         segmented_tracking=True, segment_frames=10, **options)
    assert 20 < result.performance["resumed_frames"] < result.count
    assert result.states == expected.states
    np.testing.assert_allclose(result.positions, expected.positions, atol=1e-8)
    np.testing.assert_allclose(result.homographies, expected.homographies, atol=1e-8)
    np.testing.assert_allclose(result.confidence_grids, expected.confidence_grids, atol=1e-6)
    assert {item[0] for item in phases if item} == {"分析", "计算", "检查", "缓存"}
    complete = track_video(info, anchors, 0, 100, threading.Event(),
                           segmented_tracking=True, segment_frames=10, **options)
    assert complete.performance["resumed_frames"] == complete.count
    np.testing.assert_array_equal(complete.homographies, result.homographies)
    changed = track_video(info, anchors, 0, 100, threading.Event(),
                          segmented_tracking=True, segment_frames=10, **dict(options, smoothness=4))
    assert changed.performance["resumed_frames"] == 0


def test_corrupt_checkpoint_is_ignored_and_cache_can_be_cleared(tmp_path, monkeypatch):
    from lockframe.checkpoints import clear_checkpoints
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    monkeypatch.setenv("LOCKFRAME_CACHE_DIR", str(tmp_path))
    args = (info, {5: demo_frame(5)[1]}, 0, 20)
    track_video(*args, threading.Event(), segmented_tracking=True, segment_frames=10)
    file = next((tmp_path / "tracking-checkpoints").glob("*.npz"))
    file.write_bytes(b"broken interrupted data")
    result = track_video(*args, threading.Event(), segmented_tracking=True, segment_frames=10)
    assert result.performance["resumed_frames"] == 0 and not result.lost_frames
    assert clear_checkpoints(info) == 1 and not file.exists()
    assert not file.with_suffix(".locksequence").exists()


def test_grid_project_roundtrip_and_phase_controls(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from lockframe import app
    from lockframe.models import Settings, save_project, read_project
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    anchors = {5: demo_frame(5)[1]}
    result = track_video(info, anchors, 0, 20, threading.Event(), confidence_mesh=True)
    settings = Settings(confidence_mesh=True, segmented_tracking=True, segment_frames=20)
    path = tmp_path / "grid.lockframe"
    save_project(str(path), info, settings, anchors, 0, 20, result)
    raw = read_project(str(path))
    np.testing.assert_array_equal(raw["track"]["confidence_grids"], result.confidence_grids)
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app, "QSettings", lambda *args: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    window = app.MainWindow()
    try:
        window._ready = False
        window._restore_settings(settings)
        window._ready = True
        assert window.settings().confidence_mesh and window.settings().segment_frames == 20
        window._on_progress(.5, "[计算:0.400] test")
        assert window.stage_bars["分析"].value() == 1000
        assert window.stage_bars["计算"].value() == 400
        window._on_progress(.6, "[汇总:检查:0.250]")
        assert window.stage_bars["检查"].value() == 250
        window.canvas.mode = "source"
        frame = np.full((240, 400, 3), 80, np.uint8)
        window.canvas.set_frames(frame, box=(.3,.3,.2,.2), homography=np.array([[.2,0,.3],[0,.2,.3],[0,0,1]]),
                                 confidence_grid=np.full((4,4),.8))
        window.canvas.resize(800, 500)
        assert not window.canvas.grab().isNull()
    finally:
        window.dirty = False
        window.close()


def test_grid_reports_local_occlusion_and_quality_review_never_moves_a_measured_pose():
    import cv2
    from lockframe.reference_tracking import ReferencePatch
    rng = np.random.default_rng(7)
    gray = cv2.GaussianBlur(rng.integers(0, 255, (160, 200), dtype=np.uint8), (3, 3), .6)
    reference = ReferencePatch(gray, [50, 40, 80, 80])
    first = confidence_grid(reference, gray, reference.matrix)
    hidden = gray.copy()
    hidden[40:80, 50:90] = 20
    second = confidence_grid(reference, hidden, reference.matrix)
    assert first.min() > .99
    assert np.max(second[:2, :2]) < .1
    assert np.min(second[2:, 2:]) > .99
    from lockframe.models import TrackResult
    positions = np.tile([.5, .5], (7, 1))
    positions[3] = [.8, .5]
    result = TrackResult(0, 6, positions.copy(), np.tile([.4, .4, .1, .1], (7, 1)),
                         np.full(7, .9), ["tracked"] * 7)
    report = review_track(result)
    assert 3 in report["motion_review"]
    import json
    json.dumps(report)
    np.testing.assert_array_equal(result.positions, positions)


@pytest.mark.parametrize("mode", ["normal", "precision"])
def test_resume_restores_learned_appearance_and_feature_identities(mode, tmp_path, monkeypatch):
    import cv2
    from lockframe.models import VideoInfo, file_fingerprint
    from lockframe import performance
    # Independent changing textures force real adaptive reference updates.
    rng = np.random.default_rng(421)
    patches = []
    for _ in range(2):
        patch = cv2.GaussianBlur(rng.normal(0, 1, (80, 80)).astype(np.float32), (5, 5), .9)
        patches.append(np.clip(patch / patch.std() * 34 + 127, 12, 242).astype(np.uint8))
    images = []
    for index, alpha in enumerate(np.linspace(0, .59, 66)):
        image = np.full((240, 400), 24, np.uint8)
        matrix = np.array([[1., 0., 120 + index * .6], [0., 1., 80 + index * .15]])
        rendered = cv2.warpAffine(cv2.addWeighted(patches[0], 1-alpha, patches[1], alpha, 0), matrix, (400, 240))
        support = cv2.warpAffine(np.full_like(patches[0], 255), matrix, (400, 240))
        image[support > 250] = rendered[support > 250]
        images.append(image)
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic source identity")
    info = VideoInfo(str(source), 400, 240, 30., 2.2, frames=66,
                     proxy=str(source), fingerprint=file_fingerprint(str(source)))
    monkeypatch.setenv("LOCKFRAME_CACHE_DIR", str(tmp_path / "cache"))
    interrupt = [False]
    class MemoryCache:
        def __init__(self, path, options, cancel):
            self.cancel = cancel
            self.peak_bytes = self.hits = 0
        def read(self, index):
            return images[index]
        def iter_frames(self, indices):
            for index in indices:
                if interrupt[0] and index == 37:
                    self.cancel.set()
                yield index, images[index]
        def close(self):
            pass
    monkeypatch.setattr(performance, "GrayFrameCache", MemoryCache)
    if mode == "precision":
        from lockframe import media
        monkeypatch.setattr(media, "prepare_precision_video", lambda value, *args: value)
    options = dict(mode=mode, adaptive_reference=True, confidence_mesh=True)
    anchors = {0: (.3, 1/3, .2, 1/3)}
    expected = track_video(info, anchors, 0, 65, threading.Event(), **options)
    assert expected.performance["adaptation_updates"] >= 3
    interrupt[0] = True
    with pytest.raises(Cancelled):
        track_video(info, anchors, 0, 65, threading.Event(), segmented_tracking=True, segment_frames=10, **options)
    interrupt[0] = False
    resumed = track_video(info, anchors, 0, 65, threading.Event(), segmented_tracking=True, segment_frames=10, **options)
    assert resumed.performance["resumed_frames"] >= 30
    assert resumed.performance["adaptation_updates"] == expected.performance["adaptation_updates"]
    assert resumed.states == expected.states
    np.testing.assert_allclose(resumed.positions, expected.positions, atol=1e-9)
