"""A retained stale trajectory may be reviewed/saved, but never exported as current."""
from copy import deepcopy
from dataclasses import replace
import json
import pickle
from pathlib import Path
import queue
import threading

import numpy as np
import pytest

from lockframe.batch_export import prepare_exports, run_exports, validate_clip
from lockframe.models import Settings, TrackResult, VideoInfo, file_fingerprint
from lockframe.regions import BOX_CORNERS
from lockframe.sequence import Clip, save_sequence, read_sequence


TRIANGLE = ((0., 0.), (1., 0.), (.5, 1.))
OTHER_TRIANGLE = ((0., 0.), (1., 0.), (.7, 1.))


@pytest.fixture
def clip(tmp_path):
    source = tmp_path / "真实路径占位_不解码.mp4"
    source.write_bytes(b"source identity fixture")
    info = VideoInfo(str(source), 640, 360, 25., .08, frames=2,
                     fingerprint=file_fingerprint(source))
    box = (.3, .3, .2, .2)
    track = TrackResult(0, 1, np.array([[.4, .4], [.4, .4]]), np.array([box, box]),
                        np.ones(2), ["manual", "tracked"], anchors={0: box})
    return Clip(info, Settings(acceleration="cpu", resolution=0), {0: box}, 0, 1, track)


def polygon(clip, points=TRIANGLE):
    clip = deepcopy(clip)
    clip.settings.target_shape, clip.settings.polygon_points = "polygon", points
    clip.track.performance.update(shape="polygon", polygon_points=points)
    return clip


def test_mixed_batch_skips_shape_vertices_and_anchors_but_runs_others(clip, tmp_path, monkeypatch):
    from lockframe import batch_export
    stale_shape = replace(clip, settings=replace(clip.settings, target_shape="ellipse"), id="shape")
    stale_vertices = polygon(clip)
    stale_vertices.id = "vertices"
    stale_vertices.settings.polygon_points = OTHER_TRIANGLE
    stale_anchor = replace(clip, anchors={0: (.35, .3, .2, .2)}, id="anchor")
    valid_polygon = polygon(clip)
    valid_polygon.id = "polygon"
    tasks = prepare_exports([clip, stale_shape, stale_vertices, stale_anchor, valid_polygon], tmp_path / "导出")
    assert [bool(task.reason) for task in tasks] == [False, True, True, True, False]
    assert "区域形状或顶点" in tasks[1].reason and "过期" in tasks[2].reason
    assert "关键帧已修改" in tasks[3].reason
    rendered = []
    def render(info, track, settings, destination, *args, **kwargs):
        rendered.append(settings.target_shape)
        return {"path": destination}
    monkeypatch.setattr(batch_export, "export_video", render)
    results = run_exports(tasks, threading.Event())
    assert [item["state"] for item in results] == ["完成", "跳过", "跳过", "跳过", "完成"]
    assert rendered == ["rectangle", "polygon"]


def test_material_summary_distinguishes_stale_region_and_keeps_pause_priority(clip):
    assert clip.summary == "已完成"
    clip.settings.target_shape = "ellipse"
    assert clip.summary == "区域已修改 · 待重跟踪"
    clip = polygon(clip)
    clip.settings.polygon_points = OTHER_TRIANGLE
    assert clip.summary == "区域已修改 · 待重跟踪"
    clip.track.performance.update(complete=False, tracking_pause={"frame": 1})
    assert clip.summary == "已暂停 · 修正第 2 帧后继续"
    clip.track.performance["complete"] = True
    clip.settings.polygon_points = TRIANGLE
    clip.anchors[0] = (.1, .1, .2, .2)
    assert clip.summary == "区域已修改 · 待重跟踪"
    clip.status = "读取失败：原片缺失"
    assert clip.summary == "读取失败：原片缺失"


@pytest.mark.parametrize("result_shape,setting_shape", [("circle", "ellipse"), ("ellipse", "circle")])
def test_circle_ellipse_are_equivalent_in_batch_and_direct_export(clip, tmp_path, monkeypatch,
                                                               result_shape, setting_shape):
    from lockframe import export
    clip.track.performance["shape"] = result_shape
    clip.settings.target_shape = setting_shape
    validate_clip(clip)
    monkeypatch.setattr(export, "select_encoder", lambda *args: "libx264")
    monkeypatch.setattr(export, "_export_video", lambda *args, **kwargs: {"path": args[3]})
    result = export.export_video(clip.info, clip.track, clip.settings, str(tmp_path / "result.mp4"), threading.Event())
    assert result["path"].endswith("result.mp4")


@pytest.mark.parametrize("change", ["shape", "vertices", "missing-provenance"])
def test_direct_api_rejects_stale_shape_before_encoder_or_output_io(clip, tmp_path, monkeypatch, change):
    from lockframe import export
    if change == "shape":
        clip.settings.target_shape = "ellipse"
    else:
        clip = polygon(clip)
        if change == "vertices":
            clip.settings.polygon_points = OTHER_TRIANGLE
        else:
            clip.track.performance.clear()
    output = tmp_path / "不可覆盖.mp4"
    output.write_bytes(b"finished")
    monkeypatch.setattr(export, "select_encoder", lambda *args: pytest.fail("过期区域不能启动编码器"))
    with pytest.raises(ValueError, match="区域形状或顶点.*过期.*重新跟踪"):
        export.export_video(clip.info, clip.track, clip.settings, str(output), threading.Event())
    assert output.read_bytes() == b"finished"
    assert not list(tmp_path.glob(".lockframe-export-*"))


def test_old_rectangle_without_provenance_and_out_of_range_anchors_remain_compatible(clip):
    clip.track.performance.clear()
    clip.anchors[50] = (.1, .1, .2, .2)
    validate_clip(clip)
    clip.track.anchors = None  # Old results have no saved anchor provenance.
    validate_clip(clip)


def test_queued_polygon_snapshot_is_isolated_from_later_shape_edits(clip, tmp_path, monkeypatch):
    from lockframe import batch_export
    clip = polygon(clip)
    task = prepare_exports([clip], tmp_path / "输出")[0]
    clip.settings.polygon_points = OTHER_TRIANGLE
    clip.track.performance["polygon_points"] = BOX_CORNERS
    clip.anchors[0] = (.1, .1, .3, .3)
    snapshot = pickle.loads(task.snapshot)
    assert snapshot.settings.polygon_points == TRIANGLE
    assert snapshot.track.performance["polygon_points"] == TRIANGLE
    validate_clip(snapshot)
    rendered = []
    def render(info, track, settings, destination, *args, **kwargs):
        rendered.append(settings.polygon_points)
        return {"path": destination}
    monkeypatch.setattr(batch_export, "export_video", render)
    assert run_exports([task], threading.Event())[0]["state"] == "完成"
    assert rendered == [TRIANGLE]


def test_sequence_roundtrip_retains_independent_regions_and_stale_result(clip, tmp_path):
    triangle = polygon(clip)
    triangle.id = "triangle"
    triangle.settings.polygon_points = OTHER_TRIANGLE  # Stale track intentionally retained for review.
    rectangle = deepcopy(clip)
    ellipse = replace(clip, settings=replace(clip.settings, target_shape="ellipse"), track=None)
    path = tmp_path / "区域素材.locksequence"
    save_sequence(path, [triangle, rectangle, ellipse])
    restored = read_sequence(path)
    assert [item.settings.target_shape for item in restored] == ["polygon", "rectangle", "ellipse"]
    assert restored[0].settings.polygon_points == OTHER_TRIANGLE
    assert restored[0].track.performance["polygon_points"] == [list(point) for point in TRIANGLE]
    np.testing.assert_array_equal(restored[0].track.positions, triangle.track.positions)
    tasks = prepare_exports(restored, tmp_path / "output")
    assert "区域形状或顶点" in tasks[0].reason and not tasks[1].reason and "尚未跟踪" in tasks[2].reason
    restored[0].settings.polygon_points = BOX_CORNERS
    assert restored[1].settings.polygon_points == () and triangle.settings.polygon_points == OTHER_TRIANGLE


def test_old_sequence_missing_region_fields_defaults_to_rectangle(clip, tmp_path):
    path = tmp_path / "旧工程.locksequence"
    save_sequence(path, [clip])
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clips"][0]["settings"].pop("target_shape")
    data["clips"][0]["settings"].pop("polygon_points")
    # Removed feature data remains inert through the same migration boundary.
    data["clips"][0]["settings"].update(time_remap={"reverse": True}, frame_interpolation={"enabled": True})
    path.write_text(json.dumps(data), encoding="utf-8")
    restored = read_sequence(path)
    assert restored[0].settings.target_shape == "rectangle" and restored[0].settings.polygon_points == ()
    validate_clip(restored[0])
    save_sequence(path, restored)
    saved = json.loads(path.read_text(encoding="utf-8"))["clips"][0]["settings"]
    assert "time_remap" not in saved and "frame_interpolation" not in saved


def test_sequence_rejects_invalid_polygon_data(clip, tmp_path):
    path = tmp_path / "bad.locksequence"
    save_sequence(path, [clip])
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clips"][0]["settings"].update(target_shape="polygon", polygon_points=[[0, 0], [1, 1], [0, 1], [1, 0]])
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="自交"):
        read_sequence(path)


def test_sequence_worker_passes_polygon_points_with_other_tracking_parameters(clip, monkeypatch):
    from lockframe import sequence, tracking
    clip = polygon(clip)
    # This dispatch-only fixture represents a prepared CFR clip. Empty proxy
    # now means metadata-only (estimated frames), which must not be tracked.
    clip.info = replace(clip.info, proxy=clip.info.path)
    captured = {}
    def tracker(info, anchors, start, end, cancel, progress, **kwargs):
        captured.update(kwargs)
        return clip.track
    monkeypatch.setattr(tracking, "track_video", tracker)
    monkeypatch.setattr(sequence, "_cancel", threading.Event())
    monkeypatch.setattr(sequence, "_events", queue.Queue())
    monkeypatch.setattr(sequence, "_preview_enabled", False)
    monkeypatch.setattr(sequence, "_pause", threading.Event())
    monkeypatch.setattr(sequence, "_pause_on_lost", True)
    assert sequence._track_clip(clip, 1) is clip.track
    assert captured["shape"] == "polygon" and captured["polygon_points"] == TRIANGLE
    assert captured["pause_on_lost"] is True
    assert captured["previous"] is clip.track


def test_actual_spawned_sequence_tracker_preserves_polygon_provenance():
    from lockframe.demo import demo_frame
    from lockframe.media import import_video, resource_root
    from lockframe.sequence import run_queue
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    clip = Clip(info, Settings(target_shape="polygon", polygon_points=TRIANGLE,
                               acceleration="cpu", compute_threads=2),
                {5: demo_frame(5)[1]}, 5, 5)
    result = run_queue([clip], 1, threading.Event())[clip.id]
    assert not result["error"] and result["track"] is not None
    assert result["track"].performance["shape"] == "polygon"
    assert tuple(tuple(point) for point in result["track"].performance["polygon_points"]) == TRIANGLE
    assert result["track"].performance["queue_workers"] == 1
