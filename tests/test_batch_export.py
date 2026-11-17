import os
import pickle
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lockframe.batch_export import prepare_exports, run_exports
from lockframe.demo import demo_frame
from lockframe.media import import_video, probe, resource_root, Cancelled
from lockframe.models import Settings
from lockframe.sequence import Clip, save_sequence, read_sequence
from lockframe.tracking import track_video


@pytest.fixture(scope="module")
def clip():
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    anchors = {5: demo_frame(5)[1]}
    track = track_video(info, anchors, 2, 15, threading.Event())
    return Clip(info, Settings(acceleration="cpu", resolution=0), anchors, 2, 15, track)


def test_mixed_eligibility_and_deep_frozen_snapshots(clip, tmp_path):
    valid = deepcopy(clip)
    lost = deepcopy(clip)
    lost.track.states[3] = "lost"
    missing = replace(clip, info=replace(clip.info, path=str(tmp_path / "不存在.mp4")))
    tasks = prepare_exports([valid, replace(clip, track=None), lost, missing], tmp_path)
    assert [bool(t.reason) for t in tasks] == [False, True, True, True]
    before = pickle.loads(tasks[0].snapshot)
    valid.settings.target_x = .7
    valid.track.positions[:] = .1
    valid.anchors.clear()
    after = pickle.loads(tasks[0].snapshot)
    assert after.settings.target_x == .5 and after.anchors == before.anchors
    np.testing.assert_array_equal(after.track.positions, before.track.positions)
    assert tasks[0].result_revision


def test_unique_unicode_paths_and_source_protection(clip, tmp_path):
    folder = tmp_path / "输出 中文"
    tasks = prepare_exports([clip, clip], folder)
    assert tasks[0].destination != tasks[1].destination
    Path(tasks[0].destination).write_bytes(b"existing")
    third = prepare_exports([clip], folder)[0]
    assert third.destination != tasks[0].destination
    source = Path(clip.info.path)
    assert all(Path(t.destination).resolve() != source.resolve() for t in tasks)


def test_failure_continues_and_skips_are_reported(clip, tmp_path, monkeypatch):
    from lockframe import batch_export
    tasks = prepare_exports([clip, replace(clip, track=None), clip], tmp_path)
    calls = []
    def render(*args, **kwargs):
        calls.append(args[3])
        if len(calls) == 1:
            raise RuntimeError("模拟磁盘写入失败")
        return {"path": args[3]}
    monkeypatch.setattr(batch_export, "export_video", render)
    result = run_exports(tasks, threading.Event())
    assert [r["state"] for r in result] == ["失败", "跳过", "完成"]
    assert "模拟磁盘" in result[0]["detail"] and len(calls) == 2


def test_cancel_keeps_finished_and_does_not_start_remaining(clip, tmp_path):
    tasks = prepare_exports([clip, clip], tmp_path)
    cancel = threading.Event()
    def item(index, state, value, detail):
        if index == 0 and state == "完成":
            cancel.set()
    result = run_exports(tasks, cancel, item=item)
    assert [r["state"] for r in result] == ["完成", "已取消"]
    assert Path(tasks[0].destination).is_file() and not Path(tasks[1].destination).exists()
    assert not list(tmp_path.glob(".lockframe-export-*"))


def test_cancel_during_render_cleans_partial(clip, tmp_path):
    tasks = prepare_exports([clip, clip], tmp_path)
    cancel = threading.Event()
    def item(index, state, value, detail):
        if state == "导出中" and value > 0:
            cancel.set()
    result = run_exports(tasks, cancel, item=item)
    assert all(r["state"] == "已取消" for r in result)
    assert not list(tmp_path.iterdir())


def test_real_export_parameters_audio_and_legacy_roundtrip(clip, tmp_path):
    square = replace(clip, settings=replace(clip.settings, aspect="1:1", target_x=.4, target_y=.6))
    path = tmp_path / "旧序列.locksequence"
    save_sequence(path, [clip, square])
    restored = read_sequence(path)
    tasks = prepare_exports(restored, tmp_path / "导出")
    results = run_exports(tasks, threading.Event())
    assert all(r["state"] == "完成" for r in results)
    for index, task in enumerate(tasks):
        info = probe(task.destination)
        import json
        import subprocess
        from lockframe.media import find_tool, NO_WINDOW
        metadata = json.loads(subprocess.check_output([find_tool("ffprobe"), "-v", "error", "-count_frames",
            "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", task.destination], creationflags=NO_WINDOW))
        assert int(metadata["streams"][0]["nb_read_frames"]) == clip.track.count and info.has_audio
        assert abs(info.duration - clip.track.count / clip.info.fps) < 1 / clip.info.fps
        assert (info.width, info.height) == ((960, 540) if index == 0 else (540, 540))


def test_late_output_collision_never_overwrites(clip, tmp_path):
    task = prepare_exports([clip], tmp_path)[0]
    Path(task.destination).write_bytes(b"existing finished video")
    result = run_exports([task], threading.Event())
    assert result[0]["state"] == "失败"
    assert Path(task.destination).read_bytes() == b"existing finished video"
    assert not list(tmp_path.glob(".lockframe-export-*"))


def test_post_enqueue_source_or_range_change_detected(clip, tmp_path):
    changed = replace(clip, info=replace(clip.info, fingerprint="stale"))
    bad_range = replace(clip, start=0)
    tasks = prepare_exports([changed, bad_range], tmp_path)
    assert "变化" in tasks[0].reason and "范围" in tasks[1].reason


def test_source_modified_after_queue_creation_is_rejected(clip, tmp_path):
    import shutil
    from lockframe.models import file_fingerprint
    source = tmp_path / "新原片.mp4"
    shutil.copy2(clip.info.path, source)
    copied = replace(clip, info=replace(clip.info, path=str(source), fingerprint=file_fingerprint(source)))
    copied.track = deepcopy(clip.track)
    from lockframe.models import source_identity
    copied.track.performance["source_identity"] = source_identity(copied.info)
    tasks = prepare_exports([copied], tmp_path / "output")
    assert not tasks[0].reason
    with source.open("ab") as stream:
        stream.write(b"changed")
    result = run_exports(tasks, threading.Event())
    assert result[0]["state"] == "失败" and "变化" in result[0]["detail"]
    assert not Path(tasks[0].destination).exists()


def test_results_from_another_source_are_not_exported(clip, tmp_path):
    import shutil
    from lockframe.models import file_fingerprint
    source = tmp_path / "another.mp4"
    shutil.copy2(clip.info.path, source)
    wrong = replace(clip, info=replace(clip.info, path=str(source), fingerprint=file_fingerprint(source)))
    tasks = prepare_exports([wrong], tmp_path / "output")
    assert "身份" in tasks[0].reason


def test_atomic_commit_collision_cleans_render_and_keeps_finished(clip, tmp_path, monkeypatch):
    from lockframe import export
    original_rename = os.rename
    task = prepare_exports([clip], tmp_path)[0]
    def competing_rename(source, destination, *args, **kwargs):
        if Path(destination) == Path(task.destination):
            Path(destination).write_bytes(b"competitor completed")
        return original_rename(source, destination, *args, **kwargs)
    monkeypatch.setattr(export.os, "rename", competing_rename)
    result = run_exports([task], threading.Event())
    assert result[0]["state"] == "失败"
    assert Path(task.destination).read_bytes() == b"competitor completed"
    assert not list(tmp_path.glob(".lockframe-export-*"))


def test_composition_edit_restore_and_drag_do_not_retrack(clip, tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings, QPoint, Qt
    from PySide6.QtTest import QTest
    from lockframe import app
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app, "QSettings", lambda *args: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    monkeypatch.setattr(app, "track_video", lambda *args, **kwargs: pytest.fail("构图修改不能重跑跟踪"))
    window = app.MainWindow()
    window.clips = [deepcopy(clip)]
    try:
        window._activate_clip(window.clips[0].id)
        track = window.track
        window._ready = False
        window._restore_settings(replace(clip.settings, target_x=.41412, target_y=.62345))
        window._ready = True
        assert window.settings().target_x == pytest.approx(.41412)
        assert window.settings().target_y == pytest.approx(.62345)
        window._set_composition(.5, .5)
        assert window.settings().target_x == .5 and window.settings().target_y == .5
        window.show()
        window.composition_edit.setChecked(True)
        application.processEvents()
        rect = window.canvas.output_rect
        point = QPoint(round(rect.x() + .65 * rect.width()), round(rect.y() + .4 * rect.height()))
        original_anchors = dict(window.anchors)
        QTest.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=point)
        QTest.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=point)
        window.aspect_combo.setCurrentIndex(2)
        window.resolution_combo.setCurrentIndex(1)
        assert window.track is track and window.anchors == original_anchors
        assert window.settings().target_x == pytest.approx(.65, abs=.004)
        assert window.settings().target_y == pytest.approx(.4, abs=.004)
        assert "实际" in window.crop_label.text() and "受限" in window.crop_label.text()
        assert "需要放大" in window.crop_label.toolTip() and "上限" in window.crop_label.toolTip()
    finally:
        window._confirm_discard = lambda: True
        window.close()


def test_new_anchor_recomputes_only_dependent_walks(clip, monkeypatch):
    from lockframe.tracking import FeatureTracker
    anchors = {f: demo_frame(f)[1] for f in (5, 25, 45)}
    original = track_video(clip.info, anchors, 0, 60, threading.Event())
    calls = []
    step = FeatureTracker.step
    def count(self, gray):
        calls.append(1)
        return step(self, gray)
    monkeypatch.setattr(FeatureTracker, "step", count)
    anchors[15] = demo_frame(15)[1]
    result = track_video(clip.info, anchors, 0, 60, threading.Event(), previous=original)
    assert not result.lost_frames and len(calls) == 18  # 6..14 and 16..24
    np.testing.assert_array_equal(result.positions[25:], original.positions[25:])


def test_acceptance_record_separates_measurements_from_unverified_footage(clip):
    from lockframe.acceptance import acceptance_record
    record = acceptance_record(clip)
    assert record["status"] == "待实拍验收"
    assert record["measured"]["tracking_performance"]["seconds"] > 0
    assert record["manual_review"]["manual_correction_count"] is None
    assert record["manual_review"]["confirmed_original_footage_and_authorized_use"] is None
