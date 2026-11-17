import os
import threading
from dataclasses import replace

import numpy as np
import pytest

from lockframe.sequence import Clip, run_queue, save_sequence, read_sequence, import_many
from lockframe.models import Settings
from lockframe.media import import_video, resource_root
from lockframe.demo import demo_frame


@pytest.fixture
def clip():
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    return Clip(info, Settings(acceleration="cpu", compute_threads=4), {5: demo_frame(5)[1]}, 0, 20)


def test_sequential_queue_orders_items_and_continues_after_failure_and_unset_region(clip):
    second = Clip(clip.info, replace(clip.settings), dict(clip.anchors), 0, 20)
    invalid = Clip(clip.info, replace(clip.settings), dict(clip.anchors), 0, clip.info.frames + 5)
    unset = Clip(clip.info)
    started = []
    def item_done(clip_id, result, error):
        if error == "正在跟踪":
            started.append(clip_id)
    result = run_queue([clip, invalid, unset, second], 1, threading.Event(), item_done=item_done)
    assert started == [clip.id, invalid.id, second.id]
    assert result[clip.id]["track"] is not None and not result[clip.id]["track"].lost_frames
    assert result[second.id]["track"] is not None and not result[second.id]["track"].lost_frames
    assert result[invalid.id]["error"].startswith("失败：")
    assert result[unset.id]["error"] == "未选目标，已跳过"


def test_parallel_queue_uses_distinct_processes_and_divides_resources(clip, tmp_path):
    import shutil
    second_path = tmp_path / "second.mp4"
    shutil.copy2(clip.info.path, second_path)
    second_info = import_video(str(second_path), threading.Event())
    second = Clip(second_info, replace(clip.settings, tracking_mode="precision", perspective=True,
                                    planar_region=True), dict(clip.anchors), 0, 20)
    result = run_queue([clip, second], 2, threading.Event())
    tracks = [result[item.id]["track"] for item in (clip, second)]
    assert all(track is not None and not track.lost_frames for track in tracks)
    assert len({track.performance["queue_worker_pid"] for track in tracks}) == 2
    assert all(track.performance["queue_worker_pid"] != os.getpid() for track in tracks)
    assert all(track.performance["queue_threads"] == 2 for track in tracks)
    assert all(track.performance["queue_cache_divisor"] == 2 for track in tracks)
    assert tracks[1].homographies.shape == (21, 3, 3)


def test_queue_rejects_modified_source_and_keeps_other_items(clip):
    changed = Clip(replace(clip.info, fingerprint="stale"), replace(clip.settings), dict(clip.anchors), 0, 20)
    result = run_queue([changed, clip], 1, threading.Event())
    assert "原视频已变化" in result[changed.id]["error"]
    assert result[clip.id]["track"] is not None


def test_queue_cancels_inflight_work_without_replacing_previous_track(clip):
    cancel = threading.Event()
    def done(clip_id, result, error):
        if error == "正在跟踪":
            cancel.set()
    result = run_queue([clip], 1, cancel, item_done=done)
    assert result[clip.id]["error"] == "已取消"
    assert result[clip.id]["track"] is None


def test_queue_cancellation_keeps_completed_result_and_never_starts_next(clip):
    second = Clip(clip.info, replace(clip.settings), dict(clip.anchors), 0, 20)
    cancel = threading.Event()
    started = []
    def done(clip_id, result, error):
        if error == "正在跟踪":
            started.append(clip_id)
        if result is not None:
            cancel.set()
    result = run_queue([clip, second], 1, cancel, item_done=done)
    assert result[clip.id]["track"] is not None
    assert result[second.id]["error"] == "已取消"
    assert started == [clip.id]


def test_sequence_roundtrip_retains_order_ranges_settings_anchors_and_full_matrices(clip, tmp_path):
    from lockframe.tracking import track_video
    clip.settings.perspective = clip.settings.planar_region = True
    clip.track = track_video(clip.info, clip.anchors, clip.start, clip.end, threading.Event(),
                             perspective=True, planar_region=True)
    second = Clip(clip.info, Settings(target_shape="ellipse"), {10: demo_frame(10)[1]}, 8, 18)
    path = tmp_path / "测试.locksequence"
    save_sequence(path, [second, clip])
    restored = read_sequence(path)
    assert [item.start for item in restored] == [8, 0]
    assert [item.end for item in restored] == [18, 20]
    assert restored[0].anchors == second.anchors and restored[0].settings.target_shape == "ellipse"
    assert restored[1].settings.planar_region and restored[1].settings.perspective
    np.testing.assert_array_equal(restored[1].track.homographies, clip.track.homographies)
    np.testing.assert_array_equal(restored[1].track.positions, clip.track.positions)


def test_import_many_preserves_successes_and_reports_failures(monkeypatch):
    from lockframe import sequence
    def importer(path, cancel):
        if path == "bad":
            raise ValueError("损坏视频")
        return path
    monkeypatch.setattr(sequence, "import_video", importer)
    result = import_many(["one", "bad", "two"], threading.Event())
    assert result[0]["info"] == "one" and result[2]["info"] == "two"
    assert result[1]["error"] == "损坏视频"


def test_sequence_save_cannot_overwrite_source_or_proxy(clip):
    for path in (clip.info.path, clip.info.proxy):
        with pytest.raises(ValueError, match="不能覆盖"):
            save_sequence(path, [clip])


def test_sequence_load_rejects_out_of_bounds_anchors(clip, tmp_path):
    import json
    path = tmp_path / "bad.locksequence"
    save_sequence(path, [clip])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["clips"][0]["anchors"] = {"5": [.9, .9, .5, .5]}
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="选区无效"):
        read_sequence(path)


def test_clip_switch_keeps_independent_workspaces_and_selection_order(clip, tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QSettings
    from lockframe import app
    from lockframe.media import configure_cache_preferences
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app, "QSettings", lambda *args: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    window = app.MainWindow()
    second = Clip(clip.info, Settings(target_shape="ellipse", planar_region=True),
                  {10: demo_frame(10)[1]}, 8, 18)
    window.clips = [clip, second]
    try:
        window._activate_clip(clip.id)
        assert window.anchors == clip.anchors and window.start_spin.value() == 1
        window._activate_clip(second.id)
        assert window.anchors == second.anchors and window.start_spin.value() == 9
        assert window.settings().target_shape == "ellipse" and window.settings().planar_region
        window._activate_clip(clip.id)
        assert window.anchors == clip.anchors and window.end_spin.value() == 21
        assert window.settings().target_shape == "rectangle" and not window.settings().planar_region
        window._refresh_sequence({second.id})
        window._move_clips(-1)
        assert [item.id for item in window.clips] == [second.id, clip.id]
        window.clip_list.selectAll()
        assert window._selected_clip_ids() == {clip.id, second.id}
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        window._remove_clips()
        assert window.clips == [] and window.info is None and window.reader is None
    finally:
        window.dirty = False
        for item in window.clips:
            item.dirty = False
        window._sequence_changed = False
        window.close()
        configure_cache_preferences(None)
