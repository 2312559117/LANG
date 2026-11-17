"""Production widget checks for stop, correct, explicitly continue and core UI."""
import os
from copy import deepcopy
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from lockframe.demo import demo_frame
from lockframe.media import configure_cache_preferences, import_video, resource_root
from lockframe.models import Settings
from lockframe.sequence import Clip, read_sequence, save_sequence
from lockframe.tracking import track_video
from lockframe.v072_check import (activate, check_auto_correction, check_layout,
                                 check_manual_correction, check_batch_correction)


@pytest.fixture(scope="module")
def source():
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    anchors = {10: demo_frame(10, info.width, info.height)[1]}
    result = track_video(info, anchors, 0, 35, threading.Event())
    return Clip(info, Settings(acceleration="cpu", compute_threads=2), anchors, 0, 35, result)


@pytest.fixture
def window(tmp_path, monkeypatch):
    from lockframe import app

    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app, "QSettings", lambda *args: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    window = app.MainWindow()
    window.resize(1160, 760)
    window.show()
    application.processEvents()
    yield window
    if window.busy and window.worker:
        window.worker.cancel()
        window.worker.wait(15000)
        application.processEvents()
    window._confirm_discard = lambda: True
    window.close()
    configure_cache_preferences(None)


def test_real_lost_pause_requires_paused_frame_correction_and_explicit_continue(window, tmp_path):
    report, info, result = check_auto_correction(window, tmp_path / "automatic")
    assert report["lost_frame_zero_based"] == 12
    assert report["computed_before_pause"] == 13
    assert result.count == info.frames == 48
    assert not result.lost_frames


def test_manual_stop_retains_results_and_continue_is_an_explicit_action(window, source, tmp_path):
    report = check_manual_correction(window, tmp_path / "manual", source.info)
    assert report["explicit_continue_required"] and report["continued_track_complete"]


def test_basic_actions_left_advanced_secondary_and_curve_on_preview(window, source, tmp_path):
    activate(window, [deepcopy(source)])
    report = check_layout(window, tmp_path / "layout")
    assert report["advanced_secondary_dialog"]
    assert report["persistent_monitors"]


def test_queue_stops_remaining_materials_until_corrected_then_keeps_completed_results(window, source, tmp_path):
    report = check_batch_correction(window, tmp_path / "batch", source.info)
    assert report["completed_before_pause"] == 1
    assert report["completed_after_continue"] == 3


def test_switching_material_cannot_move_correction_gate_to_another_source(window, source, tmp_path):
    from lockframe.v072_check import create_correction_source, correction_box, wait_job
    from lockframe.tracking_pause import is_complete

    info = create_correction_source(tmp_path / "source")
    switching = Clip(info, Settings(acceleration="cpu", compute_threads=2),
                     {0: correction_box(0)}, 0, info.frames - 1)
    other = deepcopy(source)
    activate(window, [switching, other])
    window.auto_pause_check.setChecked(True)
    window._start_tracking()
    wait_job(window)
    paused = window._paused_tracking
    assert paused and not is_complete(paused["result"])
    window._stash_clip()
    path = tmp_path / "暂停修正状态.locksequence"
    save_sequence(path, window.clips)
    restored = read_sequence(path)
    assert not is_complete(restored[0].track)
    assert is_complete(restored[1].track)
    window._activate_clip(other.id)
    wait_job(window)
    assert window.info.path == other.info.path
    assert not window.continue_button.isEnabled()
    assert window.track is other.track
    window._activate_clip(switching.id)
    wait_job(window)
    assert window.info.path == info.path
    assert not window.continue_button.isEnabled()
    assert window._paused_tracking and not is_complete(window._paused_tracking["result"])


def test_spawn_queue_pauses_other_running_worker_and_does_not_start_pending(source, tmp_path):
    from lockframe.sequence import run_queue
    from lockframe.v072_check import create_correction_source, correction_box
    from lockframe.tracking_pause import TrackingPaused, is_complete

    info = create_correction_source(tmp_path / "source")
    switched = Clip(info, Settings(acceleration="cpu", compute_threads=2),
                    {0: correction_box(0)}, 0, info.frames - 1)
    other = Clip(source.info, Settings(acceleration="cpu", compute_threads=2),
                 {0: demo_frame(0, source.info.width, source.info.height)[1]}, 0, source.info.frames - 1)
    pending = deepcopy(other)
    pending.id = "unstarted-after-lost"
    outcomes = run_queue([switched, other, pending], 2, threading.Event(), pause_on_lost=True)
    assert outcomes[switched.id]["error"] == "已暂停"
    assert isinstance(outcomes[switched.id]["pause"], TrackingPaused)
    assert outcomes[switched.id]["pause"].reason == "lost"
    assert not is_complete(outcomes[switched.id]["pause"].result)
    if "pause" in outcomes[other.id]:
        assert not is_complete(outcomes[other.id]["pause"].result)
        assert outcomes[other.id]["pause"].reason == "manual"
    else:
        assert is_complete(outcomes[other.id]["track"])
    assert outcomes[pending.id]["error"] == "等待继续"
    assert outcomes[pending.id]["track"] is None


def _pause_source(window, tmp_path):
    from lockframe.v072_check import create_correction_source, correction_box, wait_job
    info = create_correction_source(tmp_path / "source")
    clip = Clip(info, Settings(acceleration="cpu", compute_threads=2),
                {0: correction_box(0)}, 0, info.frames - 1)
    activate(window, [clip])
    window.auto_pause_check.setChecked(True)
    window._start_tracking()
    wait_job(window)
    assert window._paused_tracking and not window.busy
    return clip


def test_shortening_range_past_paused_frame_allows_a_new_task(window, tmp_path):
    from lockframe.v072_check import wait_job
    from lockframe.tracking_pause import is_complete
    _pause_source(window, tmp_path)
    window.end_spin.setValue(10)
    assert window._paused_tracking is None
    assert window.track_button.isVisible() and window.track_button.isEnabled()
    window._start_tracking()
    wait_job(window)
    assert is_complete(window.track) and window.track.end == 9


def test_deleting_pause_correction_closes_the_continue_gate(window, tmp_path):
    from lockframe.v072_check import correction_box
    _pause_source(window, tmp_path)
    frame = window._paused_tracking["frame"]
    window._select_roi(correction_box(frame))
    assert window.continue_button.isEnabled()
    for index in range(window.anchor_list.count()):
        from PySide6.QtCore import Qt
        if window.anchor_list.item(index).data(Qt.ItemDataRole.UserRole) == frame:
            window.anchor_list.setCurrentRow(index)
            break
    window._delete_anchor()
    assert not window.continue_button.isEnabled()
    window._continue_tracking()
    assert not window.busy and window._paused_tracking is not None


@pytest.mark.parametrize("change", ["missing", "replaced"])
def test_continuation_rejects_changed_source_and_retains_partial(window, tmp_path, change):
    from pathlib import Path
    from lockframe.v072_check import correction_box
    _pause_source(window, tmp_path)
    context = window._paused_tracking
    window._select_roi(correction_box(context["frame"]))
    path = Path(window.info.path)
    if change == "missing":
        path.unlink()
    else:
        with path.open("ab") as stream:
            stream.write(b"changed source")
    errors = []
    window._error = errors.append
    window._continue_tracking()
    assert errors and "原视频" in errors[0]
    assert not window.busy and window._paused_tracking is context
    assert window.track is context["result"]


def test_removing_an_uncorrected_paused_material_does_not_block_other_tasks(window, tmp_path, monkeypatch):
    from lockframe.v072_check import correction_box, wait_job
    from lockframe.tracking_pause import is_complete
    from PySide6.QtWidgets import QMessageBox
    clip = _pause_source(window, tmp_path)
    window._stash_clip()
    removed = deepcopy(clip)
    removed.id = "removed-paused-material"
    window.clips.append(removed)
    window._paused_by_clip[removed.id] = dict(window._paused_tracking)
    window._paused_queue_ids = [clip.id, removed.id]
    window._refresh_sequence({removed.id})
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    window._remove_clips()
    assert removed.id not in window._paused_by_clip and removed.id not in window._paused_queue_ids
    window._select_roi(correction_box(window._paused_tracking["frame"]))
    window._continue_tracking()
    wait_job(window)
    assert is_complete(window.track) and not window._paused_tracking
