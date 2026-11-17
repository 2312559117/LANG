"""Production-path checks also executed by the standalone frozen EXE."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import threading
import time

from PySide6.QtCore import QPointF, QEvent, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QFileDialog

from .batch_export import prepare_exports, run_exports
from .batch_export_ui import open_export_queue
from .media import find_tool, NO_WINDOW, probe
from .acceptance import acceptance_record


def check_phase1(window, folder, valid):
    application = QApplication.instance()
    valid = deepcopy(valid)
    valid.settings = replace(valid.settings, aspect="1:1", resolution=0, target_x=.41412, target_y=.62345)
    untracked = replace(valid, id="untracked", track=None)
    lost = deepcopy(valid)
    lost.id = "lost"
    lost.track.states[2] = "lost"
    missing = replace(valid, id="missing", info=replace(valid.info, path=str(folder / "missing.mp4")))
    window.clips = [valid, untracked, lost, missing]
    window.active_clip_id = None
    window._activate_clip(valid.id)
    assert abs(window.settings().target_y - .62345) < 1e-6
    original_track = window.track
    window.composition_edit.setChecked(True)
    window._seek(valid.start + 3)
    application.processEvents()
    rect = window.canvas.output_rect
    point = QPointF(rect.x() + rect.width() * .6, rect.y() + rect.height() * .45)
    for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                           (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
        QApplication.sendEvent(window.canvas, QMouseEvent(kind, point, point,
                              Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier))
    assert abs(window.settings().target_x - .6) < .001
    assert abs(window.settings().target_y - .45) < .001 and window.track is original_track
    assert window.frame_index == valid.start + 3  # before and after share this exact source frame
    window._set_composition(.5, .5)
    window.composition_edit.setChecked(False)
    window._refresh_sequence({c.id for c in window.clips})
    output = folder / "批量导出"
    output.mkdir(exist_ok=True)
    # A pre-existing completed output must survive automatic numbering.
    protected = output / (Path(valid.info.path).stem + "_番茄跟踪.mp4")
    protected.write_bytes(b"protected completed output")
    old_dialog = QFileDialog.getExistingDirectory
    QFileDialog.getExistingDirectory = lambda *args, **kwargs: str(output)
    try:
        open_export_queue(window, selected=True)
    finally:
        QFileDialog.getExistingDirectory = old_dialog
    queue = window.export_queue
    # Editing after enqueue must leave the queued square output unchanged.
    window.aspect_combo.setCurrentIndex(window.aspect_combo.findData("16:9"))
    window._set_composition(.7, .4)
    assert window.track is original_track
    assert not window.busy and window.open_clip_button.isEnabled()
    window._activate_clip(untracked.id)
    assert window.active_clip_id == untracked.id
    deadline = time.monotonic() + 60
    while queue.running and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.01)
    if queue.running:
        queue.cancel()
        queue.worker.wait(15000)
        raise RuntimeError("Batch export GUI timed out")
    states = [queue.table.item(i, 1).text() for i in range(4)]
    assert states == ["完成", "跳过", "跳过", "跳过"], states
    assert protected.read_bytes() == b"protected completed output"
    destination = queue.tasks[0].destination
    exported = probe(destination)
    metadata = json.loads(subprocess.check_output([find_tool("ffprobe"), "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", destination], creationflags=NO_WINDOW))
    frames = int(metadata["streams"][0]["nb_read_frames"])
    assert frames == valid.track.count
    assert exported.width == exported.height and exported.has_audio == valid.info.has_audio
    assert abs(exported.duration - frames / valid.info.fps) <= 1 / valid.info.fps
    queue.grab().save(str(folder / "batch-export-preview.png"))
    queue.close()
    # Runtime failures are isolated; cancellation keeps previously committed output.
    tasks = prepare_exports([valid, valid, valid], folder / "failure-and-cancel")
    Path(tasks[0].destination).write_bytes(b"late conflict")
    cancel = threading.Event()
    def done(index, state, value, text):
        if index == 1 and state == "完成":
            cancel.set()
    outcomes = run_exports(tasks, cancel, item=done)
    assert [o["state"] for o in outcomes] == ["失败", "完成", "已取消"]
    assert not Path(tasks[2].destination).exists()
    partial_tasks = prepare_exports([valid], folder / "cancel-partial")
    partial_cancel = threading.Event()
    def interrupt(index, state, value, text):
        if state == "导出中" and value > 0:
            partial_cancel.set()
    partial = run_exports(partial_tasks, partial_cancel, item=interrupt)
    assert partial[0]["state"] == "已取消" and not list((folder / "cancel-partial").iterdir())
    (folder / "acceptance-synthetic.json").write_text(json.dumps(acceptance_record(valid), ensure_ascii=False, indent=2), encoding="utf-8")
    window._activate_clip(valid.id)
    window._set_view("compare")
    window.inspector_tabs.setCurrentIndex(0)
    application.processEvents()
    window.grab().save(str(folder / "composition-preview.png"))
    # Exercise GUI correction -> retained dependency result -> production tracker.
    from .demo import demo_frame
    from .tracking import track_video
    source_track = window.track
    correction_frame = valid.start + 4
    window._seek(correction_frame)
    window._select_roi(demo_frame(correction_frame, valid.info.width, valid.info.height)[1])
    assert window.track is source_track and window._retrack_previous is source_track
    assert window._region_stale() and not window.export_button.isEnabled()
    window._stash_clip()
    assert window._clip().retrack_previous is source_track
    corrected = track_video(window.info, window.anchors, valid.start, valid.end,
                            threading.Event(), previous=window._retrack_previous)
    assert corrected.performance["reused_frames"] > 0 and not corrected.lost_frames
    window._tracking_done(corrected)
    # This direct callback bypasses Job.finished -> _job_finished -> _sync_ui.
    # Refresh the controls as the production worker completion path does.
    window._sync_ui()
    assert not window._region_stale() and window.export_button.isEnabled()
    return {"mixed_queue_states": states, "frozen_snapshot_and_edit_other_clip": True,
            "composition_drag_without_retracking": True, "same_source_frame_comparison": True,
            "failure_then_completion_then_cancel": [o["state"] for o in outcomes],
            "partial_cancel_cleanup": True, "frames": frames, "width": exported.width,
            "height": exported.height, "duration": exported.duration, "audio": exported.has_audio,
            "correction_reused_frames": corrected.performance["reused_frames"],
            "real_car_footage": "pending: no confirmed original footage in project"}
