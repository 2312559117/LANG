"""Real MainWindow acceptance for the compact workflow and correction gate.

The locally generated appearance-switch video has an independently known target
box. It is synthetic QA material, not original car footage or an accuracy claim.
The same checks run from Python and from the packaged executable.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import subprocess
import threading
import time
import traceback

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QSettings
from PySide6.QtWidgets import QApplication

from .demo import demo_frame
from .models import Settings
from .sequence import Clip


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def wait_job(window, timeout=90):
    application = QApplication.instance()
    deadline = time.monotonic() + timeout
    while window.busy and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.005)
    if window.busy:
        window.worker.cancel()
        window.worker.wait(15000)
        application.processEvents()
        raise RuntimeError("Compact workflow check timed out")
    application.processEvents()


def correction_box(frame):
    """Known normalized box in the generated 400 x 240 source."""
    return ((90 + frame) / 400, 76 / 240, 80 / 400, 80 / 240)


def create_correction_source(directory):
    """An abrupt texture replacement requires a new manual reference at frame 12."""
    from .media import import_video

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "合成验收_目标外观切换.avi"
    rng = np.random.default_rng(729)
    patches = []
    for marker in ("A", "B"):
        patch = rng.integers(25, 230, (80, 80, 3), dtype=np.uint8)
        patch = cv2.GaussianBlur(patch, (3, 3), .55)
        cv2.putText(patch, marker, (18, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (240, 240, 240), 3)
        patches.append(patch)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 24., (400, 240))
    if not writer.isOpened():
        raise RuntimeError("Unable to create the local synthetic correction source")
    try:
        for index in range(48):
            frame = np.full((240, 400, 3), 24, dtype=np.uint8)
            x = 90 + index
            frame[76:156, x:x + 80] = patches[int(index >= 12)]
            writer.write(frame)
    finally:
        writer.release()
    return import_video(str(path), threading.Event())


def activate(window, clips):
    window.clips, window.active_clip_id = clips, None
    window._activate_clip(clips[0].id)
    wait_job(window)
    window._refresh_sequence({clip.id for clip in clips})


def _paused_result(window):
    paused = window._paused_tracking
    _require(bool(paused), "Tracking stopped without retaining its correction state")
    return paused, paused["result"]


def check_auto_correction(window, directory):
    from .batch_export import prepare_exports
    from .tracking_pause import computed_mask, is_complete

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    info = create_correction_source(folder)
    clip = Clip(info, Settings(acceleration="cpu", compute_threads=2, resolution=0),
                {0: correction_box(0)}, 0, info.frames - 1)
    activate(window, [clip])
    window.auto_pause_check.setChecked(True)
    window.live_preview_check.setChecked(False)
    window._start_tracking()
    _require(window.busy, "A real tracking Job did not start")
    wait_job(window)
    paused, partial = _paused_result(window)
    frame = int(paused["frame"])
    mask = computed_mask(partial)
    _require(0 < frame < info.frames - 1 and partial.states[frame] == "lost",
             "The appearance-switch source did not stop at a measured lost frame")
    _require(not is_complete(partial) and 0 < int(mask.sum()) < partial.count,
             "Paused tracking claimed its uncomputed frames were complete")
    _require(not mask[frame + 1:].any(), "Tracking kept calculating after its first lost frame")
    _require(not window.busy and window.worker is None,
             "Paused tracking retained a running worker")
    _require(window.frame_index == frame and window.canvas.selectable,
             "Paused tracking did not reveal the correction frame for editing")
    _require(not window.continue_button.isEnabled() and not window.export_button.isEnabled(),
             "Uncorrected tracking allowed continuation or export")
    _require(window.tracking_progress_panel.state == "paused" and window._header_value < 1.,
             "Paused tracking was displayed as a completed task")
    window._stash_clip()
    task = prepare_exports([window._clip()], folder / "暂停禁止导出")[0]
    _require(bool(task.reason), "Batch export accepted an incomplete tracking result")
    before = int(mask.sum())
    QApplication.processEvents()
    time.sleep(.04)
    QApplication.processEvents()
    _require(int(computed_mask(partial).sum()) == before,
             "A paused task continued calculating in the background")
    window._continue_tracking()
    _require(not window.busy and bool(window._paused_tracking),
             "Clicking continue without a correction restarted tracking")
    window._set_composition(.61, .44)
    _require(not window.continue_button.isEnabled(),
             "Changing composition incorrectly satisfied the tracking correction gate")
    window.grab().save(str(folder / "lost-paused-awaiting-correction.png"))

    # A saved single-project partial must reopen at the same correction gate,
    # even without its in-memory engine snapshot or previous GUI context.
    from .models import read_project, save_project
    project_path = folder / "暂停待修正.lockframe"
    save_project(str(project_path), info, window.settings(), window.anchors, 0, info.frames - 1, partial)
    window._paused_by_clip.clear()
    window._pending_project = read_project(str(project_path))
    window._pending_project["_project_path"] = str(project_path)
    window._import_done(info)
    window._sync_ui()
    _require(window._paused_tracking and window.frame_index == frame and not window.continue_button.isEnabled(),
             "A reopened single project lost its paused-frame correction gate")

    # A reference on another frame must not accidentally unlock this pause.
    window._seek(min(info.frames - 1, frame + 2))
    window._select_roi(correction_box(window.frame_index))
    _require(not window.continue_button.isEnabled(),
             "An unrelated-frame keyframe unlocked the paused-frame correction gate")
    window._seek(frame)
    window._select_roi(correction_box(frame))
    QApplication.processEvents()
    _require(not window.busy and window.continue_button.isEnabled(),
             "Correcting the paused frame failed to wait for an explicit Continue click")
    window.grab().save(str(folder / "corrected-ready-to-continue.png"))
    window._continue_tracking()
    _require(window.busy, "An explicit Continue click did not create a real tracking Job")
    wait_job(window)
    result = window.track
    _require(result is not None and is_complete(result) and not result.lost_frames,
             "The corrected appearance-switch source did not finish with a valid track")
    _require(not window._paused_tracking and window.export_button.isEnabled(),
             "Successful continuation did not restore normal export")
    _require(window.tracking_progress_panel.state == "completed",
             "Successful continuation failed to finish the real stage panel")
    window.grab().save(str(folder / "continued-tracking-completed.png"))
    return {"footage": "local synthetic appearance-switch video; original car footage pending",
            "source": info.path, "frames": info.frames, "fps": info.fps,
            "lost_frame_zero_based": frame, "computed_before_pause": before,
            "live_preview_disabled": True, "stopped_before_next_frame": True,
            "uncorrected_continue_blocked": True, "composition_did_not_unlock": True,
            "different_frame_did_not_unlock": True, "correction_requires_explicit_continue": True,
            "single_project_pause_restored": True,
            "incomplete_batch_export_blocked": True, "continued_track_complete": True}, info, result


def check_manual_correction(window, directory, info):
    from .tracking_pause import computed_mask, is_complete

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    clip = Clip(info, Settings(acceleration="cpu", compute_threads=2, resolution=0),
                {0: demo_frame(0, info.width, info.height)[1]}, 0, info.frames - 1)
    activate(window, [clip])
    window.live_preview_check.setChecked(True)
    window._start_tracking()
    window._pause_tracking()
    wait_job(window)
    paused, result = _paused_result(window)
    frame = int(paused["frame"])
    _require(not is_complete(result), "Manual pause discarded its partial measurements")
    _require(window.continue_button.isEnabled() and not window.busy, "Ordinary pause must offer explicit Continue without forcing a correction")
    window._seek(frame)
    box = list(demo_frame(frame, info.width, info.height)[1])
    box[0] += 1 / info.width  # an actual one-source-pixel manual refinement
    window._select_roi(tuple(box))
    _require(not window.busy and window.continue_button.isEnabled(),
             "Manual correction automatically restarted, or failed to enable Continue")
    done = int(computed_mask(result).sum())
    window.grab().save(str(folder / "manually-paused-ready-to-continue.png"))
    window._continue_tracking()
    wait_job(window)
    _require(window.track is not None and is_complete(window.track) and not window.track.lost_frames,
             "Manual pause/correct/continue did not produce a valid demo track")
    return {"real_job": True, "pause_frame_zero_based": frame, "computed_before_pause": done,
            "explicit_continue_required": True, "continued_track_complete": True}


def check_batch_correction(window, directory, demo_info):
    from .tracking_pause import is_complete

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    switched_info = create_correction_source(folder)
    settings = Settings(acceleration="cpu", compute_threads=2, resolution=0)
    first = Clip(demo_info, deepcopy(settings), {0: demo_frame(0, demo_info.width, demo_info.height)[1]}, 0, 8)
    switching = Clip(switched_info, deepcopy(settings), {0: correction_box(0)}, 0, switched_info.frames - 1)
    waiting = Clip(demo_info, deepcopy(settings), {0: demo_frame(0, demo_info.width, demo_info.height)[1]}, 0, 18)
    activate(window, [first, switching, waiting])
    window.auto_pause_check.setChecked(True)
    window.live_preview_check.setChecked(False)
    window.queue_mode.setCurrentIndex(window.queue_mode.findData(1))
    window._start_queue()
    _require(window.busy, "The sequential correction queue failed to start")
    wait_job(window, 120)
    _require(first.track is not None and is_complete(first.track),
             "A queue pause discarded its previously completed material")
    completed = first.track
    _require(switching.track is not None and not is_complete(switching.track),
             "The queue did not retain the measured partial result of its lost material")
    _require(waiting.track is None, "The queue began a new material after a correction pause")
    _require(bool(window._paused_queue_ids) and window.active_clip_id == switching.id,
             "The paused queue did not select its correction material and retain the remaining work")
    _require(window.tracking_progress_panel.state == "paused" and window._header_value < 1.,
             "An incomplete paused batch was displayed as complete")
    paused, _ = _paused_result(window)
    _require(not window.continue_button.isEnabled(), "The paused batch continued before manual correction")
    window._seek(int(paused["frame"]))
    window._select_roi(correction_box(window.frame_index))
    _require(not window.busy and window.continue_button.isEnabled(),
             "A batch correction automatically restarted the queue")
    window.grab().save(str(folder / "batch-paused-ready-to-continue.png"))
    window._continue_tracking()
    _require(window.busy, "Explicit Continue did not restart the remaining batch")
    wait_job(window, 120)
    _require(all(clip.track is not None and is_complete(clip.track) and not clip.track.lost_frames
                 for clip in (first, switching, waiting)),
             "The corrected batch did not retain all completed tracking results")
    _require(first.track is completed, "Continuing a paused queue unnecessarily retracked a completed material")
    _require(not window._paused_queue_ids, "A completed queue retained stale paused task IDs")
    window.grab().save(str(folder / "batch-continued-completed.png"))
    return {"real_spawn_queue": True, "materials": 3, "completed_before_pause": 1,
            "waiting_material_not_started": True, "correction_requires_explicit_continue": True,
            "previous_result_not_retracked": True, "completed_after_continue": 3}


def check_layout(window, directory):
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    window.resize(1160, 760)
    window.show()
    QApplication.processEvents()
    _require(set(window.sidebar_sections) == {"materials", "tracking", "lock", "time_remap", "export"},
             "The workflow exposes unexpected complex top-level sections")
    _require(window.workspace_splitter.count() == 3,
             "The workflow must have materials, dual preview and task columns")
    _require(window.workspace_splitter.widget(0) is window.materials_sidebar
             and window.workspace_splitter.widget(1) is window.central_preview
             and window.workspace_splitter.widget(2) is window.task_sidebar,
             "The three workspace columns are in the wrong order")
    _require(window.asset_status_bar.isVisible(), "The permanent source status bar is missing")
    _require(window.central_preview.width() >= 560
             and window.canvas.width() > 350 and window.canvas.height() > 150,
             "The compact workflow squeezed out the working preview")
    export_bottom = window.export_button.mapTo(window.task_sidebar, QPoint(0, window.export_button.height())).y()
    _require(window.task_sidebar.isAncestorOf(window.export_button)
             and window.export_button.isVisible() and export_bottom <= window.task_sidebar.height(),
             "The export action scrolled out of the compact right workflow")
    _require(window.materials_sidebar.isAncestorOf(window.clip_list),
             "The materials list is outside the left column")
    _require(window.timeline_well.parentWidget() is window.centralWidget()
             and abs(window.timeline_well.width()-window.workspace_splitter.width()) <= 2,
             "The source-frame timebar does not span all three columns")
    window.resize(1560, 940)
    window._reveal_section("materials")
    QApplication.processEvents()
    window.grab().save(str(folder / "reference-workflow-large.png"))
    window.resize(1160, 760)
    QApplication.processEvents()
    window.inspector_tabs.setCurrentIndex(2)
    QApplication.processEvents()
    _require(window.settings_dialog.isVisible() and window.acceleration_combo.isVisible(),
             "Advanced compute settings cannot be reached through the secondary settings dialog")
    _require(not window.settings_dialog.isModal(), "Advanced settings blocked the working preview")
    window.grab().save(str(folder / "compact-workflow-small.png"))
    window.settings_dialog.hide()
    window.library_tabs.setCurrentIndex(1)
    QApplication.processEvents()
    sidebar_left = window.task_sidebar.mapTo(window, QPoint(0, 0)).x()
    sidebar_right = window.task_sidebar.mapTo(window, QPoint(window.task_sidebar.width(), 0)).x()
    for control in (window.track_button, window.pause_button, window.continue_button,
                    window.center_button, window.export_button):
        _require(window.task_sidebar.isAncestorOf(control),
                 "A core tracking/composition/export action belongs to the wrong column")
        # Idle pause/continue actions may be intentionally hidden. Their layout
        # geometry is meaningful only while the corresponding action is shown.
        if not control.isVisible():
            continue
        left = control.mapTo(window, QPoint(0, 0)).x()
        right = control.mapTo(window, QPoint(control.width(), 0)).x()
        _require(sidebar_left <= left < right <= sidebar_right,
                 f"Core action {control.text()!r} lies outside right workflow: "
                 f"{left}..{right}; sidebar {sidebar_left}..{sidebar_right}")
    return {"core_sections": list(window.sidebar_sections), "advanced_secondary_dialog": True,
            "persistent_monitors": True, "export_always_visible": True,
            "columns": 3, "materials_left": True, "core_actions_right": True,
            "full_width_timebar": True,
            "window": [window.width(), window.height()],
            "canvas": [window.canvas.width(), window.canvas.height()]}


def check_ui_v9(window, directory, info):
    from .export import export_video
    from .media import NO_WINDOW, find_tool, probe

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        automatic, switched_info, result = check_auto_correction(window, folder / "automatic")
        exported = export_video(switched_info, result, window.settings(),
                                str(folder / "corrected-synthetic-export.mp4"), threading.Event())
        output = probe(exported["path"])
        # probe() intentionally leaves frames=0 until a proxy is imported. Count
        # the actual finished stream without creating an unnecessary new proxy.
        counted = json.loads(subprocess.check_output(
            [find_tool("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", exported["path"]],
            creationflags=NO_WINDOW, timeout=30))
        output_frames = int(counted["streams"][0]["nb_read_frames"])
        _require(output_frames == result.count and abs(output.duration - result.count / switched_info.fps)
                 <= 1 / switched_info.fps + .01,
                 "The continued track export changed frame count or duration")
        _require(not output.has_audio, "Silent synthetic footage unexpectedly gained an audio track")
        manual = check_manual_correction(window, folder / "manual", info)
        batch = check_batch_correction(window, folder / "batch", info)
        layout = check_layout(window, folder)
        return {"footage": "bundled/local synthetic QA only; original car footage pending",
                "automatic_correction": automatic, "manual_correction": manual,
                "batch_correction": batch, "layout": layout, "continued_export": {"frames": output_frames,
                    "duration": output.duration, "audio": output.has_audio,
                    "width": output.width, "height": output.height}}
    finally:
        if window.busy and window.worker is not None:
            window.worker.cancel()
            window.worker.wait(15000)
            QApplication.processEvents()


def run_workflow_check(directory):
    from . import __version__, app
    from .media import configure_cache_preferences, import_video, resource_root

    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    original_settings = app.QSettings
    window = None
    try:
        app.QSettings = lambda *args: QSettings(str(folder / "isolated-check.ini"), QSettings.Format.IniFormat)
        try:
            window = app.MainWindow()
        finally:
            app.QSettings = original_settings
        info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
        started = time.monotonic()
        details = check_ui_v9(window, folder, info)
        report = {"ok": True, "version": __version__, "seconds": time.monotonic() - started,
                  "ui_v9": details}
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        (folder / "report.json").write_text(json.dumps({"ok": False, "error": traceback.format_exc()},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    finally:
        app.QSettings = original_settings
        if window is not None:
            window._confirm_discard = lambda: True
            window.close()
        configure_cache_preferences(None)
