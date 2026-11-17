"""Production three-column UI checks shared by source and portable builds.

Only the bundled synthetic demo is used. This validates actual widget wiring,
worker state and source identity, not accuracy on original car footage.
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import threading
import time
import traceback

from PySide6.QtWidgets import QApplication

from .demo import demo_frame
from .models import Settings
from .sequence import Clip


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _wait(window, timeout=90):
    application = QApplication.instance()
    deadline = time.monotonic() + timeout
    while window.busy and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.005)
    if window.busy:
        window.worker.cancel()
        window.worker.wait(15000)
        application.processEvents()
        raise RuntimeError("Three-column UI worker timed out")
    application.processEvents()


def _activate(window, clips):
    window.clips, window.active_clip_id = clips, None
    window._activate_clip(clips[0].id)
    _wait(window)
    window._refresh_sequence({clip.id for clip in clips})


def _source_clip(info, *, last=70, mode="normal"):
    last = min(last, info.frames - 1)
    seed = min(30, last)
    return Clip(info, Settings(acceleration="cpu", compute_threads=2,
                               tracking_mode=mode, resolution=0),
                {seed: demo_frame(seed, info.width, info.height)[1]}, 0, last)


def check_ui_v8(window, directory, info):
    """Drive real MainWindow jobs and return an explicit synthetic QA report.

    The caller owns window lifetime and JSON report writing. This function leaves
    completed demo clips open for final screenshot inspection. On an exception,
    it cancels and joins any live worker before allowing the caller to close.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance()
    report = {"footage": "bundled synthetic demo; original car footage pending"}
    try:
        window.resize(1160, 760)
        window.show()
        application.processEvents()
        _require(window.workspace_splitter.count() == 3,
                 "The chosen reference layout must have materials, dual preview and task columns")
        _require(window.workspace_splitter.widget(0) is window.materials_sidebar
                 and window.workspace_splitter.widget(1) is window.central_preview
                 and window.workspace_splitter.widget(2) is window.task_sidebar,
                 "The three workspace columns are in the wrong order")
        _require(hasattr(window, "asset_status_bar") and hasattr(window, "tracking_progress_panel"),
                 "Persistent asset bar or tracking progress panel is missing")
        _require(window.asset_status_bar.isVisible(), "Asset metadata bar is not persistent")

        clip = _source_clip(info)
        _activate(window, [clip])
        _require(window.asset_status_bar.source_path == info.path,
                 "Persistent metadata did not follow the selected material")
        metadata = window.asset_status_bar.metadata_label.text()
        _require(str(info.width) in metadata and str(info.height) in metadata
                 and "fps" in metadata and f"{info.frames:,} 帧" in window.asset_status_bar.metadata_label.toolTip(),
                 "Persistent metadata omitted visible resolution/fps or frame count tooltip")
        for index in range(window.library_tabs.count()):
            window.library_tabs.setCurrentIndex(index)
            application.processEvents()
            _require(window.library_tabs.currentIndex() == index,
                     "Materials/tracking/review navigation failed")
        for index in range(window.inspector_tabs.count()):
            window.inspector_tabs.setCurrentIndex(index)
            application.processEvents()
            _require(window.inspector_tabs.currentIndex() == index,
                     "Composition/region/performance/export navigation failed")
        window.settings_dialog.hide()
        application.processEvents()
        _require(window.central_preview.width() >= 560
                 and window.canvas.width() > 350 and window.canvas.height() > 150,
                 "The small-window layout squeezed out the working preview")
        _require(window.materials_sidebar.isAncestorOf(window.clip_list),
                 "The materials list is outside the left column")
        _require(all(window.task_sidebar.isAncestorOf(control) for control in
                     (window.track_button, window.pause_button, window.continue_button,
                      window.center_button, window.export_button)),
                 "Core tracking/composition/export controls are outside the right column")
        _require(window.timeline_well.parentWidget() is window.centralWidget()
                 and abs(window.timeline_well.width() - window.workspace_splitter.width()) <= 2,
                 "The source-frame timebar does not span all three columns")
        window.grab().save(str(folder / "three-column-small.png"))
        report["layout"] = {"columns": 3, "window": [window.width(), window.height()],
                            "canvas": [window.canvas.width(), window.canvas.height()],
                            "navigation_preserved": True, "materials_left": True,
                            "tracking_composition_export_right": True, "full_width_timebar": True}

        # A real complete production Job must populate both measured preview and
        # stage state. No manual progress injection is used in this check.
        observed, progress = [], []
        window.live_preview_check.setChecked(True)
        window._start_tracking()
        _require(window.busy and window.worker is not None, "Tracking did not start")
        window.worker.tracking_frame.connect(lambda payload: observed.append(deepcopy(payload)))
        window.worker.progress.connect(lambda value, text: progress.append((value, text)))
        _wait(window)
        panel = window.tracking_progress_panel
        _require(window.track is not None and not window.track.lost_frames,
                 "Bundled demo tracking did not finish with a usable result")
        _require(observed and progress, "Real tracking produced no live frames or progress events")
        _require(panel.state == "completed", "Successful tracking did not mark the stage panel complete")
        _require(all(phase.status == "completed" for phase in panel.phases.values()),
                 "Successful single-material tracking left an unfinished phase")
        _require(window.asset_status_bar.source_path == window.info.path,
                 "Completed live preview did not restore active material metadata")
        visible_canvas = window.canvas.visibleRegion().boundingRect()
        _require(window.canvas.visibleRegion().contains(window.canvas.rect()),
                 "Expanded progress controls clipped the preview canvas at the minimum window size")
        window.grab().save(str(folder / "tracking-stage-completed.png"))
        result = window.track
        report["single_tracking"] = {"live_observations": len(observed),
                                     "progress_events": len(progress), "status": panel.state,
                                     "canvas_height": window.canvas.height(),
                                     "visible_canvas_height": visible_canvas.height(),
                                     "phase_states": {key: phase.status for key, phase in panel.phases.items()}}

        window._set_composition(.57, .43)
        _require(window.track is result, "Composition edit replaced the track")
        window._step_preview(1)
        _require(window.canvas.mode == "compare", "Source-time stepping hid a monitor")
        report["source_preview"] = {"persistent_monitors": True, "composition_preserves_track": True}

        # Cancelling a new actual Job must retain already completed materials
        # and leave unstarted phases waiting instead of claiming completion.
        window._stash_clip()
        completed_clip = window._clip()
        pending_clip = _source_clip(info, last=info.frames - 1)
        window.clips.append(pending_clip)
        window._activate_clip(pending_clip.id)
        window._start_tracking()
        _require(window.busy, "Cancellation check did not start a real tracking job")
        window._cancel()
        _wait(window)
        _require(window.tracking_progress_panel.state == "cancelled",
                 "Cancelled tracking was shown as successful")
        _require(completed_clip.track is result and pending_clip.track is None,
                 "Cancellation discarded a completed material or committed an unfinished result")
        _require(any(phase.status != "completed" for phase in panel.phases.values()),
                 "Cancellation incorrectly completed every stage")
        window.grab().save(str(folder / "tracking-stage-cancelled.png"))
        report["cancellation"] = {"state": panel.state, "previous_result_preserved": True}

        # Two actual spawned workers exercise the portable Windows multiprocessing
        # path and per-material result isolation, rather than a simulated timer.
        from .media import import_video
        second_path = folder / "第二条_红色车 2.mp4"
        if not second_path.exists():
            shutil.copy2(info.path, second_path)
        second_info = import_video(str(second_path), threading.Event())
        clips = [_source_clip(info, last=35),
                 _source_clip(second_info, last=info.frames - 1, mode="precision")]
        _activate(window, clips)
        window.queue_mode.setCurrentIndex(window.queue_mode.findData(2))
        window._start_queue()
        _require(window.busy, "Parallel queue did not start")
        queue_frames, queue_progress, queue_errors, foreign_frames = [], [], [], []
        active_snapshot = None

        def queue_item_done(clip_id, track, error):
            nonlocal active_snapshot
            if clip_id == clips[0].id and track is not None:
                active_snapshot = (window.info, window.frame_index, window.frame,
                                   dict(window.anchors), window.track)

        def queue_frame(clip_id, payload):
            queue_frames.append((clip_id, deepcopy(payload)))
            if (clip_id in getattr(window, "_live_completed_ids", set()) or
                    getattr(window, "_live_follow_clip_id", None) != clip_id):
                return
            try:
                expected_info = next(clip.info for clip in clips if clip.id == clip_id)
                _require(window.asset_status_bar.source_path == expected_info.path,
                         "The persistent metadata disagreed with the material shown on the live canvas")
                if clip_id == clips[1].id:
                    _require(active_snapshot is not None,
                             "Foreign preview followed a second material before the first completed")
                    source_info, source_frame_index, source_frame, source_anchors, source_track = active_snapshot
                    _require(window.info is source_info and window.frame_index == source_frame_index
                             and window.frame is source_frame and window.anchors == source_anchors
                             and window.track is source_track,
                             "Foreign live preview replaced the active material's editing state")
                    foreign_frames.append(payload["frame"])
                    if len(foreign_frames) == 1:
                        window.grab().save(str(folder / "parallel-foreign-metadata.png"))
            except Exception as error:
                queue_errors.append(str(error))

        window.worker.item_result.connect(queue_item_done)
        window.worker.frame_event.connect(queue_frame)
        window.worker.progress.connect(lambda value, text: queue_progress.append((value, text)))
        _wait(window, 120)
        _require(all(clip.track is not None and not clip.track.lost_frames for clip in clips),
                 "Parallel queue did not retain both material results")
        pids = [clip.track.performance.get("queue_worker_pid") for clip in clips]
        _require(len(set(pids)) == 2 and all(pids), "Parallel queue failed to isolate its processes")
        _require({clip_id for clip_id, _ in queue_frames} == {clip.id for clip in clips},
                 "Parallel queue did not deliver measured observations from both workers")
        _require(not queue_errors, "; ".join(queue_errors))
        _require(foreign_frames, "The parallel check did not exercise foreign-material metadata")
        _require(panel.state == "completed" and panel.batch, "Parallel queue stage panel did not finish")
        _require(window.asset_status_bar.source_path == window.info.path,
                 "Queue completion failed to restore active material metadata")
        _require(getattr(window, "_live_foreign_reader", None) is None,
                 "Queue completion leaked a foreign preview decoder")
        window.grab().save(str(folder / "parallel-stage-completed.png"))
        report["parallel_tracking"] = {"completed": 2, "worker_pids": pids,
                                       "measured_frames": len(queue_frames),
                                       "progress_events": len(queue_progress), "status": panel.state,
                                       "different_source_paths": True,
                                       "foreign_metadata_observations": len(foreign_frames),
                                       "foreign_preview_preserves_workspace": True,
                                       "metadata_restored": True}
        return report
    finally:
        if window.busy and window.worker is not None:
            window.worker.cancel()
            window.worker.wait(15000)
            application.processEvents()


def run_ui_check(directory):
    """Standalone CLI entry; QApplication must already exist, as in app.main.

    Use an isolated INI file so a noninteractive release check never overwrites
    the user's UI or compute preferences.
    """
    from PySide6.QtCore import QSettings
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
        details = check_ui_v8(window, folder, info)
        report = {"ok": True, "version": __version__, "seconds": time.monotonic() - started,
                  "ui_v8": details}
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
