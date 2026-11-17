"""Production-window and media checks shared by source and frozen v0.7 builds."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import shutil
import subprocess
import threading
import time

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from .demo import demo_frame
from .export import export_video
from .media import FrameReader, NO_WINDOW, find_tool, import_video, prepare_precision_video, probe
from .models import Settings
from .sequence import Clip
from .stabilization import build_plan, preview_frame


def _wait_window(window, seconds=60):
    application = QApplication.instance()
    deadline = time.monotonic() + seconds
    while window.busy and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(.005)
    if window.busy:
        window.worker.cancel()
        window.worker.wait(15000)
        application.processEvents()
        raise RuntimeError("v0.7 production-window tracking timed out")
    application.processEvents()


def _verify_output(path, track, info, *, audio):
    output = probe(str(path))
    raw = subprocess.check_output([find_tool("ffprobe"), "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        creationflags=NO_WINDOW)
    frames = int(json.loads(raw)["streams"][0]["nb_read_frames"])
    if frames != track.count or abs(output.duration - (track.count / info.fps)) > 1 / info.fps:
        raise RuntimeError("Source-time export frame count or duration mismatch")
    if output.has_audio != audio or abs(output.fps - info.fps) > .001:
        raise RuntimeError("Source-time export audio policy or frame rate mismatch")
    return {"path": str(path), "frames": frames, "duration": output.duration,
            "width": output.width, "height": output.height, "audio": output.has_audio}


def _check_queue_follow(window, folder, info):
    """Two real sequential worker jobs must advance the production live canvas."""
    second_source = folder / "second-demo.mp4"
    if not second_source.exists():
        shutil.copy2(info.path, second_source)
    second_info = import_video(str(second_source), threading.Event())
    seed = min(55, info.frames - 1)
    settings = Settings(tracking_mode="precision", acceleration="cpu", compute_threads=2,
                        resolution=0)
    clips = [Clip(source, replace(settings),
                  {seed: demo_frame(seed, source.width, source.height)[1]}, 0, source.frames - 1)
             for source in (info, second_info)]
    for clip in clips:
        clip.precision_info = prepare_precision_video(clip.info, threading.Event())
    window.clips, window.active_clip_id = clips, None
    window._activate_clip(clips[0].id)
    _wait_window(window)
    window._refresh_sequence({clip.id for clip in clips})
    window.queue_mode.setCurrentIndex(window.queue_mode.findData(1))
    window.library_tabs.setCurrentIndex(1)
    observed = {clip.id: [] for clip in clips}
    errors, completed = [], []
    verification = {clips[0].id: FrameReader(clips[0].precision_info.proxy),
                    clips[1].id: FrameReader(clips[1].info.proxy)}
    baseline = None
    foreign_saved = False

    def item_done(clip_id, track, error):
        nonlocal baseline
        if track is not None:
            completed.append(clip_id)
            if clip_id == clips[0].id:
                baseline = (window.info, window.frame_index, deepcopy(window.anchors),
                            window.track, window.frame.copy())

    def frame_shown(clip_id, payload):
        nonlocal foreign_saved
        if (clip_id in getattr(window, "_live_completed_ids", set())
                or getattr(window, "_live_follow_clip_id", None) != clip_id):
            return
        try:
            if not window.busy:
                raise RuntimeError("Queue live frame arrived after completion")
            np.testing.assert_array_equal(window.canvas.source, verification[clip_id].read(payload["frame"]))
            if window.canvas.state != payload["state"]:
                raise RuntimeError("Queue live canvas state differs from worker observation")
            if Path(window._clip(clip_id).info.path).name not in window.live_preview_status.text():
                raise RuntimeError("Queue live preview omitted the currently followed filename")
            if clip_id == clips[1].id:
                if baseline is None or completed != [clips[0].id]:
                    raise RuntimeError("Sequential live preview followed the next clip before the first finished")
                original_info, original_index, original_anchors, original_track, original_frame = baseline
                if (window.info is not original_info or window.frame_index != original_index
                        or window.anchors != original_anchors or window.track is not original_track):
                    raise RuntimeError("Foreign queue preview replaced the active clip's editing workspace")
                np.testing.assert_array_equal(window.frame, original_frame)
                if not foreign_saved:
                    window.grab().save(str(folder / "queue-live-tracking-preview.png"))
                    foreign_saved = True
            observed[clip_id].append(deepcopy(payload))
        except Exception as error:
            errors.append(str(error))

    window._start_queue()
    if not window.busy or window.worker is None:
        raise RuntimeError("Production sequential queue did not start")
    window.worker.item_result.connect(item_done)
    window.worker.frame_event.connect(frame_shown)
    try:
        _wait_window(window, seconds=120)
    finally:
        for reader in verification.values():
            reader.close()
    if errors:
        raise RuntimeError("Sequential queue live-follow check failed: " + "; ".join(errors))
    if completed != [clip.id for clip in clips] or any(clip.track is None or clip.track.lost_frames for clip in clips):
        raise RuntimeError("Sequential queue failed to retain both measured tracking results")
    if any(not observed[clip.id] for clip in clips):
        raise RuntimeError("Sequential precision queue did not visibly follow both working clips")
    for clip in clips:
        for payload in observed[clip.id]:
            np.testing.assert_array_equal(payload["box"], clip.track.boxes[payload["frame"] - clip.start])
    if getattr(window, "_live_foreign_reader", None) is not None:
        raise RuntimeError("Sequential queue left its foreign preview reader open")
    return {"workers": 1, "completed": 2, "different_source_files": True,
            "displayed_observations": [len(observed[clip.id]) for clip in clips],
            "displayed_frames": [[value["frame"] for value in observed[clip.id]] for clip in clips],
            "followed_next_after_first_completed": True,
            "foreign_preview_preserved_active_workspace": True, "foreign_reader_closed": True}


def check_v7(window, folder, info):
    """Exercise real MainWindow actions; synthetic footage is explicitly reported."""
    folder = Path(folder)
    application = QApplication.instance()
    first, last, seed = 20, min(100, info.frames - 1), min(55, info.frames - 1)
    if last < seed:
        raise RuntimeError("Synthetic demo is too short for v0.7 release checks")
    anchor = demo_frame(seed, info.width, info.height)[1]
    clip = Clip(info, Settings(acceleration="cpu", compute_threads=2, resolution=0),
                {seed: anchor}, first, last)
    window.clips, window.active_clip_id = [clip], None
    window._activate_clip(clip.id)
    _wait_window(window)
    window._refresh_sequence({clip.id})
    window.library_tabs.setCurrentIndex(1)
    window.inspector_tabs.setCurrentIndex(0)
    window.live_preview_check.setChecked(True)
    saved_anchors = deepcopy(window.anchors)
    observations, observation_errors = [], []
    verification_reader = FrameReader(info.proxy)
    live_saved = False

    def observe(payload):
        nonlocal live_saved
        try:
            if not window.busy:
                raise RuntimeError("Live observation arrived after tracking completion")
            index = int(payload["frame"])
            if window.frame_index != index or window.canvas.state != payload["state"]:
                raise RuntimeError("Live preview is not displaying its measured tracking frame")
            np.testing.assert_array_equal(window.frame, verification_reader.read(index))
            if window.anchors != saved_anchors:
                raise RuntimeError("Live preview modified tracking anchors")
            observations.append(deepcopy(payload))
            if not live_saved:
                window.grab().save(str(folder / "live-tracking-preview.png"))
                live_saved = True
        except Exception as error:
            observation_errors.append(str(error))

    window._start_tracking()
    if not window.busy or window.worker is None:
        raise RuntimeError("MainWindow did not start the production tracking job")
    window.worker.tracking_frame.connect(observe)
    try:
        _wait_window(window)
    finally:
        verification_reader.close()
    if observation_errors:
        raise RuntimeError("Live preview check failed: " + "; ".join(observation_errors))
    if not observations or window.track is None or window.track.lost_frames:
        raise RuntimeError("Production tracking did not emit live measured observations")
    result = window.track
    if window.anchors != saved_anchors:
        raise RuntimeError("Production tracking changed manual anchors")
    for payload in observations:
        index = payload["frame"] - result.start
        np.testing.assert_array_equal(payload["box"], result.boxes[index])
        if payload["state"] != result.states[index] or payload["confidence"] != result.confidence[index]:
            raise RuntimeError("Live observation differs from the completed tracking result")
    original_positions = result.positions.copy()

    settings = window.settings()
    output = export_video(info, result, settings, str(folder / "source-time-audio.mp4"), threading.Event())
    output_report = _verify_output(output["path"], result, info, audio=info.has_audio)
    window._seek(result.start + 1)
    window._step_preview(1)
    if window.frame_index != result.start + 2 or window.timeline.value() != window.frame_index:
        raise RuntimeError("Source-time preview skipped or remapped a frame")
    window._set_composition(.57, .43)
    if window.track is not result or not np.array_equal(result.positions, original_positions):
        raise RuntimeError("Composition edit or export mutated tracking")
    window._set_view("compare")
    application.processEvents()
    window.grab().save(str(folder / "workflow-preview.png"))
    queue_follow = _check_queue_follow(window, folder, info)
    return {"footage": "synthetic demo; original car footage acceptance remains pending",
            "live_preview": {"observations_before_completion": len(observations),
                             "source_frames": [payload["frame"] for payload in observations],
                             "exact_proxy_pixels": True, "matches_completed_observations": True,
                             "anchors_preserved": True},
            "source_time_preview": True, "composition_preserves_tracking": True,
            "sequential_queue_follow": queue_follow, "source_export": output_report}
