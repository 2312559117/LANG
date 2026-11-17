"""Non-interactive release check, using only the supplied synthetic demo."""
from __future__ import annotations

from pathlib import Path
import json
import threading
import time
import traceback

import numpy as np
import cv2
from PySide6.QtWidgets import QApplication

from . import __version__
from .demo import demo_frame
from .export import export_video
from .media import find_tool, import_video, prepare_precision_video, probe, resource_root, cache_root
from .models import Settings
from .tracking import track_video
from .performance import ComputeOptions, TemplateMatcher, memory_status


def check_appearance_learning():
    """Compare fixed/learned references against independently known target centers."""
    from .tracking import FeatureTracker
    from .precision_tracking import PrecisionFeatureTracker
    from .performance import compute_scope
    rng = np.random.default_rng(421)
    patches = []
    for _ in range(2):
        patch = cv2.GaussianBlur(rng.normal(0, 1, (80, 80)).astype(np.float32), (5, 5), .9)
        patches.append(np.clip(patch / patch.std() * 34 + 127, 12, 242).astype(np.uint8))
    images, centers = [], []
    for index, alpha in enumerate(np.linspace(0, .59, 66)):
        center = np.array([160. + index * .6, 120. + index * .15])
        matrix = np.array([[1., 0., center[0] - 40], [0., 1., center[1] - 40]])
        patch = cv2.addWeighted(patches[0], 1 - alpha, patches[1], alpha, 0)
        image = np.full((240, 400), 24, np.uint8)
        rendered = cv2.warpAffine(patch, matrix, (400, 240))
        mask = cv2.warpAffine(np.full_like(patch, 255), matrix, (400, 240))
        image[mask > 250] = rendered[mask > 250]
        images.append(image)
        centers.append(center)
    report = {}
    with compute_scope(ComputeOptions(acceleration="cpu")):
        for mode, cls in (("normal", FeatureTracker), ("precision", PrecisionFeatureTracker)):
            modes = {}
            for enabled in (False, True):
                tracker = cls(images[0], [120, 80, 80, 80], adaptive_reference=enabled)
                errors, lost = [], 0
                before = time.perf_counter()
                for image, expected in zip(images[1:], centers[1:]):
                    value = tracker.step(image)
                    if value.state == "lost":
                        lost += 1
                    else:
                        errors.append(float(np.linalg.norm(value.box[:2] + value.box[2:] / 2 - expected)))
                modes["adaptive" if enabled else "fixed"] = {
                    "analyzed_frames": 65, "valid_frames": 65 - lost,
                    "lost_frames": lost, "max_error_px": max(errors, default=float("inf")),
                    "updates": tracker.adaptation_updates, "seconds": time.perf_counter() - before}
            # The v0.7.24 persistent adjacent-frame kernel can carry the
            # ordinary fixed-reference tracker through this complete blend.
            # Adaptive learning still has to remain stable and active; the
            # precision path keeps the stronger comparative requirement.
            valid = (modes["adaptive"]["valid_frames"] >= 60
                     if mode == "normal" else
                     modes["adaptive"]["valid_frames"] >= max(
                         60, modes["fixed"]["valid_frames"] + 3))
            if (not valid or modes["adaptive"]["max_error_px"] >= .8
                    or modes["adaptive"]["updates"] < 3):
                raise RuntimeError("Adaptive reference failed its measured appearance-change check")
            report[mode] = modes
    return report


def check_projective_region():
    """Measure full surface corners, including recovery after complete occlusion."""
    from .tracking import FeatureTracker
    from .precision_tracking import PrecisionFeatureTracker
    from .planar import UNIT_CORNERS, project
    from .performance import compute_scope
    rng = np.random.default_rng(702)
    patch = cv2.GaussianBlur(rng.integers(25, 230, (100, 120), dtype=np.uint8), (3, 3), .55)
    cv2.putText(patch, "PLANE", (6, 55), cv2.FONT_HERSHEY_SIMPLEX, .65, 245, 2)
    frames, targets = [], []
    for index in range(25):
        t = index / 24
        corners = np.array([[110 + 20*t, 70 + 5*t], [230 - 8*t, 70 + 14*t],
                            [230 + 12*t, 170 - 7*t], [110 - 10*t, 170 + 9*t]], np.float32)
        matrix = cv2.getPerspectiveTransform((UNIT_CORNERS * [120, 100]).astype(np.float32), corners)
        image = np.full((260, 400), 20, np.uint8)
        rendered = cv2.warpPerspective(patch, matrix, (400, 260))
        support = cv2.warpPerspective(np.full_like(patch, 255), matrix, (400, 260))
        image[support > 250] = rendered[support > 250]
        frames.append(image)
        targets.append(corners)
    report = {}
    with compute_scope(ComputeOptions(acceleration="cpu")):
        for mode, cls in (("normal", FeatureTracker), ("precision", PrecisionFeatureTracker)):
            tracker = cls(frames[0], [110, 70, 120, 100], shape="ellipse",
                          perspective=True, adaptive_reference=True, planar_region=True)
            errors, recovered = [], False
            for index, image in enumerate(frames[1:], 1):
                value = tracker.step(np.full_like(image, 20) if index == 14 else image)
                if index == 14:
                    if value.state != "lost":
                        raise RuntimeError("Fully hidden plane was falsely accepted")
                    continue
                if value.state == "lost" or value.homography is None:
                    raise RuntimeError("Projective masked tracking or recovery failed")
                errors.extend(np.linalg.norm(project(UNIT_CORNERS, value.homography) - targets[index], axis=1))
                recovered |= index == 15
            if max(errors) >= 1.5 or not recovered:
                raise RuntimeError("Projective corner error exceeded 1.5 analysis pixels")
            report[mode] = {"visible_frames": 23, "occluded_frames_rejected": 1,
                            "recovered_next_frame": recovered, "max_corner_error_px": float(max(errors))}
    normalized = np.diag([1/400, 1/260, 1]) @ value.homography
    return report, cv2.cvtColor(frames[-1], cv2.COLOR_GRAY2BGR), value.box / [400, 260, 400, 260], normalized


def run_check(directory: str) -> int:
    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        info = import_video(str(resource_root() / "assets" / "demo.mp4"), threading.Event())
        seed = min(55, info.frames - 1)
        anchor = demo_frame(seed)[1]
        track = track_video(info, {seed: anchor}, 0, info.frames - 1, threading.Event())
        expected = []
        for index in range(info.frames):
            box = demo_frame(index)[1]
            expected.append([box[0] + box[2] / 2, box[1] + box[3] / 2])
        errors = np.linalg.norm((track.positions - expected) * [info.width, info.height], axis=1)
        if track.lost_frames or errors.max() > 3:
            raise RuntimeError("Packaged tracking failed the known-trajectory check")
        result = export_video(info, track, Settings(), str(folder / "demo-stabilized.mp4"), threading.Event())
        exported = probe(result["path"])
        if not exported.has_audio or abs(exported.duration - info.frames / info.fps) > 0.1:
            raise RuntimeError("Packaged export lost audio or changed duration")
        from .app import MainWindow
        application = QApplication.instance()
        window = MainWindow()
        if any(window.inspector_tabs.tabText(i) == "AI" for i in range(window.inspector_tabs.count())):
            raise RuntimeError("Removed cloud page is still present")
        if window.preferences.childGroups().count("ai"):
            raise RuntimeError("Retired cloud settings were not cleared")
        original_adaptive = window.adaptive_reference_check.isChecked()
        original_planar = window.planar_region_check.isChecked()
        window._ready = False
        window.adaptive_reference_check.setChecked(False)
        window.planar_region_check.setChecked(False)
        window._ready = True
        window.resize(1620, 960)
        window.show()
        application.processEvents()
        window.grab().save(str(folder / "empty-preview.png"))
        window._import_done(info)
        window.anchors = {seed: anchor}
        window._refresh_anchors()
        window._tracking_done(track)
        window._sync_ui()
        window._seek(65)
        application.processEvents()
        window.grab().save(str(folder / "release-preview.png"))
        # Exercise the new trim UI and its real export path in the packaged EXE.
        from .trim import TrimDialog
        first, last = 15, min(info.frames - 1, 149)
        dialog = TrimDialog(info, first, last, seed, window)
        dialog.show()
        application.processEvents()
        dialog.grab().save(str(folder / "trim-preview.png"))
        dialog.accept()
        if dialog.reader is not None or dialog.timer.isActive():
            raise RuntimeError("Trim dialog did not release its preview resources")
        window._apply_trim(dialog.start_frame, dialog.end_frame)
        trimmed = track_video(info, {seed: anchor}, first, last, threading.Event())
        if trimmed.lost_frames:
            raise RuntimeError("Trimmed tracking failed")
        trimmed_output = export_video(info, trimmed, Settings(), str(folder / "demo-trimmed.mp4"), threading.Event())
        trimmed_info = probe(trimmed_output["path"])
        if (not trimmed_info.has_audio or
                abs(trimmed_info.duration - trimmed.count / info.fps) > .06):
            raise RuntimeError("Trimmed export lost audio or changed duration")
        window._tracking_done(trimmed)
        window._sync_ui()
        window._seek(65)
        application.processEvents()
        window.grab().save(str(folder / "trimmed-studio-preview.png"))
        # Run the optional algorithm inside the frozen build, in both time
        # directions. The UI uses the same high-detail analysis cache to pick.
        precise_first, precise_last = max(0, seed - 15), min(info.frames - 1, seed + 20)
        before = time.perf_counter()
        precise = track_video(info, {seed: anchor}, precise_first, precise_last,
                              threading.Event(), mode="precision", resources=ComputeOptions(acceleration="gpu"))
        precise_seconds = time.perf_counter() - before
        precise_expected = np.asarray(expected)[precise_first:precise_last + 1]
        precise_errors = np.linalg.norm((precise.positions - precise_expected)
                                       * [info.width, info.height], axis=1)
        if precise.lost_frames or precise_errors.max() > 3 or precise.tracking_mode != "precision":
            raise RuntimeError("Packaged precision mode failed its known-trajectory check")
        precise_output = export_video(info, precise, Settings(tracking_mode="precision"),
                                      str(folder / "demo-precision.mp4"), threading.Event())
        precise_exported = probe(precise_output["path"])
        if not precise_exported.has_audio or abs(precise_exported.duration - precise.count / info.fps) > .06:
            raise RuntimeError("Precision export lost audio or changed duration")
        window._ready = False
        window.tracking_combo.setCurrentIndex(window.tracking_combo.findData("precision"))
        window._ready = True
        window._precision_preview_done(prepare_precision_video(info, threading.Event()))
        window._apply_trim(precise_first, precise_last)
        window._tracking_done(precise)
        window._set_view("source")
        window._seek(seed)
        window.canvas.set_zoom(3.)
        window.pick_mode.setCurrentIndex(window.pick_mode.findData("point"))
        window._sync_ui()
        application.processEvents()
        if abs(window.preview_zoom.value() - 3.) > .001:
            raise RuntimeError("Preview zoom controls are not synchronized")
        window.grab().save(str(folder / "precision-zoom-preview.png"))
        window.shape_combo.setCurrentIndex(window.shape_combo.findData("circle"))
        window.pick_mode.setCurrentIndex(window.pick_mode.findData("adjust"))
        window.inspector_tabs.setCurrentIndex(1)
        window.resize(1160, 760)
        application.processEvents()
        window.grab().save(str(folder / "circle-edit-small-preview.png"))
        circular = track_video(info, window.anchors, precise_first, precise_last,
                               threading.Event(), mode="precision", shape="circle",
                               resources=ComputeOptions(acceleration="gpu"))
        circle_errors = np.linalg.norm((circular.positions - precise_expected) * [info.width, info.height], axis=1)
        if circular.lost_frames or circle_errors.max() > 3:
            raise RuntimeError("Circular masked tracking failed")
        window._tracking_done(circular)
        window._sync_ui()
        window.inspector_tabs.setCurrentIndex(2)
        window._hardware_updated({"memory": memory_status()})
        window.resize(1620, 960)
        application.processEvents()
        window.grab().save(str(folder / "performance-preview.png"))
        window.adaptive_reference_check.setChecked(True)
        if not window.settings().adaptive_reference or window.track is not None:
            raise RuntimeError("Adaptive setting did not invalidate the old track")
        window.resize(1160, 760)
        application.processEvents()
        window.grab().save(str(folder / "adaptive-settings-preview.png"))
        projective_report, plane_image, plane_box, plane_matrix = check_projective_region()
        window._apply_planar_preset()
        if not (window.settings().planar_region and window.settings().perspective
                and window.settings().adaptive_reference and window.settings().tracking_mode == "precision"):
            raise RuntimeError("Planar preset did not enable the four required settings")
        window.canvas.set_zoom(1.)
        window.canvas.selection_shape = "ellipse"
        window.canvas.set_frames(plane_image, box=plane_box, confidence=.99,
                                 state="tracked", homography=plane_matrix)
        application.processEvents()
        window.grab().save(str(folder / "projective-region-preview.png"))
        # Restore the user's original preference after the UI self-check.
        window._ready = False
        window.adaptive_reference_check.setChecked(original_adaptive)
        window.planar_region_check.setChecked(original_planar)
        window.preferences.setValue("adaptive_reference", original_adaptive)
        window.preferences.setValue("planar_region", original_planar)
        window._ready = True
        learned = track_video(info, {seed: anchor}, precise_first, precise_last,
                              threading.Event(), mode="precision", adaptive_reference=True)
        if learned.lost_frames or not learned.performance["adaptive_reference"]:
            raise RuntimeError("Packaged adaptive tracking failed")
        appearance_report = check_appearance_learning()
        gpu_available = TemplateMatcher("gpu").available
        if gpu_available and not precise.performance.get("gpu_calls"):
            raise RuntimeError("Compatible GPU detected but precision search did not execute GPU calls")
        # Exercise the real Qt queue and spawned trackers, including frozen Windows.
        from .sequence import Clip, run_queue, save_sequence, read_sequence
        queue_clips = [Clip(info, Settings(acceleration="cpu", compute_threads=4),
                            {seed: anchor}, precise_first, precise_last) for _ in range(2)]
        import shutil
        second_source = folder / "second-demo.mp4"
        shutil.copy2(info.path, second_source)
        queue_clips[1].info = import_video(str(second_source), threading.Event())
        window.clips, window.active_clip_id = queue_clips, None
        window._activate_clip(queue_clips[0].id)
        window._refresh_sequence({clip.id for clip in queue_clips})
        window.queue_mode.setCurrentIndex(window.queue_mode.findData(2))
        window._start_queue()
        deadline = time.monotonic() + 60
        while window.busy and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(.01)
        if window.busy:
            window.worker.cancel()
            window.worker.wait(15000)
            raise RuntimeError("Batch queue timed out")
        queue_tracks = [clip.track for clip in window.clips]
        if any(value is None or value.lost_frames for value in queue_tracks):
            raise RuntimeError("Batch queue did not preserve each clip's tracking result")
        process_ids = [value.performance["queue_worker_pid"] for value in queue_tracks]
        if len(set(process_ids)) != 2:
            raise RuntimeError("Parallel tracking did not use isolated workers")
        sequential = run_queue([Clip(info, Settings(acceleration="cpu", compute_threads=4),
                                     {seed: anchor}, precise_first, precise_last)], 1, threading.Event())
        sequential_track = next(iter(sequential.values()))["track"]
        if sequential_track is None or sequential_track.lost_frames:
            raise RuntimeError("Sequential batch tracking failed")
        np.testing.assert_allclose(sequential_track.positions, queue_tracks[0].positions, atol=1e-5)
        save_sequence(folder / "demo.locksequence", window.clips)
        restored_sequence = read_sequence(folder / "demo.locksequence")
        np.testing.assert_array_equal(restored_sequence[1].track.positions, queue_tracks[1].positions)
        window.resize(1620, 960)
        window._seek(seed)
        application.processEvents()
        window.grab().save(str(folder / "sequence-preview.png"))
        batch_report = {"parallel_completed": 2, "worker_pids": process_ids,
                        "sequential_completed": 1, "sequence_roundtrip": True}
        from .checkpoints import clear_checkpoints
        from .quality import parse_phase
        from .media import Cancelled
        checkpoint_info = queue_clips[1].info
        clear_checkpoints(checkpoint_info)
        checkpoint_options = dict(mode="precision", perspective=True, planar_region=True,
                                  adaptive_reference=True, confidence_mesh=True,
                                  segmented_tracking=True, segment_frames=10)
        checkpoint_cancel = threading.Event()
        def interrupt_checkpoint(value, text):
            phase = parse_phase(text)
            if phase and phase[0] == "计算" and phase[1] > .55:
                checkpoint_cancel.set()
        interrupted = False
        try:
            track_video(checkpoint_info, {seed: anchor}, precise_first, precise_last,
                         checkpoint_cancel, interrupt_checkpoint, **checkpoint_options)
        except Cancelled:
            interrupted = True
        phase_settings = Settings(tracking_mode="precision", perspective=True, planar_region=True,
                                  adaptive_reference=True, confidence_mesh=True,
                                  segmented_tracking=True, segment_frames=10)
        resume_clip = Clip(checkpoint_info, phase_settings, {seed: anchor}, precise_first, precise_last)
        resumed_outcomes = run_queue([resume_clip], 1, threading.Event(), window._on_progress,
                                     cache_directory=str(cache_root()))
        checkpoint_result = resumed_outcomes[resume_clip.id]["track"]
        if checkpoint_result is None:
            raise RuntimeError("Checkpoint worker failed: " + resumed_outcomes[resume_clip.id]["error"])
        if (not interrupted or checkpoint_result.performance["resumed_frames"] < 10
                or checkpoint_result.confidence_grids is None or checkpoint_result.lost_frames):
            raise RuntimeError("Packaged checkpoint resume or confidence grid failed")
        checkpoint_errors = np.linalg.norm((checkpoint_result.positions - precise_expected) * [info.width, info.height], axis=1)
        if checkpoint_errors.max() > 3:
            raise RuntimeError("Resumed tracking exceeded known-trajectory error")
        resume_clip.track = checkpoint_result
        save_sequence(folder / "checkpoint.locksequence", [resume_clip])
        np.testing.assert_array_equal(read_sequence(folder / "checkpoint.locksequence")[0].track.confidence_grids,
                                      checkpoint_result.confidence_grids)
        window._ready = False
        window._restore_settings(phase_settings)
        window._ready = True
        window._reset_stages()
        window._update_stage("缓存", 1.)
        window._tracking_done(checkpoint_result)
        window._set_view("source")
        window._seek(seed + 8)
        application.processEvents()
        window.grab().save(str(folder / "checkpoint-grid-preview.png"))
        checkpoint_report = {"interrupted": interrupted,
                             "resumed_worker_pid": checkpoint_result.performance["queue_worker_pid"],
                             "resumed_frames": checkpoint_result.performance["resumed_frames"],
                             "max_error_px": float(checkpoint_errors.max()), "grid_roundtrip": True,
                             "frames": checkpoint_result.count, "grid_shape": list(checkpoint_result.confidence_grids.shape),
                             "quality": checkpoint_result.performance["quality"]}
        from .phase1_check import check_phase1
        phase1_report = check_phase1(window, folder, queue_clips[1])
        from .v7_check import check_v7
        v7_report = check_v7(window, folder, info)
        from .ui_v8_check import check_ui_v8
        ui_report = check_ui_v8(window, folder / "single-column", info)
        from .v072_check import check_ui_v9
        workflow_report = check_ui_v9(window, folder / "correction-workflow", info)
        window.preferences.setValue("adaptive_reference", original_adaptive)
        window.preferences.setValue("planar_region", original_planar)
        for clip in window.clips:
            clip.dirty = False
        window._sequence_changed = False
        window.dirty = False
        window.close()
        report = {"ok": True, "version": __version__, "frames": track.count, "lost_frames": len(track.lost_frames),
                  "mean_tracking_error_px": float(errors.mean()), "max_tracking_error_px": float(errors.max()),
                  "export_has_audio": exported.has_audio, "export_size": [exported.width, exported.height],
                  "export_duration": exported.duration, "ffmpeg": find_tool("ffmpeg"),
                  "encoder": result["encoder"], "gpu_available": gpu_available,
                  "appearance_learning": appearance_report,
                  "projective_region": projective_report,
                  "cloud_feature_removed": True,
                  "batch_queue": batch_report,
                  "phase1": phase1_report,
                  "v7": v7_report,
                  "single_column": ui_report,
                  "correction_workflow": workflow_report,
                  "checkpoint_grid": checkpoint_report,
                  "ffprobe": find_tool("ffprobe"), "export": result["path"],
                  "precision": {"frames": precise.count, "lost_frames": len(precise.lost_frames),
                                "analysis_size": precise.analysis_size, "seconds": precise_seconds,
                                "performance": precise.performance,
                                "max_tracking_error_px": float(precise_errors.max()),
                                "duration": precise_exported.duration, "has_audio": precise_exported.has_audio,
                                "export": precise_output["path"]},
                  "circle": {"lost_frames": len(circular.lost_frames), "max_error_px": float(circle_errors.max()),
                             "performance": circular.performance},
                  "preview_zoom": window.canvas.zoom_factor,
                  "trim": {"start_frame": first, "end_frame": last, "frames": trimmed.count,
                           "duration": trimmed_info.duration, "has_audio": trimmed_info.has_audio,
                           "export": trimmed_output["path"]}}
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        (folder / "report.json").write_text(json.dumps({"ok": False, "error": traceback.format_exc()},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
