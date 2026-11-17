"""Independent clip workspaces and a bounded, cancellable tracking queue."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
import queue
import time
import uuid

import numpy as np

from .models import VideoInfo, Settings, TrackResult, file_fingerprint
from .media import import_video, inspect_video, check_cancel, Cancelled
from .performance import ComputeOptions
from .tracking_pause import TrackingPaused


@dataclass
class Clip:
    info: VideoInfo
    settings: Settings = field(default_factory=Settings)
    anchors: dict = field(default_factory=dict)
    start: int = 0
    end: int = -1
    track: TrackResult | None = None
    frame: int = 0
    project_path: str | None = None
    dirty: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = ""
    precision_info: VideoInfo | None = None
    retrack_previous: TrackResult | None = None
    tracking_resume: dict | None = field(default=None, repr=False)
    timebar_state: dict = field(default_factory=dict)
    section_tracking: bool = False
    tracking_sections: list = field(default_factory=list)
    review_marks: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.end == -1:
            self.end = self.info.frames - 1

    @property
    def ready(self):
        if self.section_tracking:
            from .section_workflow import tracking_ranges
            try:
                tracking_ranges(True, self.tracking_sections, self.anchors, self.start, self.end)
                return True
            except ValueError:
                return False
        return any(self.start <= frame <= self.end for frame in self.anchors)

    @property
    def summary(self):
        if self.status:
            return self.status
        if self.section_tracking and not self.ready and self.track is None:
            return "分段待设置 · 补齐范围及各段选区"
        if not self.ready and self.track is None:
            return "待选区"
        if self.track is None:
            return "就绪"
        if not self.track.performance.get("complete", True):
            frame = self.track.performance.get("tracking_pause", {}).get("frame", self.start)
            from .tracking_pause import can_resume_without_edit
            if can_resume_without_edit(dict(result=self.track,frame=frame,reason=self.track.performance.get('tracking_pause',{}).get('reason'))):
                return '已暂停检查 · 可直接继续'
            return f"已暂停 · 修正第 {frame + 1} 帧后继续"
        from .regions import shape_matches, anchors_match
        from .section_workflow import sections_match
        if (not shape_matches(self.track, self.settings)
                or not sections_match(self.track, self.section_tracking, self.tracking_sections, self.start, self.end)
                or not anchors_match(self.track, self.anchors, self.start, self.end)):
            return "区域已修改 · 待重跟踪"
        return f"待修正 {len(self.track.lost_frames)} 帧" if self.track.lost_frames else "已完成"


def import_many(paths, cancel, progress=None, *, metadata_only=False):
    results = []
    for index, path in enumerate(paths):
        if cancel.is_set():
            break
        try:
            def report(value):
                check_cancel(cancel)
                progress((index + max(0., min(.99, value))) / len(paths),
                         f"导入 {index + 1}/{len(paths)} · {Path(path).name} · 当前 {value:.0%}")
            if progress:report(0.)
            importer = inspect_video if metadata_only else import_video
            info = importer(path, cancel, report) if progress else importer(path, cancel)
            results.append({"info": info, "path": path})
        except Cancelled:
            break
        except Exception as error:
            results.append({"path": path, "error": str(error)})
        if progress:
            progress((index + 1) / len(paths), f"准备素材 {index + 1}/{len(paths)} · {Path(path).name}")
    return results


def _payload(clip):
    result = asdict(clip)
    result.pop("precision_info")
    result.pop("status")
    result.pop("retrack_previous")
    result.pop("tracking_resume")
    result["dirty"] = False
    return result


def save_sequence(path, clips):
    if not 1 <= len(clips) <= 500:
        raise ValueError("一个素材序列应包含 1–500 条素材")
    target = Path(path)
    protected = {Path(value).resolve() for clip in clips for value in (clip.info.path, clip.info.proxy) if value}
    if target.resolve() in protected:
        raise ValueError("序列文件不能覆盖原视频或分析预览")
    for clip in clips:
        clip.settings.validate()
        from .section_workflow import section_configuration
        section_configuration(clip.section_tracking, clip.tracking_sections, clip.info.frames)
        if clip.track is not None:
            clip.track.validate(clip.info.frames)
    data = {"format": "tomato-sequence", "version": 1, "clips": [_payload(clip) for clip in clips]}
    content = json.dumps(data, ensure_ascii=False, default=lambda value: value.tolist(), separators=(",", ":"))
    if len(content.encode("utf-8")) > 100 * 1024**2:
        raise ValueError("序列结果超过 100 MB，请分成多个序列保存")
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def read_sequence(path):
    target = Path(path)
    if target.stat().st_size > 100 * 1024**2:
        raise ValueError("序列文件过大")
    data = json.loads(target.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or data.get("format") != "tomato-sequence"
            or data.get("version") != 1 or not isinstance(data.get("clips"), list)
            or not 1 <= len(data["clips"]) <= 500):
        raise ValueError("不是有效的番茄素材序列")
    clips = []
    for raw in data["clips"]:
        info = VideoInfo(**raw["info"])
        settings = Settings.from_dict(raw["settings"])
        settings.validate()
        anchors = {int(k): tuple(v) for k, v in raw["anchors"].items()}
        start, end = int(raw["start"]), int(raw["end"])
        if not 0 <= start <= end < info.frames:
            raise ValueError("素材片段范围无效")
        for frame, box in anchors.items():
            if (not 0 <= frame < info.frames or len(box) != 4 or not np.isfinite(box).all()
                    or min(box[:2]) < 0 or min(box[2:]) <= 0
                    or box[0] + box[2] > 1.001 or box[1] + box[3] > 1.001):
                raise ValueError("序列选区无效")
        track = None
        if raw.get("track"):
            values = dict(raw["track"])
            for key in ("positions", "boxes", "confidence", "homographies", "confidence_grids"):
                if values.get(key) is not None:
                    values[key] = np.asarray(values[key], dtype=float)
            if values.get("anchors") is not None:
                values["anchors"] = {int(k): tuple(v) for k, v in values["anchors"].items()}
            track = TrackResult(**values)
            track.validate(info.frames)
            if (track.start, track.end) != (start, end):
                raise ValueError("序列轨迹与片段范围不一致")
        clip = Clip(info, settings, anchors, start, end, track,
                    max(start, min(end, int(raw.get("frame", start)))), raw.get("project_path"))
        from .section_workflow import section_configuration
        clip.section_tracking, clip.tracking_sections = section_configuration(
            raw.get("section_tracking", False), raw.get("tracking_sections", []), info.frames)
        # Treat imported identifiers as labels, never as paths or output filenames.
        clip.id = uuid.uuid4().hex
        from .timebar_core import clean_state
        clip.timebar_state = clean_state(raw.get('timebar_state', {}), info.frames, start, end)
        clips.append(clip)
        from .review_records import clean_marks
        clip.review_marks=clean_marks(raw.get('review_marks',{}))
    return clips


def restore_sequence(clips, cancel, progress=None):
    restored = []
    for index, clip in enumerate(clips):
        if cancel.is_set():
            break
        try:
            # Unopened imported assets have no exact CFR proxy yet. Restoring
            # their list must not eagerly transcode the entire library.
            importer = import_video if clip.info.proxy else inspect_video
            info = importer(clip.info.path, cancel)
            if info.fingerprint != clip.info.fingerprint:
                clip.anchors, clip.track = {}, None
                clip.timebar_state = {}
                clip.section_tracking, clip.tracking_sections = False, []
                clip.start, clip.end, clip.frame = 0, info.frames - 1, 0
                clip.status = "原片已变化 · 待选区"
                clip.dirty = True
            clip.info = info
        except Cancelled:
            break
        except Exception as error:
            clip.status = "读取失败：" + str(error)
        restored.append(clip)
        if progress:
            progress((index + 1) / len(clips), f"恢复素材 {index + 1}/{len(clips)}")
    return restored


_cancel = _events = None
_preview_enabled = False
_pause = None
_pause_on_lost = False


def _initialize(cancel, events, cache_directory, preview_enabled=False, pause=None, pause_on_lost=False):
    global _cancel, _events, _preview_enabled, _pause, _pause_on_lost
    _cancel, _events = cancel, events
    _preview_enabled = preview_enabled
    _pause, _pause_on_lost = pause, pause_on_lost
    if cache_directory:
        import os
        os.environ["LOCKFRAME_CACHE_DIR"] = cache_directory


def _track_clip(clip, workers):
    from .tracking import track_video
    if file_fingerprint(clip.info.path) != clip.info.fingerprint:
        raise ValueError("原视频已变化，请重新导入该素材后跟踪")
    if not clip.info.proxy:
        raise ValueError("请先打开此素材完成预览准备，再确认跟踪选区")
    settings = clip.settings
    def progress(value, *details):
        # Throttle IPC to at most about 100 updates per clip.
        bucket = int(value * 100)
        if bucket != progress.last:
            progress.last = bucket
            _events.put((clip.id, value, str(details[0]) if details else ""))
    progress.last = -1
    def frame_event(payload):
        now = time.monotonic()
        complete = payload["completed"] == payload["total"]
        if now - frame_event.last >= .125 or (complete and not frame_event.complete_reported):
            _events.put(("__frame__", clip.id, payload))
            frame_event.last = now
            frame_event.complete_reported |= complete
    frame_event.last = 0.
    frame_event.complete_reported = False
    resources = ComputeOptions(settings.acceleration, settings.memory_percent,
                               max(1, settings.compute_threads // workers), cache_divisor=workers)
    try:
        from .section_workflow import tracking_ranges
        ranges = tracking_ranges(clip.section_tracking, clip.tracking_sections, clip.anchors, clip.start, clip.end)
        result = track_video(clip.info, clip.anchors, clip.start, clip.end, _cancel, progress,
                       mode=settings.tracking_mode, resources=resources, shape=settings.target_shape,
                       polygon_points=settings.polygon_points,
                       perspective=settings.perspective, smoothness=settings.smoothness,
                       adaptive_reference=settings.adaptive_reference, planar_region=settings.planar_region,
                       segmented_tracking=settings.segmented_tracking, segment_frames=settings.segment_frames,
                       confidence_mesh=settings.confidence_mesh,
                       previous=clip.track if clip.track is not None else clip.retrack_previous,
                       pause_on_lost=_pause_on_lost, pause_event=_pause,
                       resume=clip.tracking_resume,
                       drift_review=settings.drift_review,
                       **({"tracking_sections": ranges} if ranges else {}),
                       **({"on_frame": frame_event} if _preview_enabled else {}))
    except TrackingPaused as paused:
        import os
        paused.result.performance.update(queue_worker_pid=os.getpid(), queue_workers=workers,
                                         queue_threads=resources.threads, queue_cache_divisor=resources.cache_divisor)
        raise
    import os
    result.performance.update(queue_worker_pid=os.getpid(), queue_workers=workers,
                              queue_threads=resources.threads, queue_cache_divisor=resources.cache_divisor)
    return result


def run_queue(clips, workers, cancel, progress=None, item_done=None, cache_directory=None,
              on_frame=None, on_item_progress=None, pause_on_lost=False, pause_event=None):
    """Spawn isolated trackers; keep completed results on cancellation or item failure."""
    if workers not in {1, 2}:
        raise ValueError("同时运行数应为 1 或 2")
    outcomes, pending, running = {}, list(clips), {}
    fractions = {clip.id: 0. for clip in clips}
    from .quality import STAGES, parse_phase
    stages = {clip.id: dict.fromkeys(STAGES, 0.) for clip in clips}
    context = mp.get_context("spawn")
    shared_cancel, events = context.Event(), context.Queue()
    shared_pause = context.Event()
    def notify(clip, result, error):
        paused = result if isinstance(result, TrackingPaused) else None
        outcomes[clip.id] = {"track": None if paused else result, "error": error}
        if paused:
            outcomes[clip.id]["pause"] = paused
        if not paused and error != "等待继续":
            fractions[clip.id] = 1.
        if item_done:
            item_done(clip.id, result, error)
    def drain_events():
        try:
            while True:
                clip_id, value, text = events.get_nowait()
                if clip_id == "__frame__":
                    if on_frame is not None:
                        on_frame(value, text)
                    continue
                fractions[clip_id] = max(fractions.get(clip_id, 0), min(1., value))
                if on_item_progress is not None and clip_id not in outcomes:
                    on_item_progress(clip_id, float(value), text)
                parsed = parse_phase(text)
                if parsed:
                    stage, fraction, _ = parsed
                    for name in STAGES[:STAGES.index(stage)]:
                        stages[clip_id][name] = 1.
                    stages[clip_id][stage] = fraction
        except queue.Empty:
            pass
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_initialize,
                                 initargs=(shared_cancel, events, cache_directory, on_frame is not None,
                                           shared_pause, pause_on_lost)) as executor:
            while pending or running:
                if pause_event is not None and pause_event.is_set():
                    shared_pause.set()
                if cancel.is_set():
                    shared_cancel.set()
                    for clip in pending:
                        notify(clip, None, "已取消")
                    pending.clear()
                elif shared_pause.is_set():
                    for clip in pending:
                        notify(clip, None, "等待继续")
                    pending.clear()
                while pending and len(running) < workers and not cancel.is_set() and not shared_pause.is_set():
                    clip = pending.pop(0)
                    if not clip.ready:
                        notify(clip, None, "未选目标，已跳过")
                        continue
                    if item_done:
                        item_done(clip.id, None, "正在跟踪")
                    running[executor.submit(_track_clip, clip, workers)] = clip
                drain_events()
                if running:
                    finished, _ = wait(running, timeout=.1, return_when=FIRST_COMPLETED)
                    drain_events()
                    for future in finished:
                        clip = running.pop(future)
                        try:
                            notify(clip, future.result(), "")
                        except TrackingPaused as paused:
                            shared_pause.set()
                            notify(clip, paused, "已暂停")
                        except Cancelled:
                            notify(clip, None, "已取消")
                        except Exception as error:
                            notify(clip, None, "失败：" + str(error))
                if progress:
                    progress(sum(fractions.values()) / max(1, len(clips)),
                             f"素材任务 {len(outcomes)}/{len(clips)} · 运行 {len(running)} · 等待 {len(pending)}")
                    for name in STAGES:
                        fraction = sum(value[name] for value in stages.values()) / max(1, len(clips))
                        if fraction > 0:
                            progress(sum(fractions.values()) / max(1, len(clips)), f"[汇总:{name}:{fraction:.3f}]")
        # Process shutdown flushes multiprocessing feeder threads. A very short
        # clip may finish before the first poll; retain its genuine observations.
        drain_events()
        return outcomes
    finally:
        events.close()
        events.join_thread()


def clip_thumbnail(clip, width=160, height=90):
    """Return JPEG bytes for the clip's first retained frame, without Qt state.

    Call in a background worker. The UI owns caching and placeholders; errors
    must not prevent importing or selecting a clip. No source files are written.
    """
    import cv2
    from .media import FrameReader

    if (type(width) is not int or type(height) is not int
            or not 1 <= width <= 1024 or not 1 <= height <= 1024):
        raise ValueError("缩略图尺寸应在 1–1024 像素之间")
    if not 0 <= clip.start <= clip.end < clip.info.frames:
        raise ValueError("素材片段范围无效")
    reader = FrameReader(clip.info.proxy or clip.info.path)
    try:
        frame = reader.read(clip.start)
    finally:
        reader.close()
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    frame = cv2.resize(frame, (max(1, round(source_width * scale)),
                               max(1, round(source_height * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("无法生成素材缩略图")
    return encoded.tobytes()
