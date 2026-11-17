"""Sequential exports through the single renderer, with private queue snapshots."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import pickle
import time

from .export import export_video
from .models import file_fingerprint, source_identity
from .media import Cancelled
from .stabilization import build_plan
from .logging import get_logger
from .regions import shape_matches, anchors_match


@dataclass(frozen=True)
class ExportTask:
    clip_id: str
    name: str
    snapshot: bytes
    destination: str
    result_revision: str
    audio_policy: str
    reason: str = ""


def prepare_exports(clips, directory):
    """Serialize trusted in-memory data, never load a pickle from a file/user input."""
    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # Test permissions before constructing any jobs, without touching existing files.
    import tempfile
    with tempfile.TemporaryFile(dir=folder):
        pass
    protected = {Path(p).resolve() for c in clips for p in (c.info.path, c.info.proxy) if p}
    reserved, tasks = set(), []
    for clip in clips:
        reason = ""
        stem = Path(clip.info.path).stem[:100]
        destination = folder / (stem + "_番茄跟踪.mp4")
        suffix = 1
        while destination.exists() or destination in protected or destination in reserved:
            destination = folder / f"{stem}_番茄跟踪_{suffix}.mp4"
            suffix += 1
        reserved.add(destination)
        try:
            validate_clip(clip)
        except Exception as error:
            reason = str(error)
        # Bytes are immutable. Arrays, dictionaries and settings remain isolated even
        # when the active clip is edited in place after this call.
        snapshot = pickle.dumps(deepcopy(clip), protocol=5)
        revision = hashlib.sha256(snapshot).hexdigest()
        audio = "保留首条音轨" if clip.info.has_audio else "无音轨"
        if clip.settings.retime.get('enabled') and clip.info.has_audio:
            audio='映射同步；倒放/定格自动静音' if clip.settings.retime.get('audio_policy')=='mapped' else '变速静音'
        tasks.append(ExportTask(clip.id, Path(clip.info.path).name, snapshot, str(destination),
                                revision, audio, reason))
    return tasks


def validate_clip(clip):
    if not Path(clip.info.path).is_file():
        raise ValueError("源文件不存在")
    if file_fingerprint(clip.info.path) != clip.info.fingerprint:
        raise ValueError("源文件已变化，请重新导入并跟踪")
    clip.settings.validate()
    if clip.track is None:
        raise ValueError("尚未跟踪")
    if not clip.track.performance.get("complete", True):
        raise ValueError("跟踪已暂停，仍有未分析帧；请修正并继续跟踪后导出")
    clip.track.validate(clip.info.frames)
    if clip.track.performance.get("source_identity", source_identity(clip.info)) != source_identity(clip.info):
        raise ValueError("源文件身份与跟踪结果不匹配")
    if (clip.start, clip.end) != (clip.track.start, clip.track.end):
        raise ValueError("处理范围与跟踪结果不匹配")
    if not anchors_match(clip.track, clip.anchors, clip.start, clip.end):
        raise ValueError("关键帧已修改，请重新跟踪")
    from .section_workflow import sections_match
    if not sections_match(clip.track, clip.section_tracking, clip.tracking_sections, clip.start, clip.end):
        raise ValueError("独立分段已修改，请重新跟踪")
    if not shape_matches(clip.track, clip.settings):
        raise ValueError("区域形状或顶点已修改，结果已过期，请重新跟踪")
    if clip.track.tracking_mode != clip.settings.tracking_mode:
        raise ValueError("跟踪算法与结果不匹配")
    for name, default in (("perspective", False), ("smoothness", 3),
                          ("adaptive_reference", False), ("planar_region", False)):
        if clip.track.performance.get(name, default) != getattr(clip.settings, name):
            raise ValueError("跟踪参数与结果不匹配，请重新跟踪")
    if clip.track.lost_frames:
        raise ValueError(f"仍有 {len(clip.track.lost_frames)} 帧真实跟丢，请补关键帧")
    build_plan(clip.info, clip.track, clip.settings)
    if clip.settings.retime.get('enabled'):
        from .time_remap_map import output_frame_map
        output_frame_map(clip.settings.retime,clip.track.count)


def run_exports(tasks, cancel, progress=None, item=None):
    outcomes = []
    total = max(1, len(tasks))
    def report(index, task, state, value=0., detail=""):
        if item:
            item(index, state, value, detail)
        if progress:
            progress((index + value) / total, f"{index + 1}/{len(tasks)} · {task.name} · {state} · {detail}")
    for index, task in enumerate(tasks):
        state, detail, result = "等待", "", None
        started = time.perf_counter()
        try:
            if task.reason:
                state, detail = "跳过", task.reason
            elif cancel.is_set():
                state = "已取消"
            else:
                clip = pickle.loads(task.snapshot)  # internal trusted bytes only
                validate_clip(clip)  # detect changes after enqueue, before rendering
                report(index, task, "导出中")
                result = export_video(clip.info, clip.track, clip.settings, task.destination, cancel,
                                      lambda v, text: report(index, task, "导出中", v, text),
                                      overwrite=False)
                state = "完成"
                detail=result.get('audio_reason','')
        except Cancelled:
            state = "已取消"
        except Exception as error:
            state, detail = "失败", str(error)
            get_logger("batch-export").exception("批量导出失败 %s", task.name)
        outcomes.append(dict(clip_id=task.clip_id, state=state, detail=detail, result=result,
                             seconds=time.perf_counter() - started, revision=task.result_revision,
                             audio_policy=task.audio_policy, destination=task.destination))
        report(index, task, state, 1., detail)
    return outcomes
