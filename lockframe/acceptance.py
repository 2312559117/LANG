"""Evidence worksheet: measured facts are separate from manual acceptance."""
from dataclasses import asdict
from datetime import datetime

from . import __version__
from .performance import memory_status
from .stabilization import build_plan


def acceptance_record(clip):
    track = clip.track
    plan = build_plan(clip.info, track, clip.settings) if track is not None else None
    lost_intervals = []
    if track is not None:
        for frame in track.lost_frames:
            if lost_intervals and frame == lost_intervals[-1][1] + 1:
                lost_intervals[-1][1] = frame
            else:
                lost_intervals.append([frame, frame])
    return {
        "version": __version__, "created_local": datetime.now().isoformat(timespec="seconds"),
        "status": "待实拍验收", "source_kind": "待人工确认：原始实拍 / 合成 / 软件录屏",
        "source": asdict(clip.info), "settings": asdict(clip.settings),
        "range_frames_zero_based_inclusive": [clip.start, clip.end],
        "anchors": {str(k): list(v) for k, v in clip.anchors.items()},
        "measured": {
            "lost_intervals_zero_based_inclusive": lost_intervals,
            "tracking_performance": track.performance if track is not None else None,
            "actual_zoom": plan.zoom if plan else None,
            "zoom_limit": clip.settings.max_zoom,
            "composition_limited_frames": plan.limited_frames if plan else None,
            "system_memory_at_record_time": memory_status(),
        },
        "manual_review": {
            "confirmed_original_footage_and_authorized_use": None,
            "suspicious_drift_intervals_and_pixels": [], "manual_correction_count": None,
            "lock_effect_and_subject_detail": None, "black_borders_intervals": [],
            "zoom_limit_visual_effect": None, "export_seconds": None,
            "export_frames_size_duration_audio": None,
            "process_peak_memory_mb": None, "cache_disk_mb": None,
            "reviewer": None, "result": "未验收",
        },
        "notes": "录屏不能代替原始汽车素材；内存快照不是峰值。空值表示尚未测量，不能视为通过。",
    }
