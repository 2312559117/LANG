from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math

import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool = False
    frames: int = 0
    proxy: str = ""
    fingerprint: str = ""
    hdr: bool = False
    pixel_format: str = "yuv420p"
    bit_depth: int = 8
    color_primaries: str = "unknown"
    color_transfer: str = "unknown"
    color_space: str = "unknown"
    color_range: str = "unknown"
    mastering_display: str = ""
    content_light: str = ""


@dataclass
class Settings:
    mode: str = "lock"
    strength: float = 1.0
    smoothing: float = 0.35
    max_zoom: float = 1.8
    aspect: str = "original"
    resolution: int = 1920
    target_x: float = 0.5
    target_y: float = 0.5
    crf: int = 18
    tracking_mode: str = "normal"
    target_shape: str = "rectangle"
    acceleration: str = "auto"
    memory_percent: int = 80
    compute_threads: int = 8
    # 运动模型开关：开启后跟踪器解算透视形变（单应矩阵），适合三维角度变化的表面。
    perspective: bool = False
    # 搜索/匹配平滑等级 1..5：数值越高越假设运动平滑连续、抗噪；越低越允许突变。
    smoothness: int = 3
    adaptive_reference: bool = False
    planar_region: bool = False
    segmented_tracking: bool = False
    segment_frames: int = 300
    confidence_mesh: bool = False
    # Clip-wide footprint in the local 0..1 coordinates of each ROI anchor.
    polygon_points: tuple[tuple[float, float], ...] = ()
    # Version 2 is a new contract; retired time_remap fields remain inert.
    retime: dict = field(default_factory=dict)
    drift_review: str = 'off'

    @classmethod
    def from_dict(cls, values):
        # Retired project fields are inert, including malformed legacy values.
        values = dict(values)
        values.pop("time_remap", None)
        values.pop("frame_interpolation", None)
        if values.get("retime"):
            from .time_remap_model import normalize_config
            values["retime"] = normalize_config(values["retime"])
        from .regions import shape_data
        shape, points = shape_data(values.get('target_shape', 'rectangle'), values.get('polygon_points', ()))
        values.update(target_shape=shape, polygon_points=points)
        return cls(**values)

    def validate(self) -> None:
        if not isinstance(self.drift_review,str) or self.drift_review not in ('off','relaxed','standard','strict'):
            raise ValueError('未知的漂移复核灵敏度')
        from .time_remap_model import normalize_config
        normalize_config(None if type(self.retime) is dict and not self.retime else self.retime)
        from .regions import shape_data
        shape_data(self.target_shape, self.polygon_points)
        from .performance import ComputeOptions
        ComputeOptions(self.acceleration, self.memory_percent, self.compute_threads).validate()
        if self.tracking_mode not in {"normal", "precision"}:
            raise ValueError("未知的跟踪算法")
        if self.mode not in {"lock", "follow"}:
            raise ValueError("未知的稳定模式")
        if self.aspect not in {"original", "9:16", "16:9", "1:1"}:
            raise ValueError("未知的画面比例")
        for value, lo, hi in (
            (self.strength, 0, 1), (self.smoothing, 0.01, 3),
            (self.max_zoom, 1, 4), (self.target_x, 0.1, 0.9),
            (self.target_y, 0.1, 0.9), (self.crf, 12, 30),
        ):
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError("稳定参数超出允许范围")
        if self.resolution not in {0, 1080, 1920, 2560, 3840}:
            raise ValueError("未知的输出分辨率")
        if not isinstance(self.perspective, bool):
            raise ValueError("透视开关必须为布尔值")
        if not isinstance(self.adaptive_reference, bool):
            raise ValueError("自适应参考开关必须为布尔值")
        if not isinstance(self.planar_region, bool):
            raise ValueError("平面区域开关必须为布尔值")
        if not isinstance(self.segmented_tracking, bool) or not isinstance(self.confidence_mesh, bool):
            raise ValueError("分段和网格开关必须为布尔值")
        if type(self.segment_frames) is not int or not 10 <= self.segment_frames <= 3600:
            raise ValueError("分段长度应为 10–3600 帧")
        if not isinstance(self.smoothness, int) or self.smoothness not in {1, 2, 3, 4, 5}:
            raise ValueError("平滑等级应在 1–5 之间")


@dataclass
class TrackResult:
    start: int
    end: int
    positions: np.ndarray
    boxes: np.ndarray
    confidence: np.ndarray
    states: list[str]
    tracking_mode: str = "normal"
    analysis_size: tuple[int, int] | None = None
    performance: dict = field(default_factory=dict)
    anchors: dict | None = None
    homographies: np.ndarray | None = None
    confidence_grids: np.ndarray | None = None

    @property
    def count(self) -> int:
        return self.end - self.start + 1

    @property
    def lost_frames(self) -> list[int]:
        from .tracking_pause import computed_mask
        measured = computed_mask(self)
        return [self.start + i for i, state in enumerate(self.states) if state == "lost" and measured[i]]

    def validate(self, total_frames: int) -> None:
        if self.tracking_mode not in {"normal", "precision"}:
            raise ValueError("未知的跟踪算法")
        if not 0 <= self.start <= self.end < total_frames:
            raise ValueError("跟踪区间与视频不匹配")
        n = self.count
        if self.positions.shape != (n, 2) or self.boxes.shape != (n, 4):
            raise ValueError("跟踪数据尺寸不正确")
        if self.confidence.shape != (n,) or len(self.states) != n:
            raise ValueError("跟踪数据长度不正确")
        if not all(np.isfinite(a).all() for a in (self.positions, self.boxes, self.confidence)):
            raise ValueError("跟踪数据含无效数值")
        if not set(self.states) <= {"manual", "tracked", "recovered", "lost"}:
            raise ValueError("跟踪数据含未知状态")
        if np.any(self.positions < 0) or np.any(self.positions > 1):
            raise ValueError("跟踪位置超出画面")
        if np.any(self.boxes[:, 2:] <= 0) or np.any(self.boxes < 0) or np.any(self.boxes > 1):
            raise ValueError("目标框数据无效")
        if np.any(self.confidence < 0) or np.any(self.confidence > 1):
            raise ValueError("跟踪可信度数据无效")
        if self.homographies is not None:
            from .planar import UNIT_CORNERS, project
            matrices = np.asarray(self.homographies)
            if matrices.shape != (n, 3, 3) or not np.isfinite(matrices).all():
                raise ValueError("平面跟踪矩阵尺寸或数值无效")
            for matrix in matrices:
                denominators = np.column_stack([UNIT_CORNERS, np.ones(4)]) @ matrix[2]
                if (abs(np.linalg.det(matrix)) < 1e-12 or np.any(denominators <= 0)
                        or not np.isfinite(project(UNIT_CORNERS, matrix)).all()):
                    raise ValueError("平面跟踪矩阵退化或跨越无穷远")
        if self.confidence_grids is not None:
            values = np.asarray(self.confidence_grids)
            if (values.shape != (n, 4, 4) or not np.isfinite(values).all()
                    or np.any(values < -1) or np.any(values > 1)):
                raise ValueError("置信度网格无效")
        # 校验增量重跟踪依赖的 anchors 字段：帧号必须在区间内、box 合法。
        # 增量复用判定 (tracking.py:_track_video::reusable) 直接对比 anchors，
        # 若 anchors 与 track 数据不一致会导致静默复用错误区间。
        if self.anchors is not None:
            for frame, box in self.anchors.items():
                if not isinstance(frame, int) or not self.start <= frame <= self.end:
                    raise ValueError("跟踪结果中的关键帧不在跟踪区间内")
                values = list(box)
                if len(values) != 4 or not all(isinstance(v, (int, float)) for v in values):
                    raise ValueError("跟踪结果中的关键帧格式无效")
                x, y, w, h = (float(v) for v in values)
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError("跟踪结果中的关键帧含无效数值")
                if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.001 or y + h > 1.001:
                    raise ValueError("跟踪结果中的关键帧超出画面")


@dataclass
class RenderPlan:
    width: int
    height: int
    matrices: np.ndarray
    zoom: float
    required_zoom: float
    limited_frames: int


def file_fingerprint(path: str) -> str:
    stat = Path(path).stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def source_identity(info: VideoInfo) -> dict:
    return dict(path=str(Path(info.path).resolve()).casefold(), fingerprint=info.fingerprint,
                width=info.width, height=info.height, fps=info.fps, frames=info.frames)


def save_project(path: str, info: VideoInfo, settings: Settings,
                 anchors: dict[int, tuple], start: int, end: int,
                 track: TrackResult | None, *, timebar: dict | None = None,
                 section_tracking=False, tracking_sections=None,review_marks=None) -> None:
    settings.validate()
    from .section_workflow import section_configuration
    section_tracking, tracking_sections = section_configuration(section_tracking, tracking_sections, info.frames)
    if file_fingerprint(info.path) != info.fingerprint:
        raise ValueError("原视频已经被修改，请重新导入后再保存工程")
    destination = Path(path)
    if destination.resolve() == Path(info.path).resolve():
        raise ValueError("工程不能覆盖原视频")
    payload = {
        "format": "lockframe-project", "version": 1,
        "tracker_revision": 4,
        "source": info.path, "fingerprint": info.fingerprint,
        "settings": asdict(settings), "start": start, "end": end,
        "section_tracking": section_tracking, "tracking_sections": tracking_sections,
        "anchors": {str(k): list(v) for k, v in anchors.items()},
        "track": None if track is None else {
            "start": track.start, "end": track.end,
            "positions": track.positions.tolist(), "boxes": track.boxes.tolist(),
            "confidence": track.confidence.tolist(), "states": track.states,
            "tracking_mode": track.tracking_mode, "analysis_size": track.analysis_size,
            "performance": track.performance,
            "homographies": None if track.homographies is None else track.homographies.tolist(),
            "confidence_grids": None if track.confidence_grids is None else track.confidence_grids.tolist(),
            # 序列化 track.anchors 让读回时不依赖外部 self.anchors 同步（P1-1）。
            # tracker_revision>=4 的工程必带此字段；旧工程由 app 回退到顶层 anchors。
            "anchors": ({str(k): list(v) for k, v in track.anchors.items()}
                         if track.anchors is not None else None),
        },
    }
    if timebar is not None:
        from .timebar_core import clean_state
        payload['timebar'] = clean_state(timebar, info.frames, start, end)
    if review_marks is not None:
        from .review_records import clean_marks
        payload['review_marks']=clean_marks(review_marks)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)


def read_project(path: str) -> dict:
    if Path(path).stat().st_size > 100_000_000:
        raise ValueError("工程文件过大")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "lockframe-project" or payload.get("version") != 1:
        raise ValueError("不是受支持的番茄跟踪工程")
    payload.pop("time_remap", None)
    payload.pop("frame_interpolation", None)
    from .section_workflow import section_configuration
    payload["section_tracking"], payload["tracking_sections"] = section_configuration(
        payload.get("section_tracking", False), payload.get("tracking_sections", []), 2**31)
    settings = Settings.from_dict(payload["settings"])
    settings.validate()
    payload["settings"] = asdict(settings)
    if not Path(payload["source"]).is_file():
        raise ValueError("原视频已移动，请将视频放回原位置后再打开工程")
    for frame, box in payload["anchors"].items():
        if int(frame) < 0 or len(box) != 4:
            raise ValueError("工程中的关键帧无效")
        x, y, w, h = [float(v) for v in box]
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError("工程中的目标框无效")
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.001 or y + h > 1.001:
            raise ValueError("工程中的目标框超出画面")
    return payload
