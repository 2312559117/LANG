"""Compact tracking progress driven only by the existing measured phase events.

Queue phase events describe the whole batch, not the displayed clip.  Keep those
two quantities separate so a batch's 47% cannot become a clip's alleged 47%.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout

from .quality import STAGES, parse_phase


PHASE_TITLES = {
    "分析": "准备分析",
    "计算": "逐帧跟踪",
    "检查": "质量检查",
    "缓存": "写入结果",
}
STATUS_TITLES = {
    "idle": "准备跟踪", "running": "正在跟踪", "completed": "跟踪完成",
    "failed": "跟踪失败", "cancelled": "跟踪已取消",
    "paused": "已暂停 · 等待修正",
}
STAGE_STATUS_TITLES = {
    "waiting": "等待", "running": "进行中", "completed": "完成",
    "failed": "失败", "cancelled": "已取消",
    "paused": "已暂停",
}


def _fraction(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("进度必须是有限数值")
    return max(0., min(1., value))


@dataclass
class PhaseState:
    fraction: float = 0.
    status: str = "waiting"


class _ElidedLabel(QLabel):
    """Long filenames should not force the preview wider than the window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.full_text = ""
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)

    def setText(self, text):
        self.full_text = str(text)
        self.setToolTip(self.full_text)
        self._elide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        QLabel.setText(self, self.fontMetrics().elidedText(
            self.full_text, Qt.TextElideMode.ElideMiddle, max(0, self.contentsRect().width())))


class TrackingProgressPanel(QFrame):
    """An embeddable phase panel; all public methods run on the GUI thread.

    ``begin`` starts a fresh run. By default ``set_task`` retains batch aggregate
    rows; ``follow_task_phases=True`` instead clears them to show the new clip's
    actual phases. ``update_progress`` accepts the Job/QueueJob signal unchanged;
    ``update_task_progress`` consumes a separately routed per-clip phase event.
    Terminal detail folds away without changing its measured values and can be
    opened again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trackingProgressPanel")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet("""
            QFrame#trackingProgressPanel { background: #191918; border: 1px solid #32302d; border-radius: 3px; }
            QFrame#trackingProgressPanel QLabel { background: transparent; border: none; }
            QFrame#trackingProgressPanel QLabel#progressTask { color: #eee8df; font-weight: 600; }
            QFrame#trackingProgressPanel QLabel#progressNumbers { color: #e88c51; font-weight: 600; }
            QFrame#trackingProgressPanel QLabel#progressDetail, QFrame#trackingProgressPanel QLabel#progressScope { color: #aaa49b; font-size: 11px; }
            QFrame#trackingProgressPanel QProgressBar { min-height: 4px; max-height: 4px; background: #33312e; border: none; border-radius: 2px; }
            QFrame#trackingProgressPanel QProgressBar::chunk { background: #e88c51; border-radius: 2px; }
            QFrame#trackingProgressPanel QPushButton#progressExpand { background: transparent; color: #e88c51; border: 1px solid #3d3934; padding: 2px 7px; min-height: 18px; max-height: 18px; font-size: 11px; }
            QFrame#trackingProgressPanel QPushButton#progressExpand:hover { background: #30271f; border-color: #e88c51; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 5, 12, 5)
        layout.setSpacing(3)
        heading = QHBoxLayout()
        self.task_label = _ElidedLabel()
        self.task_label.setObjectName("progressTask")
        heading.addWidget(self.task_label, 1)
        self.progress_label = QLabel()
        self.progress_label.setObjectName("progressNumbers")
        self.expand_button = QPushButton("收起阶段")
        self.expand_button.setObjectName("progressExpand")
        self.expand_button.setCheckable(True)
        self.expand_button.setAccessibleName("展开或收起跟踪阶段明细")
        self.expand_button.clicked.connect(self.set_expanded)
        heading.addWidget(self.expand_button)
        layout.addLayout(heading)
        layout.addWidget(self.progress_label)
        self.overall_bar = self._bar()
        layout.addWidget(self.overall_bar)
        self.details_widget = QFrame()
        self.details_widget.setFrameShape(QFrame.Shape.NoFrame)
        details_layout = QVBoxLayout(self.details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(5)
        self.scope_label = QLabel()
        self.scope_label.setObjectName("progressScope")
        details_layout.addWidget(self.scope_label)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        self.stage_bars, self.stage_labels = {}, {}
        for index, name in enumerate(STAGES):
            label = QLabel()
            label.setTextFormat(Qt.TextFormat.PlainText)
            bar = self._bar()
            grid.addWidget(label, index, 0)
            grid.addWidget(bar, index, 1)
            self.stage_labels[name], self.stage_bars[name] = label, bar
        grid.setColumnStretch(1, 1)
        details_layout.addLayout(grid)
        self.detail_label = _ElidedLabel()
        self.detail_label.setObjectName("progressDetail")
        details_layout.addWidget(self.detail_label)
        layout.addWidget(self.details_widget)
        self.reset()

    @staticmethod
    def _bar():
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(0)
        bar.setTextVisible(False)
        return bar

    def reset(self, hide=True):
        self.state = "idle"
        self.filename, self.task_index, self.task_total = "", 1, 1
        self.batch = False
        self.follow_task_phases = False
        self.overall_progress, self.task_progress = 0., None
        self.phases = {stage: PhaseState() for stage in STAGES}
        self.detail = "等待跟踪任务"
        self.set_expanded(not getattr(self,"compact_default",False))
        self._render()
        self.setVisible(not hide)

    def begin(self, filename="", index=1, total=1, batch=False, follow_task_phases=False):
        self.reset(hide=False)
        self.batch = bool(batch)
        self.follow_task_phases = bool(follow_task_phases)
        self.state = "running"
        self.detail = "等待分析开始"
        self.set_task(filename, index, total)

    def set_task(self, filename, index=1, total=1, reset=True):
        total, index = max(1, int(total)), int(index)
        self.filename = str(filename)
        self.task_total, self.task_index = total, max(1, min(total, index))
        if reset:
            self.task_progress = None if self.batch else 0.
            if not self.batch or self.follow_task_phases:
                self.phases = {stage: PhaseState() for stage in STAGES}
            if self.follow_task_phases:
                self.detail = "等待当前素材的阶段进度"
            if not self.batch:
                self.overall_progress = 0.
        self._render()

    def set_task_progress(self, value):
        if self.state != "running":
            return
        self.task_progress = _fraction(value) if value is not None else None
        self._render()

    def update_task_progress(self, value, message):
        """Update the followed clip with its own genuine weighted phase event.

        The caller routes events by clip identity. This path deliberately does
        not update batch overall progress, which arrives through the queue.
        """
        if self.state != "running":
            return
        self.task_progress = _fraction(value)
        text = str(message)
        phase = parse_phase(text)
        if phase:
            self.set_phase(*phase[:2])
            self.detail = phase[2]
        elif text and not text.startswith("[汇总:"):
            self.detail = text
        self._render()

    def set_expanded(self, expanded):
        """Change only presentation; terminal status and measured data survive."""
        self.expanded = bool(expanded)
        self.details_widget.setVisible(self.expanded)
        self.expand_button.setChecked(self.expanded)
        self.expand_button.setText("收起阶段" if self.expanded else "展开阶段")
        self.updateGeometry()

    def set_phase(self, stage, fraction, *, aggregate=False):
        if stage not in STAGES:
            raise ValueError("未知跟踪阶段：" + str(stage))
        if self.state != "running":
            return
        fraction = _fraction(fraction)
        if aggregate:
            phase = self.phases[stage]
            phase.fraction = fraction
            phase.status = "completed" if fraction >= 1. else "running" if fraction > 0. else "waiting"
        else:
            current = STAGES.index(stage)
            for earlier in STAGES[:current]:
                self.phases[earlier] = PhaseState(1., "completed")
            self.phases[stage] = PhaseState(fraction, "completed" if fraction >= 1. else "running")
        self._render()

    def update_progress(self, value, message):
        if self.state != "running":
            return
        self.overall_progress = _fraction(value)
        if not self.batch:
            self.task_progress = self.overall_progress
        text = str(message)
        if text.startswith("[汇总:") and text.endswith("]"):
            if self.follow_task_phases:
                self._render()
                return
            try:
                _, stage, fraction = text[1:-1].split(":")
                self.set_phase(stage, _fraction(fraction), aggregate=True)
            except (ValueError, TypeError):
                pass  # malformed status text must not turn into a fabricated phase
        else:
            if self.batch and self.follow_task_phases:
                # Queue-wide summary text must not obscure the followed clip's
                # phase detail. Per-clip events use update_task_progress.
                self._render()
                return
            phase = parse_phase(text)
            if phase:
                self.set_phase(*phase[:2])
                self.detail = phase[2]
            elif text:
                self.detail = text
        self._render()

    def finish(self, status="completed", detail=""):
        if status not in {"completed", "failed", "cancelled", "paused"}:
            raise ValueError("未知任务结束状态：" + str(status))
        self.state = status
        if status == "completed" and not self.batch:
            self.overall_progress = self.task_progress = 1.
            self.phases = {stage: PhaseState(1., "completed") for stage in STAGES}
        elif status != "completed":
            # Unstarted stages stay waiting; a failure never claims they ran.
            for phase in self.phases.values():
                if phase.status == "running":
                    phase.status = status
        self.detail = str(detail) or {
            "completed": "任务已结束，请复核跟踪结果",
            "failed": "任务失败，已完成结果保留",
            "cancelled": "任务已取消，已完成结果保留",
            "paused": "请修正暂停帧，再点击继续跟踪",
        }[status]
        self._render()
        self.set_expanded(False)
        self.show()

    def _render(self):
        title = STATUS_TITLES[self.state]
        if self.filename:
            title += f" {self.task_index}/{self.task_total}：{self.filename}"
        self.task_label.setText(title)
        self.overall_bar.setValue(round(self.overall_progress * 1000))
        if self.batch:
            current = "—" if self.task_progress is None else f"{self.task_progress:.0%}"
            self.progress_label.setText(f"当前 {current} · 总体 {self.overall_progress:.0%}")
        else:
            self.progress_label.setText(f"{self.overall_progress:.0%}")
        if self.batch and self.follow_task_phases:
            scope = "当前素材阶段 · 总体进度单独统计"
        else:
            scope = "本批阶段汇总 · 当前与总体进度分别计算" if self.batch else "当前素材 · 实际计算阶段"
        self.scope_label.setText(scope)
        for stage, phase in self.phases.items():
            suffix = STAGE_STATUS_TITLES[phase.status]
            if phase.status == "running":
                suffix = f"{phase.fraction:.0%}"
            elif phase.status in {"failed", "cancelled"}:
                suffix = f"{suffix} · {phase.fraction:.0%}"
            self.stage_labels[stage].setText(f"{PHASE_TITLES[stage]}  ·  {suffix}")
            self.stage_bars[stage].setValue(round(phase.fraction * 1000))
            self.stage_labels[stage].setToolTip(f"算法阶段：{stage}；{STAGE_STATUS_TITLES[phase.status]}；{phase.fraction:.1%}")
        self.detail_label.setText(self.detail)
