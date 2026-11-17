"""Select a frame-accurate video excerpt without modifying the source video."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .media import FrameReader
from .models import VideoInfo
from .widgets import qimage


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _nearest_frame(seconds: float, fps: float) -> int:
    # Avoid Python's even-number rounding at exact half-frame boundaries.
    return int(math.floor(max(0.0, seconds) * fps + 0.5))


def _timestamp(seconds: float) -> str:
    milliseconds = max(0, int(math.floor(seconds * 1000 + 0.5)))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    prefix = f"{hours:02d}:" if hours else ""
    return f"{prefix}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class TrimTimeline(QWidget):
    """A two-handle range with an independently scrubbable playhead.

    The handles are frame *boundaries*: the right handle is ``end + 1``.
    This leaves a visible, selectable range even for a single-frame excerpt.
    Dragging the bottom rail always moves the playhead, including near handles.
    """

    rangeChanged = Signal(int, int)
    seekRequested = Signal(int)

    HANDLE_Y = 25
    RAIL_Y = 62
    MARGIN = 15

    def __init__(self, frames: int, parent=None):
        super().__init__(parent)
        self.frames = max(1, int(frames))
        self.start_frame = 0
        self.end_frame = self.frames - 1
        self.current_frame = 0
        self._dragging = None
        self._active = "current"
        self.setObjectName("trimTimeline")
        self.setFixedHeight(83)
        self.setMinimumWidth(250)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("片段范围与播放头")
        self.setToolTip("上方两端手柄选择入点和出点；下方轨道拖动播放头。方向键可逐帧调整。")

    def _width(self) -> float:
        return max(1.0, self.width() - 2 * self.MARGIN)

    def boundary_x(self, boundary: int) -> float:
        return self.MARGIN + _clamp(boundary, 0, self.frames) / self.frames * self._width()

    def frame_x(self, frame: int) -> float:
        return self.MARGIN + (_clamp(frame, 0, self.frames - 1) + 0.5) / self.frames * self._width()

    def _boundary_at(self, x: float) -> int:
        return _clamp(math.floor((x - self.MARGIN) / self._width() * self.frames + 0.5),
                      0, self.frames)

    def _frame_at(self, x: float) -> int:
        return _clamp(math.floor((x - self.MARGIN) / self._width() * self.frames),
                      0, self.frames - 1)

    def set_values(self, start: int, end: int, current: int | None = None) -> None:
        self.start_frame = _clamp(start, 0, self.frames - 1)
        self.end_frame = _clamp(end, self.start_frame, self.frames - 1)
        if current is not None:
            self.current_frame = _clamp(current, 0, self.frames - 1)
        self.update()

    def set_position(self, frame: int) -> None:
        self.current_frame = _clamp(frame, 0, self.frames - 1)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left = self.boundary_x(0)
        right = self.boundary_x(self.frames)
        start = self.boundary_x(self.start_frame)
        end = self.boundary_x(self.end_frame + 1)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#3b362e"))
        painter.drawRoundedRect(QRectF(left, 14, right - left, 27), 4, 4)
        painter.setBrush(QColor("#684639"))
        painter.drawRect(QRectF(start, 14, max(1, end - start), 27))
        painter.setPen(QColor("#ebad96"))
        painter.drawLine(QPointF(start, 14), QPointF(end, 14))
        painter.drawLine(QPointF(start, 41), QPointF(end, 41))
        for x in (start, end):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e9c46a"))
            painter.drawRoundedRect(QRectF(x - 5, 9, 10, 38), 3, 3)
            painter.setPen(QPen(QColor("#4d3429"), 1))
            painter.drawLine(QPointF(x - 1, 21), QPointF(x - 1, 34))
            painter.drawLine(QPointF(x + 2, 21), QPointF(x + 2, 34))
        painter.setPen(QPen(QColor("#534a3e"), 3, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(left, self.RAIL_Y), QPointF(right, self.RAIL_Y))
        current = self.frame_x(self.current_frame)
        painter.setPen(QPen(QColor("#e8ddd0"), 1))
        painter.drawLine(QPointF(current, 48), QPointF(current, 75))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e8f0fa"))
        painter.drawEllipse(QPointF(current, self.RAIL_Y), 5, 5)
        if self.hasFocus():
            painter.setPen(QPen(QColor("#6b9dcc"), 1, Qt.PenStyle.DotLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 5, 5)

    def _move_to(self, x: float) -> None:
        if self._dragging == "start":
            start = _clamp(self._boundary_at(x), 0, self.end_frame)
            if start != self.start_frame:
                self.start_frame = start
                self.rangeChanged.emit(self.start_frame, self.end_frame)
            self.seekRequested.emit(self.start_frame)
        elif self._dragging == "end":
            end = _clamp(self._boundary_at(x) - 1, self.start_frame, self.frames - 1)
            if end != self.end_frame:
                self.end_frame = end
                self.rangeChanged.emit(self.start_frame, self.end_frame)
            self.seekRequested.emit(self.end_frame)
        else:
            self.seekRequested.emit(self._frame_at(x))
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        start = self.boundary_x(self.start_frame)
        end = self.boundary_x(self.end_frame + 1)
        distances = (abs(event.position().x() - start), abs(event.position().x() - end))
        if event.position().y() < 49 and min(distances) <= 13:
            self._dragging = "start" if distances[0] <= distances[1] else "end"
        else:
            self._dragging = "current"
        self._active = self._dragging
        self._move_to(event.position().x())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._move_to(event.position().x())
        elif event.position().y() < 49 and min(
            abs(event.position().x() - self.boundary_x(self.start_frame)),
            abs(event.position().x() - self.boundary_x(self.end_frame + 1)),
        ) <= 13:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._move_to(event.position().x())
            self._dragging = None

    def keyPressEvent(self, event):
        key = event.key()
        if key not in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End}:
            super().keyPressEvent(event)
            return
        value = {"start": self.start_frame, "end": self.end_frame,
                 "current": self.current_frame}[self._active]
        value = (0 if key == Qt.Key.Key_Home else self.frames - 1 if key == Qt.Key.Key_End
                 else value + (-1 if key == Qt.Key.Key_Left else 1))
        if self._active == "start":
            self.start_frame = _clamp(value, 0, self.end_frame)
            self.rangeChanged.emit(self.start_frame, self.end_frame)
            value = self.start_frame
        elif self._active == "end":
            self.end_frame = _clamp(value, self.start_frame, self.frames - 1)
            self.rangeChanged.emit(self.start_frame, self.end_frame)
            value = self.end_frame
        self.seekRequested.emit(_clamp(value, 0, self.frames - 1))
        self.update()


class _TrimPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.message = "正在读取画面…"
        self.setMinimumSize(280, 130)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, frame):
        self.image = qimage(frame)
        self.message = ""
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17191d"))
        if self.image is None:
            painter.setPen(QColor("#a9b0bb"))
            painter.drawText(self.rect().adjusted(20, 12, -20, -12),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.message)
            return
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        painter.drawImage(QRectF((self.width() - width) / 2, (self.height() - height) / 2,
                                width, height), self.image)


class TrimDialog(QDialog):
    """Edit an inclusive frame interval; read the result only after Accepted.

    ``VideoInfo`` and source files are never mutated. Preview reads ``info.proxy``
    silently through its own FrameReader; ``finished`` releases it and the timer.
    ``end_input`` is the exclusive end time, so duration is always frame-exact.
    """

    def __init__(self, info: VideoInfo, start: int, end: int, current: int = 0, parent=None):
        super().__init__(parent)
        if not math.isfinite(info.fps) or info.fps <= 0 or info.frames < 1:
            raise ValueError("视频帧率或帧数无效")
        self.info = info
        self.start_frame = _clamp(start, 0, info.frames - 1)
        self.end_frame = _clamp(end, self.start_frame, info.frames - 1)
        self.current_frame = _clamp(current, 0, info.frames - 1)
        self.reader = None
        self._disposed = False
        self._syncing = False
        self._play_started = 0.0
        self._play_first = 0
        self.setWindowTitle("截取视频片段 · 番茄跟踪")
        self.setModal(True)
        self.setMinimumSize(660, 530)
        screen = self.screen()
        available = screen.availableGeometry() if screen else None
        self.resize(min(980, max(660, available.width() - 60)) if available else 980,
                    min(720, max(530, available.height() - 80)) if available else 720)
        from .theme import STYLE
        self.setStyleSheet(STYLE)
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._play_tick)
        self._build_ui()
        self.finished.connect(self._release_resources)
        self._sync_fields()
        try:
            if info.proxy:
                self.reader = FrameReader(info.proxy)
        except Exception:
            self.reader = None
        self.play_button.setEnabled(self.reader is not None)
        self.seek(self.current_frame)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)
        title = QLabel("先留下想要的镜头")
        title.setObjectName("trimTitle")
        layout.addWidget(title)
        description = QLabel("拖动两端选择保留的片段，再进行目标跟踪与增稳。")
        description.setObjectName("trimMuted")
        layout.addWidget(description)
        self.preview = _TrimPreview()
        layout.addWidget(self.preview, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("播放所选片段")
        self.play_button.setObjectName("trimPlayButton")
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        self.loop_check = QCheckBox("循环播放")
        self.loop_check.setObjectName("trimLoopCheck")
        self.loop_check.setChecked(True)
        controls.addWidget(self.loop_check)
        controls.addStretch()
        self.position_label = QLabel()
        self.position_label.setObjectName("trimCurrentTimeLabel")
        controls.addWidget(self.position_label)
        layout.addLayout(controls)

        self.timeline = TrimTimeline(self.info.frames)
        self.timeline.rangeChanged.connect(self._range_changed)
        self.timeline.seekRequested.connect(self._scrub)
        layout.addWidget(self.timeline)

        timing = QHBoxLayout()
        self.start_input = self._time_input("trimStartTime")
        self.end_input = self._time_input("trimEndTime")
        for caption, widget in (("开始时间", self.start_input), ("结束时间", self.end_input)):
            column = QVBoxLayout()
            column.setSpacing(4)
            column.addWidget(QLabel(caption))
            column.addWidget(widget)
            timing.addLayout(column, 1)
        self.start_input.valueChanged.connect(self._start_time_changed)
        self.end_input.valueChanged.connect(self._end_time_changed)
        duration_column = QVBoxLayout()
        duration_column.setSpacing(4)
        duration_column.addWidget(QLabel("保留时长"))
        self.duration_label = QLabel()
        self.duration_label.setObjectName("trimDuration")
        duration_column.addWidget(self.duration_label)
        timing.addSpacing(16)
        timing.addLayout(duration_column, 1)
        layout.addLayout(timing)

        actions = QHBoxLayout()
        self.set_start_button = QPushButton("当前画面设为开始")
        self.set_start_button.setObjectName("trimSetStartButton")
        self.set_start_button.clicked.connect(self._set_start_here)
        actions.addWidget(self.set_start_button)
        self.set_end_button = QPushButton("当前画面设为结束")
        self.set_end_button.setObjectName("trimSetEndButton")
        self.set_end_button.clicked.connect(self._set_end_here)
        actions.addWidget(self.set_end_button)
        actions.addStretch()
        self.reset_button = QPushButton("恢复完整视频")
        self.reset_button.setObjectName("trimResetButton")
        self.reset_button.clicked.connect(self._reset)
        actions.addWidget(self.reset_button)
        layout.addLayout(actions)

        self.status_label = QLabel("下方细轨道可拖动查看画面 · 方向键逐帧调整 · 静音预览")
        self.status_label.setObjectName("trimMuted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        footer = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel |
                                  QDialogButtonBox.StandardButton.Ok)
        self.apply_button = footer.button(QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText("应用片段")
        self.apply_button.setObjectName("trimApplyButton")
        self.apply_button.setDefault(True)
        self.cancel_button = footer.button(QDialogButtonBox.StandardButton.Cancel)
        self.cancel_button.setText("取消")
        self.cancel_button.setObjectName("trimCancelButton")
        footer.accepted.connect(self.accept)
        footer.rejected.connect(self.reject)
        layout.addWidget(footer)

    def _time_input(self, name):
        control = QDoubleSpinBox()
        control.setObjectName(name)
        control.setDecimals(3)
        control.setRange(0.0, self.info.frames / self.info.fps)
        control.setSingleStep(1.0 / self.info.fps)
        control.setSuffix(" 秒")
        control.setKeyboardTracking(False)
        control.setAccelerated(True)
        return control

    def _sync_fields(self):
        self._syncing = True
        try:
            self.start_input.setRange(0.0, self.end_frame / self.info.fps)
            self.end_input.setRange((self.start_frame + 1) / self.info.fps,
                                    self.info.frames / self.info.fps)
            self.start_input.setValue(self.start_frame / self.info.fps)
            self.end_input.setValue((self.end_frame + 1) / self.info.fps)
            self.timeline.set_values(self.start_frame, self.end_frame, self.current_frame)
            duration = (self.end_frame - self.start_frame + 1) / self.info.fps
            self.duration_label.setText(f"{_timestamp(duration)}  ·  {self.end_frame - self.start_frame + 1} 帧")
            self.duration_label.setToolTip(f"精确时长：{duration:.6f} 秒")
        finally:
            self._syncing = False

    def _range_changed(self, start: int, end: int):
        self.stop()
        self.start_frame = _clamp(start, 0, self.info.frames - 1)
        self.end_frame = _clamp(end, self.start_frame, self.info.frames - 1)
        self._sync_fields()

    def _start_time_changed(self, seconds: float):
        if self._syncing:
            return
        self._range_changed(_clamp(_nearest_frame(seconds, self.info.fps), 0, self.end_frame),
                            self.end_frame)
        self.seek(self.start_frame)

    def _end_time_changed(self, seconds: float):
        if self._syncing:
            return
        self._range_changed(self.start_frame,
                            _clamp(_nearest_frame(seconds, self.info.fps) - 1,
                                   self.start_frame, self.info.frames - 1))
        self.seek(self.end_frame)

    def _set_start_here(self):
        # Setting a point outside the previous range expands the other boundary
        # by a frame instead of silently selecting a different visible frame.
        self._range_changed(self.current_frame, max(self.current_frame, self.end_frame))

    def _set_end_here(self):
        self._range_changed(min(self.start_frame, self.current_frame), self.current_frame)

    def _reset(self):
        self._range_changed(0, self.info.frames - 1)

    def _scrub(self, frame: int):
        self.stop()
        self.seek(frame)

    def seek(self, frame: int):
        if self._disposed:
            return False
        self.current_frame = _clamp(frame, 0, self.info.frames - 1)
        self.timeline.set_position(self.current_frame)
        self.position_label.setText(
            f"{_timestamp(self.current_frame / self.info.fps)} / "
            f"{_timestamp(self.info.frames / self.info.fps)}  ·  "
            f"第 {self.current_frame + 1} 帧")
        if self.reader is None:
            self.preview.message = "暂时无法预览画面，仍可通过时间选择片段。"
            self.preview.update()
            return False
        try:
            self.preview.set_frame(self.reader.read(self.current_frame))
            self.play_button.setEnabled(True)
            self.status_label.setText("下方细轨道可拖动查看画面 · 方向键逐帧调整 · 静音预览")
            return True
        except Exception:
            self.stop()
            self.preview.image = None
            self.preview.message = "这帧画面暂时无法读取，请试试相邻画面。"
            self.preview.update()
            self.status_label.setText("这帧预览读取失败，仍可调整片段或查看相邻画面。")
            return False

    def _toggle_play(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.play()

    def play(self):
        if self._disposed or self.reader is None:
            return
        if not self.start_frame <= self.current_frame < self.end_frame:
            if not self.seek(self.start_frame):
                return
        self._play_first = self.current_frame
        self._play_started = time.monotonic()
        self.timer.start()
        self.play_button.setText("暂停")

    def stop(self):
        self.timer.stop()
        self.play_button.setText("播放所选片段")

    def _play_tick(self):
        if self._disposed or not self.timer.isActive():
            return
        frame = self._play_first + int((time.monotonic() - self._play_started) * self.info.fps)
        if frame > self.end_frame:
            if self.loop_check.isChecked():
                frame = self.start_frame + (frame - self.start_frame) % (
                    self.end_frame - self.start_frame + 1)
            else:
                self.seek(self.end_frame)
                self.stop()
                return
        if frame != self.current_frame:
            self.seek(frame)

    def accept(self):
        # Commit any typed value before returning the inclusive frame interval.
        self.start_input.interpretText()
        self.end_input.interpretText()
        super().accept()

    def _release_resources(self, result=None):
        if self._disposed:
            return
        self.stop()
        self._disposed = True
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def closeEvent(self, event):
        self._release_resources()
        super().closeEvent(event)
