from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from .media import Cancelled
from .logging import get_logger
from .tracking_pause import TrackingPaused


class _FrameMailbox(QObject):
    """Latest observation per clip; GUI timers coalesce rather than queue frames."""

    def __init__(self, parent, deliver):
        super().__init__(parent)
        self.deliver = deliver
        self.lock = threading.Lock()
        self.pending = {}
        self.timer = QTimer(self)
        self.timer.setInterval(125)
        self.timer.timeout.connect(self.flush)

    def put(self, payload, key=None):
        with self.lock:
            self.pending[key] = payload

    @Slot()
    def start(self):
        self.timer.start()

    @Slot()
    def flush(self):
        with self.lock:
            pending, self.pending = self.pending, {}
        for key, payload in pending.items():
            if key is None:
                self.deliver(payload)
            else:
                self.deliver(key, payload)

    @Slot()
    def finish(self):
        self.timer.stop()
        self.flush()


class Job(QThread):
    progress = Signal(float, str)
    result = Signal(object)
    failed = Signal(str)
    aborted = Signal()
    paused = Signal(object)
    tracking_frame = Signal(object)
    _preview_complete = Signal()

    def __init__(self, function, *args, parent=None, keep_partial=False, live_preview=False, **kwargs):
        super().__init__(parent)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.keep_partial = keep_partial
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.supports_pause = bool(kwargs.pop("supports_pause", False))
        self.frames = _FrameMailbox(self, self.tracking_frame.emit) if live_preview else None
        if self.frames is not None:
            self.started.connect(self.frames.start)
            self._preview_complete.connect(self.frames.finish)

    def cancel(self):
        self.cancel_event.set()

    def pause(self):
        if self.supports_pause:
            self.pause_event.set()

    def run(self):
        try:
            kwargs = dict(self.kwargs)
            if self.supports_pause:
                kwargs["pause_event"] = self.pause_event
            if self.frames is not None:
                kwargs["on_frame"] = self.frames.put
            result = self.function(*self.args, cancel=self.cancel_event,
                                   progress=self.report, **kwargs)
            self._preview_complete.emit()
            if self.cancel_event.is_set() and not self.keep_partial:
                self.aborted.emit()
            else:
                self.result.emit(result)
        except TrackingPaused as paused:
            self._preview_complete.emit()
            self.paused.emit(paused)
        except Cancelled:
            self._preview_complete.emit()
            self.aborted.emit()
        except Exception as error:
            self._preview_complete.emit()
            get_logger("worker").error("%s\n%s", error,
                                       traceback.format_exc())
            self.failed.emit(str(error))

    def report(self, value, *details):
        if len(details) == 2:
            frame, state = details
            label = {"manual": "关键帧", "tracked": "稳定跟踪", "recovered": "重新匹配", "lost": "待修正"}.get(state, state)
            text = f"第 {frame + 1} 帧 · {label}"
        else:
            text = str(details[0]) if details else "正在准备视频"
        self.progress.emit(float(value), text)


class QueueJob(QThread):
    progress = Signal(float, str)
    item_result = Signal(str, object, str)
    result = Signal(object)
    failed = Signal(str)
    frame_event = Signal(str, object)
    item_progress = Signal(str, float, str)
    _preview_complete = Signal()

    def __init__(self, clips, workers, cache_directory, parent=None, live_preview=False, pause_on_lost=False):
        super().__init__(parent)
        self.clips, self.workers, self.cache_directory = clips, workers, cache_directory
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.supports_pause = True
        self.pause_on_lost = pause_on_lost
        self.frames = _FrameMailbox(self, self.frame_event.emit) if live_preview else None
        if self.frames is not None:
            self.started.connect(self.frames.start)
            self._preview_complete.connect(self.frames.finish)

    def report_frame(self, clip_id, payload):
        if self.frames is not None:
            self.frames.put(payload, clip_id)

    def cancel(self):
        self.cancel_event.set()

    def pause(self):
        self.pause_event.set()

    def run(self):
        from .sequence import run_queue
        try:
            result = run_queue(self.clips, self.workers, self.cancel_event,
                               self.progress.emit, self.item_result.emit, self.cache_directory,
                               on_frame=self.report_frame if self.frames is not None else None,
                               on_item_progress=self.item_progress.emit,
                               pause_on_lost=self.pause_on_lost, pause_event=self.pause_event)
            self._preview_complete.emit()
            self.result.emit(result)
        except Exception as error:
            self._preview_complete.emit()
            get_logger("queue").exception("素材任务失败")
            self.failed.emit(str(error))
