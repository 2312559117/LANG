"""Bounded, cancellable original-resolution inspection on the analysis CFR clock."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import subprocess
import threading
import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .media import preview_tonemap_filter, Cancelled, NO_WINDOW, check_cancel, find_tool
from .models import file_fingerprint


def read_original_frames(info, indices, cancel):
    indices = tuple(sorted(set(indices)))
    if not indices or len(indices) > 3 or any(i < 0 or i >= info.frames for i in indices):
        raise ValueError("高清预览帧范围无效")
    check_cancel(cancel)
    if file_fingerprint(info.path) != info.fingerprint:
        raise ValueError("原视频已变化，请重新导入")
    # Match import/export's fps filter exactly, including VFR conversion. Input
    # -ss/native CAP_PROP_POS_FRAMES can select a different source instant.
    select = '+'.join(f'eq(n\\,{i})' for i in indices)
    vf = (preview_tonemap_filter(info) + f'fps={info.fps:.10f},select={select},'
          f'scale={info.width}:{info.height},setsar=1')
    command = [find_tool('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-i', info.path, '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', vf,
               '-frames:v', str(len(indices)), '-threads', '2', '-fps_mode', 'passthrough',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    deadline = time.monotonic() + 45
    try:
        while True:
            check_cancel(cancel)
            if time.monotonic() > deadline:
                raise TimeoutError("原片高清读取超时，当前保留流畅预览")
            try:
                pixels, errors = process.communicate(timeout=.08)
                break
            except subprocess.TimeoutExpired:
                pass
        check_cancel(cancel)
        if process.returncode:
            raise ValueError('无法读取原片高清帧：' + errors.decode('utf-8', 'replace')[-500:])
        stride = info.width * info.height * 3
        if len(pixels) != stride * len(indices):
            raise ValueError("原片高清帧数量不匹配，当前保留流畅预览")
        if file_fingerprint(info.path) != info.fingerprint:
            raise ValueError("原视频已变化，请重新导入")
        return {i: np.frombuffer(pixels[n * stride:(n + 1) * stride], np.uint8)
                .reshape(info.height, info.width, 3).copy() for n, i in enumerate(indices)}
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


class OriginalPreview(QObject):
    ready = Signal(object)

    def __init__(self, parent=None, *, decoder=read_original_frames, budget=96 * 1024 * 1024):
        super().__init__(parent)
        self.decoder, self.budget = decoder, budget
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tomato-original')
        self._future = self._pending = self._running = self._wanted = None
        self._failed = None
        self.error = ''
        self._cache = OrderedDict()
        self._cancel = threading.Event()
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._poll)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._start)

    @staticmethod
    def identity(info):
        return (info.path, info.fingerprint, info.width, info.height, info.fps, info.frames)

    def request(self, info, indices):
        token = (self.identity(info), tuple(sorted(set(indices))))
        if self._closed:
            return None
        self._wanted = token
        if all((token[0], i) in self._cache for i in token[1]):
            frames = {i: self._cache[(token[0], i)] for i in token[1]}
            for i in token[1]:
                self._cache.move_to_end((token[0], i))
            self._pending = None
            self._debounce.stop()
            if self._running != token:
                self._cancel.set()
            self.error = ''
            return frames
        if token == self._failed or (token == self._running and not self._cancel.is_set()) or (self._pending and self._pending[2] == token):
            return None
        self.error = ''
        self._cancel.set()
        self._pending = (replace(info), token[1], token)
        self._debounce.start()
        return None

    def _start(self):
        if self._closed or self._future is not None or self._pending is None:
            return
        info, indices, self._running = self._pending
        self._pending = None
        self._cancel = threading.Event()
        self._future = self._executor.submit(self.decoder, info, indices, self._cancel)
        self._timer.start()

    def _poll(self):
        if self._future is None or not self._future.done():
            return
        future, token = self._future, self._running
        self._future = self._running = None
        try:
            frames = future.result()
            if not self._closed and token == self._wanted:
                for i, frame in frames.items():
                    self._cache[(token[0], i)] = frame
                while len(self._cache) > 3 or sum(frame.nbytes for frame in self._cache.values()) > self.budget:
                    self._cache.popitem(last=False)
                # One unusually large frame may exceed the cache budget. Keep
                # its current response only; do not repeatedly decode on paint.
                self._response = (token, frames)
                self._failed = None
                self.ready.emit(token)
        except Cancelled:
            pass
        except Exception as error:
            if token == self._wanted and not self._closed:
                self._failed, self.error = token, str(error)
                from .logging import get_logger
                get_logger('preview').exception('原片高清读取失败，保留代理预览')
                self.ready.emit(token)
        self._timer.stop()
        if self._pending is not None and not self._debounce.isActive():
            self._start()

    def frames(self, info, indices):
        token = (self.identity(info), tuple(sorted(set(indices))))
        response = getattr(self, '_response', None)
        if response is not None and response[0] == token:
            self._wanted = token
            return response[1]
        self._response = None
        return self.request(info, indices)

    def clear(self):
        self._wanted = self._pending = self._failed = self._response = None
        self.error = ''
        self._cache.clear()
        self._cancel.set()
        self._debounce.stop()

    def close(self):
        self._closed = True
        self.clear()
        self._timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
