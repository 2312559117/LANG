"""Bounded, cancellable source-media overviews for the source-time ruler.

Only RGB arrays and audio peaks cross the worker boundary.  Qt images/pixmaps
are deliberately constructed by the widget on the GUI thread.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import subprocess
import threading
import zipfile

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from .media import Cancelled, NO_WINDOW, cache_root, find_tool


@dataclass(frozen=True)
class MediaSource:
    identity: str
    path: str
    video_path: str
    fps: float
    frames: int
    has_audio: bool

    @classmethod
    def from_info(cls, info):
        if info is None:
            return None
        path = Path(info.path).resolve()
        stat = path.stat()
        proxy = Path(info.proxy) if info.proxy else path
        if not proxy.is_file():
            proxy = path
        proxy = proxy.resolve()
        pstat = proxy.stat()
        fps = float(info.fps)
        frames = int(info.frames or round(info.duration * fps))
        if not np.isfinite(fps) or fps <= 0 or frames < 1:
            raise ValueError('素材帧率或帧数无效')
        stamp = (str(path), stat.st_size, stat.st_mtime_ns,
                 str(proxy), pstat.st_size, pstat.st_mtime_ns,
                 info.fingerprint, fps, frames, bool(info.has_audio))
        identity = hashlib.sha256(json.dumps(stamp, ensure_ascii=False).encode()).hexdigest()
        return cls(identity, str(path), str(proxy), fps, frames, bool(info.has_audio))


@dataclass(frozen=True)
class OverviewRequest:
    source: MediaSource
    start: int
    end: int  # exclusive source frame
    film_frames: tuple[int, ...]
    bins: int
    tile_width: int = 320
    tile_height: int = 120
    include_audio: bool = True

    @property
    def key(self):
        raw = ('original-film-v3', self.source.identity, self.start, self.end, self.film_frames, self.bins, self.tile_width, self.tile_height, self.include_audio)
        return hashlib.sha256(repr(raw).encode()).hexdigest()


@dataclass
class MediaOverview:
    key: str
    identity: str
    start: int
    end: int
    frames: tuple[int, ...]
    images: tuple[np.ndarray, ...]
    peaks: np.ndarray
    has_audio: bool
    video_error: str = ''
    audio_error: str = ''


class WorkCancel:
    def __init__(self):
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._process = None

    def check(self):
        if self.event.is_set():
            raise Cancelled()

    def register(self, process):
        with self._lock:
            self._process = process
            if self.event.is_set() and process.poll() is None:
                process.terminate()

    def cancel(self):
        self.event.set()
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                try:
                    self._process.terminate()
                except OSError:
                    pass


def make_request(source, start, end, count=10, bins=1400, pixel_size=(320,120), *, include_audio=True):
    start = max(0, min(source.frames - 1, int(start)))
    end = max(start + 1, min(source.frames, int(end)))
    count = max(1, min(16, int(count), end - start))
    frames = tuple(int(f) for f in np.linspace(start, end - 1, count).round())
    return OverviewRequest(source, start, end, frames, max(16, min(4096, int(bins))),
                           max(16,min(768,int(pixel_size[0]))),max(16,min(384,int(pixel_size[1]))),bool(include_audio))


def read_film(request, cancel):
    """Crop full-resolution originals before reducing to physical tile pixels."""
    cancel.check()
    capture = cv2.VideoCapture(request.source.path)
    if not capture.isOpened():
        capture.release()
        raise ValueError('无法读取视频胶片')
    images = []
    try:
        for frame in request.film_frames:
            cancel.check()
            capture.set(cv2.CAP_PROP_POS_MSEC, frame / request.source.fps * 1000)
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError('无法读取视频胶片的源帧')
            height, width = image.shape[:2]
            ratio=request.tile_width/request.tile_height
            if width/height>ratio:
                crop=max(1,round(height*ratio));left=(width-crop)//2;image=image[:,left:left+crop]
            else:
                crop=max(1,round(width/ratio));top=(height-crop)//2;image=image[top:top+crop,:]
            height,width=image.shape[:2]
            scale=min(1.,request.tile_width/width,request.tile_height/height)
            image=cv2.resize(image,(max(1,round(width*scale)),max(1,round(height*scale))),interpolation=cv2.INTER_AREA)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return tuple(images)
    finally:
        capture.release()


def read_peaks(request, cancel, sample_rate=2000):
    """Decode mono audio incrementally; memory does not grow with clip length."""
    if not request.source.has_audio or not request.include_audio:
        return np.empty(0, np.float32)
    cancel.check()
    duration = (request.end - request.start) / request.source.fps
    samples = max(1, round(duration * sample_rate))
    peaks = np.zeros(request.bins, np.float32)
    command = [find_tool('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-threads', '2', '-ss', f'{request.start / request.source.fps:.9f}', '-i', request.source.path,
               '-t', f'{duration:.9f}', '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(sample_rate),
               '-f', 'f32le', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               creationflags=NO_WINDOW)
    cancel.register(process)
    offset = 0
    remainder = b''
    try:
        while True:
            cancel.check()
            chunk = process.stdout.read(8192)
            if not chunk:
                break
            chunk = remainder + chunk
            count = len(chunk) // 4 * 4
            remainder = chunk[count:]
            data = np.frombuffer(chunk[:count], dtype='<f4')
            indices = np.minimum(request.bins - 1,
                                 (np.arange(len(data), dtype=np.int64) + offset) * request.bins // samples)
            np.maximum.at(peaks, indices, np.nan_to_num(np.abs(data), nan=0, posinf=1))
            offset += len(data)
        cancel.check()
        if process.wait(timeout=10) or offset == 0:
            raise ValueError('音频波形读取失败')
        return np.clip(peaks, 0, 1)
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        process.stdout.close()


def build_overview(request, cancel):
    images = ()
    peaks = np.empty(0, np.float32)
    video_error = audio_error = ''
    try:
        images = read_film(request, cancel)
    except Cancelled:
        raise
    except Exception as error:
        video_error = str(error)
    cancel.check()
    try:
        peaks = read_peaks(request, cancel)
    except Cancelled:
        raise
    except Exception as error:
        audio_error = str(error)
    cancel.check()
    return MediaOverview(request.key, request.source.identity, request.start, request.end,
                         request.film_frames, images, peaks, request.source.has_audio,
                         video_error, audio_error)


class TimelineMedia(QObject):
    """Latest-request-wins worker with a small memory and bounded disk cache."""
    ready = Signal(object)

    def __init__(self, parent=None, directory=None):
        super().__init__(parent)
        self.directory = Path(directory) if directory else cache_root() / 'timeline-media-v2'
        self.directory.mkdir(parents=True, exist_ok=True)
        self._condition = threading.Condition()
        self._pending = None
        self._active = None
        self._wanted = None
        self._closed = self._busy = False
        self._memory = OrderedDict()
        self._thread = threading.Thread(target=self._run, name='timeline-media', daemon=True)
        self._thread.start()

    def request(self, request):
        with self._condition:
            if self._closed or request.key == self._wanted:
                return
            self._wanted = request.key
            self._pending = request
            if self._active:
                self._active.cancel()
            self._condition.notify_all()

    def set_busy(self, busy):
        with self._condition:
            self._busy = bool(busy)
            if busy and self._active:
                self._active.cancel()
            self._condition.notify_all()

    def clear(self):
        with self._condition:
            self._wanted = None
            self._pending = None
            if self._active:
                self._active.cancel()

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            if self._active:
                self._active.cancel()
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def _load(self, request):
        if request.key in self._memory:
            self._memory.move_to_end(request.key)
            return self._memory[request.key]
        path = self.directory / (request.key + '.npz')
        try:
            # Never allow object/pickle arrays in persistent media caches.
            with np.load(path, allow_pickle=False) as data:
                count = int(data['count'])
                peaks = data['peaks']
                if not 0 <= count <= 16 or peaks.size > 4096:
                    return None
                images = tuple(data[f'image{i}'] for i in range(count))
                if any(a.ndim != 3 or a.shape[2] != 3 or a.dtype!=np.uint8
                       or a.shape[1]>request.tile_width or a.shape[0]>request.tile_height for a in images):
                    return None
                result = MediaOverview(request.key, request.source.identity, request.start, request.end,
                                       request.film_frames, images, peaks, request.source.has_audio)
                self._remember(result)
                return result
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    def _remember(self, result):
        self._memory[result.key] = result
        self._memory.move_to_end(result.key)
        while len(self._memory) > 8 or sum(sum(a.nbytes for a in r.images)+r.peaks.nbytes for r in self._memory.values())>32*1024*1024:
            self._memory.popitem(last=False)

    def _save(self, result):
        if result.video_error or result.audio_error:
            return  # Failed reads are retryable after reopening the clip.
        self._remember(result)
        path = self.directory / (result.key + '.npz')
        temporary = self.directory / (result.key + '.partial')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, count=len(result.images), peaks=result.peaks,
                                    **{f'image{i}': a for i, a in enumerate(result.images)})
            os.replace(temporary, path)
            entries = sorted(self.directory.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
            size = 0
            for entry in entries:
                size += entry.stat().st_size
                if size > 64 * 1024 * 1024:
                    entry.unlink(missing_ok=True)
        except OSError:
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or (self._pending is not None and not self._busy))
                if self._closed:
                    return
                request = self._pending
                self._pending = None
                token = WorkCancel()
                self._active = token
            result = None
            try:
                result = self._load(request)
                if result is None:
                    result = build_overview(request, token)
                    token.check()
                    # Compression stays outside the control lock. Cache hits
                    # already have a disk entry and only need to be delivered.
                    self._save(result)
            except Cancelled:
                result = None
            except Exception as error:
                # A damaged cache/decoder must not silently kill future requests.
                result = MediaOverview(request.key, request.source.identity,
                    request.start, request.end, request.film_frames, (),
                    np.empty(0,np.float32), request.source.has_audio,
                    str(error), str(error) if request.source.has_audio else '')
            finally:
                with self._condition:
                    self._active = None
                    if not self._closed and self._wanted == request.key and self._pending is None:
                        # Playback may already have stopped when the worker
                        # observes cancellation. The token still needs a retry.
                        if token.event.is_set():
                            self._pending = request
                        elif result is not None:
                            self.ready.emit(result)
