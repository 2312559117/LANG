"""One background decoder, bounded JPEG cache; Qt images stay on the GUI thread."""
from collections import OrderedDict, deque
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import os
import threading

from PySide6.QtCore import QObject, QTimer, Signal

from .sequence import clip_thumbnail


class ThumbnailQueue(QObject):
    ready = Signal(str, str, object, str)

    def __init__(self, parent=None, *, decoder=clip_thumbnail, entries=128, budget=8*1024*1024):
        super().__init__(parent)
        self.decoder, self.entries, self.budget = decoder, entries, budget
        self._condition = threading.Condition()
        self._wanted, self._queue = {}, OrderedDict()
        self._cache, self._results = OrderedDict(), deque()
        self._running = self._thread = None
        self._closed = self._paused = False
        self._stats = dict(decoded=0, cache_hits=0, failures=0, probes=0)
        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._poll)

    @staticmethod
    def key(clip):
        info=clip.info
        data=(os.path.normcase(info.path),info.fingerprint,info.proxy,info.frames,
              info.width,info.height,info.fps,clip.start,clip.end)
        return hashlib.sha256(repr(data).encode('utf-8')).hexdigest()

    def sync(self, clips):
        wanted={clip.id:self.key(clip) for clip in clips}
        with self._condition:
            if self._closed:return
            for id in list(self._queue):
                if id not in wanted or self._queue[id][0]!=wanted[id]:self._queue.pop(id)
            for clip in clips:
                key=wanted[clip.id]
                if self._wanted.get(clip.id)!=key:
                    self._queue[clip.id]=(key,SimpleNamespace(info=replace(clip.info),start=clip.start,end=clip.end))
            self._wanted=wanted
            self._start()

    def recheck(self, clips):
        """Visible rows get fresh file-stat validation, with all IO off the UI."""
        with self._condition:
            if self._closed:return
            for clip in reversed(clips):
                key=self.key(clip)
                if self._wanted.get(clip.id)!=key or self._running==(clip.id,key):continue
                self._queue[clip.id]=(key,SimpleNamespace(info=replace(clip.info),start=clip.start,end=clip.end))
                self._queue.move_to_end(clip.id,last=False)
            self._start()

    def _start(self):
        if self._queue and self._thread is None:
            self._thread=threading.Thread(target=self._work,name='tomato-thumbnails',daemon=True)
            self._thread.start()
        self._condition.notify_all()
        if self._queue and not self._paused:self._timer.start()

    @staticmethod
    def _stamp(path):
        if not path:return None
        try:
            stat=os.stat(path)
            return stat.st_size,stat.st_mtime_ns
        except OSError as error:return ('unavailable',str(error))

    def _read(self, key, clip):
        source,proxy=self._stamp(clip.info.path),self._stamp(clip.info.proxy)
        cache_key=(key,source,proxy)
        with self._condition:
            self._stats['probes']+=1
            if cache_key in self._cache:
                self._stats['cache_hits']+=1
                value=self._cache.pop(cache_key);self._cache[cache_key]=value
                return value
        try:
            if source[0]=='unavailable':raise ValueError('源文件无法访问')
            if clip.info.fingerprint and f'{source[0]}:{source[1]}'!=clip.info.fingerprint:
                raise ValueError('源文件已变化，请重新导入素材')
            jpeg=self.decoder(clip,160,90)
            if self._stamp(clip.info.path)!=source or self._stamp(clip.info.proxy)!=proxy:
                raise ValueError('读取时素材已变化，请重新导入')
            value=(jpeg,'')
            with self._condition:self._stats['decoded']+=1
        except Exception as error:
            value=(b'',str(error))
            with self._condition:self._stats['failures']+=1
            from .logging import get_logger
            get_logger('thumbnails').warning('缩略图不可用 %s: %s',clip.info.path,error)
        with self._condition:
            if not self._closed:
                self._cache[cache_key]=value
                while len(self._cache)>self.entries or sum(len(v[0]) for v in self._cache.values())>self.budget:
                    self._cache.popitem(last=False)
        return value

    def _work(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._closed or (self._queue and not self._paused))
                if self._closed:return
                id,(key,clip)=self._queue.popitem(last=False)
                self._running=(id,key)
            jpeg,error=self._read(key,clip)
            with self._condition:
                self._condition.wait_for(lambda:self._closed or len(self._results)<16)
                if self._closed:return
                if self._wanted.get(id)==key:self._results.append((id,key,jpeg,error))
                self._running=None

    def _poll(self):
        with self._condition:
            results=list(self._results);self._results.clear()
            wanted=dict(self._wanted)
            active=self._running is not None or (self._queue and not self._paused)
            self._condition.notify_all()
        for id,key,jpeg,error in results:
            if not self._closed and wanted.get(id)==key:self.ready.emit(id,key,jpeg,error)
        if not active:self._timer.stop()

    def set_paused(self, paused):
        with self._condition:
            self._paused=paused
            self._start()

    def statistics(self):
        with self._condition:
            return dict(self._stats,cache_entries=len(self._cache),cache_bytes=sum(len(v[0]) for v in self._cache.values()),
                        pending=len(self._queue),results=len(self._results))

    def close(self):
        with self._condition:
            self._closed=True
            self._queue.clear();self._wanted.clear();self._results.clear();self._cache.clear()
            self._condition.notify_all()
        self._timer.stop()
