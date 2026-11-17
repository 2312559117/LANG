"""One active/one pending stop-frame preview; playback uses explicit nearest frames."""
from collections import OrderedDict
import threading
import numpy as np
import cv2
from PySide6.QtCore import QObject, Signal
from .media import FrameReader
from .flow.frame import YuvFrame, parse_format
from .flow.base import create_interpolator
from .stabilization import preview_frame


class RetimePreview(QObject):
    ready=Signal(object,object,str)

    def __init__(self,parent=None):
        super().__init__(parent);self._lock=threading.Lock();self._pending=None
        self._cancel=None;self._thread=None;self._closed=False;self.wanted=None;self.generation=0

    def request(self,token,info,plan,left,alpha,config):
        with self._lock:
            if self._closed or token==self.wanted:return
            self.wanted=token
            if self._cancel:self._cancel.set()
            self._pending=(token,info,plan,left,alpha,config)
            if self._thread is None:
                self._thread=threading.Thread(target=self._run,daemon=True,name='retime-preview')
                self._thread.start()

    def invalidate(self):
        with self._lock:
            self.generation+=1;self.wanted=None;self._pending=None
            if self._cancel:self._cancel.set()

    def _run(self):
        # Decoder and pairs are worker-local. A changed clip/plan clears both.
        reader=None;engine=None;identity=None;cache=OrderedDict()
        try:
            while True:
                with self._lock:
                    if self._closed or self._pending is None:
                        self._thread=None;return
                    token,info,plan,left,alpha,config=self._pending;self._pending=None
                    cancel=threading.Event();self._cancel=cancel
                try:
                    key=(info.proxy,info.fingerprint,id(plan),config['flow_backend'],config['allow_frame_fallback'])
                    if key!=identity:
                        if reader:reader.close()
                        if engine:engine.close()
                        reader=FrameReader(info.proxy);cache.clear();identity=key
                        engine=create_interpolator(config['flow_backend'],config['allow_frame_fallback'],cancel)
                    # Cancellation belongs to this request, including reused engine.
                    engine.cancel=cancel
                    def frame(index):
                        if cancel.is_set():raise InterruptedError('预览已更新')
                        if index not in cache:
                            bgr=reader.read(index)
                            if plan is not None:
                                bgr=preview_frame(bgr,info,plan,index-token[2],max_side=960)
                            elif max(bgr.shape[:2])>960:
                                scale=960/max(bgr.shape[:2]);bgr=cv2.resize(bgr,None,fx=scale,fy=scale)
                            yuv=cv2.cvtColor(bgr,cv2.COLOR_BGR2YUV)
                            cache[index]=YuvFrame(tuple(np.ascontiguousarray(yuv[:,:,i]) for i in range(3)),parse_format('yuv444p'))
                            while len(cache)>2:cache.popitem(last=False)
                        return cache[index]
                    a=frame(left);b=frame(left+1)
                    from .retime_renderer import _scene_cut
                    if _scene_cut(a,b):raise ValueError('相邻帧疑似切镜，不跨切镜合成')
                    pair=engine.prepare(a,b);out=pair.compose(alpha)
                    image=cv2.cvtColor(np.stack(out.planes,axis=-1),cv2.COLOR_YUV2BGR)
                    if not cancel.is_set():self.ready.emit(token,image,'DIS 停帧代理光流预览 · 导出使用原生位深')
                except Exception as error:
                    if cancel.is_set():
                        if engine:engine.close()
                        engine=None;identity=None;cache.clear()
                    else:
                        from .logging import get_logger
                        get_logger('retime-preview').exception('停帧光流预览失败')
                        self.ready.emit(token,None,f'光流预览失败：{error}；当前显示最近源帧')
        finally:
            if reader:reader.close()
            if engine:engine.close()

    def close(self):
        with self._lock:
            self._closed=True;self._pending=None;self.wanted=None
            if self._cancel:self._cancel.set()
        # A Python daemon has no QThread destruction hazard. Never block Qt on DIS.
