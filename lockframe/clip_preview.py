"""A cancellable three-second render through the real native export pipeline."""
from copy import deepcopy
from pathlib import Path
import json,hashlib,shutil,tempfile
import numpy as np
from PySide6.QtCore import QObject,Signal
from .media import FrameReader,check_cancel,run_ffmpeg,preview_tonemap_filter
from .retime_frames import temp_root,probe_native
from .retime_renderer import render_retimed_video
from .workers import Job
from .logging import get_logger


def render_preview_clip(info,track,settings,mapping,first,key,folder,cancel,progress=None):
    folder=Path(folder);settings=deepcopy(settings)
    # Workspace previews are silent, with the exact spatial/flow/encoding
    # settings. Only the final display copy is downscaled/tone-mapped.
    settings.retime=deepcopy(settings.retime);settings.retime['audio_policy']='mute'
    native=folder/'native.mp4';display=folder/'display.mp4'
    report=render_retimed_video(info,track,settings,np.asarray(mapping,dtype=np.float64),str(native),
        cancel,(lambda value,text:progress(.9*value,text)) if progress else None,overwrite=False)
    check_cancel(cancel)
    factor=min(1.,1280/max(report['width'],report['height']))
    width=max(2,round(report['width']*factor/2)*2);height=max(2,round(report['height']*factor/2)*2)
    if progress:progress(.92,'光流片段已渲染，正在生成流畅显示副本')
    run_ffmpeg(['-i',str(native),'-map','0:v:0','-an','-vf',
        preview_tonemap_filter(info,opencv_proxy=True)+f'scale={width}:{height},setsar=1',
        '-frames:v',str(len(mapping)),'-c:v','libx264','-preset','ultrafast','-crf','16','-g','1',
        '-pix_fmt','yuv420p','-map_metadata','-1',str(display)],cancel,len(mapping)/info.fps)
    check_cancel(cancel)
    if int(probe_native(display,cancel).get('nb_frames',-1))!=len(mapping):
        raise ValueError('预渲染显示副本帧数不完整，未交付预览')
    if display.stat().st_size>256*1024**2:raise ValueError('三秒预览超过256MiB缓存预算，请降低输出尺寸')
    native.unlink(missing_ok=True)
    if progress:progress(1.,'三秒预渲染完成，可播放检查')
    return dict(key=key,path=str(display),first=first,last=first+len(mapping)-1,
        frames=len(mapping),width=width,height=height,report=report)


class ClipPreview(QObject):
    changed=Signal()

    def __init__(self,parent):
        super().__init__(parent);self.worker=None;self.cache=None;self.reader=None
        self.wanted=None;self.folder=None;self.work_folder=None;self.message=''
        self.track_ref=None
        self.root=temp_root()/'preview-v0730';self.root.mkdir(parents=True,exist_ok=True)

    @property
    def running(self):return self.worker is not None

    @staticmethod
    def key(info,track,settings,start,end,anchors,clip_id):
        from dataclasses import asdict
        values=asdict(settings);values.pop('drift_review',None)
        payload=dict(source=[info.path,info.fingerprint],track=id(track),range=[start,end],
            anchors=sorted(anchors.items()),clip_id=clip_id,settings=values)
        return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

    def want(self,key):
        if key==self.wanted:return
        self.wanted=key
        if self.worker:self.worker.cancel()
        self.clear_cache();self.message='参数已改变，请重新预渲染' if key else ''

    def _clean(self,folder):
        if folder is None:return
        path=Path(folder).resolve();root=self.root.resolve()
        if path==root or not path.is_relative_to(root) or not path.name.startswith('clip-'):
            raise ValueError('拒绝清理非预览缓存目录')
        try:shutil.rmtree(path)
        except FileNotFoundError:pass
        except OSError:get_logger('clip-preview').exception('无法清理预览缓存')

    def clear_cache(self):
        if self.reader:self.reader.close();self.reader=None
        self.cache=None;self.track_ref=None
        if self.folder:self._clean(self.folder);self.folder=None

    def prepare(self,key,info,track,settings,mapping,first):
        if self.running:return False
        self.want(key);self.work_folder=Path(tempfile.mkdtemp(prefix='clip-',dir=self.root))
        self.track_ref=track
        self.worker=Job(render_preview_clip,deepcopy(info),deepcopy(track),deepcopy(settings),
            np.array(mapping,dtype=np.float64,copy=True),first,key,str(self.work_folder),parent=self)
        self.worker.progress.connect(self._progress);self.worker.result.connect(self._done)
        self.worker.failed.connect(self._failed);self.worker.aborted.connect(lambda:self._failed('预渲染已取消'))
        self.worker.finished.connect(self._finished)
        self.message='正在预渲染当前三秒，完成后可检查真实光流效果';self.changed.emit()
        self.worker.start();return True

    def _progress(self,value,text):self.message=f'预渲染 {value:.0%} · {text}';self.changed.emit()

    def _done(self,payload):
        if payload['key']!=self.wanted:return
        reader=None
        try:
            reader=FrameReader(payload['path']);reader.read(0)
        except Exception as error:
            if reader:reader.close()
            self._failed('预览副本无法读取：'+str(error));return
        track_ref=self.track_ref
        self.clear_cache();self.reader=reader;self.cache=payload;self.track_ref=track_ref
        self.folder=self.work_folder;self.work_folder=None
        self.message=f'已缓存 {payload["frames"]} 帧 · 静音显示代理，可播放检查'
        self.changed.emit()

    def _failed(self,message):self.message=str(message);self.changed.emit()

    def _finished(self):
        worker=self.worker;self.worker=None
        if self.work_folder:self._clean(self.work_folder);self.work_folder=None
        if self.cache is None:self.track_ref=None
        if worker:worker.deleteLater()
        self.changed.emit()

    def frame(self,key,step):
        if (key!=self.wanted or self.cache is None or self.reader is None or
                not self.cache['first']<=step<=self.cache['last']):return None
        try:return self.reader.read(step-self.cache['first'])
        except Exception as error:
            self.message='预览缓存读取失败：'+str(error);self.clear_cache();self.changed.emit();return None

    def cancel(self):
        if self.worker:self.worker.cancel();self.message='正在取消预渲染…';self.changed.emit()

    def close(self):
        if self.running:self.cancel();return False
        self.wanted=None;self.clear_cache();return True
