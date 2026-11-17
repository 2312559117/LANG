"""Bounded native-YUV disk store for non-monotonic source-frame access.

The analysis clock remains the product's existing CFR clock. This module does
not claim an original VFR-PTS reconstruction or change the tracking timeline.
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

from .media import NO_WINDOW, check_cancel, find_tool,first_frame_side_data

MAX_STORE_BYTES = 64 * 1024**3


def temp_root():
    configured = os.environ.get('LOCKFRAME_RETIME_TEMP')
    folder = Path(configured) if configured else Path(tempfile.gettempdir())/'tomato-retime'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def fps_text(fps):
    rate = Fraction(str(float(fps))).limit_denominator(1_000_000)
    if rate <= 0:
        raise ValueError('输出帧率无效')
    return f'{rate.numerator}/{rate.denominator}'


def run_process(command, cancel, *, output_limit=2*1024**2, timeout=120, cwd=None):
    """Small-output cancellable child, always killed and reaped on failure."""
    check_cancel(cancel)
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, cwd=cwd, creationflags=NO_WINDOW)
    deadline = time.monotonic()+timeout if timeout else None
    try:
        while True:
            check_cancel(cancel)
            if deadline is not None and time.monotonic()>deadline:
                raise TimeoutError('变速媒体处理长时间未结束，请检查磁盘和源文件')
            try:
                output, errors = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                continue
        check_cancel(cancel)
        if len(output)+len(errors)>output_limit:
            raise RuntimeError('媒体子进程诊断输出超出限制')
        if process.returncode:
            raise RuntimeError('变速媒体处理失败：'+errors.decode('utf-8','replace')[-1800:])
        return output
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


def probe_native(path, cancel):
    raw=run_process([find_tool('ffprobe'),'-v','error','-select_streams','v:0',
                     '-show_streams','-of','json',str(path)],cancel,timeout=30)
    streams=json.loads(raw).get('streams',[])
    if not streams:
        raise ValueError('源文件没有视频画面')
    video=streams[0]
    if video.get('color_transfer') in ('smpte2084','arib-std-b67'):
        video=dict(video,side_data_list=[*video.get('side_data_list',[]),
                                        *first_frame_side_data(path,cancel)])
    return video


def canonical_format(name):
    """Planar layout conversion preserves component depth/subsampling."""
    value=str(name).lower()
    if value in ('yuv420p','yuv422p','yuv444p'):
        return value
    if value in ('yuvj420p','yuvj422p','yuvj444p'):
        return value.replace('yuvj','yuv')
    if re.fullmatch(r'yuv(?:420|422|444)p(?:10|12|16)le',value):
        return value
    aliases={'nv12':'yuv420p','nv21':'yuv420p','p010le':'yuv420p10le',
             'p016le':'yuv420p16le','nv16':'yuv422p','p210le':'yuv422p10le',
             'p216le':'yuv422p16le','p410le':'yuv444p10le','p416le':'yuv444p16le'}
    if value in aliases:
        return aliases[value]
    # RGB conversions are explicit, preserve the reported component precision,
    # and never downgrade an HDR/10-bit source to an 8-bit RGB intermediate.
    if value in ('rgb24','bgr24','rgba','bgra','gbrp'):
        return 'yuv444p'
    match=re.fullmatch(r'gbrp(10|12|16)le',value)
    if match:
        return 'yuv444p'+match[1]+'le'
    raise ValueError(f'原生变速暂不支持像素格式 {name}，请先转换为同位深YUV格式')


def read_native_frame(info,index,cancel,*,metadata=None):
    """One CFR analysis-clock source frame, for bounded preview pair reads.

    No raw-file cache is created. Seeking here prioritises matching the existing
    CFR analysis clock over approximate keyframe seeking; late frames may take
    time to decode. Preview workers should cancel stale requests.
    """
    from .flow.frame import YuvFrame,parse_format
    if type(index) is not int or not 0<=index<info.frames:
        raise IndexError('预览源帧超出范围')
    metadata=metadata or probe_native(info.path,cancel)
    fmt=parse_format(canonical_format(metadata.get('pix_fmt',getattr(info,'pixel_format','yuv420p'))))
    vf=(f'fps={fps_text(info.fps)},trim=start_frame={index}:end_frame={index+1},'
        f'setpts=PTS-STARTPTS,scale={info.width}:{info.height}:flags=bicubic,setsar=1')
    count=fmt.frame_bytes(info.width,info.height)
    raw=run_process([find_tool('ffmpeg'),'-v','error','-nostdin','-i',info.path,'-map','0:v:0',
        '-an','-sn','-dn','-vf',vf,'-frames:v','1','-pix_fmt',fmt.canonical_name,'-f','rawvideo','pipe:1'],
        cancel,output_limit=count+2*1024**2,timeout=None)
    if len(raw)!=count:raise ValueError('预览源帧解码不完整')
    return YuvFrame.from_bytes(raw,info.width,info.height,fmt)


def native_preview_bgr(frame,info,cancel):
    """Display-only native YUV conversion; HDR tone mapping never enters export."""
    check_cancel(cancel)
    color=[]
    for field,flag in (('color_primaries','-color_primaries'),('color_transfer','-color_trc'),
                       ('color_space','-colorspace'),('color_range','-color_range')):
        value=getattr(info,field,'unknown')
        if value not in (None,'','unknown','unspecified','reserved') and re.fullmatch(r'[a-zA-Z0-9_-]{1,40}',str(value)):
            color.extend([flag,str(value)])
    vf='format=bgr24'
    if getattr(info,'hdr',False):
        vf='zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable,zscale=p=bt709:t=bt709:m=bt709:r=tv,format=bgr24'
    # communicate(input=...) can deadlock without cancellation while native
    # filters run; the watcher kills the child and the finally always reaps it.
    process=subprocess.Popen([find_tool('ffmpeg'),'-v','error','-nostdin','-f','rawvideo',
        '-pix_fmt',frame.format.canonical_name,'-s',f'{frame.width}x{frame.height}',*color,
        '-i','pipe:0','-frames:v','1','-vf',vf,'-f','rawvideo','-pix_fmt','bgr24','pipe:1'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=NO_WINDOW)
    try:
        with cancel_guard(process,cancel):output,errors=process.communicate(frame.to_bytes())
        check_cancel(cancel)
        if process.returncode:raise RuntimeError('原生预览色彩转换失败：'+errors.decode('utf-8','replace')[-1000:])
        if len(output)!=frame.width*frame.height*3:raise ValueError('原生预览像素不完整')
        return np.frombuffer(output,dtype=np.uint8).reshape(frame.height,frame.width,3).copy()
    finally:
        if process.poll() is None:process.kill()
        process.communicate()


def check_disk_budget(folder, required, *, limit=MAX_STORE_BYTES):
    if required<0 or required>limit:
        raise ValueError(f'变速源帧缓存预计 {required/1024**3:.2f} GiB，超出 {limit/1024**3:.0f} GiB预算；请缩短片段')
    free=shutil.disk_usage(folder).free
    reserve=max(256*1024**2, int(required*.05))
    if required+reserve>free:
        raise OSError(f'变速临时磁盘空间不足：需要约 {(required+reserve)/1024**3:.2f} GiB，可用 {free/1024**3:.2f} GiB')


class NativeFrameStore:
    """Read-only memmap with one-frame owned copies, never a full RAM decode."""
    def __init__(self, path, width, height, count, pixel_format):
        from .flow.frame import parse_format
        self.path=Path(path)
        self.width,self.height,self.count=int(width),int(height),int(count)
        self.format=parse_format(pixel_format)
        self.frame_bytes=self.format.frame_bytes(self.width,self.height)
        if self.path.stat().st_size!=self.count*self.frame_bytes:
            raise ValueError('原生源帧缓存帧数或字节数不完整')
        self._map=np.memmap(self.path,mode='r',dtype=np.uint8)

    def read(self,index):
        from .flow.frame import YuvFrame
        if type(index) is not int or not 0<=index<self.count:
            raise IndexError('源帧索引超出变速缓存范围')
        start=index*self.frame_bytes
        # from_bytes owns plane arrays, so eviction/closing does not invalidate a
        # frame already handed to a flow pair or preview worker.
        return YuvFrame.from_bytes(self._map[start:start+self.frame_bytes].tobytes(),
                                   self.width,self.height,self.format)

    def close(self):
        if self._map is not None:
            self._map._mmap.close()
            self._map=None

    @classmethod
    @contextmanager
    def open(cls,info,track,folder,cancel,progress=None,*,metadata=None):
        from .flow.frame import parse_format
        metadata=metadata or probe_native(info.path,cancel)
        pixel_format=canonical_format(metadata.get('pix_fmt',getattr(info,'pixel_format','yuv420p')))
        fmt=parse_format(pixel_format)
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        required=fmt.frame_bytes(info.width,info.height)*track.count
        check_disk_budget(folder,required)
        path=folder/'selected-source.yuv'
        range_option=':in_range=full:out_range=full' if metadata.get('color_range')=='pc' or str(metadata.get('pix_fmt','')).startswith('yuvj') else ''
        vf=(f'fps={fps_text(info.fps)},trim=start_frame={track.start}:end_frame={track.end+1},'
            f'setpts=PTS-STARTPTS,scale={info.width}:{info.height}:flags=bicubic{range_option},setsar=1')
        if progress:
            progress(0.,'准备原生源帧 · 支持倒放与回跳的D盘缓存')
        # File output avoids a pipe/RAM queue whose length grows with the clip.
        run_process([find_tool('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-y',
                     '-i',info.path,'-map','0:v:0','-an','-sn','-dn','-vf',vf,
                     '-frames:v',str(track.count),'-pix_fmt',pixel_format,'-f','rawvideo',str(path)],
                    cancel,timeout=None)
        store=cls(path,info.width,info.height,track.count,pixel_format)
        try:
            if progress:
                progress(1.,'原生源帧已准备')
            yield store
        finally:
            store.close()


@contextmanager
def cancel_guard(process,cancel):
    """Unblock a pipe writer if cancellation arrives during encoder backpressure."""
    stopped=threading.Event()
    def watch():
        while not stopped.wait(.1):
            if cancel.is_set():
                if process.poll() is None:
                    try:process.kill()
                    except OSError:pass
                return
    thread=threading.Thread(target=watch,name='retime-encoder-cancel',daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set();thread.join(timeout=1)
