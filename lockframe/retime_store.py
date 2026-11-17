"""Access-pattern-aware native frames without changing the existing CFR clock."""
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path
import subprocess
import time
import numpy as np
from .media import check_cancel,find_tool,NO_WINDOW
from .flow.frame import parse_format,YuvFrame
from .retime_frames import NativeFrameStore,canonical_format,fps_text,cancel_guard


class StreamingFrameStore:
    """Forward native decode, at most two owned source frames and no YUV file."""
    def __init__(self,info,track,folder,cancel,metadata,first=0,last=None):
        self.info,self.track,self.cancel=info,track,cancel
        self._metadata=metadata
        self.width,self.height,self.count=info.width,info.height,track.count
        self.format=parse_format(canonical_format(metadata.get('pix_fmt',info.pixel_format)))
        self.frame_bytes=self.format.frame_bytes(self.width,self.height)
        self.first_requested=first;self.last_requested=track.count-1 if last is None else last
        self.frames=OrderedDict();self.last=first-1;self.decoded_frames=0;self.decode_seconds=0.
        self.bytes_written=0;self.peak_cache_bytes=0;self.strategy='stream'
        self.log_path=Path(folder)/'stream-decode.log';self.log=self.log_path.open('wb')
        range_option=':in_range=full:out_range=full' if metadata.get('color_range')=='pc' or str(metadata.get('pix_fmt','')).startswith('yuvj') else ''
        vf=(f'fps={fps_text(info.fps)},trim=start_frame={track.start+first}:end_frame={track.start+self.last_requested+1},'
            f'setpts=PTS-STARTPTS,scale={self.width}:{self.height}:flags=bicubic{range_option},setsar=1')
        self.process=None;self.guard=None
        try:
            check_cancel(cancel)
            self.process=subprocess.Popen([find_tool('ffmpeg'),'-v','error','-nostdin','-i',info.path,
                '-map','0:v:0','-an','-sn','-dn','-vf',vf,'-frames:v',str(self.last_requested-first+1),
                '-pix_fmt',self.format.canonical_name,'-f','rawvideo','pipe:1'],
                stdout=subprocess.PIPE,stderr=self.log,stdin=subprocess.DEVNULL,creationflags=NO_WINDOW)
            self.guard=cancel_guard(self.process,cancel);self.guard.__enter__()
        except BaseException:
            self.close();raise

    def read(self,index):
        check_cancel(self.cancel)
        if type(index) is not int or not 0<=index<self.count:raise IndexError('源帧索引超出处理范围')
        if index in self.frames:return self.frames[index].copy()
        if index<self.last:raise ValueError('顺序解码窗口无法读取已释放源帧')
        began=time.perf_counter()
        try:
            while self.last<index:
                check_cancel(self.cancel)
                raw=self.process.stdout.read(self.frame_bytes)
                check_cancel(self.cancel)
                if len(raw)!=self.frame_bytes:
                    self.log.flush()
                    raise ValueError('原生源帧解码中断：'+self.log_path.read_text(encoding='utf-8',errors='replace')[-600:])
                self.last+=1;self.decoded_frames+=1
                # Skipped forward frames never enter the native-plane cache.
                if self.last>=index-1:
                    self.frames[self.last]=YuvFrame.from_bytes(raw,self.width,self.height,self.format)
                    while len(self.frames)>2:self.frames.popitem(last=False)
                    self.peak_cache_bytes=max(self.peak_cache_bytes,sum(f.nbytes for f in self.frames.values()))
            return self.frames[index].copy()
        finally:self.decode_seconds+=time.perf_counter()-began

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:self.process.kill()
            self.process.wait()
            if self.process.stdout:self.process.stdout.close()
        if self.guard is not None:self.guard.__exit__(None,None,None);self.guard=None
        if not self.log.closed:self.log.close()
        self.frames.clear()

    def rewind(self):
        decoded,seconds,peak=self.decoded_frames,self.decode_seconds,self.peak_cache_bytes
        self.close()
        self.__init__(self.info,self.track,self.log_path.parent,self.cancel,self._metadata,self.first_requested,self.last_requested)
        self.decoded_frames+=decoded;self.decode_seconds+=seconds;self.peak_cache_bytes=peak


class ChunkedFrameStore:
    """Sparse/repeated source access with bounded, lazily decoded disk blocks.

    Each block uses the proven absolute CFR trim. Broad reverse traversal uses
    one-pass disk mode instead, avoiding repeated full-prefix decoding here.
    """
    def __init__(self,info,track,folder,cancel,metadata):
        self.info,self.track,self.folder,self.cancel,self.metadata=info,track,Path(folder),cancel,metadata
        self.width,self.height,self.count=info.width,info.height,track.count
        self.format=parse_format(canonical_format(metadata.get('pix_fmt',info.pixel_format)))
        self.frame_bytes=self.format.frame_bytes(self.width,self.height)
        # At most two blocks, normally <=128 MiB total; a giant source frame
        # still needs one real frame and is never silently downconverted.
        self.block_frames=max(1,min(32,(64*1024**2)//self.frame_bytes))
        self.blocks=OrderedDict();self.strategy='chunks';self.bytes_written=0
        self.peak_cache_bytes=0;self.decoded_frames=0;self.decode_seconds=0.

    def read(self,index):
        check_cancel(self.cancel)
        if type(index) is not int or not 0<=index<self.count:raise IndexError('源帧索引超出处理范围')
        block=index//self.block_frames
        if block not in self.blocks:
            while len(self.blocks)>=2:self._drop(next(iter(self.blocks)))
            first=block*self.block_frames;count=min(self.block_frames,self.count-first)
            sub=SimpleNamespace(start=self.track.start+first,end=self.track.start+first+count-1,count=count)
            folder=self.folder/f'block-{block}'
            context=NativeFrameStore.open(self.info,sub,folder,self.cancel,metadata=self.metadata)
            began=time.perf_counter()
            try:store=context.__enter__()
            except BaseException:
                (folder/'selected-source.yuv').unlink(missing_ok=True)
                raise
            finally:self.decode_seconds+=time.perf_counter()-began
            self.blocks[block]=(context,store)
            self.bytes_written+=store.count*store.frame_bytes;self.decoded_frames+=store.count
            self.peak_cache_bytes=max(self.peak_cache_bytes,sum(s.count*s.frame_bytes for _,s in self.blocks.values()))
        self.blocks.move_to_end(block)
        return self.blocks[block][1].read(index%self.block_frames)

    def _drop(self,index):
        context,store=self.blocks.pop(index)
        context.__exit__(None,None,None);store.path.unlink(missing_ok=True)
        try:store.path.parent.rmdir()
        except OSError:pass

    def close(self):
        for index in list(self.blocks):self._drop(index)


def store_strategy(values,count,frame_bytes):
    if np.all(np.diff(values)>=0):return 'stream'
    block=max(1,min(32,(64*1024**2)//frame_bytes))
    needed=np.unique(np.concatenate((np.floor(values).astype(np.int64)//block,
        np.minimum(count-1,np.ceil(values)).astype(np.int64)//block)))
    all_blocks=(count+block-1)//block
    return 'chunks' if len(needed)<=8 and len(needed)<all_blocks*.5 else 'disk'


@contextmanager
def open_frame_store(info,track,values,folder,cancel,progress=None,*,metadata):
    fmt=parse_format(canonical_format(metadata.get('pix_fmt',info.pixel_format)))
    strategy=store_strategy(values,track.count,fmt.frame_bytes(info.width,info.height))
    if strategy=='disk':
        began=time.perf_counter()
        with NativeFrameStore.open(info,track,folder,cancel,progress,metadata=metadata) as store:
            store.strategy='disk';store.bytes_written=store.count*store.frame_bytes
            store.peak_cache_bytes=store.bytes_written;store.decoded_frames=store.count
            store.decode_seconds=time.perf_counter()-began
            yield store
        return
    if progress:progress(1.,'按需顺序解码，不写全段原始帧' if strategy=='stream' else '按需分块缓存，保留原生位深')
    if strategy=='stream':
        first=int(np.floor(np.min(values)));last=min(track.count-1,int(np.ceil(np.max(values))))
        store=StreamingFrameStore(info,track,folder,cancel,metadata,first,last)
    else:store=ChunkedFrameStore(info,track,folder,cancel,metadata)
    try:yield store
    finally:store.close()
