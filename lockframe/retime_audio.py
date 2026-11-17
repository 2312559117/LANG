"""Explicit audio policies for a source-bound frame map.

Fixed speed uses atempo and retains pitch. Strictly increasing variable maps
use linear PCM resampling, following the same map while changing pitch. Reverse,
repeat and freeze are explicitly muted until a suitable editing model exists.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from .media import check_cancel,find_tool
from .retime_frames import run_process,check_disk_budget


def atempo_chain(speed):
    if not math.isfinite(speed) or speed<=0:
        raise ValueError('音频速度无效')
    values=[]
    while speed>2:
        values.append(2.);speed/=2
    while speed<.5:
        values.append(.5);speed/=.5
    values.append(speed)
    return ','.join(f'atempo={value:.14g}' for value in values)


def audio_strategy(info,track,frame_map,config):
    if not info.has_audio:
        return dict(policy='none',method='none',reason='原片没有音轨')
    if not config.get('enabled',False):
        return dict(policy='original',method='atempo',reason='',speed=1.,
                    start=track.start/info.fps,source_duration=track.count/info.fps)
    if config.get('audio_policy','mute')=='mute':
        return dict(policy='muted',method='none',reason='用户选择变速静音')
    differences=np.diff(frame_map)
    if len(frame_map)<2 or np.any(differences<=1e-10):
        return dict(policy='muted',method='none',reason='曲线含倒放、回跳或定格，当前音频策略自动静音')
    duration=len(frame_map)/info.fps
    if np.allclose(differences,differences[0],rtol=1e-7,atol=1e-9):
        # MS-style frame-centre maps include both endpoint frames. Audio uses
        # the inclusive source interval and exact output duration, rather than
        # treating the difference between centres as the audio boundary rate.
        source_duration=(float(frame_map[-1])-float(frame_map[0])+1)/info.fps
        return dict(policy='mapped',method='atempo',reason='固定倍速保留音调',
                    speed=source_duration/duration,start=(track.start+float(frame_map[0]))/info.fps,
                    source_duration=source_duration)
    return dict(policy='mapped',method='linear_pcm',reason='动态映射同步重采样；音调随速度改变')


def _resample_pcm(info,track,frame_map,folder,cancel):
    folder=Path(folder);source=folder/'source-audio.f32';output=folder/'mapped-audio.f32'
    rate,channels=48000,2
    source_samples=max(1,round(track.count/info.fps*rate))
    out_samples=max(1,round(len(frame_map)/info.fps*rate))
    check_disk_budget(folder,(source_samples+out_samples)*channels*4)
    run_process([find_tool('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-y',
                 '-ss',f'{track.start/info.fps:.14f}','-i',info.path,'-map','0:a:0',
                 '-t',f'{track.count/info.fps:.14f}','-vn','-ac',str(channels),'-ar',str(rate),
                 '-f','f32le',str(source)],cancel,timeout=None)
    size=source.stat().st_size
    if size<channels*4 or size%(channels*4):
        raise ValueError('原片音轨PCM不完整')
    pcm=np.memmap(source,dtype='<f4',mode='r',shape=(size//(channels*4),channels))
    times=np.arange(len(frame_map)+1,dtype=np.float64)
    positions=np.concatenate((frame_map,[min(float(track.count),float(frame_map[-1])+1.)]))
    try:
        with output.open('wb') as stream:
            for first in range(0,out_samples,65536):
                check_cancel(cancel)
                indices=np.arange(first,min(out_samples,first+65536),dtype=np.float64)
                source_position=np.interp(indices/rate*info.fps,times,positions)/info.fps*rate
                left=np.floor(source_position).astype(np.int64)
                valid=(left>=0)&(left<len(pcm))
                safe=np.clip(left,0,len(pcm)-1)
                right=np.minimum(safe+1,len(pcm)-1)
                alpha=(source_position-left)[:,None]
                samples=((1-alpha)*pcm[safe]+alpha*pcm[right]).astype('<f4')
                samples[~valid]=0
                stream.write(samples.tobytes())
    finally:
        pcm._mmap.close()
    return output


def mux_audio(info,track,frame_map,config,silent,finished,folder,cancel):
    strategy=audio_strategy(info,track,frame_map,config)
    duration=len(frame_map)/info.fps
    if strategy['method']=='none':
        Path(silent).replace(finished)
        return strategy
    if strategy['method']=='linear_pcm':
        audio=_resample_pcm(info,track,frame_map,folder,cancel)
        input_args=['-f','f32le','-ar','48000','-ac','2','-i',str(audio)]
        filters='apad'
    else:
        input_args=['-ss',f"{strategy['start']:.14f}",'-t',f"{strategy['source_duration']:.14f}",'-i',info.path]
        filters=atempo_chain(strategy['speed'])+',apad'
    run_process([find_tool('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-y',
                 '-i',str(silent),*input_args,'-map','0:v:0','-map','1:a:0',
                 '-c:v','copy','-c:a','aac','-b:a','192k','-af',filters,
                 '-t',f'{duration:.14f}','-map_metadata','0','-movflags','+faststart',str(finished)],
                cancel,timeout=None)
    return strategy
