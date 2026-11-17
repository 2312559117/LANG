"""Native-YUV time remapping, spatial lock, optical-flow synthesis and encoding.

Spatial ordering is an explicit product design: each measured adjacent source
frame is locked first, then flow is computed in the common output coordinates.
The recovered evidence does not establish that MS uses the same ordering.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import os
import re
import subprocess
import tempfile
import uuid
import time

import cv2
import numpy as np

from .encoding import HardwareEncodingError,encoder_options,select_encoder
from .media import NO_WINDOW,check_cancel,find_tool
from .models import file_fingerprint
from .stabilization import build_plan
from .retime_frames import NativeFrameStore,cancel_guard,canonical_format,fps_text,probe_native,temp_root
from .retime_audio import mux_audio
from .logging import get_logger
from .retime_store import open_frame_store

log=get_logger('retime_renderer')


def validate_frame_map(frame_map,source_count):
    values=np.asarray(frame_map,dtype=np.float64)
    if values.ndim!=1 or not len(values) or len(values)>10_000_000:
        raise ValueError('变速frame-map必须是非空的一维映射，且不超过一千万输出帧')
    bad=np.flatnonzero(~np.isfinite(values))
    if len(bad):
        raise ValueError(f'变速frame-map第 {int(bad[0])+1} 行不是有限数值')
    bad=np.flatnonzero((values<0)|(values>source_count-1))
    if len(bad):
        raise ValueError(f'变速frame-map第 {int(bad[0])+1} 行超出源片段')
    result=values.copy();result.flags.writeable=False
    return result


def transform_frame(frame,matrix,width,height):
    """Apply the same measured source transform to luma and chroma centres."""
    from .flow.frame import YuvFrame
    transform=np.eye(3,dtype=np.float64)
    transform[:2]=np.asarray(matrix,dtype=np.float64)
    if not np.isfinite(transform).all():
        raise ValueError('构图矩阵存在无效数值')
    shapes=frame.format.plane_shapes(width,height)
    planes=[]
    max_value=(1<<frame.format.depth)-1
    for index,(plane,shape) in enumerate(zip(frame.planes,shapes)):
        sx,sy=(1,1) if index==0 else (frame.format.sub_x,frame.format.sub_y)
        centres=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]],dtype=np.float64)
        mapped=(np.linalg.inv(centres)@transform@centres)[:2]
        # Float sampling followed by explicit native-depth quantization avoids
        # unsigned-16 clipping at 65535 instead of e.g. the 10-bit maximum 1023.
        shift=frame.format.shift
        source=plane.astype(np.float32)/(1<<shift)
        warped=cv2.warpAffine(source,mapped,(shape[1],shape[0]),flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
        quantized=np.rint(np.clip(warped,0,max_value)).astype(frame.format.dtype)
        if shift:quantized=quantized<<shift
        planes.append(quantized)
    return YuvFrame(tuple(planes),frame.format)


def _color_value(info,metadata,field):
    value=metadata.get(field)
    if value in (None,'','unknown','unspecified','reserved'):
        value=getattr(info,field,'unknown')
    if value in (None,'','unknown','unspecified','reserved'):return None
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}',str(value)):
        raise ValueError('源片色彩元数据格式无效')
    return str(value)


def _metadata_args(info,metadata):
    arguments=[]
    for field,flag in (('color_primaries','-color_primaries'),('color_transfer','-color_trc'),
                       ('color_space','-colorspace'),('color_range','-color_range')):
        value=_color_value(info,metadata,field)
        if value is not None:arguments.extend([flag,value])
    return arguments


def _x265_metadata(info,metadata):
    values=['pools=2','frame-threads=2','repeat-headers=1']
    # Explicit codec VUI as well as container/AVCodec flags. Some FFmpeg/x265
    # builds omit primaries/transfer when only AVCodec flags are provided.
    for field,option in (('color_primaries','colorprim'),('color_transfer','transfer'),('color_space','colormatrix')):
        value=_color_value(info,metadata,field)
        if value is not None:values.append(option+'='+value)
    mastering=getattr(info,'mastering_display','') or ''
    light=getattr(info,'content_light','') or ''
    if mastering:
        if not re.fullmatch(r'G\(\d{1,10},\d{1,10}\)B\(\d{1,10},\d{1,10}\)R\(\d{1,10},\d{1,10}\)WP\(\d{1,10},\d{1,10}\)L\(\d{1,12},\d{1,12}\)',mastering):
            raise ValueError('HDR mastering display 元数据格式无效')
        values.append('master-display='+mastering)
    if light:
        if not re.fullmatch(r'\d{1,7},\d{1,7}',light):
            raise ValueError('HDR content light 元数据格式无效')
        values.append('max-cll='+light)
    if getattr(info,'hdr',False):
        values.append('hdr10=1')
    return ':'.join(values)


def _scene_cut(left,right):
    maximum=(1<<left.format.depth)-1
    a=cv2.resize(left.planes[0].astype(np.float32)/(maximum*(1<<left.format.shift)),(32,32))
    b=cv2.resize(right.planes[0].astype(np.float32)/(maximum*(1<<right.format.shift)),(32,32))
    # Conservative rejection, rather than a promise of comprehensive scene-cut
    # detection. Native flow/vector validation remains an additional gate.
    return float(np.mean(np.abs(a-b)))>.45


def _render_attempt(info,track,settings,frame_map,store,plan,silent,cancel,progress,encoder,metadata):
    from .flow.base import create_interpolator
    config=getattr(settings,'retime',{}) or {}
    flow_enabled=bool(config.get('enabled') and config.get('optical_flow'))
    allow_fallback=bool(config.get('allow_frame_fallback',False))
    flow=create_interpolator(config.get('flow_backend','auto'),allow_fallback=allow_fallback,cancel=cancel) if flow_enabled else None
    depth=store.format.depth
    hdr=bool(getattr(info,'hdr',False))
    if hdr and depth<10:
        if flow:flow.close()
        raise ValueError('HDR源片低于10位，当前不能保证HDR输出；未静默编码为8位HDR')
    if depth not in (8,10,12):
        if flow:flow.close()
        raise ValueError(f'当前MP4编码不能无损保留 {depth} 位采样，请使用8/10/12位源片；未自动降位')
    pixel_format=store.format.canonical_name
    if encoder=='libx265':
        options=['-c:v','libx265','-preset','fast','-crf',str(settings.crf),
                 '-x265-params',_x265_metadata(info,metadata),'-tag:v','hvc1']
    else:options=encoder_options(encoder,settings.crf)
    warped=OrderedDict();fallbacks=[]
    timing=dict(warp_seconds=0.,synthesis_seconds=0.,encode_write_seconds=0.)
    boundaries={int(s['start'])-track.start for s in track.performance.get('tracking_sections',[])[1:]}
    def read(index):
        if index in warped:
            warped.move_to_end(index)
            return warped[index]
        original=store.read(index);began=time.perf_counter()
        frame=transform_frame(original,plan.matrices[index],plan.width,plan.height)
        timing['warp_seconds']+=time.perf_counter()-began
        warped[index]=frame
        while len(warped)>2:warped.popitem(last=False)
        return frame
    encode_log=silent.with_suffix('.encode.log')
    process=None
    stats={}
    status={}
    try:
        with encode_log.open('wb') as errors:
            process=subprocess.Popen([find_tool('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-y',
                '-f','rawvideo','-pix_fmt',pixel_format,'-s',f'{plan.width}x{plan.height}',
                '-r',fps_text(info.fps),'-i','pipe:0','-an',*options,*_metadata_args(info,metadata),
                '-pix_fmt',pixel_format,'-movflags','+faststart',str(silent)],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors,creationflags=NO_WINDOW)
            with cancel_guard(process,cancel):
                for output_index,position in enumerate(frame_map):
                    check_cancel(cancel)
                    left_index=int(np.floor(position));right_index=min(left_index+1,track.count-1)
                    alpha=float(position-left_index)
                    if not flow_enabled or alpha<1e-10 or left_index==right_index:
                        frame=read(min(track.count-1,int(np.floor(position+.5))))
                    else:
                        left,right=read(left_index),read(right_index)
                        reason='跨独立分段边界，不能合成连续运动' if right_index in boundaries else '相邻帧疑似切镜，停止光流合成' if _scene_cut(left,right) else ''
                        if reason:
                            if not allow_fallback:
                                raise ValueError(reason+'；关闭光流或明确允许最近帧回退后再导出')
                            if reason not in fallbacks:fallbacks.append(reason)
                            frame=left if alpha<.5 else right
                        else:
                            pair=flow.prepare(left,right,key=(left_index,right_index))
                            began=time.perf_counter();frame=pair.compose(alpha)
                            timing['synthesis_seconds']+=time.perf_counter()-began
                    began=time.perf_counter()
                    try:process.stdin.write(frame.to_bytes())
                    except (BrokenPipeError,OSError) as error:
                        check_cancel(cancel)
                        raise HardwareEncodingError('变速视频编码器提前退出，请检查输出格式、磁盘空间和驱动') from error
                    timing['encode_write_seconds']+=time.perf_counter()-began
                    if progress and (output_index%4==0 or output_index==len(frame_map)-1):
                        backend=flow.backend_name if flow else 'nearest'
                        progress(.12+.73*(output_index+1)/len(frame_map),
                                 f'变速渲染 {output_index+1}/{len(frame_map)} · {backend} · {encoder}')
                process.stdin.close()
                while process.poll() is None:
                    check_cancel(cancel);cancel.wait(.1)
                check_cancel(cancel)
                if process.returncode:
                    detail=encode_log.read_text(encoding='utf-8',errors='replace')[-1000:]
                    raise HardwareEncodingError('变速编码失败：'+detail)
        stats=dict(flow.stats) if flow else {}
        status=dict(flow.status) if flow else dict(backend='nearest',requested_backend='off',fallback_reason='')
        if status.get('fallback_reason'):
            fallbacks.insert(0,status['fallback_reason'])
        return dict(actual_backend=flow.backend_name if flow else 'nearest',flow_pairs=stats.get('prepared_pairs',0),
                    flow_statistics=stats,flow_status=status,fallbacks=fallbacks,encoder=encoder,
                    pixel_format=pixel_format,bit_depth=depth,hdr=hdr,stage_timing=timing)
    finally:
        if process is not None:
            if process.poll() is None:process.kill()
            process.wait()
            if process.stdin and not process.stdin.closed:
                try:process.stdin.close()
                except (BrokenPipeError,OSError):pass
        if flow:flow.close()


def _commit(finished,target,overwrite,cancel):
    """Stage beside destination; only a fully copied movie can become a product."""
    target=Path(target);staging=target.with_name(f'.{target.stem}.{uuid.uuid4().hex}.partial.mp4')
    try:
        with Path(finished).open('rb') as source,staging.open('xb') as output:
            while True:
                check_cancel(cancel);chunk=source.read(1024*1024)
                if not chunk:break
                output.write(chunk)
            output.flush();os.fsync(output.fileno())
        check_cancel(cancel)
        if overwrite:os.replace(staging,target)
        elif os.name=='nt':os.rename(staging,target)
        else:os.link(staging,target);staging.unlink()
    finally:
        staging.unlink(missing_ok=True)


def _validate_product(path,info,metadata,report,frame_count,cancel):
    """A zero encoder exit code alone cannot commit an incomplete product."""
    from .flow.frame import parse_format
    actual=probe_native(path,cancel)
    if int(actual.get('nb_frames',-1))!=frame_count:
        raise ValueError('导出文件帧数不完整，未提交成品')
    if (int(actual.get('width',0)),int(actual.get('height',0)))!=(report['width'],report['height']):
        raise ValueError('导出画幅与渲染计划不一致，未提交成品')
    if abs(float(actual.get('duration',0))-frame_count/info.fps)>1/info.fps+.001:
        raise ValueError('导出时长与时间映射不一致，未提交成品')
    actual_depth=parse_format(canonical_format(actual.get('pix_fmt',''))).depth
    if actual_depth!=report['bit_depth']:
        raise ValueError('编码器改变了原生采样位深，未提交成品')
    if report['hdr']:
        for field in ('color_primaries','color_transfer','color_space'):
            expected=_color_value(info,metadata,field)
            if expected is not None and actual.get(field)!=expected:
                raise ValueError('编码器未保留HDR色彩元数据，未提交成品：'+field)


def render_retimed_video(info,track,settings,frame_map,destination,cancel,progress=None,*,overwrite=True):
    total_began=time.perf_counter()
    check_cancel(cancel)
    settings.validate();track.validate(info.frames)
    from .tracking_pause import is_complete
    if not is_complete(track) or track.lost_frames:
        raise ValueError('跟踪未完成或存在真实跟丢，请修正后再导出')
    target=Path(destination).resolve()
    if target.suffix.lower()!='.mp4':raise ValueError('请导出为 .mp4 文件')
    protected={Path(info.path).resolve()}
    if info.proxy:protected.add(Path(info.proxy).resolve())
    if target in protected:raise ValueError('输出文件不能覆盖原视频或分析预览')
    if target.exists() and not overwrite:raise FileExistsError('输出文件已存在，重新入队将自动编号')
    if file_fingerprint(info.path)!=info.fingerprint:raise ValueError('原视频已变化，请重新导入和跟踪')
    values=validate_frame_map(frame_map,track.count)
    metadata=probe_native(info.path,cancel)
    for side in metadata.get('side_data_list',[]):
        kind=str(side.get('side_data_type','')).lower()
        if any(term in kind for term in ('dovi','dolby vision','smpte2094','hdr10+')):
            raise ValueError('当前原生变速尚不能保留动态HDR/Dolby Vision元数据，未静默降级')
    plan=build_plan(info,track,settings)
    target.parent.mkdir(parents=True,exist_ok=True)
    config=getattr(settings,'retime',{}) or {}
    log.info('开始原生变速 源帧=%d 输出帧=%d 光流=%s',track.count,len(values),bool(config.get('optical_flow')))
    with tempfile.TemporaryDirectory(prefix='render-',dir=temp_root()) as directory:
        folder=Path(directory)
        silent,finished=folder/'video.mp4',folder/'finished.mp4'
        with open_frame_store(info,track,values,folder,cancel,
                (lambda value,text:progress(.12*value,text)) if progress else None,metadata=metadata) as store:
            native_high=store.format.depth>8 or bool(getattr(info,'hdr',False))
            encoder='libx265' if native_high else select_encoder(settings.acceleration,cancel)
            try:
                report=_render_attempt(info,track,settings,values,store,plan,silent,cancel,progress,encoder,metadata)
            except HardwareEncodingError:
                if encoder in ('libx264','libx265'):raise
                if progress:progress(.12,'编码器不可用，保持时间映射与光流设置并改用CPU编码')
                if hasattr(store,'rewind'):store.rewind()
                report=_render_attempt(info,track,settings,values,store,plan,silent,cancel,progress,'libx264',metadata)
                report['hardware_fallback']=True
            report['source_store_bytes']=store.bytes_written
            report['source_cache_strategy']=store.strategy
            report['source_cache_peak_bytes']=0 if store.strategy=='stream' else store.peak_cache_bytes
            report['source_window_peak_bytes']=store.peak_cache_bytes if store.strategy=='stream' else 0
            report['source_frames_delivered']=store.decoded_frames
            report['stage_timing']['decode_seconds']=store.decode_seconds
        check_cancel(cancel)
        if progress:progress(.88,'按时间映射处理音频')
        began=time.perf_counter();audio=mux_audio(info,track,values,config,silent,finished,folder,cancel)
        report['stage_timing']['audio_seconds']=time.perf_counter()-began
        report.update(width=plan.width,height=plan.height)
        _validate_product(finished,info,metadata,report,len(values),cancel)
        if file_fingerprint(info.path)!=info.fingerprint:raise ValueError('原视频在导出时发生变化，未提交成品')
        _commit(finished,target,overwrite,cancel)
    report.update(path=str(target),frames=len(values),duration=len(values)/info.fps,
                  width=plan.width,height=plan.height,zoom=plan.zoom,limited_frames=plan.limited_frames,
                  audio_policy=audio['policy'],audio_reason=audio['reason'],audio_method=audio['method'],
                  source_timebase='existing analysis CFR; original VFR PTS not reconstructed',
                  spatial_order='warp measured neighbours separately, then synthesize; product design')
    if progress:progress(1.,'变速导出完成')
    report['stage_timing']['total_seconds']=time.perf_counter()-total_began
    report['stage_timing']['flow_prepare_seconds']=report['flow_statistics'].get('flow_seconds',0.)
    log.info('原生渲染性能 缓存=%s 原始帧磁盘字节=%d 峰值磁盘字节=%d 交付源帧=%d 阶段=%s',
        report['source_cache_strategy'],report['source_store_bytes'],report['source_cache_peak_bytes'],
        report['source_frames_delivered'],report['stage_timing'])
    log.info('原生变速完成 帧=%d 后端=%s 编码=%s 音频=%s',len(values),report['actual_backend'],report['encoder'],report['audio_policy'])
    return report
