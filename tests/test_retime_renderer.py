from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import threading

import numpy as np
import pytest

from lockframe.models import VideoInfo,Settings,TrackResult,file_fingerprint
from lockframe.media import find_tool,NO_WINDOW,Cancelled
from lockframe.retime_frames import NativeFrameStore,check_disk_budget,run_process,read_native_frame,native_preview_bgr
from lockframe.retime_renderer import render_retimed_video,validate_frame_map,transform_frame
from lockframe.retime_audio import audio_strategy,atempo_chain


def streams(path):
    return json.loads(subprocess.check_output([find_tool('ffprobe'),'-v','error','-count_frames',
        '-show_streams','-of','json',str(path)],creationflags=NO_WINDOW))['streams']


def tracked(info):
    n=info.frames
    return TrackResult(0,n-1,np.tile([.5,.5],(n,1)),np.tile([.25,.25,.5,.5],(n,1)),
                       np.ones(n),['manual']+['tracked']*(n-1),anchors={0:(.25,.25,.5,.5)})


def config(**kwargs):
    value=dict(version=2,enabled=True,mode='basic',optical_flow=False,flow_backend='dis',
               allow_frame_fallback=False,audio_policy='mapped')
    value.update(kwargs)
    return value


def settings(**kwargs):
    value=Settings(acceleration='cpu',resolution=0,aspect='original',strength=0,max_zoom=1,compute_threads=2)
    value.retime=config(**kwargs)
    return value


@pytest.fixture
def clip(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCKFRAME_RETIME_TEMP',str(tmp_path/'retime-temp'))
    path=tmp_path/'原片8位.mkv'
    run_process([find_tool('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=s=96x64:r=12:d=1',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1','-map','0:v','-map','1:a',
        '-c:v','ffv1','-pix_fmt','yuv420p','-c:a','pcm_s16le','-shortest',str(path)],threading.Event())
    info=VideoInfo(str(path),96,64,12.,1.,True,12,str(path),file_fingerprint(path))
    return info,tracked(info),tmp_path


def test_native_store_random_access_is_exact_and_releases_file(clip):
    info,track,folder=clip
    with NativeFrameStore.open(info,track,folder/'store',threading.Event()) as store:
        a=store.read(0).to_bytes();z=store.read(11).to_bytes()
        assert a!=z and store.read(0).to_bytes()==a and store.read(11).to_bytes()==z
        assert store.path.stat().st_size==96*64*3//2*12
        saved=store.path
    saved.unlink()
    assert not saved.exists()


def test_native_preview_uses_same_cfr_frame_and_cancellable_conversion(clip):
    info,track,folder=clip
    frame=read_native_frame(info,8,threading.Event())
    with NativeFrameStore.open(info,track,folder/'preview-store',threading.Event()) as store:
        assert frame.to_bytes()==store.read(8).to_bytes()
    bgr=native_preview_bgr(frame,info,threading.Event())
    assert bgr.shape==(64,96,3) and bgr.dtype==np.uint8
    cancelled=threading.Event();cancelled.set()
    with pytest.raises(Cancelled):native_preview_bgr(frame,info,cancelled)


def test_native_chroma_translation_uses_same_physical_displacement():
    from lockframe.flow.frame import YuvFrame,parse_format
    # A two-luma-pixel translation is one chroma sample in 4:2:0.
    y=np.zeros((8,12),dtype='<u2');u=np.zeros((4,6),dtype='<u2');v=u.copy()
    y[2:4,2:4]=801;u[1,1]=531;v[1,1]=719
    frame=YuvFrame((y,u,v),parse_format('yuv420p10le'))
    shifted=transform_frame(frame,np.array([[1.,0,2],[0,1.,0]]),12,8)
    assert shifted.planes[0][2,4]==801
    assert shifted.planes[1][1,2]==531 and shifted.planes[2][1,2]==719
    assert shifted.planes[0].max()<=1023


def test_actual_fixed_speed_export_frames_duration_audio_and_source_protection(clip):
    info,track,folder=clip
    identity=file_fingerprint(info.path)
    output=folder/'两倍速.mp4'
    report=render_retimed_video(info,track,settings(),np.linspace(0,11,6),output,threading.Event(),overwrite=False)
    data=streams(output);video=next(v for v in data if v['codec_type']=='video')
    assert report['frames']==6 and report['duration']==.5 and video['nb_read_frames']=='6'
    assert abs(float(video['duration'])-.5)<1/12 and any(v['codec_type']=='audio' for v in data)
    assert report['audio_method']=='atempo' and report['actual_backend']=='nearest'
    assert file_fingerprint(info.path)==identity
    with pytest.raises(FileExistsError):
        render_retimed_video(info,track,settings(),np.arange(12),output,threading.Event(),overwrite=False)
    info.proxy=str(folder/'analysis.mp4');Path(info.proxy).write_bytes(b'protected proxy')
    with pytest.raises(ValueError,match='覆盖'):
        render_retimed_video(info,track,settings(),np.arange(12),info.proxy,threading.Event())
    assert Path(info.proxy).read_bytes()==b'protected proxy'


def test_actual_reverse_and_freeze_are_random_source_reads_with_explicit_mute(clip):
    info,track,folder=clip
    for name,mapping in [('倒放',np.arange(11,-1,-1,dtype=float)),('定格',np.ones(8)*5)]:
        report=render_retimed_video(info,track,settings(),mapping,folder/f'{name}.mp4',threading.Event())
        data=streams(report['path'])
        assert report['audio_policy']=='muted' and '倒放' in report['audio_reason']
        assert not any(s['codec_type']=='audio' for s in data)
        assert next(s for s in data if s['codec_type']=='video')['nb_read_frames']==str(len(mapping))


def test_actual_variable_audio_is_mapped_not_original_audio_copy(clip):
    info,track,folder=clip
    mapping=np.array([0.,.2,.5,1.,2.,4.,7.,11.])
    report=render_retimed_video(info,track,settings(),mapping,folder/'动态声音.mp4',threading.Event())
    assert report['audio_method']=='linear_pcm' and '音调' in report['audio_reason']
    assert any(v['codec_type']=='audio' for v in streams(report['path']))


def test_dynamic_pcm_hits_known_source_sample_positions(clip):
    from lockframe.retime_audio import _resample_pcm
    info,track,folder=clip
    mapping=np.array([0.,1.,4.,5.,9.,11.])
    mapped=_resample_pcm(info,track,mapping,folder,threading.Event())
    source=np.fromfile(folder/'source-audio.f32',dtype='<f4').reshape(-1,2)
    output=np.fromfile(mapped,dtype='<f4').reshape(-1,2)
    assert len(output)==24000
    # Exact knot and halfway between output knots: source frame 4 and 7.
    np.testing.assert_allclose(output[8000],source[16000],atol=1e-6)
    np.testing.assert_allclose(output[14000],source[28000],atol=1e-6)


def test_actual_optical_flow_pair_reuse_and_declared_segment_boundary_failure(clip):
    info,track,folder=clip
    report=render_retimed_video(info,track,settings(optical_flow=True),
        np.array([0.,.25,.5,.75,1.,1.5,2.]),folder/'真DIS.mp4',threading.Event())
    assert report['actual_backend']=='dis' and report['flow_pairs']==2
    assert report['flow_statistics']['cache_hits']>=2
    track.performance['tracking_sections']=[dict(start=0,end=5),dict(start=6,end=11)]
    refused=folder/'不能伪装.mp4'
    with pytest.raises(ValueError,match='分段边界'):
        render_retimed_video(info,track,settings(optical_flow=True),[5.5],refused,threading.Event())
    assert not refused.exists() and not list(folder.glob('.*.partial.mp4'))
    assert not list((folder/'retime-temp').glob('render-*'))


def test_actual_cancellation_cleans_temp_and_encoder_fallback_preserves_mapping(clip,monkeypatch):
    import lockframe.retime_renderer as module
    info,track,folder=clip
    cancel=threading.Event()
    def progress(value,text):
        if value>.3:cancel.set()
    with pytest.raises(Cancelled):
        render_retimed_video(info,track,settings(),np.linspace(0,11,24),folder/'已取消.mp4',cancel,progress)
    assert not (folder/'已取消.mp4').exists() and not list((folder/'retime-temp').glob('render-*'))
    original=module._render_attempt;calls=[]
    def attempt(info,track,settings,mapping,*args,**kwargs):
        encoder=args[-2]
        calls.append((encoder,mapping.copy()))
        if len(calls)==1:raise module.HardwareEncodingError('injected hardware failure')
        return original(info,track,settings,mapping,*args,**kwargs)
    monkeypatch.setattr(module,'select_encoder',lambda *args:'h264_nvenc')
    monkeypatch.setattr(module,'_render_attempt',attempt)
    mapping=np.linspace(0,11,5)
    report=render_retimed_video(info,track,settings(),mapping,folder/'回退.mp4',threading.Event())
    assert report['hardware_fallback'] and report['encoder']=='libx264'
    assert calls[0][0]=='h264_nvenc' and calls[1][0]=='libx264'
    np.testing.assert_array_equal(calls[0][1],calls[1][1])


def test_invalid_maps_disk_budget_and_pcm_policy_do_not_fake_success(clip,monkeypatch):
    import lockframe.retime_frames as module
    info,track,folder=clip
    for values in ([],[[0,1]],[0,float('nan')],[-1,1],[0,12]):
        with pytest.raises(ValueError):validate_frame_map(values,12)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=1024))
    with pytest.raises(OSError,match='空间不足'):check_disk_budget(folder,96*64*12)
    assert atempo_chain(8)=='atempo=2,atempo=2,atempo=2'
    assert audio_strategy(info,track,np.array([1.,1.]),config())['policy']=='muted'
    assert audio_strategy(info,track,np.array([0.,2.,4.]),config())['speed']==pytest.approx(5/3)


def test_actual_10bit_hdr_export_never_downgrades_to_rgb8(tmp_path,monkeypatch):
    from lockframe.flow.frame import YuvFrame,parse_format
    monkeypatch.setenv('LOCKFRAME_RETIME_TEMP',str(tmp_path/'temp'))
    width,height,count=96,64,6
    fmt=parse_format('yuv420p10le')
    raw=tmp_path/'tenbit.yuv'
    with raw.open('wb') as stream:
        for i in range(count):
            y=(128+np.arange(width*height).reshape(height,width)%733+i*3).astype('<u2')
            planes=(y,np.full((32,48),470+i,dtype='<u2'),np.full((32,48),550-i,dtype='<u2'))
            stream.write(YuvFrame(planes,fmt).to_bytes())
    source=tmp_path/'HDR原片.mkv'
    run_process([find_tool('ffmpeg'),'-v','error','-y','-f','rawvideo','-pix_fmt','yuv420p10le',
        '-s','96x64','-r','12','-i',str(raw),'-c:v','ffv1','-level','3',
        '-color_primaries','bt2020','-color_trc','smpte2084','-colorspace','bt2020nc',str(source)],threading.Event())
    info=VideoInfo(str(source),width,height,12.,.5,False,count,str(source),file_fingerprint(source),True)
    info.pixel_format='yuv420p10le';info.bit_depth=10
    info.color_primaries='bt2020';info.color_transfer='smpte2084';info.color_space='bt2020nc'
    info.mastering_display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    info.content_light='1000,400'
    output=tmp_path/'HDR成品.mp4'
    report=render_retimed_video(info,tracked(info),settings(optical_flow=True),[0.,.5,1.,2.,3.,4.,5.],output,threading.Event())
    stream=next(v for v in streams(output) if v['codec_type']=='video')
    assert report['bit_depth']==10 and report['encoder']=='libx265'
    assert stream['pix_fmt']=='yuv420p10le' and stream['color_transfer']=='smpte2084'
    assert stream['color_primaries']=='bt2020' and stream['color_space']=='bt2020nc'
    assert stream['nb_read_frames']=='7'
    decoded=run_process([find_tool('ffmpeg'),'-v','error','-i',str(output),'-frames:v','1',
                         '-pix_fmt','yuv420p10le','-f','rawvideo','pipe:1'],threading.Event())
    samples=np.frombuffer(decoded,dtype='<u2')
    assert samples.max()>255 and np.any(samples%4)
    frames=json.loads(run_process([find_tool('ffprobe'),'-v','error','-select_streams','v:0',
        '-read_intervals','%+#1','-show_frames','-of','json',str(output)],threading.Event()))['frames']
    side=frames[0].get('side_data_list',[])
    assert any(v['side_data_type']=='Mastering display metadata' for v in side)
    assert any(v['side_data_type']=='Content light level metadata' and v['max_content']==1000 for v in side)
    bgr=native_preview_bgr(read_native_frame(info,2,threading.Event()),info,threading.Event())
    assert bgr.dtype==np.uint8 and bgr.shape==(64,96,3)
