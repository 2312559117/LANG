"""Actual-media integration of HDR imports, SDR previews and public exports."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import threading

import cv2
import numpy as np
import pytest

from lockframe.export import export_video
from lockframe.media import (find_tool,import_video,native_metadata,prepare_precision_video,
                             probe,preview_tonemap_filter)
from lockframe.models import Settings,TrackResult,file_fingerprint
from lockframe.original_preview import read_original_frames
from lockframe.retime_frames import run_process
from lockframe.time_remap_model import default_config


def _streams(path):
    return json.loads(run_process([find_tool('ffprobe'),'-v','error','-count_frames',
        '-show_streams','-of','json',str(path)],threading.Event()))['streams']


def _track(info):
    count=info.frames
    return TrackResult(0,count-1,np.tile([.5,.5],(count,1)),np.tile([.25,.25,.5,.5],(count,1)),
        np.ones(count),['manual']+['tracked']*(count-1),anchors={0:(.25,.25,.5,.5)})


def _settings():
    return Settings(acceleration='cpu',resolution=0,strength=0,max_zoom=1,compute_threads=2)


@pytest.fixture(scope='module')
def source_media(tmp_path_factory):
    folder=tmp_path_factory.mktemp('retime-media-source')
    result={}
    for hdr in (False,True):
        path=folder/('HDR实测.mp4' if hdr else '普通实测.mp4')
        args=[find_tool('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=s=96x64:r=12:d=0.5',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=0.5',
            '-map','0:v','-map','1:a','-c:a','aac','-shortest']
        if hdr:
            args+=['-pix_fmt','yuv420p10le','-c:v','libx265','-preset','ultrafast','-x265-params',
                'pools=2:frame-threads=2:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:hdr10=1:'
                'master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,400',
                '-color_primaries','bt2020','-color_trc','smpte2084','-colorspace','bt2020nc']
        else:args+=['-pix_fmt','yuv420p','-c:v','libx264','-preset','ultrafast']
        run_process([*args,str(path)],threading.Event())
        result[hdr]=path
    return result


@pytest.fixture
def media_env(tmp_path,monkeypatch):
    monkeypatch.setenv('LOCKFRAME_CACHE_DIR',str(tmp_path/'cache'))
    monkeypatch.setenv('LOCKFRAME_RETIME_TEMP',str(tmp_path/'retime-temp'))
    return tmp_path


@pytest.mark.parametrize('pixel_format,reported,expected',[
    ('p010le','0',10),('yuv420p10le','0',10),('yuv444p12le','0',12),
    ('p016le','0',16),('yuv420p','0',8),('yuv420p10le','10',10)])
def test_raw_sample_zero_does_not_hide_native_precision(pixel_format,reported,expected):
    result=native_metadata(dict(pix_fmt=pixel_format,bits_per_raw_sample=reported))
    assert result['pixel_format']==pixel_format and result['bit_depth']==expected


def test_native_static_hdr_metadata_has_exact_codec_units():
    result=native_metadata(dict(pix_fmt='yuv420p10le',side_data_list=[
        dict(side_data_type='Mastering display metadata',green_x='13250/50000',green_y='34500/50000',
            blue_x='7500/50000',blue_y='3000/50000',red_x='34000/50000',red_y='16000/50000',
            white_point_x='15635/50000',white_point_y='16450/50000',
            max_luminance='1000/1',min_luminance='1/200'),
        dict(side_data_type='Content light level metadata',max_content=1000,max_average=400)]))
    assert result['mastering_display']=='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50)'
    assert result['content_light']=='1000,400'


def test_real_hdr_probe_preserves_static_metadata_from_native_stream(source_media):
    info=probe(str(source_media[True]),threading.Event())
    assert info.hdr and info.bit_depth==10 and info.pixel_format=='yuv420p10le'
    assert (info.color_primaries,info.color_transfer,info.color_space)==('bt2020','smpte2084','bt2020nc')
    assert info.mastering_display.endswith('L(10000000,50)')
    assert info.content_light=='1000,400'


def test_real_hdr_import_precision_and_original_preview_share_sdr_display(source_media,media_env):
    source=source_media[True];before=file_fingerprint(source)
    progress=[]
    info=import_video(str(source),threading.Event(),progress.append)
    precise=prepare_precision_video(info,threading.Event())
    assert info.frames==precise.frames==6 and progress[-1]==1.
    assert precise.hdr and precise.bit_depth==10 and precise.path==str(source)
    assert info.proxy!=precise.proxy and 'zscale' in preview_tonemap_filter(info)
    for proxy in (info.proxy,precise.proxy):
        stream=next(s for s in _streams(proxy) if s['codec_type']=='video')
        assert stream['pix_fmt']=='yuv420p' and stream['nb_read_frames']=='6'
        assert stream.get('color_transfer') not in ('smpte2084','arib-std-b67')
    original=read_original_frames(info,[0,3,5],threading.Event())
    capture=cv2.VideoCapture(info.proxy)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES,3);ok,proxy=capture.read()
    finally:capture.release()
    assert ok and original[3].shape==proxy.shape==(64,96,3)
    assert float(np.mean(np.abs(original[3].astype(float)-proxy.astype(float))))<4
    assert file_fingerprint(source)==before


def test_sdr_precision_cache_name_remains_baseline_and_reuses_valid_cache(source_media,media_env,monkeypatch):
    info=import_video(str(source_media[False]),threading.Event())
    prepared=prepare_precision_video(info,threading.Event())
    key=f'{info.path}{info.fingerprint}precision-v1:{info.fps:.10f}:{info.frames}:96x64'
    assert Path(prepared.proxy).name=='precision-'+hashlib.sha256(key.encode()).hexdigest()[:24]+'.mp4'
    def forbidden(*args,**kwargs):raise AssertionError('valid baseline precision cache was regenerated')
    monkeypatch.setattr('lockframe.media.run_ffmpeg',forbidden)
    assert prepare_precision_video(info,threading.Event()).proxy==prepared.proxy
    assert not info.hdr and preview_tonemap_filter(info)==''


def test_public_disabled_hdr_export_keeps_original_audio_and_native_color(source_media,media_env):
    info=import_video(str(source_media[True]),threading.Event())
    report=export_video(info,_track(info),_settings(),str(media_env/'原速HDR.mp4'),threading.Event())
    streams=_streams(report['path']);video=next(s for s in streams if s['codec_type']=='video')
    assert report['audio_policy']=='original' and report['audio_method']=='atempo'
    assert report['frames']==6 and report['duration']==.5
    assert any(s['codec_type']=='audio' for s in streams)
    assert video['pix_fmt']=='yuv420p10le' and video['color_transfer']=='smpte2084'


@pytest.mark.parametrize('hdr',[False,True])
def test_public_enabled_mapped_export_runs_new_dispatch_and_audio_strategy(source_media,media_env,hdr):
    info=import_video(str(source_media[hdr]),threading.Event())
    settings=_settings();settings.retime=default_config()
    settings.retime.update(enabled=True,mode='advanced',audio_policy='mapped',points=[
        dict(position=0.,source=0.,interpolation='linear',incoming_interpolation='linear'),
        dict(position=.5,source=1.,interpolation='linear',incoming_interpolation='linear')])
    report=export_video(info,_track(info),settings,str(media_env/f'映射{hdr}.mp4'),threading.Event())
    assert report['frames']==3 and report['duration']==.25
    assert report['audio_policy']=='mapped' and report['audio_method']=='atempo'
    assert any(s['codec_type']=='audio' for s in _streams(report['path']))


def test_retired_fields_cannot_activate_retime_or_change_old_sdr_audio(source_media,media_env,monkeypatch):
    info=import_video(str(source_media[False]),threading.Event())
    settings=Settings.from_dict(dict(asdict(_settings()),time_remap={'enabled':True,'speed':100},
        frame_interpolation={'backend':'rife','enabled':True}))
    assert settings.retime=={} and 'time_remap' not in asdict(settings)
    def forbidden(*args,**kwargs):raise AssertionError('retired fields activated new renderer')
    monkeypatch.setattr('lockframe.retime_renderer.render_retimed_video',forbidden)
    output=media_env/'旧工程原声.mp4'
    export_video(info,_track(info),settings,str(output),threading.Event())
    streams=_streams(output);video=next(s for s in streams if s['codec_type']=='video')
    assert video['nb_read_frames']=='6' and abs(float(video['duration'])-.5)<1/12
    assert any(s['codec_type']=='audio' for s in streams)
