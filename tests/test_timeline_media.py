import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from dataclasses import replace
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lockframe.media import Cancelled,NO_WINDOW,find_tool,probe
from lockframe.timeline_media import (MediaOverview,MediaSource,TimelineMedia,
    WorkCancel,build_overview,make_request,read_peaks)


@pytest.fixture(scope='module')
def media_clip(tmp_path_factory):
    directory=tmp_path_factory.mktemp('真实媒体概览')
    path=directory/'颜色与强弱音频.mkv'
    command=[find_tool('ffmpeg'),'-v','error','-nostdin','-y',
             '-f','lavfi','-i','testsrc2=size=96x64:rate=10:duration=2',
             '-f','lavfi','-i',r'aevalsrc=if(lt(t\,1)\,0.1\,0.8)*sin(2*PI*220*t):s=48000:d=2',
             '-c:v','ffv1','-c:a','pcm_s16le',str(path)]
    subprocess.run(command,check=True,capture_output=True,creationflags=NO_WINDOW,timeout=30)
    return probe(str(path))


def pump_until(predicate,timeout=5):
    app=QApplication.instance() or QApplication([])
    deadline=time.monotonic()+timeout
    while not predicate() and time.monotonic()<deadline:
        app.processEvents();time.sleep(.005)
    assert predicate()


def test_real_source_time_film_and_bounded_streamed_audio(media_clip):
    source=MediaSource.from_info(media_clip)
    request=make_request(source,0,20,count=200,bins=20000)
    assert len(request.film_frames)==16 and request.bins==4096
    result=build_overview(request,WorkCancel())
    assert not result.video_error and not result.audio_error
    assert len(result.images)==16 and result.peaks.shape==(4096,)
    assert all(image.dtype==np.uint8 and image.shape[2]==3 and image.shape[0]<=80 and image.shape[1]<=160 for image in result.images)
    assert not np.array_equal(result.images[0],result.images[-1])
    quiet=read_peaks(make_request(source,0,10,bins=100),WorkCancel())
    loud=read_peaks(make_request(source,10,20,bins=100),WorkCancel())
    assert np.median(loud)>np.median(quiet)*5
    assert 0<quiet.max()<.15 and .65<loud.max()<=1
    no_audio=build_overview(make_request(replace(source,has_audio=False),0,10),WorkCancel())
    assert no_audio.peaks.size==0 and not no_audio.has_audio and not no_audio.audio_error


def test_request_range_and_source_identity_invalidate(media_clip,tmp_path):
    first=MediaSource.from_info(media_clip)
    assert first==MediaSource.from_info(media_clip)
    assert make_request(first,0,10).key!=make_request(first,1,11).key
    assert make_request(first,0,10,count=2).key!=make_request(first,0,10,count=3).key
    copy=tmp_path/'another.mkv';copy.write_bytes(Path(media_clip.path).read_bytes())
    second=MediaSource.from_info(replace(media_clip,path=str(copy)))
    assert first.identity!=second.identity
    stat=copy.stat();os.utime(copy,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000000))
    assert second.identity!=MediaSource.from_info(replace(media_clip,path=str(copy))).identity


def test_latest_wins_cancel_busy_cache_and_close(media_clip,tmp_path,monkeypatch):
    import lockframe.timeline_media as module
    app=QApplication.instance() or QApplication([])
    source=MediaSource.from_info(media_clip);calls=[];entered=threading.Event()
    def build(request,cancel):
        calls.append(request.key);entered.set()
        for _ in range(40):
            cancel.check();time.sleep(.002)
        return MediaOverview(request.key,source.identity,request.start,request.end,
                             request.film_frames,(np.zeros((8,8,3),np.uint8),),
                             np.zeros(request.bins,np.float32),True)
    monkeypatch.setattr(module,'build_overview',build)
    controller=TimelineMedia(directory=tmp_path);received=[]
    controller.ready.connect(received.append)
    first=make_request(source,0,10,count=1);second=make_request(source,10,20,count=1)
    try:
        controller.request(first);assert entered.wait(2)
        controller.request(second)
        pump_until(lambda:len(received)==1)
        assert received[0].key==second.key and not any(r.key==first.key for r in received)
        controller.clear();controller.request(second)
        pump_until(lambda:len(received)==2)
        assert calls.count(second.key)==1  # cached, not a new decoder run
        controller.set_busy(True);controller.clear();controller.request(first)
        for _ in range(10):app.processEvents();time.sleep(.005)
        assert calls.count(first.key)==1 and len(received)==2
        controller.set_busy(False);pump_until(lambda:len(received)==3)
        assert received[-1].key==first.key
    finally:controller.close()
    assert not controller._thread.is_alive()
    assert not list(tmp_path.glob('*.partial'))


def test_cancelled_work_produces_no_audio_or_fake_complete_result(media_clip):
    token=WorkCancel();token.cancel()
    with pytest.raises(Cancelled):build_overview(make_request(MediaSource.from_info(media_clip),0,10),token)


def test_quick_playback_pause_retries_cancelled_overview(media_clip, tmp_path, monkeypatch):
    from lockframe import timeline_media

    app = QApplication.instance() or QApplication([])
    source = MediaSource.from_info(media_clip)
    request = make_request(source, 0, 10, count=1, include_audio=False)
    entered, release = threading.Event(), threading.Event()
    attempts = []

    def build(current, token):
        attempts.append(current.key)
        if len(attempts) == 1:
            entered.set()
            assert release.wait(3)
        token.check()
        return MediaOverview(current.key, source.identity, 0, 10, current.film_frames,
                             (np.zeros((8, 8, 3), np.uint8),), np.empty(0), source.has_audio)

    monkeypatch.setattr(timeline_media, 'build_overview', build)
    service = TimelineMedia(directory=tmp_path)
    received = []
    service.ready.connect(received.append)
    try:
        service.request(request)
        assert entered.wait(2)
        service.set_busy(True)
        service.set_busy(False)
        release.set()
        pump_until(lambda: bool(received), timeout=2)
        assert attempts == [request.key, request.key]
        assert received[0].key == request.key
    finally:
        release.set()
        service.close()


def test_overview_cache_hits_do_not_rewrite_disk(media_clip, tmp_path):
    app = QApplication.instance() or QApplication([])
    source = MediaSource.from_info(media_clip)
    request = make_request(source, 0, 10, count=1, include_audio=False)
    path = tmp_path / (request.key + '.npz')
    stamp = 1_700_000_000_000_000_000

    def receive(service):
        received = []
        service.ready.connect(received.append)
        service.request(request)
        pump_until(lambda: bool(received))
        service.ready.disconnect(received.append)
        return received[0]

    service = TimelineMedia(directory=tmp_path)
    try:
        first = receive(service)
        assert first.images and path.is_file()
        os.utime(path, ns=(stamp, stamp))
        service.clear()
        assert receive(service).images
        assert path.stat().st_mtime_ns == stamp
    finally:
        service.close()

    service = TimelineMedia(directory=tmp_path)
    try:
        assert receive(service).images
        assert path.stat().st_mtime_ns == stamp
    finally:
        service.close()


def test_missing_source_is_not_replaced_by_placeholder_media(media_clip):
    with pytest.raises(FileNotFoundError):MediaSource.from_info(replace(media_clip,path='D:/不存在的时间轴素材.mp4'))


def test_hd_cache_obeys_byte_budget_not_just_entry_count(media_clip,tmp_path):
    app=QApplication.instance() or QApplication([])
    source=MediaSource.from_info(media_clip);service=TimelineMedia(directory=tmp_path)
    try:
        keys=[]
        for index in range(10):
            request=make_request(source,0,20,count=16,bins=100+index,pixel_size=(512,256));keys.append(request.key)
            service._save(MediaOverview(request.key,source.identity,0,20,request.film_frames,
                tuple(np.zeros((256,512,3),np.uint8) for _ in range(16)),np.zeros(100+index,np.float32),True))
        size=sum(sum(a.nbytes for a in r.images)+r.peaks.nbytes for r in service._memory.values())
        assert size<=32*1024*1024 and len(service._memory)<=8
        assert keys[0] not in service._memory and keys[-1] in service._memory
    finally:service.close()


def test_corrupt_cache_recovers_and_decoder_errors_are_explicit(media_clip,tmp_path,monkeypatch):
    import lockframe.timeline_media as module
    app=QApplication.instance() or QApplication([])
    request=make_request(MediaSource.from_info(media_clip),0,10,count=1)
    (tmp_path/(request.key+'.npz')).write_bytes(b'broken')
    monkeypatch.setattr(module,'read_film',lambda *a:(_ for _ in ()).throw(ValueError('读取失败')))
    monkeypatch.setattr(module,'read_peaks',lambda *a:(_ for _ in ()).throw(ValueError('音频失败')))
    controller=TimelineMedia(directory=tmp_path);received=[];controller.ready.connect(received.append)
    try:
        controller.request(request);pump_until(lambda:bool(received))
        assert received[0].images==() and received[0].peaks.size==0
        assert received[0].video_error=='读取失败' and received[0].audio_error=='音频失败'
    finally:controller.close()
