from dataclasses import replace
import threading
import time

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lockframe.media import Cancelled, FrameReader, import_video, run_ffmpeg
from lockframe.original_preview import OriginalPreview, read_original_frames
from lockframe.models import VideoInfo
from lockframe.preview_quality_check import check_quality
from test_workflow_v9 import window, source


def spin(predicate, timeout=6):
    application = QApplication.instance() or QApplication([])
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        application.processEvents()
        if predicate(): return
        time.sleep(.005)
    pytest.fail('Background preview did not reach expected state')


def test_real_4k_source_and_effect_with_playback_fallback(tmp_path):
    result = check_quality(tmp_path/'4K清晰度')
    assert result['original_pixel_mae']==0
    assert result['proxy_upscaled_mae']>5


def test_native_decode_follows_exact_import_cfr_clock_on_vfr_source(tmp_path):
    path=tmp_path/'变帧率.mkv'
    run_ffmpeg(['-f','lavfi','-i','testsrc2=size=160x90:rate=12', '-frames:v','8',
                '-vf','setpts=if(lt(N\\,3)\\,N/12/TB\\,(N-3)/6/TB+0.25/TB)',
                '-fps_mode','vfr','-c:v','ffv1',str(path)],threading.Event())
    info=import_video(str(path),threading.Event())
    cfr=tmp_path/'完整CFR.mkv'
    run_ffmpeg(['-i',str(path),'-vf',f'fps={info.fps:.10f},scale={info.width}:{info.height},setsar=1',
                '-c:v','ffv1','-pix_fmt','bgr0',str(cfr)],threading.Event())
    indices=(0,2,info.frames-1)
    originals=read_original_frames(info,indices,threading.Event())
    reader=FrameReader(str(cfr))
    try:
        for index in indices: assert np.array_equal(originals[index],reader.read(index))
    finally:reader.close()
    changed=replace(info,fingerprint='wrong')
    with pytest.raises(ValueError,match='变化'):read_original_frames(changed,(0,),threading.Event())
    cancel=threading.Event();cancel.set()
    with pytest.raises(Cancelled):read_original_frames(info,(0,),cancel)


def test_latest_request_cancels_old_and_cannot_paint_foreign_clip():
    application=QApplication.instance() or QApplication([])
    started=threading.Event();cancelled=threading.Event();calls=[];received=[]
    def decode(info,indices,cancel):
        calls.append(info.path)
        if info.path=='old':
            started.set()
            assert cancel.wait(3)
            cancelled.set()
            raise Cancelled()
        return {i:np.full((2,2,3),7,np.uint8) for i in indices}
    preview=OriginalPreview(decoder=decode)
    preview.ready.connect(received.append)
    info=VideoInfo('old',2,2,12,1,frames=12)
    try:
        preview.frames(info,(0,));spin(started.is_set)
        preview.frames(replace(info,path='new'),(3,));spin(lambda:bool(received))
        assert cancelled.is_set() and len(received)==1 and received[0][0][0]=='new'
        assert calls==['old','new']
        assert preview.frames(replace(info,path='new'),(3,))[3][0,0,0]==7
    finally:preview.close()


def test_clear_then_request_same_running_frame_restarts_instead_of_losing_request():
    application=QApplication.instance() or QApplication([])
    started=threading.Event();calls=[];received=[]
    def decode(info,indices,cancel):
        calls.append(1)
        if len(calls)==1:
            started.set();assert cancel.wait(3);raise Cancelled()
        return {0:np.zeros((2,2,3),np.uint8)}
    preview=OriginalPreview(decoder=decode)
    preview.ready.connect(received.append)
    info=VideoInfo('same',2,2,12,1,frames=12)
    try:
        preview.frames(info,(0,));spin(started.is_set)
        preview.clear();preview.frames(info,(0,));spin(lambda:bool(received))
        assert len(calls)==2 and preview.frames(info,(0,)) is not None
    finally:preview.close()


def test_failure_does_not_retry_on_every_paint_and_can_retry_after_clear():
    application=QApplication.instance() or QApplication([])
    calls=[]
    def fail(*args):calls.append(1);raise OSError('source unavailable')
    preview=OriginalPreview(decoder=fail)
    info=VideoInfo('missing',2,2,12,1,frames=12)
    try:
        preview.frames(info,(0,));spin(lambda:bool(preview.error))
        for _ in range(5):assert preview.frames(info,(0,)) is None
        assert len(calls)==1
        preview.clear();preview.frames(info,(0,));spin(lambda:len(calls)==2)
    finally:preview.close()


def test_byte_budget_and_only_three_cached_frames():
    application=QApplication.instance() or QApplication([])
    def decode(info,indices,cancel):return {i:np.zeros((8,8,3),np.uint8) for i in indices}
    preview=OriginalPreview(decoder=decode,budget=400)
    info=VideoInfo('cache',8,8,12,1,frames=12)
    try:
        for index in range(4):
            preview.frames(info,(index,));spin(lambda:bool(getattr(preview,'_response',None)) and preview._wanted==preview._response[0])
        assert len(preview._cache)<=2
        assert sum(f.nbytes for f in preview._cache.values())<=400
        preview.clear();assert not preview._cache and preview._response is None
    finally:preview.close()


def test_failed_original_read_keeps_proxy_and_tracking_intact(window,source):
    from copy import deepcopy
    from lockframe.v072_check import activate
    clip=deepcopy(source);activate(window,[clip])
    track,anchors=window.track,dict(window.anchors)
    def fail(*args):raise OSError('test missing original')
    window._original_preview.clear();window._original_preview.decoder=fail
    window._show_current();spin(lambda:bool(window._original_preview.error))
    assert window.canvas.source is window.frame
    assert '高清不可用，已回退' in window.preview_note.text()
    assert window.track is track and window.anchors==anchors




def test_original_arrival_waits_for_pan_release_and_preserves_view(window,source):
    from copy import deepcopy
    from PySide6.QtCore import QPointF
    from lockframe.v072_check import activate
    activate(window,[deepcopy(source)])
    window._original_preview.clear()
    window.canvas.set_zoom(3.)
    window.canvas._pan_start=QPointF(1,1)
    window._show_current()
    pixels=window.canvas.source
    spin(lambda:bool(getattr(window._original_preview,'_response',None)))
    assert window.canvas.source is pixels
    window.canvas._pan_start=None
    spin(lambda:'原片高清 · 左' in window.preview_note.text())
    assert window.canvas.zoom_factor==3.
