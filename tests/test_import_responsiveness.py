import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from test_workflow_v9 import window
from lockframe import media, sequence
from lockframe.media import Cancelled
from lockframe.models import VideoInfo
from lockframe.timeline_media import TimelineMedia, MediaSource, MediaOverview, make_request


def test_slow_disk_cache_does_not_block_import_ui_and_cancellation_drops_result(window, tmp_path, monkeypatch):
    import lockframe.timeline_media as module
    path=tmp_path/'source.mp4';path.write_bytes(b'source')
    source=MediaSource.from_info(VideoInfo(str(path),160,90,10,1,frames=10))
    entered=threading.Event();release=threading.Event();received=[]
    request=make_request(source,0,10,count=1)
    monkeypatch.setattr(module,'build_overview',lambda r,c:MediaOverview(r.key,source.identity,0,10,r.film_frames,
        (np.zeros((8,8,3),np.uint8),),np.zeros(16,np.float32),False))
    service=TimelineMedia(directory=tmp_path/'cache');original=service._save
    def slow_save(result):
        entered.set();assert release.wait(3);original(result)
    monkeypatch.setattr(service,'_save',slow_save)
    service.ready.connect(received.append);window.strip._media_service=service
    timer=threading.Timer(.8,release.set);timer.start()
    try:
        service.request(request);assert entered.wait(2)
        start=time.monotonic();window.busy=True;window._sync_ui();elapsed=time.monotonic()-start
        assert elapsed<.3,elapsed
        assert not release.is_set()  # Control lock is available while IO is still pending.
        service.clear();release.set();service._thread.join(.02)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents();assert not received
    finally:
        release.set();timer.join();service.close();window.busy=False


def test_probe_cancel_kills_and_reaps_actual_slow_child(tmp_path,monkeypatch):
    original=media.subprocess.Popen;children=[]
    def slow(*args,**kwargs):
        child=original([sys.executable,'-c','import time;time.sleep(30)'],**kwargs);children.append(child);return child
    monkeypatch.setattr(media.subprocess,'Popen',slow)
    cancel=threading.Event();timer=threading.Timer(.15,cancel.set);timer.start();start=time.monotonic()
    try:
        with pytest.raises(Cancelled):media.probe('slow.mp4',cancel)
        assert time.monotonic()-start<2 and children[0].poll() is not None
    finally:timer.join()


def test_import_watchdog_cleans_up_stalled_actual_child(tmp_path,monkeypatch):
    original=media.subprocess.Popen;children=[]
    def stalled(*args,**kwargs):
        child=original([sys.executable,'-c','import time;time.sleep(30)'],**kwargs);children.append(child);return child
    monkeypatch.setattr(media.subprocess,'Popen',stalled)
    with pytest.raises(TimeoutError,match='无进展'):
        media.run_ffmpeg([],threading.Event(),stall_timeout=.05)
    assert children and children[0].poll() is not None


def test_batch_reports_inside_each_clip_and_cancel_preserves_finished_only(monkeypatch):
    cancel=threading.Event();progress=[];calls=[]
    def importer(path,cancel,report=None):
        calls.append(path);report(.1);report(.5)
        if path=='second':cancel.set();report(.7)
        return path
    monkeypatch.setattr(sequence,'import_video',importer)
    result=sequence.import_many(['first','second','third'],cancel,lambda v,s:progress.append((v,s)))
    assert result==[{'info':'first','path':'first'}] and calls==['first','second']
    assert any(0<v<1/3 for v,_ in progress) and any('导入 2/3' in s for _,s in progress)
    assert all(a[0]<=b[0] for a,b in zip(progress,progress[1:]))


def test_import_check_actual_workflow_and_failures(tmp_path):
    from lockframe.import_check import check_imports
    report=check_imports(tmp_path)
    assert report['mixed_valid']==2 and report['failed']==2
    assert report['cancelled_preserves_completed'] and report['heartbeat_ticks']>=3
    assert report['duplicate_feedback'] and report['filter_reset'] and report['slow_probe_cancelled']
