from dataclasses import replace
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lockframe.models import VideoInfo,file_fingerprint
from lockframe.sequence import Clip
from lockframe.thumbnail_queue import ThumbnailQueue
from lockframe.thumbnail_check import check_thumbnails,row_data,spin
from test_workflow_v9 import source,window


def clip_at(tmp_path):
    path=tmp_path/'source.mp4';path.write_bytes(b'source')
    return Clip(VideoInfo(str(path),80,40,10,.3,frames=3,fingerprint=file_fingerprint(str(path))))


def jpeg(color=40):
    return cv2.imencode('.jpg',np.full((45,80,3),color,np.uint8))[1].tobytes()


def test_changed_trim_and_removed_clip_discard_late_results(tmp_path):
    app=QApplication.instance() or QApplication([])
    clip=clip_at(tmp_path);started=threading.Event();release=threading.Event();received=[]
    def decode(c,*args):
        if c.start==0:started.set();assert release.wait(3)
        return jpeg(c.start*100)
    queue=ThumbnailQueue(decoder=decode);queue.ready.connect(lambda *args:received.append(args))
    try:
        queue.sync([clip]);spin(started.is_set)
        changed=replace(clip,start=1)
        queue.sync([changed]);release.set();spin(lambda:bool(received))
        assert len(received)==1 and received[0][1]==queue.key(changed)
        assert not received[0][3]
        queue.sync([]);queue.recheck([changed]);app.processEvents()
        assert queue.statistics()['pending']==0 and len(received)==1
    finally:release.set();queue.close()


def test_cache_validates_proxy_and_source_changes_off_gui_thread(tmp_path):
    app=QApplication.instance() or QApplication([])
    clip=clip_at(tmp_path);proxy=tmp_path/'proxy.mp4';proxy.write_bytes(b'proxy')
    clip.info.proxy=str(proxy)
    calls=[];received=[];main_thread=threading.get_ident()
    def decode(*args):calls.append(threading.get_ident());return jpeg(len(calls)*30)
    queue=ThumbnailQueue(decoder=decode);queue.ready.connect(lambda *args:received.append(args))
    try:
        queue.sync([clip]);spin(lambda:len(received)==1)
        queue.recheck([clip]);spin(lambda:len(received)==2)
        assert len(calls)==1 and calls[0]!=main_thread
        proxy.write_bytes(b'changed proxy size')
        queue.recheck([clip]);spin(lambda:len(received)==3)
        assert len(calls)==2 and received[-1][2]!=received[0][2]
        Path(clip.info.path).write_bytes(b'changed original size')
        queue.recheck([clip]);spin(lambda:len(received)==4)
        assert not received[-1][2] and '变化' in received[-1][3]
        assert len(calls)==2 and queue.statistics()['cache_hits']>=1
    finally:queue.close()


def test_removed_inflight_clip_never_delivers_a_result(tmp_path):
    app=QApplication.instance() or QApplication([])
    clip=clip_at(tmp_path);started=threading.Event();release=threading.Event();received=[]
    def slow(*args):started.set();assert release.wait(3);return jpeg()
    queue=ThumbnailQueue(decoder=slow);queue.ready.connect(lambda *args:received.append(args))
    try:
        queue.sync([clip]);spin(started.is_set)
        queue.sync([]);release.set()
        spin(lambda:queue._running is None)
        app.processEvents()
        assert not received and queue.statistics()['results']==0
    finally:release.set();queue.close()


def test_cache_entry_byte_limit_and_pause_close_do_not_block_ui(tmp_path):
    app=QApplication.instance() or QApplication([])
    clip=clip_at(tmp_path);received=[];calls=[]
    def decode(*args):calls.append(1);return jpeg(100)
    queue=ThumbnailQueue(decoder=decode,entries=2,budget=1600)
    queue.ready.connect(lambda *args:received.append(args))
    try:
        queue.set_paused(True);queue.sync([clip])
        for _ in range(5):app.processEvents();time.sleep(.005)
        assert not calls
        queue.set_paused(False);spin(lambda:len(received)==1)
        for start in (1,2):
            queue.sync([replace(clip,start=start)]);spin(lambda:len(received)==start+1)
        assert queue.statistics()['cache_entries']<=2 and queue.statistics()['cache_bytes']<=1600
    finally:queue.close()
    started=threading.Event();release=threading.Event()
    def slow(*args):started.set();assert release.wait(3);return jpeg()
    queue=ThumbnailQueue(decoder=slow);queue.ready.connect(lambda *args:pytest.fail('Closed thumbnail queue delivered'))
    try:
        queue.sync([clip]);spin(started.is_set)
        before=time.monotonic();queue.close();assert time.monotonic()-before<.1
        release.set();queue._thread.join(1);app.processEvents()
        assert not queue._thread.is_alive() and queue.statistics()['cache_bytes']==0
    finally:release.set();queue.close()


def test_actual_import_sequence_trim_and_failure_list_flow(tmp_path):
    report=check_thumbnails(tmp_path/'素材列表验收')
    assert report['all_thumbnails_without_clicking'] and report['sequence_reopened']
    assert report['missing_source_placeholder'] and report['tracking_state_unchanged']


def test_500_rows_decode_in_background_keep_selection_and_frame_work(window,source):
    from copy import deepcopy
    from lockframe.v072_check import activate
    from PySide6.QtCore import Qt
    first=deepcopy(source);activate(window,[first])
    window._thumbnails.close()
    calls=[]
    def decode(*args):calls.append(threading.get_ident());return jpeg(60)
    queue=ThumbnailQueue(window,decoder=decode)
    queue.ready.connect(window._thumbnail_ready);window._thumbnails=queue
    queue.set_paused(True)
    window.clips=[first]+[Clip(first.info,start=first.start,end=first.end) for _ in range(499)]
    selected={window.clips[0].id,window.clips[-1].id}
    before=time.monotonic();window._refresh_sequence(selected)
    elapsed=time.monotonic()-before
    assert elapsed<1 and not calls
    assert window._selected_clip_ids()==selected and window.active_clip_id==first.id
    timer=QTimer();timer.setInterval(10);ticks=[];timer.timeout.connect(lambda:ticks.append(1));timer.start()
    queue.set_paused(False)
    try:
        spin(lambda:all(row_data(window,c.id)['thumbnail_state']=='ready' for c in window.clips))
        assert ticks and calls==[calls[0]] and calls[0]!=threading.get_ident()
        assert queue.statistics()['cache_entries']==1
        assert window._selected_clip_ids()==selected
        assert window.clip_list.currentItem().data(Qt.ItemDataRole.UserRole)==first.id
        window._seek(3);window._step_preview(1)
        assert window.frame_index==4 and window.track is first.track
    finally:timer.stop()
