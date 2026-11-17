import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import threading
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest


def events_until(predicate,seconds=30):
    from PySide6.QtWidgets import QApplication
    deadline=time.monotonic()+seconds
    while not predicate() and time.monotonic()<deadline:
        QApplication.processEvents();time.sleep(.01)
    QApplication.processEvents();assert predicate()


def test_public_prerender_review_and_cached_playback(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from lockframe.performance_check import check_performance
    application=QApplication.instance() or QApplication([])
    monkeypatch.setenv('LOCKFRAME_CACHE_DIR',str(tmp_path/'cache'))
    monkeypatch.setenv('LOCKFRAME_RETIME_TEMP',str(tmp_path/'native'))
    report=check_performance(tmp_path)
    assert report['explicit_confirmation_seed'] and report['cached_playback']
    assert report['prerender_frames']==45 and report['source_yuv_disk_bytes']==0


def test_snapshot_edits_cancel_stale_worker_and_clear_its_files(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication,QWidget
    from lockframe import clip_preview
    from lockframe.media import check_cancel
    from lockframe.models import Settings
    from test_retime_integration import tracked_clip
    from lockframe.models import VideoInfo
    application=QApplication.instance() or QApplication([]);owner=QWidget()
    monkeypatch.setenv('LOCKFRAME_RETIME_TEMP',str(tmp_path/'native'))
    service=clip_preview.ClipPreview(owner);started=threading.Event();captured=[]
    clip=tracked_clip(VideoInfo('unused',100,80,25,2,frames=50),0,35)
    clip.settings.retime={'enabled':True};mapping=np.arange(10,dtype=float)
    def hold(info,track,settings,values,first,key,folder,cancel,progress):
        captured.append((track.positions.copy(),deepcopy(settings.retime),values.copy()))
        (Path(folder)/'incomplete.mp4').write_bytes(b'never a completed movie')
        started.set();cancel.wait(10);check_cancel(cancel)
    monkeypatch.setattr(clip_preview,'render_preview_clip',hold)
    try:
        assert service.prepare('old',clip.info,clip.track,clip.settings,mapping,0)
        folder=service.work_folder;events_until(started.is_set)
        clip.settings.retime['enabled']=False;clip.track.positions[:]=0;mapping[:]=99
        service.want('new');events_until(lambda:not service.running)
        assert captured[0][1]=={'enabled':True} and captured[0][0].sum()>0
        np.testing.assert_array_equal(captured[0][2],np.arange(10,dtype=float))
        assert service.cache is None and not folder.exists() and service.track_ref is None
        with pytest.raises(ValueError):service._clean(tmp_path)
    finally:
        if service.worker:service.cancel();service.worker.wait(15000);application.processEvents()
        service.close();owner.close()


@pytest.mark.parametrize('policy',[None,[],{},'unknown',1])
def test_invalid_review_policy_rejected_cleanly(policy):
    from lockframe.models import Settings
    from lockframe.drift_review import DriftReview
    with pytest.raises(ValueError):Settings(drift_review=policy).validate()
    with pytest.raises(ValueError):DriftReview(policy)


def test_failure_is_readable_and_cleans_unfinished_preview(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication,QWidget
    from lockframe import clip_preview
    from lockframe.models import VideoInfo
    from test_retime_integration import tracked_clip
    application=QApplication.instance() or QApplication([]);owner=QWidget()
    monkeypatch.setenv('LOCKFRAME_RETIME_TEMP',str(tmp_path/'native'))
    service=clip_preview.ClipPreview(owner)
    clip=tracked_clip(VideoInfo('unused',100,80,25,2,frames=50),0,35)
    def fail(*args,**kwargs):raise ValueError('磁盘已满，无法预渲染')
    monkeypatch.setattr(clip_preview,'render_preview_clip',fail)
    service.prepare('key',clip.info,clip.track,clip.settings,np.arange(5),0)
    folder=service.work_folder;events_until(lambda:not service.running)
    assert service.cache is None and not folder.exists() and '磁盘已满' in service.message
    service.close();owner.close()


from test_retime_integration import window,demo_info,activate,tracked_clip


def test_close_during_prerender_waits_for_worker_cleanup(window,demo_info,monkeypatch):
    from lockframe import clip_preview
    from lockframe.media import check_cancel
    from lockframe.time_remap_model import make_constant_speed
    from PySide6.QtWidgets import QApplication
    started=threading.Event()
    def hold(info,track,settings,values,first,key,folder,cancel,progress):
        (Path(folder)/'unfinished.mp4').write_bytes(b'incomplete')
        started.set();cancel.wait(10);check_cancel(cancel)
    monkeypatch.setattr(clip_preview,'render_preview_clip',hold)
    activate(window,[tracked_clip(demo_info,0,35,make_constant_speed(None,.5))])
    window._confirm_discard=lambda:True
    window._prepare_clip_preview();events_until(started.is_set)
    service=window._clip_preview;folder=service.work_folder
    window.close()
    events_until(lambda:not service.running and not window.isVisible())
    assert not folder.exists() and service.cache is None
