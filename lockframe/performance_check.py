"""Exercise public review/preview controls in source and frozen releases."""
from copy import deepcopy
from pathlib import Path
import json
import threading
import time
import traceback
from unittest.mock import patch


def check_performance(folder):
    import numpy as np
    from PySide6.QtCore import QSettings,Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from . import app
    from .demo import create_demo,demo_frame
    from .media import import_video,configure_cache_preferences
    from .models import Settings
    from .sequence import Clip
    from .tracking import FeatureTracker,track_video
    from .tracking_pause import TrackingPaused,is_complete,computed_mask
    from .time_remap_model import make_constant_speed
    from .v072_check import activate

    directory=Path(folder);directory.mkdir(parents=True,exist_ok=True)
    source=directory/'预渲染 中文.mp4';cancel=threading.Event()
    create_demo(str(source),frames=24,width=320,height=180)
    info=import_video(str(source),cancel);anchor=demo_frame(0,320,180)[1]
    original=FeatureTracker.step;calls=[]
    def weak(self,gray):
        value=original(self,gray);calls.append(1)
        if value.state!='lost':value.confidence=.6
        return value
    with patch.object(FeatureTracker,'step',weak):
        try:
            track_video(info,{0:anchor},0,23,cancel,drift_review='standard',pause_on_lost=True)
            raise AssertionError('continuous weak evidence did not pause')
        except TrackingPaused as value:paused=value
        assert paused.reason=='review' and paused.frame==6 and len(calls)==6
        assert not paused.result.lost_frames and computed_mask(paused.result).sum()==7
        calls.clear()
        try:
            track_video(info,{0:anchor},0,23,cancel,drift_review='standard',pause_on_lost=True,
                previous=paused.result,resume=paused.resume)
            raise AssertionError('unconfirmed review was bypassed')
        except TrackingPaused as value:assert value.frame==6 and not calls

    settings=Settings(acceleration='cpu',compute_threads=2,resolution=0,drift_review='standard')
    clip=Clip(info,settings,{0:anchor},0,23,track=paused.result)
    window=None
    def wait(predicate,seconds=35):
        deadline=time.monotonic()+seconds
        while not predicate() and time.monotonic()<deadline:
            QApplication.processEvents();time.sleep(.01)
        QApplication.processEvents();assert predicate(),'preview did not finish: '+window._clip_preview.message
    try:
        with patch.object(app,'QSettings',lambda *args:QSettings(str(directory/'prefs.ini'),QSettings.Format.IniFormat)):
            window=app.MainWindow()
        window._confirm_discard=lambda:True
        window.resize(1560,960);window.show();activate(window,[clip])
        window._tracking_paused(paused);window._sync_ui()
        assert window.continue_button.isEnabled() and window.continue_button.text()=='确认并继续'
        with patch.object(window,'_start_tracking') as start:
            QTest.mouseClick(window.continue_button,Qt.MouseButton.LeftButton)
            assert start.call_count==1
        assert 6 in window.anchors and window._paused_tracking['corrected']
        # Actual numeric continuation after the explicit human-confirmed seed.
        completed=track_video(info,window.anchors,0,23,cancel,drift_review='standard',
            previous=paused.result,pause_on_lost=True)
        assert is_complete(completed) and not completed.lost_frames and completed.states[6]=='manual'
        window._tracking_done(completed);window._sync_ui()
        playback=check_playback_hotpath(window)
        positions=completed.positions.copy();anchors=deepcopy(window.anchors)
        slow=make_constant_speed(speed=.5);slow.update(optical_flow=True,flow_backend='dis')
        window._apply_retime_config(slow);QApplication.processEvents()
        window.retime_panel.clock.setCurrentIndex(window.retime_panel.clock.findData('output'))
        window._seek_output(3)
        assert window.retime_panel.prepare_clip.isEnabled()
        QTest.mouseClick(window.retime_panel.prepare_clip,Qt.MouseButton.LeftButton)
        service=window._clip_preview
        wait(lambda:not service.running)
        assert service.cache is not None,service.message
        payload=deepcopy(service.cache)
        assert payload['first']==3 and payload['last']==47 and payload['frames']==45
        assert payload['report']['actual_backend']=='dis' and payload['report']['source_store_bytes']==0
        assert payload['report']['source_cache_strategy']=='stream'
        assert service.frame(service.wanted,4) is not None
        assert window.retime_panel.play_clip.isEnabled()
        QTest.mouseClick(window.retime_panel.play_clip,Qt.MouseButton.LeftButton)
        assert window.timer.isActive() and window._retime_play_range==(3,47)
        window._play_started=time.monotonic()-.3;window._retime_tick()
        assert window._retime_output_step>3 and '预渲染缓存' in window.preview_note.text()
        window._stop_play()
        assert window.track is completed and window.anchors==anchors
        np.testing.assert_array_equal(window.track.positions,positions)
        screenshot=directory/'performance-workspace.png';window.grab().save(str(screenshot))
        cache_path=Path(payload['path']);window._apply_retime_config(make_constant_speed(speed=2.))
        assert service.cache is None and not cache_path.exists()
        # Cancel a real render immediately; its directory is cleaned on finish.
        window._apply_retime_config(slow);window._seek_output(0)
        window._prepare_clip_preview();work=service.work_folder
        assert service.running and work is not None
        window._prepare_clip_preview();wait(lambda:not service.running)
        assert service.cache is None and not work.exists()
        return dict(review_pauses_without_lost=True,unconfirmed_review_blocked=True,
            explicit_confirmation_seed=True,confirmed_continue_complete=True,
            prerender_frames=payload['frames'],cached_playback=True,parameter_invalidation=True,
            cancellation_cleanup=True,tracking_unchanged=True,dual_preview=True,
            source_cache=payload['report']['source_cache_strategy'],source_yuv_disk_bytes=0,
            stage_timing=payload['report']['stage_timing'],playback=playback,
            footage='synthetic; not manual ground-truth accuracy or long-shot evidence')
    finally:
        if window is not None:
            window._stop_play()
            if window._clip_preview.worker:
                window._clip_preview.cancel();window._clip_preview.worker.wait(15000)
                QApplication.processEvents()
            window.close();QApplication.processEvents()
        configure_cache_preferences(None)


def check_playback_hotpath(window):
    """Exercise actual transport scheduling; no physical display-FPS claim."""
    from . import ad_workspace
    from .timeline_media import MediaSource
    from PySide6.QtWidgets import QApplication
    window._stop_play();window._seek(0)
    with patch.object(window,'_retime_sync',wraps=window._retime_sync) as layout, \
         patch.object(ad_workspace,'ad_resize',wraps=ad_workspace.ad_resize) as resize, \
         patch.object(MediaSource,'from_info',wraps=MediaSource.from_info) as identity:
        window._toggle_play()
        assert window.timer.isActive() and window.strip._media_busy and window._thumbnails._paused
        for frame in range(1,7):window._seek(frame)
        assert not layout.call_count and not resize.call_count and not identity.call_count
        stats=dict(window._playback_stats)
        window._stop_play()
        assert layout.call_count and resize.call_count and identity.call_count
        assert not window.strip._media_busy and not window._thumbnails._paused
    window.strip._request_media()
    service=window.strip._media_service
    with patch.object(service,'request',wraps=service.request) as request:
        window.strip._request_media()
        assert not request.call_args.args[0].include_audio
        window.strip.set_audio_visible(True);window.strip._request_media()
        assert request.call_args.args[0].include_audio
        window.strip.set_audio_visible(False);window.strip._request_media()
    QApplication.processEvents()
    return dict(per_frame_layout_avoided=True,background_paused=True,background_resumed=True,
                hidden_waveform_not_requested=True,submitted_frames=stats['frames'],
                average_processing_ms=stats['total']/stats['frames']*1000)


def run_performance_check(folder):
    from . import __version__
    directory=Path(folder);directory.mkdir(parents=True,exist_ok=True)
    report=dict(version=__version__,ok=False)
    try:report['checks']=check_performance(directory);report['ok']=True
    except Exception:report['error']=traceback.format_exc()
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['ok'] else 1
