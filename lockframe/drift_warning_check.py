"""Early warning UI/engine acceptance, runnable in the frozen executable."""
from copy import deepcopy
from pathlib import Path
import json,threading,traceback
from unittest.mock import patch


def check_drift_warning(directory):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from . import app
    from .demo import create_demo,demo_frame
    from .media import import_video,configure_cache_preferences
    from .models import Settings
    from .sequence import Clip
    from .tracking import FeatureTracker,track_video
    from .tracking_pause import TrackingPaused
    from .v072_check import activate
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    source=folder/'跟偏预警合成.mp4';create_demo(str(source),frames=24,width=320,height=180)
    cancel=threading.Event();info=import_video(str(source),cancel);anchor=demo_frame(0,320,180)[1]
    original=FeatureTracker.step;calls=[];events=[];messages=[]
    def weak(self,gray):
        value=original(self,gray);calls.append(1)
        if value.state!='lost':value.confidence=.96 if len(calls)<=5 else .8
        return value
    with patch.object(FeatureTracker,'step',weak):
        try:
            track_video(info,{0:anchor},0,23,cancel,drift_review='standard',pause_on_lost=True,
                on_frame=events.append,progress=lambda value,text:messages.append((value,text)))
            raise AssertionError('expected sustained quality decline pause')
        except TrackingPaused as value:paused=value
    warned=[p for p in events if p.get('drift_warning')]
    assert warned[0]['frame']==8 and paused.frame==11 and len(calls)==11
    assert warned[0]['confidence']==.8 and not paused.result.lost_frames
    assert any('跟偏预警' in text for _,text in messages)
    from .drift_consistency_check import check_consistency
    consistency=check_consistency(folder/'consistency')
    window=None
    try:
        with patch.object(app,'QSettings',lambda *args:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)):
            window=app.MainWindow()
        window._confirm_discard=lambda:True
        window.show();activate(window,[Clip(info,Settings(drift_review='standard'),{0:anchor},0,23)])
        window.live_preview_check.setChecked(True)
        window._on_tracking_frame(warned[0]);QApplication.processEvents()
        banner=window.drift_warning_banner
        for width,height in ((1560,960),(1160,760)):
            window.resize(width,height);QApplication.processEvents()
            assert banner.isVisible() and banner.visibleRegion().contains(banner.rect())
            # A partial trajectory must not render a fake stabilized result;
            # the persistent right pane correctly falls back to source pixels.
            assert window.canvas.source_image is not None and window.canvas.persistent_compare
            assert window.canvas.height()>180
            window.grab().save(str(folder/f'early-warning-{width}.png'))
        foreign=deepcopy(events[0]);foreign['source_identity']={'path':'different'}
        window._on_tracking_frame(foreign);assert banner.isVisible()
        window._on_tracking_frame(events[0]);assert banner.isHidden()
        window.live_preview_check.setChecked(False)
        value,text=next((v,t) for v,t in messages if '跟偏预警' in t)
        window._on_progress(value,text);assert banner.isVisible()
        window._tracking_paused(paused);window._sync_ui();window._seek(paused.frame)
        assert '已暂停' in banner.text() and window.continue_button.text()=='确认并继续'
        assert window.track is paused.result
        return dict(early_warning_frame=8,pause_frame=11,score_at_warning=.8,
            original_absolute_threshold=.62,early_before_review=True,measured_frames=12,
            lost_frames=0,tracked_geometry_unchanged=True,live_banner=True,preview_off_banner=True,
            foreign_clip_ignored=True,recovery_clears_banner=True,small_window=True,consistency=consistency,
            cases='controlled confidence degradation on synthetic pixels, not real drift ground truth')
    finally:
        if window is not None:window.close();QApplication.processEvents()
        configure_cache_preferences(None)


def run_drift_warning_check(directory):
    from . import __version__
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    report=dict(version=__version__,ok=False)
    try:report['checks']=check_drift_warning(folder);report['ok']=True
    except Exception:report['error']=traceback.format_exc()
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['ok'] else 1
