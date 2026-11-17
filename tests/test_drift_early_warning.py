from copy import deepcopy
import threading
import numpy as np
import pytest
from lockframe.drift_review import DriftReview,POLICIES,REVIEW_REVISION
from lockframe.tracking import track_video,FeatureTracker
from lockframe.precision_tracking import PrecisionFeatureTracker
from lockframe.tracking_pause import TrackingPaused,computed_mask
from test_tracking_pause import scene
from test_retime_integration import window,demo_info,tracked_clip,activate


def feed(guard,frame,score,state='tracked'):
    return guard.observe(frame,score,state,[.4+frame*.001,.4],[.3,.3,.2,.2])


def test_relative_quality_drop_warns_before_fixed_threshold_and_before_pause():
    guard=DriftReview('standard');warnings=[];reviews=[]
    for i in range(12):
        review=feed(guard,i,.96 if i<6 else .8)
        if guard.new_warning:warnings.append(guard.new_warning.copy())
        if review:reviews.append(review)
    assert warnings[0]['frame']==8 and reviews[0]['frame']==11
    assert warnings[0]['confidence']>POLICIES['standard'][0]
    assert '持续下降' in warnings[0]['reason'] and not reviews[0]['confirmed_lost']


def test_single_glitch_does_not_warn_and_recovery_clears_banner_evidence():
    guard=DriftReview('standard')
    for i in range(8):assert feed(guard,i,.96 if i!=5 else .5) is None
    assert guard.warning is None
    for i in range(8,11):assert feed(guard,i,.6) is None
    assert guard.warning is not None and guard.new_warning is not None
    assert feed(guard,11,.96) is None and guard.warning is None
    for i in range(12,15):assert feed(guard,i,.6) is None
    assert guard.new_warning['frame']==14


def test_repeated_weak_recovery_warns_but_one_recovery_does_not():
    guard=DriftReview('standard')
    assert feed(guard,0,.78,'recovered') is None and guard.warning is None
    feed(guard,1,.96)
    for i in range(2,5):assert feed(guard,i,.78,'recovered') is None
    assert guard.new_warning['reason']=='连续重捕获，匹配不稳定'


@pytest.mark.parametrize('mode',['normal','precision'])
@pytest.mark.parametrize('backward',[False,True])
def test_pipeline_publishes_measured_warning_then_stops_and_persists(scene,monkeypatch,mode,backward):
    info,box,_=scene;kind=PrecisionFeatureTracker if mode=='precision' else FeatureTracker
    original=kind.step;calls=[];events=[];progress=[]
    def step(self,gray):
        value=original(self,gray);calls.append(int(gray[0,0]))
        if value.state!='lost':value.confidence=.96 if len(calls)<=5 else .8
        return value
    monkeypatch.setattr(kind,'step',step)
    seed=24 if backward else 0
    with pytest.raises(TrackingPaused) as raised:
        track_video(info,{seed:box(seed)},0,24,threading.Event(),mode=mode,
            drift_review='standard',pause_on_lost=True,segmented_tracking=True,segment_frames=10,
            on_frame=events.append,progress=lambda value,text:progress.append(text))
    pause=raised.value
    assert pause.reason=='review' and len(calls)==11 and not pause.result.lost_frames
    warned=[v for v in events if 'drift_warning' in v]
    assert warned[0]['frame']==(16 if backward else 8)
    assert warned[0]['confidence']==.8 and pause.frame==(13 if backward else 11)
    assert len(events)==12 and computed_mask(pause.result).sum()==12
    assert pause.result.performance['early_warning_ranges'] and any('跟偏预警' in s for s in progress)
    assert pause.result.performance['drift_review_revision']==REVIEW_REVISION
    calls.clear()
    with pytest.raises(TrackingPaused) as resumed:
        track_video(info,{seed:box(seed)},0,24,threading.Event(),mode=mode,
            drift_review='standard',pause_on_lost=True,segmented_tracking=True,segment_frames=10)
    assert not calls and resumed.value.result.performance['early_warning_ranges']==pause.result.performance['early_warning_ranges']


def test_warning_only_changes_no_measurements_and_old_review_revision_recomputes(scene,monkeypatch):
    info,box,_=scene;original=FeatureTracker.step;calls=[]
    def step(self,gray):
        value=original(self,gray);i=int(gray[0,0]);calls.append(i)
        if value.state!='lost':value.confidence=.96 if i<=5 or i>=13 else .8
        return value
    monkeypatch.setattr(FeatureTracker,'step',step)
    baseline=track_video(info,{0:box(0)},0,24,threading.Event())
    warned=track_video(info,{0:box(0)},0,24,threading.Event(),drift_review='standard')
    np.testing.assert_array_equal(warned.positions,baseline.positions)
    np.testing.assert_array_equal(warned.confidence,baseline.confidence)
    assert warned.states==baseline.states and warned.performance['early_warning_ranges']
    assert warned.performance['early_warning_ranges'][0]['start']==6
    assert warned.performance['early_warning_ranges'][0]['end']==12
    warned.performance.pop('drift_review_revision');calls.clear()
    track_video(info,{0:box(0)},0,24,threading.Event(),drift_review='standard',previous=warned)
    assert calls==list(range(1,25))


def test_warning_banner_uses_same_source_clears_and_survives_project_review(window,demo_info):
    from lockframe.models import source_identity
    clip=tracked_clip(demo_info,0,35);activate(window,[clip]);window.live_preview_check.setChecked(True)
    warning=dict(start=2,end=4,frame=4,confidence=.8,reason='匹配质量持续下降',policy='standard')
    payload=dict(frame=4,box=(.25,.25,.5,.5),position=(.5,.5),confidence=.8,state='tracked',
        completed=5,total=36,source_identity=source_identity(demo_info),drift_warning=warning)
    window._on_tracking_frame(payload)
    assert not window.drift_warning_banner.isHidden() and '跟偏预警' in window.drift_warning_banner.text()
    bad=deepcopy(payload);bad['source_identity']={'path':'different'};bad.pop('drift_warning')
    window._on_tracking_frame(bad)
    assert not window.drift_warning_banner.isHidden()
    window._on_progress(.22,'[计算:0.220] 第6帧 · 跟偏预警 · 匹配质量下降')
    assert not window.drift_warning_banner.isHidden()
    payload.pop('drift_warning');window._on_tracking_frame(payload)
    assert window.drift_warning_banner.isHidden()
    window.live_preview_check.setChecked(False)
    window._on_progress(.2,'[计算:0.200] 跟偏预警 · 第5帧 · 可暂停检查')
    assert not window.drift_warning_banner.isHidden()
    window.track.performance['early_warning_ranges']=[warning];window._refresh_issues();window._seek(4)
    assert '曾触发' in window.drift_warning_banner.text()
    assert any('跟偏预警记录' in window.issue_list.item(i).text() for i in range(window.issue_list.count()))
    window._seek(12);assert window.drift_warning_banner.isHidden()


def test_public_warning_acceptance(tmp_path,monkeypatch):
    from PySide6.QtWidgets import QApplication
    from lockframe.drift_warning_check import check_drift_warning
    application=QApplication.instance() or QApplication([])
    monkeypatch.setenv('LOCKFRAME_CACHE_DIR',str(tmp_path/'cache'))
    assert check_drift_warning(tmp_path)['early_before_review']
