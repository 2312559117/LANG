from copy import deepcopy
from dataclasses import replace
import json,threading
import numpy as np
import pytest
from lockframe.drift_review import DriftReview,retained_ranges
from lockframe.tracking import FeatureTracker,track_video
from lockframe.precision_tracking import PrecisionFeatureTracker
from lockframe.tracking_pause import TrackingPaused
from test_tracking_pause import scene
from test_retime_integration import window,demo_info,tracked_clip,activate


def feed(monitor,index,score):
    event=monitor.observe(index,score,'tracked',[.5,.5],[.3,.3,.4,.4])
    return deepcopy((event,monitor.warning,monitor.new_warning))


def test_slow_quality_decline_cannot_lower_its_own_reference():
    monitor=DriftReview('standard');warnings=[];reviews=[]
    for index in range(100):
        score=.96 if index<8 else max(.80,.96-(index-7)*.002)
        event,warning,new=feed(monitor,index,score)
        if new:warnings.append(new)
        if event:reviews.append(event)
    assert warnings and reviews and warnings[0]['frame']<reviews[0]['frame']
    assert warnings[0]['baseline_confidence']==.96 and monitor.baseline==.96
    assert warnings[0]['confidence']>.62 and not reviews[0]['confirmed_lost']


@pytest.mark.parametrize('policy',['relaxed','standard','strict'])
@pytest.mark.parametrize('cut',[30,62,67])
def test_json_snapshot_preserves_baseline_and_does_not_repeat_warning(policy,cut):
    scores=[.96]*8+[.89]*53+[.78]*12+[.97]*5+[.70]*12
    original=DriftReview(policy)
    for i,score in enumerate(scores[:cut]):feed(original,i,score)
    value=json.loads(json.dumps(original.snapshot(),allow_nan=False))
    restored=DriftReview(policy)
    assert restored.restore(value,start=0,end=100,frame=cut-1,direction=1)
    for i in range(cut,len(scores)):
        assert feed(restored,i,scores[i])==feed(original,i,scores[i])
    value['stable'][:]=[0.]
    assert restored.snapshot()==original.snapshot()


@pytest.mark.parametrize('field,value',[
    ('revision',-1),('baseline',float('nan')),('baseline',True),
    ('stable',[.95]*25),('history',[[0,[.5,.5]],[8,[.5,.5]]]),
    ('streak',[True]),('reported',1),('warning',{'confidence':float('nan')}),
])
def test_corrupt_review_snapshot_is_rejected_without_partial_mutation(field,value):
    monitor=DriftReview('standard')
    for index in range(8):feed(monitor,index,.96)
    data=monitor.snapshot();data[field]=value
    restored=DriftReview('standard');before=restored.snapshot()
    assert not restored.restore(data,start=0,end=100,frame=7,direction=1)
    assert restored.snapshot()==before


def test_reused_warning_keeps_unrecalculated_prefix_and_splits_at_manual_frames():
    item=dict(start=6,end=11,frame=11,confidence=.8,policy='standard',reason='下降')
    keep=np.ones(25,dtype=bool);keep[[8,11]]=False
    ranges=retained_ranges([item],keep,0,24)
    assert [(v['start'],v['end'],v['frame']) for v in ranges]==[(6,7,7),(9,10,10)]
    assert item['start']==6 and item['end']==11


@pytest.fixture
def long_scene(scene,monkeypatch):
    from lockframe import performance
    info,box,_=scene;original=performance.GrayFrameCache
    class Longer(original):
        def read(self,index):
            image=super().read(min(index,24)).copy();image[0,0]=index
            return image
        def iter_frames(self,indices):
            for index in indices:yield index,self.read(index)
    monkeypatch.setattr(performance,'GrayFrameCache',Longer)
    return replace(info,frames=100,duration=100/info.fps),box


@pytest.mark.parametrize('disk',[False,True])
@pytest.mark.parametrize('mode',['normal','precision'])
@pytest.mark.parametrize('backward',[False,True])
def test_long_manual_pause_continuation_matches_uninterrupted_alarm(long_scene,monkeypatch,disk,mode,backward):
    info,box=long_scene;kind=PrecisionFeatureTracker if mode=='precision' else FeatureTracker
    original=kind.step;calls=[]
    def step(self,gray):
        index=int(gray[0,0]);calls.append(index);value=original(self,gray)
        distance=99-index if backward else index
        if value.state!='lost':value.confidence=.96 if distance<=8 else .88 if distance<=60 else .83
        return value
    monkeypatch.setattr(kind,'step',step)
    opts=dict(drift_review='standard',pause_on_lost=True,mode=mode)
    seed=99 if backward else 0;anchors={seed:box(24 if backward else 0)}
    stop=39 if backward else 60;destination=33 if backward else 66
    expected=list(range(38,32,-1)) if backward else list(range(61,67))
    with pytest.raises(TrackingPaused) as baseline:
        track_video(info,anchors,0,99,threading.Event(),**opts)
    assert baseline.value.reason=='review' and baseline.value.frame==destination
    event=threading.Event()
    def observe(payload):
        if payload['frame']==stop:event.set()
    with pytest.raises(TrackingPaused) as raised:
        track_video(info,anchors,0,99,threading.Event(),pause_event=event,on_frame=observe,
            segmented_tracking=disk,segment_frames=10,**opts)
    paused=raised.value
    assert paused.reason=='manual' and paused.frame==stop
    assert paused.resume['review_monitor']['baseline']==.96
    calls.clear();events=[]
    with pytest.raises(TrackingPaused) as resumed:
        track_video(info,anchors,0,99,threading.Event(),on_frame=events.append,
            segmented_tracking=disk,segment_frames=10,
            **({} if disk else dict(previous=paused.result,resume=paused.resume)),**opts)
    result=resumed.value
    assert result.reason=='review' and result.frame==destination and calls==expected
    assert result.result.performance['review_state_restores']==1
    selected=slice(33,None) if backward else slice(None,67)
    np.testing.assert_array_equal(result.result.positions[selected],baseline.value.result.positions[selected])
    assert [v['frame'] for v in events if v['state']!='manual']==expected


@pytest.mark.parametrize('cut',[30,62])
def test_reverse_snapshot_preserves_time_direction(cut):
    scores=[.96]*8+[.89]*53+[.78]*12
    monitor=DriftReview('standard')
    for index,score in enumerate(scores[:cut]):feed(monitor,99-index,score)
    restored=DriftReview('standard')
    assert restored.restore(monitor.snapshot(),start=0,end=99,frame=100-cut,direction=-1)
    for index in range(cut,len(scores)):
        assert feed(restored,99-index,scores[index])==feed(monitor,99-index,scores[index])


def test_bad_in_memory_diagnostic_rebuilds_instead_of_skipping_early_evidence(long_scene,monkeypatch):
    info,box=long_scene;original=FeatureTracker.step;calls=[];event=threading.Event()
    def step(self,gray):
        index=int(gray[0,0]);calls.append(index);value=original(self,gray)
        if value.state!='lost':value.confidence=.96 if index<=8 else .88 if index<=60 else .83
        return value
    monkeypatch.setattr(FeatureTracker,'step',step)
    def observe(payload):
        if payload['frame']==60:event.set()
    with pytest.raises(TrackingPaused) as raised:
        track_video(info,{0:box(0)},0,99,threading.Event(),pause_event=event,on_frame=observe,
            drift_review='standard',pause_on_lost=True)
    paused=raised.value;paused.resume['review_monitor']['baseline']=float('nan');calls.clear()
    with pytest.raises(TrackingPaused) as resumed:
        track_video(info,{0:box(0)},0,99,threading.Event(),previous=paused.result,resume=paused.resume,
            drift_review='standard',pause_on_lost=True)
    assert resumed.value.reason=='review' and resumed.value.frame==66
    assert calls==list(range(1,67)) and resumed.value.result.performance['review_state_rebuilds']==1


def test_export_filename_cannot_trigger_tracking_warning(window,demo_info):
    activate(window,[tracked_clip(demo_info,0,35)])
    window._on_progress(.2,'正在导出 跟偏预警测试.mp4')
    assert window.drift_warning_banner.isHidden()


def test_unmeasured_slots_cannot_display_recorded_warning(window,demo_info):
    activate(window,[tracked_clip(demo_info,0,35)])
    window.track.performance.update(complete=False,computed_frames=[i<3 for i in range(36)],
        early_warning_ranges=[dict(start=4,end=6,frame=6,confidence=.8,reason='外部旧记录',policy='standard')])
    window.frame_index=5;window._refresh_drift_warning()
    assert window.drift_warning_banner.isHidden()


def test_manual_correction_does_not_erase_earlier_unrecalculated_warnings(scene,monkeypatch):
    info,box,_=scene;original=FeatureTracker.step
    def step(self,gray):
        value=original(self,gray)
        if value.state!='lost':value.confidence=.6
        return value
    monkeypatch.setattr(FeatureTracker,'step',step)
    with pytest.raises(TrackingPaused) as raised:
        track_video(info,{0:box(0)},0,24,threading.Event(),drift_review='standard',pause_on_lost=True)
    result=track_video(info,{0:box(0),6:box(6)},0,24,threading.Event(),drift_review='standard',previous=raised.value.result)
    assert result.performance['early_warning_ranges'][0]['start']==1
    assert result.performance['early_warning_ranges'][0]['end']==5
    assert result.states[6]=='manual' and not result.lost_frames


def test_corrupt_disk_monitor_rebuilds_current_walk(long_scene,monkeypatch,tmp_path):
    from lockframe.checkpoints import save_checkpoint
    info,box=long_scene;original=FeatureTracker.step;calls=[];pause=threading.Event()
    def step(self,gray):
        index=int(gray[0,0]);calls.append(index);value=original(self,gray)
        if value.state!='lost':value.confidence=.96 if index<=8 else .88 if index<=60 else .83
        return value
    monkeypatch.setattr(FeatureTracker,'step',step)
    def observe(payload):
        if payload['frame']==60:pause.set()
    options=dict(drift_review='standard',pause_on_lost=True,segmented_tracking=True,segment_frames=10)
    with pytest.raises(TrackingPaused):
        track_video(info,{0:box(0)},0,99,threading.Event(),pause_event=pause,on_frame=observe,**options)
    path=next((tmp_path/'cache/tracking-checkpoints').glob('*.npz'))
    with np.load(path,allow_pickle=False) as archive:
        values={name:archive[name].copy() for name in archive.files if name!='metadata'}
        metadata=json.loads(str(archive['metadata'].item()))
    metadata['review_monitor']['history']=[[9999,[.5,.5]]]
    save_checkpoint(path,metadata['key'],values,metadata);calls.clear()
    with pytest.raises(TrackingPaused) as restored:
        track_video(info,{0:box(0)},0,99,threading.Event(),**options)
    assert restored.value.reason=='review' and restored.value.frame==66
    assert calls==list(range(1,67)) and restored.value.result.performance['review_state_rebuilds']==1
