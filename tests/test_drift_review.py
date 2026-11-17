from dataclasses import replace
import threading
import numpy as np
import pytest
from lockframe.drift_review import DriftReview
from lockframe.tracking import track_video,FeatureTracker
from lockframe.tracking_pause import TrackingPaused,computed_mask,is_complete
from test_tracking_pause import scene


@pytest.mark.parametrize('policy,limit,count',[('relaxed',.52,10),('standard',.62,6),('strict',.74,4)])
def test_evidence_hysteresis_flags_suspicion_without_inventing_lost(policy,limit,count):
    guard=DriftReview(policy);messages=[]
    for i in range(count+5):
        value=guard.observe(i,limit-.02,'tracked',[.4+i*.001,.4],[.3,.3,.2,.2])
        if value:messages.append(value)
    assert len(messages)==1 and messages[0]['frame']==count-1
    assert not messages[0]['confirmed_lost']
    assert guard.observe(50,.98,'tracked',[.5,.4],[.3,.3,.2,.2]) is None


def test_high_confidence_motion_and_manual_reference_do_not_trigger_drift():
    guard=DriftReview('strict')
    for i in range(40):
        assert guard.observe(i,.99,'tracked',[i*.05,.4],[.3,.3,.2,.2]) is None
    for i in range(3):assert guard.observe(i,.5,'tracked',[.4,.4],[.3,.3,.2,.2]) is None
    assert guard.observe(3,1.,'manual',[.4,.4],[.3,.3,.2,.2]) is None
    assert guard.observe(4,.5,'tracked',[.4,.4],[.3,.3,.2,.2]) is None


def weak_scores(monkeypatch):
    original=FeatureTracker.step;calls=[]
    def step(self,gray):
        value=original(self,gray);calls.append(int(gray[0,0]))
        if value.state!='lost':value.confidence=.6
        return value
    monkeypatch.setattr(FeatureTracker,'step',step);return calls


def test_review_stops_at_measured_frame_and_requires_manual_confirmation(scene,monkeypatch):
    info,box,_=scene;calls=weak_scores(monkeypatch)
    with pytest.raises(TrackingPaused) as raised:
        track_video(info,{0:box(0)},0,24,threading.Event(),pause_on_lost=True,drift_review='standard')
    pause=raised.value
    assert pause.reason=='review' and pause.frame==6 and calls==list(range(1,7))
    assert not pause.result.lost_frames and computed_mask(pause.result).sum()==7
    assert not is_complete(pause.result) and pause.result.performance['suspected_drift_ranges']
    calls.clear()
    with pytest.raises(TrackingPaused) as same:
        track_video(info,{0:box(0)},0,24,threading.Event(),pause_on_lost=True,drift_review='standard',
            previous=pause.result,resume=pause.resume)
    assert same.value.frame==6 and calls==[]
    # An explicit corrected/confirmed manual keyframe is a new reference.
    anchors={frame:box(frame) for frame in (0,6,12,18,24)}
    result=track_video(info,anchors,0,24,threading.Event(),pause_on_lost=True,drift_review='standard',previous=pause.result)
    assert is_complete(result) and not result.lost_frames
    assert all(result.states[frame]=='manual' for frame in anchors)


def test_warning_only_preserves_actual_trajectory_and_disabled_guard_baseline(scene,monkeypatch):
    info,box,_=scene;weak_scores(monkeypatch)
    baseline=track_video(info,{0:box(0)},0,24,threading.Event())
    warned=track_video(info,{0:box(0)},0,24,threading.Event(),drift_review='standard')
    np.testing.assert_array_equal(warned.positions,baseline.positions)
    assert warned.states==baseline.states and not warned.lost_frames
    assert warned.performance['suspected_drift_ranges']
    assert 'drift_review' not in baseline.performance


def test_review_checkpoint_survives_restart_and_policy_change_does_not_reuse_it(scene,monkeypatch):
    info,box,_=scene;calls=weak_scores(monkeypatch)
    with pytest.raises(TrackingPaused):
        track_video(info,{0:box(0)},0,24,threading.Event(),drift_review='standard',
            pause_on_lost=True,segmented_tracking=True,segment_frames=10)
    calls.clear()
    with pytest.raises(TrackingPaused) as repeated:
        track_video(info,{0:box(0)},0,24,threading.Event(),drift_review='standard',
            pause_on_lost=True,segmented_tracking=True,segment_frames=10)
    assert repeated.value.reason=='review' and calls==[]
    assert repeated.value.result.performance['suspected_drift_ranges'][0]['frame']==6
    result=track_video(info,{0:box(0)},0,24,threading.Event(),drift_review='off',
        pause_on_lost=True,segmented_tracking=True,segment_frames=10)
    assert is_complete(result) and calls==list(range(1,25))
