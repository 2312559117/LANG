"""Frozen acceptance on generated pixels; no real drift accuracy claim."""
from pathlib import Path
import threading
from unittest.mock import patch


def check_consistency(folder):
    import cv2,numpy as np
    from .drift_review import DriftReview
    from .media import import_video
    from .tracking import FeatureTracker,track_video
    from .tracking_pause import TrackingPaused
    monitor=DriftReview('standard');first_warning=None;first_review=None
    for index in range(100):
        score=.96 if index<8 else max(.8,.96-(index-7)*.002)
        value=monitor.observe(index,score,'tracked',[.5,.5],[.3,.3,.4,.4])
        if first_warning is None and monitor.new_warning:first_warning=index
        if first_review is None and value:first_review=index
    assert first_warning is not None and first_review>first_warning and monitor.baseline==.96
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    source=folder/'续跑一致性_合成静止.avi'
    rng=np.random.default_rng(732);frame=np.full((180,320,3),25,np.uint8)
    frame[60:124,100:164]=cv2.GaussianBlur(rng.integers(20,235,(64,64,3),np.uint8),(3,3),.6)
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),30.,(320,180))
    if not writer.isOpened():raise RuntimeError('无法生成本地续跑验收源')
    try:
        for _ in range(96):writer.write(frame)
    finally:writer.release()
    cancel=threading.Event();info=import_video(str(source),cancel)
    anchor=(100/320,60/180,.2,64/180);original=FeatureTracker.step;counter=0;calls=[]
    def scored(self,gray):
        nonlocal counter
        counter+=1;calls.append(counter);value=original(self,gray)
        if value.state!='lost':value.confidence=.96 if counter<=8 else .88 if counter<=60 else .83
        return value
    def execute(**kwargs):
        try:
            track_video(info,{0:anchor},0,95,cancel,drift_review='standard',pause_on_lost=True,**kwargs)
            raise AssertionError('controlled degradation did not suspend')
        except TrackingPaused as value:return value
    with patch.object(FeatureTracker,'step',scored):
        baseline=execute();assert baseline.reason=='review' and baseline.frame==66
        counter=0;pause=threading.Event()
        def observe(payload):
            if payload['frame']==60:pause.set()
        saved=execute(pause_event=pause,on_frame=observe,segmented_tracking=True,segment_frames=10)
        assert saved.reason=='manual' and saved.frame==60 and saved.resume['review_monitor']['baseline']==.96
        counter=60;calls.clear();events=[]
        restored=execute(segmented_tracking=True,segment_frames=10,on_frame=events.append)
        assert restored.reason=='review' and restored.frame==66 and calls==list(range(61,67))
        assert restored.result.performance['review_state_restores']==1 and not restored.result.lost_frames
        np.testing.assert_array_equal(restored.result.positions[:67],baseline.result.positions[:67])
        warnings=[p['frame'] for p in events if p.get('drift_warning')]
        assert warnings[0]==63
    return dict(slow_decline_warning_frame=first_warning,slow_decline_review_frame=first_review,
        held_baseline=.96,manual_pause_frame=60,resumed_warning_frame=63,resumed_review_frame=66,
        exact_disk_resume=True,numeric_measurements_equal=True,
        data='generated static pixels with controlled quality scores; not real drift labels')
