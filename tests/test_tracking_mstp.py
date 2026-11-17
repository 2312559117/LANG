"""Independent geometry, continuity evidence and host resume contracts."""
import threading

import cv2
import numpy as np
import pytest

from lockframe.reference_tracking import motion_features, persistent_motion, TRACKING_REVISION
from lockframe.tracking import FeatureTracker, track_video
from lockframe.models import VideoInfo, file_fingerprint
from lockframe.tracking_pause import TrackingPaused


def scene():
    rng = np.random.default_rng(821)
    patch = cv2.GaussianBlur(rng.integers(20, 240, (72, 80), np.uint8), (3, 3), .55)
    frames = []
    for i in range(18):
        matrix = np.array([[1., 0., 70+i*.7], [0., 1., 44+i*.23]])
        image = cv2.warpAffine(patch, matrix, (260, 180), borderMode=cv2.BORDER_CONSTANT)
        frames.append(image)
    return frames, [70., 44., 80., 72.]


def test_persistent_partial_affine_measures_known_subpixel_motion():
    images, box = scene()
    mask = np.zeros_like(images[0]); mask[48:112, 74:146] = 255
    points = motion_features(images[0], mask)
    new, indices, matrix = persistent_motion(images[0], images[1], points)
    mapped = np.array([110., 80., 1.])
    np.testing.assert_allclose(matrix @ mapped - mapped[:2], [.7, .23], atol=.12)
    assert len(indices) >= 12
    assert len(new) == len(points)


def test_fb_filter_rejects_reverse_inconsistent_and_nonfinite_points(monkeypatch):
    from lockframe import reference_tracking as module
    points = np.array([[20,20],[40,20],[20,40],[40,40],[30,30],[50,30],[60,20],[60,40]],np.float32)
    calls = 0
    def flow(a,b,p,initial,**kwargs):
        nonlocal calls
        assert kwargs['winSize'] == (21,21) and kwargs['maxLevel'] == 3
        calls += 1
        result = p.copy()
        if calls == 1: result += [2., 1.]
        else:
            result -= [2., 1.]
            result[4] += [1.6, 0.]
            result[5] = np.nan
        return result, np.ones((len(p),1),np.uint8), np.zeros((len(p),1),np.float32)
    monkeypatch.setattr(module.cv2,'calcOpticalFlowPyrLK',flow)
    _,indices,matrix=persistent_motion(np.zeros((80,80),np.uint8),np.zeros((80,80),np.uint8),points)
    assert 4 not in indices and 5 not in indices and len(indices)==6
    np.testing.assert_allclose(matrix,[[1,0,2],[0,1,1]],atol=1e-6)


@pytest.mark.parametrize('shape',['rectangle','ellipse','polygon'])
def test_new_detector_stays_inside_explicit_selection_mask(shape):
    from lockframe.reference_tracking import ReferencePatch
    images,box=scene()
    ref=ReferencePatch(images[0],box,shape=shape,polygon_points=((.1,.1),(.9,.2),(.6,.9)),motion_profile=True)
    xy=np.rint(ref.points).astype(int)
    assert len(xy)>=4 and len(xy)<=120
    assert np.all(ref.mask[xy[:,1],xy[:,0]]>0)
    if shape=='rectangle': assert np.all(ref.mask==255)
    else: assert np.any(ref.mask==0)


def test_continuous_measured_patch_can_change_without_matching_first_reference(monkeypatch):
    images,box=scene(); tracker=FeatureTracker(images[0],box)
    # Explicitly isolate the new adjacent-frame evidence path.
    monkeypatch.setattr(tracker,'_verified',lambda *args:None)
    observation=tracker.step(np.roll(images[0], (1,2), axis=(0,1)))
    assert observation.state=='tracked' and observation.confidence>0
    np.testing.assert_allclose(observation.box[:2]+observation.box[2:]/2,[112.,81.],atol=.2)
    assert tracker._pending_motion is None


def test_v2_reappearance_requires_independent_evidence_after_occlusion_gap(monkeypatch):
    images,box=scene();tracker=FeatureTracker(images[0],box)
    monkeypatch.setattr(tracker,'_verified',lambda *args:None)
    observation=tracker.step(np.zeros_like(images[0]))
    assert observation.state=='lost' and observation.confidence==0
    observation=tracker.step(images[2])
    assert observation.state=='lost' and observation.confidence==0


def test_measured_points_are_retained_only_for_the_accepted_pose():
    images,box=scene();tracker=FeatureTracker(images[0],box)
    ids=np.arange(len(tracker.points)); projected=tracker.points.copy()
    measured=projected+[.13,-.17]
    tracker._pending_motion=(ids,measured,tracker.matrix.copy())
    tracker._bind_points(ids,tracker.matrix,.95)
    np.testing.assert_allclose(tracker.points,measured,atol=1e-5)
    # A different recovery winner must not inherit a rejected flow bank.
    tracker._pending_motion=(ids,measured,tracker.matrix.copy())
    other=tracker.matrix.copy();other[:,2]+=[3,1]
    tracker._bind_points(ids,other,.95)
    np.testing.assert_allclose(tracker.points,projected+[3,1],atol=1e-5)


def test_actual_point_bank_roundtrips_snapshot():
    from lockframe.checkpoints import tracker_snapshot,restore_tracker
    images,box=scene();a=FeatureTracker(images[0],box)
    for image in images[1:8]:assert a.step(image).state!='lost'
    arrays,metadata=tracker_snapshot(a)
    b=FeatureTracker(images[0],box);restore_tracker(b,arrays,metadata,images[7])
    for image in images[8:]:
        oa,ob=a.step(image),b.step(image)
        assert oa.state==ob.state
        np.testing.assert_allclose(oa.box,ob.box,atol=1e-9)
        np.testing.assert_array_equal(a.points,b.points)


@pytest.fixture
def pipeline(tmp_path,monkeypatch):
    from lockframe import performance
    images,box=scene();p=tmp_path/'source.bin';p.write_bytes(b'independent synthetic source')
    info=VideoInfo(str(p),260,180,30.,len(images)/30,frames=len(images),proxy=str(p),fingerprint=file_fingerprint(str(p)))
    class Reader:
        peak_bytes=hits=0
        def __init__(self,*args):pass
        def read(self,index):return images[index]
        def iter_frames(self,indices):
            for i in indices:yield i,images[i]
        def close(self):pass
    monkeypatch.setattr(performance,'GrayFrameCache',Reader)
    monkeypatch.setenv('LOCKFRAME_CACHE_DIR',str(tmp_path/'cache'))
    return info,{8:((70+8*.7)/260,(44+8*.23)/180,80/260,72/180)}


def test_bidirectional_pause_resume_and_old_algorithm_invalidation(pipeline,monkeypatch):
    info,anchors=pipeline
    expected=track_video(info,anchors,0,17,threading.Event())
    assert not expected.lost_frames
    pause=threading.Event()
    def observed(value):
        if value['frame']==4:pause.set()
    with pytest.raises(TrackingPaused) as exc:
        track_video(info,anchors,0,17,threading.Event(),pause_event=pause,on_frame=observed)
    pause.clear();paused=exc.value
    resumed=track_video(info,anchors,0,17,threading.Event(),previous=paused.result,resume=paused.resume)
    assert resumed.states==expected.states
    np.testing.assert_allclose(resumed.positions,expected.positions,atol=1e-10)
    expected.performance['tracking_revision']='trackstable-robust-v1'
    calls=[];step=FeatureTracker.step
    def counted(self,image):calls.append(1);return step(self,image)
    monkeypatch.setattr(FeatureTracker,'step',counted)
    fresh=track_video(info,anchors,0,17,threading.Event(),previous=expected)
    assert len(calls)==17 and fresh.performance['reused_frames']==0
    assert fresh.performance['tracking_revision']==TRACKING_REVISION


def test_plain_v2_reads_source_while_planar_keeps_proxy(pipeline,monkeypatch):
    from lockframe import performance
    info,anchors=pipeline
    info.proxy=info.path+'.proxy'
    base=performance.GrayFrameCache;opened=[]
    class CapturingReader(base):
        def __init__(self,path,*args):
            opened.append(path);super().__init__(path,*args)
    monkeypatch.setattr(performance,'GrayFrameCache',CapturingReader)
    normal=track_video(info,anchors,0,17,threading.Event())
    assert opened.pop(0)==info.path
    assert normal.performance['backend']=='CPU 光流 · MS v2 源帧'
    track_video(info,anchors,0,17,threading.Event(),planar_region=True)
    assert opened.pop(0)==info.proxy


def test_incompatible_source_retains_cfr_proxy_and_v2_motion(pipeline,monkeypatch):
    from lockframe import performance, tracking
    info,anchors=pipeline
    info.proxy=info.path+'.proxy'
    base=performance.GrayFrameCache;opened=[];closed=[]
    class CapturingReader(base):
        def __init__(self,path,*args):
            self.path=path;opened.append(path);super().__init__(path,*args)
        def close(self):
            closed.append(self.path);super().close()
    monkeypatch.setattr(performance,'GrayFrameCache',CapturingReader)
    monkeypatch.setattr(tracking,'_mstrack_source_compatible',lambda *args:False)
    result=track_video(info,anchors,0,17,threading.Event())
    assert opened==[info.path,info.proxy] and closed==opened
    assert result.performance['backend']=='CPU 光流 · MS v2 CFR兼容'
    assert not result.lost_frames


def test_disk_checkpoint_preserves_new_point_bank(pipeline,monkeypatch):
    from lockframe import performance
    from lockframe.media import Cancelled
    info,anchors=pipeline; expected=track_video(info,anchors,0,17,threading.Event())
    iterator=performance.GrayFrameCache.iter_frames;cancel=threading.Event()
    def interrupted(self,indices):
        for index,image in iterator(self,indices):
            if index==11:cancel.set()
            yield index,image
    monkeypatch.setattr(performance.GrayFrameCache,'iter_frames',interrupted)
    with pytest.raises(Cancelled):track_video(info,anchors,0,17,cancel,segmented_tracking=True,segment_frames=10)
    monkeypatch.setattr(performance.GrayFrameCache,'iter_frames',iterator)
    actual=track_video(info,anchors,0,17,threading.Event(),segmented_tracking=True,segment_frames=10)
    assert actual.performance['resumed_frames']>0
    np.testing.assert_allclose(actual.positions,expected.positions,atol=1e-10)
