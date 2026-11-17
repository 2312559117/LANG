from copy import deepcopy
import json
import threading
import uuid
import numpy as np
import pytest
from lockframe.demo import demo_frame
from lockframe.models import read_project, save_project, Settings, TrackResult, VideoInfo
from lockframe.sequence import Clip, save_sequence, read_sequence
from lockframe.section_workflow import tracking_ranges, sections_match
from lockframe.v072_check import activate, wait_job, create_correction_source, correction_box
from test_workflow_v9 import source, window


def test_split_pick_and_real_tracking_is_saved_per_clip(window, source, tmp_path):
    clip, other = deepcopy(source), deepcopy(source)
    other.id = uuid.uuid4().hex
    activate(window, [clip, other])
    window._open_tracking_sections()
    dialog = window._sections_panel
    window._seek(18)
    dialog.split()
    assert window.tracking_sections == [{'start':0,'end':17},{'start':18,'end':35}]
    assert window.section_tracking and window.track is None
    assert dialog.selector.count()==2
    dialog.selector.setCurrentIndex(1)
    dialog.selector.setCurrentIndex(1)
    dialog.pick()
    assert window.frame_index==18 and dialog.isVisible() and not dialog.isWindow()
    window._select_roi(demo_frame(18,clip.info.width,clip.info.height)[1])
    window._start_tracking();wait_job(window)
    assert window.track is not None and not window.track.lost_frames
    assert window.track.performance['tracking_sections']==window.tracking_sections
    assert window.strip.tracking_sections==window.tracking_sections
    window._stash_clip()
    assert clip.section_tracking and not other.section_tracking
    assert other.track is not None
    window._activate_clip(other.id)
    assert not window.section_tracking
    window._activate_clip(clip.id)
    assert window.section_tracking and len(window.tracking_sections)==2
    project=tmp_path/'独立分段.lockframe';window.project_path=str(project)
    assert window._save_project()
    payload=read_project(str(project))
    assert payload['section_tracking'] and payload['tracking_sections']==clip.tracking_sections
    window._confirm_discard=lambda:True
    window.load_video(clip.info.path,project=payload);wait_job(window)
    assert window.section_tracking and window.tracking_sections==clip.tracking_sections
    assert window.track is not None and window.plan is not None


def test_io_drafts_overlap_and_missing_seed_do_not_start_job(window, source, monkeypatch):
    activate(window,[deepcopy(source)]);window._open_tracking_sections()
    dialog=window._sections_panel
    window._seek(0);dialog.mark_in();window._seek(11);dialog.mark_out()
    assert window.tracking_sections==[{'start':0,'end':11}]
    old=deepcopy(window.tracking_sections)
    window._seek(4);dialog.mark_in();window._seek(19);dialog.mark_out()
    assert window.tracking_sections==old
    calls=[];monkeypatch.setattr(window,'_job',lambda *a,**k:calls.append(k))
    window._start_tracking()
    assert not calls
    assert '覆盖' in dialog.status.text()
    window._seek(12);dialog.mark_in();window._seek(35);dialog.mark_out()
    window._start_tracking()
    assert not calls and '尚未设置' in dialog.status.text()


def test_sequence_persists_drafts_and_legacy_defaults(source,tmp_path):
    clip=deepcopy(source);clip.section_tracking=True
    clip.tracking_sections=[{'start':0,'end':10},{'start':15,'end':35}]
    target=tmp_path/'分段.locksequence';save_sequence(target,[clip])
    restored=read_sequence(target)[0]
    assert restored.section_tracking and restored.tracking_sections==clip.tracking_sections
    assert not restored.ready
    raw=json.loads(target.read_text(encoding='utf-8'))
    raw['clips'][0].pop('section_tracking');raw['clips'][0].pop('tracking_sections')
    target.write_text(json.dumps(raw),encoding='utf-8')
    legacy=read_sequence(target)[0]
    assert not legacy.section_tracking and legacy.tracking_sections==[]
    raw['clips'][0].update(section_tracking=True,tracking_sections=[{'start':0,'end':20},{'start':20,'end':35}])
    target.write_text(json.dumps(raw),encoding='utf-8')
    with pytest.raises(ValueError,match='重叠'):read_sequence(target)


def test_declared_appearance_boundary_avoids_old_lost_and_batch_uses_sections(window,tmp_path):
    info=create_correction_source(tmp_path/'source')
    clip=Clip(info,Settings(acceleration='cpu',compute_threads=2),{0:correction_box(0),12:correction_box(12)},0,47,
              section_tracking=True,tracking_sections=[{'start':0,'end':11},{'start':12,'end':47}])
    activate(window,[clip]);window._start_tracking();wait_job(window)
    assert window._paused_tracking is None and not window.track.lost_frames
    assert window.track.states[12]=='manual'
    window._stash_clip()
    other=deepcopy(clip);other.id=uuid.uuid4().hex;other.track=None
    clip.track=None
    activate(window,[clip,other]);window._refresh_sequence({clip.id,other.id})
    window.queue_mode.setCurrentIndex(window.queue_mode.findData(2))
    window._start_queue();wait_job(window,120)
    assert all(c.track is not None and not c.track.lost_frames for c in (clip,other))
    assert all(c.track.performance['tracking_sections']==c.tracking_sections for c in (clip,other))


def test_section_edits_invalidate_exports_and_toggle_preserves_drafts(window,source,monkeypatch,tmp_path):
    from lockframe.batch_export import validate_clip
    clip=deepcopy(source);activate(window,[clip])
    window._set_tracking_sections(True,[{'start':0,'end':35}])
    assert window.track is None and not window.export_button.isEnabled()
    window.track=source.track;window._stash_clip();window._sync_ui()
    assert window._sections_stale() and not window.export_button.isEnabled()
    with pytest.raises(ValueError,match='分段'):validate_clip(clip)
    errors=[];monkeypatch.setattr(window,'_error',errors.append)
    window._export();assert errors and '分段' in errors[0]
    draft=deepcopy(window.tracking_sections);window._set_tracking_sections(False,draft)
    assert not window.section_tracking and window.tracking_sections==draft


def test_follow_smoothing_does_not_cross_section_boundary():
    from lockframe.stabilization import build_plan
    positions=np.array([[.3,.5]]*12+[[.7,.5]]*12)
    boxes=np.c_[positions-[.05,.05],np.full((24,2),.1)]
    track=TrackResult(0,23,positions,boxes,np.ones(24),['tracked']*24,
                      performance={'tracking_sections':[{'start':0,'end':11},{'start':12,'end':23}]})
    info=VideoInfo('synthetic',400,240,24.,1.,frames=24)
    plan=build_plan(info,track,Settings(mode='follow',resolution=0,max_zoom=4))
    assert np.ptp(plan.matrices[:,:,2],axis=0).max()<1e-6


def test_trim_intersects_sections_and_reset_keeps_drafts_but_clears_targets(window,source):
    clip=deepcopy(source);activate(window,[clip])
    sections=[{'start':0,'end':17},{'start':18,'end':35}]
    window._set_tracking_sections(True,sections)
    window._seek(20);window._select_roi(demo_frame(20,clip.info.width,clip.info.height)[1])
    window._start_tracking();wait_job(window)
    assert sections_match(window.track,True,sections,3,30)
    window._apply_trim(3,30);window._stash_clip()
    assert window.track.start==3 and window.track.end==30
    assert sections_match(window.track,True,clip.tracking_sections,3,30)
    window._reset_current_tracking()
    assert not window.anchors and window.track is None and window.tracking_sections==sections
    assert not clip.ready


def test_forced_process_exit_recovers_last_atomic_checkpoint(source,tmp_path,monkeypatch):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    from lockframe.media import import_video
    from lockframe.tracking import track_video
    from lockframe.performance import ComputeOptions
    cache=tmp_path/'crash-cache';marker=tmp_path/'ready-to-kill'
    monkeypatch.setenv('LOCKFRAME_CACHE_DIR',str(cache))
    info=source.info
    anchors={f:demo_frame(f,info.width,info.height)[1] for f in (0,18)}
    sections=[{'start':0,'end':17},{'start':18,'end':35}]
    script='''import json,sys,threading
from pathlib import Path
from lockframe.media import import_video
from lockframe.tracking import track_video
from lockframe.performance import ComputeOptions
video,anchors,sections,marker=json.loads(sys.argv[1])
info=import_video(video,threading.Event())
def observed(value):
 if value['frame']==23:
  Path(marker).write_text('checkpoint committed before this frame')
  threading.Event().wait(30)
track_video(info,{int(k):tuple(v) for k,v in anchors.items()},0,35,threading.Event(),
 tracking_sections=sections,segmented_tracking=True,segment_frames=10,
 resources=ComputeOptions('cpu',80,2),on_frame=observed)
'''
    with (tmp_path/'child.txt').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-c',script,json.dumps([info.path,anchors,sections,str(marker)])],
                                 cwd=Path(__file__).resolve().parents[1],env=os.environ.copy(),
                                 stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            deadline=time.monotonic()+20
            while not marker.exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            assert marker.exists(),(tmp_path/'child.txt').read_text(errors='replace')
            process.kill();process.wait(timeout=10)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
    kwargs=dict(tracking_sections=sections,resources=ComputeOptions('cpu',80,2))
    recovered=track_video(info,anchors,0,35,threading.Event(),segmented_tracking=True,segment_frames=10,**kwargs)
    assert recovered.performance['resumed_frames']>=19 and not recovered.lost_frames
    fresh=track_video(info,anchors,0,35,threading.Event(),**kwargs)
    np.testing.assert_allclose(recovered.positions,fresh.positions,atol=1e-10)
