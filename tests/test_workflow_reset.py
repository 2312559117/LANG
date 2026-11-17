from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import uuid
import pytest

from lockframe.demo import demo_frame
from lockframe.models import save_project, read_project
from lockframe.sequence import save_sequence, read_sequence
from lockframe.tracking_pause import TrackingPaused
from lockframe.v072_check import activate
from test_workflow_v9 import source, window


def clips_from(source):
    first, second = deepcopy(source), deepcopy(source)
    second.id = uuid.uuid4().hex
    return first, second


def test_reset_current_completed_clip_preserves_parameters_other_clip_and_saved_empty_state(window,source,tmp_path,monkeypatch):
    first, other = clips_from(source)
    first.settings=replace(first.settings,target_x=.38,aspect="9:16")
    activate(window,[first,other]);window._seek(17)
    settings=window.settings()
    other_track,other_anchors=other.track,dict(other.anchors)
    window._refresh_sequence({first.id,other.id}) # Multi-selection must still reset only active.
    window.reset_tracking_button.click()
    assert not window.anchors and window.track is window._retrack_previous is window._paused_tracking is None
    assert window.plan is None
    assert window.canvas.box is None and window.canvas.processed is None
    assert not window.track_button.isEnabled()
    assert window.continue_button.isVisible() and not window.continue_button.isEnabled() and window.track_button.text()=="开始跟踪"
    assert window.settings()==settings and window.frame_index==17
    assert (first.start,first.end)==(0,35) and first.dirty and first.project_path is None
    assert other.track is other_track and other.anchors==other_anchors and not other.dirty
    assert first.track is first.retrack_previous is first.tracking_resume is None
    window._activate_clip(other.id);window._activate_clip(first.id)
    assert not window.anchors and window.track is window._retrack_previous is None
    project=tmp_path/"重置后.lockframe"
    save_project(str(project),first.info,first.settings,first.anchors,first.start,first.end,first.track)
    payload=read_project(str(project))
    assert payload["anchors"]=={} and payload["track"] is None
    sequence=tmp_path/"重置后.locksequence"
    save_sequence(sequence,[first,other])
    restored=read_sequence(sequence)
    assert not restored[0].anchors and restored[0].track is None and restored[1].track is not None
    window._seek(20)
    window._select_roi(demo_frame(20,first.info.width,first.info.height)[1])
    calls=[]
    monkeypatch.setattr(window,"_job",lambda *args,**kwargs:calls.append(kwargs))
    window._start_tracking()
    assert len(calls)==1 and calls[0]["previous"] is None and calls[0]["resume"] is None


@pytest.mark.parametrize("reason",["lost","manual"])
def test_reset_paused_clip_clears_only_its_queue_gate(window,source,reason):
    first,other=clips_from(source)
    activate(window,[first,other])
    partial=deepcopy(first.track)
    partial.performance.update(complete=False,computed_frames=[i<=12 for i in range(partial.count)],
                               tracking_pause=dict(frame=12,direction=1,reason=reason))
    if reason=="lost":partial.states[12]="lost"
    window._tracking_paused(TrackingPaused(partial,12,1,reason,resume={"old_tracker":True}))
    other_context={"corrected":False,"reason":"lost","frame":22}
    window._paused_by_clip[other.id]=other_context
    window._paused_queue_ids=[first.id,other.id]
    window._sync_ui()
    window.reset_tracking_button.click()
    assert first.id not in window._paused_by_clip and window._paused_tracking is None
    assert window._paused_by_clip[other.id] is other_context
    assert window._paused_queue_ids==[other.id]
    assert not first.anchors and first.track is first.retrack_previous is first.tracking_resume is None
    assert not window._resume_requested and window.continue_button.isVisible() and not window.continue_button.isEnabled()


@pytest.mark.parametrize("kind",["tracking","queue","export","finishing"])
def test_reset_disabled_and_guarded_while_a_worker_can_write_back(window,source,kind):
    activate(window,[deepcopy(source)])
    result=window.track;anchors=dict(window.anchors)
    window.busy=kind!="finishing";window._job_kind=kind;window.worker=SimpleNamespace(supports_pause=True)
    try:
        window._sync_ui()
        assert not window.reset_tracking_button.isEnabled()
        window.reset_tracking_button.click();window._reset_current_tracking()
        assert window.track is result and window.anchors==anchors
    finally:
        window.busy=False;window.worker=None;window._job_kind=None
        window._sync_ui()


def test_reset_cache_error_reports_failure_and_keeps_visible_targets(window,source,monkeypatch):
    activate(window,[deepcopy(source)])
    result=window.track;anchors=dict(window.anchors);errors=[]
    monkeypatch.setattr(window,"_error",errors.append)
    def fail(clip):raise OSError("locked cache")
    monkeypatch.setattr("lockframe.tracking_pause.reset_clip_tracking",fail)
    window.reset_tracking_button.click()
    assert window.track is result and window.anchors==anchors and errors
    assert "locked cache" in errors[0] and window._clip().track is result


def test_reset_empty_clip_is_safe_and_keeps_new_selection_manual(window,source):
    first=deepcopy(source);first.anchors={};first.track=None
    activate(window,[first])
    window.reset_tracking_button.click();window.reset_tracking_button.click()
    assert window.worker is None and not window.busy and not window.anchors
    assert window.pick_mode.currentData()=="box" and window.track is None


def test_real_reset_reopen_new_tracking_and_export(tmp_path):
    from lockframe.reset_check import check_reset
    report=check_reset(tmp_path)
    assert report["all_targets_cleared"] and report["new_run_reused_frames"]==0


def test_no_material_reset_is_disabled_and_safe(window):
    assert not window.reset_tracking_button.isEnabled()
    window._reset_current_tracking()
    assert window.info is None and not window.anchors and not window.busy
