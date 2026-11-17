from copy import deepcopy
from pathlib import Path
import json,threading,uuid
import pytest
from test_retime_integration import window,demo_info,tracked_clip,activate,contract_clip


@pytest.mark.parametrize('lost',[False,True])
def test_batch_continue_after_manual_pause_keeps_anchors_and_honors_lost(window,demo_info,monkeypatch,lost):
    from unittest.mock import MagicMock
    from lockframe import sequence_ui
    from lockframe.tracking_pause import TrackingPaused
    clips=[tracked_clip(demo_info),tracked_clip(demo_info)]
    for clip in clips:
        clip.track.performance.update(complete=False,computed_frames=[i<=5 for i in range(36)],
            tracking_pause=dict(reason='manual',frame=5,direction=1))
    if lost:clips[1].track.states[5]='lost'
    activate(window,clips)
    for clip in clips:window._paused_by_clip[clip.id]=window._pause_context_for(TrackingPaused(clip.track,5,1,'manual'))
    window._paused_tracking=window._paused_by_clip[clips[0].id]
    window._paused_queue_ids=[clip.id for clip in clips]
    anchors=[dict(clip.anchors) for clip in clips];factory=MagicMock();monkeypatch.setattr(sequence_ui,'QueueJob',factory)
    try:
        window._continue_tracking()
        assert factory.call_count==(0 if lost else 1)
        assert [clip.anchors for clip in clips]==anchors
        if not lost:factory.return_value.start.assert_called_once()
    finally:
        window.worker=None;window.busy=False;window._tracking_active=False


def test_recovery_rebinds_already_open_editor_to_recovered_clip(window,demo_info,tmp_path):
    from lockframe.time_remap_model import make_constant_speed
    clip=tracked_clip(demo_info,retime=make_constant_speed(None,.5));activate(window,[clip])
    window._open_retime_editor();old=window._retime_editor
    restored=deepcopy(clip);restored.id=uuid.uuid4().hex;draft=make_constant_speed(None,2.)
    payload=dict(clips=[restored],folder=str(tmp_path/'recovery'),view=dict(active_index=0,
        editor=dict(config=draft,context=[clip.info.path,clip.info.fingerprint,0,35])))
    window._workspace_loaded(payload);window._activate_clip(restored.id);window._apply_restored_workspace_view()
    editor=window._retime_editor
    assert editor is not old and editor._workspace_context[0]==restored.id and editor.draft==draft
    applied=make_constant_speed(None,3.);editor.applied.emit(applied)
    assert window.settings().retime==applied and window.track is restored.track


@pytest.mark.parametrize('bad',[
    {'time':'broken'}, {'time':float('nan')}, {'time':1e100}, {'names':[None]}, {'count':'2'},
    {'file':'../../outside.locksequence'}, {'view':[]}, {'sha256':'invalid'},
])
def test_malformed_index_metadata_falls_back_without_breaking_listing(contract_clip,tmp_path,bad):
    from lockframe.workspace_autosave import write_workspace,list_workspaces,recover_workspace
    folder=tmp_path/uuid.uuid4().hex;cancel=threading.Event()
    write_workspace([contract_clip],{'generation':1},folder,cancel)
    write_workspace([contract_clip],{'generation':2},folder,cancel)
    path=folder/'manifest.json';data=json.loads(path.read_text(encoding='utf-8'));data['latest'].update(bad)
    path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    listed=list_workspaces(tmp_path);assert len(listed)==1 and listed[0]['latest']['view']['generation']==1
    clips,view,_=recover_workspace(folder);assert len(clips)==1 and view['generation']==1


def test_new_save_keeps_good_backup_when_previous_latest_file_is_damaged(contract_clip,tmp_path):
    from lockframe.workspace_autosave import write_workspace,recover_workspace
    folder=tmp_path/uuid.uuid4().hex;cancel=threading.Event()
    write_workspace([contract_clip],{'generation':1},folder,cancel)
    write_workspace([contract_clip],{'generation':2},folder,cancel)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    (folder/manifest['latest']['file']).write_bytes(b'damaged')
    write_workspace([contract_clip],{'generation':3},folder,cancel)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    (folder/manifest['latest']['file']).write_bytes(b'damaged again')
    _,view,_=recover_workspace(folder);assert view['generation']==1


def test_actual_batch_worker_resumes_manual_pause_without_correction(window,demo_info):
    from lockframe.workflow_complete_check import exercise_batch_resume
    assert exercise_batch_resume(window,demo_info)['no_extra_anchors']
