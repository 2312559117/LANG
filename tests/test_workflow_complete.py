from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib,json,threading,time,uuid
import numpy as np
import pytest
from test_retime_integration import contract_clip,window,demo_info,tracked_clip,activate
from test_native_flow import frame


def old_synthesis():
    path=Path(__file__).parent/'fixtures/synthesis_v0732.py.txt';text=path.read_bytes()
    assert hashlib.sha256(text).hexdigest()=='1cc7f4e04af1c3afc53312b5da3ca8852e0912821ae6be30e0903cba52efb971'
    values={'__name__':'lockframe.flow.reference','__package__':'lockframe.flow'}
    exec(compile(text,str(path),'exec'),values);return values['synthesize']


@pytest.mark.parametrize('fmt',['yuv420p','yuv422p','yuv444p','yuv420p10le','yuv422p12le','yuv444p16le','nv12','p010le'])
@pytest.mark.parametrize('grid',[(17,11),(5,7)])
def test_grouped_synthesis_and_preparation_are_byte_exact(fmt,grid):
    from lockframe.flow.synthesis import synthesize,SynthesisPreparation
    left,right=frame(fmt,17,11),frame(fmt,17,11,52);rng=np.random.default_rng(733)
    f=rng.normal(0,1.5,(*grid[::-1],2)).astype(np.float32);b=-f
    reference=old_synthesis();prepared=SynthesisPreparation(left,right,f,b,cache_float=True)
    for alpha in (0.,.13,.5,.81,1.):
        expected=reference(left,right,f,b,alpha,tile_rows=3).to_bytes()
        assert synthesize(left,right,f,b,alpha,tile_rows=3).to_bytes()==expected
        assert synthesize(left,right,f,b,alpha,tile_rows=3,_prepared=prepared).to_bytes()==expected


def test_prepared_pair_cache_counts_float_memory_and_rejects_foreign_inputs():
    from lockframe.flow.base import create_interpolator
    from lockframe.flow.synthesis import synthesize,FlowError
    a,b=frame(),frame(seed=51)
    with create_interpolator('dis',max_cache_bytes=256000,max_pairs=2) as flow:
        pair=flow.prepare(a,b);assert pair.preparation.nbytes>0
        assert pair.nbytes==pair.left.nbytes+pair.right.nbytes+pair.forward.nbytes+pair.backward.nbytes+pair.preparation.nbytes
        assert flow.stats['cache_bytes']<=flow.max_cache_bytes
        with pytest.raises(FlowError):synthesize(a,b,pair.forward,pair.backward,.5,_prepared=pair.preparation)


def test_manual_pause_continues_without_creating_anchor_but_lost_does_not(window,demo_info):
    from lockframe.tracking_pause import TrackingPaused,can_resume_without_edit
    from unittest.mock import patch
    clip=tracked_clip(demo_info,0,35);result=clip.track
    result.performance.update(complete=False,computed_frames=[i<=5 for i in range(36)],tracking_pause=dict(reason='manual',frame=5,direction=1))
    activate(window,[clip]);window._tracking_paused(TrackingPaused(result,5,1,'manual'));window._sync_ui()
    anchors=deepcopy(window.anchors)
    assert window.continue_button.isEnabled() and can_resume_without_edit(window._paused_tracking)
    with patch.object(window,'_start_tracking') as start:window._continue_tracking();assert start.call_count==1
    assert window.anchors==anchors
    result.states[5]='lost';window._paused_tracking['reason']='manual';window._sync_ui()
    assert not window.continue_button.isEnabled() and not can_resume_without_edit(window._paused_tracking)


def test_autosave_freezes_all_clips_and_falls_back_to_previous_generation(contract_clip,tmp_path):
    from lockframe.workspace_autosave import freeze_clips,write_workspace,recover_workspace,_manifest
    first=contract_clip;second=deepcopy(first);second.id=uuid.uuid4().hex;second.settings.target_x=.7
    folder=tmp_path/'recovery'/uuid.uuid4().hex;cancel=threading.Event();snapshot=freeze_clips([first,second])
    first.settings.retime['enabled']=False;first.track.positions[:]=0
    write_workspace(snapshot,dict(active_index=1,clock='output',output_step=7),folder,cancel)
    clips,view,_=recover_workspace(folder)
    assert clips[0].settings.retime['enabled'] and clips[0].track.positions.sum()>0 and clips[1].settings.target_x==.7
    assert view['active_index']==1 and view['output_step']==7
    snapshot[0].settings.target_x=.2;write_workspace(snapshot,dict(active_index=0),folder,cancel)
    manifest=_manifest(folder);(folder/manifest['latest']['file']).write_bytes(b'corrupt')
    recovered,view,_=recover_workspace(folder)
    assert recovered[0].settings.target_x!=.2 and view['active_index']==1


def test_autosave_cancel_failure_and_quota_keep_committed_workspace(contract_clip,tmp_path,monkeypatch):
    from lockframe import workspace_autosave as module
    from lockframe.media import Cancelled
    folder=tmp_path/'recovery'/uuid.uuid4().hex;cancel=threading.Event()
    module.write_workspace([contract_clip],{},folder,cancel);before=(folder/'manifest.json').read_bytes();source=Path(contract_clip.info.path).read_bytes()
    original=module.save_sequence
    def canceled(path,clips):original(path,clips);cancel.set()
    monkeypatch.setattr(module,'save_sequence',canceled)
    with pytest.raises(Cancelled):module.write_workspace([contract_clip],{'changed':True},folder,cancel)
    assert (folder/'manifest.json').read_bytes()==before and len(list(folder.glob('*.locksequence')))==1
    cancel.clear();monkeypatch.setattr(module,'save_sequence',original)
    def fail(*_):raise OSError('磁盘写入失败')
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(OSError):module.write_workspace([contract_clip],{'changed':True},folder,cancel)
    assert (folder/'manifest.json').read_bytes()==before and len(list(folder.glob('*.locksequence')))==1
    monkeypatch.setattr(module,'MAX_DISK',1)
    with pytest.raises(OSError,match='上限'):module.write_workspace([contract_clip],{},folder,cancel)
    assert Path(contract_clip.info.path).read_bytes()==source


def test_recovery_deletion_is_confined_and_unknown_files_are_kept(contract_clip,tmp_path):
    from lockframe.workspace_autosave import write_workspace,remove_workspace,list_workspaces,mark_workspace
    root=tmp_path/'recovery';folder=root/uuid.uuid4().hex
    write_workspace([contract_clip],{},folder,threading.Event());assert list_workspaces(root)
    mark_workspace(folder,clean=True);assert list_workspaces(root)[0]['clean']
    with pytest.raises(ValueError):remove_workspace(root,tmp_path)
    with pytest.raises(ValueError):remove_workspace(root,folder,current=folder)
    (folder/'keep.mp4').write_bytes(b'original')
    with pytest.raises(ValueError):remove_workspace(root,folder)
    assert (folder/'keep.mp4').read_bytes()==b'original'
    (folder/'keep.mp4').unlink();remove_workspace(root,folder);assert not folder.exists()


def test_visual_presets_keep_render_options_and_custom_roundtrip(tmp_path):
    from lockframe.curve_presets import builtins,apply_curve,PresetStore
    from lockframe.time_remap_model import default_config
    from lockframe.time_remap_map import output_frame_map
    original=default_config();original.update(enabled=True,optical_flow=True,audio_policy='mapped')
    for preset in builtins():
        value=apply_curve(original,preset['curve'])
        assert value['optical_flow'] and value['audio_policy']=='mapped' and value['enabled']
        assert np.isfinite(output_frame_map(value,120)).all()
    store=PresetStore(tmp_path/'预设 中文.json');identifier=store.save('我的脉冲',apply_curve(original,builtins()[3]['curve']))
    saved=store.read();assert saved[0]['id']==identifier and saved[0]['name']=='我的脉冲'
    with pytest.raises(ValueError):store.save('我的脉冲',original)
    store.remove(identifier);assert store.read()==[]


@pytest.mark.parametrize('mode',['basic','advanced'])
def test_playhead_and_alt_seek_do_not_edit_curve_or_tracking(window,demo_info,mode):
    from lockframe.time_remap_model import make_constant_speed
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    config=make_constant_speed(None,.5);config['mode']=mode
    clip=tracked_clip(demo_info,0,35,config);activate(window,[clip])
    before=deepcopy(window.settings().retime);track=window.track
    window._seek_output(12);assert window.retime_panel.plot.cursor_x is not None
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    plot=window.retime_panel.plot;assert plot.width()>100
    point=plot.pixel(.2,5).toPoint()
    QTest.mouseClick(plot,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.AltModifier,pos=point)
    expected=7 if mode=='basic' else 3
    assert window.settings().retime==before and window.track is track and abs(window.frame_index-expected)<=1
    window._open_retime_editor();editor=window._retime_editor;draft=deepcopy(editor.draft)
    window._seek_output(18);assert editor.canvas.cursor_x is not None and editor.draft==draft


def test_review_marks_persist_and_true_lost_cannot_be_acknowledged(window,demo_info,tmp_path):
    from lockframe.review_records import collect_issues
    from lockframe.sequence import save_sequence,read_sequence
    clip=tracked_clip(demo_info,0,35);clip.track.performance.update(early_warning_ranges=[dict(start=2,end=4,frame=4,confidence=.6)],quality={})
    clip.track.states[6]='lost';activate(window,[clip]);window._open_review_workbench();dialog=window._review_dialog
    i=next(i for i,item in enumerate(dialog.rows) if item.kind=='跟偏预警');dialog.table.selectRow(i)
    positions=window.track.positions.copy();dialog.set_checked(True)
    assert clip.review_marks and np.array_equal(positions,window.track.positions) and window.track.states[6]=='lost'
    i=next(i for i,item in enumerate(dialog.rows) if item.kind=='真实跟丢');dialog.table.selectRow(i);before=dict(clip.review_marks)
    dialog.set_checked(True);assert clip.review_marks==before and not dialog.mark.isEnabled()
    path=tmp_path/'已检查序列.locksequence';window._stash_clip();save_sequence(path,window.clips)
    restored=read_sequence(path);assert any(item.checked for item in collect_issues(restored))


def test_full_workspace_recovery_and_review_in_actual_window(window,demo_info,tmp_path):
    from lockframe.workflow_complete_check import exercise_workspace,wait_for
    try:
        result=exercise_workspace(window,demo_info,tmp_path)
        assert result['recovery_async'] and result['curve_draft_recovered']
    finally:
        window._autosave.enabled=False;window._autosave.pending=False;window._autosave.timer.stop()
        if window._autosave.worker:window._autosave.worker.cancel()
        wait_for(lambda:window._autosave.worker is None)


def test_close_waits_for_atomic_autosave_without_destroying_worker(window,demo_info,tmp_path,monkeypatch):
    from lockframe import workspace_autosave_ui as module
    from lockframe.workflow_complete_check import wait_for
    from lockframe.workspace_autosave import _manifest
    activate(window,[tracked_clip(demo_info)]);window._confirm_discard=lambda:True
    entered=threading.Event();release=threading.Event();original=module.write_workspace
    def slow(*args,**kwargs):
        entered.set();assert release.wait(10);return original(*args,**kwargs)
    monkeypatch.setattr(module,'write_workspace',slow)
    window._start_workspace_services(tmp_path/'close-recovery');window.autosave_action.setChecked(True)
    try:
        window._autosave.request(force=True);wait_for(entered.is_set)
        assert not window.close() and not window.isEnabled()
        release.set();wait_for(lambda:not window.isVisible())
        assert window._autosave.worker is None and _manifest(window._autosave.folder)['clean']
    finally:
        release.set();window._autosave.enabled=False;window._autosave.pending=False
        wait_for(lambda:window._autosave.worker is None)


def test_missing_source_recovery_preserves_slot_and_changed_source_invalidates_track(contract_clip,tmp_path,monkeypatch):
    from lockframe.workspace_autosave import write_workspace,restore_saved_workspace
    from lockframe import sequence
    from lockframe.models import file_fingerprint
    cancel=threading.Event();folder=tmp_path/'recovery'/uuid.uuid4().hex
    write_workspace([contract_clip],{},folder,cancel);source=Path(contract_clip.info.path);source.unlink()
    restored=restore_saved_workspace(folder,cancel)['clips']
    assert len(restored)==1 and restored[0].status.startswith('读取失败')
    source.write_bytes(b'changed video source')
    monkeypatch.setattr(sequence,'import_video',lambda *a,**k:replace(contract_clip.info,fingerprint=file_fingerprint(source)))
    restored=restore_saved_workspace(folder,cancel)['clips']
    assert restored[0].track is None and not restored[0].anchors


def test_corrupt_custom_presets_are_preserved_and_builtins_still_visible(window,tmp_path):
    from lockframe.curve_presets import PresetStore
    from lockframe.curve_preset_ui import PresetPicker
    path=tmp_path/'坏预设.json';path.write_text('[]',encoding='utf-8');store=PresetStore(path)
    with pytest.raises(ValueError):store.save('不覆盖',None)
    picker=PresetPicker(None,30,window,store);picker.show()
    assert picker.list.count()==5 and picker.message.text() and path.read_text()=='[]'
    picker.close()
