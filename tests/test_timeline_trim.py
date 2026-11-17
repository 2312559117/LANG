"""Source-range edits through actual Qt handles and existing export functions."""
from copy import deepcopy
import threading
import pickle

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_workflow_v9 import window, source
from lockframe.v072_check import activate, create_correction_source, correction_box, wait_job
from lockframe.sequence import Clip, save_sequence, read_sequence
from lockframe.models import Settings, read_project
from lockframe.timeline_trim import crop_measured_result
from lockframe.core_check import create_source, verify_export
from lockframe.export import export_video
from lockframe.batch_export import prepare_exports, run_exports


def drag(bar, first, last, cancel=False):
    QTest.mousePress(bar,Qt.MouseButton.LeftButton,pos=first)
    QTest.mouseMove(bar,last)
    if cancel:QTest.keyClick(bar,Qt.Key.Key_Escape)
    QTest.mouseRelease(bar,Qt.MouseButton.LeftButton,pos=last)


def test_video_handles_apply_once_seek_preview_and_undo_restores_tracking(window,source):
    clip=deepcopy(source);activate(window,[clip]);original=window.track
    bar=window.strip;row=round(bar.row_rect('video').center().y())
    old_range=(clip.start,clip.end)
    window._seek(17);window._tb_operation('play_range',(10,20));window._tb_operation('loop')
    old_play=dict(bar.state)
    history=window._tb_history();count=len(history.undo_stack)
    drag(bar,QPoint(round(bar.x(0)),row),QPoint(round(bar.x(4)),row))
    assert (window.start_spin.value(),window.end_spin.value())==(5,36)
    assert window.track.start==4 and window.track.end==35 and window.track is not original
    assert len(history.undo_stack)==count+1 and not window.busy and window.worker is None
    assert np.array_equal(window.track.positions,original.positions[4:])
    assert bar.state['play_in']==4 and bar.state['play_out']==36 and not bar.state['custom']
    window._tb_operation('undo')
    assert (clip.start,clip.end)==old_range and window.track is original
    assert bar.state==old_play and window.frame_index==17
    window._tb_operation('redo');assert clip.start==4 and window.track.start==4


def test_cancel_invalid_and_busy_trims_do_not_change_source_range(window,source):
    activate(window,[deepcopy(source)]);original=window.track;bar=window.strip
    row=round(bar.row_rect('video').center().y())
    drag(bar,QPoint(round(bar.x(36)),row),QPoint(round(bar.x(8)),row),cancel=True)
    assert window.track is original and window.end_spin.value()==36
    window._tb_operation('trim',(10,10));assert window.track is original
    window.busy=True
    try:window._tb_operation('trim',(3,9));assert window.track is original
    finally:window.busy=False;window._sync_ui()
    assert window.start_spin.value()==1 and window.end_spin.value()==36


def test_crop_brackets_and_preview_in_out_remain_distinct(window,source):
    activate(window,[deepcopy(source)]);bar=window.strip;bar.setFocus()
    window._seek(5);QTest.keyClick(bar,Qt.Key.Key_BracketLeft)
    window._seek(12);QTest.keyClick(bar,Qt.Key.Key_BracketRight)
    assert (window.start_spin.value(),window.end_spin.value())==(6,13)
    assert window.track.start==5 and window.track.end==12
    window._seek(8);QTest.keyClick(bar,Qt.Key.Key_I)
    assert bar.state['play_in']==8 and window.start_spin.value()==6
    window._tb_operation('undo');window._tb_operation('undo');window._tb_operation('undo')
    assert window.start_spin.value()==1 and window.end_spin.value()==36


def test_crop_keeps_lost_and_never_promotes_unmeasured_or_stale_poses(source):
    result=deepcopy(source.track);anchors=source.anchors
    result.states[5]='lost'
    cropped=crop_measured_result(result,4,8,source.settings,anchors)
    assert cropped.lost_frames==[5] and result.lost_frames==[5]
    assert not np.shares_memory(cropped.positions,result.positions)
    result.performance.update(complete=False,computed_frames=[i<=10 for i in range(result.count)])
    assert crop_measured_result(result,0,11,source.settings,anchors) is None
    measured=crop_measured_result(result,0,9,source.settings,anchors)
    assert measured.performance['complete'] and measured.lost_frames==[5]
    settings=deepcopy(source.settings);settings.target_shape='ellipse'
    assert crop_measured_result(result,0,9,settings,anchors) is None
    assert crop_measured_result(result,0,source.info.frames-1,source.settings,anchors) is None
    settings=deepcopy(source.settings);settings.tracking_mode='precision'
    assert crop_measured_result(result,0,9,settings,anchors) is None
    assert crop_measured_result(result,9,2,source.settings,anchors) is None


def test_single_frame_and_expanding_then_undo_do_not_start_tracking(window,source):
    activate(window,[deepcopy(source)]);original=window.track
    window._tb_operation('trim',(15,16))
    assert window.track.count==1 and window.start_spin.value()==window.end_spin.value()==16
    assert window.export_button.isEnabled() and not window._clip().ready
    assert window._clip().summary=='已完成'  # Measured result remains valid without an in-range seed.
    window._tb_operation('trim_reset')
    assert window.track is None and window._retrack_previous is not None and not window.busy
    window._tb_operation('undo');assert window.track.count==1
    window._tb_operation('undo');assert window.track is original


def test_range_project_sequence_and_other_clip_are_isolated(window,source,tmp_path):
    first=deepcopy(source);other=Clip(source.info,deepcopy(source.settings),dict(source.anchors),0,35,source.track)
    activate(window,[first,other]);old=other.track
    window._tb_operation('trim',(3,20));cropped=window.track
    window._activate_clip(other.id)
    assert other.start==0 and other.end==35 and window.track is old
    window._activate_clip(first.id)
    assert window.track is cropped and window.start_spin.value()==4 and window.end_spin.value()==20
    path=tmp_path/'裁剪序列.locksequence';window._stash_clip();save_sequence(path,window.clips)
    restored=read_sequence(path)
    assert (restored[0].track.start,restored[0].track.end)==(3,19)
    window.project_path=str(tmp_path/'裁剪工程.lockframe');assert window._save_project()
    saved=read_project(window.project_path)
    assert (saved['start'],saved['end'])==(3,19) and saved['track']['start']==3
    window._confirm_discard=lambda:True
    window.load_video(source.info.path,project=saved);wait_job(window)
    assert window.track.start==3 and window.track.end==19 and window.frame_index==3


def test_paused_prefix_can_reuse_measured_frames_and_undo_restores_gate(window,tmp_path):
    info=create_correction_source(tmp_path/'pause')
    clip=Clip(info,Settings(acceleration='cpu',compute_threads=2),{0:correction_box(0)},0,47)
    activate(window,[clip]);window.auto_pause_check.setChecked(True)
    window._start_tracking();wait_job(window)
    pause=window._paused_tracking;assert pause and pause['frame']==12
    window._paused_queue_ids=['unrelated-before',clip.id,'unrelated-after']
    window._tb_operation('trim',(0,12))
    assert window._paused_tracking is None and window.track.count==12 and window.export_button.isEnabled()
    window._tb_operation('undo')
    assert window._paused_tracking is pause and window.track is pause['result']
    assert window._paused_queue_ids==['unrelated-before',clip.id,'unrelated-after']
    assert not window.export_button.isEnabled() and not window.continue_button.isEnabled()


def test_timeline_trim_export_exact_range_audio_and_batch_snapshot(window,tmp_path):
    from lockframe.models import file_fingerprint
    clip=create_source(tmp_path/'source');activate(window,[clip]);identity=file_fingerprint(clip.info.path)
    bar=window.strip;row=round(bar.row_rect('video').center().y())
    drag(bar,QPoint(round(bar.x(2)),row),QPoint(round(bar.x(4)),row))
    drag(bar,QPoint(round(bar.x(10)),row),QPoint(round(bar.x(8)),row))
    window._stash_clip();assert (clip.start,clip.end)==(4,7)
    tasks=prepare_exports([clip],str(tmp_path/'批量输出'))
    window._tb_operation('trim',(5,7))
    assert pickle.loads(tasks[0].snapshot).start==4 and pickle.loads(tasks[0].snapshot).end==7
    window._tb_operation('undo')
    single=tmp_path/'原声裁剪.mp4'
    export_video(window.info,window.track,window.settings(),str(single),threading.Event())
    assert verify_export(single,clip)['frames']==4
    outcomes=run_exports(tasks,threading.Event());assert outcomes[0]['state']=='完成'
    assert verify_export(outcomes[0]['destination'],clip)['frames']==4
    assert file_fingerprint(clip.info.path)==identity
    window.project_path=str(tmp_path/'无区间内种子的裁剪.lockframe');assert window._save_project()
    saved=read_project(window.project_path);window._confirm_discard=lambda:True
    window.load_video(clip.info.path,project=saved);wait_job(window)
    assert window.track.start==4 and window.track.count==4 and window.frame_index==4
    assert window.export_button.isEnabled() and window._clip().summary=='已完成'
