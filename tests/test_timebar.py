import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from lockframe.timebar_core import Timebase,clean_state,shifted_keys,EditHistory
from lockframe.timebar_ui import Timebar
from test_workflow_v9 import source,window


@pytest.mark.parametrize('fps,frame,expected',[(30000/1001,1800,'00:01:00;02'),
    (30000/1001,17982,'00:10:00;00'),(30000/1001,107892,'01:00:00;00'),
    (60000/1001,3600,'00:01:00;04'),(60000/1001,35964,'00:10:00;00'),
    (24000/1001,24,'00:00:01:00'),(25,25,'00:00:01:00')])
def test_source_timecode_boundary(fps,frame,expected):
    assert Timebase.from_fps(fps).timecode(frame)==expected


def test_optional_view_schema_and_integer_frames():
    assert clean_state({},60,4,40)['play_out']==41
    assert clean_state(dict(play_in=True,play_out=30),60,4,40)['play_in']==4
    assert clean_state(dict(play_in=1.5,play_out=30),60)['play_in']==0
    state=clean_state(dict(play_in=4,play_out=5,custom=True,scale=float('nan')),60)
    assert state['custom'] and state['play_out']==5 and state['scale']==0
    assert clean_state(dict(schema=99),60)['schema']==1


def test_move_preserves_distances_and_rejects_collision():
    keys={2:(0,0,.2,.2),5:(.1,.1,.2,.2),9:(.2,.2,.2,.2)}
    moved=shifted_keys(keys,{2,5},-100,0,20)
    assert set(moved)=={0,3,9} and keys.keys()=={2,5,9}
    with pytest.raises(ValueError):shifted_keys(keys,{2,5},4,0,20)
    assert shifted_keys(keys,set(),4,0,20)==keys


def test_history_multi_redo_new_branch_and_memory_limit():
    h=EditHistory(limit=3,budget=900)
    for label in ('a','b','c'):h.push(dict(label=label,bytes=256))
    assert h.undo()['label']=='c' and h.undo()['label']=='b'
    assert h.redo()['label']=='b' and h.redo()['label']=='c'
    h.undo();h.push(dict(label='d'));assert h.redo() is None
    for _ in range(10):h.push(dict(label='e',bytes=400))
    assert len(h.undo_stack)==2


def test_qt_zoom_anchor_box_select_drag_cancel_and_readonly():
    app=QApplication.instance() or QApplication([])
    bar=Timebar();bar.resize(892,116);bar.load_state({},100,25)
    bar.anchors={20:(0,0,.2,.2),30:(0,0,.2,.2)};bar.sync_data();bar.show();app.processEvents()
    try:
        x=bar.x(45);before=bar.at(x);bar.zoom(2,x);assert bar.at(x)==pytest.approx(before)
        bar.fit();QTest.mousePress(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(18)),68))
        QTest.mouseMove(bar,QPoint(round(bar.x(32)),68));QTest.mouseRelease(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(32)),68))
        assert bar.selected=={20,30}
        events=[];bar.operation.connect(lambda *args:events.append(args))
        QTest.mousePress(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(20)),68))
        QTest.mouseMove(bar,QPoint(round(bar.x(25)),68));QTest.keyClick(bar,Qt.Key.Key_Escape)
        QTest.mouseRelease(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(25)),68))
        assert not events and set(bar.anchors)=={20,30}
        bar.editable=False;QTest.keyClick(bar,Qt.Key.Key_Delete);assert not events
    finally:bar.close()


def test_host_play_range_edits_keys_undo_and_material_isolation(window,tmp_path):
    from lockframe.media import import_video,resource_root
    from lockframe.demo import demo_frame
    from lockframe.sequence import Clip,save_sequence,read_sequence
    from lockframe.models import save_project,read_project
    import threading
    info=import_video(str(resource_root()/'assets/demo.mp4'),threading.Event())
    box=tuple(demo_frame(0)[1]);first=Clip(info,anchors={0:box,10:box,20:box})
    second=Clip(info,anchors={2:box})
    window.clips=[first,second];window.active_clip_id=None;window._activate_clip(first.id)
    original=window.track;original_range=(first.start,first.end);settings=window.settings()
    window._seek(5);window._tb_operation('in');window._seek(24);window._tb_operation('out')
    assert window.strip.state['play_in']==5 and window.strip.state['play_out']==25
    assert window.track is original and (window.start_spin.value()-1,window.end_spin.value()-1)==original_range
    window.strip.selected={10,20};window._tb_operation('move',({10,20},2))
    assert set(window.anchors)=={0,12,22}
    window._tb_operation('undo');assert set(window.anchors)=={0,10,20}
    window._tb_operation('redo');assert set(window.anchors)=={0,12,22}
    window.strip.selected={12,22};window._tb_operation('copy');window._seek(35);window._tb_operation('paste')
    assert {35,45}<=window.anchors.keys()
    window.strip.selected={35,45};window._tb_operation('delete');assert not {35,45}&window.anchors.keys()
    window._tb_operation('undo');assert {35,45}<=window.anchors.keys()
    window._tb_operation('in');assert window.strip.state['play_in']==5  # Invalid inverted range is rejected.
    window._stash_clip();saved=dict(first.timebar_state);window._activate_clip(second.id)
    assert not window.strip.state['custom'] and second.anchors=={2:box}
    window._tb_operation('undo');assert second.anchors=={2:box}
    window._activate_clip(first.id);assert window.strip.state==saved
    sequence=tmp_path/'时间线.locksequence';window._stash_clip();save_sequence(sequence,window.clips)
    restored=read_sequence(sequence);assert restored[0].timebar_state==saved
    project=tmp_path/'时间线.lockframe';save_project(str(project),info,settings,window.anchors,first.start,first.end,None,timebar=saved)
    assert read_project(str(project))['timebar']==saved
    assert not any(k in json.loads(project.read_text(encoding='utf-8')) for k in ('time_remap','frame_interpolation'))


def test_single_frame_loop_and_busy_commands_are_safe(window):
    from lockframe.media import import_video,resource_root
    from lockframe.sequence import Clip
    import threading
    clip=Clip(import_video(str(resource_root()/'assets/demo.mp4'),threading.Event()))
    window.clips=[clip];window.active_clip_id=None;window._activate_clip(clip.id)
    window._tb_operation('play_range',(3,4));window._tb_operation('loop')
    window._seek(3);window._toggle_play();window._play_started=time.monotonic()-50
    window._play_tick();assert window.frame_index==3 and window.timer.isActive()
    window._stop_play();before=dict(window.strip.state);window.busy=True
    try:window._tb_operation('clear_range');assert window.strip.state==before
    finally:window.busy=False


def test_play_edits_keep_real_tracking_result_and_anchor_undo_restores_it(window,source,monkeypatch):
    from copy import deepcopy
    from lockframe.v072_check import activate
    clip=deepcopy(source);activate(window,[clip]);result=window.track
    monkeypatch.setattr(window,'_job',lambda *a,**k:pytest.fail('timebar edit must not launch tracking'))
    window._tb_operation('play_range',(2,8));window._tb_operation('loop')
    assert window.track is result and window.start_spin.value()==1 and window.end_spin.value()==36
    window.strip.selected={10};window._tb_operation('move',({10},2))
    assert window.track is None and window._retrack_previous is result and 12 in window.anchors
    window._tb_operation('undo');assert window.track is result and set(window.anchors)=={10}
    window._tb_operation('redo');assert window.track is None and set(window.anchors)=={12}
    window.strip.selected={12};window._tb_operation('copy');window._seek(34);window._tb_operation('paste')
    keys=dict(window.anchors);window._seek(35);window._tb_operation('paste')
    assert set(window.anchors)=={12,34,35}
    window._tb_operation('paste');assert set(window.anchors)=={12,34,35}  # collision is atomic


def test_paused_anchor_edit_undo_restores_continue_gate(window,source):
    from copy import deepcopy
    from lockframe.v072_check import activate
    from lockframe.tracking_pause import TrackingPaused
    clip=deepcopy(source);activate(window,[clip])
    pause=TrackingPaused(clip.track,10,1,'lost')
    window._tracking_paused(pause)
    box=list(window.anchors[10]);box[0]+=.01
    window._select_roi(tuple(box));assert window._paused_tracking['corrected']
    window._tb_operation('undo');assert not window._paused_tracking['corrected']
    window._tb_operation('redo');assert window._paused_tracking['corrected'] and not window.busy


def test_pointer_move_submits_once_and_play_range_handle_is_independent():
    app=QApplication.instance() or QApplication([])
    bar=Timebar();bar.resize(892,116);bar.load_state({},100,25)
    bar.anchors={20:(0,0,.2,.2)};bar.sync_data();bar.show();app.processEvents();events=[]
    bar.operation.connect(lambda *args:events.append(args))
    try:
        QTest.mousePress(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(20)),68))
        QTest.mouseMove(bar,QPoint(round(bar.x(22)),68))
        assert not events
        QTest.mouseRelease(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(22)),68))
        assert len(events)==1 and events[0][0]=='move' and events[0][1][1]==2
        events.clear();QTest.mousePress(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(0)),34))
        QTest.mouseMove(bar,QPoint(round(bar.x(5)),34));QTest.mouseRelease(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(5)),34))
        assert events==[('play_range',(5,100))] and (bar.start,bar.end)==(0,99)
    finally:bar.close()


def test_noop_roi_preserves_track_and_step_stops_clock(window,source):
    from copy import deepcopy
    from lockframe.v072_check import activate
    activate(window,[deepcopy(source)]);window._seek(10)
    track=window.track;window._select_roi(tuple(window.anchors[10]))
    assert window.track is track and not window._tb_history().undo_stack
    window._toggle_play();assert window.timer.isActive()
    window._step_preview(1)
    assert not window.timer.isActive() and window.frame_index==11


def test_removing_material_and_replacing_sequence_release_histories(window,source,monkeypatch):
    from copy import deepcopy
    from lockframe.sequence import Clip
    from lockframe.v072_check import activate
    from PySide6.QtWidgets import QMessageBox
    first=deepcopy(source);second=Clip(source.info)
    activate(window,[first,second]);window._tb_operation('play_range',(2,8))
    window._timebar_clipboard=(first.id,[(0,(0.,0.,.2,.2))])
    assert first.id in window._timebar_histories
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:QMessageBox.StandardButton.Yes)
    window._refresh_sequence({first.id});window._remove_clips()
    assert first.id not in window._timebar_histories and window._timebar_clipboard==(None,[])
    window._tb_history().push(dict(kind='play'))
    window._sequence_restored([]);assert window._timebar_histories  # Rejected/empty load keeps old state.
    window._pending_sequence_path=None;window._sequence_restored([Clip(source.info)])
    assert not window._timebar_histories and window._timebar_clipboard==(None,[])
