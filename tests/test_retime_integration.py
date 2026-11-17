"""New retime contracts across project, queue and real Qt source/output clocks.

UI footage is the bundled synthetic demo, not real-camera accuracy evidence.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

from copy import deepcopy
from dataclasses import asdict
import json
import pickle
from pathlib import Path
import threading
import uuid

import numpy as np
import pytest

from lockframe.models import (Settings,TrackResult,VideoInfo,file_fingerprint,
                              read_project,save_project,source_identity)
from lockframe.sequence import Clip,read_sequence,save_sequence
from lockframe.time_remap_model import default_config,make_constant_speed
from lockframe.time_remap_map import output_frame_map


def tracked_clip(info,start=0,end=35,retime=None):
    n=end-start+1;anchors={start:(.25,.25,.5,.5)}
    track=TrackResult(start,end,np.tile([.5,.5],(n,1)),np.tile([.25,.25,.5,.5],(n,1)),
                      np.ones(n),['manual']+['tracked']*(n-1),anchors=deepcopy(anchors),
                      performance={'source_identity':source_identity(info)})
    settings=Settings(acceleration='cpu',compute_threads=2,resolution=0,
                      retime=deepcopy(retime or {}))
    return Clip(info,settings,anchors,start,end,track)


@pytest.fixture
def contract_clip(tmp_path):
    path=tmp_path/'素材 中文.mp4';path.write_bytes(b'local metadata contract fixture')
    info=VideoInfo(str(path),100,80,25.,4.,True,100,str(path),file_fingerprint(path))
    return tracked_clip(info,10,39,make_constant_speed(None,.5))


def test_default_settings_validate_and_legacy_fields_are_inert():
    Settings().validate()
    raw=asdict(Settings())
    raw.update(time_remap={'enabled':True,'position':float('nan')},
               frame_interpolation={'engine':'rife','model':'retired'},)
    settings=Settings.from_dict(raw)
    settings.validate()
    assert settings.retime=={}
    np.testing.assert_array_equal(output_frame_map(settings.retime or None,13),np.arange(13))
    assert 'time_remap' not in asdict(settings) and 'frame_interpolation' not in asdict(settings)


def test_project_v2_roundtrip_ignores_broken_legacy_fields_and_resaves_clean(contract_clip,tmp_path):
    clip=contract_clip;path=tmp_path/'新工程.lockframe'
    save_project(str(path),clip.info,clip.settings,clip.anchors,clip.start,clip.end,clip.track)
    data=json.loads(path.read_text(encoding='utf-8'))
    data.update(time_remap={'enabled':True},frame_interpolation={'engine':'rife'})
    data['settings'].update(time_remap='malformed legacy ignored',frame_interpolation=[1,2,3])
    path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
    restored=read_project(str(path))
    settings=Settings.from_dict(restored['settings'])
    assert settings.retime==clip.settings.retime
    assert len(output_frame_map(settings.retime,30))==60
    assert 'time_remap' not in restored and 'frame_interpolation' not in restored
    target=tmp_path/'再次保存.lockframe'
    save_project(str(target),clip.info,settings,clip.anchors,clip.start,clip.end,clip.track)
    clean=json.loads(target.read_text(encoding='utf-8'))
    assert 'time_remap' not in clean['settings'] and 'frame_interpolation' not in clean['settings']
    assert clean['settings']['retime']['version']==2


def test_sequence_and_batch_snapshots_keep_nested_curve_and_clip_isolation(contract_clip,tmp_path):
    from lockframe.batch_export import prepare_exports,run_exports
    from unittest.mock import patch
    first=deepcopy(contract_clip);second=deepcopy(contract_clip);second.id=uuid.uuid4().hex
    second.settings.retime=make_constant_speed(None,-2.)
    sequence=tmp_path/'多素材.locksequence';save_sequence(sequence,[first,second])
    restored=read_sequence(sequence)
    assert restored[0].settings.retime!=restored[1].settings.retime
    tasks=prepare_exports(restored,tmp_path/'输出 中文')
    assert not any(task.reason for task in tasks)
    frozen=[pickle.loads(task.snapshot).settings.retime for task in tasks]
    restored[0].settings.retime['points'][1]['source']=.2
    restored[0].settings.retime['basic']['speeds'][0]=9.
    restored[1].settings.retime['enabled']=False
    assert pickle.loads(tasks[0].snapshot).settings.retime==frozen[0]
    assert pickle.loads(tasks[1].snapshot).settings.retime==frozen[1]
    calls=[]
    def renderer(info,track,settings,destination,cancel,progress,*,overwrite):
        calls.append(deepcopy(settings.retime));return {'path':destination}
    with patch('lockframe.batch_export.export_video',renderer):
        result=run_exports(tasks,threading.Event())
    assert [item['state'] for item in result]==['完成','完成']
    assert calls==frozen
    assert len(output_frame_map(calls[0],30))==60
    assert len(output_frame_map(calls[1],30))==15


@pytest.mark.parametrize('enabled',[False,True])
def test_public_single_export_dispatch_keeps_disabled_baseline_and_shared_mapping(contract_clip,tmp_path,monkeypatch,enabled):
    from lockframe import export,retime_renderer
    clip=contract_clip;clip.settings.retime['enabled']=enabled
    captured=[]
    def retimed(info,track,settings,mapping,*args,**kwargs):
        captured.append(('retime',mapping.copy()));return {'frames':len(mapping)}
    def baseline(*args,**kwargs):
        captured.append(('baseline',None));return {'frames':clip.track.count}
    monkeypatch.setattr(retime_renderer,'render_retimed_video',retimed)
    monkeypatch.setattr(export,'_export_video',baseline)
    monkeypatch.setattr(export,'select_encoder',lambda *args:'libx264')
    result=export.export_video(clip.info,clip.track,clip.settings,str(tmp_path/'成品.mp4'),threading.Event())
    assert captured[0][0]==('retime' if enabled else 'baseline')
    assert result['frames']==(60 if enabled else 30)
    if enabled:
        np.testing.assert_array_equal(captured[0][1],output_frame_map(clip.settings.retime,clip.track.count))


@pytest.fixture(scope='module')
def demo_info():
    from lockframe.media import import_video,resource_root
    return import_video(str(resource_root()/'assets/demo.mp4'),threading.Event())


@pytest.fixture
def window(tmp_path,monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from lockframe import app
    application=QApplication.instance() or QApplication([])
    monkeypatch.setattr(app,'QSettings',lambda *args:QSettings(str(tmp_path/'prefs.ini'),QSettings.Format.IniFormat))
    monkeypatch.setattr(app,'track_video',lambda *args,**kwargs:pytest.fail('编辑变速不能重新跟踪'))
    value=app.MainWindow();value.resize(1280,800);value.show();application.processEvents()
    value._error=lambda message:pytest.fail(message)
    yield value
    if value._retime_editor is not None:value._retime_editor.close()
    value._confirm_discard=lambda:True
    value.close();application.processEvents()
    from lockframe.media import configure_cache_preferences
    configure_cache_preferences(None)


def activate(window,clips):
    from lockframe.v072_check import activate as production_activate
    production_activate(window,clips)
    assert window.info is not None and window.active_clip_id==clips[0].id


def test_curve_edits_only_update_render_clock_and_keep_measured_tracking(window,demo_info):
    clip=tracked_clip(demo_info,0,35);activate(window,[clip])
    track=window.track;anchors=deepcopy(window.anchors);positions=track.positions.copy()
    window._apply_retime_config(make_constant_speed(None,.5))
    assert window.track is track and window.anchors==anchors
    np.testing.assert_array_equal(window.track.positions,positions)
    assert window.plan is not None and len(window._retime_map())==72
    assert window.settings().retime==clip.settings.retime
    assert window.dirty and not window.busy
    original=deepcopy(window.settings().retime)
    window._retime_config['points'][0]['source']=.1
    assert clip.settings.retime==original


def test_reverse_output_steps_decrease_source_but_source_timeline_stays_source(window,demo_info):
    clip=tracked_clip(demo_info,10,29,make_constant_speed(None,-1.));activate(window,[clip])
    window._retime_clock_changed('output')
    window._seek_output(0)
    assert window.frame_index==29 and window._retime_output_step==0
    window._step_preview(1)
    assert window.frame_index==28 and window._retime_output_step==1
    assert window.timeline.value()==28
    # The old source timeline must not reinterpret a source index as output step.
    window.strip.seek.emit(15)
    assert window.frame_index==15 and window._retime_output_step==14
    window._retime_clock_changed('source');window._step_preview(1)
    assert window.frame_index==16


def test_freeze_output_clock_advances_even_when_source_frame_does_not(window,demo_info):
    config=make_constant_speed(None,1.)
    for point in config['points']:point['source']=.4
    clip=tracked_clip(demo_info,10,29,config);activate(window,[clip])
    window._retime_clock_changed('output');window._seek_output(2)
    source=window.frame_index
    window._step_preview(1)
    assert window.frame_index==source and window._retime_output_step==3
    window._step_preview(1)
    assert window.frame_index==source and window._retime_output_step==4
    window._seek(source)
    assert window._retime_output_step==4


def test_local_repeat_source_seek_chooses_nearby_output_occurrence(window,demo_info):
    config=make_constant_speed(None,1.)
    config['points']=[dict(position=0,source=0,interpolation='linear'),
                      dict(position=.5,source=1,interpolation='linear',incoming_interpolation='linear'),
                      dict(position=1,source=0,incoming_interpolation='linear')]
    clip=tracked_clip(demo_info,0,20,config);activate(window,[clip])
    window._retime_clock_changed('output');window._seek_output(15)
    assert window.frame_index==10 and window._retime_output_step==15
    window._seek(10)
    assert window._retime_output_step==15
    window._seek_output(5);window._seek(10)
    assert window._retime_output_step==5


def test_editor_draft_cannot_apply_to_another_clip_of_the_same_source(window,demo_info):
    first=tracked_clip(demo_info,0,35,make_constant_speed(None,.5))
    second=tracked_clip(demo_info,0,35,make_constant_speed(None,2.))
    activate(window,[first,second]);window._open_retime_editor()
    editor=window._retime_editor
    before=deepcopy(second.settings.retime)
    window._activate_clip(second.id)
    editor.applied.emit(make_constant_speed(None,-1.))
    assert window.active_clip_id==second.id
    assert window.settings().retime==before and second.settings.retime==before
    assert '素材或处理范围已改变' in window.status_label.text()


def test_editor_draft_cannot_apply_after_source_range_changed(window,demo_info):
    clip=tracked_clip(demo_info,0,35,make_constant_speed(None,.5));activate(window,[clip])
    window._open_retime_editor();editor=window._retime_editor
    before=deepcopy(window.settings().retime)
    window.end_spin.setValue(30)
    editor.applied.emit(make_constant_speed(None,-1.))
    assert window.settings().retime==before
    assert '素材或处理范围已改变' in window.status_label.text()
