from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys

import pytest

from lockframe.core_check import check_core,RETIRED_MODULES
from lockframe.media import app_root,NO_WINDOW
from lockframe.models import Settings,read_project,save_project
from lockframe.sequence import read_sequence,save_sequence
from test_workflow_v9 import source,window


@pytest.mark.parametrize('old',[None,'broken',{'mode':'fixed','fixed_speed':0},
    {'mode':'basic','frame_interpolation':'optical_flow'},
    {'mode':'advanced','audio':'mute','frame_interpolation':'rife','advanced':{'points':[[0,1],[1,0]]}}])
def test_legacy_project_sequence_ignore_retired_values_and_save_clean(source,tmp_path,old):
    project=tmp_path/'旧工程.lockframe'
    save_project(str(project),source.info,source.settings,source.anchors,source.start,source.end,source.track)
    payload=json.loads(project.read_text(encoding='utf-8'))
    payload['time_remap']=old
    payload['settings'].update(time_remap=old,frame_interpolation='rife')
    project.write_text(json.dumps(payload),encoding='utf-8')
    restored=read_project(str(project))
    assert 'time_remap' not in restored and restored['settings']==asdict(source.settings)
    assert restored['anchors']==payload['anchors'] and restored['track']==payload['track']
    settings=Settings.from_dict(restored['settings'])
    assert not hasattr(settings,'time_remap')
    save_project(str(project),source.info,settings,source.anchors,source.start,source.end,source.track)
    assert 'frame_interpolation' not in project.read_text(encoding='utf-8')
    sequence=tmp_path/'旧序列.locksequence'
    save_sequence(sequence,[source])
    payload=json.loads(sequence.read_text(encoding='utf-8'))
    payload['clips'][0]['settings'].update(time_remap=old,frame_interpolation='optical_flow')
    sequence.write_text(json.dumps(payload),encoding='utf-8')
    clips=read_sequence(sequence)
    assert clips[0].settings==source.settings and clips[0].anchors==source.anchors
    save_sequence(sequence,clips)
    assert 'time_remap' not in sequence.read_text(encoding='utf-8')


def test_old_fields_ignored_but_real_settings_errors_still_rejected():
    settings=Settings.from_dict(dict(time_remap={'fixed_speed':'bad'},frame_interpolation=object(),memory_percent=500))
    with pytest.raises(ValueError):settings.validate()
    with pytest.raises(TypeError):Settings.from_dict({'unexpected_setting':42})


def test_real_legacy_open_source_time_single_batch_audio_and_page(tmp_path):
    report=check_core(tmp_path/'删除功能验收')
    assert report['single_export']['frames']==8
    assert report['batch_export']['frames']==8
    assert report['single_export']['audio'] and report['batch_snapshot_isolated']


@pytest.mark.parametrize('flag',['--retime-check','--interpolation-check','--rife-worker'])
def test_removed_cli_entries_rejected_without_starting_workers(flag,tmp_path):
    result=subprocess.run([sys.executable,str(app_root()/'main.py'),flag,str(tmp_path)],
        capture_output=True,creationflags=NO_WINDOW,timeout=20)
    assert result.returncode==2 and b'unrecognized arguments' in result.stderr


def test_no_retired_ui_controls_and_preview_uses_source_frames(window,source):
    from PySide6.QtWidgets import QLabel,QAbstractButton
    from lockframe.v072_check import activate
    activate(window,[deepcopy(source)])
    assert list(window.sidebar_sections)==['materials','tracking','lock','time_remap','export']
    assert [p.findChild(QLabel,'step').text() for p in window.sidebar_sections.values()]==['01','02','03','04','05']
    visible_text=' '.join(c.text() for c in window.findChildren(QAbstractButton) if c.isVisible())
    assert '启用变速' in visible_text
    assert all(word not in visible_text for word in ('RIFE','云端','删除帧'))
    assert not hasattr(window,'time_panel') and not hasattr(window,'preview_clock')
    track=window.track
    window._seek(3);window._step_preview(1)
    assert window.frame_index==4 and window.timeline.value()==4
    window._set_composition(.6,.4)
    assert window.track is track and window.canvas.mode=='compare'


def test_retired_models_and_runtime_dependencies_not_shipped_in_current_source():
    root=app_root()
    assert not (root/'assets/models/rife-v4.6').exists()
    for name in RETIRED_MODULES:assert not (root/'lockframe'/(name+'.py')).exists()
    for filename in ('requirements.txt','requirements-lock.txt'):
        assert 'rife-' not in (root/filename).read_text(encoding='utf-8')
        assert 'Pillow' not in (root/filename).read_text(encoding='utf-8')
