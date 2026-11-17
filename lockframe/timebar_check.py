"""Actual source and frozen timebar workflow; synthetic footage explicitly labelled."""
from pathlib import Path
import json
import os
import threading
import time
import traceback

from . import __version__


def check_timebar(folder):
    from . import app
    from .core_check import create_source,verify_export
    from .sequence import Clip,save_sequence,read_sequence
    from .models import read_project
    from .export import export_video
    from .batch_export import prepare_exports,run_exports
    from .media import configure_cache_preferences,file_fingerprint
    from .v072_check import activate,wait_job
    from PySide6.QtCore import QSettings,QPoint,Qt
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    previous=os.environ.get('LOCKFRAME_CACHE_DIR');os.environ['LOCKFRAME_CACHE_DIR']=str(folder/'cache')
    window=None
    try:
        application=QApplication.instance() or QApplication([])
        factory=app.QSettings;app.QSettings=lambda *a:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)
        try:window=app.MainWindow()
        finally:app.QSettings=factory
        window.resize(1440,920);window.show();application.processEvents()
        clip=create_source(folder/'generated-source');identity=file_fingerprint(clip.info.path)
        other=Clip(clip.info,start=2,end=9)
        activate(window,[clip,other]);original=window.track;bar=window.strip
        window._tb_operation('play_range',(3,5));window._tb_operation('loop')
        assert window.track is original and (window.start_spin.value(),window.end_spin.value())==(3,10)
        anchor_x=bar.x(6);before=bar.at(anchor_x);bar.zoom(3,anchor_x);assert abs(bar.at(anchor_x)-before)<1e-8
        bar.fit();window._seek(3);bar.setFocus();QTest.keyClick(bar,Qt.Key.Key_Right)
        assert window.frame_index==4
        QTest.keyClick(bar,Qt.Key.Key_Home);assert window.frame_index==3
        window._toggle_play();window._play_started=time.monotonic()-1;window._play_tick()
        assert window.timer.isActive() and 3<=window.frame_index<5
        window._stop_play();window._seek(4);window._tb_operation('add');assert window.track is None and 4 in window.anchors
        bar.selected={4};window._tb_operation('move',({4},1));assert 5 in window.anchors and 4 not in window.anchors
        window._tb_operation('undo');window._tb_operation('undo');assert window.track is original and set(window.anchors)=={2}
        window._tb_operation('redo');window._tb_operation('redo');assert set(window.anchors)=={2,5}
        window._tb_operation('undo');window._tb_operation('undo');assert window.track is original
        window._stash_clip();saved=dict(clip.timebar_state)
        window._activate_clip(other.id);assert not bar.state['custom']
        window._activate_clip(clip.id);assert bar.state==saved and window.track is original
        sequence=folder/'时间线序列.locksequence';window._stash_clip();save_sequence(sequence,window.clips)
        assert read_sequence(sequence)[0].timebar_state==saved
        window.project_path=str(folder/'时间线工程.lockframe');assert window._save_project()
        payload=read_project(window.project_path);assert payload['timebar']==saved
        window._confirm_discard=lambda:True
        window.load_video(payload['source'],project=payload);wait_job(window)
        assert window.strip.state==saved and window.track is not None
        window._stash_clip();export_clip=window._clip()
        single=folder/'正常输出.mp4';export_video(window.info,window.track,window.settings(),str(single),threading.Event())
        single_report=verify_export(single,export_clip)
        tasks=prepare_exports([export_clip],str(folder/'批量输出'));outcomes=run_exports(tasks,threading.Event())
        assert outcomes[0]['state']=='完成';batch_report=verify_export(outcomes[0]['destination'],export_clip)
        assert file_fingerprint(clip.info.path)==identity
        application.processEvents();window.grab().save(str(folder/'帧标尺与双预览.png'))
        # Layout remains usable at the supported small desktop size.
        window.resize(1160,760);application.processEvents()
        assert window.canvas.width()>200 and window.canvas.height()>100
        assert window.canvas.visibleRegion().boundingRect().height()>100
        window.grab().save(str(folder/'小窗口时间线.png'))
        return dict(footage='synthetic indexed video with numeric test trajectory; real car footage pending',
                    independent_play_range=True,loop_and_keyboard=True,zoom_anchor=True,
                    key_move_undo_redo=True,clip_state_isolated=True,sequence_and_project_reopened=True,
                    no_tracking_job_for_edits=True,source_unchanged=True,
                    small_layout=dict(window=[window.width(),window.height()],canvas=[window.canvas.width(),window.canvas.height()]),
                    single_export=single_report,batch_export=batch_report,
                    limitations=['single correction lane; no generic mask/transform interpolation','existing CFR frame clock; not VFR PTS mapping'])
    finally:
        if window:window._confirm_discard=lambda:True;window.close();application.processEvents()
        if previous is None:os.environ.pop('LOCKFRAME_CACHE_DIR',None)
        else:os.environ['LOCKFRAME_CACHE_DIR']=previous
        configure_cache_preferences(None)


def run_timebar_check(directory):
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    try:report=dict(ok=True,version=__version__,timebar=check_timebar(folder));code=0
    except Exception:report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return code
