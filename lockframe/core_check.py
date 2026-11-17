"""Acceptance for the four-step, source-time-only product and legacy projects."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import traceback

import numpy as np

from . import __version__
from .batch_export import prepare_exports, run_exports
from .export import export_video
from .media import (FrameReader, NO_WINDOW, app_root, configure_cache_preferences, find_tool,
                    import_video, resource_root, run_ffmpeg)
from .models import Settings, TrackResult, read_project, save_project
from .sequence import Clip, read_sequence, save_sequence
from .v072_check import activate, wait_job


RETIRED_MODULES = ('time_ui','time_mapping','time_presets','retime_check',
                   'frame_interpolation','rife_interpolation','rife_worker',
                   'interpolation_preview','interpolation_check')


def create_source(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'原片带声音.mp4'
    run_ffmpeg(['-f','lavfi','-i','testsrc2=size=192x128:rate=12:duration=1',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=1',
                '-vf',"geq=lum='clip(lum(X,Y)+N*12,16,235)':cb='cb(X,Y)':cr='cr(X,Y)'",
                '-c:v','libx264','-crf','8','-pix_fmt','yuv420p',
                '-c:a','aac','-shortest',str(path)],threading.Event())
    info=import_video(str(path),threading.Event())
    first,last=2,9
    anchors={2:(.4,.4,.2,.2)}
    result=TrackResult(first,last,np.full((8,2),.5),np.tile([.4,.4,.2,.2],(8,1)),
                       np.ones(8),['tracked']*8,anchors=anchors)
    return Clip(info,Settings(resolution=0,max_zoom=1.,acceleration='cpu'),anchors,first,last,result)


def verify_export(path,clip):
    report=json.loads(subprocess.check_output([find_tool('ffprobe'),'-v','error','-count_frames',
        '-show_streams','-show_format','-of','json',str(path)],creationflags=NO_WINDOW))
    video=next(s for s in report['streams'] if s['codec_type']=='video')
    assert int(video['nb_read_frames'])==clip.track.count
    assert (video['width'],video['height'])==(clip.info.width,clip.info.height)
    duration=float(report['format']['duration'])
    assert abs(duration-clip.track.count/clip.info.fps)<=1/clip.info.fps
    assert any(s['codec_type']=='audio' for s in report['streams'])
    original,output=FrameReader(clip.info.path),FrameReader(str(path))
    errors=[]
    try:
        for n in range(clip.track.count):
            a,b=original.read(clip.start+n),output.read(n)
            errors.append(float(np.abs(a.astype(float)-b.astype(float)).mean()))
        assert max(errors)<8, errors
    finally:original.close();output.close()
    audio=subprocess.check_output([find_tool('ffmpeg'),'-v','error','-i',str(path),
        '-map','0:a:0','-f','f32le','-ar','48000','-ac','1','pipe:1'],creationflags=NO_WINDOW)
    samples=np.frombuffer(audio,np.float32)
    assert np.sqrt(np.mean(samples**2))>.01
    frequency=float(np.fft.rfftfreq(len(samples),1/48000)[np.argmax(abs(np.fft.rfft(samples)))])
    assert abs(frequency-440)<5
    return dict(frames=int(video['nb_read_frames']),duration=duration,audio=True,
                frame_errors=errors,audio_peak_hz=frequency)


def audit_removal():
    from . import app
    import ast
    source=Path(app.__file__).parent
    if not (source/'app.py').is_file():
        source=app_root()/'source/lockframe'
    forbidden={'lockframe.'+name for name in RETIRED_MODULES}|{'rife_ncnn_vulkan_python'}
    assert not any(name in sys.modules for name in forbidden)
    assert not any((source/(name+'.py')).exists() for name in RETIRED_MODULES)
    assert not (resource_root()/'assets/models/rife-v4.6').exists()
    bad=[]
    files=list(source.glob('*.py'))
    assert files,'Missing inspectable source snapshot'
    for file in files:
        for node in ast.walk(ast.parse(file.read_text(encoding='utf-8'))):
            if isinstance(node,ast.Import): names=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom): names=[node.module or '']
            else:continue
            for name in names:
                if name.split('.')[0] in set(RETIRED_MODULES)|{'rife_ncnn_vulkan_python'}:
                    bad.append((file.name,name))
    assert not bad,bad
    return dict(retired_modules_absent=True,inference_not_imported=True,models_absent=True,source_files_checked=len(files),
                live_imports_clean=True)


def check_core(folder):
    from . import app
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication,QLabel
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    previous_cache=os.environ.get('LOCKFRAME_CACHE_DIR')
    os.environ['LOCKFRAME_CACHE_DIR']=str(folder/'cache')
    window=None
    try:
        application=QApplication.instance() or QApplication([])
        factory=app.QSettings
        app.QSettings=lambda *args:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)
        try:window=app.MainWindow()
        finally:app.QSettings=factory
        window.resize(1440,920);window.show();application.processEvents()
        clip=create_source(folder)
        project=folder/'旧工程.lockframe'
        save_project(str(project),clip.info,clip.settings,clip.anchors,clip.start,clip.end,clip.track)
        payload=json.loads(project.read_text(encoding='utf-8'))
        old=dict(mode='advanced',fixed_speed=.5,audio='mute',frame_interpolation='rife',
                 advanced=dict(points=[[0,1],[.5,0],[1,0]],interpolation='hold'))
        payload['settings'].update(time_remap=old,frame_interpolation='optical_flow')
        project.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
        restored=read_project(str(project));restored['_project_path']=str(project)
        window._confirm_discard=lambda:True
        window.load_video(clip.info.path,project=restored);wait_job(window)
        assert window.track is not None and window.settings()==clip.settings
        assert not hasattr(window,'time_panel') and not hasattr(window,'_rife_preview')
        assert list(window.sidebar_sections)==['materials','tracking','lock','time_remap','export']
        numbers=[section.findChild(QLabel,'step').text() for section in window.sidebar_sections.values()]
        assert numbers==['01','02','03','04','05']
        assert not hasattr(window,'curve_editor_host') and not hasattr(window,'preview_clock')
        assert window.canvas.mode=='compare' and window.canvas.visibleRegion().contains(window.canvas.rect())
        window._seek(2);window._step_preview(1)
        assert window.frame_index==3 and window.timeline.value()==3
        positions=window.track.positions.copy();track=window.track
        window._set_composition(.6,.4)
        assert window.track is track and np.array_equal(track.positions,positions)
        window._set_composition(.5,.5)
        application.processEvents();window.grab().save(str(folder/'四步工作流双预览.png'))
        out=export_video(clip.info,window.track,window.settings(),str(folder/'旧工程正常原速.mp4'),threading.Event())
        single=verify_export(out['path'],clip)
        cleaned=folder/'清理后.lockframe'
        save_project(str(cleaned),window.info,window.settings(),window.anchors,clip.start,clip.end,window.track)
        assert 'time_remap' not in json.loads(cleaned.read_text(encoding='utf-8'))['settings']
        sequence=folder/'旧序列.locksequence'
        save_sequence(sequence,[clip])
        raw=json.loads(sequence.read_text(encoding='utf-8'))
        raw['clips'][0]['settings'].update(time_remap=old,frame_interpolation='rife')
        sequence.write_text(json.dumps(raw,ensure_ascii=False),encoding='utf-8')
        clips=read_sequence(sequence)
        tasks=prepare_exports(clips,str(folder/'批量输出'))
        clips[0].settings.target_x=.7
        outcomes=run_exports(tasks,threading.Event())
        assert outcomes[0]['state']=='完成',outcomes
        batch=verify_export(outcomes[0]['destination'],clip)
        cleaned_sequence=folder/'清理后.locksequence';save_sequence(cleaned_sequence,clips)
        assert 'time_remap' not in json.loads(cleaned_sequence.read_text(encoding='utf-8'))['clips'][0]['settings']
        audit=audit_removal()
        return dict(footage='local synthetic indexed clip only; original car footage pending',
                    steps=numbers,source_clock=True,persistent_dual_preview=True,
                    legacy_project_and_sequence_ignored=True,saved_fields_clean=True,
                    composition_keeps_track=True,batch_snapshot_isolated=True,
                    single_export=single,batch_export=batch,removal_audit=audit)
    finally:
        if window:
            window._confirm_discard=lambda:True;window.close();application.processEvents()
        if previous_cache is None:os.environ.pop('LOCKFRAME_CACHE_DIR',None)
        else:os.environ['LOCKFRAME_CACHE_DIR']=previous_cache
        configure_cache_preferences(None)


def run_core_check(directory):
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    try:report=dict(ok=True,version=__version__,core=check_core(folder));code=0
    except Exception:report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return code
