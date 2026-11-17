"""Exercise production workspace services in source and frozen Qt builds."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json,threading,time,traceback,uuid
import numpy as np


def wait_for(predicate,timeout=30):
    from PySide6.QtWidgets import QApplication
    end=time.monotonic()+timeout
    while not predicate() and time.monotonic()<end:
        QApplication.processEvents();time.sleep(.005)
    QApplication.processEvents()
    assert predicate(),'workspace acceptance timed out'


def exercise_batch_resume(window,info):
    """Real tracker pause plus production QueueJob, without a manual correction."""
    from .demo import demo_frame
    from .models import Settings
    from .sequence import Clip
    from .tracking import track_video
    from .tracking_pause import TrackingPaused,is_complete
    from .v072_check import activate,wait_job
    anchor=demo_frame(0,info.width,info.height)[1];pause=threading.Event();end=min(23,info.frames-1)
    def observed(value):
        if value['frame']==5:pause.set()
    try:
        track_video(info,{0:anchor},0,end,threading.Event(),pause_event=pause,on_frame=observed)
        raise AssertionError('expected ordinary pause')
    except TrackingPaused as value:paused=value
    assert paused.reason=='manual' and not paused.result.lost_frames
    clips=[Clip(replace(info),Settings(acceleration='cpu',compute_threads=2),{0:anchor},0,end,
        deepcopy(paused.result),retrack_previous=deepcopy(paused.result),tracking_resume=deepcopy(paused.resume)) for _ in range(2)]
    activate(window,clips)
    for clip in clips:
        window._paused_by_clip[clip.id]=window._pause_context_for(TrackingPaused(clip.track,paused.frame,paused.direction,'manual',resume=clip.tracking_resume))
    window._paused_tracking=window._paused_by_clip[clips[0].id];window._paused_queue_ids=[clip.id for clip in clips]
    window._continue_tracking();assert window.busy,'batch resume did not start'
    wait_job(window)
    assert all(is_complete(clip.track) and not clip.track.lost_frames for clip in clips)
    assert all(clip.anchors=={0:anchor} and clip.track.anchors=={0:anchor} for clip in clips)
    return dict(real_queue=True,clips=2,ordinary_pause_frame=paused.frame,complete=True,no_extra_anchors=True)


def exercise_workspace(window,info,folder):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from .models import TrackResult,Settings,source_identity,save_project,read_project
    from .sequence import Clip
    from .time_remap_model import make_constant_speed
    from .curve_presets import builtins,PresetStore,apply_curve
    from .curve_preset_ui import PresetPicker
    from .workspace_autosave import recover_workspace,_manifest
    from .review_records import collect_issues
    from .tracking_pause import TrackingPaused
    from .v072_check import activate,wait_job
    folder=Path(folder);n=min(24,info.frames);anchor=(.25,.25,.5,.5)
    def clip(x):
        track=TrackResult(0,n-1,np.tile([.5,.5],(n,1)),np.tile(anchor,(n,1)),np.ones(n),
            ['manual']+['tracked']*(n-1),anchors={0:anchor},performance={'source_identity':source_identity(info),
            'early_warning_ranges':[dict(start=3,end=5,frame=5,confidence=.7)]})
        return Clip(replace(info),Settings(target_x=x,acceleration='cpu',compute_threads=2,resolution=0,
            retime=make_constant_speed(None,.5)),{0:anchor},0,n-1,track)
    clips=[clip(.3),clip(.7)];activate(window,clips);window._activate_clip(clips[1].id);wait_job(window)
    window._retime_clock_changed('output');window._seek_output(12)
    window.canvas.set_zoom(2.);window.canvas.set_output_zoom(1.5)
    window._open_retime_editor();editor=window._retime_editor
    draft=make_constant_speed(None,2.);editor._change(draft);QApplication.processEvents()
    track=window.track;before=deepcopy(window.settings().retime)
    point=editor.canvas.pixel(.2,.5).toPoint()
    QTest.mouseClick(editor.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.AltModifier,pos=point)
    assert window.track is track and window.settings().retime==before and editor.draft==draft
    window._seek_output(12)
    window._start_workspace_services(folder/'recovery');window.autosave_action.setChecked(True)
    assert window._autosave.request(force=True)
    wait_for(lambda:window._autosave.worker is None)
    assert not window._autosave.last_error
    saved,view,_=recover_workspace(window._autosave.folder)
    assert len(saved)==2 and saved[0].settings.target_x==.3 and saved[1].settings.target_x==.7
    assert view['active_index']==1 and view['editor']['config']==draft and view['output_step']==12
    assert window.track is track
    # Mutate the live scene, then recover through the same async UI action.
    window._apply_retime_config(make_constant_speed(None,3.))
    window._confirm_discard=lambda:True
    window._restore_workspace(str(window._autosave.folder));wait_job(window)
    wait_for(lambda:window._workspace_restore_pending is None and not window.busy)
    wait_for(lambda:window._autosave.worker is None)
    assert window.active_clip_id==window.clips[1].id and window._retime_clock=='output'
    assert window._retime_output_step==12 and window.settings().retime==before
    assert window._retime_editor.draft==draft and window.canvas.zoom_factor==2.
    assert window._retime_editor is not editor and window._retime_editor._workspace_context[0]==window.active_clip_id
    window._retime_editor.applied.emit(draft);assert window.settings().retime==draft
    window._apply_retime_config(before)
    assert window.canvas.output_zoom==1.5 and len(window._selected_clip_ids())==2
    # Issue navigation crosses clips, without changing measured points.
    window._open_review_workbench();review=window._review_dialog
    assert len(review.rows)==2
    window._review_locate(review.rows[0]);wait_for(lambda:window.active_clip_id==window.clips[0].id and not window.busy)
    wait_for(lambda:window.frame_index==3)
    assert window._retime_clock=='source'
    review.table.selectRow(0);review.set_checked(True)
    wait_for(lambda:window._autosave.worker is None)
    assert window.clips[0].review_marks and window.track.states[3]=='tracked'
    project=folder/'人工复核.lockframe';window._stash_clip();current=window._clip()
    save_project(str(project),current.info,current.settings,current.anchors,current.start,current.end,current.track,review_marks=current.review_marks)
    assert read_project(str(project))['review_marks']==current.review_marks
    checked=collect_issues(window.clips);assert sum(item.checked for item in checked)==1
    window.track.positions[3,0]+=.001
    assert not any(item.checked for item in collect_issues(window.clips))
    # A custom visual preset applies only curve fields, not audio or flow.
    store=PresetStore(folder/'用户曲线.json');config=deepcopy(before);config['optical_flow']=True
    ident=store.save('我的曲线',config)
    picker=PresetPicker(config,n,window,store);chosen=[];picker.applied.connect(chosen.append)
    picker.show();QApplication.processEvents();assert picker.list.count()==6 and not picker.list.item(0).icon().isNull()
    picker.list.setCurrentRow(3);picker.choose();assert chosen and chosen[0]['optical_flow']
    assert chosen[0]==apply_curve(config,builtins()[3]['curve'])
    store.remove(ident);assert store.read()==[]
    # Ordinary inspection pause accepts Continue; actual lost never does.
    result=deepcopy(window.track);result.performance.update(complete=False,computed_frames=[i<=5 for i in range(n)])
    window._tracking_paused(TrackingPaused(result,5,1,'manual'));window._sync_ui()
    assert window.continue_button.isEnabled()
    result.states[5]='lost';window._sync_ui();assert not window.continue_button.isEnabled()
    window._paused_tracking=None
    for width,height in ((1560,960),(1160,760)):
        window.resize(width,height);QApplication.processEvents()
        assert window.canvas.isVisible() and window.canvas.persistent_compare and window.canvas.height()>180
        window.grab().save(str(folder/f'workspace-{width}.png'))
    window._autosave.pending=False;window._autosave.enabled=False;window._autosave.timer.stop()
    wait_for(lambda:window._autosave.worker is None)
    assert window._autosave.close() and _manifest(window._autosave.folder)['clean']
    return dict(autosave_multiclip=True,recovery_async=True,curve_draft_recovered=True,source_output_clock=True,
        zoom_and_selection=True,review_navigation=True,project_review_marks=True,changed_track_invalidates_review=True,
        ordinary_pause=True,lost_gate=True,visual_presets=True,custom_presets=True,small_window=True,
        synthetic_measurements=True,accuracy_ground_truth=False)


def run_workflow_complete_check(directory):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from unittest.mock import patch
    from . import app,__version__
    from .demo import create_demo
    from .media import import_video,configure_cache_preferences
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    report=dict(version=__version__,ok=False);window=None
    try:
        source=folder/'工作区合成.mp4';create_demo(str(source),frames=36,width=320,height=180)
        info=import_video(str(source),threading.Event())
        with patch.object(app,'QSettings',lambda *args:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)):
            window=app.MainWindow()
        window._confirm_discard=lambda:True;window.resize(1560,960);window.show()
        report['checks']=exercise_workspace(window,info,folder)
        report['batch_resume']=exercise_batch_resume(window,info);report['ok']=True
    except Exception:report['error']=traceback.format_exc()
    finally:
        if window is not None:
            window._autosave.enabled=False;window._autosave.pending=False;window._autosave.timer.stop()
            if window._autosave.worker:window._autosave.worker.cancel();window._autosave.worker.wait(15000)
            QApplication.processEvents();window.close();QApplication.processEvents()
        configure_cache_preferences(None)
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['ok'] else 1
