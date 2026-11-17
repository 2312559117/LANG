"""Run current-clip reset through the real GUI, tracker, persistence and export."""
from pathlib import Path
import json
import os
import threading
import traceback

from . import __version__
from .batch_export import prepare_exports
from .export import export_video
from .media import cache_root, configure_cache_preferences, file_fingerprint
from .models import Settings, save_project, read_project
from .sequence import Clip, save_sequence, read_sequence
from .tracking import track_video
from .v072_check import create_correction_source, correction_box, activate, wait_job


def check_reset(folder):
    from . import app
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    old_cache=os.environ.get("LOCKFRAME_CACHE_DIR")
    os.environ["LOCKFRAME_CACHE_DIR"]=str(folder/"cache")
    window=None
    try:
        application=QApplication.instance() or QApplication([])
        original_settings=app.QSettings
        app.QSettings=lambda *args:QSettings(str(folder/"prefs.ini"),QSettings.Format.IniFormat)
        try:
            window=app.MainWindow()
        finally:
            app.QSettings=original_settings
        window.resize(1440,920);window.show();application.processEvents()
        info=create_correction_source(folder/"source")
        fingerprint=file_fingerprint(info.path)
        cancel=threading.Event()
        # A separate material keeps a complete result throughout the reset.
        other_anchors={0:correction_box(0),12:correction_box(12)}
        other_result=track_video(info,other_anchors,0,47,cancel)
        assert not other_result.lost_frames
        other=Clip(info,Settings(acceleration="cpu",compute_threads=2),other_anchors,0,47,other_result)
        settings=Settings(acceleration="cpu",compute_threads=2,segmented_tracking=True,
                          segment_frames=10,target_x=.4,resolution=0)
        current=Clip(info,settings,{0:correction_box(0)},0,47)
        activate(window,[current,other])
        window.auto_pause_check.setChecked(True)
        window._start_tracking()
        assert window.busy and not window.reset_tracking_button.isEnabled()
        window._reset_current_tracking()
        assert window.anchors=={0:correction_box(0)}
        wait_job(window)
        assert window._paused_tracking and window._paused_tracking["reason"]=="lost"
        lost_frame=window._paused_tracking["frame"]
        assert lost_frame==12
        window._select_roi(correction_box(12))
        assert len(window.anchors)==2 and window._paused_tracking["corrected"] and not window.busy
        assert list((cache_root()/"tracking-checkpoints").glob("*.npz"))
        old_other_positions=other.track.positions.copy()
        before_settings=window.settings();before_range=(current.start,current.end)
        window.reset_tracking_button.click()
        assert not window.anchors and window.track is window._retrack_previous is window._paused_tracking is None
        assert window.plan is None
        assert window.canvas.box is None and window.canvas.processed is None
        assert window.settings()==before_settings and (current.start,current.end)==before_range
        assert other.track is other_result and other.anchors==other_anchors
        assert (other.track.positions==old_other_positions).all()
        assert window.continue_button.isVisible() and not window.continue_button.isEnabled() and not window.track_button.isEnabled()
        assert not list((cache_root()/"tracking-checkpoints").glob("*.npz"))
        assert not list((cache_root()/"tracking-checkpoints").glob("*.locksequence"))
        window._stash_clip()
        window._activate_clip(other.id);window._activate_clip(current.id)
        assert window.track is None and not window.anchors and window.canvas.box is None
        window.reset_tracking_button.click() # Idempotent, no automatic matching.
        assert not window.busy
        application.processEvents();window.grab().save(str(folder/"重置当前素材双预览.png"))
        project=folder/"重置后.lockframe"
        save_project(str(project),info,window.settings(),{},0,47,None)
        payload=read_project(str(project))
        assert payload["anchors"]=={} and payload["track"] is None
        sequence=folder/"重置后.locksequence"
        save_sequence(sequence,window.clips)
        restored=read_sequence(sequence)
        assert not restored[0].anchors and restored[0].track is None and restored[1].track is not None
        tasks=prepare_exports(window.clips,folder/"batch")
        assert tasks[0].reason and not tasks[1].reason
        window._confirm_discard=lambda:True
        window.load_video(info.path,project=payload);wait_job(window)
        assert not window.anchors and window.track is window._retrack_previous is window._paused_tracking is None
        assert window.settings()==before_settings
        # Both appearances need their own new reference. No old result/resume is passed.
        window._seek(0);window._select_roi(correction_box(0))
        window._seek(12);window._select_roi(correction_box(12))
        assert not window.busy and window.track is window._retrack_previous is None
        window._start_tracking();wait_job(window)
        assert window.track.performance["complete"] and not window.track.lost_frames
        assert window.track.performance["resumed_frames"]==0
        assert window.track.performance["reused_frames"]==0
        output=export_video(info,window.track,window.settings(),str(folder/"重新选点跟踪.mp4"),cancel)
        assert output["frames"]==48
        assert file_fingerprint(info.path)==fingerprint
        return dict(footage="synthetic appearance-switch only; original car footage pending",
                    real_lost_frame_zero_based=lost_frame,all_targets_cleared=True,
                    checkpoints_and_recovery_records_cleared=True,parameters_preserved=True,
                    other_material_result_preserved=True,empty_project_and_sequence=True,
                    empty_saved_project_reopened=True,busy_guard=True,no_automatic_restart=True,
                    new_run_reused_frames=window.track.performance["reused_frames"],
                    new_run_resumed_frames=window.track.performance["resumed_frames"],
                    original_source_unchanged=True,output=output)
    finally:
        if window is not None:
            if window.busy:
                window._cancel();wait_job(window)
            window._confirm_discard=lambda:True
            window.close();QApplication.processEvents()
        if old_cache is None:os.environ.pop("LOCKFRAME_CACHE_DIR",None)
        else:os.environ["LOCKFRAME_CACHE_DIR"]=old_cache
        configure_cache_preferences(None)


def run_reset_check(directory):
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    try:
        report=dict(ok=True,version=__version__,tracking_reset=check_reset(folder));code=0
    except Exception:
        report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return code
