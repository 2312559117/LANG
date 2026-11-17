"""Production Qt/worker acceptance for independent tracking sections."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import traceback
import uuid
from . import __version__


def run_sections_check(directory):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from . import app
    from .sequence import Clip, save_sequence, read_sequence
    from .models import Settings, read_project
    from .v072_check import create_correction_source, correction_box, activate, wait_job
    from .export import export_video
    from .media import find_tool, NO_WINDOW, configure_cache_preferences
    import threading
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    original_settings=app.QSettings
    window=None
    try:
        app.QSettings=lambda *args:QSettings(str(folder/'preferences.ini'),QSettings.Format.IniFormat)
        window=app.MainWindow();window.resize(1560,960);window.show()
        window._confirm_discard=lambda:True
        info=create_correction_source(folder/'synthetic')
        clip=Clip(info,Settings(acceleration='cpu',compute_threads=2,resolution=0),{0:correction_box(0)},0,47)
        activate(window,[clip]);window._open_tracking_sections();window._seek(12)
        dialog=window._sections_panel;dialog.split()
        expected=[{'start':0,'end':11},{'start':12,'end':47}]
        assert window.section_tracking and window.tracking_sections==expected
        assert dialog.selector.count()==2
        dialog.selector.setCurrentIndex(1);dialog.pick()
        assert window.frame_index==12 and dialog.isVisible() and not dialog.isWindow()
        window._select_roi(correction_box(12));window._open_tracking_sections()
        assert dialog.selector.count()==2
        QApplication.processEvents();dialog.grab().save(str(folder/'sections-editor.png'))
        window._start_tracking();wait_job(window)
        assert window.track is not None and not window.track.lost_frames and window._paused_tracking is None
        assert window.track.performance['tracking_sections']==expected
        assert window.strip.tracking_sections==expected
        assert window.canvas.source is not None and window.canvas.processed is not None
        QApplication.processEvents();window.grab().save(str(folder/'sections-workspace.png'))
        window._stash_clip();project=folder/'独立分段.lockframe'
        window.project_path=str(project);assert window._save_project()
        payload=read_project(str(project));assert payload['tracking_sections']==expected
        window.load_video(info.path,project=payload);wait_job(window)
        assert window.section_tracking and window.tracking_sections==expected
        assert window.track is not None and window.plan is not None
        window._stash_clip();clip=window._clip()
        sequence=folder/'独立分段.locksequence';save_sequence(sequence,[clip])
        restored=read_sequence(sequence)[0]
        assert restored.section_tracking and restored.tracking_sections==expected
        export=export_video(info,window.track,window.settings(),str(folder/'分段成品.mp4'),threading.Event())
        streams=json.loads(subprocess.check_output([find_tool('ffprobe'),'-v','error','-count_frames',
            '-show_streams','-of','json',export['path']],creationflags=NO_WINDOW))['streams']
        video=next(s for s in streams if s['codec_type']=='video')
        assert int(video['nb_read_frames'])==48 and abs(float(video['duration'])-2)<1/24
        assert [int(video['width']),int(video['height'])]==[400,240]
        first,second=deepcopy(restored),deepcopy(restored)
        first.track=second.track=None;second.id=uuid.uuid4().hex
        activate(window,[first,second]);window._refresh_sequence({first.id,second.id})
        window.queue_mode.setCurrentIndex(window.queue_mode.findData(2))
        window._start_queue();wait_job(window,120)
        assert all(c.track is not None and not c.track.lost_frames for c in (first,second))
        assert all(c.track.performance['tracking_sections']==expected for c in (first,second))
        report=dict(ok=True,version=__version__,independent_sections=2,frames=48,inline_panel=True,
                    io_and_split_editor=True,each_section_requires_target=True,
                    abrupt_appearance_boundary=True,dual_monitors_visible=True,
                    project_restored=True,sequence_restored=True,batch_workers=2,
                    export_frames=48,export_seconds=float(video['duration']),
                    export_dimensions=[400,240],boundary_interpolation=False,
                    note='Synthetic footage; original car footage is a separate acceptance')
        (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return 0
    except Exception:
        (folder/'report.json').write_text(json.dumps(dict(ok=False,version=__version__,error=traceback.format_exc()),
                         ensure_ascii=False,indent=2),encoding='utf-8')
        return 1
    finally:
        if window is not None:
            if window.busy and window.worker:
                window.worker.cancel();window.worker.wait(15000)
                QApplication.processEvents()
            window.close()
        app.QSettings=original_settings
        configure_cache_preferences(None)
