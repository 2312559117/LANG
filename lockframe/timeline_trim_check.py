"""Source and frozen Qt handle edits, sharp original film and real audio export."""
from pathlib import Path
import json
import os
import pickle
import subprocess
import threading
import time
import traceback

import cv2
import numpy as np

from . import __version__


def check_timeline_trim(directory):
    from . import app
    from .core_check import create_source,verify_export
    from .media import configure_cache_preferences,find_tool,NO_WINDOW,probe
    from .models import file_fingerprint,read_project
    from .sequence import save_sequence,read_sequence
    from .export import export_video
    from .batch_export import prepare_exports,run_exports
    from .v072_check import activate,wait_job
    from .timeline_media import MediaSource,make_request,read_film,WorkCancel
    from PySide6.QtCore import QSettings,QPoint,Qt
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    application=QApplication.instance() or QApplication([])
    factory=app.QSettings;window=None
    prior=os.environ.get('LOCKFRAME_CACHE_DIR');os.environ['LOCKFRAME_CACHE_DIR']=str(folder/'cache')
    try:
        app.QSettings=lambda *a:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)
        window=app.MainWindow();app.QSettings=factory
        clip=create_source(folder/'generated-source');identity=file_fingerprint(clip.info.path)
        activate(window,[clip]);window.resize(1586,992);window.show();application.processEvents()
        original=window.track;bar=window.strip;row=round(bar.row_rect('video').center().y())
        def drag(a,b):
            QTest.mousePress(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(a)),row))
            QTest.mouseMove(bar,QPoint(round(bar.x(b)),row))
            QTest.mouseRelease(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(b)),row))
        drag(2,4);drag(10,8)
        assert (window.start_spin.value(),window.end_spin.value())==(5,8)
        assert window.track.start==4 and window.track.count==4 and not window.busy
        window._stash_clip();project=folder/'裁剪工程.lockframe';window.project_path=str(project)
        assert window._save_project();saved=read_project(str(project))
        assert saved['start']==4 and saved['end']==7
        sequence=folder/'裁剪序列.locksequence';save_sequence(sequence,window.clips)
        assert read_sequence(sequence)[0].track.count==4
        tasks=prepare_exports([clip],str(folder/'批量原声'))
        window._tb_operation('trim',(5,7));assert pickle.loads(tasks[0].snapshot).start==4
        window._tb_operation('undo');assert window.track.count==4
        single=folder/'裁剪原声.mp4'
        export_video(window.info,window.track,window.settings(),str(single),threading.Event())
        single_report=verify_export(single,clip)
        outcomes=run_exports(tasks,threading.Event());assert outcomes[0]['state']=='完成'
        batch_report=verify_export(outcomes[0]['destination'],clip)
        assert file_fingerprint(clip.info.path)==identity
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            application.processEvents();time.sleep(.01)
            if bar.media_status['state']=='ready':break
        window.grab().save(str(folder/'时间线裁剪_1586x992.png'))
        window.resize(1160,760);application.processEvents()
        assert window.canvas.width()>=500 and window.export_button.visibleRegion().boundingRect().height()==window.export_button.height()
        window.grab().save(str(folder/'时间线裁剪_1160x760.png'))
        window._tb_operation('undo');window._tb_operation('undo');assert window.track is original
        window._confirm_discard=lambda:True;window.load_video(clip.info.path,project=saved);wait_job(window)
        assert window.track.start==4 and window.track.count==4 and window.frame_index==4 and window.export_button.isEnabled()
        # Full-resolution portrait chart makes the previous upscale loss measurable.
        chart=np.zeros((1280,720,3),np.uint8)
        for x in range(0,720,12):chart[:,x:x+6]=255
        png=folder/'原始竖屏细线.png';cv2.imencode('.png',chart)[1].tofile(png)
        video=folder/'原始竖屏细线.mkv'
        subprocess.run([find_tool('ffmpeg'),'-v','error','-y','-loop','1','-i',str(png),
                        '-frames:v','2','-r','10','-c:v','ffv1',str(video)],check=True,
                       capture_output=True,creationflags=NO_WINDOW,timeout=30)
        source=MediaSource.from_info(probe(str(video)))
        request=make_request(source,0,2,count=1,pixel_size=(320,100))
        image=read_film(request,WorkCancel())[0]
        old=cv2.resize(cv2.resize(chart,(45,80),interpolation=cv2.INTER_AREA),(320,569),interpolation=cv2.INTER_LINEAR)
        old=old[(569-100)//2:(569+100)//2]
        ratio=float(image.std()/max(.01,old.std()));assert ratio>3
        cv2.imencode('.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))[1].tofile(folder/'原片生成的新胶片.png')
        cv2.imencode('.png',old)[1].tofile(folder/'旧算法模拟对照.png')
        return dict(footage='generated indexed source/audio and original procedural resolution chart; not real-car accuracy',
                    source_range=[4,7],frames=4,mouse_handles=True,undo_track_identity_restored=True,
                    single_export=single_report,batch_export=batch_report,snapshot_isolated=True,
                    project_sequence_saved=True,cropped_project_reopened=True,source_unchanged=True,device_pixel_ratio=window.devicePixelRatioF(),
                    sharpness_chart=dict(width=image.shape[1],height=image.shape[0],
                                         new_std=float(image.std()),old_upscale_std=float(old.std()),contrast_ratio=ratio,
                                         explanation='Resolution-chart contrast only; no tracking accuracy or universal perceived sharpness claim'))
    finally:
        app.QSettings=factory
        if window:window._confirm_discard=lambda:True;window.close();application.processEvents()
        if prior is None:os.environ.pop('LOCKFRAME_CACHE_DIR',None)
        else:os.environ['LOCKFRAME_CACHE_DIR']=prior
        configure_cache_preferences(None)


def run_timeline_trim_check(directory):
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    try:report=dict(ok=True,version=__version__,timeline_trim=check_timeline_trim(folder));code=0
    except Exception:report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return code
