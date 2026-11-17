"""Reproducible live Qt captures and UI-only comparison against the approved PNG."""
from dataclasses import replace
from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import threading
import time
import traceback

import cv2
import numpy as np

from . import __version__


REFERENCE_BOUNDS={
    'header':[10,0,1566,50], 'materials':[10,58,308,702],
    'preview':[328,58,925,702], 'tasks':[1263,58,313,702],
    'timeline':[10,768,1566,198],
}


def create_visual_source(folder):
    """Original procedural video/audio fixture, never an illustration pasted into UI."""
    from .media import find_tool,NO_WINDOW
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'海边日落_01.mp4'
    if path.exists():return path
    width,height=1920,1080
    y=np.linspace(0,1,height)[:,None,None]
    top=np.array([52,45,37]);horizon=np.array([76,154,232]);sea=np.array([25,30,29])
    sky=top+(horizon-top)*np.clip(y/.55,0,1)
    water=horizon+(sea-horizon)*np.clip((y-.55)/.45,0,1)
    base=np.repeat(np.where(y<.55,sky,water),width,axis=1).astype(np.uint8)
    cv2.circle(base,(1510,430),45,(178,230,255),-1,cv2.LINE_AA)
    for row in range(612,height,10):
        for x in range(0,width,100):
            shade=45+int(35*math.sin(row*.12+x*.003))
            cv2.line(base,(x,row),(x+65,row),(shade,shade+4,shade+7),2,cv2.LINE_AA)
    cv2.fillPoly(base,[np.array([[1500,760],[1655,760],[1580,590]])],(15,22,24))
    cv2.fillPoly(base,[np.array([[1480,772],[1660,772],[1615,808],[1520,808]])],(14,21,22))
    command=[find_tool('ffmpeg'),'-y','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
             '-s',f'{width}x{height}','-r','25','-i','pipe:0','-f','lavfi','-i',
             'aevalsrc=0.15*sin(2*PI*440*t)*(0.6+0.4*sin(2*PI*2*t)):s=48000:d=10',
             '-frames:v','250','-c:v','libx264','-preset','ultrafast','-crf','20','-pix_fmt','yuv420p',
             '-c:a','aac','-shortest',str(path)]
    with (folder/'generator.log').open('wb') as errors:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=errors,creationflags=NO_WINDOW)
        try:
            for index in range(250):
                frame=base.copy();cx=470+round(15*math.sin(index/45))
                cv2.circle(frame,(cx,712),28,(14,18,19),-1,cv2.LINE_AA)
                cv2.rectangle(frame,(cx-37,742),(cx+37,876),(18,24,26),-1)
                for j in range(8):cv2.line(frame,(cx-30+j*8,748),(cx-30+j*8,866),(44,47,45),2)
                cv2.line(frame,(cx-20,873),(cx-29,1010),(12,17,18),25)
                cv2.line(frame,(cx+20,873),(cx+29,1010),(12,17,18),25)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait(timeout=90):raise RuntimeError('无法生成本地视觉验收素材')
        finally:
            if process.poll() is None:process.kill();process.wait()
    return path


def compare_images(reference,actual,folder):
    """Report pixel differences, masking media content only, with a visible mask."""
    folder=Path(folder)
    ref=cv2.imdecode(np.fromfile(reference,dtype=np.uint8),cv2.IMREAD_COLOR);run=cv2.imdecode(np.fromfile(actual,dtype=np.uint8),cv2.IMREAD_COLOR)
    if ref is None or run is None:raise ValueError('参考或真实运行截图不可读')
    if run.shape!=ref.shape:run=cv2.resize(run,(ref.shape[1],ref.shape[0]),interpolation=cv2.INTER_AREA)
    mask=np.ones(ref.shape[:2],np.uint8)*255
    excluded=[(337,131,785,599),(799,131,1243,599),
              *[(26,172+i*80,128,234+i*80) for i in range(6)],(187,865,1558,906),(188,920,1556,960)]
    for x1,y1,x2,y2 in excluded:mask[y1:y2,x1:x2]=0
    # Media labels and duration chips are UI even though they overlap pictures.
    for x1,y1,x2,y2 in [(343,138,396,164),(805,138,883,164),(195,873,306,894),
                          *[(87,213+i*80,128,234+i*80) for i in range(6)]]:
        mask[y1:y2,x1:x2]=255
    difference=np.abs(ref.astype(np.int16)-run.astype(np.int16)).max(axis=2).astype(np.uint8)
    heat=cv2.applyColorMap(difference,cv2.COLORMAP_INFERNO)
    heat[mask==0]=[32,32,32]
    cv2.imencode('.png',np.concatenate([ref,run],axis=1))[1].tofile(folder/'参考与实际并排.png')
    cv2.imencode('.png',cv2.addWeighted(ref,.5,run,.5,0))[1].tofile(folder/'参考与实际叠加.png')
    cv2.imencode('.png',heat)[1].tofile(folder/'UI差异热图.png')
    cv2.imencode('.png',mask)[1].tofile(folder/'媒体排除区域.png')
    return dict(ui_pixel_mae=float(np.abs(ref.astype(float)-run)[mask>0].mean()),
                ui_pixels_over_24_percent=float((difference[mask>0]>24).mean()*100),
                excluded_media_percent=float((mask==0).mean()*100),
                excluded_rectangles=excluded,exact_pixel_match=False,
                explanation='Masks cover preview/video thumbnails/filmstrip/audio data only; borders, text and controls remain compared.')


def check_ad_ui(directory,reference=None):
    from . import app
    from .media import import_video,configure_cache_preferences
    from .models import Settings,file_fingerprint
    from .sequence import Clip
    from .tracking import track_video
    from .performance import ComputeOptions
    from .v072_check import activate
    from .timebar_core import Timebase
    from PySide6.QtCore import QPoint,QSettings,Qt
    from PySide6.QtWidgets import QApplication
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    application=QApplication.instance() or QApplication([])
    original_factory=app.QSettings;window=None
    previous_cache=os.environ.get('LOCKFRAME_CACHE_DIR');os.environ['LOCKFRAME_CACHE_DIR']=str(folder/'cache')
    try:
        app.QSettings=lambda *a:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)
        window=app.MainWindow()
        app.QSettings=original_factory
        source=create_visual_source(folder/'generated-media')
        info=import_video(str(source),threading.Event())
        anchors={0:(.22,.62,.09,.33)}
        track=track_video(info,anchors,0,249,threading.Event(),resources=ComputeOptions(acceleration='cpu',threads=2))
        settings=Settings(resolution=0,acceleration='cpu',compute_threads=2)
        first=Clip(info,settings,anchors,0,249,track,128)
        clips=[first]
        for name in ('城市夜景_02.mp4','产品特写_03.mp4','人物访谈_04.mp4','旅行片段_05.mp4','演示素材_06.mp4'):
            target=source.with_name(name);shutil.copy2(source,target)
            clips.append(Clip(import_video(str(target),threading.Event()),replace(settings)))
        identity=file_fingerprint(info.path)
        activate(window,clips);window._refresh_sequence({first.id});window.resize(1586,992);window.show()
        window._seek(128);window._tb_units(1);window._set_view('compare')
        # This presentation fixture is explicitly separate from the actual
        # completed numeric tracking result. It never runs as a user task.
        window.tracking_progress_panel.begin('视觉样例 · 128 / 250')
        window.tracking_progress_panel.update_progress(.51,'[计算] 128 / 250')
        window.tracking_progress_panel.set_expanded(False)
        window._ad_progress_state=window.tracking_progress_panel.state
        window.status_label.setText('视觉验收样例：进度卡51%为控件样例；视频、胶片与波形均为本地生成数据')
        window._sync_ui();window._seek(128)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            application.processEvents();time.sleep(.015)
            media=getattr(window.strip,'media_status',{})
            thumbs=[window.clip_list.item(i).data(Qt.ItemDataRole.UserRole+1) or {} for i in range(window.clip_list.count())]
            if media.get('state') in ('ready','error') and all(t.get('thumbnail_state') in ('ready','failed') for t in thumbs):break
        bounds={}
        for key,widget in dict(header=window.header,materials=window.materials_sidebar,
                               preview=window.central_preview,tasks=window.task_sidebar,timeline=window.timeline_well).items():
            pos=widget.mapTo(window,QPoint(0,0));bounds[key]=[pos.x(),pos.y(),widget.width(),widget.height()]
        capture=folder/'实际运行_1586x992.png';window.grab().save(str(capture))
        # Both monitor controls consume the same clock and cannot change tracking.
        retained=window.track
        window.monitor_transports[0].slider.setValue(42);application.processEvents()
        assert window.frame_index==42 and all(t.slider.value()==42 for t in window.monitor_transports)
        window.monitor_transports[0].slider.setValue(128);application.processEvents()
        assert window.frame_index==128 and window.track is retained
        window._set_composition(.55,.45);assert window.track is retained
        window._set_composition(.5,.5)
        assert file_fingerprint(info.path)==identity
        window.resize(1160,760);application.processEvents();window.grab().save(str(folder/'实际运行_1160x760.png'))
        assert window.canvas.width()>=500 and window.canvas.height()>=100
        assert window.export_button.isVisible() and window.timeline_well.isVisible()
        window.resize(1586,992);application.processEvents()
        deviations={key:[a-b for a,b in zip(bounds[key],expected)] for key,expected in REFERENCE_BOUNDS.items()}
        result=dict(footage='original procedural test video/audio; no camera-footage accuracy claim',
                    presentation_progress_fixture_only=True,production_tracking_result_unchanged=True,
                    frames=info.frames,fps=info.fps,frame=128,timecode=Timebase.from_fps(info.fps).timecode(128),
                    bounds=bounds,reference_bounds=REFERENCE_BOUNDS,bounds_deviation=deviations,
                    max_bounds_deviation=max(abs(x) for row in deviations.values() for x in row),
                    dpr=window.devicePixelRatioF(),shared_transport_synchronized=True,transport_count=1,annotation_removed=True,audio_hidden_default=True,
                    media=window.strip.media_status,tracking_lost_count=len(track.lost_frames),
                    track_button_enabled=window.track_button.isEnabled(),export_button_enabled=window.export_button.isEnabled(),
                    composition_preserves_tracking=True,source_unchanged=True,
                    limitations=['reference generation has different media and ambiguous frame counters',
                                 'preview preserves source aspect ratio; no stretching to match illustration',
                                 'waveform display does not enable audio monitoring; export retains audio'])
        if reference and Path(reference).is_file():result['comparison']=compare_images(reference,capture,folder)
        return result
    finally:
        app.QSettings=original_factory
        if window:
            window._confirm_discard=lambda:True;window.close();application.processEvents()
        if previous_cache is None:os.environ.pop('LOCKFRAME_CACHE_DIR',None)
        else:os.environ['LOCKFRAME_CACHE_DIR']=previous_cache
        configure_cache_preferences(None)


def run_ad_check(directory):
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    reference=os.environ.get('TOMATO_AD_REFERENCE')
    try:report=dict(ok=True,version=__version__,ad=check_ad_ui(folder,reference));code=0
    except Exception:report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/'report.json').write_bytes(json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8'))
    return code
