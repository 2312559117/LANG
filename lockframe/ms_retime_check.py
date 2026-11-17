"""Source/frozen acceptance: the public UI, project, preview and renderer path."""
from pathlib import Path
import json
import threading
import time
import traceback
from copy import deepcopy


def check_ms_retime(folder):
    import numpy as np
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from . import app
    from .demo import create_demo,demo_frame
    from .media import import_video,probe
    from .models import Settings,save_project,read_project
    from .sequence import Clip
    from .tracking import track_video
    from .export import export_video
    from .batch_export import prepare_exports,run_exports
    from .time_remap_model import make_constant_speed,default_config
    from .time_remap_map import output_frame_map
    from .retime_frames import probe_native
    from .retime_frames import run_process
    from .media import find_tool
    from .models import TrackResult
    from .v072_check import activate
    directory=Path(folder);directory.mkdir(parents=True,exist_ok=True)
    import os
    os.environ['LOCKFRAME_CACHE_DIR']=str(directory/'cache')
    source=directory/'变速验收含声音.mp4';cancel=threading.Event()
    create_demo(str(source),frames=24,width=320,height=180)
    info=import_video(str(source),cancel)
    anchor=demo_frame(0,320,180)[1]
    track=track_video(info,{0:anchor},0,23,cancel)
    assert not track.lost_frames
    settings=Settings(acceleration='cpu',compute_threads=2,resolution=0)
    clip=Clip(info,settings,{0:anchor},0,23,track=track)
    original=app.QSettings;window=None
    try:
        app.QSettings=lambda *args:QSettings(str(directory/'prefs.ini'),QSettings.Format.IniFormat)
        window=app.MainWindow();window._confirm_discard=lambda:True
        window.resize(1560,960);window.show();activate(window,[clip])
        tracked=window.track;positions=tracked.positions.copy()
        # Real mouse interaction in source and frozen EXE. Signal-only tests
        # cannot detect a read-only canvas or disabled hit testing.
        from PySide6.QtCore import Qt,QPoint
        from PySide6.QtTest import QTest
        def drag(canvas,index,delta):
            point=canvas.pixel(*canvas.nodes()[index]).toPoint()
            QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=point)
            QTest.mouseMove(canvas,point+delta,20)
            QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=point+delta)
            QApplication.processEvents()
        editable=default_config();editable['enabled']=True
        window._apply_retime_config(editable);QApplication.processEvents()
        drag(window.retime_panel.plot,2,QPoint(0,-15))
        assert window.settings().retime['basic']['speeds'][2]>1.
        assert window.track is tracked
        window._open_retime_editor();draft_editor=window._retime_editor
        original=deepcopy(draft_editor.draft)
        drag(draft_editor.canvas,3,QPoint(0,-24))
        assert draft_editor.draft['basic']['speeds'][3]>1.
        assert window.settings().retime['basic']['speeds'][3]==1.
        draft_editor.undo();assert draft_editor.draft==original
        draft_editor.redo();assert draft_editor.apply_changes()
        assert window.settings().retime['basic']['speeds'][3]>1.
        draft_editor.set_constant_speed(1.)
        draft_editor.view.setCurrentIndex(draft_editor.view.findData(True))
        drag(draft_editor.canvas,0,QPoint(0,-25))
        assert draft_editor.draft['points'][0]['outgoing_speed']>1.
        assert draft_editor.apply_changes() and window.track is tracked
        draft_editor.reject()
        slow=make_constant_speed(speed=.5);slow.update(optical_flow=True,preview_flow=True,audio_policy='mapped',flow_backend='dis')
        window._apply_retime_config(slow)
        assert window.track is tracked and np.array_equal(positions,window.track.positions)
        window.retime_panel.clock.setCurrentIndex(1)
        assert len(window._retime_map())==48
        window._seek_output(7)
        wanted=window._retime_output_step
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            QApplication.processEvents();time.sleep(.02)
            if '停帧代理光流预览' in window.preview_note.text():break
        assert '停帧代理光流预览' in window.preview_note.text(),window.preview_note.text()
        assert window._retime_output_step==wanted
        window._open_retime_editor();editor=window._retime_editor
        assert editor is not None and editor.isVisible()
        editor.reject()
        for width,height in ((1560,960),(1160,760)):
            window.resize(width,height);QApplication.processEvents()
            panel=window.retime_panel
            assert panel.plot.geometry().bottom()<panel.summary.geometry().top()
            assert panel.summary.geometry().bottom()<panel.flow.geometry().top()
            last=panel.body.layout().itemAt(panel.body.layout().count()-1).widget()
            assert panel.body.rect().contains(last.geometry())
            assert window.canvas.source_image is not None and window.canvas.processed_image is not None
            assert window.export_button.visibleRegion().contains(window.export_button.rect())
            window.grab().save(str(directory/f'retime-{width}x{height}.png'))
        project=directory/'新版曲线.lockframe'
        save_project(str(project),info,window.settings(),{0:anchor},0,23,track)
        saved=read_project(str(project))
        assert saved['settings']['retime']['enabled']
        single=export_video(info,track,window.settings(),str(directory/'DIS慢放.mp4'),cancel)
        assert single['frames']==48 and single['actual_backend']=='dis' and single['flow_pairs']<=23
        metadata=probe(single['path'])
        assert metadata.has_audio and int(probe_native(single['path'],cancel)['nb_frames'])==48
        assert abs(metadata.duration-48/info.fps)<=1/info.fps
        clip.settings=deepcopy(window.settings())
        tasks=prepare_exports([clip],str(directory/'批量'))
        clip.settings.retime=make_constant_speed(speed=2.)
        outcome=run_exports(tasks,cancel)
        assert outcome[0]['state']=='完成',outcome
        assert outcome[0]['result']['frames']==48
        reverse=make_constant_speed(speed=-1.);reverse['audio_policy']='mapped'
        window._apply_retime_config(reverse)
        window._transport_boundary(True)
        assert window._retime_output_step==0 and window.frame_index==23
        window._transport_boundary(False)
        assert window._retime_output_step==23 and window.frame_index==0
        settings.retime=reverse
        backwards=export_video(info,track,settings,str(directory/'倒放静音.mp4'),cancel)
        assert backwards['frames']==24 and backwards['audio_policy']=='muted'
        assert not probe(backwards['path']).has_audio
        freeze=make_constant_speed(speed=1.)
        for point in freeze['points']:point['source']=.4
        settings.retime=freeze
        frozen=export_video(info,track,settings,str(directory/'定格.mp4'),cancel)
        assert frozen['frames']==24
        # Exercise bundled FFmpeg zscale/x265 and the native dispatch in EXE,
        # not just a source-only test. This is generated HDR, not camera HDR.
        hdr_source=directory/'合成HDR10.mp4'
        run_process([find_tool('ffmpeg'),'-v','error','-y','-f','lavfi','-i','testsrc2=s=96x64:r=12:d=0.5',
            '-pix_fmt','yuv420p10le','-c:v','libx265','-preset','ultrafast','-x265-params',
            'pools=2:frame-threads=2:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:hdr10=1:'
            'master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,50):max-cll=1000,400',
            '-color_primaries','bt2020','-color_trc','smpte2084','-colorspace','bt2020nc',str(hdr_source)],cancel)
        hdr=import_video(str(hdr_source),cancel)
        assert hdr.bit_depth==10 and hdr.hdr and hdr.content_light=='1000,400'
        hdr_track=TrackResult(0,5,np.tile([.5,.5],(6,1)),np.tile([.25,.25,.5,.5],(6,1)),
            np.ones(6),['manual']*6,anchors={i:(.25,.25,.5,.5) for i in range(6)})
        hdr_output=export_video(hdr,hdr_track,Settings(acceleration='cpu',resolution=0,strength=0,max_zoom=1),
            str(directory/'原生HDR10成片.mp4'),cancel)
        native=probe_native(hdr_output['path'],cancel)
        assert native['pix_fmt']=='yuv420p10le' and native['color_transfer']=='smpte2084'
        assert probe(hdr_output['path']).content_light=='1000,400'
        config=default_config();config['enabled']=True;config['basic']['speeds']=[2.,.7,1.,.7,2.]
        assert output_frame_map(config,24).shape[0]>0
        return dict(source_frames=24,slow_frames=48,shared_map=True,source_timeline_retained=True,
            native_dis=single['actual_backend'],flow_pairs=single['flow_pairs'],audio=single['audio_method'],
            batch_snapshot_immutable=True,reverse_muted=True,freeze=True,tracking_unchanged=True,
            editor_transaction=True,stop_frame_flow_preview=True,dual_preview=True,
            sidebar_curve_mouse_drag=True,editor_curve_mouse_drag=True,speed_curve_mouse_drag=True,
            mouse_drag_undo_apply=True,
            generated_hdr10_import_export=True,static_hdr_metadata=True,
            nvof='not implemented; DIS only',footage='synthetic, not real-car accuracy evidence')
    finally:
        if window is not None:window.close();QApplication.processEvents()
        app.QSettings=original


def run_ms_retime_check(folder):
    from . import __version__
    directory=Path(folder);directory.mkdir(parents=True,exist_ok=True)
    report=dict(version=__version__,ok=False)
    try:
        report['checks']=check_ms_retime(directory);report['ok']=True
    except Exception:
        report['error']=traceback.format_exc()
    (directory/'ms-retime-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['ok'] else 1
