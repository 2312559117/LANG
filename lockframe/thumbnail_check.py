"""Actual list thumbnails, import/sequence/trim/failed-source checks in source and EXE."""
from pathlib import Path
import json
import os
import threading
import time
import traceback
import cv2
import numpy as np

from . import __version__
from .media import configure_cache_preferences
from .sequence import read_sequence,save_sequence,restore_sequence
from .v072_check import wait_job


def row_data(window,id):
    from PySide6.QtCore import Qt
    return window._thumbnail_items[id].data(Qt.ItemDataRole.UserRole+1)


def spin(predicate,timeout=15):
    from PySide6.QtWidgets import QApplication
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        QApplication.processEvents()
        if predicate():return
        time.sleep(.005)
    raise TimeoutError('素材缩略图加载未在期限内完成')


def make_sources(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    specs=[('红色镜头.avi',(128,72),[(0,0,240),(0,240,0),(240,0,0)]),
           ('黄色镜头.avi',(128,72),[(0,240,240)]*3),
           ('竖屏镜头.avi',(72,128),[(240,240,0)]*3)]
    paths=[]
    for name,size,colors in specs:
        path=folder/name
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),5,size)
        assert writer.isOpened()
        try:
            for color in colors:writer.write(np.full((size[1],size[0],3),color,np.uint8))
        finally:writer.release()
        paths.append(str(path))
    return paths


def check_thumbnails(folder):
    from . import app
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    old_cache=os.environ.get('LOCKFRAME_CACHE_DIR')
    os.environ['LOCKFRAME_CACHE_DIR']=str(folder/'cache')
    window=None
    try:
        application=QApplication.instance() or QApplication([])
        factory=app.QSettings
        app.QSettings=lambda *args:QSettings(str(folder/'preferences.ini'),QSettings.Format.IniFormat)
        try:window=app.MainWindow()
        finally:app.QSettings=factory
        window.resize(1440,920);window.show();application.processEvents()
        window._confirm_discard=lambda:True
        paths=make_sources(folder)
        window._import_paths(paths);wait_job(window)
        assert len(window.clips)==3
        ids=[clip.id for clip in window.clips]
        spin(lambda:all(row_data(window,id)['thumbnail_state']=='ready' for id in ids))
        first=window.clips[0]
        color=row_data(window,first.id)['thumbnail'].toImage().pixelColor(0,0)
        assert color.red()>200 and color.green()<30
        portrait=row_data(window,ids[2])['thumbnail']
        assert portrait.height()>portrait.width()
        active=window.active_clip_id
        window._refresh_sequence({ids[0],ids[1]})
        assert window._selected_clip_ids()=={ids[0],ids[1]}
        assert window.active_clip_id==active
        window._move_clips(1)
        assert [clip.id for clip in window.clips]==[ids[2],ids[0],ids[1]]
        assert window._selected_clip_ids()=={ids[0],ids[1]}
        window._apply_trim(1,2)
        spin(lambda:row_data(window,first.id)['thumbnail_state']=='ready')
        color=row_data(window,first.id)['thumbnail'].toImage().pixelColor(0,0)
        assert color.green()>200 and color.red()<30
        assert window._clip(first.id).start==1
        window._stash_clip()
        sequence=folder/'素材预览.locksequence';save_sequence(sequence,window.clips)
        assert all('thumbnail' not in clip for clip in json.loads(sequence.read_text(encoding='utf-8'))['clips'])
        # Use the same restore Job/callback as the actual Open Sequence action.
        window._pending_sequence_path=str(sequence)
        window._job(restore_sequence,(read_sequence(sequence),),window._sequence_restored,'检查打开素材序列',keep_partial=True)
        wait_job(window)
        ids=[next(clip.id for clip in window.clips if clip.info.path==path) for path in paths]
        spin(lambda:all(row_data(window,id)['thumbnail_state']=='ready' for id in ids))
        assert not any(clip.track is not None or clip.anchors for clip in window.clips)
        window.grab().save(str(folder/'素材列表缩略图.png'))
        missing=window._clip(ids[1]);status=missing.status
        path=Path(missing.info.path);path.rename(path.with_name('验收临时移开的原片.avi'))
        window._refresh_visible_thumbnails()
        spin(lambda:row_data(window,missing.id)['thumbnail_state']=='failed')
        assert row_data(window,missing.id)['thumbnail'] is None and missing.status==status
        assert all(row_data(window,id)['thumbnail_state']=='ready' for id in (ids[0],ids[2]))
        return dict(footage='generated RGB clips only; original car footage pending',
                    imported_materials=3,all_thumbnails_without_clicking=True,portrait_aspect_preserved=True,
                    trim_start_updated=True,multiselect_order_active_preserved=True,sequence_reopened=True,
                    missing_source_placeholder=True,tracking_state_unchanged=True,
                    background=window._thumbnails.statistics())
    finally:
        if window:
            window._confirm_discard=lambda:True;window.close();application.processEvents()
        if old_cache is None:os.environ.pop('LOCKFRAME_CACHE_DIR',None)
        else:os.environ['LOCKFRAME_CACHE_DIR']=old_cache
        configure_cache_preferences(None)


def run_thumbnail_check(directory):
    folder=Path(directory).resolve();folder.mkdir(parents=True,exist_ok=True)
    try:report=dict(ok=True,version=__version__,thumbnails=check_thumbnails(folder));code=0
    except Exception:report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return code
