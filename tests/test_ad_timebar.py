import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint,Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from lockframe.models import VideoInfo
from lockframe.timebar_ui import Timebar
from lockframe.timeline_media import MediaOverview


@pytest.fixture
def bar():
    app=QApplication.instance() or QApplication([])
    bar=Timebar();bar.ad_style=True;bar.setFixedHeight(222);bar.resize(1566,222)
    bar.load_state({},250,25);bar.anchors={20:(0.,0.,.2,.2),80:(.1,.1,.2,.2)}
    bar.sync_data();bar.show();app.processEvents()
    yield bar
    bar.close()


def drag(bar,a,b):
    QTest.mousePress(bar,Qt.MouseButton.LeftButton,pos=a)
    QTest.mouseMove(bar,b)
    QTest.mouseRelease(bar,Qt.MouseButton.LeftButton,pos=b)


def test_two_core_rows_optional_audio_and_no_annotation_row(bar):
    assert bar.left==174 and bar.state['timecode']
    assert bar.row_rect('mark').isEmpty() and bar.row_rect('audio').isEmpty()
    for height in (222,180):
        bar.setFixedHeight(height);bar.set_compact(height==180)
        rows=[bar.row_rect(n) for n in ('correction','video')]
        assert all(row.height()>=25 for row in rows)
        assert rows[0].bottom()==rows[1].top() and rows[-1].bottom()<=height
        assert not bar.grab().isNull()
        bar.set_audio_visible(True)
        assert bar.row_rect('audio').height()>0 and bar.row_rect('mark').isEmpty()
        bar.set_audio_visible(False)
    bar.load_state({'timecode':False},250,25);assert not bar.state['timecode']


def test_dynamic_key_lane_and_play_range_keep_source_semantics(bar):
    operations=[];bar.operation.connect(lambda *args:operations.append(args))
    key=lambda frame:QPoint(round(bar.x(frame)),round(bar.key_y))
    drag(bar,key(20),key(24))
    assert operations==[('move',((20,),4))]
    assert set(bar.anchors)=={20,80}  # Only the host commits authoritative anchors.
    operations.clear()
    drag(bar,QPoint(round(bar.x(0)),round(bar.range_y)),QPoint(round(bar.x(5)),round(bar.range_y)))
    assert operations==[('play_range',(5,250))]
    assert (bar.start,bar.end)==(0,249)
    operations.clear();bar.editable=False
    drag(bar,key(20),key(25));QTest.keyClick(bar,Qt.Key.Key_Delete)
    assert not operations


def test_hand_pan_and_eye_do_not_mutate_tracking_or_export_state(bar):
    bar.zoom(4);before=bar.state['scroll'];anchors=dict(bar.anchors);operations=[]
    bar.operation.connect(lambda *args:operations.append(args))
    bar.tool_mode='pan';drag(bar,QPoint(800,60),QPoint(500,60))
    assert bar.state['scroll']>before and not operations
    old=dict(bar.state);signals=[];bar.overlayVisibilityChanged.connect(lambda *args:signals.append(args))
    QTest.mouseClick(bar,Qt.MouseButton.LeftButton,pos=QPoint(59,round(bar.row_rect('correction').center().y())))
    assert signals==[('correction',False)] and bar.anchors==anchors and bar.state==old
    bar.set_audio_visible(True)
    QTest.mouseClick(bar,Qt.MouseButton.LeftButton,pos=QPoint(59,round(bar.row_rect('audio').center().y())))
    assert bar._layer_visible['audio'] and not operations  # No fake mute command.


def test_media_switch_discards_stale_worker_results_and_releases_on_close(bar,tmp_path):
    source=tmp_path/'第一条.mp4';source.write_bytes(b'identity only; no decoder launched')
    other=tmp_path/'第二条.mp4';other.write_bytes(b'other identity')
    bar.set_media(VideoInfo(str(source),96,64,25,10,False,250))
    first=bar._media_source.identity
    bar.set_media(VideoInfo(str(other),96,64,25,10,False,250))
    revision=bar._media_revision
    stale=MediaOverview('old',first,0,250,(0,),(np.zeros((8,8,3),np.uint8),),np.zeros(32,np.float32),False)
    bar._accept_media(stale)
    assert bar._media_revision==revision and bar._media_result is None
    bar.close_media()
    assert not bar._media_timer.isActive() and bar._media_source is None


def test_source_video_ruler_clicks_seek_same_frame(bar):
    seek=[];bar.seek.connect(seek.append)
    for row in ('mark','video','audio'):
        QTest.mouseClick(bar,Qt.MouseButton.LeftButton,pos=QPoint(round(bar.x(75)),round(bar.row_rect(row).center().y())))
    assert seek==[75,75,75]


def test_no_material_pending_and_read_error_are_distinct(bar,tmp_path):
    assert bar.media_status['state']=='empty'
    missing=VideoInfo(str(tmp_path/'没有.mp4'),96,64,25,10,False,250)
    bar.set_media(missing)
    assert bar.media_status['state']=='error' and '媒体概览不可用' in bar.media_status['video_error']
    bar.set_media(None)
    assert bar.media_status['state']=='empty' and not bar.media_status['video_error']
    path=tmp_path/'pending.mp4';path.write_bytes(b'read later')
    bar.set_media(VideoInfo(str(path),96,64,25,10,False,250))
    assert bar.media_status['state']=='pending'
    bar.set_busy(True);assert bar.media_status['state']=='busy'
