import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from lockframe.widgets import PreviewCanvas

@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])

@pytest.fixture
def canvas(app):
    view = PreviewCanvas()
    view.resize(1000,680)
    view.mode = 'source'
    view.selection_mode = 'adjust'
    view.set_frames(np.full((360,640,3),40,np.uint8), box=(.4,.4,.1,.16))
    view.show()
    app.processEvents()
    yield view
    view.close()

def drag(canvas, start, end):
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas, end)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)

def test_zoomed_region_move_resize_and_keyboard_commit(canvas, app):
    canvas.set_zoom(3)
    values = []
    canvas.roi_selected.connect(values.append)
    original = np.asarray(canvas.box)
    rect = canvas._selection_rect()
    start = rect.center().toPoint()
    drag(canvas,start,start+QPoint(20,15))
    assert len(values)==1
    np.testing.assert_allclose(values[-1][:2], original[:2]+[20/canvas.source_rect.width(),15/canvas.source_rect.height()])
    corner=canvas._selection_rect().bottomRight().toPoint()
    before=np.asarray(canvas.box).copy()
    drag(canvas,corner,corner+QPoint(12,20))
    assert values[-1][2]>before[2] and values[-1][3]>before[3]
    before=np.asarray(canvas.box).copy()
    QTest.keyClick(canvas,Qt.Key.Key_Right,Qt.KeyboardModifier.AltModifier)
    assert abs(values[-1][0]-before[0]-.1/640)<1e-9
    QTest.keyClick(canvas,Qt.Key.Key_Down,Qt.KeyboardModifier.ShiftModifier)
    assert abs(values[-1][1]-before[1]-10/360)<1e-9

def test_escape_discards_uncommitted_drag(canvas):
    original=np.asarray(canvas.box).copy()
    selected=[]
    canvas.roi_selected.connect(selected.append)
    center=canvas._selection_rect().center().toPoint()
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=center)
    QTest.mouseMove(canvas,center+QPoint(20,20))
    QTest.keyClick(canvas,Qt.Key.Key_Escape)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=center+QPoint(20,20))
    np.testing.assert_array_equal(canvas.box,original)
    assert not selected

def test_circle_draw_and_resize_preserve_actual_pixel_diameter(canvas):
    canvas.selection_shape='circle'
    canvas.selection_mode='box'
    canvas.box=None
    values=[]
    canvas.roi_selected.connect(values.append)
    start=canvas.source_rect.center().toPoint()
    drag(canvas,start,start+QPoint(70,100))
    assert abs(values[-1][2]*640-values[-1][3]*360)<1e-6
    canvas.box=values[-1]
    corner=canvas._selection_rect().bottomRight().toPoint()
    drag(canvas,corner,corner+QPoint(30,10))
    assert abs(values[-1][2]*640-values[-1][3]*360)<1e-6

def test_window_numeric_adjustment_and_shape_resource_roundtrip(app):
    from lockframe.app import MainWindow
    from lockframe.models import VideoInfo, Settings
    from lockframe.media import import_video
    from pathlib import Path
    import threading
    window=MainWindow()
    info=import_video(str(Path(__file__).resolve().parents[1]/'assets/demo.mp4'),threading.Event())
    window._import_done(info)
    window._select_roi((.4,.3,.1,.2))
    window.shape_combo.setCurrentIndex(window.shape_combo.findData('circle'))
    assert window.track is None
    window.region_fields[0].setValue(300.1)
    window.region_fields[2].setValue(90.)
    window._apply_region()
    box=window.anchors[0]
    assert abs(box[0]*info.width-300.1)<1e-6
    assert abs(box[2]*info.width-90)<1e-6 and abs(box[3]*info.height-90)<1e-6
    settings=Settings(target_shape='ellipse', acceleration='cpu', memory_percent=65,compute_threads=3)
    window._ready=False
    window._restore_settings(settings)
    window._ready=True
    restored=window.settings()
    assert restored.target_shape=='ellipse' and restored.acceleration=='cpu'
    assert restored.memory_percent==65 and restored.compute_threads==3
    window.dirty=False
    window.close()
