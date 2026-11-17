"""Exercise real Qt mouse events rather than emitting model-change signals."""
from copy import deepcopy
import numpy as np
import pytest
from PySide6.QtCore import Qt,QPoint
from PySide6.QtTest import QTest,QSignalSpy
from PySide6.QtWidgets import QApplication
from lockframe.time_remap_ui import CurveCanvas,TimeRemapPanel
from lockframe.time_remap_model import default_config,make_constant_speed
from lockframe.time_remap_editor import TimeRemapEditor
from lockframe.time_remap_map import output_frame_map
from test_retime_integration import window,demo_info


@pytest.fixture(scope='module')
def application():return QApplication.instance() or QApplication([])


def drag(canvas,index,delta):
    start=canvas.pixel(*canvas.nodes()[index]).toPoint()
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=start)
    QTest.mouseMove(canvas,start+delta,20)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=start+delta)
    QApplication.processEvents()


def test_sidebar_curve_mouse_drag_emits_one_commit_and_doubleclick_opens_editor(application):
    panel=TimeRemapPanel();panel.resize(310,336)
    config=default_config();config['enabled']=True
    panel.set_config(config);panel.show();application.processEvents()
    changes=QSignalSpy(panel.configEdited);opened=QSignalSpy(panel.openEditor)
    drag(panel.plot,2,QPoint(0,-15))
    assert changes.count()==1 and changes.at(0)[0]['basic']['speeds'][2]>1.
    QTest.mouseDClick(panel.plot,Qt.MouseButton.LeftButton,pos=panel.plot.plot().center().toPoint())
    assert opened.count()==1
    panel.close()


def test_full_editor_basic_drag_preserves_selected_node_and_undo(application):
    editor=TimeRemapEditor(None,240,24.);editor.show();application.processEvents()
    before=deepcopy(editor.draft)
    drag(editor.canvas,3,QPoint(-15,-24))
    assert editor.canvas.selected==3
    assert editor.draft['basic']['speeds'][3]>1.
    assert len(editor._history)==2
    editor.undo();assert editor.draft==before
    editor.redo();assert editor.draft['basic']['speeds'][3]>1.
    editor.reject()


def test_advanced_speed_graph_can_drag_velocity_and_changes_mapping(application):
    config=make_constant_speed(speed=1.)
    editor=TimeRemapEditor(config,120,24.);editor.show()
    editor.view.setCurrentIndex(editor.view.findData(True));application.processEvents()
    before=output_frame_map(editor.draft,120).copy()
    drag(editor.canvas,0,QPoint(0,-25))
    assert editor.draft['points'][0]['outgoing_speed']>1.
    assert editor.draft['points'][0]['interpolation']=='bezier'
    assert editor.draft['points'][1]['incoming_interpolation']=='bezier'
    assert not np.allclose(before,output_frame_map(editor.draft,120))
    editor.undo();np.testing.assert_array_equal(before,output_frame_map(editor.draft,120))
    editor.reject()


def test_dragging_curve_body_adjusts_nearest_node(application):
    canvas=CurveCanvas(editable=True);canvas.resize(640,260);canvas.show();application.processEvents()
    changes=QSignalSpy(canvas.committed)
    start=canvas.pixel(.39,1.).toPoint()
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=start)
    assert canvas.drag is not None
    QTest.mouseMove(canvas,start+QPoint(0,-35),20)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=start+QPoint(0,-35))
    assert changes.count()==1 and max(canvas.config['basic']['speeds'])>1.
    canvas.close()


def test_endpoint_drag_keeps_coordinate_scale_stable_until_release(application):
    canvas=CurveCanvas(editable=True);canvas.resize(640,260)
    canvas.set_config(make_constant_speed(speed=1.));canvas.show();application.processEvents()
    start=canvas.pixel(1.,1.).toPoint();target=start-QPoint(80,0)
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=start)
    QTest.mouseMove(canvas,target,20)
    first=canvas.config['points'][-1]['position']
    QTest.mouseMove(canvas,target,20)
    assert canvas.config['points'][-1]['position']==first
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=target)
    canvas.close()


def test_click_without_drag_does_not_mark_settings_changed(application):
    canvas=CurveCanvas(editable=True);canvas.resize(640,260);canvas.show();application.processEvents()
    changes=QSignalSpy(canvas.committed)
    pos=canvas.pixel(*canvas.nodes()[2]).toPoint()
    QTest.mouseClick(canvas,Qt.MouseButton.LeftButton,pos=pos)
    assert changes.count()==0
    canvas.close()


def test_switching_config_cancels_old_drag_without_commit(application):
    canvas=CurveCanvas(editable=True);canvas.resize(640,260);canvas.show();application.processEvents()
    changes=QSignalSpy(canvas.committed);start=canvas.pixel(*canvas.nodes()[2]).toPoint()
    QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=start)
    replacement=make_constant_speed(speed=-1.);canvas.set_config(replacement)
    QTest.mouseMove(canvas,start-QPoint(0,20),20)
    QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=start-QPoint(0,20))
    assert changes.count()==0 and canvas.config['points'][0]['source']==1.
    canvas.close()


def test_advanced_bezier_handle_drag_changes_draft_and_undo(application):
    config=default_config();config.update(enabled=True,mode='advanced')
    editor=TimeRemapEditor(config,240,24.);editor.show();application.processEvents()
    initial=deepcopy(editor.draft)
    side,x,y=editor.canvas.handles()[0];start=editor.canvas.pixel(x,y).toPoint()
    QTest.mousePress(editor.canvas,Qt.MouseButton.LeftButton,pos=start)
    assert editor.canvas.drag[0]=='handle'
    QTest.mouseMove(editor.canvas,start+QPoint(12,-24),20)
    QTest.mouseRelease(editor.canvas,Qt.MouseButton.LeftButton,pos=start+QPoint(12,-24))
    assert editor.draft['points'][0][side+'_handle_source']!=initial['points'][0][side+'_handle_source']
    editor.undo();assert editor.draft==initial
    editor.reject()


def test_sidebar_curve_and_controls_do_not_overlap_after_resize(window,demo_info):
    from lockframe.sequence import Clip
    from lockframe.models import Settings
    from lockframe.v072_check import activate
    activate(window,[Clip(deepcopy(demo_info),Settings(),{},0,35)])
    config=default_config();config['enabled']=True
    window._apply_retime_config(config)
    for width,height in ((1560,960),(1160,760),(1560,960)):
        window.resize(width,height);QApplication.processEvents()
        panel=window.retime_panel
        assert panel.plot.geometry().bottom()<panel.summary.geometry().top()
        assert panel.summary.geometry().bottom()<panel.flow.geometry().top()
        last=panel.body.layout().itemAt(panel.body.layout().count()-1).widget()
        assert panel.body.rect().contains(last.geometry())
