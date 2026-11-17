"""Real Qt pointer editing must propose one canonical command, never a model edit."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt, QEvent
from PySide6.QtGui import QFocusEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from lockframe.regions import BOX_CORNERS
from lockframe.widgets import PreviewCanvas


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def canvas(app):
    view = PreviewCanvas()
    view.resize(1100, 700)
    view.mode = "source"
    view.selection_mode = "adjust"
    view.selection_shape = "polygon"
    view.region_points = BOX_CORNERS
    view.set_frames(np.full((360, 640, 3), 48, np.uint8), box=(.3, .3, .4, .4))
    view.show()
    app.processEvents()
    yield view
    view.close()
    app.processEvents()


def source_position(view, x, y):
    rect = view.source_rect
    return QPoint(round(rect.x() + x * rect.width()), round(rect.y() + y * rect.height()))


def drag(view, start, end):
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(view, end)
    QTest.mouseRelease(view, Qt.MouseButton.LeftButton, pos=end)


def apply_proposal(view, payload):
    view.box, view.region_points = payload["box"], payload["points"]
    view.homography = None
    view.update()


def proposal_source(payload):
    box = np.asarray(payload["box"])
    return box[:2] + np.asarray(payload["points"]) * box[2:]


def test_vertex_drag_has_one_release_proposal_and_no_model_mutation(canvas):
    proposals, rois = [], []
    canvas.region_selected.connect(proposals.append)
    canvas.roi_selected.connect(rois.append)
    original = (tuple(canvas.box), tuple(canvas.region_points))
    start = source_position(canvas, .3, .3)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    for distance in (8, 15, 26):
        QTest.mouseMove(canvas, start + QPoint(distance, 18))
        assert not proposals and not rois
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=start + QPoint(26, 18))
    assert len(proposals) == 1 and not rois
    assert canvas.selected_vertex == 0
    assert (tuple(canvas.box), tuple(canvas.region_points)) == original
    points = proposal_source(proposals[0])
    assert points[0, 0] > .3 and points[0, 1] > .3
    np.testing.assert_allclose(points[1:], [[.7, .3], [.7, .7], [.3, .7]], atol=1e-6)


@pytest.mark.parametrize("cancel", ["escape", "focus", "zoom", "resize", "mode", "pan"])
def test_cancel_restores_geometry_and_no_command(canvas, cancel, app):
    original = (tuple(canvas.box), tuple(canvas.region_points))
    proposals = []
    canvas.region_selected.connect(proposals.append)
    start = source_position(canvas, .3, .3)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas, start + QPoint(40, 20))
    assert canvas._region_drag is not None
    if cancel == "escape":
        QTest.keyClick(canvas, Qt.Key.Key_Escape)
    elif cancel == "focus":
        QApplication.sendEvent(canvas, QFocusEvent(QEvent.Type.FocusOut))
    elif cancel == "zoom":
        canvas.set_zoom(2)
    elif cancel == "resize":
        canvas.resize(1000, 700)
        app.processEvents()
    elif cancel == "mode":
        canvas.mode = "output"
    else:
        QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=start)
        QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=start)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=start + QPoint(40, 20))
    assert not proposals
    assert (tuple(canvas.box), tuple(canvas.region_points)) == original


def test_inside_drag_moves_whole_polygon_at_zoom_and_pan(canvas, app):
    canvas.set_zoom(2)
    middle = canvas.source_viewport.center().toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=middle)
    QTest.mouseMove(canvas, middle + QPoint(25, -10))
    QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=middle + QPoint(25, -10))
    before = canvas._region_source_points().copy()
    proposals = []
    canvas.region_selected.connect(proposals.append)
    start = source_position(canvas, .5, .5)
    drag(canvas, start, start + QPoint(30, 20))
    assert len(proposals) == 1 and canvas.selected_vertex == -1
    delta = np.array([30 / canvas.source_rect.width(), 20 / canvas.source_rect.height()])
    np.testing.assert_allclose(proposal_source(proposals[0]), before + delta, atol=1e-12)


def test_add_delete_proposals_and_triangle_minimum(canvas):
    proposals = []
    canvas.region_selected.connect(proposals.append)
    canvas.region_selected.connect(lambda payload: apply_proposal(canvas, payload))
    canvas.region_edit_mode = "add"
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=source_position(canvas, .5, .26))
    assert len(proposals) == 1 and len(canvas.region_points) == 5
    assert canvas.selected_vertex == 1 and canvas.can_delete_vertex
    assert canvas.delete_region_vertex()
    assert len(proposals) == 2 and len(canvas.region_points) == 4
    canvas.selected_vertex = 0
    assert canvas.delete_region_vertex()
    assert len(proposals) == 3 and len(canvas.region_points) == 3
    canvas.selected_vertex = 0
    assert not canvas.can_delete_vertex and not canvas.delete_region_vertex()
    assert len(proposals) == 3


def test_delete_mode_click_and_no_selection_are_safe(canvas):
    proposals, selected = [], []
    canvas.region_selected.connect(proposals.append)
    canvas.vertex_selection_changed.connect(selected.append)
    canvas.region_edit_mode = "delete"
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=source_position(canvas, .5, .5))
    assert not proposals and not canvas.can_delete_vertex
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=source_position(canvas, .3, .3))
    assert len(proposals) == 1 and selected == [0, -1]
    assert len(canvas.region_points) == 4  # Host has not committed the proposal.


def test_invalid_self_intersection_is_rejected_with_original_preserved(canvas):
    proposals, errors = [], []
    canvas.region_selected.connect(proposals.append)
    canvas.selection_rejected.connect(errors.append)
    # Moving the top-left beyond the opposite side crosses two non-adjacent edges.
    drag(canvas, source_position(canvas, .3, .3), source_position(canvas, .85, .5))
    assert not proposals and len(errors) == 1
    np.testing.assert_allclose(canvas.box, [.3, .3, .4, .4])
    assert canvas.region_points == BOX_CORNERS


def test_vertex_radius_is_screen_constant_and_projection_is_used(canvas):
    for zoom in (1, 3, 6):
        canvas.box = (.47, .47, .06, .06)
        canvas.set_zoom(zoom)
        vertex = source_position(canvas, .47, .47)
        assert canvas._hit_selection(QPointF(vertex + QPoint(-4, 0))) == ("vertex", 0)
        assert canvas._hit_selection(QPointF(vertex + QPoint(-9, 0))) is None
    canvas.reset_view()
    canvas.box = (.3, .3, .4, .4)
    canvas.homography = np.array([[.30, 0, .36], [0, .30, .34], [0, 0, 1.]])
    point = source_position(canvas, .36, .34)
    assert canvas._hit_selection(QPointF(point)) == ("vertex", 0)
    proposals = []
    canvas.region_selected.connect(proposals.append)
    drag(canvas, point, point + QPoint(12, 15))
    assert len(proposals) == 1
    np.testing.assert_allclose(proposal_source(proposals[0])[1:], [[.66, .34], [.66, .64], [.36, .64]], atol=1e-6)


def test_busy_output_and_composition_do_not_edit_region(canvas, app):
    proposals = []
    canvas.region_selected.connect(proposals.append)
    canvas.selectable = False
    drag(canvas, source_position(canvas, .3, .3), source_position(canvas, .4, .4))
    canvas.selected_vertex = 0
    assert not proposals and not canvas.can_delete_vertex and not canvas.delete_region_vertex()
    canvas.selectable = True
    canvas.mode = "output"
    canvas.update()
    app.processEvents()
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    assert not proposals and not canvas.delete_region_vertex()
    canvas.mode = "compare"
    canvas.set_frames(canvas.source, canvas.source, box=canvas.box)
    canvas.composition_editable = True
    canvas.update()
    app.processEvents()
    changed = []
    canvas.composition_changed.connect(lambda x, y: changed.append((x, y)))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.output_rect.center().toPoint())
    assert changed and not proposals


def test_polygon_maximum_and_keyboard_step(canvas):
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    canvas.region_points = tuple((.5 + .5 * np.cos(t), .5 + .5 * np.sin(t)) for t in angles)
    canvas.region_edit_mode = "add"
    proposals, rejected = [], []
    canvas.region_selected.connect(proposals.append)
    canvas.selection_rejected.connect(rejected.append)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=source_position(canvas, .5, .3))
    assert not proposals and len(rejected) == 1
    canvas.region_edit_mode = "select"
    canvas.region_points = BOX_CORNERS
    canvas.selected_vertex = 0
    QTest.keyClick(canvas, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert len(proposals) == 1
    assert proposal_source(proposals[0])[0, 0] == pytest.approx(.3 + .1 / 640)


def test_rectangle_ellipse_legacy_and_new_roi_drawing(canvas):
    proposals, rois = [], []
    canvas.region_selected.connect(proposals.append)
    canvas.roi_selected.connect(rois.append)
    for shape in ("rectangle", "ellipse"):
        canvas.selection_shape = shape
        canvas.box = (.3, .3, .4, .4)
        drag(canvas, source_position(canvas, .5, .5), source_position(canvas, .52, .52))
        assert rois[-1][0] > .3
        assert not proposals
    canvas.selection_shape = "ellipse"
    canvas.box = (.3, .3, .4, .4)
    assert canvas._hit_selection(QPointF(source_position(canvas, .32, .32))) is None
    assert len(canvas._handles(canvas._selection_rect())) == 4
    canvas.selection_shape = "polygon"
    canvas.selection_mode = "box"
    canvas.box = None
    drag(canvas, source_position(canvas, .2, .2), source_position(canvas, .4, .4))
    assert len(rois) == 3 and not proposals


def test_invalid_polygon_payloads_and_paint_do_not_emit(canvas, app):
    assert canvas._region_payload([[.2, .2], [.8, .8], [.2, .8], [.8, .2]]) is None
    assert canvas._region_payload([[.2, .2], [.2, .2], [.8, .8]]) is None
    assert canvas._region_payload([[.2, .2], [.8, .2], [.5, .2]]) is None
    assert canvas._region_payload([[.2, .2], [.8, .2], [float("nan"), .8]]) is None
    assert canvas._region_payload([[.2, .2], [1.2, .2], [.5, .8]]) is None
    proposals = []
    canvas.region_selected.connect(proposals.append)
    before = (tuple(canvas.box), tuple(canvas.region_points))
    canvas.mode = "compare"
    canvas.set_frames(canvas.source, canvas.source, box=canvas.box)
    canvas.grab()
    app.processEvents()
    assert before == (tuple(canvas.box), tuple(canvas.region_points)) and not proposals
