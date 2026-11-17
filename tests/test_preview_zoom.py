"""Source-preview transformations must never alter normalized tracking targets."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from lockframe.widgets import PreviewCanvas


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def canvas(app):
    widget = PreviewCanvas()
    widget.resize(1000, 680)
    widget.mode = "source"
    # Distinctive source/output colors expose drawing beyond the source viewport.
    source = np.full((360, 640, 3), (15, 35, 220), dtype=np.uint8)
    output = np.full_like(source, (20, 180, 15))
    widget.set_frames(source, output)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def image_point(canvas, x, y):
    rect = canvas.source_rect
    return QPoint(round(rect.left() + x * rect.width()), round(rect.top() + y * rect.height()))


def normalized_at(canvas, position):
    rect = canvas.source_rect
    return ((position.x() - rect.left()) / rect.width(),
            (position.y() - rect.top()) / rect.height())


def wheel(canvas, position, delta):
    event = QWheelEvent(QPointF(position), QPointF(canvas.mapToGlobal(position)), QPoint(), QPoint(0, delta),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(canvas, event)


def test_zoomed_box_selection_maps_back_to_original_image(canvas, app):
    selected = []
    canvas.roi_selected.connect(selected.append)
    canvas.set_zoom(4)
    app.processEvents()
    start, end = image_point(canvas, .46, .45), image_point(canvas, .54, .55)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas, end)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end)
    assert len(selected) == 1
    np.testing.assert_allclose(selected[0], [.46, .45, .08, .10], atol=.001)


def test_point_selection_keeps_pixel_size_across_zoom_and_pan(canvas, app):
    selected = []
    canvas.roi_selected.connect(selected.append)
    canvas.selection_mode = "point"
    canvas.point_size = 32
    canvas.set_zoom(5)
    center = canvas.source_viewport.center().toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=center)
    QTest.mouseMove(canvas, center + QPoint(85, 55))
    QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=center + QPoint(85, 55))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)
    assert len(selected) == 1
    roi = selected[0]
    np.testing.assert_allclose(np.array(roi[2:]) * [640, 360], [32, 32], atol=.01)
    np.testing.assert_allclose(np.array(roi[:2]) + np.array(roi[2:]) / 2,
                               normalized_at(canvas, center), atol=1e-6)


def test_mouse_wheel_preserves_image_under_pointer(canvas, app):
    canvas.set_zoom(2)
    anchor = canvas.source_viewport.center().toPoint() + QPoint(75, 30)
    before = normalized_at(canvas, anchor)
    changed = []
    canvas.zoom_changed.connect(changed.append)
    wheel(canvas, anchor, 120)
    assert canvas.zoom_factor == pytest.approx(2.4)
    np.testing.assert_allclose(normalized_at(canvas, anchor), before, atol=1e-10)
    assert changed == [canvas.zoom_factor]


@pytest.mark.parametrize("with_space", [False, True])
def test_pan_moves_source_without_selecting_target(canvas, app, with_space):
    selected = []
    canvas.roi_selected.connect(selected.append)
    canvas.set_zoom(3)
    start = canvas.source_viewport.center().toPoint()
    before = QPointF(canvas.source_rect.topLeft())
    button = Qt.MouseButton.LeftButton if with_space else Qt.MouseButton.MiddleButton
    if with_space:
        QTest.keyPress(canvas, Qt.Key.Key_Space)
    QTest.mousePress(canvas, button, pos=start)
    QTest.mouseMove(canvas, start + QPoint(80, -40))
    QTest.mouseRelease(canvas, button, pos=start + QPoint(80, -40))
    if with_space:
        QTest.keyRelease(canvas, Qt.Key.Key_Space)
    np.testing.assert_allclose([canvas.source_rect.left() - before.x(), canvas.source_rect.top() - before.y()],
                               [80, -40], atol=1e-6)
    assert not selected


def test_output_wheel_and_source_paint_are_isolated(canvas, app):
    canvas.mode = "compare"
    canvas.update()
    app.processEvents()
    output_center = QPoint(750, 340)
    before = canvas.grab().toImage().pixelColor(output_center)
    canvas.set_zoom(8)
    app.processEvents()
    after = canvas.grab().toImage().pixelColor(output_center)
    assert after == before
    wheel(canvas, output_center, -120)
    assert canvas.zoom_factor == 8
    # The enlarged source also stays below its label and above the bottom margin.
    from lockframe.theme import CANVAS
    assert canvas.grab().toImage().pixelColor(250, 675).name() == CANVAS
    canvas.mode = "output"
    canvas.update()
    app.processEvents()
    wheel(canvas, QPoint(250, 340), -120)
    assert canvas.zoom_factor == 8


def test_zoom_limits_fit_and_frame_updates(canvas, app):
    changed = []
    canvas.zoom_changed.connect(changed.append)
    canvas.set_zoom(100)
    assert canvas.zoom_factor == 8
    canvas.set_frames(canvas.source.copy(), canvas.processed)
    assert canvas.zoom_factor == 8
    canvas.set_zoom(float("nan"))
    assert canvas.zoom_factor == 8
    canvas.reset_view()
    assert canvas.zoom_factor == 1 and changed == [8, 1]
    np.testing.assert_allclose(normalized_at(canvas, canvas.source_viewport.center()), [.5, .5])
    canvas.set_zoom(-100)
    assert canvas.zoom_factor == 1
    canvas.set_zoom(3)
    canvas.set_frames(np.zeros((240, 320, 3), dtype=np.uint8))
    assert canvas.zoom_factor == 1


def test_resize_view_change_and_extreme_pan_remain_inside_image(canvas, app):
    canvas.set_zoom(8)
    center = canvas.source_viewport.center().toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=center)
    QTest.mouseMove(canvas, center + QPoint(20000, -20000))
    QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=center + QPoint(20000, -20000))
    for mode, size in [("compare", (340, 260)), ("source", (1200, 500)), ("compare", (1000, 680))]:
        canvas.mode = mode
        canvas.resize(*size)
        canvas.update()
        app.processEvents()
        assert canvas.source_rect.intersected(canvas.source_viewport) == canvas.source_viewport
        assert canvas.zoom_factor == 8


def test_point_edge_check_uses_image_edge_and_allows_viewport_edge(canvas, app):
    selected, rejected = [], []
    canvas.roi_selected.connect(selected.append)
    canvas.selection_rejected.connect(rejected.append)
    canvas.selection_mode = "point"
    canvas.point_size = 32
    canvas.set_zoom(4)
    # A target may cross the current viewport edge while staying in the source image.
    point = canvas.source_viewport.center().toPoint()
    point.setX(round(canvas.source_viewport.left() + 3))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point)
    assert len(selected) == 1 and not rejected
    canvas.reset_view()
    point = image_point(canvas, .005, .5)
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=point)
    assert len(selected) == 1 and len(rejected) == 1
    assert "边缘" in rejected[0]


def test_space_release_requests_playback_only_without_a_pan(canvas, app):
    requests = []
    canvas.playback_requested.connect(lambda: requests.append(True))
    QTest.keyPress(canvas, Qt.Key.Key_Space)
    assert not requests
    QTest.keyRelease(canvas, Qt.Key.Key_Space)
    assert requests == [True]
    canvas.set_zoom(3)
    center = canvas.source_viewport.center().toPoint()
    QTest.keyPress(canvas, Qt.Key.Key_Space)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(canvas, center + QPoint(60, 20))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(60, 20))
    assert requests == [True]
    QTest.keyRelease(canvas, Qt.Key.Key_Space)
    assert requests == [True]
    # Each independent press gets its own playback/pan decision.
    QTest.keyClick(canvas, Qt.Key.Key_Space)
    assert requests == [True, True]


def test_full_resolution_swap_preserves_zoom_and_normalized_view(canvas, app):
    canvas.set_zoom(4)
    center = canvas.source_viewport.center().toPoint()
    QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=center)
    QTest.mouseMove(canvas, center + QPoint(90, -45))
    QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=center + QPoint(90, -45))
    before = normalized_at(canvas, canvas.source_viewport.center())
    changes = []
    canvas.zoom_changed.connect(changes.append)
    canvas.set_frames(np.zeros((1080, 1920, 3), dtype=np.uint8), preserve_view=True)
    assert canvas.zoom_factor == 4
    np.testing.assert_allclose(normalized_at(canvas, canvas.source_viewport.center()), before, atol=1e-10)
    assert changes == []
    # Explicitly dropping the source still clears the view even with preservation enabled.
    canvas.set_frames(None, preserve_view=True)
    assert canvas.zoom_factor == 1 and not canvas.source_rect.isValid()
