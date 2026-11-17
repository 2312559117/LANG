import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from lockframe.region_toolbar import RegionToolbar


@pytest.fixture
def toolbar():
    app = QApplication.instance() or QApplication([])
    widget = RegionToolbar()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    app.processEvents()


def test_empty_busy_and_paused_state_gate_all_editing(toolbar):
    shapes, actions = [], []
    toolbar.shape_requested.connect(shapes.append)
    toolbar.action_requested.connect(actions.append)
    controls = [*toolbar.shape_buttons.values(), toolbar.edit_button, toolbar.add_button,
                toolbar.delete_button, toolbar.reset_button, toolbar.shape_combo, toolbar.overflow_button]
    assert all(not control.isEnabled() for control in controls)
    toolbar.set_state("polygon", ready=True, busy=True, can_delete=True)
    assert all(not control.isEnabled() for control in controls)
    QTest.mouseClick(toolbar.shape_buttons["rectangle"], Qt.MouseButton.LeftButton)
    toolbar.overflow_actions["reset"].trigger()
    assert shapes == actions == []
    # Paused correction is represented by a ready region and no active worker.
    toolbar.set_state("polygon", ready=True, busy=False, can_delete=True)
    assert all(control.isEnabled() for control in controls)
    QTest.mouseClick(toolbar.delete_button, Qt.MouseButton.LeftButton)
    assert actions == ["delete"] and not shapes


def test_shape_buttons_and_dropdown_emit_once_and_remain_exclusive(toolbar):
    shapes, actions = [], []
    toolbar.shape_requested.connect(shapes.append)
    toolbar.action_requested.connect(actions.append)
    toolbar.set_state("rectangle", ready=True, busy=False)
    QTest.mouseClick(toolbar.shape_buttons["ellipse"], Qt.MouseButton.LeftButton)
    assert shapes == ["ellipse"] and toolbar.shape_combo.currentData() == "ellipse"
    assert [name for name, button in toolbar.shape_buttons.items() if button.isChecked()] == ["ellipse"]
    toolbar.shape_combo.setCurrentIndex(toolbar.shape_combo.findData("polygon"))
    assert shapes == ["ellipse", "polygon"]
    assert [name for name, button in toolbar.shape_buttons.items() if button.isChecked()] == ["polygon"]
    assert toolbar.add_button.isEnabled() and not toolbar.delete_button.isEnabled()
    QTest.mouseClick(toolbar.edit_button, Qt.MouseButton.LeftButton)
    assert actions == ["edit"] and shapes == ["ellipse", "polygon"]
    assert toolbar.edit_button.isChecked() and toolbar.shape_combo.currentData() == "polygon"


def test_programmatic_sync_circle_compat_and_vertex_limits_never_emit(toolbar):
    received = []
    toolbar.shape_requested.connect(lambda value: received.append(("shape", value)))
    toolbar.action_requested.connect(lambda value: received.append(("action", value)))
    toolbar.set_state("circle", ready=True, busy=False)
    assert toolbar.shape_combo.currentData() == "ellipse" and toolbar.shape_buttons["ellipse"].isChecked()
    assert not toolbar.add_button.isEnabled() and not toolbar.delete_button.isEnabled()
    toolbar.set_state("polygon", ready=True, busy=False, can_delete=False, can_add=False)
    assert not toolbar.add_button.isEnabled() and not toolbar.delete_button.isEnabled()
    assert not toolbar.overflow_actions["add"].isEnabled()
    toolbar.set_state("polygon", ready=True, busy=False, can_delete=True, mode="add")
    assert toolbar.add_button.isEnabled() and toolbar.delete_button.isEnabled()
    assert toolbar.add_button.isChecked() and not toolbar.edit_button.isChecked()
    toolbar.set_state("rect", ready=True, busy=False)
    assert toolbar.shape_combo.currentData() == "rectangle"
    assert not received


def test_compact_overflow_preserves_shapes_dropdown_and_state(toolbar):
    toolbar.set_state("polygon", ready=True, busy=False, can_delete=False)
    actions = []
    toolbar.action_requested.connect(actions.append)
    toolbar.set_available_width(560)
    QApplication.instance().processEvents()
    assert toolbar.label.isHidden() and not toolbar.overflow_button.isHidden()
    assert all(not button.isHidden() for button in toolbar.shape_buttons.values())
    assert not toolbar.edit_button.isHidden() and not toolbar.shape_combo.isHidden()
    assert all(button.isHidden() for button in (toolbar.add_button, toolbar.delete_button, toolbar.reset_button))
    assert toolbar.sizeHint().width() <= 230
    toolbar.overflow_actions["delete"].trigger()
    toolbar.overflow_actions["add"].trigger()
    toolbar.overflow_actions["reset"].trigger()
    assert actions == ["add", "reset"]
    # Reset is an intent; only the host may change the actual shape/history.
    assert toolbar.shape_combo.currentData() == "polygon"
    toolbar.set_available_width(760)
    QApplication.instance().processEvents()
    assert not toolbar.label.isHidden() and toolbar.overflow_button.isHidden()
    assert all(not button.isHidden() for button in (toolbar.add_button, toolbar.delete_button, toolbar.reset_button))
    assert toolbar.sizeHint().width() <= 350


def test_icons_accessible_labels_and_full_compact_geometry(toolbar):
    toolbar.set_state("rectangle", ready=True, busy=False)
    buttons = [*toolbar.shape_buttons.values(), toolbar.edit_button, toolbar.add_button,
               toolbar.delete_button, toolbar.reset_button]
    assert len(buttons) == 7
    assert all(not button.icon().isNull() and button.toolTip() and button.accessibleName() for button in buttons)
    for width in (960, 760, 759, 560):
        toolbar.set_available_width(width)
        toolbar.resize(toolbar.sizeHint())
        QApplication.instance().processEvents()
        visible = [button for button in buttons if not button.isHidden()] + [toolbar.shape_combo]
        if not toolbar.overflow_button.isHidden():
            visible.append(toolbar.overflow_button)
        assert toolbar.height() == 24
        assert all(control.y() == 0 and control.height() == 24 for control in visible)
        assert all(toolbar.rect().contains(control.geometry()) for control in visible)
