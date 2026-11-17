"""Real-window geometry and control reuse for the three-column workspace."""
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

from lockframe.workspace_layout import install_workspace
from test_workflow_v9 import window


def _bounds(widget, relative):
    origin = widget.mapTo(relative, QPoint(0, 0))
    return origin.x(), origin.y(), origin.x() + widget.width(), origin.y() + widget.height()


def test_installs_once_and_reuses_existing_steps_controls_and_timebar(window):
    original = {name: getattr(window, name) for name in (
        "canvas", "strip", "clip_list", "track_button", "composition_edit",
        "export_button", "batch_export_button", "settings_dialog")}
    section_ids = {name: id(section) for name, section in window.sidebar_sections.items()}
    menus = {name: section.more_button.menu() for name, section in window.sidebar_sections.items()}
    install_workspace(window)
    install_workspace(window)
    QApplication.processEvents()
    assert window.workspace_splitter.count() == 3
    assert all(getattr(window, name) is control for name, control in original.items())
    assert all(id(window.sidebar_sections[name]) == identity for name, identity in section_ids.items())
    assert all(window.sidebar_sections[name].more_button.menu() is menu for name, menu in menus.items())
    assert window.materials_sidebar.isAncestorOf(window.clip_list)
    assert not window.materials_sidebar.isAncestorOf(window.track_button)
    assert window.task_sidebar.isAncestorOf(window.track_button)
    assert window.task_sidebar.isAncestorOf(window.export_button)
    assert window.central_preview.isAncestorOf(window.canvas)
    assert not window.central_preview.isAncestorOf(window.strip)
    assert window.timeline_well.parentWidget() is window.centralWidget()
    assert window.timeline_well.isAncestorOf(window.strip)


def test_full_height_materials_import_above_list_and_refresh_cannot_collapse_it(window):
    install_workspace(window)
    QApplication.processEvents()
    assert _bounds(window.import_button, window)[1] < _bounds(window.clip_list, window)[1]
    assert _bounds(window.select_all_clips, window)[1] < _bounds(window.clip_list, window)[1]
    before = window.clip_list.height()
    assert before > 208
    window._refresh_sequence()
    QApplication.processEvents()
    assert window.clip_list.height() == before
    assert window.clip_list.maximumHeight() > 10000


def test_three_columns_and_full_width_timeline_fit_both_requested_window_sizes(window):
    install_workspace(window)
    for width, height in ((1560, 940), (1160, 760)):
        window.resize(width, height)
        QApplication.processEvents()
        assert window.width() == width and window.height() == height
        left, center, right = [window.workspace_splitter.widget(index) for index in range(3)]
        assert 240 <= left.width() <= 340
        assert center.width() >= 560
        assert 260 <= right.width() <= 360
        assert window.canvas.height() >= 160 and window.canvas.width() >= 540
        assert window.canvas.visibleRegion().contains(window.canvas.rect())
        timeline = _bounds(window.timeline_well, window)
        area = _bounds(window.workspace_splitter, window)
        assert timeline[0] == area[0] and timeline[2] == area[2]
        assert timeline[1] >= area[3]
        exported = _bounds(window.export_button, right)
        assert window.export_button.isVisible() and 0 <= exported[1] < exported[3] <= right.height()
        assert window.sidebar_scroll.horizontalScrollBar().maximum() == 0
        assert window.clip_list.visibleRegion().contains(window.clip_list.rect())


def test_task_scroll_keeps_export_fixed_and_tracking_actions_inside_narrow_column(window):
    install_workspace(window)
    window.resize(1160, 760)
    window.workspace_splitter.setSizes([240, 640, 260])
    window.pause_button.show()
    window.continue_button.show()
    window.cancel_button.show()
    window.tracking_progress_panel.begin("测试素材.mp4")
    QApplication.processEvents()
    right = window.task_sidebar
    exported_before = _bounds(window.export_button, right)
    window.sidebar_scroll.verticalScrollBar().setValue(window.sidebar_scroll.verticalScrollBar().maximum())
    QApplication.processEvents()
    assert _bounds(window.export_button, right) == exported_before
    tracking = window.sidebar_sections["tracking"].content
    for control in (window.pause_button, window.continue_button, window.reset_tracking_button,
                    window.cancel_button, window.tracking_combo):
        bounds = _bounds(control, tracking)
        assert 0 <= bounds[0] < bounds[2] <= tracking.width()
    assert window.tracking_actions.count() == 3
    assert window.sidebar_content.isAncestorOf(window.sidebar_sections["lock"])
    assert not window.sidebar_content.isAncestorOf(window.sidebar_sections["export"])
