from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from lockframe.v072_check import activate
from test_workflow_v9 import window, source


def test_inline_io_is_clip_owned_and_overlap_leaves_data_unchanged(window,source):
    import uuid
    first,second=deepcopy(source),deepcopy(source);second.id=uuid.uuid4().hex
    activate(window,[first,second]);window._open_tracking_sections()
    panel=window._sections_panel
    window._seek(0);QTest.keyClick(window.strip,Qt.Key.Key_I)
    window._seek(17);QTest.keyClick(window.strip,Qt.Key.Key_O)
    assert window.tracking_sections==[dict(start=0,end=17)]
    window._seek(4);panel.mark_in();window._seek(20);panel.mark_out()
    assert window.tracking_sections==[dict(start=0,end=17)] and '重叠' in panel.status.text()
    window._activate_clip(second.id)
    assert not window.section_tracking and panel.isHidden()
    window._open_tracking_sections()
    assert not window.tracking_sections and second.id not in panel.pending
    window._activate_clip(first.id)
    assert panel.isVisible() and panel.pending[first.id]==4


def test_busy_section_edits_are_blocked_and_selected_range_is_located(window,source):
    activate(window,[deepcopy(source)]);window._open_tracking_sections()
    panel=window._sections_panel;window._seek(18);panel.split()
    assert panel.isVisible() and not panel.isWindow()
    original=deepcopy(window.tracking_sections)
    window.busy=True;window._sync_ui()
    try:
        panel.mark_in();panel.mark_out();panel.remove();panel.split()
        assert window.tracking_sections==original and not panel.in_button.isEnabled()
    finally:
        window.busy=False;window._sync_ui()
    window._seek(30);panel.selector.setCurrentIndex(0)
    assert window.frame_index<=17 and window.section_tracking


def test_zoom_limit_visible_changes_plan_without_retracking_and_shape_has_one_home(window,source):
    activate(window,[deepcopy(source)])
    result=window.track
    window.zoom_slider.setValue(230)
    assert window.track is result and window.plan is not None and window.settings().max_zoom==2.3
    assert window.sidebar_sections['lock'].isAncestorOf(window.zoom_slider)
    window.zoom_slider.setValue(400)
    assert window.settings().max_zoom==4 and window.track is result
    window._open_settings(2);QApplication.processEvents()
    assert window.shape_combo.isHidden()
    assert window.region_toolbar.isVisible()
    assert window.folder_button.isHidden() and window.hardware_brief.isHidden()


def test_explicit_preview_io_remains_separate_from_segment_io(window,source):
    activate(window,[deepcopy(source)]);window._open_tracking_sections()
    window._seek(5);window._tb_operation('preview_in')
    window._seek(20);window._tb_operation('preview_out')
    assert window.strip.state['play_in']==5 and window.strip.state['play_out']==21
    assert not window.tracking_sections and not window._sections_panel.pending


def test_production_simple_acceptance(tmp_path):
    from lockframe.simple_check import check_simple_workspace
    result=check_simple_workspace(tmp_path/'simple')
    assert result['shared_transport']==1 and result['keyboard_io']
    assert result['zoom_limit_in_composition'] and result['help_and_diagnostics_retained']
