"""Live-control contracts: two transports, one authoritative source clock."""
from copy import deepcopy

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QToolButton

from test_workflow_v9 import window, source
from lockframe.ad_workspace import install_ad_workspace, ad_sync, _filter_clips
from lockframe.v072_check import activate


def test_ad_install_reuses_controls_and_matches_approved_panel_geometry(window):
    ids = [id(x) for x in (window.canvas, window.strip, window.track_button, window.export_button)]
    install_ad_workspace(window)
    window.resize(1586, 992)
    QApplication.processEvents()
    assert ids == [id(x) for x in (window.canvas, window.strip, window.track_button, window.export_button)]
    for widget, expected in ((window.materials_sidebar, (10,58,308,702)),
                             (window.central_preview, (328,58,925,702)),
                             (window.task_sidebar, (1263,58,313,702)),
                             (window.timeline_well, (10,768,1566,198))):
        p = widget.mapTo(window, QPoint())
        assert (p.x(),p.y(),widget.width(),widget.height()) == expected
    assert window.track_legend.isHidden()
    window._layout_preview_controls()
    assert window.track_legend.isHidden()


def test_both_monitor_sliders_and_step_actions_follow_one_clock_without_retracking(window, source):
    activate(window, [deepcopy(source)])
    track = window.track
    assert len(window.monitor_transports) == 1
    shared = window.monitor_transports[0]
    shared.slider.setValue(14)
    assert window.frame_index == shared.slider.value() == 14
    buttons = shared.findChildren(QToolButton)
    window.next_button.click()
    assert window.frame_index == shared.slider.value() == 15
    shared.slider.setValue(12)
    assert window.frame_index == 12 and window.track is track
    window._toggle_play()
    assert window.timer.isActive()
    window._step_preview(1)
    assert not window.timer.isActive() and window.track is track


def test_busy_disables_both_monitor_controllers_and_keeps_source_timebar_identity(window, source):
    activate(window, [deepcopy(source)])
    identity = window.strip.media_status['identity']
    window.busy = True
    try:
        ad_sync(window)
        assert all(not transport.isEnabled() for transport in window.monitor_transports)
        assert window.strip.media_status['identity'] == identity
    finally:
        window.busy = False
        window._sync_ui()
    assert all(transport.isEnabled() for transport in window.monitor_transports)


def test_output_pixel_zoom_is_view_only_and_independent_of_original_roi(window, source):
    activate(window, [deepcopy(source)])
    canvas = window.canvas
    QApplication.processEvents()
    canvas.grab()
    original = canvas.output_rect.width()
    anchors = dict(window.anchors)
    track = window.track
    source_rect = canvas.source_rect
    canvas.set_output_zoom(2.)
    canvas.grab()
    assert abs(canvas.output_rect.width() - original * 2) < .1
    assert canvas.source_rect == source_rect
    assert window.anchors == anchors and window.track is track
    right = window.monitor_transports[0]
    next(b for b in right.findChildren(QToolButton) if b.toolTip() == '效果：适配').click()
    assert canvas.output_zoom == 1.


def test_filter_is_view_only_preserves_clips_and_export_reachable_at_minimum_size(window, source):
    first = deepcopy(source)
    from lockframe.sequence import Clip
    second = Clip(source.info)
    activate(window, [first, second])
    _filter_clips(window, 'done')
    assert len(window.clips) == 2 and window.clip_list.item(1).isHidden()
    first.settings.target_shape='ellipse'
    _filter_clips(window, 'done')
    assert window.clip_list.item(0).isHidden()  # stale geometry cannot be called complete
    _filter_clips(window, 'all')
    assert not window.clip_list.item(1).isHidden()
    window.resize(1160,760)
    QApplication.processEvents()
    button = window.export_button
    assert button.visibleRegion().boundingRect().height() == button.height()
    assert window.clip_list.horizontalScrollBar().maximum() == 0
    assert window.canvas.width() >= 500 and window.canvas.height() >= 100



def test_ad_progress_reset_does_not_expand_over_composition(window):
    panel=window.tracking_progress_panel
    panel.set_expanded(True)
    assert panel.expanded
    panel.reset()
    window._sync_ui()
    QApplication.processEvents()
    assert not panel.expanded
    assert panel.maximumHeight()==48
    assert window.sidebar_sections['tracking'].height()==216
