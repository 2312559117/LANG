"""Focused tracking workspace, reusing authoritative Qt controls and models."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout
from .sections_ui import SectionsPanel


def install_simple_workspace(w):
    if getattr(w, '_simple_workspace', False):
        return
    w._simple_workspace = True
    w.hardware_brief.hide()
    w.ad_export.hide()
    # The primary shape toolbar and timeline trim handles own these operations.
    w.shape_combo.hide()
    for label in w.shape_combo.parentWidget().findChildren(QLabel):
        if label.text() == '区域形状（与预览工具栏同步）':
            label.hide()
    w.folder_button.hide()
    for key, titles in (
        ('materials', ('截取视频片段…', '打开合成演示')),
        ('tracking', ('独立分段跟踪…', '区域形状与数值微调…')),
    ):
        menu = w.sidebar_sections[key].more_button.menu()
        for action in list(menu.actions()):
            if action.text() in titles:
                menu.removeAction(action)
    # Put support/developer utilities in Help, away from daily media work.
    w._simple_menu_actions = w.ad_menu.actions()
    file_menu = w._simple_menu_actions[0].menu()
    help_menu = w._simple_menu_actions[-1].menu()
    w._simple_support_menus = (file_menu, help_menu)
    file_menu.removeAction(w.trim_action)
    w.addAction(w.trim_action)  # Keep Ctrl+T without a duplicate file-menu entry.
    developer = help_menu.addMenu('开发与验收')
    for action in list(file_menu.actions()):
        if '验收记录' in action.text():
            file_menu.removeAction(action)
            developer.addAction(action)
    file_menu.removeAction(w.demo_action)
    help_menu.addAction(w.demo_action)
    # Keep only direct composition dragging in the compact main panel.
    composition = w.sidebar_sections['lock'].content.layout()
    for slider in w.composition_sliders:
        for index in range(composition.count()):
            item=composition.itemAt(index)
            row=item.layout()
            if row is not None and row.indexOf(slider)>=0:
                while row.count():
                    child=row.takeAt(0)
                    if child.widget(): child.widget().hide()
                composition.takeAt(index)
                break
    tracking = w.sidebar_sections['tracking'].content.layout()
    tracking.removeWidget(w.reset_tracking_button)
    w.tracking_actions.addWidget(w.reset_tracking_button)
    w.reset_tracking_button.setText('重置')
    # Move the real zoom slider together with its live value label.
    w.zoom_slider.setMaximum(400)  # Match the existing Settings 1–4x contract.
    original = w.zoom_slider.parentWidget().layout()
    index = original.indexOf(w.zoom_slider)
    heading = original.takeAt(index-1).layout()
    original.removeWidget(w.zoom_slider)
    original.removeWidget(w.crop_label)
    composition = w.sidebar_sections['lock'].content.layout()
    composition.addLayout(heading)
    composition.addWidget(w.zoom_slider)
    composition.addWidget(w.crop_label)
    w.crop_label.setWordWrap(False)
    # Keep detailed tracking evidence together, under Tracking settings.
    advanced = w.settings_tabs.widget(0).widget().layout()
    for control in (w.planar_region_check, w.planar_preset,
                    w.adaptive_reference_check, w.mesh_compute_check):
        control.parentWidget().layout().removeWidget(control)
        advanced.insertWidget(max(0, advanced.count()-1), control)
    w.sections_button.setCheckable(True)
    w.sections_button.clicked.disconnect()
    w.sections_button.clicked.connect(lambda checked: w._set_tracking_sections(checked, w.tracking_sections))
    tracking.removeWidget(w.sections_button)
    w.ad_timeline_toolbar.layout().insertWidget(4, w.sections_button)
    panel = w._sections_panel = SectionsPanel(w)
    w.timeline_well.layout().insertWidget(1, panel)
    panel.hide()
    w.strip.sectionPicked.connect(lambda index: panel.selector.setCurrentIndex(index))
    view = next(a.menu() for a in w.ad_menu.actions() if a.text() == '视图')
    waveform = view.addAction('显示音频波形（仅查看）')
    waveform.setCheckable(True)
    waveform.toggled.connect(lambda value: w.strip.set_audio_visible(value))
    w.audio_waveform_action = waveform
    w.strip.overlayVisibilityChanged.connect(
        lambda name, value: waveform.setChecked(value) if name == 'audio' else None)
    # Hidden legacy controls remain the settings API, with a single visible home.
    details = w.settings_tabs.widget(3).widget().layout()
    for title, control in (('预览画质', w.preview_quality), ('时间线显示单位', w.time_units)):
        details.insertWidget(max(0, details.count()-1), QLabel(title))
        details.insertWidget(max(0, details.count()-1), control)
        control.show()
    w._sections_sync()
    resize_simple_workspace(w)


def resize_simple_workspace(w):
    compact = w.height() < 900
    panel = getattr(w, '_sections_panel', None)
    segment_height = 66 if panel is not None and not panel.isHidden() else 0
    strip_height = (102 if segment_height else 136) if compact else 158
    w.timeline_well.setFixedHeight(40 + strip_height + segment_height)
    w.strip.setFixedHeight(strip_height)
    w.timeline_trim_label.setVisible(w.width() >= 1300)
    w.strip.set_compact(compact)
    progress = w.tracking_progress_panel
    extra = max(0, progress.sizeHint().height()-72) if progress.expanded else 0
    progress.setMinimumHeight(48 if compact else 70)
    progress.setMaximumHeight(16777215 if progress.expanded else 48 if compact else 70)
    w.sidebar_sections['tracking'].setFixedHeight((216 if compact else 254)+extra)
    w.sidebar_sections['lock'].setFixedHeight(166 if compact else 188)
    w.sidebar_sections['export'].setFixedHeight(91)
    for key in ('tracking', 'lock'):
        box = w.sidebar_sections[key]
        box.layout().setContentsMargins(12, 7, 12, 7)
        box.layout().setSpacing(5)
        box.content.layout().setSpacing(5)
    for button, height in ((w.select_target_button, 26), (w.tracking_combo, 26),
                           (w.track_button, 32), (w.pause_button, 28),
                           (w.continue_button, 28), (w.reset_tracking_button, 28)):
        button.setFixedHeight(height if compact else height+4)
    if w.plan is not None:
        w.crop_label.setText(f'实际 {w.plan.zoom:.2f}× · 构图受限 {w.plan.limited_frames} 帧')
        w.crop_label.setToolTip(f'需要放大 {w.plan.required_zoom:.2f}×；上限 {w.settings().max_zoom:.2f}×；输出 {w.plan.width}×{w.plan.height}。')
    else:
        w.crop_label.setText('跟踪完成后显示实际放大')
    w.sync_label.setVisible(w.central_preview.width() >= 720)
