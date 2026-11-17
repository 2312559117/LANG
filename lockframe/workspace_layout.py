"""Arrange existing controls into material / dual-monitor / task columns.

Only widget ownership and layout change here. Tracking, selection, menus,
signals, and the source-frame timebar remain the original instances.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLayout, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


_UNLIMITED = 16777215


def _layout_index(layout, nested):
    return next((index for index in range(layout.count())
                 if layout.itemAt(index).layout() == nested), -1)


def install_workspace(window):
    """Install once, at the end of MainWindow._build_ui after centralWidget."""
    if getattr(window, "_three_column_workspace", False):
        return
    splitter = window.workspace_splitter
    left = window.reference_sidebar
    preview = splitter.widget(1)
    timeline = getattr(window, "timeline_well", None)
    if timeline is None:
        timeline = preview.findChild(QFrame, "timelineWell")
    if timeline is None:
        raise RuntimeError("源帧时间线面板尚未创建，无法安装三栏工作区")
    window.timeline_well = timeline
    window._three_column_workspace = True

    # The original primary sections keep their controls, menus, and connections.
    sections = window.sidebar_sections
    old_scroll = window.sidebar_scroll
    left_layout = left.layout()
    left_layout.removeWidget(old_scroll)
    left_layout.removeWidget(sections["export"])
    left.setMinimumWidth(240)
    left.setMaximumWidth(340)
    left.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    window.materials_sidebar = left
    material_section = sections["materials"]
    material_content = material_section.content
    material_layout = material_content.layout()
    # Keep the existing import/select-all row, moving it above the full list.
    material_layout.removeWidget(window.clip_list)
    material_layout.insertWidget(material_layout.count(), window.clip_list, 1)
    window.clip_list.setMinimumHeight(72)
    window.clip_list.setMaximumHeight(_UNLIMITED)
    window.clip_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    material_content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
    material_section.layout().setStretch(1, 1)
    material_section.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
    left_layout.addWidget(material_section, 1)
    material_section.show()

    right = QFrame()
    right.setObjectName("referenceSidebar")
    right.setAccessibleName("跟踪、构图与导出")
    right.setMinimumWidth(260)
    right.setMaximumWidth(360)
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(0)
    right_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.horizontalScrollBar().valueChanged.connect(
        lambda value: scroll.horizontalScrollBar().setValue(0) if value else None)
    content = QWidget()
    content.setMinimumWidth(0)
    content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(0, 0, 0, 0)
    content_layout.setSpacing(0)
    content_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
    for name in ("tracking", "lock"):
        content_layout.addWidget(sections[name])
        sections[name].show()
    content_layout.addStretch(1)
    scroll.setWidget(content)
    right_layout.addWidget(scroll, 1)
    right_layout.addWidget(sections["export"])
    sections["export"].show()
    window.task_sidebar = right
    window.task_sidebar_scroll = scroll
    # Existing section-navigation methods now reveal the right-hand sections.
    window.sidebar_scroll = scroll
    window.sidebar_content = content

    # Pause and continue share a row; reset and cancellation have full width.
    # This keeps all actions reachable even at the 260 px task-column minimum.
    tracking_layout = sections["tracking"].content.layout()
    action_index = _layout_index(tracking_layout, window.tracking_actions)
    for offset, button in enumerate((window.reset_tracking_button, window.cancel_button), 1):
        window.tracking_actions.removeWidget(button)
        tracking_layout.insertWidget(max(0, action_index + offset), button)
    window.tracking_combo.setMinimumWidth(0)
    window.tracking_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

    preview.setMinimumWidth(560)
    preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    window.central_preview = preview
    splitter.addWidget(right)
    splitter.setChildrenCollapsible(False)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setStretchFactor(2, 0)
    splitter.setSizes([290, 920, 310])

    # Lift the same timebar out of the central column to span all three columns.
    preview.layout().removeWidget(timeline)
    root_layout = window.centralWidget().layout()
    root_layout.insertWidget(root_layout.indexOf(splitter) + 1, timeline)
    timeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    timeline.layout().setContentsMargins(8, 6, 8, 6)
    timeline.layout().setSpacing(4)
    timeline.show()
    old_scroll.hide()
    old_scroll.deleteLater()
