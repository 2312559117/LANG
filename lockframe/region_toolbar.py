"""Compact region tools for the source/effect monitor header.

This widget emits editing intents only. The host owns the region, undo history,
and tracking-result validity; synchronising its state never emits an intent.
Icons are small original vector drawings, rendered locally without assets.
"""
from math import cos, pi, sin

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal, QSignalBlocker
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QHBoxLayout, QLabel, QMenu, QSizePolicy,
    QToolButton, QWidget,
)


_SHAPES = (("rectangle", "矩形"), ("ellipse", "椭圆"), ("polygon", "多边形"))
_ACTION_LABELS = {"edit": "选择 / 编辑", "add": "添加多边形顶点", "delete": "删除选中的顶点", "reset": "重置区域为矩形，保留外接框"}


def _tool_icon(kind):
    """Draw a consistent 16 px icon in normal, checked, and disabled colours."""
    icon = QIcon()
    for mode, state, colour in (
        (QIcon.Mode.Normal, QIcon.State.Off, "#b7afa5"),
        (QIcon.Mode.Normal, QIcon.State.On, "#e88c51"),
        (QIcon.Mode.Disabled, QIcon.State.Off, "#68635c"),
        (QIcon.Mode.Disabled, QIcon.State.On, "#68635c"),
    ):
        # A 2x pixmap keeps the custom outlines crisp on high-DPI displays.
        pixmap = QPixmap(32, 32)
        pixmap.setDevicePixelRatio(2.0)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(colour), 1.15)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if kind == "rectangle":
            painter.drawRect(QRectF(2.6, 3.5, 10.8, 9))
        elif kind == "ellipse":
            painter.drawEllipse(QRectF(2.2, 3.6, 11.6, 8.8))
        elif kind == "polygon":
            painter.drawPolygon(QPolygonF([
                QPointF(8 + 6 * cos(-pi / 2 + i * 2 * pi / 5),
                        8 + 6 * sin(-pi / 2 + i * 2 * pi / 5))
                for i in range(5)
            ]))
        elif kind == "edit":
            painter.setBrush(QColor(colour))
            painter.drawPolygon(QPolygonF([QPointF(3.5, 2.3), QPointF(12.2, 9),
                QPointF(8.4, 9.4), QPointF(10, 13.1), QPointF(8, 14),
                QPointF(6.4, 10.1), QPointF(3.5, 12.2)]))
        elif kind in ("add", "delete"):
            painter.drawLine(QPointF(3, 7), QPointF(12, 7))
            if kind == "add":
                painter.drawLine(QPointF(7.5, 2.5), QPointF(7.5, 11.5))
            painter.setBrush(QColor(colour))
            painter.drawEllipse(QPointF(12.5, 12.5), 1.3, 1.3)
        elif kind == 'more':
            painter.setBrush(QColor(colour))
            for x in (3.5, 8., 12.5):
                painter.drawEllipse(QPointF(x, 8), .85, .85)
        elif kind == "reset":
            path = QPainterPath()
            path.moveTo(12.3, 5.2)
            path.cubicTo(8.8, .6, 2, 3, 2.5, 8.5)
            path.cubicTo(3, 13.2, 9, 14.6, 12.1, 11)
            painter.drawPath(path)
            painter.drawPolyline(QPolygonF([QPointF(12.6, 1.9), QPointF(12.6, 5.6), QPointF(8.8, 5.6)]))
        painter.end()
        icon.addPixmap(pixmap, mode, state)
    return icon


class RegionToolbar(QWidget):
    """Region intents with a compact overflow variant; never edits the model."""
    shape_requested = Signal(str)
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("regionToolbar")
        self.setAccessibleName("跟踪区域形状工具栏")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._shape = "rectangle"
        self._mode = "select"
        self._compact = False
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self.label = QLabel("● 跟踪区域", self)
        self.label.setObjectName("regionLabel")
        self.label.setToolTip("跟踪区域：选择参与跟踪的像素")
        self._layout.addWidget(self.label)
        self.shape_group = QButtonGroup(self)
        self.shape_group.setExclusive(True)
        self.shape_buttons = {}
        for shape, text in _SHAPES:
            button = self._button(shape, text + "跟踪区域", checkable=True)
            self.shape_buttons[shape] = button
            self.shape_group.addButton(button)
            button.clicked.connect(lambda _checked=False, value=shape: self._request_shape(value))
        self.edit_button = self._button("edit", _ACTION_LABELS["edit"], checkable=True)
        self.add_button = self._button("add", _ACTION_LABELS["add"], checkable=True)
        self.delete_button = self._button("delete", _ACTION_LABELS["delete"])
        self.reset_button = self._button("reset", _ACTION_LABELS["reset"])
        for action, button in self._action_buttons().items():
            button.clicked.connect(lambda _checked=False, value=action: self._request_action(value))

        self.shape_combo = QComboBox(self)
        self.shape_combo.setObjectName("regionShapeCombo")
        self.shape_combo.setAccessibleName("当前跟踪区域形状")
        self.shape_combo.setToolTip("当前跟踪区域形状")
        self.shape_combo.setFixedSize(72, 24)
        for shape, text in _SHAPES:
            self.shape_combo.addItem(text, shape)
        self.shape_combo.currentIndexChanged.connect(self._combo_changed)
        self._layout.addWidget(self.shape_combo)

        self.overflow_button = QToolButton(self)
        self.overflow_button.setObjectName("regionOverflow")
        self.overflow_button.setIcon(_tool_icon('more'))
        self.overflow_button.setAccessibleName("更多区域编辑工具")
        self.overflow_button.setToolTip("更多区域编辑工具")
        self.overflow_button.setFixedSize(24, 24)
        self.overflow_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.overflow_menu = QMenu(self.overflow_button)
        self.overflow_actions = {}
        for name in ("add", "delete", "reset"):
            action = QAction(_tool_icon(name), _ACTION_LABELS[name], self.overflow_menu)
            action.setToolTip(_ACTION_LABELS[name])
            action.triggered.connect(lambda _checked=False, value=name: self._request_action(value))
            self.overflow_menu.addAction(action)
            self.overflow_actions[name] = action
        self.overflow_button.setMenu(self.overflow_menu)
        self.overflow_button.hide()
        self._layout.addWidget(self.overflow_button)
        self.setStyleSheet("""
            QWidget#regionToolbar { background: transparent; }
            QLabel#regionLabel { color: #e88c51; font-size: 10px; padding-right: 3px; }
            QToolButton { background: transparent; border: 1px solid transparent;
                          border-radius: 2px; padding: 0; color: #b7afa5; }
            QToolButton:hover { background: #2c251f; border-color: #69513e; }
            QToolButton:checked { background: #2e241a; border-color: #e88c51; }
            QToolButton:disabled { color: #68635c; border-color: transparent; background: transparent; }
            QToolButton::menu-indicator { image: none; }
            QComboBox#regionShapeCombo { padding: 0 5px; min-height: 0; background: #191918;
                color: #dfd5c6; border: 1px solid #3d3934; border-radius: 2px; font-size: 10px; }
            QComboBox#regionShapeCombo:disabled { color: #68635c; border-color: #35302a; }
            QComboBox#regionShapeCombo::drop-down { width: 14px; border: none; }
        """)
        self.set_state("rectangle", ready=False, busy=False)

    def _button(self, icon, label, *, checkable=False):
        button = QToolButton(self)
        button.setObjectName("region" + icon.title())
        button.setIcon(_tool_icon(icon))
        button.setIconSize(QSize(16, 16))
        button.setFixedSize(24, 24)
        button.setCheckable(checkable)
        button.setToolTip(label)
        button.setAccessibleName(label)
        self._layout.addWidget(button)
        return button

    def _action_buttons(self):
        return {"edit": self.edit_button, "add": self.add_button,
                "delete": self.delete_button, "reset": self.reset_button}

    def _combo_changed(self, index):
        shape = self.shape_combo.itemData(index)
        if shape:
            self._request_shape(shape)

    def _request_shape(self, shape):
        if not self.shape_buttons[shape].isEnabled():
            return
        self.set_state(shape, self._ready, self._busy, self._can_delete, self._mode, self._can_add)
        self.shape_requested.emit(shape)

    def _request_action(self, action):
        if not self._action_buttons()[action].isEnabled():
            return
        if action in ("edit", "add"):
            self._mode = "select" if action == "edit" else "add"
            self._sync_checked()
        self.action_requested.emit(action)

    def _sync_checked(self):
        controls = [*self.shape_buttons.values(), self.shape_combo, self.edit_button, self.add_button]
        blockers = [QSignalBlocker(control) for control in controls]
        for shape, button in self.shape_buttons.items():
            button.setChecked(shape == self._shape)
        self.shape_combo.setCurrentIndex(self.shape_combo.findData(self._shape))
        self.edit_button.setChecked(self._mode not in ("add", "add_vertex", "polygon_add"))
        self.add_button.setChecked(self._mode in ("add", "add_vertex", "polygon_add"))
        del blockers

    def set_state(self, shape, ready, busy, can_delete=False, mode="select", can_add=True):
        """Reflect host state without requests; legacy circle is an ellipse view."""
        self._ready, self._busy = bool(ready), bool(busy)
        self._can_delete, self._can_add = bool(can_delete), bool(can_add)
        self._shape = {"rect": "rectangle", "circle": "ellipse"}.get(shape, shape)
        if self._shape not in self.shape_buttons:
            self._shape = "rectangle"
        self._mode = mode
        enabled = bool(ready and not busy)
        for button in self.shape_buttons.values():
            button.setEnabled(enabled)
        self.shape_combo.setEnabled(enabled)
        self.edit_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.add_button.setEnabled(enabled and self._shape == "polygon" and can_add)
        self.delete_button.setEnabled(enabled and self._shape == "polygon" and can_delete)
        for action, button in self._action_buttons().items():
            if action in self.overflow_actions:
                self.overflow_actions[action].setEnabled(button.isEnabled())
        self.overflow_button.setEnabled(enabled)
        self._sync_checked()

    def set_available_width(self, width):
        """Receive central-header width, not this widget's allocated width."""
        compact = width < 760
        self._compact = compact
        self.label.setVisible(not compact)
        for button in (self.add_button, self.delete_button, self.reset_button):
            button.setVisible(not compact)
        self.overflow_button.setVisible(compact)
        self._layout.invalidate()
        self.updateGeometry()

    def sizeHint(self):
        return self._layout.sizeHint()

    def minimumSizeHint(self):
        return QSize(0, 24)
