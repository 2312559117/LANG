"""Non-modal, transactional editor for the version 2 local retime contract."""
from copy import deepcopy
import math

from PySide6.QtCore import Qt, Signal, QSignalBlocker
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QScrollArea, QStackedWidget, QLabel, QCheckBox, QPushButton, QComboBox,
    QDoubleSpinBox, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QGroupBox,
)

from .time_remap_model import (
    normalize_config, active_points, make_constant_speed, MAX_KEYFRAMES,
)
from .time_remap_math import evaluate_source
from .time_remap_map import output_frame_count
from .time_remap_ui import CurveCanvas


class TimeRemapEditor(QDialog):
    """Edit an isolated draft; only Apply publishes a validated deep copy.

    ``source_frames`` is the selected source range, not the full media length.
    Closing or cancelling discards edits since the last Apply. No host window,
    tracking state, media reader, or inference worker is owned by this dialog.
    """

    applied = Signal(object)

    def __init__(self, config, source_frames, fps, parent=None):
        super().__init__(parent)
        self.setWindowTitle('变速与时间曲线')
        self.setModal(False)
        self.resize(1000, 810)
        self._refreshing = False
        self._selected = 0
        self._invalid_fields = set()
        self._build()
        self.set_config(config, source_frames, fps)

    @staticmethod
    def _spin(lo, hi, decimals=4, step=.1):
        control = QDoubleSpinBox()
        control.setRange(lo, hi)
        control.setDecimals(decimals)
        control.setSingleStep(step)
        control.setKeyboardTracking(False)
        return control

    @staticmethod
    def _combo(items):
        control = QComboBox()
        for text, value in items:
            control.addItem(text, value)
        return control

    def _button(self, text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build(self):
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.enabled = QCheckBox('启用变速')
        self.mode = self._combo([('基础四段速度', 'basic'), ('高级时间曲线', 'advanced')])
        self.view = self._combo([('位置曲线', False), ('速度曲线', True)])
        top.addWidget(self.enabled)
        top.addWidget(self.mode)
        presets=QPushButton('曲线预设…');presets.clicked.connect(self._open_presets);top.addWidget(presets)
        top.addStretch()
        top.addWidget(self.view)
        root.addLayout(top)
        self.preview_time=QLabel('Alt+单击曲线可定位原片；草稿仅在应用后影响效果')
        root.addWidget(self.preview_time)
        self.canvas = CurveCanvas(editable=True)
        self.canvas.setMinimumHeight(210)
        root.addWidget(self.canvas)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        content = QVBoxLayout(body)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        self.pages = QStackedWidget()
        content.addWidget(self.pages)

        basic = QWidget()
        grid = QGridLayout(basic)
        self.speeds, self.boundaries = [], []
        for i in range(5):
            spin = self._spin(.1, 10.)
            spin.setSuffix(' ×')
            grid.addWidget(QLabel(('开始', '节点一', '节点二', '节点三', '结束')[i]), 0, i)
            grid.addWidget(spin, 1, i)
            spin.valueChanged.connect(lambda value, index=i: self._basic_value('speeds', index, value))
            spin.editingFinished.connect(self._commit)
            self.speeds.append(spin)
        for i in range(3):
            spin = self._spin(8., 96., 2, 1.)
            spin.setSuffix(' %')
            grid.addWidget(QLabel(f'源位置边界 {i+1}'), 2, i)
            grid.addWidget(spin, 3, i)
            spin.valueChanged.connect(lambda value, index=i: self._basic_value('boundaries', index, value / 100.))
            spin.editingFinished.connect(self._commit)
            self.boundaries.append(spin)
        convert = QPushButton('转换为高级曲线（保留输出时长）')
        convert.clicked.connect(self.convert_basic)
        grid.addWidget(convert, 4, 0, 1, 5)
        self.pages.addWidget(basic)

        advanced = QWidget()
        layout = QVBoxLayout(advanced)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['输出位置 / 源时长', '源进度', '左侧插值', '右侧插值'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(115)
        self.table.setMaximumHeight(170)
        self.table.currentCellChanged.connect(lambda row, *_: self._select(row))
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self._button('添加节点', self.add_point, row)
        self.delete_button = self._button('删除选中节点', self.remove_point, row)
        row.addStretch()
        layout.addLayout(row)
        form = QGridLayout()
        self.point_controls = {}
        fields = [
            ('position', '输出位置（源时长倍数）', 0., 100.),
            ('source', '源进度（0–1）', 0., 1.),
            ('left_influence', '左影响比例', 0., 1.),
            ('right_influence', '右影响比例', 0., 1.),
        ]
        for index, (name, title, lo, hi) in enumerate(fields):
            spin = self._spin(lo, hi, 6, .01)
            self.point_controls[name] = spin
            form.addWidget(QLabel(title), index // 2, (index % 2) * 2)
            form.addWidget(spin, index // 2, (index % 2) * 2 + 1)
            spin.valueChanged.connect(lambda value, key=name: self._point_value(key, value))
            spin.editingFinished.connect(self._commit)
        for index, (name, title) in enumerate([
            ('incoming_interpolation', '左侧插值'), ('interpolation', '右侧插值')
        ]):
            combo = self._combo([('贝塞尔', 'bezier'), ('线性', 'linear')])
            self.point_controls[name] = combo
            form.addWidget(QLabel(title), 2, index * 2)
            form.addWidget(combo, 2, index * 2 + 1)
            combo.currentIndexChanged.connect(lambda _, key=name, control=combo: self._point_value(key, control.currentData(), True))
        optional = [
            ('left_handle_source', '左手柄源位置'), ('right_handle_source', '右手柄源位置'),
            ('incoming_speed', '入速度 ×'), ('outgoing_speed', '出速度 ×'),
            ('left_speed_tangent', '左速度切线'), ('right_speed_tangent', '右速度切线'),
        ]
        for index, (name, title) in enumerate(optional):
            edit = QLineEdit()
            edit.setPlaceholderText('自动（留空）')
            edit.setToolTip('留空使用自动值；有效值范围以数据契约为准，超出时归一化。')
            self.point_controls[name] = edit
            form.addWidget(QLabel(title), 3 + index // 2, index % 2 * 2)
            form.addWidget(edit, 3 + index // 2, index % 2 * 2 + 1)
            edit.textEdited.connect(lambda _, key=name: self._invalid_fields.discard(key))
            edit.editingFinished.connect(lambda key=name: self._optional_value(key))
        self.linked = QCheckBox('关联左右手柄')
        self.linked.toggled.connect(lambda value: self._point_value('handles_linked', value, True))
        form.addWidget(self.linked, 6, 0, 1, 4)
        layout.addLayout(form)
        layout.addWidget(QLabel('输出位置 1 = 所选源范围的完整时长；源进度可下降（倒放）或保持不变（定格）。'))
        self.pages.addWidget(advanced)

        actions = QGroupBox('快速操作')
        row = QHBoxLayout(actions)
        self.constant_speed = self._spin(.01, 100., 3)
        self.constant_speed.setValue(1.)
        self.constant_speed.setSuffix(' ×')
        row.addWidget(self.constant_speed)
        self._button('固定倍速', lambda: self.set_constant_speed(self.constant_speed.value()), row)
        self._button('倒放', lambda: self.set_constant_speed(-self.constant_speed.value()), row)
        row.addWidget(QLabel('定格源位置'))
        self.freeze_source = self._spin(0., 1., 5, .01)
        row.addWidget(self.freeze_source)
        self.freeze_seconds = self._spin(.001, 1_000_000., 3, 1.)
        self.freeze_seconds.setValue(1.)
        self.freeze_seconds.setSuffix(' 秒')
        row.addWidget(self.freeze_seconds)
        self._button('定格', self._freeze_clicked, row)
        content.addWidget(actions)

        options = QGroupBox('光流与声音')
        form = QFormLayout(options)
        self.flow = QCheckBox('启用本地双向光流插帧')
        self.backend = self._combo([('自动（当前使用 DIS）', 'auto'), ('DIS / CPU', 'dis')])
        self.preview_flow = QCheckBox('效果预览使用光流（会增加计算量）')
        self.allow_fallback = QCheckBox('我同意：光流失败时退回最近源帧（可能不流畅）')
        self.audio = self._combo([('静音', 'mute'), ('按时间映射同步声音', 'mapped')])
        form.addRow(self.flow, self.backend)
        form.addRow(self.preview_flow)
        form.addRow(self.allow_fallback)
        form.addRow('声音策略', self.audio)
        note = QLabel('NVOF 尚未实现并验证，当前不提供 GPU 光流选项。映射声音可能变调；含倒放、定格或非单调映射时静音。')
        note.setWordWrap(True)
        form.addRow(note)
        content.addWidget(options)
        content.addStretch()
        self.message = QLabel()
        self.message.setWordWrap(True)
        root.addWidget(self.message)
        buttons = QHBoxLayout()
        self.undo_button = self._button('撤销', self.undo, buttons)
        self.redo_button = self._button('重做', self.redo, buttons)
        self._button('重置', self.reset, buttons)
        buttons.addStretch()
        self._button('取消', self.reject, buttons)
        self.apply_button = self._button('应用', self.apply_changes, buttons)
        root.addLayout(buttons)
        self.enabled.toggled.connect(lambda value: self._config_value('enabled', value))
        self.mode.currentIndexChanged.connect(lambda _: self._config_value('mode', self.mode.currentData()))
        self.view.currentIndexChanged.connect(self._view_changed)
        for control, name in ((self.flow, 'optical_flow'), (self.preview_flow, 'preview_flow'), (self.allow_fallback, 'allow_frame_fallback')):
            control.toggled.connect(lambda value, key=name: self._config_value(key, value))
        for control, name in ((self.backend, 'flow_backend'), (self.audio, 'audio_policy')):
            control.currentIndexChanged.connect(lambda _, key=name, widget=control: self._config_value(key, widget.currentData()))
        self.canvas.changing.connect(self._canvas_changing)
        self.canvas.committed.connect(self._canvas_committed)
        self.canvas.selectedChanged.connect(self._select)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)

    def set_config(self, config, source_frames=None, fps=None):
        count = source_frames if source_frames is not None else self.source_frames
        rate = fps if fps is not None else self.fps
        if type(count) is not int or count < 1 or isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('源范围帧数和帧率必须有效')
        self.source_frames, self.fps = count, float(rate)
        self.draft = normalize_config(normalize_config(config or None))
        self._baseline = deepcopy(self.draft)
        self._history = [deepcopy(self.draft)]
        self._history_index = 0
        self._invalid_fields.clear()
        self._selected = 0
        self.message.clear()
        self._refresh()

    def _open_presets(self):
        from .curve_preset_ui import PresetPicker
        self._preset_picker=PresetPicker(self.draft,self.source_frames,self)
        self._preset_picker.applied.connect(self._change);self._preset_picker.show()

    def _view_changed(self):
        self.canvas.speed_view = self.view.currentData()
        self.canvas.update()

    def _refresh(self, canvas=True):
        self._refreshing = True
        controls = [self.enabled, self.mode, self.flow, self.backend, self.preview_flow, self.allow_fallback, self.audio,
                    *self.speeds, *self.boundaries, *self.point_controls.values(), self.linked, self.table]
        blockers = [QSignalBlocker(control) for control in controls]
        try:
            config = self.draft
            for control, key in ((self.enabled, 'enabled'), (self.flow, 'optical_flow'), (self.preview_flow, 'preview_flow'), (self.allow_fallback, 'allow_frame_fallback')):
                control.setChecked(config[key])
            for control, key in ((self.mode, 'mode'), (self.backend, 'flow_backend'), (self.audio, 'audio_policy')):
                control.setCurrentIndex(control.findData(config[key]))
            self.pages.setCurrentIndex(0 if config['mode'] == 'basic' else 1)
            for spin, value in zip(self.speeds, config['basic']['speeds']):
                spin.setValue(value)
            for spin, value in zip(self.boundaries, config['basic']['boundaries']):
                spin.setValue(value * 100.)
            points = config['points']
            self._selected = max(0, min(self._selected, len(points) - 1))
            self.table.setRowCount(len(points))
            for row, point in enumerate(points):
                for column, value in enumerate((f"{point['position']:.6g}", f"{point['source']:.6g}", point['incoming_interpolation'], point['interpolation'])):
                    self.table.setItem(row, column, QTableWidgetItem(value))
            self.table.selectRow(self._selected)
            point = points[self._selected]
            for key, control in self.point_controls.items():
                value = point[key]
                if isinstance(control, QComboBox):
                    control.setCurrentIndex(control.findData(value))
                elif isinstance(control, QLineEdit):
                    control.setText('' if value is None else f'{value:.10g}')
                else:
                    control.setValue(value)
            self.linked.setChecked(point['handles_linked'])
            self.delete_button.setEnabled(len(points) > 2)
            if canvas:
                self.canvas.set_config(config)
            self.canvas.selected = self._selected if config['mode'] == 'advanced' else min(self.canvas.selected, 4)
            self.canvas.update()
            planned = output_frame_count(self.source_frames, active_points(config))
            frames = planned if config['enabled'] else self.source_frames
            suffix = '' if config['enabled'] else f'；启用后 {planned} 帧 / {planned / self.fps:.3f} 秒'
            self.summary.setText(f'源范围 {self.source_frames} 帧 / {self.source_frames / self.fps:.3f} 秒 → 输出 {frames} 帧 / {frames / self.fps:.3f} 秒{suffix}')
            self.undo_button.setEnabled(self._history_index > 0)
            self.redo_button.setEnabled(self._history_index + 1 < len(self._history))
        finally:
            del blockers
            self._refreshing = False

    def _change(self, config, commit=True, canvas=True):
        if self._refreshing:
            return
        try:
            normalized = normalize_config(config)
            output_frame_count(self.source_frames, active_points(normalized))
        except (ValueError, OverflowError) as error:
            self.message.setText(str(error))
            return
        self.draft = normalized
        if commit:
            self._commit()
        self._refresh(canvas)

    def _commit(self):
        if self._refreshing or self.draft == self._history[self._history_index]:
            return
        self._history = self._history[:self._history_index + 1] + [deepcopy(self.draft)]
        self._history = self._history[-80:]
        self._history_index = len(self._history) - 1
        self.undo_button.setEnabled(self._history_index > 0)
        self.redo_button.setEnabled(False)

    def _config_value(self, key, value):
        config = deepcopy(self.draft)
        if key == 'mode' and value == 'advanced' and config['mode'] == 'basic':
            # Mode changes preserve the previously authored advanced curve.
            # Explicit conversion is available separately to replace it.
            self.message.setText('已打开保留的高级曲线；如需转换当前基础速度，请使用“转换为高级曲线”。')
        config[key] = value
        self._change(config)

    def _basic_value(self, key, index, value):
        config = deepcopy(self.draft)
        config['basic'][key][index] = value
        self._change(config, False)

    def _point_value(self, key, value, commit=False):
        config = deepcopy(self.draft)
        config['points'][self._selected][key] = value
        self._change(config, commit)

    def _optional_value(self, key):
        if self._refreshing:
            return
        text = self.point_controls[key].text().strip()
        try:
            value = float(text) if text else None
            if value is not None and not math.isfinite(value):
                raise ValueError()
        except ValueError:
            self._invalid_fields.add(key)
            self.message.setText('可选数值必须为有限数值；留空表示自动。')
            return
        self._invalid_fields.discard(key)
        self._point_value(key, value, True)

    def _select(self, index):
        if self._refreshing or index < 0:
            return
        if self.draft['mode']=='basic':
            self.canvas.selected=index;self.canvas.update();return
        self._selected = index
        self._refresh(canvas=False)

    def _canvas_changing(self, config):
        self._change(config, False, canvas=False)

    def _canvas_committed(self, config):
        self._change(config, True)

    def convert_basic(self):
        config = deepcopy(self.draft)
        config['mode'] = 'basic'
        config['points'] = active_points(config)
        config['mode'] = 'advanced'
        self._change(config)

    def add_point(self):
        config = deepcopy(self.draft)
        points = config['points']
        if len(points) >= MAX_KEYFRAMES:
            self.message.setText(f'最多允许 {MAX_KEYFRAMES} 个节点。')
            return
        index = min(self._selected, len(points) - 2)
        x = (points[index]['position'] + points[index + 1]['position']) / 2.
        if points[index + 1]['position'] - points[index]['position'] < .0002:
            self.message.setText('所选区间太短，无法再添加独立节点。')
            return
        points.append(dict(position=x, source=evaluate_source(points, x)))
        self._selected = index + 1
        self._change(config)

    def remove_point(self):
        if len(self.draft['points']) <= 2:
            return
        config = deepcopy(self.draft)
        config['points'].pop(self._selected)
        self._change(config)

    def set_constant_speed(self, speed=1.):
        self._selected = 0
        self._change(make_constant_speed(self.draft, speed))

    def set_freeze(self, source, seconds):
        if isinstance(source, bool) or isinstance(seconds, bool) or not math.isfinite(source) or not math.isfinite(seconds) or not 0 <= source <= 1 or seconds <= 0:
            raise ValueError('定格需要有效源位置与正时长')
        scale = seconds / (self.source_frames / self.fps)
        if not .01 <= scale <= 100.:
            raise ValueError('定格时长需为源范围时长的 0.01–100 倍')
        config = deepcopy(self.draft)
        config.update(enabled=True, mode='advanced', points=[
            dict(position=0., source=source, interpolation='linear', incoming_interpolation='linear'),
            dict(position=scale, source=source, interpolation='linear', incoming_interpolation='linear'),
        ])
        self._change(config)

    def _freeze_clicked(self):
        try:
            self.set_freeze(self.freeze_source.value(), self.freeze_seconds.value())
        except ValueError as error:
            self.message.setText(str(error))

    def undo(self):
        if self.draft != self._history[self._history_index]:
            self._commit()
        if self._history_index:
            self._history_index -= 1
            self.draft = deepcopy(self._history[self._history_index])
            self._invalid_fields.clear()
            self._refresh()

    def redo(self):
        if self._history_index + 1 < len(self._history):
            self._history_index += 1
            self.draft = deepcopy(self._history[self._history_index])
            self._invalid_fields.clear()
            self._refresh()

    def reset(self):
        self._selected = 0
        self._invalid_fields.clear()
        self._change(normalize_config())

    def apply_changes(self):
        # Finish editable fields even when invoked programmatically, without
        # allowing a stale valid draft to hide an invalid visible input.
        config = deepcopy(self.draft)
        invalid = set()
        for name, control in self.point_controls.items():
            if isinstance(control, QLineEdit):
                text = control.text().strip()
                try:
                    value = float(text) if text else None
                    if value is not None and not math.isfinite(value):
                        raise ValueError()
                except ValueError:
                    invalid.add(name)
                    continue
                config['points'][self._selected][name] = value
        self._invalid_fields = invalid
        if invalid:
            self.message.setText('请先修正无效数值，或留空使用自动值。')
            return False
        try:
            config = normalize_config(config)
            output_frame_count(self.source_frames, active_points(config))
        except (ValueError,OverflowError) as error:
            self.message.setText(str(error))
            return False
        self.draft = config
        self._commit()
        self._baseline = deepcopy(config)
        self._refresh()
        self.applied.emit(deepcopy(config))
        self.message.setText('已应用到当前素材。后续编辑仍需再次应用。')
        return True

    def reject(self):
        self.draft = deepcopy(self._baseline)
        self._history = [deepcopy(self.draft)]
        self._history_index = 0
        self._invalid_fields.clear()
        self._refresh()
        super().reject()
