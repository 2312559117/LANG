"""Inline, source-time segment range and target controls."""
from copy import deepcopy
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSizePolicy
from .section_workflow import section_configuration, active_sections, tracking_ranges, sections_match
from .tracking_pause import computed_mask


class SectionsMixin:
    def _open_tracking_sections(self):
        if not self.info or self.busy:
            return
        self._stop_play()
        if not self.section_tracking:
            self._set_tracking_sections(True, self.tracking_sections)
        self._sections_panel.refresh()
        self._sections_panel.setFocus()

    def _set_tracking_sections(self, enabled, sections):
        if not self.info or self.busy:
            return False
        try:
            enabled, sections = section_configuration(enabled, sections, self.info.frames)
        except ValueError as error:
            self.status_label.setText(str(error))
            return False
        if (enabled, sections) == (self.section_tracking, self.tracking_sections):
            return True
        self._stop_play()
        self._tb_clear_history()
        if self._paused_tracking:
            self._abandon_paused_tracking(sync=False)
        if self.track is not None:
            self._retrack_previous = self.track
        self.section_tracking, self.tracking_sections = enabled, deepcopy(sections)
        self.track = self.plan = None
        self.dirty = self._sequence_changed = True
        self._stash_clip()
        self._refresh_issues()
        self._show_current()
        self._update_strip()
        self._sync_ui()
        self.status_label.setText('分段已更新：在每段内选目标，再点击开始跟踪。未覆盖帧不会插值或伪装成功。'
                                  if enabled else '已切回整段跟踪，分段草稿保留；请重新开始跟踪。')
        return True

    def _sections_stale(self):
        return self.track is not None and not sections_match(
            self.track, self.section_tracking, self.tracking_sections,
            self.start_spin.value()-1, self.end_spin.value()-1)

    def _sections_sync(self):
        if not hasattr(self, 'sections_button'):
            return
        self.sections_button.setEnabled(self.info is not None and not self.busy)
        blocker = QSignalBlocker(self.sections_button)
        self.sections_button.setChecked(self.section_tracking)
        del blocker
        self.sections_button.setText('分段跟踪')
        self.sections_button.setToolTip('启用后在下方时间线上按 I 设入点、O 设出点，再在左原片框选本段目标。')
        if self.section_tracking and self._paused_tracking is None:
            self.track_button.setText('更新分段跟踪' if self.track is not None else '开始分段跟踪')
        if self._sections_stale():
            self.export_button.setEnabled(False)
        panel = getattr(self, '_sections_panel', None)
        if panel is not None:
            panel.setVisible(bool(self.info and self.section_tracking))
            panel.refresh()


class SectionsPanel(QWidget):
    """Source I/O and per-range targets, directly above the timeline."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.pending = {}
        self.setObjectName('inlineSections')
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel('独立分段'))
        self.selector = QComboBox()
        self.selector.setMinimumWidth(180)
        self.selector.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.selector.currentIndexChanged.connect(self.choose)
        row.addWidget(self.selector, 1)
        self.in_button = self.button(row, 'I  入点', self.mark_in)
        self.out_button = self.button(row, 'O  出点', self.mark_out)
        self.pick_button = self.button(row, '选本段目标', self.pick)
        self.split_button = self.button(row, '此处分段', self.split)
        self.remove_button = self.button(row, '移除本段', self.remove)
        layout.addLayout(row)
        self.status = QLabel('在时间线上定位，I设入点、O设出点，随后在左原片框选。')
        self.status.setObjectName('muted')
        self.status.setWordWrap(False)
        layout.addWidget(self.status)

    def button(self, row, title, callback):
        button = QPushButton(title)
        button.clicked.connect(lambda checked=False: callback())
        row.addWidget(button)
        return button

    def selected(self):
        return self.selector.currentData()

    def choose(self, index):
        w = self.window
        section = self.selected()
        if section is None or w.busy:
            return
        frame = w.frame_index
        if not section['start'] <= frame <= section['end']:
            frame = next((f for f in sorted(w.anchors) if section['start'] <= f <= section['end']), section['start'])
        w._seek(frame)

    def refresh(self):
        w = self.window
        if not w.info:
            return
        self.pending = {key: value for key, value in self.pending.items() if w._clip(key) is not None}
        selected = self.selected()
        blocker = QSignalBlocker(self.selector)
        self.selector.clear()
        mask = computed_mask(w.track) if w.track is not None else None
        for index, section in enumerate(w.tracking_sections):
            a, b = section['start'], section['end']
            keys = [f for f in w.anchors if a <= f <= b]
            state = '待选目标' if not keys else '已选目标'
            if w.track is not None:
                lo, hi = max(a, w.track.start), min(b, w.track.end)
                if hi >= lo:
                    done = int(mask[lo-w.track.start:hi-w.track.start+1].sum())
                    lost = sum(w.track.states[f-w.track.start]=='lost' and mask[f-w.track.start] for f in range(lo,hi+1))
                    state = f'待修正 {lost}帧' if lost else '已跟踪' if done==hi-lo+1 else f'{done}/{hi-lo+1}帧'
            self.selector.addItem(f'段{index+1} · {a+1}–{b+1}帧 · {state}', dict(section))
        if selected in w.tracking_sections:
            self.selector.setCurrentIndex(w.tracking_sections.index(selected))
        del blocker
        for button in (self.in_button, self.out_button, self.split_button):
            button.setEnabled(not w.busy)
        self.selector.setEnabled(not w.busy)
        for button in (self.pick_button, self.remove_button):
            button.setEnabled(bool(w.tracking_sections) and not w.busy)
        self.out_button.setEnabled(not w.busy and w.active_clip_id in self.pending)
        self.update_status()

    def update_status(self):
        w = self.window
        pending = self.pending.get(w.active_clip_id)
        if pending is not None:
            text = f'入点：第 {pending+1} 帧 → 定位出点后按 O。分段仅影响跟踪，不改变视频时长。'
        elif not w.tracking_sections:
            text = '定位 → I 入点 → O 出点 → 左原片选目标。各段需覆盖保留范围，随后点开始跟踪。'
        else:
            try:
                tracking_ranges(True, w.tracking_sections, w.anchors, w.start_spin.value()-1, w.end_spin.value()-1)
                text = f'{len(w.tracking_sections)}段已就绪 · 点击开始跟踪；段内独立参考，不跨段插值。'
            except ValueError as error:
                text = str(error)
        self.status.setText(text)
        self.status.setToolTip(text)

    def cursor_changed(self):
        w = self.window
        w.strip.pending_section = self.pending.get(w.active_clip_id) if w.section_tracking else None
        if not w.section_tracking:
            return
        for index, section in enumerate(w.tracking_sections):
            if section['start'] <= w.frame_index <= section['end']:
                blocker = QSignalBlocker(self.selector)
                self.selector.setCurrentIndex(index)
                del blocker
                break

    def mark_in(self):
        w = self.window
        if not w.info or w.busy:
            return
        if not w.start_spin.value()-1 <= w.frame_index <= w.end_spin.value()-1:
            self.status.setText('请在保留片段范围内设置分段入点。')
            return
        self.pending[w.active_clip_id] = w.frame_index
        w.strip.pending_section = w.frame_index
        w.strip.update()
        self.out_button.setEnabled(True)
        self.update_status()

    def mark_out(self):
        w = self.window
        first = self.pending.get(w.active_clip_id)
        if first is None or w.busy:
            return
        if w.frame_index < first:
            self.status.setText('出点不能早于入点，请重新定位。')
            return
        if not w.start_spin.value()-1 <= first <= w.frame_index <= w.end_spin.value()-1:
            self.status.setText('分段不能超出保留片段范围。')
            return
        section = dict(start=first, end=w.frame_index)
        if self.apply(w.tracking_sections+[section]):
            self.pending.pop(w.active_clip_id, None)
            w.strip.pending_section = None
            w.strip.update()
            self.refresh()
            self.selector.setCurrentIndex(w.tracking_sections.index(section))
            w._target_selection_tool()
            self.status.setText('分段已建立：请在左侧原片框选本段目标，再建立下一段或开始跟踪。')

    def apply(self, sections):
        if self.window._set_tracking_sections(True, sections):
            self.refresh()
            return True
        self.status.setText(self.window.status_label.text())
        return False

    def split(self):
        w = self.window
        if w.busy or not w.info:
            return
        sections = deepcopy(w.tracking_sections) or [dict(start=w.start_spin.value()-1,end=w.end_spin.value()-1)]
        section = next((s for s in sections if s['start'] < w.frame_index <= s['end']), None)
        if section is None:
            self.status.setText('请定位到段内下一段的首帧，首帧不能再次分段。')
            return
        sections.remove(section)
        sections.extend([dict(start=section['start'],end=w.frame_index-1),dict(start=w.frame_index,end=section['end'])])
        if self.apply(sections):
            self.cursor_changed()
            self.pick()

    def pick(self):
        if self.window.busy or self.selected() is None:
            return
        self.choose(self.selector.currentIndex())
        self.window._target_selection_tool()
        self.status.setText('在左侧原片框选当前段目标；本操作不修改构图。')

    def remove(self):
        section = self.selected()
        if section is not None and not self.window.busy:
            self.apply([s for s in self.window.tracking_sections if s != section])
