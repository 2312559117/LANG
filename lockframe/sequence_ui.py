"""Clip list, selection, ordering and queue controls for the desktop workspace."""
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPoint, QItemSelectionModel, QSignalBlocker, QSize
from PySide6.QtWidgets import (QAbstractItemView, QListWidget, QListWidgetItem, QPushButton,
                               QHBoxLayout, QComboBox, QLabel, QFileDialog, QMessageBox, QApplication, QMenu)
from PySide6.QtGui import QAction, QPixmap, QActionGroup

from .sequence import Clip, import_many, save_sequence, read_sequence, restore_sequence
from .media import cache_root
from .workers import QueueJob
from .workspace_ui import ClipList
from .tracking_pause import TrackingPaused, is_complete


class SequenceMixin:
    def _init_sequence(self):
        self.clips = []
        self.active_clip_id = None
        self.sequence_path = None
        self._switching_clip = False
        self._sequence_changed = False
        from .thumbnail_queue import ThumbnailQueue
        self._thumbnails = ThumbnailQueue(self)
        self._thumbnails.ready.connect(self._thumbnail_ready)
        self._thumbnail_list_refreshing = False
        self._thumbnail_items = {}
        self._thumbnail_probe_timer = QTimer(self)
        self._thumbnail_probe_timer.setInterval(2000)
        self._thumbnail_probe_timer.timeout.connect(self._refresh_visible_thumbnails)
        self._thumbnail_probe_timer.start()

    def _sequence_menu(self, menu):
        acceptance_action = QAction("保存当前素材验收记录…", self)
        acceptance_action.triggered.connect(self._save_acceptance_record)
        menu.addAction(acceptance_action)
        from .batch_export_ui import open_export_queue
        for title, selected in (("导出选中素材…", True), ("导出全部可导出素材…", False)):
            action = QAction(title, self)
            action.triggered.connect(lambda checked=False, choice=selected: open_export_queue(self, choice))
            menu.addAction(action)
        self.open_sequence_action = QAction("打开素材序列…", self)
        self.open_sequence_action.triggered.connect(self._open_sequence)
        menu.addAction(self.open_sequence_action)
        self.save_sequence_action = QAction("保存素材序列…", self)
        self.save_sequence_action.triggered.connect(self._save_sequence)
        menu.addAction(self.save_sequence_action)
        self.recover_sequence_action = QAction("恢复续跑任务…", self)
        self.recover_sequence_action.triggered.connect(self._recover_checkpoint_task)
        menu.addAction(self.recover_sequence_action)

    def _save_acceptance_record(self):
        if self.busy or self.info is None:
            return
        self._stash_clip()
        clip = self._clip()
        if clip is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存验收记录 · 未填写项保持待验收", "素材验收.json", "验收记录 (*.json)")
        if not path:
            return
        import json
        from .acceptance import acceptance_record
        try:
            target = Path(path).resolve()
            protected = {Path(p).resolve() for c in self.clips for p in (c.info.path, c.info.proxy) if p}
            if target in protected:
                raise ValueError("验收记录不能覆盖源视频或分析预览")
            target.write_text(json.dumps(acceptance_record(clip), ensure_ascii=False, indent=2), encoding="utf-8")
            self.status_label.setText("已保存验收记录：自动数据已填入，实拍身份与目视检查仍需人工填写。")
        except Exception as error:
            self._error(str(error))

    def _sequence_panel(self, layout):
        self.clip_list = ClipList()
        self.clip_list.setMinimumHeight(145)
        self.clip_list.setMaximumHeight(220)
        self.clip_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.clip_list.setToolTip("双击打开素材；Ctrl / Shift 多选。每条素材分别设置目标和片段范围。")
        self.clip_list.itemDoubleClicked.connect(lambda item: self._activate_clip(item.data(Qt.ItemDataRole.UserRole)))
        self.clip_list.itemClicked.connect(self._preview_clicked_clip)
        self.clip_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.clip_list.customContextMenuRequested.connect(self._clip_context_menu)
        self.clip_list.verticalScrollBar().valueChanged.connect(lambda value:self._refresh_visible_thumbnails())
        layout.addWidget(self.clip_list)
        row = QHBoxLayout()
        self.select_all_clips = QPushButton("全选")
        self.select_all_clips.clicked.connect(self.clip_list.selectAll)
        self.open_clip_button = QPushButton("打开选中")
        self.open_clip_button.clicked.connect(self._open_selected_clip)
        row.addWidget(self.select_all_clips)
        row.addWidget(self.open_clip_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.clip_up, self.clip_down, self.clip_remove = QPushButton("上移"), QPushButton("下移"), QPushButton("移除")
        self.clip_up.clicked.connect(lambda: self._move_clips(-1))
        self.clip_down.clicked.connect(lambda: self._move_clips(1))
        self.clip_remove.clicked.connect(self._remove_clips)
        for button in (self.clip_up, self.clip_down, self.clip_remove):
            row.addWidget(button)
        layout.addLayout(row)
        self.queue_mode = QComboBox()
        self.queue_mode.addItem("按序列顺序 · 逐条跟踪", 1)
        self.queue_mode.addItem("同时跟踪 · 两条并行", 2)
        layout.addWidget(self.queue_mode)
        self.queue_button = QPushButton("一键跟踪选中素材")
        self.queue_button.setObjectName("trackAction")
        self.queue_button.clicked.connect(self._start_queue)
        layout.addWidget(self.queue_button)
        self.queue_summary = QLabel("批量导入后，双击各素材设置跟踪区域。")
        self.queue_summary.setWordWrap(True)
        self.queue_summary.setObjectName("muted")
        layout.addWidget(self.queue_summary)

    def _install_clip_views(self):
        menu = self.sidebar_sections['materials'].more_button.menu().addMenu('素材显示方式')
        group = QActionGroup(self)
        group.setExclusive(True)
        self.clip_view_actions = {}
        for title, mode in (('列表 · 原有排列', 'list'), ('缩略图网格', 'grid')):
            action = menu.addAction(title)
            action.setCheckable(True)
            group.addAction(action)
            self.clip_view_actions[mode] = action
            action.triggered.connect(lambda checked=False, value=mode: self._set_clip_view_mode(value))
        self._clip_view_group = group
        self._set_clip_view_mode(str(self.preferences.value('assets/view', 'list')), persist=False)

    def _set_clip_view_mode(self, mode, persist=True):
        mode = mode if mode in ('list', 'grid') else 'list'
        grid = mode == 'grid'
        self._clip_view_mode = mode
        self.clip_list.setProperty('clipViewMode', mode)
        self.clip_list.setViewMode(QListWidget.ViewMode.IconMode if grid else QListWidget.ViewMode.ListMode)
        self.clip_list.setFlow(QListWidget.Flow.LeftToRight if grid else QListWidget.Flow.TopToBottom)
        self.clip_list.setWrapping(grid)
        self.clip_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.clip_list.setMovement(QListWidget.Movement.Static)
        self.clip_list.setSpacing(4 if grid else 0)
        if grid:self.clip_list.update_grid()
        else:self.clip_list.setGridSize(QSize())
        self.clip_list.setUniformItemSizes(True)
        for value, action in self.clip_view_actions.items():action.setChecked(value == mode)
        if hasattr(self, 'materials_sidebar'):
            self.materials_sidebar.setMaximumWidth(900 if grid else 360)
        self.clip_list.doItemsLayout()
        self.clip_list.viewport().update()
        if persist:self.preferences.setValue('assets/view', mode)
        self._refresh_visible_thumbnails()

    def _clip_context_menu(self, position):
        item = self.clip_list.itemAt(position)
        if item is None:
            return
        clip_id = item.data(Qt.ItemDataRole.UserRole)
        if self._clip(clip_id) is None:
            return
        # Right-click targets exactly this row, even after an earlier multi-selection.
        self.clip_list.setCurrentItem(item, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        menu = QMenu(self)
        menu.setObjectName('clipContextMenu')
        action = menu.addAction('从序列删除素材')
        action.setObjectName('removeClipFromSequence')
        action.setToolTip('只移除此条素材及其序列设置，保留硬盘上的原视频')
        action.setEnabled(not self.busy)
        action.triggered.connect(lambda checked=False: self._remove_clips(clip_ids={clip_id}))
        try:
            menu.exec(self.clip_list.viewport().mapToGlobal(position))
        finally:
            menu.deleteLater()

    def _preview_clicked_clip(self, item):
        if not QApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            self._activate_clip(item.data(Qt.ItemDataRole.UserRole))

    def _clip(self, clip_id=None):
        return next((clip for clip in self.clips if clip.id == (clip_id or self.active_clip_id)), None)

    def _stash_clip(self):
        if not self._ready or self._switching_clip or not self.info:
            return
        clip = self._clip()
        if clip is None:
            return
        if clip.track is not self.track or clip.anchors != self.anchors:
            clip.status = ""
        clip.settings, clip.anchors = replace(self.settings()), dict(self.anchors)
        clip.start, clip.end = self.start_spin.value() - 1, self.end_spin.value() - 1
        clip.track, clip.frame = self.track, self.frame_index
        clip.project_path, clip.dirty = self.project_path, self.dirty
        clip.timebar_state = dict(self.strip.state)
        clip.section_tracking = self.section_tracking
        clip.tracking_sections = [dict(value) for value in self.tracking_sections]
        clip.precision_info = self.precision_info
        clip.retrack_previous = self._retrack_previous
        if self._paused_tracking:
            self._paused_by_clip[clip.id] = self._paused_tracking
            clip.tracking_resume = self._paused_tracking.get("resume")

    def _register_current_clip(self):
        clip = next((value for value in self.clips if Path(value.info.path).resolve() == Path(self.info.path).resolve()), None)
        if clip is None:
            clip = Clip(self.info)
            self.clips.append(clip)
            self._sequence_changed = True
        self.active_clip_id = clip.id
        clip.info = self.info
        self._stash_clip()
        self._refresh_sequence()

    def _selected_clip_ids(self):
        return {item.data(Qt.ItemDataRole.UserRole) for item in self.clip_list.selectedItems()}

    def _refresh_sequence(self, selected=None):
        if not hasattr(self, "clip_list"):
            return
        selected = self._selected_clip_ids() if selected is None else selected
        if hasattr(self, "reference_sidebar") and not getattr(self, '_three_column_workspace', False):
            self.clip_list.setFixedHeight(68 * max(1, min(3, len(self.clips))) + 4)
        previous = {self.clip_list.item(i).data(Qt.ItemDataRole.UserRole):
                    self.clip_list.item(i).data(Qt.ItemDataRole.UserRole + 1) or {}
                    for i in range(self.clip_list.count())}
        current = self.clip_list.currentItem()
        current_id = current.data(Qt.ItemDataRole.UserRole) if current else self.active_clip_id
        self._thumbnail_list_refreshing = True
        blocker = QSignalBlocker(self.clip_list)
        scroll_position = self.clip_list.verticalScrollBar().value()
        self._thumbnail_items = {}
        self.clip_list.clear()
        for index, clip in enumerate(self.clips, 1):
            prefix = "当前 · " if clip.id == self.active_clip_id else ""
            item = QListWidgetItem(f"{prefix}{index:02d}  {Path(clip.info.path).name}\n{clip.summary}")
            item.setData(Qt.ItemDataRole.UserRole, clip.id)
            key = self._thumbnails.key(clip)
            old = previous.get(clip.id, {})
            keep = old.get("thumbnail_key") == key
            data = {"name": Path(clip.info.path).name, "state": clip.summary,
                    'width':clip.info.width, 'height':clip.info.height, 'duration':clip.info.duration,
                    "order": index, "active": clip.id == self.active_clip_id,
                    "thumbnail_key": key, "thumbnail": old.get("thumbnail") if keep else None,
                    "thumbnail_state": old.get("thumbnail_state", "loading") if keep else "loading",
                    "thumbnail_error": old.get("thumbnail_error", "") if keep else ""}
            item.setData(Qt.ItemDataRole.UserRole + 1, data)
            item.setToolTip(clip.info.path + "\n" + clip.summary + ("\n预览图："+data["thumbnail_error"] if data["thumbnail_error"] else ""))
            self.clip_list.addItem(item)
            self._thumbnail_items[clip.id] = item
            item.setSelected(clip.id in selected)
            if clip.id == current_id:
                self.clip_list.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.clip_list.verticalScrollBar().setValue(scroll_position)
        blocker.unblock()
        self._thumbnail_list_refreshing = False
        self._thumbnails.sync(self.clips)
        self._refresh_visible_thumbnails()

        if hasattr(self, '_ad_filter_mode'):
            from .ad_workspace import _filter_clips
            _filter_clips(self, self._ad_filter_mode)

    def _refresh_visible_thumbnails(self):
        if (self._thumbnail_list_refreshing or not hasattr(self, "clip_list") or
                self.busy or self.timer.isActive() or not self.clip_list.isVisible()):
            return
        viewport = self.clip_list.viewport().rect()
        ids = {self.clip_list.item(i).data(Qt.ItemDataRole.UserRole)
               for i in range(self.clip_list.count())
               if not self.clip_list.item(i).isHidden() and
               self.clip_list.visualItemRect(self.clip_list.item(i)).intersects(viewport)}
        self._thumbnails.recheck([clip for clip in self.clips if clip.id in ids])

    def _thumbnail_ready(self, clip_id, key, jpeg, error):
        clip = self._clip(clip_id)
        if clip is None or self._thumbnails.key(clip) != key:return
        item = self._thumbnail_items.get(clip_id)
        if item is None:return
        data = item.data(Qt.ItemDataRole.UserRole + 1) or {}
        if data.get("thumbnail_key") != key:return
        pixmap = QPixmap()
        if not error and not pixmap.loadFromData(jpeg, "JPG"):error = "缩略图解码失败"
        data.update(thumbnail=None if error else pixmap, thumbnail_state="failed" if error else "ready",
                    thumbnail_error=error)
        item.setData(Qt.ItemDataRole.UserRole + 1, data)
        item.setToolTip(clip.info.path+"\n"+clip.summary+("\n预览图："+error if error else ""))

    def _sync_sequence_ui(self):
        idle = not self.busy
        self._thumbnails.set_paused(not idle or self.timer.isActive())
        self.clip_list.setEnabled(bool(self.clips))
        for control in (self.select_all_clips, self.clip_up, self.clip_down,
                        self.clip_remove, self.open_clip_button, self.queue_mode):
            control.setEnabled(idle and bool(self.clips))
        self.queue_button.setEnabled(idle and bool(self.clips))
        self.open_sequence_action.setEnabled(idle)
        self.recover_sequence_action.setEnabled(idle)
        self.save_sequence_action.setEnabled(idle and bool(self.clips))
        self.save_button.setText("保存序列" if len(self.clips) > 1 or self.sequence_path else "保存工程")
        if idle:
            self._stash_clip()
            self._refresh_sequence()

    def _open_selected_clip(self):
        selected = self._selected_clip_ids()
        if selected:
            self._activate_clip(next(clip.id for clip in self.clips if clip.id in selected))

    def _activate_clip(self, clip_id):
        if self.busy:
            return
        clip = self._clip(clip_id)
        if clip is None or clip.id == self.active_clip_id:
            return
        from .logging import get_logger
        get_logger('import').info('准备打开素材 clip=%s file=%s',clip.id,Path(clip.info.path).name)
        if not clip.info.proxy or not Path(clip.info.proxy).is_file():
            # Only the chosen clip needs an exact CFR proxy. The rest of the
            # sequence remains metadata-only until opened, including on restore.
            from .media import import_video
            self._stash_clip()
            self._stop_play()
            import_feedback = getattr(self, '_import_feedback', None)
            def prepared(info):
                current = self._clip(clip_id)
                if current is None:
                    return
                full_range = current.start == 0 and current.end == current.info.frames - 1
                if info.fingerprint != current.info.fingerprint:
                    current.anchors, current.track, current.retrack_previous = {}, None, None
                    current.tracking_resume = None
                    current.precision_info = None
                    current.timebar_state = {}
                    current.section_tracking, current.tracking_sections = False, []
                    current.start, current.frame = 0, 0
                    current.dirty = True
                    self._paused_by_clip.pop(clip_id, None)
                    full_range = True
                current.info = info
                current.start = min(current.start, info.frames - 1)
                current.end = info.frames - 1 if full_range else min(current.end, info.frames - 1)
                current.frame = min(current.frame, info.frames - 1)
                current.status = ''
                self._activate_after_job = clip_id
                if import_feedback:self._import_feedback = import_feedback
                self._refresh_sequence()
            self._job(import_video, (clip.info.path,), prepared,
                      '素材已添加 · 正在为当前素材准备流畅预览（其它素材按需准备）…')
            return
        self._stash_clip()
        self._stop_play()
        self._switching_clip = True
        try:
            self._pending_project, self._pending_demo = None, False
            self._import_done(clip.info)
            self._ready = False
            self._restore_settings(clip.settings)
            self.start_spin.setValue(clip.start + 1)
            self.end_spin.setValue(clip.end + 1)
            self._ready = True
            self.anchors, self.track = dict(clip.anchors), clip.track
            self.section_tracking = clip.section_tracking
            self.tracking_sections = [dict(value) for value in clip.tracking_sections]
            self._retrack_previous = clip.retrack_previous
            self.project_path, self.dirty = clip.project_path, clip.dirty
            self.precision_info = clip.precision_info
            self.active_clip_id = clip.id
            self.strip.load_state(clip.timebar_state, clip.info.frames, clip.info.fps, clip.start, clip.end)
            self._prepare_preview_pending = False
            self._refresh_anchors()
            self._refresh_issues()
            self._rebuild_plan()
            self._seek(clip.frame)
            if self.track is not None:
                self._tracking_done(self.track)
                self.dirty = clip.dirty
            else:
                self.performance_label.setText("尚未跟踪当前素材")
            self._update_strip()
            self.track_button.setText("重新跟踪" if self.track is not None else "开始跟踪")
            self.status_label.setText(f"已打开素材 · {Path(clip.info.path).name} · {clip.summary}")
            get_logger('import').info('素材画面已打开 clip=%s frame=%d',clip.id,self.frame_index)
        except Exception as error:
            self._error(str(error))
        finally:
            self._switching_clip = False
            self._ready = True
        self._sync_ui()
        if self.settings().tracking_mode == "precision":
            self._prepare_precision_preview()

    def _import_paths(self, paths):
        if self.busy:
            return
        self._stash_clip()
        known = {str(Path(clip.info.path).resolve()).casefold() for clip in self.clips}
        unique = []
        for path in paths:
            resolved = str(Path(path).resolve())
            if resolved.casefold() not in known:
                unique.append(resolved)
                known.add(resolved.casefold())
        if not unique:
            self.queue_summary.setText("这些素材已经在序列中，可双击打开。")
            self.status_label.setText(self.queue_summary.text())
            return
        if len(self.clips) + len(unique) > 500:
            self.queue_summary.setText("每个素材序列最多 500 条，请分批处理。")
            return
        self._job(import_many, (unique,), self._imports_done, "正在读取素材信息（代理按需准备）…",
                  keep_partial=True, metadata_only=True)

    def _imports_done(self, results):
        from .logging import get_logger
        get_logger('import').info('接收素材导入结果 count=%d',len(results))
        defaults = replace(self.settings())
        new_ids, failed = set(), []
        for item in results:
            if "error" in item:
                failed.append(Path(item["path"]).name + "：" + item["error"])
                from .logging import get_logger
                get_logger('import').warning('导入失败 %s: %s', item['path'], item['error'])
                continue
            clip = Clip(item["info"], replace(defaults))
            self.clips.append(clip)
            new_ids.add(clip.id)
        self._refresh_sequence(new_ids)
        cancelled = self.worker is not None and self.worker.cancel_event.is_set()
        text = f"{'导入已取消 · ' if cancelled else ''}已添加 {len(new_ids)} 条素材 · 失败 {len(failed)} 条。"
        self.queue_summary.setText(text + "双击素材分别选区。")
        self.queue_summary.setToolTip("\n".join(failed))
        feedback = text + (failed[0][:200] if failed else '可选择目标开始跟踪')
        self._import_feedback = feedback if failed or cancelled else None
        self.status_label.setText(feedback)
        self.status_label.setToolTip("\n".join(failed))
        if new_ids:
            self._sequence_changed = True
            if hasattr(self, '_ad_filter_mode'):
                from .ad_workspace import _filter_clips
                _filter_clips(self, 'all')
        get_logger('import').info('素材列表已更新 added=%d failed=%d total=%d',len(new_ids),len(failed),len(self.clips))
        # Activation must wait until the import worker has fully stopped.
        if self.active_clip_id is None and new_ids and not cancelled:
            self._activate_after_job = next(clip.id for clip in self.clips if clip.id in new_ids)

    def _move_clips(self, direction):
        if self.busy:
            return
        selected = self._selected_clip_ids()
        indices = range(1, len(self.clips)) if direction < 0 else range(len(self.clips) - 2, -1, -1)
        for index in indices:
            other = index + direction
            if self.clips[index].id in selected and self.clips[other].id not in selected:
                self.clips[index], self.clips[other] = self.clips[other], self.clips[index]
        self._sequence_changed = True
        self._refresh_sequence(selected)

    def _remove_clips(self, checked=False, *, clip_ids=None):
        if self.busy:
            return
        selected = self._selected_clip_ids() if clip_ids is None else set(clip_ids)
        selected &= {clip.id for clip in self.clips}
        if not selected:
            return
        self._stash_clip()
        if any(clip.dirty for clip in self.clips if clip.id in selected):
            if QMessageBox.question(self, "移除素材", "选中素材有未保存调整，仍从序列移除？原视频不会删除。",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
        remaining = [clip for clip in self.clips if clip.id not in selected]
        self._paused_queue_ids = [value for value in self._paused_queue_ids if value not in selected]
        for clip_id in selected:
            self._paused_by_clip.pop(clip_id, None)
            if hasattr(self, '_timebar_histories'):self._timebar_histories.pop(clip_id, None)
        if getattr(self, '_timebar_clipboard', (None,))[0] in selected:
            self._timebar_clipboard = (None, [])
        if self.active_clip_id in selected:
            self._paused_tracking = None
        if not remaining:
            self._stop_play()
            self._original_preview.clear()
            if self.reader:
                self.reader.close()
            self.reader = self.info = self.frame = self.track = self.plan = self.precision_info = None
            self._retrack_previous = None
            self._display_info = None
            self._header_value, self._header_state = None, "就绪"
            self.tracking_progress_panel.reset()
            self.anchors, self.clips, self.active_clip_id = {}, [], None
            self.project_path, self.dirty = None, False
            self._prepare_preview_pending = False
            self._ready = False
            for spin in (self.start_spin, self.end_spin):
                spin.setRange(1, 1)
            self.timeline.setRange(0, 0)
            self._ready = True
            self.canvas.set_frames(None)
            self.file_label.setText("导入你的第一段视频")
            self.meta_label.setText("MP4 / MOV / MKV / AVI")
            self._refresh_anchors()
            self._refresh_issues()
            self._update_strip()
            self._sync_ui()
            self.queue_summary.setText("素材已从序列移除，原视频未删除。")
            self.status_label.setText(self.queue_summary.text())
            return
        self.clips = remaining
        self._sequence_changed = True
        if self.active_clip_id in selected:
            self.active_clip_id = None
            self._activate_clip(self.clips[0].id)
        self._refresh_sequence()
        self.status_label.setText(f"已从序列删除 {len(selected)} 条素材，原视频保留。")

    def _start_queue(self):
        if self.busy:
            return
        self._stash_clip()
        selected = self._selected_clip_ids()
        clips = [clip for clip in self.clips if clip.id in selected]
        if not clips:
            self.queue_summary.setText("请先多选素材，或点击「全选」。")
            return
        if not any(clip.ready for clip in clips):
            self.queue_summary.setText("选中素材尚未选区。请双击各素材，在预览中选好目标。")
            return
        from .tracking_pause import can_resume_without_edit
        blocked = next((clip for clip in clips if clip.id in self._paused_by_clip
                        and not can_resume_without_edit(self._paused_by_clip[clip.id])), None)
        if blocked is not None:
            self._activate_clip(blocked.id)
            self.queue_summary.setText("请先修正已暂停素材，再点击继续跟踪")
            return
        if self.active_clip_id not in {clip.id for clip in clips if clip.ready}:
            self._activate_clip(next(clip.id for clip in clips if clip.ready))
            if self.busy:
                self._queue_after_prepare = selected
                return
        if hasattr(self, '_timebar_histories'):
            for clip in clips:self._timebar_histories.pop(clip.id, None)
        self._begin_live_preview()
        self._mark_workflow(1)
        self._live_queue_follow = True
        self._live_follow_clip_id = next(clip.id for clip in clips if clip.ready)
        self._live_completed_ids = set()
        self._stop_play()
        self._reset_stages()
        self._tracking_active, self._job_kind = True, "queue"
        self._job_outcome = None
        self._manual_pause_requested = False
        self._cancel_progress = None
        self._header_value, self._header_state = 0., "批量总体"
        self._queue_order = [clip.id for clip in clips]
        self._queue_running_ids, self._queue_progress_by_id = set(), {}
        self._queue_panel_clip_id = self._live_follow_clip_id
        first = self._clip(self._queue_panel_clip_id)
        self.tracking_progress_panel.begin(Path(first.info.path).name,
            self._queue_order.index(first.id) + 1, len(clips), batch=True, follow_task_phases=True)
        for clip in clips:
            clip.status = "等待跟踪" if clip.ready else "未选目标，已跳过"
        self.busy = True
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self._sync_ui()
        self._refresh_sequence(selected)
        snapshots = [replace(clip, settings=replace(clip.settings), anchors=dict(clip.anchors)) for clip in clips]
        self.worker = worker = QueueJob(snapshots, self.queue_mode.currentData(), str(cache_root()), self,
                                        live_preview=self.live_preview_check.isChecked(),
                                        pause_on_lost=self.auto_pause_check.isChecked())
        self._sync_ui()
        worker.frame_event.connect(self._on_queue_tracking_frame)
        worker.progress.connect(self._on_progress)
        worker.item_progress.connect(self._queue_item_progress)
        worker.item_result.connect(self._queue_item_done)
        worker.result.connect(self._queue_done)
        worker.failed.connect(self._queue_failed)
        worker.finished.connect(self._job_finished)
        worker.start()

    def _queue_item_done(self, clip_id, track, error):
        if isinstance(track, TrackingPaused):
            clip = self._clip(clip_id)
            if clip is None:
                return
            context = self._pause_context_for(track)
            from .logging import get_logger
            from .tracking_pause import computed_mask
            get_logger('queue').info('批量跟踪暂停 素材=%s 原因=%s 帧=%d 方向=%d 已测量=%d/%d',
                Path(clip.info.path).name, track.reason, track.frame, track.direction,
                int(computed_mask(track.result).sum()), track.result.count)
            self._paused_by_clip[clip_id] = context
            clip.track = clip.retrack_previous = track.result
            clip.tracking_resume = track.resume
            clip.frame, clip.dirty, clip.status = track.frame, True, "已暂停 · 等待修正"
            self._queue_running_ids.discard(clip_id)
            self._live_completed_ids = getattr(self, "_live_completed_ids", set()) | {clip_id}
            if self._queue_panel_clip_id == clip_id:
                self._queue_panel_clip_id = None
            if getattr(self, "_live_follow_clip_id", None) == clip_id:
                self._live_follow_clip_id = None
            if clip_id == self.active_clip_id:
                self._tracking_paused(track)
            self._refresh_sequence()
            return
        if error == "正在跟踪":
            self._queue_running_ids.add(clip_id)
            if self._queue_panel_clip_id is None:
                self._show_queue_task(clip_id)
        else:
            self._queue_running_ids.discard(clip_id)
            if self._queue_panel_clip_id == clip_id:
                if track is not None:
                    self.tracking_progress_panel.update_task_progress(1., "[缓存:1.000] 当前素材结果已写入")
                self._queue_panel_clip_id = None
                next_id = next((value for value in self._queue_order if value in self._queue_running_ids), None)
                if next_id:
                    self._show_queue_task(next_id)
        if error != "正在跟踪":
            self._live_completed_ids = getattr(self, "_live_completed_ids", set()) | {clip_id}
            if getattr(self, "_live_follow_clip_id", None) == clip_id:
                self._live_follow_clip_id = None
        clip = self._clip(clip_id)
        if clip is None:
            return
        clip.status = error
        if track is not None:
            self._paused_by_clip.pop(clip_id, None)
            clip.tracking_resume = None
            clip.track, clip.dirty, clip.status = track, True, ""
            clip.retrack_previous = None
            if clip.id == self.active_clip_id:
                self._tracking_done(track)
        self._refresh_sequence()

    def _show_queue_task(self, clip_id):
        clip = self._clip(clip_id)
        if clip is None:
            return
        self._queue_panel_clip_id = clip_id
        if getattr(self, "_live_queue_follow", False):
            self._live_follow_clip_id = clip_id
        self.tracking_progress_panel.set_task(Path(clip.info.path).name,
            self._queue_order.index(clip_id) + 1, len(self._queue_order))
        if clip_id in self._queue_progress_by_id:
            self.tracking_progress_panel.update_task_progress(*self._queue_progress_by_id[clip_id])

    def _queue_item_progress(self, clip_id, value, message):
        if clip_id in getattr(self, "_live_completed_ids", set()):
            return
        self._queue_progress_by_id[clip_id] = (value, message)
        if self._queue_panel_clip_id is None:
            self._show_queue_task(clip_id)
        if self._queue_panel_clip_id == clip_id:
            self.tracking_progress_panel.update_task_progress(value, message)

    def _queue_done(self, outcomes):
        completed = sum(item["track"] is not None for item in outcomes.values())
        review = sum(item["track"] is not None and bool(item["track"].lost_frames) for item in outcomes.values())
        failed = len(outcomes) - completed
        paused_ids = [clip_id for clip_id, item in outcomes.items() if item.get("pause") is not None]
        if paused_ids:
            self._paused_queue_ids = [clip.id for clip in self.clips if clip.id in outcomes and
                                      (clip.id in paused_ids or outcomes[clip.id].get("error") == "等待继续")]
            primary = next((clip_id for clip_id in paused_ids if
                            self._paused_by_clip[clip_id]["reason"] == "lost"), paused_ids[0])
            clip = self._clip(primary)
            self._job_outcome, self._header_state = "paused", "已暂停"
            if self.active_clip_id == primary:
                self._paused_tracking = self._paused_by_clip[primary]
            else:
                self._activate_after_job = primary
            overall = self._header_value or 0.
            self._paused_batch_progress = overall
            self.tracking_progress_panel.begin(Path(clip.info.path).name,
                self._queue_order.index(primary) + 1, len(self._queue_order), batch=True, follow_task_phases=True)
            self.tracking_progress_panel.update_progress(overall, "队列已暂停")
            if primary in self._queue_progress_by_id:
                self.tracking_progress_panel.update_task_progress(*self._queue_progress_by_id[primary])
            self.queue_summary.setText(f"队列已暂停 · 已完成 {completed} 条，剩余 {len(self._paused_queue_ids)} 条等待修正后继续")
            self.tracking_progress_panel.finish("paused", self.queue_summary.text())
            self.status_label.setText(self.queue_summary.text())
            self._sequence_changed = True
            return
        cancelled = self.worker is not None and self.worker.cancel_event.is_set()
        errors = any(item.get("error", "").startswith("失败") for item in outcomes.values())
        self._job_outcome = "cancelled" if cancelled else "failed" if errors else "completed"
        self._header_state = "已取消" if cancelled else "部分失败" if errors else "队列结束"
        self._sequence_changed = True
        self.queue_summary.setText(f"本轮完成 {completed} 条 · 其中待修正 {review} 条 · 跳过 / 取消 / 失败 {failed} 条。双击素材检查结果。")
        self.tracking_progress_panel.finish(self._job_outcome, self.queue_summary.text())
        self.status_label.setText("素材队列已结束 · 已完成结果保留，跟丢帧可逐条补关键帧")

    def _queue_failed(self, message):
        for clip in self.clips:
            if clip.status in {"等待跟踪", "正在跟踪"}:
                clip.status = "任务中断，可重试"
        self._refresh_sequence()
        self._error(message)

    def _save_workspace(self):
        return self._save_sequence() if len(self.clips) > 1 or self.sequence_path else self._save_project()

    def _save_sequence(self):
        if self.busy or not self.clips:
            return False

        self._stash_clip()
        path, _ = QFileDialog.getSaveFileName(self, "保存素材序列", self.sequence_path or "素材序列.locksequence",
                                             "番茄素材序列 (*.locksequence)")
        if not path:
            return False
        if not path.lower().endswith(".locksequence"):
            path += ".locksequence"
        try:
            save_sequence(path, self.clips)
            self.sequence_path, self._sequence_changed = path, False
            for clip in self.clips:
                clip.dirty = False
            self.dirty = False
            self.status_label.setText("素材序列已保存 · 顺序、选区、参数和跟踪结果可继续使用")
            self._sync_ui()
            return True
        except Exception as error:
            self._error(str(error))
            return False

    def _open_sequence(self):
        if self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开素材序列", "", "番茄素材序列 (*.locksequence)")
        self._load_sequence_file(path)

    def _recover_checkpoint_task(self):
        if self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, "恢复续跑任务 · 选择对应视频名称的记录",
                                             str(cache_root() / "tracking-checkpoints"), "续跑任务 (*.locksequence)")
        self._load_sequence_file(path, recovery=True)

    def _load_sequence_file(self, path, recovery=False):
        if not path or not self._confirm_discard():
            return
        try:
            clips = read_sequence(path)
            self._pending_sequence_path = None if recovery else path
            self._job(restore_sequence, (clips,), self._sequence_restored, "正在恢复素材序列…", keep_partial=True)
        except Exception as error:
            self._error(str(error))

    def _sequence_restored(self, clips):
        if not clips:
            return
        first_readable = next((clip for clip in clips if not clip.status.startswith("读取失败")), None)
        if first_readable is None:
            raise ValueError("所有原片都无法读取，请恢复原片路径后重试。当前工作区已保留。")
        self._timebar_histories = {}
        self._timebar_clipboard = (None, [])
        self.clips, self.active_clip_id = clips, None
        self.sequence_path = self._pending_sequence_path
        self._sequence_changed = self.sequence_path is None
        if self.sequence_path is None:
            for clip in clips:
                clip.dirty = True
        self._refresh_sequence({clip.id for clip in clips})
        self._activate_after_job = first_readable.id
