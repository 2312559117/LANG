"""One authoritative clip footprint, integrated with the existing edit history."""
from dataclasses import replace

from .regions import BOX_CORNERS, TrackingRegion, shape_data, shape_matches, anchors_match


class RegionMixin:
    def _region_snapshot(self):
        return dict(shape=self._region_shape, points=self._polygon_points,
                    anchors=dict(self.anchors), track=self.track, previous=self._retrack_previous)

    def _region_stale(self):
        return bool(self.track is not None and (
            not shape_matches(self.track, self.settings()) or
            not anchors_match(self.track, self.anchors, self.start_spin.value()-1, self.end_spin.value()-1)))

    def _region_sync(self):
        if not hasattr(self, 'region_toolbar'):
            return
        self.canvas.selection_shape = self._region_shape
        self.canvas.region_points = self._polygon_points
        ready = self.info is not None and self.canvas.box is not None
        self.region_toolbar.set_state(self._region_shape, ready, self.busy,
            can_delete=self.canvas.can_delete_vertex, mode=self.canvas.region_edit_mode,
            can_add=len(self._polygon_points)<64)
        self.shape_combo.setEnabled(bool(ready and not self.busy))
        self._region_header_layout()

    def _region_header_layout(self):
        if not hasattr(self, 'preview_head'):
            return
        width = self.preview_head.width()
        self.region_toolbar.set_available_width(width)
        self.sync_label.setText('同步检查' if width < 760 else '原片 / 效果预览 · 同步检查')
        self.monitor_filename.setMaximumWidth(130 if width < 760 else 200)

    def _region_restore(self, snapshot):
        self._region_shape, self._polygon_points = snapshot['shape'], snapshot['points']
        self.anchors = dict(snapshot['anchors'])
        self.track, self._retrack_previous = snapshot['track'], snapshot['previous']
        blocked = self.shape_combo.blockSignals(True)
        self.shape_combo.setCurrentIndex(self.shape_combo.findData(self._region_shape))
        self.shape_combo.blockSignals(blocked)
        self.canvas.region_edit_mode = 'select'
        self.canvas.selected_vertex = -1
        self.canvas.selection_shape, self.canvas.region_points = self._region_shape, self._polygon_points
        self._tb_refresh_anchor_edit()

    def _region_commit(self, shape, points=(), box=None, *, label='修改跟踪区域'):
        if not self._ready or self.busy or not self.info:
            self._region_sync()
            return False
        try:
            shape, points = shape_data(shape, points)
            if box is not None:
                region = TrackingRegion.from_box(box, shape, points)
                box = region.box
                if self.frame is not None and min(box[2]*self.frame.shape[1], box[3]*self.frame.shape[0]) < 12:
                    raise ValueError('目标框太小，请包含清晰边缘或纹理')
                if not self.start_spin.value()-1 <= self.frame_index <= self.end_spin.value()-1:
                    raise ValueError('请先将播放位置移到处理区间内')
            elif self.canvas.box is None:
                raise ValueError('请先在原片选择跟踪区域')
        except (TypeError, ValueError) as error:
            self.status_label.setText(str(error))
            blocked=self.shape_combo.blockSignals(True)
            self.shape_combo.setCurrentIndex(self.shape_combo.findData(self._region_shape))
            self.shape_combo.blockSignals(blocked)
            self._region_sync()
            return False
        self._stop_play()
        before=self._region_snapshot()
        if box is not None:
            self.anchors[self.frame_index]=box
        changed=(shape!=before['shape'] or points!=before['points'] or self.anchors!=before['anchors'])
        if not changed:
            if box is not None and self._paused_tracking and self.frame_index==self._paused_tracking['frame']:
                self._paused_tracking['corrected']=True
            self._sync_ui()
            return False
        # Legacy results lack provenance. Freeze their original anchors before editing,
        # so save/reopen and batch export cannot mistake them for the new geometry.
        if self.track is not None:
            if self.track.anchors is None:
                self.track=replace(self.track, anchors={f:box for f,box in before['anchors'].items()
                    if self.track.start <= f <= self.track.end})
            self._retrack_previous=self.track
        self._region_shape,self._polygon_points=shape,points
        blocked=self.shape_combo.blockSignals(True)
        self.shape_combo.setCurrentIndex(self.shape_combo.findData(shape))
        self.shape_combo.blockSignals(blocked)
        self.canvas.region_edit_mode='select'
        self.canvas.selected_vertex=-1
        self.canvas.selection_shape,self.canvas.region_points=shape,points
        self.composition_edit.setChecked(False)
        self.pick_mode.setCurrentIndex(self.pick_mode.findData('adjust'))
        self._tb_history().push(dict(kind='region', label=label, before=before,
                                    after=self._region_snapshot(), bytes=1024+len(points)*32+len(self.anchors)*192))
        self._tb_refresh_anchor_edit()
        if self._paused_tracking and (box is not None and self.frame_index==self._paused_tracking['frame']):
            self._paused_tracking['corrected']=True
            self._sync_ui()
        self.status_label.setText('区域已修改；旧结果已过期，请重新跟踪后导出' if self._region_stale() else '跟踪区域已更新')
        return True

    def _region_choose_shape(self, shape):
        points=self._polygon_points if self._region_shape=='polygon' and shape=='polygon' else BOX_CORNERS if shape=='polygon' else ()
        self._region_commit(shape,points,label='切换跟踪区域形状')

    def _region_proposal(self, data):
        if not isinstance(data,dict):
            return
        self._region_commit(data.get('shape'),data.get('points',()),data.get('box'))

    def _region_action(self, action):
        if self.busy or not self.info or self.canvas.box is None:
            return
        self._stop_play()
        if action=='delete':
            self.canvas.delete_region_vertex()
        elif action=='reset':
            self._region_commit('rectangle',(),label='重置区域为矩形')
        elif action in ('edit','add'):
            self.composition_edit.setChecked(False)
            self.canvas.region_edit_mode='add' if action=='add' and self._region_shape=='polygon' else 'select'
            self.pick_mode.setCurrentIndex(self.pick_mode.findData('adjust'))
        self._region_sync()
        self.canvas.update()
