"""Small host adapters for local recovery, presets and centralized review."""
from copy import deepcopy
from pathlib import Path
import math
from PySide6.QtCore import QTimer,QPointF
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel
from .workspace_autosave_ui import WorkspaceAutosave,WorkspaceRecoveryDialog
from .workspace_autosave import restore_saved_workspace,list_workspaces,mark_workspace
from .logging import get_logger


class WorkspaceFeaturesMixin:
    def _install_workspace_features(self):
        self._workspace_services_started=False;self._workspace_close_saved=False;self._workspace_close_approved=False
        self._recovery_dialog=None;self._review_dialog=None;self._preset_dialog=None;self._workspace_restore_pending=None
        self._autosave=WorkspaceAutosave(self);self._autosave.changed.connect(self._autosave_status)
        self.autosave_status_label=QLabel();self.autosave_status_label.hide();self.footer_row.insertWidget(1,self.autosave_status_label)
        file_menu=self.ad_menu.actions()[0].menu()
        self.autosave_action=QAction('自动保存工作区（每15秒）',self);self.autosave_action.setCheckable(True)
        self.autosave_action.setChecked(str(self.preferences.value('workspace_autosave',True)).lower() in ('1','true'))
        self.autosave_action.toggled.connect(self._set_workspace_autosave);file_menu.addAction(self.autosave_action)
        self.recovery_action=file_menu.addAction('自动保存与工作区恢复…',self._open_workspace_recovery)
        self.review_action=QAction('集中复核工作台…',self);self.review_action.setShortcut('F8');self.review_action.triggered.connect(self._open_review_workbench)
        self.addAction(self.review_action)
        menu=next(a.menu() for a in self.ad_menu.actions() if a.text()=='跟踪');menu.addAction(self.review_action)

    def _start_workspace_services(self,root=None):
        if root is not None:
            self._autosave.root=Path(root);self._autosave.folder=Path(root)/self._autosave.folder.name
        self._workspace_services_started=True
        self._set_workspace_autosave(self.autosave_action.isChecked())

    def _set_workspace_autosave(self,enabled):
        self.preferences.setValue('workspace_autosave',bool(enabled))
        if not self._workspace_services_started:return
        if enabled:self._autosave.start()
        else:
            self._autosave.enabled=False;self._autosave.timer.stop();self._autosave.pending=False
            if self._autosave.worker:self._autosave.worker.cancel()
        self._autosave_status()

    def _autosave_status(self):
        if not self._workspace_services_started:return
        self.autosave_status_label.setText(self._autosave.message if self._autosave.enabled else '自动保存已关闭')
        self.autosave_status_label.setToolTip(str(self._autosave.folder)+(('\n'+self._autosave.last_error) if self._autosave.last_error else ''))
        self.autosave_status_label.show()

    def _workspace_view_snapshot(self):
        ids=[clip.id for clip in self.clips]
        view=dict(active_index=ids.index(self.active_clip_id) if self.active_clip_id in ids else 0,
            selected=[i for i,clip in enumerate(self.clips) if clip.id in self._selected_clip_ids()],
            sequence_path=self.sequence_path,clock=self._retime_clock,output_step=self._retime_output_step,
            source_zoom=self.canvas.zoom_factor,output_zoom=self.canvas.output_zoom,
            source_center=[self.canvas._view_center.x(),self.canvas._view_center.y()],
            export_folder=str(self.preferences.value('export_folder','')),
            paused_queue=[i for i,clip in enumerate(self.clips) if clip.id in self._paused_queue_ids])
        editor=self._retime_editor
        expected=(self.active_clip_id,self.info.path,self.info.fingerprint,*self._retime_range()) if self.info else None
        if editor is not None and editor.isVisible() and getattr(editor,'_workspace_context',None)==expected:
            view['editor']=dict(config=deepcopy(editor.draft),context=[self.info.path,self.info.fingerprint,*self._retime_range()])
        return view

    def _open_workspace_recovery(self):
        if self._recovery_dialog is not None:self._recovery_dialog.close()
        dialog=self._recovery_dialog=WorkspaceRecoveryDialog(self._autosave,self)
        dialog.restore.connect(self._restore_workspace);dialog.show()

    def _offer_workspace_recovery(self):
        if self.busy or self.clips:return
        if any(not value.get('clean') and not value.get('acknowledged') for value in list_workspaces(self._autosave.root)):
            self._open_workspace_recovery()

    def _restore_workspace(self,folder):
        if self.busy or not self._confirm_discard():return
        if self._autosave.worker:
            self.status_label.setText('正在提交自动保存，请稍后再恢复工作区');return
        self._stop_play();self._job(restore_saved_workspace,(folder,),self._workspace_loaded,'正在恢复自动保存的完整工作区…')

    def _workspace_loaded(self,payload):
        clips=payload['clips'];view=payload.get('view',{})
        if not isinstance(view,dict):view={}
        valid=[i for i,clip in enumerate(clips) if Path(clip.info.path).is_file() and not clip.status.startswith('读取失败')]
        if not valid:raise ValueError('所有原片都无法读取，请先恢复原片路径；自动保存记录已保留，当前工作区未替换')
        wanted=view.get('active_index',0);chosen=wanted if type(wanted) is int and wanted in valid else valid[0]
        # Old dialogs bind Apply to the old clip ID. Recovery creates new IDs;
        # never pour a recovered draft into an editor with stale callbacks.
        if self._retime_editor is not None:
            self._retime_editor.close();self._retime_editor.deleteLater();self._retime_editor=None
        if self._preset_dialog is not None:
            self._preset_dialog.close();self._preset_dialog.deleteLater();self._preset_dialog=None
        self._paused_by_clip={};self._paused_tracking=None;self._paused_queue_ids=[]
        self._pending_sequence_path=view.get('sequence_path') if isinstance(view.get('sequence_path'),str) else None
        self._sequence_restored(clips);self._sequence_changed=True
        for clip in clips:clip.dirty=True
        self._activate_after_job=clips[chosen].id
        self._workspace_restore_pending=dict(payload,view=view,chosen=chosen,valid_view=chosen==wanted)

    def _apply_restored_workspace_view(self):
        pending=self._workspace_restore_pending
        if pending is None:return
        if self.busy:QTimer.singleShot(100,self._apply_restored_workspace_view);return
        self._workspace_restore_pending=None;view=pending['view'];clips=pending['clips']
        def indices(key):
            values=view.get(key,[])
            return [i for i in values if type(i) is int and 0<=i<len(clips)] if isinstance(values,list) else []
        self._paused_queue_ids=[clips[i].id for i in indices('paused_queue')]
        self._queue_order=[clip.id for clip in clips]
        self._refresh_sequence({clips[i].id for i in indices('selected')})
        if isinstance(view.get('export_folder'),str):self.preferences.setValue('export_folder',view['export_folder'])
        if pending['valid_view']:
            clock=view.get('clock','source')
            if clock not in ('source','output'):clock='source'
            self.retime_panel.clock.setCurrentIndex(self.retime_panel.clock.findData(clock))
            step=view.get('output_step',0)
            if self._retime_active() and type(step) is int:self._seek_output(step)
            for key,method in (('source_zoom',self.canvas.set_zoom),('output_zoom',self.canvas.set_output_zoom)):
                value=view.get(key,1.)
                if type(value) in (float,int) and math.isfinite(value):method(max(1.,min(8.,value)))
            center=view.get('source_center')
            if isinstance(center,list) and len(center)==2 and all(type(v) in (float,int) and math.isfinite(v) and 0<=v<=1 for v in center):self.canvas._view_center=QPointF(*center)
            draft=view.get('editor')
            if isinstance(draft,dict) and draft.get('context')==[self.info.path,self.info.fingerprint,*self._retime_range()]:
                try:
                    self._open_retime_editor();self._retime_editor._change(draft['config']);self._retime_editor.message.setText('已恢复未应用草稿；点击应用后才改变当前素材')
                except Exception as error:get_logger('autosave').warning('未应用草稿无法恢复 %s',error)
        try:mark_workspace(pending['folder'],acknowledged=True)
        except Exception as error:get_logger('autosave').warning('恢复标记无法保存 %s',error)
        self.dirty=True;self._sync_ui();self.status_label.setText('工作区已恢复 · 自动保存不覆盖手动工程；缺失原片会保留为读取失败素材')
        self._autosave.request(force=True)

    def _open_curve_presets(self):
        if not self.info or self.busy:return
        from .curve_preset_ui import PresetPicker
        identity=(self.active_clip_id,self.info.path,self.info.fingerprint,*self._retime_range())
        dialog=self._preset_dialog=PresetPicker(self._retime_config or None,self.end_spin.value()-self.start_spin.value()+1,self)
        def apply(config):
            actual=(self.active_clip_id,self.info.path,self.info.fingerprint,*self._retime_range()) if self.info else None
            if actual==identity:self._apply_retime_config(config)
            else:self.status_label.setText('素材或范围已变，请重新打开曲线预设')
        dialog.applied.connect(apply);dialog.show()

    def _open_review_workbench(self):
        if self._review_dialog is not None and self._review_dialog.isVisible():self._review_dialog.refresh();self._review_dialog.raise_();return
        from .review_workbench import ReviewWorkbench
        self._review_dialog=ReviewWorkbench(self);self._review_dialog.locate.connect(self._review_locate)
        self._review_dialog.changed.connect(lambda:self._autosave.request(force=True));self._review_dialog.show()

    def _review_locate(self,item):
        if self.busy:self.status_label.setText('请先暂停当前任务，再定位复核区间');return
        if self._clip(item.clip_id) is None:return
        self._stop_play()
        if item.clip_id!=self.active_clip_id:self._activate_clip(item.clip_id)
        def finish():
            if self.busy:QTimer.singleShot(100,finish);return
            if self.active_clip_id!=item.clip_id:return
            self.retime_panel.clock.setCurrentIndex(self.retime_panel.clock.findData('source'))
            self._seek(item.first)
            self.status_label.setText(f'{item.kind} · 第 {item.first+1}–{item.last+1} 帧 · '+item.reason)
        QTimer.singleShot(0,finish)
