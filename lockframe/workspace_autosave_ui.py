from copy import deepcopy
from pathlib import Path
import time,uuid
from PySide6.QtCore import QObject,QTimer,Signal
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QListWidget,QPushButton,QLabel
from .workspace_data import data_root
from .workspace_autosave import freeze_clips,write_workspace,list_workspaces,remove_workspace,mark_workspace
from .workers import Job
from .logging import get_logger


class WorkspaceAutosave(QObject):
    changed=Signal()

    def __init__(self,host,root=None):
        super().__init__(host);self.host=host
        self.root=Path(root) if root else data_root()/'workspace-recovery'
        self.folder=self.root/uuid.uuid4().hex;self.enabled=False;self.worker=None;self.pending=False
        self.message='自动保存待命';self.last_error=''
        self.timer=QTimer(self);self.timer.setInterval(15000);self.timer.timeout.connect(self.request)

    def start(self):self.enabled=True;self.timer.start()

    def request(self,force=False):
        if not self.enabled or not self.host.clips or self.host.busy:return False
        if not force and self.host.timer.isActive():return False
        if self.worker:self.pending=True;return False
        try:
            self.host._stash_clip();clips=freeze_clips(self.host.clips);view=self.host._workspace_view_snapshot()
        except Exception as error:self.failed(str(error));return False
        self.worker=Job(write_workspace,clips,view,str(self.folder),parent=self,keep_partial=True)
        self.worker.result.connect(self.done);self.worker.failed.connect(self.failed);self.worker.aborted.connect(self.aborted);self.worker.finished.connect(self.finished)
        self.message='正在自动保存…';self.changed.emit();self.worker.start();return True

    def done(self,value):
        self.message='已自动保存 '+time.strftime('%H:%M:%S',time.localtime(value['time']));self.last_error='';self.changed.emit()

    def failed(self,message):
        self.last_error=str(message);self.message='自动保存失败（上次记录保留）';self.changed.emit()
        get_logger('autosave').error('工作区自动保存失败 %s',message)

    def aborted(self):
        self.message='本次自动保存已取消（上次记录保留）';self.changed.emit()

    def finished(self):
        worker=self.worker;self.worker=None
        if worker:worker.deleteLater()
        if self.pending:self.pending=False;self.request(force=True)
        self.changed.emit()

    def close(self):
        self.timer.stop()
        if self.worker:return False
        try:mark_workspace(self.folder,clean=True)
        except Exception as error:get_logger('autosave').warning('无法标记正常退出 %s',error)
        return True


class WorkspaceRecoveryDialog(QDialog):
    restore=Signal(str)

    def __init__(self,service,parent=None):
        super().__init__(parent);self.service=service;self.entries=[]
        self.setWindowTitle('自动保存与工作区恢复');self.resize(680,370);self.setModal(False)
        layout=QVBoxLayout(self);layout.addWidget(QLabel('自动保存保留素材、选区、跟踪结果、构图、曲线和工作位置；不会覆盖手动工程。'))
        self.list=QListWidget();layout.addWidget(self.list);row=QHBoxLayout()
        for title,callback in (('恢复选中工作区',self.restore_selected),('忽略启动提醒',self.ignore),('清理选中旧记录',self.remove),('打开文件夹',self.open_folder)):
            button=QPushButton(title);button.clicked.connect(callback);row.addWidget(button)
        layout.addLayout(row);self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note);self.refresh()

    def refresh(self):
        self.entries=list_workspaces(self.service.root);self.list.clear()
        for value in self.entries:
            latest=value['latest'];state='正常退出' if value.get('clean') else '已恢复/忽略' if value.get('acknowledged') else '未正常结束'
            try:stamp=time.strftime('%m-%d %H:%M:%S',time.localtime(latest['time']))
            except (ValueError,OverflowError,OSError):stamp='时间未知'
            self.list.addItem(stamp+f' · {latest["count"]} 条素材 · {state}\n'+ '、'.join(latest['names']))
        if self.entries:self.list.setCurrentRow(0)
        self.note.setText(self.service.message+' · '+str(self.service.root)+(('\n'+self.service.last_error) if self.service.last_error else ''))

    def current(self):
        index=self.list.currentRow()
        return self.entries[index] if 0<=index<len(self.entries) else None

    def restore_selected(self):
        value=self.current()
        if value:self.restore.emit(value['folder']);self.accept()

    def ignore(self):
        value=self.current()
        if value:
            try:mark_workspace(value['folder'],acknowledged=True);self.refresh()
            except Exception as error:self.note.setText(str(error))

    def remove(self):
        value=self.current()
        if value:
            try:remove_workspace(self.service.root,value['folder'],self.service.folder);self.refresh()
            except Exception as error:self.note.setText(str(error))

    def open_folder(self):
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        self.service.root.mkdir(parents=True,exist_ok=True);QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.service.root)))
