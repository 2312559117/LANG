"""Modeless diagnostic export, independent of the tracking/export job."""
from pathlib import Path
from datetime import datetime
from PySide6.QtCore import QUrl,Qt,Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFileDialog,QProgressBar

from . import __version__
from .diagnostics_bundle import capture_state,export_diagnostics
from .logging import log_folder,SESSION_ID
from .workers import Job


class DiagnosticsDialog(QDialog):
    export_finished=Signal()
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('日志与诊断');self.resize(620,245)
        self.worker=None;self.running=False;self.last_bundle=None;self._close_requested=False
        layout=QVBoxLayout(self)
        text=QLabel('出现问题后，导出诊断包并把 ZIP 发给开发者。\n包含最近日志、版本、基础环境和任务状态；不添加视频或工程。')
        text.setWordWrap(True);layout.addWidget(text)
        self.path_label=QLabel(str(log_folder()));self.path_label.setWordWrap(True);self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        self.status=QLabel('本地日志已自动启用 · 最近记录自动轮换');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.progress=QProgressBar();self.progress.setRange(0,1000);self.progress.hide();layout.addWidget(self.progress)
        row=QHBoxLayout();layout.addLayout(row)
        self.export_button=QPushButton('导出诊断包…');self.export_button.clicked.connect(self.choose_export);row.addWidget(self.export_button)
        logs=QPushButton('打开日志文件夹');logs.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_folder()))));row.addWidget(logs)
        self.folder_button=QPushButton('打开诊断包目录');self.folder_button.setEnabled(False);self.folder_button.clicked.connect(self.open_folder);row.addWidget(self.folder_button)
        self.cancel_button=QPushButton('取消打包');self.cancel_button.hide();self.cancel_button.clicked.connect(self.cancel);row.addWidget(self.cancel_button)

    def choose_export(self):
        folder=log_folder().parent/'diagnostics'
        try:folder.mkdir(parents=True,exist_ok=True)
        except OSError as error:
            self.status.setText('无法创建诊断目录：'+str(error));return
        name=f'番茄跟踪_诊断_v{__version__}_{datetime.now():%Y%m%d-%H%M%S}_{SESSION_ID}.zip'
        path,_=QFileDialog.getSaveFileName(self,'导出诊断包',str(folder/name),'ZIP 诊断包 (*.zip)',options=QFileDialog.Option.DontConfirmOverwrite)
        if path:self.begin_export(path if path.lower().endswith('.zip') else path+'.zip')

    def begin_export(self,path):
        if self.running:return
        state,protected=capture_state(self.parent())
        self.running=True;self.export_button.setEnabled(False);self.progress.setValue(0);self.progress.show();self.cancel_button.show();self.cancel_button.setEnabled(True)
        self.status.setText('正在打包，可以继续使用素材与跟踪功能…')
        worker=Job(export_diagnostics,path,state,parent=self,protected_paths=protected,keep_partial=True)
        self.worker=worker
        worker.progress.connect(lambda value,text:self.progress.setValue(round(value*1000)))
        worker.result.connect(self._done)
        worker.failed.connect(lambda message:self.status.setText('诊断包未保存：'+message))
        worker.aborted.connect(lambda:self.status.setText('打包已取消，原日志保留'))
        worker.finished.connect(self._finished);worker.start()

    def _done(self,path):
        self.last_bundle=path;self.folder_button.setEnabled(True)
        self.status.setText('已保存，可将此 ZIP 发给开发者：\n'+path)

    def _finished(self):
        worker,self.worker=self.worker,None;self.running=False
        self.export_button.setEnabled(True);self.cancel_button.hide();self.progress.hide()
        if worker:worker.deleteLater()
        self.export_finished.emit()
        if self._close_requested:self.close()

    def cancel(self):
        if self.worker:self.worker.cancel();self.cancel_button.setEnabled(False)

    def open_folder(self):
        if self.last_bundle:QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_bundle).parent)))

    def closeEvent(self,event):
        if self.running:
            self._close_requested=True;self.cancel();event.ignore()
        else:event.accept()


def open_diagnostics(window):
    dialog=getattr(window,'diagnostics_dialog',None)
    if dialog is None:
        dialog=DiagnosticsDialog(window);window.diagnostics_dialog=dialog
    dialog.show();dialog.raise_();dialog.activateWindow()
    return dialog
