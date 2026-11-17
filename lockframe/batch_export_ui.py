"""A modeless export queue; workspace editing does not mutate its snapshots."""
import json
from collections import Counter
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
                               QFileDialog, QMessageBox)

from .batch_export import prepare_exports, run_exports
from .workers import Job
from .logging import log_folder


class ExportQueueDialog(QDialog):
    def __init__(self, tasks, directory, parent):
        super().__init__(parent)
        self.setWindowTitle("批量导出 · 参数已冻结")
        self.resize(900, 460)
        self.directory, self.tasks = directory, tasks
        self.running = True
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("逐条导出；可继续编辑素材。排队任务使用入队时的构图、范围、画幅和声音策略。"))
        self.table = QTableWidget(len(tasks), 4)
        self.table.setHorizontalHeaderLabels(["素材", "状态 / 单条进度", "输出文件", "原因"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, task in enumerate(tasks):
            for col, text in enumerate((task.name, "跳过" if task.reason else "等待", Path(task.destination).name, task.reason)):
                cell = QTableWidgetItem(text)
                cell.setToolTip(task.destination if col == 2 else text)
                self.table.setItem(row, col, cell)
        layout.addWidget(self.table)
        self.summary = QLabel("等待开始")
        layout.addWidget(self.summary)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        layout.addWidget(self.progress)
        row = QHBoxLayout()
        self.cancel_button = QPushButton("取消剩余任务")
        self.cancel_button.clicked.connect(self.cancel)
        row.addWidget(self.cancel_button)
        button = QPushButton("打开输出目录")
        button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(directory)))
        row.addWidget(button)
        layout.addLayout(row)
        # Signal transport ensures widgets are updated only on the GUI thread.
        from PySide6.QtCore import QObject, Signal
        class Updates(QObject):
            item = Signal(int, str, float, str)
        self.updates = Updates(self)
        self.updates.item.connect(self.update_item)
        self.worker = Job(run_exports, tasks, parent=self, item=self.updates.item.emit, keep_partial=True)
        self.worker.progress.connect(self.update_progress)
        self.worker.result.connect(self.done_exports)
        self.worker.failed.connect(self.failed_exports)
        self.worker.finished.connect(self.stopped)

    def start(self):
        self.show()
        self.worker.start()

    def update_item(self, row, state, value, detail):
        self.table.item(row, 1).setText(f"{state} · {value:.0%}" if state == "导出中" else state)
        self.table.item(row, 3).setText(detail)
        self.table.item(row, 3).setToolTip(detail)

    def update_progress(self, value, text):
        self.progress.setValue(round(value * 1000))
        self.summary.setText(text)

    def done_exports(self, outcomes):
        counts = Counter(o["state"] for o in outcomes)
        self.summary.setText(" · ".join(f"{state} {count} 条" for state, count in counts.items()))
        from datetime import datetime
        path = log_folder() / ("batch-export-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
        try:
            path.write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
            self.summary.setToolTip(f"任务日志：{path}")
        except OSError as error:
            self.summary.setToolTip(f"任务日志写入失败：{error}")

    def failed_exports(self, error):
        self.summary.setText("队列失败：" + error)

    def stopped(self):
        self.running = False
        self.cancel_button.setEnabled(False)

    def cancel(self):
        self.worker.cancel()
        self.cancel_button.setEnabled(False)
        self.summary.setText("正在取消；已完成的成品保留，当前临时文件将清理。")

    def closeEvent(self, event):
        if self.running:
            self.hide()  # queue is accessible again through workspace buttons
            event.ignore()
        else:
            event.accept()


def open_export_queue(window, selected=True):
    if hasattr(window, "_mark_workflow"):
        window._mark_workflow(3)
    existing = getattr(window, "export_queue", None)
    if existing is not None and existing.running:
        existing.show()
        existing.raise_()
        return
    if window.busy:
        return
    window._stash_clip()
    ids = window._selected_clip_ids()
    clips = [c for c in window.clips if not selected or c.id in ids]
    if not clips:
        window.queue_summary.setText("请先选中需要导出的素材。")
        return
    directory = QFileDialog.getExistingDirectory(window, "批量输出目录 · 已有文件自动编号",
                                                 str(window.preferences.value("export_folder", "")))
    if not directory:
        return
    try:
        tasks = prepare_exports(clips, directory)
    except Exception as error:
        QMessageBox.warning(window, "无法建立导出队列", str(error))
        return
    window.preferences.setValue("export_folder", directory)
    window.export_queue = ExportQueueDialog(tasks, directory, window)
    window.export_queue.start()
