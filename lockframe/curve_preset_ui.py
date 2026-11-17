import numpy as np
from PySide6.QtCore import Qt,QSize,QPointF,Signal
from PySide6.QtGui import QPixmap,QPainter,QPen,QColor,QIcon,QPainterPath
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QListWidget,QListWidgetItem,QPushButton,QLabel,QInputDialog,QListView
from .curve_presets import PresetStore,builtins,apply_curve
from .time_remap_model import normalize_config,active_points
from .time_remap_math import basic_speed,evaluate_speed
from .time_remap_map import output_frame_map


def preview_icon(config):
    config=normalize_config(config);points=active_points(config)
    xs=np.linspace(0,1 if config['mode']=='basic' else points[-1]['position'],81)
    ys=[basic_speed(config['basic'],float(x)) if config['mode']=='basic' else evaluate_speed(points,float(x)) for x in xs]
    pixmap=QPixmap(160,68);pixmap.fill(QColor('#181817'));p=QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing);p.setPen(QPen(QColor('#37312a'),1))
    for i in range(1,4):p.drawLine(8,8+i*13,152,8+i*13)
    path=QPainterPath();lo=min(0.,min(ys));hi=max(2.,max(ys))
    for i,y in enumerate(ys):
        point=QPointF(8+i/80*144,59-(y-lo)/(hi-lo)*50)
        if i:path.lineTo(point)
        else:path.moveTo(point)
    p.setPen(QPen(QColor('#eca56b'),2));p.drawPath(path);p.end();return QIcon(pixmap)


class PresetPicker(QDialog):
    applied=Signal(object)

    def __init__(self,config,frames,parent=None,store=None):
        super().__init__(parent);self.config=normalize_config(config);self.frames=frames;self.store=store or PresetStore()
        self.setWindowTitle('曲线预设');self.setModal(True);self.resize(650,380)
        layout=QVBoxLayout(self);layout.addWidget(QLabel('选择曲线预览后应用；保留当前音频、光流与其他设置。'))
        self.list=QListWidget();self.list.setViewMode(QListView.ViewMode.IconMode);self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setIconSize(QSize(160,68));self.list.setGridSize(QSize(194,106));layout.addWidget(self.list)
        self.message=QLabel();self.message.setWordWrap(True);layout.addWidget(self.message);row=QHBoxLayout()
        save=QPushButton('保存当前曲线…');save.clicked.connect(self.save_current);row.addWidget(save)
        self.delete=QPushButton('删除自定义预设');self.delete.clicked.connect(self.remove_current);row.addWidget(self.delete)
        row.addStretch();apply=QPushButton('应用曲线');apply.clicked.connect(self.choose);row.addWidget(apply);layout.addLayout(row)
        self.list.itemDoubleClicked.connect(lambda *_:self.choose());self.list.currentRowChanged.connect(self.sync);self.reload()

    def reload(self):
        self.list.clear();self.items=builtins()
        try:self.items+=self.store.read();self.message.clear()
        except Exception as error:self.message.setText(str(error))
        for value in self.items:
            item=QListWidgetItem(preview_icon(value['curve']),value['name']+(' · 自定义' if not value['builtin'] else ''))
            item.setData(Qt.ItemDataRole.UserRole,value['id']);self.list.addItem(item)
        if self.items:self.list.setCurrentRow(0)
        self.sync()

    def sync(self,*_):
        i=self.list.currentRow();self.delete.setEnabled(0<=i<len(self.items) and not self.items[i]['builtin'])

    def choose(self):
        index=self.list.currentRow()
        if not 0<=index<len(self.items):return
        try:
            config=apply_curve(self.config,self.items[index]['curve']);output_frame_map(config,self.frames)
            self.applied.emit(config);self.accept()
        except Exception as error:self.message.setText(str(error))

    def save_current(self):
        name,ok=QInputDialog.getText(self,'保存当前曲线','预设名称')
        if not ok:return
        try:self.store.save(name,self.config);self.reload();self.list.setCurrentRow(len(self.items)-1)
        except Exception as error:self.message.setText(str(error))

    def remove_current(self):
        index=self.list.currentRow()
        if not 0<=index<len(self.items) or self.items[index]['builtin']:return
        try:self.store.remove(self.items[index]['id']);self.reload()
        except Exception as error:self.message.setText(str(error))
