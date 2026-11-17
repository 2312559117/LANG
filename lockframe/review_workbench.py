from PySide6.QtCore import Qt,Signal
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QComboBox,QPushButton,QLabel,
    QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView)
from .review_records import collect_issues,clean_marks


class ReviewWorkbench(QDialog):
    locate=Signal(object)
    changed=Signal()

    def __init__(self,host):
        super().__init__(host);self.host=host;self.rows=[]
        self.setWindowTitle('集中复核工作台');self.setModal(False);self.resize(980,480)
        layout=QVBoxLayout(self);row=QHBoxLayout()
        self.scope=QComboBox();self.scope.addItem('全部素材',None)
        for clip in host.clips:self.scope.addItem(clip.info.path.split('\\')[-1].split('/')[-1],clip.id)
        self.filter=QComboBox()
        for title,key in (('待检查','pending'),('已人工检查','checked'),('修正/关键帧记录','anchors'),('全部','all')):self.filter.addItem(title,key)
        row.addWidget(self.scope,1);row.addWidget(self.filter);refresh=QPushButton('刷新');refresh.clicked.connect(self.refresh);row.addWidget(refresh);layout.addLayout(row)
        self.table=QTableWidget(0,5);self.table.setHorizontalHeaderLabels(['素材','类型','源帧区间','状态','说明'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4,QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(lambda *_:self.jump());self.table.itemSelectionChanged.connect(self.sync)
        layout.addWidget(self.table);row=QHBoxLayout()
        for name,callback in (('上一个问题',lambda:self.step(-1)),('下一个问题',lambda:self.step(1)),('定位原片',self.jump)):
            button=QPushButton(name);button.clicked.connect(callback);row.addWidget(button)
        row.addStretch();self.mark=QPushButton('标记已检查');self.mark.clicked.connect(lambda:self.set_checked(True));row.addWidget(self.mark)
        self.unmark=QPushButton('恢复待检查');self.unmark.clicked.connect(lambda:self.set_checked(False));row.addWidget(self.unmark)
        layout.addLayout(row);self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        self.scope.currentIndexChanged.connect(self.refresh);self.filter.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def selected(self):
        index=self.table.currentRow()
        return self.rows[index] if 0<=index<len(self.rows) else None

    def refresh(self):
        self.host._stash_clip();selected=self.selected();key=selected.key if selected else None
        scope=self.scope.currentData();self.scope.blockSignals(True);self.scope.clear();self.scope.addItem('全部素材',None)
        from pathlib import Path
        for clip in self.host.clips:self.scope.addItem(Path(clip.info.path).name,clip.id)
        self.scope.setCurrentIndex(max(0,self.scope.findData(scope)));self.scope.blockSignals(False)
        clips=[clip for clip in self.host.clips if self.scope.currentData() in (None,clip.id)]
        mode=self.filter.currentData();all_rows=collect_issues(clips)
        self.rows=[v for v in all_rows if mode=='all' or (mode=='pending' and not v.checked and not v.historical) or (mode=='checked' and v.checked) or (mode=='anchors' and v.historical)]
        self.table.setRowCount(len(self.rows))
        for i,item in enumerate(self.rows):
            status='历史记录' if item.historical else '已检查（人工）' if item.checked else '待修正' if item.kind=='真实跟丢' else '待检查'
            for j,text in enumerate((item.filename,item.kind,f'{item.first+1}–{item.last+1}',status,item.reason)):
                cell=QTableWidgetItem(text);cell.setToolTip(text);self.table.setItem(i,j,cell)
        if self.rows:self.table.selectRow(next((i for i,v in enumerate(self.rows) if key and v.key==key),0))
        self.note.setText(f'当前 {len(self.rows)} 项 · 上限2000项，可按素材筛选。标记已检查只记录人工复核，不修改轨迹或解除真实跟丢。')
        self.sync()

    def sync(self):
        item=self.selected();idle=not self.host.busy
        self.mark.setEnabled(bool(item and item.checkable and not item.checked and idle))
        self.unmark.setEnabled(bool(item and item.checkable and item.checked and idle))

    def jump(self):
        item=self.selected()
        if item:self.locate.emit(item)

    def step(self,delta):
        if not self.rows:return
        self.table.selectRow((self.table.currentRow()+delta)%len(self.rows));self.jump()

    def set_checked(self,checked):
        item=self.selected()
        if item is None or not item.checkable or self.host.busy:return
        clip=self.host._clip(item.clip_id)
        if clip is None:return
        clip.review_marks=clean_marks(clip.review_marks)
        if checked:
            if len(clip.review_marks)>=2048 and item.key not in clip.review_marks:
                self.note.setText('该素材复核记录已满，请先清理过期记录或另存工程。');return
            clip.review_marks[item.key]=True
        else:clip.review_marks.pop(item.key,None)
        clip.dirty=True;self.host._sequence_changed=True
        if clip.id==self.host.active_clip_id:self.host.dirty=True
        self.changed.emit();self.refresh()
