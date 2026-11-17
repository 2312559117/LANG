"""Single-column editing workspace and persistent source metadata."""
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, QSize, QRect, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget, QSizePolicy, QToolButton
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QListWidget

from .ui_icons import icon


class ClipList(QListWidget):
    """Adapt grid cells to the material panel without changing the item model."""
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_grid()

    def update_grid(self):
        if self.property('clipViewMode') != 'grid':return
        width=max(80,self.viewport().width())
        columns=max(1,(width+67)//134)
        cell=max(72,(width-4)//columns)
        size=QSize(cell,round((cell-12)*.65)+38)
        if self.gridSize()!=size:self.setGridSize(size)


def duration_text(seconds):
    """Format actual media duration for the thumbnail badge and metadata."""
    try:
        value=max(0,int(float(seconds)))
    except (TypeError,ValueError,OverflowError):
        return '—'
    return f'{value//3600:02d}:{value//60%60:02d}:{value%60:02d}' if value>=3600 else f'{value//60:02d}:{value%60:02d}'


class SectionNavigator(QObject):
    """Named page registry whose navigation reveals a real sidebar section."""
    currentChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pages, self.titles = [], []
        self.index = 0
        self.reveal = None

    def addTab(self, page, title):
        self.pages.append(page)
        self.titles.append(title)
        return len(self.pages) - 1

    def count(self):
        return len(self.pages)

    def tabText(self, index):
        return self.titles[index]

    def widget(self, index):
        return self.pages[index]

    def currentIndex(self):
        return self.index

    def setCurrentIndex(self, index):
        if not 0 <= index < self.count():
            return
        self.index = index
        if self.reveal:
            self.reveal(index)
        self.currentChanged.emit(index)


class ClipDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        return self.parent().gridSize() if self.parent() is not None and self.parent().property('clipViewMode') == 'grid' else QSize(0, 80)

    def paint(self, painter, option, index):
        data = index.data(Qt.ItemDataRole.UserRole + 1) or {}
        if self.parent() is not None and self.parent().property('clipViewMode') == 'grid':
            self._paint_grid(painter, option, data)
            return
        name = data.get("name", index.data(Qt.ItemDataRole.DisplayRole) or "")
        state = data.get("state", "")
        row = QRectF(option.rect.adjusted(0, 2, 0, -2))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected=bool(option.state & QStyle.StateFlag.State_Selected)
        hovered=bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.setPen(QPen(QColor('#e88c51' if selected else '#242320'),1))
        painter.setBrush(QColor('#23221f' if selected else '#242320' if hovered else '#20201f'))
        painter.drawRoundedRect(row.adjusted(.5,.5,-.5,-.5),4,4)
        thumbnail = QRect(option.rect.x()+8,option.rect.y()+10,102,60)
        painter.fillRect(thumbnail, QColor("#151412"))
        pixmap = data.get("thumbnail")
        clip_path=QPainterPath();clip_path.addRoundedRect(QRectF(thumbnail),3,3)
        painter.save();painter.setClipPath(clip_path)
        if pixmap is not None and not pixmap.isNull():
            image = pixmap.scaled(thumbnail.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(thumbnail.x()+(thumbnail.width()-image.width())//2,
                               thumbnail.y()+(thumbnail.height()-image.height())//2, image)
        else:
            painter.setPen(QColor("#948a7b"))
            painter.drawText(thumbnail, Qt.AlignmentFlag.AlignCenter,
                             "无预览" if data.get("thumbnail_state")=="failed" else "加载中")
        painter.restore()
        duration=duration_text(data.get('duration'))
        if duration!='—':
            badge=QRect(thumbnail.right()-39,thumbnail.bottom()-18,39,18)
            font=QFont('Microsoft YaHei UI');font.setPixelSize(12);painter.setFont(font)
            if len(duration)>5:badge.setLeft(thumbnail.right()-painter.fontMetrics().horizontalAdvance(duration)-6)
            painter.setPen(Qt.PenStyle.NoPen);painter.setBrush(QColor(12,11,10,215));painter.drawRoundedRect(QRectF(badge),2,2)
            painter.setPen(QColor('#fff2df'));painter.drawText(badge,Qt.AlignmentFlag.AlignCenter,duration)
        area=QRect(thumbnail.right()+15,option.rect.y()+12,max(10,option.rect.right()-thumbnail.right()-21),56)
        font=QFont('Microsoft YaHei UI');font.setPixelSize(14);font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font);metrics=painter.fontMetrics();painter.setPen(QColor('#eee8df'))
        painter.drawText(QRect(area.x(),area.y()+2,area.width(),24),Qt.AlignmentFlag.AlignVCenter,
                         metrics.elidedText(name,Qt.TextElideMode.ElideMiddle,max(10,area.width())))
        font.setPixelSize(11);font.setWeight(QFont.Weight.Normal);painter.setFont(font)
        width,height=data.get('width'),data.get('height')
        detail=f'{width} × {height}   {duration}' if width and height else state
        painter.setPen(QColor('#a8a196'))
        painter.drawText(QRect(area.x(),area.y()+30,area.width(),21),Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(detail,Qt.TextElideMode.ElideRight,max(10,area.width())))
        painter.restore()

    def _paint_grid(self, painter, option, data):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        card = QRectF(option.rect.adjusted(2, 2, -2, -2))
        painter.setPen(QPen(QColor('#e88c51' if selected else '#302e29'), 1))
        painter.setBrush(QColor('#2d2822' if selected else '#272622' if hovered else '#1c1c1b'))
        painter.drawRoundedRect(card.adjusted(.5,.5,-.5,-.5), 5, 5)
        photo = QRect(option.rect.x()+6, option.rect.y()+6, option.rect.width()-12, option.rect.height()-38)
        failed = data.get('thumbnail_state') == 'failed'
        painter.fillRect(photo, QColor('#401b19' if failed else '#111110'))
        pixmap = data.get('thumbnail')
        if pixmap is not None and not pixmap.isNull():
            image = pixmap.scaled(photo.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(photo.x()+(photo.width()-image.width())//2,
                               photo.y()+(photo.height()-image.height())//2, image)
        else:
            painter.setPen(QColor('#e8bbb4' if failed else '#aca394'))
            painter.drawText(photo.adjusted(2,16,-2,-2), Qt.AlignmentFlag.AlignCenter,
                             '预览不可用' if failed else '加载中')
        font = QFont('Microsoft YaHei UI');font.setPixelSize(10);painter.setFont(font)
        state = '当前' if data.get('active') else data.get('state', '已添加')
        state = painter.fontMetrics().elidedText(state, Qt.TextElideMode.ElideRight, max(22,photo.width()-48))
        for text, right in ((state, False), (duration_text(data.get('duration')), True)):
            width=painter.fontMetrics().horizontalAdvance(text)+7
            badge=QRect(photo.right()-width+1 if right else photo.left(),photo.top(),width,16)
            painter.fillRect(badge,QColor(15,14,12,215));painter.setPen(QColor('#f4e6d4'))
            painter.drawText(badge,Qt.AlignmentFlag.AlignCenter,text)
        font.setPixelSize(11);painter.setFont(font);painter.setPen(QColor('#e3dbcf'))
        name=data.get('name','')
        painter.drawText(QRect(photo.x(),photo.bottom()+6,photo.width(),22),Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(name,Qt.TextElideMode.ElideMiddle,photo.width()))
        painter.restore()


class SidebarSection(QFrame):
    def __init__(self, title, content, *, expanded=False, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarSection")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.button = QPushButton()
        self.button.setObjectName("sectionHeader")
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.title = title
        self.content = content
        layout.addWidget(self.button)
        layout.addWidget(content)
        self.button.toggled.connect(self.set_expanded)
        self.set_expanded(expanded)

    def set_expanded(self, expanded):
        self.button.blockSignals(True)
        self.button.setChecked(expanded)
        self.button.blockSignals(False)
        self.button.setText(("▾  " if expanded else "▸  ") + self.title)
        self.content.setVisible(expanded)


class AssetStatusBar(QFrame):
    """Metadata follows the displayed source, including a foreign queue preview."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("assetStatusBar")
        self.setMinimumWidth(150)
        self.full_filename = "尚未导入素材"
        self.source_path = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        column = QVBoxLayout()
        column.setSpacing(4)
        self.filename_label = QLabel(self.full_filename)
        self.filename_label.setTextFormat(Qt.TextFormat.PlainText)
        self.filename_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.filename_label.setObjectName("assetFilename")
        self.filename_label.setMinimumWidth(40)
        self.metadata_label = QLabel("分辨率 —  ·  fps —  ·  帧数 —")
        self.metadata_label.setObjectName("assetMetadata")
        self.metadata_label.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Preferred)
        self.metadata_label.setMinimumWidth(0)
        column.addWidget(self.filename_label)
        column.addWidget(self.metadata_label)
        layout.addLayout(column, 1)
        self.progress_label = QLabel("就绪 · —")
        self.progress_label.setObjectName("assetProgress")
        layout.addWidget(self.progress_label)

    def set_info(self, info, *, context=""):
        self.source_path = info.path if info else None
        self.full_filename = Path(info.path).name if info else "尚未导入素材"
        self.filename_label.setToolTip(str(info.path) if info else "导入视频或打开合成演示")
        self.metadata_label.setText(
            f"{info.width} × {info.height}   {info.fps:g} fps   {duration_text(info.frames/info.fps)}"
            if info else "分辨率 —   fps —   时长 —")
        self.metadata_label.setToolTip(
            f'{info.width} × {info.height} · {info.fps:g} fps · {info.frames:,} 帧' + (f' · {context}' if context else '')
            if info else '导入素材后显示分辨率、帧率与时长')
        self._elide()

    def set_progress(self, value=None, *, state="就绪"):
        self.progress_label.setText(f"{state} · {max(0, min(100, round(value * 100)))}%"
                                    if value is not None else f"{state} · —")

    def _elide(self):
        self.filename_label.setText(self.filename_label.fontMetrics().elidedText(
            self.full_filename, Qt.TextElideMode.ElideMiddle, max(40, self.filename_label.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()


class ReferenceStep(QFrame):
    """Always-visible basic controls; the header menu opens secondary settings."""
    def __init__(self, number, title, content, menu, parent=None):
        super().__init__(parent)
        self.setObjectName("referenceStep")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        row = QHBoxLayout()
        tag = QLabel(number, self)
        tag.setObjectName("step")
        tag.hide()  # Retained for the established four-step accessibility checks.
        symbol=QLabel()
        symbol.setObjectName('sectionIcon')
        symbol.setFixedSize(24,24)
        symbol.setPixmap(icon({'01':'folder','02':'target','03':'compose','04':'export'}.get(number,'more'),colour='#e88c51',size=24).pixmap(24,24))
        row.addWidget(symbol)
        heading = QLabel(title)
        heading.setObjectName("panelTitle")
        row.addWidget(heading)
        row.addStretch()
        self.more_button = QToolButton()
        self.more_button.setObjectName('sectionMore')
        self.more_button.setIcon(icon('more'))
        self.more_button.setIconSize(QSize(18,18))
        self.more_button.setFixedSize(25,25)
        self.more_button.setAccessibleName(title + "更多设置")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_button.setMenu(menu)
        row.addWidget(self.more_button)
        layout.addLayout(row)
        layout.addWidget(content)
        content.setObjectName("adStepContent")
        self.content = content

    def set_expanded(self, expanded):
        self.content.setVisible(expanded)
