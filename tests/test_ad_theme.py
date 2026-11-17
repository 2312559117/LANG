"""AD presentation keeps actual metadata and Qt control-state contracts."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QMenu, QPushButton, QStyle, QStyleOptionViewItem, QWidget

from lockframe.ui_icons import icon
from lockframe.workspace_ui import AssetStatusBar, ClipDelegate, ReferenceStep, duration_text
from lockframe.theme import STYLE


@pytest.fixture(scope='module')
def app():
    instance=QApplication.instance() or QApplication([])
    yield instance


def test_vector_icons_render_normal_checked_and_disabled_states(app):
    for name in ('undo','redo','layout','export','folder','import','filter','play','pause',
                 'first','last','prev','next','fit','one','pointer','hand','target','compose',
                 'more','link','zoom-in','zoom-out'):
        drawing=icon(name)
        normal=drawing.pixmap(18,18,QIcon.Mode.Normal,QIcon.State.Off).toImage()
        disabled=drawing.pixmap(18,18,QIcon.Mode.Disabled,QIcon.State.Off).toImage()
        selected=drawing.pixmap(18,18,QIcon.Mode.Normal,QIcon.State.On).toImage()
        assert not drawing.isNull()
        assert any(normal.pixelColor(x,y).alpha() for x in range(normal.width()) for y in range(normal.height()))
        assert normal!=disabled and normal!=selected


def test_material_row_renders_unicode_thumbnail_and_truthful_duration(app):
    model=QStandardItemModel()
    item=QStandardItem('海边日落_01.mp4');model.appendRow(item)
    thumbnail=QPixmap(192,108);thumbnail.fill(QColor('#764d32'))
    unchanged=thumbnail.toImage()
    item.setData(dict(name='海边日落_01.mp4',width=1920,height=1080,duration=74.9,
                      thumbnail=thumbnail,state='已跟踪'),Qt.ItemDataRole.UserRole+1)
    option=QStyleOptionViewItem();option.rect=QRect(0,0,296,80)
    option.state=QStyle.StateFlag.State_Enabled|QStyle.StateFlag.State_Selected
    canvas=QPixmap(296,80);canvas.fill(QColor('#141414'))
    painter=QPainter(canvas);ClipDelegate().paint(painter,option,model.index(0,0));painter.end()
    assert thumbnail.toImage()==unchanged
    assert duration_text(74.9)=='01:14' and duration_text(3604)=='01:00:04'
    assert duration_text(None)=='—'
    assert max(canvas.toImage().pixelColor(x,35).red() for x in range(3))>170
    assert canvas.toImage().pixelColor(20,25)==QColor('#764d32')


def test_reference_header_keeps_number_and_menu_actions(app):
    menu=QMenu();action=menu.addAction('区域与数值微调')
    step=ReferenceStep('02','跟踪',QWidget(),menu)
    assert step.findChild(QLabel,'step').text()=='02'
    assert step.findChild(QLabel,'step').isHidden()
    assert step.more_button.menu() is menu and action in step.more_button.menu().actions()
    assert not step.more_button.icon().isNull()


def test_metadata_compacts_without_hiding_true_frame_count(app):
    from types import SimpleNamespace
    bar=AssetStatusBar();info=SimpleNamespace(path='D:/素材/试拍.mp4',width=1920,height=1080,fps=25,frames=250)
    bar.set_info(info,context='队列正在跟踪')
    assert '1920 × 1080' in bar.metadata_label.text()
    assert '25 fps' in bar.metadata_label.text() and '00:10' in bar.metadata_label.text()
    assert '250 帧' in bar.metadata_label.toolTip()
    assert '队列正在跟踪' in bar.metadata_label.toolTip()
    assert '#63c6ca' not in STYLE and '#e88c51' in STYLE


def test_checked_indicator_and_disabled_continue_render_clear_states(app):
    from pathlib import Path
    check=QCheckBox('跟丢时自动暂停');check.setStyleSheet(STYLE)
    check.setChecked(True);check.resize(180,32);check.show();app.processEvents()
    rendered=check.grab().toImage()
    # A real SVG check glyph has dark ink inside the orange indicator.
    assert any(rendered.pixelColor(x,y).red()>180 for x in range(2,16) for y in range(9,23))
    assert any(rendered.pixelColor(x,y).red()<50 for x in range(5,14) for y in range(11,21))
    assert (Path(__file__).parents[1]/'assets/check-mark.svg').exists()
    button=QPushButton('继续跟踪');button.setObjectName('adSecondaryContinue')
    button.setStyleSheet(STYLE);button.setEnabled(False);button.resize(142,38)
    button.show();app.processEvents()
    assert button.grab().toImage().pixelColor(10,10)==QColor('#1e1e1d')
    check.close();button.close()
