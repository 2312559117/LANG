"""Approved A workspace geometry / D obsidian and warm-orange palette."""
from .media import resource_root
BACKGROUND = '#141414'
SURFACE = '#1b1b1b'
CANVAS = '#111110'
TEXT = '#eee8df'
MUTED = '#aaa49b'
ACCENT = '#e88c51'
CREAM = '#eadfce'
BORDER = '#32302d'
STYLE = """
QMainWindow, QDialog, QWidget { background: #141414; color: #eee8df; font-family: 'Microsoft YaHei UI'; font-size: 13px; }
QLabel { background: transparent; border: none; }
QLabel#muted, QLabel#trimMuted { color: #aaa49b; }
QLabel#eyebrow { color: #aaa49b; font-size: 10px; }
QLabel#title { color: #f3ece2; font-size: 21px; font-weight: 700; }
QLabel#panelTitle { color: #eee8df; font-size: 16px; font-weight: 600; }
QLabel#accent, QLabel#trimDuration, QLabel#step { color: #e88c51; }
QLabel#fileName, QLabel#assetFilename { font-weight: 600; font-size: 14px; }
QLabel#timecode, QLabel#monitorClock { color: #bdb6ab; font-family: 'Consolas', 'Microsoft YaHei UI'; font-size: 12px; }
QLabel#trimTitle { font-size: 18px; font-weight: 600; }
QLabel#localBadge { color: #bbb2a5; border: 1px solid #423d37; border-radius: 3px; padding: 5px 9px; font-size: 10px; }
QFrame#sidebar, QFrame#referenceSidebar, QFrame#taskSidebar { background: #1b1b1b; border: 1px solid #32302d; border-radius: 4px; }
QFrame#header { background: transparent; border-bottom: 1px solid #32302d; }
QFrame#divider { background: #32302d; max-height: 1px; }
QFrame#fileWell, QFrame#sidebarSection { background: #1b1b1b; border: 1px solid #32302d; border-radius: 4px; }
QFrame#monitor, QWidget#adPreviewPanel { background: #1b1b1b; border: 1px solid #32302d; border-radius: 4px; }
QFrame#timelineWell, QFrame#adTimeline { background: #191918; border: 1px solid #32302d; border-radius: 4px; }
QFrame#referenceStep { background: #1b1b1b; border: 1px solid #32302d; border-radius: 4px; }
QWidget#adStepContent { background: transparent; }
QFrame#referenceStep QLabel#panelTitle { font-size: 16px; font-weight: 600; }
QFrame#referenceStep QToolButton { background: transparent; color: #b7afa5; border: none; padding: 2px; }
QFrame#referenceStep QToolButton:hover { background: #2b2824; }
QFrame#referenceStep QListWidget { background: transparent; border: none; }
QPushButton, QToolButton { background: #212120; color: #e0d9ce; border: 1px solid #3d3934; border-radius: 4px; padding: 5px 10px; min-height: 18px; font-weight: 500; }
QPushButton:hover, QToolButton:hover { background: #2b2824; border-color: #655446; }
QPushButton:pressed, QToolButton:pressed { background: #30271f; border-color: #e88c51; }
QPushButton:focus, QToolButton:focus { border-color: #e88c51; }
QPushButton:disabled, QToolButton:disabled { background: #1e1e1d; color: #746f68; border-color: #32302d; }
QPushButton#primary, QPushButton#trimApplyButton, QPushButton#trackAction, QPushButton#adGlobalExport { background: #eadfce; color: #191714; border: 1px solid #eadfce; font-weight: 600; }
QPushButton#primary:hover, QPushButton#trimApplyButton:hover, QPushButton#trackAction:hover, QPushButton#adGlobalExport:hover { background: #f6ead7; border-color: #f6ead7; }
QPushButton#primary:pressed, QPushButton#trackAction:pressed, QPushButton#adGlobalExport:pressed { background: #d9c7ad; }
QPushButton#primary:disabled, QPushButton#trackAction:disabled, QPushButton#adGlobalExport:disabled { background: #49443d; color: #aaa196; border-color: #49443d; }
QPushButton#quiet, QPushButton#toolButton { background: transparent; color: #bcb4a8; }
QPushButton#viewButton:checked, QPushButton#workflowStep:checked { background: #30271f; color: #f0d5b8; border-color: #e88c51; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #191918; selection-background-color: #57412f; border: 1px solid #3d3934; border-radius: 4px; padding: 4px 8px; min-height: 19px; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover { border-color: #655446; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: #e88c51; }
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled { color: #746f68; border-color: #32302d; }
QComboBox::drop-down { border: none; width: 21px; }
QComboBox QAbstractItemView { background: #242320; selection-background-color: #57412f; selection-color: #f6ecdd; border: 1px solid #574b3e; padding: 5px; }
QSlider { background: transparent; border: none; min-height: 20px; max-height: 20px; }
QSlider::groove:horizontal { background: #34322f; height: 4px; border: none; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #e88c51; height: 4px; border: none; border-radius: 2px; }
QSlider::handle:horizontal { background: #f2e8d8; border: 2px solid #e88c51; width: 10px; height: 10px; margin: -5px 0; border-radius: 7px; }
QSlider::handle:horizontal:hover { background: #fff4e4; }
QSlider::sub-page:horizontal:disabled { background: #79624e; }
QSlider::handle:horizontal:disabled { background: #aaa094; border-color: #79624e; }
QSlider#monitorSeek::groove:horizontal { height: 6px; border-radius: 3px; }
QSlider#monitorSeek::handle:horizontal { border: 1px solid #35312c; width: 13px; height: 13px; margin: -4px 0; }
QListWidget { background: #1b1b1b; border: 1px solid #32302d; border-radius: 4px; outline: none; }
QListWidget::item { padding: 0; border: none; }
QListWidget::item:selected, QListWidget::item:hover { background: transparent; }
QProgressBar { background: #33312e; border: none; border-radius: 3px; max-height: 7px; }
QProgressBar::chunk { background: #e88c51; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 5px; background: transparent; margin: 3px 0; }
QScrollBar::handle:vertical { background: #504940; min-height: 28px; border-radius: 2px; }
QScrollBar::handle:vertical:hover { background: #806951; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QTabWidget::pane { border: none; background: transparent; padding-top: 12px; }
QTabBar::tab { background: transparent; border: none; border-bottom: 2px solid #423b32; color: #aaa196; padding: 8px 9px; margin-right: 2px; }
QTabBar::tab:selected { color: #efd3b1; border-bottom-color: #e88c51; }
QTabBar::tab:hover { color: #f1e7d8; background: #2f2a24; }
QCheckBox { spacing: 8px; background: transparent; color: #c2baae; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #b5a996; border-radius: 3px; background: #191918; }
QCheckBox::indicator:checked { border: 2px solid #e88c51; background: #e88c51; }
QCheckBox::indicator:disabled { border-color: #63594b; background: #26231f; }
QToolTip { color: #eee5d9; background: #2c2721; border: 1px solid #6e5842; padding: 6px; }
QMenuBar { background: transparent; color: #b9b1a6; padding: 0; font-size: 14px; }
QMenuBar::item { padding: 8px 12px; background: transparent; }
QMenuBar::item:selected, QMenu::item:selected { background: #403124; color: #f4e7d5; }
QMenu { background: #24211d; border: 1px solid #55493b; padding: 5px; }
QMenu::item { padding: 8px 25px; }
QMenu::separator { height: 1px; background: #44392e; margin: 5px; }
QMessageBox { background: #211f1c; }
QSplitter::handle { background: #141414; width: 10px; }
QSplitter::handle:hover { background: #40352a; }
QPushButton#sectionHeader { text-align: left; background: #24211d; border: none; border-bottom: 1px solid #3c342b; padding: 9px 10px; font-weight: 600; }
QPushButton#sectionHeader:checked { color: #efd3b1; background: #2c241d; }
QFrame#assetStatusBar { background: transparent; border: none; }
QLabel#assetMetadata { color: #a8a196; font-family: 'Microsoft YaHei UI'; font-size: 11px; }
QLabel#assetProgress { color: #e88c51; font-family: 'Consolas'; font-size: 12px; }
QTableWidget { background: #1b1b1b; gridline-color: #3d352b; selection-background-color: #403124; }
QHeaderView::section { background: #2b251e; color: #c9bdac; padding: 7px; border: none; border-bottom: 1px solid #554739; }
QFrame#adHeader { background: #161615; border-bottom: 1px solid #32302d; }
QWidget#adPreviewHead, QWidget#adTransport, QWidget#adDualTransport, QWidget#adMonitor { background: transparent; border: none; }
QWidget#adPreviewHead { border-bottom: 1px solid #2d2a25; }
QWidget#adTimelineToolbar { background: #1b1b1b; border-bottom: 1px solid #32302d; }
QToolButton#adTransportButton, QPushButton#adTransportButton { background: transparent; padding: 0; border: 1px solid transparent; min-height: 0; }
QToolButton#adTransportButton:hover, QPushButton#adTransportButton:hover { background: #30271f; border-color: #65503b; }
QToolButton#adTransportButton:checked, QPushButton#adTransportButton:checked { background: #282018; border-color: #e88c51; }
QPushButton#adSecondaryContinue { background: #212120; border-color: #3d3934; color: #e0d9ce; }
QPushButton#adSecondaryContinue:disabled { background: #1e1e1d; border-color: #32302d; color: #746f68; }
QPushButton#adSecondaryContinue:hover { background: #2b2824; border-color: #655446; }
QPushButton#adSecondaryContinue:pressed { background: #30271f; border-color: #e88c51; }
"""
_asset_root = (resource_root() / 'assets').as_posix()
STYLE += """
QCheckBox::indicator:checked { image: url('%s/check-mark.svg'); }
QComboBox::down-arrow { image: url('%s/chevron-down.png'); width: 12px; height: 8px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 17px; border: none; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 17px; border: none; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url('%s/chevron-up.png'); width: 12px; height: 8px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url('%s/chevron-down.png'); width: 12px; height: 8px; }
""" % (_asset_root, _asset_root, _asset_root, _asset_root)
