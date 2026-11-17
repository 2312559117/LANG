from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
import time

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, QUrl, QEvent
from PySide6.QtGui import QAction, QDesktopServices, QFont, QFontDatabase, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QDialog, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSlider, QSpinBox, QDoubleSpinBox, QTabWidget, QVBoxLayout, QWidget,
    QSplitter, QMenu,
    QSizePolicy, QLayout,
)

from . import __version__
from .demo import create_demo, demo_frame
from .export import export_video
from .media import FrameReader, cache_root, configure_cache_preferences, import_video, prepare_precision_video, resource_root, set_cache_dir
from .models import Settings, TrackResult, file_fingerprint, read_project, save_project
from .stabilization import build_plan, preview_frame
from .tracking import track_video
from .widgets import PreviewCanvas, TrackStrip
from .workers import Job
from .sequence_ui import SequenceMixin
from .live_preview_ui import LivePreviewMixin
from .quality import STAGES, parse_phase, phase_progress
from .trim import TrimDialog
from .performance import ComputeOptions, cache_budget
from .hardware_monitor import HardwareMonitor
from .logging import configure_logging, get_logger, log_folder


from .theme import STYLE
from .branding import application_icon, brand_pixmap
from .workspace_ui import AssetStatusBar, SectionNavigator, SidebarSection, ReferenceStep, ClipDelegate
from .progress_ui import TrackingProgressPanel
from .tracking_pause import TrackingPaused, computed_mask, is_complete
from .timebar_ui import Timebar
from .timebar_host import TimebarMixin
from .region_host import RegionMixin
from .region_toolbar import RegionToolbar
from .sections_ui import SectionsMixin
from .time_remap_controller import RetimeMixin
from .workspace_features import WorkspaceFeaturesMixin
from copy import deepcopy


def label(text, name=None, wrap=False):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def section(layout, number, title):
    row = QHBoxLayout()
    row.addWidget(label(number, "step"))
    row.addWidget(label(title, "panelTitle"))
    row.addStretch()
    layout.addLayout(row)


def divider(layout):
    widget = QFrame()
    widget.setObjectName("divider")
    widget.setFixedHeight(1)
    layout.addWidget(widget)


def timestamp(frame, fps):
    seconds = max(0, frame / fps)
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


class MainWindow(WorkspaceFeaturesMixin, RetimeMixin, SectionsMixin, RegionMixin, TimebarMixin, LivePreviewMixin, SequenceMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        # Offscreen Qt has no platform font database; use installed Windows fonts
        # for both visual tests and minimal Windows environments. No font is copied.
        if os.name == "nt" and "Microsoft YaHei UI" not in QFontDatabase.families():
            fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
            for name in ("msyh.ttc", "msyhbd.ttc", "segoeui.ttf"):
                candidate = fonts / name
                if candidate.exists():
                    QFontDatabase.addApplicationFont(str(candidate))
        self._ready = False
        self.info = None
        self.track = None
        self._retrack_previous = None
        self.plan = None
        from .original_preview import OriginalPreview
        self._original_preview = OriginalPreview(self)
        self._original_preview.ready.connect(self._original_preview_ready)
        self.anchors = {}
        self.section_tracking, self.tracking_sections = False, []
        self._region_shape = 'rectangle'
        self._polygon_points = ()
        self.reader = None
        self.frame = None
        self.frame_index = 0
        self.project_path = None
        self.last_export = None
        self.worker = None
        self.busy = False
        self.dirty = False
        self._pending_project = None
        self._pending_demo = False
        self._close_pending = False
        self.precision_info = None
        self._prepare_preview_pending = False
        self._preserve_preview_view = False
        self._tracking_active = False
        self._job_outcome = None
        self._cancel_progress = None
        self._job_kind = None
        self._header_value = None
        self._header_state = "就绪"
        self._display_info = None
        self._paused_tracking = None
        self._paused_by_clip = {}
        self._paused_queue_ids = []
        self._init_sequence()
        self._play_started = 0
        self._play_first = 0
        self.preferences = QSettings("Yongzheng", "LockFrameStudio")
        # Retire the removed cloud feature without reading or decrypting keys.
        self.preferences.remove("ai")
        self.preferences.sync()
        configure_cache_preferences(self.preferences)
        self.setWindowTitle("番茄跟踪")
        self.setWindowIcon(application_icon())
        self.resize(1586, 992)
        self.setMinimumSize(1160, 760)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._play_tick)
        self._build_menu()
        self._build_ui()
        self._ready = True
        self._sync_ui()
        self.hardware_monitor = HardwareMonitor(self)
        self.hardware_monitor.updated.connect(self._hardware_updated)
        self._install_workspace_features()

    def _build_menu(self):
        menu = self.menuBar().addMenu("文件")
        self.import_action = QAction("导入视频…", self)
        self.import_action.setShortcut(QKeySequence.StandardKey.Open)
        self.import_action.triggered.connect(self._choose_video)
        menu.addAction(self.import_action)
        self.open_action = QAction("打开工程…", self)
        self.open_action.setShortcut("Ctrl+Shift+O")
        self.open_action.triggered.connect(self._open_project)
        menu.addAction(self.open_action)
        self._sequence_menu(menu)
        self.save_action = QAction("保存工程…", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._save_workspace)
        menu.addAction(self.save_action)
        self.trim_action = QAction("裁剪视频片段…", self)
        self.trim_action.setShortcut("Ctrl+T")
        self.trim_action.triggered.connect(self._edit_trim)
        menu.addAction(self.trim_action)
        menu.addSeparator()
        self.demo_action = QAction("打开合成演示", self)
        self.demo_action.triggered.connect(self.open_demo)
        menu.addAction(self.demo_action)
        help_menu = self.menuBar().addMenu("帮助")
        guide = help_menu.addAction("使用说明")
        guide.triggered.connect(self._show_help)
        about = help_menu.addAction("关于与开源许可")
        about.triggered.connect(self._show_about)
        help_menu.addSeparator()
        self.diagnostics_action=help_menu.addAction('导出诊断包…')
        self.diagnostics_action.triggered.connect(self._open_diagnostics)
        help_menu.addAction('打开日志文件夹').triggered.connect(self._open_log_folder)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 0, 10, 8)
        layout.setSpacing(8)
        header = self.header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(2, 6, 2, 10)
        header_layout.setSpacing(12)
        mark = QLabel()
        mark.setPixmap(brand_pixmap(32))
        mark.setFixedSize(32, 32)
        header_layout.addWidget(mark)
        header_layout.addWidget(label("番茄跟踪", "title"))
        self.asset_status_bar = AssetStatusBar()
        header_layout.addWidget(self.asset_status_bar, 1)
        self.import_button = QPushButton("导入视频")
        self.import_button.clicked.connect(self._choose_video)
        self.save_button = QPushButton("保存工程")
        self.save_button.setObjectName("quiet")
        self.save_button.clicked.connect(self._save_workspace)
        self.export_button = QPushButton("导出当前")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self._export)
        self.batch_export_button = QPushButton("批量导出")
        batch_menu = QMenu(self)
        from .batch_export_ui import open_export_queue
        for title, selected in (("导出选中素材…", True), ("导出全部可导出素材…", False)):
            batch_menu.addAction(title).triggered.connect(lambda checked=False, choice=selected: open_export_queue(self, choice))
        self.batch_export_button.setMenu(batch_menu)
        layout.addWidget(header)
        self.workflow_bar = QFrame()
        self.workflow_bar.setObjectName("workflowBar")
        workflow = QHBoxLayout(self.workflow_bar)
        workflow.setContentsMargins(8, 4, 8, 4)
        self.workflow_buttons = []
        for index, title in enumerate(("素材", "跟踪", "构图", "导出")):
            button = QPushButton(title)
            button.setObjectName("workflowStep")
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.clicked.connect(lambda checked=False, step=index: self._workflow_step(step))
            workflow.addWidget(button)
            self.workflow_buttons.append(button)
        workflow.addStretch()
        self.workflow_hint = label("从导入素材开始", "muted", True)
        self.tracking_progress_panel = TrackingProgressPanel()
        self.stage_widget = self.tracking_progress_panel
        self.stage_bars = self.tracking_progress_panel.stage_bars
        self.stage_labels = self.tracking_progress_panel.stage_labels
        self.stage_widget.hide()
        self.tracking_progress_panel.expand_button.toggled.connect(lambda: self._layout_preview_controls())
        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self._left_panel()
        preview = self._center_panel()
        self._right_panel()
        self.workspace_splitter.addWidget(self._reference_sidebar())
        self.workspace_splitter.addWidget(preview)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([308, 1222])
        layout.addWidget(self.workspace_splitter, 1)
        footer = QHBoxLayout()
        self.footer_row=footer
        self.status_label = label("准备就绪  ·  导入素材后选择跟踪目标", "muted")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        self.hardware_brief = label("RAM —  ·  GPU —", "timecode")
        footer.addWidget(self.hardware_brief)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1000)
        self.progress.hide()
        footer.addWidget(self.progress)
        self.cancel_button = QPushButton("取消任务")
        self.cancel_button.setObjectName("quiet")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        self.tracking_actions.addWidget(self.cancel_button)
        footer.addWidget(label(f"TOMATO  /  v{__version__}", "eyebrow"))
        layout.addLayout(footer)
        self.setCentralWidget(root)
        from .workspace_layout import install_workspace
        install_workspace(self)
        from .ad_workspace import install_ad_workspace
        install_ad_workspace(self)
        from .simple_workspace import install_simple_workspace
        install_simple_workspace(self)
        self._install_retime()

    def _panel(self, width):
        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setMinimumWidth(230)
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(13)
        return panel, layout

    def _reference_sidebar(self):
        """Four compact primary steps, with detailed settings one level deeper."""
        panel, outer = self._panel(308)
        panel.setObjectName("referenceSidebar")
        panel.setMinimumWidth(284)
        panel.setMaximumWidth(410)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.workflow_bar.hide()
        self.workflow_hint.hide()
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("番茄跟踪 · 更多设置")
        self.settings_dialog.setModal(False)
        self.settings_dialog.resize(760, 700)
        advanced_layout = QVBoxLayout(self.settings_dialog)
        self.settings_tabs = QTabWidget()
        advanced_layout.addWidget(self.settings_tabs)
        for page, title in ((self.library_tabs.widget(1), "跟踪参数"),
                            (self.library_tabs.widget(2), "复核与关键帧"),
                            (self.inspector_tabs.widget(1), "选区微调"),
                            (self.inspector_tabs.widget(2), "计算与缓存"),
                            (self.inspector_tabs.widget(0), "构图参数"),
                            (self.inspector_tabs.widget(3), "输出设置"),
                            (self.library_tabs.widget(0), "素材与片段")):
            self.settings_tabs.addTab(page, title)
        self.library_tabs.widget(0).widget().layout().addWidget(self.save_button)
        self.library_tabs.widget(0).widget().layout().addWidget(self.range_tools)
        self.library_tabs.widget(1).widget().layout().addWidget(self.preview_toolbar)
        self.library_tabs.widget(1).widget().layout().addWidget(self.preview_tools)
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.horizontalScrollBar().valueChanged.connect(
            lambda value: self.sidebar_scroll.horizontalScrollBar().setValue(0) if value else None)
        self.sidebar_content = QWidget()
        sections = QVBoxLayout(self.sidebar_content)
        sections.setContentsMargins(0, 0, 0, 0)
        sections.setSpacing(0)
        self.sidebar_sections = {}

        def add(key, number, title, page, actions):
            page.setMinimumWidth(0)
            page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            if page.layout():
                page.layout().setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
            menu = QMenu(self)
            for text, callback in actions:
                menu.addAction(text).triggered.connect(lambda checked=False, action=callback: action())
            box = ReferenceStep(number, title, page, menu)
            self.sidebar_sections[key] = box
            sections.addWidget(box)

        materials = QWidget()
        material_layout = QVBoxLayout(materials)
        material_layout.setContentsMargins(0, 0, 0, 0)
        self.clip_list.setMinimumHeight(72)
        self.clip_list.setMaximumHeight(208)
        self.clip_list.setItemDelegate(ClipDelegate(self.clip_list))
        self.clip_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        material_layout.addWidget(self.clip_list)
        row = QHBoxLayout()
        self.import_button.setText("导入素材")
        row.addWidget(self.import_button)
        row.addWidget(self.select_all_clips)
        material_layout.addLayout(row)
        add("materials", "01", "素材", materials,
            (("素材排序、移除与片段…", lambda: self._open_settings(6)),
             ("截取视频片段…", self._edit_trim), ("保存工程 / 序列…", self._save_workspace),
             ("打开合成演示", self.open_demo)))
        self._install_clip_views()
        tracking = QWidget()
        tracking_layout = QVBoxLayout(tracking)
        tracking_layout.setContentsMargins(0, 0, 0, 0)
        tracking_layout.setSpacing(7)
        row = QHBoxLayout()
        self.select_target_button = QPushButton("选跟踪目标")
        self.select_target_button.clicked.connect(self._target_selection_tool)
        row.addWidget(self.select_target_button)
        row.addWidget(self.tracking_combo, 1)
        self.review_button=QPushButton('复核');self.review_button.setToolTip('集中查看预警、跟丢与关键帧（F8）')
        self.review_button.clicked.connect(self._open_review_workbench);row.addWidget(self.review_button)
        tracking_layout.addLayout(row)
        tracking_layout.addWidget(self.track_button)
        self.tracking_actions = QHBoxLayout()
        self.pause_button = QPushButton("暂停检查")
        self.pause_button.clicked.connect(self._pause_tracking)
        self.continue_button = QPushButton("继续跟踪")
        self.continue_button.setObjectName("primary")
        self.continue_button.clicked.connect(self._continue_tracking)
        self.tracking_actions.addWidget(self.pause_button)
        self.tracking_actions.addWidget(self.continue_button)
        self.reset_tracking_button = QPushButton("重置跟踪")
        self.reset_tracking_button.setObjectName("quiet")
        self.reset_tracking_button.setToolTip("清空当前素材的全部跟踪点、轨迹、暂停进度和续跑缓存；保留原片、裁剪、构图参数。运行时请先暂停。")
        self.reset_tracking_button.clicked.connect(self._reset_current_tracking)
        self.tracking_actions.addWidget(self.reset_tracking_button)
        tracking_layout.addLayout(self.tracking_actions)
        self.auto_pause_check = QCheckBox("跟丢 / 持续预警时暂停")
        self.auto_pause_check.setChecked(str(self.preferences.value("pause_on_lost", True)).lower() in {"1", "true"})
        self.auto_pause_check.setToolTip("真实跟丢立即暂停；开启疑似漂移复核后，也会因连续弱匹配暂停。真实跟丢必须重新框选；疑似帧可修正或人工确认。")
        self.auto_pause_check.toggled.connect(lambda checked: self.preferences.setValue("pause_on_lost", checked))
        tracking_layout.addWidget(self.auto_pause_check)
        self.sections_button = QPushButton("独立分段跟踪…")
        self.sections_button.clicked.connect(self._open_tracking_sections)
        tracking_layout.addWidget(self.sections_button)
        tracking_layout.addWidget(self.tracking_progress_panel)
        add("tracking", "02", "跟踪", tracking,
            (("一键跟踪选中素材", self._start_queue),
             ("独立分段跟踪…", self._open_tracking_sections),
             ("跟踪参数与并行设置…", lambda: self._open_settings(0)),
             ("复核与手动关键帧…", lambda: self._open_settings(1)),
             ("放弃暂停，重新跟踪", self._abandon_paused_tracking),
             ("区域形状与数值微调…", lambda: self._open_settings(2)),
             ("硬件、缓存与分段续跑…", lambda: self._open_settings(3))))
        from PySide6.QtGui import QActionGroup
        review_menu=self.sidebar_sections['tracking'].more_button.menu().addMenu('疑似漂移复核')
        self._drift_review_group=QActionGroup(self);self._drift_review_group.setExclusive(True)
        self._drift_review_actions=[]
        for i in range(self.drift_review_combo.count()):
            action=review_menu.addAction(self.drift_review_combo.itemText(i));action.setParent(self)
            action.setCheckable(True);self._drift_review_group.addAction(action)
            action.setChecked(i==self.drift_review_combo.currentIndex())
            action.triggered.connect(lambda checked=False,index=i:self.drift_review_combo.setCurrentIndex(index))
            self._drift_review_actions.append(action)
        self.drift_review_combo.currentIndexChanged.connect(lambda i:[a.setChecked(j==i) for j,a in enumerate(self._drift_review_actions)])
        lock = QWidget()
        lock_layout = QVBoxLayout(lock)
        lock_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.center_button = QPushButton("一键居中")
        self.center_button.clicked.connect(lambda: self._set_composition(.5, .5))
        row.addWidget(self.center_button)
        self.composition_edit.setText("画面拖构图")
        row.addWidget(self.composition_edit)
        lock_layout.addLayout(row)
        self.composition_sliders = []
        for title, spin in (("X", self.target_x_spin), ("Y", self.target_y_spin)):
            row = QHBoxLayout()
            row.addWidget(label(title, "timecode"))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(10000, 90000)
            slider.setValue(round(spin.value() * 1000))
            slider.valueChanged.connect(lambda value, target=spin: target.setValue(value / 1000))
            value_label = label(f"{spin.value():g}%", "timecode")
            value_label.setMinimumWidth(37)
            def sync(value, control=slider, output=value_label):
                control.blockSignals(True)
                control.setValue(round(value * 1000))
                control.blockSignals(False)
                output.setText(f"{value:g}%")
            spin.valueChanged.connect(sync)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            lock_layout.addLayout(row)
            self.composition_sliders.append(slider)
        add("lock", "03", "构图", lock,
            (("锁定方式、放大与比例…", lambda: self._open_settings(4)),))
        output = QWidget()
        output_layout = QHBoxLayout(output)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.export_button.setObjectName("toolButton")
        self.batch_export_button.setMenu(None)
        self.batch_export_button.setText("导出选中")
        from .batch_export_ui import open_export_queue
        self.batch_export_button.clicked.connect(lambda: open_export_queue(self, True))
        output_layout.addWidget(self.export_button)
        output_layout.addWidget(self.batch_export_button)
        add("export", "04", "导出", output,
            (("输出尺寸、画质与声音…", lambda: self._open_settings(5)),
             ("导出全部可导出素材…", lambda: open_export_queue(self, False)),
             ("打开输出目录", self._open_export_folder)))
        sections.addStretch()
        self.sidebar_scroll.setWidget(self.sidebar_content)
        outer.addWidget(self.sidebar_scroll, 1)
        sections.removeWidget(self.sidebar_sections["export"])
        outer.addWidget(self.sidebar_sections["export"])
        self.library_tabs.reveal = lambda index: self._reveal_section("materials") if index == 0 else self._open_settings(index - 1)
        self.inspector_tabs.reveal = lambda index: self._open_settings((4, 2, 3, 5)[index])
        self.reference_sidebar = panel
        return panel

    def _open_settings(self, index):
        self.settings_tabs.setCurrentIndex(index)
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _target_selection_tool(self):
        self.composition_edit.setChecked(False)
        self.pick_mode.setCurrentIndex(self.pick_mode.findData("box"))
        self.canvas.setFocus()

    def _reveal_section(self, key):
        section_widget = self.sidebar_sections[key]
        section_widget.set_expanded(True)
        if self.sidebar_content.isAncestorOf(section_widget):
            self.sidebar_scroll.ensureWidgetVisible(section_widget, 0, 12)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'header') and not getattr(self, '_ad_installed', False):
            self.header.setFixedHeight(max(64, min(88, round(self.height() * .08))))
        self._layout_preview_controls()
        self._region_header_layout()
        if getattr(self, '_ad_installed', False):
            from .ad_workspace import ad_resize
            ad_resize(self)
        if hasattr(self,'_retime_preview'):
            self._retime_sync()

    def _layout_preview_controls(self):
        if not hasattr(self, "range_tools"):
            return
        expanded_small = (self.height() < 880 and not self.stage_widget.isHidden()
                          and self.tracking_progress_panel.expanded)
        for editing in (self.preview_toolbar, self.preview_tools, self.range_tools, self.track_legend):
            editing.setVisible(not self._tracking_active and not expanded_small and
                               not (editing is self.track_legend and getattr(self, "_ad_installed", False)))

    def _left_panel(self):
        panel, outer = self._panel(266)
        outer.setContentsMargins(10, 10, 10, 10)
        self.library_tabs = SectionNavigator(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(12)
        scroll.setWidget(content)
        self.library_tabs.addTab(scroll, "素材")
        section(layout, "01", "素材库")
        self._sequence_panel(layout)
        self.file_label = self.asset_status_bar.filename_label
        self.meta_label = self.asset_status_bar.metadata_label
        self.trim_button = QPushButton("截取视频片段")
        self.trim_button.clicked.connect(self._edit_trim)
        layout.addWidget(self.trim_button)
        self.trim_summary = label("先留下需要的镜头，再开始跟踪。", "muted", True)
        layout.addWidget(self.trim_summary)
        self.demo_button = QPushButton("试用合成演示")
        self.demo_button.setObjectName("quiet")
        self.demo_button.clicked.connect(self.open_demo)
        layout.addWidget(self.demo_button)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 8, 5, 0)
        layout.setSpacing(12)
        scroll.setWidget(content)
        self.library_tabs.addTab(scroll, "跟踪")
        section(layout, "02", "目标跟踪")
        self.tracking_combo = QComboBox()
        self.tracking_combo.addItem("普通跟踪", "normal")
        self.tracking_combo.addItem("精准跟踪 · 加强", "precision")
        self.tracking_combo.currentIndexChanged.connect(self._tracking_mode_changed)
        layout.addWidget(self.tracking_combo)
        self.tracking_hint = label("快速分析清晰目标，适合日常镜头。", "muted", True)
        layout.addWidget(self.tracking_hint)
        motion_row = QHBoxLayout()
        motion_row.addWidget(label("运动模型", "muted"))
        self.perspective_check = QCheckBox("透视形变")
        self.perspective_check.setToolTip(
            "勾选后跟踪器额外解算画面透视形变（近大远小、四边形扭曲），\n"
            "适合翻转广告牌、侧转屏幕、地面等三维角度变化的表面。\n"
            "勾选分量越多解算越复杂，简单平面运动时建议关闭以更稳。")
        self.perspective_check.toggled.connect(self._tracking_mode_changed)
        motion_row.addWidget(self.perspective_check)
        motion_row.addStretch()
        layout.addLayout(motion_row)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(label("平滑等级", "muted"))
        self.smoothness_combo = QComboBox()
        for value, title in ((1, "1 · 极敏感"), (2, "2 · 敏感"), (3, "3 · 中等"),
                              (4, "4 · 平稳"), (5, "5 · 极平稳")):
            self.smoothness_combo.addItem(title, value)
        self.smoothness_combo.setCurrentIndex(2)
        self.smoothness_combo.setToolTip(
            "控制跟踪器假设帧间运动变化的剧烈程度。\n"
            "数值越高：假设运动平滑连续，能滤除噪点、抖动干扰，路径更稳，适合匀速镜头。\n"
            "数值越低：允许运动在帧间突变，适合目标运动极快或方向突变，但更易受噪声影响跟丢。")
        self.smoothness_combo.currentIndexChanged.connect(self._tracking_mode_changed)
        smooth_row.addWidget(self.smoothness_combo)
        smooth_row.addStretch()
        layout.addLayout(smooth_row)
        self.track_button = QPushButton("开始跟踪")
        self.segmented_check = QCheckBox("分段检查点 · 自动保存续跑点")
        self.segmented_check.setToolTip("按段落盘，取消或异常退出后，使用相同视频、选区和参数重新跟踪可继续。未完成段会重算，不重置跟踪参考。")
        self.segmented_check.toggled.connect(self._compute_changed)
        layout.addWidget(self.segmented_check)
        self.segment_frames = QSpinBox()
        self.segment_frames.setRange(10, 3600)
        self.segment_frames.setValue(300)
        self.segment_frames.setSuffix(" 帧 / 段")
        self.segment_frames.valueChanged.connect(self._compute_changed)
        layout.addWidget(self.segment_frames)
        self.drift_review_combo=QComboBox()
        for text,value in (('关闭疑似漂移复核','off'),('宽松 · 少打断','relaxed'),('标准 · 推荐','standard'),('严格 · 更早复核','strict')):
            self.drift_review_combo.addItem(text,value)
        self.drift_review_combo.setCurrentIndex(2)
        self.drift_review_combo.setToolTip('根据连续匹配证据提示疑似漂移；不是目标识别概率。开启跟丢自动暂停时，也会停在疑似帧等待复核。')
        self.drift_review_combo.currentIndexChanged.connect(self._compute_changed)
        layout.addWidget(label('疑似漂移复核灵敏度'))
        layout.addWidget(self.drift_review_combo)
        self.track_button.setObjectName("trackAction")
        self.track_button.clicked.connect(self._start_tracking)
        layout.addWidget(self.track_button)
        layout.addWidget(label("批量跟踪", "panelTitle"))
        layout.addWidget(self.queue_mode)
        layout.addWidget(self.queue_button)
        self.live_preview_check = QCheckBox("实时预览计算中的画面")
        self.live_preview_check.setChecked(True)
        self.live_preview_check.setToolTip("显示实际分析到的帧与目标框；界面限速刷新，避免预览拖慢跟踪。")
        layout.addWidget(self.live_preview_check)
        self.live_preview_status = label("开始跟踪后，预览随实际计算帧更新。", "muted", True)
        layout.addWidget(self.live_preview_status)
        self.summary_label = label("在预览中选中车标、文字或清晰边缘。", "muted", True)
        layout.addWidget(self.summary_label)
        layout.addSpacing(5)
        layout.addStretch()
        review_page = QWidget()
        layout = QVBoxLayout(review_page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(label("点击问题帧检查，重新框选后重算。", "muted", True))
        self.library_tabs.addTab(review_page, "复核")
        self.marker_tabs = QTabWidget()
        anchors_page = QWidget()
        anchors_layout = QVBoxLayout(anchors_page)
        anchors_layout.setContentsMargins(0, 0, 0, 0)
        anchors_layout.setSpacing(8)
        anchors_layout.addWidget(label("重新选点可纠正当前位置。", "muted", True))
        self.anchor_list = QListWidget()
        self.anchor_list.setMinimumHeight(86)
        self.anchor_list.itemClicked.connect(lambda item: self._seek(item.data(Qt.ItemDataRole.UserRole)))
        anchors_layout.addWidget(self.anchor_list, 1)
        self.remove_anchor = QPushButton("删除选中关键帧")
        self.remove_anchor.setObjectName("quiet")
        self.remove_anchor.clicked.connect(self._delete_anchor)
        anchors_layout.addWidget(self.remove_anchor)
        self.marker_tabs.addTab(anchors_page, "关键帧")
        issues_page = QWidget()
        issues_layout = QVBoxLayout(issues_page)
        issues_layout.setContentsMargins(0, 0, 0, 0)
        issues_layout.setSpacing(8)
        issues_layout.addWidget(label("点击问题位置，补关键帧后重新跟踪。", "muted", True))
        self.issue_list = QListWidget()
        self.issue_list.setMinimumHeight(86)
        self.issue_list.itemClicked.connect(lambda item: self._seek(item.data(Qt.ItemDataRole.UserRole)))
        issues_layout.addWidget(self.issue_list, 1)
        self.marker_tabs.addTab(issues_page, "待修正")
        layout.addWidget(self.marker_tabs, 1)
        return panel

    def _center_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.preview_head = QFrame()
        self.preview_head.installEventFilter(self)
        self.preview_head.setObjectName('preview-head')
        heading = QHBoxLayout(self.preview_head)
        heading.setContentsMargins(8, 4, 8, 4)
        heading.setSpacing(6)
        self.monitor_filename = label("尚未导入素材", "muted")
        self.monitor_filename.setMinimumWidth(44)
        self.monitor_filename.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        heading.addWidget(self.monitor_filename, 1)
        self.region_toolbar = RegionToolbar()
        self.region_toolbar.shape_requested.connect(self._region_choose_shape)
        self.region_toolbar.action_requested.connect(self._region_action)
        heading.addWidget(self.region_toolbar)
        heading.addStretch()
        self.compare_hold_button = QPushButton("按住对比")
        self.compare_hold_button.pressed.connect(self._hold_source)
        self.compare_hold_button.released.connect(self._release_source)
        self.compare_hold_button.hide()
        self.sync_label = label('原片 / 效果预览 · 同步检查', 'muted')
        heading.addWidget(self.sync_label)
        layout.addWidget(self.preview_head)
        self.preview_toolbar = QWidget()
        toolbar = QHBoxLayout(self.preview_toolbar)
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(label("预览", "panelTitle"))
        self.pick_mode = QComboBox()
        self.pick_mode.addItem("框选细节", "box")
        self.pick_mode.addItem("点选细节", "point")
        self.pick_mode.addItem("调整选区", "adjust")
        self.pick_mode.setToolTip("借助原片缩放和局部放大镜，点击车标、文字等明确的细节")
        self.pick_mode.currentIndexChanged.connect(self._pick_mode_changed)
        toolbar.addWidget(self.pick_mode)
        self.point_size = QSpinBox()
        self.point_size.setRange(24, 160)
        self.point_size.setSingleStep(8)
        self.point_size.setValue(64)
        self.point_size.setSuffix(" px")
        self.point_size.setToolTip("点选时的跟踪区域大小；尽量贴住目标细节，减少框内背景")
        self.point_size.valueChanged.connect(self._pick_mode_changed)
        toolbar.addWidget(self.point_size)
        self.mesh_overlay_check = QCheckBox("网格")
        self.mesh_overlay_check.setChecked(True)
        self.mesh_overlay_check.setToolTip("显示已计算的区域证据网格；绿较强、红较弱、灰无足够纹理，不代表成功率。")
        self.mesh_overlay_check.toggled.connect(lambda: self._show_current() if self.info else None)
        toolbar.addWidget(self.mesh_overlay_check)
        toolbar.addStretch()
        self.view_buttons = {}
        for key, text in (("source", "原片"), ("output", "稳定后"), ("compare", "对比")):
            button = QPushButton(text)
            button.setObjectName("viewButton")
            button.setCheckable(True)
            button.setChecked(key == "compare")
            button.clicked.connect(lambda checked=False, mode=key: self._set_view(mode))
            button.hide()
            self.view_buttons[key] = button
        layout.addWidget(self.preview_toolbar)
        self.canvas = PreviewCanvas()
        self.canvas.setMinimumHeight(160)
        self.canvas.persistent_compare = True
        self.canvas.mode = "compare"
        self.canvas.setObjectName("previewCanvas")
        self.canvas.roi_selected.connect(self._select_roi)
        self.canvas.region_selected.connect(self._region_proposal)
        self.canvas.vertex_selection_changed.connect(lambda index: self._region_sync())
        self.canvas.edit_started.connect(self._stop_play)
        self.canvas.selection_rejected.connect(lambda text: self.status_label.setText(text))
        self.canvas.playback_requested.connect(self._toggle_play)
        for title, key, command in (('撤销区域编辑', 'Ctrl+Z', 'undo'), ('重做区域编辑', 'Ctrl+Shift+Z', 'redo')):
            action = QAction(title, self.canvas)
            action.setShortcut(key)
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            action.triggered.connect(lambda checked=False,c=command:self._tb_operation(c))
            self.canvas.addAction(action)
        monitor = self.preview_host = QFrame()
        monitor.setObjectName("monitor")
        monitor.setMinimumHeight(190)
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(1, 1, 1, 1)
        monitor_layout.addWidget(self.canvas)
        layout.addWidget(monitor, 1)
        self.preview_tools = QWidget()
        zoom_row = QHBoxLayout(self.preview_tools)
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_row.addWidget(label("原片缩放", "muted"))
        self.preview_minus = QPushButton("−")
        self.preview_minus.setFixedWidth(32)
        self.preview_minus.setObjectName("toolButton")
        self.preview_minus.clicked.connect(lambda: self.canvas.set_zoom(self.canvas.zoom_factor / 1.25))
        zoom_row.addWidget(self.preview_minus)
        self.preview_zoom = QDoubleSpinBox()
        self.preview_zoom.setRange(1., 8.)
        self.preview_zoom.setSingleStep(.25)
        self.preview_zoom.setDecimals(2)
        self.preview_zoom.setSuffix(" ×")
        self.preview_zoom.setFixedWidth(98)
        self.preview_zoom.valueChanged.connect(lambda value: self.canvas.set_zoom(value))
        self.canvas.zoom_changed.connect(self._preview_zoom_changed)
        zoom_row.addWidget(self.preview_zoom)
        self.preview_plus = QPushButton("+")
        self.preview_plus.setFixedWidth(32)
        self.preview_plus.setObjectName("toolButton")
        self.preview_plus.clicked.connect(lambda: self.canvas.set_zoom(self.canvas.zoom_factor * 1.25))
        zoom_row.addWidget(self.preview_plus)
        self.preview_fit = QPushButton("适应画面")
        self.preview_fit.setObjectName("toolButton")
        self.preview_fit.clicked.connect(self.canvas.reset_view)
        zoom_row.addWidget(self.preview_fit)
        zoom_row.addStretch()
        zoom_row.addWidget(label("跟踪区域", "muted"))
        self.shape_combo = QComboBox()
        for text, data in (("矩形", "rectangle"), ("圆形（兼容）", "circle"), ("椭圆", "ellipse"), ('多边形', 'polygon')):
            self.shape_combo.addItem(text, data)
        self.shape_combo.currentIndexChanged.connect(self._shape_changed)
        zoom_row.addWidget(self.shape_combo)
        self.preview_tools.setToolTip("滚轮缩放 · 中键拖动查看；选区控制点可拖动，方向键微调，Shift 加速。")
        layout.addWidget(self.preview_tools)
        self.drift_warning_banner=QLabel()
        self.drift_warning_banner.setObjectName('driftWarningBanner')
        self.drift_warning_banner.setWordWrap(True)
        self.drift_warning_banner.setTextFormat(Qt.TextFormat.PlainText)
        self.drift_warning_banner.setMinimumHeight(34)
        self.drift_warning_banner.setStyleSheet('QLabel#driftWarningBanner {color:#ffe3a0;background:#493719;border:1px solid #b78435;border-radius:5px;padding:6px;font-weight:600;}')
        layout.addWidget(self.drift_warning_banner);self.drift_warning_banner.hide()
        self.preview_note = label("框选跟踪目标；左侧按素材、跟踪、构图、导出依次调整。", "muted", True)
        layout.addWidget(self.preview_note)
        timeline_well = self.timeline_well = QFrame()
        timeline_well.setObjectName("timelineWell")
        layout.addWidget(timeline_well)
        layout = QVBoxLayout(timeline_well)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        transport = QHBoxLayout()
        self.previous_button = QPushButton("上一帧")
        self.previous_button.clicked.connect(lambda: self._step_preview(-1))
        self.play_button = QPushButton("播放")
        self.play_button.setObjectName("transport")
        self.play_button.clicked.connect(self._toggle_play)
        self.next_button = QPushButton("下一帧")
        self.next_button.clicked.connect(lambda: self._step_preview(1))
        for button in (self.previous_button, self.play_button, self.next_button):
            transport.addWidget(button)
        self.loop_button = QCheckBox("循环")
        self.loop_button.toggled.connect(lambda value:self._tb_operation('loop'))
        transport.addWidget(self.loop_button)
        self.play_range_button = QPushButton("范围 / 关键帧…")
        self.play_range_button.clicked.connect(self._tb_menu)
        transport.addWidget(self.play_range_button)
        for text, factor in (("−", .8), ("适配", 0), ("+", 1.25)):
            button = QPushButton(text)
            button.setMaximumWidth(42)
            button.clicked.connect(lambda checked=False,f=factor:self.strip.zoom(f) if f else self.strip.fit())
            transport.addWidget(button)
        self.time_units = QComboBox()
        self.time_units.addItems(["帧", "时间码"])
        self.time_units.currentIndexChanged.connect(self._tb_units)
        transport.addWidget(self.time_units)
        transport.addStretch()
        self.preview_quality = QComboBox()
        self.preview_quality.addItem("自动高清", "auto")
        self.preview_quality.addItem("流畅预览", "proxy")
        self.preview_quality.setToolTip("自动高清：暂停/逐帧/选点时后台读取原片，播放与跟踪时使用流畅代理。只影响查看，不改变跟踪或导出。")
        self.preview_quality.setCurrentIndex(1 if self.preferences.value("preview/quality", "auto") == "proxy" else 0)
        self.preview_quality.currentIndexChanged.connect(self._preview_quality_changed)
        transport.addWidget(self.preview_quality)
        self.time_label = label("00:00.000 / 00:00.000", "timecode")
        transport.addWidget(self.time_label)
        layout.addLayout(transport)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.sliderPressed.connect(self._stop_play)
        self.timeline.valueChanged.connect(self._timeline_seek)
        self.timeline.hide()  # Existing navigation API; the visible ruler owns pointer interaction.
        self.strip = Timebar()
        self.strip.seek.connect(self._seek)
        self.strip.scrubStarted.connect(self._stop_play)
        self.strip.operation.connect(self._tb_operation)
        self.strip.viewChanged.connect(self._tb_view_changed)
        layout.addWidget(self.strip)
        self.range_tools = QWidget()
        range_row = QHBoxLayout(self.range_tools)
        range_row.setContentsMargins(0, 0, 0, 0)
        range_row.addWidget(label("处理区间", "muted"))
        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(1, 1)
            spin.setSuffix(" 帧")
            spin.valueChanged.connect(self._range_changed)
        range_row.addWidget(self.start_spin)
        self.in_button = QPushButton("入点")
        self.in_button.clicked.connect(lambda: self._set_range_point(True))
        range_row.addWidget(self.in_button)
        range_row.addWidget(label("—", "muted"))
        range_row.addWidget(self.end_spin)
        self.out_button = QPushButton("出点")
        self.out_button.clicked.connect(lambda: self._set_range_point(False))
        range_row.addWidget(self.out_button)
        range_row.addStretch()
        layout.addWidget(self.range_tools)
        self.track_legend = label("Alt+滚轮缩放 · 中键平移 · I/O播放范围 · 菱形为修正点（Delete只删修正点，不删视频帧）", "muted", True)
        layout.addWidget(self.track_legend)
        return panel

    def eventFilter(self, watched, event):
        if watched is getattr(self, 'preview_head', None) and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._region_header_layout)
        return super().eventFilter(watched, event)

    def _hold_source(self):
        self._before_hold_view = self.canvas.mode
        if not self.busy:
            self.canvas.mode = "source"
            self._show_current()

    def _release_source(self):
        if not self.busy:
            self.canvas.mode = getattr(self, "_before_hold_view", "source")
            self._show_current()

    def _slider(self, layout, title, low, high, value, formatter):
        row = QHBoxLayout()
        row.addWidget(label(title))
        row.addStretch()
        value_label = label(formatter(value), "accent")
        row.addWidget(value_label)
        layout.addLayout(row)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(lambda v: value_label.setText(formatter(v)))
        slider.valueChanged.connect(self._settings_changed)
        layout.addWidget(slider)
        return slider

    def _right_panel(self):
        panel, outer = self._panel(252)
        section(outer, "03", "画面设置")
        self.inspector_tabs = SectionNavigator(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(13)
        layout.addWidget(label("稳定方式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("主体锁定", "lock")
        self.mode_combo.addItem("平滑跟随", "follow")
        self.mode_combo.currentIndexChanged.connect(self._settings_changed)
        layout.addWidget(self.mode_combo)
        self.mode_hint = label("把选中的细节固定在画面指定位置。", "muted", True)
        layout.addWidget(self.mode_hint)
        self.strength_slider = self._slider(layout, "跟随强度", 0, 100, 100, lambda v: f"{v}%")
        self.smoothing_slider = self._slider(layout, "平滑时间", 5, 150, 35, lambda v: f"{v / 100:.2f} 秒")
        self.smoothing_slider.setToolTip("用于平滑跟随模式。主体锁定直接补偿目标位置，不使用这项参数。")
        layout.addWidget(label("锁定位置"))
        self.target_combo = QComboBox()
        self.target_combo.addItem("画面中央", (0.5, 0.5))
        self.target_combo.addItem("左侧三分线", (1 / 3, 0.5))
        self.target_combo.addItem("右侧三分线", (2 / 3, 0.5))
        self.target_combo.currentIndexChanged.connect(self._target_preset)
        layout.addWidget(self.target_combo)
        self.target_x_spin, self.target_y_spin = QDoubleSpinBox(), QDoubleSpinBox()
        for title, spin in (("X · 构图位置", self.target_x_spin), ("Y · 构图位置", self.target_y_spin)):
            spin.setRange(10, 90)
            spin.setDecimals(3)
            spin.setValue(50)
            spin.setSuffix(" %")
            spin.valueChanged.connect(self._settings_changed)
            layout.addWidget(label(title))
            layout.addWidget(spin)
        center = QPushButton("一键居中")
        center.clicked.connect(lambda: self._set_composition(.5, .5))
        layout.addWidget(center)
        self.composition_edit = QCheckBox("效果预览拖动构图锚点")
        self.composition_edit.setToolTip("只修改输出构图；原片中的框选和关键帧保持独立。")
        self.composition_edit.toggled.connect(self._composition_edit_changed)
        self.canvas.composition_changed.connect(self._set_composition)
        layout.addWidget(self.composition_edit)
        divider(layout)
        self.zoom_slider = self._slider(layout, "放大上限", 100, 300, 180, lambda v: f"{v / 100:.2f}×")
        self.crop_label = label("跟踪完成后计算所需裁切。", "muted", True)
        layout.addWidget(self.crop_label)
        layout.addWidget(label("画面比例"))
        self.aspect_combo = QComboBox()
        for text, value in (("保持原比例", "original"), ("竖屏 9:16", "9:16"),
                            ("横屏 16:9", "16:9"), ("方形 1:1", "1:1")):
            self.aspect_combo.addItem(text, value)
        self.aspect_combo.currentIndexChanged.connect(self._settings_changed)
        layout.addWidget(self.aspect_combo)
        layout.addStretch()
        scroll.setWidget(content)
        self.inspector_tabs.addTab(scroll, "构图")
        self._region_panel()
        self._performance_panel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(13)
        layout.addWidget(label("视频输出", "panelTitle"))
        from .batch_export_ui import open_export_queue
        for title, selected in (("导出选中素材…", True), ("导出全部可导出素材…", False)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, choice=selected: open_export_queue(self, choice))
            layout.addWidget(button)
        self.export_audio_hint = label("从原片渲染，保留所选片段的声音。", "muted", True)
        layout.addWidget(self.export_audio_hint)
        divider(layout)
        layout.addWidget(label("输出尺寸"))
        self.resolution_combo = QComboBox()
        for text, value in (("长边 1920", 1920), ("长边 1080", 1080), ("长边 3840", 3840), ("原片尺寸", 0)):
            self.resolution_combo.addItem(text, value)
        self.resolution_combo.currentIndexChanged.connect(self._settings_changed)
        layout.addWidget(self.resolution_combo)
        layout.addWidget(label("不会主动放大超出原片像素尺寸。", "muted", True))
        layout.addWidget(label("画质"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItem("高画质 · 文件较大", 18)
        self.quality_combo.addItem("均衡 · 日常分享", 21)
        self.quality_combo.addItem("最高画质 · 文件更大", 15)
        self.quality_combo.currentIndexChanged.connect(self._settings_changed)
        layout.addWidget(self.quality_combo)
        divider(layout)
        self.audio_policy_label = label("MP4 / H.264  ·  保留原片声音", "muted", True)
        layout.addWidget(self.audio_policy_label)
        layout.addWidget(label("预览不播放声音；成品声音按当前时间映射策略处理。", "muted", True))
        layout.addStretch()
        self.folder_button = QPushButton("打开导出文件夹")
        self.folder_button.setObjectName("quiet")
        self.folder_button.clicked.connect(self._open_export_folder)
        layout.addWidget(self.folder_button)
        scroll.setWidget(content)
        self.inspector_tabs.addTab(scroll, "导出")
        return panel

    def settings(self):
        x, y = self.target_x_spin.value() / 100, self.target_y_spin.value() / 100
        return Settings(self.mode_combo.currentData(), self.strength_slider.value() / 100,
                        self.smoothing_slider.value() / 100, self.zoom_slider.value() / 100,
                        self.aspect_combo.currentData(), self.resolution_combo.currentData(),
                        x, y, self.quality_combo.currentData(), self.tracking_combo.currentData(),
                        self.shape_combo.currentData(), self.acceleration_combo.currentData(),
                        self.memory_spin.value(), self.threads_spin.value(),
                        self.perspective_check.isChecked(), self.smoothness_combo.currentData(),
                        self.adaptive_reference_check.isChecked(), self.planar_region_check.isChecked(),
                        self.segmented_check.isChecked(), self.segment_frames.value(), self.mesh_compute_check.isChecked(),
                        self._polygon_points, retime=deepcopy(getattr(self,'_retime_config',{})),
                        drift_review=self.drift_review_combo.currentData())

    def _region_panel(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.addWidget(label("选区微调", "panelTitle"))
        layout.addWidget(label('区域形状（与预览工具栏同步）'))
        layout.addWidget(self.shape_combo)
        layout.addWidget(label("以原片像素设置位置和大小；应用后成为当前帧的手动关键帧。", "muted", True))
        self.region_fields = []
        for title in ("X · 左边", "Y · 顶边", "W · 宽度", "H · 高度"):
            layout.addWidget(label(title))
            spin = QDoubleSpinBox()
            spin.setRange(0, 100000)
            spin.setDecimals(1)
            spin.setSingleStep(.1)
            spin.setSuffix(" px")
            layout.addWidget(spin)
            self.region_fields.append(spin)
        self.region_apply = QPushButton("应用选区调整")
        self.region_apply.clicked.connect(self._apply_region)
        layout.addWidget(self.region_apply)
        layout.addWidget(label("圆形以宽度为直径。\n方向键：1 px\nShift + 方向键：10 px\nAlt + 方向键：0.1 px\nEsc：取消当前拖动", "muted", True))
        layout.addStretch()
        self.inspector_tabs.addTab(page, "区域")

    def _performance_panel(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.addWidget(label("计算设备", "panelTitle"))
        self.acceleration_combo = QComboBox()
        for title, value in (("自动 · 推荐", "auto"), ("GPU 优先", "gpu"), ("仅 CPU · 兼容", "cpu")):
            self.acceleration_combo.addItem(title, value)
        layout.addWidget(self.acceleration_combo)
        layout.addWidget(label("系统内存预算上限"))
        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(20, 80)
        self.memory_spin.setValue(80)
        self.memory_spin.setSuffix(" %")
        layout.addWidget(self.memory_spin)
        layout.addWidget(label("CPU 计算线程"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(min(8, os.cpu_count() or 1))
        layout.addWidget(self.threads_spin)
        self.planar_region_check = QCheckBox("平面区域跟踪 · 形状遮罩")
        self.planar_region_check.setChecked(
            str(self.preferences.value("planar_region", False)).lower() in {"true", "1"})
        self.planar_region_check.setToolTip("使用形状内像素细化与独立区域验证；关闭时保留原点跟踪路径。")
        self.planar_region_check.toggled.connect(self._planar_region_changed)
        layout.addWidget(self.planar_region_check)
        self.planar_preset = QPushButton("使用平面跟踪预设")
        self.planar_preset.setToolTip("开启精准模式、形状遮罩区域跟踪、自适应外观和完整单应矩阵。")
        self.planar_preset.clicked.connect(self._apply_planar_preset)
        layout.addWidget(self.planar_preset)
        self.adaptive_reference_check = QCheckBox("自适应参考外观")
        self.adaptive_reference_check.setChecked(
            str(self.preferences.value("adaptive_reference", False)).lower() in {"true", "1"})
        self.adaptive_reference_check.setToolTip(
            "仅在连续高可信度帧缓慢更新外观，并始终保留首次选区作校验。模糊、跟丢和重新捕获时不学习。修改后需要重新跟踪。")
        self.adaptive_reference_check.toggled.connect(self._adaptive_reference_changed)
        layout.addWidget(self.adaptive_reference_check)
        self.mesh_compute_check = QCheckBox("计算置信度网格 · 4 × 4")
        self.mesh_compute_check.setToolTip("计算遮罩内各单元的图像相关证据。低纹理显示灰色，不将网格值当成正确概率。修改需重新跟踪。")
        self.mesh_compute_check.toggled.connect(self._tracking_mode_changed)
        layout.addWidget(self.mesh_compute_check)
        self.clear_checkpoint_button = QPushButton("清除当前素材续跑缓存")
        self.clear_checkpoint_button.clicked.connect(self._clear_tracking_checkpoints)
        layout.addWidget(self.clear_checkpoint_button)
        layout.addWidget(label("外观渐变时可开启；原始参考会持续校验，严重遮挡仍需补关键帧。", "muted", True))
        layout.addWidget(label("内存按需缓存，接近预算时释放；系统和其他软件的占用无法强制限制。GPU 按任务需要使用，最高可用 100%，不设置负载上限。", "muted", True))
        layout.addWidget(label("精准搜索使用 OpenCL；导出自动尝试硬件编码。不支持时回退 CPU。", "muted", True))
        divider(layout)
        self.hardware_details = label("正在读取本机硬件…", "muted", True)
        layout.addWidget(self.hardware_details)
        self.performance_label = label("完成跟踪后显示实际计算速度。", "muted", True)
        layout.addWidget(self.performance_label)
        self.log_button = QPushButton("打开日志文件夹")
        self.log_button.setObjectName("quiet")
        self.log_button.clicked.connect(self._open_log_folder)
        layout.addWidget(self.log_button)
        divider(layout)
        layout.addWidget(label("缓存位置", "panelTitle"))
        layout.addWidget(label("分析预览和高清代理存放在此目录。默认在系统盘，可改到其他盘避免堆积。", "muted", True))
        self.cache_dir_label = label("…", "muted", True)
        self.cache_dir_label.setWordWrap(True)
        layout.addWidget(self.cache_dir_label)
        cache_row = QHBoxLayout()
        self.cache_change_button = QPushButton("更改…")
        self.cache_change_button.setObjectName("toolButton")
        self.cache_change_button.clicked.connect(self._change_cache_dir)
        cache_row.addWidget(self.cache_change_button)
        self.cache_default_button = QPushButton("恢复默认")
        self.cache_default_button.setObjectName("toolButton")
        self.cache_default_button.clicked.connect(self._reset_cache_dir)
        cache_row.addWidget(self.cache_default_button)
        cache_row.addStretch()
        layout.addLayout(cache_row)
        self._refresh_cache_label()
        layout.addStretch()
        for control in (self.acceleration_combo, self.memory_spin, self.threads_spin):
            signal = control.currentIndexChanged if isinstance(control, QComboBox) else control.valueChanged
            signal.connect(self._compute_changed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.inspector_tabs.addTab(scroll, "性能")

    def _adaptive_reference_changed(self, *args):
        if not self._ready:
            return
        self.preferences.setValue("adaptive_reference", self.adaptive_reference_check.isChecked())
        self._tracking_mode_changed()

    def _clear_tracking_checkpoints(self):
        if self.busy or not self.info:
            return
        from .checkpoints import clear_checkpoints
        count = clear_checkpoints(self.info)
        self.status_label.setText(f"已清除当前素材的 {count} 个续跑缓存；工程和视频保留")

    def _planar_region_changed(self, *args):
        if not self._ready:
            return
        self.preferences.setValue("planar_region", self.planar_region_check.isChecked())
        self._tracking_mode_changed()

    def _apply_planar_preset(self):
        if self.busy:
            return
        self._ready = False
        self.planar_region_check.setChecked(True)
        self.adaptive_reference_check.setChecked(True)
        self.perspective_check.setChecked(True)
        self.tracking_combo.setCurrentIndex(self.tracking_combo.findData("precision"))
        self._ready = True
        self.preferences.setValue("planar_region", True)
        self.preferences.setValue("adaptive_reference", True)
        self._tracking_mode_changed()

    def _compute_changed(self, *args):
        if self._ready:
            self.dirty = self.info is not None
            self._sync_ui()

    def _hardware_updated(self, counters):
        memory = counters["memory"]
        gpu = counters.get("gpu")
        ram = f"RAM {memory['percent']:.0f}%" if memory["total"] else "RAM —"
        self.hardware_brief.setText(ram + (f"  ·  GPU {gpu['percent']:.0f}%" if gpu else "  ·  GPU —"))
        details = (f"系统内存 {memory['used'] / 1024**3:.1f} / {memory['total'] / 1024**3:.1f} GB" if memory["total"] else "内存计数器不可用")
        details += (f"\n{gpu['name']}\n显存 {gpu['used_mb'] / 1024:.1f} / {gpu['total_mb'] / 1024:.1f} GB" if gpu else "\nGPU 占用计数器不可用；加速能力会在任务中检测。")
        if memory["total"] and not cache_budget(self.memory_spin.value(), memory):
            details += "\n系统内存余量不足，暂停新增帧缓存。"
        self.hardware_details.setText(details)

    def _shape_changed(self, *args):
        if not self._ready:
            return
        self._region_choose_shape(self.shape_combo.currentData())

    def _apply_region(self):
        if not self.info or self.busy or self.canvas.box is None:
            return
        x, y, w, h = [spin.value() for spin in self.region_fields]
        if self.shape_combo.currentData() == "circle":
            w = h = min(w, self.info.width - x, self.info.height - y)
        w, h = min(w, self.info.width - x), min(h, self.info.height - y)
        if w <= 0 or h <= 0:
            self.status_label.setText("选区必须位于画面内，且宽高大于零")
            return
        self._select_roi((x / self.info.width, y / self.info.height, w / self.info.width, h / self.info.height))

    def _sync_ui(self):
        has_info = self.info is not None
        idle = not self.busy
        self.import_button.setEnabled(idle)
        self.import_action.setEnabled(idle)
        self.open_action.setEnabled(idle)
        self.demo_action.setEnabled(idle)
        self.demo_button.setEnabled(idle)
        self.trim_button.setEnabled(has_info and idle)
        self.trim_action.setEnabled(has_info and idle)
        self.save_button.setEnabled(has_info and idle)
        self.save_action.setEnabled(has_info and idle)
        active_anchors = has_info and any(self.start_spin.value() - 1 <= f <= self.end_spin.value() - 1 for f in self.anchors)
        self.track_button.setEnabled(bool(active_anchors) and idle)
        self.export_button.setEnabled(has_info and is_complete(self.track) and not self.track.lost_frames and idle and not self._region_stale())
        self.batch_export_button.setEnabled(bool(self.clips) and idle)
        paused = self._paused_tracking
        self.pause_button.setVisible(self._tracking_active)
        self.pause_button.setEnabled(self._tracking_active and self.worker is not None and self.worker.supports_pause)
        self.continue_button.setVisible(paused is not None)
        from .tracking_pause import can_resume_without_edit
        self.continue_button.setEnabled(bool(paused and (can_resume_without_edit(paused) or paused.get('reason')=='review')) and idle)
        self.continue_button.setText('确认并继续' if paused and paused.get('reason')=='review' and not paused.get('corrected') else '继续跟踪')
        self.continue_button.setToolTip('确认此帧目标框正确，将其作为人工关键帧继续；若跟偏，请先重新框选。' if paused and paused.get('reason')=='review' else '暂停检查可直接继续，也可重新框选修正。' if can_resume_without_edit(paused) else '修正暂停帧后继续跟踪')
        self.drift_review_combo.setEnabled(idle)
        for action in getattr(self,'_drift_review_actions',[]):action.setEnabled(idle)
        self.track_button.setVisible(paused is None)
        self.auto_pause_check.setEnabled(idle)
        self.select_target_button.setEnabled(has_info and idle)
        self.reset_tracking_button.setEnabled(has_info and idle and self.worker is None and self._clip() is not None)
        self.center_button.setEnabled(has_info and idle)
        self.live_preview_check.setEnabled(idle)
        self.remove_anchor.setEnabled(bool(self.anchors) and idle)
        self.canvas.selectable = has_info and idle
        self.pick_mode.setEnabled(has_info and idle)
        self.point_size.setEnabled(has_info and idle and self.pick_mode.currentData() == "point")
        self.tracking_combo.setEnabled(idle)
        for control in (self.segmented_check, self.segment_frames, self.mesh_compute_check):
            control.setEnabled(idle)
        self.clear_checkpoint_button.setEnabled(has_info and idle)
        for control in (self.adaptive_reference_check, self.planar_region_check, self.planar_preset,
                        self.perspective_check, self.smoothness_combo):
            control.setEnabled(idle)
        for control in (self.shape_combo, self.acceleration_combo, self.memory_spin, self.threads_spin):
            control.setEnabled(idle)
        for control in (*self.region_fields, self.region_apply):
            control.setEnabled(has_info and idle and self.canvas.box is not None)
        tracking_mode = self.tracking_combo.currentData()
        if tracking_mode == "precision":
            self.tracking_hint.setText("更精细地匹配小目标和复杂运动，分析耗时更长。")
        else:
            self.tracking_hint.setText("快速分析清晰目标，适合日常镜头。")
        for widget in (self.preview_minus, self.preview_plus, self.preview_zoom, self.preview_fit):
            widget.setEnabled(has_info and idle and self.canvas.mode != "output")
        self.strip.editable = bool(has_info and idle)
        for control in (self.loop_button, self.play_range_button, self.time_units):
            control.setEnabled(bool(has_info and idle))
        for widget in (self.timeline, self.previous_button, self.next_button, self.play_button,
                       self.start_spin, self.end_spin, self.in_button, self.out_button):
            widget.setEnabled(has_info and idle)
        for widget in (self.mode_combo, self.strength_slider, self.zoom_slider, self.aspect_combo,
                       self.resolution_combo, self.quality_combo):
            widget.setEnabled(has_info and idle)
        is_follow = self.mode_combo.currentData() == "follow"
        self.smoothing_slider.setEnabled(has_info and idle and is_follow)
        self.target_combo.setEnabled(has_info and idle and not is_follow)
        for widget in (self.target_x_spin, self.target_y_spin, self.composition_edit):
            widget.setEnabled(has_info and idle and not is_follow)
        for slider in self.composition_sliders:
            slider.setEnabled(has_info and idle and not is_follow)
        self.canvas.composition_editable = self.composition_edit.isChecked() and has_info and idle and not is_follow
        self.mode_hint.setText("平滑目标的运动，保留自然运镜。" if is_follow else "把选中的细节固定在画面指定位置。")
        self.folder_button.setEnabled(bool(self.last_export))
        self.cancel_button.setVisible(self.busy)
        # Tracking has its own persistent measured phase panel above the canvas.
        self.progress.setVisible(self.busy and not self._tracking_active)
        self._layout_preview_controls()
        self.setWindowTitle(("● " if self.dirty else "") + f"番茄跟踪 v{__version__}" +
                            ("  /  " + Path(self.info.path).name if has_info else ""))
        self._update_trim_summary()
        if not has_info:
            self.workflow_hint.setText("导入视频，开始一个镜头")
        elif self.busy:
            self.workflow_hint.setText("正在处理 · 可查看分析实况")
        elif not active_anchors:
            self.workflow_hint.setText("下一步：在原片上选中清晰细节")
        elif self.track is None:
            self.workflow_hint.setText("下一步：开始跟踪")
        elif self._region_stale():
            self.workflow_hint.setText('区域已修改，旧结果已过期；请重新跟踪')
        elif self.track.lost_frames:
            self.workflow_hint.setText(f"下一步：复核 {len(self.track.lost_frames)} 帧跟丢")
        else:
            self.workflow_hint.setText("可调整构图或导出")
        self._sync_sequence_ui()
        self.asset_status_bar.set_info(self._display_info or self.info,
                                       context="分析实况" if self._display_info else "")
        shown_info = self._display_info or self.info
        self.monitor_filename.setText(Path(shown_info.path).name if shown_info else "尚未导入素材")
        self.monitor_filename.setToolTip(shown_info.path if shown_info else '')
        self.compare_hold_button.setEnabled(has_info and self.plan is not None and idle)
        self.asset_status_bar.set_progress(self._header_value, state=self._header_state)
        self._region_sync()
        self._sections_sync()
        self._retime_sync()
        if getattr(self, '_ad_installed', False):
            from .ad_workspace import ad_sync
            ad_sync(self)

    def _pick_mode_changed(self, *args):
        if not self._ready:
            return
        self.canvas.selection_mode = self.pick_mode.currentData()
        self.canvas.point_size = self.point_size.value()
        if self.canvas.selection_mode == "point" and self.canvas.mode == "output":
            self._set_view("source")
        self.canvas.update()
        self._sync_ui()

    def _update_trim_summary(self):
        if not self.info:
            self.trim_summary.setText("先截取需要的片段，再开始增稳")
            return
        start, end = self.start_spin.value() - 1, self.end_spin.value() - 1
        duration = (end - start + 1) / self.info.fps
        self.trim_summary.setText(f"保留 {timestamp(start, self.info.fps)} — {timestamp(end + 1, self.info.fps)}\n"
                                  f"{duration:.3f} 秒 · {end - start + 1} 帧")

    def _edit_trim(self):
        if self.busy or not self.info:
            return
        self._stop_play()
        dialog = None
        try:
            dialog = TrimDialog(self.info, self.start_spin.value() - 1, self.end_spin.value() - 1,
                                self.frame_index, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._apply_trim(dialog.start_frame, dialog.end_frame)
        except Exception as error:
            self._error(str(error))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _apply_trim(self, start, end):
        if not self.info or self.busy:
            return
        start = max(0, min(self.info.frames - 1, int(start)))
        end = max(start, min(self.info.frames - 1, int(end)))
        self._tb_trim_change(start,end+1)
        self._seek(start)

    def _set_view(self, mode):
        self.canvas.mode = "compare"
        for key, button in self.view_buttons.items():
            button.setChecked(key == "compare")
        self.canvas.update()
        self._sync_ui()

    def _preview_zoom_changed(self, factor):
        self.preview_zoom.blockSignals(True)
        self.preview_zoom.setValue(factor)
        self.preview_zoom.blockSignals(False)

    def _tracking_mode_changed(self, *args):
        if not self._ready or self.busy:
            return
        self._stop_play()
        self._tb_clear_history()
        self.track = self.plan = None
        self.dirty = self.info is not None
        self._refresh_issues()
        self._rebuild_plan()
        self._show_current()
        self._update_strip()
        self._sync_ui()
        if not self.info:
            return
        if self.tracking_combo.currentData() == "precision":
            self._prepare_precision_preview()
        else:
            self._switch_preview(self.info)
            self.status_label.setText("已切换普通跟踪 · 保留关键帧，请重新跟踪")

    def _switch_preview(self, analysis_info):
        reader = FrameReader(analysis_info.proxy)
        if self.reader:
            self.reader.close()
        self.reader = reader
        self._preserve_preview_view = True
        self._seek(self.frame_index)

    def _prepare_precision_preview(self):
        if not self.info or self.busy:
            return
        if self.precision_info is not None:
            self._switch_preview(self.precision_info)
            return
        self._job(prepare_precision_video, (self.info,), self._precision_preview_done,
                  "正在准备精准跟踪的高清原片预览…")

    def _precision_preview_done(self, info):
        self.precision_info = info
        self._switch_preview(info)
        self.status_label.setText("精准跟踪已就绪 · 可放大原片选择小细节，再开始跟踪")

    def _settings_changed(self, *args):
        if not self._ready:
            return
        self._stop_play()
        self.dirty = self.info is not None
        self._rebuild_plan()
        self._sync_ui()
        self._show_current()

    def _target_preset(self, *args):
        if hasattr(self, "target_x_spin"):
            self._set_composition(*self.target_combo.currentData())

    def _set_composition(self, x, y):
        for spin, value in ((self.target_x_spin, x), (self.target_y_spin, y)):
            spin.blockSignals(True)
            spin.setValue(value * 100)
            spin.blockSignals(False)
        self._settings_changed()

    def _composition_edit_changed(self, enabled):
        self.canvas.composition_editable = enabled
        self.canvas._composition_drag = False
        if enabled:
            self._set_view("compare")
            if self.track is not None and not self.track.start <= self.frame_index <= self.track.end:
                self._seek(self.track.start)
        self.status_label.setText("效果画面：拖动构图锚点；原片选区暂停编辑。" if enabled else "原片：选择或修正跟踪区域。")

    def _rebuild_plan(self):
        self.plan = build_plan(self.info, self.track, self.settings()) if self.info and is_complete(self.track) and not self._sections_stale() else None
        self.audio_policy_label.setText("MP4 / H.264  ·  " + ("保留首条原片音轨" if self.info and self.info.has_audio else "无音轨"))
        if not self.plan:
            self.crop_label.setText("跟踪完成后计算所需裁切。")
            return
        text = (f"实际放大 {self.plan.zoom:.2f}× · 上限 {self.settings().max_zoom:.2f}×"
                f"\n构图受限 {self.plan.limited_frames} 帧 · 输出 {self.plan.width} × {self.plan.height}")
        if self.plan.limited_frames:
            text += f"\n{self.plan.limited_frames} 帧达到裁切上限，锁定会适当让位于构图。可提高放大上限。"
        else:
            text += "\n当前裁切余量足够。"
        self.crop_label.setText(text)

    def _choose_video(self):
        if self.busy:
            return
        self._stop_play()
        get_logger('import').info('打开素材选择窗口')
        paths, _ = QFileDialog.getOpenFileNames(self, "批量导入视频", str(self.preferences.value("last_folder", "")),
                                             "视频 (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.mts *.m2ts);;所有文件 (*)")
        get_logger('import').info('素材选择返回 数量=%d', len(paths))
        if paths:
            self.preferences.setValue("last_folder", str(Path(paths[0]).parent))
            self._import_paths(paths)

    def _confirm_discard(self):
        self._stash_clip()
        if (len(self.clips) > 1 or self.sequence_path) and (self._sequence_changed or any(clip.dirty for clip in self.clips)):
            answer = QMessageBox.question(self, "保留素材序列？", "序列或素材调整尚未保存，是否保存整个序列后继续？",
                                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                          QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            return self._save_sequence() if answer == QMessageBox.StandardButton.Save else answer == QMessageBox.StandardButton.Discard
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "保留当前工程？", "当前调整尚未保存，是否保存后继续？",
                                      QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                      QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self._save_project()
        return answer == QMessageBox.StandardButton.Discard

    def load_video(self, path, project=None, is_demo=False):
        if self.busy or not self._confirm_discard():
            return
        self._pending_project = project
        self._pending_demo = is_demo
        self._stop_play()
        self.preferences.setValue("last_folder", str(Path(path).parent))
        self._job(import_video, (path,), self._import_done, "正在准备可逐帧分析的预览…")

    def open_demo(self):
        if self.busy or not self._confirm_discard():
            return
        path = resource_root() / "assets" / "demo.mp4"
        if path.is_file():
            # Already confirmed above.
            self.dirty = False
            self.load_video(str(path), is_demo=True)
            return
        self._pending_project = None
        self._pending_demo = True

        def generate(cancel, progress):
            target = cache_root() / "lockframe-demo.mp4"
            if not target.exists():
                create_demo(str(target), cancel=cancel)
            return import_video(str(target), cancel, progress)

        self._job(generate, (), self._import_done, "正在生成合成演示…")

    def _import_done(self, info):
        project_marks=(self._pending_project or {}).get('review_marks',{}) if self._pending_project and self._pending_project.get('fingerprint')==info.fingerprint else None
        timeline_raw = (self._pending_project or {}).get('timebar', {})
        if self._pending_project and self._pending_project.get('fingerprint') != info.fingerprint:
            timeline_raw = {}
        if not self._switching_clip:
            self._timebar_histories = {}
        self._original_preview.clear()
        self._paused_tracking = None
        if not self._switching_clip:
            self._stash_clip()
        self._retrack_previous = None
        if self.reader:
            self.reader.close()
        self.info = info
        self._display_info = None
        self._restore_retime({})
        if not self.busy:
            self._header_value, self._header_state = None, "就绪"
            self.tracking_progress_panel.reset()
        self.precision_info = None
        self._prepare_preview_pending = False
        self._preserve_preview_view = False
        self.canvas.reset_view()
        self.reader = FrameReader(info.proxy)
        self.track = self.plan = None
        self.anchors = {}
        self.section_tracking, self.tracking_sections = False, []
        self.frame_index = 0
        self.project_path = None
        self._ready = False
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(1, info.frames)
        self.start_spin.setValue(1)
        self.end_spin.setValue(info.frames)
        self.timeline.setRange(0, info.frames - 1)
        self._ready = True
        self.file_label.setText(Path(info.path).name)
        self.file_label.setToolTip(info.path)
        self.meta_label.setText(f"{info.width} × {info.height}  /  {info.fps:.2f} fps\n"
                                f"{info.duration:.2f} 秒  ·  {info.frames} 帧  ·  {'含声音' if info.has_audio else '无声音'}")
        self.issue_list.clear()
        self.dirty = False
        status = "视频已就绪，在原片画面上框选目标即可开始"
        if self._pending_demo:
            _, box = demo_frame(0)
            self.anchors[0] = box
            status = "合成演示已就绪 · 已选中目标，点击「开始跟踪」体验效果"
        if self._pending_project:
            payload = self._pending_project
            self.project_path = payload.get("_project_path")
            self._ready = False
            self.start_spin.setValue(max(1, min(info.frames, int(payload["start"]) + 1)))
            self.end_spin.setValue(max(self.start_spin.value(), min(info.frames, int(payload["end"]) + 1)))
            self._restore_settings(Settings.from_dict(payload["settings"]))
            self._ready = True
            if payload.get("fingerprint") == info.fingerprint:
                from .section_workflow import section_configuration
                self.section_tracking, self.tracking_sections = section_configuration(
                    payload.get("section_tracking", False), payload.get("tracking_sections", []), info.frames)
                self.anchors = {int(k): tuple(v) for k, v in payload["anchors"].items() if int(k) < info.frames}
                raw = payload.get("track")
                selected_mode = self.tracking_combo.currentData()
                compatible = (payload.get("tracker_revision") in {2, 3, 4}
                              and raw and raw.get("tracking_mode", "normal") == selected_mode
                              and (payload.get("tracker_revision") >= 3 or selected_mode == "normal"))
                if compatible:
                    restored = TrackResult(raw["start"], raw["end"], np.asarray(raw["positions"], dtype=float),
                                           np.asarray(raw["boxes"], dtype=float),
                                           np.asarray(raw["confidence"], dtype=float), raw["states"],
                                           raw.get("tracking_mode", "normal"), raw.get("analysis_size"),
                                           raw.get("performance", {}))
                    if raw.get("homographies") is not None:
                        restored.homographies = np.asarray(raw["homographies"], dtype=float)
                    if raw.get("confidence_grids") is not None:
                        restored.confidence_grids = np.asarray(raw["confidence_grids"], dtype=float)
                    restored.anchors = ({int(k): tuple(v) for k, v in raw["anchors"].items()}
                                       if raw.get("anchors") is not None else
                                       {f:box for f,box in self.anchors.items() if restored.start<=f<=restored.end})
                    restored.validate(info.frames)
                    if restored.start == self.start_spin.value() - 1 and restored.end == self.end_spin.value() - 1:
                        self.track = restored
                status = "工程已恢复，可以继续调整或导出"
                if raw and not compatible:
                    status = "工程已恢复 · 跟踪方式已更新，请点击开始跟踪重新计算"
            else:
                status = "原视频已经改变，旧跟踪和关键帧已清除，请重新框选"
        self.strip.load_state(timeline_raw, info.frames, info.fps,
                              self.start_spin.value()-1, self.end_spin.value()-1)
        self._pending_demo = False
        self._pending_project = None
        self._prepare_preview_pending = self.tracking_combo.currentData() == "precision"
        self._refresh_anchors()
        self._refresh_issues()
        self._rebuild_plan()
        self._seek(self.start_spin.value()-1)
        self.status_label.setText(status)
        if not self._switching_clip:
            self._register_current_clip()
            if project_marks is not None and self._clip() is not None:
                from .review_records import clean_marks
                self._clip().review_marks=clean_marks(project_marks)
            if self.track is not None and not is_complete(self.track):
                self._restore_pause_context(self.track)
                self._seek(self._paused_tracking["frame"])
                self.status_label.setText("暂停工程已恢复；修正暂停帧后点击继续跟踪。未分析帧不能导出。")

    def _restore_settings(self, settings):
        self.drift_review_combo.setCurrentIndex(self.drift_review_combo.findData(settings.drift_review))
        self._region_shape = settings.target_shape
        self._polygon_points = settings.polygon_points
        for combo, data in ((self.mode_combo, settings.mode), (self.aspect_combo, settings.aspect),
                            (self.resolution_combo, settings.resolution), (self.quality_combo, settings.crf),
                            (self.tracking_combo, settings.tracking_mode), (self.shape_combo, settings.target_shape),
                            (self.acceleration_combo, settings.acceleration)):
            index = combo.findData(data)
            if index >= 0:
                combo.setCurrentIndex(index)
        self.strength_slider.setValue(round(settings.strength * 100))
        self.smoothing_slider.setValue(round(settings.smoothing * 100))
        self.zoom_slider.setValue(round(settings.max_zoom * 100))
        self.memory_spin.setValue(settings.memory_percent)
        self.threads_spin.setValue(settings.compute_threads)
        self.perspective_check.setChecked(settings.perspective)
        self.adaptive_reference_check.setChecked(settings.adaptive_reference)
        self.planar_region_check.setChecked(settings.planar_region)
        self.segmented_check.setChecked(settings.segmented_tracking)
        self.segment_frames.setValue(settings.segment_frames)
        self.mesh_compute_check.setChecked(settings.confidence_mesh)
        index = self.smoothness_combo.findData(settings.smoothness)
        if index >= 0:
            self.smoothness_combo.setCurrentIndex(index)
        self.canvas.selection_shape = settings.target_shape
        self.canvas.region_points = settings.polygon_points
        self.canvas.region_edit_mode = 'select'
        self.canvas.selected_vertex = -1
        nearest = min(range(self.target_combo.count()), key=lambda i: abs(self.target_combo.itemData(i)[0] - settings.target_x))
        self.target_combo.setCurrentIndex(nearest)
        self.target_x_spin.setValue(settings.target_x * 100)
        self.target_y_spin.setValue(settings.target_y * 100)
        self._restore_retime(settings.retime)

    def _seek(self, index):
        if not self.info or not self.reader:
            return
        index = max(0, min(self.info.frames - 1, int(index)))
        self.frame_index = index
        self._retime_source_seek(index)
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        preview_started = time.perf_counter()
        was_playing = self.timer.isActive()
        try:
            self.frame = self.reader.read(index)
            self._show_current()
        except Exception as error:
            self._stop_play()
            self.status_label.setText(str(error))
        self.time_label.setText(f"{timestamp(index, self.info.fps)} / {timestamp(self.info.frames, self.info.fps)}  ·  {index + 1} 帧")
        self._update_strip()
        self._tb_update_labels()
        if was_playing:
            stats = getattr(self, '_playback_stats', None)
            if stats is None:
                stats = self._playback_stats = dict(started=preview_started, frames=0, total=0., maximum=0.)
            cost = time.perf_counter() - preview_started
            stats['frames'] += 1
            stats['total'] += cost
            stats['maximum'] = max(stats['maximum'], cost)

    def _workflow_step(self, step):
        self._mark_workflow(step)

        if step == 0:
            self.library_tabs.setCurrentIndex(0)
            if self.info is None and not self.busy:
                self._choose_video()
        elif step == 1:
            self.library_tabs.setCurrentIndex(2 if self.track is not None else 1)
            if self.track is not None:
                self.marker_tabs.setCurrentIndex(1)
        elif step == 2:
            self.inspector_tabs.setCurrentIndex(0)
            self._set_view("compare")
        else:
            self.inspector_tabs.setCurrentIndex(3)

    def _mark_workflow(self, step):
        for index, button in enumerate(self.workflow_buttons):
            button.setChecked(index == step)







    def _timeline_seek(self, index):
        self._seek(index)

    def _step_preview(self, direction):
        if self._retime_step(direction):return
        self._stop_play()
        self._seek(self.frame_index + direction)


    def _show_current(self):
        if self.frame is None or not self.info:
            return
        box, confidence, state = None, None, ""
        homography = None
        processed = None
        grid = None
        source_frame = self.frame
        indices = (self.frame_index,)
        original = None
        inspecting = self.preview_quality.currentData() == "auto" and not self.busy and not self.timer.isActive()
        if inspecting:
            original = self._original_preview.frames(self.info, indices)
            if original is not None:
                source_frame = original[self.frame_index]
        else:
            self._original_preview.clear()
        max_side = None if original is not None else 1280
        if (self.track and self.track.start <= self.frame_index <= self.track.end
                and (self.track.performance.get("complete", True)
                     or computed_mask(self.track)[self.frame_index - self.track.start])
                and not (self.frame_index in self.anchors and tuple(self.anchors[self.frame_index]) !=
                         tuple((self.track.anchors or {}).get(self.frame_index, ())))):
            index = self.frame_index - self.track.start
            box = self.track.boxes[index]
            confidence = self.track.confidence[index]
            state = self.track.states[index]
            if self.track.homographies is not None:
                homography = self.track.homographies[index]
            if self.track.confidence_grids is not None and self.mesh_overlay_check.isChecked():
                grid = self.track.confidence_grids[index]
            if self.plan:
                processed = preview_frame(source_frame, self.info, self.plan, index, max_side=max_side)
            self.preview_note.setText("这一帧需要修正，请重新框选，再点击继续跟踪。" if state == "lost" else
                                       f"第 {self.frame_index + 1} 帧 · 匹配可信度 {confidence:.0%} · 可逐帧检查目标框是否贴合")
        elif self.frame_index in self.anchors:
            box, confidence, state = self.anchors[self.frame_index], 1.0, "manual"
            self.preview_note.setText("关键帧已设置，开始跟踪后可查看稳定效果。")
        else:
            self.preview_note.setText("在原片上拖出目标框；跟偏时可在当前帧补一个关键帧。")
        self.canvas.target = (self.target_x_spin.value() / 100, self.target_y_spin.value() / 100)
        quality = "原片高清" if original is not None else "流畅代理"
        quality += f" · 左 {source_frame.shape[1]}×{source_frame.shape[0]}"
        if processed is not None:
            quality += f" / 右 {processed.shape[1]}×{processed.shape[0]}"
        if inspecting and original is None:
            quality += " · 高清不可用，已回退" if self._original_preview.error else " · 正在读取原片高清…"
        self.preview_note.setText(self.preview_note.text() + " · " + quality)
        if self._region_stale():
            self.preview_note.setText('区域已修改 · 旧结果仅供复核，请重新跟踪后导出 · ' + self.preview_note.text())
        self.preview_quality.setToolTip(quality + ("\n" + self._original_preview.error if self._original_preview.error else "\n只改变预览，不改变跟踪和导出。"))
        self.canvas.set_frames(source_frame, processed, box, confidence, state,
                               preserve_view=self._preserve_preview_view, homography=homography,
                               confidence_grid=grid)
        self._preserve_preview_view = False

        if box is not None:
            for spin, value in zip(self.region_fields, (box[0] * self.info.width, box[1] * self.info.height,
                                                        box[2] * self.info.width, box[3] * self.info.height)):
                spin.setValue(float(value))
        for control in (*self.region_fields, self.region_apply):
            control.setEnabled(box is not None and not self.busy)
        if not self.timer.isActive():self._region_sync()

        if getattr(self, '_ad_installed', False):
            from .ad_workspace import ad_sync
            ad_sync(self, playback=self.timer.isActive())
        self._retime_show_current()
        if not self.busy:self._refresh_drift_warning()

    def _original_preview_ready(self, token):
        if (self._original_preview._closed or not self.info or self.busy or self.timer.isActive() or
                self.preview_quality.currentData() != "auto" or token != self._original_preview._wanted):
            return
        # Do not replace images while the user is dragging an ROI or composing.
        if (self.canvas._edit is not None or self.canvas.drag_start is not None or
                self.canvas._pan_start is not None or self.canvas._composition_drag):
            QTimer.singleShot(80, lambda: self._original_preview_ready(token))
            return
        self._preserve_preview_view = True
        self._show_current()

    def _preview_quality_changed(self):
        self.preferences.setValue("preview/quality", self.preview_quality.currentData())
        self._original_preview.clear()
        self._preserve_preview_view = True
        self._show_current()


    def _select_roi(self, box):
        self._region_commit(self._region_shape, self._polygon_points, box)

    def _refresh_anchors(self):
        self.anchor_list.clear()
        for frame in sorted(self.anchors):
            suffix = " · 区间外" if not self.start_spin.value() - 1 <= frame <= self.end_spin.value() - 1 else ""
            item = QListWidgetItem(f"{timestamp(frame, self.info.fps)}  ·  第 {frame + 1} 帧{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, frame)
            self.anchor_list.addItem(item)

    def _delete_anchor(self):
        item = self.anchor_list.currentItem()
        if item is None:
            return
        timebar_before = self._tb_before_roi()
        removed_frame = item.data(Qt.ItemDataRole.UserRole)
        self.anchors.pop(removed_frame, None)
        if self.track is not None:
            self._retrack_previous = self.track
        self.track = self._paused_tracking["result"] if self._paused_tracking else None
        if self._paused_tracking and removed_frame == self._paused_tracking["frame"]:
            self._paused_tracking["corrected"] = False
        self.plan = None
        self.dirty = True
        self._refresh_anchors()
        self._refresh_issues()
        self._rebuild_plan()
        self._show_current()
        self._update_strip()
        self._sync_ui()
        self._tb_finish_roi(timebar_before, "删除修正关键帧")

    def _range_changed(self, *args):
        if not self._ready or not self.info:
            return
        self._stop_play()
        if self.start_spin.value() > self.end_spin.value():
            other = self.start_spin if self.sender() is self.end_spin else self.end_spin
            other.blockSignals(True)
            other.setValue(self.end_spin.value() if other is self.start_spin else self.start_spin.value())
            other.blockSignals(False)
        self._tb_clear_history()
        self._abandon_paused_tracking(sync=False)
        self.track = self.plan = None
        self.dirty = True
        self._refresh_anchors()
        self._refresh_issues()
        self._rebuild_plan()
        self._update_strip()
        self._show_current()
        self._sync_ui()

    def _set_range_point(self, is_start):
        if not self.info:
            return
        if is_start:
            self.start_spin.setValue(self.frame_index + 1)
            if self.end_spin.value() < self.start_spin.value():
                self.end_spin.setValue(self.start_spin.value())
        else:
            self.end_spin.setValue(self.frame_index + 1)
            if self.start_spin.value() > self.end_spin.value():
                self.start_spin.setValue(self.end_spin.value())

    def _update_strip(self):
        self.strip.frames = self.info.frames if self.info else 0
        self.strip.current = self.frame_index
        self.strip.anchors = self.anchors
        self.strip.track = self.track
        self.strip.tracking_sections = list(self.tracking_sections) if self.section_tracking else []
        self.strip.section_mode = self.section_tracking
        self.strip.start, self.strip.end = self.start_spin.value() - 1, self.end_spin.value() - 1
        self.strip.sync_data()
        if hasattr(self, "_sections_panel"):
            self._sections_panel.cursor_changed()
        self._tb_update_labels()

    def _toggle_play(self):
        if not self.info or self.busy:
            return
        if self._retime_toggle_play():return
        if self.timer.isActive():
            self._stop_play()
        else:
            first, last = self.strip.state["play_in"], self.strip.state["play_out"]-1
            if not first <= self.frame_index < last:
                self._seek(first)
            self._play_started = time.monotonic()
            self._play_first = self.frame_index
            self.timer.start()
            self.play_button.setText("暂停")
            self._show_current()

    def _stop_play(self):
        playing = self.timer.isActive()
        self.timer.stop()
        stats = getattr(self, '_playback_stats', None)
        self._playback_stats = None
        if stats and stats['frames']:
            elapsed = max(.001, time.perf_counter() - stats['started'])
            get_logger('preview').info(
                '播放统计 素材=%s 时钟=%s 源fps=%.3f 提交帧=%d 墙钟秒=%.3f 提交fps=%.2f 平均处理毫秒=%.2f 最大处理毫秒=%.2f（非显示器呈现fps）',
                Path(self.info.path).name if self.info else '', self._retime_clock,
                self.info.fps if self.info else 0., stats['frames'], elapsed,
                stats['frames']/elapsed, stats['total']/stats['frames']*1000, stats['maximum']*1000)
        self._retime_play_range=None
        if hasattr(self, "play_button"):
            self.play_button.setText("播放")
        if playing and not self.busy and self.info:
            self._show_current()

    def _play_tick(self):
        if self._retime_tick():return
        index = self._play_first + int((time.monotonic() - self._play_started) * self.info.fps)
        first, last = self.strip.state['play_in'], self.strip.state['play_out']-1
        if self.strip.state['loop'] and index > last:
            index = first+(index-first) % (last-first+1)
        index = min(last, index)
        if index != self.frame_index:
            self._seek(index)
        if index >= last and not self.strip.state["loop"]:
            self._stop_play()

    def _start_tracking(self):
        if not self.info or self.busy:
            return
        try:
            if file_fingerprint(self.info.path) != self.info.fingerprint:
                raise ValueError("原视频已变化，请重新导入后跟踪；当前暂停结果已保留")
        except OSError:
            self._error("原视频无法访问，请恢复源文件或重新导入后跟踪；当前暂停结果已保留")
            return
        except ValueError as error:
            self._error(str(error))
            return
        if self._paused_tracking and not getattr(self, "_resume_requested", False):
            self.status_label.setText("当前任务已暂停，请修正暂停帧后点击继续跟踪")
            return
        start, end = self.start_spin.value() - 1, self.end_spin.value() - 1
        if end < start:
            self._error("出点需要在入点之后")
            return
        from .section_workflow import tracking_ranges
        try:
            sections = tracking_ranges(self.section_tracking, self.tracking_sections, self.anchors, start, end)
        except ValueError as error:
            self.status_label.setText(str(error))
            self._open_tracking_sections()
            self._sections_panel.status.setText(str(error))
            return
        self._stop_play()
        self._mark_workflow(1)
        self._tb_clear_history()
        self._job(track_video, (self.info, dict(self.anchors), start, end),
                  self._tracking_done,
                  "正在精准跟踪，逐帧精细匹配…" if self.settings().tracking_mode == "precision"
                  else "正在跟踪目标…", mode=self.settings().tracking_mode,
                  resources=ComputeOptions(self.settings().acceleration, self.settings().memory_percent,
                                           self.settings().compute_threads),
                  shape=self.settings().target_shape,
                  polygon_points=self.settings().polygon_points,
                  perspective=self.settings().perspective,
                  smoothness=self.settings().smoothness,
                  adaptive_reference=self.settings().adaptive_reference,
                  planar_region=self.settings().planar_region,
                  segmented_tracking=self.settings().segmented_tracking,
                  segment_frames=self.settings().segment_frames,
                  confidence_mesh=self.settings().confidence_mesh,
                  drift_review=self.settings().drift_review,
                  pause_on_lost=self.auto_pause_check.isChecked(),
                  resume=self._paused_tracking.get("resume") if self._paused_tracking else None,
                  previous=self.track if self.track is not None else self._retrack_previous,
                  **({"tracking_sections": sections} if sections else {}))

    def _pause_tracking(self):
        if self.worker is None or not self._tracking_active:
            return
        self._manual_pause_requested = True
        self.worker.pause()
        self.pause_button.setEnabled(False)
        self.status_label.setText("正在停止计算并保留已分析帧…")

    def _reset_current_tracking(self):
        """Forget this stopped clip's targets and every path that could resume them."""
        if self.busy or self.worker is not None or not self.info:
            return
        self._stop_play()
        self._stash_clip()
        clip = self._clip()
        if clip is None:
            return
        from .tracking_pause import reset_clip_tracking
        try:
            removed = reset_clip_tracking(clip)
        except OSError as error:
            self._error(f"无法重置跟踪：{error}。当前跟踪点和结果仍保留。")
            return
        self._tb_clear_history()
        # Clear UI references before any refresh can stash them back into Clip.
        self.anchors = {}
        self.track = self._retrack_previous = self._paused_tracking = None
        self.plan = None
        self._paused_by_clip.pop(clip.id, None)
        self._paused_queue_ids = [value for value in self._paused_queue_ids if value != clip.id]
        self._resume_requested = self._manual_pause_requested = False
        self._original_preview.clear()
        self._live_preview_last_frame = None
        self._live_preview_previous_mode = None
        self._close_foreign_preview()
        self._display_info = None
        self._header_value, self._header_state = None, "待选区"
        self.tracking_progress_panel.reset()
        self.track_button.setText("开始跟踪")
        self.performance_label.setText("跟踪已重置 · 重新选点后开始新的跟踪")
        self.live_preview_status.setText("尚未开始新的跟踪")
        self.dirty = self._sequence_changed = True
        self._refresh_anchors()
        self._refresh_issues()
        self._rebuild_plan()
        self._target_selection_tool()
        self._show_current()
        self._update_strip()
        self._mark_workflow(1)
        self._sync_ui()
        self.status_label.setText("已重置当前素材的全部跟踪点、轨迹和暂停进度。请重新选择目标，再点击开始跟踪。")
        get_logger("app").info("重置当前素材跟踪 clip=%s 已清除检查点=%d", clip.id, removed)

    def _abandon_paused_tracking(self, *, sync=True):
        """A new processing range is a new task; never keep an unreachable gate."""
        if self.busy or self._paused_tracking is None:
            return
        self._retrack_previous = self._paused_tracking["result"]
        self._paused_tracking = None
        self._paused_by_clip.pop(self.active_clip_id, None)
        self._paused_queue_ids = [value for value in self._paused_queue_ids if value != self.active_clip_id]
        clip = self._clip()
        if clip is not None:
            clip.tracking_resume = None
        self.track = self.plan = None
        self._header_state, self._header_value = "等待跟踪", None
        self.dirty = True
        self.status_label.setText("已放弃暂停任务；已分析帧仅在兼容时复用。点击开始跟踪运行当前范围。")
        if sync:
            self._refresh_issues()
            self._show_current()
            self._update_strip()
            self._sync_ui()

    @staticmethod
    def _pause_context_for(paused):
        return {"result": paused.result, "frame": paused.frame, "reason": paused.reason,
                   "direction": paused.direction, "resume": paused.resume,
                   "anchors": dict(paused.result.anchors or {}), "corrected": False}

    def _tracking_paused(self, paused):
        context = self._pause_context_for(paused)
        self._paused_tracking = context
        if self.active_clip_id:
            self._paused_by_clip[self.active_clip_id] = context
        self.track = self._retrack_previous = paused.result
        self.plan = None
        self._job_outcome, self._header_state = "paused", "已暂停"
        self._live_preview_previous_mode = "source"
        self.canvas.mode = "source"
        self._seek(paused.frame)
        self._refresh_issues()
        self.status_label.setText(f"已暂停在第 {paused.frame + 1} 帧 · 已分析 {paused.done_count}/{paused.result.count} 帧。重新框选目标后点击继续跟踪。")
        if paused.reason=='review':
            self.status_label.setText(f'第 {paused.frame+1} 帧疑似漂移，未判为跟丢。请检查：跟偏则重新框选；目标正确可点“确认并继续”。')
        elif paused.reason=='manual':
            from .tracking_pause import can_resume_without_edit
            if can_resume_without_edit(context):self.status_label.setText(f'已暂停检查第 {paused.frame+1} 帧 · 可直接继续；需要修正时在原片重新框选。')
        self.tracking_progress_panel.finish("paused", self.status_label.text())
        self._reveal_section("tracking")
        self.dirty = True

    def _restore_pause_context(self, result):
        metadata = result.performance.get("tracking_pause", {})
        context = self._paused_by_clip.get(self.active_clip_id)
        if context is None or context["result"] is not result:
            frame = int(metadata.get("frame", result.start))
            context = {"result": result, "frame": frame, "reason": metadata.get("reason", "manual"),
                       "direction": metadata.get("direction", 1), "resume": None,
                       "anchors": dict(result.anchors or {}),
                       "corrected": frame in self.anchors and tuple(self.anchors[frame]) !=
                                    tuple((result.anchors or {}).get(frame, ()))}
            self._paused_by_clip[self.active_clip_id] = context
        self._paused_tracking = context
        self._retrack_previous = result
        self.plan = None
        self._header_state = "已暂停"
        self.canvas.mode = "source"
        batch = self.active_clip_id in self._paused_queue_ids
        order = getattr(self, "_queue_order", [self.active_clip_id])
        self.tracking_progress_panel.begin(Path(self.info.path).name,
            order.index(self.active_clip_id) + 1 if self.active_clip_id in order else 1,
            len(order) if batch else 1, batch=batch, follow_task_phases=batch)
        stage = metadata.get("stage", "计算")
        fraction = metadata.get("phase_fraction", float(computed_mask(result).sum()) / result.count)
        values = []
        phase_progress(lambda value, text: values.append((value, text)), stage, fraction, "修正暂停帧后点击继续跟踪")
        value, text = values[0]
        if batch:
            self._header_value = getattr(self, "_paused_batch_progress", value)
            self.tracking_progress_panel.update_progress(self._header_value, "队列已暂停")
            self.tracking_progress_panel.update_task_progress(value, text)
        else:
            self._header_value = value
            self.tracking_progress_panel.update_progress(value, text)
        from .tracking_pause import can_resume_without_edit
        self.tracking_progress_panel.finish("paused", "疑似漂移：重新框选，或确认目标正确后继续" if context.get('reason')=='review' else "已暂停检查，可直接继续或重新框选修正" if can_resume_without_edit(context) else "重新框选暂停帧后，点击继续跟踪")

    def _continue_tracking(self):
        context = self._paused_tracking
        if self.busy or context is None:
            return
        if context.get('reason')=='review' and not context.get('corrected'):
            self._seek(context['frame'])
            box=context['result'].boxes[context['frame']-context['result'].start]
            self._region_commit(self._region_shape,self._polygon_points,tuple(float(v) for v in box),label='人工确认疑似漂移帧')
        from .tracking_pause import can_resume_without_edit
        if not can_resume_without_edit(context):
            self.status_label.setText(f"请先在第 {context['frame'] + 1} 帧重新框选目标，再继续")
            return
        self._paused_queue_ids = [value for value in self._paused_queue_ids if self._clip(value) is not None]
        if self._paused_queue_ids:
            blocked = next((clip_id for clip_id in self._paused_queue_ids if
                clip_id in self._paused_by_clip and not can_resume_without_edit(self._paused_by_clip[clip_id])), None)
            if blocked:
                self._activate_clip(blocked)
                self.status_label.setText("还有已暂停素材需要修正，修正后再继续批量任务")
                return
            self._stash_clip()
            self._refresh_sequence(set(self._paused_queue_ids))
            self._paused_queue_ids = []
            self._start_queue()
            return
        self._retrack_previous = context["result"]
        self._resume_requested = True
        try:
            self._start_tracking()
        finally:
            self._resume_requested = False

    def _tracking_done(self, result):
        if not is_complete(result):
            self.track = result
            self._restore_pause_context(result)
            self._refresh_issues()
            self._show_current()
            return
        self._paused_tracking = None
        self._paused_by_clip.pop(self.active_clip_id, None)
        self._retrack_previous = None
        self.track = result
        self._mark_workflow(1 if result.lost_frames else 2)
        performance = result.performance
        self.performance_label.setText(f"本次 {performance.get('fps', 0):.1f} 帧/秒\n"
                                       f"{performance.get('device') or 'CPU'}\n"
                                       f"GPU 匹配 {performance.get('gpu_calls', 0)} 次\n"
                                       f"帧缓存峰值 {performance.get('cache_peak_mb', 0):.1f} MB\n"
                                       f"外观更新 {performance.get('adaptation_updates', 0)} 次")
        self.performance_label.setText(self.performance_label.text() + f"\n续跑复用 {performance.get('resumed_frames', 0)} 帧")
        self.dirty = True
        self.track_button.setText("重新跟踪")
        self._refresh_issues()
        self._rebuild_plan()
        self._show_current()
        self._update_strip()
        if result.lost_frames:
            self.marker_tabs.setCurrentIndex(1)
            self.status_label.setText(f"跟踪完成 · {len(result.lost_frames)} 帧待修正，已暂停导出；点击左侧问题位置补关键帧")
        else:
            quality = performance.get("quality", {})
            review = set(quality.get("low_confidence", [])) | set(quality.get("motion_review", []))
            review.update(item['frame'] for item in performance.get('early_warning_ranges',[]) if isinstance(item,dict) and type(item.get('frame')) is int)
            if review:
                self.marker_tabs.setCurrentIndex(1)
                self.status_label.setText(f"跟踪完成 · {len(review)} 帧建议复核，原始轨迹已保留；可在左侧跳转检查")
            else:
                self.status_label.setText("跟踪完成 · 请播放检查效果，可以调整参数或导出")

    def _refresh_issues(self):
        self.issue_list.clear()
        self.marker_tabs.setTabText(1, "待修正")
        if not self.track:
            self.summary_label.setText(f"已设置 {len(self.anchors)} 个手动关键帧" if self.anchors else "尚未设置跟踪目标")
            return
        lost = self.track.lost_frames
        self.marker_tabs.setTabText(1, f"待修正 {len(lost)}" if lost else "待修正")
        measured = int(computed_mask(self.track).sum())
        valid = measured - len(lost)
        mode_name = "精准跟踪" if self.track.tracking_mode == "precision" else "普通跟踪"
        self.summary_label.setText(f"{mode_name} · 有效匹配 {valid} / {self.track.count} 帧\n"
                                   f"{len(lost)} 帧待修正 · 未分析 {self.track.count - measured} 帧 · {len(self.anchors)} 个关键帧")
        groups = []
        for frame in lost:
            if groups and frame == groups[-1][1] + 1:
                groups[-1][1] = frame
            else:
                groups.append([frame, frame])
        for first, last in groups:
            item = QListWidgetItem(f"待修正 · 第 {first + 1}–{last + 1} 帧")
            item.setData(Qt.ItemDataRole.UserRole, first)
            self.issue_list.addItem(item)
        quality = self.track.performance.get("quality", {})
        from .drift_review import review_ranges
        suspicious=review_ranges(self.track.performance.get('suspected_drift_ranges',[]),self.track.start,self.track.end)
        for interval in suspicious[:200]:
            item=QListWidgetItem(f'疑似漂移待复核 · 第 {interval["start"]+1}–{interval["end"]+1} 帧')
            item.setData(Qt.ItemDataRole.UserRole,interval['frame']);self.issue_list.addItem(item)
        from .drift_review import review_ranges
        early=review_ranges(self.track.performance.get('early_warning_ranges',[]),self.track.start,self.track.end)
        for interval in early[:200]:
            if any(r['start']<=interval['frame']<=r['end'] for r in suspicious):continue
            item=QListWidgetItem(f'跟偏预警记录 · 第 {interval["start"]+1}–{interval["end"]+1} 帧')
            item.setData(Qt.ItemDataRole.UserRole,interval['frame']);self.issue_list.addItem(item)
        review = set(quality.get("low_confidence", [])) | set(quality.get("motion_review", []))
        review.update(item['frame'] for item in early)
        if not lost and review:
            self.marker_tabs.setTabText(1, f"待复核 {len(review)}")
        for title, key in (("低可信度", "low_confidence"), ("运动突变待复核", "motion_review")):
            frames = quality.get(key, [])
            for frame in frames[:200]:
                item = QListWidgetItem(f"{title} · 第 {frame + 1} 帧")
                item.setData(Qt.ItemDataRole.UserRole, frame)
                self.issue_list.addItem(item)
        if not groups and not suspicious and not early and not quality.get("low_confidence") and not quality.get("motion_review"):
            item = QListWidgetItem("未发现跟丢 · 仍建议目视检查")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.issue_list.addItem(item)

    def _job(self, function, args, callback, message, **kwargs):
        get_logger('job').info('任务开始 function=%s args_count=%d', getattr(function,'__name__',type(function).__name__),len(args))
        self._progress_log_time=0.
        self._tracking_active = function is track_video
        self._job_kind = "tracking" if self._tracking_active else "other"
        self._job_outcome = None
        self._import_feedback = None
        self._cancel_progress = None
        self._display_info = None
        self._header_value = 0.
        self._header_state = "跟踪" if self._tracking_active else "处理中"
        if function is track_video:
            self._begin_live_preview()
            kwargs["live_preview"] = self.live_preview_check.isChecked()
            kwargs["supports_pause"] = True
            self._manual_pause_requested = False
        self._reset_stages(function is track_video)
        self.busy = True
        self._original_preview.clear()
        self._stop_play()
        self.cancel_button.setEnabled(True)
        self.progress.setValue(0)
        self.status_label.setText(message)
        self._sync_ui()
        worker = Job(function, *args, parent=self, **kwargs)
        self.worker = worker
        self._sync_ui()
        worker.progress.connect(self._on_progress)
        worker.tracking_frame.connect(self._on_tracking_frame)
        worker.paused.connect(self._tracking_paused)

        def complete(result):
            try:
                get_logger('job').info('任务结果到达 function=%s callback=%s',getattr(function,'__name__',type(function).__name__),getattr(callback,'__name__',type(callback).__name__))
                callback(result)
                get_logger('job').info('任务结果已应用 function=%s',getattr(function,'__name__',type(function).__name__))
                cancelled = worker.cancel_event.is_set()
                self._job_outcome = "cancelled" if cancelled else "completed"
                if self._tracking_active:
                    self.tracking_progress_panel.finish(self._job_outcome, self.status_label.text())
                if cancelled:self._header_state = "已取消"
                else:self._header_value, self._header_state = 1., "完成"
            except Exception as error:
                get_logger('job').exception('应用任务结果失败')
                self._error(str(error))

        worker.result.connect(complete)
        worker.failed.connect(self._error)
        worker.aborted.connect(self._job_aborted)
        worker.finished.connect(self._job_finished)
        worker.start()

    def _on_progress(self, value, message):
        now=time.monotonic()
        if now-getattr(self,'_progress_log_time',0.)>=10.:
            self._progress_log_time=now
            get_logger('job').info('任务进度 kind=%s progress=%.3f stage=%s',self._job_kind,value,message[:300])
        if self._job_kind == "queue" and self._cancel_progress is not None:
            value = min(value, self._cancel_progress)
        self._header_value = max(0., min(1., value))
        self.asset_status_bar.set_progress(self._header_value, state=self._header_state)
        if self._tracking_active or parse_phase(message) or message.startswith("[汇总:"):
            if self.tracking_progress_panel.state == "idle":
                self.tracking_progress_panel.begin(Path(self.info.path).name if self.info else "")
            self.tracking_progress_panel.update_progress(value, message)
        if message.startswith("[汇总:"):
            self.progress.setValue(round(value * 1000))
            return
        phase = parse_phase(message)
        if phase:
            message = phase[2]
        if phase and phase[0]=='计算' and '跟偏预警' in message:
            self._set_drift_warning(message=message)
        elif phase and phase[0]=='计算' and message.startswith('第'):
            self._set_drift_warning()
        self.progress.setValue(round(value * 1000))
        self.status_label.setText(f"{message}  ·  {value:.0%}")

    def _reset_stages(self, visible=True):
        if visible:
            self.tracking_progress_panel.begin(Path(self.info.path).name if self.info else "")
        else:
            self.tracking_progress_panel.reset()

    def _update_stage(self, stage, fraction):
        self.tracking_progress_panel.set_phase(stage, fraction)

    def _job_aborted(self):
        self._job_outcome = "cancelled"
        self._header_state = "已取消"
        self.status_label.setText("任务已取消，原视频和已导出文件保持完整")
        if self._tracking_active:
            self.tracking_progress_panel.finish("cancelled", self.status_label.text())

    def _job_finished(self):
        get_logger('job').info('任务线程结束 kind=%s outcome=%s',self._job_kind,self._job_outcome)
        worker = self.worker
        self.worker = None
        self.busy = False
        self._tracking_active = False
        self._display_info = None
        self._job_kind = None
        self._end_live_preview()
        self._sync_ui()
        if worker:
            worker.deleteLater()
        if self._close_pending:
            self._close_pending = False
            self.close()
        elif getattr(self, "_activate_after_job", None):
            clip_id, self._activate_after_job = self._activate_after_job, None
            self._activate_clip(clip_id)
        elif self._prepare_preview_pending:
            self._prepare_preview_pending = False
            self._prepare_precision_preview()
        elif getattr(self, "_queue_after_prepare", None):
            selected, self._queue_after_prepare = self._queue_after_prepare, None
            self._refresh_sequence(selected)
            self._start_queue()
        feedback = getattr(self, '_import_feedback', None)
        self._import_feedback = None
        if feedback:self.status_label.setText(feedback)
        if not self.busy and not self._close_pending and self.info and self.isVisible():
            self._show_current()
        if getattr(self,'_workspace_restore_pending',None):QTimer.singleShot(0,self._apply_restored_workspace_view)
        if hasattr(self,'_autosave'):QTimer.singleShot(0,lambda:self._autosave.request(force=True))
        if getattr(self,'_review_dialog',None) is not None and self._review_dialog.isVisible():self._review_dialog.refresh()

    def _cancel(self):
        self._queue_after_prepare = None
        if self.worker:
            self._cancel_progress = self._header_value or 0.
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("正在安全取消任务…")

    def _error(self, message):
        self._queue_after_prepare = None
        if self.busy:
            self._job_outcome = "failed"
            self._header_state = "失败"
            if self._tracking_active:
                self.tracking_progress_panel.finish("failed", message)
        get_logger("app").error("界面错误提示: %s", message)
        self.status_label.setText(message)
        QMessageBox.warning(self, "处理提示", message)

    def _export(self):
        if not self.info or not self.track or self.busy:
            return
        if self._sections_stale():
            self._error('独立分段已修改，请重新跟踪后导出')
            return
        if self._region_stale():
            self._error('跟踪区域已修改，旧结果已过期；请重新跟踪后导出')
            return
        if not is_complete(self.track):
            self._error("跟踪已暂停，请先修正并继续跟踪，再导出")
            return
        if self.track.lost_frames:
            self._error("请先修正跟丢的位置，再导出视频")
            return
        self._mark_workflow(3)
        folder = str(self.preferences.value("export_folder", str(Path(self.info.path).parent)))
        suggested = str(Path(folder) / (Path(self.info.path).stem + "_番茄跟踪.mp4"))
        path, _ = QFileDialog.getSaveFileName(self, "导出稳定视频", suggested, "MP4 视频 (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        self.preferences.setValue("export_folder", str(Path(path).parent))
        self.cancel_button.setEnabled(True)
        self._job(export_video, (self.info, self.track, self.settings(), path),
                  self._export_done, "正在从原片渲染，请稍候…")

    def _export_done(self, result):
        self.last_export = result["path"]
        self.status_label.setText(f"导出完成 · {result['width']} × {result['height']} · {result['frames']} 帧 · {result['path']}")
        QMessageBox.information(self, "导出完成", f"稳定视频已保存：\n{result['path']}\n\n可以点击「打开导出文件夹」查看。")

    def _open_export_folder(self):
        if self.last_export:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_export).parent)))

    def _open_diagnostics(self):
        from .diagnostics_ui import open_diagnostics
        return open_diagnostics(self)

    def _open_log_folder(self):
        try:
            log_folder().mkdir(parents=True,exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_folder())))
        except OSError as error:self.status_label.setText('日志目录暂不可用：'+str(error))

    def _refresh_cache_label(self):
        try:
            path = cache_root()
        except Exception:
            path = Path(os.environ.get("LOCALAPPDATA", "")) / "LockFrameStudio" / "cache"
        configured = self.preferences.value("cache/dir", "")
        if configured:
            self.cache_dir_label.setText(f"当前：{path}")
        else:
            self.cache_dir_label.setText(f"当前（默认）：{path}")

    def _change_cache_dir(self):
        current = self.preferences.value("cache/dir", "")
        start = current if current else str(Path(os.environ.get("LOCALAPPDATA", "")) / "LockFrameStudio" / "cache")
        choice = QFileDialog.getExistingDirectory(self, "选择缓存目录", start)
        if not choice:
            return
        set_cache_dir(self.preferences, choice)
        self._refresh_cache_label()
        self.status_label.setText(f"缓存已切换到 {choice} · 下次导入视频时生效，旧缓存需手动清理")

    def _reset_cache_dir(self):
        set_cache_dir(self.preferences, None)
        self._refresh_cache_label()
        self.status_label.setText("缓存已恢复默认位置 · 下次导入视频时生效")

    def _save_project(self):
        if not self.info or self.busy:
            return False
        path = self.project_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "保存番茄跟踪工程", str(Path(self.info.path).with_suffix(".lockframe")),
                                                 "番茄跟踪工程 (*.lockframe)")
        if not path:
            return False
        if not path.lower().endswith(".lockframe"):
            path += ".lockframe"
        try:
            save_project(path, self.info, self.settings(), self.anchors,
                         self.start_spin.value() - 1, self.end_spin.value() - 1, self.track,
                         timebar=dict(self.strip.state), section_tracking=self.section_tracking,
                         tracking_sections=self.tracking_sections,review_marks=self._clip().review_marks if self._clip() else {})
            self.project_path = path
            self.dirty = False
            self.status_label.setText("工程已保存 · 原视频和跟踪调整可下次继续使用")
            self._sync_ui()
            return True
        except Exception as error:
            self._error(str(error))
            return False

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开番茄跟踪工程", "", "番茄跟踪工程 (*.lockframe)")
        if not path:
            return
        try:
            payload = read_project(path)
            payload["_project_path"] = path
            self.load_video(payload["source"], project=payload)
        except Exception as error:
            self._error(str(error))

    def _show_help(self):
        QMessageBox.information(self, "使用番茄跟踪", "1. 导入 SDR 视频，或打开合成演示。\n"
            "2. 点击「裁剪视频片段」，预览并选择需要保留的起止时间。\n"
            "3. 选择普通 / 精准跟踪；精准模式耗时更长，适合较难镜头。\n"
            "4. 原片滚轮缩放或使用缩放按钮，中键拖动；框选或点选细节后开始跟踪。\n"
            "5. 预览上方可切换矩形、椭圆、多边形；选择顶点后可加点、删点或重置。\n"
            "6. 跟偏时暂停修正，修改后主动继续；改区域会保留并标记旧结果过期，重新跟踪后才能导出。\n"
            "7. 调整主体锁定 / 平滑跟随、裁切和画面比例，再导出。\n\n"
            "快捷键：预览区空格播放/暂停，空格+左键或中键拖动；Ctrl+O 导入，Ctrl+T 裁剪，Ctrl+S 保存。\n"
            "原片缩放只影响查看，不改变导出构图。\n"
            "红色区间必须修正后才能导出。可信度是匹配指标，不保证内容识别正确。\n"
            "按素材、跟踪、构图、变速、导出操作。变速可关闭；预览不播放声音。变速声音按映射策略处理，倒放和定格静音。HDR 显示映射为 SDR，支持原生 10 位 HDR 导出。")

    def _show_about(self):
        QMessageBox.information(self, "关于番茄跟踪", f"番茄跟踪  v{__version__}\n\n"
            "为本地视频主体锁定与平滑跟随制作。\n"
            "基于 lockOn 的光流锁定思路，参考 StableFlow Pro 的跟踪工作流。\n"
            "Windows / Python / OpenCV / Qt / FFmpeg\n\n"
            "本应用采用 GPL-3.0-only；StableFlow Pro 的 MIT 声明随附。\n"
            "完整许可、第三方来源与源码见软件目录中的说明文件。\n"
            "软件按现状提供，不保证所有镜头均能自动跟踪成功。")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self.busy:
            event.acceptProposedAction()

    def dropEvent(self, event):
        if not self.busy:
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if path.lower().endswith(".lockframe"):
                    try:
                        payload = read_project(path)
                        payload["_project_path"] = path
                        self.load_video(payload["source"], project=payload)
                    except Exception as error:
                        self._error(str(error))
                else:
                    self._import_paths([url.toLocalFile() for url in urls if url.isLocalFile()])

    def closeEvent(self, event):
        diagnostics=getattr(self,'diagnostics_dialog',None)
        if diagnostics is not None and diagnostics.running:
            diagnostics.cancel()
            if not getattr(self,'_closing_diagnostics',False):
                self._closing_diagnostics=True
                diagnostics.export_finished.connect(self._close_after_diagnostics)
            event.ignore();return
        self._closing_diagnostics=False
        queue = getattr(self, "export_queue", None)
        if queue is not None and queue.running:
            answer = QMessageBox.question(self, "批量导出正在进行", "取消剩余导出并关闭？已完成结果会保留。",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                queue.cancel()
                queue.worker.finished.connect(self.close)
            event.ignore()
            return
        if self.busy:
            answer = QMessageBox.question(self, "任务正在进行", "是否取消当前任务并关闭软件？",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if answer == QMessageBox.StandardButton.Yes:
                self._close_pending = True
                self._cancel()
            event.ignore()
            return
        if hasattr(self,'_clip_preview') and not self._clip_preview.close():
            if not getattr(self,'_closing_clip_preview',False):
                self._closing_clip_preview=True
                self._clip_preview.worker.finished.connect(self.close)
                # finished may already be queued for this GUI thread.
                if self._clip_preview.worker.isFinished():QTimer.singleShot(0,self.close)
            event.ignore();return
        self._closing_clip_preview=False
        if not getattr(self,'_workspace_close_approved',False) and not self._confirm_discard():
            event.ignore()
            return
        self._workspace_close_approved=True
        self._stop_play()
        if hasattr(self,'_autosave'):
            if not self._workspace_close_saved:
                self._workspace_close_saved=True;self._autosave.request(force=True)
            if not self._autosave.close():
                self.setEnabled(False);QTimer.singleShot(80,self.close);event.ignore();return
        if self.reader:
            self.reader.close()
        self._original_preview.close()
        self._retime_debounce.stop()
        self._retime_preview.close()
        if hasattr(self.strip, 'close_media'):
            self.strip.close_media()
        self._thumbnails.close()
        self._thumbnail_probe_timer.stop()
        event.accept()
        self.hardware_monitor.stop()
        get_logger('app').info('正常关闭界面')

    def _close_after_diagnostics(self):
        self.diagnostics_dialog.export_finished.disconnect(self._close_after_diagnostics)
        self._closing_diagnostics=False
        self.close()


def main():
    import argparse
    import sys

    configure_logging()
    parser = argparse.ArgumentParser(description="番茄跟踪 · 本地视频主体锁定工具")
    parser.add_argument("video", nargs="?")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--self-check", metavar="DIRECTORY", help="检查发行版的跟踪、渲染和界面")
    parser.add_argument("--ui-check", metavar="DIRECTORY", help="检查三栏工作流、状态栏与实时阶段")
    parser.add_argument("--workflow-check", metavar="DIRECTORY", help="检查精简界面与暂停修正继续跟踪")
    parser.add_argument("--reset-check", metavar="DIRECTORY", help="检查当前素材重置与重新跟踪")
    parser.add_argument("--preview-quality-check", metavar="DIRECTORY", help="检查原片高清双预览与流畅回退")
    parser.add_argument("--core-check", metavar="DIRECTORY", help="检查四步流程、旧工程兼容和正常导出")
    parser.add_argument("--timebar-check", metavar="DIRECTORY", help="检查源帧时间线、修正点和独立播放范围")
    parser.add_argument("--thumbnails-check", metavar="DIRECTORY", help="检查素材列表缩略图与缓存隔离")
    parser.add_argument('--region-check', metavar='DIRECTORY', help='检查三栏布局、区域形状、撤销与掩码跟踪导出')
    parser.add_argument('--timeline-trim-check',metavar='DIRECTORY',help='检查时间线裁剪、清晰胶片与原声导出')
    parser.add_argument('--ad-ui-check', metavar='DIRECTORY', help='检查批准AD布局、双预览控制及真实媒体时间轴')
    parser.add_argument('--import-check',metavar='DIRECTORY',help='检查素材导入完成、取消、错误及界面响应')
    parser.add_argument('--diagnostics-check',metavar='DIRECTORY',help='检查本地日志、诊断包与独立导出')
    parser.add_argument('--sections-check',metavar='DIRECTORY',help='检查独立分段、逐段选区、保存恢复与批量导出')
    parser.add_argument('--simple-check',metavar='DIRECTORY',help='检查简洁工作区、共享播放和主页面分段')
    parser.add_argument('--ms-retime-check',metavar='DIRECTORY',help='检查新版变速、DIS光流、同步预览与单片批量导出')
    parser.add_argument('--performance-check',metavar='DIRECTORY',help='检查疑似漂移复核、短段预渲染和按需导出')
    parser.add_argument('--drift-warning-check',metavar='DIRECTORY',help='检查黄色跟偏预警、持续下降暂停和复核')
    parser.add_argument('--workflow-complete-check',metavar='DIRECTORY',help='检查自动保存恢复、集中复核、曲线预设与暂停检查')
    args = parser.parse_args()
    if args.ms_retime_check or args.performance_check or args.drift_warning_check or args.workflow_complete_check:
        os.environ['QT_QPA_PLATFORM']='offscreen'
    if args.simple_check or args.sections_check or args.diagnostics_check or args.self_check or args.ui_check or args.workflow_check or args.reset_check or args.preview_quality_check or args.core_check or args.thumbnails_check or args.timebar_check or args.region_check or args.ad_ui_check or args.timeline_trim_check or args.import_check:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    application = QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    application.setApplicationName("番茄跟踪")
    application.setWindowIcon(application_icon())
    application.setOrganizationName("Yongzheng")
    application.setFont(QFont("Microsoft YaHei UI", 9))
    get_logger("app").info("界面初始化完成 版本=%s", __version__)
    if args.workflow_complete_check:
        from .workflow_complete_check import run_workflow_complete_check
        return run_workflow_complete_check(args.workflow_complete_check)
    if args.drift_warning_check:
        from .drift_warning_check import run_drift_warning_check
        return run_drift_warning_check(args.drift_warning_check)
    if args.performance_check:
        from .performance_check import run_performance_check
        return run_performance_check(args.performance_check)
    if args.ms_retime_check:
        from .ms_retime_check import run_ms_retime_check
        return run_ms_retime_check(args.ms_retime_check)
    if args.simple_check:
        from .simple_check import run_simple_check
        return run_simple_check(args.simple_check)
    if args.sections_check:
        from .sections_check import run_sections_check
        return run_sections_check(args.sections_check)
    if args.diagnostics_check:
        from .diagnostics_check import run_diagnostics_check
        return run_diagnostics_check(args.diagnostics_check)
    if args.import_check:
        from .import_check import run_import_check
        return run_import_check(args.import_check)
    if args.timeline_trim_check:
        from .timeline_trim_check import run_timeline_trim_check
        return run_timeline_trim_check(args.timeline_trim_check)
    if args.ad_ui_check:
        from .ad_ui_check import run_ad_check
        return run_ad_check(args.ad_ui_check)
    if args.region_check:
        from .region_check import run_region_check
        return run_region_check(args.region_check)
    if args.timebar_check:
        from .timebar_check import run_timebar_check
        return run_timebar_check(args.timebar_check)
    if args.thumbnails_check:
        from .thumbnail_check import run_thumbnail_check
        return run_thumbnail_check(args.thumbnails_check)
    if args.core_check:
        from .core_check import run_core_check
        return run_core_check(args.core_check)
    if args.preview_quality_check:
        from .preview_quality_check import run_quality_check
        return run_quality_check(args.preview_quality_check)
    if args.reset_check:
        from .reset_check import run_reset_check
        return run_reset_check(args.reset_check)
    if args.workflow_check:
        from .v072_check import run_workflow_check
        return run_workflow_check(args.workflow_check)
    if args.ui_check:
        from .ui_v8_check import run_ui_check
        return run_ui_check(args.ui_check)
    if args.self_check:
        from .diagnostics import run_check
        return run_check(args.self_check)
    window = MainWindow()
    window._start_workspace_services()
    window.show()
    if not args.video and not args.demo:QTimer.singleShot(200,window._offer_workspace_recovery)
    if args.demo:
        QTimer.singleShot(100, window.open_demo)
    elif args.video:
        QTimer.singleShot(100, lambda: window.load_video(args.video))
    return application.exec()
