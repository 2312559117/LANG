"""Approved A geometry / D palette, composed from the existing live controls."""
import math

from PySide6.QtCore import Qt,QSize
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (QApplication,QComboBox,QFrame,QHBoxLayout,QLabel,QMenu,
    QMenuBar,QPushButton,QSizePolicy,QSlider,QToolButton,QVBoxLayout,QWidget)

from .ui_icons import icon


def tool(name,title,callback,parent=None):
    button=QToolButton(parent)
    button.setObjectName('adTransportButton')
    button.setIcon(icon(name));button.setIconSize(QSize(20,20))
    button.setToolTip(title)
    button.setAccessibleName(title)
    button.setFixedSize(30,30)
    button.clicked.connect(callback)
    return button


class MonitorTransport(QWidget):
    """One shared source clock, with independent source/output view controls."""
    def __init__(self,window,primary=False):
        super().__init__()
        self.window=window
        self.setObjectName('adTransport')
        layout=QVBoxLayout(self)
        layout.setContentsMargins(10,5,10,9)
        layout.setSpacing(3)
        self.slider=QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName('monitorSeek')
        self.slider.sliderPressed.connect(window._stop_play)
        self.slider.valueChanged.connect(window._transport_seek)
        layout.addWidget(self.slider)
        self.clock=QLabel('00:00:00:00 / 00:00:00:00')
        self.clock.setObjectName('monitorClock')
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter);self.clock.setFixedHeight(18)
        layout.addWidget(self.clock)
        row=QHBoxLayout();row.setSpacing(4)
        row.addWidget(tool('first','跳到播放入点',lambda:window._transport_boundary(True)))
        previous=window.previous_button if primary else tool('prev','上一帧',lambda:window._step_preview(-1))
        play=window.play_button if primary else tool('play','播放 / 暂停',window._toggle_play)
        next_button=window.next_button if primary else tool('next','下一帧',lambda:window._step_preview(1))
        self.play_button=play
        for button,name,title in ((previous,'prev','上一帧'),(play,'play','播放 / 暂停'),(next_button,'next','下一帧')):
            button.setObjectName('adTransportButton')
            button.setText('');button.setIcon(icon(name));button.setIconSize(QSize(20,20));button.setToolTip(title)
            button.setFixedSize(30,30);button.setIconSize(QSize(20,20));row.addWidget(button)
        row.addWidget(tool('last','跳到播放出点',lambda:window._transport_boundary(False)))
        row.addStretch()
        row.addWidget(QLabel('原片'))
        row.addWidget(tool('fit','原片：适配',window.canvas.reset_view if primary else lambda:window.canvas.set_output_zoom(1.)))
        def pixel_view():
            canvas=window.canvas
            if primary and window.info:
                canvas.set_zoom(min(8.,max(1.,window.info.width/max(1.,canvas.source_viewport.width()))))
            elif canvas.processed_image or canvas.source_image:
                image=canvas.processed_image or canvas.source_image
                canvas.set_output_zoom(min(8.,max(1.,image.width()/max(1.,(canvas.width()-12)/2))))
        one=tool('one','当前预览按像素查看（最高8倍，居中显示）',pixel_view)
        one.setToolTip('原片：按像素查看')
        row.addWidget(one)
        row.addWidget(QLabel('效果'))
        row.addWidget(tool('fit','效果：适配',lambda:window.canvas.set_output_zoom(1.)))
        def output_pixels():
            canvas=window.canvas
            image=canvas.processed_image or canvas.source_image
            if image:
                canvas.set_output_zoom(min(8.,max(1.,image.width()/max(1.,(canvas.width()-12)/2))))
        row.addWidget(tool('one','效果：按像素查看',output_pixels))
        row.addWidget(window.loop_button)
        layout.addLayout(row)

    def sync(self):
        w=self.window;info=w._display_info or w.info
        index=w._live_preview_last_frame if w.busy and w._display_info and getattr(w,'_live_preview_last_frame',None) is not None else w.frame_index
        self.slider.blockSignals(True)
        self.slider.setRange(0,max(0,info.frames-1) if info else 0)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        if info:
            from .timebar_core import Timebase
            tb=Timebase.from_fps(info.fps)
            self.clock.setText(f'{tb.timecode(index)}  /  {tb.timecode(info.frames)}')
            if getattr(w,'_retime_config',{}).get('enabled') and w._retime_active():
                values=w._retime_map()
                self.slider.blockSignals(True)
                self.slider.setRange(0,len(values)-1);self.slider.setValue(w._retime_output_step)
                self.slider.blockSignals(False)
                self.clock.setText(f'输出 {tb.timecode(w._retime_output_step)} / {tb.timecode(len(values))} · 源 {tb.timecode(index)}')
        self.play_button.setText('')
        playing=w.timer.isActive()
        if getattr(self,'_playing_icon',None)!=playing:
            self._playing_icon=playing
            self.play_button.setIcon(icon('pause' if playing else 'play',colour='#eadfce',size=21))
        self.setEnabled(bool(w.info and not w.busy))


def _clear_layout(layout):
    while layout.count():
        item=layout.takeAt(0)
        if item.widget():item.widget().hide()


def install_ad_workspace(w):
    """Reparent, never duplicate the authoritative model or processing controls."""
    if getattr(w,'_ad_installed',False):return
    w._ad_installed=True
    root=w.centralWidget().layout()
    root.setContentsMargins(10,0,10,0);root.setSpacing(8)
    w.header.setObjectName('adHeader');w.header.setFixedHeight(50)
    header=w.header.layout();header.setContentsMargins(10,0,10,0);header.setSpacing(14)
    old_menu=w.menuBar();old_menu.hide();old_menu.setFixedHeight(0)
    w._ad_original_menu = old_menu
    original_actions = old_menu.actions()
    # QActions retain their callbacks; menus receive explicit ownership rather
    # than transferring menuAction wrappers between two different menu bars.
    file_actions = original_actions[0].menu().actions()
    help_actions = original_actions[-1].menu().actions()
    for action in file_actions+help_actions:
        action.setParent(w)
    w._ad_support_actions = file_actions+help_actions
    menu=QMenuBar(w.header);menu.setObjectName('adMenu')
    file_menu=menu.addMenu('文件');file_menu.addActions(file_actions)
    for title,items in (
        ('编辑',(('撤销',lambda:w._tb_operation('undo')),('重做',lambda:w._tb_operation('redo')))),
        ('视图',(('适配原片',w.canvas.reset_view),('区域与数值微调',lambda:w._open_settings(2)),
                 ('时间轴显示帧号',lambda:w._tb_units(0)),('时间轴显示时间码',lambda:w._tb_units(1)))),
        ('播放',(('播放 / 暂停',w._toggle_play),('播放范围 / 关键帧',w._tb_menu))),
        ('跟踪',(('开始跟踪',w._start_tracking),('暂停修正',w._pause_tracking),('继续跟踪',w._continue_tracking))),
        ('窗口',(('跟踪设置',lambda:w._open_settings(0)),('构图设置',lambda:w._open_settings(4)),('输出设置',lambda:w._open_settings(5))))):
        dropdown=menu.addMenu(title)
        for text,callback in items:dropdown.addAction(text).triggered.connect(lambda checked=False,c=callback:c())
    help_menu=menu.addMenu("帮助");help_menu.addActions(help_actions)
    w._ad_source_menus=(file_menu,help_menu)
    w.ad_menu = menu
    header.removeWidget(w.asset_status_bar)
    header.addWidget(menu)
    header.addStretch()
    w.ad_undo=tool('undo','撤销当前素材编辑',lambda:w._tb_operation('undo'))
    w.ad_redo=tool('redo','重做当前素材编辑',lambda:w._tb_operation('redo'))
    header.addWidget(w.ad_undo);header.addWidget(w.ad_redo)
    header.addWidget(tool('layout','恢复批准的三栏布局',lambda:w.workspace_splitter.setSizes([308,925,313])))
    w.ad_export=QPushButton('导出视频');w.ad_export.setIcon(icon('export',colour='#171614'))
    w.ad_export.setObjectName('adGlobalExport');w.ad_export.setFixedSize(110,36)
    w.ad_export.clicked.connect(w._export);header.addWidget(w.ad_export)
    # Asset metadata is always visible in the central monitor header.
    head=w.preview_head.layout();_clear_layout(head)
    head.setContentsMargins(6,6,6,6);head.setSpacing(10)
    w.monitor_filename.hide()
    w.asset_status_bar.setMinimumWidth(170);w.asset_status_bar.setMaximumWidth(244)
    w.asset_status_bar.layout().setContentsMargins(0,0,0,0)
    w.asset_status_bar.progress_label.hide()
    w.asset_status_bar.show();head.addWidget(w.asset_status_bar,1)
    head.addWidget(w.region_toolbar);w.region_toolbar.show()
    head.addStretch()
    link=QLabel();link.setPixmap(icon('link',colour='#e88c51').pixmap(22,22));head.addWidget(link)
    w.sync_label.setText('原片 / 效果预览  |  同步播放');w.sync_label.show();head.addWidget(w.sync_label)
    w.preview_head.setObjectName('adPreviewHead');w.preview_head.setFixedHeight(72)
    preview=w.central_preview;preview.setObjectName('adPreviewPanel');preview.setAttribute(Qt.WidgetAttribute.WA_StyledBackground,True)
    preview.layout().setContentsMargins(8,0,8,0);preview.layout().setSpacing(0)
    w.preview_note.hide()
    w.preview_host.setObjectName('adMonitor')
    w.preview_host.layout().setContentsMargins(0,0,0,0)
    w.canvas.ad_style=True
    w.monitor_transports=[]
    controllers=QWidget();controllers.setObjectName('adDualTransport')
    controls=QHBoxLayout(controllers);controls.setContentsMargins(0,0,0,0);controls.setSpacing(12)
    for primary in (True,):
        control=MonitorTransport(w,primary);controls.addWidget(control,1);w.monitor_transports.append(control)
    controllers.setFixedHeight(70);preview.layout().addWidget(controllers)
    # Four existing sections retain their menus, settings, and live actions.
    materials=w.sidebar_sections['materials'];materials.layout().setContentsMargins(6,14,6,8);materials.layout().setSpacing(16)
    materials.content.layout().setSpacing(10)
    w.import_button.setIcon(icon('import',colour='#eadfce'));w.import_button.setFixedHeight(37)
    w.select_all_clips.setText('全部');w.select_all_clips.setFixedHeight(37)
    w.select_all_clips.clicked.disconnect()
    filter_menu=QMenu(w.select_all_clips)
    for title,mode in (('显示全部素材','all'),('仅显示待跟踪','pending'),('仅显示已完成','done')):
        filter_menu.addAction(title).triggered.connect(lambda checked=False,m=mode:_filter_clips(w,m))
    filter_menu.addSeparator();filter_menu.addAction('全选素材').triggered.connect(w.clip_list.selectAll)
    # Keep the original full-selection button behaviour. Filter is a secondary
    # context menu, so batch multi-select never turns into a destructive filter.
    w.select_all_clips.setMenu(filter_menu)
    w.select_all_clips.setToolTip('筛选素材；菜单可全选，Ctrl/Shift可多选')
    w.clip_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    w.materials_sidebar.setMinimumWidth(220);w.materials_sidebar.setMaximumWidth(900 if getattr(w, '_clip_view_mode', 'list')=='grid' else 360)
    w.task_sidebar.setMinimumWidth(260);w.task_sidebar.setMaximumWidth(360)
    w.workspace_splitter.setHandleWidth(10);w.workspace_splitter.setSizes([308,925,313])
    w.task_sidebar.layout().setSpacing(6)
    w.sidebar_content.layout().setSpacing(5)
    for button,height in ((w.select_target_button,30),(w.tracking_combo,30),(w.track_button,40),
                          (w.pause_button,38),(w.continue_button,38),(w.reset_tracking_button,38)):
        button.setFixedHeight(height)
    w.select_target_button.setFlat(True)
    w.track_button.setIcon(icon('play',colour='#171614',size=20))
    w.continue_button.setObjectName('adSecondaryContinue')
    w.sidebar_sections['tracking'].content.layout().setSpacing(8)
    w.sidebar_sections['tracking'].layout().setContentsMargins(12,10,12,12)
    w.sidebar_sections['tracking'].layout().setSpacing(12)
    w.auto_pause_check.setFixedHeight(25)
    w.pause_button.setText('暂停');w.composition_edit.setText('画面构图')
    w.center_button.setFixedHeight(36)
    w.sidebar_sections['lock'].layout().setContentsMargins(14,12,14,12)
    w.sidebar_sections['lock'].layout().setSpacing(14)
    w.sidebar_sections['lock'].content.layout().setSpacing(10)
    w.export_button.setText('导出视频');w.batch_export_button.setText('导出选中序列')
    for button in (w.export_button,w.batch_export_button):button.setFixedHeight(36)
    export_layout=w.sidebar_sections['export'].content.layout()
    export_layout.removeWidget(w.export_button);export_layout.addWidget(w.export_button)
    # Original transport controls remain reachable in secondary settings.
    timeline_layout=w.timeline_well.layout()
    transport_item=timeline_layout.takeAt(0)
    if transport_item and transport_item.layout():
        legacy=QWidget(w);legacy.setLayout(transport_item.layout());legacy.hide();w._ad_legacy_transport=legacy
    timeline_layout.setContentsMargins(0,0,0,0);timeline_layout.setSpacing(0)
    w.timeline_well.setObjectName('adTimeline');w.track_legend.hide()
    toolbar=QWidget();toolbar.setObjectName('adTimelineToolbar');toolbar.setFixedHeight(40)
    tools=QHBoxLayout(toolbar);tools.setContentsMargins(6,2,6,2);tools.setSpacing(6)
    w.ad_pointer=tool('pointer','选择和移动修正关键帧',lambda:setattr(w.strip,'tool_mode','select'))
    w.ad_pointer.setCheckable(True);w.ad_pointer.setChecked(True);tools.addWidget(w.ad_pointer)
    tools.addWidget(tool('target','跟踪参数',lambda:w._open_settings(0)))
    tools.addWidget(tool('link','添加当前帧修正点',lambda:w._tb_operation('add')))
    w.ad_hand=tool('hand','拖动平移时间轴',lambda:setattr(w.strip,'tool_mode','pan'));w.ad_hand.setCheckable(True)
    w.ad_hand.clicked.connect(lambda checked:w.ad_pointer.setChecked(not checked))
    w.ad_pointer.clicked.connect(lambda checked:w.ad_hand.setChecked(not checked));tools.addWidget(w.ad_hand)
    w.play_range_button.setText('');w.play_range_button.setIcon(icon('more'))
    w.play_range_button.setObjectName('adTransportButton');w.play_range_button.setFixedSize(30,30)
    tools.addWidget(w.play_range_button)
    tools.addWidget(w.time_label)
    fit=QComboBox();fit.setFixedWidth(72);fit.addItems(['适应','2×','4×','8×'])
    fit.currentIndexChanged.connect(lambda index:(w.strip.fit(),w.strip.zoom(2**index) if index else None))
    tools.addWidget(fit)
    w.timeline_trim_label=QLabel('拖动V1两端裁剪');w.timeline_trim_label.setObjectName('muted')
    tools.addWidget(w.timeline_trim_label);tools.addStretch()
    tools.addWidget(tool('layout','复制当前双预览画面',lambda:QApplication.clipboard().setPixmap(w.canvas.grab())))
    tools.addWidget(tool('zoom-out','缩小时间轴',lambda:w.strip.zoom(.8)))
    zoom=QSlider(Qt.Orientation.Horizontal);zoom.setFixedWidth(102);zoom.setRange(0,100);zoom.setValue(0)
    zoom.valueChanged.connect(lambda value:(w.strip.fit(),w.strip.zoom(2**(value/25)) if value else None));tools.addWidget(zoom)
    tools.addWidget(tool('zoom-in','放大时间轴',lambda:w.strip.zoom(1.25)))
    timeline_layout.insertWidget(0,toolbar)
    w.ad_timeline_toolbar=toolbar;w.strip.ad_style=True
    w.strip.state['timecode']=True
    w.status_label.setFixedHeight(18);w.status_label.setWordWrap(False)
    w.status_label.setToolTip('就绪');w.hardware_brief.setFixedHeight(18)
    w.continue_button.style().unpolish(w.continue_button);w.continue_button.style().polish(w.continue_button)
    w.tracking_progress_panel.compact_default=True
    w.tracking_progress_panel.expand_button.toggled.connect(lambda checked:ad_resize(w))
    for widget in (preview,w.preview_head,w.header,w.timeline_well):
        widget.style().unpolish(widget);widget.style().polish(widget)
    ad_resize(w);ad_sync(w)


def _filter_clips(w,mode):
    from .regions import shape_matches,anchors_match
    w._ad_filter_mode=mode
    for i,clip in enumerate(w.clips):
        if i<w.clip_list.count():
            done=(clip.track is not None and (clip.start,clip.end)==(clip.track.start,clip.track.end) and clip.track.performance.get('complete',True)
                  and not clip.track.lost_frames and shape_matches(clip.track,clip.settings)
                  and anchors_match(clip.track,clip.anchors,clip.start,clip.end))
            w.clip_list.item(i).setHidden(mode=='done' and not done or mode=='pending' and done)
    w.select_all_clips.setText({'all':'全部','pending':'待跟踪','done':'已完成'}[mode])


def ad_resize(w):
    if not getattr(w,'_ad_installed',False):return
    if getattr(w, '_simple_workspace', False):
        from .simple_workspace import resize_simple_workspace
        resize_simple_workspace(w)
        return
    w.header.setFixedHeight(50)
    compact=w.height()<900
    w.timeline_well.setFixedHeight(180 if compact else 262)
    w.strip.setFixedHeight(140 if compact else 222)
    if hasattr(w.strip,'set_compact'):w.strip.set_compact(compact)
    progress=w.tracking_progress_panel
    extra=max(0,progress.sizeHint().height()-88) if progress.expanded else 0
    progress.setMinimumHeight(50 if compact else 88);progress.setMaximumHeight(16777215 if progress.expanded else 50 if compact else 88)
    w.sidebar_sections['tracking'].setFixedHeight((248 if compact else 363)+extra)
    w.sidebar_sections['lock'].setFixedHeight(134 if compact else 173)
    w.sidebar_sections['export'].setFixedHeight(91)
    for button,height in ((w.select_target_button,26),(w.tracking_combo,26),(w.track_button,30),
                          (w.pause_button,28),(w.continue_button,28),(w.reset_tracking_button,28)):
        button.setFixedHeight(height if compact else (30 if button in (w.select_target_button,w.tracking_combo) else 40 if button is w.track_button else 38))
    w.sidebar_sections['tracking'].layout().setContentsMargins(12,6 if compact else 10,12,6 if compact else 12)
    w.sidebar_sections['tracking'].layout().setSpacing(4 if compact else 12)
    w.sidebar_sections['tracking'].content.layout().setSpacing(4 if compact else 8)
    w.sidebar_sections['lock'].layout().setSpacing(6 if compact else 14)
    w.sidebar_sections['lock'].content.layout().setSpacing(4 if compact else 10)
    w.sync_label.setVisible(w.central_preview.width()>=720)


def ad_sync(w, *, playback=False):
    if not getattr(w,'_ad_installed',False):return
    for transport in w.monitor_transports:transport.sync()
    background_busy=w.busy or w.timer.isActive()
    w._thumbnails.set_paused(background_busy)
    if hasattr(w.strip,'set_busy'):w.strip.set_busy(background_busy)
    # Transport and canvas move every frame; media identity and sidebar geometry
    # only change through the full UI sync (or when playback stops).
    if playback:return
    w.ad_export.setEnabled(w.export_button.isEnabled())
    w.pause_button.show();w.continue_button.show();w.track_button.show()
    w.track_button.setEnabled(w.track_button.isEnabled() and not w._paused_tracking)
    progress=w.tracking_progress_panel
    if getattr(w,'_ad_progress_state',None)!=progress.state:
        w._ad_progress_state=progress.state
        progress.set_expanded(False)
    progress.show()
    if hasattr(w.strip,'set_media'):w.strip.set_media(w.info)
    w.status_label.setToolTip(w.status_label.text()+'\n'+w.preview_note.text())
    w.track_legend.hide()
    w.sync_label.setText('原片 / 效果预览  |  同步播放')
    ad_resize(w)
