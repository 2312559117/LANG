"""Real MainWindow region workflow, reusable from source and the frozen EXE.

All generated test footage is synthetic. This check establishes UI integration,
source-time/audio export and actual invocation of the local tracker; it does not
measure tracking accuracy on the user's original car footage.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import os
import threading
import time
import traceback

from . import __version__


EDITED_POINTS = ((.08, .08), (.90, .05), (.95, .90), (.55, 1.), (.05, .85))
TRACK_BOX = (.3, .3, .4, .4)


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def footprint(window):
    settings = window.settings()
    return settings.target_shape, tuple(tuple(point) for point in settings.polygon_points)


def check_layout(window, folder):
    """Check coordinates/visibility as well as ownership, at both target sizes."""
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QApplication, QLabel

    require(window.workspace_splitter.count() == 3, '工作区不是素材、双预览、任务三栏')
    require(window.workspace_splitter.widget(0) is window.materials_sidebar, '左栏不是素材列表')
    require(window.workspace_splitter.widget(1) is window.central_preview, '中栏不是双预览')
    require(window.workspace_splitter.widget(2) is window.task_sidebar, '右栏不是跟踪与构图')
    require(window.timeline_well.parentWidget() is window.centralWidget(), '时间线仍局限在中栏')
    require(list(window.sidebar_sections) == ['materials', 'tracking', 'lock', 'time_remap', 'export'], '五步工作流不连续')
    require([section.findChild(QLabel, 'step').text() for section in window.sidebar_sections.values()]
            == ['01', '02', '03', '04', '05'], '步骤编号不连续')
    measurements = []
    for width, height in ((1560, 940), (1160, 760)):
        window.resize(width, height)
        window.show()
        QApplication.processEvents()
        window._show_current()
        QApplication.processEvents()
        timeline_top = window.timeline_well.mapTo(window, QPoint(0, 0)).y()
        columns_bottom = window.workspace_splitter.mapTo(window, QPoint(0, window.workspace_splitter.height())).y()
        require(timeline_top >= columns_bottom, '时间线覆盖了上方工作区')
        require(window.timeline_well.width() >= window.workspace_splitter.width() - 2,
                '时间线没有横跨三个工作栏')
        require(window.canvas.isVisible() and window.canvas.width() >= 300 and window.canvas.height() >= 120,
                '窗口缩小时双预览不可用')
        require(window.canvas.mode == 'compare' and window.canvas.source_viewport.isValid()
                and window.canvas.output_rect.isValid(), '原片和效果预览没有同时显示')
        require(window.canvas.source_viewport.right() <= window.canvas.output_rect.left(), '两个预览发生重叠')
        require(window.asset_status_bar.isVisible(), '顶部素材状态条消失')
        require(window.export_button.isVisible() and window.task_sidebar.isAncestorOf(window.export_button),
                '导出入口没有常驻右栏')
        require(window.materials_sidebar.isAncestorOf(window.clip_list), '素材序列脱离了左栏')
        require(window.task_sidebar.isAncestorOf(window.track_button), '跟踪入口脱离了右栏')
        require(window.central_preview.isAncestorOf(window.region_toolbar), '区域工具栏不在预览标题区域')
        image = folder / f'三栏区域工作区_{width}x{height}.png'
        require(window.grab().save(str(image)), '无法保存实际页面截图')
        measurements.append(dict(requested=[width, height], actual=[window.width(), window.height()],
                                 canvas=[window.canvas.width(), window.canvas.height()],
                                 columns=window.workspace_splitter.sizes(), timeline_width=window.timeline_well.width()))
    return measurements


def check_regions(directory):
    from . import app
    from .batch_export import prepare_exports, run_exports
    from .core_check import create_source, verify_export
    from .export import export_video
    from .media import configure_cache_preferences
    from .models import Settings, file_fingerprint, read_project
    from .regions import BOX_CORNERS
    from .sequence import Clip, read_sequence, restore_sequence, save_sequence
    from .v072_check import activate, wait_job
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    previous_cache = os.environ.get('LOCKFRAME_CACHE_DIR')
    os.environ['LOCKFRAME_CACHE_DIR'] = str(folder / 'cache')
    window = None
    errors = []
    try:
        application = QApplication.instance() or QApplication([])
        factory = app.QSettings
        app.QSettings = lambda *args: QSettings(str(folder / 'preferences.ini'), QSettings.Format.IniFormat)
        try:
            window = app.MainWindow()
        finally:
            app.QSettings = factory
        window._confirm_discard = lambda: True
        # A self-check must fail with diagnostics rather than block on a modal
        # error dialog if a Job/import callback unexpectedly fails.
        window._error = lambda message: errors.append(str(message))
        clip = create_source(folder / 'generated-source')
        clip.settings = replace(clip.settings, strength=0., compute_threads=2)
        identity = file_fingerprint(clip.info.path)
        other = Clip(clip.info, Settings(target_shape='ellipse', acceleration='cpu', resolution=0),
                     {2: TRACK_BOX}, 2, 9, frame=2)
        clip.frame = clip.start
        activate(window, [clip, other])
        window._seek(clip.start)
        window.show()
        application.processEvents()
        original = window.track
        require(original is not None and window.export_button.isEnabled(), '原有矩形结果无法正常使用')
        before_boxes = original.boxes.copy()
        original_anchors = dict(window.anchors)
        window.region_toolbar.shape_buttons['polygon'].click()
        require(footprint(window) == ('polygon', BOX_CORNERS), '切换多边形未从四个外接框角点初始化')
        require(window.track is original and window._region_stale() and not window.export_button.isEnabled(),
                '切换形状未保留并标记过期结果，或错误地允许导出')
        window._tb_operation('undo')
        require(footprint(window) == ('rectangle', ()) and window.track is original
                and not window._region_stale() and window.export_button.isEnabled(), '撤销形状修改没有恢复原结果')
        window._tb_operation('redo')
        require(footprint(window) == ('polygon', BOX_CORNERS) and window._region_stale(), '重做形状修改失败')

        window.composition_edit.setChecked(True)
        require(window.canvas.composition_editable, '构图拖动模式不可用')
        window._region_action('edit')
        require(not window.canvas.composition_editable and not window.composition_edit.isChecked(),
                '区域编辑没有退出构图拖动模式')

        window._region_proposal(dict(shape='polygon', box=TRACK_BOX, points=EDITED_POINTS))
        require(footprint(window) == ('polygon', EDITED_POINTS)
                and tuple(window.anchors[clip.start]) == TRACK_BOX, '顶点和外接框提案未进入同一主状态')
        require(window.track is original and (original.boxes == before_boxes).all(), '区域编辑改写了已测量轨迹')
        window._tb_operation('undo')
        require(footprint(window) == ('polygon', BOX_CORNERS) and window.anchors == original_anchors,
                '撤销没有同时还原顶点和外接框')
        window._tb_operation('redo')
        require(footprint(window) == ('polygon', EDITED_POINTS) and tuple(window.anchors[clip.start]) == TRACK_BOX,
                '重做没有同时重建顶点和外接框')
        window._stash_clip()
        rejected = prepare_exports([window._clip()], folder / '过期结果禁止批量导出')[0]
        require(bool(rejected.reason), '批量导出接受了形状或外接框已改变的旧跟踪结果')

        # Clip activation/restoration must not let the shared toolbar become a
        # second authoritative shape model or contaminate another clip.
        window._activate_clip(other.id)
        require(footprint(window) == ('ellipse', ()), '另一素材没有恢复自己的椭圆区域')
        window._activate_clip(clip.id)
        window._seek(clip.start)
        require(footprint(window) == ('polygon', EDITED_POINTS), '切回素材后多边形数据丢失')
        window.busy = True
        try:
            window._sync_ui()
            require(not any(button.isEnabled() for button in window.region_toolbar.shape_buttons.values())
                    and not window.region_toolbar.reset_button.isEnabled(), '运行中仍可编辑区域')
            window.region_toolbar.shape_buttons['rectangle'].click()
            require(footprint(window) == ('polygon', EDITED_POINTS), '禁用工具栏仍修改了区域')
        finally:
            window.busy = False
            window._sync_ui()

        window.project_path = str(folder / '多边形区域.lockframe')
        require(window._save_project(), '多边形工程保存失败')
        payload = read_project(window.project_path)
        require(payload['settings']['target_shape'] == 'polygon'
                and tuple(tuple(p) for p in payload['settings']['polygon_points']) == EDITED_POINTS,
                '工程没有保存区域形状和顶点')
        payload['_project_path'] = window.project_path
        window.load_video(payload['source'], project=payload)
        wait_job(window)
        window._seek(clip.start)
        require(footprint(window) == ('polygon', EDITED_POINTS) and window._region_stale(),
                '重新打开工程后区域或过期状态丢失')
        window._stash_clip()
        sequence = folder / '多边形序列.locksequence'
        save_sequence(sequence, window.clips)
        restored = read_sequence(sequence)
        require(any(c.settings.target_shape == 'polygon' and c.settings.polygon_points == EDITED_POINTS
                    for c in restored), '素材序列没有保存多边形')
        window._pending_sequence_path = str(sequence)
        window._job(restore_sequence, (restored,), window._sequence_restored, '验收重开区域序列', keep_partial=True)
        wait_job(window)
        polygon_clip = next(c for c in window.clips if c.settings.target_shape == 'polygon')
        window._activate_clip(polygon_clip.id)
        window._seek(polygon_clip.start)
        require(footprint(window) == ('polygon', EDITED_POINTS), '重开序列后主页面没有恢复顶点')

        empty = Clip(polygon_clip.info, start=polygon_clip.start, end=polygon_clip.end)
        window.clips.append(empty)
        window._activate_clip(empty.id)
        require(window.canvas.box is None and not window.region_toolbar.shape_combo.isEnabled()
                and not any(button.isEnabled() for button in window.region_toolbar.shape_buttons.values()),
                '没有跟踪区域时仍可使用区域工具')
        window._activate_clip(polygon_clip.id)
        window._seek(polygon_clip.start)
        before_reset = dict(window.anchors)
        box_before_reset = tuple(window.canvas.box)
        window._region_action('reset')
        require(footprint(window) == ('rectangle', ()) and window.anchors == before_reset
                and tuple(window.canvas.box) == box_before_reset, '重置区域没有保留外接框或错误删除了关键帧')
        window._tb_operation('undo')
        require(footprint(window) == ('polygon', EDITED_POINTS), '区域重置不能撤销')

        # The indexed source contains clipping/animation later in the clip.
        # Use its known four-frame stable interval for an integration smoke
        # check, with three actual measured non-manual observations.
        window.end_spin.setValue(6)
        window._seek(2)
        window._region_proposal(dict(shape='polygon', box=TRACK_BOX, points=EDITED_POINTS))
        window.live_preview_check.setChecked(True)
        window.auto_pause_check.setChecked(True)
        window._start_tracking()
        require(window.busy, '多边形跟踪没有创建实际 Job')
        require(not window.region_toolbar.shape_combo.isEnabled(), '实际跟踪期间区域工具没有禁用')
        wait_job(window)
        result = window.track
        require(result is not None and result.count == 4 and result.performance.get('complete')
                and not result.lost_frames and result.states.count('manual') == 1,
                '实际多边形跟踪未完成已知稳定区间')
        require(result.performance.get('shape') == 'polygon'
                and tuple(tuple(p) for p in result.performance.get('polygon_points', ())) == EDITED_POINTS,
                '实际跟踪任务没有收到所选掩码')
        require(not window._region_stale() and window.export_button.isEnabled(), '新结果仍被错误标为过期')
        layout = check_layout(window, folder)
        window._stash_clip()
        export_clip = deepcopy(window._clip())
        single_path = folder / '多边形正常音频输出.mp4'
        export_video(window.info, result, window.settings(), str(single_path), threading.Event())
        single_report = verify_export(single_path, export_clip)
        tasks = prepare_exports([window._clip()], folder / '多边形批量输出')
        require(not tasks[0].reason, '有效多边形结果不能进入批量导出队列')
        window._region_choose_shape('ellipse')
        window._stash_clip()
        require(window._region_stale(), '入队后编辑区域未标记当前结果过期')
        outcomes = run_exports(tasks, threading.Event())
        require(outcomes[0]['state'] == '完成', f'冻结的多边形导出任务失败：{outcomes}')
        batch_report = verify_export(outcomes[0]['destination'], export_clip)
        require(file_fingerprint(export_clip.info.path) == identity, '验收修改了生成的源文件')
        require(not errors, f'区域流程出现错误回调：{errors}')
        return dict(footage='local synthetic indexed/audio source; no original-footage accuracy claim',
                    layout=layout, three_columns=True, timeline_full_width=True, persistent_dual_preview=True,
                    polygon_initialized_from_corners=True, region_proposal_undo_redo=True,
                    old_measured_results_preserved=True, stale_export_blocked=True,
                    clip_shapes_isolated=True, busy_and_no_region_disabled=True,
                    sequence_and_project_reopened=True, reset_keeps_box_and_anchors=True,
                    actual_local_tracking=dict(frames=result.count, measured_nonmanual=result.count-1,
                                               states=list(result.states), shape=result.performance['shape']),
                    queued_polygon_snapshot_isolated=True, source_unchanged=True,
                    single_export=single_report, batch_export=batch_report)
    finally:
        if window is not None:
            if window.busy and window.worker is not None:
                window.worker.cancel()
                window.worker.wait(15000)
                QApplication.processEvents()
            window._confirm_discard = lambda: True
            window.close()
            QApplication.processEvents()
        if previous_cache is None:
            os.environ.pop('LOCKFRAME_CACHE_DIR', None)
        else:
            os.environ['LOCKFRAME_CACHE_DIR'] = previous_cache
        configure_cache_preferences(None)


def run_region_check(directory):
    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        report = dict(ok=True, version=__version__, regions=check_regions(folder))
        code = 0
    except Exception:
        report = dict(ok=False, version=__version__, error=traceback.format_exc())
        code = 1
    report['seconds'] = time.monotonic() - started
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return code
