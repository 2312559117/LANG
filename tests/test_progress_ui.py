import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from lockframe.progress_ui import TrackingProgressPanel
from lockframe.quality import STAGES, phase_progress


@pytest.fixture
def panel():
    application = QApplication.instance() or QApplication([])
    widget = TrackingProgressPanel()
    widget.resize(720, widget.sizeHint().height())
    yield widget
    widget.close()
    application.processEvents()


def test_real_tracking_phase_protocol_and_waiting_rows(panel):
    assert panel.isHidden()
    panel.begin("汽车 <01>.mov")
    phase_progress(panel.update_progress, "分析", 0., "检查视频与选区")
    assert not panel.isHidden()
    assert panel.phases["分析"].status == "running"
    assert all(panel.phases[name].status == "waiting" for name in STAGES[1:])
    assert panel.detail == "检查视频与选区"
    phase_progress(panel.update_progress, "计算", .68, "双向分析 · 第 420 帧")
    assert panel.phases["分析"].status == "completed"
    assert panel.stage_bars["计算"].value() == 680
    assert "68%" in panel.stage_labels["计算"].text()
    assert panel.phases["检查"].status == panel.phases["缓存"].status == "waiting"
    assert panel.overall_progress == pytest.approx(.12 + .73 * .68)
    assert panel.task_progress == panel.overall_progress
    assert "汽车 <01>.mov" in panel.task_label.toolTip()


def test_batch_aggregate_and_current_clip_progress_never_conflate(panel):
    panel.begin("红色车3.mov", 3, 5, batch=True)
    panel.update_progress(.47, "[汇总:计算:0.680]")
    assert panel.task_progress is None
    assert "当前 —" in panel.progress_label.text()
    assert "总体 47%" in panel.progress_label.text()
    assert "本批阶段汇总" in panel.scope_label.text()
    # Parallel clips can contribute to different phases; a later aggregate
    # must never invent completion of preceding phases for the whole batch.
    panel.update_progress(.48, "[汇总:检查:0.200]")
    assert panel.phases["分析"].fraction == 0.
    assert panel.phases["计算"].fraction == .68
    panel.set_task_progress(.91)
    assert "当前 91%" in panel.progress_label.text()
    panel.set_task("车4.mov", 4, 5)
    assert panel.task_progress is None
    assert panel.overall_progress == .48
    assert panel.phases["计算"].fraction == .68
    assert "4/5" in panel.task_label.toolTip()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failure_or_cancel_preserves_measured_progress_and_final_status(panel, status):
    panel.begin("跟踪中.mov")
    phase_progress(panel.update_progress, "计算", .32)
    panel.finish(status, "磁盘写入失败" if status == "failed" else "用户停止")
    assert panel.state == status and not panel.isHidden()
    assert panel.phases["分析"].status == "completed"
    assert panel.phases["计算"].status == status
    assert panel.stage_bars["计算"].value() == 320
    assert panel.phases["检查"].status == panel.phases["缓存"].status == "waiting"
    before = panel.overall_progress
    # Late delivery from a stopped worker must not paint a successful result.
    panel.update_progress(1., "[缓存:1.000] 迟到消息")
    panel.set_task_progress(1.)
    assert panel.state == status and panel.overall_progress == before
    assert panel.stage_bars["缓存"].value() == 0


def test_next_run_cannot_reuse_terminal_values_or_clip_identity(panel):
    panel.begin("上一批.mov", 5, 5, batch=True)
    panel.update_progress(1., "[汇总:缓存:0.800]")
    panel.set_task_progress(1.)
    panel.finish("failed", "一条素材失败")
    panel.begin("新的素材.mp4", 1, 2, batch=True)
    assert panel.state == "running"
    assert panel.task_progress is None and panel.overall_progress == 0.
    assert all(phase.status == "waiting" and phase.fraction == 0. for phase in panel.phases.values())
    assert all(bar.value() == 0 for bar in panel.stage_bars.values())
    assert "上一批" not in panel.task_label.toolTip()
    assert "新的素材" in panel.task_label.toolTip()
    assert panel.detail == "等待分析开始"


def test_completed_batch_does_not_hide_partial_phase_evidence(panel):
    panel.begin("可用素材.mov", 1, 3, batch=True)
    panel.update_progress(1., "[汇总:缓存:0.333]")
    panel.finish("completed", "完成 1 条，跳过 2 条")
    assert panel.phases["缓存"].fraction == .333
    assert panel.phases["检查"].status == "waiting"
    assert "跳过 2 条" in panel.detail


def test_single_success_and_explicit_reset(panel):
    panel.begin("单条.mov")
    panel.finish()
    assert all(phase.status == "completed" for phase in panel.phases.values())
    assert panel.overall_bar.value() == 1000
    panel.reset()
    assert panel.isHidden() and panel.state == "idle"
    assert panel.overall_bar.value() == 0
    assert not panel.filename


def test_malformed_stage_message_and_invalid_measurement_cannot_invent_progress(panel):
    panel.begin(batch=True)
    panel.update_progress(.3, "[汇总:光流匹配:0.900]")
    assert all(phase.status == "waiting" for phase in panel.phases.values())
    panel.update_progress(.3, "[汇总:计算:invalid]")
    assert all(phase.fraction == 0. for phase in panel.phases.values())
    with pytest.raises(ValueError, match="有限"):
        panel.set_task_progress(float("nan"))
    with pytest.raises(ValueError, match="有限"):
        panel.update_progress(float("inf"), "未测量")
    assert panel.task_progress is None and panel.overall_progress == .3
    with pytest.raises(ValueError, match="未知跟踪阶段"):
        panel.set_phase("光流匹配", .9)


def test_long_unicode_filename_does_not_expand_canvas_minimum_width(panel):
    panel.begin("红色汽车与车标/" + "绕车素材镜头" * 80 + ".mov")
    panel.show()
    panel.resize(420, panel.sizeHint().height())
    QApplication.instance().processEvents()
    assert panel.task_label.minimumWidth() == 0
    assert panel.task_label.fontMetrics().horizontalAdvance(panel.task_label.text()) <= panel.task_label.width()
    assert "绕车素材镜头" in panel.task_label.toolTip()


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_terminal_collapses_only_presentation_and_can_reopen(panel, status):
    panel.begin("正在检查的汽车.mov", 3, 5, batch=True)
    panel.update_progress(.47, "[汇总:计算:0.680]")
    panel.set_task_progress(.81)
    panel.show()
    QApplication.instance().processEvents()
    expanded_height = panel.sizeHint().height()
    assert panel.expanded and not panel.details_widget.isHidden()
    panel.finish(status, "完成与问题区间仍需复核")
    QApplication.instance().processEvents()
    measured = {key: (value.fraction, value.status) for key, value in panel.phases.items()}
    assert not panel.expanded and panel.details_widget.isHidden()
    assert panel.expand_button.text() == "展开阶段"
    assert panel.sizeHint().height() <= 64
    assert panel.sizeHint().height() < expanded_height
    assert panel.state == status and panel.overall_progress == .47 and panel.task_progress == .81
    assert "3/5" in panel.task_label.toolTip() and "总体 47%" in panel.progress_label.text()
    panel.expand_button.click()
    QApplication.instance().processEvents()
    assert panel.expanded and not panel.details_widget.isHidden()
    assert panel.detail_label.toolTip() == "完成与问题区间仍需复核"
    assert {key: (value.fraction, value.status) for key, value in panel.phases.items()} == measured
    assert panel.state == status
    panel.expand_button.click()
    assert not panel.expanded
    panel.begin("下一轮.mov")
    assert panel.expanded and not panel.details_widget.isHidden()
    assert panel.state == "running" and panel.overall_progress == 0.
    assert all(value.status == "waiting" for value in panel.phases.values())


def test_followed_clip_phase_isolated_from_batch_aggregate_and_overall(panel):
    panel.begin("当前汽车.mov", 2, 5, batch=True, follow_task_phases=True)
    panel.update_task_progress(.61, "[计算:0.680] 当前素材双向匹配")
    assert panel.task_progress == .61 and panel.overall_progress == 0.
    assert panel.phases["分析"].status == "completed"
    assert panel.phases["计算"].fraction == .68
    assert panel.phases["检查"].status == "waiting"
    panel.update_progress(.47, "[汇总:检查:0.880]")
    panel.update_progress(.48, "素材任务 1/5 · 运行 2 · 等待 2")
    assert panel.overall_progress == .48 and panel.task_progress == .61
    assert panel.phases["计算"].fraction == .68
    assert panel.phases["检查"].status == "waiting"
    assert panel.detail == "当前素材双向匹配"
    assert panel.scope_label.text() == "当前素材阶段 · 总体进度单独统计"
    assert "当前 61%" in panel.progress_label.text() and "总体 48%" in panel.progress_label.text()


def test_switching_followed_clip_clears_only_current_phase_then_replays_real_event(panel):
    panel.begin("第一条.mov", 1, 3, batch=True, follow_task_phases=True)
    panel.update_progress(.34, "[汇总:缓存:0.333]")
    panel.update_task_progress(.95, "[缓存:0.000] 第一条写入中")
    panel.set_task("第二条.mov", 2, 3)
    assert panel.overall_progress == .34 and panel.task_progress is None
    assert all(value.status == "waiting" and value.fraction == 0. for value in panel.phases.values())
    assert "第一条" not in panel.detail
    panel.update_task_progress(.38, "[计算:0.350] 第二条的真实最新事件")
    assert panel.overall_progress == .34 and panel.task_progress == .38
    assert panel.phases["分析"].fraction == 1.
    assert panel.phases["计算"].fraction == .35
    assert panel.phases["缓存"].fraction == 0.
    panel.finish("cancelled")
    panel.update_task_progress(1., "[缓存:1.000] 迟到事件")
    assert panel.task_progress == .38 and panel.phases["缓存"].fraction == 0.
    panel.begin("新批次.mov", batch=True)
    assert not panel.follow_task_phases
    panel.update_progress(.2, "[汇总:计算:0.300]")
    assert panel.phases["计算"].fraction == .3
