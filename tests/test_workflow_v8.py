"""Integration contracts for the chosen reference-sidebar workflow."""
import os
from copy import deepcopy
from dataclasses import replace
import shutil
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QSettings
from PySide6.QtWidgets import QApplication, QLabel

from lockframe.demo import demo_frame
from lockframe.media import configure_cache_preferences, import_video, resource_root
from lockframe.models import Settings
from lockframe.sequence import Clip
from lockframe.tracking import track_video


@pytest.fixture(scope="module")
def source():
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    anchors = {10: demo_frame(10, info.width, info.height)[1]}
    result = track_video(info, anchors, 0, 24, threading.Event())
    return Clip(info, Settings(acceleration="cpu", compute_threads=2), anchors, 0, 24, result)


@pytest.fixture
def window(source, tmp_path, monkeypatch):
    from lockframe import app
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app, "QSettings", lambda *args: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    window = app.MainWindow()
    window.clips = [deepcopy(source)]
    window._activate_clip(window.clips[0].id)
    window.resize(1160, 760)
    window.show()
    application.processEvents()
    yield window
    if window.busy and window.worker:
        window.worker.cancel()
        window.worker.wait(15000)
        application.processEvents()
    window._confirm_discard = lambda: True
    window.dirty, window._sequence_changed = False, False
    for clip in window.clips:
        clip.dirty = False
    window.close()
    configure_cache_preferences(None)


def test_three_columns_retain_real_step_navigation_and_small_preview(window):
    assert window.workspace_splitter.count() == 3
    assert window.workspace_splitter.widget(0) is window.materials_sidebar
    assert window.workspace_splitter.widget(1) is window.central_preview
    assert window.workspace_splitter.widget(2) is window.task_sidebar
    assert window.asset_status_bar.isVisible()
    for navigator, controls in ((window.library_tabs, (window.clip_list, window.track_button, window.marker_tabs)),
                                (window.inspector_tabs, (window.target_x_spin, window.region_apply,
                                                         window.acceleration_combo, window.resolution_combo))):
        for index in range(navigator.count()):
            navigator.setCurrentIndex(index)
            QApplication.processEvents()
            assert navigator.currentIndex() == index
            assert controls[index].isVisible()
    assert window.central_preview.width() >= 560
    assert window.canvas.width() > 350 and window.canvas.height() > 150
    assert window.canvas.visibleRegion().contains(window.canvas.rect())
    window.inspector_tabs.setCurrentIndex(0)
    QApplication.processEvents()
    composition_labels = [label for label in window.findChildren(QLabel) if "构图位置" in label.text()]
    assert len(composition_labels) == 2
    assert all(window.settings_dialog.isAncestorOf(label) and label.isVisible() for label in composition_labels)
    window.settings_dialog.hide()
    QApplication.processEvents()
    assert window.materials_sidebar.isAncestorOf(window.clip_list)
    sidebar_left = window.task_sidebar.mapTo(window, QPoint(0, 0)).x()
    sidebar_right = sidebar_left + window.task_sidebar.width()
    for control in (window.track_button, window.pause_button, window.continue_button,
                    window.center_button, window.composition_edit, *window.composition_sliders,
                    window.export_button):
        assert window.task_sidebar.isAncestorOf(control)
        if not control.isVisible():
            continue  # Idle pause/continue actions retain their existing visibility policy.
        left = control.mapTo(window, QPoint(0, 0)).x()
        right = control.mapTo(window, QPoint(control.width(), 0)).x()
        assert sidebar_left <= left < right <= sidebar_right
    export_bottom = window.export_button.mapTo(window.task_sidebar, QPoint(0, window.export_button.height())).y()
    assert window.track_button.isVisible() and window.center_button.isVisible() and window.export_button.isVisible()
    assert export_bottom <= window.task_sidebar.height()
    assert window.timeline_well.parentWidget() is window.centralWidget()
    assert abs(window.timeline_well.width()-window.workspace_splitter.width()) <= 2


def test_persistent_bar_switches_unicode_long_material_without_widening_window(window, source, tmp_path):
    path = tmp_path / ("红色汽车_" * 12 + ".mp4")
    shutil.copy2(source.info.path, path)
    info = import_video(str(path), threading.Event())
    other = replace(deepcopy(source), info=info, id="long-unicode-source")
    window.clips.append(other)
    previous_width = window.width()
    window._activate_clip(other.id)
    QApplication.processEvents()
    bar = window.asset_status_bar
    assert bar.source_path == str(path)
    assert bar.full_filename == path.name
    assert bar.filename_label.toolTip() == str(path)
    assert f"{info.width} × {info.height}" in bar.metadata_label.text()
    assert f"{info.fps:g} fps" in bar.metadata_label.text()
    assert window.width() == previous_width
    assert bar.geometry().right() <= window.centralWidget().width()
    window._activate_clip(window.clips[0].id)
    assert bar.source_path == source.info.path and bar.full_filename == "demo.mp4"


def test_composition_preserves_measured_track(window, monkeypatch):
    from lockframe import app
    monkeypatch.setattr(app, "track_video", lambda *args, **kwargs: pytest.fail("构图不应重跑跟踪"))
    result = window.track
    positions = result.positions.copy()
    window._set_composition(.63, .42)
    assert window.track is result and (result.positions == positions).all()


def test_failed_tracking_is_not_rendered_as_success(window, monkeypatch):
    from lockframe import app
    monkeypatch.setattr(app.QMessageBox, "warning", lambda *args: None)
    def fail(*args, progress, **kwargs):
        progress(.32, "[计算:0.274] 第 10 帧")
        raise RuntimeError("测试解码失败")
    monkeypatch.setattr(app, "track_video", fail)
    previous_track = window.track
    window._start_tracking()
    deadline = time.monotonic() + 10
    while window.busy and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(.005)
    assert not window.busy
    panel = window.tracking_progress_panel
    assert panel.state == "failed"
    assert panel.phases["计算"].fraction < 1.
    assert panel.phases["检查"].status == "waiting"
    assert window.track is previous_track


def test_persistent_monitors_survive_view_requests_and_tracking(window):
    from PySide6.QtCore import QPoint
    assert window.canvas.mode == "compare" and window.canvas.persistent_compare
    for mode in ("source", "output", "compare"):
        window._set_view(mode)
        assert window.canvas.mode == "compare"
    window._hold_source()
    assert window.canvas.mode == "compare"
    window._release_source()
    payload = {"frame": 10, "box": list(window.track.boxes[10]), "confidence": 1., "state": "tracked", "completed": 11, "total": 25}
    window._display_live_observation(payload, window.info, window.reader)
    QApplication.processEvents()
    assert window.canvas.mode == "compare" and window.canvas.processed is None
    assert window.canvas.source_rect.isValid() and window.canvas.output_rect.isValid()
    assert window.canvas.output_rect.left() >= window.canvas.width() / 2


