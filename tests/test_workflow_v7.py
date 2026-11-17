import os
from copy import deepcopy
from dataclasses import replace
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

from lockframe.demo import demo_frame
from lockframe.media import import_video, resource_root, configure_cache_preferences
from lockframe.models import Settings
from lockframe.sequence import Clip, save_sequence, read_sequence
from lockframe.tracking import track_video


@pytest.fixture(scope="module")
def source():
    info = import_video(str(resource_root() / "assets/demo.mp4"), threading.Event())
    anchors = {10: demo_frame(10)[1]}
    result = track_video(info, anchors, 0, 29, threading.Event())
    return Clip(info, Settings(acceleration="cpu"), anchors, 0, 29, result)


@pytest.fixture
def window(source, tmp_path, monkeypatch):
    from lockframe import app
    application = QApplication.instance() or QApplication([])
    monkeypatch.setattr(app, "QSettings", lambda *args: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    window = app.MainWindow()
    window.clips = [deepcopy(source)]
    window._activate_clip(window.clips[0].id)
    window.show()
    application.processEvents()
    yield window
    window._confirm_discard = lambda: True
    window.close()
    configure_cache_preferences(None)


def test_steps_expose_tracking_and_review_without_long_scroll(window):
    assert [window.library_tabs.tabText(i) for i in range(3)] == ["素材", "跟踪", "复核"]
    window._workflow_step(1)
    assert window.library_tabs.currentIndex() == 2 and window.issue_list.isVisible()
    window.library_tabs.setCurrentIndex(1)
    assert window.track_button.isVisible() and window.live_preview_check.isVisible()
    window._workflow_step(2)
    assert window.inspector_tabs.currentIndex() == 0 and window.target_x_spin.isVisible()
    window._workflow_step(3)
    assert window.inspector_tabs.currentIndex() == 3
    assert window.batch_export_button.isVisible()


