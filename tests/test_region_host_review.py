"""Legacy ROI provenance and per-material history at the real MainWindow boundary."""
from dataclasses import replace
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest


@pytest.fixture
def legacy_window(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from lockframe import app
    from lockframe.core_check import create_source
    from lockframe.v072_check import activate

    application = QApplication.instance() or QApplication([])
    monkeypatch.setenv('LOCKFRAME_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr(app, 'QSettings', lambda *args: QSettings(str(tmp_path / 'preferences.ini'), QSettings.Format.IniFormat))
    window = app.MainWindow()
    errors = []
    window._error = lambda message: errors.append(str(message))
    window._confirm_discard = lambda: True
    clip = create_source(tmp_path / 'source')
    clip.track.anchors = None
    activate(window, [clip])
    window._seek(clip.start)
    try:
        yield window, clip, errors
    finally:
        window.dirty = False
        window._sequence_changed = False
        window.close()
        application.processEvents()


@pytest.mark.parametrize('edit', ['shape', 'geometry'])
def test_legacy_result_freezes_only_processing_anchors_and_remains_stale_after_reopen(legacy_window, tmp_path, edit):
    from lockframe.models import read_project
    from lockframe.sequence import save_sequence, read_sequence
    from lockframe.v072_check import wait_job

    window, clip, errors = legacy_window
    original = window.track
    original_active = dict(window.anchors)
    # This is legal clip data: the processing range need not span all ROI keys.
    window.anchors[0] = (.1, .1, .2, .2)
    assert not window._region_stale()
    if edit == 'shape':
        window._region_choose_shape('ellipse')
    else:
        window._select_roi((.3, .3, .4, .4))
    assert window._region_stale() and not window.export_button.isEnabled()
    assert window.track.anchors == original_active
    assert original.anchors is None  # History must retain the unmodified legacy object.
    window.track.validate(window.info.frames)
    window._tb_operation('undo')
    assert window.track is original and not window._region_stale()
    assert 0 in window.anchors
    window._tb_operation('redo')
    assert window._region_stale() and window.track.anchors == original_active
    window._stash_clip()
    sequence_path = tmp_path / '含区间外修正点.locksequence'
    save_sequence(sequence_path, window.clips)
    assert read_sequence(sequence_path)[0].track.anchors == original_active
    window.project_path = str(tmp_path / '修改后过期.lockframe')
    assert window._save_project()
    payload = read_project(window.project_path)
    payload['_project_path'] = window.project_path
    window.load_video(payload['source'], project=payload)
    wait_job(window)
    assert not errors
    assert window._region_stale() and not window.export_button.isEnabled()
    assert window.track.anchors == original_active and 0 in window.anchors


def test_explicit_empty_result_provenance_survives_project_reopen_and_blocks_export(legacy_window, tmp_path):
    from lockframe.models import read_project
    from lockframe.v072_check import wait_job

    window, clip, errors = legacy_window
    window.anchors = {}  # A legacy measured path can be loaded with no saved ROI keys.
    window._show_current()
    window._select_roi((.3, .3, .4, .4))
    assert window._region_stale() and window.track.anchors == {}
    window.project_path = str(tmp_path / '旧轨迹缺少修正点.lockframe')
    assert window._save_project()
    payload = read_project(window.project_path)
    assert payload['track']['anchors'] == {}  # Empty provenance is not missing provenance.
    payload['_project_path'] = window.project_path
    window.load_video(payload['source'], project=payload)
    wait_job(window)
    assert not errors
    assert window.track.anchors == {} and window._region_stale()
    assert not window.export_button.isEnabled()
    window._stash_clip()
    from lockframe.batch_export import prepare_exports
    assert '关键帧已修改' in prepare_exports([window._clip()], tmp_path / '不可导出')[0].reason


def test_shape_undo_history_is_isolated_per_clip_after_repeated_activation(legacy_window):
    from lockframe.sequence import Clip

    window, clip, errors = legacy_window
    other = Clip(clip.info, replace(clip.settings), dict(clip.anchors), clip.start, clip.end,
                 replace(clip.track, anchors=dict(clip.anchors)), frame=clip.start)
    window.clips.append(other)
    window._region_choose_shape('polygon')
    window._activate_clip(other.id)
    assert window.settings().target_shape == 'rectangle'
    window._region_choose_shape('ellipse')
    window._tb_operation('undo')
    assert window.settings().target_shape == 'rectangle'
    assert window._clip(clip.id).settings.target_shape == 'polygon'
    window._activate_clip(clip.id)
    assert window.settings().target_shape == 'polygon'
    window._tb_operation('undo')
    assert window.settings().target_shape == 'rectangle' and not window._region_stale()
    window._tb_operation('redo')
    assert window.settings().target_shape == 'polygon' and window._region_stale()
    window._activate_clip(other.id)
    assert window.settings().target_shape == 'rectangle' and not window._region_stale()
    window._tb_operation('redo')
    assert window.settings().target_shape == 'ellipse' and window._region_stale()
    assert not errors


def test_composition_mode_blocks_roi_arrow_edits_until_region_editing_resumes(legacy_window):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    window, clip, errors = legacy_window
    window.show()
    window.pick_mode.setCurrentIndex(window.pick_mode.findData('adjust'))
    window.composition_edit.setChecked(True)
    window.canvas.setFocus()
    QApplication.processEvents()
    before_anchors = dict(window.anchors)
    before_track = window.track
    before_shape = (window.settings().target_shape, window.settings().polygon_points)
    QTest.keyClick(window.canvas, Qt.Key.Key_Right)
    assert window.anchors == before_anchors and window.track is before_track
    assert (window.settings().target_shape, window.settings().polygon_points) == before_shape
    assert window.composition_edit.isChecked() and window.canvas.composition_editable
    assert not window._region_stale()
    window.composition_edit.setChecked(False)
    QTest.keyClick(window.canvas, Qt.Key.Key_Right)
    assert window.anchors != before_anchors and window._region_stale()
    assert not errors
