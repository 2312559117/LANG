"""Transactional Qt editor coverage; no host window or video I/O required."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QSignalSpy
from lockframe.time_remap_editor import TimeRemapEditor
from lockframe.time_remap_model import normalize_config, active_points
from lockframe.time_remap_map import output_frame_count, output_frame_map


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor(app):
    widget = TimeRemapEditor(None, 240, 24.)
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_transaction_apply_cancel_isolation(app):
    config = normalize_config()
    original = deepcopy(config)
    widget = TimeRemapEditor(config, 240, 24.)
    spy = QSignalSpy(widget.applied)
    widget.set_constant_speed(2.)
    assert config == original and spy.count() == 0
    assert not widget.isModal()
    assert widget.apply_changes()
    assert spy.count() == 1
    applied = deepcopy(spy.at(0)[0])
    spy.at(0)[0]['basic']['speeds'][0] = 7.
    assert widget.draft['basic']['speeds'][0] == 1.
    widget.set_constant_speed(-3.)
    widget.reject()
    assert widget.draft == applied
    assert spy.count() == 1
    widget.deleteLater()


def test_canvas_intermediate_changes_form_single_undo(editor):
    initial = deepcopy(editor.draft)
    for speed in (1.2, 1.4, 2.):
        draft = deepcopy(editor.draft)
        draft['basic']['speeds'][2] = speed
        editor.canvas.changing.emit(draft)
    assert len(editor._history) == 1
    editor.canvas.committed.emit(deepcopy(editor.draft))
    assert len(editor._history) == 2
    editor.undo()
    assert editor.draft == initial
    editor.redo()
    assert editor.draft['basic']['speeds'][2] == 2.


def test_basic_conversion_preserves_duration_and_shape(editor):
    config = normalize_config()
    config['enabled'] = True
    config['basic']['speeds'] = [2.] * 5
    editor.set_config(config)
    before = output_frame_map(editor.draft, 240)
    editor.convert_basic()
    after = output_frame_map(editor.draft, 240)
    assert editor.draft['mode'] == 'advanced'
    assert len(after) == 120
    np.testing.assert_allclose(after, before)
    assert editor.draft['points'][-1]['position'] == pytest.approx(.5)
    editor.undo()
    assert editor.draft['mode'] == 'basic'


def test_advanced_full_fields_and_optional_auto_round_trip(editor):
    config = normalize_config(dict(version=2, enabled=True, mode='advanced', points=[
        dict(position=0., source=0., outgoing_speed=2., right_speed_tangent=4., right_influence=.2, right_handle_source=.3, handles_linked=True),
        dict(position=.5, source=.8, incoming_speed=.7, outgoing_speed=-.4, left_speed_tangent=2., right_speed_tangent=-3., left_influence=.4, right_influence=.6, left_handle_source=.6, right_handle_source=.7),
        dict(position=1., source=.6, incoming_speed=-1., left_speed_tangent=1.),
    ], optical_flow=True, preview_flow=True, allow_frame_fallback=True, audio_policy='mapped', flow_backend='dis'))
    editor.set_config(config)
    assert editor.draft == config
    editor._select(1)
    assert editor.point_controls['incoming_speed'].text() == '.7' or float(editor.point_controls['incoming_speed'].text()) == .7
    editor.point_controls['incoming_speed'].setText('')
    editor.point_controls['incoming_speed'].editingFinished.emit()
    assert editor.draft['points'][1]['incoming_speed'] is None
    editor.point_controls['right_speed_tangent'].setText('-8.5')
    editor.point_controls['right_speed_tangent'].editingFinished.emit()
    assert editor.draft['points'][1]['right_speed_tangent'] == -8.5
    assert editor.apply_changes()
    assert editor.draft['optical_flow'] and editor.draft['preview_flow']
    assert editor.draft['allow_frame_fallback'] and editor.draft['audio_policy'] == 'mapped'


def test_reverse_freeze_and_mode_switch_keep_advanced_data(editor):
    editor.set_constant_speed(-2.)
    points = deepcopy(editor.draft['points'])
    assert output_frame_count(240, active_points(editor.draft)) == 120
    editor.mode.setCurrentIndex(editor.mode.findData('basic'))
    editor.mode.setCurrentIndex(editor.mode.findData('advanced'))
    assert editor.draft['points'] == points
    editor.set_freeze(.4, 2.)
    frame_map = output_frame_map(editor.draft, 240)
    assert len(frame_map) == 48
    np.testing.assert_allclose(frame_map, .4 * 239)
    editor.undo()
    assert editor.draft['points'] == points


def test_node_table_selection_add_remove_and_undo(editor):
    editor.convert_basic()
    size = len(editor.draft['points'])
    editor.table.setCurrentCell(2, 0)
    editor.add_point()
    assert len(editor.draft['points']) == size + 1
    assert editor.table.currentRow() == 3
    editor.point_controls['source'].setValue(.8)
    editor.point_controls['source'].editingFinished.emit()
    assert editor.draft['points'][3]['source'] == .8
    editor.remove_point()
    assert len(editor.draft['points']) == size
    editor.undo()
    assert len(editor.draft['points']) == size + 1


def test_invalid_visible_input_blocks_apply(editor):
    editor.convert_basic()
    spy = QSignalSpy(editor.applied)
    editor.point_controls['outgoing_speed'].setText('nan')
    editor.point_controls['outgoing_speed'].editingFinished.emit()
    assert not editor.apply_changes()
    assert spy.count() == 0
    editor.point_controls['outgoing_speed'].setText('')
    editor.point_controls['outgoing_speed'].editingFinished.emit()
    assert editor.apply_changes()
    assert spy.count() == 1


def test_reset_history_and_supported_backend(editor):
    assert [editor.backend.itemData(i) for i in range(editor.backend.count())] == ['auto', 'dis']
    editor.set_constant_speed(3.)
    editor.reset()
    assert editor.draft == normalize_config(normalize_config())
    editor.undo()
    assert editor.draft['points'][-1]['position'] == pytest.approx(1 / 3)
    editor.redo()
    assert editor.draft == normalize_config(normalize_config())


@pytest.mark.parametrize('count,fps', [(0, 24.), (True, 24.), (10, 0.), (10, float('nan'))])
def test_invalid_source_contract(app, count, fps):
    with pytest.raises(ValueError):
        TimeRemapEditor(None, count, fps)
