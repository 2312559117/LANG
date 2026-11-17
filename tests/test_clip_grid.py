from copy import deepcopy
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget
from test_workflow_v9 import window, source
from test_clip_context import context
from lockframe.v072_check import activate


def test_grid_toggle_preserves_clip_order_selection_tracking_and_plan(window,source):
    first,second=deepcopy(source),deepcopy(source);second.id='grid-second'
    activate(window,[first,second]);original=window.track;plan=window.plan
    selected=window._selected_clip_ids();active=window.active_clip_id
    assert window._clip_view_mode=='list' and window.clip_list.viewMode()==QListWidget.ViewMode.ListMode
    window.clip_view_actions['grid'].trigger()
    assert window._clip_view_mode=='grid' and window.clip_list.viewMode()==QListWidget.ViewMode.IconMode
    assert window._selected_clip_ids()==selected and window.active_clip_id==active
    assert window.track is original and window.plan is plan and window.worker is None and not window.busy
    assert [c.id for c in window.clips]==[first.id,second.id]
    assert window.preferences.value('assets/view')=='grid'
    assert window.clip_view_actions['grid'].isChecked() and not window.clip_view_actions['list'].isChecked()
    window.clip_view_actions['list'].trigger()
    assert window.clip_list.viewMode()==QListWidget.ViewMode.ListMode and not window.clip_list.isWrapping()
    assert window.track is original and window.plan is plan and window._selected_clip_ids()==selected


def test_grid_preference_restores_on_reopen_and_context_removes_only_one(window,source):
    from lockframe.app import MainWindow
    window._set_clip_view_mode('grid')
    other=MainWindow()
    try:
        assert other._clip_view_mode=='grid' and other.materials_sidebar.maximumWidth()==900
        assert other.clip_view_actions['grid'].isChecked()
    finally:
        other._confirm_discard=lambda:True;other.close()
    first,second=deepcopy(source),deepcopy(source);second.id='grid-remove'
    activate(window,[first,second]);window.clip_list.selectAll()
    context(window,1)
    assert [c.id for c in window.clips]==[first.id] and window.active_clip_id==first.id
    assert window._clip_view_mode=='grid'


def test_grid_rechecks_every_visible_cell_including_after_scroll(window,source,monkeypatch):
    clips=[]
    for i in range(35):
        clip=deepcopy(source);clip.id=f'cell-{i}';clips.append(clip)
    activate(window,clips);window._set_clip_view_mode('grid')
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    batches=[];monkeypatch.setattr(window._thumbnails,'recheck',lambda rows:batches.append({c.id for c in rows}))
    def verify():
        window._refresh_visible_thumbnails()
        viewport=window.clip_list.viewport().rect()
        expected={window.clip_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(35)
                  if window.clip_list.visualItemRect(window.clip_list.item(i)).intersects(viewport)}
        assert batches[-1]==expected and len(expected)>4
    verify()
    window.clip_list.scrollToBottom();QApplication.processEvents();verify()
    assert 'cell-34' in batches[-1]
