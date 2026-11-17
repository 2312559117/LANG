from copy import deepcopy
from pathlib import Path
import hashlib

from PySide6.QtCore import QTimer, Qt, QPoint
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from test_workflow_v9 import window, source
from lockframe.v072_check import activate


def context(window, row, *, trigger=True, before=None):
    """Enter the actual menu event loop; capture the fixed target before any refresh."""
    item=window.clip_list.item(row)
    point=window.clip_list.visualItemRect(item).center()
    observed={}
    def choose():
        menu=QApplication.activePopupWidget()
        assert isinstance(menu,QMenu) and menu.objectName()=='clipContextMenu'
        action=menu.actions()[0]
        observed.update(enabled=action.isEnabled(),text=action.text(),selected=window._selected_clip_ids())
        if before:before()
        if trigger and action.isEnabled():action.trigger()
        menu.close()
    QTimer.singleShot(30,choose)
    window.clip_list.customContextMenuRequested.emit(point)
    return observed


def test_context_targets_only_clicked_clip_and_preserves_other_tracks_and_source(window,source):
    first,second=deepcopy(source),deepcopy(source);second.id='second-context'
    activate(window,[first,second]);window.clip_list.selectAll()
    fingerprint=hashlib.sha256(Path(source.info.path).read_bytes()).hexdigest()
    original=window.track
    # The callback must not depend on a mutable selection while a menu is open.
    report=context(window,1,before=window.clip_list.selectAll)
    assert report['selected']=={second.id} and report['enabled'] and '删除' in report['text']
    assert [c.id for c in window.clips]==[first.id]
    assert window.active_clip_id==first.id and window.track is original
    assert hashlib.sha256(Path(source.info.path).read_bytes()).hexdigest()==fingerprint
    assert '原视频保留' in window.status_label.text()


def test_context_cancel_busy_and_unsaved_decline_leave_clip_untouched(window,source,monkeypatch):
    clip=deepcopy(source);activate(window,[clip]);track=window.track
    context(window,0,trigger=False)
    assert len(window.clips)==1 and window.track is track
    window.busy=True
    try:assert not context(window,0)['enabled']
    finally:window.busy=False
    window.dirty=True
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:QMessageBox.StandardButton.No)
    context(window,0)
    assert len(window.clips)==1 and window.active_clip_id==clip.id and window.track is track
    # Empty list area has no menu and cannot affect selection or source.
    window._clip_context_menu(QPoint(-5,-5))
    assert window.track is track


def test_context_removing_current_then_last_clears_preview_and_clip_history(window,source,monkeypatch):
    first,second=deepcopy(source),deepcopy(source);second.id='remaining-context'
    activate(window,[first,second]);window._tb_history();window._timebar_clipboard=(first.id,[(0,(0,0,1,1))])
    window._paused_by_clip[first.id]={'frame':3};window._paused_queue_ids=[first.id,second.id]
    context(window,0)
    assert window.active_clip_id==second.id and window.info==second.info
    assert first.id not in window._timebar_histories and first.id not in window._paused_by_clip
    assert window._paused_queue_ids==[second.id] and window._timebar_clipboard==(None,[])
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:QMessageBox.StandardButton.Yes)
    context(window,0)
    assert not window.clips and window.clip_list.count()==0 and window.active_clip_id is None
    assert window.info is window.reader is window.frame is window.track is window.plan is None
    assert not window.export_button.isEnabled() and not window.track_button.isEnabled()
