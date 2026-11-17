from pathlib import Path
import json
import threading
import zipfile
from test_workflow_v9 import window,source
from lockframe.diagnostics_check import wait_for


def test_real_help_export_independent_of_active_job(window,tmp_path):
    entered=threading.Event();release=threading.Event()
    def task(cancel,progress):entered.set();release.wait(10);return True
    window._job(task,(),lambda result:None,'test')
    try:
        wait_for(entered.is_set);worker=window.worker
        window.diagnostics_action.trigger();dialog=window.diagnostics_dialog
        dialog.begin_export(str(tmp_path/'诊断.zip'));wait_for(lambda:not dialog.running)
        assert window.worker is worker and window.busy
        with zipfile.ZipFile(dialog.last_bundle) as archive:
            assert json.loads(archive.read('diagnostics.json'))['state']['busy']
    finally:
        release.set();wait_for(lambda:not window.busy)


def test_capture_snapshot_isolated_from_later_material_edits(window,source):
    from copy import deepcopy
    from lockframe.v072_check import activate
    from lockframe.diagnostics_bundle import capture_state
    activate(window,[deepcopy(source)])
    state,protected=capture_state(window)
    original=state['media']['start'];mode=state['processing']['tracking_mode']
    window.start_spin.setValue(3);window.status_label.setText('changed')
    assert state['media']['start']==original and state['processing']['tracking_mode']==mode
    assert state['status']!='changed' and source.info.path in protected


def test_close_main_while_diagnostics_running_waits_for_worker(window,tmp_path,monkeypatch):
    import lockframe.diagnostics_ui as ui
    from lockframe.media import check_cancel
    entered=threading.Event()
    def slow(*a,cancel,**k):entered.set();cancel.wait(5);check_cancel(cancel)
    monkeypatch.setattr(ui,'export_diagnostics',slow)
    window._open_diagnostics();dialog=window.diagnostics_dialog
    dialog.begin_export(str(tmp_path/'cancel.zip'));wait_for(entered.is_set)
    window._confirm_discard=lambda:True
    window.close();wait_for(lambda:not dialog.running)
    wait_for(lambda:not window.isVisible())
    assert not (tmp_path/'cancel.zip').exists() and window.worker is None


def test_declined_close_does_not_reappear_after_next_diagnostic_export(window,tmp_path,monkeypatch):
    import lockframe.diagnostics_ui as ui
    from lockframe.media import check_cancel
    original=ui.export_diagnostics;entered=threading.Event();confirmations=[]
    def slow(*a,cancel,**k):entered.set();cancel.wait(5);check_cancel(cancel)
    monkeypatch.setattr(ui,'export_diagnostics',slow)
    window._confirm_discard=lambda:confirmations.append(True) or False
    dialog=window._open_diagnostics()
    dialog.begin_export(str(tmp_path/'cancel.zip'));wait_for(entered.is_set)
    window.close();wait_for(lambda:not dialog.running)
    assert window.isVisible() and len(confirmations)==1
    monkeypatch.setattr(ui,'export_diagnostics',original)
    dialog.begin_export(str(tmp_path/'next.zip'));wait_for(lambda:not dialog.running)
    assert dialog.last_bundle and window.isVisible() and len(confirmations)==1
