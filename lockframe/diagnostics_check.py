"""Exercise the actual help entry and background ZIP exporter in source and EXE."""
from pathlib import Path
import json
import threading
import time
import traceback
import zipfile

from . import __version__


def wait_for(predicate,seconds=15):
    from PySide6.QtWidgets import QApplication
    deadline=time.monotonic()+seconds
    while not predicate():
        QApplication.processEvents();time.sleep(.01)
        if time.monotonic()>deadline:raise TimeoutError('diagnostics check timed out')
    QApplication.processEvents()


def check_diagnostics(folder):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from . import app as ui
    from .logging import get_logger,log_folder,SESSION_ID
    from .diagnostics_bundle import export_diagnostics
    application=QApplication.instance() or QApplication([])
    original=ui.QSettings
    ui.QSettings=lambda *a:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)
    try:w=ui.MainWindow()
    finally:ui.QSettings=original
    w.resize(1160,760);w.show();application.processEvents()
    try:
        try:raise RuntimeError('diagnostic-check api_key=sample-private-token')
        except RuntimeError:get_logger('diagnostics-check').exception('诊断异常样本')
        # Keep another genuine Job running while the dialog exports independently.
        entered=threading.Event();release=threading.Event();values=[]
        def task(cancel,progress):
            entered.set()
            if not release.wait(10):raise TimeoutError('test task not released')
            return 'done'
        w._job(task,(),values.append,'诊断并行验收')
        wait_for(entered.is_set);active=w.worker
        w.diagnostics_action.trigger();application.processEvents()
        dialog=w.diagnostics_dialog
        assert dialog.isVisible() and w.busy and w.worker is active
        destination=folder/'诊断包.zip'
        destination.write_bytes(b'keep-existing')
        dialog.begin_export(str(destination));wait_for(lambda:not dialog.running)
        bundle=Path(dialog.last_bundle)
        assert bundle!=destination and destination.read_bytes()==b'keep-existing'
        assert w.busy and w.worker is active
        with zipfile.ZipFile(bundle) as archive:
            assert archive.testzip() is None
            names=archive.namelist();report=json.loads(archive.read('diagnostics.json'))
            assert report['version']==__version__ and report['session']==SESSION_ID
            assert report['state']['busy'] and report['state']['clip_count']==0
            text='\n'.join(archive.read(n).decode('utf-8') for n in names)
            assert 'sample-private-token' not in text and '[REDACTED]' in text
            assert 'RuntimeError' in text and '诊断异常样本' in text
            assert 'thread-stacks.txt' in names and any(n.startswith('logs/') for n in names)
            assert all(n in ('diagnostics.json','thread-stacks.txt','说明.txt') or n.startswith('logs/') for n in names)
        protected=folder/'素材.zip';protected.write_bytes(b'original')
        try:export_diagnostics(protected,{},threading.Event(),protected_paths=[protected])
        except ValueError:pass
        else:raise AssertionError('protected output was not rejected')
        assert protected.read_bytes()==b'original' and not list(folder.glob('*.partial.zip'))
        dialog.grab().save(str(folder/'日志与诊断.png'))
        release.set();wait_for(lambda:not w.busy)
        assert values==['done']
        dialog.close()
        return dict(help_entry=True,background_export=True,independent_job=True,
                    version_session=True,exception_trace=True,credential_filter=True,
                    no_overwrite=True,source_protection=True,bundle=str(bundle),logs=str(log_folder()))
    finally:
        release.set() if 'release' in locals() else None
        if w.worker:
            w.worker.cancel();w.worker.wait(15000);application.processEvents()
        dialog=getattr(w,'diagnostics_dialog',None)
        if dialog and dialog.worker:
            dialog.cancel();dialog.worker.wait(15000);application.processEvents()
        w._confirm_discard=lambda:True;w.close();application.processEvents()


def run_diagnostics_check(directory):
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    report=dict(version=__version__,ok=False)
    try:report['diagnostics']=check_diagnostics(folder);report['ok']=True
    except Exception:report['error']=traceback.format_exc()
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['ok'] else 1
