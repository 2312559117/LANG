import json
import logging
import threading
import zipfile
from pathlib import Path

import pytest
from lockframe import diagnostics_bundle as bundle
from lockframe.logging import LocalFormatter,redact
from lockframe.media import Cancelled


@pytest.mark.parametrize('value',[
    'sk-123456789abcdef','Bearer abcdefghijkl','api_key=privatevalue',
    'password: privatevalue','"api_key": "privatevalue"',"secret='privatevalue'",
])
def test_common_credentials_filtered(value):
    result=redact(value)
    assert '[REDACTED]' in result
    assert all(token not in result for token in ('123456789abcdef','abcdefghijkl','privatevalue'))


def test_formatter_redacts_tracebacks_and_records_session_thread():
    try:raise RuntimeError('password=privatevalue')
    except RuntimeError:
        import sys
        record=logging.LogRecord('lockframe.test',logging.ERROR,__file__,1,'failed',(),sys.exc_info())
    text=LocalFormatter('%(session)s %(threadName)s %(message)s').format(record)
    assert 'RuntimeError' in text and '[REDACTED]' in text and 'privatevalue' not in text
    assert 'MainThread' in text


def test_bounded_allowlist_unicode_collision_and_json_strings(tmp_path,monkeypatch):
    logs=tmp_path/'日志';logs.mkdir()
    (logs/'tomato-tracker.log').write_text('X'*2_100_000+'\napi_key=privatevalue\n',encoding='utf-8')
    (logs/'tomato-tracker.log.1').write_text('previous',encoding='utf-8')
    (logs/'secret-video.mp4').write_bytes(b'not-collected')
    (logs/'unknown.txt').write_text('not-collected')
    (logs/'native-crash-1.log').write_text('trace',encoding='utf-8')
    (logs/'native-crash-user.log').write_text('not-collected',encoding='utf-8')
    target=tmp_path/'诊断.zip';target.write_bytes(b'preserve')
    monkeypatch.setenv('MY_PRIVATE_ENV','not-collected')
    output=Path(bundle.export_diagnostics(target,{'status':'password="privatevalue" \\ 路径'},threading.Event(),logs_directory=logs))
    assert output.name=='诊断_2.zip' and target.read_bytes()==b'preserve'
    with zipfile.ZipFile(output) as archive:
        names=archive.namelist()
        assert 'logs/native-crash-1.log' in names and not any('unknown' in n or 'video' in n or 'user.log' in n for n in names)
        assert len(archive.read('logs/tomato-tracker.log'))<=2_000_100
        report=json.loads(archive.read('diagnostics.json'))
        assert report['state']['status']=='password="[REDACTED]" \\ 路径'
        assert 'privatevalue' not in archive.read('logs/tomato-tracker.log').decode()
        assert 'MY_PRIVATE_ENV' not in archive.read('diagnostics.json').decode()
    assert not list(tmp_path.glob('*.partial.zip'))


def test_missing_logs_still_exports_environment_and_state(tmp_path):
    target=bundle.export_diagnostics(tmp_path/'empty.zip',{'busy':False},threading.Event(),logs_directory=tmp_path/'missing')
    with zipfile.ZipFile(target) as archive:
        assert json.loads(archive.read('diagnostics.json'))['included_logs']==[]
        assert 'thread-stacks.txt' in archive.namelist()


def test_source_protection_rejects_before_touching_source(tmp_path):
    target=tmp_path/'source.zip';target.write_bytes(b'original')
    with pytest.raises(ValueError,match='不能覆盖'):
        bundle.export_diagnostics(target,{},threading.Event(),protected_paths=[target])
    assert target.read_bytes()==b'original'
    assert not list(tmp_path.glob('*.partial.zip'))


def test_cancel_during_collection_removes_partial_and_keeps_completed(tmp_path):
    logs=tmp_path/'logs';logs.mkdir();(logs/'tomato-tracker.log').write_text('record')
    completed=tmp_path/'done.zip';completed.write_bytes(b'complete')
    cancel=threading.Event()
    def progress(value,text):cancel.set()
    with pytest.raises(Cancelled):
        bundle.export_diagnostics(tmp_path/'cancel.zip',{},cancel,progress,logs_directory=logs)
    assert completed.read_bytes()==b'complete' and not (tmp_path/'cancel.zip').exists()
    assert not list(tmp_path.glob('*.partial.zip'))


def test_write_failure_cleans_partial_and_keeps_existing(tmp_path,monkeypatch):
    original=zipfile.ZipFile.writestr
    def fail(self,*a,**k):
        original(self,*a,**k)
        raise OSError('disk full test')
    monkeypatch.setattr(zipfile.ZipFile,'writestr',fail)
    target=tmp_path/'existing.zip';target.write_bytes(b'good')
    with pytest.raises(OSError,match='disk full'):
        bundle.export_diagnostics(target,{},threading.Event(),logs_directory=tmp_path/'none')
    assert target.read_bytes()==b'good' and not (tmp_path/'existing_2.zip').exists()
    assert not list(tmp_path.glob('*.partial.zip'))


def test_unreadable_log_reported_without_failing_bundle(tmp_path,monkeypatch):
    log=tmp_path/'tomato-tracker.log';log.write_text('record')
    original=Path.open
    def fail(self,*a,**k):
        if self==log and a and a[0]=='rb':raise PermissionError('locked')
        return original(self,*a,**k)
    monkeypatch.setattr(Path,'open',fail)
    result=bundle.export_diagnostics(tmp_path/'result.zip',{},threading.Event(),logs_directory=tmp_path)
    with zipfile.ZipFile(result) as archive:
        report=json.loads(archive.read('diagnostics.json'))
        assert report['unavailable_logs']==[{'file':log.name,'error':'PermissionError'}]
