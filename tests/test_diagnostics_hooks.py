"""Use child processes so exception hooks and native trace files cannot leak into pytest."""
import os
from pathlib import Path
import subprocess
import sys


def run_child(code,tmp_path):
    return subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=Path(__file__).resolve().parents[1],
                          env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8',timeout=20)


def test_actual_python_thread_and_nonfatal_native_trace_destinations(tmp_path):
    result=run_child('''
import faulthandler,sys,threading
from pathlib import Path
from lockframe import logging as logs
folder=Path(sys.argv[1]);logs.configure_logging(folder);logs.install_excepthook()
try:raise ValueError('python-hook api_key=privatevalue')
except ValueError:sys.excepthook(*sys.exc_info())
def fail():raise RuntimeError('thread-hook password=privatevalue')
t=threading.Thread(target=fail,name='diagnostic-thread');t.start();t.join()
faulthandler.dump_traceback(file=logs._native_stream,all_threads=True)
logs._native_stream.flush()
for h in logs.configure_logging().handlers:h.flush()
''',tmp_path)
    assert result.returncode==0,result.stderr
    text=(tmp_path/'tomato-tracker.log').read_text(encoding='utf-8')
    assert 'python-hook' in text and 'thread-hook' in text and 'diagnostic-thread' in text
    assert 'privatevalue' not in text and '[REDACTED]' in text
    assert 'ValueError' in text and 'RuntimeError' in text
    traces=list(tmp_path.glob('native-crash-*.log'))
    assert len(traces)==1 and 'Current thread' in traces[0].read_text(encoding='utf-8')


def test_unavailable_log_directory_falls_back_and_startup_error_redacts(tmp_path):
    result=run_child('''
from pathlib import Path
import sys,tempfile
from lockframe import logging as logs
folder=Path(sys.argv[1]);blocked=folder/'blocked';blocked.write_bytes(b'file')
tempfile.tempdir=str(folder)
logger=logs.configure_logging(blocked/'logs')
assert logs.log_folder()==folder/'tomato-tracker-logs'
logs.get_logger('fallback').info('fallback active')
logs.startup_error('startup password=privatevalue')
for h in logger.handlers:h.flush()
''',tmp_path)
    assert result.returncode==0,result.stderr
    folder=tmp_path/'tomato-tracker-logs'
    assert 'fallback active' in (folder/'tomato-tracker.log').read_text(encoding='utf-8')
    assert (folder/'startup-error.log').read_text(encoding='utf-8')=='startup password=[REDACTED]'


def test_retention_failure_keeps_registered_native_descriptor_open(tmp_path):
    result=run_child('''
import faulthandler,sys
from pathlib import Path
from lockframe import logging as logs
folder=Path(sys.argv[1]);logs.configure_logging(folder)
original=Path.glob
def denied(self,pattern):raise PermissionError('retention scan denied')
Path.glob=denied
try:logs.enable_native_trace()
finally:Path.glob=original
assert logs._native_stream is not None and not logs._native_stream.closed
faulthandler.dump_traceback(file=logs._native_stream)
logs._native_stream.flush()
''',tmp_path)
    assert result.returncode==0,result.stderr
    assert 'Current thread' in next(tmp_path.glob('native-crash-*.log')).read_text(encoding='utf-8')
