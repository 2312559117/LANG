"""Local file logging so a user can share exactly what the tracker saw.

Common credential patterns are filtered. Logs contain task diagnostics and
file names, not media content or a dump of user settings/environment.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path


_configured = False
_folder: Path | None = None
SESSION_ID = uuid.uuid4().hex[:12]
SESSION_STARTED = datetime.now(timezone.utc).isoformat()
_native_stream = None


def redact(text):
    """Filter common credential forms, including legacy log lines, without reading env secrets."""
    text = re.sub(r"(?i)\bsk-[a-z0-9_-]{8,}", "[REDACTED]", str(text))
    text = re.sub(r"(?i)\bBearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    return re.sub(r"(?i)(\b(?:api[_-]?key|access[_-]?token|password|secret)[\"']?\s*[:=]\s*[\"']?)[^\s,;\"']+",
                  lambda match:match.group(1)+'[REDACTED]', text)


class LocalFormatter(logging.Formatter):
    def format(self, record):
        record.session = SESSION_ID
        return redact(super().format(record))



def log_dir() -> Path:
    """Directory that holds the log files (overridable for tests)."""
    override = os.environ.get("LOCKFRAME_LOG_DIR")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA", tempfile.gettempdir())
    return Path(base) / "LockFrameStudio" / "logs"


def configure_logging(directory: Path | str | None = None) -> logging.Logger:
    """Idempotent setup of a rotating file handler under the app data folder."""
    global _configured, _folder
    root = logging.getLogger("lockframe")
    if _configured:
        return root
    _folder = Path(directory) if directory is not None else log_dir()
    try:
        _folder.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            _folder / "tomato-tracker.log", maxBytes=2_000_000, backupCount=4, encoding="utf-8")
    except OSError:
        # Logging must not prevent startup when a selected directory is unavailable.
        _folder = Path(tempfile.gettempdir()) / 'tomato-tracker-logs'
        try:
            _folder.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                _folder / 'tomato-tracker.log', maxBytes=2_000_000, backupCount=4, encoding='utf-8')
        except OSError:
            handler = logging.NullHandler()
    handler.setFormatter(LocalFormatter(
        "%(asctime)s %(levelname)s pid=%(process)d thread=%(threadName)s session=%(session)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured = True
    from . import __version__, __build__
    root.info('日志已启用 version=%s build=%s session=%s frozen=%s', __version__, __build__, SESSION_ID, getattr(sys, 'frozen', False))
    return root


def get_logger(name: str = "app") -> logging.Logger:
    return logging.getLogger("lockframe." + name)


def log_folder() -> Path:
    return _folder or log_dir()


def install_excepthook() -> None:
    """Persist uncaught exceptions to the log while keeping default handling."""
    def handle(exc_type, exc_value, exc_traceback):
        import traceback
        get_logger("crash").error(
            "未捕获异常: %s\n%s",
            exc_value,
            "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.excepthook = handle
    previous = threading.__excepthook__
    def thread_error(args):
        get_logger('thread').error('后台线程未捕获异常 %s', args.thread.name if args.thread else '',
                                   exc_info=(args.exc_type,args.exc_value,args.exc_traceback))
        previous(args)
    threading.excepthook = thread_error
    enable_native_trace()


def enable_native_trace():
    """Keep a small set of native-fault trace destinations; no periodic stack dumping."""
    global _native_stream
    if _native_stream is not None:return
    try:
        import faulthandler
        folder=log_folder();folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'native-crash-{os.getpid()}.log'
        _native_stream=path.open('a',encoding='utf-8')
        faulthandler.enable(file=_native_stream,all_threads=True)
    except (OSError,RuntimeError):
        if _native_stream is not None:_native_stream.close()
        _native_stream=None
        return
    # Retention errors must not close an fd already registered with faulthandler.
    try:
        older=sorted((p for p in folder.glob('native-crash-*.log') if p!=path and re.fullmatch(r'native-crash-\d+\.log',p.name)),
                     key=lambda p:p.stat().st_mtime,reverse=True)
        for old in older[3:]:
            try:old.unlink()
            except OSError:pass
    except OSError:pass


def startup_error(details):
    folder=log_folder();folder.mkdir(parents=True,exist_ok=True)
    target=folder/'startup-error.log'
    target.write_text(redact(details[-256000:]),encoding='utf-8')
    return target
