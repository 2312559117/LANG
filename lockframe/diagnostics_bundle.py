"""Bounded, local, explicitly exported diagnostics. Never collect media or projects."""
from pathlib import Path
from datetime import datetime, timezone
import json
import logging
import os
import platform
import re
import sys
import threading
import traceback
import uuid
import zipfile

from . import __version__, __build__
from .logging import SESSION_ID, SESSION_STARTED, log_folder, redact, get_logger
from .media import check_cancel


def capture_state(window):
    """Read a small immutable snapshot on the GUI thread, without filesystem IO."""
    info=window.info
    state=dict(busy=window.busy,job_kind=window._job_kind,progress=window._header_value,
               status=window.status_label.text()[:1000],clip_count=len(window.clips),
               selected_count=len(window._selected_clip_ids()),frame=window.frame_index,
               asset_view=getattr(window,'_clip_view_mode','list'))
    if info:
        state['media']=dict(name=Path(info.path).name,width=info.width,height=info.height,
                            fps=info.fps,frames=info.frames,duration=info.duration,
                            audio=info.has_audio,start=window.start_spin.value()-1,end=window.end_spin.value()-1)
        settings=window.settings()
        state['processing']={name:getattr(settings,name) for name in
            ('tracking_mode','target_shape','acceleration','memory_percent','compute_threads','aspect','resolution')}
    protected=[p for c in window.clips for p in (c.info.path,c.info.proxy,c.project_path) if p]
    protected.extend(p for p in (window.project_path,window.sequence_path,window.last_export) if p)
    queue=getattr(window,'export_queue',None)
    if queue:protected.extend(t.destination for t in queue.tasks)
    # The paths used for protection are never included in the exported state.
    return state,tuple(protected)


def runtime_report():
    import cv2
    import numpy as np
    import PySide6
    from .performance import memory_status
    return dict(schema=1,version=__version__,build=__build__,session=SESSION_ID,session_started_utc=SESSION_STARTED,
                exported_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
                frozen=bool(getattr(sys,'frozen',False)),system=platform.system(),release=platform.release(),
                architecture=platform.machine(),python=platform.python_version(),qt=PySide6.__version__,
                opencv=cv2.__version__,numpy=np.__version__,cpu_count=os.cpu_count(),memory=memory_status(),
                collection='Local logs and basic state only; no video, image, project, model, cache or environment dump')


def _clean(value):
    if isinstance(value,str):return redact(value)
    if isinstance(value,dict):return {key:_clean(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(item) for item in value]
    return value


def _log_entries(folder):
    names=['tomato-tracker.log',*[f'tomato-tracker.log.{i}' for i in range(1,5)],'startup-error.log']
    files=[folder/n for n in names]
    try:
        native=sorted((p for p in folder.glob('native-crash-*.log')
                       if re.fullmatch(r'native-crash-\d+\.log',p.name)),
                      key=lambda p:p.stat().st_mtime,reverse=True)[:4]
        files.extend(native)
    except OSError:pass
    return files


def export_diagnostics(destination, state, cancel, progress=None, *, protected_paths=(), logs_directory=None):
    target=Path(destination).resolve()
    if target.suffix.lower()!='.zip':raise ValueError('诊断包应保存为 ZIP 文件')
    protected={Path(p).resolve() for p in protected_paths}
    if target in protected:raise ValueError('诊断包不能覆盖原素材、工程或成品')
    check_cancel(cancel)
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(f'.{target.stem}.{uuid.uuid4().hex}.partial.zip')
    folder=Path(logs_directory) if logs_directory is not None else log_folder()
    omitted=[];included=[]
    try:
        for handler in logging.getLogger('lockframe').handlers:
            try:handler.flush()
            except OSError:pass
        report=runtime_report();report['state']=state
        frames=sys._current_frames()
        stacks='\n\n'.join(f'Thread {thread.name}\n'+''.join(traceback.format_stack(frames[thread.ident],limit=35))
                           for thread in threading.enumerate()[:12] if thread.ident in frames)
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as output:
            files=_log_entries(folder)
            for index,path in enumerate(files):
                check_cancel(cancel)
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_size==0:continue
                    limit=2_000_000 if path.name.startswith('tomato-tracker') else 256_000
                    with path.open('rb') as stream:
                        stream.seek(max(0,path.stat().st_size-limit))
                        content=stream.read(limit).decode('utf-8','replace')
                    check_cancel(cancel)
                    output.writestr('logs/'+path.name,redact(content))
                    included.append(path.name)
                except OSError as error:omitted.append(dict(file=path.name,error=type(error).__name__))
                if progress:progress((index+1)/(len(files)+2),'正在打包本地日志')
            report.update(included_logs=included,unavailable_logs=omitted)
            output.writestr('diagnostics.json',json.dumps(_clean(report),ensure_ascii=False,indent=2))
            output.writestr('thread-stacks.txt',redact(stacks))
            output.writestr('说明.txt','番茄跟踪诊断包\n请在出现问题后导出并把本 ZIP 发给开发者。\n包含版本、基础环境、任务状态和最近日志；不包含原视频或工程。\n')
        check_cancel(cancel)
        base=target;counter=1
        while True:
            check_cancel(cancel)
            if target in protected:
                counter+=1;target=base.with_name(f'{base.stem}_{counter}{base.suffix}');continue
            try:
                if os.name=='nt':os.rename(temporary,target)
                else:os.link(temporary,target)
                break
            except FileExistsError:
                counter+=1;target=base.with_name(f'{base.stem}_{counter}{base.suffix}')
        get_logger('diagnostics').info('诊断包已导出 %s 日志数=%d',target.name,len(included))
        if progress:progress(1.,'诊断包已保存')
        return str(target)
    finally:temporary.unlink(missing_ok=True)
