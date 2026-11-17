"""Exercise import completion, partial cancellation and UI responsiveness in EXE."""
from pathlib import Path
import json
import threading
import time
import traceback

from . import __version__


def check_imports(folder):
    from PySide6.QtCore import QSettings, QTimer
    from PySide6.QtWidgets import QApplication
    from . import app as ui, media, sequence
    from .thumbnail_check import make_sources
    from .v072_check import wait_job
    from .ad_workspace import _filter_clips

    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    paths=make_sources(folder/'sources')
    broken=folder/'损坏素材.mp4';broken.write_bytes(b'not a video')
    missing=folder/'缺失素材.mp4'
    application=QApplication.instance() or QApplication([])
    settings_factory=ui.QSettings
    ui.QSettings=lambda *a:QSettings(str(folder/'prefs.ini'),QSettings.Format.IniFormat)
    try:w=ui.MainWindow()
    finally:ui.QSettings=settings_factory
    w.resize(1160,760);w.show();application.processEvents()
    ticks=[];timer=QTimer(w);timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(time.monotonic()));timer.start()
    original=sequence.inspect_video;events=[];phase=['mixed']
    entered=threading.Event()
    def importer(path,cancel,progress=None):
        if phase[0]=='cancel' and path==paths[1]:
            entered.set()
            if progress:progress(.25)
            if not cancel.wait(5):raise TimeoutError('cancel test did not request cancellation')
            media.check_cancel(cancel)
        # A deterministic slow clip proves GUI timers run while import works.
        if progress:progress(.01)
        if cancel.wait(.12):media.check_cancel(cancel)
        return original(path,cancel,progress)
    sequence.inspect_video=importer
    try:
        w._import_paths([paths[0],str(broken),paths[1],str(missing)])
        w.worker.progress.connect(lambda value,text:events.append((value,text)))
        wait_job(w,30)
        assert not w.busy and w.worker is None and len(w.clips)==2 and w.info is not None
        assert w._clip().info.path==str(Path(paths[0]).resolve())
        assert w.clips[0].info.proxy and not w.clips[1].info.proxy
        assert '失败 2 条' in w.status_label.text(),w.status_label.text()
        assert len(ticks)>=3 and any(0<value<.25 for value,_ in events)
        # Existing list filters must not hide the newly imported, untracked clip.
        _filter_clips(w,'done');w._import_paths([paths[2]]);wait_job(w,20)
        assert w._ad_filter_mode=='all' and len(w.clips)==3
        assert all(not w.clip_list.item(i).isHidden() for i in range(w.clip_list.count()))
        w._import_paths(paths)
        assert not w.busy and '已经在序列' in w.status_label.text()
        # Partial cancellation returns the completed first item, never the active or later item.
        phase[0]='cancel';cancelled=threading.Event();partial=[]
        def collect(cancel,progress):
            return sequence.import_many(paths,cancel,progress,metadata_only=True)
        def done(result):partial.extend(result)
        w._job(collect,(),done,'验收导入取消',keep_partial=True)
        def cancel_when_entered():
            if entered.is_set():w._cancel();cancelled.set();poll.stop()
        poll=QTimer(w);poll.setInterval(10);poll.timeout.connect(cancel_when_entered);poll.start()
        wait_job(w,20);poll.stop()
        assert cancelled.is_set() and len(partial)==1 and partial[0]['path']==paths[0]
        assert w._job_outcome=='cancelled' and w.worker is None
        # A cache hit must finish and emit progress rather than remain at 100% without a result.
        values=[];info=media.import_video(paths[0],threading.Event(),values.append)
        stamp=Path(info.proxy).stat().st_mtime_ns
        again=media.import_video(paths[0],threading.Event(),values.append)
        assert Path(again.proxy).stat().st_mtime_ns==stamp and values[-1]==1.
        sequence.inspect_video=original
        # The next clip prepares only when opened, without transcoding the third.
        w._activate_clip(w.clips[1].id);wait_job(w,20)
        assert w.clips[1].info.proxy and not w.clips[2].info.proxy
        w._activate_clip(w.clips[0].id)
        # Cancel an actual long-running child during the probe communication loop.
        popen=media.subprocess.Popen;children=[];tool=media.find_tool('ffmpeg')
        def slow_probe(command,**kwargs):
            child=popen([tool,'-v','error','-nostdin','-re','-f','lavfi','-i',
                         'color=black:s=64x64:r=1','-t','30','-f','null','-'],**kwargs)
            children.append(child);return child
        media.subprocess.Popen=slow_probe
        cancel=threading.Event();alarm=threading.Timer(.15,cancel.set);alarm.start();started=time.monotonic()
        try:
            try:media.probe(paths[0],cancel)
            except media.Cancelled:pass
            else:raise AssertionError('slow probe was not cancelled')
        finally:media.subprocess.Popen=popen;alarm.join()
        elapsed=time.monotonic()-started
        assert elapsed<2. and children and children[0].poll() is not None
        w.grab().save(str(folder/'导入完成_素材列表.png'))
        grid_report=check_grid_view(w,folder)
        context_report=check_clip_context(w,folder)
        return dict(mixed_valid=2,failed=2,heartbeat_ticks=len(ticks),
                    inside_clip_progress=True,completion_activated=True,filter_reset=True,
                    duplicate_feedback=True,cancelled_preserves_completed=True,
                    cache_hit_completed=True,lazy_proxy=True,slow_probe_cancelled=True,probe_cancel_seconds=elapsed,
                    clip_context=context_report,clip_grid=grid_report,
                    note='Generated local clips, corruption/missing input and controlled slow IO; not all Windows shell handlers.')
    finally:
        sequence.inspect_video=original
        if w.worker is not None:
            w.worker.cancel();w.worker.wait(10000);application.processEvents()
        timer.stop();w._confirm_discard=lambda:True;w.close();media.configure_cache_preferences(None)


def check_grid_view(window, folder):
    from PySide6.QtWidgets import QApplication, QListWidget
    from PySide6.QtCore import Qt
    from .thumbnail_check import spin
    from .sequence import Clip
    from .models import file_fingerprint
    from dataclasses import replace
    original=list(window.clips);selected=window._selected_clip_ids()
    identity=window.active_clip_id;track,plan=window.track,window.plan
    # Extra cards use copies of generated local colour clips solely for layout QA.
    generated=folder/'grid-fixtures';generated.mkdir(exist_ok=True)
    extras=[]
    for n in range(12):
        base=original[n%len(original)];path=generated/f'网格测试_{n+1:02d}.avi'
        path.write_bytes(Path(base.info.path).read_bytes())
        extras.append(Clip(replace(base.info,path=str(path),fingerprint=file_fingerprint(path))))
    window.clips=original+extras;window._refresh_sequence(selected)
    window.clip_view_actions['grid'].trigger();QApplication.processEvents()
    assert window.clip_list.viewMode()==QListWidget.ViewMode.IconMode
    spin(lambda:all((window._thumbnail_items[c.id].data(Qt.ItemDataRole.UserRole+1) or {}).get('thumbnail_state')=='ready' for c in window.clips))
    window.grab().save(str(folder/'素材网格_默认宽度.png'))
    window.resize(1900,1000);window.workspace_splitter.setSizes([720,870,290]);QApplication.processEvents()
    window.grab().save(str(folder/'素材网格_展开宽度.png'))
    assert window.active_clip_id==identity and window.track is track and window.plan is plan and window.worker is None
    assert window._selected_clip_ids()==selected and window.preferences.value('assets/view')=='grid'
    window.clip_view_actions['list'].trigger();QApplication.processEvents()
    assert window.clip_list.viewMode()==QListWidget.ViewMode.ListMode
    window.clips=original;window._refresh_sequence(selected)
    window.resize(1160,760);window.workspace_splitter.setSizes([275,575,280]);QApplication.processEvents()
    window.grab().save(str(folder/'素材列表_原有排列.png'))
    return dict(list_preserved=True,grid_available=True,selection_preserved=True,track_plan_unchanged=True,
                preference_saved=True,context_compatible=True,layout_fixture_cards=15,
                fixture='Copied generated local colour clips for layout only; not camera tracking evidence')


def check_clip_context(window, folder):
    """Real viewport context events and menu actions, including current/last removal."""
    import hashlib
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication, QMenu
    paths={c.info.path:hashlib.sha256(Path(c.info.path).read_bytes()).hexdigest() for c in window.clips}
    ids=[c.id for c in window.clips]
    assert len(ids)==3
    def context(row, execute=True, screenshot=False):
        row_item=window.clip_list.item(row)
        point=window.clip_list.visualItemRect(row_item).center()
        observed={}
        def choose():
            menu=QApplication.activePopupWidget()
            try:
                assert isinstance(menu,QMenu) and menu.objectName()=='clipContextMenu'
                action=menu.actions()[0]
                observed.update(enabled=action.isEnabled(),selected=window._selected_clip_ids())
                if screenshot:
                    window.grab().save(str(folder/'右键删除_素材列表.png'))
                    menu.grab().save(str(folder/'右键删除_菜单.png'))
                if execute and action.isEnabled():action.trigger()
            except Exception as error:observed['error']=repr(error)
            finally:
                if isinstance(menu,QMenu):menu.close()
        QTimer.singleShot(30,choose)
        event=QContextMenuEvent(QContextMenuEvent.Reason.Mouse,point,window.clip_list.viewport().mapToGlobal(point))
        QApplication.sendEvent(window.clip_list.viewport(),event)
        assert 'error' not in observed and observed,observed
        return observed
    window.clip_list.selectAll();observed=context(2,screenshot=True)
    assert observed['selected']=={ids[2]} and [c.id for c in window.clips]==ids[:2]
    assert window.active_clip_id==ids[0]
    context(1,False);assert len(window.clips)==2
    window.busy=True
    try:assert not context(0)['enabled'] and len(window.clips)==2
    finally:window.busy=False
    context(0);assert window.active_clip_id==ids[1] and len(window.clips)==1
    context(0)
    assert not window.clips and window.clip_list.count()==0 and window.info is None and window.reader is None
    assert not window.export_button.isEnabled()
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha for path,sha in paths.items())
    return dict(single_target=True,retained_other_clips=True,cancel_keeps_clip=True,busy_disabled=True,
                current_switches=True,last_clears_preview=True,original_files_unchanged=True)


def run_import_check(directory):
    folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
    try:report=dict(ok=True,version=__version__,imports=check_imports(folder));code=0
    except Exception:report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return code
