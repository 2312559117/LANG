"""Atomic, bounded local workspace generations; never overwrite user projects."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib,json,re,shutil,time,uuid,math
from .media import check_cancel
from .sequence import save_sequence,read_sequence
from .workspace_data import atomic_json,data_root

MAX_MEMORY=64*1024**2
MAX_DISK=1024**3
GENERATION=re.compile(r'[0-9a-f]{32}\.locksequence')


def _entry(value):
    if not isinstance(value,dict):return None
    stamp=value.get('time');names=value.get('names');count=value.get('count')
    if type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<=stamp<=32503680000:return None
    if type(count) is not int or not 1<=count<=500 or not isinstance(value.get('view'),dict):return None
    if not isinstance(names,list) or len(names)>3 or not all(isinstance(name,str) and len(name)<=512 for name in names):return None
    if not isinstance(value.get('file'),str) or not GENERATION.fullmatch(value['file']):return None
    if any(not isinstance(value.get(key),str) or not re.fullmatch('[0-9a-f]{64}',value[key]) for key in ('digest','sha256')):return None
    return value


def freeze_clips(clips):
    total=0
    for clip in clips:
        if clip.track is not None:
            track=clip.track
            total+=sum(value.nbytes for value in (track.positions,track.boxes,track.confidence,track.homographies,track.confidence_grids) if value is not None)
            total+=track.count*16
    if total>MAX_MEMORY:raise ValueError('自动保存快照超过64 MiB预算，请分开保存素材序列；当前工程未修改')
    memo={}
    return [replace(clip,info=replace(clip.info),settings=deepcopy(clip.settings),anchors=dict(clip.anchors),
        track=deepcopy(clip.track,memo),precision_info=None,retrack_previous=None,tracking_resume=None,
        timebar_state=deepcopy(clip.timebar_state),tracking_sections=deepcopy(clip.tracking_sections),
        review_marks=dict(clip.review_marks)) for clip in clips]


def _manifest(folder):
    path=Path(folder)/'manifest.json'
    if not path.exists():return None
    if path.stat().st_size>1024**2:raise ValueError('自动保存索引超过读取预算')
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict) or value.get('format')!='tomato-workspace-recovery' or value.get('version')!=1:raise ValueError('自动保存索引版本不支持')
    latest,previous=_entry(value.get('latest')),_entry(value.get('previous'))
    if latest is None:
        latest,previous=previous,None
    if latest is None:raise ValueError('自动保存索引没有有效记录，原文件已保留')
    value=dict(value,latest=latest,previous=previous)
    for key in ('clean','acknowledged'):value[key]=value.get(key) is True
    return value


def _generation(folder,name):
    if not isinstance(name,str) or not GENERATION.fullmatch(name):raise ValueError('自动保存文件名无效')
    path=(Path(folder)/name).resolve()
    if path.parent!=Path(folder).resolve():raise ValueError('自动保存文件不在所属目录中')
    return path


def write_workspace(clips,view,folder,cancel,progress=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);check_cancel(cancel)
    if len(json.dumps(view,ensure_ascii=False).encode())>480*1024:raise ValueError('工作区视图和草稿过大，未覆盖上次自动保存')
    previous=_manifest(folder)
    prior_entry=None
    if previous:
        for entry in (previous['latest'],previous['previous']):
            if entry is None:continue
            try:
                path=_generation(folder,entry['file'])
                if path.stat().st_size<=100*1024**2 and hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']:
                    prior_entry=entry;break
            except (OSError,ValueError):continue
    root=folder.parent
    used=sum(path.stat().st_size for path in root.glob('*/*.locksequence') if path.is_file())
    if used+100*1024**2>MAX_DISK:raise OSError('自动保存空间接近1 GiB上限，请在恢复管理中清理不需要的旧记录')
    if shutil.disk_usage(folder).free<256*1024**2:raise OSError('磁盘空间不足256 MiB，已保留上次自动保存')
    generation=folder/(uuid.uuid4().hex+'.locksequence');committed=False
    try:
        save_sequence(generation,clips);check_cancel(cancel)
        digest=hashlib.sha256(generation.read_bytes()+json.dumps(view,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        if previous and prior_entry is previous['latest'] and prior_entry['digest']==digest:
            return dict(saved=False,time=previous['latest']['time'],folder=str(folder))
        stamp=time.time();entry=dict(file=generation.name,digest=digest,sha256=hashlib.sha256(generation.read_bytes()).hexdigest(),
            view=view,time=stamp,count=len(clips),names=[Path(clip.info.path).name for clip in clips[:3]])
        manifest=dict(format='tomato-workspace-recovery',version=1,clean=False,acknowledged=False,
            latest=entry,previous=prior_entry)
        check_cancel(cancel);atomic_json(folder/'manifest.json',manifest);committed=True
        keep={entry['file'],(manifest['previous'] or {}).get('file')}
        for path in folder.glob('*.locksequence'):
            if path.name not in keep and GENERATION.fullmatch(path.name):
                try:_generation(folder,path.name).unlink(missing_ok=True)
                except OSError as error:
                    from .logging import get_logger
                    get_logger('autosave').warning('新快照已提交，旧代暂未清理 %s',error)
        return dict(saved=True,time=stamp,folder=str(folder),file=str(generation))
    finally:
        if not committed:generation.unlink(missing_ok=True)


def recover_workspace(folder):
    manifest=_manifest(folder)
    if manifest is None:raise ValueError('没有找到自动保存索引')
    errors=[]
    for entry in (manifest.get('latest'),manifest.get('previous')):
        if not isinstance(entry,dict):continue
        try:
            path=_generation(folder,entry['file'])
            if path.stat().st_size>100*1024**2:raise ValueError('自动保存文件过大')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry.get('sha256'):raise ValueError('自动保存校验失败')
            return read_sequence(path),entry.get('view',{}),entry
        except Exception as error:errors.append(str(error))
    raise ValueError('自动保存及备用代均无法读取：'+'；'.join(errors))


def list_workspaces(root=None):
    root=Path(root) if root else data_root()/'workspace-recovery';result=[]
    if root.is_dir():
        for folder in root.iterdir():
            if not folder.is_dir() or not re.fullmatch('[0-9a-f]{32}',folder.name):continue
            try:
                manifest=_manifest(folder)
                if manifest and isinstance(manifest.get('latest'),dict):result.append(dict(folder=str(folder),**manifest))
            except (OSError,ValueError,TypeError):continue
    return sorted(result,key=lambda item:item['latest'].get('time',0),reverse=True)


def mark_workspace(folder,**fields):
    manifest=_manifest(folder)
    if manifest is not None:
        manifest.update({k:bool(v) for k,v in fields.items() if k in ('clean','acknowledged')})
        atomic_json(Path(folder)/'manifest.json',manifest)


def remove_workspace(root,folder,current=None):
    root=Path(root).resolve();folder=Path(folder).resolve()
    if folder.parent!=root or not re.fullmatch('[0-9a-f]{32}',folder.name) or current and folder==Path(current).resolve():
        raise ValueError('不能清理当前会话或所属目录外的恢复记录')
    # Remove only known data files; never traverse an imported path or link.
    for path in folder.iterdir():
        if path.is_symlink() or path.is_dir():raise ValueError('恢复目录含未知条目，未清理')
        if path.name!='manifest.json' and not GENERATION.fullmatch(path.name):raise ValueError('恢复目录含未知文件，未清理')
    for path in folder.iterdir():path.unlink()
    folder.rmdir()


def restore_saved_workspace(folder,cancel,progress=None):
    from .sequence import restore_sequence
    check_cancel(cancel);clips,view,entry=recover_workspace(folder)
    restored=restore_sequence(clips,cancel,progress);check_cancel(cancel)
    return dict(clips=restored,view=view,entry=entry,folder=str(folder))
