"""Curve-only templates: never overwrite audio, flow or tracking settings."""
from copy import deepcopy
from pathlib import Path
import json,math,uuid
from .workspace_data import data_root,atomic_json
from .time_remap_model import normalize_config


def curve_part(config):
    value=normalize_config({'version':2,**(config or {})})
    return {key:deepcopy(value[key]) for key in ('version','mode','basic','points')}


def apply_curve(current,curve):
    result=normalize_config(current or None);result.update(curve_part(curve))
    return normalize_config(result)


def builtins():
    result=[]
    for key,name,speeds,bounds in (
        ('ease','柔降入镜',[6.,2.2,1.2,.8,.5],[.12,.33,.65]),
        ('peak','中心突进',[1.,1.4,7.,1.4,1.],[.38,.5,.62]),
        ('swoop','两端疾行',[5.,1.6,.6,1.6,5.],[.15,.5,.85]),
        ('even','匀速基准',[1.]*5,[.25,.5,.75])):
        config=normalize_config();config['basic'].update(speeds=speeds,boundaries=bounds)
        result.append(dict(id=key,name=name,curve=curve_part(config),builtin=True))
    points=[]
    for i in range(17):
        u=i/16;speed=1+5*math.sin(4*math.pi*u)**2
        points.append(dict(position=u/3.5,source=u-5/(56*math.pi)*math.sin(8*math.pi*u),
            interpolation='bezier',incoming_interpolation='bezier',incoming_speed=speed,outgoing_speed=speed,
            left_influence=1/3,right_influence=1/3))
    result.insert(3,dict(id='pulse',name='四段脉冲',curve=curve_part(dict(version=2,mode='advanced',points=points)),builtin=True))
    return result


class PresetStore:
    def __init__(self,path=None):self.path=Path(path) if path else data_root()/'curve-presets.json'

    def read(self):
        if not self.path.exists():return []
        if self.path.stat().st_size>1024**2:raise ValueError('自定义曲线预设文件过大，未读取')
        raw=json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(raw,dict) or raw.get('format')!='tomato-curve-presets' or raw.get('version')!=1 or not isinstance(raw.get('items'),list) or len(raw['items'])>64:
            raise ValueError('自定义预设格式不支持，原文件已保留')
        result=[]
        for item in raw['items']:
            identifier=item['id'];name=item['name']
            if not isinstance(identifier,str) or len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier) or not isinstance(name,str) or not 1<=len(name)<=32:
                raise ValueError('自定义预设内容损坏，原文件已保留')
            result.append(dict(id=identifier,name=name,curve=curve_part(item['curve']),builtin=False))
        return result

    def save(self,name,config):
        name=str(name).strip();items=self.read()
        if not name or len(name)>32:raise ValueError('预设名称需要1–32个字符')
        if len(items)>=64:raise ValueError('最多保存64个自定义曲线，请先删除不用的预设')
        if any(v['name']==name for v in items):raise ValueError('名称已存在，请使用不同名称')
        item=dict(id=uuid.uuid4().hex,name=name,curve=curve_part(config));items.append(item)
        if len(json.dumps(items,ensure_ascii=False).encode())>1000*1024:raise ValueError('自定义预设超过1 MiB预算，原预设已保留')
        atomic_json(self.path,dict(format='tomato-curve-presets',version=1,items=items));return item['id']

    def remove(self,identifier):
        items=self.read();kept=[v for v in items if v['id']!=identifier]
        if len(kept)!=len(items):atomic_json(self.path,dict(format='tomato-curve-presets',version=1,items=kept))
