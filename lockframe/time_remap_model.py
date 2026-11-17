"""Versioned per-clip retime data, independent of tracking state.

The old ``time_remap``/``frame_interpolation`` project fields are deliberately
not consumed here. The new contract is explicit version 2 under ``retime``.
"""
from copy import deepcopy
import math

MAX_KEYFRAMES = 512
POINT_DEFAULTS = dict(position=0., source=0., left_influence=.33,
                      right_influence=.33, left_handle_source=None,
                      right_handle_source=None, interpolation='bezier',
                      incoming_interpolation='bezier', handles_linked=False,
                      left_speed_tangent=None, right_speed_tangent=None,
                      incoming_speed=None, outgoing_speed=None)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{name}必须是有限数值')
    try:
        number=float(value)
    except OverflowError as error:
        raise ValueError(f'{name}必须是有限数值') from error
    if not math.isfinite(number):
        raise ValueError(f'{name}必须是有限数值')
    return number


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def default_points():
    return [dict(POINT_DEFAULTS, position=0., source=0., left_influence=0.),
            dict(POINT_DEFAULTS, position=1., source=1., right_influence=0.)]


def _bezier(start, end):
    return start['interpolation']=='bezier' and end['incoming_interpolation']=='bezier'


def normalize_keyframes(points=None):
    """Normalize valid finite points using the retained managed algorithm.

    Unlike the reference's permissive damaged-data import, the product boundary
    rejects non-finite values and unknown fields rather than silently losing edits.
    """
    if points is None:
        return default_points()
    if not isinstance(points, (list, tuple)) or len(points)>MAX_KEYFRAMES:
        raise ValueError(f'变速曲线最多允许{MAX_KEYFRAMES}个关键帧')
    result=[]
    original_positions=[]
    for raw in points:
        if not isinstance(raw, dict) or set(raw)-set(POINT_DEFAULTS):
            raise ValueError('变速关键帧字段无效')
        if 'position' not in raw or 'source' not in raw:
            raise ValueError('变速关键帧缺少输出位置或源位置')
        point=dict(POINT_DEFAULTS, **raw)
        original_positions.append(_number(point['position'],'position'))
        for name,lo,hi in (('position',0.,100.),('source',0.,1.),
                           ('left_influence',.01,1.),('right_influence',.01,1.)):
            point[name]=_clamp(_number(point[name],name),lo,hi)
        for name in ('left_handle_source','right_handle_source',
                     'left_speed_tangent','right_speed_tangent',
                     'incoming_speed','outgoing_speed'):
            if point[name] is not None:
                lo,hi=(-1.,2.) if 'handle' in name else (-100000.,100000.) if 'tangent' in name else (-100.,100.)
                point[name]=_clamp(_number(point[name],name),lo,hi)
        if type(point['handles_linked']) is not bool:
            raise ValueError('变速手柄关联状态无效')
        if point['interpolation'] not in ('linear','bezier') or point['incoming_interpolation'] not in ('linear','bezier'):
            raise ValueError('未知的变速曲线插值方式')
        result.append(point)
    # The reference orders BEFORE clamping: two out-of-range positions must
    # retain their original order when both become the same endpoint.
    result=[point for _,point in sorted(zip(original_positions,result),key=lambda item:item[0])]
    if not result:
        return default_points()
    if len(result)==1:
        result.append(default_points()[1 if result[0]['position']<=.5 else 0])
        result.sort(key=lambda point:point['position'])
    for index in range(1,len(result)):
        result[index]['position']=max(result[index]['position'],result[index-1]['position']+.0001)
    if result[-1]['position']>100:
        result[-1]['position']=100.
        for index in range(len(result)-2,-1,-1):
            result[index]['position']=max(0.,min(result[index]['position'],result[index+1]['position']-.0001))
    for index,point in enumerate(result):
        before=index>0
        after=index+1<len(result)
        left_curve=before and _bezier(result[index-1],point)
        right_curve=after and _bezier(point,result[index+1])
        left=point['left_influence'] if before else 0.
        right=point['right_influence'] if after else 0.
        y0=result[index-1]['source'] if before else point['source']
        y1=result[index+1]['source'] if after else point['source']
        span0=point['position']-result[index-1]['position'] if before else 1.
        span1=result[index+1]['position']-point['position'] if after else 1.
        left_y=point['left_handle_source']
        right_y=point['right_handle_source']
        left_y=point['source']+(y0-point['source'])*left if left_y is None else left_y
        right_y=point['source']+(y1-point['source'])*right if right_y is None else right_y
        point['left_influence'],point['right_influence']=left,right
        for side,enabled,value,span,influence in (('left',left_curve,left_y,span0,left),
                                                  ('right',right_curve,right_y,span1,right)):
            distance=100.*max(.0001,span)*max(.01,influence)
            point[f'{side}_handle_source']=_clamp(_clamp(value,point['source']-distance,point['source']+distance),-1.,2.) if enabled else None
            if not enabled:
                point[f'{side}_speed_tangent']=None
                point['incoming_speed' if side=='left' else 'outgoing_speed']=None
    return result


def normalize_basic(raw=None):
    raw={} if raw is None else raw
    if not isinstance(raw,dict) or set(raw)-{'speeds','boundaries'}:
        raise ValueError('基础变速配置无效')
    speeds=raw.get('speeds',[1.]*5)
    boundaries=raw.get('boundaries',[.25,.5,.75])
    if not isinstance(speeds,(list,tuple)) or len(speeds)!=5:
        raise ValueError('基础变速需要5个速度锚点')
    if not isinstance(boundaries,(list,tuple)) or len(boundaries)!=3:
        raise ValueError('基础变速需要3个区间边界')
    speeds=[_clamp(_number(value,'基础速度'),.1,10.) for value in speeds]
    positions=[_number(value,'基础边界') for value in boundaries]
    # Reserve .08 for each following section and .04 at the end. The
    # recovered independent upper bounds can otherwise create min>max.
    positions[0]=_clamp(positions[0],.08,.80)
    positions[1]=_clamp(positions[1],positions[0]+.08,.88)
    positions[2]=_clamp(positions[2],positions[1]+.08,.96)
    return dict(speeds=speeds,boundaries=positions)


def default_config():
    return dict(version=2,enabled=False,mode='basic',basic=normalize_basic(),
                points=default_points(),optical_flow=False,flow_backend='auto',
                allow_frame_fallback=False,audio_policy='mute',preview_flow=False)


def normalize_config(raw=None):
    if raw is None:
        return default_config()
    if not isinstance(raw,dict) or set(raw)-set(default_config()):
        raise ValueError('变速配置字段无效')
    config=default_config()
    config.update(deepcopy(raw))
    if 'version' not in raw or type(config['version']) is not int or config['version']!=2:
        raise ValueError('不支持的变速配置版本')
    for name in ('enabled','optical_flow','allow_frame_fallback','preview_flow'):
        if type(config[name]) is not bool:
            raise ValueError(f'变速配置{name}需要布尔值')
    if config['mode'] not in ('basic','advanced'):
        raise ValueError('未知的变速曲线模式')
    if config['flow_backend'] not in ('auto','dis','nvof'):
        raise ValueError('未知的光流后端')
    if config['audio_policy'] not in ('mute','mapped'):
        raise ValueError('未知的变速音频策略')
    config['basic']=normalize_basic(config['basic'])
    config['points']=normalize_keyframes(config['points'])
    return config


def active_points(config):
    config=normalize_config(config)
    if config['mode']=='advanced':
        return config['points']
    from .time_remap_math import basic_time_points
    return normalize_keyframes(basic_time_points(config['basic']))


def make_constant_speed(config=None,speed=1.):
    config=normalize_config(config)
    speed=_number(speed,'播放速度')
    if abs(speed)<.01 or abs(speed)>100:
        raise ValueError('恒定速度绝对值需在0.01–100之间，定格请设置源位置和输出时长')
    start,end=(0.,1.) if speed>0 else (1.,0.)
    config.update(enabled=True,mode='advanced',points=[
        dict(POINT_DEFAULTS,position=0.,source=start,interpolation='linear',incoming_interpolation='linear'),
        dict(POINT_DEFAULTS,position=1./abs(speed),source=end,interpolation='linear',incoming_interpolation='linear')])
    return normalize_config(config)
