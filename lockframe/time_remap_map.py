"""Bounded output-step -> relative source-frame maps shared by preview/export."""
from functools import lru_cache
import json
import math
import numpy as np
from .time_remap_model import normalize_config,active_points,normalize_keyframes
from .time_remap_math import evaluate_sources

MAX_OUTPUT_FRAMES = 2_000_000


def _count(value):
    if type(value) is not int or not 1<=value<=MAX_OUTPUT_FRAMES:
        raise ValueError(f'源片段帧数应在1–{MAX_OUTPUT_FRAMES}之间')
    return value


def output_frame_count(source_frame_count,points):
    _count(source_frame_count)
    points=normalize_keyframes(points)
    scale=max(.01,min(100.,points[-1]['position']))
    result=max(1,math.floor(source_frame_count*scale+.5))
    if result>MAX_OUTPUT_FRAMES:
        raise ValueError(f'变速输出超过{MAX_OUTPUT_FRAMES}帧，请缩短片段或输出时长')
    return result


def build_frame_map(source_frame_count,points):
    """MS normalized endpoint convention, with safe product allocation limits."""
    _count(source_frame_count)
    points=normalize_keyframes(points)
    if source_frame_count==1:
        return np.zeros(1,dtype=np.float64)
    count=output_frame_count(source_frame_count,points)
    scale=max(.01,min(100.,points[-1]['position']))
    positions=np.arange(count,dtype=np.float64)/max(1,count-1)*scale
    return evaluate_sources(points,positions)*(source_frame_count-1)


def nearest_source_indices(frame_map):
    values=np.asarray(frame_map,dtype=np.float64)
    if values.ndim!=1 or not values.size or values.size>MAX_OUTPUT_FRAMES or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('frame-map需要有限非负的一维源帧位置')
    if np.any(values>MAX_OUTPUT_FRAMES-1):
        raise ValueError('frame-map源帧位置超出允许范围')
    return np.floor(values+.5).astype(np.int64)


def _identity(points):
    if abs(points[0]['position'])>1e-12 or abs(points[0]['source'])>1e-12 or abs(points[-1]['position']-1.)>1e-12:
        return False
    for index,point in enumerate(points):
        if abs(point['position']-point['source'])>1e-12:
            return False
        for name in ('incoming_speed','outgoing_speed'):
            if point[name] is not None and abs(point[name]-1.)>1e-12:
                return False
        for name in ('left_speed_tangent','right_speed_tangent'):
            if point[name] is not None and abs(point[name])>1e-12:
                return False
        for side,next_index,sign in (('left',index-1,-1),('right',index+1,1)):
            value=point[f'{side}_handle_source']
            if value is None or not 0<=next_index<len(points):
                continue
            expected=point['source']+sign*abs(points[next_index]['position']-point['position'])*point[f'{side}_influence']
            if abs(value-expected)>1e-12:
                return False
    return True


@lru_cache(maxsize=4)
def _cached_map(count,enabled,points_json):
    points=json.loads(points_json)
    if not enabled or _identity(points):
        result=np.arange(count,dtype=np.float64)
    elif count==1:
        # The extracted compatibility helper above returns one lookup value
        # for a singleton. Product output still needs the requested hold time.
        result=np.zeros(output_frame_count(count,points),dtype=np.float64)
    else:
        result=build_frame_map(count,points)
    result.flags.writeable=False
    return result


def output_frame_map(config,source_frame_count):
    """Cached read-only frame map. Audio, flow and composition don't invalidate it.

    Straight identity is deliberately exact so enabling flow at original speed
    never invents interpolation work solely from 24-step inverse roundoff.
    """
    _count(source_frame_count)
    config=normalize_config(config)
    points=active_points(config)
    key=json.dumps(points,sort_keys=True,separators=(',',':'),allow_nan=False)
    return _cached_map(source_frame_count,config['enabled'],key)
