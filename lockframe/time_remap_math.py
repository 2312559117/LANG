"""Independent time-remap math matched to retained managed-source evidence.

All coordinates are normalized to the selected source range: x is output time
in units of source duration, y is source progress. No tracking data is changed.
"""
import math
import numpy as np
from .time_remap_model import normalize_keyframes,normalize_basic,POINT_DEFAULTS,_number


def cubic(a,b,c,d,t):
    one=1.-t
    return one*one*one*a+3.*one*one*t*b+3.*one*t*t*c+t*t*t*d


def cubic_derivative(a,b,c,d,t):
    one=1.-t
    return 3.*one*one*(b-a)+6.*one*t*(c-b)+3.*t*t*(d-c)


def cubic_second_derivative(a,b,c,d,t):
    return 6.*(1.-t)*(c-2.*b+a)+6.*t*(d-2.*c+b)


def basic_speed(basic,position):
    basic=normalize_basic(basic)
    positions=[0.,*basic['boundaries'],1.]
    speeds=basic['speeds']
    position=_number(position,'源位置')
    position=max(0.,min(1.,position))
    index=1
    while index<len(positions)-1 and position>positions[index]:
        index+=1
    previous=max(0,index-1)
    span=max(.0001,positions[index]-positions[previous])
    t=max(0.,min(1.,(position-positions[previous])/span))
    p0,p1,p2,p3=(speeds[max(0,previous-1)],speeds[previous],speeds[index],speeds[min(len(speeds)-1,index+1)])
    t2=t*t;t3=t2*t
    value=.5*(2.*p1+(-p0+p2)*t+(2.*p0-5.*p1+4.*p2-p3)*t2+(-p0+3.*p1-3.*p2+p3)*t3)
    return max(.1,min(10.,value))


def basic_time_points(basic):
    """Product mapping: keep the integrated physical output duration."""
    basic=normalize_basic(basic)
    points=[dict(POINT_DEFAULTS,position=0.,source=0.)]
    accumulated=0.
    for index in range(1,9):
        left,right=(index-1)/8.,index/8.
        previous=basic_speed(basic,left)
        for sample in range(1,25):
            current=basic_speed(basic,left+(right-left)*sample/24.)
            accumulated+=(right-left)/24./max(.1,(previous+current)*.5)
            previous=current
        points.append(dict(POINT_DEFAULTS,position=accumulated,source=right))
    return points


def convert_basic_to_advanced(config):
    """Retained MS legacy-conversion normalization: nine points ending at x=1.

    Use ``active_points(config)`` when switching the product editor to advanced
    mode so its actual integrated duration is not silently discarded.
    """
    basic=config.get('basic',config)
    points=basic_time_points(basic)
    total=points[-1]['position']
    for point in points:
        point['position']/=total
    return points


def _bezier(start,end):
    return start['interpolation']=='bezier' and end['incoming_interpolation']=='bezier'


def _controls(start,end):
    span=max(.0001,end['position']-start['position'])
    outgoing=max(.01,min(1.,start['right_influence']))
    incoming=max(.01,min(1.,end['left_influence']))
    x1=start['position']+span*outgoing
    x2=end['position']-span*incoming
    dy=end['source']-start['source']
    y1=start['right_handle_source']
    y2=end['left_handle_source']
    y1=start['source']+dy*outgoing if y1 is None else y1
    y2=end['source']-dy*incoming if y2 is None else y2
    return x1,x2,y1,y2


def _acceleration(start,end,at_end):
    x1,x2,y1,y2=_controls(start,end)
    t=1. if at_end else 0.
    dx=cubic_derivative(start['position'],x1,x2,end['position'],t)
    if abs(dx)<=1e-7:
        return 0.
    dy=cubic_derivative(start['source'],y1,y2,end['source'],t)
    ddx=cubic_second_derivative(start['position'],x1,x2,end['position'],t)
    ddy=cubic_second_derivative(start['source'],y1,y2,end['source'],t)
    acceleration=(ddy*dx-dy*ddx)/(dx*dx*dx)
    return max(-100000.,min(100000.,acceleration)) if math.isfinite(acceleration) else 0.


def _quintic_coefficients(start,end,controls):
    x1,x2,y1,y2=controls
    dx0=cubic_derivative(start['position'],x1,x2,end['position'],0.)
    dx1=cubic_derivative(start['position'],x1,x2,end['position'],1.)
    ddx0=cubic_second_derivative(start['position'],x1,x2,end['position'],0.)
    ddx1=cubic_second_derivative(start['position'],x1,x2,end['position'],1.)
    v0=start['outgoing_speed']
    v1=end['incoming_speed']
    v0=(cubic_derivative(start['source'],y1,y2,end['source'],0.)/dx0 if abs(dx0)>1e-7 else 0.) if v0 is None else v0
    v1=(cubic_derivative(start['source'],y1,y2,end['source'],1.)/dx1 if abs(dx1)>1e-7 else 0.) if v1 is None else v1
    a0=start['right_speed_tangent'];a1=end['left_speed_tangent']
    a0=_acceleration(start,end,False) if a0 is None else a0
    a1=_acceleration(start,end,True) if a1 is None else a1
    d0=v0*dx0;d1=v1*dx1
    dd0=a0*dx0*dx0+v0*ddx0;dd1=a1*dx1*dx1+v1*ddx1
    c0=start['source'];c1=d0;c2=dd0/2.
    delta=end['source']-c0-c1-c2
    first=d1-c1-2.*c2;second=dd1-2.*c2
    return c0,c1,c2,10.*delta-4.*first+second/2.,-15.*delta+7.*first-second,6.*delta-3.*first+second/2.


def _has_quintic(start,end):
    return any(value is not None for value in (start['right_speed_tangent'],end['left_speed_tangent'],start['outgoing_speed'],end['incoming_speed']))


def _source_segment(start,end,t,controls):
    if not _has_quintic(start,end):
        return cubic(start['source'],controls[2],controls[3],end['source'],t)
    c0,c1,c2,c3,c4,c5=_quintic_coefficients(start,end,controls)
    return ((((c5*t+c4)*t+c3)*t+c2)*t+c1)*t+c0


def _source_derivative(start,end,t,controls):
    if not _has_quintic(start,end):
        return cubic_derivative(start['source'],controls[2],controls[3],end['source'],t)
    _,c1,c2,c3,c4,c5=_quintic_coefficients(start,end,controls)
    return (((5.*c5*t+4.*c4)*t+3.*c3)*t+2.*c2)*t+c1


def _parameter(start,end,position,controls):
    lower=0.;upper=1.;t=0.
    for _ in range(24):
        t=(lower+upper)*.5
        if cubic(start['position'],controls[0],controls[1],end['position'],t)<position:
            lower=t
        else:
            upper=t
    return t


def evaluate_source(points,position):
    points=normalize_keyframes(points)
    position=_number(position,'输出位置')
    position=max(0.,position)
    if position<=points[0]['position']:
        return points[0]['source']
    for start,end in zip(points,points[1:]):
        if position>end['position']:
            continue
        span=max(.0001,end['position']-start['position'])
        t=max(0.,min(1.,(position-start['position'])/span))
        if not _bezier(start,end):
            value=start['source']+(end['source']-start['source'])*t
        else:
            controls=_controls(start,end)
            value=_source_segment(start,end,_parameter(start,end,position,controls),controls)
        return max(0.,min(1.,float(value)))
    return points[-1]['source']


def _anchor_speed(points,index,side):
    point=points[index]
    explicit=point['incoming_speed' if side<0 else 'outgoing_speed']
    if explicit is not None:
        return explicit
    if side<0 and index>0:
        start,end=points[index-1],point
        span=max(.0001,end['position']-start['position'])
        if start['interpolation']=='linear' and end['incoming_interpolation']=='linear':
            return (end['source']-start['source'])/span
        influence=max(.01,min(1.,end['left_influence']))
        handle=end['left_handle_source']
        return (end['source']-(end['source'] if handle is None else handle))/(span*max(.001,influence))
    if side>0 and index+1<len(points):
        start,end=point,points[index+1]
        span=max(.0001,end['position']-start['position'])
        if start['interpolation']=='linear' and end['incoming_interpolation']=='linear':
            return (end['source']-start['source'])/span
        influence=max(.01,min(1.,start['right_influence']))
        handle=start['right_handle_source']
        return ((start['source'] if handle is None else handle)-start['source'])/(span*max(.001,influence))
    return 0.


def evaluate_speed(points,position):
    points=normalize_keyframes(points)
    position=_number(position,'输出位置')
    position=max(0.,position)
    if position<=points[0]['position']:
        value=_anchor_speed(points,0,1)
    else:
        value=None
        for start,end in zip(points,points[1:]):
            if position>end['position']:
                continue
            if not _bezier(start,end):
                value=(end['source']-start['source'])/max(.0001,end['position']-start['position'])
            else:
                controls=_controls(start,end);t=_parameter(start,end,position,controls)
                dx=cubic_derivative(start['position'],controls[0],controls[1],end['position'],t)
                if abs(dx)>1e-7:
                    value=_source_derivative(start,end,t,controls)/dx
                else:
                    left=max(start['position'],position-1./4096.)
                    right=min(end['position'],position+1./4096.)
                    value=(evaluate_source(points,right)-evaluate_source(points,left))/max(.0001,right-left)
            break
        if value is None:
            value=_anchor_speed(points,len(points)-1,-1)
    return max(-100.,min(100.,float(value))) if math.isfinite(value) else 0.


def evaluate_sources(points,positions):
    """Vectorized version of the same 24-step inverse, bounded by the caller."""
    points=normalize_keyframes(points)
    positions=np.asarray(positions,dtype=np.float64)
    if not np.isfinite(positions).all():
        raise ValueError('输出位置需要有限数值')
    positions=np.maximum(0.,positions)
    output=np.full(positions.shape,points[-1]['source'],dtype=np.float64)
    output[positions<=points[0]['position']]=points[0]['source']
    for start,end in zip(points,points[1:]):
        selected=(positions>start['position'])&(positions<=end['position'])
        x=positions[selected]
        if not x.size:
            continue
        span=max(.0001,end['position']-start['position'])
        if not _bezier(start,end):
            y=start['source']+(end['source']-start['source'])*np.clip((x-start['position'])/span,0.,1.)
        else:
            controls=_controls(start,end)
            lower=np.zeros(x.size);upper=np.ones(x.size)
            for _ in range(24):
                t=(lower+upper)*.5
                left=cubic(start['position'],controls[0],controls[1],end['position'],t)<x
                lower=np.where(left,t,lower);upper=np.where(left,upper,t)
            y=_source_segment(start,end,t,controls)
        output[selected]=np.clip(y,0.,1.)
    return output
