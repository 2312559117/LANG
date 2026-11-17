"""Single clip-wide region footprint, separate from source-frame ROI anchors."""
from dataclasses import dataclass
import math
from numbers import Real


BOX_CORNERS=((0.,0.),(1.,0.),(1.,1.),(0.,1.))


def validate_polygon(points):
    """Return a simple closed footprint (closure is implicit), preserving order."""
    if not isinstance(points,(tuple,list)) or not 3<=len(points)<=64:
        raise ValueError('多边形需要3–64个顶点')
    result=[]
    for point in points:
        if not isinstance(point,(tuple,list)) or len(point)!=2:
            raise ValueError('多边形顶点格式无效')
        if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in point):
            raise ValueError('多边形顶点必须是有限坐标')
        x,y=map(float,point)
        if not -1e-9<=x<=1+1e-9 or not -1e-9<=y<=1+1e-9:
            raise ValueError('多边形顶点超出区域外接框')
        result.append((max(0.,min(1.,x)),max(0.,min(1.,y))))
    n=len(result)
    if any(math.dist(result[i],result[j])<1e-7 for i in range(n) for j in range(i)):
        raise ValueError('多边形不能包含重合顶点')
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c):return abs(cross(a,b,c))<1e-10 and min(a[0],b[0])-1e-10<=c[0]<=max(a[0],b[0])+1e-10 and min(a[1],b[1])-1e-10<=c[1]<=max(a[1],b[1])+1e-10
    def intersects(a,b,c,d):
        ab,ad,ca,cb=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
        return (ab*ad<0 and ca*cb<0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)
    for i in range(n):
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1):continue
            if intersects(result[i],result[(i+1)%n],result[j],result[(j+1)%n]):
                raise ValueError('多边形边不能自交或相碰')
    area=abs(sum(result[i][0]*result[(i+1)%n][1]-result[(i+1)%n][0]*result[i][1] for i in range(n)))/2
    if area<1e-6:raise ValueError('多边形面积太小或已退化')
    return tuple(result)


def shape_data(shape,points=()):
    if shape=='rect':shape='rectangle'
    if shape not in ('rectangle','ellipse','circle','polygon'):raise ValueError('未知的区域形状')
    return shape,validate_polygon(points) if shape=='polygon' else ()


def shape_matches(result,settings):
    """Legacy results without shape provenance are rectangular, never polygonal."""
    if result is None:return True
    try:
        old_shape,old_points=shape_data(result.performance.get('shape','rectangle'),result.performance.get('polygon_points',()))
        shape,points=shape_data(settings.target_shape,settings.polygon_points)
        old_shape='ellipse' if old_shape=='circle' else old_shape
        shape='ellipse' if shape=='circle' else shape
        return old_shape==shape and old_points==points
    except (ValueError,TypeError):return False


def anchors_match(result,anchors,start,end):
    if result is None or result.anchors is None:return True
    current={int(f):tuple(box) for f,box in anchors.items() if start<=int(f)<=end}
    old={int(f):tuple(box) for f,box in result.anchors.items()}
    return old==current


@dataclass(frozen=True)
class TrackingRegion:
    """Source-normalized geometry for an edit; only the footprint is clip-wide."""
    shape: str
    x: float
    y: float
    w: float
    h: float
    points: tuple=()

    @classmethod
    def from_box(cls,box,shape='rectangle',points=()):
        shape,points=shape_data(shape,points)
        if len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) for v in box):
            raise ValueError('区域外接框无效')
        x,y,w,h=map(float,box)
        if min(x,y)<-1e-9 or min(w,h)<=0 or x+w>1+1e-9 or y+h>1+1e-9:
            raise ValueError('区域外接框超出原片')
        return cls(shape,x,y,w,h,tuple((x+u*w,y+v*h) for u,v in points))

    @property
    def box(self):return self.x,self.y,self.w,self.h

    @property
    def local_points(self):
        return validate_polygon(tuple(((x-self.x)/self.w,(y-self.y)/self.h) for x,y in self.points)) if self.shape=='polygon' else ()
