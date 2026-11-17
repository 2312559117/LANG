"""Bidirectional backward warps on native YUV planes, Keys cubic a=-0.75."""
import math
import cv2
import numpy as np
from .frame import YuvFrame


class FlowError(RuntimeError):
    pass


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        from ..media import Cancelled
        raise Cancelled()


def matching_frames(left, right):
    if not isinstance(left, YuvFrame) or not isinstance(right, YuvFrame):
        raise ValueError("光流插帧要求原生YUV帧")
    if left.format != right.format or (left.width, left.height) != (right.width, right.height):
        raise ValueError("光流插帧要求左右原生YUV位深、布局和尺寸一致")


def validate_flow(flow, width, height):
    array = np.asarray(flow)
    if array.ndim != 3 or array.shape[2] != 2 or min(array.shape[:2]) < 1 or array.dtype.kind != "f":
        raise FlowError("光流向量必须是有限浮点H×W×2数组")
    if (not np.isfinite(array).all()
            or np.max(np.abs(array[..., 0])) > max(1024, 8 * array.shape[1])
            or np.max(np.abs(array[..., 1])) > max(1024, 8 * array.shape[0])):
        raise FlowError("光流向量无效或超出安全范围")
    return np.ascontiguousarray(array, dtype=np.float32)


def _plane_flow(flow, width, height, first, last):
    """Bilinear grid sampling at (pixel+.5)/scale-.5, clamped at edges."""
    gh, gw = flow.shape[:2]
    if (height,width)==(gh,gw):return flow[first:last]
    sx, sy = width / gw, height / gh
    x = np.clip((np.arange(width, dtype=np.float32) + .5) / sx - .5, 0., gw - 1.)
    y = np.clip((np.arange(first, last, dtype=np.float32) + .5) / sy - .5, 0., gh - 1.)
    x0, y0 = x.astype(np.intp), y.astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, gw - 1), np.minimum(y0 + 1, gh - 1)
    wx, wy = (x - x0).astype(np.float32)[None, :, None], (y - y0).astype(np.float32)[:, None, None]
    top = flow[y0[:, None], x0[None, :]] * (1 - wx) + flow[y0[:, None], x1[None, :]] * wx
    bottom = flow[y1[:, None], x0[None, :]] * (1 - wx) + flow[y1[:, None], x1[None, :]] * wx
    sampled = top * (1 - wy) + bottom * wy
    sampled *= np.asarray([sx, sy], dtype=np.float32)
    return sampled


def quantize(values, fmt):
    """Round-even storage sample, semantic-depth rounding, then left alignment."""
    storage_max = (1 << fmt.storage_bits) - 1
    value = np.rint(np.clip(values, 0, storage_max)).astype(np.uint32)
    if fmt.shift:
        value = (value + (1 << (fmt.shift - 1))) >> fmt.shift
    value = np.minimum(value, (1 << fmt.depth) - 1)
    if fmt.shift:
        value = value << fmt.shift
    return value.astype(fmt.dtype)


class SynthesisPreparation:
    def __init__(self,left,right,forward,backward,*,cache_float=False):
        matching_frames(left,right)
        self.left,self.right,self.forward,self.backward=left,right,forward,backward
        self.f=validate_flow(forward,left.width,left.height);self.b=validate_flow(backward,left.width,left.height)
        if self.f.shape!=self.b.shape:raise FlowError('前后向光流网格尺寸不一致')
        self.planes=None;self.nbytes=0
        if cache_float:
            self.planes=tuple((lp.astype(np.float32),rp.astype(np.float32)) for lp,rp in zip(left.planes,right.planes))
            for pair in self.planes:
                for plane in pair:plane.flags.writeable=False;self.nbytes+=plane.nbytes


def synthesize(left, right, forward, backward, alpha, *, cancel=None, tile_rows=96,_prepared=None):
    """Pure synthesis from supplied grid-unit flows; output keeps native format.

    OpenCV's cubic image sampler uses a=-.75 and its documented fractional
    interpolation table. This is the same cubic family, not a byte-exact claim
    about MS's embedded floating-point OpenCL sampler.
    """
    matching_frames(left, right)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.floating)) or not math.isfinite(float(alpha)) or not 0 <= alpha <= 1:
        raise ValueError("插帧时间权重必须在0–1之间")
    check_cancel(cancel)
    if alpha == 0:
        return left.copy()
    if alpha == 1:
        return right.copy()
    if _prepared is not None:
        if not isinstance(_prepared,SynthesisPreparation) or not (_prepared.left is left and _prepared.right is right and _prepared.forward is forward and _prepared.backward is backward):
            raise FlowError('合成准备状态与原生帧对不匹配')
        f,b=_prepared.f,_prepared.b
    else:
        f = validate_flow(forward, left.width, left.height)
        b = validate_flow(backward, left.width, left.height)
    if f.shape != b.shape:
        raise FlowError("前后向光流网格尺寸不一致")
    if type(tile_rows) is not int or not 1 <= tile_rows <= 1024:
        raise ValueError("插帧工作块大小无效")
    output = [None]*len(left.planes)
    shapes={}
    for index,plane in enumerate(left.planes):shapes.setdefault(plane.shape,[]).append(index)
    for (h,w),indices in shapes.items():
      # Share coordinate maps between equal-sized native components (U/V).
      # Bound uncached float conversion to one plane group, rather than growing
      # a full-frame RGB or unbounded workspace.
      group_limit=len(indices) if _prepared is not None and _prepared.planes is not None else max(1,(32*1024**2)//(h*w*8))
      for offset in range(0,len(indices),group_limit):
        group=indices[offset:offset+group_limit]
        planes={index:(_prepared.planes[index] if _prepared is not None and _prepared.planes is not None else
            (left.planes[index].astype(np.float32),right.planes[index].astype(np.float32))) for index in group}
        for index in group:output[index]=np.empty_like(left.planes[index])
        for first in range(0, h, tile_rows):
            check_cancel(cancel)
            last = min(h, first + tile_rows)
            x = np.broadcast_to(np.arange(w, dtype=np.float32), (last-first, w))
            y = np.broadcast_to(np.arange(first, last, dtype=np.float32)[:, None], (last-first, w))
            ff, bb = _plane_flow(f, w, h, first, last), _plane_flow(b, w, h, first, last)
            lx,ly=x-float(alpha)*ff[...,0],y-float(alpha)*ff[...,1]
            rx,ry=x-(1-float(alpha))*bb[...,0],y-(1-float(alpha))*bb[...,1]
            for index in group:
              left_float,right_float=planes[index]
              l = cv2.remap(left_float, lx,ly,
                          cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
              r = cv2.remap(right_float, rx,ry,
                          cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
              output[index][first:last] = quantize((1-float(alpha))*l + float(alpha)*r, left.format)
    check_cancel(cancel)
    return YuvFrame(tuple(output), left.format)
