"""Validated native integer YUV planes; no RGB conversion or depth reduction."""
from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np


MAX_FRAME_BYTES = 1024 ** 3


@dataclass(frozen=True)
class PixelFormat:
    name: str
    depth: int
    sub_x: int
    sub_y: int
    shift: int = 0

    def __post_init__(self):
        if (not isinstance(self.name, str) or type(self.depth) is not int or self.depth not in (8, 10, 12, 16)
                or type(self.sub_x) is not int or type(self.sub_y) is not int
                or (self.sub_x, self.sub_y) not in ((2, 2), (2, 1), (1, 1))
                or type(self.shift) is not int or self.shift < 0
                or self.depth + self.shift > self.storage_bits):
            raise ValueError("不支持的原生YUV位深或采样布局")

    @property
    def storage_bits(self):
        return 8 if self.depth == 8 else 16

    @property
    def dtype(self):
        return np.dtype("u1" if self.depth == 8 else "<u2")

    @property
    def max_value(self):
        return ((1 << self.depth) - 1) << self.shift

    @property
    def canonical_name(self):
        sampling = {(2, 2): "420", (2, 1): "422", (1, 1): "444"}[self.sub_x, self.sub_y]
        return f"yuv{sampling}p" + ("" if self.depth == 8 else f"{self.depth}le")

    def plane_shapes(self, width, height):
        if type(width) is not int or type(height) is not int or not 1 <= width < 32767 or not 1 <= height < 32767:
            raise ValueError("原生YUV尺寸应为1–32766像素")
        chroma = ((height + self.sub_y - 1) // self.sub_y,
                  (width + self.sub_x - 1) // self.sub_x)
        return ((height, width), chroma, chroma)

    def frame_bytes(self, width, height):
        size = sum(h * w for h, w in self.plane_shapes(width, height)) * self.dtype.itemsize
        if size > MAX_FRAME_BYTES:
            raise ValueError("单帧原生YUV数据超过1GiB工作上限")
        return size


def parse_format(name):
    if isinstance(name, PixelFormat):
        return name
    if not isinstance(name, str):
        raise ValueError("原生像素格式无效")
    return _parse_format(name)


@lru_cache(maxsize=32)
def _parse_format(name):
    if name in ("nv12", "nv21"):
        return PixelFormat(name, 8, 2, 2)
    if name in ("p010le", "p010"):
        return PixelFormat("p010le", 10, 2, 2, 6)
    if name in ("p016le", "p016"):
        return PixelFormat("p016le", 16, 2, 2)
    match = re.fullmatch(r"yuv(?:j)?(420|422|444)p(?:(10|12|16)le)?", name)
    if not match:
        raise ValueError(f"不支持原生像素格式 {name!r}；请规范化为同位深planar YUV")
    sub = {"420": (2, 2), "422": (2, 1), "444": (1, 1)}[match[1]]
    return PixelFormat(name, int(match[2] or 8), *sub)


@dataclass(frozen=True)
class YuvFrame:
    planes: tuple
    format: PixelFormat

    def __post_init__(self):
        object.__setattr__(self, "format", parse_format(self.format))
        if not isinstance(self.planes, (tuple, list)) or len(self.planes) != 3:
            raise ValueError("原生YUV帧必须包含Y/U/V三个平面")
        values = tuple(np.asarray(plane) for plane in self.planes)
        if values[0].ndim != 2:
            raise ValueError("原生Y平面尺寸无效")
        h, w = values[0].shape
        expected = self.format.plane_shapes(w, h)
        self.format.frame_bytes(w, h)
        for plane, shape in zip(values, expected):
            if plane.ndim != 2 or plane.shape != shape or plane.dtype != self.format.dtype:
                raise ValueError("原生YUV平面的形状或整数位深不匹配")
            if self.format.depth < self.format.storage_bits and np.any(plane > self.format.max_value):
                raise ValueError("原生YUV样本超出声明位深")
            if self.format.shift and np.any(plane & ((1 << self.format.shift) - 1)):
                raise ValueError("高位对齐YUV样本含无效低位")
        object.__setattr__(self, "planes", values)

    @property
    def width(self):
        return self.planes[0].shape[1]

    @property
    def height(self):
        return self.planes[0].shape[0]

    @property
    def nbytes(self):
        return sum(plane.nbytes for plane in self.planes)

    @classmethod
    def from_bytes(cls, raw, width, height, pixel_format):
        fmt = parse_format(pixel_format)
        size = fmt.frame_bytes(width, height)
        if len(raw) != size:
            raise ValueError(f"原生YUV帧字节数不匹配：需要{size}，得到{len(raw)}")
        samples = np.frombuffer(raw, dtype=fmt.dtype)
        shapes = fmt.plane_shapes(width, height)
        y_count = width * height
        c_count = shapes[1][0] * shapes[1][1]
        y = samples[:y_count].reshape(shapes[0])
        if fmt.name in ("nv12", "nv21", "p010le", "p016le"):
            uv = samples[y_count:].reshape(c_count, 2)
            u, v = uv[:, 0].reshape(shapes[1]), uv[:, 1].reshape(shapes[2])
            if fmt.name == "nv21":
                u, v = v, u
        else:
            u = samples[y_count:y_count + c_count].reshape(shapes[1])
            v = samples[y_count + c_count:].reshape(shapes[2])
        return cls((y, u, v), fmt)

    def to_bytes(self):
        y, u, v = self.planes
        if self.format.name in ("nv12", "nv21", "p010le", "p016le"):
            if self.format.name == "nv21":
                u, v = v, u
            uv = np.empty((u.shape[0], u.shape[1] * 2), dtype=self.format.dtype)
            uv[:, 0::2], uv[:, 1::2] = u, v
            return y.tobytes() + uv.tobytes()
        return b"".join(plane.tobytes() for plane in self.planes)

    def copy(self):
        return YuvFrame(tuple(plane.copy() for plane in self.planes), self.format)

    def luma8(self):
        if self.format.depth == 8:
            return np.ascontiguousarray(self.planes[0])
        return np.ascontiguousarray((self.planes[0] >> (self.format.shift + self.format.depth - 8)).astype(np.uint8))
