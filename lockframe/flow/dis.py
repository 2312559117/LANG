"""Independent bidirectional DIS with explicit, bounded prepared-pair reuse."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import hashlib
import time

import cv2
import numpy as np

from .frame import YuvFrame
from .synthesis import FlowError, check_cancel, matching_frames, synthesize, validate_flow,SynthesisPreparation


DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
DIS_PRESET_DESCRIPTION = "MEDIUM (番茄产品选择；MS原生preset未知)"


class PreparedFlowPair:
    def __init__(self, left, right, forward, backward, owner):
        self.left, self.right = left.copy(), right.copy()
        self.forward, self.backward = forward.copy(), backward.copy()
        for value in (*self.left.planes, *self.right.planes, self.forward, self.backward):
            value.flags.writeable = False
        self.owner = owner
        self.backend_name = "dis"
        self.nbytes = self.left.nbytes + self.right.nbytes + self.forward.nbytes + self.backward.nbytes
        float_bytes=sum(p.size*4 for p in (*self.left.planes,*self.right.planes))
        self.preparation=SynthesisPreparation(self.left,self.right,self.forward,self.backward,
            cache_float=bool(owner.max_pairs and self.nbytes+float_bytes<=owner.max_cache_bytes))
        self.nbytes+=self.preparation.nbytes

    def compose(self, alpha):
        check_cancel(self.owner.cancel)
        output = synthesize(self.left, self.right, self.forward, self.backward, alpha, cancel=self.owner.cancel,_prepared=self.preparation)
        self.owner.stats["composed_frames"] += 1
        return output


class NearestPair:
    def __init__(self, left, right, owner, reason):
        self.left, self.right, self.owner = left.copy(), right.copy(), owner
        self.backend_name = "nearest"
        self.reason = reason
        self.nbytes = self.left.nbytes + self.right.nbytes

    def compose(self, alpha):
        if (isinstance(alpha, bool) or not np.isscalar(alpha) or not np.isfinite(alpha)
                or not 0 <= alpha <= 1):
            raise ValueError("插帧时间权重必须在0–1之间")
        check_cancel(self.owner.cancel)
        self.owner.stats["composed_frames"] += 1
        return (self.left if alpha < .5 else self.right).copy()


class DISInterpolator:
    """One render-job-owned interpolator; do not call prepare concurrently.

    Two native DIS objects run independently, one per direction. Cancellation is
    observed while waiting and at synthesis tiles; native calc itself has no
    OpenCV cancellation callback and completes before its resources are freed.
    """
    def __init__(self, *, cancel=None, allow_fallback=False, max_cache_bytes=128 * 1024**2, max_pairs=2,
                 requested_backend="dis", fallback_reason=""):
        if type(max_cache_bytes) is not int or max_cache_bytes < 0 or type(max_pairs) is not int or not 0 <= max_pairs <= 8:
            raise ValueError("光流帧对缓存预算无效")
        if type(allow_fallback) is not bool:
            raise ValueError("光流降级开关无效")
        self.cancel, self.allow_fallback = cancel, allow_fallback
        self.max_cache_bytes, self.max_pairs = max_cache_bytes, max_pairs
        self.backend_name = "dis"
        self.status = dict(requested_backend=requested_backend, backend="dis", fallback_reason=fallback_reason,
                           nvof_available=False, dis_preset=DIS_PRESET_DESCRIPTION, synthesis_backend="cpu")
        self.stats = dict(prepared_pairs=0, cache_hits=0, cache_bytes=0, cache_pairs=0,
                          cache_peak_bytes=0, fallback_pairs=0, composed_frames=0, flow_seconds=0.)
        self._cache = OrderedDict()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retime-dis")
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _flow(first, second):
        if not hasattr(cv2, "DISOpticalFlow_create"):
            raise FlowError("当前OpenCV运行环境未提供DISOpticalFlow")
        algorithm = cv2.DISOpticalFlow_create(DIS_PRESET)
        return algorithm.calc(first, second, None)

    def _result(self, future):
        while True:
            check_cancel(self.cancel)
            try:
                return future.result(timeout=.05)
            except FutureTimeout:
                pass

    def _key(self, left, right, key):
        if key is None:
            digest = hashlib.blake2b(digest_size=20)
            for plane in (*left.planes, *right.planes):
                digest.update(np.ascontiguousarray(plane).view(np.uint8))
            key = digest.digest()
        try:
            hash(key)
        except TypeError as error:
            raise ValueError("光流帧对缓存key必须不可变") from error
        return (key, left.width, left.height, left.format)

    def prepare(self, left, right, key=None):
        if self._closed:
            raise RuntimeError("光流插帧器已关闭")
        check_cancel(self.cancel)
        matching_frames(left, right)
        cache_key = self._key(left, right, key)
        if cache_key in self._cache:
            pair = self._cache.pop(cache_key)
            self._cache[cache_key] = pair
            self.stats["cache_hits"] += 1
            return pair
        started = time.perf_counter()
        try:
            first, second = left.luma8(), right.luma8()
            # Futures retain their own input references until calc finishes.
            forward = self._pool.submit(self._flow, first, second)
            backward = self._pool.submit(self._flow, second, first)
            f = validate_flow(self._result(forward), left.width, left.height)
            b = validate_flow(self._result(backward), left.width, left.height)
            if f.shape != b.shape:
                raise FlowError("前后向DIS光流网格尺寸不匹配")
            pair = PreparedFlowPair(left, right, f, b, self)
            self.stats["prepared_pairs"] += 1
        except (cv2.error, FlowError, RuntimeError) as error:
            check_cancel(self.cancel)
            if not self.allow_fallback:
                raise FlowError(f"DIS双向光流失败，已停止；未允许退回逐帧：{error}") from error
            reason = f"DIS失败；按用户配置退回最近帧：{error}"
            self.status.update(backend="nearest", fallback_reason=reason)
            self.backend_name = "nearest"
            self.stats["fallback_pairs"] += 1
            pair = NearestPair(left, right, self, reason)
        finally:
            self.stats["flow_seconds"] += time.perf_counter() - started
        check_cancel(self.cancel)
        if self.max_pairs and pair.nbytes <= self.max_cache_bytes:
            self._cache[cache_key] = pair
            while len(self._cache) > self.max_pairs or sum(p.nbytes for p in self._cache.values()) > self.max_cache_bytes:
                self._cache.popitem(last=False)
        self.stats["cache_bytes"] = sum(p.nbytes for p in self._cache.values())
        self.stats["cache_peak_bytes"] = max(self.stats["cache_peak_bytes"], self.stats["cache_bytes"])
        self.stats["cache_pairs"] = len(self._cache)
        return pair

    def clear(self):
        self._cache.clear()
        self.stats.update(cache_bytes=0, cache_pairs=0)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        self.clear()
