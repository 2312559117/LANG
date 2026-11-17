"""Bounded working memory and measured OpenCL template matching.

The percentage is a ceiling for total machine memory, not a request to fill it.
No synthetic GPU workload is created to inflate utilization counters.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from contextlib import contextmanager
import ctypes
import os
import queue
import threading
import time

import cv2
import numpy as np

from .media import FrameReader, check_cancel


@dataclass(frozen=True)
class ComputeOptions:
    acceleration: str = "auto"
    memory_percent: int = 80
    threads: int = min(8, os.cpu_count() or 1)
    cache_divisor: int = 1

    def validate(self):
        if self.acceleration not in {"auto", "gpu", "cpu"}:
            raise ValueError("未知的计算设备")
        if not 20 <= self.memory_percent <= 80:
            raise ValueError("内存上限应在 20%–80% 之间")
        if not 1 <= self.threads <= 64:
            raise ValueError("计算线程应在 1–64 之间")
        if self.cache_divisor not in {1, 2}:
            raise ValueError("缓存分配数应为 1 或 2")


def memory_status():
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        *[(name, ctypes.c_ulonglong) for name in
                          ("total", "available", "page_total", "page_free", "virtual_total", "virtual_free", "extended")]]
        value = Memory()
        value.length = ctypes.sizeof(value)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return {"total": value.total, "available": value.available,
                    "used": value.total - value.available, "percent": value.load}
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        available = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        return {"total": total, "available": available, "used": total - available,
                "percent": round((total - available) / total * 100)}
    except (AttributeError, ValueError, OSError):
        return {"total": 0, "available": 0, "used": 0, "percent": 0}


def cache_budget(percent, memory=None):
    memory = memory if memory is not None else memory_status()
    # Keep room for frame decoding, ECC scratch buffers and the desktop.
    reserve = 384 * 1024 ** 2
    return max(0, int(min(memory["total"] * percent / 100 - memory["used"] - reserve,
                          memory["available"] - reserve)))


@contextmanager
def compute_scope(options):
    options.validate()
    previous_threads = cv2.getNumThreads()
    previous_opencl = cv2.ocl.useOpenCL()
    cv2.setNumThreads(options.threads)
    cv2.ocl.setUseOpenCL(options.acceleration != "cpu")
    try:
        yield
    finally:
        cv2.setNumThreads(previous_threads)
        cv2.ocl.setUseOpenCL(previous_opencl)


class GrayFrameCache:
    """Decode lazily, reuse gray pixels for both directions, evict under pressure."""
    def __init__(self, path, options, cancel):
        self.reader = FrameReader(path)
        self.options, self.cancel = options, cancel
        self.capacity = cache_budget(options.memory_percent) // options.cache_divisor
        self.items = OrderedDict()
        self.bytes = self.peak_bytes = self.hits = self.misses = 0
        self.last_pressure_check = 0.
        self.max_frames = None

    def read(self, index):
        check_cancel(self.cancel)
        now = time.monotonic()
        if now - self.last_pressure_check > .5:
            # A fresh budget excludes memory already held by this cache.
            memory = memory_status()
            memory = dict(memory, used=max(0, memory["used"] - self.bytes),
                          available=memory["available"] + self.bytes)
            self.capacity = cache_budget(self.options.memory_percent, memory) // self.options.cache_divisor
            self.last_pressure_check = now
            self._evict()
        if index in self.items:
            self.hits += 1
            self.items.move_to_end(index)
            return self.items[index]
        frame = cv2.cvtColor(self.reader.read(index), cv2.COLOR_BGR2GRAY)
        self.misses += 1
        if frame.nbytes <= self.capacity:
            self.items[index] = frame
            self.bytes += frame.nbytes
            self._evict()
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return frame

    def _evict(self):
        while self.items and (self.bytes > self.capacity or (self.max_frames and len(self.items) > self.max_frames)):
            _, frame = self.items.popitem(last=False)
            self.bytes -= frame.nbytes

    def close(self):
        self.reader.close()
        self.items.clear()
        self.bytes = 0

    def iter_frames(self, indices):
        """Overlap sequential decode with tracking, with a bounded lookahead."""
        pending = queue.Queue(maxsize=4 if self.capacity else 1)
        stopped = threading.Event()

        def publish(item):
            while not stopped.is_set():
                try:
                    pending.put(item, timeout=.1)
                    return
                except queue.Full:
                    check_cancel(self.cancel)

        def decode():
            try:
                for index in indices:
                    if stopped.is_set():
                        return
                    publish((index, self.read(index), None))
                publish((None, None, None))
            except Exception as error:
                while not stopped.is_set():
                    try:
                        pending.put((None, None, error), timeout=.1)
                        return
                    except queue.Full:
                        pass

        thread = threading.Thread(target=decode, name="tomato-frame-prefetch", daemon=True)
        thread.start()
        try:
            while True:
                check_cancel(self.cancel)
                try:
                    index, frame, error = pending.get(timeout=.1)
                except queue.Empty:
                    continue
                if error is not None:
                    raise error
                if index is None:
                    return
                yield index, frame
        finally:
            stopped.set()
            thread.join()


class TemplateMatcher:
    """Masked zero-mean NCC via OpenCL correlations, with a CPU fallback.

    OpenCV's masked matchTemplate is CPU-only in the standard build. Three
    unmasked CCORR operations express the same binary-mask statistic and can
    execute through its existing OpenCL backend. Final tracking verification
    remains on the immutable reference and rejects unreliable matches.
    """
    def __init__(self, acceleration="auto"):
        self.acceleration = acceleration
        self.gpu_calls = self.cpu_calls = 0
        self.reason = "CPU"
        self.device = ""
        self.available = False
        if acceleration != "cpu":
            try:
                cv2.ocl.setUseOpenCL(True)
                device = cv2.ocl.Device_getDefault()
                self.available = bool(cv2.ocl.useOpenCL() and device.available() and int(device.type()) & 4)
                self.device = device.name() if self.available else ""
                if self.available:
                    self.reason = "OpenCL GPU"
            except cv2.error:
                self.reason = "OpenCL 不可用，CPU 回退"
        self._search = None
        self._gpu_search = self._gpu_squared = None

    def prepare(self, search):
        self._search = search
        self._gpu_search = self._gpu_squared = None

    def scores(self, search, template, mask):
        # Kernel dispatch is slower than CPU on tiny crops. Explicit GPU mode
        # permits these as well; auto avoids the transfer overhead.
        gpu = self.available and (self.acceleration == "gpu" or search.size >= 120_000)
        if gpu:
            try:
                if self._search is not search:
                    self.prepare(search)
                if self._gpu_search is None:
                    self._gpu_search = cv2.UMat(search.astype(np.float32))
                    self._gpu_squared = cv2.multiply(self._gpu_search, self._gpu_search)
                weights = (mask > 0).astype(np.float32)
                count = float(weights.sum())
                centered = (template.astype(np.float32) - float((template * weights).sum()) / max(count, 1.)) * weights
                power = float((centered * centered).sum())
                if count < 4 or power < 1:
                    return np.full((search.shape[0] - template.shape[0] + 1,
                                    search.shape[1] - template.shape[1] + 1), -1., np.float32)
                gpu_mask = cv2.UMat(weights)
                sums = cv2.matchTemplate(self._gpu_search, gpu_mask, cv2.TM_CCORR)
                squares = cv2.matchTemplate(self._gpu_squared, gpu_mask, cv2.TM_CCORR)
                variance = cv2.subtract(squares, cv2.multiply(sums, sums, scale=1. / count))
                numerator = cv2.matchTemplate(self._gpu_search, cv2.UMat(centered), cv2.TM_CCORR)
                denominator = cv2.sqrt(cv2.multiply(cv2.max(variance, 0.), power))
                result = cv2.divide(numerator, denominator).get()
                # Suppress zero-variance/near-uniform windows (including finite
                # overflows) instead of treating them as confident evidence.
                variance_cpu = variance.get()
                invalid = (~np.isfinite(result) | (variance_cpu < count * 1.)
                           | (result > 1.001) | (result < -1.001))
                result[invalid] = -1.
                self.gpu_calls += 1
                return result
            except cv2.error:
                self.available = False
                self.reason = "OpenCL 执行失败，CPU 回退"
                self._gpu_search = self._gpu_squared = None
        self.cpu_calls += 1
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        result = np.nan_to_num(result, nan=-1., posinf=-1., neginf=-1.)
        result[(result > 1.001) | (result < -1.001)] = -1.
        return result
