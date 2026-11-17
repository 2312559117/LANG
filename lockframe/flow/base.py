"""Select a validated flow backend; encoding selection is deliberately separate."""
from .dis import DISInterpolator
from .nvof import availability


def create_interpolator(backend="auto", allow_fallback=False, cancel=None, *,
                        max_cache_bytes=128 * 1024**2, max_pairs=2):
    if backend not in ("auto", "dis", "nvof"):
        raise ValueError("未知的光流插帧后端")
    fallback = availability()["reason"] if backend in ("auto", "nvof") else ""
    return DISInterpolator(cancel=cancel, allow_fallback=allow_fallback,
                           max_cache_bytes=max_cache_bytes, max_pairs=max_pairs,
                           requested_backend=backend, fallback_reason=fallback)
