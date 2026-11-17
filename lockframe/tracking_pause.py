"""Explicit, pickle-safe tracking suspension without a live worker or decoder.

Unmeasured slots in a suspended TrackResult are display placeholders only. Their
computed mask and ``complete=False`` must be honoured by preview/export callers.
"""
from __future__ import annotations

import numpy as np


class PauseAwareCancel:
    """Allow a preparation subprocess to stop, then distinguish pause/cancel."""

    def __init__(self, cancel, pause):
        self.cancel, self.pause = cancel, pause

    def is_set(self):
        return self.cancel.is_set() or self.pause.is_set()

    def wait(self, timeout=None):
        # FFmpeg's existing poll loop waits at most 150 ms. Preserve that bound
        # and wake immediately on ordinary cancellation.
        return self.cancel.wait(timeout) or self.pause.is_set()


def computed_mask(result):
    """Return an independent mask; legacy complete results measured every slot."""
    if result.performance.get("complete", True):
        return np.ones(result.count, dtype=bool)
    values = result.performance.get("computed_frames")
    if (not isinstance(values, list) or len(values) != result.count
            or not all(type(value) is bool for value in values)):
        return np.zeros(result.count, dtype=bool)
    return np.asarray(values, dtype=bool)


def is_complete(result):
    return result is not None and result.performance.get("complete", True) is True


def can_resume_without_edit(context):
    if not context:return False
    if context.get('corrected'):return True
    if context.get('reason')!='manual':return False
    result=context.get('result');frame=context.get('frame')
    if result is None or type(frame) is not int or not result.start<=frame<=result.end:return False
    index=frame-result.start
    return bool(computed_mask(result)[index] and result.states[index]!='lost')


class TrackingPaused(Exception):
    """A finished worker's partial result and optional numeric continuation.

    ``resume`` contains only dictionaries, arrays and scalar values; it is safe
    to transport through the batch ProcessPoolExecutor, with no OpenCV objects.
    Adding/changing an anchor invalidates dependent continuation automatically.
    """

    def __init__(self, result, frame, direction, reason="manual", resume=None, metadata=None):
        self.result = result
        self.frame = int(frame)
        self.direction = int(direction)
        self.reason = str(reason)
        self.resume = resume
        self.metadata = dict(metadata or result.performance.get("tracking_pause", {}))
        self.done_count = int(computed_mask(result).sum())
        label = "跟踪丢失，等待手动修正" if reason == "lost" else "疑似漂移，等待人工复核" if reason=='review' else "已暂停跟踪"
        super().__init__(f"第 {self.frame + 1} 帧 · {label} · 已测量 {self.done_count}/{result.count} 帧")

    def __reduce__(self):
        return (type(self), (self.result, self.frame, self.direction, self.reason,
                             self.resume, self.metadata))


def reset_clip_tracking(clip):
    """Discard a stopped clip's targets/results and on-disk continuation.

    The caller must stop workers first and clear its preview/pause references
    before refreshing or stashing UI state. Settings, range, source and playhead
    are preserved. Checkpoints are source-scoped (including duplicate clips of
    that source); other clips' in-memory results are never changed.

    Cache errors propagate before any clip state is changed, so the caller must
    report failure rather than claim a successful reset.
    """
    from .checkpoints import clear_checkpoints

    removed = clear_checkpoints(clip.info)
    clip.anchors = {}
    clip.track = None
    clip.retrack_previous = None
    clip.tracking_resume = None
    clip.status = ""
    clip.dirty = True
    return removed
