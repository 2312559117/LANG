"""Atomic, versioned NPZ checkpoints with explicit numeric state (no pickle)."""
from pathlib import Path
import hashlib
import json
import uuid
import numpy as np
from .media import cache_root

REVISION = 2


def checkpoint_path(info, anchors, start, end, configuration):
    source = str(Path(info.path).resolve())
    signature = {"revision": REVISION, "source": source, "fingerprint": info.fingerprint,
                 "size": [info.width, info.height, info.frames, info.fps], "range": [start, end],
                 "anchors": sorted((int(k), list(v)) for k, v in anchors.items()), **configuration}
    source_key = hashlib.sha256(source.encode()).hexdigest()[:16]
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    name = Path(source).stem[:40]
    return cache_root() / "tracking-checkpoints" / f"{source_key}-{name}-{key}.npz", key


def save_task_record(path, info, anchors, start, end, settings, *, tracking_sections=None):
    """A regular sequence file lets the UI recover source/ROI even after a crash."""
    from .sequence import Clip, save_sequence
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = Clip(info, settings, dict(anchors), start, end)
    clip.section_tracking = bool(tracking_sections)
    clip.tracking_sections = [dict(value) for value in tracking_sections or []]
    save_sequence(path.with_suffix(".locksequence"), [clip])


def clear_checkpoints(info):
    source = str(Path(info.path).resolve())
    prefix = hashlib.sha256(source.encode()).hexdigest()[:16]
    folder = cache_root() / "tracking-checkpoints"
    count = 0
    if folder.is_dir():
        for path in folder.glob(prefix + "-*.npz"):
            path.unlink(missing_ok=True)
            count += 1
        for path in folder.glob(prefix + "-*.locksequence"):
            path.unlink(missing_ok=True)
    return count


def save_checkpoint(path, key, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays,
                                metadata=np.asarray(json.dumps(dict(metadata, key=key, revision=REVISION))))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, key, count):
    if not path.is_file():
        return None
    try:
        import zipfile
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > max(128 * 1024**2, count * 512):
                return None
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("key") != key or metadata.get("revision") != REVISION:
                return None
            arrays = {name: data[name].copy() for name in data.files if name != "metadata"}
        if any(value.dtype.kind not in "biufU" for value in arrays.values()):
            return None
        expected = {"positions": (count, 2), "boxes": (count, 4), "confidence": (count,),
                    "states": (count,), "computed": (count,)}
        if any(name not in arrays or arrays[name].shape != shape for name, shape in expected.items()):
            return None
        if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fiu"):
            return None
        if not set(arrays["states"]) <= {"manual", "tracked", "recovered", "lost"}:
            return None
        if arrays["computed"].dtype.kind != "b" or type(metadata.get("walk")) is not int:
            return None
        selected = arrays["computed"]
        if (np.any(arrays["boxes"][selected, 2:] <= 0) or np.any(arrays["boxes"][selected] < 0)
                or np.any(arrays["boxes"][selected] > 1) or np.any(arrays["positions"][selected] < 0)
                or np.any(arrays["positions"][selected] > 1)
                or np.any(arrays["confidence"][selected] < 0) or np.any(arrays["confidence"][selected] > 1)):
            return None
        counters = ("fast_frames", "full_frames", "adaptation_updates",
                    "replenishment_events", "quality_rejections", "scale_guards")
        if any(type(metadata.get(name)) is not int or metadata[name] < 0 for name in counters):
            return None
        tracker = metadata.get("tracker")
        if tracker:
            required = ["tracker_" + name for name in ("matrix", "box", "velocity", "point_ids")]
            required += ["reference_" + name for name in ("image", "blurred_image", "points", "core", "weights")]
            if any(name not in arrays for name in required):
                return None
            points = arrays["reference_points"]
            image = arrays["reference_image"]
            if (image.ndim != 2 or image.dtype != np.uint8 or arrays["reference_blurred_image"].shape != image.shape
                    or points.ndim != 2 or points.shape[1] != 2 or len(points) < 4
                    or arrays["reference_core"].shape != (len(points),)
                    or arrays["reference_weights"].shape != (len(points),)
                    or arrays["tracker_matrix"].shape not in {(2, 3), (3, 3)}
                    or arrays["tracker_box"].shape != (4,) or arrays["tracker_velocity"].shape != (2,)):
                return None
            ids = arrays["tracker_point_ids"]
            if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= len(points)):
                return None
            if not tracker.get("points_none") and arrays.get("tracker_points", np.empty(0)).shape != (len(ids), 2):
                return None
            if type(metadata.get("last_frame")) is not int or metadata["last_frame"] < 0:
                return None
            tracker_counters = ("age", "missing", "adaptation_updates", "fast_frames", "full_frames",
                                "replenishment_events", "quality_rejections", "scale_guards")
            if any(type(tracker.get(name)) is not int or tracker[name] < 0
                   for name in tracker_counters):
                return None
            if tracker.get("mstrack_v2"):
                motion_required = ("motion_box", "motion_reference", "motion_adaptive")
                if any(name not in arrays for name in motion_required):
                    return None
                if (arrays["motion_box"].shape != (4,)
                        or arrays["motion_reference"].ndim != 2
                        or arrays["motion_adaptive"].ndim != 2
                        or arrays["motion_reference"].dtype != np.uint8
                        or arrays["motion_adaptive"].dtype != np.uint8):
                    return None
                if not tracker.get("motion_points_none"):
                    points = arrays.get("motion_points", np.empty(0))
                    if points.ndim != 3 or points.shape[1:] != (1, 2):
                        return None
        return arrays, metadata
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return None


def tracker_snapshot(tracker):
    values = {}
    for name in ("matrix", "box", "velocity", "points", "point_ids"):
        value = getattr(tracker, name)
        if value is not None:
            values["tracker_" + name] = np.asarray(value).copy()
    for name in ("image", "blurred_image", "points", "core", "weights"):
        values["reference_" + name] = np.asarray(getattr(tracker.reference, name)).copy()
    metadata = {name: int(getattr(tracker, name, 0)) for name in
                ("age", "missing", "adaptation_updates", "fast_frames", "full_frames",
                 "replenishment_events", "quality_rejections", "scale_guards")}
    metadata["points_none"] = tracker.points is None
    metadata["mstrack_v2"] = bool(getattr(tracker, "_mstrack_v2", False))
    if metadata["mstrack_v2"]:
        for name in ("box", "reference", "adaptive", "points"):
            value = getattr(tracker, "_motion_" + name)
            if value is not None:
                values["motion_" + name] = np.asarray(value).copy()
        metadata["motion_points_none"] = tracker._motion_points is None
    return values, metadata


def restore_tracker(tracker, arrays, metadata, previous_gray):
    for name in ("image", "blurred_image", "points", "core", "weights"):
        setattr(tracker.reference, name, arrays["reference_" + name].copy())
    tracker.template = tracker.reference.image
    for name in ("matrix", "box", "velocity", "point_ids"):
        setattr(tracker, name, arrays["tracker_" + name].copy())
    tracker.points = None if metadata["points_none"] else arrays["tracker_points"].copy()
    if metadata.get("mstrack_v2") and getattr(tracker, "_mstrack_v2", False):
        tracker._motion_box = arrays["motion_box"].copy()
        tracker._motion_reference = arrays["motion_reference"].copy()
        tracker._motion_adaptive = arrays["motion_adaptive"].copy()
        tracker._motion_points = (None if metadata.get("motion_points_none")
                                  else arrays["motion_points"].copy())
        tracker._motion_previous = tracker._motion_gray(previous_gray)
    for name in ("age", "missing", "adaptation_updates", "fast_frames", "full_frames",
                 "replenishment_events", "quality_rejections", "scale_guards"):
        if hasattr(tracker, name):
            setattr(tracker, name, metadata[name])
    tracker.previous = previous_gray
    tracker._reference_updated()
