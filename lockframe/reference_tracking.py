"""Immutable-reference checks for the lock-on optical-flow tracker.

The two-template/reference-anchor idea also appears in StableFlow Pro's
RectangleTemplateTracker. This implementation uses OpenCV similarity geometry
and never learns a new target from an unverified, potentially drifting box.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from .planar import REGIONAL_NCC_THRESHOLD


TRACKING_REVISION = "mstrackpro-offline-approx-v2"


@dataclass(frozen=True)
class MotionQuality:
    """Independent geometry evidence used by the per-frame quality gate."""

    inlier_count: int
    total_points: int
    inlier_ratio: float
    reprojection_mean: float
    reprojection_median: float
    point_spread: float
    confidence: float
    quality_ok: bool


def transform_points(points, matrix):
    """Apply a similarity (2×3) or perspective (3×3) transform to 2D points."""
    points = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 3):
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        mapped = homogeneous @ matrix.T
        return mapped[:, :2] / mapped[:, 2:3]
    return points @ matrix[:, :2].T + matrix[:, 2]


def balanced_features(gray, region_mask=None):
    """Keep interior detail in every region; let the selected center lead."""
    height, width = gray.shape
    margin = max(2, round(min(width, height) * 0.10))
    xs = np.linspace(margin, width - margin, 4).astype(int)
    ys = np.linspace(margin, height - margin, 4).astype(int)
    result = []
    # Detect the center first rather than allowing a detailed background edge
    # to consume all of goodFeaturesToTrack's global corner budget.
    cells = [(1, 1)] + [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    for row, column in cells:
        mask = np.zeros_like(gray)
        mask[ys[row]:ys[row + 1], xs[column]:xs[column + 1]] = 255
        if region_mask is not None:
            mask = cv2.bitwise_and(mask, region_mask)
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=24 if (row, column) == (1, 1) else 10,
            qualityLevel=0.015, minDistance=max(2, min(width, height) / 24),
            mask=mask, blockSize=3)
        if points is not None:
            result.append(points.reshape(-1, 2))
    return np.concatenate(result).astype(np.float32) if result else np.empty((0, 2), np.float32)

def motion_features(gray, mask=None):
    """Independent MS approximation defaults, respecting the caller's ROI mask.

    A rectangle stays a rectangle: only ellipse selections use an ellipse mask.
    Coordinates are refined in the analysis image, not snapped to integer pixels.
    """
    points = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=.01,
                                    minDistance=6, mask=mask, blockSize=5)
    if points is None:
        return np.empty((0, 2), np.float32)
    cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), (3, 20, .01))
    points = points.reshape(-1, 2)
    good = np.isfinite(points).all(axis=1)
    good &= (points[:, 0] >= 0) & (points[:, 0] < gray.shape[1])
    good &= (points[:, 1] >= 0) & (points[:, 1] < gray.shape[0])
    if mask is not None:
        # Subpixel refinement can cross the footprint's boundary.
        xy = np.rint(points).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, gray.shape[1]-1)
        xy[:, 1] = np.clip(xy[:, 1], 0, gray.shape[0]-1)
        good &= mask[xy[:, 1], xy[:, 0]] > 0
    return points[good].astype(np.float32)


def mstp_v2_features(gray, box, max_corners=120):
    """Reproduce the REA-observed MS v2 feature bank in image coordinates.

    The recovered client uses an ellipse inscribed in the selected rectangle,
    even though the UI exposes a rectangular ROI.  Keeping the mask in the
    full analysis image also preserves LK neighbourhood pixels at its edge.
    """
    x, y, width, height = np.asarray(box, dtype=np.float64)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.ellipse(
        mask,
        (round(x + width / 2), round(y + height / 2)),
        (max(1, round(width / 2)), max(1, round(height / 2))),
        0, 0, 360, 255, -1,
    )
    points = cv2.goodFeaturesToTrack(
        gray, maxCorners=int(max_corners), qualityLevel=.01,
        minDistance=6, mask=mask, blockSize=5)
    if points is None:
        return None
    cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), (3, 20, .01))
    return points.astype(np.float32)


def mstp_v2_crop(gray, box):
    """Extract the v2 template with the same rounded dimensions and centre."""
    x, y, width, height = np.asarray(box, dtype=np.float64)
    return cv2.getRectSubPix(
        gray,
        (max(8, int(round(width))), max(8, int(round(height)))),
        (float(x + width / 2), float(y + height / 2)),
    )


def _mstp_v2_template_fallback(previous, gray, box, points, reference, adaptive):
    """Local template reacquisition observed in the recovered offline v2."""
    x, y, width, height = np.asarray(box, dtype=np.float64)
    prediction = np.asarray(box, dtype=np.float64).copy()
    lk_confidence = 0.
    if points is not None and len(points) >= 4:
        options = dict(
            winSize=(25, 25), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01),
        )
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **options)
        if nxt is not None and status is not None:
            back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, previous, nxt, None, **options)
            if back is not None and reverse is not None:
                valid = ((status.ravel() != 0) & (reverse.ravel() != 0)
                         & (np.linalg.norm(points - back, axis=2).ravel() < 1.5))
                if valid.sum() >= 4:
                    source = points[valid].reshape(-1, 2)
                    destination = nxt[valid].reshape(-1, 2)
                    prediction[:2] += np.median(destination - source, axis=0)
                    lk_confidence = float(valid.sum() / len(points))

    px, py, width, height = prediction
    margin = max(16, int(max(width, height) * 1.4))
    left, top = max(0, int(px - margin)), max(0, int(py - margin))
    right = min(gray.shape[1], int(px + width + margin))
    bottom = min(gray.shape[0], int(py + height + margin))
    search = gray[top:bottom, left:right]
    candidates = []
    for template in (reference, adaptive):
        if template is None:
            continue
        for scale in (.96, 1., 1.04):
            tw, th = max(8, int(width * scale)), max(8, int(height * scale))
            if tw > search.shape[1] or th > search.shape[0]:
                continue
            candidate = cv2.resize(template, (tw, th), interpolation=cv2.INTER_LINEAR)
            if float(np.std(candidate)) < 2:
                continue
            response = cv2.matchTemplate(search, candidate, cv2.TM_CCOEFF_NORMED)
            scores = response.copy()
            for _ in range(2):
                _, score, _, location = cv2.minMaxLoc(scores)
                distance = math.hypot(left + location[0] - px,
                                      top + location[1] - py) / max(width, height)
                score = float(score) - .025 * distance
                candidates.append((score, np.array(
                    [left + location[0], top + location[1], tw, th], np.float64)))
                lx, ly = location
                mx, my = max(4, round(tw * .35)), max(4, round(th * .35))
                scores[max(0, ly - my):ly + my + 1,
                       max(0, lx - mx):lx + mx + 1] = -1.

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_box = candidates[0] if candidates else (-1., prediction.copy())
    # Keep Tomato's explicit drift protection: the recovered v2 picked the
    # single maximum, but two almost-equal distant motifs are not evidence of
    # a unique target.  Reporting lost here lets the UI pause for correction.
    for score, other in candidates[1:]:
        separation = np.linalg.norm((best_box[:2] - other[:2])
                                    / np.maximum(best_box[2:], 1.))
        if separation > .55 and best_score - score < .065:
            return (np.asarray(box, dtype=np.float64).copy(),
                    mstp_v2_features(gray, box), 0., "held")

    if best_score >= .48:
        result = best_box
        state = "tracked" if lk_confidence > .2 else "reacquired"
    elif lk_confidence >= .5:
        result = prediction
        state = "tracked"
    else:
        result = np.asarray(box, dtype=np.float64).copy()
        state = "held"
    result[2] = min(max(8, result[2]), gray.shape[1])
    result[3] = min(max(8, result[3]), gray.shape[0])
    result[0] = np.clip(result[0], 0, gray.shape[1] - result[2])
    result[1] = np.clip(result[1], 0, gray.shape[0] - result[3])
    confidence = (max(0., min(1., .7 * max(0., best_score) + .3 * lk_confidence))
                  if state != "held" else 0.)
    return result, mstp_v2_features(gray, result), confidence, state


def mstp_v2_follow(previous, gray, box, points, reference, adaptive):
    """One exact offline-v2 motion step, with held kept distinguishable.

    Affine rotation/scale contribute only to the transformed ROI centre.  The
    ordinary MS path keeps the selected box dimensions unchanged until local
    template reacquisition explicitly returns a nearby scale.
    """
    result = np.asarray(box, dtype=np.float64).copy()
    confidence, state, good = 0., "held", None
    if points is not None and len(points) >= 3:
        options = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(3, 30, .01), minEigThreshold=.0001,
        )
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **options)
        if nxt is not None and status is not None:
            back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, previous, nxt, None, **options)
            if back is not None and reverse is not None:
                valid = ((status.ravel() != 0) & (reverse.ravel() != 0)
                         & (np.linalg.norm(points - back, axis=2).ravel() < 1.5))
                if valid.sum() >= 3:
                    source = points[valid].reshape(-1, 2)
                    destination = nxt[valid].reshape(-1, 2)
                    matrix, inliers = cv2.estimateAffinePartial2D(
                        source, destination, method=cv2.RANSAC,
                        ransacReprojThreshold=3, maxIters=2000,
                        confidence=.99, refineIters=10)
                    center = np.array(
                        [box[0] + box[2] / 2, box[1] + box[3] / 2, 1.],
                        dtype=np.float64)
                    if matrix is not None and inliers is not None:
                        predicted = matrix @ center
                        delta = predicted - center[:2]
                        good = destination[inliers.ravel() != 0].reshape(-1, 1, 2)
                    else:
                        delta = np.median(destination - source, axis=0)
                        good = destination.reshape(-1, 1, 2)
                    result[:2] += delta
                    confidence = float(len(good) / len(points))
                    state = "tracked"
    if state == "held":
        return _mstp_v2_template_fallback(
            previous, gray, box, points, reference, adaptive)
    result[:2] = np.clip(
        result[:2], [0, 0],
        np.asarray([gray.shape[1], gray.shape[0]]) - result[2:])
    if good is None or len(good) < 8:
        good = mstp_v2_features(gray, result)
    return result, good, confidence, state


def persistent_motion(previous, current, points, initial=None):
    """Measured frame-to-frame LK and partial affine, independent of templates.

    Return actual next-frame points, original-bank inlier indices and a 2x3
    transform. A failed observation returns None; never synthesize a success.
    """
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    if len(points) < 4:
        return None
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(3, 30, .01), minEigThreshold=.0001)
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points,
        None if initial is None else np.asarray(initial, np.float32).reshape(-1, 1, 2),
        flags=0 if initial is None else cv2.OPTFLOW_USE_INITIAL_FLOW, **options)
    if nxt is None or status is None:
        return None
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None, **options)
    if back is None or reverse is None:
        return None
    good = (status.ravel() != 0) & (reverse.ravel() != 0)
    good &= np.isfinite(nxt).all(axis=(1, 2)) & np.isfinite(back).all(axis=(1, 2))
    good &= np.linalg.norm(points-back, axis=2).ravel() < 1.5
    good &= (nxt[:, 0, 0] >= 0) & (nxt[:, 0, 0] < current.shape[1])
    good &= (nxt[:, 0, 1] >= 0) & (nxt[:, 0, 1] < current.shape[0])
    indices = np.flatnonzero(good)
    if len(indices) < 4:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        points[good].reshape(-1, 2), nxt[good].reshape(-1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000,
        confidence=.99, refineIters=10)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None
    indices = indices[inliers.ravel() != 0]
    if len(indices) < 4:
        return None
    return nxt.reshape(-1, 2), indices, matrix

def adaptive_fb_limit(values, base=1.0, absolute_cap=6.0, multiplier=3.0):
    """Return a robust forward/backward threshold based on median + MAD."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float(base)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return float(min(absolute_cap, max(base, median + multiplier * 1.4826 * mad)))


def checked_flow(previous, current, points, *, initial=None, levels=3, window=21,
                 max_fb=1.1, absolute_fb_cap=6.0):
    if len(points) < 4:
        return None
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(window, window), maxLevel=levels,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    new, status, error = cv2.calcOpticalFlowPyrLK(
        previous, current, points,
        None if initial is None else np.asarray(initial, np.float32).reshape(-1, 1, 2),
        flags=0 if initial is None else cv2.OPTFLOW_USE_INITIAL_FLOW, **options)
    if new is None:
        return None
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, new, None, **options)
    if back is None:
        return None
    old = points.reshape(-1, 2)
    new = new.reshape(-1, 2)
    fb = np.linalg.norm(old - back.reshape(-1, 2), axis=1)
    good = (status.ravel() == 1) & (back_status.ravel() == 1)
    good &= np.isfinite(new).all(axis=1) & np.isfinite(fb) & (error.ravel() < 30)
    good &= (new[:, 0] >= 0) & (new[:, 0] < current.shape[1])
    good &= (new[:, 1] >= 0) & (new[:, 1] < current.shape[0])
    # A fixed threshold discards nearly every point under motion blur. Use the
    # coherent population to relax the limit, while retaining an absolute cap
    # so an occluder or independently moving reflection cannot drag the fit.
    limit = adaptive_fb_limit(fb[good], max_fb, absolute_fb_cap)
    good &= fb <= limit
    return new, good, fb


def fit_similarity(source, destination):
    if len(source) < 4:
        return None
    source = np.asarray(source, np.float32)
    destination = np.asarray(destination, np.float32)
    matrix, first = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=2000, confidence=0.995, refineIters=10)
    if matrix is None or first is None or not np.isfinite(matrix).all():
        return None
    first = first.ravel().astype(bool)
    if first.sum() < 4:
        return None
    residual = np.linalg.norm(transform_points(source, matrix) - destination, axis=1)
    threshold = float(np.clip(np.median(residual[first]) * 2.0, 0.75, 3.0))
    refined, second = cv2.estimateAffinePartial2D(
        source[first], destination[first], method=cv2.RANSAC,
        ransacReprojThreshold=threshold, maxIters=2000,
        confidence=0.995, refineIters=15)
    if refined is None or second is None or not np.isfinite(refined).all():
        return matrix, first
    inliers = np.zeros(len(source), dtype=bool)
    inliers[np.flatnonzero(first)[second.ravel().astype(bool)]] = True
    return (refined, inliers) if inliers.sum() >= 4 else (matrix, first)


def fit_geometry(source, destination, perspective=False):
    """Fit a similarity (2×3) or perspective (3×3) model to point pairs.

    Perspective mode fits a homography via RANSAC. The caller is responsible
    for independent geometry-validity checks (no degenerate planar config)
    before trusting the result; verification still goes through the immutable reference.
    """
    source = np.asarray(source, np.float32)
    destination = np.asarray(destination, np.float32)
    if perspective:
        if len(source) < 8:
            return None
        matrix, first = cv2.findHomography(source, destination, cv2.RANSAC, 3.,
                                           maxIters=2000, confidence=.995)
        if matrix is None or first is None or not np.isfinite(matrix).all():
            return None
        first = first.ravel().astype(bool)
        if first.sum() < 8:
            return None
        residual = np.linalg.norm(transform_points(source, matrix) - destination, axis=1)
        threshold = float(np.clip(np.median(residual[first]) * 2.0, 1.0, 3.0))
        refined, second = cv2.findHomography(source[first], destination[first], cv2.RANSAC,
                                             threshold, maxIters=2000, confidence=.995)
        if refined is None or second is None or not np.isfinite(refined).all():
            return matrix, first
        inliers = np.zeros(len(source), dtype=bool)
        inliers[np.flatnonzero(first)[second.ravel().astype(bool)]] = True
        return (refined, inliers) if inliers.sum() >= 8 else (matrix, first)
    return fit_similarity(source, destination)


def evaluate_motion_quality(source, destination, matrix, inliers, reference_size,
                            min_points=4):
    """Measure support, residual and spatial distribution without smoothing."""
    source = np.asarray(source, np.float64)
    destination = np.asarray(destination, np.float64)
    inliers = np.asarray(inliers, dtype=bool)
    total = len(source)
    count = int(inliers.sum())
    if not total or count == 0:
        return MotionQuality(count, total, 0., float("inf"), float("inf"), 0., 0., False)
    residual = np.linalg.norm(transform_points(source, matrix) - destination, axis=1)
    selected = residual[inliers]
    size = np.maximum(np.asarray(reference_size, np.float64), 1.)
    span = np.ptp(source[inliers], axis=0) / size
    spread = float(np.sqrt(max(0., span[0] * span[1])))
    ratio = count / total
    mean = float(selected.mean())
    median = float(np.median(selected))
    required = min(12, max(min_points, int(np.ceil(total * .35))))
    ok = (count >= required and ratio >= .45 and mean <= 2.5 and
          np.isfinite(spread) and spread >= .035)
    count_score = min(1., count / max(required, 1))
    error_score = max(0., 1. - mean / 3.)
    spread_score = min(1., spread / .35)
    confidence = float(np.clip(ratio * .45 + error_score * .30 +
                               spread_score * .15 + count_score * .10, 0., .99))
    return MotionQuality(count, total, ratio, mean, median, spread, confidence, ok)


def guard_similarity_scale(matrix, previous, center, quality, hard_limit=.10):
    """Freeze an implausible one-frame scale change while preserving its center."""
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    previous = np.asarray(previous, dtype=np.float64)
    if matrix.shape != (2, 3) or previous.shape != (2, 3):
        return matrix
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    old_scale = float(np.hypot(previous[0, 0], previous[1, 0]))
    if not np.isfinite(scale) or not np.isfinite(old_scale) or min(scale, old_scale) <= 1e-6:
        return None
    ratio = scale / old_scale
    if (1. - hard_limit <= ratio <= 1. + hard_limit and
            quality.inlier_ratio >= .5 and quality.reprojection_mean <= 3.):
        return matrix
    mapped_center = matrix[:, :2] @ center + matrix[:, 2]
    angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    c, s = np.cos(angle) * old_scale, np.sin(angle) * old_scale
    matrix[:, :2] = [[c, -s], [s, c]]
    matrix[:, 2] = mapped_center - matrix[:, :2] @ center
    return matrix


# 平滑等级 → LK 光流参数映射。等级越高窗口与金字塔层数越大，越抗噪、越假设运动平滑。
_SMOOTHNESS_PROFILES = {
    1: (15, 2),  # 极敏感：允许突变，易受噪声影响
    2: (19, 3),
    3: (21, 3),  # 默认中等
    4: (25, 4),
    5: (31, 4),  # 极平滑：稳健抗抖，不适合急速运动
}


def flow_profile(smoothness):
    """Return (window, levels) for the given smoothness level (1..5)."""
    return _SMOOTHNESS_PROFILES.get(int(smoothness), _SMOOTHNESS_PROFILES[3])


class ReferencePatch:
    def __init__(self, gray, box, shape="rectangle", perspective=False, planar_region=False,
                 polygon_points=(), motion_profile=False):
        x, y, w, h = box
        self.width, self.height = round(w), round(h)
        self.center = np.array([self.width / 2, self.height / 2])
        self.perspective = bool(perspective)
        self.planar_region = bool(planar_region)
        self.polygon = shape == "polygon"
        self._shape_masked = shape != "rectangle"
        self.matrix = np.array([[1., 0., x + w / 2 - self.center[0]],
                                [0., 1., y + h / 2 - self.center[1]]])
        if self.perspective:
            # 3×3 单应矩阵：左上 2×3 复用上面的位移，最后一行 [0,0,1]
            self.matrix = np.vstack([self.matrix, [0., 0., 1.]])
        self.image = self.warp(gray, self.matrix)
        self.mask = np.full_like(self.image, 255)
        if shape in {"circle", "ellipse"}:
            yy, xx = np.mgrid[:self.height, :self.width]
            self.mask = np.where(((xx - (self.width - 1) / 2) / (self.width / 2)) ** 2 +
                                 ((yy - (self.height - 1) / 2) / (self.height / 2)) ** 2 <= 1,
                                 255, 0).astype(np.uint8)
        elif self.polygon:
            from .regions import validate_polygon
            points = validate_polygon(polygon_points)
            contour = np.rint(np.asarray(points) * [self.width - 1, self.height - 1]).astype(np.int32)
            self.mask.fill(0)
            cv2.fillPoly(self.mask, [contour], 255)
            if np.count_nonzero(self.mask) < 16:
                raise ValueError("多边形区域太小，请选择有足够纹理的区域")
        # Excluded image pixels must not leak through LK windows, SIFT
        # descriptors, Gaussian blur or exposure normalization.
        self.image = self.masked_image(self.image)
        self.blurred_image = self.blur(self.image)
        self.points = (motion_features(self.image, self.mask) if motion_profile
                       else balanced_features(self.image, self.mask if shape != "rectangle" else None))
        if motion_profile and len(self.points) < 4:
            # Preserve the host's small-detail support when a six-pixel spacing
            # cannot supply enough independent correspondences.
            self.points = balanced_features(self.image, self.mask)
        normal = (self.points - self.center) / [self.width, self.height]
        self.core = (np.abs(normal) <= 0.27).all(axis=1)
        self.weights = np.where(self.core, 3., 1.)
        yy, xx = np.mgrid[:self.height, :self.width]
        distance = ((xx - self.center[0]) / self.width) ** 2 + ((yy - self.center[1]) / self.height) ** 2
        self.pixel_weights = np.exp(-distance * 10).astype(np.float32)
        self.pixel_weights *= self.mask > 0
        total = float(self.pixel_weights.sum())
        self.pixel_weights = self.pixel_weights / total if total > 0 else self.pixel_weights
        # Drift anchor: the immutable first-frame appearance. adapt() reads
        # this via the caller-supplied original_reference to reject updates
        # that drift too far from the user's original selection.
        self.original_image = self.image.copy()

    def masked_image(self, image):
        """Make excluded shape pixels inert before neighborhood operations.

        Rectangle keeps its legacy pixel path unchanged.
        """
        if not self._shape_masked:
            return image
        return np.where(self.mask > 0, image, 0).astype(image.dtype)

    def blur(self, image):
        if not self._shape_masked:
            return cv2.GaussianBlur(image, (0, 0), 1.0)
        # Normalized convolution uses included samples only, including near
        # concave edges. Fill excluded pixels with zero afterwards.
        mask = (self.mask > 0).astype(np.float32)
        values = cv2.GaussianBlur(image.astype(np.float32) * mask, (0, 0), 1.)
        weights = cv2.GaussianBlur(mask, (0, 0), 1.)
        return self.masked_image(values / np.maximum(weights, 1e-6))

    def warp(self, gray, matrix):
        if self.perspective:
            return cv2.warpPerspective(gray, matrix, (self.width, self.height),
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return cv2.warpAffine(gray, matrix, (self.width, self.height),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def correlation(self, aligned):
        a, b = self.image.astype(np.float32), aligned.astype(np.float32)
        weights = self.pixel_weights
        a -= float((a * weights).sum())
        b -= float((b * weights).sum())
        energy = float((a * a * weights).sum() * (b * b * weights).sum())
        if energy < 25:
            return -1.
        return float((a * b * weights).sum() / np.sqrt(energy))

    def regional_correlation(self, aligned, reference=None, blurred=False, region_mask=None):
        """Independent whole-region NCC, decoupled from point fitting.

        Equal-weight NCC across included pixels supplies independent evidence.
        Circle/ellipse masks exclude pixels outside the selected shape. Blur
        fallback smooths both images before comparison.
        """
        mask = self.mask > 0
        if region_mask is not None:
            mask &= region_mask
        if not mask.any():
            return -1.
        source = self.image if reference is None else reference
        target = aligned
        if blurred:
            source = self.blur(source)
            target = self.blur(target)
        a = source.astype(np.float32)[mask]
        b = target.astype(np.float32)[mask]
        if a.size < 16:
            return -1.
        a -= a.mean()
        b -= b.mean()
        energy = float(np.linalg.norm(a) * np.linalg.norm(b))
        if energy < 1.:
            return -1.
        return float(a @ b / energy)

    def regional_evidence(self, aligned, point_ids, blurred=False, original=False):
        """Check the full region; a distributed center may exclude moving background."""
        source = self.original_image if original else self.image
        score = self.regional_correlation(aligned, reference=source, blurred=blurred)
        if np.isfinite(score) and score >= REGIONAL_NCC_THRESHOLD:
            return score
        # A user box can include an independently moving background ring.
        # Require measured features in three center quadrants before using the
        # central region as evidence; a single matching corner cannot suffice.
        points = self.points[point_ids]
        inside = (np.abs((points - self.center) / [self.width, self.height]) <= .27).all(axis=1)
        central = points[inside]
        if len(central) < 4 or len(np.unique(central >= self.center, axis=0)) < 3:
            return -1.
        yy, xx = np.mgrid[:self.height, :self.width]
        region = ((np.abs(xx - self.center[0]) <= self.width * .27)
                  & (np.abs(yy - self.center[1]) <= self.height * .27))
        core = self.regional_correlation(aligned, reference=source, blurred=blurred, region_mask=region)
        return core if np.isfinite(core) and core >= .80 else -1.

    def adapt(self, gray, matrix, confidence, original_reference=None,
               min_confidence=.85, max_drift=.55, redundancy=.995):
        """Progressively learn appearance change with drift guards.

        Learning guards:
        1) Only consider adapting on high-confidence verified frames.
        2) Skip if the new appearance is too similar to the current one
           (redundancy) — no point refreshing.
        3) Reject updates that drift too far from the immutable original.
        4) Keep initial pixels internally; no deep copy of a reference object.
        Returns True if the reference was updated, False if skipped.
        """
        if not np.isfinite(confidence) or confidence < min_confidence:
            return False
        aligned = self.masked_image(self.warp(gray, matrix))
        # Redundancy gate: a near-identical update wastes effort and
        # amplifies subpixel noise instead of helping.
        current_corr = self.regional_correlation(aligned)
        if not np.isfinite(current_corr) or current_corr >= redundancy or current_corr < .85:
            return False
        # Check initial pixels, rather than the latest learned appearance.
        original = self.original_image if original_reference is None else original_reference.original_image
        original_corr = self.regional_correlation(aligned, reference=original)
        if not np.isfinite(original_corr) or original_corr < max_drift:
            return False
        # Update the working reference, but keep `original_template` and
        # `original_image` (if present) immutable as drift anchors.
        image = cv2.addWeighted(self.image, .8, aligned, .2, 0.)
        points = balanced_features(image, self.mask if self._shape_masked else None)
        if len(points) < (8 if self.perspective else 4) or float(image[self.mask > 0].std()) < 7:
            return False
        self.image = image
        self.blurred_image = self.blur(image)
        self.points = points
        normal = (self.points - self.center) / [self.width, self.height]
        self.core = (np.abs(normal) <= 0.27).all(axis=1)
        self.weights = np.where(self.core, 3., 1.)
        return True

    def correlation_blurred(self, aligned):
        """Match against a slightly softened reference for motion blur tolerance."""
        a = self.blurred_image.astype(np.float32)
        b = self.blur(aligned.astype(np.float32))
        weights = self.pixel_weights
        a -= float((a * weights).sum())
        b -= float((b * weights).sum())
        energy = float((a * a * weights).sum() * (b * b * weights).sum())
        if energy < 25:
            return -1.
        return float((a * b * weights).sum() / np.sqrt(energy))

    def best_correlation(self, aligned):
        sharp = self.correlation(aligned)
        if sharp >= .67:
            return sharp, False
        return self.correlation_blurred(aligned), True

    def verify(self, gray, matrix):
        """Reproject into the fixed reference, then remove registration error.

        A locally consistent LK fit can still follow background. The original
        appearance and original feature identities must independently agree.
        Returning a measured correction also removes accumulated subpixel drift.
        """
        aligned = self.masked_image(self.warp(gray, matrix))
        flowed = checked_flow(self.image, aligned, self.points, levels=2,
                               window=15, max_fb=0.9)
        if flowed is None:
            return None
        new, good, _ = flowed
        good &= np.linalg.norm(new - self.points, axis=1) < max(4, min(self.width, self.height) * .12)
        fit = fit_geometry(self.points[good], new[good], perspective=self.perspective)
        if fit is None:
            return None
        residual, inliers = fit
        good_indices = np.flatnonzero(good)[inliers]
        if len(good_indices) < 4:
            return None
        # Center evidence is mandatory when it existed at initialization.
        if self.core.sum() >= 4 and self.core[good_indices].sum() < 3:
            return None
        support = float(self.weights[good_indices].sum() / self.weights.sum())
        if support < .36:
            return None
        if self.perspective:
            corrected = matrix @ residual
        else:
            correction = np.vstack([residual, [0, 0, 1]])
            corrected = matrix @ correction
        aligned = self.warp(gray, corrected)
        correlation, blurred = self.best_correlation(aligned)
        if not np.isfinite(correlation) or correlation < .67:
            return None
        regional = self.regional_evidence(aligned, good_indices, blurred=blurred) if self.planar_region else 1.
        if not np.isfinite(regional) or regional < REGIONAL_NCC_THRESHOLD:
            return None
        # Strong correlation alone is insufficient: a partial occluder may
        # leave a few excellent corners but not enough independent evidence.
        confidence = min(.99, max(0., correlation) * min(1., support / .80))
        confidence = min(confidence, regional)
        if blurred:
            confidence = min(confidence, .78)
        return corrected, confidence, good_indices
