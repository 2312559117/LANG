"""Slower, fixed-reference registration for the optional precision mode.

This extends the existing GPL-3.0-only lockOn-derived tracker without changing
its normal-mode behavior. Every accepted location is measured in the current
frame: no trajectory filtering or extrapolation is reported as a match.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from .media import check_cancel
from .reference_tracking import checked_flow, fit_geometry, fit_similarity, transform_points
from .tracking import FeatureTracker, Observation
from .performance import TemplateMatcher
from .planar import REGIONAL_NCC_THRESHOLD, template_pose, shifted, project, valid_surface


def _normalise(image, mask=None):
    """Remove an additive/linear exposure change while retaining fine detail."""
    values = image.astype(np.float32)
    samples = values if mask is None else values[mask > 0]
    result = np.clip((values - float(samples.mean())) * (42. / max(8., float(samples.std())))
                     + 128., 0, 255).astype(np.uint8)
    return result if mask is None else np.where(mask > 0, result, 0).astype(np.uint8)


def _similarity(matrix):
    """Project a dense affine fit onto a rotation + uniform scale model."""
    if not np.isfinite(matrix).all():
        return None
    linear = np.asarray(matrix[:, :2], np.float64)
    u, singular, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) <= 0 or singular.min() < .05:
        return None
    # Strong shear/perspective is outside this tracker's model; accepting it
    # tends to deform a selected logo towards whatever background is nearby.
    if singular.max() / singular.min() > 1.16:
        return None
    result = np.asarray(matrix, np.float64).copy()
    result[:, :2] = rotation * singular.mean()
    return result


class PrecisionFeatureTracker(FeatureTracker):
    """Independent flow/descriptor/template hypotheses plus subpixel ECC.

    The immutable user selection is checked after every dense fit, including
    center features, forward/backward correspondence, original appearance and
    competing repeated motifs. Runtime grows with target/search size. Small
    or textureless details and strong perspective changes can still be lost.
    """

    _dense_region_tracker = True

    def __init__(self, gray: np.ndarray, box, cancel: threading.Event | None = None,
                 shape="rectangle", matcher=None, perspective=False, smoothness=3,
                 adaptive_reference=False, planar_region=False, polygon_points=()):
        self.cancel = cancel
        self._check()
        super().__init__(gray, box, shape=shape, perspective=perspective, smoothness=smoothness,
                         adaptive_reference=adaptive_reference, planar_region=planar_region,
                         polygon_points=polygon_points)
        self.matcher = matcher if matcher is not None else TemplateMatcher("cpu")
        self.fast_frames = self.full_frames = 0
        self.sift = cv2.SIFT_create(nfeatures=650, contrastThreshold=.018, edgeThreshold=12)
        self.descriptor_factor = 2. if min(self.original_size) < 90 else 1.
        self._reference_updated()
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

    def _reference_updated(self):
        """Keep dense registration and descriptors aligned with the feature bank."""
        self.normalized_reference = _normalise(
            self.reference.image, self.reference.mask if self.reference._shape_masked else None)
        self.verify_reference = cv2.resize(self.normalized_reference, None, fx=2, fy=2,
                                            interpolation=cv2.INTER_CUBIC)
        ref = cv2.resize(self.reference.image, None,
                         fx=self.descriptor_factor, fy=self.descriptor_factor,
                         interpolation=cv2.INTER_CUBIC)
        ref_mask = cv2.resize(self.reference.mask, ref.shape[::-1], interpolation=cv2.INTER_NEAREST)
        keypoints, self.descriptors = self.sift.detectAndCompute(ref, ref_mask)
        self.descriptor_points = np.asarray(
            [key.pt for key in keypoints], np.float32).reshape(-1, 2) / self.descriptor_factor

    def _check(self):
        if self.cancel is not None:
            check_cancel(self.cancel)

    def _search(self, gray):
        center = self.box[:2] + self.box[2:] / 2
        predicted = center + self.velocity * min(self.missing + 1, 4)
        size = self.box[2:]
        radius = np.maximum(85., size * min(2.6, 1.65 + .16 * self.missing))
        low = np.maximum(0, np.floor(predicted - radius - size / 2)).astype(int)
        high = np.minimum([self.width, self.height],
                          np.ceil(predicted + radius + size / 2)).astype(int)
        return gray[low[1]:high[1], low[0]:high[0]], low

    def _descriptor_candidates(self, search, origin):
        self._check()
        if self.descriptors is None or len(self.descriptors) < 4:
            return []
        enlarged = cv2.resize(search, None, fx=self.descriptor_factor,
                              fy=self.descriptor_factor, interpolation=cv2.INTER_CUBIC)
        keys, descriptors = self.sift.detectAndCompute(enlarged, None)
        self._check()
        if descriptors is None or len(descriptors) < 5:
            return []
        forward = self._matcher.knnMatch(self.descriptors, descriptors, k=2)
        backward = self._matcher.match(descriptors, self.descriptors)
        reverse = {match.queryIdx: match.trainIdx for match in backward}
        matches = [pair[0] for pair in forward if len(pair) == 2
                   and pair[0].distance < .72 * pair[1].distance
                   and reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
        if len(matches) < 4:
            return []
        source = np.asarray([self.descriptor_points[m.queryIdx] for m in matches])
        destination = np.asarray([keys[m.trainIdx].pt for m in matches])
        destination = destination / self.descriptor_factor + origin
        fitted = fit_geometry(source, destination, perspective=self.perspective)
        if fitted is None or fitted[1].sum() < 4:
            return []
        matrix, inliers = fitted
        # At least three distinct quadrants must support a descriptor recovery.
        quadrants = (source[inliers] >= self.reference.center).astype(int)
        if len(np.unique(quadrants, axis=0)) < 3:
            return []
        return [matrix]

    def _template_candidates(self, search, origin, broad):
        self.matcher.prepare(search)
        scale = float(np.hypot(self.matrix[0, 0], self.matrix[1, 0]))
        angle = float(np.arctan2(self.matrix[1, 0], self.matrix[0, 0]))
        factors = (.78, .90, 1., 1.12, 1.25) if broad else (.94, 1., 1.06)
        angles = (-24., -12., 0., 12., 24.) if broad else (-6., 0., 6.)
        corners = np.array([[0, 0], [self.reference.width, 0],
                            [0, self.reference.height], self.original_size])
        candidates = []
        for factor in factors:
            for delta in angles:
                self._check()
                theta = angle + np.deg2rad(delta)
                a, b = scale * factor * np.cos(theta), scale * factor * np.sin(theta)
                linear = np.array([[a, -b], [b, a]])
                transformed = corners @ linear.T
                low = np.floor(transformed.min(axis=0))
                high = np.ceil(transformed.max(axis=0))
                width, height = (high - low).astype(int)
                if not self.perspective and (min(width, height) < 10 or width >= search.shape[1] or height >= search.shape[0]):
                    continue
                render = np.column_stack((linear, -low))
                appearance = self.reference.image if self.adaptive_reference else self.original_template
                projective_pose = None
                if self.perspective:
                    posed = template_pose(self.matrix, self.original_size, factor, delta)
                    if posed is None:
                        continue
                    render, low, dimensions, projective_pose = posed
                    width, height = dimensions
                    if min(width, height) < 10 or width >= search.shape[1] or height >= search.shape[0]:
                        continue
                    template = cv2.warpPerspective(appearance, render, (int(width), int(height)))
                    mask = cv2.warpPerspective(self.reference.mask, render, (int(width), int(height)))
                else:
                    template = cv2.warpAffine(appearance, render, (width, height))
                    mask = cv2.warpAffine(self.reference.mask, render, (width, height))
                scores = self.matcher.scores(search, template, mask)
                scores = np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.)
                # Masked normalized matching can also return finite numerical
                # overflow on almost uniform pixels; these are not evidence.
                scores[(scores > 1.001) | (scores < -1.001)] = -1.
                for _ in range(2):
                    _, score, _, location = cv2.minMaxLoc(scores)
                    if score < .57:
                        break
                    left, top = location
                    matrix = np.column_stack((linear, origin + location - low))
                    if projective_pose is not None:
                        matrix = shifted(projective_pose, origin + location - low)
                        center = project([self.reference.center], matrix)[0]
                    else:
                        center = linear @ self.reference.center + matrix[:, 2]
                    candidates.append((float(score), matrix, center))
                    margin = np.maximum(4, self.original_size * scale * .42).astype(int)
                    scores[max(0, top - margin[1]):top + margin[1] + 1,
                           max(0, left - margin[0]):left + margin[0] + 1] = -1.
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _dense_refine(self, gray, matrix):
        """Coarse-to-fine ECC on the fixed patch, then constrain its geometry."""
        if not self._valid_box(self._box_from_matrix(matrix)):
            return None
        aligned = self.reference.warp(gray, matrix)
        ref = self.normalized_reference
        current = _normalise(aligned, self.reference.mask if self.reference._shape_masked else None)
        if self.perspective:
            residual = np.eye(3, 3, dtype=np.float32)
            motion = cv2.MOTION_HOMOGRAPHY
        else:
            residual = np.eye(2, 3, dtype=np.float32)
            motion = cv2.MOTION_AFFINE
        factors = (.5, 1.) if min(self.original_size) >= 44 else (1., 2.)
        last_factor = 1.
        for factor in factors:
            self._check()
            ratio = factor / last_factor
            residual[:2, 2] *= ratio
            if self.perspective:
                residual[2, :2] /= ratio
            last_factor = factor
            target = cv2.resize(ref, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
            image = cv2.resize(current, (target.shape[1], target.shape[0]),
                               interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(self.reference.mask, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
            border = max(2, round(2 * factor))
            mask[:border] = mask[-border:] = 0
            mask[:, :border] = mask[:, -border:] = 0
            try:
                _, residual = cv2.findTransformECC(
                    target, image, residual, motion,
                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 75, 1e-6),
                    mask, 1 if self.reference._shape_masked else 3)
            except cv2.error:
                return None
        self._check()
        residual[:2, 2] /= last_factor
        if self.perspective:
            residual[2, :2] *= last_factor
        if self.perspective:
            # 透视模式接受 ECC 单应矩阵输出，仅做位移合理性检查防止发散。
            corrected = matrix @ residual
            displacement = np.linalg.norm(transform_points(self.reference.points, residual)
                                          - self.reference.points, axis=1)
            if np.median(displacement) > max(8., min(self.original_size) * .23):
                return None
            return corrected if self._valid_box(self._box_from_matrix(corrected)) else None
        affine = residual.copy()
        residual = _similarity(residual)
        if residual is None:
            return None
        # Project around the selected center, not coordinate (0, 0), otherwise
        # removing a tiny affine shear itself introduces a false translation.
        residual[:, 2] += (affine[:, :2] - residual[:, :2]) @ self.reference.center
        displacement = np.linalg.norm(transform_points(self.reference.points, residual)
                                      - self.reference.points, axis=1)
        if np.median(displacement) > max(8., min(self.original_size) * .23):
            return None
        corrected = matrix @ np.vstack([residual, [0, 0, 1]])
        if not self._valid_box(self._box_from_matrix(corrected)):
            return None
        return corrected

    def _precision_verify(self, gray, matrix):
        """Photometric AND independent, bidirectional fixed-feature evidence."""
        self._check()
        aligned = self.reference.warp(gray, matrix)
        correlation, blurred = self.reference.best_correlation(aligned)
        if not np.isfinite(correlation) or correlation < .70:
            return None
        # Verify at twice patch resolution: local interpolation permits a
        # tighter subpixel correspondence test without throwing away tiny logos.
        original = self.verify_reference
        warped = cv2.resize(_normalise(aligned, self.reference.mask if self.reference._shape_masked else None), original.shape[::-1],
                            interpolation=cv2.INTER_CUBIC)
        flowed = checked_flow(original, warped, self.reference.points * 2,
                               levels=3, window=21, max_fb=1.1)
        if flowed is None:
            return None
        new, good, _ = flowed
        residual_error = np.linalg.norm(new / 2 - self.reference.points, axis=1)
        good &= residual_error < 1.65
        ids = np.flatnonzero(good)
        if len(ids) < 4:
            return None
        fitted = fit_geometry(self.reference.points[ids], new[ids] / 2,
                            perspective=self.perspective)
        if fitted is None:
            return None
        _, inliers = fitted
        ids = ids[inliers]
        if len(ids) < 4 or (self.reference.core.sum() >= 4 and self.reference.core[ids].sum() < 3):
            return None
        support = float(self.reference.weights[ids].sum() / self.reference.weights.sum())
        if support < .40:
            return None
        # A preserved outside rim cannot stand in for an occluded central mark.
        height, width = aligned.shape
        y0, y1 = round(height * .23), round(height * .77)
        x0, x1 = round(width * .23), round(width * .77)
        if self.reference._shape_masked:
            region = np.zeros_like(self.reference.mask, dtype=bool)
            region[y0:y1, x0:x1] = True
            # An intentionally concave/off-center polygon may exclude the
            # central rectangle. It must then prove the selected region,
            # rather than using the excluded center as evidence.
            if np.count_nonzero(region & (self.reference.mask > 0)) < 16:
                region = None
            core_correlation = self.reference.regional_correlation(aligned, region_mask=region)
        else:
            a = self.reference.image[y0:y1, x0:x1].astype(np.float32).ravel()
            b = aligned[y0:y1, x0:x1].astype(np.float32).ravel()
            a -= a.mean()
            b -= b.mean()
            energy = float(np.linalg.norm(a) * np.linalg.norm(b))
            core_correlation = float(a @ b / energy) if energy > 1. else -1.
        if core_correlation < .62:
            return None
        regional = self.reference.regional_evidence(aligned, ids, blurred=blurred) if self.planar_region else 1.
        if not np.isfinite(regional) or regional < REGIONAL_NCC_THRESHOLD:
            return None
        if self.adaptive_reference and self.reference.regional_evidence(aligned, ids, original=True) < .55:
            return None
        confidence = min(.99, correlation * min(1., support / .80))
        confidence = min(confidence, regional)
        if blurred:
            confidence = min(confidence, .78)
        # The dense optimization already determines the measured position;
        # this feature fit is an independent check, not a second noisy shift.
        return matrix, self._box_from_matrix(matrix), confidence, ids

    def step(self, gray: np.ndarray) -> Observation:
        self._check()
        self.age += 1
        candidates = []
        local = self._flow(gray)
        # Only skip alternative hypotheses after BOTH the fixed-reference flow
        # and dense subpixel registration pass strict independent checks. Audit
        # every eighth frame; uncertain/occluded frames always use full search.
        if local is not None and local[2] >= .86 and not self.missing and self.age % 8:
            refined = self._dense_refine(gray, local[0])
            verified = self._precision_verify(gray, refined) if refined is not None else None
            if verified is not None and verified[2] >= .86:
                self.fast_frames += 1
                return self._accept(gray, verified, "tracked")
        self.full_frames += 1
        if local is not None:
            candidates.append(local[0])
        search, origin = self._search(gray)
        candidates.extend(self._descriptor_candidates(search, origin))
        templates = self._template_candidates(search, origin, broad=local is None)
        # If continuity is unavailable, equally plausible repeated details must
        # remain lost even if one candidate would pass dense registration.
        ambiguous = False
        if local is None and templates:
            best_score, _, point = templates[0]
            ambiguous = any(
                np.linalg.norm((point - other) / np.maximum(self.box[2:], 1)) > .55
                and best_score - score < .065
                for score, _, other in templates[1:])
        if not ambiguous:
            # Several initializations reach different ECC basins; optimize the
            # best six, excluding duplicate poses from nearby template scales.
            candidates.extend(item[1] for item in templates[:6])
        else:
            candidates = []
        if self.perspective:
            # 候选可能混合 2×3（来自模板匹配）和 3×3，统一提升为 3×3 才能比较与精化。
            candidates = [np.vstack([c, [0., 0., 1.]]) if c.shape == (2, 3) else c
                           for c in candidates]
        verified = []
        optimized = []
        for candidate in candidates:
            self._check()
            if any(np.max(np.abs(candidate - existing)) < .05 for existing in optimized):
                continue
            optimized.append(candidate)
            refined = self._dense_refine(gray, candidate)
            if refined is None:
                continue
            result = self._precision_verify(gray, refined)
            if result is not None:
                verified.append(result)
        if not verified:
            self.missing += 1
            self.previous = gray
            self.points = None
            return self._observation(0., "lost")
        verified.sort(key=lambda result: result[2], reverse=True)
        winner = verified[0]
        center = winner[1][:2] + winner[1][2:] / 2
        if local is None:
            for other in verified[1:]:
                other_center = other[1][:2] + other[1][2:] / 2
                if (np.linalg.norm((center - other_center) / self.box[2:]) > .55
                        and winner[2] - other[2] < .08):
                    self.missing += 1
                    self.previous = gray
                    self.points = None
                    return self._observation(0., "lost")
        return self._accept(gray, winner, "tracked" if local is not None and not self.missing else "recovered")

    def _accept(self, gray, winner, state):
        matrix, box, confidence, ids = winner
        center = box[:2] + box[2:] / 2
        old_center = self.box[:2] + self.box[2:] / 2
        self.velocity = .5 * self.velocity + .5 * (center - old_center)
        self.matrix, self.box, self.previous = matrix, box, gray
        self._bind_points(ids, matrix, confidence)
        self.missing = 0
        self._maybe_adapt(gray, confidence, state)
        return self._observation(confidence, state)
