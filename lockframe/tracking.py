"""Multi-point lock-on tracking, adapted from mo3rfan/lockOn's LK workflow.

Reimplemented for Windows/Python on 2026-10-17. Unlike the upstream single-point
demo, checks forward/backward flow and immutable-reference registration, with
center-led robust similarity fits and ambiguity-rejecting template recovery.
See THIRD_PARTY_NOTICES.md and third_party/lockOn. GPL-3.0-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import subprocess
import threading
import time

import cv2
import numpy as np

from .media import FrameReader, check_cancel, find_tool, NO_WINDOW
from .models import TrackResult, VideoInfo, source_identity
from .reference_tracking import (
    TRACKING_REVISION,
    persistent_motion,
    ReferencePatch,
    checked_flow,
    evaluate_motion_quality,
    fit_geometry,
    fit_similarity,
    flow_profile,
    guard_similarity_scale,
    mstp_v2_crop,
    mstp_v2_features,
    mstp_v2_follow,
    transform_points,
)
from .logging import get_logger
from .planar import UNIT_CORNERS, as_homography, project, valid_surface, dense_refine, template_pose, shifted


log = get_logger("tracking")


@dataclass
class Observation:
    box: np.ndarray
    confidence: float
    state: str
    homography: np.ndarray | None = None
    center: np.ndarray | None = None
    grid: np.ndarray | None = None


def normalize_box(box, width: int, height: int):
    return np.asarray(box, dtype=np.float64) / np.array([width, height, width, height])


def _mstrack_source_compatible(reader, info, cancel):
    """Use raw indexing only where it agrees with the existing CFR timeline.

    VFR, rotated or non-square-pixel sources retain the normalized proxy clock.
    Unknown metadata also takes that compatibility path.
    """
    if not hasattr(reader, "reader"):
        return True  # injected test/third-party reader owns its frame contract
    capture = reader.reader.capture
    actual = (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if actual != (info.width, info.height, info.frames):
        return False
    command = [find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,sample_aspect_ratio",
               "-of", "json", info.path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               creationflags=NO_WINDOW)
    try:
        deadline = time.monotonic() + 15
        while True:
            check_cancel(cancel)
            if time.monotonic() >= deadline:
                return False
            try:
                output, _ = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            return False
        stream = json.loads(output)["streams"][0]
        average = float(Fraction(stream["avg_frame_rate"]))
        nominal = float(Fraction(stream["r_frame_rate"]))
        sar = stream.get("sample_aspect_ratio", "1:1")
        return (abs(average - nominal) < 1e-5 and abs(average - info.fps) < 1e-5
                and (stream["width"], stream["height"]) == (info.width, info.height)
                and sar in {"1:1", "0:1", "N/A"})
    except (ValueError, KeyError, IndexError, ZeroDivisionError, OSError):
        return False
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


class FeatureTracker:
    """Local motion cross-checked against an immutable, center-led reference.

    The original manual selection is the reference for the entire segment.
    A later manual keyframe creates a new tracker, so changing appearance is
    corrected explicitly instead of learning a drifting patch as a new target.
    """

    def __init__(self, gray: np.ndarray, box, shape="rectangle", perspective=False, smoothness=3,
                 adaptive_reference=False, planar_region=False, polygon_points=()):
        self.height, self.width = gray.shape
        self.box = np.asarray(box, dtype=np.float64).copy()
        self.perspective = bool(perspective)
        self.smoothness = int(smoothness)
        self.adaptive_reference = bool(adaptive_reference)
        self.planar_region = bool(planar_region)
        self.confidence_mesh_enabled = False
        self.adaptation_updates = 0
        self.replenishment_events = 0
        self.quality_rejections = 0
        self.scale_guards = 0
        x, y, w, h = self.box
        if min(w, h) < 12:
            raise ValueError("目标框太小，请框选至少 12 × 12 分析像素的清晰细节")
        self.reference = ReferencePatch(gray, self.box, shape, perspective=self.perspective,
                                         planar_region=self.planar_region, polygon_points=polygon_points,
                                         motion_profile=(not self.perspective
                                                         and not self.adaptive_reference
                                                         and shape != "polygon"))
        self.template = self.reference.image
        self.original_template = self.template.copy()
        self.original_size = np.array([self.reference.width, self.reference.height])
        samples = self.template[self.reference.mask > 0] if self.reference._shape_masked else self.template
        if (self.template.size == 0 or float(samples.std()) < 7 or
                len(self.reference.points) < 4):
            raise ValueError("这个区域纹理太少，请框选车标、文字、灯组边缘等清晰细节")
        self.matrix = self.reference.matrix.copy()
        self.previous = gray
        self.point_ids = np.arange(len(self.reference.points))
        self.points = transform_points(self.reference.points, self.matrix).astype(np.float32)
        self._pending_motion = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.missing = 0
        self.age = 0
        # Flow profile chosen from the user's smoothness setting.
        self._flow_window, self._flow_levels = flow_profile(self.smoothness)
        # The recovered MS client has a distinct ordinary-mode state machine:
        # it analyses at a 960-pixel long edge, keeps image-space points across
        # frames and applies affine motion to the ROI centre only.  Restrict it
        # to the plain local tracker so precision/planar/adaptive/polygon paths
        # retain their stronger, already validated contracts.
        self._mstrack_v2 = (type(self) is FeatureTracker and not self.perspective
                            and not self.planar_region and not self.adaptive_reference
                            and shape in {"rectangle", "circle", "ellipse"})
        if self._mstrack_v2:
            self._motion_scale = min(1., 960. / max(self.height, self.width))
            self._motion_previous = self._motion_gray(gray)
            self._motion_box = self.box * self._motion_scale
            self._motion_base_size = self._motion_box[2:].copy()
            self._motion_reference = mstp_v2_crop(self._motion_previous, self._motion_box)
            self._motion_adaptive = self._motion_reference.copy()
            self._motion_points = mstp_v2_features(self._motion_previous, self._motion_box)

    def _motion_gray(self, gray):
        if self._motion_scale >= 1.:
            return gray
        dimensions = (max(16, round(gray.shape[1] * self._motion_scale)),
                      max(16, round(gray.shape[0] * self._motion_scale)))
        return cv2.resize(gray, dimensions, interpolation=cv2.INTER_AREA)

    def _motion_pose(self, box):
        """Represent the v2 centre/scale result in the host affine format."""
        center = box[:2] + box[2:] / 2
        ratios = box[2:] / np.maximum(self.original_size, 1.)
        scale = float(np.mean(ratios))
        linear = np.array([[scale, 0.], [0., scale]], dtype=np.float64)
        return np.column_stack([linear, center - linear @ self.reference.center])

    def _step_mstrack_v2(self, gray):
        """Run the REA-recovered v2 ordinary tracker without fake success."""
        self.age += 1
        motion_gray = self._motion_gray(gray)
        box, points, confidence, state = mstp_v2_follow(
            self._motion_previous, motion_gray, self._motion_box,
            self._motion_points, self._motion_reference, self._motion_adaptive)
        if state != "held":
            before = mstp_v2_crop(self._motion_previous, self._motion_box)
            after = mstp_v2_crop(motion_gray, box)
            if after.shape != before.shape:
                after = cv2.resize(after, (before.shape[1], before.shape[0]))
            a = before.astype(np.float32).ravel()
            b = after.astype(np.float32).ravel()
            a -= a.mean()
            b -= b.mean()
            energy = float(np.linalg.norm(a) * np.linalg.norm(b))
            correlation = float(a @ b / energy) if energy >= 1. else -1.
            # A coherent LK fit can survive a texture replacement by following
            # the unchanged ROI boundary. Require independent appearance
            # evidence before accepting that fit as the selected target.
            if correlation < .45 and self._verified(
                    gray, self._motion_pose(np.asarray(box) / self._motion_scale)) is None:
                state, confidence = "held", 0.
                box, points = self._motion_box.copy(), None
                self.quality_rejections += 1
        self._motion_previous = motion_gray
        self._motion_box = np.asarray(box, dtype=np.float64)
        self._motion_points = points
        self.previous = gray
        if state == "held":
            # The standalone reconstruction used "held" to keep rendering.
            # Tomato's correction workflow must expose the missing measurement
            # as lost so tracking pauses instead of spending work on drift.
            self.missing += 1
            return self._observation(0., "lost")

        if confidence > .8 and self.age % 10 == 0:
            self._motion_adaptive = mstp_v2_crop(motion_gray, self._motion_box)
            self.adaptation_updates += 1
        old_center = self.box[:2] + self.box[2:] / 2
        box = self._motion_box / self._motion_scale
        box[2:] = np.minimum(box[2:], [self.width, self.height])
        box[:2] = np.clip(box[:2], [0, 0],
                          np.asarray([self.width, self.height]) - box[2:])
        new_center = box[:2] + box[2:] / 2
        self.velocity = .5 * self.velocity + .5 * (new_center - old_center)
        self.box = box
        self.matrix = self._motion_pose(box)
        self.missing = 0
        host_state = "recovered" if state == "reacquired" else "tracked"
        return self._observation(confidence, host_state)

    def _valid_box(self, box):
        x, y, w, h = box
        return (np.isfinite(box).all() and min(w, h) >= 8 and
                0 <= x <= self.width - w and 0 <= y <= self.height - h and
                .3 <= w / self.original_size[0] <= 4 and
                .3 <= h / self.original_size[1] <= 4)

    def _box_from_matrix(self, matrix):
        if self.perspective:
            if not valid_surface(matrix, self.original_size, (self.height, self.width)):
                return np.full(4, np.nan)
            # 透视下用四角变换得到外接矩形，保持 TrackResult 数据结构兼容。
            corners = np.array([[0, 0, 1], [self.reference.width, 0, 1],
                                 [0, self.reference.height, 1],
                                 [self.reference.width, self.reference.height, 1]])
            mapped = (matrix @ corners.T).T
            mapped = mapped[:, :2] / mapped[:, 2:3]
            x, y = mapped[:, 0].min(), mapped[:, 1].min()
            w = mapped[:, 0].max() - x
            h = mapped[:, 1].max() - y
            return np.r_[x, y, w, h]
        center = matrix[:, :2] @ self.reference.center + matrix[:, 2]
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        size = self.original_size * scale
        return np.r_[center - size / 2, size]

    def _verified(self, gray, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if self.perspective and matrix.shape == (2, 3):
            # 把线性变换候选提升为单应矩阵进入透视 verify。
            matrix = np.vstack([matrix, [0., 0., 1.]])
        if not self._valid_box(self._box_from_matrix(matrix)):
            return None
        if self.planar_region and not getattr(self, "_dense_region_tracker", False):
            matrix = dense_refine(self.reference, gray, matrix)
            if matrix is None or not self._valid_box(self._box_from_matrix(matrix)):
                return None
        result = self.reference.verify(gray, matrix)
        if result is None:
            return None
        corrected, confidence, point_ids = result
        if self.adaptive_reference:
            aligned = self.reference.warp(gray, corrected)
            if self.reference.regional_evidence(aligned, point_ids, original=True) < .55:
                return None
        box = self._box_from_matrix(corrected)
        if not self._valid_box(box):
            return None
        return corrected, box, confidence, point_ids

    def _verify_predicted(self, gray):
        """Bridge a single-frame dropout by checking the velocity guess.

        A brief blur or a motion jump can defeat one LK step even though the
        target is still exactly where its measured velocity predicted. Only
        accept that guessed pose after the normal fixed-reference verification,
        and only when it is close to the previous box, so a genuinely missing
        target still reports lost instead of being teleported.
        """
        if self.perspective:
            # 透视模式下，单帧速度预测的相似变换无法表达 3D 角度变化；
            # 跳过此 bridge，由 _recover 兜底重新匹配。
            return None
        linear = self.matrix[:, :2].copy()
        center = self.box[:2] + self.box[2:] / 2
        predicted = center + self.velocity
        matrix = np.column_stack([linear, predicted - linear @ self.reference.center])
        verified = self._verified(gray, matrix)
        if verified is None:
            return None
        _, box, _, _ = verified
        corrected_center = box[:2] + box[2:] / 2
        if np.linalg.norm(corrected_center - center) > max(32., 2.5 * min(self.original_size)):
            return None
        return verified

    def _flow(self, gray):
        if self.points is None or len(self.points) < 4 or self.missing:
            return None
        result = self._flow_once(gray, None)
        if result is None and np.any(self.velocity):
            # Fast motion can exceed the LK search window in a single frame;
            # seed the pyramid with the measured velocity before declaring a
            # dropout. The result still passes the immutable reference check.
            result = self._flow_once(gray, self.points + self.velocity)
        return result

    def _persistent_flow(self, gray, initial):
        """Try the transplanted incremental kernel before reference geometry.

        Keep the host's independent appearance/occlusion checks; the MS demo's
        held position is deliberately not accepted as a measured observation.
        """
        motion = persistent_motion(self.previous, gray, self.points, initial)
        if motion is None:
            return None
        new, selected, delta = motion
        ids = self.point_ids[selected]
        if self.reference.core.sum() >= 4 and self.reference.core[ids].sum() < 3:
            return None
        quality = evaluate_motion_quality(
            self.points, new, delta, np.isin(np.arange(len(new)), selected),
            self.original_size, min_points=4)
        if not quality.quality_ok:
            self.quality_rejections += 1
            return None
        candidate = delta @ as_homography(self.matrix)
        unguarded = candidate
        candidate = guard_similarity_scale(candidate, self.matrix, self.reference.center, quality)
        if candidate is None:
            return None
        if not np.allclose(candidate, unguarded, atol=1e-9):
            self.scale_guards += 1
        result = self._verified(gray, candidate)
        if (result is None and not self.planar_region and type(self) is FeatureTracker
                and len(selected) / len(self.points) >= .70 and not self.missing):
            # Continuous flow is measured against the immediately preceding
            # frame; a changing 3D appearance need not match the first image.
            # This never runs after a gap, during reacquisition, or in the
            # planar/precision/perspective paths with stronger contracts.
            old_patch = self.reference.masked_image(self.reference.warp(self.previous, self.matrix))
            aligned = self.reference.masked_image(self.reference.warp(gray, candidate))
            correlation = self.reference.regional_correlation(aligned, reference=old_patch)
            box = self._box_from_matrix(candidate)
            if self._valid_box(box) and correlation >= .90:
                result = candidate, box, min(.95, correlation, quality.confidence), ids
                log.debug("MS连续运动 相邻帧相关=%.4f 内点=%d/%d", correlation, len(selected), len(self.points))
        if result is None:
            return None
        self._pending_motion = (ids.copy(), new[selected].copy(), result[0].copy())
        return (*result[:2], min(result[2], quality.confidence), result[3])

    def _flow_once(self, gray, initial):
        if not self.perspective:
            result = self._persistent_flow(gray, initial)
            if result is not None:
                return result
        result = checked_flow(self.previous, gray, self.points,
                              levels=self._flow_levels, window=self._flow_window,
                              initial=initial)
        if result is None:
            return None
        new, good, _ = result
        ids = self.point_ids[good]
        if len(ids) < 4:
            return None
        source = self.reference.points[ids]
        destination = new[good]
        core = self.reference.core[ids]
        hypotheses = []
        # Center first: a richly textured background must not outvote the logo.
        for selected in (core, np.ones(len(ids), bool)):
            fit = fit_geometry(source[selected], destination[selected], perspective=self.perspective)
            if fit is None:
                continue
            matrix, fitted_inliers = fit
            errors = np.linalg.norm(transform_points(source, matrix) - destination, axis=1)
            inliers = errors < 1.8
            selected_indices = np.flatnonzero(selected)
            fitted_ids = selected_indices[fitted_inliers]
            fitted_mask = np.zeros(len(ids), dtype=bool)
            fitted_mask[fitted_ids] = True
            # RANSAC defines the initial consensus. Other points may join only
            # when their residual independently agrees with that model.
            inliers &= fitted_mask | (errors <= 1.15)
            if inliers.sum() < 4:
                continue
            quality = evaluate_motion_quality(
                source, destination, matrix, inliers, self.original_size,
                min_points=8 if self.perspective else 4)
            if not quality.quality_ok:
                self.quality_rejections += 1
                continue
            if self.reference.core.sum() >= 4 and core[inliers].sum() < 3:
                continue
            support = self.reference.weights[ids[inliers]].sum() / self.reference.weights.sum()
            if support < .34:
                continue
            if not self.perspective:
                guarded = guard_similarity_scale(
                    matrix, self.matrix, self.reference.center, quality)
                if guarded is None:
                    continue
                if not np.allclose(guarded, matrix, atol=1e-9):
                    self.scale_guards += 1
                matrix = guarded
            verified = self._verified(gray, matrix)
            if verified is not None:
                # Confidence is bounded by both local evidence and the fixed
                # reference, rather than receiving a high constant baseline.
                verified = (*verified[:2], min(verified[2], float(support / .8),
                                               quality.confidence), verified[3])
                hypotheses.append(verified)
        return max(hypotheses, key=lambda item: item[2]) if hypotheses else None

    def _recover(self, gray):
        """Reacquire the original selection; reject competing repeated motifs."""
        x, y, w, h = self.box
        center = self.box[:2] + self.box[2:] / 2
        predicted = center + self.velocity * min(self.missing + 1, 4)
        radius = min(3., 1. + self.missing * .14)
        rx, ry = max(45, w * radius), max(45, h * radius)
        left = max(0, int(predicted[0] - rx - w / 2))
        top = max(0, int(predicted[1] - ry - h / 2))
        right = min(self.width, int(predicted[0] + rx + w / 2))
        bottom = min(self.height, int(predicted[1] + ry + h / 2))
        search = gray[top:bottom, left:right]
        scale = np.hypot(self.matrix[0, 0], self.matrix[1, 0])
        angle = np.arctan2(self.matrix[1, 0], self.matrix[0, 0])
        candidates = []
        # A small bank covers scale/rotation changes without adding a model
        # download or accepting an unconstrained whole-frame best match.
        broad = self.missing >= 2
        factors = (.85, .92, 1., 1.08, 1.16) if broad else (.92, 1., 1.08)
        deltas = (-12., -6., 0., 6., 12.) if broad else (-8., 0., 8.)
        for factor in factors:
            for delta in deltas:
                theta = angle + np.deg2rad(delta)
                a, b = scale * factor * np.cos(theta), scale * factor * np.sin(theta)
                linear = np.array([[a, -b], [b, a]])
                corners = np.array([[0, 0], [self.reference.width, 0],
                                    [0, self.reference.height],
                                    [self.reference.width, self.reference.height]]) @ linear.T
                low, high = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
                tw, th = (high - low).astype(int)
                if not self.perspective and (min(tw, th) < 8 or search.shape[1] < tw + 1 or search.shape[0] < th + 1):
                    continue
                render = np.column_stack([linear, -low])
                appearance = self.reference.image if self.adaptive_reference else self.original_template
                projective_pose = None
                if self.perspective:
                    posed = template_pose(self.matrix, self.original_size, factor, delta)
                    if posed is None:
                        continue
                    render, low, dimensions, projective_pose = posed
                    tw, th = dimensions
                    if min(tw, th) < 8 or tw >= search.shape[1] or th >= search.shape[0]:
                        continue
                    template = cv2.warpPerspective(appearance, render, (int(tw), int(th)))
                    mask = cv2.warpPerspective(self.reference.mask, render, (int(tw), int(th)))
                else:
                    template = cv2.warpAffine(appearance, render, (tw, th))
                    mask = cv2.warpAffine(self.reference.mask, render, (tw, th))
                # Mask padding introduced by rotation. NaN/Inf can occur for
                # uniform search regions in OpenCV's masked normalized metric.
                scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED, mask=mask)
                scores = np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.)
                for _ in range(2):
                    _, score, _, loc = cv2.minMaxLoc(scores)
                    if score < .76:
                        break
                    lx, ly = loc
                    matrix = np.column_stack([linear, np.array([left + lx, top + ly]) - low])
                    if projective_pose is not None:
                        matrix = shifted(projective_pose, np.array([left + lx, top + ly]) - low)
                        point = project([self.reference.center], matrix)[0]
                    else:
                        point = matrix[:, :2] @ self.reference.center + matrix[:, 2]
                    candidates.append((float(score), matrix, point))
                    margin = np.maximum(4, self.original_size * scale * .35).astype(int)
                    scores[max(0, ly - margin[1]):ly + margin[1] + 1,
                           max(0, lx - margin[0]):lx + margin[0] + 1] = -1.
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        best_score, matrix, point = candidates[0]
        for score, _, other_point in candidates[1:]:
            separation = np.linalg.norm((point - other_point) / np.maximum([w, h], 1))
            if separation > .55 and best_score - score < .065:
                return None
        verified = self._verified(gray, matrix)
        if verified is None:
            return None
        return (*verified[:2], min(verified[2], best_score), verified[3])

    def step(self, gray: np.ndarray) -> Observation:
        if self._mstrack_v2:
            return self._step_mstrack_v2(gray)
        self.age += 1
        self._pending_motion = None
        result = self._flow(gray)
        state = "tracked"
        if result is None:
            result = self._verify_predicted(gray)
            if result is None:
                result = self._recover(gray)
            if result is not None:
                state = "recovered"
        if result is None:
            self.missing += 1
            self.points = None
            self.previous = gray
            return self._observation(0., "lost")
        matrix, box, confidence, point_ids = result
        old_center = self.box[:2] + self.box[2:] / 2
        new_center = box[:2] + box[2:] / 2
        # Velocity only narrows the next search. It never replaces a measured
        # position and never smooths an actual abrupt subject movement.
        self.velocity = .5 * self.velocity + .5 * (new_center - old_center)
        self.matrix = matrix
        self.box = box
        self.previous = gray
        self.missing = 0
        self._bind_points(point_ids, matrix, confidence)
        self._maybe_adapt(gray, confidence, state)
        return self._observation(confidence, state)

    def _bind_points(self, point_ids, matrix, confidence):
        """Replenish the active bank from immutable reference identities."""
        point_ids = np.unique(np.asarray(point_ids, dtype=np.int64))
        total = len(self.reference.points)
        target = min(total, max(8, int(np.ceil(total * .65))))
        if confidence >= .72 and len(point_ids) < target:
            missing = np.setdiff1d(np.arange(total), point_ids, assume_unique=True)
            # Reintroduce center-weighted features first. Their current-frame
            # positions come from the independently verified pose, not from a
            # learned or extrapolated template.
            order = np.argsort(-self.reference.weights[missing], kind="stable")
            missing = missing[order]
            projected = transform_points(self.reference.points[missing], matrix)
            inside = ((projected[:, 0] >= 0) & (projected[:, 0] < self.width) &
                      (projected[:, 1] >= 0) & (projected[:, 1] < self.height))
            additions = missing[inside][:max(0, target - len(point_ids))]
            if len(additions):
                point_ids = np.r_[point_ids, additions]
                self.replenishment_events += 1
        self.point_ids = point_ids
        self.points = transform_points(self.reference.points[point_ids], matrix).astype(np.float32)
        pending = self._pending_motion
        if (pending is not None and type(self) is FeatureTracker
                and np.allclose(matrix, pending[2], atol=1e-8, rtol=0)):
            ids, measured, _ = pending
            lookup = {int(point): value for point, value in zip(ids, measured)}
            for index, point in enumerate(point_ids):
                if int(point) in lookup:
                    self.points[index] = lookup[int(point)]
        self._pending_motion = None

    def _observation(self, confidence, state):
        """Expose the full pose in unit-plane coordinates."""
        grid = None
        if self.confidence_mesh_enabled:
            from .quality import confidence_grid
            grid = confidence_grid(self.reference, self.previous, self.matrix, state == "lost")
        if not (self.planar_region or self.perspective or self.confidence_mesh_enabled):
            return Observation(self.box.copy(), float(confidence), state)
        homography = as_homography(self.matrix) @ np.diag([*self.original_size, 1.])
        center = project([[.5, .5]], homography)[0]
        return Observation(self.box.copy(), float(confidence), state, homography, center, grid)

    def _maybe_adapt(self, gray, confidence, state):
        """Learn only verified continuous frames, then rebind every feature ID."""
        if not self.adaptive_reference or state != "tracked" or self.age % 5:
            return
        if not self.reference.adapt(gray, self.matrix, float(confidence)):
            return
        self.adaptation_updates += 1
        self.template = self.reference.image
        self.point_ids = np.arange(len(self.reference.points))
        self.points = transform_points(self.reference.points, self.matrix).astype(np.float32)
        self._reference_updated()

    def _reference_updated(self):
        """Refresh derived registration caches in trackers that keep them."""

def track_video(info: VideoInfo, anchors: dict[int, tuple], start: int, end: int,
                cancel: threading.Event, progress=None, mode="normal", resources=None,
                shape="rectangle", perspective=False, smoothness=3,
                previous: TrackResult | None = None, adaptive_reference=False, planar_region=False,
                segmented_tracking=False, segment_frames=300, confidence_mesh=False,
                on_frame=None, pause_on_lost=False, pause_event=None, resume=None,
                polygon_points=(), tracking_sections=None, drift_review='off') -> TrackResult:
    """Track a range, optionally ending the worker at a correction boundary.

    Interactive callers enable ``pause_on_lost`` and catch TrackingPaused.
    Legacy callers retain full-range recovery/analysis by default. ``pause_event``
    is separate from cancellation so an intentional pause retains measured work.
    """
    from .performance import ComputeOptions, compute_scope
    options = resources or ComputeOptions()
    from .drift_review import POLICIES
    if not isinstance(drift_review,str) or drift_review not in POLICIES:raise ValueError('未知的漂移复核灵敏度')
    if shape not in {"rectangle", "circle", "ellipse", "polygon"}:
        raise ValueError("未知的目标形状")
    if shape == "polygon":
        from .regions import validate_polygon
        polygon_points = validate_polygon(polygon_points)
    else:
        polygon_points = ()
    if type(segment_frames) is not int or not 10 <= segment_frames <= 3600:
        raise ValueError("分段长度应为 10–3600 帧")
    with compute_scope(options):
        return _track_video(info, anchors, start, end, cancel, progress, mode, options,
                            shape, perspective, smoothness, previous, adaptive_reference, planar_region,
                            segmented_tracking, segment_frames, confidence_mesh, on_frame,
                            pause_on_lost, pause_event, resume, polygon_points, tracking_sections, drift_review)


def _track_video(info, anchors, start, end, cancel, progress, mode, options, shape,
                 perspective, smoothness, previous=None, adaptive_reference=False, planar_region=False,
                 segmented_tracking=False, segment_frames=300, confidence_mesh=False, on_frame=None,
                 pause_on_lost=False, pause_event=None, resume=None, polygon_points=(), tracking_sections=None, drift_review='off'):
    from .performance import GrayFrameCache, TemplateMatcher
    from .quality import phase_progress, review_track
    from .checkpoints import checkpoint_path, load_checkpoint, save_checkpoint, tracker_snapshot, restore_tracker, save_task_record
    from .tracking_pause import TrackingPaused, computed_mask, PauseAwareCancel
    phase_progress(progress, "分析", 0., "检查视频与选区")
    if mode not in {"normal", "precision"}:
        raise ValueError("未知的跟踪算法")
    check_cancel(cancel)
    if not 0 <= start <= end < info.frames:
        raise ValueError("请设置有效的处理区间")
    from .tracking_sections import normalize_sections, section_anchors, section_walks
    tracking_sections = normalize_sections(tracking_sections, start, end)
    groups = section_anchors(tracking_sections, anchors, start, end)
    active = [frame for section, seeds in groups for frame in seeds]
    preparation_paused = False
    preparation_fraction = 0.
    if mode == "precision":
        from .media import prepare_precision_video, Cancelled
        from .precision_tracking import PrecisionFeatureTracker
        preparation_cancel = PauseAwareCancel(cancel, pause_event) if pause_event is not None else cancel
        def report_preparation(value):
            nonlocal preparation_fraction
            preparation_fraction = max(0., min(.9, float(value) * .9))
            phase_progress(progress, "分析", preparation_fraction, "准备高精度分析画面")
        try:
            info = prepare_precision_video(
                info, preparation_cancel,
                report_preparation)
        except Cancelled:
            if cancel.is_set() or pause_event is None or not pause_event.is_set():
                raise
            preparation_paused = True
            # Preparation owns and cleans its partial FFmpeg file. Build only
            # manual observations using the ordinary proxy, then suspend below;
            # no precision matching runs while the pause remains requested.
    n = end - start + 1
    positions = np.zeros((n, 2), dtype=np.float64)
    boxes = np.zeros((n, 4), dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    states = ["lost"] * n
    homographies = np.zeros((n, 3, 3), dtype=np.float64) if (perspective or planar_region or confidence_mesh) else None
    grids = np.full((n, 4, 4), -1., np.float32) if confidence_mesh else None
    mstrack_v2 = (mode == "normal" and not perspective and not planar_region
                  and not adaptive_reference and shape in {"rectangle", "circle", "ellipse"})
    # The recovered engine samples the source frames directly, then performs
    # one INTER_AREA reduction to a 960-pixel long edge.  Re-resampling the
    # existing 1280-pixel H.264 proxy shifted the reference trajectory by
    # several pixels, so ordinary v2 uses the source while all other modes keep
    # their established proxy/precision inputs.
    source_frames = mstrack_v2 and not info.hdr
    try:
        reader = GrayFrameCache(info.path if source_frames else info.proxy, options, cancel)
    except (ValueError, cv2.error):
        if not source_frames:
            raise
        reader = GrayFrameCache(info.proxy, options, cancel)
        source_frames = False
    if source_frames:
        try:
            compatible = _mstrack_source_compatible(reader, info, cancel)
        except BaseException:
            reader.close()
            raise
        if not compatible:
            reader.close()
            reader = GrayFrameCache(info.proxy, options, cancel)
            source_frames = False
    reader.max_frames = segment_frames if segmented_tracking else None
    matcher = TemplateMatcher(options.acceleration) if mode == "precision" else None
    log.info("开始跟踪 模式=%s 形状=%s 帧=%d..%d 关键帧=%s 设备=%s",
             mode, shape, start, end, active, options.acceleration)
    started = time.perf_counter()
    fast_frames = full_frames = adaptation_updates = 0
    replenishment_events = quality_rejections = scale_guards = 0
    completed = 0
    last_update = 0.0
    preview_identity = source_identity(info) if on_frame is not None else None
    # Masked neighborhood/normalization evidence changed; do not reuse an old
    # ellipse/circle working bank. Rectangular results retain their revision.
    reference_revision = 5 if mstrack_v2 else (4 if shape != "rectangle" else 3)
    continuation_signature = {
        "source_identity": source_identity(info), "range": (start, end), "mode": mode,
        "anchors": {frame: tuple(anchors[frame]) for frame in active},
        "shape": shape, "perspective": perspective, "smoothness": smoothness,
        "polygon_points": polygon_points,
        "adaptive_reference": adaptive_reference, "planar_region": planar_region,
        "confidence_mesh": confidence_mesh, "reference_revision": reference_revision,
        "tracking_revision": TRACKING_REVISION,
    }
    if tracking_sections:
        continuation_signature["tracking_sections"] = tracking_sections
    from .drift_review import REVIEW_REVISION
    if drift_review!='off':
        continuation_signature.update(drift_review=drift_review,drift_review_revision=REVIEW_REVISION)
    if not isinstance(resume, dict) or resume.get("signature") != continuation_signature:
        resume = None

    def walk_interval(seed, destination, direction):
        if direction < 0:
            return list(range(destination, seed))
        return list(range(seed + 1, destination + 1))

    walk_list = section_walks(groups)

    from .regions import shape_data
    previous_footprint = None
    if previous is not None:
        try:
            previous_footprint = shape_data(previous.performance.get("shape", "rectangle"),
                                             previous.performance.get("polygon_points", ()))
        except (ValueError, TypeError):
            pass
    reusable = (previous is not None and previous.start == start and previous.end == end
                and previous.performance.get("source_identity", source_identity(info)) == source_identity(info)
                and previous.tracking_mode == mode and previous.count == n
                and previous_footprint == (shape, polygon_points)
                and previous.performance.get("perspective", False) == perspective
                and previous.performance.get("smoothness", 3) == smoothness
                and previous.performance.get("adaptive_reference", False) == adaptive_reference
                and previous.performance.get("planar_region", False) == planar_region
                and (homographies is None or previous.homographies is not None)
                and (grids is None or previous.confidence_grids is not None)
                and previous.performance.get("reference_revision") == reference_revision
                and previous.performance.get("tracking_revision") == TRACKING_REVISION)
    reusable = reusable and previous.performance.get("tracking_sections", []) == tracking_sections
    reusable = reusable and previous.performance.get('drift_review','off')==drift_review
    if drift_review!='off':
        reusable = reusable and previous.performance.get('drift_review_revision')==REVIEW_REVISION
    if not reusable:
        resume = None
    recompute = np.zeros(n, dtype=bool)
    if not reusable:
        recompute[:] = True
    else:
        previous_anchors = previous.anchors or {}
        # Each walk starts from an independent manual reference. Compare its
        # boundaries AND reference box, so adding/removing/moving an anchor
        # invalidates only dependent walks, including changed walk boundaries.
        try:
            old_walks = set(section_walks(section_anchors(tracking_sections, previous_anchors, start, end)))
        except ValueError:
            # Old or externally constructed results may lack anchor metadata.
            # Their numeric poses remain valid, but cannot prove a dependency.
            old_walks = set()
        previous_computed = computed_mask(previous)
        partial_previous = previous.performance.get("complete", True) is False
        for frame_index in active:
            recompute[frame_index - start] = True
        for seed, destination, direction in walk_list:
            interval = walk_interval(seed, destination, direction)
            if not interval:
                continue
            if partial_previous:
                # A partial walk can be shortened by a new manual correction.
                # Its already-measured prefix still depends on the same seed;
                # the new anchor's suffix depends on the new reference instead.
                old_walk = next((walk for walk in old_walks
                                 if walk[0] == seed and walk[2] == direction), None)
                same_reference = (old_walk is not None
                                  and tuple(previous_anchors.get(seed, ())) == tuple(anchors[seed]))
                old_interval = set(walk_interval(*old_walk)) if old_walk is not None else set()
                for frame in interval:
                    index = frame - start
                    recompute[index] = (not same_reference or frame not in old_interval
                                        or not previous_computed[index]
                                        or previous.states[index] == "lost")
                continue
            needs = any(previous.states[frame - start] == "lost" for frame in interval)
            if not needs:
                needs = ((seed, destination, direction) not in old_walks
                         or tuple(previous_anchors.get(seed, ())) != tuple(anchors[seed]))
            if needs:
                for frame in interval:
                    recompute[frame - start] = True
    if reusable:
        for index in range(n):
            if not recompute[index]:
                positions[index] = previous.positions[index]
                boxes[index] = previous.boxes[index]
                confidence[index] = previous.confidence[index]
                states[index] = previous.states[index]
                if homographies is not None:
                    homographies[index] = previous.homographies[index]
                if grids is not None:
                    grids[index] = previous.confidence_grids[index]
        completed = int(np.count_nonzero(~recompute))
    computed = ~recompute.copy()
    from .drift_review import review_ranges,retained_ranges,merge_range
    drift_ranges=[];early_ranges=[]
    if reusable:
        drift_ranges=retained_ranges(previous.performance.get('suspected_drift_ranges',[]),~recompute,start,end)
        early_ranges=retained_ranges(previous.performance.get('early_warning_ranges',[]),~recompute,start,end)
    review_runtime=None;review_state_restores=review_state_rebuilds=0
    resume_data = None
    saved_chunks = resumed_frames = 0
    checkpoint_file = key = None
    if segmented_tracking:
        from .models import file_fingerprint
        if file_fingerprint(info.path) != info.fingerprint:
            raise ValueError("原视频已变化，不能使用续跑缓存")
        checkpoint_file, key = checkpoint_path(info, anchors, start, end, {
            "mode": mode, "shape": shape, "perspective": perspective, "smoothness": smoothness,
            "polygon_points": polygon_points,
            "adaptive_reference": adaptive_reference, "planar_region": planar_region,
            "confidence_mesh": confidence_mesh, "reference_revision": reference_revision,
            "tracking_revision": TRACKING_REVISION,
            "opencv": cv2.__version__, "acceleration": options.acceleration,
            **({"tracking_sections": tracking_sections} if tracking_sections else {}),
            **({'drift_review':drift_review,'drift_review_revision':REVIEW_REVISION} if drift_review!='off' else {})})
        from .models import Settings
        save_task_record(checkpoint_file, info, anchors, start, end,
                         Settings(tracking_mode=mode, target_shape=shape, perspective=perspective,
                                  polygon_points=polygon_points,
                                  smoothness=smoothness, adaptive_reference=adaptive_reference,
                                  planar_region=planar_region, confidence_mesh=confidence_mesh,
                                  segmented_tracking=True, segment_frames=segment_frames,
                                  acceleration=options.acceleration, memory_percent=options.memory_percent,
                                  compute_threads=options.threads,drift_review=drift_review),
                         **({"tracking_sections": tracking_sections} if tracking_sections else {}))

    def persist(walk_index, last_frame=None, tracker=None, pause=None):
        nonlocal saved_chunks
        if checkpoint_file is None:
            return
        arrays = {"positions": positions, "boxes": boxes, "confidence": confidence,
                  "states": np.asarray(states, dtype="U10"), "computed": computed}
        if homographies is not None:
            arrays["homographies"] = homographies
        if grids is not None:
            arrays["grids"] = grids
        tracker_meta = None
        if tracker is not None:
            values, tracker_meta = tracker_snapshot(tracker)
            arrays.update(values)
        save_checkpoint(checkpoint_file, key, arrays, {
            "walk": len(walk_list) if computed.all() and tracker is None else walk_index,
            "last_frame": last_frame, "tracker": tracker_meta,
            "fast_frames": fast_frames, "full_frames": full_frames,
            "adaptation_updates": adaptation_updates, "complete": bool(computed.all()),
            "replenishment_events": replenishment_events,
            "quality_rejections": quality_rejections, "scale_guards": scale_guards,
            "pause": pause,"suspected_drift_ranges":drift_ranges,"early_warning_ranges":early_ranges,
            **({'review_monitor':review_runtime[1].snapshot()} if drift_review!='off' and tracker is not None
                and review_runtime is not None and review_runtime[0]==walk_index else {})})
        saved_chunks += 1

    def build_result(complete, current_tracker=None):
        # Unknown slots remain explicitly unmeasured. Finite fallback geometry
        # keeps the existing project format valid without pretending success.
        result_positions, result_boxes = positions.copy(), boxes.copy()
        result_homographies = None if homographies is None else homographies.copy()
        for index in np.flatnonzero(~computed):
            frame = index + start
            seeds = next(seeds for section, seeds in groups if section["start"] <= frame <= section["end"])
            nearest = min(seeds, key=lambda seed: abs(seed - frame))
            box = np.asarray(anchors[nearest], dtype=np.float64)
            result_boxes[index] = box
            result_positions[index] = box[:2] + box[2:] / 2
            if result_homographies is not None:
                x, y, w, h = box
                result_homographies[index] = [[w, 0., x], [0., h, y], [0., 0., 1.]]
        result = TrackResult(start, end, result_positions, result_boxes, confidence.copy(), list(states),
                             mode, (width, height),
                             anchors={frame: tuple(anchors[frame]) for frame in active},
                             homographies=result_homographies,
                             confidence_grids=None if grids is None else grids.copy())
        elapsed = time.perf_counter() - started
        result.performance = {"seconds": elapsed, "fps": completed / max(elapsed, .001),
                              "source_identity": source_identity(info), "shape": shape,
                              "polygon_points": [list(point) for point in polygon_points],
                              "perspective": perspective, "smoothness": smoothness,
                              "adaptive_reference": adaptive_reference, "planar_region": planar_region,
                              "reference_revision": reference_revision,
                              "tracking_revision": TRACKING_REVISION,
                              "adaptation_updates": adaptation_updates + getattr(current_tracker, "adaptation_updates", 0),
                              "replenishment_events": replenishment_events + getattr(current_tracker, "replenishment_events", 0),
                              "quality_rejections": quality_rejections + getattr(current_tracker, "quality_rejections", 0),
                              "scale_guards": scale_guards + getattr(current_tracker, "scale_guards", 0),
                              "gpu_calls": matcher.gpu_calls if matcher else 0,
                              "cpu_match_calls": matcher.cpu_calls if matcher else 0,
                              "device": matcher.device if matcher else "CPU",
                              "backend": (matcher.reason if matcher else
                                          ("CPU 光流 · MS v2 源帧" if source_frames else
                                           ("CPU 光流 · MS v2 CFR兼容" if mstrack_v2 else "CPU 光流"))),
                              "cache_peak_mb": reader.peak_bytes / 1024 ** 2,
                              "cache_hits": reader.hits, "threads": options.threads,
                              "fast_frames": fast_frames + getattr(current_tracker, "fast_frames", 0),
                              "full_frames": full_frames + getattr(current_tracker, "full_frames", 0),
                              "resumed_frames": resumed_frames, "checkpoint_writes": saved_chunks,
                              "reused_frames": int(np.count_nonzero(~recompute)),
                              "segmented_tracking": segmented_tracking, "segment_frames": segment_frames,
                              "confidence_mesh": confidence_mesh, "complete": bool(complete)}
        if tracking_sections:
            result.performance["tracking_sections"] = [dict(section) for section in tracking_sections]
        if drift_review!='off':
            result.performance['drift_review']=drift_review
            result.performance['drift_review_revision']=REVIEW_REVISION
            result.performance['suspected_drift_ranges']=[dict(item) for item in drift_ranges]
            result.performance['early_warning_ranges']=[dict(item) for item in early_ranges]
            result.performance.update(review_state_restores=review_state_restores,review_state_rebuilds=review_state_rebuilds)
        if not complete:
            result.performance["computed_frames"] = computed.tolist()
        return result

    def suspend(frame_index, direction, walk_index, reason="manual", tracker=None, last_frame=None,
                stage="计算", fraction=None):
        phase_fraction = completed / n if fraction is None else float(fraction)
        metadata = {"frame": int(frame_index), "direction": int(direction), "walk": int(walk_index),
                    "reason": reason, "done_count": int(computed.sum()), "total": n,
                    "stage": stage, "phase_fraction": phase_fraction}
        persist(walk_index, last_frame, tracker, metadata)
        continuation = None
        if tracker is not None:
            arrays, tracker_meta = tracker_snapshot(tracker)
            continuation = {"signature": continuation_signature, "walk": walk_list[walk_index],
                            "last_frame": int(last_frame), "arrays": arrays, "tracker": tracker_meta}
            if drift_review!='off' and review_runtime is not None and review_runtime[0]==walk_index:
                continuation['review_monitor']=review_runtime[1].snapshot()
        result = build_result(False, tracker)
        result.performance["tracking_pause"] = metadata
        result.validate(info.frames)
        if progress:
            label = "跟踪丢失，等待手动修正" if reason == "lost" else "疑似漂移，等待人工复核" if reason=='review' else "已暂停，保留已测轨迹"
            phase_progress(progress, stage, phase_fraction, f"第 {frame_index + 1} 帧 · {label}")
        log.info("跟踪暂停 原因=%s 帧=%d 方向=%d 已测量=%d/%d", reason, frame_index, direction, completed, n)
        raise TrackingPaused(result, frame_index, direction, reason, continuation, metadata)

    def store(frame_index, box, score, state, homography=None, measured_center=None, grid=None,*,notify=True):
        nonlocal completed
        normalized = normalize_box(box, width, height)
        index = frame_index - start
        boxes[index] = normalized
        positions[index] = normalized[:2] + normalized[2:] / 2
        if measured_center is not None:
            positions[index] = measured_center / [width, height]
        if homographies is not None:
            if homography is None:
                x, y, w, h = normalized
                homographies[index] = [[w, 0., x], [0., h, y], [0., 0., 1.]]
            else:
                homographies[index] = np.diag([1 / width, 1 / height, 1.]) @ homography
        confidence[index] = score
        states[index] = state
        if grids is not None and grid is not None:
            grids[index] = grid
        if not computed[index]:
            completed += 1
        computed[index] = True
        if notify:publish_frame(frame_index,score,state)

    def publish_frame(frame_index,score,state,warning=None):
        nonlocal last_update
        index=frame_index-start
        if on_frame is not None:
            # Only measured observations enter the preview channel. Copies keep
            # later tracker writes and UI edits independent; no decoded image is
            # sent through Qt or multiprocessing queues.
            on_frame({"frame": int(frame_index), "source_time": frame_index / info.fps,
                      "source_identity": dict(preview_identity),
                      "box": tuple(float(value) for value in boxes[index]),
                      "position": tuple(float(value) for value in positions[index]),
                      "confidence": float(score), "state": state,
                      "homography": None if homographies is None else homographies[index].tolist(),
                      "grid": None if grids is None else grids[index].tolist(),
                      "completed": int(completed), "total": n,
                      **({'drift_warning':dict(warning)} if warning is not None else {})})
        now = time.monotonic()
        if progress and not preparation_paused and (now - last_update > 0.08 or completed == n):
            elapsed = max(.001, time.perf_counter() - started)
            speed = completed / elapsed
            remaining = max(0, n - completed) / speed
            label = {"manual": "关键帧", "tracked": "跟踪", "recovered": "重新匹配", "lost": "待修正"}.get(state, state)
            if warning is not None:label='跟偏预警 · '+warning['reason']
            phase_progress(progress, "计算", completed / n,
                           f"第 {frame_index + 1} 帧 · {label} · {speed:.1f} 帧/秒 · 约剩 {remaining:.0f} 秒")
            last_update = now

    def walk(seed: int, destination: int, direction: int, walk_index):
        nonlocal fast_frames, full_frames, adaptation_updates
        nonlocal replenishment_events, quality_rejections, scale_guards
        nonlocal review_runtime,review_state_restores,review_state_rebuilds
        check_cancel(cancel)
        if len(anchors[seed]) != 4:
            raise ValueError("关键帧格式无效，应为 (x, y, w, h) 四元组")
        gray = reader.read(seed)
        box = np.array(anchors[seed]) * np.array([width, height, width, height])
        if mode == "precision":
            tracker = PrecisionFeatureTracker(gray, box, cancel=cancel, shape=shape,
                                               matcher=matcher, perspective=perspective,
                                               smoothness=smoothness, adaptive_reference=adaptive_reference,
                                               planar_region=planar_region, polygon_points=polygon_points)
        else:
            tracker = FeatureTracker(gray, box, shape=shape, perspective=perspective,
                                     smoothness=smoothness, adaptive_reference=adaptive_reference,
                                     planar_region=planar_region, polygon_points=polygon_points)
        tracker.confidence_mesh_enabled = confidence_mesh
        first_frame = seed + direction
        last_frame = seed
        from .drift_review import DriftReview
        monitor=DriftReview(drift_review)
        continuation=None
        if resume is not None and resume.get("walk") == (seed, destination, direction):
            continuation=(resume['arrays'],resume['tracker'],resume['last_frame'],resume.get('review_monitor'))
        elif resume_data is not None and walk_index == resume_data[1]["walk"] and resume_data[1].get("tracker"):
            continuation=(resume_data[0],resume_data[1]['tracker'],resume_data[1]['last_frame'],resume_data[1].get('review_monitor'))
        if continuation is not None:
            values,tracker_meta,candidate,diagnostic=continuation
            if drift_review=='off' or monitor.restore(diagnostic,start=min(seed,destination),end=max(seed,destination),frame=candidate,direction=direction):
                last_frame=int(candidate)
                restore_tracker(tracker,values,tracker_meta,reader.read(last_frame))
                first_frame=last_frame+direction
                if drift_review!='off':review_state_restores+=1
            else:
                review_state_rebuilds+=1
                log.warning('预警续跑状态无效，将从当前关键帧重建预警与跟踪器，不跳过已测前缀 帧=%s',candidate)
        review_runtime=(walk_index,monitor)
        if pause_event is not None and pause_event.is_set():
            suspend(last_frame, direction, walk_index, tracker=tracker, last_frame=last_frame)
        frames = reader.iter_frames(range(first_frame, destination + direction, direction))
        chunk_count = 0
        active_warning=None
        if monitor.warning is not None:
            active_warning=next((item for item in reversed(early_ranges)
                if item['start']<=monitor.warning['end'] and monitor.warning['start']<=item['end']),None)
        try:
            for frame_index, frame in frames:
                check_cancel(cancel)
                if pause_event is not None and pause_event.is_set():
                    suspend(last_frame, direction, walk_index, tracker=tracker, last_frame=last_frame)
                observation = tracker.step(frame)
                last_frame = frame_index
                stored=False
                if (recompute[frame_index - start] or not computed[frame_index - start]
                        or (pause_on_lost and observation.state == "lost")):
                    store(frame_index, observation.box, observation.confidence, observation.state,
                          observation.homography, observation.center, observation.grid,notify=False)
                    stored=True
                check_cancel(cancel)
                suspicion=monitor.observe(frame_index,confidence[frame_index-start],states[frame_index-start],
                    positions[frame_index-start],boxes[frame_index-start])
                if monitor.new_warning is not None:
                    active_warning=merge_range(early_ranges,monitor.new_warning)
                    log.info('跟偏提前预警 帧=%d 模式=%s 原因=%s 当前分数=%.3f 基线=%s',
                        frame_index,drift_review,active_warning['reason'],active_warning['confidence'],active_warning.get('baseline_confidence'))
                    phase_progress(progress,'计算',completed/n,f'跟偏预警 · 第 {frame_index+1} 帧 · {active_warning["reason"]}；可暂停检查')
                if monitor.warning is not None and active_warning is not None:
                    first=min(active_warning['start'],monitor.warning['start'])
                    last=max(active_warning['end'],monitor.warning['end'])
                    active_warning.update(monitor.warning)
                    active_warning.update(start=first,end=last)
                if monitor.warning is None:active_warning=None
                if stored:publish_frame(frame_index,confidence[frame_index-start],states[frame_index-start],monitor.warning)
                if pause_on_lost and observation.state == "lost":
                    suspend(frame_index, direction, walk_index, "lost", tracker, last_frame)
                if suspicion is not None:
                    merge_range(drift_ranges,suspicion)
                    log.info('疑似漂移 待复核 帧=%d 模式=%s 证据=%s',frame_index,drift_review,suspicion['reason'])
                    if pause_on_lost:
                        suspend(frame_index,direction,walk_index,'review',tracker,last_frame)
                    phase_progress(progress,'计算',completed/n,f'疑似漂移 · 第 {frame_index+1} 帧 · {suspicion["reason"]}')
                if pause_event is not None and pause_event.is_set():
                    suspend(last_frame, direction, walk_index, tracker=tracker, last_frame=last_frame)
                chunk_count += 1
                if segmented_tracking and chunk_count % segment_frames == 0:
                    persist(walk_index, frame_index, tracker)
        finally:
            frames.close()
        fast_frames += getattr(tracker, "fast_frames", 0)
        full_frames += getattr(tracker, "full_frames", 0)
        adaptation_updates += tracker.adaptation_updates
        replenishment_events += tracker.replenishment_events
        quality_rejections += tracker.quality_rejections
        scale_guards += tracker.scale_guards
        persist(walk_index + 1)

    try:
        first = reader.read(active[0])
        height, width = first.shape[:2]
        if checkpoint_file is not None:
            resume_data = load_checkpoint(checkpoint_file, key, n)
            if resume_data is not None:
                values, meta = resume_data
                valid = 0 <= meta.get("walk", -1) <= len(walk_list)
                valid &= meta.get("walk") != len(walk_list) or bool(values["computed"].all())
                if meta.get("tracker") and valid:
                    seed, destination, direction = walk_list[meta["walk"]] if meta["walk"] < len(walk_list) else (0, 0, 0)
                    last_frame = meta.get("last_frame", -1)
                    valid &= direction != 0 and min(seed, destination) <= last_frame <= max(seed, destination)
                valid &= homographies is None or values.get("homographies", np.empty(0)).shape == homographies.shape
                valid &= grids is None or values.get("grids", np.empty(0)).shape == grids.shape
                if not valid:
                    resume_data = None
                else:
                    positions[:], boxes[:], confidence[:] = values["positions"], values["boxes"], values["confidence"]
                    states[:] = values["states"].tolist()
                    computed[:] = values["computed"]
                    if homographies is not None:
                        homographies[:] = values["homographies"]
                    if grids is not None:
                        grids[:] = values["grids"]
                    resumed_frames = completed = int(computed.sum())
                    fast_frames, full_frames = meta["fast_frames"], meta["full_frames"]
                    adaptation_updates = meta["adaptation_updates"]
                    replenishment_events = meta["replenishment_events"]
                    quality_rejections = meta["quality_rejections"]
                    scale_guards = meta["scale_guards"]
                    drift_ranges[:]=review_ranges(meta.get('suspected_drift_ranges',[]),start,end)
                    early_ranges[:]=review_ranges(meta.get('early_warning_ranges',[]),start,end)
        phase_progress(progress, "分析", preparation_fraction if preparation_paused else 1.,
                       "高精度分析画面准备已暂停" if preparation_paused
                       else f"选区就绪 · 续跑复用 {resumed_frames} 帧")
        for frame_index in active:
            box = np.array(anchors[frame_index], dtype=np.float64)
            x, y, w, h = box
            if not np.isfinite(box).all() or min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.001 or y + h > 1.001:
                raise ValueError("目标框超出画面，请重新框选")
            store(frame_index, box * np.array([width, height, width, height]), 1.0, "manual")
        previous_pause = previous.performance.get("tracking_pause", {}) if reusable else {}
        if (pause_on_lost and previous_pause.get("reason") in ("lost","review")
                and previous_pause.get("frame") not in active):
            # Clicking Continue without correcting the failed observation must
            # never silently advance to the next frame or retry the whole walk.
            failed_frame = int(previous_pause["frame"])
            index = failed_frame - start
            if 0 <= index < n:
                positions[index], boxes[index] = previous.positions[index], previous.boxes[index]
                confidence[index], states[index] = previous.confidence[index], previous.states[index]
                if homographies is not None:
                    homographies[index] = previous.homographies[index]
                if grids is not None:
                    grids[index] = previous.confidence_grids[index]
                if not computed[index]:
                    computed[index] = True
                    completed += 1
                suspend(failed_frame, previous_pause.get("direction", 1),
                        min(previous_pause.get("walk", 0), len(walk_list)), previous_pause['reason'])
        if pause_on_lost and resume_data is not None:
            cached_pause = resume_data[1].get("pause") or {}
            if cached_pause.get("reason") in ("lost","review"):
                suspend(cached_pause["frame"], cached_pause["direction"],
                        cached_pause["walk"], cached_pause['reason'])
        if pause_event is not None and pause_event.is_set():
            suspend(active[0], -1 if active[0] > start else 1, 0,
                    stage="分析" if preparation_paused else "计算",
                    fraction=preparation_fraction if preparation_paused else None)
        if segmented_tracking and resume_data is None:
            persist(0)
        for walk_index, (seed, destination, direction) in enumerate(walk_list):
            interval = walk_interval(seed, destination, direction)
            if resume_data is not None and walk_index < resume_data[1]["walk"]:
                continue
            if interval and (resume_data is not None or any(recompute[frame - start] for frame in interval)):
                walk(seed, destination, direction, walk_index)
        if not computed.all():
            raise ValueError("跟踪仍含未计算帧，不能提交为完整结果")
        phase_progress(progress, "计算", 1., "逐帧匹配完成")
        result = build_result(True)
        phase_progress(progress, "检查", 0., "检查低可信度、遮挡与可疑突变")
        if tracking_sections:
            from copy import copy
            quality = {"low_confidence": [], "motion_review": [], "lost_count": 0,
                       "raw_positions_preserved": True}
            for section in tracking_sections:
                part = copy(result)
                part.start, part.end = section["start"], section["end"]
                interval = slice(part.start - start, part.end - start + 1)
                part.positions, part.confidence, part.states = (result.positions[interval],
                                                               result.confidence[interval], result.states[interval])
                report = review_track(part)
                for name in ("low_confidence", "motion_review"):
                    quality[name].extend(report[name])
                quality["lost_count"] += report["lost_count"]
            result.performance["quality"] = quality
        else:
            result.performance["quality"] = review_track(result)
        result.validate(info.frames)
        phase_progress(progress, "检查", 1., "原始轨迹保留，质量报告已生成")
        phase_progress(progress, "缓存", 0., "保存分段检查点" if segmented_tracking else "整理跟踪结果")
        persist(len(walk_list))
        result.performance["checkpoint_writes"] = saved_chunks
        phase_progress(progress, "缓存", 1., "续跑缓存已保存" if segmented_tracking else "结果已就绪")
        lost = result.lost_frames
        log.info("跟踪完成 模式=%s 总帧=%d 有效=%d 跟丢=%d%s",
                 mode, n, n - len(lost), len(lost),
                 " 跟丢帧=" + str(lost) if lost else "")
        return result
    finally:
        reader.close()
