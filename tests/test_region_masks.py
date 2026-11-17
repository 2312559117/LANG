"""Selected-shape evidence, independent of pixels excluded by the user's ROI."""
import json
import threading

import cv2
import numpy as np
import pytest

from lockframe.models import VideoInfo, file_fingerprint
from lockframe.reference_tracking import ReferencePatch
from lockframe.tracking import FeatureTracker, track_video
from lockframe.precision_tracking import PrecisionFeatureTracker, _normalise
from lockframe.quality import confidence_grid


POLYGON = ((.08, .08), (.92, .08), (.92, .42), (.62, .42), (.62, .92), (.08, .92))
OTHER_POLYGON = ((.08, .08), (.92, .08), (.92, .6), (.62, .6), (.62, .92), (.08, .92))


def texture(seed=124, size=80):
    noise = np.random.default_rng(seed).normal(0, 1, (size, size)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (3, 3), .6)
    return np.clip(noise / noise.std() * 34 + 127, 8, 246).astype(np.uint8)


def selected_reference(shape="polygon"):
    pixels = texture()
    return ReferencePatch(pixels, (0, 0, 80, 80), shape,
                          polygon_points=POLYGON if shape == "polygon" else ())


@pytest.mark.parametrize("shape", ["polygon", "ellipse", "circle"])
def test_outside_pixels_cannot_change_features_blur_normalization_or_region_evidence(shape):
    pixels = texture()
    first = selected_reference(shape)
    changed = pixels.copy()
    changed[first.mask == 0] = 255 - changed[first.mask == 0]
    second = ReferencePatch(changed, (0, 0, 80, 80), shape,
                            polygon_points=POLYGON if shape == "polygon" else ())
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.image, second.image)
    np.testing.assert_array_equal(first.blurred_image, second.blurred_image)
    np.testing.assert_array_equal(_normalise(pixels, first.mask), _normalise(changed, first.mask))
    assert first.regional_correlation(changed) == pytest.approx(1., abs=1e-6)
    assert first.regional_correlation(changed, blurred=True) == pytest.approx(1., abs=1e-6)
    assert first.correlation_blurred(changed) == pytest.approx(1., abs=1e-6)
    assert not first.adapt(changed, first.matrix, .99)
    assert np.all(first.mask[np.rint(first.points[:, 1]).astype(int),
                             np.rint(first.points[:, 0]).astype(int)] > 0)


@pytest.mark.parametrize("shape", ["polygon", "ellipse"])
def test_masked_precision_descriptors_and_dense_refinement_ignore_excluded_pixels(shape):
    image = texture()
    reference = selected_reference(shape)
    changed = image.copy()
    changed[reference.mask == 0] = 255 - changed[reference.mask == 0]
    kwargs = dict(shape=shape, polygon_points=POLYGON if shape == "polygon" else ())
    # Keep the box away from the frame border so geometry checks can refine.
    frames = []
    for patch in (image, changed):
        frame = np.full((180, 220), 20, np.uint8)
        frame[50:130, 60:140] = patch
        frames.append(frame)
    first = PrecisionFeatureTracker(frames[0], (60, 50, 80, 80), **kwargs)
    second = PrecisionFeatureTracker(frames[1], (60, 50, 80, 80), **kwargs)
    np.testing.assert_array_equal(first.descriptors, second.descriptors)
    np.testing.assert_array_equal(first.normalized_reference, second.normalized_reference)
    np.testing.assert_array_equal(first._dense_refine(frames[0], first.matrix),
                                  first._dense_refine(frames[1], first.matrix))
    for frame in frames:
        result = first._precision_verify(frame, first.matrix)
        assert result is not None
        assert result[2] > .98


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
@pytest.mark.parametrize("shape", ["polygon", "ellipse"])
def test_excluded_texture_cannot_rescue_a_genuinely_lost_target(cls, shape):
    patch = texture()
    frame = np.full((180, 220), 24, np.uint8)
    frame[50:130, 60:140] = patch
    tracker = cls(frame, (60, 50, 80, 80), shape=shape,
                  polygon_points=POLYGON if shape == "polygon" else (), planar_region=True)
    missing = frame.copy()
    region = missing[50:130, 60:140]
    region[tracker.reference.mask > 0] = 110
    observation = tracker.step(missing)
    assert observation.state == "lost" and observation.confidence == 0.


@pytest.mark.parametrize("shape", ["polygon", "ellipse", "circle"])
def test_excluded_texture_cannot_initialize_an_empty_selection(shape):
    pixels = texture()
    reference = selected_reference(shape)
    pixels[reference.mask > 0] = 110
    with pytest.raises(ValueError, match="纹理太少"):
        FeatureTracker(pixels, (0, 0, 80, 80), shape=shape,
                       polygon_points=POLYGON if shape == "polygon" else ())


def test_concave_mask_grid_and_adaptation_use_only_included_texture():
    reference = selected_reference()
    assert reference.mask[65, 65] == 0 and reference.mask[65, 25] == 255
    image = texture()
    changed = cv2.addWeighted(image, .75, texture(89), .25, 0)
    changed[reference.mask == 0] = 255
    assert reference.adapt(changed, reference.matrix, .99)
    assert np.all(reference.image[reference.mask == 0] == 0)
    assert np.all(reference.original_image[reference.mask == 0] == 0)
    grid = confidence_grid(reference, image, reference.matrix)
    assert grid[3, 3] == -1.
    assert grid[1, 1] > .9


@pytest.mark.parametrize("cls", [FeatureTracker, PrecisionFeatureTracker])
@pytest.mark.parametrize("perspective", [False, True])
def test_polygon_measured_motion_survives_independent_excluded_background(cls, perspective):
    patch = texture()
    mask = selected_reference().mask > 0
    def frame(index):
        pixels = np.random.default_rng(700 + index).integers(8, 246, (180, 220), dtype=np.uint8)
        region = pixels[50:130, 60 + index:140 + index]
        region[mask] = patch[mask]
        return pixels
    tracker = cls(frame(0), (60, 50, 80, 80), shape="polygon", polygon_points=POLYGON,
                  perspective=perspective, planar_region=True)
    for index in range(1, 7):
        observed = tracker.step(frame(index))
        assert observed.state != "lost"
        center = observed.center if observed.center is not None else observed.box[:2] + observed.box[2:] / 2
        np.testing.assert_allclose(center, [100 + index, 90], atol=.3)


def test_rectangle_reference_and_default_normalization_keep_original_pixel_path():
    image = texture()
    reference = ReferencePatch(image, (0, 0, 80, 80))
    np.testing.assert_array_equal(reference.image, image)
    np.testing.assert_array_equal(reference.blurred_image, cv2.GaussianBlur(image, (0, 0), 1.))
    expected = image.astype(np.float32)
    expected = np.clip((expected - float(expected.mean())) *
                       (42. / max(8., float(expected.std()))) + 128., 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(_normalise(image), expected)


@pytest.fixture
def memory_sequence(tmp_path, monkeypatch):
    """Real tracker/pose/checkpoints, substituting only filesystem video decode."""
    from lockframe import media, performance
    patch = texture()
    frames = []
    for index in range(24):
        frame = np.full((180, 220), 24, np.uint8)
        frame[50:130, 60 + index:140 + index] = patch
        frames.append(frame)
    source = tmp_path / "原始源身份.bin"
    source.write_bytes(b"generated test source")
    info = VideoInfo(str(source), 220, 180, 24., 1., frames=24,
                     proxy=str(source), fingerprint=file_fingerprint(str(source)))
    class Cache:
        def __init__(self, path, options, cancel):
            self.peak_bytes = self.hits = 0
        def read(self, index):
            return frames[index]
        def iter_frames(self, indices):
            for index in indices:
                yield index, frames[index]
        def close(self):
            pass
    monkeypatch.setattr(performance, "GrayFrameCache", Cache)
    monkeypatch.setattr(media, "prepare_precision_video", lambda info, *args: info)
    monkeypatch.setenv("LOCKFRAME_CACHE_DIR", str(tmp_path / "cache"))
    return info, {0: (60 / 220, 50 / 180, 80 / 220, 80 / 180)}, tmp_path / "cache"


@pytest.mark.parametrize("mode", ["normal", "precision"])
def test_changed_polygon_invalidates_result_reuse_and_complete_checkpoint(memory_sequence, mode):
    info, anchors, cache = memory_sequence
    options = dict(mode=mode, shape="polygon", polygon_points=POLYGON,
                   planar_region=True, segmented_tracking=True, segment_frames=10)
    first = track_video(info, anchors, 0, 23, threading.Event(), **options)
    assert not first.lost_frames
    assert first.performance["polygon_points"] == [list(point) for point in POLYGON]
    json.dumps(first.performance)
    same = track_video(info, anchors, 0, 23, threading.Event(), **options)
    assert same.performance["resumed_frames"] == 24
    changed = track_video(info, anchors, 0, 23, threading.Event(), previous=first,
                          **dict(options, polygon_points=OTHER_POLYGON))
    assert changed.performance["resumed_frames"] == 0
    assert changed.performance["reused_frames"] == 0
    assert len(list((cache / "tracking-checkpoints").glob("*.npz"))) == 2
    from lockframe.sequence import read_sequence
    records = [read_sequence(path)[0] for path in (cache / "tracking-checkpoints").glob("*.locksequence")]
    assert {tuple(tuple(point) for point in record.settings.polygon_points) for record in records} == {
        POLYGON, OTHER_POLYGON}


@pytest.mark.parametrize("mode", ["normal", "precision"])
@pytest.mark.parametrize("shape", ["polygon", "ellipse"])
def test_shaped_manual_pause_resume_retains_measured_result(memory_sequence, mode, shape):
    from lockframe.tracking_pause import TrackingPaused
    info, anchors, _ = memory_sequence
    options = dict(mode=mode, shape=shape, polygon_points=POLYGON if shape == "polygon" else (), planar_region=True)
    expected = track_video(info, anchors, 0, 23, threading.Event(), **options)
    pause = threading.Event()
    def reached(value):
        if value["frame"] == 10:
            pause.set()
    with pytest.raises(TrackingPaused) as captured:
        track_video(info, anchors, 0, 23, threading.Event(), pause_event=pause,
                    on_frame=reached, **options)
    partial = captured.value
    assert partial.frame == 10
    pause.clear()
    resumed = track_video(info, anchors, 0, 23, threading.Event(), previous=partial.result,
                          resume=partial.resume, **options)
    assert not resumed.lost_frames
    np.testing.assert_array_equal(resumed.positions, expected.positions)
    np.testing.assert_array_equal(resumed.confidence, expected.confidence)
    changed = track_video(info, anchors, 0, 23, threading.Event(), previous=partial.result,
                          resume=partial.resume, **dict(options, shape="polygon", polygon_points=OTHER_POLYGON))
    assert changed.performance["reused_frames"] == 0


@pytest.mark.parametrize("mode", ["normal", "precision"])
def test_polygon_partial_checkpoint_restores_masked_reference_bank_exactly(memory_sequence, monkeypatch, mode):
    from lockframe import performance
    from lockframe.media import Cancelled
    info, anchors, _ = memory_sequence
    options = dict(mode=mode, shape="polygon", polygon_points=POLYGON,
                   planar_region=True, adaptive_reference=True, confidence_mesh=True)
    expected = track_video(info, anchors, 0, 23, threading.Event(), **options)
    original = performance.GrayFrameCache.iter_frames
    cancel = threading.Event()
    def interrupted(cache, indices):
        for index, image in original(cache, indices):
            if index == 18:
                cancel.set()
            yield index, image
    monkeypatch.setattr(performance.GrayFrameCache, "iter_frames", interrupted)
    with pytest.raises(Cancelled):
        track_video(info, anchors, 0, 23, cancel, segmented_tracking=True, segment_frames=10, **options)
    monkeypatch.setattr(performance.GrayFrameCache, "iter_frames", original)
    resumed = track_video(info, anchors, 0, 23, threading.Event(),
                          segmented_tracking=True, segment_frames=10, **options)
    assert 10 < resumed.performance["resumed_frames"] < 24
    assert resumed.states == expected.states
    np.testing.assert_array_equal(resumed.positions, expected.positions)
    np.testing.assert_array_equal(resumed.confidence_grids, expected.confidence_grids)


@pytest.mark.parametrize("points", [(), ((0., 0.), (1., 1.)),
                                     ((0., 0.), (1., 1.), (0., 1.), (1., 0.))])
def test_direct_tracking_rejects_invalid_polygon_before_video_decode(points):
    with pytest.raises(ValueError):
        track_video(None, {}, 0, 1, threading.Event(), shape="polygon", polygon_points=points)
