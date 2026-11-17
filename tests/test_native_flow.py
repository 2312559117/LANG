import threading
import time

import cv2
import numpy as np
import pytest

from lockframe.flow.frame import PixelFormat, YuvFrame, parse_format
from lockframe.flow.synthesis import FlowError, quantize, synthesize, _plane_flow
from lockframe.flow.base import create_interpolator
from lockframe.flow.nvof import availability


def frame(fmt="yuv420p", width=96, height=64, seed=41):
    fmt = parse_format(fmt)
    rng = np.random.default_rng(seed)
    planes = tuple((rng.integers(0, 1 << fmt.depth, shape, dtype=np.uint32) << fmt.shift).astype(fmt.dtype)
                   for shape in fmt.plane_shapes(width, height))
    return YuvFrame(planes, fmt)


@pytest.mark.parametrize("fmt", ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p12le",
                                  "yuv444p16le", "nv12", "nv21", "p010le", "p016le"])
def test_native_roundtrip_preserves_layout_depth_and_odd_chroma(fmt):
    original = frame(fmt, 17, 11)
    data = original.to_bytes()
    restored = YuvFrame.from_bytes(data, 17, 11, fmt)
    assert len(data) == original.format.frame_bytes(17, 11)
    assert restored.format == original.format
    assert restored.width == 17 and restored.height == 11
    for a, b in zip(original.planes, restored.planes):
        np.testing.assert_array_equal(a, b)
    assert restored.to_bytes() == data


def test_native_frames_reject_wrong_lengths_layout_dtype_and_depth():
    with pytest.raises(ValueError, match="字节数"):
        YuvFrame.from_bytes(b"short", 17, 11, "yuv420p10le")
    with pytest.raises(ValueError, match="不支持"):
        parse_format("rgb24")
    fmt = parse_format("yuv420p10le")
    valid = frame(fmt, 8, 8)
    wrong = list(valid.planes)
    wrong[0] = wrong[0].astype(np.float32)
    with pytest.raises(ValueError, match="形状或整数"):
        YuvFrame(tuple(wrong), fmt)
    wrong[0] = np.full((8, 8), 1024, np.uint16)
    with pytest.raises(ValueError, match="超出"):
        YuvFrame(tuple(wrong), fmt)
    with pytest.raises(ValueError, match="采样"):
        PixelFormat("invalid", 10, 3, 2)
    with pytest.raises(ValueError, match="采样"):
        PixelFormat("invalid", 8, True, 2)
    with pytest.raises(ValueError, match="格式"):
        parse_format({})


@pytest.mark.parametrize("fmt", ["yuv420p", "yuv422p10le", "yuv444p12le", "yuv420p16le", "p010le"])
def test_alpha_endpoints_are_exact_native_samples_with_no_flow_required(fmt):
    left, right = frame(fmt, 19, 13), frame(fmt, 19, 13, seed=67)
    for alpha, expected in ((0, left), (1, right)):
        result = synthesize(left, right, None, None, alpha)
        assert result.to_bytes() == expected.to_bytes()
        assert result.format == expected.format
        assert not np.shares_memory(result.planes[0], expected.planes[0])


def moving_square(fmt="yuv444p", offset=8):
    fmt = parse_format(fmt)
    planes = [np.zeros(shape, fmt.dtype) for shape in fmt.plane_shapes(64, 48)]
    level = min((1 << fmt.depth) - 1, 900 if fmt.depth > 8 else 220) << fmt.shift
    planes[0][16:32, 10:22] = level
    for plane in planes[1:]:
        plane.fill(((1 << fmt.depth) // 2) << fmt.shift)
    left = YuvFrame(tuple(planes), fmt)
    right = YuvFrame(tuple(cv2.warpAffine(p, np.array([[1., 0., offset], [0., 1., 0.]]),
                                         (p.shape[1], p.shape[0]), flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_REPLICATE) for p in planes), fmt)
    flow = np.zeros((48, 64, 2), np.float32)
    flow[..., 0] = offset
    return left, right, flow


@pytest.mark.parametrize("fmt", ["yuv444p", "yuv444p10le", "yuv444p16le"])
def test_known_bidirectional_flow_moves_detail_without_crossfade_ghost(fmt):
    left, right, flow = moving_square(fmt)
    middle = synthesize(left, right, flow, -flow, .5)
    level = int(left.planes[0].max())
    expected = np.zeros((48, 64), left.format.dtype)
    expected[16:32, 14:26] = level
    np.testing.assert_array_equal(middle.planes[0], expected)
    # Exchanging endpoints and flow directions is a true reverse traversal.
    np.testing.assert_array_equal(synthesize(right, left, -flow, flow, .25).planes[0],
                                  synthesize(left, right, flow, -flow, .75).planes[0])


def test_lower_resolution_flow_grid_uses_half_pixel_sampling_and_pixel_scale():
    grid = np.zeros((2, 2, 2), np.float32)
    grid[..., 0] = 1
    sampled = _plane_flow(grid, 8, 4, 0, 4)
    assert sampled.shape == (4, 8, 2)
    assert np.all(sampled[..., 0] == 4) and np.all(sampled[..., 1] == 0)
    grid[0, 0, 1], grid[0, 1, 1], grid[1, 0, 1], grid[1, 1, 1] = (0, 1, 2, 3)
    value = _plane_flow(grid, 2, 2, 0, 2)
    np.testing.assert_array_equal(value[..., 1], grid[..., 1])


def test_depth_quantization_clips_and_uses_round_even_with_p010_alignment():
    fmt = parse_format("yuv420p10le")
    np.testing.assert_array_equal(quantize(np.array([-2, .5, 1.5, 2.5, 1024, 90000]), fmt),
                                  [0, 0, 2, 2, 1023, 1023])
    shifted = parse_format("p010le")
    values = quantize(np.array([0, 64, 96, 1023 * 64, 65535], np.float32), shifted)
    np.testing.assert_array_equal(values, [0, 64, 128, 65472, 65472])
    assert np.all(values % 64 == 0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1e20])
def test_invalid_vectors_are_rejected_not_synthesized(invalid):
    left, right, flow = moving_square()
    flow[0, 0, 0] = invalid
    with pytest.raises(FlowError, match="无效"):
        synthesize(left, right, flow, flow, .5)


def test_mismatched_native_layout_is_an_error_even_if_fallback_was_allowed():
    with create_interpolator("dis", allow_fallback=True) as service:
        with pytest.raises(ValueError, match="一致"):
            service.prepare(frame("yuv420p"), frame("yuv422p"))
    with pytest.raises(ValueError):
        synthesize(frame(), frame(), np.zeros((4, 4, 2), np.float32),
                   np.zeros((4, 4, 2), np.float32), float("nan"))


def test_prepare_pair_is_reused_for_multiple_alphas_and_cache_is_bounded(monkeypatch):
    calls = []
    def zero(a, b):
        calls.append(1)
        return np.zeros((*a.shape, 2), np.float32)
    with create_interpolator("dis", max_pairs=2) as service:
        monkeypatch.setattr(service, "_flow", zero)
        left, right = frame(), frame(seed=51)
        pair = service.prepare(left, right, key=(0, 1))
        for alpha in (.2, .4, .6):
            pair.compose(alpha)
        assert len(calls) == 2
        assert service.prepare(left, right, key=(0, 1)) is pair
        assert service.stats["cache_hits"] == 1
        before = pair.compose(0).to_bytes()
        left.planes[0].fill(0)
        assert pair.compose(0).to_bytes() == before
        service.prepare(left, right, key=(1, 2));service.prepare(left, right, key=(2, 3))
        assert service.stats["cache_pairs"] == 2
        assert service.stats["cache_bytes"] <= service.max_cache_bytes
        service.prepare(left, right, key=(0, 1))
        assert len(calls) == 8
    assert service.stats["cache_pairs"] == 0


def test_zero_cache_budget_never_retains_a_pair(monkeypatch):
    with create_interpolator("dis", max_cache_bytes=0) as service:
        monkeypatch.setattr(service, "_flow", lambda a, b: np.zeros((*a.shape, 2), np.float32))
        left, right = frame(), frame(seed=52)
        service.prepare(left, right);service.prepare(left, right)
        assert service.stats["cache_pairs"] == 0 and service.stats["prepared_pairs"] == 2


def test_dis_failure_only_falls_back_after_explicit_opt_in(monkeypatch):
    def failure(*args):
        raise FlowError("injected invalid vectors")
    left, right = frame(), frame(seed=71)
    with create_interpolator("dis") as service:
        monkeypatch.setattr(service, "_flow", failure)
        with pytest.raises(FlowError, match="未允许退回"):
            service.prepare(left, right)
    with create_interpolator("dis", allow_fallback=True) as service:
        monkeypatch.setattr(service, "_flow", failure)
        pair = service.prepare(left, right)
        assert pair.backend_name == service.status["backend"] == "nearest"
        assert "用户配置" in service.status["fallback_reason"]
        assert pair.compose(.25).to_bytes() == left.to_bytes()
        assert pair.compose(.75).to_bytes() == right.to_bytes()


def test_auto_and_requested_nvof_never_claim_an_unimplemented_gpu_backend():
    assert not availability()["available"] and not availability()["implemented"]
    for backend in ("auto", "nvof"):
        with create_interpolator(backend) as service:
            assert service.backend_name == "dis" and service.status["requested_backend"] == backend
            assert not service.status["nvof_available"]
            assert "尚未实现并验证" in service.status["fallback_reason"]


def test_missing_dis_implementation_has_a_readable_failure(monkeypatch):
    monkeypatch.delattr(cv2, "DISOpticalFlow_create")
    with create_interpolator("dis") as service:
        with pytest.raises(FlowError, match="未提供DIS"):
            service.prepare(frame(), frame(seed=123))


def test_cancelled_prepare_and_tiled_compose_use_pipeline_cancel_contract(monkeypatch):
    from lockframe.media import Cancelled
    event = threading.Event();event.set()
    left, right, flow = moving_square()
    with create_interpolator("dis", cancel=event) as service:
        with pytest.raises(Cancelled):
            service.prepare(left, right)
    with pytest.raises(Cancelled):
        synthesize(left, right, flow, -flow, .5, cancel=event)
    event.clear()
    with create_interpolator("dis", cancel=event) as service:
        def slow(a, b):
            time.sleep(.08)
            return np.zeros((*a.shape, 2), np.float32)
        monkeypatch.setattr(service, "_flow", slow)
        timer = threading.Timer(.015, event.set);timer.start()
        try:
            with pytest.raises(Cancelled):
                service.prepare(left, right)
        finally:
            timer.join()


def test_actual_dis_midpoint_follows_textured_translation_without_depth_loss():
    rng = np.random.default_rng(19)
    y = cv2.GaussianBlur(rng.integers(0, 1024, (96, 128), dtype=np.uint16), (3, 3), .4)
    y[30:65, 30:60] = np.minimum(y[30:65, 30:60].astype(np.uint32) + 160, 1023).astype(np.uint16)
    u = np.full((48, 64), 513, np.uint16);v = np.full((48, 64), 417, np.uint16)
    left = YuvFrame((y, u, v), parse_format("yuv420p10le"))
    right = YuvFrame((cv2.warpAffine(y, np.array([[1., 0., 4.], [0., 1., 0.]]), (128, 96),
                                    borderMode=cv2.BORDER_REPLICATE), u.copy(), v.copy()), left.format)
    with create_interpolator("dis") as service:
        pair = service.prepare(left, right)
        middle = pair.compose(.5)
        expected = cv2.warpAffine(y, np.array([[1., 0., 2.], [0., 1., 0.]]), (128, 96),
                                  borderMode=cv2.BORDER_REPLICATE)
        error = np.abs(middle.planes[0].astype(float) - expected)
        assert np.mean(error[12:-12, 12:-12]) < 8
        assert middle.format.depth == 10 and middle.planes[0].dtype == np.uint16
        assert np.any(middle.planes[0] > 255)
        np.testing.assert_array_equal(middle.planes[1], u)
        np.testing.assert_array_equal(middle.planes[2], v)
