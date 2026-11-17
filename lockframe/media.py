from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import cv2

from .models import VideoInfo, file_fingerprint
from .logging import get_logger


log = get_logger("media")

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class Cancelled(Exception):
    pass


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", app_root()))


def find_tool(name: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [
        app_root() / "bin" / (name + suffix),
        resource_root() / "bin" / (name + suffix),
    ]
    configured = os.environ.get("LOCKFRAME_FFMPEG_DIR")
    if configured:
        candidates.insert(0, Path(configured) / (name + suffix))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    discovered = shutil.which(name)
    if discovered:
        return discovered
    raise RuntimeError(f"没有找到 {name}。请将 FFmpeg 和 FFprobe 放在软件的 bin 文件夹。")


def check_cancel(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled()


def probe(path: str, cancel: threading.Event | None = None) -> VideoInfo:
    executable = find_tool("ffprobe")
    if cancel is not None:check_cancel(cancel)
    process = subprocess.Popen(
        [executable, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    deadline = time.monotonic() + 30
    try:
        while True:
            if cancel is not None:check_cancel(cancel)
            if time.monotonic() >= deadline:
                raise TimeoutError("读取视频信息超时，请检查素材是否可访问或复制到本地后重试")
            try:
                output, errors = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:pass
        if cancel is not None:check_cancel(cancel)
        if process.returncode:
            raise ValueError("无法读取视频：" + errors.decode("utf-8", "replace")[-800:])
        data = json.loads(output)
    finally:
        if process.poll() is None:process.kill()
        process.communicate()
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video" and
                  not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ValueError("文件中没有可读取的视频画面")
    width, height = int(video["width"]), int(video["height"])
    sar = video.get("sample_aspect_ratio", "1:1").replace(":", "/")
    try:
        width = round(width * float(Fraction(sar)))
    except (ValueError, ZeroDivisionError):
        pass
    rotation = float(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    if abs(round(rotation / 90)) % 2:
        width, height = height, width
    try:
        fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0
    if not 1 <= fps <= 240:
        try:
            fps = float(Fraction(video.get("r_frame_rate", "30/1")))
        except (ValueError, ZeroDivisionError):
            fps = 30
    if not 1 <= fps <= 240:
        raise ValueError("第一版支持 1–240 fps 的视频")
    duration = float(video.get("duration", data.get("format", {}).get("duration", 0)))
    if duration <= 0 or width < 32 or height < 32:
        raise ValueError("视频尺寸或时长无效")
    hdr = video.get("color_transfer") in {"smpte2084", "arib-std-b67"}
    if hdr:
        # HEVC frequently stores mastering/CLL in frame SEI, absent from the
        # stream dictionary. Read one bounded frame, never scan the whole clip.
        video=dict(video,side_data_list=[*video.get('side_data_list',[]),
                                        *first_frame_side_data(path,cancel)])
    return VideoInfo(str(Path(path).resolve()), width, height, fps, duration,
                     any(s.get("codec_type") == "audio" for s in streams),
                     fingerprint=file_fingerprint(path), hdr=hdr, **native_metadata(video))


def first_frame_side_data(path, cancel=None):
    """Read HDR SEI from one video packet/frame, with cancellation and timeout."""
    if cancel is not None:check_cancel(cancel)
    process=subprocess.Popen([find_tool('ffprobe'),'-v','error','-select_streams','v:0',
        '-read_intervals','%+#1','-show_frames','-show_entries','frame=side_data_list',
        '-of','json',str(path)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,creationflags=NO_WINDOW)
    deadline=time.monotonic()+30
    try:
        while True:
            if cancel is not None:check_cancel(cancel)
            if time.monotonic()>deadline:raise TimeoutError('读取HDR首帧元数据超时')
            try:
                output,errors=process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:pass
        if cancel is not None:check_cancel(cancel)
        if len(output)+len(errors)>2*1024**2:raise ValueError('HDR首帧元数据超出读取上限')
        if process.returncode:raise ValueError('无法读取HDR首帧元数据：'+errors.decode('utf-8','replace')[-500:])
        frames=json.loads(output).get('frames',[])
        return frames[0].get('side_data_list',[]) if frames else []
    finally:
        if process.poll() is None:process.kill()
        process.communicate()



def native_metadata(video):
    """Retain explicitly reported color facts; never invent HDR mastering data."""
    import re
    pixel_format = video.get('pix_fmt') or 'yuv420p'
    match = re.search(r'(?:p|p0)(10|12|14|16)(?:le|be)$', pixel_format)
    try:
        depth = int(video.get('bits_per_raw_sample') or (match.group(1) if match else 8))
    except (TypeError, ValueError):
        depth = 8
    if depth not in (8, 10, 12, 14, 16):
        depth = int(match.group(1)) if match else 8
    result = dict(pixel_format=pixel_format, bit_depth=depth,
                  color_primaries=video.get('color_primaries', 'unknown'),
                  color_transfer=video.get('color_transfer', 'unknown'),
                  color_space=video.get('color_space', 'unknown'),
                  color_range=video.get('color_range', 'unknown'))
    for side in video.get('side_data_list', []):
        try:
            if side.get('side_data_type') == 'Mastering display metadata':
                def chroma(key): return round(float(Fraction(str(side[key])))*50000)
                def light(key): return round(float(Fraction(str(side[key])))*10000)
                result['mastering_display'] = (f"G({chroma('green_x')},{chroma('green_y')})"
                    f"B({chroma('blue_x')},{chroma('blue_y')})R({chroma('red_x')},{chroma('red_y')})"
                    f"WP({chroma('white_point_x')},{chroma('white_point_y')})"
                    f"L({light('max_luminance')},{light('min_luminance')})")
            elif side.get('side_data_type') == 'Content light level metadata':
                result['content_light'] = f"{int(side['max_content'])},{int(side['max_average'])}"
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            log.warning('无法解析部分HDR静态元数据')
    return result


def preview_tonemap_filter(info, *, opencv_proxy=False):
    # Tracking/display proxies are SDR; original native planes remain on disk.
    if not info.hdr:
        return ''
    if (info.color_transfer not in ('smpte2084','arib-std-b67') or
            info.color_primaries in ('unknown','','unspecified') or
            info.color_space in ('unknown','','unspecified')):
        raise ValueError('HDR颜色元数据不完整，无法安全生成显示代理，请重新导入有效HDR素材')
    # OpenCV's capture uses a 601 YUV->BGR matrix. Encode display proxies in
    # that matrix, with explicit 709 primaries/transfer, so switching to the
    # FFmpeg full-resolution display does not shift the visible colours.
    matrix='bt470bg' if opencv_proxy else 'bt709'
    return ('zscale=t=linear:npl=100,format=gbrpf32le,'
            f'tonemap=hable:desat=0,zscale=p=bt709:t=bt709:m={matrix}:r=tv,format=yuv420p,')


_CACHE_PREFERENCES = None


def configure_cache_preferences(preferences) -> None:
    """Register the app's QSettings so cache_root() reads the user's cache dir.

    Called once at app startup. Tests that don't call this use the env var or
    the default %LOCALAPPDATA% path.
    """
    global _CACHE_PREFERENCES
    _CACHE_PREFERENCES = preferences


def cache_root(preferences=None) -> Path:
    """Resolve the cache directory.

    Priority: LOCKFRAME_CACHE_DIR env var > explicit `preferences` arg >
    module-registered preferences (set via configure_cache_preferences) >
    %LOCALAPPDATA%\\LockFrameStudio\\cache (default).

    The caller is responsible for migrating existing cache files if the user
    changes the directory; this function only resolves and creates the path.
    """
    configured = os.environ.get("LOCKFRAME_CACHE_DIR")
    if not configured:
        prefs = preferences if preferences is not None else _CACHE_PREFERENCES
        if prefs is not None:
            configured = prefs.value("cache/dir", "")
    if configured:
        path = Path(configured).expanduser()
    else:
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        path = base / "LockFrameStudio" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_cache_dir(preferences, path: str | None) -> None:
    """Persist the user's chosen cache directory, or clear it to use the default."""
    if path:
        preferences.setValue("cache/dir", str(Path(path).expanduser()))
    else:
        preferences.remove("cache/dir")


def run_ffmpeg(args: list[str], cancel: threading.Event, duration: float = 0,
               progress=None, *, stall_timeout=None) -> None:
    with tempfile.TemporaryDirectory(prefix="lockframe-process-") as directory:
        progress_path = Path(directory) / "progress.txt"
        error_path = Path(directory) / "stderr.txt"
        command = [find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-progress", str(progress_path), "-y", *args]
        with error_path.open("wb") as errors:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=errors,
                                       creationflags=NO_WINDOW)
            try:
                last_activity = time.monotonic()
                last_stamp = None
                while process.poll() is None:
                    check_cancel(cancel)
                    if stall_timeout is not None:
                        stamp = progress_path.stat() if progress_path.exists() else None
                        stamp = (stamp.st_size, stamp.st_mtime_ns) if stamp else None
                        if stamp != last_stamp:
                            last_stamp, last_activity = stamp, time.monotonic()
                        if time.monotonic() - last_activity > stall_timeout:
                            raise TimeoutError("生成分析预览长时间无进展，请检查素材、磁盘空间后重试")
                    if duration and progress and progress_path.exists():
                        with progress_path.open("rb") as stream:
                            stream.seek(max(0, progress_path.stat().st_size - 2048))
                            lines = stream.read().decode("utf-8", "ignore").splitlines()
                        for line in reversed(lines):
                            if line.startswith("out_time_us="):
                                try:
                                    progress(min(0.99, int(line.split("=", 1)[1]) / 1e6 / duration))
                                except ValueError:
                                    pass
                                break
                    cancel.wait(0.15)
                check_cancel(cancel)
                if process.returncode:
                    message = error_path.read_text(encoding="utf-8", errors="replace")[-1800:]
                    raise RuntimeError("视频处理失败：" + message)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        if progress:
            progress(1.0)


def inspect_video(path: str, cancel: threading.Event, progress=None) -> VideoInfo:
    """Register metadata without transcoding every unopened clip in a batch.

    The estimated CFR range is provisional until import_video prepares this
    clip. No tracking or source-frame reads use this metadata-only object.
    """
    check_cancel(cancel)
    if progress: progress(0.)
    info = probe(path, cancel)
    info.frames = max(2, round(info.duration * info.fps))
    check_cancel(cancel)
    if progress: progress(1.)
    log.info("素材信息已读取 %s 尺寸=%dx%d 帧率=%.3f 估计帧数=%d 代理=按需准备",
             Path(info.path).name, info.width, info.height, info.fps, info.frames)
    return info


def import_video(path: str, cancel: threading.Event, progress=None) -> VideoInfo:
    check_cancel(cancel)
    if progress:progress(0.)
    info = probe(path, cancel)
    check_cancel(cancel)
    digest = hashlib.sha256((info.path + info.fingerprint + ("proxy-hdr-v4" if info.hdr else "proxy-v2")).encode()).hexdigest()[:24]
    proxy = cache_root() / (digest + ".mp4")
    if not proxy.is_file():
        partial = proxy.with_name(f"{proxy.stem}.{uuid.uuid4().hex}.partial.mp4")
        try:
            # Intra-only proxy permits accurate, responsive backward tracking and seeking.
            scale = min(1.0, 1280 / max(info.width, info.height))
            width = max(2, round(info.width * scale / 2) * 2)
            height = max(2, round(info.height * scale / 2) * 2)
            run_ffmpeg([
                "-i", info.path, "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", preview_tonemap_filter(info,opencv_proxy=True) + f"fps={info.fps:.10f},scale={width}:{height},setsar=1",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-g", "1",
                "-pix_fmt", "yuv420p", "-map_metadata", "-1", "-movflags", "+faststart",
                str(partial),
            ], cancel, info.duration,
                (lambda value: progress(.02 + .94 * value)) if progress else None,
                stall_timeout=120)
            check_cancel(cancel)
            if file_fingerprint(info.path) != info.fingerprint:
                raise ValueError("原视频在导入时发生变化，请重新导入")
            try:
                partial.replace(proxy)
            except PermissionError:
                if not proxy.is_file():raise
        finally:
            partial.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(proxy))
    try:
        if not capture.isOpened():
            proxy.unlink(missing_ok=True)
            raise ValueError("分析预览生成失败，请重新导入")
        info.frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if info.frames < 2:
            raise ValueError("视频至少需要两帧画面")
        info.duration = info.frames / info.fps
        info.proxy = str(proxy)
    finally:
        capture.release()
    check_cancel(cancel)
    if file_fingerprint(info.path) != info.fingerprint:
        raise ValueError("原视频在导入时发生变化，请重新导入")
    if progress:progress(1.)
    check_cancel(cancel)
    log.info("导入视频 %s 尺寸=%dx%d 帧率=%.3f 帧数=%d 音轨=%s",
             Path(info.path).name, info.width, info.height, info.fps, info.frames,
             info.has_audio)
    return info


def prepare_precision_video(info: VideoInfo, cancel: threading.Event, progress=None) -> VideoInfo:
    """Prepare a source-derived, higher-detail proxy without changing normal import.

    Both proxies use the same frame-rate filter, so normalized manual anchors and
    inclusive frame ranges refer to exactly the same source instants.  A unique
    staging file keeps simultaneous app instances from reading unfinished work.
    """
    check_cancel(cancel)
    if file_fingerprint(info.path) != info.fingerprint:
        raise ValueError("原视频已经被修改，请重新导入后再进行精准跟踪")
    if info.frames < 2 or info.width < 2 or info.height < 2 or not 1 <= info.fps <= 240:
        raise ValueError("视频分析信息无效，请重新导入")
    scale = min(1.0, 2560 / max(info.width, info.height))
    # Round down to even dimensions: never invent detail by upscaling odd sources.
    width = max(2, int(info.width * scale) // 2 * 2)
    height = max(2, int(info.height * scale) // 2 * 2)
    key = f"{info.path}{info.fingerprint}precision-{'hdr-v3' if info.hdr else 'v1'}:{info.fps:.10f}:{info.frames}:{width}x{height}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    proxy = cache_root() / ("precision-" + digest + ".mp4")

    def validate(path: Path) -> None:
        check_cancel(cancel)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError("精准分析视频无法读取，请重新生成")
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if frames != info.frames or actual_size != (width, height):
                raise ValueError("精准分析视频与普通预览帧数或尺寸不匹配，请重新导入")
            # Metadata alone can describe a truncated file.  Check both ends as
            # well, while leaving decoding of intervening frames to the tracker.
            for index in (0, info.frames - 1):
                check_cancel(cancel)
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError("精准分析视频不完整，请重新生成")
        finally:
            capture.release()
        check_cancel(cancel)

    if proxy.is_file():
        try:
            validate(proxy)
        except ValueError:
            proxy.unlink(missing_ok=True)
        else:
            if file_fingerprint(info.path) != info.fingerprint:
                raise ValueError("原视频已经被修改，请重新导入后再进行精准跟踪")
            check_cancel(cancel)
            if progress:
                progress(1.0)
            check_cancel(cancel)
            return replace(info, proxy=str(proxy))

    partial = proxy.with_name(f"{proxy.stem}.{uuid.uuid4().hex}.partial.mp4")
    try:
        run_ffmpeg([
            "-i", info.path, "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", preview_tonemap_filter(info,opencv_proxy=True) + f"fps={info.fps:.10f},scale={width}:{height},setsar=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "10", "-g", "1",
            "-pix_fmt", "yuv420p", "-map_metadata", "-1", "-movflags", "+faststart",
            str(partial),
        ], cancel, info.duration,
            (lambda value: progress(min(0.98, value * 0.98))) if progress else None)
        check_cancel(cancel)
        validate(partial)
        if file_fingerprint(info.path) != info.fingerprint:
            raise ValueError("原视频已经被修改，请重新导入后再进行精准跟踪")
        check_cancel(cancel)
        try:
            partial.replace(proxy)
        except PermissionError:
            # Another instance may have just published this same cache and
            # opened it for preview on Windows. Reuse its completed work.
            if not proxy.is_file():
                raise
            validate(proxy)
    finally:
        partial.unlink(missing_ok=True)
    check_cancel(cancel)
    if progress:
        progress(1.0)
    check_cancel(cancel)
    return replace(info, proxy=str(proxy))


class FrameReader:
    def __init__(self, path: str):
        self.capture = cv2.VideoCapture(path)
        self.next_frame = 0
        self._last_frame = None
        if not self.capture.isOpened():
            raise ValueError("无法打开预览视频")

    def read(self, index: int):
        if index == self.next_frame - 1 and self._last_frame is not None:
            return self._last_frame.copy()
        # Playback may skip a few source frames to keep the source clock. A
        # short forward grab avoids re-seeking and re-decoding the previous GOP.
        if self.next_frame < index <= self.next_frame + 8:
            while self.next_frame < index:
                if not self.capture.grab():
                    raise ValueError(f"无法解码第 {self.next_frame + 1} 帧")
                self.next_frame += 1
        if index != self.next_frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = self.capture.read()
        if not success:
            self._last_frame = None
            raise ValueError(f"无法解码第 {index + 1} 帧")
        self.next_frame = index + 1
        self._last_frame = frame.copy()
        return frame

    def close(self):
        self._last_frame = None
        self.capture.release()
