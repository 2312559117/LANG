from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import os
import subprocess
import tempfile
import threading
import cv2
import numpy as np
from .media import NO_WINDOW, check_cancel, find_tool, run_ffmpeg
from .models import Settings, TrackResult, VideoInfo, file_fingerprint
from .stabilization import build_plan
from .encoding import encoder_options, select_encoder, HardwareEncodingError
from .logging import get_logger
from .regions import shape_matches

log = get_logger("export")


@contextmanager
def _source_frames(info, track, folder, cancel):
    """Stream each selected original frame once on the existing CFR clock."""
    vf = (f"fps={info.fps:.10f},trim=start_frame={track.start}:end_frame={track.end + 1},"
          f"setpts=PTS-STARTPTS,scale={info.width}:{info.height},setsar=1")
    with (folder / "decode.log").open("wb") as errors:
        decoder = subprocess.Popen([
            find_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", info.path, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf,
            "-frames:v", str(track.count), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ], stdout=subprocess.PIPE, stderr=errors, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        def frames():
            size = info.width * info.height * 3
            for absolute in range(track.start, track.end + 1):
                check_cancel(cancel)
                raw = decoder.stdout.read(size)
                if len(raw) != size:
                    raise RuntimeError(f"原视频在第 {absolute + 1} 帧解码中断")
                yield absolute, np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
        try:
            yield frames()
            decoder.stdout.close()
            while decoder.poll() is None:
                check_cancel(cancel)
                cancel.wait(.1)
            if decoder.returncode:
                raise RuntimeError("原视频解码失败，请检查原视频和磁盘空间")
        finally:
            if decoder.poll() is None: decoder.kill()
            decoder.wait()
            decoder.stdout.close()


def export_video(info: VideoInfo, track: TrackResult, settings: Settings,
                 destination: str, cancel: threading.Event, progress=None, *, overwrite=True) -> dict:
    if track is None or not track.performance.get("complete", True):
        raise ValueError("跟踪尚未完成，请修正并继续跟踪后导出")
    settings.validate()
    if not shape_matches(track, settings):
        raise ValueError("区域形状或顶点已修改，结果已过期，请重新跟踪")
    if settings.retime.get('enabled') or getattr(info,'hdr',False) or getattr(info,'bit_depth',8) > 8:
        from .time_remap_map import output_frame_map
        from .retime_renderer import render_retimed_video
        mapping=output_frame_map(settings.retime or None,track.count)
        return render_retimed_video(info,track,settings,mapping,destination,cancel,progress,overwrite=overwrite)
    log.info("开始导出 帧=%s 分辨率=%d 设备=%s 目标=%s",
             f"{track.start}..{track.end}" if track is not None else "-",
             settings.resolution, settings.acceleration,
             Path(destination).name)
    encoder = select_encoder(settings.acceleration, cancel)
    try:
        return _export_video(info, track, settings, destination, cancel, progress, encoder, overwrite=overwrite)
    except HardwareEncodingError:
        if encoder == "libx264":
            raise
        if progress:
            progress(0., "硬件编码不可用，正在改用 CPU 重新渲染")
        result = _export_video(info, track, settings, destination, cancel, progress, "libx264", overwrite=overwrite)
        result["hardware_fallback"] = True
        return result

def _export_video(info, track, settings, destination, cancel, progress, encoder_name, overwrite=True):
    check_cancel(cancel)
    track.validate(info.frames)
    if track.lost_frames:
        raise ValueError(f"有 {len(track.lost_frames)} 帧跟丢。请补关键帧并重新跟踪后导出。")
    target = Path(destination).resolve()
    if not overwrite and target.exists():
        raise FileExistsError("输出文件已存在，重新入队将自动编号")
    if target == Path(info.path).resolve() or target == Path(info.proxy).resolve():
        raise ValueError("输出文件不能覆盖原视频或分析预览")
    if target.suffix.lower() != ".mp4":
        raise ValueError("请导出为 .mp4 文件")
    if file_fingerprint(info.path) != info.fingerprint:
        raise ValueError("原视频已经被修改，请重新导入并跟踪")
    plan = build_plan(info, track, settings)
    duration = track.count / info.fps
    ffmpeg = find_tool("ffmpeg")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lockframe-export-", dir=target.parent) as directory:
        folder = Path(directory)
        silent, finished = folder / "video.mp4", folder / "finished.mp4"
        with (folder / "encode.log").open("wb") as encode_log:
            encoder = None
            try:
                with _source_frames(info, track, folder, cancel) as frames:
                    encoder = subprocess.Popen([
                        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                        "-pix_fmt", "bgr24", "-s", f"{plan.width}x{plan.height}", "-r", f"{info.fps:.10f}",
                        "-i", "pipe:0", "-an", *encoder_options(encoder_name, settings.crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent),
                    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encode_log, creationflags=NO_WINDOW)
                    for index, (absolute, frame) in enumerate(frames):
                        check_cancel(cancel)
                        result = cv2.warpAffine(frame, plan.matrices[absolute - track.start],
                            (plan.width, plan.height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                        try: encoder.stdin.write(result.tobytes())
                        except BrokenPipeError as error:
                            raise HardwareEncodingError("编码器异常退出，请检查磁盘空间和输出路径") from error
                        if progress and (index % 4 == 0 or index == track.count - 1):
                            progress(.9 * (index + 1) / track.count, f"正在从原片顺序渲染 · {encoder_name}")
                try: encoder.stdin.close()
                except BrokenPipeError as error:
                    raise HardwareEncodingError("视频编码器提前退出") from error
                while encoder.poll() is None:
                    check_cancel(cancel)
                    cancel.wait(.1)
                if encoder.returncode: raise HardwareEncodingError("视频编码失败")
            finally:
                if encoder is not None:
                    if encoder.poll() is None: encoder.kill()
                    encoder.wait()
                    if encoder.stdin and not encoder.stdin.closed:
                        try: encoder.stdin.close()
                        except BrokenPipeError: pass
        check_cancel(cancel)
        if info.has_audio:
            run_ffmpeg([
                "-ss", f"{track.start / info.fps:.10f}", "-t", f"{duration:.10f}",
                "-i", info.path, "-i", str(silent),
                "-map", "1:v:0", "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac",
                "-b:a", "192k", "-af", "apad", "-t", f"{duration:.10f}",
                "-map_metadata", "-1", "-movflags", "+faststart", str(finished),
            ], cancel, duration,
                (lambda value: progress(.9 + .09 * value, "正在保留原片声音")) if progress else None)
        else: silent.replace(finished)
        check_cancel(cancel)
        if overwrite: os.replace(finished, target)
        elif os.name == "nt": os.rename(finished, target)
        else: os.link(finished, target)
    if progress: progress(1., "导出完成")
    log.info("导出完成 编码=%s 尺寸=%dx%d 帧=%d 文件=%s", encoder_name, plan.width, plan.height, track.count, target.name)
    return dict(path=str(target),width=plan.width,height=plan.height,frames=track.count,
                zoom=plan.zoom,limited_frames=plan.limited_frames,encoder=encoder_name,
                duration=duration,audio_policy="original" if info.has_audio else "none")
