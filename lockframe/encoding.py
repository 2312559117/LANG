"""Probe working hardware encoders, not just advertised build capabilities."""
import subprocess

from .media import NO_WINDOW, check_cancel, find_tool

_verified = {}


def encoder_options(name, quality):
    if name == "h264_nvenc":
        return ["-c:v", name, "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
    if name == "h264_amf":
        return ["-c:v", name, "-quality", "quality", "-rc", "cqp", "-qp_i", str(quality), "-qp_p", str(quality)]
    if name == "h264_qsv":
        return ["-c:v", name, "-preset", "medium", "-global_quality", str(quality)]
    return ["-c:v", "libx264", "-crf", str(quality), "-preset", "fast"]


def select_encoder(acceleration, cancel):
    check_cancel(cancel)
    if acceleration == "cpu":
        return "libx264"
    ffmpeg = find_tool("ffmpeg")
    if ffmpeg in _verified:
        return _verified[ffmpeg]
    for name in ("h264_nvenc", "h264_amf", "h264_qsv"):
        check_cancel(cancel)
        try:
            result = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                                     "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=0.1",
                                     "-frames:v", "1", *encoder_options(name, 18),
                                     "-pix_fmt", "yuv420p", "-f", "null", "-"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=4, creationflags=NO_WINDOW)
            check_cancel(cancel)
            if result.returncode == 0:
                _verified[ffmpeg] = name
                return name
        except (OSError, subprocess.TimeoutExpired):
            continue
    _verified[ffmpeg] = "libx264"
    return "libx264"


class HardwareEncodingError(RuntimeError):
    pass
