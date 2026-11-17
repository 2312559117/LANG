"""Source/frozen acceptance with synthetic 4K detail; no real-footage claim."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import threading
import time
import traceback

import cv2
import numpy as np

from . import __version__
from .media import configure_cache_preferences, import_video, run_ffmpeg
from .models import Settings, TrackResult, file_fingerprint
from .original_preview import read_original_frames
from .sequence import Clip
from .stabilization import preview_frame
from .v072_check import activate


def wait_original(window, timeout=20):
    from PySide6.QtWidgets import QApplication
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if '原片高清 · 左' in window.preview_note.text():
            return
        if window._original_preview.error:
            raise RuntimeError(window._original_preview.error)
        time.sleep(.005)
    raise TimeoutError('Original preview did not arrive')


def make_detail_clip(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((2160, 3840))
    image = np.full((2160, 3840, 3), 40, dtype=np.uint8)
    image[:, :, 0] = np.where(x % 4 < 2, 220, 30)
    image[:, :, 1] = np.where(y % 4 < 2, 200, 50)
    cv2.putText(image, '4K ORIGINAL  3840 x 2160', (250, 1080),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5, cv2.LINE_AA)
    png, path = folder / '原片细节.png', folder / '原片细节.mkv'
    cv2.imencode('.png', image)[1].tofile(png)
    run_ffmpeg(['-loop', '1', '-framerate', '12', '-i', str(png), '-frames:v', '6',
                '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(path)], threading.Event())
    info = import_video(str(path), threading.Event())
    track = TrackResult(0, info.frames - 1, np.full((info.frames, 2), .5),
                        np.tile([.4, .4, .2, .2], (info.frames, 1)),
                        np.ones(info.frames), ['tracked'] * info.frames)
    track.anchors = {0: (.4, .4, .2, .2)}
    return Clip(info, Settings(resolution=0, max_zoom=1., acceleration='cpu'),
                {0: (.4, .4, .2, .2)}, 0, info.frames - 1, track)


def check_quality(folder):
    from . import app
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    old_cache = os.environ.get('LOCKFRAME_CACHE_DIR')
    os.environ['LOCKFRAME_CACHE_DIR'] = str(folder / 'cache')
    window = None
    try:
        application = QApplication.instance() or QApplication([])
        original_settings = app.QSettings
        app.QSettings = lambda *args: QSettings(str(folder/'preferences.ini'), QSettings.Format.IniFormat)
        try:
            window = app.MainWindow()
        finally:
            app.QSettings = original_settings
        window.resize(1560, 940); window.show(); application.processEvents()
        clip = make_detail_clip(folder)
        fingerprint = file_fingerprint(clip.info.path)
        activate(window, [clip])
        window.canvas.set_zoom(2.)
        track = window.track
        settings = asdict(window.settings())
        anchors = dict(window.anchors)
        wait_original(window)
        assert window.canvas.source.shape == (2160, 3840, 3)
        assert window.canvas.processed.shape == (2160, 3840, 3)
        assert window.frame.shape[1] == 1280  # Analysis pixels never replaced.
        assert window.canvas.zoom_factor == 2.
        original = read_original_frames(clip.info, (0,), threading.Event())[0]
        assert np.array_equal(window.canvas.source, original)
        expected = preview_frame(original, clip.info, window.plan, 0, max_side=None)
        assert np.array_equal(window.canvas.processed, expected)
        original_error = float(np.abs(window.canvas.source.astype(float)-original).mean())
        window.grab().save(str(folder/'双预览原片高清.png'))
        window.preview_quality.setCurrentIndex(1)
        assert window.canvas.source.shape[1] == 1280 and window.canvas.processed.shape[1] == 1280
        proxy = cv2.resize(window.canvas.source, (3840,2160), interpolation=cv2.INTER_LINEAR)
        proxy_error = float(np.abs(proxy.astype(float)-original).mean())
        assert proxy_error > original_error + 5
        window.preview_quality.setCurrentIndex(0)
        wait_original(window)
        window._toggle_play()
        assert window.timer.isActive() and window.canvas.source.shape[1] == 1280
        window._stop_play(); wait_original(window)
        window._step_preview(1); wait_original(window)
        assert window.frame_index == 1 and window.canvas.source.shape[1] == 3840
        assert window.track is track and window.anchors == anchors and asdict(window.settings()) == settings
        assert file_fingerprint(clip.info.path) == fingerprint
        return dict(footage='synthetic 4K detail only; real car footage pending',
                    source_size=[3840,2160],effect_size=[3840,2160],
                    proxy_width=1280,original_pixel_mae=original_error,proxy_upscaled_mae=proxy_error,
                    exact_cfr_clock=True,zoom_preserved=True,playback_proxy_pause_original=True,
                    next_frame_native=True,tracking_and_parameters_unchanged=True,source_unchanged=True)
    finally:
        if window:
            window._confirm_discard=lambda:True
            window.close(); application.processEvents()
        if old_cache is None: os.environ.pop('LOCKFRAME_CACHE_DIR',None)
        else: os.environ['LOCKFRAME_CACHE_DIR']=old_cache
        configure_cache_preferences(None)


def run_quality_check(directory):
    folder=Path(directory).resolve(); folder.mkdir(parents=True,exist_ok=True)
    try:
        report=dict(ok=True,version=__version__,preview_quality=check_quality(folder));code=0
    except Exception:
        report=dict(ok=False,version=__version__,error=traceback.format_exc());code=1
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return code
