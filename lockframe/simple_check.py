"""Real Qt acceptance of the simplified workspace and inline MS-style ranges."""
from pathlib import Path
import json
import traceback


def check_simple_workspace(folder):
    from PySide6.QtCore import QSettings, Qt, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from . import app
    from .models import Settings
    from .sequence import Clip
    from .v072_check import create_correction_source, correction_box, activate, wait_job
    directory = Path(folder)
    directory.mkdir(parents=True, exist_ok=True)
    original = app.QSettings
    window = None
    try:
        app.QSettings = lambda *args: QSettings(str(directory/'preferences.ini'), QSettings.Format.IniFormat)
        window = app.MainWindow()
        window.resize(1560, 960)
        window.show()
        window._confirm_discard = lambda: True
        info = create_correction_source(directory/'synthetic')
        activate(window, [Clip(info, Settings(acceleration='cpu',compute_threads=2,resolution=0),
                               {0:correction_box(0)}, 0, 47)])
        assert len(window.monitor_transports)==1
        assert window.hardware_brief.isHidden()
        assert window.strip.row_rect('mark').isEmpty() and window.strip.row_rect('audio').isEmpty()
        window._open_tracking_sections()
        panel = window._sections_panel
        assert panel.isVisible() and not panel.isWindow()
        def mark(frame, key):
            window._seek(frame)
            QTest.keyClick(window.strip, key)
        mark(0, Qt.Key.Key_I);mark(11, Qt.Key.Key_O)
        mark(12, Qt.Key.Key_I);mark(47, Qt.Key.Key_O)
        expected = [dict(start=0,end=11), dict(start=12,end=47)]
        assert window.tracking_sections==expected
        window._select_roi(correction_box(47))
        window._start_tracking();wait_job(window)
        assert window.track is not None and not window.track.lost_frames
        result = window.track
        window.zoom_slider.setValue(220)
        assert window.track is result and window.settings().max_zoom==2.2
        window.monitor_transports[0].slider.setValue(20)
        assert window.frame_index==20 and window.canvas.source is not None and window.canvas.processed is not None
        window.canvas.set_zoom(2)
        window.canvas.set_output_zoom(3)
        assert window.canvas.zoom_factor==2 and window.canvas.output_zoom==3
        window.canvas.reset_view();window.canvas.set_output_zoom(1)
        window.audio_waveform_action.setChecked(True)
        assert not window.strip.row_rect('audio').isEmpty() and window.track is result
        window.audio_waveform_action.setChecked(False)
        for width,height in ((1560,960),(1160,760)):
            window.resize(width,height);QApplication.processEvents()
            assert (window.width(),window.height())==(width,height)
            assert window.canvas.source is not None and window.canvas.processed is not None
            assert panel.isVisible() and window.canvas.height()>=100
            assert window.export_button.visibleRegion().contains(window.export_button.rect())
            assert window.sidebar_sections['lock'].isAncestorOf(window.zoom_slider)
            assert window.zoom_slider.visibleRegion().contains(window.zoom_slider.rect())
            window.grab().save(str(directory/f'workspace-{width}x{height}.png'))
        file_actions = [a.text() for a in window._simple_menu_actions[0].menu().actions()]
        assert not any('验收记录' in text or '合成演示' in text for text in file_actions)
        help_actions = window._simple_menu_actions[-1].menu().actions()
        assert any('合成演示' in a.text() for a in help_actions)
        assert any('诊断包' in a.text() for a in help_actions)
        assert any('开发与验收' in a.text() for a in help_actions)
        return dict(shared_transport=1,persistent_dual_preview=True,annotation_removed=True,
                    waveform_hidden_default=True,original_audio_policy='retained',
                    inline_segments=expected,keyboard_io=True,independent_target_tracking=True,
                    zoom_limit_in_composition=True,composition_preserves_tracking=True,
                    independent_view_zoom=True,help_and_diagnostics_retained=True,
                    compact_size=[1160,760],synthetic_footage=True)
    finally:
        if window is not None:
            if window.busy and window.worker:
                window.worker.cancel();window.worker.wait(15000)
                QApplication.processEvents()
            window.close()
        app.QSettings = original


def run_simple_check(folder):
    from . import __version__
    directory=Path(folder);directory.mkdir(parents=True,exist_ok=True)
    try:
        report=dict(ok=True,version=__version__,**check_simple_workspace(directory))
    except Exception:
        report=dict(ok=False,version=__version__,error=traceback.format_exc())
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report['ok'] else 1
