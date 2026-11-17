"""Late preview protection includes curve generation, even at the same clock."""
from copy import deepcopy
import numpy as np
from PySide6.QtCore import QCoreApplication,QEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid
from test_retime_integration import window,demo_info
from lockframe.time_remap_model import make_constant_speed


def test_changing_curve_rejects_late_preview_at_same_output_position(window,demo_info,monkeypatch):
    from lockframe.sequence import Clip
    from lockframe.models import Settings
    from lockframe.v072_check import activate
    activate(window,[Clip(deepcopy(demo_info),Settings(),{},0,demo_info.frames-1)])
    config=make_constant_speed(speed=.5);config.update(optical_flow=True,preview_flow=True)
    window._apply_retime_config(config);window._retime_clock_changed('output');window._seek_output(1)
    pending=[]
    def request(token,*args):
        pending.append(token);window._retime_preview.wanted=token
    monkeypatch.setattr(window._retime_preview,'request',request)
    window._retime_request_preview();old=pending[-1]
    # Same source identity, range, output step and RenderPlan. Only the curve
    # changed; a queued signal from the previous generation must be rejected.
    config=make_constant_speed(speed=.6);config.update(optical_flow=True,preview_flow=True)
    window._apply_retime_config(config);window._seek_output(1);window._retime_request_preview()
    new=pending[-1]
    assert old[:5]==new[:5] and old!=new
    before=window.canvas.processed_image
    window._retime_preview_ready(old,np.zeros((32,32,3),np.uint8),'stale generation')
    assert window.canvas.processed_image is before
    assert 'stale generation' not in window.preview_note.text()


def test_reopening_editor_releases_previous_closed_dialog(window,demo_info):
    from lockframe.sequence import Clip
    from lockframe.models import Settings
    from lockframe.v072_check import activate
    activate(window,[Clip(deepcopy(demo_info),Settings(),{},0,demo_info.frames-1)])
    window._open_retime_editor();old=window._retime_editor;old.reject()
    window._open_retime_editor();assert window._retime_editor is not old
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    assert not isValid(old)
