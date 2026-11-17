"""Contracts for on-demand CFR preparation and foreground playback scheduling."""
from dataclasses import replace
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from test_workflow_v9 import window, source
from lockframe import media, sequence
from lockframe.models import VideoInfo
from lockframe.sequence import Clip
from lockframe.timeline_media import MediaSource, make_request, read_peaks, WorkCancel
from lockframe.v072_check import activate, wait_job


def test_metadata_batch_never_transcodes_and_keeps_partial_cancel(monkeypatch):
    calls=[];cancel=threading.Event()
    def inspect(path,event,report=None):
        calls.append(path)
        if path=='two':event.set();media.check_cancel(event)
        return VideoInfo(path,3840,2160,60,10,frames=600)
    monkeypatch.setattr(sequence,'inspect_video',inspect)
    monkeypatch.setattr(sequence,'import_video',lambda *a:pytest.fail('eager transcoding'))
    result=sequence.import_many(['one','two','three'],cancel,metadata_only=True)
    assert calls==['one','two'] and len(result)==1 and not result[0]['info'].proxy


def test_unopened_sequence_restore_does_not_transcode(tmp_path,monkeypatch):
    info=VideoInfo(str(tmp_path/'a.mov'),1080,1920,60,1,frames=60,fingerprint='same')
    clip=Clip(info);calls=[]
    monkeypatch.setattr(sequence,'inspect_video',lambda *a:calls.append(a[0]) or replace(info))
    monkeypatch.setattr(sequence,'import_video',lambda *a:pytest.fail('restored unopened proxy'))
    restored=sequence.restore_sequence([clip],threading.Event())
    assert calls==[info.path] and restored[0].info.proxy=='' and restored[0].end==59


def test_deferred_prepare_cancel_retry_and_range_reconciliation(window,source,monkeypatch):
    # Provisional range may differ from the exact proxy frame count.
    metadata=replace(source.info,proxy='',frames=source.info.frames+3)
    clip=Clip(metadata);window.clips=[clip]
    entered=threading.Event()
    def cancelled(path,cancel,progress=None):
        entered.set();cancel.wait(3);media.check_cancel(cancel)
    monkeypatch.setattr(media,'import_video',cancelled)
    window._activate_clip(clip.id)
    assert entered.wait(2)
    window._cancel();wait_job(window)
    assert not clip.info.proxy and window.active_clip_id is None and not window.busy
    monkeypatch.setattr(media,'import_video',lambda *a,**k:replace(source.info))
    window._activate_clip(clip.id);wait_job(window)
    assert window.active_clip_id==clip.id and clip.end==source.info.frames-1
    assert window.end_spin.value()==source.info.frames and window.reader is not None


def test_hidden_audio_launches_no_process_and_toggle_changes_cache_key(tmp_path,monkeypatch):
    path=tmp_path/'sound.mov';path.write_bytes(b'identity only')
    info=VideoInfo(str(path),160,90,30,1,frames=30,has_audio=True)
    source=MediaSource.from_info(info)
    hidden=make_request(source,0,30,include_audio=False)
    shown=make_request(source,0,30,include_audio=True)
    assert hidden.key!=shown.key
    monkeypatch.setattr(media.subprocess,'Popen',lambda *a,**k:pytest.fail('hidden audio decoded'))
    assert read_peaks(hidden,WorkCancel()).size==0


def test_source_change_during_deferred_prepare_discards_old_measurements(window,source,monkeypatch):
    clip=Clip(replace(source.info,proxy='',fingerprint='old'),source.settings,
              dict(source.anchors),source.start,source.end,source.track)
    clip.precision_info=source.info
    window.clips=[clip]
    monkeypatch.setattr(media,'import_video',lambda *a,**k:replace(source.info))
    window._activate_clip(clip.id);wait_job(window)
    assert not window.anchors and window.track is None and clip.track is None
    assert clip.precision_info is None
    assert (clip.start,clip.end)==(0,source.info.frames-1) and clip.dirty


def test_sequential_skip_duplicate_and_reverse_match_exact_decoder(tmp_path):
    path=tmp_path/'精确源帧.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(96,64))
    assert writer.isOpened()
    for i in range(25):writer.write(np.full((64,96,3),i*8,np.uint8))
    writer.release();reader=media.FrameReader(str(path));reference=cv2.VideoCapture(str(path))
    try:
        for index in (0,1,5,5,14,24,2,3):
            reference.set(cv2.CAP_PROP_POS_FRAMES,index);ok,expected=reference.read();assert ok
            actual=reader.read(index);np.testing.assert_array_equal(actual,expected)
            actual[:]=0  # a caller cannot corrupt the repeated-frame cache
    finally:reader.close();reference.release()


def test_playback_pauses_background_and_avoids_per_frame_layout(window,source):
    from lockframe.performance_check import check_playback_hotpath
    activate(window,[Clip(source.info,source.settings,dict(source.anchors),source.start,source.end,source.track)])
    report=check_playback_hotpath(window)
    assert report['per_frame_layout_avoided'] and report['submitted_frames']>=6


def test_metadata_only_tracking_cannot_use_estimated_range(tmp_path,monkeypatch):
    path=tmp_path/'source.mov';path.write_bytes(b'fixture')
    from lockframe.models import file_fingerprint
    info=VideoInfo(str(path),160,90,30,1,frames=30,fingerprint=file_fingerprint(path))
    with pytest.raises(ValueError,match='先打开'):
        sequence._track_clip(Clip(info,anchors={0:(.1,.1,.2,.2)}),1)


def test_cancelled_batch_retains_list_without_launching_proxy(window, source, tmp_path, monkeypatch):
    from lockframe import sequence_ui

    second = tmp_path / 'second.mov'
    second.write_bytes(b'fixture')
    inspecting_second = threading.Event()
    proxy_calls = []

    def inspect(paths, cancel, progress=None, *, metadata_only=False):
        assert metadata_only
        inspecting_second.set()
        assert cancel.wait(3)
        return [dict(info=replace(source.info, proxy=''), path=paths[0])]

    def prepare(path, cancel, progress=None):
        proxy_calls.append(path)
        return replace(source.info)

    monkeypatch.setattr(sequence_ui, 'import_many', inspect)
    monkeypatch.setattr(media, 'import_video', prepare)
    window._import_paths([source.info.path, str(second)])
    assert inspecting_second.wait(2)
    window._cancel()
    wait_job(window)
    assert not proxy_calls and window.active_clip_id is None
    assert len(window.clips) == 1 and not window.clips[0].info.proxy
    assert '导入已取消' in window.status_label.text()
    window._activate_clip(window.clips[0].id)
    wait_job(window)
    assert proxy_calls == [source.info.path] and window.active_clip_id == window.clips[0].id


def test_restore_opens_readable_clip_and_keeps_missing_entry(window, source, tmp_path, monkeypatch):
    missing = Clip(replace(source.info, path=str(tmp_path / 'missing.mov'), proxy=''))
    valid = Clip(source.info)
    path = tmp_path / 'mixed.locksequence'
    sequence.save_sequence(path, [missing, valid])
    errors = []
    monkeypatch.setattr(window, '_error', errors.append)
    monkeypatch.setattr(window, '_confirm_discard', lambda: True)
    window._load_sequence_file(str(path))
    wait_job(window)
    assert not errors and window.info.path == valid.info.path
    assert len(window.clips) == 2 and window.clips[0].status.startswith('读取失败')

    unavailable = tmp_path / 'missing-only.locksequence'
    sequence.save_sequence(unavailable, [missing])
    active = window.active_clip_id
    window._load_sequence_file(str(unavailable))
    wait_job(window)
    assert len(errors) == 1 and '所有原片' in errors[0]
    assert window.active_clip_id == active and len(window.clips) == 2
