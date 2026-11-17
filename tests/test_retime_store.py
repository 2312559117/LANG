from pathlib import Path
import threading
import numpy as np
import pytest
from lockframe.retime_store import open_frame_store,store_strategy,ChunkedFrameStore
from lockframe.retime_frames import NativeFrameStore,probe_native
from lockframe.retime_renderer import render_retimed_video
from lockframe.media import Cancelled
from test_retime_renderer import clip,settings
from test_retime_media import source_media


def test_stream_is_exact_and_writes_no_full_yuv_cache(clip):
    info,track,folder=clip;cancel=threading.Event();metadata=probe_native(info.path,cancel)
    (folder/'stream').mkdir()
    with NativeFrameStore.open(info,track,folder/'reference',cancel,metadata=metadata) as expected:
        with open_frame_store(info,track,np.arange(track.count,dtype=float),folder/'stream',cancel,metadata=metadata) as actual:
            for i in (0,1,0,4,8,11):
                assert actual.read(i).to_bytes()==expected.read(i).to_bytes()
            assert actual.strategy=='stream' and actual.bytes_written==0
            assert actual.peak_cache_bytes<=2*actual.frame_bytes
        assert not list((folder/'stream').glob('*.yuv'))


def test_subset_freeze_delivers_only_needed_native_frame(clip):
    info,track,folder=clip;cancel=threading.Event();metadata=probe_native(info.path,cancel)
    (folder/'one').mkdir()
    with open_frame_store(info,track,np.full(48,8.),folder/'one',cancel,metadata=metadata) as store:
        first=store.read(8).to_bytes()
        for _ in range(47):assert store.read(8).to_bytes()==first
        assert store.decoded_frames==1 and store.bytes_written==0


def test_lazy_chunks_are_exact_bounded_and_cleanup_evicted_files(clip):
    info,track,folder=clip;cancel=threading.Event();metadata=probe_native(info.path,cancel)
    store=ChunkedFrameStore(info,track,folder/'chunks',cancel,metadata);store.block_frames=2
    with NativeFrameStore.open(info,track,folder/'reference',cancel,metadata=metadata) as expected:
        for index in (0,5,10,1):
            assert store.read(index).to_bytes()==expected.read(index).to_bytes()
            assert len(store.blocks)<=2 and len(list((folder/'chunks').rglob('*.yuv')))<=2
        assert store.peak_cache_bytes<=4*store.frame_bytes
    store.close();assert not list((folder/'chunks').rglob('*.yuv'))


def test_access_pattern_selection_avoids_quadratic_reverse_decode():
    assert store_strategy(np.arange(100.),100,100)=='stream'
    assert store_strategy(np.arange(99.,-1.,-1.),100,100)=='disk'
    assert store_strategy(np.array([0.,3.,1.,3.]),1000,100)=='chunks'


def test_cancellation_closes_stream_and_never_commits_product(clip):
    info,track,folder=clip;cancel=threading.Event()
    def progress(value,text):
        if value>.15:cancel.set()
    target=folder/'cancel-stream.mp4'
    with pytest.raises(Cancelled):
        render_retimed_video(info,track,settings(),np.linspace(0,11,24),str(target),cancel,progress)
    assert not target.exists() and not list(folder.glob('*.partial.mp4'))


def test_hardware_retry_rewinds_stream_preserving_map_and_flow(clip,monkeypatch):
    import lockframe.retime_renderer as module
    from lockframe.encoding import HardwareEncodingError
    info,track,folder=clip;original=module._render_attempt;seen=[]
    monkeypatch.setattr(module,'select_encoder',lambda *args:'h264_nvenc')
    def attempt(info,track,settings,mapping,store,plan,*args):
        seen.append(mapping.copy())
        if len(seen)==1:
            store.read(8);raise HardwareEncodingError('simulated encoder failure after decoding')
        return original(info,track,settings,mapping,store,plan,*args)
    monkeypatch.setattr(module,'_render_attempt',attempt)
    mapping=np.linspace(0,11,24)
    result=render_retimed_video(info,track,settings(optical_flow=True),mapping,str(folder/'retry-stream.mp4'),threading.Event())
    assert result['hardware_fallback'] and result['actual_backend']=='dis'
    assert result['source_cache_strategy']=='stream' and result['source_store_bytes']==0
    assert result['frames']==24 and result['stage_timing']['total_seconds']>0
    np.testing.assert_array_equal(seen[0],seen[1])


def test_trimmed_hdr10_stream_matches_native_store_without_downconversion(source_media,tmp_path):
    from lockframe.media import probe
    from test_retime_renderer import tracked
    from dataclasses import replace
    info=probe(str(source_media[True]),threading.Event());track=tracked(info)
    track=replace(track,start=1,end=4,positions=track.positions[1:5],boxes=track.boxes[1:5],
        confidence=track.confidence[1:5],states=track.states[1:5],anchors={1:(.25,.25,.5,.5)})
    cancel=threading.Event();metadata=probe_native(info.path,cancel)
    (tmp_path/'stream10').mkdir()
    with NativeFrameStore.open(info,track,tmp_path/'reference10',cancel,metadata=metadata) as reference:
        with open_frame_store(info,track,np.array([.2,1.,2.7]),tmp_path/'stream10',cancel,metadata=metadata) as actual:
            assert actual.format.depth==10
            for i in range(4):assert actual.read(i).to_bytes()==reference.read(i).to_bytes()
            assert actual.bytes_written==0 and actual.decoded_frames==4
