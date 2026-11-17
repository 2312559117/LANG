"""Real lossless portrait resolution chart and physical-pixel cache keys."""
from dataclasses import replace
import subprocess

import cv2
import numpy as np

from lockframe.media import probe,find_tool,NO_WINDOW
from lockframe.timeline_media import MediaSource,make_request,read_film,WorkCancel


def test_original_portrait_detail_survives_and_low_res_proxy_is_not_enlarged(tmp_path):
    chart=np.zeros((1280,720,3),np.uint8)
    for x in range(0,720,12):chart[:,x:x+6]=255
    png=tmp_path/'竖屏清晰度图.png';cv2.imencode('.png',chart)[1].tofile(png)
    path=tmp_path/'原片.mkv';proxy=tmp_path/'低清分析.mkv'
    subprocess.run([find_tool('ffmpeg'),'-v','error','-y','-loop','1','-i',str(png),'-frames:v','2',
                    '-r','10','-c:v','ffv1',str(path)],check=True,capture_output=True,creationflags=NO_WINDOW)
    subprocess.run([find_tool('ffmpeg'),'-v','error','-y','-f','lavfi','-i','color=red:size=90x160:rate=10',
                    '-frames:v','2','-c:v','ffv1',str(proxy)],check=True,capture_output=True,creationflags=NO_WINDOW)
    source=MediaSource.from_info(replace(probe(str(path)),proxy=str(proxy)))
    request=make_request(source,0,2,count=1,pixel_size=(320,100))
    image=read_film(request,WorkCancel())[0]
    assert image.shape==(100,320,3)
    assert np.max(np.abs(image[:,:,0].astype(int)-image[:,:,1]))<2  # Original monochrome, not red proxy.
    old=cv2.resize(cv2.resize(chart,(45,80),interpolation=cv2.INTER_AREA),(320,569),interpolation=cv2.INTER_LINEAR)
    old=old[(569-100)//2:(569+100)//2]
    assert image.std()>old.std()*3 and image.std()>60
    assert request.key!=make_request(source,0,2,count=1,pixel_size=(640,200)).key
    bounded=make_request(source,0,2,pixel_size=(100000,100000))
    assert bounded.tile_width<=768 and bounded.tile_height<=384
