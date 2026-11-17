"""Review annotations never alter measurements or resolve true lost frames."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,re
import numpy as np
from .tracking_pause import computed_mask
from .drift_review import review_ranges


def clean_marks(value):
    if not isinstance(value,dict):return {}
    return {key:True for key,mark in list(value.items())[:2048]
        if isinstance(key,str) and re.fullmatch('[0-9a-f]{64}',key) and mark is True}


@dataclass(frozen=True)
class ReviewIssue:
    clip_id:str
    filename:str
    kind:str
    first:int
    last:int
    key:str
    checked:bool
    reason:str

    @property
    def historical(self):return self.kind=='关键帧'

    @property
    def checkable(self):return self.kind not in ('真实跟丢','关键帧')


def _runs(mask,start):
    edges=np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return zip(start+edges[::2],start+edges[1::2]-1)


def collect_issues(clips,limit=2000):
    result=[]
    for clip in clips:
        filename=Path(clip.info.path).name;track=clip.track
        marks=clean_marks(getattr(clip,'review_marks',{}))
        if track is not None:
            measured=computed_mask(track);n=track.count
            masks={name:np.zeros(n,dtype=bool) for name in ('真实跟丢','跟偏预警','运动突变','低匹配质量')}
            masks['真实跟丢']=measured & (np.asarray(track.states)=='lost')
            for item in review_ranges(track.performance.get('early_warning_ranges',[]),track.start,track.end)+review_ranges(track.performance.get('suspected_drift_ranges',[]),track.start,track.end):
                masks['跟偏预警'][item['start']-track.start:item['end']-track.start+1]=True
            quality=track.performance.get('quality',{})
            for name,key in (('运动突变','motion_review'),('低匹配质量','low_confidence')):
                for frame in quality.get(key,[]):
                    if isinstance(frame,(int,np.integer)) and track.start<=frame<=track.end:masks[name][frame-track.start]=True
            covered=np.zeros(n,dtype=bool)
            identity=json.dumps(dict(source=[clip.info.path,clip.info.fingerprint],range=[track.start,track.end],
                anchors=sorted(clip.anchors.items()),sections=clip.tracking_sections,
                algorithm=[track.tracking_mode,track.performance.get('tracking_revision'),track.performance.get('reference_revision')]),sort_keys=True)
            for kind,mask in masks.items():
                chosen=mask & measured & ~covered;covered|=chosen
                for first,last in _runs(chosen,track.start):
                    first,last=int(first),int(last);sl=slice(first-track.start,last-track.start+1)
                    digest=hashlib.sha256((identity+kind+str((first,last))).encode())
                    for values in (track.positions[sl],track.boxes[sl],track.confidence[sl]):digest.update(np.ascontiguousarray(values,dtype='<f8').tobytes())
                    key=digest.hexdigest()
                    reason={'真实跟丢':'需重新框选并重算；人工检查不能将跟丢标为成功','跟偏预警':'历史匹配证据变弱，请在原片确认目标位置',
                        '运动突变':'轨迹变化较大，请核对是否为真实运动','低匹配质量':'匹配质量偏低，建议目视复核'}[kind]
                    result.append(ReviewIssue(clip.id,filename,kind,first,last,key,bool(marks.get(key)) if kind!='真实跟丢' else False,reason))
                    if len(result)>=limit:return result
        for frame in sorted(clip.anchors):
            result.append(ReviewIssue(clip.id,filename,'关键帧',int(frame),int(frame),'',False,'人工选区/修正点；不等于问题帧'))
            if len(result)>=limit:return result
    order={clip.id:index for index,clip in enumerate(clips)}
    return sorted(result,key=lambda v:(order[v.clip_id],v.first,v.historical,v.kind))
