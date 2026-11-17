"""Conservative evidence accumulation; suspicion is never a measured lost state."""
from collections import deque
import math
import numpy as np

POLICIES={'off':None,'relaxed':(.52,10),'standard':(.62,6),'strict':(.74,4)}
REVIEW_REVISION=3
QUALITY_DROP={'relaxed':.16,'standard':.12,'strict':.08}


def review_ranges(values,start,end):
    """Bounded diagnostic metadata from projects/checkpoints, never geometry."""
    if not isinstance(values,list):return []
    result=[]
    for item in values[:1000]:
        if not isinstance(item,dict):continue
        first,last,frame=item.get('start'),item.get('end'),item.get('frame')
        if (any(type(v) is not int for v in (first,last,frame)) or
                not start<=first<=frame<=last<=end):continue
        score=item.get('confidence',0.)
        if not isinstance(score,(float,int)) or not math.isfinite(score) or not 0<=score<=1:continue
        cleaned=dict(start=first,end=last,frame=frame,confidence=float(score),
            reason=str(item.get('reason','连续匹配证据偏弱'))[:120],
            policy=str(item.get('policy','standard'))[:16],confirmed_lost=False)
        baseline=item.get('baseline_confidence')
        if type(baseline) in (float,int) and math.isfinite(baseline) and 0<=baseline<=1:
            cleaned['baseline_confidence']=float(baseline)
        result.append(cleaned)
    return result


def retained_ranges(values,keep,start,end):
    """Keep measured, unaffected portions instead of dropping a whole warning."""
    flags=np.asarray(keep,dtype=bool)
    if flags.shape!=(end-start+1,):raise ValueError('预警复用位图与范围不一致')
    edges=np.flatnonzero(np.diff(np.r_[False,flags,False]))
    firsts=start+edges[::2];lasts=start+edges[1::2]-1;result=[]
    for item in review_ranges(values,start,end):
        index=int(np.searchsorted(lasts,item['start']))
        while index<len(firsts) and firsts[index]<=item['end']:
            a=max(item['start'],int(firsts[index]));b=min(item['end'],int(lasts[index]))
            result.append(dict(item,start=a,end=b,frame=max(a,min(b,item['frame']))))
            if len(result)>=1000:return result
            index+=1
    return result


def merge_range(values,event):
    candidate=next((item for item in reversed(values)
        if item['start']<=event['end'] and event['start']<=item['end']),None)
    if candidate is None:
        candidate=dict(event)
        if len(values)<1000:values.append(candidate)
    else:
        a=min(candidate['start'],event['start']);b=max(candidate['end'],event['end'])
        candidate.update(event);candidate.update(start=a,end=b)
    return candidate


class DriftReview:
    def __init__(self,policy='off'):
        if not isinstance(policy,str) or policy not in POLICIES:raise ValueError('未知的漂移复核灵敏度')
        self.policy=policy;self.history=deque(maxlen=12);self.streak=[];self.reported=False
        self.stable=deque(maxlen=24);self.warning=None;self.new_warning=None
        self.baseline=None

    def snapshot(self):
        return dict(revision=REVIEW_REVISION,policy=self.policy,
            history=[[int(frame),[float(v) for v in position]] for frame,position in self.history],
            stable=[float(v) for v in self.stable],baseline=self.baseline,
            streak=list(self.streak),reported=self.reported,
            warning=dict(self.warning) if self.warning is not None else None)

    def restore(self,value,*,start,end,frame,direction):
        """Validate before mutating; malformed metadata requires a safe rebuild."""
        if (not isinstance(value,dict) or value.get('revision')!=REVIEW_REVISION or
                value.get('policy')!=self.policy or type(frame) is not int or
                type(direction) is not int or direction not in (-1,1) or not start<=frame<=end):return False
        history=value.get('history');stable=value.get('stable');streak=value.get('streak')
        if (not isinstance(history,list) or len(history)>12 or not isinstance(stable,list) or len(stable)>24 or
                not isinstance(streak,list) or len(streak)>(POLICIES[self.policy] or (0,0))[1] or
                type(value.get('reported')) is not bool):return False
        def number(v):return type(v) in (int,float) and math.isfinite(v) and 0<=v<=1
        if any(not number(v) for v in stable):return False
        for item in history:
            if (not isinstance(item,list) or len(item)!=2 or type(item[0]) is not int or not start<=item[0]<=end or
                    not isinstance(item[1],list) or len(item[1])!=2 or not all(number(v) for v in item[1])):return False
        frames=[v[0] for v in history]
        if not frames and stable:return False
        if frames and (frames[-1]!=frame or any(b-a!=direction for a,b in zip(frames,frames[1:]))):return False
        if any(type(v) is not int or not start<=v<=end for v in streak):return False
        if streak and (not frames or streak[-1]!=frame or any(b-a!=direction for a,b in zip(streak,streak[1:]))):return False
        baseline=value.get('baseline')
        if baseline is None:
            if len(stable)>=5:return False
        elif not number(baseline) or len(stable)<5 or baseline+1e-9<float(np.median(stable)):return False
        count=(POLICIES[self.policy] or (0,0))[1]
        if value['reported'] and len(streak)<count:return False
        if count and len(streak)==count and not value['reported']:return False
        warning=value.get('warning');clean=None
        if warning is not None:
            if not isinstance(warning,dict) or not number(warning.get('confidence')):return False
            cleaned=review_ranges([warning],start,end)
            if (len(cleaned)!=1 or warning.get('policy')!=self.policy or warning.get('frame')!=frame or
                    len(streak)<max(2,(count+1)//2) or
                    (warning.get('start'),warning.get('end'))!=(min(streak),max(streak))):return False
            if 'baseline_confidence' in warning and (not number(warning['baseline_confidence']) or
                    baseline is None or abs(warning['baseline_confidence']-baseline)>1e-9):return False
            clean=cleaned[0]
        elif len(streak)>=max(2,(count+1)//2):return False
        self.history=deque(((f,np.array(p,dtype=float)) for f,p in history),maxlen=12)
        self.stable=deque((float(v) for v in stable),maxlen=24)
        self.baseline=None if baseline is None else float(baseline)
        self.streak=list(streak);self.reported=value['reported'];self.warning=clean;self.new_warning=None
        return True

    def observe(self,frame,score,state,position,box):
        self.new_warning=None
        if self.policy=='off':return None
        score=float(score);position=np.asarray(position,dtype=float)
        box=np.asarray(box,dtype=float)
        if state in ('manual','lost') or not math.isfinite(score):
            self.history.clear();self.stable.clear();self.streak=[];self.reported=False
            self.warning=None;self.baseline=None;return None
        limit,count=POLICIES[self.policy]
        # Subject movement alone is legitimate. Only weak matching evidence
        # can produce suspicion; never flag a high-confidence moving target.
        weak=score<limit
        reason='连续匹配证据偏弱'
        # Establish a measured baseline first; never teach a decline back as
        # the new normal. A high-confidence moving subject alone is not drift.
        baseline=self.baseline
        if baseline is not None and score<.90 and score<=baseline-QUALITY_DROP[self.policy]+1e-9:
            weak=True;reason='匹配质量相对稳定阶段持续下降'
        if state=='recovered' and score<min(.85,limit+.20):
            weak=True;reason='连续重捕获，匹配不稳定'
        if self.history and score<min(.82,limit+.12):
            previous=self.history[-1][1]
            scale=max(.005,float(np.linalg.norm(box[2:])))
            if float(np.linalg.norm(position-previous))/scale>.35:
                weak=True;reason='弱匹配伴随目标位置突变'
        self.history.append((frame,position.copy()))
        if weak:self.streak.append(int(frame))
        else:
            self.streak=[];self.reported=False;self.warning=None
            if score>=max(.80,limit+.10):self.stable.append(score)
            if len(self.stable)>=5:
                measured=float(np.median(self.stable))
                self.baseline=measured if self.baseline is None else max(self.baseline,measured)
        if len(self.streak)>count:self.streak=self.streak[-count:]
        if len(self.streak)>=max(2,(count+1)//2):
            first=self.warning is None
            self.warning=dict(start=min(self.streak),end=max(self.streak),frame=int(frame),
                reason=reason,policy=self.policy,confidence=score,confirmed_lost=False)
            if baseline is not None:self.warning['baseline_confidence']=baseline
            if first:self.new_warning=dict(self.warning)
        if len(self.streak)>=count and not self.reported:
            self.reported=True
            return dict(self.warning)
        return None
