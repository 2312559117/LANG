"""Source-frame navigation only; never retime footage or synthesize frames."""
from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class Timebase:
    num: int
    den: int = 1

    def __post_init__(self):
        if type(self.num) is not int or type(self.den) is not int or min(self.num, self.den) <= 0:
            raise ValueError('帧率必须是正有理数')

    @classmethod
    def from_fps(cls, fps):
        if not math.isfinite(fps) or fps <= 0: raise ValueError('无效帧率')
        for num in (24000, 30000, 60000, 120000):
            if abs(fps-num/1001) < .0001: return cls(num, 1001)
        value=Fraction(str(fps)).limit_denominator(100000)
        return cls(value.numerator, value.denominator)

    def timecode(self, frame):
        if type(frame) is not int or frame < 0: raise ValueError('帧号必须是非负整数')
        nominal=max(1,round(self.num/self.den))
        drop=(nominal//15) if self.den==1001 and self.num in (30000,60000) else 0
        if drop:
            ten=nominal*600-drop*9; minute=nominal*60-drop
            blocks,rest=divmod(frame,ten)
            frame += drop*9*blocks+drop*max(0,(rest-drop)//minute)
        seconds,ff=divmod(frame,nominal); minutes,ss=divmod(seconds,60); hh,mm=divmod(minutes,60)
        return f'{hh:02}:{mm:02}:{ss:02}{";" if drop else ":"}{ff:02}'


def clean_state(raw, frames, start=0, end=None):
    """Strict integer playback boundaries; discard only invalid optional UI state."""
    end=frames-1 if end is None else end
    default=dict(schema=1, play_in=start, play_out=end+1, custom=False, loop=False,
                 scale=0., scroll=0., timecode=False)
    if not isinstance(raw,dict) or raw.get('schema',1)!=1: return default
    a,b=raw.get('play_in',start),raw.get('play_out',end+1)
    if type(a) is not int or type(b) is not int or not 0<=a<b<=frames: return default
    default.update(play_in=a,play_out=b,custom=raw.get('custom') is True,
                   loop=raw.get('loop') is True,timecode=raw.get('timecode') is True)
    for key in ('scale','scroll'):
        v=raw.get(key,0.)
        if type(v) in (int,float) and math.isfinite(v) and v>=0:
            default[key]=min(float(v),64. if key=='scale' else max(0,frames-1))
    return default


def shifted_keys(anchors, selected, delta, start, end):
    selected=set(selected)&anchors.keys()
    if not selected:return dict(anchors)
    if max(selected)-min(selected)>end-start:raise ValueError('选中关键帧跨度超出处理范围，未作修改')
    delta=max(start-min(selected),min(end-max(selected),int(delta)))
    targets={f+delta for f in selected}
    if targets & (anchors.keys()-selected):raise ValueError('目标帧已有修正关键帧，请先移开或删除')
    result={f:box for f,box in anchors.items() if f not in selected}
    result.update({f+delta:anchors[f] for f in selected})
    return result


class EditHistory:
    """Bounded incremental edits. Playback/selection/view updates are not commands."""
    def __init__(self, limit=200, budget=50*1024*1024):
        self.undo_stack=[];self.redo_stack=[];self.limit=limit;self.budget=budget

    def push(self, command):
        self.redo_stack.clear();self.undo_stack.append(command)
        while len(self.undo_stack)>self.limit or sum(c.get('bytes',256) for c in self.undo_stack)>self.budget:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:return None
        command=self.undo_stack.pop();self.redo_stack.append(command);return command

    def redo(self):
        if not self.redo_stack:return None
        command=self.redo_stack.pop();self.undo_stack.append(command);return command

    def clear(self):
        self.undo_stack.clear();self.redo_stack.clear()
