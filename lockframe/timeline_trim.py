"""Non-destructive source-range edits; only reuse actually measured samples."""
from dataclasses import replace
from copy import deepcopy

from .regions import shape_matches, anchors_match
from .tracking_pause import computed_mask
from .quality import review_track


def crop_measured_result(result, start, end, settings, anchors):
    """Closed source range. Return None when any requested pose is unmeasured."""
    if type(start) is not int or type(end) is not int or not 0<=start<=end:return None
    if result is None or start < result.start or end > result.end:
        return None
    if not shape_matches(result, settings) or not anchors_match(result, anchors, result.start, result.end):
        return None
    if result.tracking_mode!=settings.tracking_mode:return None
    for name,default in (('perspective',False),('smoothness',3),('adaptive_reference',False),('planar_region',False)):
        if result.performance.get(name,default)!=getattr(settings,name):return None
    part = slice(start-result.start, end-result.start+1)
    if not computed_mask(result)[part].all():
        return None
    if (start,end)==(result.start,result.end) and result.performance.get('complete',True):
        return result
    performance=deepcopy(result.performance)
    for key in ('tracking_pause','computed_frames'):
        performance.pop(key,None)
    performance.update(complete=True,trimmed_from=[result.start,result.end],
                       statistics_scope='original tracking run; retained measured samples only')
    cropped=replace(result,start=start,end=end,positions=result.positions[part].copy(),
                    boxes=result.boxes[part].copy(),confidence=result.confidence[part].copy(),
                    states=list(result.states[part]),performance=performance,
                    anchors=None if result.anchors is None else {f:tuple(box) for f,box in result.anchors.items() if start<=f<=end},
                    homographies=None if result.homographies is None else result.homographies[part].copy(),
                    confidence_grids=None if result.confidence_grids is None else result.confidence_grids[part].copy())
    performance['quality']=review_track(cropped)
    return cropped


def retained_bytes(snapshot):
    """Conservative history accounting for referenced numeric snapshots."""
    total=512
    for result in (snapshot['track'],snapshot['previous']):
        if result is not None:
            total+=sum(getattr(getattr(result,name,None),'nbytes',0) for name in
                       ('positions','boxes','confidence','homographies','confidence_grids'))+result.count*16
    return total
