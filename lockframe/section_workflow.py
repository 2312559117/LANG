"""Clip-owned independent-section definitions, validation and result freshness."""
from .tracking_sections import normalize_sections, section_anchors


def section_configuration(enabled, sections, frames):
    if type(enabled) is not bool:
        raise ValueError("独立分段跟踪开关无效")
    return enabled, normalize_sections(sections, 0, frames - 1, require_coverage=False)


def active_sections(sections, start, end):
    """Preserve drafts outside the current trim; intersect only for processing."""
    return [{"start": max(start, item["start"]), "end": min(end, item["end"])}
            for item in sections if item["start"] <= end and item["end"] >= start]


def tracking_ranges(enabled, sections, anchors, start, end):
    if not enabled:
        return []
    active = active_sections(sections, start, end)
    if not active:
        raise ValueError("请先建立独立跟踪分段，再为每段框选目标")
    active = normalize_sections(active, start, end)
    section_anchors(active, anchors, start, end)
    return active


def sections_match(track, enabled, sections, start, end):
    if track is None:
        return False
    expected = active_sections(sections, start, end) if enabled else []
    actual = active_sections(track.performance.get("tracking_sections", []), start, end)
    return actual == expected
