"""Independent source-frame sections for one contiguous tracking result."""


def normalize_sections(sections, start, end, *, require_coverage=True):
    """Validate inclusive sections; an empty value retains whole-range behavior.

    Draft UI sections may have gaps, but may never overlap or leave the clip's
    processing range. Tracking requires every frame to belong to one section.
    """
    if sections is None:
        return []
    if not isinstance(sections, (list, tuple)):
        raise ValueError("跟踪分段格式无效，请重新设置分段")
    canonical = []
    for index, section in enumerate(sections, 1):
        if not isinstance(section, dict):
            raise ValueError(f"第 {index} 段格式无效")
        first, last = section.get("start"), section.get("end")
        if type(first) is not int or type(last) is not int or first > last:
            raise ValueError(f"第 {index} 段起止帧无效")
        if first < start or last > end:
            raise ValueError(f"第 {index} 段超出处理范围（{start + 1}–{end + 1} 帧）")
        canonical.append({"start": first, "end": last})
    canonical.sort(key=lambda value: value["start"])
    for previous, current in zip(canonical, canonical[1:]):
        if current["start"] <= previous["end"]:
            raise ValueError("跟踪分段不能重叠，请调整分段边界")
        if require_coverage and current["start"] != previous["end"] + 1:
            raise ValueError("跟踪分段之间有未覆盖帧，请补齐后再跟踪")
    if require_coverage and canonical and (canonical[0]["start"] != start or canonical[-1]["end"] != end):
        raise ValueError("跟踪分段必须完整覆盖处理范围，请补齐首尾分段")
    return canonical


def section_anchors(sections, anchors, start, end):
    """Return each section and its own manual seeds, never borrowing a seed."""
    ranges = sections or [{"start": start, "end": end}]
    groups = []
    for index, section in enumerate(ranges, 1):
        active = sorted(frame for frame in anchors if section["start"] <= frame <= section["end"])
        if not active:
            if sections:
                raise ValueError(f"第 {index} 段（{section['start'] + 1}–{section['end'] + 1} 帧）尚未设置跟踪目标，请在该段内框选目标")
            raise ValueError("请在处理区间内框选一个跟踪目标")
        groups.append((section, active))
    return groups


def section_walks(groups):
    """Build existing reference walks, bounded independently by each section."""
    walks = []
    for section, active in groups:
        if active[0] > section["start"]:
            walks.append((active[0], section["start"], -1))
        for index, seed in enumerate(active):
            destination = active[index + 1] - 1 if index + 1 < len(active) else section["end"]
            if destination > seed:
                walks.append((seed, destination, 1))
    return walks
