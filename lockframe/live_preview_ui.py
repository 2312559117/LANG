"""Display measured tracker observations without editing tracking workspaces."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .logging import get_logger
from .media import FrameReader
from .models import source_identity


def _timestamp(frame, fps):
    seconds = max(0, frame / fps)
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


class LivePreviewMixin:
    def _set_drift_warning(self,warning=None,*,message=None,paused=False):
        banner=getattr(self,'drift_warning_banner',None)
        if banner is None:return
        if warning is None and not message:
            if banner.text():banner.clear()
            if not banner.isHidden():banner.hide()
            return
        if message is None:
            frame=int(warning['frame'])+1
            prefix='已暂停 · 疑似跟偏待复核' if paused else '跟偏预警'
            message=f'{prefix} · 第 {frame} 帧 · {str(warning.get("reason","匹配证据正在变弱"))[:120]}'
            if not paused:message+='；可点击暂停修正'
        if banner.text()!=message:banner.setText(message)
        if banner.isHidden():banner.show()

    def _refresh_drift_warning(self):
        from .drift_review import review_ranges
        context=getattr(self,'_paused_tracking',None)
        if context and context.get('reason')=='review' and not context.get('corrected') and self.frame_index==context['frame']:
            self._set_drift_warning(dict(frame=self.frame_index,reason='重新框选，或确认目标正确后继续'),paused=True)
            return
        ranges = self.track.performance.get('early_warning_ranges', []) if self.track is not None else []
        if ranges and not self._region_stale():
            from .tracking_pause import computed_mask
            index=self.frame_index-self.track.start
            if 0 <= index < self.track.count and (self.track.performance.get('complete', True)
                                                  or computed_mask(self.track)[index]):
                for item in review_ranges(ranges, self.track.start, self.track.end):
                    if item['start'] <= self.frame_index <= item['end']:
                        self._set_drift_warning(message=f'此区间曾触发跟偏预警 · 第 {self.frame_index+1} 帧 · {item["reason"]}；请目视复核')
                        return
        self._set_drift_warning()

    def _begin_live_preview(self):
        if hasattr(self,'_autosave'):self._autosave.request(force=True)
        self._set_drift_warning()
        original_preview = getattr(self, "_original_preview", None)
        if original_preview is not None:
            original_preview.clear()
        self._close_foreign_preview()
        self._live_preview_previous_mode = self.canvas.mode
        self._live_preview_last_frame = None

    def _on_tracking_frame(self, payload):
        check = getattr(self, "live_preview_check", None)
        if (check is not None and not check.isChecked()) or not self.info or not self.reader:
            return
        if payload.get("source_identity", source_identity(self.info)) != source_identity(self.info):
            return
        self._display_live_observation(payload, self.info, self.reader)

    def _display_live_observation(self, payload, info, reader, *, foreign=False):
        index = int(payload["frame"])
        if not 0 <= index < info.frames:
            return
        try:
            frame = reader.read(index)
        except Exception:
            # Preview failure must not abort an otherwise valid analysis job.
            get_logger("preview").exception("实时跟踪预览无法读取第 %s 帧", index + 1)
            return
        if not foreign:
            self.frame, self.frame_index = frame, index
        self._live_preview_last_frame = index
        self._display_info = info
        if hasattr(self, "asset_status_bar"):
            self.asset_status_bar.set_info(info, context="分析实况")
        if hasattr(self, "monitor_filename"):
            self.monitor_filename.setText(Path(info.path).name)
        # Future-frame smoothing and safe zoom need the completed trajectory.
        # During analysis show the source and its actual measured overlay.
        self.canvas.mode = "source"
        for key, button in getattr(self, "view_buttons", {}).items():
            button.setChecked(key == "source")
        if not foreign:
            self.timeline.blockSignals(True)
            self.timeline.setValue(index)
            self.timeline.blockSignals(False)
        self.canvas.set_frames(frame, box=payload["box"], confidence=payload["confidence"],
                               state=payload["state"], preserve_view=True,
                               homography=None if payload.get("homography") is None else
                               np.asarray(payload["homography"], dtype=float),
                               confidence_grid=np.asarray(payload["grid"], dtype=float)
                               if payload.get("grid") is not None and self.mesh_overlay_check.isChecked() else None)
        label = {"manual": "关键帧", "tracked": "稳定跟踪", "recovered": "重新匹配", "lost": "跟丢 · 待修正"}.get(
            payload["state"], payload["state"])
        self.preview_note.setText(f"分析实况 · {Path(info.path).name} · 第 {index + 1} 帧 · {label} · "
                                  f"已分析 {payload['completed']}/{payload['total']} 帧 · 最多 8 次/秒刷新")
        self._set_drift_warning(payload.get('drift_warning'))
        if hasattr(self, "live_preview_status"):
            self.live_preview_status.setText(f"正在计算：{Path(info.path).name} · 第 {index + 1} 帧 · {label}")
        self.time_label.setText(f"{_timestamp(index, info.fps)} / "
                                f"{_timestamp(info.frames, info.fps)}  ·  {index + 1} 帧")
        if not foreign:
            self._update_strip()
        if getattr(self, '_ad_installed', False):
            from .ad_workspace import ad_sync
            ad_sync(self)

    def _on_queue_tracking_frame(self, clip_id, payload):
        # Opt-in queue following paints one worker at a time while retaining the
        # active workspace's source, timeline, anchors and completed trajectory.
        if not getattr(self, "_live_queue_follow", False):
            if clip_id == self.active_clip_id:
                self._on_tracking_frame(payload)
            return
        check = getattr(self, "live_preview_check", None)
        if check is not None and not check.isChecked():
            return
        if clip_id in getattr(self, "_live_completed_ids", set()):
            return
        followed = getattr(self, "_live_follow_clip_id", None)
        if followed is None:
            self._live_follow_clip_id = followed = getattr(self, "_queue_panel_clip_id", None) or clip_id
        if clip_id != followed:
            return
        if clip_id == self.active_clip_id:
            self._on_tracking_frame(payload)
            return
        clip = self._clip(clip_id)
        if clip is None or payload.get("source_identity", source_identity(clip.info)) != source_identity(clip.info):
            return
        cached = getattr(self, "_live_foreign_reader", None)
        if cached is None or cached[0] != clip_id:
            self._close_foreign_preview()
            try:
                cached = self._live_foreign_reader = (clip_id, FrameReader(clip.info.proxy))
            except Exception:
                get_logger("preview").exception("无法打开队列分析实况：%s", Path(clip.info.path).name)
                return
        self._display_live_observation(payload, clip.info, cached[1], foreign=True)

    def _close_foreign_preview(self):
        cached = getattr(self, "_live_foreign_reader", None)
        if cached is not None:
            cached[1].close()
            self._live_foreign_reader = None

    def _end_live_preview(self):
        self._close_foreign_preview()
        self._live_queue_follow = False
        self._live_follow_clip_id = None
        if hasattr(self, "live_preview_status"):
            self.live_preview_status.setText("分析任务已结束 · 可在复核页检查目标，或继续调整构图与时间。")
        previous = getattr(self, "_live_preview_previous_mode", None)
        if previous is not None:
            self.canvas.mode = previous
            for key, button in getattr(self, "view_buttons", {}).items():
                button.setChecked(key == previous)
            self._live_preview_previous_mode = None
            # A followed clip never replaced the active workspace frame. Reload
            # it so the canvas and source time label both return to that clip.
            if self.info and self.reader:
                self._seek(self.frame_index)
