"""Per-clip retime settings and output clock, isolated from source tracking time."""
from copy import deepcopy
import time
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel,QMenu
from .time_remap_model import normalize_config
from .time_remap_map import output_frame_map
from .time_remap_ui import TimeRemapPanel
from .retime_preview import RetimePreview
from .widgets import qimage


class RetimeMixin:
    def _install_retime(self):
        from .workspace_ui import ReferenceStep
        self._retime_config={};self._retime_clock='source';self._retime_output_step=0
        self._retime_seeking=False;self._retime_editor=None
        self._retime_cached=None
        self.retime_panel=TimeRemapPanel(self)
        menu=QMenu(self);menu.addAction('完整曲线编辑器…',self._open_retime_editor)
        step=ReferenceStep('04','变速',self.retime_panel,menu)
        from .ui_icons import icon
        step.findChild(QLabel,'sectionIcon').setPixmap(icon('curve',colour='#e88c51',size=24).pixmap(24,24))
        layout=self.sidebar_content.layout();layout.insertWidget(max(0,layout.count()-1),step)
        export=self.sidebar_sections.pop('export');export.findChild(QLabel,'step').setText('05')
        self.sidebar_sections.update(time_remap=step,export=export)
        self.retime_panel.configEdited.connect(self._apply_retime_config)
        self.retime_panel.openEditor.connect(self._open_retime_editor)
        self.retime_panel.clockChanged.connect(self._retime_clock_changed)
        self.retime_panel.presetRequested.connect(self._open_curve_presets)
        self.retime_panel.plot.seekRequested.connect(self._seek_curve_position)
        self._retime_preview=RetimePreview(self);self._retime_preview.ready.connect(self._retime_preview_ready)
        self._retime_debounce=QTimer(self);self._retime_debounce.setSingleShot(True)
        self._retime_debounce.setInterval(140);self._retime_debounce.timeout.connect(self._retime_request_preview)
        from .clip_preview import ClipPreview
        self._clip_preview=ClipPreview(self);self._clip_preview.changed.connect(self._clip_preview_changed)
        self.retime_panel.prepareRequested.connect(self._prepare_clip_preview)
        self.retime_panel.playPrepared.connect(self._play_clip_preview)
        self._retime_play_range=None
        self._retime_sync()

    def _retime_range(self):
        return self.start_spin.value()-1,self.end_spin.value()-1

    def _retime_map(self):
        start,end=self._retime_range()
        key=(id(self._retime_config),start,end)
        if self._retime_cached is None or self._retime_cached[0]!=key:
            self._retime_cached=(key,output_frame_map(self._retime_config or None,max(1,end-start+1)))
        return self._retime_cached[1]

    def _retime_active(self):
        return bool(self.info and self._retime_config.get('enabled') and self._retime_clock=='output')

    def _restore_retime(self,config):
        self._retime_config=deepcopy(config or {})
        self._retime_cached=None
        if hasattr(self,'retime_panel'):
            self._retime_preview.invalidate();self._retime_output_step=0
            self.retime_panel.set_config(config);self._retime_sync()

    def _apply_retime_config(self,config):
        try:
            config=normalize_config(config)
            if self.info:
                start,end=self._retime_range();output_frame_map(config,max(1,end-start+1))
        except ValueError as error:
            self.status_label.setText(str(error));return
        reveal=config['enabled'] and not self._retime_config.get('enabled')
        self._stop_play();self._retime_config=deepcopy(config);self._retime_cached=None
        self.retime_panel.set_config(config);self._retime_preview.invalidate()
        self.dirty=True
        if self.info:
            self._stash_clip();self._retime_source_seek(self.frame_index);self._show_current()
        self._retime_sync()
        if reveal:QTimer.singleShot(0,lambda:self.sidebar_scroll.ensureWidgetVisible(self.retime_panel,0,8))

    def _open_retime_editor(self):
        if not self.info:
            self.status_label.setText('请先导入并选择素材');return
        from .time_remap_editor import TimeRemapEditor
        if self._retime_editor is not None and self._retime_editor.isVisible():
            self._retime_editor.raise_();self._retime_editor.activateWindow();return
        if self._retime_editor is not None:self._retime_editor.deleteLater()
        identity=(self.info.path,self.info.fingerprint,self.active_clip_id)
        start,end=self._retime_range()
        editor=TimeRemapEditor(self._retime_config or None,max(1,end-start+1),self.info.fps,self)
        def apply(config):
            actual=(self.info.path,self.info.fingerprint,self.active_clip_id) if self.info else None
            if identity!=actual or (start,end)!=self._retime_range():
                self.status_label.setText('素材或处理范围已改变，请关闭后重新打开曲线编辑器');return
            self._apply_retime_config(config)
        editor._workspace_context=(self.active_clip_id,self.info.path,self.info.fingerprint,start,end)
        def seek(position):
            actual=(self.info.path,self.info.fingerprint,self.active_clip_id) if self.info else None
            if identity!=actual or (start,end)!=self._retime_range():return
            from .time_remap_model import active_points
            from .time_remap_math import evaluate_source
            source=position if editor.draft['mode']=='basic' else evaluate_source(active_points(editor.draft),position)
            self._stop_play();self._seek(start+round(max(0.,min(1.,source))*max(0,end-start)))
        editor.canvas.seekRequested.connect(seek)
        editor.applied.connect(apply);self._retime_editor=editor;editor.show();self._sync_curve_playheads()

    def _seek_curve_position(self,position):
        if not self.info or self.busy:return
        self._stop_play();start,end=self._retime_range()
        if self._retime_config.get('mode','basic')=='basic':self._seek(start+round(max(0.,min(1.,position))*max(0,end-start)))
        else:
            self.retime_panel.clock.setCurrentIndex(self.retime_panel.clock.findData('output'))
            from .time_remap_model import active_points
            span=active_points(self._retime_config)[-1]['position']
            self._seek_output(round(position/max(.01,span)*max(0,len(self._retime_map())-1)))

    def _sync_curve_playheads(self):
        if not hasattr(self,'retime_panel'):return
        if not self.info:
            self.retime_panel.plot.set_playhead(None);return
        from .time_remap_model import active_points
        start,end=self._retime_range();values=self._retime_map()
        step=max(0,min(len(values)-1,self._retime_output_step))
        source=float(values[step])/max(1,end-start) if self._retime_active() else (self.frame_index-start)/max(1,end-start)
        # Basic curves always span normalized source 0..1. Generating their
        # 200-point time map here previously ran even with retime disabled.
        if self._retime_config.get('mode','basic')=='basic':span=1.
        else:
            if getattr(self,'_curve_span_config',None) is not self._retime_config:
                self._curve_span_config=self._retime_config
                self._curve_span=active_points(self._retime_config)[-1]['position']
            span=self._curve_span
        output=step/max(1,len(values)-1)*span
        self.retime_panel.plot.set_playhead(source if self._retime_config.get('mode','basic')=='basic' else output)
        editor=self._retime_editor
        if editor is not None and editor.isVisible():
            if getattr(editor,'_workspace_context',None)!=(self.active_clip_id,self.info.path,self.info.fingerprint,start,end):
                editor.canvas.set_playhead(None);editor.preview_time.setText('主预览已切换素材；返回原素材后继续编辑，当前草稿保留。');return
            editor.canvas.set_playhead(source if editor.draft['mode']=='basic' else output)
            editor.preview_time.setText(f'主预览：源 {(self.frame_index)/self.info.fps:.2f}s · 输出 {step/self.info.fps:.2f}s（草稿未自动应用）')

    def _retime_clock_changed(self,clock):
        self._stop_play();self._retime_clock=clock
        self._retime_source_seek(self.frame_index)
        if self._retime_active():self._seek_output(self._retime_output_step)
        else:self._show_current()
        self._retime_sync()

    def _retime_source_seek(self,index):
        if hasattr(self,'_retime_config') and not self._retime_seeking and self.info:
            if not self._retime_config.get('enabled'):
                self._retime_output_step=max(0,index-self.start_spin.value()+1);return
            start,_=self._retime_range();values=self._retime_map()
            # Nearest output time to the existing playhead preserves reverse/repeat locality.
            distances=np.abs(values-(index-start));best=np.flatnonzero(distances==distances.min())
            self._retime_output_step=int(best[np.argmin(np.abs(best-self._retime_output_step))])

    def _transport_seek(self,index):
        if self._retime_active():self._seek_output(index)
        else:self._seek(index)

    def _transport_boundary(self,first):
        self._stop_play()
        if self._retime_active():self._seek_output(0 if first else len(self._retime_map())-1)
        else:self._seek(self.strip.state['play_in'] if first else self.strip.state['play_out']-1)

    def _seek_output(self,index):
        values=self._retime_map();self._retime_output_step=max(0,min(len(values)-1,int(index)))
        start,_=self._retime_range();source=start+int(np.floor(values[self._retime_output_step]+.5))
        self._retime_seeking=True
        try:self._seek(source)
        finally:self._retime_seeking=False

    def _retime_step(self,direction):
        if not self._retime_active():return False
        self._stop_play();self._seek_output(self._retime_output_step+direction);return True

    def _retime_toggle_play(self):
        if not self._retime_active():return False
        if self.timer.isActive():self._stop_play()
        else:
            self._retime_play_range=None
            if self._retime_output_step>=len(self._retime_map())-1:self._seek_output(0)
            self._retime_play_first=self._retime_output_step;self._play_started=time.monotonic()
            self.timer.start();self._show_current()
        return True

    def _retime_tick(self):
        if not self._retime_active():return False
        first,last=self._retime_play_range or (0,len(self._retime_map())-1)
        index=self._retime_play_first+int((time.monotonic()-self._play_started)*self.info.fps)
        if self.strip.state['loop']:index=first+(index-first)%(last-first+1)
        index=min(last,index)
        if index!=self._retime_output_step:self._seek_output(index)
        if index>=last and not self.strip.state['loop']:self._stop_play()
        return True

    def _retime_sync(self):
        if not hasattr(self,'retime_panel'):return
        self._sync_curve_playheads()
        self.retime_panel.setEnabled(bool(self.info and not self.busy))
        if self.info:
            codec=f'MP4 / HEVC {self.info.bit_depth} 位' if self.info.hdr or self.info.bit_depth>8 else 'MP4 / H.264'
            audio='保留首条原片音轨' if self.info.has_audio else '无音轨'
            if self._retime_config.get('enabled'):
                from .retime_audio import audio_strategy
                from types import SimpleNamespace
                start,end=self._retime_range()
                strategy=audio_strategy(self.info,SimpleNamespace(start=start,count=end-start+1),self._retime_map(),self._retime_config)
                audio=strategy['reason'] or strategy['policy']
            self.audio_policy_label.setText(codec+' · '+audio)
            self.export_audio_hint.setText('从原片渲染；'+audio+'。')
        if self.info and self._retime_config.get('enabled'):
            values=self._retime_map();start,end=self._retime_range()
            self._retime_output_step=min(self._retime_output_step,len(values)-1)
            self.retime_panel.summary.setText(f'源 {(end-start+1)/self.info.fps:.2f} 秒 → 输出 {len(values)/self.info.fps:.2f} 秒 · {len(values)} 帧\n跟踪与时间线始终按源时间；变速不重跑跟踪。')
        section=self.sidebar_sections['time_remap']
        if hasattr(self,'_clip_preview'):
            self._clip_preview.want(self._clip_preview_key())
            self._update_clip_preview_ui()
        # Qt caches ancestor height-for-width values when a hidden body is
        # revealed or its wrapped summary changes. Refresh them before sizing.
        self.retime_panel.body.layout().invalidate()
        self.retime_panel.layout().invalidate()
        section.layout().invalidate()
        required=section.layout().totalHeightForWidth(max(1,section.width()))
        section.setFixedHeight(max(336,required) if self._retime_config.get('enabled') else 84)

    def _retime_show_current(self):
        if not hasattr(self,'_retime_preview'):return
        self._retime_preview.invalidate();self._retime_debounce.stop()
        if self.timer.isActive():self._sync_curve_playheads()
        else:self._retime_sync()
        if self.info and self.info.hdr:
            self.preview_note.setText(self.preview_note.text()+' · HDR 预览映射为 SDR；导出保留原生位深')
        if not self._retime_active():return
        values=self._retime_map();start,_=self._retime_range();value=float(values[self._retime_output_step])
        self.preview_note.setText(self.preview_note.text()+f' · 输出 {self._retime_output_step/self.info.fps:.3f}s → 源 {(start+value)/self.info.fps:.3f}s')
        cached=self._clip_preview.frame(self._clip_preview.wanted,self._retime_output_step)
        if cached is not None:
            self.canvas.processed_image=qimage(cached);self.canvas.update()
            self.preview_note.setText(self.preview_note.text()+' · 预渲染缓存：同导出光流与构图，当前为显示代理')
            return
        config=normalize_config(self._retime_config)
        if config['optical_flow']:
            if not self.timer.isActive() and config['preview_flow'] and value%1>1e-6:
                self._retime_debounce.start()
            else:self.preview_note.setText(self.preview_note.text()+' · 最近源帧预览；导出按光流设置')

    def _retime_request_preview(self):
        if not self._retime_active() or self.busy or self.timer.isActive() or self._clip_preview.running:return
        start,_=self._retime_range();value=float(self._retime_map()[self._retime_output_step]);left=int(np.floor(value))
        if value-left<1e-6:return
        boundaries={int(s['start']) for s in self.track.performance.get('tracking_sections',[])[1:]} if self.track else set()
        if start+left+1 in boundaries:
            self.preview_note.setText(self.preview_note.text()+' · 独立分段边界：显示最近源帧，不跨目标合成')
            return
        token=(self.info.path,self.info.fingerprint,start,self._retime_output_step,id(self.plan),self._retime_preview.generation)
        self._retime_preview.request(token,self.info,self.plan,start+left,value-left,deepcopy(self._retime_config))

    def _retime_preview_ready(self,token,image,message):
        if token!=self._retime_preview.wanted or self.timer.isActive() or not self._retime_active():return
        if image is not None:self.canvas.processed_image=qimage(image);self.canvas.update()
        self.preview_note.setText(self.preview_note.text()+' · '+message)

    def _clip_preview_key(self):
        from .tracking_pause import is_complete
        if (not self.info or not self._retime_config.get('enabled') or self.busy or
                not is_complete(self.track) or self.track.lost_frames or self._region_stale()):return None
        start,end=self._retime_range()
        if (start,end)!=(self.track.start,self.track.end):return None
        return self._clip_preview.key(self.info,self.track,self.settings(),start,end,self.anchors,self.active_clip_id)

    def _update_clip_preview_ui(self):
        service=self._clip_preview;panel=self.retime_panel
        panel.prepare_clip.setText('取消预渲染' if service.running else '预渲染当前3秒')
        panel.prepare_clip.setEnabled(service.running or service.wanted is not None)
        panel.play_clip.setEnabled(service.cache is not None and service.wanted is not None and not service.running)
        panel.clip_status.setText(service.message or '完成跟踪后，可预渲染检查真实光流效果。')

    def _prepare_clip_preview(self):
        service=self._clip_preview
        if service.running:service.cancel();return
        key=self._clip_preview_key()
        if key is None:
            self.status_label.setText('请先完成跟踪并修正跟丢，再预渲染效果');return
        self._stop_play()
        self.retime_panel.clock.setCurrentIndex(self.retime_panel.clock.findData('output'))
        first=self._retime_output_step;mapping=self._retime_map()
        stop=min(len(mapping),first+max(1,round(self.info.fps*3)))
        self._retime_debounce.stop();self._retime_preview.invalidate()
        service.prepare(key,self.info,self.track,self.settings(),mapping[first:stop],first)

    def _clip_preview_changed(self):
        self._update_clip_preview_ui()
        if self._clip_preview.cache is not None and not self.timer.isActive() and not self.busy:
            self._show_current()

    def _play_clip_preview(self):
        service=self._clip_preview
        if service.cache is None or service.wanted!=self._clip_preview_key():return
        self._stop_play()
        self.retime_panel.clock.setCurrentIndex(self.retime_panel.clock.findData('output'))
        first,last=service.cache['first'],service.cache['last']
        self._seek_output(first);self._retime_play_range=(first,last)
        self._retime_play_first=first;self._play_started=time.monotonic()
        self.timer.start();self._show_current()
