"""Adapt the source timebar to the existing clip/ROI/pause/export workflow."""
from .timebar_core import EditHistory, shifted_keys


class TimebarMixin:
    def _tb_history(self):
        if not hasattr(self,'_timebar_histories'):self._timebar_histories={}
        return self._timebar_histories.setdefault(self.active_clip_id,EditHistory())

    def _tb_clear_history(self):
        if hasattr(self,'_timebar_histories'):self._timebar_histories.pop(self.active_clip_id,None)
        if hasattr(self,'strip'):self.strip.cancel_drag()

    def _tb_trim_snapshot(self):
        return dict(start=self.start_spin.value()-1,end=self.end_spin.value()-1,
                    track=self.track,previous=self._retrack_previous,frame=self.frame_index,
                    play=dict(self.strip.state),paused=self._paused_tracking,
                    queued=self.active_clip_id in self._paused_queue_ids,
                    queued_index=self._paused_queue_ids.index(self.active_clip_id) if self.active_clip_id in self._paused_queue_ids else 0)

    def _tb_restore_trim(self,snapshot):
        self._stop_play()
        self._ready=False
        try:
            self.start_spin.setValue(snapshot['start']+1);self.end_spin.setValue(snapshot['end']+1)
        finally:self._ready=True
        self.track,self._retrack_previous=snapshot['track'],snapshot['previous']
        self.strip.state=dict(snapshot['play']);self.strip.cancel_drag()
        self._paused_tracking=snapshot['paused'];self._resume_requested=False
        self._paused_queue_ids=[i for i in self._paused_queue_ids if i!=self.active_clip_id]
        clip=self._clip()
        if self._paused_tracking:
            self._paused_by_clip[self.active_clip_id]=self._paused_tracking
            if snapshot['queued']:self._paused_queue_ids.insert(min(snapshot.get('queued_index',0),len(self._paused_queue_ids)),self.active_clip_id)
            if clip:clip.tracking_resume=self._paused_tracking.get('resume')
        else:
            self._paused_by_clip.pop(self.active_clip_id,None)
            if clip:clip.tracking_resume=None
        self.plan=None;self.dirty=True;self._sequence_changed=True
        self._refresh_anchors();self._refresh_issues();self._rebuild_plan();self._update_strip()
        self._seek(snapshot['frame']);self._sync_ui()
        if self._paused_tracking:self._restore_pause_context(self.track)
        self._stash_clip();self._refresh_sequence()

    def _tb_trim_change(self,start,exclusive_end,original_frame=None):
        if not self.info or self.busy or self.worker is not None:return False
        if not (type(start) is int and type(exclusive_end) is int and 0<=start<exclusive_end<=self.info.frames):
            self.status_label.setText('裁剪范围无效：至少保留一帧，且不能超出原片');return False
        if (start,exclusive_end-1)==(self.start_spin.value()-1,self.end_spin.value()-1):return False
        from .timeline_trim import crop_measured_result,retained_bytes
        before=self._tb_trim_snapshot()
        if original_frame is not None:before["frame"]=max(0,min(self.info.frames-1,int(original_frame)))
        candidate=self.track if self.track is not None else self._retrack_previous
        track=crop_measured_result(candidate,start,exclusive_end-1,self.settings(),self.anchors)
        after=dict(before,start=start,end=exclusive_end-1,track=track,
                   previous=candidate,paused=None,queued=False,
                   frame=max(start,min(exclusive_end-1,self.frame_index)))
        after['play']=dict(before['play'],play_in=start,play_out=exclusive_end,custom=False)
        self._tb_restore_trim(after)
        self._tb_history().push(dict(kind='trim',before=before,after=after,label='裁剪视频保留范围',
                                     bytes=retained_bytes(before)+retained_bytes(after)))
        self.tracking_progress_panel.reset()
        self._sync_ui()
        state='已复用保留区间的实测轨迹' if track is not None else '保留范围需跟踪；未测量帧不会当成成功'
        self.status_label.setText(f'保留 {(exclusive_end-start)/self.info.fps:.3f} 秒 · {exclusive_end-start} 帧 · '+state)
        return True

    def _tb_view_changed(self):
        if self.info and not self.busy:self.dirty=True;self._sequence_changed=True

    def _tb_before_roi(self):
        return (dict(self.anchors),self.track,self._retrack_previous)

    def _tb_finish_roi(self,before,label='修改修正关键帧'):
        old,track,previous=before
        changes={f:(old.get(f),self.anchors.get(f)) for f in old.keys()|self.anchors.keys()
                 if old.get(f)!=self.anchors.get(f)}
        if changes:
            self._tb_history().push(dict(kind='anchors',label=label,changes=changes,
                before=(track,previous),after=(self.track,self._retrack_previous),bytes=len(changes)*192+256))
            self.strip.sync_data()

    def _tb_set_anchors(self,anchors,selected=None):
        self.anchors=dict(anchors)
        if self.track is not None:self._retrack_previous=self.track
        self.track=self._paused_tracking['result'] if self._paused_tracking else None
        self._tb_refresh_anchor_edit()
        if selected is not None:self.strip.selected=set(selected)&self.anchors.keys();self.strip.update()

    def _tb_refresh_anchor_edit(self):
        if self._paused_tracking:
            frame=self._paused_tracking['frame']
            self._paused_tracking['corrected']=(frame in self.anchors and
                tuple(self.anchors[frame])!=tuple(self._paused_tracking['anchors'].get(frame,())))
        self.plan=None;self.dirty=True;self._sequence_changed=True
        self._refresh_anchors();self._refresh_issues();self._rebuild_plan();self._show_current();self._update_strip();self._sync_ui()

    def _tb_play_change(self,updates):
        keys=('play_in','play_out','custom','loop')
        before={k:self.strip.state[k] for k in keys};after=dict(before);after.update(updates)
        if not 0<=after['play_in']<after['play_out']<=self.info.frames:
            self.status_label.setText('播放入点必须早于出点，且至少保留一帧');return
        if before==after:return
        self._stop_play();self.strip.state.update(after)
        self._tb_history().push(dict(kind='play',before=before,after=after,label='修改播放范围/循环'))
        self._tb_view_changed();self.strip.update();self._tb_update_labels()

    def _tb_update_labels(self):
        if not self.info:return
        state=self.strip.state
        self.loop_button.blockSignals(True);self.loop_button.setChecked(state['loop']);self.loop_button.blockSignals(False)
        self.time_units.blockSignals(True);self.time_units.setCurrentIndex(1 if state['timecode'] else 0);self.time_units.blockSignals(False)
        self.play_range_button.setToolTip(f"播放范围：第 {state['play_in']+1}–{state['play_out']} 帧（仅预览）\n处理/跟踪/导出范围：第 {self.start_spin.value()}–{self.end_spin.value()} 帧\nI/O设播放入出点；Shift+I/O清除；Home/End跳转。")
        if hasattr(self,'timeline_trim_label'):
            count=self.end_spin.value()-self.start_spin.value()+1
            self.timeline_trim_label.setText(f'保留 {count/self.info.fps:.2f} 秒 · {count} 帧')
            self.timeline_trim_label.setToolTip('拖动V1视频条两端裁剪时长；Ctrl+Z撤销。仅保留范围参与跟踪/导出。\n'+self.trim_summary.text())
        if state['timecode']:
            self.time_label.setText(f'{self.strip.tb.timecode(self.frame_index)} · {self.frame_index+1} / {self.info.frames} 帧')
        else:self.time_label.setText(f'{self.frame_index+1} / {self.info.frames} 帧 · {self.frame_index/self.info.fps:.3f} 秒')

    def _tb_units(self,index):
        self.strip.state['timecode']=index==1;self._tb_view_changed();self.strip.update();self._tb_update_labels()

    def _tb_menu(self):
        from PySide6.QtWidgets import QMenu
        menu=QMenu(self)
        for title,command in (('当前帧设为裁剪入点  [','trim_in'),('当前帧设为裁剪出点  ]','trim_out'),('恢复完整原片范围','trim_reset'),('当前帧设为预览入点','preview_in'),('当前帧设为预览出点','preview_out'),
                              ('清除播放范围（恢复全片）','clear_range'),('跳到播放入点','goto_in'),('跳到播放出点','goto_out'),
                              ('上一修正关键帧  ,','previous_key'),('下一修正关键帧  .','next_key'),
                              ('添加当前帧修正点','add'),('撤销时间线编辑','undo'),('重做时间线编辑','redo')):
            action=menu.addAction(title);action.triggered.connect(lambda checked=False,c=command:self._tb_operation(c,None))
        menu.exec(self.play_range_button.mapToGlobal(self.play_range_button.rect().bottomLeft()))

    def _tb_operation(self,command,data=None):
        if not self.info or self.busy or self.worker is not None:return
        if command not in ('play','loop'):self._stop_play()
        try:
            state=self.strip.state;frame=self.frame_index
            if command in ('in','out') and self.section_tracking:
                (self._sections_panel.mark_in if command=='in' else self._sections_panel.mark_out)()
                return
            command={'preview_in':'in','preview_out':'out'}.get(command,command)
            if command=='play':self._toggle_play();return
            if command=='loop':self._tb_play_change(dict(loop=not state['loop']));return
            if command in ('trim','trim_in','trim_out','trim_reset'):
                a,b=self.start_spin.value()-1,self.end_spin.value()
                if command=='trim':a,b=data[:2]
                elif command=='trim_in':a=min(frame,b-1)
                elif command=='trim_out':b=max(a+1,frame+1)
                else:a,b=0,self.info.frames
                self._tb_trim_change(a,b,data[2] if command=='trim' and len(data)>2 else None);return
            if command in ('in','out','clear_in','clear_out','clear_range','play_range'):
                a,b=state['play_in'],state['play_out']
                if command=='in':a=frame
                elif command=='out':b=frame+1
                elif command=='clear_in':a=0
                elif command=='clear_out':b=self.info.frames
                elif command=='clear_range':a,b=0,self.info.frames
                else:a,b=data
                self._tb_play_change(dict(play_in=a,play_out=b,custom=True));return
            if command in ('goto_in','goto_out'):self._seek(state['play_in'] if command=='goto_in' else state['play_out']-1);return
            if command in ('previous_key','next_key'):
                keys=sorted(self.anchors);values=[f for f in keys if f<frame] if command=='previous_key' else [f for f in keys if f>frame]
                if values:self._seek(values[-1] if command=='previous_key' else values[0])
                return
            if command in ('undo','redo'):
                history=self._tb_history();edit=history.undo() if command=='undo' else history.redo()
                if edit is None:return
                forward=command=='redo'
                if edit['kind']=='play':self.strip.state.update(edit['after' if forward else 'before']);self._tb_view_changed();self.strip.update();self._tb_update_labels()
                elif edit['kind']=='trim':self._tb_restore_trim(edit['after' if forward else 'before'])
                elif edit['kind']=='region':
                    self._region_restore(edit['after' if forward else 'before'])
                else:
                    anchors=dict(self.anchors)
                    for f,pair in edit['changes'].items():
                        value=pair[1 if forward else 0]
                        if value is None:anchors.pop(f,None)
                        else:anchors[f]=value
                    self.anchors=anchors;self.track,self._retrack_previous=edit['after' if forward else 'before']
                    self._tb_refresh_anchor_edit()
                self.status_label.setText(('已重做：' if forward else '已撤销：')+edit['label']);return
            selected=set(self.strip.selected)&self.anchors.keys()
            if command=='copy':
                if selected:
                    first=min(selected);self._timebar_clipboard=(self.active_clip_id,[(f-first,tuple(self.anchors[f])) for f in sorted(selected)])
                    self.status_label.setText(f'已复制 {len(selected)} 个修正点；在本素材目标帧粘贴')
                return
            before=self._tb_before_roi();anchors=dict(self.anchors);new_selected=None
            low,high=self.start_spin.value()-1,self.end_spin.value()-1
            if command=='delete':
                for f in selected:anchors.pop(f,None)
            elif command=='move':
                selected,delta=data
                if not selected:return
                anchors=shifted_keys(anchors,selected,delta,low,high);new_selected=set(anchors)-(set(self.anchors)-set(selected))
            elif command=='paste':
                clip_id,values=getattr(self,'_timebar_clipboard',(None,[]))
                if clip_id!=self.active_clip_id:raise ValueError('请先复制当前素材的修正关键帧')
                targets={frame+offset:box for offset,box in values}
                if any(not low<=f<=high for f in targets):raise ValueError('粘贴超出处理/跟踪范围，未作修改')
                if targets.keys()&anchors.keys():raise ValueError('粘贴位置已有关键帧，未覆盖原有修正点')
                anchors.update(targets);new_selected=set(targets)
            elif command=='add':
                if not low<=frame<=high:raise ValueError('请先进入处理/跟踪范围')
                if frame in anchors:return
                from .tracking_pause import computed_mask
                if self.track is None or not self.track.start<=frame<=self.track.end:raise ValueError('当前帧没有已测量目标，请在左侧原片框选修正点')
                index=frame-self.track.start
                if not computed_mask(self.track)[index] or self.track.states[index]=='lost':raise ValueError('当前帧未测量或跟丢，请在原片手动框选；不能用插值冒充修正')
                anchors[frame]=tuple(self.track.boxes[index]);new_selected={frame}
            else:return
            if anchors==self.anchors:return
            self._tb_set_anchors(anchors,new_selected);self._tb_finish_roi(before,{'move':'移动修正关键帧','paste':'粘贴修正关键帧','delete':'删除修正关键帧','add':'添加修正关键帧'}[command])
            self.status_label.setText('修正关键帧已更新；暂停任务仍需点击继续跟踪' if self._paused_tracking else '修正关键帧已更新，点击开始跟踪重算受影响区间')
        except (ValueError,TypeError) as error:self.status_label.setText(str(error))
