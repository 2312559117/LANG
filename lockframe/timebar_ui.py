"""Qt source-time ruler with independent playback range and correction keys."""
from bisect import bisect_left, bisect_right
import math
import numpy as np
from PySide6.QtCore import Qt,QPointF,QRectF,Signal,QTimer
from PySide6.QtGui import QColor,QPainter,QPen,QPolygonF,QPixmap,QImage,QFont
from PySide6.QtWidgets import QWidget,QMenu
from .timebar_core import Timebase,clean_state


class Timebar(QWidget):
    sectionPicked=Signal(int)
    seek=Signal(int)
    scrubStarted=Signal()
    operation=Signal(str,object)
    viewChanged=Signal()
    overlayVisibilityChanged=Signal(str,bool)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setFixedHeight(116);self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.frames=0;self.current=0;self.anchors={};self.track=None;self.start=self.end=0
        self.tracking_sections=[]
        self.section_mode=False;self.pending_section=None
        self.editable=True;self.selected=set();self.state=clean_state({},1)
        self.tb=Timebase(30);self._keys=[];self._key_signature=None;self._base=None;self._base_token=None
        self._drag=None;self._ghost=None;self._selection=None;self._snap=None;self._status_token=None;self._status=[]
        self.ad_style=False;self.tool_mode='select';self._compact=False
        self._layer_visible=dict(mark=False,correction=True,video=True,audio=False)
        self._media_source=None;self._media_service=None;self._media_result=None;self._media_pixmaps=()
        self._media_key=None;self._media_dpr=None;self._media_error='';self._media_revision=0;self._media_busy=False
        self._media_timer=QTimer(self);self._media_timer.setSingleShot(True);self._media_timer.setInterval(120)
        self._media_timer.timeout.connect(self._request_media)
        self.setToolTip('标尺点击/拖动定位；Alt+滚轮缩放，中键/手形平移，F适配。\n标尺细条：独立播放范围；跟踪框条：处理范围；菱形：人工修正关键帧。\n眼睛仅切换时间轴内容显示，不改变导出；音频波形来自原片，预览暂不播放声音。\nDelete仅删修正关键帧，不删视频帧。')

    @property
    def left(self):return 174 if self.ad_style else 92

    @property
    def key_y(self):return self.row_rect('correction').center().y() if self.ad_style else 68

    @property
    def range_y(self):return 28 if self.ad_style else 34

    def row_rect(self,name):
        if not self.ad_style:
            return {'mark':QRectF(0,20,self.width(),22),'correction':QRectF(0,59,self.width(),23),
                    'video':QRectF(0,84,self.width(),14),'audio':QRectF(0,99,self.width(),5)}[name]
        if name == 'mark' or name == 'audio' and not self._layer_visible['audio']:
            return QRectF()
        names = ('correction','video','audio') if self._layer_visible['audio'] else ('correction','video')
        weights = (.30,.42,.28) if self._layer_visible['audio'] else (.38,.62)
        top = 32
        available = max(60, self.height()-top-6)
        index = names.index(name)
        return QRectF(0,top+round(available*sum(weights[:index])),self.width(),
                      round(available*sum(weights[:index+1]))-round(available*sum(weights[:index])))

    def set_audio_visible(self, visible):
        visible = bool(visible)
        if self._layer_visible['audio'] == visible:
            return
        self.cancel_drag()
        self._layer_visible['audio'] = visible
        self._base_token = None
        self._schedule_media()
        self.overlayVisibilityChanged.emit('audio', visible)
        self.update()

    def set_compact(self,compact):
        if self._compact==bool(compact):return
        self._compact=bool(compact);self._base_token=None;self.update()

    def set_busy(self,busy):
        changed=self._media_busy!=bool(busy);self._media_busy=bool(busy)
        if changed and self._media_service:self._media_service.set_busy(busy)
        if changed:self._base_token=None;self.update()

    @property
    def media_status(self):
        result=self._media_result
        if not self._media_source:state='error' if self._media_error else 'empty'
        elif result is not None and result.key==self._media_key:
            state='error' if result.video_error or result.audio_error else 'ready'
        else:state='busy' if self._media_busy else 'pending'
        return dict(state=state,identity=self._media_source.identity if self._media_source else None,
                    key=self._media_key,images=len(self._media_pixmaps),peaks=int(result.peaks.size) if result else 0,
                    video_error=result.video_error if result else self._media_error,
                    audio_error=result.audio_error if result else '',
                    film_pixels=[[p.width(),p.height()] for p in self._media_pixmaps],device_pixel_ratio=self.devicePixelRatioF())

    def set_media(self,info):
        from .timeline_media import MediaSource
        old_error=self._media_error
        try:source=MediaSource.from_info(info)
        except (OSError,ValueError) as error:
            source=None;self._media_error='媒体概览不可用：'+str(error)
        else:self._media_error=''
        if source==self._media_source:
            if old_error!=self._media_error:self._base_token=None;self.update()
            return
        self._media_source=source;self._media_result=None;self._media_key=None;self._media_pixmaps=()
        self._media_revision+=1;self._base_token=None
        if self._media_service:self._media_service.clear()
        if source:self._media_error='';self._schedule_media()
        self.update()

    def _schedule_media(self):
        if self.ad_style and self._media_source:self._media_timer.start()

    def _request_media(self):
        if not self.ad_style or not self._media_source:return
        from .timeline_media import TimelineMedia,make_request
        if self._media_service is None:
            self._media_service=TimelineMedia(self);self._media_service.ready.connect(self._accept_media)
            self._media_service.set_busy(self._media_busy)
        start=max(0,math.floor(self.at(self.left)));end=min(self._media_source.frames,math.ceil(self.at(self.width()-10)))
        count=min(16,max(1,math.ceil(self.span/145)))
        dpr=self.devicePixelRatioF();height=max(16,self.row_rect('video').height()-8)
        tile_width=self.span/max(1,count-1)
        request=make_request(self._media_source,start,end,count=count,bins=min(4096,max(64,self.span)),
                             pixel_size=(math.ceil(tile_width*dpr),math.ceil(height*dpr)),
                             include_audio=self._layer_visible['audio'])
        self._media_dpr=dpr
        self._media_key=request.key;self._media_service.request(request)

    def _accept_media(self,result):
        if not self._media_source or result.identity!=self._media_source.identity or result.key!=self._media_key:return
        self._media_result=result
        self._media_pixmaps=tuple(QPixmap.fromImage(QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QImage.Format.Format_RGB888).copy()) for a in result.images)
        self._media_revision+=1;self._base_token=None;self.update()

    def close_media(self):
        self._media_timer.stop();self._media_source=None;self._media_key=None
        self._media_result=None;self._media_pixmaps=()
        if self._media_service:self._media_service.close();self._media_service=None

    def closeEvent(self,event):
        self.close_media();super().closeEvent(event)

    @property
    def span(self):return max(1,self.width()-self.left-10)

    @property
    def scale(self):return self.state['scale'] or self.span/max(1,self.frames)

    def x(self,frame):return self.left+(frame-self.state['scroll'])*self.scale

    def at(self,x):return self.state['scroll']+(x-self.left)/self.scale

    def frame_at(self,x):return max(0,min(self.frames-1,int(math.floor(self.at(x)+.5))))

    def load_state(self,raw,frames,fps,start=0,end=None):
        self.frames=frames;self.start=start;self.end=frames-1 if end is None else end
        self.state=clean_state(raw,max(1,frames),start,self.end)
        if self.ad_style and (not isinstance(raw,dict) or 'timecode' not in raw):self.state['timecode']=True
        self.tb=Timebase.from_fps(fps);self.selected.clear();self.cancel_drag();self._clamp_view();self._schedule_media();self.update()

    def sync_data(self):
        if not self.state['custom']:
            self.state.update(play_in=self.start,play_out=self.end+1)
        if self.frames and not 0<=self.state['play_in']<self.state['play_out']<=self.frames:
            self.state=clean_state({},self.frames,self.start,self.end)
        signature=(id(self.anchors),len(self.anchors))
        if signature!=self._key_signature:
            self._keys=sorted(self.anchors);self._key_signature=signature;self.selected.intersection_update(self.anchors)
        # Tracking results are immutable after delivery. A partial result is a new object.
        token=(id(self.track),self.frames)
        if token!=self._status_token:
            self._status_token=token;self._status=[]
            if self.track is not None:
                from .tracking_pause import computed_mask
                measured=computed_mask(self.track)
                states=np.asarray(self.track.states)
                codes=np.zeros(len(states),dtype=np.uint8)
                for code,name in enumerate(('tracked','recovered','manual','lost'),1):codes[(states==name)&measured]=code
                edges=np.r_[0,np.flatnonzero(codes[1:]!=codes[:-1])+1,len(codes)]
                self._status=[(self.track.start+int(a),self.track.start+int(b),int(codes[a]))
                              for a,b in zip(edges[:-1],edges[1:]) if codes[a]]
        self.update()

    def _clamp_view(self):
        self.state['scroll']=max(0.,min(self.state['scroll'],max(0.,self.frames-self.span/self.scale)))

    def fit(self):
        self.state.update(scale=0.,scroll=0.);self.viewChanged.emit();self._schedule_media();self.update()

    def zoom(self,factor,anchor=None):
        anchor=self.left+self.span/2 if anchor is None else max(self.left,min(self.width()-10,anchor))
        frame=self.at(anchor);scale=max(self.span/max(1,self.frames),min(64.,self.scale*factor))
        self.state.update(scale=scale,scroll=frame-(anchor-self.left)/scale)
        self._clamp_view();self.viewChanged.emit();self._schedule_media();self.update()

    def visible_keys(self):
        a=bisect_left(self._keys,self.at(self.left)-1);b=bisect_right(self._keys,self.at(self.width())+1)
        return self._keys[a:b]

    def hit_key(self,x):
        index=bisect_left(self._keys,self.at(x))
        candidates=self._keys[max(0,index-1):index+1]
        return min(candidates,key=lambda f:abs(self.x(f)-x)) if candidates and min(abs(self.x(f)-x) for f in candidates)<=7 else None

    def _draw_ad(self,p):
        """Four views of one source clip, not four editable media tracks."""
        p.fillRect(self.rect(),QColor('#171716'))
        p.setRenderHint(QPainter.RenderHint.Antialiasing,True)
        for code,name,title in (('V2','correction','跟踪'),('V1','video','裁剪'),('A1','audio','音频')):
            if name == 'audio' and not self._layer_visible['audio']:
                continue
            row=self.row_rect(name);p.fillRect(row,QColor('#1b1b1a') if name=='audio' else QColor('#181817'))
            p.setPen(QColor('#32302d'));p.drawLine(QPointF(0,row.top()),QPointF(self.width(),row.top()))
            p.drawLine(QPointF(self.left,row.top()),QPointF(self.left,row.bottom()))
            badge=QRectF(10,row.center().y()-11,25,22);p.setBrush(QColor('#2d2c2a'));p.setPen(Qt.PenStyle.NoPen);p.drawRoundedRect(badge,3,3)
            p.setPen(QColor('#d2ccc4'));p.drawText(badge,Qt.AlignmentFlag.AlignCenter,code)
            p.drawText(QRectF(84,row.top(),68,row.height()),Qt.AlignmentFlag.AlignVCenter,title)
            cx,cy=59,row.center().y()
            if name=='audio':
                p.setPen(QPen(QColor('#716a61'),1.6));p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(QPolygonF([QPointF(cx-7,cy-4),QPointF(cx-3,cy-4),QPointF(cx+1,cy-8),QPointF(cx+1,cy+8),QPointF(cx-3,cy+4),QPointF(cx-7,cy+4)]))
                p.drawArc(QRectF(cx-1,cy-9,11,18),-60*16,120*16)
            else:
                visible=self._layer_visible[name];p.setPen(QPen(QColor('#e88c51' if visible else '#716b64'),1.5));p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QRectF(cx-8,cy-4.5,16,9));p.setBrush(QColor('#e88c51' if visible else '#716b64'));p.drawEllipse(QRectF(cx-2.5,cy-2.5,5,5))
                if not visible:p.drawLine(QPointF(cx-8,cy+7),QPointF(cx+8,cy-7))
            p.setPen(QPen(QColor('#a39b91'),1));p.drawPolyline(QPolygonF([QPointF(151,cy-2),QPointF(155,cy+2),QPointF(159,cy-2)]))
        p.save();p.setClipRect(QRectF(self.left,0,self.span+1,self.height()-6))
        ideal=120/self.scale;power=10**math.floor(math.log10(max(1.,ideal)))
        step=next(v*power for v in (1,2,5,10) if v*power>=ideal)
        if self.state['timecode']:
            fps=float(self.tb.num/self.tb.den)
            seconds=max(1,round(ideal/fps))
            step=max(1,round(seconds*fps))
        first=max(0,math.ceil(self.at(self.left)/step)*step)
        bottom=self.row_rect('audio' if self._layer_visible['audio'] else 'video').bottom()
        for frame in range(int(first),min(self.frames+1,math.ceil(self.at(self.width()))+1),int(step)):
            x=self.x(frame);p.setPen(QColor('#292724'));p.drawLine(QPointF(x,30),QPointF(x,bottom))
            p.setPen(QColor('#aaa298'));p.drawText(QRectF(x+3,2,118,17),str(frame+1) if not self.state['timecode'] else self.tb.timecode(frame))
            p.setPen(QColor('#57514a'));p.drawLine(QPointF(x,22),QPointF(x,30))
            for minor in range(1,10):
                mx=self.x(frame+step*minor/10)
                p.setPen(QColor('#3b3732'));p.drawLine(QPointF(mx,26),QPointF(mx,30))
        a,b=self.state['play_in'],self.state['play_out']
        if self._ghost and self._drag and self._drag[0]=='range':a,b=self._ghost
        p.fillRect(QRectF(self.x(a),27,max(1,self.x(b)-self.x(a)),3),QColor('#9b7252'))
        for edge in (a,b):p.fillRect(QRectF(self.x(edge)-2,23,4,8),QColor('#e88c51'))
        keyrow=self.row_rect('correction');key_y=keyrow.center().y()
        if self._layer_visible['correction']:
            target=QRectF(self.x(self.start),keyrow.top()+4,max(1,self.x(self.end+1)-self.x(self.start)),keyrow.height()-8)
            p.setPen(QPen(QColor('#bbab79'),1));p.setBrush(QColor('#454335'));p.drawRoundedRect(target,3,3)
            p.setPen(QColor('#e1d9c7'));p.drawText(target.adjusted(10,0,-8,0),Qt.AlignmentFlag.AlignVCenter,'跟踪框' if self.anchors else '处理范围')
            for a,b,code in self._status:
                if b<self.at(self.left) or a>self.at(self.width()):continue
                color={1:'#727a56',2:'#b39d6f',3:'#e8a960',4:'#bb7464'}[code]
                p.fillRect(QRectF(self.x(a),keyrow.bottom()-6,max(1,self.x(b)-self.x(a)),3),QColor(color))
            for selected,color in ((False,'#efaf67'),(True,'#f2dfbf')):
                values=[f for f in self.visible_keys() if (f in self.selected)==selected]
                pixels=np.unique(np.asarray([round(self.x(f)) for f in values],dtype=int))
                p.setPen(QPen(QColor('#f0c78e'),.7));p.setBrush(QColor(color));size=6 if selected else 5
                for x in pixels:p.drawPolygon(QPolygonF([QPointF(float(x),key_y-size),QPointF(float(x)+size,key_y),QPointF(float(x),key_y+size),QPointF(float(x)-size,key_y)]))
        self._draw_media(p)
        p.restore()
        p.fillRect(QRectF(self.left,self.height()-4,self.span,3),QColor('#2b2925'))
        visible=min(1.,self.span/self.scale/max(1,self.frames))
        p.fillRect(QRectF(self.left+self.state['scroll']/max(1,self.frames)*self.span,self.height()-5,max(8,self.span*visible),4),QColor('#6b6254'))

    def _draw_media(self,p):
        result=self._media_result;video=self.row_rect('video').adjusted(self.left+3,4,-10,-4)
        audio=self.row_rect('audio').adjusted(self.left+3,4,-10,-4)
        p.setPen(QColor('#38372e'));p.setBrush(QColor('#282d24'));p.drawRoundedRect(audio,3,3)
        if self._layer_visible['video']:
            p.setPen(QColor('#a19b8b'));p.setBrush(QColor('#22211f'));p.drawRoundedRect(video,3,3)
            p.save();p.setClipRect(video);p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            if result and self._media_pixmaps:
                frames=result.frames
                # Tiles occupy source-time cells, including portrait/letterboxed media.
                edges=[result.start]+[round((a+b)/2) for a,b in zip(frames[:-1],frames[1:])]+[result.end]
                for index,pixmap in enumerate(self._media_pixmaps):
                    cell=QRectF(self.x(edges[index]),video.top(),max(1,self.x(edges[index+1])-self.x(edges[index])),video.height())
                    # Draw original physical pixels directly; do not shrink to logical
                    # pixels and then let Qt enlarge them again on high-DPI displays.
                    source=QRectF(pixmap.rect());ratio=cell.width()/max(1.,cell.height())
                    if source.width()/source.height()>ratio:
                        width=source.height()*ratio;source.setLeft((source.width()-width)/2);source.setWidth(width)
                    else:
                        height=source.width()/ratio;source.setTop((source.height()-height)/2);source.setHeight(height)
                    p.save();p.setClipRect(cell,Qt.ClipOperation.IntersectClip)
                    p.drawPixmap(cell,pixmap,source)
                    p.restore()
            else:
                p.setPen(QColor('#aaa194'));text=self._media_error or (result.video_error if result else '')
                p.drawText(video.adjusted(8,0,-8,0),Qt.AlignmentFlag.AlignVCenter,text or ('导入素材后显示真实视频胶片' if not self._media_source else '跟踪期间暂停读取胶片' if self._media_busy else '正在读取真实视频胶片…'))
            if self._media_source:
                font=QFont(p.font());font.setPixelSize(11);p.setFont(font)
                title=p.fontMetrics().elidedText(__import__('pathlib').Path(self._media_source.path).name,Qt.TextElideMode.ElideMiddle,min(200,round(video.width()-24)))
                tag=QRectF(video.left()+8,video.top(),min(video.width()-16,p.fontMetrics().horizontalAdvance(title)+12),17)
                p.fillRect(tag,QColor(20,20,19,180));p.setPen(QColor('#eee4d6'));p.drawText(tag.adjusted(6,0,-6,0),Qt.AlignmentFlag.AlignVCenter,title)
            p.restore()
            self._draw_trim(p,video)
        if self._layer_visible['audio']:
            if result and result.peaks.size:
                p.save();p.setClipRect(audio);p.setPen(QPen(QColor('#969d74'),1))
                count=len(result.peaks);frames=np.linspace(result.start,result.end,count,endpoint=False)
                # A square-root display keeps quiet real audio visible, without inventing peaks.
                amplitude=np.sqrt(result.peaks)*max(1,audio.height()/2-3)
                cy=audio.center().y()
                for frame,amp in zip(frames,amplitude):
                    x=self.x(float(frame));p.drawLine(QPointF(x,cy-float(amp)),QPointF(x,cy+float(amp)))
                p.restore()
            else:
                p.setPen(QColor('#aaa58e'))
                text='无音轨' if self._media_source and not self._media_source.has_audio else (result.audio_error if result else '')
                p.drawText(audio.adjusted(8,0,-8,0),Qt.AlignmentFlag.AlignVCenter,text or (self._media_error if self._media_error else '导入素材后显示音频波形' if not self._media_source else '跟踪期间暂停读取波形' if self._media_busy else '正在读取真实音频波形…'))

    def _draw_trim(self,p,video):
        a,b=(self._ghost if self._drag and self._drag[0]=='trim' and self._ghost is not None else (self.start,self.end+1))
        left,right=self.x(a),self.x(b)
        p.save();p.setClipRect(video)
        p.fillRect(QRectF(video.left(),video.top(),max(0,left-video.left()),video.height()),QColor(12,12,12,175))
        p.fillRect(QRectF(right,video.top(),max(0,video.right()-right),video.height()),QColor(12,12,12,175))
        p.setPen(QPen(QColor('#e88c51'),1.5));p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(left,video.top()+1,max(1,right-left),video.height()-2))
        for x,side in ((left,0),(right,1)):
            if video.left()-3<=x<=video.right()+3:
                handle=QRectF(x if side==0 else x-8,video.top(),8,video.height())
                p.fillRect(handle,QColor('#e88c51'));p.setPen(QPen(QColor('#35291e'),1.3))
                p.drawLine(QPointF(handle.center().x(),video.center().y()-5),QPointF(handle.center().x(),video.center().y()+5))
        p.restore()
        if self._drag and self._drag[0]=='trim':
            text=f'保留 {(b-a)*self.tb.den/self.tb.num:.3f}秒 · {b-a}帧  第{a+1}–{b}帧'
            tag=QRectF(self.left+8,max(32,video.top()-24),min(360,self.span-16),22)
            p.setPen(QColor('#eadfce'));p.setBrush(QColor('#30271f'));p.drawRoundedRect(tag,3,3)
            p.drawText(tag.adjusted(6,0,-6,0),Qt.AlignmentFlag.AlignCenter,text)

    def _draw_base(self,p):
        if self.ad_style:self._draw_ad(p);return
        p.fillRect(self.rect(),QColor('#161f26'))
        p.setPen(QColor('#78909b'))
        for y,text in ((22,'源帧 / TC'),(40,'播放范围'),(55,'处理/跟踪'),(77,'修正关键帧'),(97,'跟踪结果')):
            p.drawText(QRectF(4,y-15,86,18),Qt.AlignmentFlag.AlignVCenter,text)
        p.save();p.setClipRect(QRectF(self.left,0,self.span+1,116))
        ideal=70/self.scale;power=10**math.floor(math.log10(max(1.,ideal)))
        step=next(v*power for v in (1,2,5,10) if v*power>=ideal)
        first=max(0,math.ceil(self.at(self.left)/step)*step)
        for frame in range(int(first),min(self.frames,math.ceil(self.at(self.width()))),int(step)):
            x=self.x(frame);p.setPen(QColor('#38464f'));p.drawLine(QPointF(x,20),QPointF(x,103))
            p.setPen(QColor('#aebdc7'));p.drawText(QRectF(x+3,1,100,19),str(frame+1) if not self.state['timecode'] else self.tb.timecode(frame))
        for row,a,b,color in ((29,self.state['play_in'],self.state['play_out'],'#49a9c0'),(45,self.start,self.end+1,'#62727b')):
            if row==29 and self._ghost and self._drag and self._drag[0]=='range':a,b=self._ghost
            p.fillRect(QRectF(self.x(a),row,max(1,self.x(b)-self.x(a)),10),QColor(color))
            for x in (self.x(a),self.x(b)):
                p.fillRect(QRectF(x-3,row-2,6,14),QColor(color))
        for a,b,code in self._status:
            if b<self.at(self.left) or a>self.at(self.width()):continue
            color={1:'#517f70',2:'#bca37c',3:'#e9c46a',4:'#cd6868'}[code]
            p.fillRect(QRectF(self.x(a),86,max(1,self.x(b)-self.x(a)),12),QColor(color))
        # Aggregate dense keys by screen pixel. Selected keys stay visibly distinct.
        visible=self.visible_keys()
        for selected,color in ((False,'#e6c478'),(True,'#6edee6')):
            values=[f for f in visible if (f in self.selected)==selected]
            pixels=np.unique(np.asarray([round(self.x(f)) for f in values],dtype=int))
            p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor(color))
            size=5 if selected else 4
            for x in pixels:
                p.drawPolygon(QPolygonF([QPointF(float(x),68-size),QPointF(float(x)+size,68),QPointF(float(x),68+size),QPointF(float(x)-size,68)]))
        p.restore()
        p.fillRect(QRectF(self.left,107,self.span,4),QColor('#303c45'))
        visible=min(1.,self.span/self.scale/max(1,self.frames))
        p.fillRect(QRectF(self.left+self.state['scroll']/max(1,self.frames)*self.span,106,max(8,self.span*visible),6),QColor('#6d949e'))

    def paintEvent(self,event):
        if self.ad_style and self._media_source and self._media_dpr!=self.devicePixelRatioF() and not self._media_timer.isActive():self._schedule_media()
        p=QPainter(self)
        if not self.frames:
            if self.ad_style:self._draw_ad(p)
            else:p.fillRect(self.rect(),QColor('#161f26'))
            p.setPen(QColor('#aaa194'));p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,'导入素材后显示源帧时间线');return
        token=(self.size().width(),self.devicePixelRatioF(),self.frames,self.scale,self.state['scroll'],
               self.state['play_in'],self.state['play_out'],self.start,self.end,self.state['timecode'],
               self._key_signature,frozenset(self.selected),id(self.track),self._ghost, self._drag[0] if self._drag else None,self.height(),self.ad_style,
               self._media_revision,tuple(self._layer_visible.values()))
        if token!=self._base_token:
            dpr=self.devicePixelRatioF();self._base=QPixmap(round(self.width()*dpr),round(self.height()*dpr));self._base.setDevicePixelRatio(dpr)
            base=QPainter(self._base);self._draw_base(base);base.end();self._base_token=token
        p.drawPixmap(0,0,self._base);p.save();p.setClipRect(QRectF(self.left,0,self.span+1,self.height()-5 if self.ad_style else 104))
        for index, section in enumerate(self.tracking_sections):
            a,b=max(self.start,section['start']),min(self.end,section['end'])
            if b<a:continue
            row=self.row_rect('correction')
            left,right=self.x(a),self.x(b+1)
            if right<self.left or left>self.width():continue
            color=QColor('#eea66b' if index%2==0 else '#80bdc3')
            fill=QColor(color);fill.setAlpha(22)
            rect=QRectF(left,row.top(),max(1,right-left),row.height())
            p.fillRect(rect,fill);p.setPen(QPen(color,1,Qt.PenStyle.DashLine));p.drawRect(rect)
            p.setPen(color)
            if right-left>38:p.drawText(QRectF(max(left,self.left)+4,row.top(),max(1,right-max(left,self.left)-8),15),
                                       Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter,f'段 {index+1}')
        if self.section_mode and self.pending_section is not None:
            a,b=sorted((self.pending_section,self.current))
            row=self.row_rect('correction')
            p.fillRect(QRectF(self.x(a),row.top(),max(1,self.x(b+1)-self.x(a)),row.height()),QColor(232,140,81,40))
        x=self.x(self.current);bottom=self.height()-5 if self.ad_style else 103
        p.setPen(QPen(QColor('#eadfce' if self.ad_style else '#ff7c70'),1));p.drawLine(QPointF(x,26 if self.ad_style else 0),QPointF(x,bottom))
        p.setBrush(QColor('#eadfce' if self.ad_style else '#ff7c70'));p.drawPolygon(QPolygonF([QPointF(x-5,25 if self.ad_style else 0),QPointF(x+5,25 if self.ad_style else 0),QPointF(x,31 if self.ad_style else 6)]))
        if self.ad_style:
            tag=QRectF(x-35,0,70,20);p.setPen(Qt.PenStyle.NoPen);p.drawRoundedRect(tag,3,3)
            p.setPen(QColor('#211d18'));p.drawText(tag,Qt.AlignmentFlag.AlignCenter,self.tb.timecode(self.current))
        if self._selection:
            p.fillRect(self._selection,QColor(232,140,81,45));p.setPen(QColor('#e88c51'));p.drawRect(self._selection)
        if self._snap is not None:
            p.setPen(QPen(QColor('#edd9b4'),1,Qt.PenStyle.DashLine));p.drawLine(QPointF(self.x(self._snap),0),QPointF(self.x(self._snap),bottom))
        if self._drag and self._drag[0]=='keys' and self._ghost is not None:
            p.setPen(QPen(QColor('#efd9b6'),1));p.setBrush(QColor(232,140,81,80))
            for frame in self._drag[3]:
                if not self.at(self.left)-1<=frame+self._ghost<=self.at(self.width())+1:continue
                x=self.x(frame+self._ghost)
                y=self.key_y;p.drawPolygon(QPolygonF([QPointF(x,y-6),QPointF(x+6,y),QPointF(x,y+6),QPointF(x-6,y)]))
        p.restore()

    def mousePressEvent(self,event):
        if not self.frames:return
        self.setFocus();x,y=event.position().x(),event.position().y()
        if event.button()==Qt.MouseButton.MiddleButton or (event.button()==Qt.MouseButton.LeftButton and self.tool_mode=='pan' and x>=self.left):
            self._drag=('pan',x,self.state['scroll']);return
        if self.ad_style and event.button()==Qt.MouseButton.LeftButton and x<self.left:
            for name in ('correction','video','audio'):
                if self.row_rect(name).contains(QPointF(x,y)) and (43<=x<=75 or 145<=x<=166):
                    if name=='audio' and x<145:return
                    if name == 'audio':
                        self.set_audio_visible(False)
                        return
                    self._layer_visible[name]=not self._layer_visible[name]
                    self.overlayVisibilityChanged.emit(name,self._layer_visible[name]);self._base_token=None;self.update();return
        if event.button()!=Qt.MouseButton.LeftButton or not self.editable:return
        if x<self.left:return
        self.scrubStarted.emit()
        if (23<=y<=32 if self.ad_style else 26<=y<=42):
            a,b=self.state['play_in'],self.state['play_out']
            side=0 if abs(x-self.x(a))<=abs(x-self.x(b)) else 1
            if abs(x-self.x((a,b)[side]))<=8:self._drag=('range',side,(a,b));return
        if self.ad_style and self._layer_visible['video'] and self.row_rect('video').contains(QPointF(x,y)):
            a,b=self.start,self.end+1;side=0 if abs(x-self.x(a))<=abs(x-self.x(b)) else 1
            if abs(x-self.x((a,b)[side]))<=10:
                self._drag=('trim',side,(a,b),self.current);self._ghost=(a,b);self.update();return
        key_rect=self.row_rect('correction')
        if key_rect.top()<=y<=key_rect.bottom():
            if self.tracking_sections and y <= key_rect.top()+14:
                frame = self.frame_at(x)
                index = next((i for i,s in enumerate(self.tracking_sections) if s['start']<=frame<=s['end']), None)
                if index is not None:
                    self.sectionPicked.emit(index)
                    self._drag=('seek',)
                    self.seek.emit(frame)
                    return
            key=self.hit_key(x)
            shift=bool(event.modifiers()&Qt.KeyboardModifier.ShiftModifier)
            if key is not None:
                if shift:
                    if key in self.selected:self.selected.remove(key)
                    else:self.selected.add(key)
                elif key not in self.selected:self.selected={key}
                if key in self.selected:self._drag=('keys',x,key,tuple(sorted(self.selected)))
                self.seek.emit(key);self.update();return
            if not shift:self.selected.clear()
            self._drag=('box',x,set(self.selected));self._selection=QRectF(x,key_rect.top(),0,key_rect.height());self.update();return
        if y>=(self.height()-6 if self.ad_style else 104):self._drag=('scroll',x,self.state['scroll']);return
        self._drag=('seek',);self.seek.emit(self.frame_at(x))

    def mouseMoveEvent(self,event):
        if not self._drag:
            if self.ad_style:
                self.unsetCursor()
                point=event.position()
                if point.x()<self.left and self.row_rect('audio').contains(point):
                    self.setToolTip('原片真实音频波形；预览暂不播放声音，导出按原设置保留音轨。扬声器不可切换静音；右侧箭头仅收起波形显示。')
                elif point.x()<self.left:self.setToolTip('眼睛/箭头仅切换本时间轴轨道内容显示，不修改跟踪结果和导出。')
                elif self.row_rect('video').contains(point):
                    near=min(abs(point.x()-self.x(self.start)),abs(point.x()-self.x(self.end+1)))<=10
                    self.setCursor(Qt.CursorShape.SizeHorCursor if near and self.editable else Qt.CursorShape.ArrowCursor)
                    self.setToolTip('拖动V1视频条左右橙色手柄裁剪前后时长；松开生效，Esc取消，Ctrl+Z撤销。\n[ / ]将当前帧设为裁剪入/出点；I/O按当前普通/分段模式设置预览/分段范围。')
                elif self.row_rect('correction').contains(point):
                    self.unsetCursor();self.setToolTip('菱形：人工修正关键帧；底边：跟踪质量。拖动关键帧可撤销，Delete仅删修正点，不删视频帧。')
                elif point.y()<=32:self.setToolTip('拖动播放头定位；标尺底部暖橙细条为独立播放范围，拖动两端调整；I/O在普通模式设预览范围，在分段模式设分段范围。')
                else:self.setToolTip('真实原片胶片/音频波形；点击定位源帧。Alt+滚轮缩放；中键或手形平移；F适配。')
            return
        x=event.position().x();kind=self._drag[0]
        if kind in ('pan','scroll'):
            factor=1/self.scale if kind=='pan' else -max(1,self.frames)/self.span
            self.state['scroll']=self._drag[2]-(x-self._drag[1])*factor;self._clamp_view();self.viewChanged.emit();self._schedule_media()
        elif kind=='seek':self.seek.emit(self.frame_at(x))
        elif kind=='trim':
            side=self._drag[1];a,b=self._drag[2];frame=max(0,min(self.frames,round(self.at(x))))
            self._ghost=(min(frame,b-1),b) if side==0 else (a,max(a+1,frame))
            edge=self._ghost[0] if side==0 else self._ghost[1]-1
            self.seek.emit(edge)
        elif kind=='range':
            side=self._drag[1];a,b=self._drag[2];frame=max(0,min(self.frames,round(self.at(x))))
            self._ghost=(min(frame,b-1),b) if side==0 else (a,max(a+1,frame))
        elif kind=='box':
            a,b=sorted((self._drag[1],x));row=self.row_rect('correction');self._selection=QRectF(a,row.top(),b-a,row.height())
            self.selected=self._drag[2]|{f for f in self.visible_keys() if a<=self.x(f)<=b}
        elif kind=='keys':
            delta=round((x-self._drag[1])/self.scale);key=self._drag[2];chosen=self._drag[3]
            delta=max(self.start-min(chosen),min(self.end-max(chosen),delta));self._snap=None
            if not event.modifiers()&Qt.KeyboardModifier.ControlModifier:
                candidates=[self.current,self.state['play_in'],self.state['play_out']-1,self.start,self.end]
                index=bisect_left(self._keys,key+delta)
                candidates+=self._keys[max(0,index-2):index+3]
                candidates=[f for f in candidates if f not in chosen and self.start<=f<=self.end]
                if candidates:
                    nearest=min(candidates,key=lambda f:abs(f-(key+delta)))
                    proposed=nearest-key
                    if abs((key+delta-nearest)*self.scale)<=8 and self.start<=min(chosen)+proposed<=max(chosen)+proposed<=self.end:
                        delta=proposed;self._snap=nearest
            self._ghost=delta
            self.setToolTip(f'移动 {len(chosen)} 个修正关键帧：{delta:+d} 帧；松开提交，Esc取消；Ctrl关闭吸附')
        self.update()

    def mouseReleaseEvent(self,event):
        if not self._drag:return
        drag,ghost=self._drag,self._ghost
        selected=set(self.selected)
        self.cancel_drag()
        if drag[0]=='box':self.selected=selected;self.update()
        if not self.editable:return
        if drag[0]=='trim' and ghost is not None:self.operation.emit('trim',(ghost[0],ghost[1],drag[3]))
        if drag[0]=='range' and ghost is not None:self.operation.emit('play_range',ghost)
        if drag[0]=='keys' and ghost:
            self.operation.emit('move',(drag[3],ghost))

    def cancel_drag(self):
        if self._drag and self._drag[0]=='box':self.selected=self._drag[2]
        self._drag=None;self._ghost=None;self._selection=None;self._snap=None;self.update()

    def wheelEvent(self,event):
        amount=event.angleDelta().y() or event.angleDelta().x()
        if event.modifiers()&Qt.KeyboardModifier.AltModifier:self.zoom(1.25**(amount/120),event.position().x())
        else:
            self.state['scroll']-=amount/120*60/self.scale;self._clamp_view();self.viewChanged.emit();self._schedule_media();self.update()
        event.accept()

    def contextMenuEvent(self,event):
        if not self.frames or not self.editable:return
        row=self.row_rect('correction');key=self.hit_key(event.pos().x()) if row.top()<=event.pos().y()<=row.bottom() else None
        if key is not None and key not in self.selected:self.selected={key};self.update()
        menu=QMenu(self)
        for text,command in (('当前帧设为裁剪入点  [','trim_in'),('当前帧设为裁剪出点  ]','trim_out'),('恢复完整原片范围','trim_reset'),('上一修正关键帧  ,','previous_key'),('下一修正关键帧  .','next_key'),
                             ('在当前帧添加修正点','add'),('复制所选修正点  Ctrl+C','copy'),
                             ('粘贴到当前帧  Ctrl+V','paste'),('删除所选修正点（不删视频帧）','delete'),
                             ('撤销  Ctrl+Z','undo'),('重做  Ctrl+Shift+Z','redo'),
                             ('当前帧设为播放入点  I','in'),('当前帧设为播放出点  O','out'),
                             ('清除播放范围','clear_range'),('切换循环播放  L','loop')):
            action=menu.addAction(text);action.triggered.connect(lambda checked=False,c=command:self.operation.emit(c,None))
        menu.addSeparator();menu.addAction('适配全部帧  F',self.fit);menu.exec(event.globalPos())

    def keyPressEvent(self,event):
        key=event.key();ctrl=event.modifiers()&Qt.KeyboardModifier.ControlModifier;shift=event.modifiers()&Qt.KeyboardModifier.ShiftModifier
        if key==Qt.Key.Key_Escape:self.cancel_drag();event.accept();return
        if not self.editable or not self.frames:return
        if ctrl and key==Qt.Key.Key_A:self.selected=set(self._keys);self.update()
        elif ctrl and key in (Qt.Key.Key_Z,Qt.Key.Key_Y):self.operation.emit('redo' if shift or key==Qt.Key.Key_Y else 'undo',None)
        elif ctrl and key in (Qt.Key.Key_C,Qt.Key.Key_V):self.operation.emit('copy' if key==Qt.Key.Key_C else 'paste',None)
        elif key in (Qt.Key.Key_Left,Qt.Key.Key_Right):self.scrubStarted.emit();self.seek.emit(max(0,min(self.frames-1,self.current+(-1 if key==Qt.Key.Key_Left else 1)*(10 if shift else 1))))
        elif key in (Qt.Key.Key_Home,Qt.Key.Key_End):self.scrubStarted.emit();self.seek.emit(self.state['play_in'] if key==Qt.Key.Key_Home else self.state['play_out']-1)
        elif key==Qt.Key.Key_F:self.fit()
        else:
            commands={Qt.Key.Key_Space:'play',Qt.Key.Key_I:'clear_in' if shift else 'in',Qt.Key.Key_O:'clear_out' if shift else 'out',
                      Qt.Key.Key_BracketLeft:'trim_in',Qt.Key.Key_BracketRight:'trim_out',Qt.Key.Key_L:'loop',Qt.Key.Key_Comma:'previous_key',Qt.Key.Key_Period:'next_key',Qt.Key.Key_Delete:'delete',Qt.Key.Key_Backspace:'delete'}
            if key in commands:self.operation.emit(commands[key],None)
            else:super().keyPressEvent(event);return
        event.accept()

    def focusOutEvent(self,event):
        self.cancel_drag();super().focusOutEvent(event)

    def resizeEvent(self,event):
        self._clamp_view();self._base_token=None;self._schedule_media();super().resizeEvent(event)
