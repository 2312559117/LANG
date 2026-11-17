"""Compact source/output-time controls; the editor owns an unapplied draft."""
from copy import deepcopy
import math
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, Signal, QSignalBlocker
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QCheckBox,
                               QPushButton,QComboBox,QSizePolicy)
from .time_remap_model import normalize_config, active_points, default_points
from .time_remap_math import evaluate_source,evaluate_speed,basic_speed


class CurveCanvas(QWidget):
    committed=Signal(object)
    changing=Signal(object)
    selectedChanged=Signal(int)
    openRequested=Signal()
    seekRequested=Signal(float)

    def __init__(self,parent=None,editable=False,compact=False):
        super().__init__(parent)
        self.config=normalize_config()
        self.editable=editable;self.compact=compact;self.speed_view=False;self.selected=0;self.drag=None
        self._drag_domain=None;self._drag_before=None;self._drag_offset=QPointF()
        self.cursor_x=None;self._seek_drag=False
        self.setMinimumHeight(90);self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip('拖动圆点或曲线调整最近节点；小图双击放大编辑。Alt+单击/拖动可定位播放位置。')
        self.setCursor(Qt.CursorShape.OpenHandCursor if editable else Qt.CursorShape.ArrowCursor)

    def set_config(self,config):
        if self.drag is not None:
            self.drag=None;self._drag_domain=None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self.editable else Qt.CursorShape.ArrowCursor)
        self.config=normalize_config(config)
        self.selected=min(self.selected,len(active_points(self.config))-1)
        self.update()

    def sizeHint(self):return QSize(240,max(90,self.minimumHeight()))

    def set_playhead(self,value):
        value=float(value) if value is not None and math.isfinite(float(value)) else None
        if value!=self.cursor_x:self.cursor_x=value;self.update()

    def minimumSizeHint(self):return QSize(80,max(90,self.minimumHeight()))

    def plot(self):return QRectF(35,14,max(1,self.width()-49),max(1,self.height()-38))

    def domain(self):
        if self.drag is not None and self._drag_domain is not None:return self._drag_domain
        if self.config['mode']=='basic':return 1.,0.,10.
        return max(.01,active_points(self.config)[-1]['position']),(-10. if self.speed_view else 0.),(10. if self.speed_view else 1.)

    def pixel(self,x,y):
        rect=self.plot();end,lo,hi=self.domain()
        return QPointF(rect.left()+x/end*rect.width(),rect.bottom()-(y-lo)/(hi-lo)*rect.height())

    def value(self,pos):
        rect=self.plot();end,lo,hi=self.domain()
        return ((pos.x()-rect.left())/rect.width()*end,
                lo+(rect.bottom()-pos.y())/rect.height()*(hi-lo))

    def nodes(self):
        if self.config['mode']=='basic':
            return list(zip([0.,*self.config['basic']['boundaries'],1.],self.config['basic']['speeds']))
        points=active_points(self.config)
        return [(p['position'],evaluate_speed(points,p['position']) if self.speed_view else p['source']) for p in points]

    def handles(self):
        if self.config['mode']!='advanced' or self.speed_view:return []
        points=active_points(self.config);index=min(self.selected,len(points)-1);point=points[index]
        result=[]
        for side,direction,neighbor in (('left',-1,index-1),('right',1,index+1)):
            if not 0<=neighbor<len(points):continue
            other=points[neighbor]
            if side=='right' and (point['interpolation']!='bezier' or other['incoming_interpolation']!='bezier'):continue
            if side=='left' and (other['interpolation']!='bezier' or point['incoming_interpolation']!='bezier'):continue
            influence=point[f'{side}_influence']
            x=point['position']+direction*abs(point['position']-other['position'])*influence
            y=point[f'{side}_handle_source']
            if y is None:y=point['source']+(other['source']-point['source'])*influence
            result.append((side,x,y))
        return result

    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(),QColor('#181817'))
        rect=self.plot();painter.setPen(QPen(QColor('#34312c'),1))
        for index in range(5):
            x=rect.left()+rect.width()*index/4;y=rect.top()+rect.height()*index/4
            painter.drawLine(QPointF(x,rect.top()),QPointF(x,rect.bottom()))
            painter.drawLine(QPointF(rect.left(),y),QPointF(rect.right(),y))
        end,lo,hi=self.domain();points=active_points(self.config)
        painter.setPen(QColor('#a59d91'))
        painter.drawText(QRectF(1,3,32,18),Qt.AlignmentFlag.AlignRight,f'{hi:g}'+('×' if self.config['mode']=='basic' or self.speed_view else ''))
        painter.drawText(QRectF(1,rect.bottom()-12,32,18),Qt.AlignmentFlag.AlignRight,f'{lo:g}')
        title='源进度 → 速度' if self.config['mode']=='basic' else '输出时间 → 速度' if self.speed_view else '输出时间 → 源进度'
        painter.drawText(QRectF(35,self.height()-20,self.width()-40,18),title)
        painter.save();painter.setClipRect(rect.adjusted(-1,-1,1,1))
        path=QPainterPath()
        for index,x in enumerate(np.linspace(0,end,max(64,min(300,self.width())))):
            y=(basic_speed(self.config['basic'],float(x)) if self.config['mode']=='basic' else
               evaluate_speed(points,float(x)) if self.speed_view else evaluate_source(points,float(x)))
            pos=self.pixel(float(x),y)
            if index:path.lineTo(pos)
            else:path.moveTo(pos)
        painter.setPen(QPen(QColor('#eea36a'),2));painter.drawPath(path)
        if self.cursor_x is not None:
            x=max(0.,min(end,self.cursor_x));pixel=self.pixel(x,lo).x()
            painter.setPen(QPen(QColor('#73d2d2'),1.5));painter.drawLine(QPointF(pixel,rect.top()),QPointF(pixel,rect.bottom()))
            painter.setBrush(QColor('#73d2d2'));painter.drawEllipse(QPointF(pixel,rect.top()+3),3,3)
        if self.editable or not self.speed_view or self.config['mode']=='basic':
            for index,(x,y) in enumerate(self.nodes()):
                painter.setBrush(QColor('#f1dfc1' if index==self.selected else '#2b2721'))
                painter.drawEllipse(self.pixel(x,y),6.,6.)
            if self.editable:
                nodes=self.nodes()
                origin=self.pixel(*nodes[min(self.selected,len(nodes)-1)])
                for side,x,y in self.handles():
                    painter.setPen(QPen(QColor('#9d927e'),1));painter.drawLine(origin,self.pixel(x,y))
                    p=self.pixel(x,y)
                    painter.setBrush(QColor('#d2b58b'));painter.drawRect(QRectF(p.x()-3,p.y()-3,6,6))
        painter.restore()

    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton or not self.editable:return
        point=event.position()
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            self._seek_drag=True;self.seekRequested.emit(max(0.,min(self.domain()[0],self.value(point)[0])));event.accept();return
        domain=self.domain()
        self._drag_before=deepcopy(self.config);self._drag_domain=domain;self._drag_offset=QPointF()
        for side,x,y in self.handles():
            if (self.pixel(x,y)-point).manhattanLength()<14:
                self.drag=('handle',self.selected,side);self._drag_offset=point-self.pixel(x,y)
                self.setCursor(Qt.CursorShape.ClosedHandCursor);event.accept();return
        nodes=self.nodes();index=min(range(len(nodes)),key=lambda i:(self.pixel(*nodes[i])-point).manhattanLength())
        hit=(self.pixel(*nodes[index])-point).manhattanLength()<20
        if not hit and self.plot().adjusted(-10,-10,10,10).contains(point):
            points=active_points(self.config)
            for x in np.linspace(0,domain[0],max(64,min(600,self.width()*2))):
                y=(basic_speed(self.config['basic'],float(x)) if self.config['mode']=='basic' else
                    evaluate_speed(points,float(x)) if self.speed_view else evaluate_source(points,float(x)))
                if (self.pixel(float(x),y)-point).manhattanLength()<12:
                    hit=True;break
        if hit:
            self.selected=index;self._drag_offset=point-self.pixel(*nodes[index])
            self.selectedChanged.emit(index);self.drag=('speed' if self.speed_view and self.config['mode']=='advanced' else 'point',index)
            self.setCursor(Qt.CursorShape.ClosedHandCursor);self.update();event.accept()

    def mouseMoveEvent(self,event):
        if self._seek_drag:
            self.seekRequested.emit(max(0.,min(self.domain()[0],self.value(event.position())[0])));event.accept();return
        if self.drag is None:return
        config=deepcopy(self.config);x,y=self.value(event.position()-self._drag_offset);kind,index,*side=self.drag
        if config['mode']=='basic':
            config['basic']['speeds'][index]=max(.1,min(10.,y))
            if 0<index<4:
                bounds=[0.,*config['basic']['boundaries'],1.]
                config['basic']['boundaries'][index-1]=max(bounds[index-1]+.08,min(bounds[index+1]-.08,x))
        else:
            points=config['points'];point=points[index]
            if kind in ('point','speed'):
                low=points[index-1]['position']+.0001 if index else 0.
                high=points[index+1]['position']-.0001 if index+1<len(points) else 100.
                point['position']=0. if index==0 else max(low,min(high,x))
                if kind=='speed':
                    speed=max(-100.,min(100.,y))
                    point['incoming_speed']=point['outgoing_speed']=speed
                    point['incoming_interpolation']=point['interpolation']='bezier'
                    if index:points[index-1]['interpolation']='bezier'
                    if index+1<len(points):points[index+1]['incoming_interpolation']='bezier'
                else:
                    point['source']=max(0.,min(1.,y))
                    point['left_handle_source']=point['right_handle_source']=None
            else:
                direction=side[0];neighbor=index-1 if direction=='left' else index+1
                span=abs(points[neighbor]['position']-point['position'])
                amount=(point['position']-x if direction=='left' else x-point['position'])/max(.0001,span)
                point[f'{direction}_influence']=max(.01,min(.99,amount))
                point[f'{direction}_handle_source']=max(-1.,min(2.,y))
                point['incoming_speed' if direction=='left' else 'outgoing_speed']=None
                point[f'{direction}_speed_tangent']=None
                if point['handles_linked']:
                    other='right' if direction=='left' else 'left'
                    point[f'{other}_handle_source']=2*point['source']-point[f'{direction}_handle_source']
        self.config=normalize_config(config);self.changing.emit(deepcopy(self.config));self.update()

    def mouseReleaseEvent(self,event):
        if self._seek_drag:self._seek_drag=False;event.accept();return
        if self.drag is not None:
            changed=self.config!=self._drag_before
            self.drag=None;self._drag_domain=None
            self.setCursor(Qt.CursorShape.OpenHandCursor);self.update()
            if changed:self.committed.emit(deepcopy(self.config))
            event.accept()

    def mouseDoubleClickEvent(self,event):
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:return
        if not self.editable or self.compact:self.openRequested.emit();return
        if self.config['mode']!='advanced' or self.speed_view:return
        x,y=self.value(event.position());config=deepcopy(self.config)
        config['points'].append(dict(position=max(0.,min(100.,x)),source=max(0.,min(1.,y))))
        self.config=normalize_config(config);self.committed.emit(deepcopy(self.config));self.update()


class TimeRemapPanel(QWidget):
    configEdited=Signal(object)
    openEditor=Signal()
    clockChanged=Signal(str)
    prepareRequested=Signal()
    playPrepared=Signal()
    presetRequested=Signal()

    def __init__(self,parent=None):
        super().__init__(parent);self.config=normalize_config();layout=QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0);layout.setSpacing(5)
        row=QHBoxLayout();self.enabled=QCheckBox('启用变速');row.addWidget(self.enabled)
        reset=QPushButton('重置');reset.clicked.connect(self.reset);row.addWidget(reset);layout.addLayout(row)
        self.body=QWidget();body=QVBoxLayout(self.body);body.setContentsMargins(0,0,0,0);body.setSpacing(5)
        self.mode=QComboBox();self.mode.addItem('基础四段速度','basic');self.mode.addItem('高级时间曲线','advanced')
        body.addWidget(self.mode)
        self.plot=CurveCanvas(editable=True,compact=True);self.plot.setFixedHeight(110);self.plot.openRequested.connect(self.openEditor)
        self.plot.committed.connect(self.configEdited)
        body.addWidget(self.plot)
        self.summary=QLabel();self.summary.setWordWrap(True);self.summary.setObjectName('muted');body.addWidget(self.summary)
        self.flow=QCheckBox('光流插帧（DIS）');body.addWidget(self.flow)
        self.clock=QComboBox();self.clock.addItem('源时间预览','source');self.clock.addItem('输出时间预览','output');body.addWidget(self.clock)
        row=QHBoxLayout();edit=QPushButton('打开完整曲线编辑器…');edit.clicked.connect(self.openEditor);row.addWidget(edit,1)
        self.presets=QPushButton('预设…');self.presets.clicked.connect(self.presetRequested);row.addWidget(self.presets);body.addLayout(row)
        row=QHBoxLayout()
        self.prepare_clip=QPushButton('预渲染当前3秒');self.prepare_clip.clicked.connect(self.prepareRequested)
        self.play_clip=QPushButton('播放缓存片段');self.play_clip.clicked.connect(self.playPrepared)
        self.play_clip.setEnabled(False);row.addWidget(self.prepare_clip);row.addWidget(self.play_clip);body.addLayout(row)
        self.clip_status=QLabel('完成跟踪后，可预渲染检查真实光流效果。')
        self.clip_status.setWordWrap(True);self.clip_status.setObjectName('muted');body.addWidget(self.clip_status)
        layout.addWidget(self.body)
        self.enabled.toggled.connect(lambda v:self.edit('enabled',v));self.flow.toggled.connect(lambda v:self.edit('optical_flow',v))
        self.mode.currentIndexChanged.connect(lambda i:self.edit('mode',self.mode.currentData()))
        self.clock.currentIndexChanged.connect(lambda i:self.clockChanged.emit(self.clock.currentData()))
        self.set_config(None)

    def set_config(self,config):
        self.config=normalize_config(config or None)
        controls=(self.enabled,self.flow,self.mode)
        blockers=[QSignalBlocker(x) for x in controls]
        self.enabled.setChecked(self.config['enabled']);self.flow.setChecked(self.config['optical_flow'])
        self.mode.setCurrentIndex(self.mode.findData(self.config['mode']))
        self.body.setVisible(self.config['enabled']);self.plot.set_config(self.config)
        del blockers

    def edit(self,key,value):
        config=deepcopy(self.config)
        if key=='mode' and value=='advanced' and config['mode']=='basic' and config['points']==normalize_config(normalize_config())['points']:
            config['points']=active_points(config)
        config[key]=value
        self.configEdited.emit(normalize_config(config))

    def reset(self):
        config=normalize_config();config['enabled']=self.enabled.isChecked();self.configEdited.emit(config)
