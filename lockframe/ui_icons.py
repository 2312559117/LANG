"""Original local vector icons; no platform glyph or network dependency."""
from math import cos, pi, sin
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QFont


def _draw(p, name, colour):
    p.setPen(QPen(QColor(colour),1.45,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap,Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    def line(x1,y1,x2,y2):p.drawLine(QPointF(x1,y1),QPointF(x2,y2))
    def poly(points):p.drawPolyline(QPolygonF([QPointF(x,y) for x,y in points]))
    if name in ('play','pause','first','last','prev','next'):
        if name=='pause':
            p.setBrush(QColor(colour));p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(6,5,4,14),.6,.6);p.drawRoundedRect(QRectF(14,5,4,14),.6,.6)
        else:
            left=name in ('first','prev')
            if name in ('first','last'):line(5 if left else 19,5,5 if left else 19,19)
            if name in ('prev','next'):line(6 if left else 18,6,6 if left else 18,18)
            if name=='play':p.setBrush(QColor(colour))
            p.drawPolygon(QPolygonF([QPointF(x,y) for x,y in ([(16,5),(7,12),(16,19)] if left else [(8,5),(18,12),(8,19)])]))
    elif name in ('undo','redo','reset'):
        if name=='redo':p.translate(24,0);p.scale(-1,1)
        path=QPainterPath(QPointF(6,8));path.cubicTo(21,3,23,21,8,19);p.drawPath(path);poly([(8,4),(4,8),(8,12)])
    elif name in ('folder','import'):
        path=QPainterPath(QPointF(3,8));path.lineTo(3,5);path.lineTo(10,5);path.lineTo(12,8);path.lineTo(21,8);path.lineTo(21,20);path.lineTo(3,20);path.closeSubpath();p.drawPath(path)
        if name=='import':line(12,17,12,10);poly([(9,13),(12,10),(15,13)])
        else:line(3,10,21,10)
    elif name=='curve':
        path=QPainterPath(QPointF(3,19));path.cubicTo(7,19,7,5,12,5);path.cubicTo(17,5,17,17,21,17);p.drawPath(path)
        for x,y in ((3,19),(12,5),(21,17)):p.drawEllipse(QRectF(x-1.5,y-1.5,3,3))
    elif name=='export':
        poly([(8,8),(4,8),(4,20),(20,20),(20,8),(16,8)]);line(12,14,12,3);poly([(8,7),(12,3),(16,7)])
    elif name in ('target','compose'):
        if name=='compose':
            for x,y in [(4,4),(14,4),(4,14),(14,14)]:p.drawRoundedRect(QRectF(x,y,6,6),.8,.8)
        else:
            p.drawEllipse(QRectF(5,5,14,14));p.drawEllipse(QRectF(9,9,6,6))
            for a,b,c,d in [(12,2,12,6),(12,18,12,22),(2,12,6,12),(18,12,22,12)]:line(a,b,c,d)
    elif name in ('fit','layout','one'):
        if name=='layout':p.drawRoundedRect(QRectF(3,4,18,16),1,1);line(9,4,9,20);line(9,10,21,10)
        elif name=='one':
            p.drawRoundedRect(QRectF(2,5,20,14),1,1);f=QFont('Consolas');f.setPixelSize(10);p.setFont(f);p.drawText(QRectF(3,5,18,14),Qt.AlignmentFlag.AlignCenter,'1:1')
        else:
            for points in [[(8,3),(4,3),(4,7)],[(16,3),(20,3),(20,7)],[(4,17),(4,21),(8,21)],[(16,21),(20,21),(20,17)]]:poly(points)
            p.drawRect(QRectF(8,8,8,8))
    elif name in ('pointer','edit'):
        p.setBrush(QColor(colour));p.drawPolygon(QPolygonF([QPointF(x,y) for x,y in [(5,3),(19,13),(12,14),(9,21),(5,3)]]))
    elif name=='hand':
        path=QPainterPath(QPointF(8,14));path.lineTo(8,8);path.cubicTo(8,6,10,6,10,8);path.lineTo(10,5);path.cubicTo(10,3,12,3,12,5);path.lineTo(12,8);path.lineTo(12,6);path.cubicTo(12,4,14,4,14,6);path.lineTo(14,9);path.cubicTo(14,6,16,6,16,8);path.lineTo(17,13);path.cubicTo(19,20,10,23,7,18);path.lineTo(4,13);path.cubicTo(3,10,6,10,8,14);p.drawPath(path)
    elif name in ('more','filter'):
        for y in (6,12,18):line(5,y,19,y)
        if name=='filter':p.drawEllipse(QRectF(16,4,4,4))
    elif name=='link':
        path=QPainterPath(QPointF(10,15));path.cubicTo(5,19,1,13,5,10);path.lineTo(9,6);path.cubicTo(13,2,18,6,14,10);p.drawPath(path)
        path=QPainterPath(QPointF(14,9));path.cubicTo(19,5,23,11,19,14);path.lineTo(15,18);path.cubicTo(11,22,6,18,10,14);p.drawPath(path);line(9,15,15,9)
    elif name in ('zoom-in','zoom-out'):
        p.drawEllipse(QRectF(3,3,13,13));line(14,14,21,21);line(6,9.5,13,9.5)
        if name=='zoom-in':line(9.5,6,9.5,13)
    elif name in ('rectangle','ellipse','polygon'):
        if name=='rectangle':p.drawRect(QRectF(4,5,16,14))
        elif name=='ellipse':p.drawEllipse(QRectF(3,5,18,14))
        else:p.drawPolygon(QPolygonF([QPointF(12+9*cos(-pi/2+i*2*pi/5),12+9*sin(-pi/2+i*2*pi/5)) for i in range(5)]))
    elif name in ('add','delete'):
        line(5,12,19,12)
        if name=='add':line(12,5,12,19)
    else:
        p.drawEllipse(QRectF(5,5,14,14));line(12,8,12,13);line(12,16,12,16.2)


def icon(name, colour='#b7afa5', size=18):
    """Create a crisp 2x QIcon with explicit checked and disabled states."""
    result=QIcon();size=max(8,int(size))
    for mode,state,tint in ((QIcon.Mode.Normal,QIcon.State.Off,colour),(QIcon.Mode.Normal,QIcon.State.On,'#e88c51'),(QIcon.Mode.Disabled,QIcon.State.Off,'#68635c'),(QIcon.Mode.Disabled,QIcon.State.On,'#68635c')):
        pixmap=QPixmap(size*2,size*2);pixmap.setDevicePixelRatio(2.);pixmap.fill(Qt.GlobalColor.transparent)
        p=QPainter(pixmap);p.setRenderHint(QPainter.RenderHint.Antialiasing);p.scale(size/24,size/24);_draw(p,name,tint);p.end();result.addPixmap(pixmap,mode,state)
    return result
