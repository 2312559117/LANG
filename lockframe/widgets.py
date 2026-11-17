from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QPainterPath
from PySide6.QtWidgets import QWidget
from .branding import brand_pixmap
from .theme import CANVAS


def qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()


class PreviewCanvas(QWidget):
    roi_selected = Signal(tuple)
    region_selected = Signal(object)
    vertex_selection_changed = Signal(int)
    selection_rejected = Signal(str)
    zoom_changed = Signal(float)
    playback_requested = Signal()
    edit_started = Signal()
    composition_changed = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.source = None
        self.processed = None
        self.source_image = None
        self.processed_image = None
        self.box = None
        self.homography = None
        self.confidence = None
        self.state = ""
        self.persistent_compare = False
        self.mode = "compare"
        self.selectable = True
        self.drag_start = None
        self.drag_end = None
        self.target = (0.5, 0.5)
        self.composition_editable = False
        self._composition_drag = False
        self.output_rect = QRectF()
        self.source_rect = QRectF()
        self.source_viewport = QRectF()
        self.selection_mode = "box"
        self.selection_shape = "rectangle"
        self.ad_style = False
        self.output_zoom = 1.
        self.region_overlay_visible = True
        # View-only geometry. The host owns persisted settings/history and applies
        # region_selected proposals only after canonical validation.
        self.region_points = ()
        self.region_edit_mode = "select"
        self._selected_vertex = -1
        self._region_drag = None
        self._edit = None
        self.point_size = 64
        self.hover_position = None
        self._zoom_factor = 1.0
        self._view_center = QPointF(0.5, 0.5)
        self._pan_start = None
        self._pan_center = None
        self._pan_button = None
        self._space_pressed = False
        self._space_pan_moved = False

    @property
    def selected_vertex(self):
        return self._selected_vertex

    @selected_vertex.setter
    def selected_vertex(self, value):
        value = int(value)
        if value != self._selected_vertex:
            self._selected_vertex = value
            self.vertex_selection_changed.emit(value)
            self.update()

    @property
    def can_delete_vertex(self):
        return (self.selectable and self.mode != "output" and not self.composition_editable
                and self.source_image is not None and self.selection_shape == "polygon"
                and len(self.region_points) > 3
                and 0 <= self.selected_vertex < len(self.region_points))

    @property
    def mode(self):
        return "compare" if self.persistent_compare else self._mode

    @mode.setter
    def mode(self, value):
        if hasattr(self, "_region_drag"):
            self._cancel_edit()
        self._mode = "compare" if self.persistent_compare else value

    @property
    def zoom_factor(self):
        """Source-preview magnification relative to the fitted image (1..8)."""
        return self._zoom_factor

    def set_zoom(self, factor, anchor=None):
        """Zoom around a widget-space point; the default anchor is the view center."""
        factor = float(factor)
        if not math.isfinite(factor) or self.source_image is None:
            return
        factor = max(1.0, min(8.0, factor))
        self._sync_source_geometry()
        if math.isclose(factor, self._zoom_factor):
            return
        self._cancel_edit()
        if not self.source_rect.isValid():
            # The output-only view must not change its own scale.
            self._zoom_factor = factor
        else:
            anchor = QPointF(anchor) if anchor is not None else self.source_viewport.center()
            if not self.source_viewport.contains(anchor):
                anchor = self.source_viewport.center()
            image_x = (anchor.x() - self.source_rect.left()) / self.source_rect.width()
            image_y = (anchor.y() - self.source_rect.top()) / self.source_rect.height()
            ratio = factor / self._zoom_factor
            width, height = self.source_rect.width() * ratio, self.source_rect.height() * ratio
            center = self.source_viewport.center()
            self._view_center = QPointF(image_x - (anchor.x() - center.x()) / width,
                                        image_y - (anchor.y() - center.y()) / height)
            self._zoom_factor = factor
        self.drag_start = self.drag_end = None
        self._pan_start = self._pan_center = self._pan_button = None
        self._sync_source_geometry()
        self.zoom_changed.emit(self._zoom_factor)
        self.update()

    def reset_view(self):
        self._cancel_edit()
        changed = self._zoom_factor != 1.0
        self._zoom_factor = 1.0
        self._view_center = QPointF(0.5, 0.5)
        self._pan_start = self._pan_center = self._pan_button = None
        self.drag_start = self.drag_end = None
        self.hover_position = None
        self._sync_source_geometry()
        if changed:
            self.zoom_changed.emit(1.0)
        self.update()

    def _sync_source_geometry(self):
        """Keep image coordinates independent of viewport clipping and paint timing."""
        self.source_rect = QRectF()
        self.source_viewport = QRectF()
        if self.source_image is None or self.mode not in ("compare", "source"):
            return
        area = QRectF(0, 0, (self.width()-12)/2 if self.mode == 'compare' and self.ad_style else self.width()/2 if self.mode == 'compare' else self.width(), self.height())
        viewport = area if self.ad_style else area.adjusted(12, 42, -12, -18)
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        fitted = self.fitted(viewport, self.source_image)
        width, height = fitted.width() * self._zoom_factor, fitted.height() * self._zoom_factor
        # The image covers the viewport on enlarged axes; fitted axes stay centered.
        # This also clamps the view after resize or switching source/compare modes.
        minimum_x = min(0.5, viewport.width() / (2 * width))
        minimum_y = min(0.5, viewport.height() / (2 * height))
        cx = max(minimum_x, min(1 - minimum_x, self._view_center.x()))
        cy = max(minimum_y, min(1 - minimum_y, self._view_center.y()))
        self._view_center = QPointF(cx, cy)
        self.source_viewport = viewport
        self.source_rect = QRectF(viewport.center().x() - cx * width,
                                  viewport.center().y() - cy * height, width, height)

    def _in_source(self, position):
        return self.source_viewport.contains(position) and self.source_rect.contains(position)

    def set_frames(self, source, processed=None, box=None, confidence=None, state="", preserve_view=False,
                   homography=None, confidence_grid=None):
        """Set preview pixels; preserve_view supports proxy/full-resolution swaps."""
        self._cancel_edit()
        previous_size = None if self.source_image is None else self.source_image.size()
        self.source = source
        self.processed = processed
        self.source_image = qimage(source) if source is not None else None
        self.processed_image = qimage(processed) if processed is not None else None
        self.box, self.confidence, self.state = box, confidence, state
        self.homography = homography
        self.confidence_grid = confidence_grid
        current_size = None if self.source_image is None else self.source_image.size()
        keep_view = preserve_view and previous_size is not None and current_size is not None
        if current_size != previous_size and not keep_view:
            self.reset_view()
        else:
            self._sync_source_geometry()
        self.update()

    @staticmethod
    def fitted(area: QRectF, image: QImage):
        scale = min(area.width() / image.width(), area.height() / image.height())
        w, h = image.width() * scale, image.height() * scale
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def set_output_zoom(self, factor):
        self.output_zoom=max(1.,min(8.,float(factor)))
        self.update()

    def _draw_image(self, painter, image, area, label):
        if self.ad_style:
            rect = self.fitted(area, image)
            zoom = self.output_zoom
            rect = QRectF(area.center().x()-rect.width()*zoom/2, area.center().y()-rect.height()*zoom/2, rect.width()*zoom, rect.height()*zoom)
            painter.save()
            painter.setClipRect(area)
            painter.drawImage(rect, image)
            painter.restore()
            self._ad_image_tag(painter, area, '效果预览' if self.processed_image else '效果预览 · 待跟踪')
            painter.setPen(QPen(QColor('#32302d'), 1))
            painter.drawRect(area.adjusted(.5,.5,-.5,-.5))
            return rect
        painter.setPen(QColor("#aaa49b"))
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        painter.drawText(QRectF(area.x() + 15, area.y() + 8, area.width() - 30, 24), label)
        content = area.adjusted(12, 42, -12, -18)
        rect = self.fitted(content, image)
        painter.drawImage(rect, image)
        painter.setPen(QPen(QColor("#32302d"), 1))
        painter.drawRect(rect)
        return rect

    def _draw_source(self, painter, area):
        self._sync_source_geometry()
        if self.ad_style:
            painter.save()
            painter.setClipRect(self.source_viewport)
            painter.drawImage(self.source_rect, self.source_image)
            painter.restore()
            self._ad_image_tag(painter, area, '原片')
            painter.setPen(QPen(QColor('#32302d'), 1))
            painter.drawRect(area.adjusted(.5,.5,-.5,-.5))
            return
        painter.setPen(QColor("#aaa49b"))
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        label = "原片  /  在此框选目标"
        if self._zoom_factor > 1:
            label = f"原片  /  {self._zoom_factor:.1f}×  ·  中键或空格拖动"
        painter.drawText(QRectF(area.x() + 15, area.y() + 8, area.width() - 30, 24), label)
        painter.save()
        painter.setClipRect(self.source_viewport)
        painter.drawImage(self.source_rect, self.source_image)
        painter.setPen(QPen(QColor("#32302d"), 1))
        painter.drawRect(self.source_rect)
        painter.restore()

    def _ad_image_tag(self, painter, area, text):
        font = QFont('Microsoft YaHei UI')
        font.setPixelSize(13)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text)+20
        tag = QRectF(area.x()+8,area.y()+8,width,25)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(16,16,16,220))
        painter.drawRoundedRect(tag,3,3)
        painter.setPen(QColor('#eadfce'))
        painter.drawText(tag,Qt.AlignmentFlag.AlignCenter,text)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(CANVAS))
        if self.source_image is not None and not self.ad_style:
            painter.setPen(QPen(QColor(52, 71, 82, 38), 1))
            for x in range(0, self.width(), 48):
                painter.drawLine(x, 0, x, self.height())
            for y in range(0, self.height(), 48):
                painter.drawLine(0, y, self.width(), y)
        if self.source_image is None and self.persistent_compare:
            mid = (self.width()-12)/2 if self.ad_style else self.width()/2
            painter.setFont(QFont("Microsoft YaHei UI", 10))
            for x, name, message in ((0, "原片", "导入视频，在左侧选择跟踪目标"),
                                     (mid+(12 if self.ad_style else 0), "效果预览", "跟踪后在此查看构图效果")):
                area = QRectF(x, 0, mid, self.height())
                painter.setPen(QColor("#c1b9af"))
                painter.drawRect(area.adjusted(.5,.5,-.5,-.5))
                painter.drawText(area.adjusted(15, 8, -15, -8), name)
                painter.drawText(area.adjusted(15, 42, -15, -18), Qt.AlignmentFlag.AlignCenter, message)
            painter.setPen(QColor("#32302d"))
            if not self.ad_style:painter.drawLine(QPointF(mid, 0), QPointF(mid, self.height()))
            return
        if self.source_image is None:
            cx, cy = self.width() / 2, self.height() / 2 - 64
            painter.drawPixmap(round(cx - 29), round(cy - 29), brand_pixmap(58))
            painter.setPen(QColor("#efebe4"))
            painter.setFont(QFont("Microsoft YaHei UI", 19, QFont.Weight.DemiBold))
            painter.drawText(QRectF(0, cy + 56, self.width(), 40), Qt.AlignmentFlag.AlignCenter,
                             "先选镜头，再锁定主体")
            painter.setPen(QColor("#aaa49b"))
            painter.setFont(QFont("Microsoft YaHei UI", 9))
            painter.drawText(QRectF(15, cy + 108, self.width() - 30, 24), Qt.AlignmentFlag.AlignCenter,
                             "导入或拖入视频，选中想锁住的细节")
            painter.setPen(QColor("#827b72"))
            painter.drawText(QRectF(15, cy + 140, self.width() - 30, 24), Qt.AlignmentFlag.AlignCenter,
                             "裁剪片段  /  跟踪复核  /  调整构图  /  导出")
            return
        whole = QRectF(self.rect())
        self._sync_source_geometry()
        if self.mode == "compare":
            half = (self.width()-12)/2 if self.ad_style else self.width()/2
            left = QRectF(0, 0, half, self.height())
            right = QRectF(half+(12 if self.ad_style else 0), 0, half, self.height())
            self._draw_source(painter, left)
            output_rect = self._draw_image(painter, self.processed_image or self.source_image,
                                            right, "效果预览  /  与左侧同一源时刻" if self.processed_image else "效果预览  /  待跟踪完成，当前显示原片")
            if not self.ad_style:
                painter.setPen(QPen(QColor('#32302d'), 1))
                painter.drawLine(QPointF(self.width()/2,0),QPointF(self.width()/2,self.height()))
        elif self.mode == "source":
            self._draw_source(painter, whole)
            output_rect = QRectF()
        else:
            output_rect = self._draw_image(painter, self.processed_image or self.source_image,
                                            whole, "稳定后" if self.processed_image else "请先框选并跟踪目标")
        painter.save()
        painter.setClipRect(self.source_viewport)
        if (self.region_overlay_visible and self.source_rect.isValid() and self.box is not None and self.homography is not None
                and getattr(self, "confidence_grid", None) is not None
                and self._edit is None and self._region_drag is None):
            from .planar import project
            painter.save()
            painter.setClipRect(self.source_viewport)
            surface = self._surface_polygon()
            if surface is not None:
                clip_path = QPainterPath()
                clip_path.addPolygon(surface)
                painter.setClipPath(clip_path, Qt.ClipOperation.IntersectClip)
            for row in range(4):
                for col in range(4):
                    value = float(self.confidence_grid[row, col])
                    color = QColor(120, 116, 108, 45) if value < 0 else QColor.fromHsvF(.25*value, .32, .64, .23)
                    unit = np.array([[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]]) / 4
                    points = project(unit, self.homography)
                    polygon = QPolygonF([QPointF(self.source_rect.x() + x * self.source_rect.width(),
                                                self.source_rect.y() + y * self.source_rect.height()) for x, y in points])
                    painter.setBrush(color)
                    painter.setPen(QPen(color.lighter(150), 1))
                    painter.drawPolygon(polygon)
            painter.restore()
        if self.region_overlay_visible and self.source_rect.isValid() and self.box is not None:
            x, y, w, h = self.box
            rect = QRectF(self.source_rect.x() + x * self.source_rect.width(),
                          self.source_rect.y() + y * self.source_rect.height(),
                          w * self.source_rect.width(), h * self.source_rect.height())
            color = QColor('#d67d61' if self.state == 'lost' else '#e88c51')
            painter.setPen(QPen(color, 2))
            polygon = (self._surface_polygon() if self._edit is None and self._region_drag is None
                       and self.homography is not None else None)
            if self.selection_shape == "polygon" and polygon is None:
                polygon = self._region_screen_polygon(projected=False)
            if polygon is not None:
                painter.setBrush(QColor(232, 140, 81, 20))
                if self.selection_shape == "polygon":
                    painter.setPen(QPen(color, 1.5, Qt.PenStyle.DashLine))
                painter.drawPolygon(polygon)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
                painter.drawRect(rect)
            elif self.selection_shape in {"circle", "ellipse"}:
                painter.drawEllipse(rect)
                painter.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
                painter.drawRect(rect)
            else:
                painter.drawRect(rect)
            painter.setPen(QPen(color, 1))
            center = rect.center()
            if polygon is not None and self.homography is not None and self._region_drag is None:
                from .planar import project
                x, y = project([[.5, .5]], self.homography)[0]
                center = QPointF(self.source_rect.x() + x * self.source_rect.width(),
                                 self.source_rect.y() + y * self.source_rect.height())
            painter.drawLine(center + QPointF(-7, 0), center + QPointF(7, 0))
            painter.drawLine(center + QPointF(0, -7), center + QPointF(0, 7))
            if self.selectable:
                handles = (enumerate(polygon) if self.selection_shape == "polygon" and polygon is not None
                           else self._handles(rect))
                for index, point in handles:
                    active = self.selection_shape == "polygon" and index == self.selected_vertex
                    handle = QRectF(point.x() - 3.5, point.y() - 3.5, 7, 7)
                    painter.fillRect(handle, color if self.ad_style or active else QColor('#202020'))
                    painter.drawRect(handle)
            text = "待修正" if self.state == "lost" else "手动关键帧" if self.state == "manual" else "跟踪目标"
            painter.setPen(color)
            painter.drawText(QPointF(rect.left(), max(self.source_viewport.top() + 16, rect.top() - 9)), text)
        if self.drag_start is not None and self.drag_end is not None:
            rect = self._point_rect(self.drag_end) if self.selection_mode == "point" else self._drag_rect(self.drag_end)
            painter.fillRect(rect, QColor(234, 161, 143, 25))
            painter.setPen(QPen(QColor("#e88c51"), 1.5, Qt.PenStyle.DashLine))
            if self.selection_shape == "polygon" and len(self.region_points) >= 3:
                painter.drawPolygon(QPolygonF([QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())
                                               for x, y in self.region_points]))
            elif self.selection_shape in {"circle", "ellipse"}:
                painter.drawEllipse(rect)
            else:
                painter.drawRect(rect)
        painter.restore()
        self.output_rect = QRectF(output_rect)
        if output_rect.isValid() and self.processed_image:
            target = QPointF(output_rect.x() + self.target[0] * output_rect.width(),
                             output_rect.y() + self.target[1] * output_rect.height())
            painter.setPen(QPen(QColor('#e88c51'), 1.5))
            if self.ad_style:
                painter.drawEllipse(target,14,14)
            painter.drawLine(target + QPointF(-10, 0), target + QPointF(10, 0))
            painter.drawLine(target + QPointF(0, -10), target + QPointF(0, 10))
        if (self.selectable and self.hover_position is not None and self.source_rect.isValid()
                and self._in_source(self.hover_position) and self._pan_start is None and not self._space_pressed
                and self.selection_mode != "adjust"):
            self._draw_magnifier(painter)

    def _surface_polygon(self):
        """Project the selected mask contour through the saved full homography."""
        from .planar import UNIT_CORNERS, project
        if self.homography is None or not self.source_rect.isValid():
            return None
        points = UNIT_CORNERS
        if self.selection_shape in {"circle", "ellipse"}:
            angles = np.linspace(0, 2 * np.pi, 65)[:-1]
            points = np.column_stack([.5 + .5 * np.cos(angles), .5 + .5 * np.sin(angles)])
        elif self.selection_shape == "polygon":
            points = self.region_points
            if len(points) < 3:
                return None
        points = project(points, self.homography)
        if not np.isfinite(points).all():
            return None
        return QPolygonF([QPointF(self.source_rect.x() + x * self.source_rect.width(),
                                 self.source_rect.y() + y * self.source_rect.height()) for x, y in points])

    def _selection_rect(self):
        if self.box is None or not self.source_rect.isValid():
            return QRectF()
        x, y, w, h = self.box
        return QRectF(self.source_rect.x() + x * self.source_rect.width(),
                      self.source_rect.y() + y * self.source_rect.height(),
                      w * self.source_rect.width(), h * self.source_rect.height())

    def _region_source_points(self, projected=True):
        if self.box is None or len(self.region_points) < 3:
            return None
        points = np.asarray(self.region_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            return None
        if projected and self.homography is not None and self._region_drag is None:
            from .planar import project
            points = project(points, self.homography)
        else:
            box = np.asarray(self.box, dtype=float)
            points = box[:2] + points * box[2:]
        return points if np.isfinite(points).all() else None

    def _region_screen_polygon(self, projected=True):
        if not self.source_rect.isValid():
            return None
        points = self._region_source_points(projected)
        if points is None:
            return None
        return QPolygonF([QPointF(self.source_rect.x() + x * self.source_rect.width(),
                                 self.source_rect.y() + y * self.source_rect.height()) for x, y in points])

    @staticmethod
    def _nearest_edge(position, polygon):
        nearest, distance = 0, float("inf")
        point = np.array([position.x(), position.y()])
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            a, b = np.array([start.x(), start.y()]), np.array([end.x(), end.y()])
            edge = b - a
            denominator = float(edge @ edge)
            t = 0. if denominator <= 1e-20 else float(np.clip((point - a) @ edge / denominator, 0, 1))
            gap = float(np.linalg.norm(point - (a + t * edge)))
            if gap < distance:
                nearest, distance = index, gap
        return nearest, distance

    @staticmethod
    def _region_payload(points):
        """Make a bounded, closed simple-polygon proposal without mutating a model."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 64:
            return None
        if not np.isfinite(points).all() or np.any(points < -1e-10) or np.any(points > 1 + 1e-10):
            return None
        points = np.clip(points, 0, 1)
        lower, upper = points.min(axis=0), points.max(axis=0)
        size = upper - lower
        if np.any(size <= 1e-8):
            return None
        from .regions import validate_polygon
        try:
            local = validate_polygon(((points - lower) / size).tolist())
        except (ValueError, TypeError):
            return None
        return {"shape": "polygon", "box": tuple(float(v) for v in (*lower, *size)),
                "points": tuple(tuple(float(v) for v in point) for point in local)}

    def _propose_region(self, points):
        payload = self._region_payload(points)
        if payload is None:
            self.selection_rejected.emit("区域不能自交、重合或退化，请调整顶点位置")
            return False
        self.region_selected.emit(payload)
        return True

    def delete_region_vertex(self):
        """Propose one deletion. The host commits it through the shared edit history."""
        if not self.can_delete_vertex or self.mode == "output" or self.composition_editable:
            return False
        points = self._region_source_points()
        if points is None:
            return False
        self.edit_started.emit()
        accepted = self._propose_region(np.delete(points, self.selected_vertex, axis=0))
        if accepted:
            self.selected_vertex = -1
        return accepted

    def _add_region_vertex(self, position):
        points = self._region_source_points()
        polygon = self._region_screen_polygon()
        if points is None or polygon is None or len(points) >= 64:
            if points is not None and len(points) >= 64:
                self.selection_rejected.emit("多边形最多保留 64 个顶点")
            return
        index, _ = self._nearest_edge(position, polygon)
        point = [(position.x() - self.source_rect.x()) / self.source_rect.width(),
                 (position.y() - self.source_rect.y()) / self.source_rect.height()]
        self.edit_started.emit()
        if self._propose_region(np.insert(points, index + 1, point, axis=0)):
            self.selected_vertex = index + 1

    def _update_region_drag(self, position):
        drag = self._region_drag
        points = drag["source"].copy()
        delta = np.array([(position.x() - drag["start"].x()) / self.source_rect.width(),
                          (position.y() - drag["start"].y()) / self.source_rect.height()])
        if drag["action"] == "move":
            delta = np.clip(delta, -points.min(axis=0), 1 - points.max(axis=0))
            points += delta
        else:
            points[drag["action"][1]] = np.clip(points[drag["action"][1]] + delta, 0, 1)
        drag["candidate"] = points
        payload = self._region_payload(points)
        if payload is not None:
            self.box, self.region_points = payload["box"], payload["points"]
        self.update()

    def _handles(self, rect):
        points = [rect.topLeft(), QPointF(rect.center().x(), rect.top()), rect.topRight(),
                  QPointF(rect.right(), rect.center().y()), rect.bottomRight(),
                  QPointF(rect.center().x(), rect.bottom()), rect.bottomLeft(),
                  QPointF(rect.left(), rect.center().y())]
        return [(i, p) for i, p in enumerate(points)
                if self.selection_shape not in {"circle", "ellipse"} or i % 2 == 0]

    def _hit_selection(self, point):
        rect = self._selection_rect()
        if not rect.isValid() or not self.source_viewport.contains(point):
            return None
        if self.selection_shape == "polygon":
            polygon = self._region_screen_polygon()
            if polygon is None:
                return None
            for index, vertex in enumerate(polygon):
                if (point.x() - vertex.x()) ** 2 + (point.y() - vertex.y()) ** 2 <= 7 ** 2:
                    return ("vertex", index)
            if polygon.containsPoint(point, Qt.FillRule.OddEvenFill):
                return "move"
            if self._nearest_edge(point, polygon)[1] <= 7:
                return "move"
            return None
        for index, handle in self._handles(rect):
            if abs(point.x() - handle.x()) <= 7 and abs(point.y() - handle.y()) <= 7:
                return index
        if self.selection_shape in {"circle", "ellipse"}:
            if self.selection_mode != "adjust":
                return None
            x = (point.x() - rect.center().x()) / (rect.width() / 2)
            y = (point.y() - rect.center().y()) / (rect.height() / 2)
            if x * x + y * y <= 1:
                return "move"
            outline = QPolygonF([QPointF(rect.center().x() + math.cos(t) * rect.width() / 2,
                                        rect.center().y() + math.sin(t) * rect.height() / 2)
                                 for t in np.linspace(0, 2 * np.pi, 65)[:-1]])
            return "move" if self._nearest_edge(point, outline)[1] <= 7 else None
        return "move" if self.selection_mode == "adjust" and rect.contains(point) else None

    def _drag_rect(self, position):
        bounds = self.source_rect.intersected(self.source_viewport)
        x = min(bounds.right(), max(bounds.left(), position.x()))
        y = min(bounds.bottom(), max(bounds.top(), position.y()))
        if self.selection_shape == "circle":
            dx, dy = x - self.drag_start.x(), y - self.drag_start.y()
            side = min(abs(dx), abs(dy))
            x = self.drag_start.x() + math.copysign(side, dx)
            y = self.drag_start.y() + math.copysign(side, dy)
        return QRectF(self.drag_start, QPointF(x, y)).normalized().intersected(bounds)

    def _update_edit(self, position):
        action, start, original = self._edit
        delta = position - start
        width, height = self.source_image.width(), self.source_image.height()
        dx = delta.x() / self.source_rect.width() * width
        dy = delta.y() / self.source_rect.height() * height
        x, y, w, h = original * [width, height, width, height]
        if action == "move":
            x = min(width - w, max(0., x + dx))
            y = min(height - h, max(0., y + dy))
        else:
            right, bottom = x + w, y + h
            if action in (0, 6, 7):
                x = max(0., min(right - 12., x + dx))
            if action in (2, 3, 4):
                right = min(width, max(x + 12., right + dx))
            if action in (0, 1, 2):
                y = max(0., min(bottom - 12., y + dy))
            if action in (4, 5, 6):
                bottom = min(height, max(y + 12., bottom + dy))
            w, h = right - x, bottom - y
            if self.selection_shape == "circle":
                side = min(w, h)
                if action in (0, 6):
                    x = right - side
                if action in (0, 2):
                    y = bottom - side
                w = h = side
        self.box = np.array([x / width, y / height, w / width, h / height])
        self.update()

    def _point_rect(self, position):
        if self.source_image is None or not self.source_rect.isValid():
            return QRectF()
        width = self.point_size / self.source_image.width() * self.source_rect.width()
        height = self.point_size / self.source_image.height() * self.source_rect.height()
        return QRectF(position.x() - width / 2, position.y() - height / 2, width, height)

    def _draw_magnifier(self, painter):
        position = self.hover_position
        image = self.source_image
        px = (position.x() - self.source_rect.x()) / self.source_rect.width() * image.width()
        py = (position.y() - self.source_rect.y()) / self.source_rect.height() * image.height()
        radius = 22
        left, top = max(0, int(px) - radius), max(0, int(py) - radius)
        right, bottom = min(image.width(), int(px) + radius + 1), min(image.height(), int(py) + radius + 1)
        if right <= left or bottom <= top:
            return
        viewport = self.source_viewport
        width = min(172.0, viewport.width() - 16, viewport.height() - 42)
        if width < 60:
            return
        image_size = width - 16
        box = QRectF(viewport.right() - width - 8, viewport.top() + 8, width, image_size + 42)
        if box.contains(position):
            box.moveLeft(viewport.left() + 8)
        painter.save()
        painter.setClipRect(viewport)
        painter.fillRect(box, QColor("#24211d"))
        painter.setPen(QPen(QColor("#e88c51"), 1))
        painter.drawRect(box)
        region = QRectF(box.x() + 8, box.y() + 8, image_size, image_size)
        painter.drawImage(region, image.copy(left, top, right - left, bottom - top))
        cross = QPointF(region.x() + (px - left) / (right - left) * region.width(),
                        region.y() + (py - top) / (bottom - top) * region.height())
        painter.drawLine(cross + QPointF(-13, 0), cross + QPointF(13, 0))
        painter.drawLine(cross + QPointF(0, -13), cross + QPointF(0, 13))
        painter.setFont(QFont("Microsoft YaHei UI", 9))
        painter.drawText(QRectF(box.x() + 8, region.bottom() + 6, image_size, 22), Qt.AlignmentFlag.AlignCenter,
                         "局部放大 · 精准选点")
        if self.selection_mode == "point":
            painter.setPen(QPen(QColor("#e88c51"), 1, Qt.PenStyle.DashLine))
            painter.drawRect(self._point_rect(position).intersected(self.source_rect))
        painter.restore()

    def _move_composition(self, point):
        rect = self.output_rect
        if not rect.isValid():
            return
        x = max(.1, min(.9, (point.x() - rect.x()) / rect.width()))
        y = max(.1, min(.9, (point.y() - rect.y()) / rect.height()))
        self.target = (x, y)
        self.composition_changed.emit(x, y)
        self.update()

    def mousePressEvent(self, event):
        self._sync_source_geometry()
        if self.composition_editable:
            if (event.button() == Qt.MouseButton.LeftButton and self.processed_image is not None
                    and self.output_rect.contains(event.position())):
                self._composition_drag = True
                self._move_composition(event.position())
            event.accept()
            return
        pan = (event.button() == Qt.MouseButton.MiddleButton or
               (event.button() == Qt.MouseButton.LeftButton and self._space_pressed))
        if pan and self.source_viewport.contains(event.position()):
            self._cancel_edit()
            self._pan_start = event.position()
            self._pan_center = QPointF(self._view_center)
            self._pan_button = event.button()
            self.drag_start = self.drag_end = None
            self.hover_position = None
            self.setFocus()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (event.button() == Qt.MouseButton.LeftButton and self.selectable and
                self._in_source(event.position())):
            hit = self._hit_selection(event.position())
            if self.selection_shape == "polygon" and self.box is not None:
                if self.region_edit_mode == "add":
                    self.setFocus()
                    self._add_region_vertex(event.position())
                    event.accept()
                    return
                if self.region_edit_mode == "delete":
                    self.selected_vertex = hit[1] if isinstance(hit, tuple) and hit[0] == "vertex" else -1
                    self.setFocus()
                    self.delete_region_vertex()
                    event.accept()
                    return
                if hit is not None and not event.modifiers() & Qt.KeyboardModifier.AltModifier:
                    points = self._region_source_points()
                    if points is not None:
                        self.selected_vertex = hit[1] if isinstance(hit, tuple) else -1
                        self._region_drag = {"action": hit, "start": event.position(), "source": points.copy(),
                                             "box": tuple(self.box), "points": tuple(self.region_points),
                                             "candidate": points.copy()}
                        self.edit_started.emit()
                        self.setFocus()
                        event.accept()
                        return
                self.selected_vertex = -1
            if hit is not None and not event.modifiers() & Qt.KeyboardModifier.AltModifier:
                self._edit = (hit, event.position(), np.asarray(self.box).copy())
                self.edit_started.emit()
                self.setFocus()
                event.accept()
                return
            self.edit_started.emit()
            self.drag_start = event.position()
            self.drag_end = event.position()
            self.hover_position = event.position()
            self.setFocus()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._sync_source_geometry()
        if self.composition_editable:
            if self._composition_drag:
                self._move_composition(event.position())
            self.setCursor(Qt.CursorShape.SizeAllCursor if self.output_rect.contains(event.position()) else Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self._region_drag is not None:
            if not self.selectable:
                self._cancel_edit()
            else:
                self._update_region_drag(event.position())
            event.accept()
            return
        if self._edit is not None:
            self._update_edit(event.position())
            event.accept()
            return
        if self._pan_start is not None and self.source_rect.isValid():
            delta = event.position() - self._pan_start
            if self._space_pressed and abs(delta.x()) + abs(delta.y()) >= 3:
                self._space_pan_moved = True
            self._view_center = QPointF(self._pan_center.x() - delta.x() / self.source_rect.width(),
                                        self._pan_center.y() - delta.y() / self.source_rect.height())
            self._sync_source_geometry()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()
            event.accept()
            return
        self.hover_position = event.position()
        if self.drag_start is not None:
            self.drag_end = event.position()
            self.update()
        elif self._space_pressed and self.source_viewport.contains(event.position()):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif self.selectable and self._in_source(event.position()):
            hit = self._hit_selection(event.position())
            cursor = (Qt.CursorShape.CrossCursor if self.selection_shape == "polygon" and self.region_edit_mode == "add" else
                      Qt.CursorShape.PointingHandCursor if self.selection_shape == "polygon" and isinstance(hit, tuple) else
                      Qt.CursorShape.SizeAllCursor if hit == "move" else
                      Qt.CursorShape.SizeFDiagCursor if hit in (0, 4) else
                      Qt.CursorShape.SizeBDiagCursor if hit in (2, 6) else
                      Qt.CursorShape.SizeVerCursor if hit in (1, 5) else
                      Qt.CursorShape.SizeHorCursor if hit in (3, 7) else Qt.CursorShape.CrossCursor)
            self.setCursor(cursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def leaveEvent(self, event):
        self.hover_position = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        self._sync_source_geometry()
        if self._composition_drag:
            self._move_composition(event.position())
            self._composition_drag = False
            event.accept()
            return
        if self._region_drag is not None and event.button() == Qt.MouseButton.LeftButton:
            if self.selectable:
                self._update_region_drag(event.position())
            drag = self._region_drag
            changed = not np.allclose(drag["candidate"], drag["source"], atol=1e-12)
            # Restore the authoritative view before notifying the host. A host
            # rejection never leaves a partly committed polygon on screen.
            self.box, self.region_points = drag["box"], drag["points"]
            self._region_drag = None
            if self.selectable and changed:
                self._propose_region(drag["candidate"])
            self.update()
            event.accept()
            return
        if self._edit is not None and event.button() == Qt.MouseButton.LeftButton:
            self._update_edit(event.position())
            original = self._edit[2]
            self._edit = None
            if not np.allclose(self.box, original, atol=1e-12):
                self.roi_selected.emit(tuple(self.box))
            event.accept()
            return
        if self._pan_start is not None and event.button() == self._pan_button:
            self._pan_start = self._pan_center = self._pan_button = None
            self.setCursor(Qt.CursorShape.OpenHandCursor if self._space_pressed else Qt.CursorShape.ArrowCursor)
            self.update()
            event.accept()
            return
        if self.drag_start is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        rect = self._drag_rect(event.position())
        point_selection = self.selection_mode == "point" or (rect.width() < 5 and rect.height() < 5)
        if point_selection:
            rect = self._point_rect(event.position())
            if not self._in_source(event.position()):
                rect = QRectF()
            elif not self.source_rect.contains(rect):
                self.selection_rejected.emit("选点太靠近边缘，请缩小跟踪区域或换一个清晰细节")
                rect = QRectF()
        if rect.isValid() and rect.width() > 0 and rect.height() > 0:
            roi = ((rect.left() - self.source_rect.left()) / self.source_rect.width(),
                   (rect.top() - self.source_rect.top()) / self.source_rect.height(),
                   rect.width() / self.source_rect.width(), rect.height() / self.source_rect.height())
            self.roi_selected.emit(roi)
        self.drag_start = self.drag_end = None
        self.update()

    def wheelEvent(self, event):
        self._sync_source_geometry()
        if self.source_viewport.contains(event.position()):
            delta = event.angleDelta().y()
            if not delta:
                delta = event.pixelDelta().y()
            if delta:
                self.set_zoom(self._zoom_factor * (1.2 ** (delta / 120)), event.position())
            event.accept()
            return
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cancel_edit()
            self.drag_start = self.drag_end = None
            self.update()
            event.accept()
            return
        movement = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                    Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if self.selectable and not self.composition_editable and self.mode != "output" and self.source_image is not None and self.selection_mode == "adjust" and self.box is not None and event.key() in movement:
            step = 10. if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else .1 if event.modifiers() & Qt.KeyboardModifier.AltModifier else 1.
            width, height = self.source_image.width(), self.source_image.height()
            dx, dy = movement[event.key()]
            if self.selection_shape == "polygon":
                points = self._region_source_points()
                if points is not None:
                    delta = np.array([dx * step / width, dy * step / height])
                    if 0 <= self.selected_vertex < len(points):
                        points[self.selected_vertex] = np.clip(points[self.selected_vertex] + delta, 0, 1)
                    else:
                        points += np.clip(delta, -points.min(axis=0), 1 - points.max(axis=0))
                    self.edit_started.emit()
                    self._propose_region(points)
                event.accept()
                return
            x, y, w, h = self.box
            self.edit_started.emit()
            box = (min(1 - w, max(0., x + dx * step / width)),
                   min(1 - h, max(0., y + dy * step / height)), w, h)
            self.box = box
            self.roi_selected.emit(box)
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self._space_pan_moved = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.update()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            request_playback = self._space_pressed and not self._space_pan_moved
            self._space_pressed = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor if self._pan_start is not None
                           else Qt.CursorShape.ArrowCursor)
            self.update()
            if request_playback:
                self.playback_requested.emit()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self._cancel_edit()
        self._space_pressed = False
        self._space_pan_moved = False
        self._pan_start = self._pan_center = self._pan_button = None
        self.drag_start = self.drag_end = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
        super().focusOutEvent(event)

    def resizeEvent(self, event):
        self._cancel_edit()
        self.drag_start = self.drag_end = None
        self._pan_start = self._pan_center = self._pan_button = None
        self._sync_source_geometry()
        super().resizeEvent(event)

    def _cancel_edit(self):
        if self._region_drag is not None:
            self.box = self._region_drag["box"]
            self.region_points = self._region_drag["points"]
        self._region_drag = None
        if self._edit is not None:
            self.box = self._edit[2]
        self._edit = None


class TrackStrip(QWidget):
    seek = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(27)
        self.track = None
        self.frames = 0
        self.current = 0
        self.anchors = {}
        self.start = 0
        self.end = 0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#242a32"))
        if not self.frames:
            return
        count = self.frames
        if self.end >= self.start:
            left = self.start / count * self.width()
            width = (self.end - self.start + 1) / count * self.width()
            painter.fillRect(QRectF(left, 6, width, 15), QColor("#32302d"))
        if self.track:
            from .tracking_pause import computed_mask
            measured = computed_mask(self.track)
            for i, state in enumerate(self.track.states):
                if not measured[i]:
                    continue
                color = {"tracked": "#577d72", "recovered": "#bd9c88", "manual": "#e88c51", "lost": "#cd786d"}[state]
                x = (self.track.start + i) / count * self.width()
                painter.fillRect(QRectF(x, 7, max(1.0, self.width() / count), 13), QColor(color))
        painter.setPen(QPen(QColor("#e88c51"), 2))
        for frame in self.anchors:
            x = (frame + 0.5) / count * self.width()
            painter.drawLine(QPointF(x, 2), QPointF(x, 24))
        x = (self.current + 0.5) / count * self.width()
        painter.setPen(QPen(QColor("#e8f0fa"), 1))
        painter.drawLine(QPointF(x, 0), QPointF(x, 27))

    def mousePressEvent(self, event):
        if self.frames:
            self.seek.emit(max(0, min(self.frames - 1, int(event.position().x() / self.width() * self.frames))))
