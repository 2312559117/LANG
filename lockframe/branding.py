"""Original vector mark; also supplies the Windows application icon."""
from pathlib import Path

from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def brand_pixmap(size=64):
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / 64, size / 64)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor('#e95532'))
    tomato = QPainterPath()
    tomato.moveTo(32, 18)
    tomato.cubicTo(5, 8, 2, 39, 13, 51)
    tomato.cubicTo(22, 62, 43, 62, 52, 49)
    tomato.cubicTo(65, 30, 52, 9, 32, 18)
    painter.drawPath(tomato)
    painter.setBrush(QColor('#f17b4e'))
    painter.drawEllipse(QRectF(13,19,8,18))
    painter.setBrush(QColor('#7f885d'))
    leaf = QPainterPath()
    leaf.moveTo(31, 25)
    leaf.lineTo(23, 18)
    leaf.lineTo(30, 19)
    leaf.lineTo(34, 12)
    leaf.lineTo(35, 20)
    leaf.lineTo(43, 18)
    leaf.lineTo(36, 26)
    leaf.closeSubpath()
    painter.drawPath(leaf)
    painter.setPen(QPen(QColor('#92996d'),3,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap))
    painter.drawLine(33,19,38,5)
    painter.end()
    return pixmap


def application_icon():
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(brand_pixmap(size))
    return icon


def write_assets(directory):
    import struct
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "tomato.png"
    brand_pixmap(256).save(str(target))
    sizes = (16, 24, 32, 48, 64, 128, 256)
    entries, payloads = [], []
    offset = 6 + 16 * len(sizes)
    for size in sizes:
        buffer_data = QByteArray()
        buffer = QBuffer(buffer_data)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        brand_pixmap(size).save(buffer, "PNG")
        buffer.close()
        data = bytes(buffer_data)
        entries.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset))
        payloads.append(data)
        offset += len(data)
    (folder / "tomato.ico").write_bytes(struct.pack("<HHH", 0, 1, len(sizes)) + b"".join(entries + payloads))
    for direction in ("up", "down"):
        arrow = QPixmap(12, 8)
        arrow.fill(Qt.GlobalColor.transparent)
        painter = QPainter(arrow)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#bfb5a7"), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        y0, y1 = (5, 2) if direction == "up" else (2, 5)
        painter.drawLine(3, y0, 6, y1)
        painter.drawLine(6, y1, 9, y0)
        painter.end()
        arrow.save(str(folder / f"chevron-{direction}.png"))


if __name__ == "__main__":
    import os
    import sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    write_assets(sys.argv[1] if len(sys.argv) > 1 else "assets")
