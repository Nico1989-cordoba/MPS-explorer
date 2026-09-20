# -*- coding: utf-8 -*-
"""
Draw the program's icon, and write it as assets/mps_explorer.ico.

The icon is the thing the program measures: the centres of the spectrin
clusters around the axon's perimeter, with the contour through them and
its centre. Its colours are the palette's own roles, so the icon and the
plots agree -- localizations sky blue, the contour neutral, the centre
its blue -- on the dark grey a panel is painted.

Run:  python tools/make_icon.py

An icon is drawn rather than kept as a binary blob so it can be changed
with the palette instead of in an image editor, and so a reviewer can
see what it is from the source.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import struct
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mps_plot_style import PANEL_BG, neutral, role  # noqa: E402

# Windows asks for these; the small ones are what the taskbar and the
# file list actually show, and they are why the drawing has to survive
# being 16 pixels across.
SIZES = (256, 128, 64, 48, 32, 16)
# How many cluster centres go round the ring. Eight is what still reads
# as "a ring of separate things" at 16 pixels; the real axons have
# between forty and a hundred.
N_CLUSTERS = 8


def draw(size: int) -> QtGui.QImage:
    """One square of the icon, drawn at ``size`` pixels."""
    image = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

    s = size / 256.0          # everything below is written for 256 px

    # The rounded square behind it, so the ring reads on a light desktop
    # and on a dark one alike.
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(PANEL_BG))
    radius = 48 * s
    painter.drawRoundedRect(QtCore.QRectF(0, 0, size, size), radius, radius)

    centre = size / 2.0
    ring = 76 * s             # radius of the ring of centres

    # The contour through the centres: neutral and thin, as in the plots.
    # At 16 pixels it is left out and the dots are drawn larger instead:
    # a line and eight dots on it come to the same few pixels there, and
    # what should survive is "a ring of separate things", not the line.
    tiny = size < 24
    if not tiny:
        pen = QtGui.QPen(QtGui.QColor(neutral(dark=True)))
        pen.setWidthF(max(1.0, 6 * s))
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(centre, centre), ring, ring)

    # The cluster centres on it.
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(role("locs")))
    dot = max(1.6, (38 if tiny else 26) * s)
    for i in range(N_CLUSTERS):
        angle = 2 * math.pi * i / N_CLUSTERS - math.pi / 2
        painter.drawEllipse(
            QtCore.QPointF(centre + ring * math.cos(angle),
                           centre + ring * math.sin(angle)),
            dot / 2, dot / 2)

    # The centre of the contour, which every plot marks with a cross.
    # Below 32 pixels it would close the middle of the ring into a blob,
    # so it is left out there.
    if size >= 32:
        pen = QtGui.QPen(QtGui.QColor(role("centre")))
        pen.setWidthF(max(1.0, 10 * s))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)
        arm = 20 * s
        painter.drawLine(QtCore.QPointF(centre - arm, centre),
                         QtCore.QPointF(centre + arm, centre))
        painter.drawLine(QtCore.QPointF(centre, centre - arm),
                         QtCore.QPointF(centre, centre + arm))

    painter.end()
    return image


def _png_bytes(image: QtGui.QImage) -> bytes:
    """One size, PNG-compressed, as an icon entry carries it."""
    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QIODevice.WriteOnly)
    if not image.save(buffer, "PNG"):
        raise RuntimeError(f"could not encode the {image.width()} px size")
    return bytes(buffer.data())


def write(path: str) -> str:
    """Write every size into one .ico and return the path.

    The container is assembled here rather than through QImageWriter:
    its ICO handler writes ONE image and indexes only that one, so a
    second call left 100 kB of unreferenced bytes at the end of the file
    and Windows was down-scaling the 256 px drawing for the 16 px slot --
    which is exactly the size the drawing is made legible for.

    Each size goes in PNG-compressed, which every Windows since Vista
    reads and which brings the file from 370 kB to a few.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payloads = [_png_bytes(draw(n)) for n in SIZES]

    # ICONDIR: reserved, type 1 (icon), how many images.
    header = struct.pack("<HHH", 0, 1, len(SIZES))
    offset = len(header) + 16 * len(SIZES)
    entries, body = b"", b""
    for size, png in zip(SIZES, payloads):
        # 256 is written as 0: the field is one byte.
        entries += struct.pack(
            "<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(png), offset)
        body += png
        offset += len(png)

    with open(path, "wb") as handle:
        handle.write(header + entries + body)
    return path


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "assets", "mps_explorer.ico")
    write(out)
    print(f"{out}  ({os.path.getsize(out):,} bytes, "
          f"{len(SIZES)} sizes: {', '.join(str(n) for n in SIZES)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
