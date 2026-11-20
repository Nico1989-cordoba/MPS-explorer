# -*- coding: utf-8 -*-
"""
Saving a plot as a figure: the one on screen, at the size a journal asks
for, on the background it will be printed on.

A screenshot of the window is 96 dots per inch and carries the panel's
dark theme into a paper that will be printed on white. This renders the
plot itself instead, through pyqtgraph's own exporters, at a width given
in millimetres and a resolution given in dots per inch -- and, when the
figure is meant for a journal, redraws it on white first, which is what
``dark=False`` through the palette is for.

PNG or SVG, by the extension. SVG is the one to send a journal: the
curves stay curves, so the figure can be scaled and its text edited. PNG
is what a slide or a draft needs, and 600 dpi is the usual floor for line
art.

Since 2026-10-22 (F6): the title can be edited for the figure only, a
legend of the rows the plot draws (the methods on the axon map among them)
can be put in it (``dressed``: both are taken away again after the export),
the dialog never proposes or writes over an existing file unless the
user chose it through Browse... (``free_path``: _2, _3...), and a widget
that is not a plot (the table of 'Compare methods...') is written at a
print resolution (``write_widget``).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from pyqtgraph.exporters import ImageExporter, SVGExporter

# What journals ask for, in millimetres: one column, one and a half, two.
COLUMN_WIDTHS_MM: Tuple[Tuple[str, float], ...] = (
    ("single column (85 mm)", 85.0),
    ("1.5 columns (114 mm)", 114.0),
    ("double column (174 mm)", 174.0),
)
DEFAULT_WIDTH_MM = 85.0
# 600 for line art, 300 for anything with an image in it. Below 300 no
# journal takes it.
DPI_CHOICES: Tuple[int, ...] = (300, 600, 1200)
DEFAULT_DPI = 600

MM_PER_INCH = 25.4


def pixels_for(width_mm: float, dpi: int) -> int:
    """How many pixels wide that is."""
    return max(1, int(round(width_mm / MM_PER_INCH * dpi)))


@dataclass
class FigureRequest:
    """What the user asked for."""

    plot: str
    path: str
    width_mm: float = DEFAULT_WIDTH_MM
    dpi: int = DEFAULT_DPI
    white: bool = True
    # F6 (2026-10-22): the figure's title as typed in the dialog (None: the plot's own; lines joined by " | "), and
    # whether a legend of the rows drawn is put in the figure (the caller gives its entries to ``write``).
    title: Optional[str] = None
    legend: bool = False

    @property
    def is_svg(self) -> bool:
        return os.path.splitext(self.path)[1].lower() == ".svg"


# The extensions a figure is written as: SVG, or a raster format Qt writes
# on every platform. Anything else - no extension, a typo, ".v2" in a name -
# gets ".png" appended, so the file always has a format (QImage.save cannot
# guess one, and returned False without a word).
RASTER_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
KNOWN_EXTENSIONS: Tuple[str, ...] = (".svg",) + RASTER_EXTENSIONS


class FigureNotWritten(OSError):
    """The figure could not be written; the message says why, in words."""


def with_known_extension(path: str) -> str:
    """``path`` with ".png" appended when its extension is missing or not
    one a figure is written as (manual test 2, item 1)."""
    ext = os.path.splitext(path)[1].lower()
    return path if ext in KNOWN_EXTENSIONS else path + ".png"


def _why_not_written(path: str) -> str:
    """Why ``path`` could not be written: the folder, a locked or read-only
    file, or (when nothing else explains it) the format."""
    folder = os.path.dirname(os.path.abspath(path)) or os.getcwd()
    if not os.path.isdir(folder):
        return f"there is no folder {folder}"
    if os.path.exists(path):
        if not os.access(path, os.W_OK):
            return (f"the file {os.path.basename(path)} is read-only: choose another name, or clear its "
                    f"read-only attribute")
        try:
            with open(path, "ab"):
                pass
        except OSError:
            return (f"the file {os.path.basename(path)} is open in another program (locked): close it there, "
                    f"or choose another name")
    probe = os.path.join(folder, f".mps_write_probe_{os.getpid()}")
    try:
        with open(probe, "wb"):
            pass
        os.remove(probe)
    except OSError:
        return f"the folder {folder} cannot be written to: choose another folder"
    return (f"the image could not be encoded as {os.path.splitext(path)[1] or 'an image'}: "
            f"choose .png or .svg")


# A legend with more rows than this is laid out in two columns [ad hoc: keeps it within a map's height].
LEGEND_ONE_COLUMN = 6
# Title lines are typed on one line, joined by this (the plots' titles are HTML lines joined by <br>).
TITLE_LINE_JOIN = " | "


def plain_title(plot_item: Any) -> str:
    """A plot's title as one line of plain text, its lines joined by TITLE_LINE_JOIN (for the dialog's field)."""
    import re
    text = str(getattr(getattr(plot_item, "titleLabel", None), "text", "") or "")
    lines = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", part)).strip() for part in text.split("<br>")]
    return TITLE_LINE_JOIN.join(line for line in lines if line)


def free_path(path: str) -> str:
    """``path``, or ``stem_2.ext``, ``stem_3.ext``... : the first that does not exist (F6: an export never
    overwrites a file it was not told to)."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


@contextlib.contextmanager
def dressed(plot_item: Any, title: Optional[str] = None,
            legend: Optional[Sequence[Tuple[str, Any]]] = None, white: bool = True) -> Iterator[None]:
    """Inside this block the plot carries ``title`` (lines joined by TITLE_LINE_JOIN) and a legend of
    ``legend`` ((label, sample item) pairs, tools.mps_axon_map.legend_entries); both are taken away after it, so
    the window shows what it showed. ``white``: the legend's text and box for a white background."""
    import pyqtgraph as pg
    old_title = None
    old_colour = None
    added = None
    try:
        if title is not None:
            old_title = str(getattr(plot_item.titleLabel, "text", "") or "")
            old_colour = getattr(plot_item.titleLabel, "opts", {}).get("color")
            # the figure's own colour (manual test of 2026-10-23: a plot titled for the dark screen kept its pale
            # grey title on the white figure)
            plot_item.setTitle("<br>".join(part.strip() for part in str(title).split(TITLE_LINE_JOIN.strip()))
                               if title.strip() else None, color="#000000" if white else "#ffffff")
        if legend:
            text = "#000000" if white else "#ffffff"
            fill = (255, 255, 255, 220) if white else (0, 0, 0, 200)
            # compact [ad hoc: 7 pt, two columns past LEGEND_ONE_COLUMN rows], top right, over the data
            added = pg.LegendItem(offset=(-6, 6), labelTextColor=text, brush=pg.mkBrush(*fill),
                                  pen=pg.mkPen(text, width=0.5), labelTextSize="7pt", verSpacing=-6,
                                  colCount=2 if len(legend) > LEGEND_ONE_COLUMN else 1)
            added.setParentItem(plot_item.vb)
            for label, sample in legend:
                added.addItem(sample, label)
        yield
    finally:
        if added is not None:
            scene = added.scene()
            if scene is not None:
                scene.removeItem(added)
        if old_title is not None:
            if old_colour is not None:
                plot_item.setTitle(old_title or None, color=old_colour)
            else:
                plot_item.setTitle(old_title or None)


def write_widget(widget: Any, path: str, dpi: int = DEFAULT_DPI, white: bool = True) -> str:
    """Write a widget (a table, a dialog's content) as a PNG at ``dpi`` (the screen is taken as 96 dpi): for what
    is not a plot, e.g. the table of 'Compare methods...'. Never over an existing file (``free_path``)."""
    from PyQt5 import QtCore, QtGui
    target = free_path(with_known_extension(path))
    scale = max(1.0, float(dpi) / 96.0)
    size = widget.size()
    image = QtGui.QImage(int(size.width() * scale), int(size.height() * scale), QtGui.QImage.Format.Format_ARGB32)
    image.setDotsPerMeterX(int(round(dpi / MM_PER_INCH * 1000)))
    image.setDotsPerMeterY(int(round(dpi / MM_PER_INCH * 1000)))
    image.fill(QtGui.QColor("white") if white else QtGui.QColor("black"))
    painter = QtGui.QPainter(image)
    painter.scale(scale, scale)
    widget.render(painter, QtCore.QPoint(0, 0))
    painter.end()
    if not image.save(target):
        raise FigureNotWritten(f"The figure was not written: {_why_not_written(target)}.")
    return target


def write(plot_item: Any, request: FigureRequest, legend: Optional[List[Tuple[str, Any]]] = None) -> str:
    """
    Write one plot to ``request.path`` and return the path written.

    ``plot_item`` is a PlotItem (``widget.getPlotItem()``), not the
    widget: exporting the widget would include the frame Qt draws around
    it, which is part of the application and not of the figure.

    A path without a known extension gets ".png" (``request.path`` is
    updated, so ``describe`` names the file written). The file is checked
    on disk afterwards: pyqtgraph's image exporter returns QImage.save's
    False without raising, and "Image written" was then said for a file
    that did not exist (manual test 2, item 1). Raises
    ``FigureNotWritten`` with the reason - no folder, a folder that cannot
    be written, a read-only or locked file, a format that cannot be
    encoded.
    """
    request.path = with_known_extension(request.path)
    folder = os.path.dirname(os.path.abspath(request.path))
    if folder and not os.path.isdir(folder):
        raise FigureNotWritten(f"There is no folder {folder}")
    before = (os.path.getmtime(request.path), os.path.getsize(request.path))         if os.path.exists(request.path) else None
    with dressed(plot_item, request.title, legend if request.legend else None, white=request.white):
        ok = _export(plot_item, request)
    written = os.path.isfile(request.path) and os.path.getsize(request.path) > 0
    if written and before is not None and not ok:
        written = (os.path.getmtime(request.path), os.path.getsize(request.path)) != before
    if not (ok and written):
        raise FigureNotWritten(
            f"The figure was not written: {_why_not_written(request.path)}.")
    return request.path


def _settle(plot_item: Any) -> None:
    """Lay out a plot drawn (or restyled) just before its export, so that its items sit where its axes say: a
    ViewBox maps its items to the view range only when it is next painted, and a plot drawn while its tab was
    hidden was exported with axes and no curve (manual test of 2026-10-23, H1 after the null)."""
    from PyQt5 import QtWidgets
    QtWidgets.QApplication.processEvents()
    vb = plot_item.getViewBox() if hasattr(plot_item, "getViewBox") else None
    if vb is not None and hasattr(vb, "prepareForPaint"):
        vb.prepareForPaint()


def _export(plot_item: Any, request: FigureRequest) -> bool:
    """pyqtgraph's exporter for the request's format; True when it says it wrote."""
    if request.is_svg:
        _settle(plot_item)
        exporter = SVGExporter(plot_item)
        # SVG has no resolution: its width is in points, and the journal
        # scales it. The width asked for in millimetres is kept by
        # exporting at 72 points per inch, which is what a point is.
        exporter.parameters()["width"] = pixels_for(request.width_mm, 72)
        try:
            exporter.export(request.path)
        except OSError:
            raise FigureNotWritten(
                f"The figure was not written: {_why_not_written(request.path)}.") from None
        return True
    _settle(plot_item)
    exporter = ImageExporter(plot_item)
    exporter.parameters()["width"] = pixels_for(request.width_mm, request.dpi)
    return bool(exporter.export(request.path))


def describe(request: FigureRequest) -> str:
    """One line saying what was written, for the message box."""
    if request.is_svg:
        return (f"{os.path.basename(request.path)}: {request.width_mm:.0f} mm "
                f"wide, as curves (SVG). Scale it and edit its text in "
                f"Illustrator or Inkscape.")
    pixels = pixels_for(request.width_mm, request.dpi)
    return (f"{os.path.basename(request.path)}: {request.width_mm:.0f} mm at "
            f"{request.dpi} dpi, i.e. {pixels} pixels wide.")


def suggested_name(source: str, plot: str, folder: str = "") -> str:
    """Where to propose writing a figure of this plot of this axon."""
    stem = os.path.splitext(os.path.basename(str(source or "axon")))[0]
    safe = "".join(c if c.isalnum() else "_" for c in plot).strip("_").lower()
    name = f"{stem}_{safe}.png"
    where = folder if folder and os.path.isdir(folder) else (
        os.path.dirname(str(source)) or os.getcwd())
    return os.path.join(where, name)
