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

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


def write(plot_item: Any, request: FigureRequest) -> str:
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

    if request.is_svg:
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
        ok = True
    else:
        exporter = ImageExporter(plot_item)
        exporter.parameters()["width"] = pixels_for(request.width_mm,
                                                    request.dpi)
        ok = bool(exporter.export(request.path))
    written = os.path.isfile(request.path) and os.path.getsize(request.path) > 0
    if written and before is not None and not ok:
        written = (os.path.getmtime(request.path), os.path.getsize(request.path)) != before
    if not (ok and written):
        raise FigureNotWritten(
            f"The figure was not written: {_why_not_written(request.path)}.")
    return request.path


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
