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
from typing import Any, Callable, Dict, Optional, Tuple

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


def write(plot_item: Any, request: FigureRequest) -> str:
    """
    Write one plot to ``request.path`` and return the path.

    ``plot_item`` is a PlotItem (``widget.getPlotItem()``), not the
    widget: exporting the widget would include the frame Qt draws around
    it, which is part of the application and not of the figure.
    """
    folder = os.path.dirname(os.path.abspath(request.path))
    if folder and not os.path.isdir(folder):
        raise OSError(f"There is no folder {folder}")

    if request.is_svg:
        exporter = SVGExporter(plot_item)
        # SVG has no resolution: its width is in points, and the journal
        # scales it. The width asked for in millimetres is kept by
        # exporting at 72 points per inch, which is what a point is.
        exporter.parameters()["width"] = pixels_for(request.width_mm, 72)
    else:
        exporter = ImageExporter(plot_item)
        exporter.parameters()["width"] = pixels_for(request.width_mm,
                                                    request.dpi)
    exporter.export(request.path)
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


def on_white(redraw: Callable[[bool], None]) -> Callable[[], None]:
    """
    Redraw on white, and give back the call that puts the panel back.

    The panels draw on black because that is what a screen is read on;
    a figure is printed on white. Rather than translating colours at
    export time -- which is where a white contour line on a white page
    comes from -- the plot is drawn again with the light palette, saved,
    and drawn again as it was.
    """
    redraw(False)

    def restore() -> None:
        redraw(True)

    return restore
