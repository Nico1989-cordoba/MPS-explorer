# -*- coding: utf-8 -*-
"""
Dark styling for the MPS analysis panels' plots.

``MPS_explorer`` sets pyqtgraph's GLOBAL background to white and its
foreground to black, so any plot built from inside the running application
comes out light -- even code that renders dark when run standalone. These
panels are styled dark per widget instead of changing that global, which
would recolour every plot in the application, all of whose palettes were
chosen against white.

The consequence is the thing to remember when editing either panel:
nothing may rely on the global foreground any more. Axis pens, tick text
and every title have to be set explicitly, or they are drawn in the
application's black and vanish against the dark background. That is what
``set_title`` and ``style_dark`` are for; a title set with a bare
``setTitle`` is invisible.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg

PLOT_BG = "k"
AXIS_FG = "#b0b0b0"
TITLE_FG = "#e0e0e0"


def set_title(plot: Any, text: str) -> None:
    """Set a plot title in the panel's title colour.

    Every title in these panels goes through here. One set with a plain
    ``setTitle`` picks up the application's global black foreground and
    disappears against the dark background.
    """
    plot.setTitle(text, color=TITLE_FG)


def style_dark(plot: Any) -> None:
    """Dark background and legible axes for one PlotWidget or PlotItem."""
    if hasattr(plot, "setBackground"):
        plot.setBackground(PLOT_BG)
    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    for side in ("left", "bottom", "right", "top"):
        axis = item.getAxis(side)
        if axis is None:
            continue
        axis.setPen(pg.mkPen(AXIS_FG))
        axis.setTextPen(pg.mkPen(AXIS_FG))
