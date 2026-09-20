# -*- coding: utf-8 -*-
"""
One palette for every plot in the program, and the styling around it.

Colours
-------
The eight of Okabe and Ito (2008), which are chosen so that the common
forms of colour blindness -- deuteranopia and protanopia, together about
8 % of men -- can still tell them apart. They are assigned here by ROLE,
once: a role has the same colour in every panel, and two roles a reader
has to tell apart in one plot never share one unless a symbol does the
telling. Yellow is left out of the roles on purpose: it
is the brightest of the eight on a black background and the weakest on a
white one, and these plots are read on both.

Colour is never the only difference. What is excluded from the analysis
is drawn with its own symbol as well (noise is a dot, a curated-away
cluster an x), because a reader who cannot separate two hues can always
separate a dot from a cross, and because a figure printed in grey keeps
its meaning.

``validate_plot_colours.py`` checks what is written here: every pair of
roles that appears in one plot is simulated under normal vision and under
the three dichromacies and must stay apart in CIE Lab, and every role
must stand off both backgrounds.

Dark and light
--------------
``MPS_explorer`` sets pyqtgraph's GLOBAL background to white and its
foreground to black, so any plot built from inside the running
application comes out light -- even code that renders dark when run
standalone. These panels are styled dark per widget instead of changing
that global, which would recolour every plot in the application.

The consequence is the thing to remember when editing a panel: nothing
may rely on the global foreground any more. Axis pens, tick text and
every title have to be set explicitly, or they are drawn in the
application's black and vanish against the dark background. That is what
``set_title`` and ``style_dark`` are for; a title set with a bare
``setTitle`` is invisible.

``style_light`` is the same styling for a white background, which is what
an exported figure uses: a journal expects black on white, and the one
role whose colour depends on the background -- the neutral line and
outline -- is read through ``neutral()`` rather than written into each
plot.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Dict

import pyqtgraph as pg

# --------------------------------------------------------------------------
# The eight, by their names
# --------------------------------------------------------------------------
OKABE_ITO: Dict[str, str] = {
    "black": "#000000",
    "orange": "#e69f00",
    "sky_blue": "#56b4e9",
    "bluish_green": "#009e73",
    "yellow": "#f0e442",
    "blue": "#0072b2",
    "vermillion": "#d55e00",
    "reddish_purple": "#cc79a7",
}

# --------------------------------------------------------------------------
# What each colour means. One role, one colour, everywhere.
# --------------------------------------------------------------------------
ROLES: Dict[str, str] = {
    # --- localizations -----------------------------------------------
    # In a cluster the analysis kept. The data itself.
    "locs": OKABE_ITO["sky_blue"],
    # In no cluster at all: DBSCAN's noise. Context, drawn faint.
    "noise": "#9a9a9a",
    # In a cluster the automatic curation removed (edge, DBCV). The same
    # grey as noise, on purpose: to a reader both mean "not in the
    # analysis", and the x that draws it is what says which of the two it
    # is. A hue spent here would be one fewer for the marks that ARE in
    # the analysis, and there are only eight.
    "curated": "#9a9a9a",
    # In a cluster the axoplasm panel discarded: inside the axon by both
    # widefield images. Vermillion against the sky blue of the rest,
    # which is the pair that survives every dichromacy; what it comes
    # close to is the orange of the occupied stretches, and those are a
    # line along the ring while these are diamonds inside it.
    "discarded": OKABE_ITO["vermillion"],
    # --- what is built from them --------------------------------------
    # A cluster's centre of mass.
    "centroid": OKABE_ITO["bluish_green"],
    # The stretches of the perimeter within reach of a cluster: the
    # occupancy, which is the paper's central claim.
    "occupied": OKABE_ITO["orange"],
    # The centre of the contour (its area centroid).
    "centre": OKABE_ITO["blue"],
    # --- second channel, second image ---------------------------------
    # The partner protein, against channel 1's localizations. Vermillion
    # rather than another blue: blue against orange-red is the pair that
    # survives every dichromacy, and the two channels are the comparison
    # the whole panel is for.
    "channel_b": OKABE_ITO["vermillion"],
    # The randomized control, against the observed data.
    "randomized": "#9a9a9a",
    "observed": OKABE_ITO["blue"],
    # What the paper reports, drawn dashed for comparison.
    "paper": OKABE_ITO["vermillion"],
    # A median or another summary line over a histogram.
    "summary": OKABE_ITO["orange"],
    # The axial slab, as a band behind the axial histogram. A highlight,
    # not a category: neutral, so it does not read as another thing.
    "slab": "#7f7f7f",
    # The fitted axial components.
    "fit": OKABE_ITO["orange"],
    # --- the axoplasm panel, which draws on a grey widefield image -----
    # The edge of the axon in the tubulin image, and the clusters only
    # that image puts inside: one colour, because they say one thing.
    "image_tubulin": OKABE_ITO["bluish_green"],
    # The same for the dark inside of the spectrin ring.
    "image_spectrin": OKABE_ITO["reddish_purple"],
    # The contour through every cluster, dashed. Orange rather than the
    # neutral the MPS window uses: over the bright inside of the axon a
    # white line disappears.
    "contour_all": OKABE_ITO["orange"],
    # The contour without the discarded clusters.
    "contour_kept": OKABE_ITO["blue"],
}

# Backgrounds, and the colour of everything structural on them: the
# contour of the axon, the outline of a marker, an axis. It is the one
# thing that has to change with the background, so it is asked for.
DARK_BG = "k"
LIGHT_BG = "w"
_NEUTRAL_ON_DARK = "#e8e8e8"
_NEUTRAL_ON_LIGHT = "#1a1a1a"

PLOT_BG = DARK_BG
AXIS_FG = "#b0b0b0"
TITLE_FG = "#e0e0e0"
AXIS_FG_LIGHT = "#333333"
TITLE_FG_LIGHT = "#000000"


def neutral(dark: bool = True) -> str:
    """The contour, outline and marker-edge colour for this background."""
    return _NEUTRAL_ON_DARK if dark else _NEUTRAL_ON_LIGHT


def role(name: str) -> str:
    """The colour of one role. Raises for a role that does not exist, so
    a typo is not silently drawn in pyqtgraph's default white."""
    return ROLES[name]


def rgba(name: str, alpha: int) -> tuple:
    """One role as (r, g, b, alpha), for a fill that has to let what is
    under it through."""
    value = ROLES[name].lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16),
            int(alpha))


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

def set_title(plot: Any, text: str, dark: bool = True) -> None:
    """Set a plot title in the panel's title colour.

    Every title in these panels goes through here. One set with a plain
    ``setTitle`` picks up the application's global black foreground and
    disappears against the dark background.
    """
    plot.setTitle(text, color=TITLE_FG if dark else TITLE_FG_LIGHT)


def _style(plot: Any, background: str, axis_fg: str) -> None:
    if hasattr(plot, "setBackground"):
        plot.setBackground(background)
    item = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
    for side in ("left", "bottom", "right", "top"):
        axis = item.getAxis(side)
        if axis is None:
            continue
        axis.setPen(pg.mkPen(axis_fg))
        axis.setTextPen(pg.mkPen(axis_fg))


def style_dark(plot: Any) -> None:
    """Dark background and legible axes for one PlotWidget or PlotItem."""
    _style(plot, PLOT_BG, AXIS_FG)


def style_light(plot: Any) -> None:
    """The same, on white: what a figure for a journal is drawn on."""
    _style(plot, LIGHT_BG, AXIS_FG_LIGHT)
