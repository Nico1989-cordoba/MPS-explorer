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

A verdict has no symbol to be told apart by -- it is a line of text --
so it carries a MARK instead: ``marked("warn", ...)`` puts a "!" in
front of it and ``marked("bad", ...)`` an "x". That matters for one pair
in particular: orange and vermillion, the colours of a warning and of a
failure, are only 18 apart in CIE Lab for a deuteranope, and the two are
read one under the other in the same list.

There is one place where eight colours are not enough: the axial
segments of an axon, which are ordered and can be five. Measured, no
more than three of the eight stay apart from each other under every
dichromacy, so SEGMENT_CYCLE only promises that a segment and the
segment NEXT to it are apart -- which is the comparison that panel is
for -- and a symbol and the segment's own number carry the rest.

``validate_plot_colours.py`` checks what is written here, with two
different measurements. CIE Lab distance, simulated under normal vision
and the three dichromacies, asks whether two MARKS can be told apart.
WCAG contrast asks whether TEXT can be read, which is the harder bar and
the reason ``verdict`` takes a background: on white the colours these
panels use reach 2.3:1 for a warning against the 4.5:1 a sentence needs.

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

from typing import Any, Dict, Tuple

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
    # --- the quality panels, which report verdicts rather than data ----
    # A check that passed, one that needs attention, one that failed.
    # Green and orange are 52 apart under every dichromacy, and green and
    # vermillion 37; orange and vermillion are only 18 apart for a
    # deuteranope, which is why a finding also carries a MARK.
    "good": OKABE_ITO["bluish_green"],
    "warn": OKABE_ITO["orange"],
    "bad": OKABE_ITO["vermillion"],
    # Explanation, and what could not be checked: present but not a
    # result. The same grey as noise, for the same reason.
    "dim": "#9a9a9a",
    # The second of two series of the same kind drawn in one plot -- sy
    # beside sx in the fitting-box check. Vermillion against the sky blue
    # of the first, which is the pair that survives every dichromacy. It
    # is the colour of channel_b for that same reason, and the two never
    # appear in one plot.
    "paired": OKABE_ITO["vermillion"],
}

# What a finding of each kind is prefixed with. Colour says it twice; the
# mark says it once in a way that survives a photocopy, a projector and a
# deuteranope reading orange against vermillion.
MARKS: Dict[str, str] = {
    "good": "ok  ",
    "warn": "!  ",
    "bad": "x  ",
    "dim": "-  ",
}


def marked(kind: str, text: str) -> str:
    """``text`` with the mark that says what kind of finding it is."""
    return MARKS[kind] + text


# The same four verdicts for text on the application's OWN white chrome:
# the tables of the MPS and rings windows, the dialogs. Same hues, each
# darkened until it reaches the contrast a reader needs to READ it,
# which is a higher bar than a mark in a plot has to clear. Measured on
# white, the panel's orange reaches 2.3:1 against the 4.5:1 that body
# text needs, and its green 3.4:1; all four of these are above 5.
_VERDICT_ON_LIGHT: Dict[str, str] = {
    "good": "#006b52",
    "warn": "#8a5f00",
    "bad": "#a04000",
    "dim": "#6a6a6a",
}


def verdict(kind: str, dark: bool = True) -> str:
    """The colour of a verdict of this kind, for the background it is
    read on. Darkening compresses the hues towards each other -- a
    warning and a failure come out 2 apart for a deuteranope -- which is
    why the mark from ``marked`` is what actually separates them."""
    return ROLES[kind] if dark else _VERDICT_ON_LIGHT[kind]


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
# Explanatory prose in a panel: quieter than a title, still readable.
# On a panel's dark grey, and on the application's own white chrome --
# tables, a list of warnings -- where the first would be unreadable.
TEXT_DIM = "#9a9a9a"
TEXT_DIM_LIGHT = "#6a6a6a"
# What a panel's own window is painted, behind the plots, and the
# furniture on it: the tab bar of the quality, DNA-PAINT and Two
# channels panels, which was the same four lines written out in each.
# None of it carries information -- nobody has to tell two tabs apart by
# colour -- but it is one look, so it is written once, and the validator
# still checks that the text on it can be read.
PANEL_BG = "#1a1a1a"
PANEL_BORDER = "#383838"
PANEL_TAB_BG = "#262626"
PANEL_TAB_BG_SELECTED = "#3a3a3a"


# --------------------------------------------------------------------------
# Segments of one axon, which are ordered and can outnumber the palette
# --------------------------------------------------------------------------
# An axon is cut into axial segments and the rings panel draws several at
# once. This is the one place where the palette runs out: measured, the
# largest set of the eight that stays mutually apart under all three
# dichromacies is THREE -- sky blue, yellow and vermillion -- and yellow
# is not a role. Five segments cannot be told apart by hue, and no
# ordering of them fixes that.
#
# What an ordering does fix is the comparison the panel exists for, which
# is a segment against the one next to it. In this order every
# consecutive pair stays 52 apart or more; only the wrap, a fifth segment
# back round to the first, is close, and those two are the ends of the
# axon. The order the panel used before had three consecutive pairs
# collapsed, including green against purple at 18.
#
# The rest is carried by SEGMENT_SYMBOLS, and by the segment's number,
# which the panel draws in the table, on every stacked track, on every
# subplot title and on the band in the z distribution.
SEGMENT_CYCLE: Tuple[str, ...] = (
    OKABE_ITO["blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
)
SEGMENT_SYMBOLS: Tuple[str, ...] = ("o", "s", "t", "d", "+")
# The same symbols as characters, for a table cell, which cannot draw a
# pyqtgraph marker. They are in the same order on purpose: the two drift
# apart the moment one is edited without the other, and the validator
# checks that there are as many of each.
SEGMENT_GLYPHS: Tuple[str, ...] = ("\u25cf", "\u25a0", "\u25bc",
                                   "\u25c6", "+")


def segment_colour(index: int) -> str:
    """The colour of segment ``index``, cycling when there are more
    segments than colours."""
    return SEGMENT_CYCLE[index % len(SEGMENT_CYCLE)]


def segment_symbol(index: int) -> str:
    """Its symbol, for the plot that draws every segment at once."""
    return SEGMENT_SYMBOLS[index % len(SEGMENT_SYMBOLS)]


def segment_glyph(index: int) -> str:
    """The same symbol as a character, for a table cell.

    The cell prints it beside the segment's number, in the window's own
    black: a segment's colour is 2.3 to 3.9 against white and a number
    is something a reader has to read, not merely distinguish. The
    colour stays where it is legible, which is the plots.
    """
    return SEGMENT_GLYPHS[index % len(SEGMENT_GLYPHS)]


TAB_STYLE = (
    f"QTabWidget::pane {{ border: 1px solid {PANEL_BORDER}; }} "
    f"QTabBar::tab {{ background: {PANEL_TAB_BG}; color: {AXIS_FG}; "
    f"padding: 6px 14px; }} "
    f"QTabBar::tab:selected {{ background: {PANEL_TAB_BG_SELECTED}; "
    f"color: {TITLE_FG}; }}"
)


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
