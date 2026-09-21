# -*- coding: utf-8 -*-
"""
What each control of the main window says when the mouse rests on it.

The texts live here rather than beside the widgets so they can be read
as a set: a beginner meets them in the order of the window, and a
sentence that contradicts the one next to it is easier to catch when
the two are on the same screen. ``apply_tooltips`` puts them on, and
the harness in the scratchpad checks both directions -- every name here
exists in the window, and every control of the window has a text.

They are in English, like the rest of the interface, because that is
the language the user asked for ("es el idioma más utilizado en
ciencia").

How they are written
--------------------
Each one says what the control DOES, and then the thing a beginner
cannot guess: what it is for, or what it does NOT do. "Cluster the
selection" is the first half; "the per-axon analysis runs on channel 1
only" is the half that saves an afternoon.

Nothing here describes a number the program does not use, and nothing
promises a behaviour that is not in the code: a tooltip is
documentation that ships inside the program, and a wrong one is worse
than none.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Dict, List

from PyQt5 import QtCore

# --------------------------------------------------------------------------
# The main window, in the order the window itself reads
# --------------------------------------------------------------------------
MAIN_WINDOW: Dict[str, str] = {
    # --- loading ------------------------------------------------------
    "comboBox_fileformat":
        "How to read the channel-1 file.\n\n"
        "Picasso hdf5 is the usual one: its pixel size is read from the "
        "YAML file that sits beside it. ThunderSTORM csv and custom csv "
        "are read by their column names, and then the pixel size has to "
        "be entered by hand.",
    "pushButton_browsefile":
        "Choose the localization file for channel 1.\n\n"
        "This is the channel everything is measured on. A file can also "
        "be dropped onto the window instead.",
    "comboBox_fileformat_2":
        "How to read the channel-2 file. The same three formats as "
        "channel 1.",
    "pushButton_browsefile_2":
        "Choose the localization file for channel 2, a second protein "
        "imaged in the same field.\n\n"
        "Optional. The per-axon analysis runs on channel 1; channel 2 is "
        "for the Two channels panel.",
    # --- the selection -------------------------------------------------
    "pushButton_scatter":
        "Draw the loaded channel-1 file and put a selection shape on it.\n\n"
        "Nothing is analysed yet. This is where you pick which axon to "
        "work on: move and resize the shape until it holds one axon.",
    "radioButton_circROI":
        "Select with a circle. Drag the middle to move it, drag the "
        "handle to resize it.\n\n"
        "The usual choice for a cross-sectioned axon.",
    "radioButton_squareROI":
        "Select with a rectangle. Drag the middle to move it, drag the "
        "handle to resize it.",
    "radioButton_polygonROI":
        "Select by clicking one vertex at a time; ENTER closes the "
        "shape.\n\n"
        "For an axon that is not round, or when a neighbouring one has "
        "to be left out of a circle that would otherwise contain it.",
    "lineEdit_zmin":
        "Lower edge of the axial slab, in nanometres.\n\n"
        "Filled in from the main peak of the z distribution when a file "
        "is loaded, and left alone once you edit it. Localizations below "
        "it are not in the selection. Leave both empty to keep the whole "
        "depth.",
    "lineEdit_zmax":
        "Upper edge of the axial slab, in nanometres.\n\n"
        "Filled in from the main peak of the z distribution when a file "
        "is loaded, and left alone once you edit it. Localizations above "
        "it are not in the selection. Leave both empty to keep the whole "
        "depth.",
    "pushButton_zrange":
        "Apply the z range above to the selection and redraw it.\n\n"
        "The selection shape and the z range are one cut: this is what "
        "puts both into effect.",
    # --- clustering ----------------------------------------------------
    "comboBox_algorithm":
        "Which algorithm groups the localizations into clusters.\n\n"
        "DBSCAN uses Epsilon and Min Pts. HDBSCAN uses Min Cluster Size "
        "instead and finds clusters of differing density. Auto picks "
        "DBSCAN below 100,000 localizations and HDBSCAN above, which is "
        "a choice about speed, not about the data.",
    "lineEdit_eps":
        "DBSCAN's epsilon, in nanometres: how far apart two "
        "localizations can be and still belong to the same cluster.\n\n"
        "Too small breaks one cluster into several; too large merges "
        "neighbouring ones into a blob. Not used by HDBSCAN.",
    "lineEdit_minsamples":
        "DBSCAN's Min Pts: how many localizations have to sit within "
        "epsilon of a point for it to start a cluster.\n\n"
        "Raising it leaves more localizations as noise, which the plots "
        "draw as grey crosses. Not used by HDBSCAN.",
    "lineEdit_minclustersize":
        "HDBSCAN's smallest number of localizations a group must have to "
        "count as a cluster.\n\n"
        "Used only when the algorithm above is HDBSCAN.",
    "pushButton_clusterch1":
        "Cluster the channel-1 selection, then run the whole per-axon "
        "analysis and open the MPS analysis window.\n\n"
        "Everything downstream -- the perimeter, the occupancy, the "
        "randomization, the export -- comes from this one press.",
    "lineEdit_eps_2":
        "DBSCAN's epsilon for channel 2, in nanometres.\n\n"
        "'auto' means nobody has chosen one: the cluster button will "
        "estimate it from channel 2's own points, and the Two channels "
        "panel will ask before it compares. It is never filled in from "
        "channel 1's.",
    "lineEdit_minsamples_2":
        "DBSCAN's Min Pts for channel 2.\n\n"
        "This one counts LOCALIZATIONS, and the two channels are rarely "
        "sampled alike: across five real adducin files channel 2 carried "
        "1 to 138 times more of them per 20 nm cell than channel 1. That "
        "is why the Two channels panel will not borrow channel 1's.",
    "pushButton_clusterch2":
        "Cluster the channel-2 selection, to look at.\n\n"
        "The Two channels panel clusters channel 2 again from whatever "
        "is in the two boxes above, so these clusters are not the ones "
        "it compares. With 'auto' in the boxes this button is how you "
        "find a number to put in them. The per-axon analysis stays on "
        "channel 1.",
    # --- the distance histogram ----------------------------------------
    "lineEdit_Nneighbor":
        "Which neighbour the distance histogram shows: 1 is the nearest "
        "cluster centre, 2 the second nearest, and so on.\n\n"
        "The per-axon analysis always reports the first; this is how to "
        "look at the others.",
    "pushButton_Distances":
        "Measure the distances between cluster centres and draw the "
        "histogram below.\n\n"
        "Centre to centre, in the plane.",
    "lineEdit_bin":
        "How many bars the distance histogram is drawn with.",
    "lineEdit_latmin":
        "Left edge of the distance histogram, in nanometres.\n\n"
        "Distances outside the range are left out of the DRAWING only; "
        "what is exported is unchanged.",
    "lineEdit_latmax":
        "Right edge of the distance histogram, in nanometres.\n\n"
        "Distances outside the range are left out of the DRAWING only; "
        "what is exported is unchanged.",
}


def apply_tooltips(window: Any, texts: Dict[str, str]) -> List[str]:
    """Put ``texts`` on the controls of ``window`` named by its keys.

    ``window`` can be the generated ``Ui_MainWindow``, which holds its
    widgets as plain attributes, or a real widget, which holds them as
    children -- a panel built in code has no Ui object, and the main
    window's is not a QObject and has no ``findChild``.

    Returns the names that were not found, so a control renamed in Qt
    Designer shows up as a list rather than as a tooltip that quietly
    stopped appearing. The caller decides what to do with it: the
    program logs it and carries on, the harness fails on it.
    """
    missing = []
    for name, text in texts.items():
        widget = getattr(window, name, None)
        if widget is None and hasattr(window, "findChild"):
            widget = window.findChild(QtCore.QObject, name)
        if widget is None or not hasattr(widget, "setToolTip"):
            missing.append(name)
            continue
        widget.setToolTip(text)
        # An editable combo box draws its current item in a QLineEdit of
        # its own, and the mouse rests on THAT, not on the combo: without
        # this the text appears over the arrow and nowhere else.
        line_edit = getattr(widget, "lineEdit", None)
        if callable(line_edit):
            inner = line_edit()
            if inner is not None:
                inner.setToolTip(text)
    return missing
