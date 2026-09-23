# -*- coding: utf-8 -*-
"""
The axoplasm panel: is each betaII-spectrin localization at the membrane
or inside the axon, according to a widefield betaIII-tubulin image?

Top to bottom:
  1. the images. The tubulin one, and optionally a widefield image of the
     super-resolved protein together with the localizations of the whole
     movie, to measure the shift between the acquisitions. Section 5 also
     needs that spectrin image: it finds the ring's dark inside in it;
  2. the alignment, measured and adjustable by hand;
  3. the mask, with Otsu's threshold on the axon's region, adjustable;
  4. the margin the user sets, where the localizations are -- inside only
     through a cluster both images put inside -- and the export;
  5. the MPS analysis' clusters that are not anchored to the membrane --
     the ones both widefield images put inside the axon -- and the contour
     rebuilt without them. The main window is told, and repeats the whole
     MPS analysis without them beside the original.

Dual view (2026-09-22). Either image can instead be a localization file
of that protein acquired AT THE SAME TIME as the movie, on the other half
of one camera -- the 2023 sciatic-nerve stainings of betaII-spectrin with
betaIII-tubulin are like that. Its localizations are counted on the
movie's own camera grid, so it is placed by construction: no widefield
image, no shift, and section 5 runs without one; the table records
"same acquisition". For the spectrin, that can be the movie's own file.
Measured on five picked axons of 230911 ROI 1, the tubulin mask drawn
from localizations and the one drawn from the widefield image of the
same axons overlap with an IoU of 0.53 to 0.70, and the widefield one is
larger -- its edge a median 246 nm further out in effective radius,
86 to 356 nm -- so the same margin discards less against a mask drawn
from localizations. Which to use is the user's call; the table says
which it was.

The calculations are in ``tools.mps_axoplasm``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from scipy import ndimage

from tools import mps_axoplasm as ax
from tools import mps_file_drop
from tools.mps_io import load_localizations
from tools.mps_plot_style import (
    AXIS_FG, PANEL_BG, TEXT_DIM, TITLE_FG, marked, neutral, rgba, role,
    set_title, style_dark)
from tools.cluster_quality import describe_roi

# Every colour is a role from tools.mps_plot_style, so a cluster the
# discard leaves out is the same vermillion here and in the MPS analysis
# window. The old palette had the kept contour in green and the discarded
# clusters in red, which is the one pair a deuteranope cannot separate,
# and cyan against yellow, which is the one a tritanope cannot.
#
# This panel draws on a grey widefield image rather than on black, so the
# two contours are the saturated blue and orange rather than the neutral
# the MPS window uses: a white line is lost over a bright patch of image
# and a dark one over a dim patch.
# A finding is text, not a mark on the image: the verdict roles, the
# same three as in every other panel, each with the mark for its kind.
_TEXT_OK = role("good")
_WARN = role("warn")
_DIM = TEXT_DIM
# Localizations of the clusters the discard leaves out.
_INTERIOR = role("discarded")
# Localizations of the clusters that stay: the data.
_MEMBRANE = role("locs")
_OUTLINE = rgba("image_tubulin", 255)      # the same role, as an image
_SPECTRIN_OUTLINE = rgba("image_spectrin", 255)
_DISCARDED = role("discarded")
# Clusters only one image puts inside, in the colour of that image's own
# edge: green for the tubulin mask, purple for the spectrin interior.
_TUBULIN_ONLY = role("image_tubulin")
_SPECTRIN_ONLY = role("image_spectrin")
# The contour through every cluster: orange, dashed.
_ALL_CONTOUR = role("contour_all")
# The contour without the discarded clusters.
_KEPT_CONTOUR = role("contour_kept")
# The centre of that contour: white with a dark rim, so it is legible
# wherever on the image it falls. Its shape is what names it.
_CENTRE = neutral(dark=True)
# Localizations in no cluster, or not placed yet: light, and small.
_NO_CLUSTER = neutral(dark=True)
# The ring around a cluster neither image decided about. Structural, like
# the centre and the unplaced localizations, and told apart from them by
# being a ring of 9 pixels rather than a cross or a 2 pixel dot.
_NO_DECISION = neutral(dark=True)
def _cased_edge(edge: "np.ndarray", colour: tuple) -> "np.ndarray":
    """
    An edge image with a dark casing, so it is legible wherever on the
    widefield it falls.

    A line drawn in one colour over a photograph is only as visible as
    that colour is different from whatever grey is under it, and the two
    edges here are drawn in hues whose lightness is close to a mid grey.
    A one-pixel dark casing makes both of them stand off any background,
    the way a map draws a road.
    """
    rim = ndimage.binary_dilation(edge) & ~edge
    rgba = np.zeros(edge.shape + (4,), dtype=np.ubyte)
    rgba[rim] = (0, 0, 0, 200)
    rgba[edge] = colour
    return rgba


# Contours rebuilt with every 2-opt start, kept per set of cluster centres.
CONTOUR_CACHE_SIZE = 32

# How long the panel waits, after the last step of a drag, before it
# recomputes the mask and the discard. Measured on axon 7 (34 clusters,
# 10,034 localizations): one control event costs 110-135 ms of panel work
# even when nothing is discarded, and 2.6-2.8 s when the discarded set
# changes, of which 2.46 s is the 1000-iteration randomization. A slider
# emits about 60 of those a second, so without this the window is not
# slow -- it is frozen.
RECOMPUTE_DELAY_MS = 250

# Points drawn per class; the classification itself uses all of them.
MAX_DRAWN = 20000
# Histogram of the distances to the edge.
HIST_RANGE_NM = 1500.0
HIST_BIN_NM = 25.0


def _swatch(colour: str, shape: str, size: int = 14) -> QtGui.QIcon:
    """
    The mark the image draws a layer with, for its box in the legend:
    "disc" (filled, dark rim), "ring", "dot", "line", "dash" or "plus".
    """
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    ink = QtGui.QColor(colour)
    if shape == "disc":
        painter.setPen(QtGui.QPen(QtGui.QColor("#000000"), 1))
        painter.setBrush(QtGui.QBrush(ink))
        painter.drawEllipse(2, 2, size - 4, size - 4)
    elif shape == "ring":
        painter.setPen(QtGui.QPen(ink, 2))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawEllipse(2, 2, size - 4, size - 4)
    elif shape == "dot":
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(ink))
        painter.drawEllipse(size // 2 - 2, size // 2 - 2, 5, 5)
    elif shape == "plus":
        painter.setPen(QtGui.QPen(ink, 3))
        painter.drawLine(size // 2, 1, size // 2, size - 1)
        painter.drawLine(1, size // 2, size - 1, size // 2)
    else:
        pen = QtGui.QPen(ink, 2)
        if shape == "dash":
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(0, size // 2, size, size // 2)
    painter.end()
    return QtGui.QIcon(pixmap)


def _label(text: str, colour: str = TITLE_FG,
           bold: bool = False) -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setStyleSheet(
        f"color: {colour}; font-weight: {'bold' if bold else 'normal'};")
    widget.setWordWrap(True)
    widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    return widget


def _spin(low: float, high: float, step: float, decimals: int = 1,
          suffix: str = "") -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    # Only a finished edit recomputes, not every keystroke.
    spin.setKeyboardTracking(False)
    return spin


@dataclass
class AxoplasmInputs:
    """What the panel needs from the main window."""

    loc: Any                    # the channel-1 selection (Localizations)
    movie: Any                  # the whole channel-1 file (Localizations)
    roi: Optional[Any] = None   # tools.cluster_quality ROI shape
    # Centroids (nm) of the clusters the MPS analysis kept for this
    # selection, or None when it has not been analysed.
    clusters: Callable[[], Optional[np.ndarray]] = field(
        default=lambda: None)
    # Identifies the main window's selection, to tell whether a panel
    # reopened later still shows it.
    selection_key: Any = None
    # The contour the MPS analysis built from those clusters
    # (tools.mps_geometry.PerimeterResult), or None.
    contour: Callable[[], Optional[Any]] = field(default=lambda: None)
    # Told the clusters not anchored to the membrane each time they are
    # found again -- the AnchoredClusters and the centres (nm) they were
    # found for -- or (None, None) when there are none to tell.
    anchored_changed: Callable[[Optional[ax.AnchoredClusters],
                                Optional[np.ndarray]], None] = field(
        default=lambda found, centroids: None)
    # For localizations of the selection (x, y, z in nm), the index of the
    # kept cluster each belongs to, in the order of ``clusters`` (-1: none),
    # or None when the selection has not been analysed.
    cluster_of: Callable[[np.ndarray, np.ndarray, np.ndarray],
                         Optional[np.ndarray]] = field(
        default=lambda x, y, z: None)
    # The axial cut the main window applied to this selection, and where
    # it came from ("typed", "axial peak" or "none"). Every count in this
    # panel is a count of the localizations that survived it: clearing the
    # Z fields on axon 7 took the selection from 10,034 to 23,743 and
    # fraction_inside from 0.0141 to 0.0060, with nothing in the table to
    # say why.
    z_range: Optional[Tuple[float, float]] = None
    z_range_source: str = "none"
    # The DBSCAN label of each kept cluster, in the order of ``clusters``,
    # or None when the selection has not been analysed. Every table names
    # a cluster by that label, the main window's own export included.
    cluster_labels: Callable[[], Optional[np.ndarray]] = field(
        default=lambda: None)


@dataclass
class _Measurement:
    registration: ax.ImageRegistration
    path: str           # the localizations it was measured with
    reference: str      # the widefield image it was measured against
    notes: List[str]


class _Relay(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object, int)


class AxoplasmWindow(QtWidgets.QMainWindow):
    """Membrane or interior, from tubulin and spectrin images -- widefield,
    or drawn from localizations acquired with the movie."""

    def __init__(self, inputs: AxoplasmInputs,
                 parent: Optional[QtWidgets.QWidget] = None,
                 export_callback: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(parent)
        self.inputs = inputs
        # Writes this axon's tables. The main window's single export: it
        # sees this panel, the analysis and the discard at once, which is
        # what keeps the rows of one axon describing one state.
        self.export_callback = export_callback
        self.tubulin: Optional[ax.WidefieldImage] = None
        self.reference: Optional[ax.WidefieldImage] = None
        self.tubulin_offset: Tuple[float, float] = (0.0, 0.0)
        self.reference_offset: Tuple[float, float] = (0.0, 0.0)
        self.tubulin_notes: List[str] = []
        self.reference_notes: List[str] = []
        self.region_notes: List[str] = []
        self.measured: Optional[_Measurement] = None
        self.axoplasm: Optional[ax.AxoplasmMask] = None
        self.result: Optional[ax.Classification] = None
        # Where each localization is through its cluster
        # (ax.localization_labels), once both images placed the clusters.
        self.located: Optional[np.ndarray] = None
        self.loc_cluster: Optional[np.ndarray] = None
        self.spectrin_interior: Optional[ax.AxoplasmMask] = None
        self.anchored: Optional[ax.AnchoredClusters] = None
        # The cluster centres (nm) the anchored result was computed for.
        self.anchored_centroids: Optional[np.ndarray] = None
        self.anchored_notes: List[str] = []
        self._contour_cache: Dict[bytes, Any] = {}
        self.selection_note: Optional[str] = None
        self._manual_threshold: Optional[float] = None
        self._threshold_range: Tuple[float, float] = (0.0, 1.0)
        self._updating = False
        self._thread: Optional[threading.Thread] = None
        self._progress: Optional[QtWidgets.QProgressDialog] = None
        # Bumped when the panel closes; a measurement started before is
        # discarded when it ends.
        self._generation = 0
        self._relay = _Relay()
        self._relay.finished.connect(self._measured)

        # What a drag has asked for and not yet been given. The timer is
        # parented to the window so it dies with it on the paths that
        # delete the panel rather than keep it.
        self._pending: Optional[str] = None
        self._recompute = QtCore.QTimer(self)
        self._recompute.setSingleShot(True)
        self._recompute.setInterval(RECOMPUTE_DELAY_MS)
        self._recompute.timeout.connect(self._run_pending)

        self.setWindowTitle(
            "Axoplasm - " + os.path.basename(str(inputs.movie.path)))
        self.resize(1320, 900)
        self.setStyleSheet(
            f"QMainWindow {{ background: {PANEL_BG}; }} "
            f"QLabel, QCheckBox {{ color: {TITLE_FG}; }} "
            f"QGroupBox {{ color: {TITLE_FG}; }}")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        self.label_header = _label("")
        root.addWidget(self.label_header)
        body = QtWidgets.QHBoxLayout()
        root.addLayout(body, stretch=1)

        # The controls, the results and the warnings scroll together: with
        # five sections they no longer fit a 900-pixel window.
        left = QtWidgets.QVBoxLayout()
        left_box = QtWidgets.QWidget()
        left_box.setObjectName("controls")
        left_box.setStyleSheet(f"#controls {{ background: {PANEL_BG}; }}")
        left_box.setLayout(left)
        left_box.setFixedWidth(450)
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet(
            f"QScrollArea {{ background: {PANEL_BG}; }}")
        left_scroll.setWidget(left_box)
        left_scroll.setFixedWidth(
            450 + left_scroll.verticalScrollBar().sizeHint().width() + 4)
        body.addWidget(left_scroll)
        left.addWidget(self._build_images())
        left.addWidget(self._build_alignment())
        left.addWidget(self._build_mask())
        left.addWidget(self._build_classification())
        left.addWidget(self._build_anchored())
        self.findings = QtWidgets.QVBoxLayout()
        holder = QtWidgets.QWidget()
        holder.setObjectName("findings")
        holder.setStyleSheet(f"#findings {{ background: {PANEL_BG}; }}")
        holder.setLayout(self.findings)
        left.addWidget(holder)
        left.addStretch(1)

        right = QtWidgets.QVBoxLayout()
        body.addLayout(right, stretch=1)
        self.check_reference = QtWidgets.QCheckBox(
            "Show the betaII-spectrin widefield image instead")
        self.check_reference.setToolTip(
            "The spectrin rings of the localizations should lie on the "
            "widefield membranes when the alignment is right.")
        self.check_reference.toggled.connect(lambda _on: self._draw())
        right.addWidget(self.check_reference)
        self.plot_image = pg.PlotWidget()
        style_dark(self.plot_image)
        self.plot_image.setAspectLocked(True)
        self.plot_image.setLabel("bottom", "x [nm]", color=AXIS_FG)
        self.plot_image.setLabel("left", "y [nm]", color=AXIS_FG)
        self.image_item = pg.ImageItem()
        self.outline_item = pg.ImageItem()
        self.spectrin_outline_item = pg.ImageItem()
        self.membrane_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(_MEMBRANE), size=2)
        self.interior_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(_INTERIOR), size=2)
        self.contour_all_item = pg.PlotDataItem(
            pen=pg.mkPen(_ALL_CONTOUR, width=2,
                         style=QtCore.Qt.PenStyle.DashLine))
        self.contour_item = pg.PlotDataItem(
            pen=pg.mkPen(_KEPT_CONTOUR, width=2))
        self.centre_item = pg.ScatterPlotItem(
            pen=pg.mkPen("k", width=2), brush=pg.mkBrush(_CENTRE), size=18,
            symbol="+")
        legend = self._build_layers()
        for item in (self.image_item, self.outline_item,
                     self.spectrin_outline_item, self.free_item,
                     self.membrane_item, self.interior_item,
                     self.contour_all_item,
                     self.contour_item, self.cluster_item,
                     self.tubulin_only_item, self.spectrin_only_item,
                     self.discarded_item, self.centre_item):
            self.plot_image.addItem(item)
        # The image is square, so the legend takes the room beside it.
        image_row = QtWidgets.QHBoxLayout()
        image_row.addWidget(self.plot_image, stretch=1)
        image_row.addWidget(legend)
        right.addLayout(image_row, stretch=3)
        self.plot_hist = pg.PlotWidget()
        style_dark(self.plot_hist)
        set_title(self.plot_hist,
                  "Distance to the tubulin mask's edge (positive inside)")
        self.plot_hist.setLabel("bottom", "distance [nm]", color=AXIS_FG)
        self.plot_hist.setLabel("left", "localizations", color=AXIS_FG)
        right.addWidget(self.plot_hist, stretch=1)

        self._show_header()
        self._refresh()

    # ------------------------------------------------------------ layout
    def _build_layers(self) -> QtWidgets.QWidget:
        """
        The legend beside the image: one box per thing drawn on it, to show
        or hide each one, and two buttons to show or hide them all.

        The clusters come in the four groups the two images put them in.
        The groups do not overlap: together they are every cluster.
        """
        self.cluster_item = pg.ScatterPlotItem(
            pen=pg.mkPen(_NO_DECISION), brush=None, size=9)
        # Circles for one image, squares for the other: on a grey
        # photograph the two hues are close for a deuteranope, and the
        # shape is what a reader can always separate.
        self.tubulin_only_item = pg.ScatterPlotItem(
            pen=pg.mkPen(_TUBULIN_ONLY, width=2), brush=None, size=10)
        self.spectrin_only_item = pg.ScatterPlotItem(
            pen=pg.mkPen(_SPECTRIN_ONLY, width=2), brush=None, size=10,
            symbol="s")
        self.discarded_item = pg.ScatterPlotItem(
            pen=pg.mkPen("k"), brush=pg.mkBrush(_DISCARDED), size=10)
        clusters = (
            ("both", self.discarded_item, _DISCARDED, "disc",
             "Clusters both widefield images put more than the margin inside\n"
             "the axon: inside the betaIII-tubulin mask AND inside the dark\n"
             "area the betaII-spectrin ring encloses. These are the ones the\n"
             "discard leaves out of the MPS analysis."),
            ("tubulin", self.tubulin_only_item, _TUBULIN_ONLY, "ring",
             "Clusters only the betaIII-tubulin mask puts more than the margin\n"
             "inside the axon. The spectrin image does not agree, so they are\n"
             "kept. Green, like the edge of the tubulin mask."),
            ("spectrin", self.spectrin_only_item, _SPECTRIN_ONLY, "ring",
             "Clusters only the dark inside of the betaII-spectrin ring puts\n"
             "more than the margin inside the axon. The tubulin mask does not\n"
             "agree, so they are kept. Purple, like the edge of the\n"
             "spectrin interior."),
            ("neither", self.cluster_item, _NO_DECISION, "ring",
             "Clusters neither image puts more than the margin inside the\n"
             "axon: on the membrane, or off the region an image was analysed\n"
             "in. They are kept."),
        )
        lines = (
            ("tubulin_edge", self.outline_item, _TUBULIN_ONLY, "line",
             "Edge of the axoplasm mask found in the betaIII-tubulin image\n"
             "(section 3)."),
            ("spectrin_edge", self.spectrin_outline_item, _SPECTRIN_ONLY,
             "line",
             "Edge of the dark area the betaII-spectrin ring encloses in its\n"
             "widefield image (section 5)."),
            ("contour_all", self.contour_all_item, _ALL_CONTOUR, "dash",
             "The contour through every cluster, as the MPS analysis\n"
             "connects them."),
            ("contour_kept", self.contour_item, _KEPT_CONTOUR, "line",
             "The contour without the discarded clusters: the one the MPS\n"
             "analysis window uses for its 'Discard applied' column."),
        )
        self.free_item = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush(_NO_CLUSTER), size=2)
        points = (
            ("locs_inside", self.interior_item, _INTERIOR, "dot",
             "Localizations inside the axon: those of the clusters both\n"
             "widefield images put inside (the discarded ones). The tubulin\n"
             "mask alone never makes a localization count as inside."),
            ("locs_membrane", self.membrane_item, _MEMBRANE, "dot",
             "Localizations at the membrane: those of the kept clusters."),
            ("locs_free", self.free_item, _NO_CLUSTER, "dot",
             "Localizations in no cluster: left out by DBSCAN or in a\n"
             "cluster the automatic curation removed. Before both images\n"
             "have placed the clusters, every localization is drawn here."),
            ("centre", self.centre_item, _CENTRE, "plus",
             "The centre of the contour without the discarded clusters:\n"
             "its area centroid, as the MPS analysis reports it."),
        )
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(4, 0, 0, 0)
        self.cluster_toggles: Dict[str, QtWidgets.QCheckBox] = {}
        self.layer_toggles: Dict[str, QtWidgets.QCheckBox] = {}
        for title, rows, into in (("Clusters", clusters, self.cluster_toggles),
                                  ("Lines", lines, self.layer_toggles),
                                  ("Localizations and centre", points,
                                   self.layer_toggles)):
            lay.addWidget(_label(title, TITLE_FG, bold=True))
            for key, item, colour, shape, tip in rows:
                box = QtWidgets.QCheckBox()
                box.setChecked(True)
                box.setIcon(_swatch(colour, shape))
                box.setToolTip(tip)
                box.toggled.connect(lambda on, drawn=item: drawn.setVisible(on))
                into[key] = box
                lay.addWidget(box)
            lay.addSpacing(6)
        buttons = QtWidgets.QHBoxLayout()
        show_all = QtWidgets.QPushButton("Show all")
        show_all.setToolTip("Draw everything the legend lists.")
        show_all.clicked.connect(lambda: self._show_layers(True))
        hide_all = QtWidgets.QPushButton("Hide all")
        hide_all.setToolTip(
            "Leave only the image, then tick what you want to see.")
        hide_all.clicked.connect(lambda: self._show_layers(False))
        buttons.addWidget(show_all)
        buttons.addWidget(hide_all)
        lay.addLayout(buttons)
        lay.addStretch(1)
        self.layer_toggles["centre"].setText("Centre of the contour")
        self.layer_toggles["tubulin_edge"].setText("Tubulin mask edge")
        self.layer_toggles["spectrin_edge"].setText("Spectrin interior edge")
        self.layer_toggles["contour_all"].setText("Contour, all clusters")
        self.layer_toggles["contour_kept"].setText(
            "Contour without the discarded")
        self._sync_cluster_toggles(None)
        self._sync_localization_toggles(None)
        return panel

    def _show_layers(self, on: bool) -> None:
        for box in (*self.cluster_toggles.values(),
                    *self.layer_toggles.values()):
            box.setChecked(on)

    def _sync_localization_toggles(self, located: Optional[np.ndarray]
                                   ) -> None:
        """The localization boxes' texts, with how many each holds."""
        boxes = self.layer_toggles
        if located is None:
            # Not placed yet: every localization is drawn plain.
            for key in ("locs_inside", "locs_membrane"):
                boxes[key].setEnabled(False)
            boxes["locs_inside"].setText("Localizations inside")
            boxes["locs_membrane"].setText("Localizations at the membrane")
            boxes["locs_free"].setText("Localizations (not placed yet)")
            return
        for key in ("locs_inside", "locs_membrane"):
            boxes[key].setEnabled(True)
        for key, label, text in (
                ("locs_inside", ax.LOC_INSIDE, "Localizations inside"),
                ("locs_membrane", ax.LOC_MEMBRANE,
                 "Localizations at the membrane"),
                ("locs_free", ax.LOC_NO_CLUSTER,
                 "Localizations in no cluster")):
            boxes[key].setText(f"{text} ({int(np.sum(located == label)):,})")

    def _sync_cluster_toggles(self, found: Optional[ax.AnchoredClusters]
                              ) -> None:
        """The boxes' texts, with how many clusters each group holds."""
        boxes = self.cluster_toggles
        if found is None:
            # Not classified yet: every cluster is drawn plain, under the
            # last box, and the three groups that need both images wait.
            for key in ("both", "tubulin", "spectrin"):
                boxes[key].setEnabled(False)
            boxes["both"].setText("Discarded: inside both images")
            boxes["tubulin"].setText("Inside the tubulin mask only")
            boxes["spectrin"].setText("Inside the spectrin interior only")
            boxes["neither"].setText("Clusters (not classified yet)")
            return
        for key in ("both", "tubulin", "spectrin"):
            boxes[key].setEnabled(True)
        boxes["both"].setText(
            f"Discarded: inside both images ({found.n_discarded})")
        boxes["tubulin"].setText(
            f"Inside the tubulin mask only ({found.n_tubulin_only})")
        boxes["spectrin"].setText(
            f"Inside the spectrin interior only ({found.n_spectrin_only})")
        boxes["neither"].setText(
            f"Inside neither: on the membrane "
            f"({int(found.inside_neither.sum())})")

    def _build_images(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("1. Images")
        lay = QtWidgets.QGridLayout(box)
        self.edit_tubulin = QtWidgets.QLineEdit()
        self.edit_reference = QtWidgets.QLineEdit()
        self.edit_movie = QtWidgets.QLineEdit(str(self.inputs.movie.path))
        images = mps_file_drop.IMAGE_SUFFIXES
        # A widefield image, or -- dual view -- the localizations of the
        # same protein acquired together with the movie, which need no
        # alignment at all.
        either = (tuple(mps_file_drop.IMAGE_SUFFIXES)
                  + tuple(mps_file_drop.LOCALIZATION_SUFFIXES))
        pattern = ("Widefield image or localizations "
                   "(*.tif *.tiff *.hdf5 *.h5 *.csv)")
        rows = (
            ("betaIII-tubulin, widefield image or localizations:",
             self.edit_tubulin, "betaIII-tubulin", pattern,
             self._load_tubulin, either),
            ("betaII-spectrin, widefield image or localizations (to align; "
             "section 5 needs it):",
             self.edit_reference, "betaII-spectrin", pattern,
             self._load_reference, either),
            ("Localizations of the whole movie (to align):", self.edit_movie,
             "Localizations of the whole movie",
             "Localizations (*.hdf5 *.h5 *.csv)", None,
             mps_file_drop.LOCALIZATION_SUFFIXES),
        )
        for i, (text, edit, title, pattern, loader, suffixes) in enumerate(rows):
            lay.addWidget(_label(text, _DIM), 2 * i, 0, 1, 2)
            button = QtWidgets.QPushButton("Browse...")
            button.setToolTip(f"Choose the file for: {text}\n\n"
                              f"It can also be dropped onto the box beside "
                              f"this button.")
            button.clicked.connect(
                lambda _=False, e=edit, t=title, p=pattern, f=loader:
                self._browse(e, t, p, f))
            if loader is not None:
                edit.editingFinished.connect(loader)
            mps_file_drop.accept_files(
                edit, suffixes, self._dropped(edit, loader),
                hint="Drop the file here, or press Browse")
            lay.addWidget(edit, 2 * i + 1, 0)
            lay.addWidget(button, 2 * i + 1, 1)
        self.edit_movie.setToolTip(
            "The alignment needs the whole field of the movie the selection "
            "was picked from: one picked axon alone rarely aligns.")
        for edit in (self.edit_tubulin, self.edit_reference):
            edit.setToolTip(
                "A widefield image taken with the movie, placed on the "
                "localizations by the shift of section 2.\n\n"
                "Or, for dual-view data, a localization file of the same "
                "protein acquired AT THE SAME TIME as the movie, on the "
                "other half of the camera: its localizations are counted on "
                "the movie's own camera grid, so it needs no shift and no "
                "widefield image. For the spectrin, that can be the movie's "
                "own file.")
        return box

    def _build_alignment(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("2. Alignment")
        lay = QtWidgets.QGridLayout(box)
        self.btn_measure = QtWidgets.QPushButton("Measure the shift")
        self.btn_measure.setToolTip(
            "Cross-correlates the spectrin widefield image with the "
            "localizations. The tubulin image is taken to share the "
            "widefield image's position, since they were acquired together.")
        self.btn_measure.clicked.connect(self.measure)
        self.btn_back = QtWidgets.QPushButton("Back to the measured shift")
        self.btn_back.setToolTip(
            "Undo a shift typed by hand and go back to the one the "
            "correlation measured.\n\n"
            "The exported row records which of the two it was, so a "
            "shift moved by hand is never reported as a measured one.")
        self.btn_back.clicked.connect(self._back_to_measured)
        lay.addWidget(self.btn_measure, 0, 0, 1, 2)
        lay.addWidget(self.btn_back, 0, 2, 1, 2)
        self.spin_dx = _spin(-20000, 20000, 10, suffix=" nm")
        self.spin_dy = _spin(-20000, 20000, 10, suffix=" nm")
        for spin in (self.spin_dx, self.spin_dy):
            spin.setToolTip("Added to the localizations to land them on the "
                            "widefield images.")
            spin.valueChanged.connect(self._shift_edited)
        lay.addWidget(_label("Shift x:", _DIM), 1, 0)
        lay.addWidget(self.spin_dx, 1, 1)
        lay.addWidget(_label("y:", _DIM), 1, 2)
        lay.addWidget(self.spin_dy, 1, 3)
        self.label_alignment = _label("")
        lay.addWidget(self.label_alignment, 2, 0, 1, 4)
        return box

    def _build_mask(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("3. Axoplasm mask")
        lay = QtWidgets.QGridLayout(box)
        self.slider_threshold = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_threshold.setRange(0, 1000)
        self.slider_threshold.valueChanged.connect(self._slider_moved)
        self.spin_threshold = _spin(-1e9, 1e9, 1)
        self.spin_threshold.setToolTip(
            "The grey level that separates the inside of the axon from "
            "the background in the tubulin image.\n\n"
            "Everything brighter is axoplasm. The slider beside it does "
            "the same thing; Otsu picks a value from the image itself.")
        self.spin_threshold.valueChanged.connect(self._threshold_typed)
        self.btn_otsu = QtWidgets.QPushButton("Otsu")
        self.btn_otsu.setToolTip(
            "Otsu's threshold (1979) for the region around the axon.")
        self.btn_otsu.clicked.connect(self._use_otsu)
        lay.addWidget(_label("Threshold:", _DIM), 0, 0)
        lay.addWidget(self.slider_threshold, 0, 1)
        lay.addWidget(self.spin_threshold, 0, 2)
        lay.addWidget(self.btn_otsu, 0, 3)
        self.spin_smooth = _spin(0, 2000, 10, suffix=" nm")
        self.spin_smooth.setValue(ax.DEFAULT_SMOOTH_SIGMA_PX * self.pixel_nm)
        self.spin_smooth.setToolTip(
            "Gaussian smoothing of the tubulin image before thresholding "
            "(sigma).")
        self.spin_smooth.valueChanged.connect(
            lambda _v: self._schedule("rebuild"))
        lay.addWidget(_label("Smoothing:", _DIM), 1, 0)
        lay.addWidget(self.spin_smooth, 1, 1)
        self.label_mask = _label("")
        lay.addWidget(self.label_mask, 2, 0, 1, 4)
        return box

    def _build_classification(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("4. Membrane or interior")
        lay = QtWidgets.QGridLayout(box)
        self.spin_margin = _spin(0, 5000, 10, suffix=" nm")
        self.spin_margin.setValue(ax.DEFAULT_MARGIN_NM)
        self.spin_margin.setToolTip(
            "How far inside a widefield edge a cluster must be before that "
            "image puts it inside the axon. The edge is blurred by the "
            "diffraction limit, so a spectrin ring on the membrane falls "
            "half inside without a margin. A cluster is inside only when "
            "both images put it inside (section 5), and a localization only "
            "when its cluster is.")
        self.spin_margin.valueChanged.connect(
            lambda _v: self._schedule("reclassify"))
        lay.addWidget(_label("Margin:", _DIM), 0, 0)
        lay.addWidget(self.spin_margin, 0, 1)
        self.btn_export = QtWidgets.QPushButton("Export axon...")
        self.btn_export.setToolTip(
            "Write this axon to the tables: the analysis, this panel and "
            "the discard in one row, and the per-cluster and "
            "per-localization tables if you ask for them. The same button "
            "as in the main window, so nothing is written twice.")
        self.btn_export.clicked.connect(self._export_clicked)
        lay.addWidget(self.btn_export, 0, 2)
        self.btn_image = QtWidgets.QPushButton("Export image...")
        self.btn_image.setToolTip(
            "Write the image above, or the distance histogram, as a figure: "
            "at the width a journal asks for, at 300 to 1200 dpi, or as an "
            "SVG. What is drawn is what the checkboxes leave on.")
        self.btn_image.clicked.connect(self._on_export_image)
        lay.addWidget(self.btn_image, 0, 3)
        self.label_result = _label("")
        lay.addWidget(self.label_result, 1, 0, 1, 3)
        return box

    def _build_anchored(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("5. Clusters anchored to the membrane")
        box.setToolTip(
            "A cluster of the MPS analysis is discarded only when BOTH "
            "widefield images put it inside the axon by more than the "
            "margin: inside the tubulin mask, and inside the dark area the "
            "betaII-spectrin ring encloses. The contour is then rebuilt "
            "from the rest with 2-opt from every starting point, as the "
            "MPS analysis builds its own, and the MPS analysis window "
            "repeats every parameter with and without them. The cross is "
            "the centre of that contour, its area centroid.")
        lay = QtWidgets.QVBoxLayout(box)
        self.label_anchored = _label("")
        lay.addWidget(self.label_anchored)
        return box

    # ------------------------------------------------------------ state
    @property
    def pixel_nm(self) -> float:
        return float(self.inputs.loc.pixel_size_nm)

    def _show_header(self) -> None:
        loc = self.inputs.loc
        self.label_header.setText(
            f"Channel 1: {os.path.basename(str(self.inputs.movie.path))}. "
            f"Selection: {loc.n:,} localizations, ROI "
            f"{describe_roi(self.inputs.roi)}; pixel {self.pixel_nm:g} nm.")

    def update_selection(self, inputs: AxoplasmInputs) -> None:
        """The main window applied another selection."""
        self.inputs = inputs
        if self._manual_threshold is not None:
            # A threshold chosen for one axon is not carried to another. The
            # note stays until the threshold is touched again: dragging an
            # ROI can apply several selections in a row.
            self._manual_threshold = None
            self.selection_note = (
                "The selection changed in the main window: the threshold "
                "set by hand was replaced by Otsu's for the new selection.")
        self._show_header()
        self._rebuild()

    def refresh_clusters(self) -> None:
        """The main window analysed the selection's clusters again."""
        self._reclassify()

    def closeEvent(self, event: Any) -> None:
        self._generation += 1
        # A timer armed just before the close would otherwise fire on a
        # hidden window and push a new discard into the main window. The
        # panel is kept alive and reopened on one of the three close
        # paths, so it cannot be left to deletion.
        self._recompute.stop()
        self._pending = None
        self._close_progress()
        super().closeEvent(event)

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.hide()
            self._progress.deleteLater()
            self._progress = None

    def _dropped(self, edit: QtWidgets.QLineEdit,
                 loader: Optional[Callable[[], None]]
                 ) -> Callable[[List[str]], None]:
        """Take the first file of a drag into one of the fields."""
        def handler(paths: List[str]) -> None:
            edit.setText(paths[0])
            if loader is not None:
                loader()
        return handler

    def _browse(self, edit: QtWidgets.QLineEdit, title: str, pattern: str,
                loader: Optional[Callable[[], None]]) -> None:
        start = os.path.dirname(edit.text() or str(self.inputs.movie.path))
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, title, start, pattern)
        if chosen:
            edit.setText(chosen)
            if loader is not None:
                loader()

    def _load_image(self, edit: QtWidgets.QLineEdit, what: str
                    ) -> Tuple[Optional[ax.WidefieldImage], Tuple[float, float],
                               List[str]]:
        path = edit.text().strip()
        if not path:
            return None, (0.0, 0.0), []
        try:
            if path.lower().endswith(tuple(
                    mps_file_drop.LOCALIZATION_SUFFIXES)):
                movie = self.inputs.movie
                extent = (0.0, 0.0)
                if movie is not None and len(movie.x_nm):
                    extent = (float(np.max(movie.x_nm)) / self.pixel_nm,
                              float(np.max(movie.y_nm)) / self.pixel_nm)
                image = ax.image_from_localizations(
                    path, self.pixel_nm, min_extent_px=extent)
                offset = (0.0, 0.0)
                notes: List[str] = []
            else:
                image = ax.load_widefield(path)
                offset, notes = ax.camera_offset(
                    image, self.inputs.loc.info, self.pixel_nm)
        except (OSError, ValueError) as error:
            QtWidgets.QMessageBox.critical(
                self, "Axoplasm", f"The {what} image was not loaded:\n{error}")
            edit.setText("")
            return None, (0.0, 0.0), []
        return image, offset, list(image.notes) + notes

    def load_tubulin(self, path: str) -> None:
        self.edit_tubulin.setText(path)
        self._load_tubulin()

    def load_reference(self, path: str) -> None:
        self.edit_reference.setText(path)
        self._load_reference()

    def _load_tubulin(self) -> None:
        current = self.tubulin.path if self.tubulin else ""
        if self.edit_tubulin.text().strip() == current:
            return
        (self.tubulin, self.tubulin_offset,
         self.tubulin_notes) = self._load_image(self.edit_tubulin, "tubulin")
        self._manual_threshold = None
        self.selection_note = None
        self._rebuild()

    def _load_reference(self) -> None:
        current = self.reference.path if self.reference else ""
        if self.edit_reference.text().strip() == current:
            return
        (self.reference, self.reference_offset,
         self.reference_notes) = self._load_image(self.edit_reference,
                                                  "spectrin widefield")
        # The spectrin image gives section 5 its ring interior: finding the
        # discarded clusters again, with this image or without one, also
        # tells the main window.
        self._reclassify()

    # -------------------------------------------------------- alignment
    def _shift_px(self) -> Tuple[float, float]:
        return (self.spin_dx.value() / self.pixel_nm,
                self.spin_dy.value() / self.pixel_nm)

    def registration(self) -> ax.ImageRegistration:
        """The shift in use, with the measurement it came from."""
        if self._placed_by_construction():
            # Every image is the movie's own kind of data, acquired with it:
            # nothing was measured and nothing needs to be.
            return ax.ImageRegistration(shift_px=(0.0, 0.0),
                                        source="same acquisition")
        shift = self._shift_px()
        measured = self.measured.registration if self.measured else None
        if measured is None:
            source = "none" if shift == (0.0, 0.0) else "manual"
            return ax.ImageRegistration(shift_px=shift, source=source)
        same = (abs(shift[0] - measured.shift_px[0]) * self.pixel_nm < 0.05
                and abs(shift[1] - measured.shift_px[1]) * self.pixel_nm < 0.05)
        return ax.ImageRegistration(
            shift_px=shift, source="measured" if same else "adjusted",
            peak=measured.peak, zero=measured.zero,
            runner_up=measured.runner_up, score=measured.score,
            n_localizations=measured.n_localizations,
            warnings=list(measured.warnings))

    def _set_shift(self, shift_px: Tuple[float, float]) -> None:
        self._updating = True
        try:
            self.spin_dx.setValue(shift_px[0] * self.pixel_nm)
            self.spin_dy.setValue(shift_px[1] * self.pixel_nm)
        finally:
            self._updating = False

    def _shift_edited(self, _value: float) -> None:
        # The shift spins step by 10 nm and are held down the same way
        # the margin is.
        if not self._updating:
            self._schedule("rebuild")

    def _back_to_measured(self) -> None:
        if self.measured is not None:
            self._set_shift(self.measured.registration.shift_px)
            self._rebuild()

    def measure(self) -> None:
        """Measure the shift in a worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self.reference is None:
            QtWidgets.QMessageBox.warning(
                self, "Axoplasm",
                "Choose the widefield betaII-spectrin image first.")
            return
        path = self.edit_movie.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(
                self, "Axoplasm",
                "Choose the localization file of the whole movie.")
            return
        reference = self.reference
        movie = self.inputs.movie
        selection = self.inputs.loc
        generation = self._generation

        def work() -> _Measurement:
            notes: List[str] = []
            if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
                    os.path.abspath(str(movie.path))):
                loc = movie
            else:
                loc = load_localizations(path)
            pixel = loc.pixel_size_nm
            if not pixel or abs(pixel - selection.pixel_size_nm) > 1e-6:
                raise ValueError(
                    f"{os.path.basename(path)} gives a pixel size of {pixel} "
                    f"nm and the selection {selection.pixel_size_nm:g} nm.")
            source = ax.movie_geometry(loc.info)[3]
            own = ax.movie_geometry(selection.info)[3]
            if source is None or own is None:
                notes.append(
                    "Picasso's metadata does not name the movie of one of "
                    "the files: check that the alignment localizations come "
                    "from the same movie as the selection.")
            elif os.path.basename(source) != os.path.basename(own):
                notes.append(
                    f"The alignment localizations come from "
                    f"{os.path.basename(source)} and the selection from "
                    f"{os.path.basename(own)}: the shift may not apply.")
            offset, more = ax.camera_offset(reference, loc.info, pixel)
            notes.extend(more)
            registration = ax.measure_shift(
                reference.image, loc.x_nm / pixel + offset[0],
                loc.y_nm / pixel + offset[1])
            return _Measurement(registration=registration, path=path,
                                reference=reference.path, notes=notes)

        def body() -> None:
            try:
                result, error = work(), None
            except Exception as exc:  # noqa: BLE001 - shown to the user
                result, error = None, exc
            self._relay.finished.emit(result, error, generation)

        self._progress = QtWidgets.QProgressDialog(
            "Measuring the shift...", None, 0, 0, self)
        self._progress.setWindowTitle("Axoplasm")
        self._progress.setWindowModality(QtCore.Qt.WindowModal)
        self._progress.setMinimumDuration(0)
        self._progress.show()
        self.btn_measure.setEnabled(False)
        self._thread = threading.Thread(target=body, daemon=True)
        self._thread.start()

    def wait(self, timeout: float = 600.0) -> None:
        """Block until a measurement finishes (for scripts and tests)."""
        self.flush()
        if self._thread is not None:
            self._thread.join(timeout)
        QtWidgets.QApplication.processEvents()

    def _measured(self, result: Optional[_Measurement],
                  error: Optional[BaseException], generation: int) -> None:
        # The worker is done once its result arrives, even if its thread
        # has not quite exited: the Measure button must come back.
        self._thread = None
        if generation != self._generation:
            self._refresh()
            return
        self._close_progress()
        if error is not None or result is None:
            self._refresh()
            QtWidgets.QMessageBox.critical(
                self, "Axoplasm", f"The shift could not be measured:\n{error}")
            return
        self.measured = result
        self._set_shift(result.registration.shift_px)
        self._rebuild()

    # ------------------------------------------------------ computation
    def _coordinates(self, x_nm: np.ndarray, y_nm: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Tubulin-image pixel coordinates of positions in nm."""
        return self._coordinates_for(self.tubulin, self.tubulin_offset,
                                     x_nm, y_nm)

    def _coordinates_for(self, image: Optional[ax.WidefieldImage],
                         offset: Tuple[float, float], x_nm: np.ndarray,
                         y_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates in ``image`` of positions in nm.

        An image drawn from localizations acquired with the movie sits on
        the movie's own camera grid: no camera offset and no shift, which
        belong to widefield images taken at another moment.
        """
        if image is not None and image.from_localizations:
            return (np.asarray(x_nm, float) / self.pixel_nm,
                    np.asarray(y_nm, float) / self.pixel_nm)
        return self._coordinates_in(offset, x_nm, y_nm)

    def _placed_by_construction(self) -> bool:
        """Every image loaded was drawn from localizations of the movie."""
        loaded = [i for i in (self.tubulin, self.reference) if i is not None]
        return bool(loaded) and all(i.from_localizations for i in loaded)

    def _needs_shift(self) -> bool:
        """Some image in use is a widefield image placed by a shift."""
        return any(i is not None and not i.from_localizations
                   for i in (self.tubulin, self.reference))

    def _coordinates_in(self, offset: Tuple[float, float], x_nm: np.ndarray,
                        y_nm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates, in an image at ``offset``, of positions in nm."""
        sx, sy = self._shift_px()
        return (np.asarray(x_nm, float) / self.pixel_nm + offset[0] + sx,
                np.asarray(y_nm, float) / self.pixel_nm + offset[1] + sy)

    def _schedule(self, what: str) -> None:
        """Recompute once the drag stops, not on every step of it.

        A rebuild subsumes a reclassify -- it ends in one -- so a pending
        rebuild is never downgraded.
        """
        self._pending = ("rebuild" if (what == "rebuild"
                                       or self._pending == "rebuild")
                         else "reclassify")
        self._recompute.start()

    def _run_pending(self) -> None:
        """What the timer fires. Clears the request before dispatching,
        because the work below re-enters the event loop."""
        what, self._pending = self._pending, None
        if what == "rebuild":
            self._rebuild()
        elif what == "reclassify":
            self._reclassify()

    def flush(self) -> None:
        """Do now whatever a drag has left pending.

        Anything that READS what the panel computed has to call this
        first: what is exported, drawn into a figure or compared against
        the analysis must be what the controls currently say, not what
        they said 250 ms ago.
        """
        if self._pending is not None:
            self._recompute.stop()
            self._run_pending()

    def _rebuild(self) -> None:
        """Mask and classification from the current controls."""
        self.axoplasm = self.result = self.located = self.loc_cluster = None
        self.region_notes = []
        if self.tubulin is not None and self.inputs.loc.n:
            col, row = self._coordinates(self.inputs.loc.x_nm,
                                         self.inputs.loc.y_nm)
            centre, radius, reach = ax.axon_centre(col, row)
            try:
                self.axoplasm = ax.build_mask(
                    self.tubulin.image, centre, radius, self.pixel_nm,
                    reach_px=reach, threshold=self._manual_threshold,
                    smooth_sigma_px=self.spin_smooth.value() / self.pixel_nm)
            except ValueError as error:
                self.region_notes = [str(error)]
            else:
                self._sync_threshold()
        self._reclassify()

    def _reclassify(self) -> None:
        # Seven callers reach here directly -- loading an image, pressing
        # Otsu, going back to the measured shift, a new selection. Any of
        # them can land inside the pause a drag opened, and without this
        # the timer would fire afterwards and run the whole chain a
        # second time, including the 2.6 s discard comparison.
        self._recompute.stop()
        self._pending = None
        self.result = self.located = self.loc_cluster = None
        self.spectrin_interior = self.anchored = None
        self.anchored_centroids = None
        self.anchored_notes = []
        if self.axoplasm is not None:
            margin = self.spin_margin.value()
            loc = self.inputs.loc
            col, row = self._coordinates(loc.x_nm, loc.y_nm)
            self.result = ax.classify(self.axoplasm, col, row, margin)
            centroids = self.inputs.clusters()
            if centroids is not None:
                c = np.asarray(centroids, float)
                if c.size == 0:
                    c = np.empty((0, 2))
                self._find_anchored(c[:, :2], margin)
            if self.anchored is not None:
                cluster_of = self.inputs.cluster_of(loc.x_nm, loc.y_nm,
                                                    loc.z_nm)
                if cluster_of is not None and len(cluster_of) == loc.n:
                    self.loc_cluster = np.asarray(cluster_of, dtype=np.intp)
                    self.located = ax.localization_labels(
                        self.loc_cluster, self.anchored.discarded)
        self._refresh()
        # After drawing: the main window may take seconds to repeat the MPS
        # analysis without the discarded clusters.
        self.inputs.anchored_changed(self.anchored, self.anchored_centroids)

    def _registration_state(self) -> str:
        """How the widefield images are placed right now, in one phrase."""
        reg = self.registration()
        words = {"none": "no shift", "manual": "set by hand",
                 "measured": "measured",
                 "adjusted": "measured, then adjusted by hand",
                 "same acquisition": "placed by construction: every image "
                                     "drawn from localizations acquired "
                                     "with the movie"}
        state = words.get(reg.source, reg.source)
        if reg.score is not None:
            state += f", score {reg.score:.1f}"
            if reg.score < ax.MIN_REGISTRATION_SCORE:
                state += " (low)"
        return state

    def _find_anchored(self, centroids: np.ndarray, margin: float) -> None:
        """The clusters both images put inside, and the contour without."""
        if self.reference is None or self.axoplasm is None \
                or len(centroids) < 3:
            return
        # Which clusters are inside depends entirely on where the images
        # sit: with no shift, axon 7 gave 4 discarded and a 19.65 um
        # contour against 5 and 18.48 um once the shift was measured.
        # Sorting them before anything is placed would hand the results
        # window, and the tables, a discard measured on an unplaced image.
        # Only a widefield image needs placing. Localizations acquired with
        # the movie are placed by construction, and asking for a shift
        # there would block the one case that needs none.
        if self._needs_shift() and self.registration().source == "none":
            self.anchored_notes = [
                "The widefield images are not placed on the localizations "
                "yet: measure the shift in section 2 (or set one by hand) "
                "before the clusters can be sorted."]
            return
        offset = self.reference_offset
        col, row = self._coordinates_for(self.reference, offset,
                                         self.inputs.loc.x_nm,
                                         self.inputs.loc.y_nm)
        centre, radius, reach = ax.axon_centre(col, row)
        scol, srow = self._coordinates_for(self.reference, offset,
                                           centroids[:, 0], centroids[:, 1])
        try:
            interior = ax.build_ring_interior(
                self.reference.image, centre, radius, self.pixel_nm,
                ring_col=scol, ring_row=srow, reach_px=reach)
        except ValueError as error:
            self.anchored_notes = [str(error)]
            return
        if len(self._contour_cache) > CONTOUR_CACHE_SIZE:
            self._contour_cache.clear()
        tcol, trow = self._coordinates(centroids[:, 0], centroids[:, 1])
        self.spectrin_interior = interior
        self.anchored_centroids = centroids
        self.anchored = ax.anchored_clusters(
            centroids, self.axoplasm, tcol, trow, interior, scol, srow,
            margin, contour_all=self.inputs.contour(),
            contour_cache=self._contour_cache,
            registration=self._registration_state())

    def _sync_threshold(self) -> None:
        mask = self.axoplasm
        if mask is None:
            return
        low = float(np.min(mask.smoothed))
        high = float(np.max(mask.smoothed))
        self._updating = True
        try:
            self._threshold_range = (low, high)
            self.spin_threshold.setValue(mask.threshold)
            span = high - low
            position = 0 if span <= 0 else int(round(
                1000 * (mask.threshold - low) / span))
            self.slider_threshold.setValue(max(0, min(1000, position)))
        finally:
            self._updating = False

    def _slider_moved(self, position: int) -> None:
        if self._updating or self.axoplasm is None:
            return
        low, high = self._threshold_range
        # The threshold itself follows the slider at once; only the mask
        # built from it waits for the drag to stop.
        self._manual_threshold = low + (high - low) * position / 1000.0
        self.selection_note = None
        self._schedule("rebuild")

    def _threshold_typed(self, value: float) -> None:
        if self._updating or self.axoplasm is None:
            return
        self._manual_threshold = float(value)
        self.selection_note = None
        self._schedule("rebuild")

    def _use_otsu(self) -> None:
        self._manual_threshold = None
        self.selection_note = None
        self._rebuild()

    # ---------------------------------------------------------- display
    def _clear(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def warnings(self) -> List[str]:
        out = self.tubulin_notes + self.reference_notes + self.region_notes
        if self.selection_note:
            out.append(self.selection_note)
        if self.measured is not None:
            out.extend(self.measured.notes)
            out.extend(self.measured.registration.warnings)
        if self.axoplasm is not None:
            out.extend(self.axoplasm.warnings)
        out.extend(self.anchored_notes)
        if self.spectrin_interior is not None:
            out.extend(self.spectrin_interior.warnings)
        if self.anchored is not None:
            out.extend(self.anchored.warnings)
        return out

    def _refresh(self) -> None:
        self.btn_measure.setEnabled(
            self.reference is not None
            and not self.reference.from_localizations
            and not (self._thread is not None and self._thread.is_alive()))
        for widget in (self.spin_dx, self.spin_dy):
            widget.setEnabled(self._needs_shift())
        self.btn_back.setEnabled(self.measured is not None)
        self.btn_export.setEnabled(self.result is not None)
        for widget in (self.slider_threshold, self.spin_threshold,
                       self.btn_otsu):
            widget.setEnabled(self.axoplasm is not None)
        self._clear(self.findings)
        messages = self.warnings()
        if self.tubulin is None:
            self.findings.addWidget(_label(marked(
                "dim",
                "Choose the betaIII-tubulin widefield image, or its "
                "localizations when they were acquired with the movie."),
                _DIM))
        elif not messages:
            self.findings.addWidget(_label(
                marked("good", "No warnings."), _TEXT_OK))
        for text in messages:
            self.findings.addWidget(_label(marked("warn", text), _WARN))
        self.findings.addStretch(1)
        self._write_labels()
        self._draw()

    def _anchored_text(self) -> str:
        if self.tubulin is None:
            return ""
        if self.reference is None:
            return ("Load the betaII-spectrin widefield image (section 1): "
                    "a cluster is discarded only when it and the tubulin "
                    "both put it inside the axon.")
        found = self.anchored
        if found is None:
            # Why there is nothing to show, when there is a reason: an
            # empty section 5 left "measure the shift first" unsaid.
            if self.anchored_notes:
                return self.anchored_notes[0]
            if self.axoplasm is None:
                return ""
            if self.inputs.clusters() is None:
                return ("Run 'cluster Ch1' on this selection: the clusters "
                        "are the ones the MPS analysis kept.")
            return "Fewer than three clusters: there is no ring to look inside."
        lines = []
        interior = self.spectrin_interior
        if interior is not None and interior.mask.any():
            levels = interior.levels
            how = ("half way between the axon's dark inside and its ring"
                   if interior.threshold_source == "half maximum" else
                   "short of half way, where the ring opens")
            lines.append(
                f"Spectrin interior (purple): {interior.area_um2:.2f} µm², "
                f"cut at {interior.threshold:.0f} counts, {how} (inside "
                f"{levels['interior']:.0f}, ring {levels['ring']:.0f}).")
        else:
            lines.append("No spectrin interior was found, so no cluster is "
                         "discarded.")
        lines.append(
            f"Discarded (orange discs): {found.n_discarded} of {found.n} "
            f"clusters, "
            f"inside both images by more than {found.margin_nm:.0f} nm. "
            f"Kept although one image alone puts them inside: "
            f"{found.n_tubulin_only} by the tubulin (green), "
            f"{found.n_spectrin_only} by the spectrin (purple).")
        parts = []
        same = found.contour_all_starts is found.contour_all
        if found.contour_all is not None:
            parts.append(f"{found.contour_all.perimeter_um:.2f} µm with all "
                         f"clusters, as the MPS analysis connects them"
                         + (" (2-opt from every start; orange, dashed)"
                            if same else " (orange, dashed)"))
        if found.contour_all_starts is not None and not same:
            parts.append(f"{found.contour_all_starts.perimeter_um:.2f} µm with "
                         f"all clusters and 2-opt from every start")
        new = found.contour_anchored
        if new is not None:
            spread = new.start_spread_um
            parts.append(
                f"{new.perimeter_um:.2f} µm without the discarded ones "
                f"(blue; 2-opt from all {new.n_starts} starts"
                + (f", which a single start could have missed by up to "
                   f"{spread:.2f} µm" if spread else "") + ")")
        if parts:
            lines.append("Perimeter: " + "; ".join(parts) + ".")
        if new is not None and new.centre is not None:
            c = new.centre
            lines.append(
                f"Centre of the contour (+, its area centroid): "
                f"x {c.x_nm:.0f}, y {c.y_nm:.0f} nm"
                + ("" if c.max_shift_nm is None else
                   f"; it moves at most {c.max_shift_nm:.0f} nm when one "
                   f"cluster is left out") + ".")
        if new is not None and found.contour_all_starts is not None:
            lines.append("The MPS analysis window repeats every parameter "
                         "without the discarded clusters, next to the "
                         "measured ones.")
        return "\n".join(lines)

    def _write_labels(self) -> None:
        self.label_anchored.setText(self._anchored_text())
        reg = self.registration()
        sx, sy = reg.shift_nm(self.pixel_nm)
        words = {"none": "no shift", "manual": "set by hand",
                 "measured": "as measured", "adjusted": "measured, then "
                 "adjusted by hand",
                 "same acquisition": "none needed: the images are drawn "
                                     "from localizations acquired with the "
                                     "movie"}.get(reg.source, reg.source)
        lines = [f"Shift in use: x {sx:+.0f} nm, y {sy:+.0f} nm ({words})."]
        if self.measured is not None:
            m = self.measured.registration
            msx, msy = m.shift_nm(self.pixel_nm)
            lines.append(
                f"Measured against "
                f"{os.path.basename(self.measured.reference)}: "
                f"({msx:+.0f}, {msy:+.0f}) nm on "
                f"{m.n_localizations:,} localizations; correlation "
                f"{m.peak:.2f} (unshifted {m.zero:.2f}, next peak "
                f"{'-' if m.runner_up is None else f'{m.runner_up:.2f}'}); "
                f"score {'-' if m.score is None else f'{m.score:.0f}'}.")
        times = []
        if self.tubulin is not None and self.tubulin.acquired:
            times.append(f"tubulin at {self.tubulin.acquired}")
        if self.reference is not None and self.reference.acquired:
            times.append(f"spectrin at {self.reference.acquired}")
        movie_time = ax.movie_geometry(self.inputs.loc.info)[2]
        if movie_time:
            times.append(f"the movie started at {movie_time}")
        if times:
            lines.append("Acquired: " + "; ".join(times) + ".")
        self.label_alignment.setText("\n".join(lines))

        mask = self.axoplasm
        if mask is None:
            self.label_mask.setText("")
        else:
            ratio = mask.area_ratio
            if mask.threshold_source == "otsu":
                source = "Otsu's"
            else:
                source = f"set by hand; Otsu's is {mask.otsu:.1f}"
            self.label_mask.setText(
                f"Threshold {mask.threshold:.1f} ({source}). "
                f"Mask {mask.area_um2:.2f} µm²"
                + ("" if ratio is None else
                   f", {ratio:.2f} times the area inside the spectrin ring")
                + ".")

        self.label_result.setText(self._located_text())

    def _located_text(self) -> str:
        """
        Section 4: where the localizations are. A localization is counted
        inside only through its cluster, when both images put that cluster
        inside; the tubulin mask alone decides nothing here.
        """
        result = self.result
        if result is None:
            return ""
        text = f"{result.n:,} localizations"
        outside = result.count(ax.LABEL_OUTSIDE)
        if outside:
            text += f", {outside:,} of them outside the analysed region"
        text += "; their distances to the tubulin mask's edge are below. "
        located = self.located
        found = self.anchored
        centroids = self.inputs.clusters()
        if located is not None and found is not None:
            inside = int(np.sum(located == ax.LOC_INSIDE))
            membrane = int(np.sum(located == ax.LOC_MEMBRANE))
            free = int(np.sum(located == ax.LOC_NO_CLUSTER))
            share = inside / result.n if result.n else 0.0
            text += (
                f"Inside the axon: {inside:,} ({share:.1%}), the "
                f"localizations of the {found.n_discarded} clusters both "
                f"images put inside. At the membrane: {membrane:,}, those "
                f"of the {found.n - found.n_discarded} kept clusters. In no "
                f"cluster: {free:,}.")
        elif centroids is None:
            text += ("Run 'cluster Ch1' on this selection: a localization "
                     "is counted inside only through its cluster.")
        elif len(centroids) == 0:
            text += "The MPS analysis kept none of this selection's clusters."
        elif self.reference is None:
            text += ("Load the betaII-spectrin widefield image: a "
                     "localization is counted inside only when both images "
                     "put its cluster inside (section 5).")
        else:
            text += "Section 5 has not placed the clusters yet."
        return text

    def _region_rect(self, rows: Tuple[int, int], cols: Tuple[int, int],
                     offset: Tuple[float, float]) -> QtCore.QRectF:
        """Where image pixels [rows) x [cols) lie, in localization nm."""
        sx, sy = self._shift_px()
        px = self.pixel_nm
        return QtCore.QRectF(
            (cols[0] - 0.5 - offset[0] - sx) * px,
            (rows[0] - 0.5 - offset[1] - sy) * px,
            (cols[1] - cols[0]) * px, (rows[1] - rows[0]) * px)

    def _draw(self) -> None:
        mask = self.axoplasm
        for item in (self.membrane_item, self.interior_item, self.free_item,
                     self.cluster_item, self.tubulin_only_item,
                     self.spectrin_only_item, self.discarded_item,
                     self.contour_all_item, self.contour_item,
                     self.centre_item):
            item.setData([], [])
        self.spectrin_outline_item.clear()
        self._sync_cluster_toggles(self.anchored
                                   if self.anchored_centroids is not None
                                   else None)
        self._sync_localization_toggles(self.located)
        if mask is None or self.tubulin is None:
            self.image_item.clear()
            self.outline_item.clear()
            self.plot_hist.clear()
            set_title(self.plot_image, "betaIII-tubulin around the axon")
            return
        r0, r1, c0, c1 = mask.region
        image, offset, title = (
            self.tubulin.image, self.tubulin_offset,
            "betaIII-tubulin around the axon")
        rows, cols = (r0, r1), (c0, c1)
        if self.check_reference.isChecked() and self.reference is not None:
            # The same area, in the reference image's pixels.
            dr = self.reference_offset[1] - self.tubulin_offset[1]
            dc = self.reference_offset[0] - self.tubulin_offset[0]
            height, width = self.reference.shape
            rows = (int(max(0, r0 + dr)), int(min(height, r1 + dr)))
            cols = (int(max(0, c0 + dc)), int(min(width, c1 + dc)))
            image, offset = self.reference.image, self.reference_offset
            title = "betaII-spectrin widefield around the axon"
        region = image[rows[0]:rows[1], cols[0]:cols[1]]
        set_title(self.plot_image, title)
        if region.size:
            low, high = np.percentile(region, (1, 99.7))
            self.image_item.setImage(region.T, levels=(low, max(high, low + 1)))
            self.image_item.setRect(self._region_rect(rows, cols, offset))
        else:
            self.image_item.clear()

        edge = mask.mask & ~ndimage.binary_erosion(mask.mask)
        self.outline_item.setImage(
            np.transpose(_cased_edge(edge, _OUTLINE), (1, 0, 2)),
            levels=(0, 255))
        self.outline_item.setRect(self._region_rect((r0, r1), (c0, c1),
                                                    self.tubulin_offset))

        result = self.result
        loc = self.inputs.loc
        if result is not None:
            rng = np.random.default_rng(0)
            # Where each localization is through its cluster; before both
            # images have placed the clusters, all of them are drawn plain.
            located = (self.located if self.located is not None
                       else np.full(loc.n, ax.LOC_NO_CLUSTER, dtype=object))
            for label, item in ((ax.LOC_NO_CLUSTER, self.free_item),
                                (ax.LOC_MEMBRANE, self.membrane_item),
                                (ax.LOC_INSIDE, self.interior_item)):
                idx = np.nonzero(located == label)[0]
                if idx.size > MAX_DRAWN:
                    idx = rng.choice(idx, MAX_DRAWN, replace=False)
                item.setData(loc.x_nm[idx], loc.y_nm[idx])
        interior = self.spectrin_interior
        if interior is not None and interior.mask.any():
            ring = interior.mask & ~ndimage.binary_erosion(interior.mask)
            cased = _cased_edge(ring, _SPECTRIN_OUTLINE)
            self.spectrin_outline_item.setImage(
                np.transpose(cased, (1, 0, 2)), levels=(0, 255))
            ir0, ir1, ic0, ic1 = interior.region
            self.spectrin_outline_item.setRect(self._region_rect(
                (ir0, ir1), (ic0, ic1), self.reference_offset))
        found = self.anchored
        if found is not None and self.anchored_centroids is not None:
            c = self.anchored_centroids
            # The four groups the boxes above the image show or hide.
            for item, group in ((self.discarded_item, found.discarded),
                                (self.tubulin_only_item, found.tubulin_only),
                                (self.spectrin_only_item,
                                 found.spectrin_only),
                                (self.cluster_item, found.inside_neither)):
                members = np.asarray(group, dtype=bool)
                item.setData(c[members, 0], c[members, 1])
            for item, drawn in ((self.contour_all_item, found.contour_all),
                                (self.contour_item, found.contour_anchored)):
                if drawn is not None:
                    closed = np.vstack([drawn.contour, drawn.contour[:1]])
                    item.setData(closed[:, 0], closed[:, 1])
            new = found.contour_anchored
            if new is not None and new.centre is not None:
                self.centre_item.setData([new.centre.x_nm],
                                         [new.centre.y_nm])
        else:
            centroids = self.inputs.clusters()
            if centroids is not None and len(centroids):
                c = np.asarray(centroids, float)
                self.cluster_item.setData(c[:, 0], c[:, 1])

        self.plot_hist.clear()
        if result is not None:
            d = result.distance_nm[np.isfinite(result.distance_nm)]
            if d.size:
                edges = np.arange(-HIST_RANGE_NM, HIST_RANGE_NM + HIST_BIN_NM,
                                  HIST_BIN_NM)
                counts, _ = np.histogram(np.clip(d, edges[0], edges[-1]),
                                         bins=edges)
                self.plot_hist.plot(edges, counts, stepMode="center",
                                    pen=pg.mkPen(TITLE_FG, width=1.5))
            self.plot_hist.addItem(pg.InfiniteLine(
                pos=0, angle=90,
                pen=pg.mkPen(neutral(dark=True), style=QtCore.Qt.DashLine)))
            self.plot_hist.addItem(pg.InfiniteLine(
                pos=result.margin_nm, angle=90, pen=pg.mkPen(_INTERIOR)))

    # ------------------------------------------------------------ export
    def summary_row(self) -> Dict[str, Any]:
        assert self.axoplasm is not None and self.result is not None
        return ax.summary_row(
            localizations=str(self.inputs.movie.path),
            tubulin=self.tubulin.path if self.tubulin else "",
            # The image the shift was measured against, not whichever is
            # loaded now; without a measurement, the one shown to align by
            # hand.
            reference=(self.measured.reference if self.measured
                       else self.reference.path if self.reference else ""),
            registration_file=self.measured.path if self.measured else "",
            roi=describe_roi(self.inputs.roi),
            pixel_size_nm=self.pixel_nm,
            offset_px=self.tubulin_offset,
            registration=self.registration(),
            mask=self.axoplasm, result=self.result,
            spectrin=self.spectrin_interior, anchored=self.anchored,
            tubulin_source=(self.tubulin.source if self.tubulin
                            else "widefield image"),
            spectrin_source=(self.reference.source
                             if self.reference is not None
                             and self.spectrin_interior is not None
                             else ""),
            spectrin_image=self._interior_image(), located=self.located,
            n_clusters=self._n_clusters(),
            z_range=self.inputs.z_range,
            z_range_source=self.inputs.z_range_source,
            warnings=self.warnings())

    def _n_clusters(self) -> Optional[int]:
        """How many clusters the MPS analysis kept here; None before."""
        centroids = self.inputs.clusters()
        return None if centroids is None else int(len(centroids))

    def _interior_image(self) -> str:
        """The spectrin image the ring interior was found in: the one
        loaded, since loading one finds the interior again."""
        if self.spectrin_interior is None or self.reference is None:
            return ""
        return self.reference.path

    def cluster_rows(self) -> List[Dict[str, Any]]:
        assert self.anchored is not None and self.anchored_centroids is not None
        return ax.cluster_rows(
            localizations=str(self.inputs.movie.path),
            roi=describe_roi(self.inputs.roi),
            centroids_nm=self.anchored_centroids, anchored=self.anchored,
            spectrin_image=self._interior_image(),
            labels=self.inputs.cluster_labels())

    def localization_rows(self) -> List[Dict[str, Any]]:
        assert self.result is not None
        loc = self.inputs.loc
        source = str(self.inputs.movie.path)
        roi = describe_roi(self.inputs.roi)
        # The cluster each localization is in (numbered as in the clusters
        # table) and where it is through that cluster; empty until both
        # images have placed the clusters.
        placed = self.located is not None and self.loc_cluster is not None
        cluster = (self.loc_cluster if placed
                   else np.full(loc.n, -1, dtype=np.intp))
        # The label the rest of the program calls this cluster by, not its
        # position among the kept ones: the two differ as soon as the
        # automatic curation removes a cluster.
        names = self.inputs.cluster_labels()
        where = (self.located if placed
                 else np.full(loc.n, "", dtype=object))
        return [
            {"source_localizations": source, "roi": roi,
             "x_nm": round(float(x), 2), "y_nm": round(float(y), 2),
             "z_nm": round(float(z), 2),
             # Empty where there is no distance: off the analysed region,
             # or no mask at all. Written as "nan" and "-inf", both were
             # read back as numbers.
             "distance_to_tubulin_edge_nm": (
                 round(float(d), 1) if np.isfinite(d) else None),
             "cluster_label": ("" if c < 0 else int(c) if names is None
                               else int(names[c])),
             "label": str(label)}
            for x, y, z, d, c, label in zip(loc.x_nm, loc.y_nm, loc.z_nm,
                                            self.result.distance_nm,
                                            cluster, where)
        ]

    def _on_export_image(self) -> None:
        """
        Write the image plot, or the histogram, as a figure.

        Not redrawn on white, unlike the MPS window's plots: what this
        panel draws on is a widefield photograph, and a figure of it is
        that photograph with the edges and the clusters on top.
        """
        # A figure written within the pause a drag opened would show the
        # mask from before the drag.
        self.flush()
        from tools import figure_export
        from tools.figure_export_ui import ask

        plots = {"Axon with the masks and the clusters": self.plot_image,
                 "Distance to the tubulin mask's edge": self.plot_hist}
        suggested = figure_export.suggested_name(
            str(self.inputs.movie.path), "axoplasm")
        request = ask(list(plots), suggested, parent=self, offer_white=False)
        if request is None:
            return
        plot = plots.get(request.plot)
        if plot is None:
            return
        try:
            written = figure_export.write(plot.getPlotItem(), request)
        except Exception as error:                        # noqa: BLE001
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Could not write the image:\n\n{error}")
            return
        QtWidgets.QMessageBox.information(
            self, "Image written",
            f"{figure_export.describe(request)}\n\n{written}")

    def _export_clicked(self) -> None:
        """
        Ask the main window to write this axon.

        This panel used to write three tables of its own, from its own
        state, while the results window wrote another from its own -- which
        is how one axon came to be described twice, differently. What this
        panel measures is now part of the axon's single row, and the
        export that writes it is the one place that sees both.
        """
        if self.export_callback is None:
            QtWidgets.QMessageBox.information(
                self, "Export axon",
                "This panel was opened on its own, so it cannot write the "
                "tables. Use 'Export axon' in the main window.")
            return
        self.export_callback()


def show_axoplasm_window(
    inputs: AxoplasmInputs,
    parent: Optional[QtWidgets.QWidget] = None,
    export_callback: Optional[Callable[[], Any]] = None,
) -> AxoplasmWindow:
    """Open the axoplasm panel."""
    window = AxoplasmWindow(inputs, parent=parent,
                            export_callback=export_callback)
    window.show()
    window.raise_()
    window.activateWindow()
    return window
