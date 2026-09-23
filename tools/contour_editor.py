# -*- coding: utf-8 -*-
"""
Drawing the contour by hand: a path along the membrane, and the clusters
joined in the order they fall along it.

The automatic contour joins the cluster centres by 2-opt, and on an axon
with centres off the membrane or a deep concavity it can run through the
wrong ones -- on axon 7, 6 of 94 centres sit deep inside the hull and the
contour comes out 1.79 times longer than the hull of the same centres.

The fix a person can make is not to reorder 94 centres one by one. It is
to say where the membrane goes: drag a closed path along it, and let the
program join the centres in the order they fall along that path. That is
what this box does, and the path -- not an order of cluster numbers -- is
what the analysis keeps, because a path still means something after a
change of eps, min samples or the slab changes which clusters there are,
and it orders the clusters the axoplasm discard leaves too.

The path starts ON THE CONTOUR THE PROGRAM MEASURED, with every centre
as a handle, so that using it without a single drag changes nothing:
the box can only move what a person moves. That was measured, not
assumed. The first version started on the convex hull of the centres --
a clean outline that looks like the obvious place to begin -- and on the
18 real April axons, applying it without a drag lengthened every contour,
by a median of 12 % and up to 49 %, with half the centres or more over
60 nm off it: in this data the membrane is not the hull. The automatic
contour's own vertices changed nothing in 18 of 18 and never crossed
themselves; the automatic contour resampled to 24 handles still moved
the perimeter by a median of 1.7 %.

That is also why the box cannot make a contour shorter. The automatic
one is 2-opt's shortest tour through every centre, and every centre has
to be visited, so another order is only ever longer. What a drawn path
changes is WHICH way round the centres are joined -- the thing that is
wrong when the shortest tour cuts across a concavity the membrane goes
round -- and the perimeter it gives is shown beside the current one
before anything is applied, with the centres the path does not run
through marked.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PyQt5 import QtCore, QtWidgets
from scipy.spatial import ConvexHull, QhullError

from tools import mps_plot_style as plot_style
from tools.mps_geometry import (
    GUIDE_FAR_NM,
    check_guide,
    order_along_guide,
)
from tools.mps_plot_style import AXIS_FG, style_dark

# The most handles the starting path gets. Enough to follow a peanut or a
# dent after a few drags, few enough that each one is worth dragging.
MAX_START_HANDLES = 24

# What the box answers.
APPLY, AUTOMATIC, CANCEL = "apply", "automatic", "cancel"


def starting_path(centroids: NDArray[np.float64],
                  n_max: int = MAX_START_HANDLES) -> NDArray[np.float64]:
    """
    A path for when there is no contour to start from: the convex hull of
    the centres, resampled to at most ``n_max`` points spread evenly.

    Only a fallback. When there is a contour the box starts on it instead,
    because on real axons the hull lengthened the contour by a median of
    12 % when applied untouched.

    The hull because it never crosses itself and is already the membrane
    of a round axon; evenly spread because a hull of 94 centres can have
    3 vertices on one flank and 20 on the other, and a handle is only
    useful where it can be dragged.
    """
    pts = np.asarray(centroids, dtype=float).reshape(-1, 2)
    try:
        hull = pts[ConvexHull(pts).vertices]
    except (QhullError, ValueError):
        # Too few or collinear centres: a small ring around their mean.
        centre = pts.mean(axis=0)
        radius = max(float(np.ptp(pts)), 100.0) / 2.0
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        return np.column_stack([centre[0] + radius * np.cos(angles),
                                centre[1] + radius * np.sin(angles)])
    closed = np.vstack([hull, hull[:1]])
    step = np.hypot(*np.diff(closed, axis=0).T)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    n = int(min(n_max, max(len(hull), 3)))
    targets = np.linspace(0.0, arc[-1], n, endpoint=False)
    x = np.interp(targets, arc, closed[:, 0])
    y = np.interp(targets, arc, closed[:, 1])
    return np.column_stack([x, y])


def tour_length_nm(points: NDArray[np.float64]) -> float:
    closed = np.vstack([points, points[:1]])
    return float(np.hypot(*np.diff(closed, axis=0).T).sum())


class ContourGuideDialog(QtWidgets.QDialog):
    """Drag a closed path along the membrane; the contour follows it."""

    def __init__(self, centroids: NDArray[np.float64], *,
                 current_contour: Optional[NDArray[np.float64]] = None,
                 current_guide: Optional[NDArray[np.float64]] = None,
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Draw the contour along the membrane")
        self.resize(760, 820)
        self.centroids = np.asarray(centroids, dtype=float).reshape(-1, 2)
        self.current_contour = (None if current_contour is None
                                else np.asarray(current_contour, dtype=float))
        self.answer: Tuple[str, Optional[NDArray[np.float64]]] = (CANCEL, None)
        lay = QtWidgets.QVBoxLayout(self)

        how = QtWidgets.QLabel(
            "The dashed path starts on the contour the program measured, so "
            "nothing changes until you move it. Drag its handles where the "
            "membrane goes; click a stretch of it to add a handle, "
            "right-click a handle to remove one. The clusters are joined in "
            "the order they fall along the path, and the contour that would "
            "be measured is drawn as you go. It can only be longer than the "
            "program's, which is already the shortest way through every "
            "centre: what a path changes is which way round they are "
            "joined.")
        how.setWordWrap(True)
        lay.addWidget(how)

        self.plot = pg.PlotWidget()
        style_dark(self.plot)
        self.plot.setAspectLocked(True)
        self.plot.setLabels(bottom="x [nm]", left="y [nm]")
        lay.addWidget(self.plot, stretch=1)

        if self.current_contour is not None and len(self.current_contour):
            closed = np.vstack([self.current_contour, self.current_contour[:1]])
            self.plot.addItem(pg.PlotDataItem(
                closed[:, 0], closed[:, 1],
                pen=pg.mkPen(plot_style.role("dim"), width=1,
                             style=QtCore.Qt.PenStyle.DotLine),
                name="contour now"))
        self.preview = pg.PlotDataItem(
            [], [], pen=pg.mkPen(plot_style.role("contour_kept"), width=2))
        self.plot.addItem(self.preview)
        self.dots = pg.ScatterPlotItem(
            self.centroids[:, 0], self.centroids[:, 1], size=8,
            brush=pg.mkBrush(plot_style.role("centroid")), pen=pg.mkPen(None))
        self.plot.addItem(self.dots)
        # The centres the path does not run through, drawn on top in the
        # warning colour AND as a hollow square: a mark that also reads in
        # grey and for someone who cannot separate the two hues.
        self.far = pg.ScatterPlotItem(
            [], [], size=14, symbol="s", brush=pg.mkBrush(None),
            pen=pg.mkPen(plot_style.verdict("warn"), width=2))
        self.plot.addItem(self.far)

        # What was drawn before, if anything; otherwise the contour the
        # program measured, every centre a handle, so that applying it
        # untouched changes nothing (see the module docstring for the
        # measurement that ruled out starting on the hull).
        if current_guide is not None:
            start = np.asarray(current_guide, dtype=float)
        elif (self.current_contour is not None
              and len(self.current_contour) >= 3):
            start = np.asarray(self.current_contour, dtype=float)
        else:
            start = starting_path(self.centroids)
        self.path = pg.PolyLineROI(
            [tuple(p) for p in start], closed=True,
            pen=pg.mkPen(plot_style.role("guide"), width=2,
                         style=QtCore.Qt.PenStyle.DashLine))
        self.plot.addItem(self.path)
        self.path.sigRegionChanged.connect(self._schedule_preview)

        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)
        lay.addWidget(self.label)

        buttons = QtWidgets.QDialogButtonBox()
        self.btn_apply = buttons.addButton(
            "Use this contour", QtWidgets.QDialogButtonBox.AcceptRole)
        self.btn_automatic = buttons.addButton(
            "Back to automatic", QtWidgets.QDialogButtonBox.ResetRole)
        self.btn_automatic.setToolTip(
            "Forget the drawn path and let the program join the centres "
            "again, by 2-opt from every start.")
        self.btn_automatic.setEnabled(current_guide is not None)
        self.btn_cancel = buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_automatic.clicked.connect(self._automatic)
        self.btn_cancel.clicked.connect(self.reject)
        lay.addWidget(buttons)

        # Dragging a handle emits many changes a second; the ordering is
        # cheap but the redraw is not free on 94 centres.
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self.update_preview)
        self.update_preview()

    # ------------------------------------------------------------- state
    def guide(self) -> NDArray[np.float64]:
        """The path as drawn, in nm, in the plot's own coordinates."""
        out = []
        for _, local in self.path.getLocalHandlePositions():
            p = self.path.mapToParent(local)
            out.append((float(p.x()), float(p.y())))
        return np.asarray(out, dtype=float)

    def _schedule_preview(self) -> None:
        self._timer.start()

    def update_preview(self) -> None:
        """Order the centres along the path and draw what would be measured."""
        path = self.guide()
        try:
            check_guide(path)
        except ValueError as error:
            self.preview.setData([], [])
            self.far.setData([], [])
            self.label.setText(str(error))
            self.label.setStyleSheet(
                f"color: {plot_style.verdict('bad', dark=False)};")
            self.btn_apply.setEnabled(False)
            return
        order, distance = order_along_guide(self.centroids, path)
        tour = self.centroids[order]
        closed = np.vstack([tour, tour[:1]])
        self.preview.setData(closed[:, 0], closed[:, 1])
        off = distance > GUIDE_FAR_NM
        self.far.setData(self.centroids[off, 0], self.centroids[off, 1])
        length = tour_length_nm(tour) / 1000.0
        text = f"Contour along this path: {length:.2f} um"
        if self.current_contour is not None and len(self.current_contour):
            text += (f" (now {tour_length_nm(self.current_contour) / 1000.0:.2f}"
                     f" um).")
        else:
            text += "."
        if off.any():
            text += (f" {int(off.sum())} of the {len(self.centroids)} centres "
                     f"(hollow squares) sit more than {GUIDE_FAR_NM:g} nm from "
                     f"the path: the contour still passes through them, and "
                     f"if they are not on the membrane the axoplasm panel is "
                     f"where they are discarded.")
        self.label.setText(text)
        self.label.setStyleSheet(
            f"color: {plot_style.verdict('warn' if off.any() else 'dim', dark=False)};")
        self.btn_apply.setEnabled(True)

    # ----------------------------------------------------------- answers
    def _apply(self) -> None:
        path = self.guide()
        try:
            check_guide(path)
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "Contour", str(error))
            return
        self.answer = (APPLY, path)
        self.accept()

    def _automatic(self) -> None:
        self.answer = (AUTOMATIC, None)
        self.accept()


def draw_contour(parent: Optional[QtWidgets.QWidget], centroids: NDArray[np.float64],
                 *, current_contour: Optional[NDArray[np.float64]] = None,
                 current_guide: Optional[NDArray[np.float64]] = None,
                 ) -> Tuple[str, Optional[NDArray[np.float64]]]:
    """Open the box; return (APPLY, path), (AUTOMATIC, None) or (CANCEL, None)."""
    dialog = ContourGuideDialog(centroids, current_contour=current_contour,
                                current_guide=current_guide, parent=parent)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return CANCEL, None
    return dialog.answer
