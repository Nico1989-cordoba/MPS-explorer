# -*- coding: utf-8 -*-
"""
Rings panel: every MPS segment of the loaded axon, its gaps and patches,
and the relationship between consecutive segments.

The per-axon panel (``mps_results_window``) analyses ONE axial slab, the
way Gazal et al. (2026) do. This one analyses every dominant axial
component and compares neighbours, which is what the thesis plan's item
(e) asks for.

What the panel is built around
------------------------------
Whether two segments are two rings at all is a property of the data, not
of the software, and on the 18-axon test set it usually fails: 19 of 33
boundaries have no interior minimum in the axial density, because the
fitted components (sigma 55-112 nm) are too broad for their ~190 nm
spacing. The segment table therefore leads with the boundary depth and
says "unresolved" in orange where there is no valley, so that a number
measured between two halves of one axial distribution can never be read
as a number measured between two rings.

The correlation is reported with the p-value of its rotation null, never
with a textbook Pearson p-value, for the reason given in ``mps_gaps``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from tools.mps_gaps import RingAnalysis, analyze_rings
from tools.mps_plot_style import PLOT_BG, set_title, style_dark

_C_ORANGE = "#d55e00"
_C_GREEN = "#009e73"
_C_BLUE = "#0072b2"
_C_GREY = "#888888"
_C_PURPLE = "#cc79a7"

# One colour per segment, reused by every plot in the panel so a segment
# is the same colour in the table, the tracks, the scatter and the
# histograms.
_SEGMENT_COLOURS = (_C_BLUE, _C_ORANGE, _C_GREEN, _C_PURPLE, "#56b4e9")


def _seg_colour(i: int) -> str:
    return _SEGMENT_COLOURS[i % len(_SEGMENT_COLOURS)]


class MPSRingsWindow(QtWidgets.QMainWindow):
    """Panel showing every axial segment, its gaps and patches, and the
    correlation between consecutive segments."""

    def __init__(
        self,
        ms: Any,
        rings: Optional[RingAnalysis] = None,
        rerun_callback: Optional[Callable[..., Any]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """
        Parameters
        ----------
        ms : a ``MultiSegmentAnalysis`` for the loaded axon.
        rings : its ``RingAnalysis``; computed here when not supplied.
        rerun_callback : called with (mode=..., guard_nm=...) when the user
            edits a control, and must return a fresh MultiSegmentAnalysis.
            None leaves the controls disabled and the panel read-only.
        """
        super().__init__(parent)
        self.ms = ms
        self.rings = rings if rings is not None else analyze_rings(ms)
        self.rerun_callback = rerun_callback
        # Target (x, y) range shared by the overlay and the small multiples,
        # re-applied whenever one of them is resized. See _build_spatial_tab.
        self._spatial_range: Optional[Tuple[Tuple[float, float],
                                            Tuple[float, float]]] = None
        self._spatial_viewboxes: List[Any] = []

        self.setWindowTitle("MPS analysis - rings and gap/patch correlation")
        self.resize(1320, 880)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.addWidget(self._build_controls())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        # Stretch factors only divide LEFTOVER space, and the tables ask for
        # far more than their share through their size hints -- left to it,
        # the plots open about 120 px wide. Set the split explicitly.
        splitter.setSizes([int(self.width() * 0.45), int(self.width() * 0.55)])
        root.addWidget(splitter, stretch=1)

        self.refresh()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox(
            "Axial segmentation (the analysis re-runs on change)")
        lay = QtWidgets.QHBoxLayout(box)

        lay.addWidget(QtWidgets.QLabel("Slab boundaries:"))
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItem("valley - cut at the density minimum", "valley")
        self.combo_mode.addItem("paper - 180 nm around each peak", "paper")
        self.combo_mode.addItem("partition - cut at the midpoint", "partition")
        self.combo_mode.setMinimumWidth(270)
        self.combo_mode.setToolTip(
            "How each segment's axial slab is bounded.\n\n"
            "valley: clip where the fitted axial density is lowest, so no\n"
            "  localization belongs to two segments. Preferred for\n"
            "  comparing neighbours.\n"
            "paper: the paper's 180 nm window around every peak. Slabs\n"
            "  OVERLAP when the periodicity is shorter than 180 nm, and\n"
            "  shared localizations inflate any similarity between them.\n"
            "partition: cut at the midpoint between peaks, which assumes a\n"
            "  symmetry the two rings do not have. Kept for comparison."
        )
        lay.addWidget(self.combo_mode)

        lay.addSpacing(12)
        lay.addWidget(QtWidgets.QLabel("Guard band [nm]:"))
        self.spin_guard = QtWidgets.QDoubleSpinBox()
        self.spin_guard.setRange(0.0, 400.0)
        self.spin_guard.setDecimals(0)
        self.spin_guard.setSingleStep(10.0)
        self.spin_guard.setToolTip(
            "A dead zone centred on each boundary, excluded from both\n"
            "neighbouring slabs. A control, not a better segmentation:\n"
            "axial precision (~50-80 nm) is comparable to the slab, so one\n"
            "ring can be detected in both. If the correlation survives a\n"
            "wide guard band, it is not axial bleed-through.\n"
            "Ignored in 'paper' mode, which has no boundary."
        )
        lay.addWidget(self.spin_guard)

        lay.addStretch(1)

        self.btn_export = QtWidgets.QPushButton("Export CSV...")
        self.btn_export.setToolTip(
            "Write one row per segment and one row per segment pair.")
        self.btn_export.clicked.connect(self._on_export)
        lay.addWidget(self.btn_export)

        enabled = self.rerun_callback is not None
        for w in (self.combo_mode, self.spin_guard):
            w.setEnabled(enabled)
        self.combo_mode.currentIndexChanged.connect(self._on_param_changed)
        self.spin_guard.valueChanged.connect(self._on_param_changed)
        return box

    def _build_left_column(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        self.lbl_header = QtWidgets.QLabel()
        self.lbl_header.setWordWrap(True)
        self.lbl_header.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.lbl_header)

        lay.addWidget(QtWidgets.QLabel("Segments"))
        self.table_seg = QtWidgets.QTableWidget(0, 8)
        self.table_seg.setHorizontalHeaderLabels([
            "Seg", "z centre", "slab", "locs", "clusters",
            "occupancy", "patch", "gap"])
        self._prepare_table(self.table_seg)
        lay.addWidget(self.table_seg, stretch=2)

        lay.addWidget(QtWidgets.QLabel(
            "Consecutive segment pairs (r at zero rotation, p from the "
            "rotation null)"))
        self.table_pair = QtWidgets.QTableWidget(0, 6)
        self.table_pair.setHorizontalHeaderLabels([
            "Pair", "dz", "boundary", "r(0)", "p", "z vs null"])
        self._prepare_table(self.table_pair)
        self.table_pair.itemSelectionChanged.connect(self._draw_correlation)
        lay.addWidget(self.table_pair, stretch=2)

        lay.addWidget(QtWidgets.QLabel("Warnings"))
        self.list_warnings = QtWidgets.QListWidget()
        self.list_warnings.setWordWrap(True)
        lay.addWidget(self.list_warnings, stretch=2)
        return w

    @staticmethod
    def _prepare_table(table: QtWidgets.QTableWidget) -> None:
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)
        # Not stretching the last section: these tables are narrow, and
        # stretching turns the final column into a band of empty space
        # wider than every other column put together.
        table.horizontalHeader().setStretchLastSection(False)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)

    def _build_plots(self) -> QtWidgets.QWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_axial_tab(), "Axial + correlation")
        tabs.addTab(self._build_spatial_tab(), "Spatial (x,y)")
        tabs.addTab(self._build_zhist_tab(), "Z histograms per ring")
        return tabs

    def _build_axial_tab(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(0, 0, 0, 0)

        self.plot_z = pg.PlotWidget()
        style_dark(self.plot_z)
        set_title(self.plot_z,
                   "Axial distribution: components, slabs and boundaries")
        self.plot_z.setLabels(bottom="z [nm]", left="density")
        grid.addWidget(self.plot_z, 0, 0)

        self.plot_profiles = pg.PlotWidget()
        style_dark(self.plot_profiles)
        set_title(self.plot_profiles,
                   "Patches around the perimeter, one track per segment "
                   "(filled = covered by spectrin)")
        self.plot_profiles.setLabels(
            bottom="angle about the axon centre [deg]", left="segment")
        self.plot_profiles.setXRange(0, 360)
        grid.addWidget(self.plot_profiles, 1, 0)

        self.plot_corr = pg.PlotWidget()
        style_dark(self.plot_corr)
        set_title(self.plot_corr,
                   "Cross-correlation of the selected pair, over all rotations")
        self.plot_corr.setLabels(bottom="rotation [deg]", left="r")
        # A correlation is already dimensionless and of order 0.1; the
        # automatic SI prefix relabels the axis "r (x0.001)" and prints
        # 0.2 as 200, which invites reading the effect as 1000x its size.
        self.plot_corr.getAxis("left").enableAutoSIPrefix(False)
        grid.addWidget(self.plot_corr, 2, 0)

        grid.setRowStretch(0, 2)
        grid.setRowStretch(1, 3)
        grid.setRowStretch(2, 2)
        return w

    def _build_spatial_tab(self) -> QtWidgets.QWidget:
        """Real (x, y) localizations of every segment: superimposed, to see
        directly whether patches at the same angle really sit at the same
        physical spot, and individually, on the SAME range so the
        superimposed view and the small multiples are one consistent
        picture rather than independently zoomed crops."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        self.plot_overlay = pg.PlotWidget()
        style_dark(self.plot_overlay)
        set_title(self.plot_overlay,
                   "Every segment's localizations, superimposed")
        self.plot_overlay.setAspectLocked(True)
        self.plot_overlay.setLabels(bottom="x [nm]", left="y [nm]")
        # An aspect-locked view keeps nm-per-pixel fixed across a resize,
        # which means it EXPANDS the visible range as the widget grows. The
        # range is set while these widgets are still at their pre-layout
        # size, so by the time the window is shown every spatial plot is
        # about twice as zoomed out as asked, with the axon adrift in empty
        # space. Re-applying the target range on resize is what keeps the
        # fit tight; panning and zooming do not emit this signal, so it
        # does not fight the user.
        self.plot_overlay.getViewBox().sigResized.connect(
            self._reapply_spatial_range)
        lay.addWidget(self.plot_overlay, stretch=3)

        lay.addWidget(QtWidgets.QLabel(
            "Each segment on its own, in nm, same x/y range as above "
            "(linked pan/zoom)"))
        self.spatial_grid = pg.GraphicsLayoutWidget()
        self.spatial_grid.setBackground(PLOT_BG)
        lay.addWidget(self.spatial_grid, stretch=2)
        return w

    def _build_zhist_tab(self) -> QtWidgets.QWidget:
        """One Z histogram per segment, on a shared axial axis so their
        relative position along the axon is visible, unlike the pooled
        histogram in the axial tab which cannot show one segment's own
        internal shape separately from the others."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel(
            "Axial (z) distribution of each segment's own slab, dashed "
            "lines mark its boundaries"))
        self.zhist_grid = pg.GraphicsLayoutWidget()
        self.zhist_grid.setBackground(PLOT_BG)
        lay.addWidget(self.zhist_grid)
        return w

    # ------------------------------------------------------------------
    # refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Redraw everything from ``self.ms`` and ``self.rings``."""
        self._sync_controls()
        self._fill_header()
        self._fill_segment_table()
        self._fill_pair_table()
        self._fill_warnings()
        self._draw_z()
        self._draw_profiles()
        self._draw_correlation()
        self._draw_spatial()
        self._draw_zhist()

    def _sync_controls(self) -> None:
        for w in (self.combo_mode, self.spin_guard):
            w.blockSignals(True)
        i = self.combo_mode.findData(getattr(self.ms, "mode", "valley"))
        if i >= 0:
            self.combo_mode.setCurrentIndex(i)
        self.spin_guard.setValue(float(getattr(self, "_guard_nm", 0.0)))
        for w in (self.combo_mode, self.spin_guard):
            w.blockSignals(False)

    def _fill_header(self) -> None:
        ms = self.ms
        n_res = 0
        if ms.valleys is not None:
            n_res = int(np.count_nonzero(ms.valleys.is_true_valley))
        n_bounds = 0 if ms.valleys is None else ms.valleys.n_boundaries
        dz = ms.z_result.mean_delta_z_nm
        self.lbl_header.setText(
            f"<b>{os.path.basename(str(ms.source_name)) or 'axon'}</b> &mdash; "
            f"{ms.n_segments} segment(s), {ms.n_analyzed} analysed, "
            f"{len(self.rings.pairs)} pair(s) compared. "
            f"Mean &Delta;Z = {'n/a' if dz is None else f'{dz:.0f} nm'}. "
            f"Axially resolved boundaries: <b>{n_res} of {n_bounds}</b> "
            f"(a boundary with no density minimum means the two segments "
            f"are two halves of one axial distribution, not two rings)."
        )

    def _fill_segment_table(self) -> None:
        t = self.table_seg
        t.setRowCount(0)
        for k, seg in enumerate(self.ms.segments):
            an = self.ms.analyses[k] if k < len(self.ms.analyses) else None
            run = self.rings.runs[k] if k < len(self.rings.runs) else None
            row = t.rowCount()
            t.insertRow(row)

            def put(col: int, text: str, colour: Optional[str] = None) -> None:
                item = QtWidgets.QTableWidgetItem(text)
                if colour:
                    item.setForeground(QtGui.QColor(colour))
                t.setItem(row, col, item)

            put(0, str(seg.index), _seg_colour(k))
            put(1, f"{seg.center_nm:,.0f}")
            put(2, f"{seg.zmin_nm:,.0f} .. {seg.zmax_nm:,.0f}")
            put(3, f"{seg.n_locs:,}")
            put(4, "n/a" if an is None else f"{an.n_clusters_kept:,}")
            put(5, "n/a" if an is None or an.occupancy_percent is None
                else f"{an.occupancy_percent:.1f} %")
            put(6, "n/a" if run is None or run.median_patch_nm is None
                else f"{run.median_patch_nm:,.0f} nm")
            put(7, "n/a" if run is None or run.median_gap_nm is None
                else f"{run.median_gap_nm:,.0f} nm")

    def _fill_pair_table(self) -> None:
        t = self.table_pair
        t.setRowCount(0)
        for pair in self.rings.pairs:
            c = pair.correlation
            row = t.rowCount()
            t.insertRow(row)

            def put(col: int, text: str, colour: Optional[str] = None,
                    italic: bool = False) -> None:
                item = QtWidgets.QTableWidgetItem(text)
                if colour:
                    item.setForeground(QtGui.QColor(colour))
                if italic:
                    f = item.font()
                    f.setItalic(True)
                    item.setFont(f)
                t.setItem(row, col, item)

            depth = pair.boundary_relative_depth
            if depth is None:
                boundary, colour, italic = "no shared boundary", _C_GREY, True
            elif not pair.boundary_is_true_valley:
                boundary, colour, italic = "unresolved", _C_ORANGE, True
            else:
                boundary = f"valley, depth {depth:.2f}"
                colour = _C_GREEN if depth >= 0.10 else _C_ORANGE
                italic = depth < 0.10

            put(0, f"{pair.index_a}-{pair.index_b}")
            put(1, f"{pair.delta_z_nm:,.0f} nm")
            put(2, boundary, colour, italic)
            put(3, f"{c.r_at_zero:+.3f}")
            put(4, f"{c.p_rotation:.4f}",
                _C_GREEN if c.p_rotation < 0.05 else None)
            put(5, "n/a" if c.z_vs_null is None else f"{c.z_vs_null:+.2f}")

        if t.rowCount():
            t.selectRow(0)

    def _fill_warnings(self) -> None:
        self.list_warnings.clear()
        msgs: List[str] = list(self.ms.warnings) + list(self.rings.warnings)
        if not msgs:
            item = QtWidgets.QListWidgetItem("No warnings.")
            item.setForeground(QtGui.QColor(_C_GREY))
            self.list_warnings.addItem(item)
            return
        for m in msgs:
            item = QtWidgets.QListWidgetItem(m)
            item.setForeground(QtGui.QColor(_C_ORANGE))
            self.list_warnings.addItem(item)

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------

    def _draw_z(self) -> None:
        self.plot_z.clear()
        ms = self.ms
        zr = ms.z_result

        zs = [a.z_slab for a in ms.analyses if a is not None and a.z_slab.size]
        if zs:
            allz = np.concatenate(zs)
            counts, edges = np.histogram(allz, bins=80, density=True)
            centres = (edges[:-1] + edges[1:]) / 2
            self.plot_z.addItem(pg.BarGraphItem(
                x=centres, height=counts, width=float(np.mean(np.diff(edges))),
                brush=pg.mkBrush(136, 136, 136, 80), pen=None))

            grid = np.linspace(allz.min(), allz.max(), 1024)
            dens = zr.mixture_density(grid)
            if np.any(dens > 0):
                self.plot_z.addItem(pg.PlotDataItem(
                    grid, dens, pen=pg.mkPen(_C_GREY, width=2)))

        for k, seg in enumerate(ms.segments):
            region = pg.LinearRegionItem(
                values=(seg.zmin_nm, seg.zmax_nm), movable=False)
            col = QtGui.QColor(_seg_colour(k))
            col.setAlpha(55)
            region.setBrush(pg.mkBrush(col))
            region.setZValue(-10)
            self.plot_z.addItem(region)
            self.plot_z.addItem(pg.InfiniteLine(
                pos=seg.center_nm, angle=90,
                pen=pg.mkPen(_seg_colour(k), width=2,
                             style=QtCore.Qt.DotLine)))

        if ms.valleys is not None:
            for pos, real, depth in zip(ms.valleys.positions_nm,
                                        ms.valleys.is_true_valley,
                                        ms.valleys.relative_depth):
                self.plot_z.addItem(pg.InfiniteLine(
                    pos=float(pos), angle=90,
                    pen=pg.mkPen(
                        _C_GREEN if real else _C_ORANGE, width=2,
                        style=QtCore.Qt.SolidLine if real
                        else QtCore.Qt.DashLine),
                    label=(f"valley {depth:.2f}" if real else "no valley"),
                    labelOpts={"position": 0.08,
                               "color": _C_GREEN if real else _C_ORANGE}))

    def _draw_profiles(self) -> None:
        """One filled track per segment, stacked.

        Overlaying the profiles as lines is unreadable: coverage is
        effectively binary (a ~60 nm patch spans ~14 bins of the 0.1 deg
        grid), so three traces of 3,600 points become a forest of vertical
        strokes. Stacked tracks answer the question the plot exists for --
        are the patches of consecutive segments at the same angles? -- by
        making it a matter of reading down a column.
        """
        self.plot_profiles.clear()
        ticks: List[Any] = []
        drawn = 0
        for k, prof in enumerate(self.rings.profiles):
            if prof is None:
                continue
            colour = QtGui.QColor(_seg_colour(k))
            fill = QtGui.QColor(colour)
            fill.setAlpha(170)
            # 0.82 leaves a gap between tracks so a full patch in one
            # segment cannot be mistaken for part of the next track.
            self.plot_profiles.plot(
                prof.theta_deg, prof.coverage * 0.82 + k,
                pen=pg.mkPen(colour, width=1),
                fillLevel=float(k), brush=pg.mkBrush(fill))
            ticks.append((k + 0.41, str(self.ms.segments[k].index)))
            drawn += 1

        if not drawn:
            return
        axis = self.plot_profiles.getAxis("left")
        axis.setTicks([ticks])
        self.plot_profiles.setYRange(-0.1, drawn + 0.05)

    def _draw_correlation(self) -> None:
        self.plot_corr.clear()
        rows = {i.row() for i in self.table_pair.selectedIndexes()}
        if not rows:
            return
        k = min(rows)
        if k >= len(self.rings.pairs):
            return
        pair = self.rings.pairs[k]
        c = pair.correlation
        if c.offsets_deg.size == 0:
            return

        self.plot_corr.plot(c.offsets_deg, c.correlation,
                            pen=pg.mkPen(_C_GREY, width=1))
        # The rotation null IS this whole curve, so showing its spread next
        # to r(0) is what makes the p-value legible.
        for lvl, col in ((c.null_mean, _C_GREY),
                         (c.null_mean + 2 * c.null_sd, _C_BLUE),
                         (c.null_mean - 2 * c.null_sd, _C_BLUE)):
            self.plot_corr.addItem(pg.InfiniteLine(
                pos=float(lvl), angle=0,
                pen=pg.mkPen(col, width=1, style=QtCore.Qt.DotLine)))
        self.plot_corr.addItem(pg.ScatterPlotItem(
            [0.0], [c.r_at_zero], size=11,
            brush=pg.mkBrush(_C_ORANGE), pen=None))
        self.plot_corr.addItem(pg.InfiniteLine(
            pos=0.0, angle=90,
            pen=pg.mkPen(_C_ORANGE, width=2),
            label=(f"r(0) = {c.r_at_zero:+.3f},  p = {c.p_rotation:.4f}"),
            labelOpts={"position": 0.92, "color": _C_ORANGE}))
        set_title(
            self.plot_corr,
            f"Segments {pair.index_a}-{pair.index_b}: cross-correlation over "
            f"all {c.n_rotations:,} rotations (dotted: null mean +/- 2 SD)")

    def _segments_with_locs(self, field: str) -> List[Any]:
        """(k, segment, analysis) for every segment whose ``field`` (a
        localization array on AxonAnalysis, e.g. "x_slab") is non-empty."""
        out = []
        for k, (seg, an) in enumerate(zip(self.ms.segments, self.ms.analyses)):
            if an is not None and getattr(an, field).size:
                out.append((k, seg, an))
        return out

    def _reapply_spatial_range(self, *_args: Any) -> None:
        """Put every spatial view back on the shared range. Called on each
        resize because aspect-locking rescales the range with the widget."""
        if self._spatial_range is None:
            return
        xr, yr = self._spatial_range
        for vb in self._spatial_viewboxes:
            vb.setRange(xRange=xr, yRange=yr, padding=0)

    def _draw_spatial(self) -> None:
        self.plot_overlay.clear()
        self.spatial_grid.clear()
        self._spatial_range = None
        self._spatial_viewboxes = []

        rows = self._segments_with_locs("x_slab")
        if not rows:
            return

        # One shared bounding box for the overlay AND every small multiple.
        # Without it, each subplot auto-ranges to its own data and a patch
        # that looks the same size in two segments could actually be at two
        # different physical scales -- the plots would agree with each
        # other by construction, which defeats the point of comparing them.
        all_x = np.concatenate([an.x_slab for _, _, an in rows])
        all_y = np.concatenate([an.y_slab for _, _, an in rows])
        pad_x = 0.05 * max(float(all_x.max() - all_x.min()), 1.0)
        pad_y = 0.05 * max(float(all_y.max() - all_y.min()), 1.0)
        xr = (float(all_x.min() - pad_x), float(all_x.max() + pad_x))
        yr = (float(all_y.min() - pad_y), float(all_y.max() + pad_y))

        self._spatial_range = (xr, yr)
        self._spatial_viewboxes = [self.plot_overlay.getViewBox()]

        for k, seg, an in rows:
            self.plot_overlay.addItem(pg.ScatterPlotItem(
                an.x_slab, an.y_slab, pen=pg.mkPen(_seg_colour(k), width=1),
                brush=None, size=4))

        # setXLink/setYLink only sync FUTURE range changes (they fire off
        # the linked view's sigRangeChanged), not the range already in
        # place when the link is made -- verified directly: a subplot
        # linked to one whose range never changes again afterward is left
        # at pg's default [0,1], not the target's actual range. The range
        # is therefore set explicitly on every subplot; the links stay so
        # an interactive pan/zoom in one still moves the rest together.
        #
        # Aspect-locking then independently recomputes one axis from each
        # widget's own pixel aspect ratio, so the displayed range can still
        # differ slightly (a percent or so) between columns of different
        # pixel width. No per-subplot y-axis label is set, since that title
        # is what was making column 0 measurably narrower than the rest.
        first: Optional[Any] = None
        for i, (k, seg, an) in enumerate(rows):
            p = self.spatial_grid.addPlot(row=0, col=i)
            style_dark(p)
            set_title(p, f"segment {seg.index}")
            p.setAspectLocked(True)
            p.addItem(pg.ScatterPlotItem(
                an.x_slab, an.y_slab, pen=pg.mkPen(_seg_colour(k), width=1),
                brush=None, size=3))
            p.setLabels(bottom="x [nm]")
            self._spatial_viewboxes.append(p.vb)
            # These view boxes are rebuilt on every redraw, so connecting
            # here cannot accumulate handlers the way the overlay's would.
            p.vb.sigResized.connect(self._reapply_spatial_range)
            if first is None:
                first = p
            else:
                p.setXLink(first)
                p.setYLink(first)

        self._reapply_spatial_range()

    def _draw_zhist(self) -> None:
        self.zhist_grid.clear()

        rows = self._segments_with_locs("z_slab")
        if not rows:
            return

        # Shared range from the SLAB BOUNDS, not the data extent: locs
        # thin out near the edges of a slab, so ranging on the data would
        # crop each subplot to a different window and hide exactly the
        # relative axial position the shared axis exists to show.
        lo = min(seg.zmin_nm for _, seg, _ in rows)
        hi = max(seg.zmax_nm for _, seg, _ in rows)
        pad = 0.05 * max(hi - lo, 1.0)
        zr = (lo - pad, hi + pad)

        first: Optional[Any] = None
        for i, (k, seg, an) in enumerate(rows):
            colour = _seg_colour(k)
            p = self.zhist_grid.addPlot(row=0, col=i)
            style_dark(p)
            set_title(p, f"segment {seg.index}")
            counts, edges = np.histogram(an.z_slab, bins=40)
            centres = (edges[:-1] + edges[1:]) / 2
            width = float(np.mean(np.diff(edges))) if edges.size > 1 else 1.0
            fill = QtGui.QColor(colour)
            fill.setAlpha(170)
            p.addItem(pg.BarGraphItem(
                x=centres, height=counts, width=width,
                brush=pg.mkBrush(fill), pen=None))
            for bound in (seg.zmin_nm, seg.zmax_nm):
                p.addItem(pg.InfiniteLine(
                    pos=float(bound), angle=90,
                    pen=pg.mkPen(colour, width=1, style=QtCore.Qt.DashLine)))
            p.setLabels(bottom="z [nm]", left="count" if i == 0 else "")
            p.setXRange(*zr, padding=0)
            if first is None:
                first = p
            else:
                p.setXLink(first)

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------

    def _on_param_changed(self) -> None:
        if self.rerun_callback is None:
            return
        mode = self.combo_mode.currentData()
        guard = float(self.spin_guard.value())
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            ms = self.rerun_callback(mode=mode, guard_nm=guard)
        except Exception as exc:                          # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Ring analysis failed", f"{exc}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        if ms is None:
            return
        self.ms = ms
        self._guard_nm = guard
        self.rings = analyze_rings(ms)
        self.refresh()

    def _on_export(self) -> None:
        """Write one row per segment and one row per segment pair."""
        base = os.path.splitext(os.path.basename(
            str(self.ms.source_name)))[0] or "axon"
        default = f"{base}_mps_rings.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export ring analysis", default, "CSV Files (*.csv)")
        if not path:
            return

        seg_rows = self.ms.export_rows()
        for k, run in enumerate(self.rings.runs):
            if run is None or k >= len(seg_rows):
                continue
            seg_rows[k].update(run.export_dict())
        pair_rows = [p.export_dict() for p in self.rings.pairs]

        # Two shapes of row go to two files rather than one ragged table: a
        # per-segment row and a per-pair row are different observations, and
        # mixing them is what makes a CSV impossible to load later.
        stem, ext = os.path.splitext(path)
        pair_path = f"{stem}_pairs{ext or '.csv'}"
        try:
            written = [self._write_csv(path, seg_rows)]
            if pair_rows:
                written.append(self._write_csv(pair_path, pair_rows))
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not write:\n\n{exc}")
            return

        QtWidgets.QMessageBox.information(
            self, "Exported", "\n".join(w for w in written if w))

    @staticmethod
    def _write_csv(path: str, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        fields = list(rows[0].keys())
        existing: Optional[List[str]] = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", newline="") as f:
                existing = next(csv.reader(f), None)
        append = existing == fields
        with open(path, "a" if append else "w",
                  encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields,
                                    extrasaction="ignore")
            if not append:
                writer.writeheader()
            writer.writerows(rows)
        return (f"{'Appended' if append else 'Wrote'} {len(rows)} row(s) to "
                f"{path}")
