# -*- coding: utf-8 -*-
"""
Each clustering method's own diagnostic plot, where one is already computed or cheap (F5, the user's request of
2026-10-22: "check that every method has its plot").

The map draws every method's clusters (tools.method_overlay.MethodLayers). Some methods also have a plot of their
own that says how the clusters came about; this module offers the ones that cost no new clustering, or one cheap
fit, and says for the others why there is none:

* DBSCAN -- the k-distance graph (each localization's distance to its (min_samples - 1)-th neighbour, sorted), with
  eps as a horizontal line: the localizations above it are not core points [lit: Ester et al. 1996; the same graph
  as the k-distance dialog, tools.mps_kdistance]. Cost: one k-d tree query on the slab.
* voronoi_factor / voronoi_csr -- the histogram of the cells' log density, with the threshold the method applied
  [lit: Levet et al. 2015, SR-Tesseler]. Cost: none; the densities are the fit's own (cluster_diagnostics).
* ToMATo -- the persistence diagram (birth against death density of every peak) with the line ``birth - death =
  tau``: the peaks between it and the diagonal were merged [lit: Chazal et al. 2013; Pike et al. 2020]. Cost: one
  ToMATo fit on the slab with the analysis' own parameters (the analysis does not keep its diagram).

Not offered, and why (NOT_OFFERED): OPTICS (its reachability plot needs OPTICS to run again, the slow method), the
two HDBSCAN (the condensed tree is not kept by the analysis and would need a refit with the hdbscan package), FOCAL
(no standard diagnostic plot), GraphCommunity (a stand-in for StormGraph: its degree plot would describe the stand-in),
External labels (the labels come from outside), Emitters (its diagnostics are its own map layer and the xi checks in
the analysis notes).

``diagnostic_for(method, analysis)`` is pure (no Qt); ``MethodDiagnosticsDialog`` shows one tab per method, each with
"Export image..." (tools.figure_export: PNG at 300-1200 dpi or SVG, on white, never over an existing file).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets

NOT_OFFERED: Dict[str, str] = {
    "OPTICS": "its reachability plot needs OPTICS to run again on the slab, and OPTICS is the slow method",
    "HDBSCAN": "the condensed tree is not kept by the analysis; drawing it would mean a refit with the hdbscan "
               "package",
    "HDBSCAN (scikit-learn)": "scikit-learn's HDBSCAN does not expose its condensed tree",
    "FOCAL": "FOCAL has no standard diagnostic plot",
    "GraphCommunity": "it is a stand-in for StormGraph (see its documentation): a plot of its degrees would describe "
                      "the stand-in, not StormGraph",
    "External labels": "the labels come from outside the program: there is nothing of its own to diagnose",
    "emitters_bagol_like": "its diagnostics are its own map layer (the Emitters layer) and the xi checks in the "
                           "analysis notes",
}
# Bins of the Voronoi density histogram [ad hoc: about the square root of a few thousand cells, coarse enough to read].
VORONOI_BINS = 60
PER_UM2 = 1.0e6          # nm^-2 -> um^-2


@dataclass
class Diagnostic:
    """One plot: curves / points, reference lines, axis labels and a caption."""
    method: str
    title: str
    x_label: str
    y_label: str
    kind: str                                   # "curve" | "histogram" | "points"
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hlines: List[Tuple[float, str]] = field(default_factory=list)
    vlines: List[Tuple[float, str]] = field(default_factory=list)
    diagonal: Optional[Tuple[float, str]] = None    # y = x - offset, with its words (ToMATo's tau line)
    caption: str = ""
    log_y: bool = False


def _dbscan(an: Any) -> Diagnostic:
    from tools import mps_kdistance as K
    m = int(getattr(an, "min_samples", 10))
    eps = float(getattr(an, "eps_nm", 25.0))
    kd = K.k_distances(an.x_slab, an.y_slab, m)
    curve = K.sorted_curve(kd)
    above = float(np.mean(kd > eps)) if kd.size else float("nan")
    return Diagnostic(
        method="DBSCAN", title=f"DBSCAN: k-distance graph (k = {m - 1}, min samples {m})",
        x_label="localizations, sorted", y_label=f"distance to the {m - 1}th neighbour [nm]", kind="curve",
        x=np.arange(curve.size, dtype=float), y=curve, hlines=[(eps, f"eps {eps:g} nm")],
        caption=(f"{curve.size:,} localizations of the slab; {100 * above:.1f} % lie above eps (not core points). "
                 f"The same graph as the k-distance dialog (docs/kdistance.md)."))


def _voronoi(an: Any) -> Optional[Diagnostic]:
    diag = getattr(an, "cluster_diagnostics", None)
    if diag is None:
        return None
    d = np.asarray(diag.density, dtype=float) * PER_UM2
    d = d[np.isfinite(d) & (d > 0)]
    if not d.size:
        return None
    logd = np.log10(d)
    counts, edges = np.histogram(logd, bins=VORONOI_BINS)
    thr = float(diag.threshold_density) * PER_UM2
    mean = float(diag.mean_density) * PER_UM2
    v: List[Tuple[float, str]] = []
    if np.isfinite(thr) and thr > 0:
        v.append((float(np.log10(thr)), f"threshold {thr:,.0f} loc/um^2 ({diag.threshold_rule or diag.threshold_mode})"))
    if np.isfinite(mean) and mean > 0:
        v.append((float(np.log10(mean)), f"mean {mean:,.0f}"))
    method = str(an.cluster_method)
    return Diagnostic(
        method=method, title=f"{method}: density of the Voronoi cells",
        x_label="log10 density [loc/um^2]", y_label="localizations", kind="histogram",
        x=edges, y=counts.astype(float), vlines=v,
        caption=(f"{d.size:,} cells, the fit's own densities (nothing is clustered again); the cells right of the "
                 f"threshold are dense, and dense cells that touch form the clusters."))


def _tomato(an: Any) -> Optional[Diagnostic]:
    from tools.clustering_strategies import make_strategy
    strategy = make_strategy("ToMATo", dict(getattr(an, "cluster_params", {}) or {}))
    pts = np.column_stack([np.asarray(an.x_slab, float), np.asarray(an.y_slab, float)])
    strategy.fit(pts)
    pairs = getattr(strategy, "persistence_pairs_", None)
    if pairs is None or not len(pairs):
        return None
    pairs = np.asarray(pairs, dtype=float)
    finite = np.isfinite(pairs).all(axis=1)
    tau = float(getattr(strategy, "tau_used_", 0.0))
    return Diagnostic(
        method="ToMATo", title="ToMATo: persistence diagram", x_label="birth density (peak)",
        y_label="death density (merge)", kind="points", x=pairs[finite, 0], y=pairs[finite, 1],
        diagonal=(tau, f"birth - death = tau ({tau:.3g})"),
        caption=(f"{int(finite.sum())} peaks of the density. A peak between the diagonal (thin) and the dashed line "
                 f"has a prominence (birth - death) below tau and was merged into a higher one; a peak below the "
                 f"dashed line is a cluster. One ToMATo fit on the slab with the analysis' parameters."))


def diagnostic_for(method: str, analysis: Any) -> Tuple[Optional[Diagnostic], str]:
    """(the plot, '') or (None, why there is none) for ``method``'s run ``analysis``."""
    name = str(method)
    if name in NOT_OFFERED:
        return None, NOT_OFFERED[name]
    if analysis is None or getattr(analysis, "x_slab", None) is None:
        return None, "the run has no localizations"
    try:
        if name == "DBSCAN":
            return _dbscan(analysis), ""
        if name in ("voronoi_factor", "voronoi_csr"):
            d = _voronoi(analysis)
            return (d, "") if d is not None else (None, "the run kept no tessellation (re-run it)")
        if name == "ToMATo":
            d = _tomato(analysis)
            return (d, "") if d is not None else (None, "no density peak in the slab")
    except Exception as exc:                                    # noqa: BLE001 - said in the tab
        return None, f"the plot could not be computed: {exc}"
    return None, "no diagnostic plot is offered for this method"


# ---------------------------------------------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------------------------------------------
# What the main series of each kind of diagnostic is, in the legend.
MAIN_WORDS = {"curve": "This axon (sorted k-distances)", "histogram": "This axon's cells (density histogram)",
              "points": "This axon's density peaks (persistence)"}


def draw(plot: Any, d: Diagnostic, dark: bool = True) -> None:
    """Draw ``d`` on a pyqtgraph PlotWidget (cleared first)."""
    import pyqtgraph as pg
    from tools.mps_plot_style import neutral, role, style_dark, style_light
    (style_dark if dark else style_light)(plot)
    plot.clear()
    pi = plot.getPlotItem()
    pi.setTitle(d.title)
    pi.setLabel("bottom", d.x_label)
    pi.setLabel("left", d.y_label)
    main = role("locs")
    if d.kind == "curve":
        plot.plot(d.x, d.y, pen=pg.mkPen(main, width=2))
    elif d.kind == "histogram":
        plot.plot(d.x, d.y, stepMode="center", fillLevel=0, brush=pg.mkBrush(86, 180, 233, 120),
                  pen=pg.mkPen(main, width=1.5))
    else:
        plot.plot(d.x, d.y, pen=None, symbol="o", symbolSize=5, symbolBrush=pg.mkBrush(main), symbolPen=None)
    ref = role("occupied")
    for y, words in d.hlines:
        plot.addItem(pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(ref, width=2, style=2), label=words,
                                     labelOpts={"position": 0.15, "color": ref}))
    for k, (x, words) in enumerate(d.vlines):
        colour = ref if k == 0 else neutral(dark)
        plot.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(colour, width=2, style=2), label=words,
                                     labelOpts={"position": 0.9 - 0.12 * k, "color": colour}))
    # Manual test of 2026-10-23: every new plot carries a legend that names each element (top right, on a
    # semi-transparent box, in the theme's colours).
    previous = getattr(plot, "legend_box", None)
    if previous is not None and previous.scene() is not None:
        previous.scene().removeItem(previous)
    text = "#ffffff" if dark else "#000000"
    legend = pg.LegendItem(offset=(-4, 4), labelTextColor=text,
                           brush=pg.mkBrush(*((0, 0, 0, 150) if dark else (255, 255, 255, 190))),
                           pen=pg.mkPen(text, width=0.5), labelTextSize="7pt", verSpacing=-6)
    legend.setParentItem(pi.vb)
    legend.setZValue(1e6)
    if d.kind == "points":
        legend.addItem(pg.ScatterPlotItem(symbol="o", size=6, brush=pg.mkBrush(main), pen=None), MAIN_WORDS[d.kind])
    else:
        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(main, width=2)), MAIN_WORDS.get(d.kind, "This axon"))
    for _v, words in d.hlines:
        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(ref, width=2, style=2)), words)
    for k, (_v, words) in enumerate(d.vlines):
        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(ref if k == 0 else neutral(dark), width=2, style=2)), words)
    if d.diagonal is not None and d.x.size:
        lo = float(min(np.min(d.x), np.min(d.y)))
        hi = float(max(np.max(d.x), np.max(d.y)))
        tau, words = d.diagonal
        plot.plot([lo, hi], [lo, hi], pen=pg.mkPen(neutral(dark), width=1))
        plot.plot([lo, hi], [lo - tau, hi - tau], pen=pg.mkPen(ref, width=2, style=2))
        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(neutral(dark), width=1)), "birth = death (no persistence)")
        legend.addItem(pg.PlotDataItem(pen=pg.mkPen(ref, width=2, style=2)), words)
    plot.legend_box = legend


DIAGNOSTICS_SIZE_KEY = "method_diagnostics_dialog.size"


class MethodDiagnosticsDialog(QtWidgets.QDialog):
    """One tab per method: its plot and its caption, or why it has none; 'Export image...' writes the plot shown."""

    def __init__(self, runs: Sequence[Tuple[str, Any]], parent: Optional[Any] = None) -> None:
        import pyqtgraph as pg
        super().__init__(parent)
        self.runs = list(runs)
        self.setObjectName("method_diagnostics_dialog")
        self.setWindowTitle("Method diagnostics")
        # maximize / resize, the size the user left it (tools.mps_ui_layout, 2026-10-23); one plot needs less room
        # than the comparison [ad hoc: 60 % of the screen]
        from tools.mps_ui_layout import open_sized
        open_sized(self, DIAGNOSTICS_SIZE_KEY, share=0.6)
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel("Each method's own plot, where one is already computed or cheap; for the others, "
                                 "why there is none. The map shows every method's clusters.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("diagnostics_tabs")
        lay.addWidget(self.tabs, 1)
        # tab index -> (plot widget, its diagnostic)
        self.plots: Dict[int, Tuple[Any, Diagnostic]] = {}
        for method, an in self.runs:
            d, why = diagnostic_for(method, an)
            page = QtWidgets.QWidget()
            pl = QtWidgets.QVBoxLayout(page)
            if d is None:
                label = QtWidgets.QLabel(f"No diagnostic plot for {method}: {why}.")
                label.setWordWrap(True)
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
                pl.addWidget(label)
            else:
                plot = pg.PlotWidget()
                plot.setObjectName(f"diagnostic_plot_{method}")
                draw(plot, d, dark=True)
                pl.addWidget(plot, 1)
                cap = QtWidgets.QLabel(d.caption)
                cap.setWordWrap(True)
                pl.addWidget(cap)
                self.plots[self.tabs.count()] = (plot, d)
            self.tabs.addTab(page, method)
        row = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export image...")
        self.btn_export.setObjectName("btn_diagnostics_export")
        self.btn_export.setToolTip("Write the plot of the tab shown as a figure: PNG (300-1200 dpi) or SVG, on "
                                   "white, with an editable title; an existing file is never overwritten (a _2, "
                                   "_3... is added).")
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addStretch(1)
        row.addWidget(self.btn_export)
        row.addWidget(close)
        lay.addLayout(row)
        self.tabs.currentChanged.connect(lambda _i: self._sync())
        self.btn_export.clicked.connect(self._export)
        self._sync()
        from tools import ui_search
        ui_search.install(self)                       # F3: Ctrl+F

    def _sync(self) -> None:
        self.btn_export.setEnabled(self.tabs.currentIndex() in self.plots)

    def export_tab(self, index: int, request: Any) -> str:
        """Write the plot of tab ``index`` as ``request`` asks (figure_export.FigureRequest)."""
        plot, d = self.plots[index]
        return export_plot(plot, d, request)

    def _export(self) -> None:
        from tools import figure_export
        from tools.figure_export_ui import ask
        entry = self.plots.get(self.tabs.currentIndex())
        if entry is None:
            return
        plot, d = entry
        source = str(getattr(self.runs[0][1], "source_name", "") or "axon")
        name = f"{d.method} diagnostic"
        request = ask([name], figure_export.suggested_name(source, name), parent=self,
                      titles={name: figure_export.plain_title(plot.getPlotItem())})
        if request is None:
            return
        try:
            written = export_plot(plot, d, request)
        except Exception as error:                               # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not write the image:\n\n{error}")
            return
        QtWidgets.QMessageBox.information(self, "Image written", f"{figure_export.describe(request)}\n\n{written}")


def export_plot(plot: Any, d: Diagnostic, request: Any) -> str:
    """Redraw ``d`` on white (when asked), write it with the request's title (figure_export.write), and draw it back
    on black."""
    from tools import figure_export
    try:
        if getattr(request, "white", True):
            draw(plot, d, dark=False)
        return figure_export.write(plot.getPlotItem(), request)
    finally:
        draw(plot, d, dark=True)


__all__ = ["Diagnostic", "NOT_OFFERED", "diagnostic_for", "draw", "export_plot", "MethodDiagnosticsDialog"]
