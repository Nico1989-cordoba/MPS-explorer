"""The Perimeter arrangement tab draws H1 / H2 in a view set from the data it draws (manual test of 2026-10-23: H1
kept a range of +/-2e8 from an earlier drawing until pyqtgraph's 'A' was pressed), a value far off or not finite
never sets the view, and the Working scale / nulls' table sit above the plots behind a divider.

Synthetic curves only. Offscreen Qt. Run: py -3 -m pytest test_arrangement_draw.py -q
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = []
_KEEP_PANELS = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


R = np.arange(20.0, 401.0, 10.0)


def _env(scale: float, spike: float = 0.0):
    obs = -scale * R / 400.0
    med = obs * 0.8
    env = SimpleNamespace(r=R, observed=obs, median=med, pointwise_lo=med - 30 * scale / 200,
                          pointwise_hi=med + 30 * scale / 200, global_lo=med - 60 * scale / 200,
                          global_hi=med + 60 * scale / 200, r_star=110.0, h_max=-50.0, p_global=0.4, n_sim=199)
    if spike:
        env.global_hi = env.global_hi.copy()
        env.global_hi[5] = spike                       # one value far off the others
        env.pointwise_lo = env.pointwise_lo.copy()
        env.pointwise_lo[7] = np.inf                   # and one not finite
    return env


def _outcome(env):
    res = SimpleNamespace(h1=env, h2=env, interpretable=True, headline=lambda: "ok", global_r_min_nm=80.0,
                          scale=SimpleNamespace(fusion_nm=60.0, d50_nm=80.0), example=None)
    return SimpleNamespace(ok=True, status="ok", message="", result=res, gap=None, key="A")


def test_a_new_drawing_never_keeps_the_previous_range():
    _app()
    from tools.arrangement_ui import ArrangementPanel
    p = ArrangementPanel()
    try:
        p.outcomes = {"huge": _outcome(_env(2e8)), "small": _outcome(_env(300.0))}
        p._draw("huge")
        (x0, x1), (y0, y1) = p.plot_h1.getViewBox().viewRange()
        assert y1 - y0 > 1e7
        p.plot_h1.getViewBox().setRange(xRange=(-200, 50), yRange=(-2e8, 2e8))   # a zoom left behind
        p._draw("small")
        for plot in (p.plot_h1, p.plot_h2):
            (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
            assert -20 <= x0 <= 1 and 395 <= x1 <= 420, (x0, x1)
            assert -500 < y0 < -300 and -40 < y1 < 200, (y0, y1)
    finally:
        p.deleteLater()


def test_a_value_far_off_or_not_finite_never_sets_the_view():
    _app()
    from tools.arrangement_ui import ArrangementPanel, envelope_y_range
    p = ArrangementPanel()
    try:
        p.outcomes = {"spiky": _outcome(_env(300.0, spike=4e7))}
        p._draw("spiky")
        (_x0, _x1), (y0, y1) = p.plot_h1.getViewBox().viewRange()
        assert y1 < 1000, "a single far value set the view"
        assert "off the scale" in p.plot_h1.getPlotItem().titleLabel.text
        rng, off = envelope_y_range([np.array([1.0, np.nan, 3.0])], [np.array([2.0, 1e9, -np.inf])])
        assert rng == (1.0, 3.0) and off == 1
        assert envelope_y_range([None, np.array([np.nan])]) == (None, 0)
    finally:
        p.deleteLater()


def test_the_tables_sit_above_the_plots_behind_a_divider():
    _app()
    from PyQt5 import QtWidgets
    from tools.arrangement_ui import ArrangementPanel
    p = ArrangementPanel()
    try:
        head = QtWidgets.QLabel("Working scale")
        p.set_header(head)
        assert p.splitter.count() == 2 and p.splitter.widget(1) is p.plots
        assert p.upper.isAncestorOf(head) and p.upper.isAncestorOf(p.table)
        p.resize(1200, 800)
        p.show()
        _app().processEvents()
        p.place_split()
        upper, plots = p.splitter.sizes()
        assert plots >= upper and upper > 0
    finally:
        p.deleteLater()


def _labels(legend):
    return [label.text for _sample, label in legend.items]


def test_every_plot_of_the_tab_has_a_legend_with_its_elements():
    _app()
    from tools import arrangement_ui as A
    p = A.ArrangementPanel()
    try:
        p.outcomes = {"small": _outcome(_env(300.0))}
        p._draw("small")
        for name in (A.PLOT_H1, A.PLOT_H2):
            labels = _labels(p._legends[name])
            for word in (A.LEG_OBSERVED, A.LEG_MEDIAN, A.LEG_POINTWISE, A.LEG_GLOBAL, "Not testable: below d50",
                         A.LEG_FUSION, A.LEG_D50, A.LEG_RSTAR):
                assert word in labels, (name, word, labels)
            assert [lab for lab, _s in p.legend_entries(name)] == labels       # the same entries for Export image
        assert "Only the global envelope decides" in p.lbl_h_help.text()
        assert "Above the null = more neighbours than random" in p.lbl_h_help.text()
        assert p.lbl_regularity_help.text()
        p.chk_legend.setChecked(False)
        assert not any(leg.isVisible() for leg in p._legends.values())
        p.chk_legend.setChecked(True)
        with p.legends_hidden():
            assert not any(leg.isVisible() for leg in p._legends.values())
        assert all(leg.isVisible() for leg in p._legends.values())
        p._draw("small")                                      # a redraw never piles up legends
        from pyqtgraph import LegendItem
        n = [it for it in p.plot_h1.getPlotItem().vb.childItems() if isinstance(it, LegendItem)]
        assert len(n) == 1
    finally:
        p.deleteLater()


def test_the_regularity_legend_and_the_method_diagnostics_legend():
    _app()
    import pyqtgraph as pg
    from tools import arrangement_ui as A
    from tools import method_diagnostics as D
    labels = [lab for lab, _s in A.regularity_legend_entries()]
    assert labels == [A.LEG_OBSERVED, A.LEG_MEDIAN, A.LEG_BAND]
    plot = pg.PlotWidget()
    d = D.Diagnostic(method="DBSCAN", title="k-distance", x_label="rank", y_label="d [nm]", kind="curve",
                     x=np.arange(10.0), y=np.arange(10.0), hlines=[(25.0, "eps 25 nm")])
    D.draw(plot, d)
    D.draw(plot, d, dark=False)                           # redrawn: one legend, not two
    legends = [it for it in plot.getPlotItem().vb.childItems() if isinstance(it, pg.LegendItem)]
    assert len(legends) == 1
    assert _labels(legends[0]) == [D.MAIN_WORDS["curve"], "eps 25 nm"]
    plot.deleteLater()


def test_a_white_redraw_keeps_the_curves_and_a_dark_title_without_a_selection():
    """Manual test of 2026-10-23 (i15): Export image of H1 on white came out with axes only and the screen's pale
    title: the white redraw followed the table's selection, and there was none."""
    _app()
    import pyqtgraph as pg
    from tools import arrangement_ui as A
    from tools.mps_plot_style import TITLE_FG, TITLE_FG_LIGHT
    p = A.ArrangementPanel()
    try:
        p.outcomes = {"small": _outcome(_env(300.0))}
        p._draw("small")
        p.table.clearSelection()
        assert p.selected_key() is None
        p.set_dark(False)                                    # what Export image does for a white figure
        for plot in (p.plot_h1, p.plot_h2):
            items = [it for it in plot.getPlotItem().items if isinstance(it, pg.PlotDataItem)]
            assert items, "the white redraw left the plot without curves"
            assert str(plot.getPlotItem().titleLabel.opts.get("color")).lower() == TITLE_FG_LIGHT.lower()
        for plot in (p.plot_gaps, p.plot_g, p.plot_s):
            assert str(plot.getPlotItem().titleLabel.opts.get("color")).lower() == TITLE_FG_LIGHT.lower()
        p.set_dark(True)
        assert str(p.plot_g.getPlotItem().titleLabel.opts.get("color")).lower() == TITLE_FG.lower()
    finally:
        _KEEP_PANELS.append(p)
