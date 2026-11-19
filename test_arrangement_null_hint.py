"""The Perimeter arrangement tab says why H1 / H2 are empty before the null runs (manual test of 2026-11-19: empty
0-1 axes, nothing said, the 'Null (per axon)' row pushed out of view by the Working scale panel).

Presentation only: a notice centred on H1 / H2 while no null is drawn (with B and the estimate), gone when one is
drawn and back when another axon is shown; the help and the plots' tooltip name the button; the Working scale panel
sits below the null's row. Synthetic objects only. Offscreen Qt.
Run: QT_QPA_PLATFORM=offscreen py -3 -m pytest test_arrangement_null_hint.py -q
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_APP = []
_KEEP = []


def _app():
    from PyQt5 import QtWidgets
    if not _APP:
        _APP.append(QtWidgets.QApplication.instance() or QtWidgets.QApplication([]))
    return _APP[0]


R = np.arange(20.0, 401.0, 10.0)


def _outcome():
    med = -R / 2.0
    env = SimpleNamespace(r=R, observed=med * 1.1, median=med, pointwise_lo=med - 30, pointwise_hi=med + 30,
                          global_lo=med - 60, global_hi=med + 60, r_star=110.0, h_max=-50.0, p_global=0.4, n_sim=199)
    res = SimpleNamespace(h1=env, h2=env, interpretable=True, headline=lambda: "ok", global_r_min_nm=80.0,
                          scale=SimpleNamespace(fusion_nm=60.0, d50_nm=80.0), example=None)
    return SimpleNamespace(ok=True, status="ok", message="", result=res, gap=None, key="DBSCAN 25/10")


def _run(n_locs=4000):
    an = SimpleNamespace(n_locs_slab=n_locs)
    return an, SimpleNamespace(key="DBSCAN 25/10", method="DBSCAN", params={"eps": 25.0, "min_samples": 10},
                               ok=True, analysis=an)


def _texts(plot):
    import pyqtgraph as pg
    return [it for it in plot.getPlotItem().items if isinstance(it, pg.TextItem)]


def test_before_any_analysis_the_plots_say_what_to_do():
    _app()
    from tools import arrangement_ui as A
    p = A.ArrangementPanel()
    try:
        assert p.null_hint_shown() == f"No null yet: run the MPS analysis, then press '{A.NULL_BUTTON}' above."
        for plot in (p.plot_h1, p.plot_h2):
            assert len(_texts(plot)) == 1
            (x0, x1), (y0, y1) = plot.getViewBox().viewRange()
            assert (x0, x1, y0, y1) == (0.0, 1.0, 0.0, 1.0)
        assert not _texts(p.plot_g)                          # the regularity plots keep their own preview
    finally:
        _KEEP.append(p)


def test_no_null_yet_names_b_and_the_estimate_and_follows_b():
    _app()
    from tools import arrangement_ui as A
    from tools import clustering_integration as ci
    p = A.ArrangementPanel()
    try:
        an, run = _run()
        p.set_context(an, [run], 12.0, 14.0)
        est = ci.words_estimate(ci.null_estimate_seconds(run, 199))
        assert p.null_hint_shown() == f"No null yet: press 'Null (per axon)' above (B = 199, {est}).", \
            p.null_hint_shown()
        p.combo_b.setCurrentIndex(p.combo_b.findData(999))
        assert "(B = 999, " in p.null_hint_shown()
        # one notice per plot, however often the state is redrawn
        p.set_context(an, [run], 12.0, 14.0)
        p.set_dark(False)
        assert len(_texts(p.plot_h1)) == 1 and len(_texts(p.plot_h2)) == 1
        # lines short enough for half the tab
        assert max(len(line) for line in _texts(p.plot_h1)[0].toPlainText().splitlines()) <= A.HINT_WIDTH
    finally:
        _KEEP.append(p)


def test_a_drawn_null_clears_the_notice_and_another_axon_brings_it_back():
    _app()
    import pyqtgraph as pg
    from tools import arrangement_ui as A
    p = A.ArrangementPanel()
    try:
        an, run = _run()
        p.set_context(an, [run], 12.0, 14.0)
        p.outcomes = {run.key: _outcome()}
        p._draw(run.key)                                      # what selecting the method's row draws
        assert p.null_hint_shown() == ""
        for plot in (p.plot_h1, p.plot_h2):
            assert not _texts(plot)
            assert any(isinstance(it, pg.PlotDataItem) for it in plot.getPlotItem().items)
            (x0, x1), _y = plot.getViewBox().viewRange()
            assert x1 > 300                                   # the view follows the curves, not the notice's 0-1
        an2, run2 = _run(5000)                                # another axon: its nulls are not this one's
        p.set_context(an2, [run2], 12.0, 14.0)
        assert p.outcomes == {} and p.null_hint_shown().startswith("No null yet: press 'Null (per axon)' above")
        # a null that did not run says so instead
        p.outcomes = {run2.key: SimpleNamespace(ok=False, status="refused", message="slow", result=None, gap=None,
                                                key=run2.key)}
        p._draw(run2.key)
        assert p.null_hint_shown() == (f"No null drawn for {run2.key}: refused (see the message under the "
                                       "estimate).")
    finally:
        _KEEP.append(p)


def test_the_notice_steps_aside_for_export_image():
    _app()
    from tools import arrangement_ui as A
    p = A.ArrangementPanel()
    try:
        items = _texts(p.plot_h1) + _texts(p.plot_h2)
        assert items and all(it.isVisible() for it in items)
        with p.legends_hidden():
            assert not any(it.isVisible() for it in items)
        assert all(it.isVisible() for it in items)
    finally:
        _KEEP.append(p)


def test_the_help_and_the_tooltip_say_the_null_is_needed():
    _app()
    from tools import arrangement_ui as A
    for text in (A.H_HELP, A.GLOBAL_TIP):
        assert "Null (per axon)" in text and "Randomization box (KS test) does not compute" in text
    p = A.ArrangementPanel()
    try:
        assert A.NULL_NEEDED in p.lbl_h_help.text() and A.NULL_NEEDED in p.plot_h1.toolTip()
    finally:
        _KEEP.append(p)


def test_the_working_scale_sits_below_the_null_row_and_above_the_table():
    _app()
    from PyQt5 import QtWidgets
    from tools import arrangement_ui as A
    p = A.ArrangementPanel()
    try:
        head = QtWidgets.QLabel("Working scale")
        head.setMinimumHeight(400)                            # a tall panel, as the real one
        p.set_header(head)
        lay = p.upper.layout()
        top = next(i for i in range(lay.count()) if lay.itemAt(i).layout() is not None
                   and lay.itemAt(i).layout().indexOf(p.btn_null) >= 0)
        order = [lay.indexOf(w) for w in (p.lbl_estimate, p.progress, p.banner, head, p.table)]
        assert top == 0 and order == sorted(order) and order[3] == order[2] + 1, (top, order)
        # a short upper pane: the null's row and its estimate are what shows, at the top of its scroll area
        p.resize(1200, 600)
        p.show()
        _app().processEvents()
        p.splitter.setSizes([90, 510])
        _app().processEvents()
        assert p.upper_scroll.verticalScrollBar().value() == 0
        view = p.upper_scroll.viewport().rect()
        btn_at = p.btn_null.mapTo(p.upper_scroll.viewport(), p.btn_null.rect().topLeft())
        assert view.contains(btn_at), (btn_at, view)
        assert head.mapTo(p.upper, head.rect().topLeft()).y() > p.lbl_estimate.mapTo(
            p.upper, p.lbl_estimate.rect().topLeft()).y()
    finally:
        p.hide()
        _KEEP.append(p)
