# -*- coding: utf-8 -*-
"""
The presentation mode of the perimeter density window ("Modo presentación", tour ``perimeter_demo``), offscreen.

The simulated example axon is opened the way Help opens it (``MPS_explorer.open_demo_axon``), the perimeter window
is fed by the program's own ``_perimeter_inputs`` (the same path the "Perimeter..." button uses), Run is pressed
with the default bar, and then:

* the numbers the tour quotes (``mps_tour_steps.PERIMETER_DEMO_FACTS``) are the ones this run measures
  (``mps_perimeter_window.perimeter_demo_facts``), and the window's own lines say the same;
* without a result the tour still opens, every card saying why nothing was computed, and the status asks for Run;
* after Run every card's example is the open axon's answer ("En este axón: ...", tools.mps_perimeter_demo_live);
* the tour walks every step: each anchor exists and is visible (the tab switches by itself), the counter is right,
  every text is filled in both languages and every example says "simulated";
* the original "perimeter" tour still resolves.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import functools
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

from tools import mps_perimeter_window as mpw  # noqa: E402
from tools import mps_tour  # noqa: E402
from tools import mps_tour_steps as data  # noqa: E402


def _pump(app, seconds=0.1):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture(scope="module")
def demo():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    work = tempfile.mkdtemp(prefix="test_perimeter_demo_tour_")
    settings_dir = os.path.join(work, "settings")
    os.makedirs(settings_dir, exist_ok=True)
    os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(work, "selection_log")
    from tools import mps_settings

    original = mps_settings.settings_path
    old_cwd = os.getcwd()
    mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
    import MPS_explorer

    saved = (MPS_explorer.load_settings, MPS_explorer.save_settings)
    MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
    MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
    os.chdir(work)
    mw = MPS_explorer.MPS_explorer()
    mw.columns_review_store_dir = os.path.join(work, "review_store")
    mw.resize(1400, 860)
    mw.show()
    _pump(app, 0.2)
    assert mw.open_demo_axon(), "the simulated example axon did not load"
    _pump(app, 0.2)
    w = mpw.MPSPerimeterWindow(inputs_provider=mw._perimeter_inputs, parent=mw)
    w.resize(1280, 860)
    w.show()
    _pump(app, 0.2)
    try:
        yield app, mw, w
    finally:
        mw.tour.close()
        w.close()
        mw.close()
        _pump(app, 0.1)
        MPS_explorer.load_settings, MPS_explorer.save_settings = saved
        mps_settings.settings_path = original
        os.chdir(old_cwd)


def test_button_asks_for_run_first_then_runs(demo):
    app, mw, w = demo
    assert w.btn_demo.text() == mpw.DEMO_BUTTON == "Modo presentación"
    assert w.btn_demo.toolTip() == mpw.DEMO_TOOLTIP and w.btn_demo.objectName() == "perimeter_demo_tour"
    assert w.result is None
    ov = w.start_demo_tour()
    _pump(app, 0.1)
    assert ov is not None and ov.is_open()
    assert w.status.full().endswith(mpw.DEMO_NEEDS_RUN) or mpw.DEMO_NEEDS_RUN in w.status.full()
    live = {s.key: s for s in w.demo_steps()}
    assert "C no se calculó porque todavía no se apretó Run" in live["pdd_c"].text("es", "example")
    ov.close_tour("closed")
    _pump(app, 0.05)
    assert mw.tour.overlay() is None
    assert w.start_run()
    assert w.wait_run(300), "the run did not finish"
    _pump(app, 0.2)
    assert w.result is not None


def test_the_facts_the_tour_quotes_are_what_the_window_shows(demo):
    _app, _mw, w = demo
    r = w.result
    measured = mpw.perimeter_demo_facts(r)
    assert measured == data.PERIMETER_DEMO_FACTS, (measured, data.PERIMETER_DEMO_FACTS)
    f = data.PERIMETER_DEMO_FACTS
    # On this simulated axon the automatic contour is rejected by its own gate: A, B and C are not computed.
    kind, _text = mpw.contour_state(r)
    assert kind == "bad" and r.a == {} and r.b is None and r.c == {}
    assert w.line_contour_state.full().endswith(
        f"length {f['pd_len_nm']} nm is 2.44x the perimeter of its convex hull (limit {f['pd_len_limit']})"
        ": the curve snakes inside the ring") or "implausible length" in w.line_contour_state.full()
    assert f"z0 = {f['pd_z0']} nm (peak)" in w.line_z0.full()
    assert (f"{f['pd_pct_minus']}% (z-1) and {f['pd_pct_plus']}% (z+1) of the central slice's "
            f"{f['pd_n_central']:,} localizations") in w.line_slices.full()
    assert w.line_b_reading.full() == "B: not computable for this axon (see Analysis notes)."
    assert w.line_a_reading.full() == "" and w.line_c_reading.full() == ""
    assert w.table_a.rowCount() == 0 or all(
        (w.table_a.item(i, 1) is None or w.table_a.item(i, 1).text() in ("", "n/a"))
        for i in range(w.table_a.rowCount()))


def test_every_text_is_filled_and_every_example_is_simulated():
    steps = mps_tour.steps_for(mpw.DEMO_TOUR_ID)
    assert 9 <= len(steps) <= 11
    for step in steps:
        for lang in data.LANGUAGES:
            for name in data.FIELDS:
                text = step.text(lang, name)
                assert text.strip(), (step.key, lang, name)
                assert "{" not in text and "}" not in text, (step.key, lang, name, text)
            assert data.SIMULATED_WORD[lang] in step.text(lang, "example").lower(), (step.key, lang)
        assert step.text("es", "what").startswith("¿Qué pregunta responde?:")
        assert step.text("es", "why").startswith("Explicación sencilla:")
        assert step.text("es", "read").startswith("Cómo se lee:")
        assert step.text("es", "example").startswith("La respuesta en el axón simulado")
    es = {s.key: s for s in steps}
    assert "34.540 nm" in es["pdd_contour"].text("es", "example")
    assert "z0 = 4 nm" in es["pdd_slices"].text("es", "example")
    assert "4,6–5,6 %" in es["pdd_validation"].text("es", "example")


def test_the_tour_walks_every_step_on_the_right_tab(demo):
    app, mw, w = demo
    assert w.result is not None, "run first (test order)"
    ov = w.btn_demo.click() or mw.tour.overlay()
    _pump(app, 0.2)
    assert ov is not None and ov.is_open()
    steps = mps_tour.steps_for(mpw.DEMO_TOUR_ID)
    # the cards carry the open axon's answers: on this simulated axon the contour is rejected (the same numbers as
    # PERIMETER_DEMO_FACTS) and A, B, C say why they were not computed
    live = {s.key: s for s in w.demo_steps()}
    assert [s.key for s in w.demo_steps()] == [s.key for s in steps]
    assert live["pdd_contour"].text("es", "example").startswith("En este axón: El contorno fue RECHAZADO")
    assert "34.540 nm" in live["pdd_contour"].text("es", "example")
    assert "z0 = 4 nm (pico)" in live["pdd_slices"].text("es", "example")
    for key, name in (("pdd_a", "A"), ("pdd_b", "B"), ("pdd_c", "C")):
        assert f"{name} no se calculó porque el contorno fue rechazado" in live[key].text("es", "example"), key
    assert ov.card.counter.text() == f"Paso 1 de {len(steps)}"
    tab_of = {}
    for i in range(w.tabs.count()):
        page = w.tabs.widget(i)
        for child in page.findChildren(QtWidgets.QWidget):
            if child.objectName():
                tab_of.setdefault(child.objectName(), i)
    visited = 0
    while True:
        step = ov.step()
        assert step is not None
        assert not ov.card.note.isVisible(), f"{step.key}: its control is said not to be in view"
        if step.target:
            target = mps_tour.find_target(w, step.target)
            assert target is not None, step.key
            assert target.isVisible(), step.key
            if step.target in tab_of:
                assert w.tabs.currentIndex() == tab_of[step.target], (step.key, w.tabs.currentIndex())
            assert not ov.hole().isNull(), step.key
        assert ov.card.title.text() == step.text("es", "title")
        visited += 1
        if ov.index() == len(steps) - 1:
            break
        ov.next()
        _pump(app, 0.05)
    assert visited == len(steps)
    assert ov.card.btn_next.text() == "Terminar"
    ov.next()
    _pump(app, 0.05)
    assert mw.tour.overlay() is None


def test_the_controls_tour_still_resolves(demo):
    _app, _mw, w = demo
    steps = mps_tour.steps_for(mpw.TOUR_ID)
    assert len(steps) >= 3
    for step in steps:
        if step.target:
            assert mps_tour.find_target(w, step.target) is not None, step.target
