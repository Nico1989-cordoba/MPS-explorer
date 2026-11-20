# -*- coding: utf-8 -*-
"""
The presentation mode of the perimeter density window with the answers of the OPEN axon
(``tools.mps_perimeter_demo_live``), on synthetic results (no file, no run):

* reliable contour: every card of A, B and C quotes this result's numbers (A: contrast and period with z and p; B:
  Spearman, z, p, minimum tau and the tau_vs_offset control; C: r of each pair with its expected leak, z-1 / z+1 as
  the main reading), in both languages, with the "En este axón:" heading;
* contour rejected: A, B and C say "no se calculó porque" the contour was rejected, the contour card gives the
  reason in plain words, and the general explanation stays;
* no result (Run not pressed, or running): every card says why nothing was computed;
* not reliable contour: the numbers carry the warning;
* the window opens the tour with these texts (offscreen, private tour controller).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

from tools import mps_perimeter_bc as bc  # noqa: E402
from tools import mps_perimeter_demo_live as live  # noqa: E402
from tools import mps_perimeter_types as pdt  # noqa: E402
from tools import mps_perimeter_window as mpw  # noqa: E402
from tools import mps_tour  # noqa: E402
from tools import mps_tour_steps as data  # noqa: E402

LOW_QUALITY = "CONTOUR NOT RELIABLE (left out of pooled analyses): "
REJECTED = ("implausible length: length 30000 nm is 2.50x the perimeter of its convex hull (limit 1.5): the curve "
            "snakes inside the ring")


def _t(value, z, p, n_null=999):
    return pdt.TestReadout(value=value, null_mean=0.0, null_sd=1.0, z=z, p=p, n_null=n_null, sided="two")


def _result(kind: str = "reliable") -> pdt.PerimeterResult:
    """A synthetic result: "reliable", "unreliable" (accepted, quality gate failed) or "rejected"."""
    trace = pdt.TraceLabel(label="2026.R1.a1", year="2026", roi_number="1", axon_number="1", acquisition_date=None,
                           complete=True, missing=(), origin="test")
    z0 = pdt.Z0Choice(z0_nm=12.0, criterion=pdt.Z0Criterion.PEAK, peak_nm=12.0, focus_nominal_nm=None,
                      focus_empirical_nm=None, peak_minus_focus_nm=None, peak_minus_focus_nominal_nm=None,
                      peak_minus_focus_empirical_nm=None)
    slices = pdt.SliceSet(bounds_nm=((-243.0, -73.0), (-73.0, 97.0), (97.0, 267.0)),
                          label=np.zeros(10, dtype=np.int8), n_locs=(2000, 2500, 2250), pct_of_central=(80.0, 90.0),
                          z_coverage=(1.0, 1.0, 1.0), frac_in_three=0.9)
    params = pdt.PerimeterParams()
    if kind == "rejected":
        ct = pdt.AutoContour(path=None, length_nm=float("nan"), method="density_nodes", knot_nm=400.0,
                             residual_sd_nm=float("nan"), n_self_crossings=0, nodes_nm=np.zeros((0, 2)),
                             polar_distance_nm=float("nan"), star_shaped=False, status=pdt.Status.CONTOUR_FAILED,
                             warnings=(REJECTED,))
        return pdt.PerimeterResult(axon_id="a1", trace=trace, params=params, z0=z0, slices=slices, contour=ct,
                                   band=None, profiles=None, status=pdt.Status.CONTOUR_FAILED,
                                   failure="the contour was rejected")
    warn = (LOW_QUALITY + "slice misfit 60 nm (limit 40)",) if kind == "unreliable" else ()
    ct = pdt.AutoContour(path=object(), length_nm=3400.0, method="density_nodes", knot_nm=400.0,
                         residual_sd_nm=21.0, n_self_crossings=0, nodes_nm=np.zeros((0, 2)), polar_distance_nm=10.0,
                         star_shaped=True, status=pdt.Status.OK, warnings=warn)
    band = pdt.BandWidth(mode="fixed", half_width_nm=90.0, fwhm_nm=120.0, robust_sd_nm=50.0,
                         frac_within={50.0: 0.5, 100.0: 0.8, 150.0: 0.9}, frac_inside=0.1, frac_outside=0.1)
    prof0 = pdt.DensityProfile(counts=np.ones(68, dtype=np.int64), relative=np.ones(68), measured=np.ones(68, bool),
                               n_bins=68, bin_nm_effective=50.0, origin_arc_nm=0.0, max_bin=0, max_tie=False,
                               n_locs=2100)
    profiles = pdt.SliceProfiles(by_slice={-1: prof0, 0: prof0, 1: prof0}, length_nm=3400.0)
    diag = pdt.PerimeterDiagnostics(unsupported_fraction=0.04, a_period_nm=190.0, a_exclusion_nm=120.0,
                                    a_usable=True, window_bins=4, leak_expected_r=0.27,
                                    c_reading="registry between slices", sigma_z_source=bc.SIGMA_Z_SOURCE_PILOT,
                                    sigma_z_widths=(36.0, 47.0, 51.0))
    a = {"contrast": _t(1.35, 2.41, 0.012), "period": _t(0.4, 0.81, 0.21)}
    b = pdt.BReadout(test=_t(-0.35, -2.10, 0.018, n_null=1999), tau_min=0.934, resolved=True, tau_vs_offset=0.12)
    c = {(-1, 0): _t(0.42, 3.1, 0.002), (0, 1): _t(0.38, 2.9, 0.004), (-1, 1): _t(0.31, 2.85, 0.004)}
    return pdt.PerimeterResult(axon_id="a1", trace=trace, params=params, z0=z0, slices=slices, contour=ct, band=band,
                               profiles=profiles, a=a, b=b, c=c, status=pdt.Status.OK, diagnostics=diag,
                               extras={"tau_vs_offset": 0.12})


def _steps(result, reason=live.NO_RESULT_NOT_RUN):
    return {s.key: s for s in live.live_steps(mps_tour.steps_for(mpw.DEMO_TOUR_ID), result, reason)}


def test_reliable_result_quotes_this_axon_in_every_card():
    r = _result("reliable")
    steps = _steps(r)
    for step in steps.values():
        for lang in data.LANGUAGES:
            ex = step.text(lang, "example")
            assert ex.startswith(live.LIVE_HEADING[lang] + ": "), (step.key, lang, ex)
            assert "{" not in ex and "}" not in ex
        # the general explanation stays as it was
        assert step.text("es", "what").startswith("¿Qué pregunta responde?:")
        assert step.text("es", "why").startswith("Explicación sencilla:")
    assert "CONFIABLE" in steps["pdd_contour"].text("es", "example")
    assert "3.400 nm" in steps["pdd_contour"].text("es", "example")
    assert "z0 = 12 nm (pico)" in steps["pdd_slices"].text("es", "example")
    assert "2.500 (central)" in steps["pdd_slices"].text("es", "example")
    a = steps["pdd_a"].text("es", "example")
    assert "1,35 (z = 2,41; p = 0,012)" in a and "período = 190 nm (z = 0,81; p = 0,210)" in a
    assert "se atrae" in a and "120 nm" in a
    b = steps["pdd_b"].text("es", "example")
    assert "Spearman = -0,35 (z = -2,10; p = 0,018)" in b and "1.999 rotaciones" in b
    assert "τ mínimo = 0,934" in b and "r = +0,12" in b and "más curvados" in b
    c = steps["pdd_c"].text("es", "example")
    leak = bc.leak_expected_r_pairs((2000, 2500, 2250), 47.0, r.params.slice_thickness_nm, r.params.slice_guard_nm)
    assert c.index("LECTURA PRINCIPAL, z−1 / z+1") < c.index("Pares vecinos")
    assert "r = +0,31 (z = 2,85; p = 0,004)" in c
    assert f"fuga esperada para este par r = {leak[(-1, 1)]:.2f}".replace(".", ",") in c
    assert f"fuga esperada {leak[(-1, 0)]:.2f}".replace(".", ",") in c
    assert f"fuga esperada {leak[(0, 1)]:.2f}".replace(".", ",") in c
    assert "hay registro entre rodajas" in c and "registro entre rodajas" in c
    c_en = steps["pdd_c"].text("en", "example")
    assert c_en.startswith("On this axon: MAIN READING, z-1 / z+1") and "r = +0.31 (z = 2.85, p = 0.004)" in c_en
    assert "Ojo" not in a + b + c


def test_rejected_contour_says_why_nothing_was_computed():
    steps = _steps(_result("rejected"))
    contour = steps["pdd_contour"].text("es", "example")
    assert "RECHAZADO" in contour and "30.000 nm" in contour and "2,5 veces" in contour
    for key, name in (("pdd_a", "A"), ("pdd_b", "B"), ("pdd_c", "C")):
        for lang, words in (("es", f"{name} no se calculó porque el contorno fue rechazado"),
                            ("en", f"{name} was not computed because the contour was rejected")):
            assert words in steps[key].text(lang, "example"), (key, lang)
        assert steps[key].text("es", "why").startswith("Explicación sencilla:")
    assert "no se calculó porque el contorno fue rechazado" in steps["pdd_profile"].text("es", "example")
    assert "el contorno fue rechazado" in steps["pdd_missing"].text("es", "example")


def test_no_result_says_run_was_not_pressed_or_is_running():
    steps = _steps(None)
    for key in ("pdd_slices", "pdd_contour", "pdd_unanchored", "pdd_profile", "pdd_a", "pdd_b", "pdd_c"):
        ex = steps[key].text("es", "example")
        assert "no se calculó porque todavía no se apretó Run" in ex, (key, ex)
        assert "was not computed because Run has not been pressed" in steps[key].text("en", "example"), key
    running = _steps(None, live.NO_RESULT_RUNNING)
    assert "el cálculo todavía está corriendo" in running["pdd_c"].text("es", "example")
    # the validation card never claims numbers of this axon
    assert "no se miden en este axón" in steps["pdd_validation"].text("es", "example")


def test_unreliable_contour_marks_every_number():
    steps = _steps(_result("unreliable"))
    assert "NO CONFIABLE" in steps["pdd_contour"].text("es", "example")
    assert "slice misfit 60 nm" in steps["pdd_contour"].text("es", "example")
    for key in ("pdd_a", "pdd_b", "pdd_c"):
        assert "Ojo: el contorno quedó marcado como NO CONFIABLE" in steps[key].text("es", "example"), key


def test_c_reading_words_translate_the_range_suffix():
    label = "leakage only (within the measured range of sigma_z 36-51 nm: neighbours above leakage at 36 nm)"
    es = live.c_reading_words(label, "es")
    assert es.startswith("sólo fuga axial") and "36–51 nm" in es and "vecinas por encima de la fuga a 36 nm" in es


def _pump(app, seconds=0.1):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.fixture()
def window():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = mpw.MPSPerimeterWindow()
    w.resize(1280, 860)
    w.show()
    _pump(app, 0.1)
    try:
        yield app, w
    finally:
        tour = w._tour_controller()
        tour.close()
        w.close()
        _pump(app, 0.05)


def _go_to(app, ov, key):
    while ov.step().key != key:
        ov.next()
        _pump(app, 0.02)


def test_window_opens_the_tour_with_the_open_axons_answers(window):
    app, w = window
    ov = w.start_demo_tour()                      # no result: it opens all the same and says why
    _pump(app, 0.1)
    assert ov is not None and ov.is_open()
    assert mpw.DEMO_NEEDS_RUN in w.status.full()
    _go_to(app, ov, "pdd_a")
    html = ov.card.sections["example"].text()
    assert "<b>En este axón</b>" in html and "no se calculó porque todavía no se apretó Run" in html
    ov.close_tour("closed")
    _pump(app, 0.05)

    w.result = _result("reliable")                # what a finished Run leaves
    ov = w.start_demo_tour()
    _pump(app, 0.1)
    _go_to(app, ov, "pdd_c")
    html = ov.card.sections["example"].text()
    assert "LECTURA PRINCIPAL, z−1 / z+1" in html and "r = +0,31" in html
    w._tour_controller().toggle_language()
    _pump(app, 0.05)
    assert "MAIN READING, z-1 / z+1" in ov.card.sections["example"].text()
    w._tour_controller().toggle_language()
    ov.close_tour("closed")
