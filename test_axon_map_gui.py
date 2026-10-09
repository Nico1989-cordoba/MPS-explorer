# -*- coding: utf-8 -*-
"""
Offscreen test of the stage-2 widgets on their own (UI stage 2, design 3-5, 8.4; IMPL-B).

``AxonMap``, ``AxialView`` and ``NearestNeighboursPanel`` are built here from the analysis of a SIMULATED axon
(``batch_columns.write_simulated_input`` + three interior clusters, analysed with ``analyze_axon``; the interior
clusters discarded with ``compare_discard``; the rings with ``analyze_all_segments``) and a synthetic Axoplasm state.
No window uses them yet (IMPL-C moves the plots); this test checks the widgets' own contract:

  1. the map: the three views (source, colouring, ticked rows); every row shows or hides exactly its items; the
     sources are exclusive and a header change makes the view Custom; row state through redraws, radio changes and
     re-runs; grey rows disabled (and not drawn) over the image, back when the image is unticked; the segments'
     colours disable the centres and the occupied stretches; a stale analysis disables the axoplasm classes; the empty
     state; the header selectors survive redraws; the export message (visible names, hidden count, provenance,
     origins); no white redraw over the image; the light redraw; the SVG of the map in each view contains every
     title line;
  2. the axial view: its two views, the three-line title (in the SVG too), the left axis, channel 2 hiding the
     components while it is drawn, the cut disabled while stale, the details panel;
  3. the nearest-neighbours panel: N capped at clusters - 1, N = 1 is the old 1NN histogram, N > 1 disables the
     median and the reference, the reference starts off and its line is the paper role, a range counts what falls
     outside, the CDF's legend sits bottom-right and its crossing is neutral, "Save distances..." writes the
     builder's bytes under the default name, no KS p in the details.

R8: simulated data only; every file in a temporary folder.

Run:  venv\\Scripts\\python.exe test_axon_map_gui.py      (offscreen; about 1 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import html  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_axon_map_gui_")
os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(WORK, "selection_log")


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-5:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def need(st: Dict[str, Any], key: str) -> Any:
    if key not in st:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return st[key]


def pump(app: Any, seconds: float = 0.05) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def visible(items: List[Any]) -> List[bool]:
    return [bool(it.isVisible()) for it in items]


def svg_text(plot_item: Any, name: str) -> str:
    from tools import figure_export
    path = os.path.join(WORK, name + ".svg")
    figure_export.write(plot_item, figure_export.FigureRequest(plot=name, path=path))
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # The SVG's text nodes in document order, joined: a title wrapped to a narrow plot (IMPL-F) is split into
    # several text elements at its spaces; its words, in order, are what must travel with the figure.
    import re as _re
    words = ""
    for t in _re.findall(r">([^<>]+)<", raw):
        t = " ".join(t.split())
        if t:
            words += ("" if not words or words.endswith("-") else " ") + t
    return html.unescape(raw) + "\n" + html.unescape(words)


def main() -> int:
    print("=" * 100)
    print("STAGE-2 WIDGETS: AXON MAP, AXIAL VIEW, NEAREST NEIGHBOURS (simulated axon, offscreen)")
    print("=" * 100)
    from PyQt5 import QtWidgets
    import pyqtgraph as pg
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from tools import mps_axon_map_layers as L
    from tools.mps_axial_view import AxialView
    from tools.mps_axon_map import AxonMap
    from tools.mps_nn_panel import NearestNeighboursPanel
    from tools.mps_plot_style import neutral, role
    st: Dict[str, Any] = {}

    print("\n0. A simulated axon, its discard comparison, its rings, a synthetic Axoplasm state")

    def simulate() -> str:
        import batch_columns as bc
        import golden_ui_stage2 as gold
        from tools import mps_zquality_window as zw
        from tools.mps_analysis import analyze_axon, compare_discard
        from tools.mps_multisegment import analyze_all_segments
        contour, seed, leak = zw.DEMO_CASES["viable"]
        npz = bc.write_simulated_input(os.path.join(WORK, "sim.npz"), contour, seed, leak)
        with np.load(npz) as d:
            x, y, z = (np.asarray(d[q], float) for q in ("x", "y", "z"))
        cx, cy = float(np.median(x)), float(np.median(y))
        r_ring = float(np.median(np.hypot(x - cx, y - cy)))
        hist, edges = np.histogram(z, bins=40)
        top = np.argsort(hist)[::-1][:3]
        extra = gold._interior_clusters(np.random.default_rng(1039), cx, cy, r_ring, (edges[top] + edges[top + 1]) / 2)
        x, y, z = (np.concatenate([x, extra["x"]]), np.concatenate([y, extra["y"]]), np.concatenate([z, extra["z"]]))
        a = analyze_axon(x, y, z, source_name="simulated", pixel_size_nm=130.0, pixel_size_source="override")
        c = np.asarray(a.centroids)
        flags = np.hypot(c[:, 0] - cx, c[:, 1] - cy) < 0.6 * r_ring
        assert flags.any(), "no interior cluster to discard"
        comp = compare_discard(a, flags, margin_nm=250.0, registration="simulated")
        ms = analyze_all_segments(x, y, z, source_name="simulated", pixel_size_nm=130.0,
                                  pixel_size_source="override")
        # a synthetic Axoplasm state: a grey image, two ring edges, localizations by class, the four groups
        rng = np.random.default_rng(5)
        img = rng.normal(1000.0, 30.0, (40, 40))
        rr, ccol = np.mgrid[0:40, 0:40]
        ring = np.abs(np.hypot(rr - 20, ccol - 20) - 12) < 0.7
        inner = np.abs(np.hypot(rr - 20, ccol - 20) - 9) < 0.7
        rect = (cx - 20 * 130.0, cy - 20 * 130.0, 40 * 130.0, 40 * 130.0)
        located = np.where(np.hypot(x - cx, y - cy) < 0.6 * r_ring, "inside",
                           np.where(rng.random(x.size) < 0.7, "membrane", "no cluster")).astype(object)
        anchored = SimpleNamespace(discarded=flags, tubulin_only=np.zeros_like(flags),
                                   spectrin_only=np.zeros_like(flags), inside_neither=~flags, margin_nm=250.0)
        state = L.AxoplasmMapState(tubulin_region=img, tubulin_rect=rect, spectrin_region=img[::-1], spectrin_rect=rect,
                                   tubulin_edge=ring, tubulin_edge_rect=rect, spectrin_edge=inner,
                                   spectrin_edge_rect=rect, loc_x=x, loc_y=y, located=located, anchored=anchored,
                                   anchored_centroids=c, clusters=c, threshold_source="otsu", margin_nm=250.0,
                                   registration="simulated", has_mask=True)
        st.update(a=a, comp=comp, ms=ms, state=state, z=z)
        return (f"{a.n_clusters_kept} clusters, {int(flags.sum())} interior discarded; {len(ms.segments)} segments; "
                f"{x.size:,} localizations")

    check("simulated inputs", simulate)

    def inputs(**kw: Any) -> Any:
        base = dict(analysis=need(st, "a"), comparison=need(st, "comp"), axoplasm=need(st, "state"),
                    rings=need(st, "ms"))
        base.update(kw)
        return L.MapInputs(**base)

    # ------------------------------------------------------------------ 1. the map
    print("\n1. The axon map")

    def views() -> str:
        m = AxonMap()
        m.resize(1200, 800)
        m.show()
        st["map"] = m
        m.set_inputs(inputs(), view="mps")
        pump(app, 0.05)
        assert m.view() == "mps" and m.combo_source.currentData() == "slab" and m.combo_colour.currentData() == "status"
        loc_keys = m.layers.group_rows("localizations")
        assert loc_keys == ["slab_kept", "slab_noise", "slab_curated", "slab_discarded"], loc_keys
        shown = set(m.layers.shown_keys())
        assert {"slab_kept", "c_kept", "contour", "occupied", "centre"} <= shown and "image" not in shown
        assert "slab_discarded" not in shown, "discard-only rows are not drawn with the measured analysis"
        assert len(m.title_lines()) == 2 and m.title_lines()[0].startswith("MPS analysis, measured:")
        m.set_view("axoplasm")
        assert m.combo_source.currentData() == "selection" and m.combo_colour.currentData() == "images"
        assert m.layers.group_rows("localizations") == ["sel_inside", "sel_membrane", "sel_free"]
        shown = set(m.layers.shown_keys())
        assert {"image", "tubulin_edge", "spectrin_edge", "c_both", "c_neither", "sel_inside"} <= shown
        assert "occupied" not in shown and "image: betaIII-tubulin" in m.title()
        m.set_view("segments")
        rows = m.layers.group_rows("localizations")
        assert rows and all(k.startswith("seg") for k in rows)
        assert set(m.layers.shown_keys()) == set(rows), m.layers.shown_keys()
        for key in ("c_kept", "occupied"):
            assert not m.layers.checkbox(key).isEnabled() and L.SEGMENT_COLOUR_REASON in m.layers.checkbox(
                key).toolTip()
            assert not any(visible(m.items(key))), f"{key} is drawn over the segments"
        m.set_view("mps")
        return "MPS analysis / Axoplasm / Segments presets: source, colouring, ticked rows; segments disable colours"

    def rows_drive_items() -> str:
        m = need(st, "map")
        n = 0
        for view in ("mps", "axoplasm"):
            m.set_view(view)
            keys = [k for k in m.layers.keys() if m.layers.checkbox(k).isEnabled() and m.items(k)]
            for key in keys:
                box = m.layers.checkbox(key)
                before = {k: visible(m.items(k)) for k in m.layers.keys()}
                box.setChecked(not box.isChecked())
                after = {k: visible(m.items(k)) for k in m.layers.keys()}
                changed = [k for k in before if before[k] != after[k]]
                # the image row also takes the grey rows off the map while it is drawn (3.5), and only them
                allowed = {key} | (set(L.OVER_IMAGE_DISABLED) if key == "image" else set())
                assert key in changed and set(changed) <= allowed, (view, key, changed)
                assert all(v == box.isChecked() for v in after[key]), (view, key)
                assert m.view() == "custom", "a row change makes the view Custom"
                box.setChecked(not box.isChecked())
                n += 1
        m.set_view("mps")
        return f"{n} rows each show/hide exactly their own items; any change makes the view Custom"

    def exclusive_and_state() -> str:
        m = need(st, "map")
        m.set_view("mps")
        m.layers.checkbox("c_kept").setChecked(False)
        m.set_inputs(inputs(shown="discard"))                 # a radio change
        assert not m.layers.checkbox("c_kept").isChecked() and not any(visible(m.items("c_kept")))
        assert m.layers.checkbox("slab_discarded").isEnabled() and m.title_lines()[0].startswith("Discard applied")
        import dataclasses
        rerun = dataclasses.replace(need(st, "a"))            # a re-run: a new analysis object
        m.set_inputs(L.MapInputs(analysis=rerun, comparison=None, axoplasm=need(st, "state"), rings=need(st, "ms"),
                                 source=m.inputs().source, colour_by=m.inputs().colour_by))
        assert not m.layers.checkbox("c_kept").isChecked(), "a row's state survives a re-run"
        m.layers.checkbox("c_kept").setChecked(True)
        m.set_inputs(inputs())
        combo = m.combo_source
        combo.setCurrentIndex(combo.findData("selection"))
        assert m.layers.group_rows("localizations") == ["sel_inside", "sel_membrane", "sel_free"]
        assert not any(k.startswith("slab_") for k in m.layers.keys()), "one source at a time"
        assert m.view() == "custom"
        assert m.layers.group_header("localizations") is combo and combo.count() == 3, "the selector survives redraws"
        combo.setCurrentIndex(combo.findData("slab"))
        return "a hidden row stays hidden through a radio change and a re-run; one localization source at a time"

    def over_image() -> str:
        m = need(st, "map")
        m.set_view("mps")
        m.layers.set_visible("image", True)
        for key in L.OVER_IMAGE_DISABLED:
            box = m.layers.checkbox(key)
            assert not box.isEnabled() and L.OVER_IMAGE_REASON in box.toolTip(), key
            assert box.isChecked() and not any(visible(m.items(key))), f"{key} drawn over the image"
        assert not m.offers_white(), "no white redraw over a photograph"
        m.layers.set_visible("image", False)
        for key in L.OVER_IMAGE_DISABLED:
            assert m.layers.checkbox(key).isEnabled() and all(visible(m.items(key))), key
        assert m.offers_white()
        return "noise and curated disabled (and hidden) while the image is drawn, back when it is not"

    def stale_and_empty() -> str:
        m = need(st, "map")
        m.set_inputs(inputs(stale=True, shown="discard"), view="axoplasm")
        assert not m.layers.is_group_enabled("localizations") and "previous" in m.layers.group_reason("localizations")
        assert not any(visible(m.items("sel_inside")))
        assert m.title_lines()[0].endswith("(previous selection)")
        m.set_inputs(L.MapInputs(), view="mps")
        assert m.title() == "No MPS analysis of this selection yet"
        assert not m.layers.is_group_enabled("contour")
        m.set_inputs(inputs(), view="mps")
        return "stale analysis: classes disabled, title says so; empty state"

    def export_and_light() -> str:
        m = need(st, "map")
        m.set_inputs(inputs(shown="discard"), view="mps")
        m.set_provenance("<b>simulated</b><br>Pixel size: 130 nm", L.provenance_line2(
            need(st, "comp").discard_applied, roi_words="ROI circle", cut=(100.0, 280.0)))
        msg = m.export_message()
        assert msg.startswith("Visible layers: In kept clusters (") and "hidden or not available" in msg
        assert "z as fitted, no z correction applied" in msg and "Origins: eps 25 nm (paper)" in msg
        assert "Title: Discard applied" in msg
        texts = {}
        for view in ("mps", "axoplasm", "segments"):
            m.set_view(view)
            pump(app, 0.05)
            svg = svg_text(m.plot_item(), f"map_{view}")
            for line in m.title_lines():
                for part in line.split("; "):
                    assert part in svg, (view, part)
            texts[view] = len(m.title_lines())
        m.set_view("mps")
        m.set_dark(False)
        assert m.plot.backgroundBrush().color().name() == "#ffffff"
        contour = [it for it in m.items("contour") if isinstance(it, pg.PlotDataItem)][-1]
        assert contour.opts["pen"].color().name() == neutral(False)
        m.set_dark(True)
        return f"export message; SVG title lines {texts}; light redraw uses the light neutral"

    check("views: the three presets", views)
    check("each row drives exactly its items", rows_drive_items)
    check("exclusive sources; state through radio changes and re-runs; selectors survive", exclusive_and_state)
    check("grey rows disabled over the image", over_image)
    check("stale and empty states", stale_and_empty)
    check("export message, SVG titles, light redraw", export_and_light)

    # ------------------------------------------------------------------ 2. the axial view
    print("\n2. The axial view")

    def axial() -> str:
        v = AxialView()
        v.resize(900, 600)
        v.show()
        a, ms, z = need(st, "a"), need(st, "ms"), need(st, "z")
        v.set_inputs(L.AxialInputs(z_roi=z, analysis=a, cut=(a.slab_zmin_nm + 0.1, a.slab_zmax_nm - 0.1),
                                   ch2_z=z + 40.0, rings=ms), view="slab")
        pump(app, 0.05)
        assert len(v.title_lines()) == 3 and v.title_lines()[2] == "z as fitted; no z correction applied"
        assert v.plot_item().getAxis("left").labelText == L.AXIAL_LEFT_LABEL
        on = set(v.layers.shown_keys())
        assert {"roi_in", "roi_out", "components", "mixture", "slab", "main_cut"} <= on and "ch2_roi" not in on
        v.layers.set_visible("ch2_roi", True)
        assert not v.layers.checkbox("components").isEnabled() and not any(visible(v.items("components")))
        assert all(visible(v.items("mixture"))), "the neutral mixture stays"
        v.layers.set_visible("ch2_roi", False)
        assert v.layers.checkbox("components").isEnabled() and all(visible(v.items("components")))
        svg = svg_text(v.plot_item(), "axial_slab")
        assert all(line.split(" (")[0] in svg for line in v.title_lines()), v.title_lines()
        v.set_view("segments")
        on = set(v.layers.shown_keys())
        assert any(k.startswith("segband") for k in on) and "valleys" in on and "roi_in" not in v.layers.keys()
        assert not v.layers.checkbox("components").isEnabled()
        assert v.title_lines()[1].startswith(f"segments: mode {ms.mode}")
        svg = svg_text(v.plot_item(), "axial_segments")
        assert all(line.split(" (")[0] in svg for line in v.title_lines())
        v.set_inputs(L.AxialInputs(z_roi=z, analysis=a, cut=(1.0, 2.0), cut_stale=True), view="slab")
        assert not v.layers.checkbox("main_cut").isEnabled() and not any(visible(v.items("main_cut")))
        rows = [r[0] for r in v.details.rows()]
        assert "components chosen by BIC" in rows and "density peak" in rows
        v.close()
        return "3-line title (also in the SVG), left axis, channel 2 hides the components, segments view, stale cut"

    check("axial view", axial)

    # ------------------------------------------------------------------ 3. nearest neighbours
    print("\n3. The nearest-neighbours panel")

    def nn() -> str:
        a, comp = need(st, "a"), need(st, "comp")
        p = NearestNeighboursPanel(root_name=lambda: os.path.join(WORK, "sim"))
        p.resize(1100, 500)
        p.show()
        p.set_analysis(a, a, comp)
        pump(app, 0.05)
        assert p.spin_neighbours.maximum() == min(10, a.n_clusters_kept - 1)
        bars = [it for it in p.items_of("nn_bars") if isinstance(it, pg.BarGraphItem)][0]
        counts, edges = np.histogram(a.nn.first_nn_nm, bins=30)
        assert np.array_equal(bars.opts["height"], counts), "N = 1, automatic range: the old 1NN histogram"
        assert p.nn_title() == "1NN distance between cluster centres (measured)"
        ref = p.layers.checkbox("nn_reference")
        assert not ref.isChecked() and not any(visible(p.items_of("nn_reference"))), "the reference starts off"
        ref.setChecked(True)
        line = p.items_of("nn_reference")[0]
        assert line.isVisible() and line.pen.color().name() == role("paper")
        p.spin_neighbours.setValue(3)
        assert not p.layers.checkbox("nn_median").isEnabled() and L.NN_FIRST_ONLY in p.layers.checkbox(
            "nn_reference").toolTip()
        assert p.nn_title().startswith("Distances to the 1st..3rd nearest centre, pooled (N = 3; measured")
        p.radio_range_set.setChecked(True)
        p.spin_range_min.setValue(0.0)
        p.spin_range_max.setValue(100.0)
        assert "outside the range not shown" in p.nn_title()
        p.radio_range_auto.setChecked(True)
        legend = p.legend
        assert legend.opts["offset"] == (-10, -10) and [lab.text for _s, lab in legend.items] == ["randomized",
                                                                                                 "observed"]
        cross = [it for it in p.plot_cdf.getPlotItem().items if isinstance(it, pg.InfiniteLine)]
        if cross:
            assert cross[0].pen.color().name() == neutral(True) != role("paper")
        out = os.path.join(WORK, "saved.csv")
        ref_path = os.path.join(WORK, "builder.csv")
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(  # type: ignore
            lambda *args, **kw: (out, "CSV Files (*.csv)"))
        st["default_seen"] = p.default_name()
        assert p.on_save() == out
        L.write_distances_csv(ref_path, a, a, 3)
        with open(out, "rb") as f1, open(ref_path, "rb") as f2:
            assert f1.read() == f2.read()
        assert st["default_seen"].endswith("sim_3neighbor_distances.csv")
        p.set_analysis(comp.discard_applied, a, comp)
        assert p.default_name().endswith("_discard_applied.csv")
        assert p.cdf_title().startswith("1NN CDF: observed vs randomized (discard applied")
        cd = [r[0] for r in p.details_cdf.rows()]
        assert "KS statistic D" in cd and not any("p" in x.split() for x in cd), cd
        p.close()
        return "N cap, the old 1NN histogram, reference off / paper role, N > 1, range, legend, crossing, Save"

    check("nearest neighbours panel", nn)

    try:
        need(st, "map").close()
    except Exception:  # noqa: BLE001
        pass
    pump(app, 0.1)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.0f} s; work folder {WORK}")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
