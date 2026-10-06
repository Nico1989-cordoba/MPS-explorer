# -*- coding: utf-8 -*-
"""
Offscreen test of the Z quality view's cluster panel (H6 clusters, GUI-1, 2026-10-02: the histogram of the cluster
MEDIAN z' under the localization profile, with its own SiZer strip; descriptive, no verdict), on the three SIMULATED
demo axons of tools.mps_zquality_window.DEMO_CASES only (R8):

  1. the panel is shown, its histogram counts ONE per cluster for both cluster sets (slab-free 3D clusters = the
     DBSCAN of tools.mps_sizer.groups_3d without its noise; ring clusters = build_rings' clusters), the selector
     switches between them, the legend, the tooltips;
  2. the z' axis: the cluster plots follow the localization profile's range (one way: switching the cluster set
     never moves the profile) and sit at the same x on screen;
  3. nothing that existed changes: rule v2's verdicts equal viability_v2, the localization legend is the same list,
     and the existing export (7 files) is byte-identical with and without the cluster profile;
  4. the NEW export files: a CSV of the medians (both sets) and a PNG of the panel, never overwritten; nothing when
     there is no cluster profile (and then the panel is hidden).

Run:  venv\\Scripts\\python.exe test_zquality_clusters_gui.py      (offscreen; ~1 min)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
WORK = tempfile.mkdtemp(prefix="test_zquality_clusters_gui_")
OLD_LEGEND = ["localizations (10 nm bars)", "smoothed density<br>(h = 17 nm = 0.10 P)", "significant peak",
              "significant valley", "SiZer: density rises (significant)", "SiZer: density falls (significant)",
              "SiZer: no significant slope", "ring centre: % of the central ring<br>(thick solid = central)"]


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def need(st: Dict[str, Any], key: str) -> Any:
    if key not in st:
        raise RuntimeError(f"'{key}' unavailable: the check producing it failed")
    return st[key]


def pump(app: Any, seconds: float = 0.1) -> None:
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        app.processEvents()
        time.sleep(0.005)


def bar_total(plot: Any) -> float:
    """The summed heights of the BarGraphItems of a plot (the histogram's counts)."""
    import pyqtgraph as pg
    tot = 0.0
    for it in plot.items:
        if isinstance(it, pg.BarGraphItem):
            h = it.opts.get("height")
            if h is not None:
                tot += float(np.sum(np.asarray(h, dtype=float)))
    return tot


def main() -> int:
    print("=" * 100)
    print("Z-QUALITY VIEW: CLUSTER-MEDIAN PANEL (H6 clusters, GUI-1) OFFSCREEN TEST on SIMULATED axons")
    print("=" * 100)
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    import batch_columns as bc
    from tools import mps_axial_clusters as mac
    from tools import mps_zquality_window as zw
    from tools.mps_axial_precision import viability_v2, z_quality
    from tools.mps_columns_window import build_review_rings
    st: Dict[str, Any] = {}

    print("\n0. The simulated axons and their views")

    def open_case(name: str) -> Callable[[], str]:
        def fn() -> str:
            contour, seed, s = zw.DEMO_CASES[name]
            p = bc.write_simulated_input(os.path.join(WORK, f"sim_{name}.npz"), contour, seed, s)
            assert bc.input_kind(p)[0] == "simulated"
            inp = zw.inputs_from_npz(p)
            w = zw.open_z_quality_for_inputs(inp)
            w.resize(1600, 1000)
            w.wait(300)
            assert w.error is None and w.report is not None, w.error
            pump(app, 0.3)
            st[f"inp_{name}"], st[f"view_{name}"] = inp, w
            rep = w.report
            assert rep.clusters is not None, rep.clusters_error
            return (f"{inp.x_nm.size:,} locs; clusters: "
                    + ", ".join(f"{k} {s.n_clusters}" for k, s in rep.clusters.sets.items())
                    + f"; {rep.clusters.seconds:.2f} s")
        return fn

    for name in ("viable", "marginal", "none"):
        check(f"{name}: the view with its cluster profile", open_case(name))

    print("\n1. The cluster panel")

    def counts(name: str) -> Callable[[], str]:
        def fn() -> str:
            w = need(st, f"view_{name}")
            inp = need(st, f"inp_{name}")
            rep = w.report
            assert w.cluster_panel.isVisible() and not w.cluster_panel.isHidden()
            res, _p, _l, _w = build_review_rings(inp)
            k_ring = sum(len(r.clusters) for r in res.rings)
            lab = mac.groups_3d_labels(inp.x_nm, inp.y_nm, inp.z_nm)
            k_3d = int(np.unique(lab[lab >= 0]).size)
            # groups_3d (rule v2's variance groups) = the same DBSCAN: its groups larger than one point are these
            from tools.mps_sizer import groups_3d
            g = groups_3d(inp.x_nm, inp.y_nm, inp.z_nm)
            assert int(np.count_nonzero(np.bincount(g) >= 2)) == k_3d
            sets = rep.clusters.sets
            assert sets["ring_slab"].n_clusters == k_ring, (sets["ring_slab"].n_clusters, k_ring)
            assert sets["groups_3d"].n_clusters == k_3d, (sets["groups_3d"].n_clusters, k_3d)
            per_ring = [len(r.clusters) for r in res.rings]
            assert [sets["ring_slab"].count_in_ring(r.index) for r in res.rings] == per_ring
            shown = {}
            for key in mac.CLUSTER_SET_ORDER:
                s = sets[key]
                assert float(np.sum(s.hist_counts)) == s.n_clusters
                w.cluster_source_combo.setCurrentIndex(mac.CLUSTER_SET_ORDER.index(key))
                pump(app, 0.05)
                assert w.cluster_source() == key
                shown[key] = bar_total(w.cluster_plot)
                assert shown[key] == s.n_clusters, (key, shown[key], s.n_clusters)
                assert s.profile is not None and abs(s.profile.h_nm - 0.10 * rep.v2.p_ref_nm) < 1e-9
                assert not s.profile.robust          # iid: one point per cluster
                assert len(s.median_z_nm) == len(s.n_locs) == len(s.ring_index)
                assert np.all(s.n_locs >= 1) and np.all(s.q25_z_nm <= s.median_z_nm + 1e-9)
            # ring-slab medians stay inside their slab (trap (i): by construction)
            rs = sets["ring_slab"]
            slab = {i: (lo, hi) for i, lo, hi in rep.clusters.slabs}
            inside = all(slab[int(k)][0] - 1e-6 <= m <= slab[int(k)][1] + 1e-6
                         for m, k in zip(rs.median_z_nm, rs.ring_index))
            assert inside, "a ring-slab median outside its slab"
            w.cluster_source_combo.setCurrentIndex(0)
            pump(app, 0.05)
            assert w.cluster_source() == "groups_3d"
            labels = [lab_.text for _s, lab_ in w.cluster_legend.items]
            assert any("cluster medians" in t for t in labels) and any("slab cut" in t for t in labels), labels
            assert not w.missing_tooltips, w.missing_tooltips
            assert "cluster_source_combo" in zw.ZQUALITY_WINDOW_TOOLTIPS and w.cluster_source_combo.toolTip()
            assert w.graphics_clusters.toolTip() and "cannot cross" in w.cluster_note.text()
            return (f"3D {k_3d} = {shown['groups_3d']:.0f} bars, ring {k_ring} = {shown['ring_slab']:.0f} bars "
                    f"(per ring {per_ring})")
        return fn

    for name in ("viable", "marginal", "none"):
        check(f"{name}: one count per cluster for both sets, selector, legend, tooltips", counts(name))

    print("\n2. The z' axis")

    def axis() -> str:
        w = need(st, "view_marginal")
        pp, cp, cs = w.profile_plot, w.cluster_plot, w.cluster_strip
        before = list(pp.viewRange()[0])
        for i in (1, 0):
            w.cluster_source_combo.setCurrentIndex(i)
            pump(app, 0.1)
        assert np.allclose(pp.viewRange()[0], before), (pp.viewRange()[0], before)
        assert np.allclose(cp.viewRange()[0], before) and np.allclose(cs.viewRange()[0], before)
        pp.setXRange(-100.0, 150.0, padding=0)
        pump(app, 0.1)
        assert np.allclose(cp.viewRange()[0], [-100.0, 150.0]) and np.allclose(cs.viewRange()[0], [-100.0, 150.0])
        pp.setXRange(before[0], before[1], padding=0)
        pump(app, 0.2)
        a, b = pp.vb.sceneBoundingRect(), cp.vb.sceneBoundingRect()
        assert abs(a.x() - b.x()) < 1.0 and abs(a.width() - b.width()) < 1.0, (a, b)
        return f"profile range {[round(v, 1) for v in before]} kept; viewboxes x {a.x():.0f}+{a.width():.0f} both"

    check("the cluster plots follow the profile's z' range, one way, at the same x", axis)

    print("\n3. What existed is unchanged")

    def unchanged(name: str) -> Callable[[], str]:
        def fn() -> str:
            w = need(st, f"view_{name}")
            inp = need(st, f"inp_{name}")
            rep = w.report
            res, _p, _l, _w = build_review_rings(inp)
            direct = viability_v2(res, z_quality(res), xyz_lab_nm=(inp.x_nm, inp.y_nm, inp.z_nm))
            assert [p.verdict for p in rep.v2.pairs] == [p.verdict for p in direct.pairs]
            assert [lab_.text for _s, lab_ in w.legend.items] == OLD_LEGEND, [lab_.text for _s, lab_ in w.legend.items]
            # the same report with and without its cluster profile writes byte-identical existing files
            bare = copy.copy(rep)
            bare.clusters = None
            fa, fb = os.path.join(WORK, f"with_{name}"), os.path.join(WORK, f"without_{name}")
            a = zw.write_report_files(rep, fa)
            b = zw.write_report_files(bare, fb)
            assert len(a) == len(b) == 5
            for x, y in zip(a, b):
                assert os.path.basename(x) == os.path.basename(y)
                assert open(x, "rb").read() == open(y, "rb").read(), os.path.basename(x)
            # and compute_z_quality_clusters leaves every rule-v2 field of a fresh report as it was
            fresh = zw.compute_z_quality(res, (inp.x_nm, inp.y_nm, inp.z_nm))
            snap = (list(fresh.warnings), fresh.seconds, fresh.computed_at, zw.plain_summary(fresh))
            zw.compute_z_quality_clusters(res, (inp.x_nm, inp.y_nm, inp.z_nm), fresh)
            assert fresh.clusters is not None
            assert snap == (list(fresh.warnings), fresh.seconds, fresh.computed_at, zw.plain_summary(fresh))
            return f"verdicts {[p.verdict for p in rep.v2.pairs]}; 5 files byte-identical; old legend kept"
        return fn

    for name in ("viable", "marginal", "none"):
        check(f"{name}: rule v2, the old legend and the existing files unchanged", unchanged(name))

    def old_export() -> str:
        w = need(st, "view_none")
        folder = os.path.join(WORK, "export")
        a = w.export_report(folder)
        assert len(a) == 7, len(a)
        assert not any("clusters" in os.path.basename(p) for p in a)
        return f"{len(a)} files, none of them a cluster file"

    check("export_report still writes exactly its 7 files", old_export)

    print("\n4. The new export files")

    def new_export() -> str:
        w = need(st, "view_none")
        rep = w.report
        folder = os.path.join(WORK, "export")
        first = [p for p in os.listdir(folder) if p.endswith("_zquality_rings.csv")][0]
        stem = first[:-len("_zquality_rings.csv")]
        a = w.export_cluster_report(folder, stem=stem)
        b = w.export_cluster_report(folder, stem=stem)
        assert len(a) == 2 and len(b) == 2 and not set(a) & set(b), (a, b)
        assert all(os.path.basename(p).startswith(stem + "_zquality_clusters") for p in a), a
        png = [p for p in a if p.endswith(".png")][0]
        assert png.endswith("_zquality_clusters_histogram.png") and os.path.getsize(png) > 5000
        with open([p for p in a if p.endswith(".csv")][0], encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        n = {k: s.n_clusters for k, s in rep.clusters.sets.items()}
        assert len(rows) == sum(n.values())
        for k in n:
            sub = [r for r in rows if r["cluster_set"] == k]
            assert len(sub) == n[k]
            med = np.array([float(r["median_z_nm"]) for r in sub])
            assert np.allclose(np.sort(med), np.sort(rep.clusters.sets[k].median_z_nm))
        need_cols = {"axon_id", "cluster_set", "cluster", "median_z_nm", "q25_z_nm", "q75_z_nm", "n_locs",
                     "ring_slab_of_median", "hist_bin_nm", "sizer_h_nm"}
        assert need_cols <= set(rows[0]), set(rows[0])
        return f"{[os.path.basename(p) for p in a]}, then a _2 set; {len(rows)} rows ({n})"

    check("export_cluster_report: medians CSV + panel PNG, never overwritten", new_export)

    def no_clusters() -> str:
        w = need(st, "view_viable")
        rep = copy.copy(w.report)
        rep.clusters = None
        w.set_report(rep)
        pump(app, 0.05)
        assert w.cluster_panel.isHidden()
        assert w.export_cluster_report(os.path.join(WORK, "none")) == []
        assert zw.write_cluster_report_files(rep, os.path.join(WORK, "none")) == []
        return "panel hidden, no cluster files"

    check("a report without a cluster profile: the panel hidden, no new files", no_clusters)

    for name in ("viable", "marginal", "none"):
        if f"view_{name}" in st:
            st[f"view_{name}"].close()
    print(f"\n{PASSED} passed, {FAILED} failed in {time.perf_counter() - T0:.0f} s")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
