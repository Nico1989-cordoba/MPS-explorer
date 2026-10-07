# -*- coding: utf-8 -*-
"""
Offscreen test of the Z quality view's rule v2-clusters table (H6 clusters IMPLEMENT, 2026-10-02: EXPLORATORY,
D-42 -- the four criteria of rule v2 read on the clusters, shown next to rule v2 and never selecting pairs), on the
three SIMULATED demo axons of tools.mps_zquality_window.DEMO_CASES only (R8):

  1. the table is shown under the pair table, one row per pair and cluster set (A, B, C), its verdicts and numbers
     equal tools.mps_axial_clusters.viability_v2c computed directly, every verdict labelled "v2-clusters", the
     tooltips (Q-38 said), and the summary view adds its block AFTER rule v2's text;
  2. nothing that existed changes: the ring and pair tables are the same with and without rule v2-clusters, rule
     v2's summary text and its 5 report files are byte-identical, export_report still writes its 7 files,
     compute_z_quality_v2c leaves every earlier field of the report as it was;
  3. the NEW export file <stem>_zquality_clusters_pairs.csv: its columns (SPEC_v2c section 5), one row per pair x
     cluster set x bandwidth, never overwritten; the full "Export z-quality report..." writes 7 + 2 + 1 files;
  4. a report without rule v2-clusters: the table hidden, no file.

Run:  venv\\Scripts\\python.exe test_zquality_v2c_gui.py      (offscreen; ~1 min)

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
WORK = tempfile.mkdtemp(prefix="test_zquality_v2c_gui_")
PAIRS_CSV_COLUMNS = ["axon_id", "source", "roi", "p_ref_nm", "computed_at", "program", "rule", "variant",
                     "variant_label", "ring_a", "ring_b", "h_over_p", "peak_a", "peak_b", "valley", "valley_at_cut",
                     "k_a", "k_b", "k_ratio_a", "k_ratio_b", "k_min", "count_ok", "exp_spur_frac_clusters",
                     "exp_spur_frac_d39", "n_points", "n_outside", "verdict", "reasons"]


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


def table_texts(t: Any) -> List[List[str]]:
    return [[(t.item(i, j).text() if t.item(i, j) is not None else "") for j in range(t.columnCount())]
            for i in range(t.rowCount())]


def main() -> int:
    print("=" * 100)
    print("Z-QUALITY VIEW: RULE v2-CLUSTERS TABLE (H6 clusters, EXPLORATORY) OFFSCREEN TEST on SIMULATED axons")
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
            assert rep.v2c is not None, rep.v2c_error
            return (f"{inp.x_nm.size:,} locs; v2 {[p.verdict for p in rep.v2.pairs]}; v2c "
                    + ", ".join(f"{k} {[p.verdict for p in rep.v2c.pairs_of(k)]}" for k in mac.V2C_VARIANT_ORDER)
                    + f"; {rep.v2c.seconds:.2f} s")
        return fn

    for name in ("viable", "marginal", "none"):
        check(f"{name}: the view with rule v2-clusters", open_case(name))

    print("\n1. The rule v2-clusters table")

    def table(name: str) -> Callable[[], str]:
        def fn() -> str:
            w = need(st, f"view_{name}")
            inp = need(st, f"inp_{name}")
            rep = w.report
            assert w.v2c_panel.isVisible() and w.v2c_table.isVisible() and w.v2c_title.isVisible()
            res, _p, _l, _w = build_review_rings(inp)
            xyz = (inp.x_nm, inp.y_nm, inp.z_nm)
            v2 = viability_v2(res, z_quality(res), xyz_lab_nm=xyz)
            direct = mac.viability_v2c(res, v2, xyz_lab_nm=xyz)
            n_pairs = len(rep.v2.pairs)
            assert w.v2c_table.rowCount() == 3 * n_pairs == len(rep.v2c.pairs)
            got = [(p.variant, p.ring_a, p.ring_b, p.verdict, p.peak_a, p.peak_b, p.valley_raw, p.valley_at_cut,
                    p.count_ok) for p in rep.v2c.pairs]
            want = [(p.variant, p.ring_a, p.ring_b, p.verdict, p.peak_a, p.peak_b, p.valley_raw, p.valley_at_cut,
                     p.count_ok) for p in direct.pairs]
            assert got == want, (got, want)
            assert np.allclose([lk.exp_spur_frac_clusters for lk in rep.v2c.leaks],
                               [lk.exp_spur_frac_clusters for lk in direct.leaks], rtol=1e-12, atol=0)
            # rule v2's D-39 number beside 4c is the pair table's
            assert [lk.exp_spur_frac_d39 for lk in rep.v2c.leaks] == [p.exp_spur_frac for p in rep.v2.pairs]
            texts = table_texts(w.v2c_table)
            vcol = zw.V2C_COLUMNS.index("Verdict (v2-clusters)")
            want_txt = {"viable": "VIABLE", "marginal": "MARGINAL", "not viable": "NOT VIABLE"}
            for row, p in zip(texts, rep.v2c.pairs):
                assert row[vcol].startswith("v2-clusters: ") and want_txt[p.verdict] in row[vcol], row[vcol]
                assert row[0] == f"{p.ring_a}-{p.ring_b}" and row[1] == zw.V2C_SHORT_NAMES[p.variant]
            assert [r[1][0] for r in texts] == ["A", "B", "C"] * n_pairs
            assert not w.missing_tooltips, w.missing_tooltips
            tip = w.v2c_table.toolTip()
            assert "EXPLORATORY" in tip and "Q-38" in tip and "never" in tip.lower()
            assert "never selects pairs" in w.v2c_title.text()
            summary = w.summary_view.toPlainText()
            base = "\n".join(zw.plain_summary(rep))
            assert summary.startswith(base + "\n"), "rule v2's summary text is no longer first and whole"
            assert "RULE v2-CLUSTERS (EXPLORATORY" in summary.split(base, 1)[1]
            return (f"{w.v2c_table.rowCount()} rows; verdicts {[p.verdict for p in rep.v2c.pairs]}; 4c "
                    f"{[round(100 * lk.exp_spur_frac_clusters, 1) for lk in rep.v2c.leaks]} %")
        return fn

    for name in ("viable", "marginal", "none"):
        check(f"{name}: one row per pair x cluster set, = viability_v2c, labelled, tooltips, summary block",
              table(name))

    print("\n2. What existed is unchanged")

    def unchanged(name: str) -> Callable[[], str]:
        def fn() -> str:
            w = need(st, f"view_{name}")
            inp = need(st, f"inp_{name}")
            rep = w.report
            rings_with, pairs_with = table_texts(w.ring_table), table_texts(w.pair_table)
            bare = copy.copy(rep)
            bare.v2c = None
            w.set_report(bare)
            pump(app, 0.05)
            assert w.v2c_panel.isHidden()
            assert table_texts(w.ring_table) == rings_with and table_texts(w.pair_table) == pairs_with
            assert w.summary_view.toPlainText() == "\n".join(zw.plain_summary(bare))
            assert zw.write_v2c_report_files(bare, os.path.join(WORK, "nov2c")) == []
            assert w.export_v2c_report(os.path.join(WORK, "nov2c")) == []
            w.set_report(rep)
            pump(app, 0.05)
            assert w.v2c_table.isVisible()
            fa, fb = os.path.join(WORK, f"with_{name}"), os.path.join(WORK, f"without_{name}")
            a = zw.write_report_files(rep, fa)
            b = zw.write_report_files(bare, fb)
            assert len(a) == len(b) == 5
            for x, y in zip(a, b):
                assert os.path.basename(x) == os.path.basename(y)
                assert open(x, "rb").read() == open(y, "rb").read(), os.path.basename(x)
            res, _p, _l, _w = build_review_rings(inp)
            xyz = (inp.x_nm, inp.y_nm, inp.z_nm)
            fresh = zw.compute_z_quality_clusters(res, xyz, zw.compute_z_quality(res, xyz))
            snap = (list(fresh.warnings), fresh.seconds, fresh.computed_at, zw.plain_summary(fresh), fresh.clusters,
                    fresh.clusters_error, fresh.v2)
            zw.compute_z_quality_v2c(res, xyz, fresh)
            assert fresh.v2c is not None and fresh.v2c_error == ""
            assert snap == (list(fresh.warnings), fresh.seconds, fresh.computed_at, zw.plain_summary(fresh),
                            fresh.clusters, fresh.clusters_error, fresh.v2)
            return "ring/pair tables, summary text and 5 files identical; earlier report fields untouched"
        return fn

    for name in ("viable", "marginal", "none"):
        check(f"{name}: rule v2's tables, summary and files unchanged", unchanged(name))

    def old_export() -> str:
        w = need(st, "view_marginal")
        a = w.export_report(os.path.join(WORK, "export"))
        assert len(a) == 7 and not any("clusters" in os.path.basename(p) for p in a), a
        return f"{len(a)} files"

    check("export_report still writes exactly its 7 files", old_export)

    print("\n3. The new export file")

    def new_export() -> str:
        w = need(st, "view_marginal")
        rep = w.report
        folder = os.path.join(WORK, "export")
        first = [p for p in os.listdir(folder) if p.endswith("_zquality_rings.csv")][0]
        stem = first[:-len("_zquality_rings.csv")]
        a = w.export_v2c_report(folder, stem=stem)
        b = w.export_v2c_report(folder, stem=stem)
        assert len(a) == len(b) == 1 and a != b, (a, b)
        assert os.path.basename(a[0]) == f"{stem}_zquality_clusters_pairs.csv", a
        assert os.path.basename(b[0]) == f"{stem}_2_zquality_clusters_pairs.csv", b
        with open(a[0], encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert list(rows[0]) == PAIRS_CSV_COLUMNS, list(rows[0])
        n_pairs = len(rep.v2.pairs)
        assert len(rows) == 2 * 3 * n_pairs
        assert {r["rule"] for r in rows} == {mac.V2C_RULE} and {r["variant"] for r in rows} == {"A", "B", "C"}
        assert sorted({r["h_over_p"] for r in rows}) == ["0.1", "0.2"]
        main_rows = [r for r in rows if r["h_over_p"] == "0.1"]
        assert [r["verdict"] for r in main_rows] == [p.verdict for p in rep.v2c.pairs]
        assert all(r["reasons"].startswith("sensitivity only") for r in rows if r["h_over_p"] == "0.2")
        assert "n_chained" not in rows[0]
        # GUI-1's medians CSV keeps its columns
        cols = list(mac.cluster_median_rows(rep.clusters, {"axon_id": "x"})[0])
        assert cols == ["axon_id", "cluster_set", "cluster_set_label", "cluster", "median_z_nm", "q25_z_nm",
                        "q75_z_nm", "n_locs", "ring_slab_of_median", "hist_bin_nm", "sizer_h_nm", "params"], cols
        return f"{os.path.basename(a[0])}, then _2; {len(rows)} rows"

    check("export_v2c_report: the pairs CSV (SPEC columns), never overwritten", new_export)

    def full_export() -> str:
        w = need(st, "view_none")
        folder = os.path.join(WORK, "full")
        os.makedirs(folder, exist_ok=True)
        orig_dir, orig_info = QtWidgets.QFileDialog.getExistingDirectory, QtWidgets.QMessageBox.information
        seen: Dict[str, str] = {}
        try:
            setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: folder))
            setattr(QtWidgets.QMessageBox, "information",
                    staticmethod(lambda *a, **k: seen.setdefault("text", str(a[2]))))
            w._on_export()
        finally:
            setattr(QtWidgets.QFileDialog, "getExistingDirectory", orig_dir)
            setattr(QtWidgets.QMessageBox, "information", orig_info)
        files = sorted(os.listdir(folder))
        assert len(files) == 10, files
        assert sum(f.endswith("_zquality_clusters_pairs.csv") for f in files) == 1
        stems = {f.split("_zquality_")[0] for f in files}
        assert len(stems) == 1, stems
        assert "_zquality_clusters_pairs.csv" in seen.get("text", ""), seen
        return f"{len(files)} files with one stem"

    check("Export z-quality report...: 7 + 2 + 1 files with the same stem", full_export)

    for name in ("viable", "marginal", "none"):
        if f"view_{name}" in st:
            st[f"view_{name}"].close()
    print(f"\n{PASSED} passed, {FAILED} failed in {time.perf_counter() - T0:.0f} s")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
