# -*- coding: utf-8 -*-
"""
Headless tests of the clusters cut by the Picasso pick (decision P-R59; docs/cut_clusters.md).

Synthetic data only (R8): a ring of Gaussian clusters drawn here, and pick files written here in Picasso v0.9.10's
format (``picasso/gui/render.py``: ``save_picks`` and ``save_picked_locs``).

  1. the rule on geometry: a cluster far inside, one at the edge, one with a localization ON the edge, polygon and
     circle; a boundary that does not hold the localizations is refused;
  2. the multiple of NeNA: a cluster 1.5 NeNA from the edge is cut at 2 NeNA, not at 1 or 0.5;
  3. no boundary / no NeNA / no context -> "not evaluated", with the reason, never a number;
  4. the switch: flagged only, every number equals the analysis without the rule; excluded, the cut clusters are
     left out (and counted as removed), the analysis id changes;
  5. the export: the cut_ columns last in the axon row, cut_by_pick last in the clusters table, the dictionary
     describes them, a row without a boundary says "boundary unknown: not evaluated";
  6. the pick reader: polygon (with an unclosed one), circle (Diameter (nm) and the older Diameter in pixels),
     rectangle, square; a localization file's metadata and a malformed file are not pick files;
  7. the match: unique, several picks that the file's own record (group, shape, area) tells apart, copies of one
     pick file, an ambiguous match (asked, never chosen), a contradicted one, none;
  8. the session: a match by hand, a candidate chosen, the NeNA stored for the folder before the file's own, a
     boundary set aside;
  9. the derivation's number: Phi(-1/sqrt 2) = 0.24.

Run:  venv\\Scripts\\python.exe test_cut_clusters.py      (a few seconds)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from tools import mps_cut_clusters as cut_rule  # noqa: E402
from tools import mps_picks  # noqa: E402

PASSED = 0
FAILED = 0
T0 = time.perf_counter()
PIXEL = 130.0


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-6:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


# --------------------------------------------------------------------------- synthetic data
def ring(seed: int = 0, n_clusters: int = 24, radius: float = 900.0, centre=(5000.0, 5000.0),
         per_cluster: int = 40, spread: float = 12.0, sigma: float = 8.0) -> Dict[str, Any]:
    """A ring of clusters in nm (x, y, z), with the true cluster of each localization."""
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    cx = centre[0] + radius * np.cos(ang)
    cy = centre[1] + radius * np.sin(ang)
    xs, ys, truth = [], [], []
    for k in range(n_clusters):
        mol = rng.normal(0, spread, (per_cluster // 4, 2))
        pos = np.repeat(mol, 4, axis=0) + rng.normal(0, sigma, (per_cluster // 4 * 4, 2))
        xs.append(cx[k] + pos[:, 0])
        ys.append(cy[k] + pos[:, 1])
        truth.append(np.full(pos.shape[0], k))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = rng.normal(0.0, 25.0, x.size)
    return dict(x=x, y=y, z=z, truth=np.concatenate(truth), cx=cx, cy=cy)


def cutting_polygon_nm(centre=(5000.0, 5000.0)) -> np.ndarray:
    """A square around the ring whose right side runs through it (x = centre + 880 nm)."""
    c = np.asarray(centre)
    return np.array([[c[0] - 1500, c[1] - 1500], [c[0] + 880, c[1] - 1500], [c[0] + 880, c[1] + 1500],
                     [c[0] - 1500, c[1] + 1500]])


def boundary(vertices_nm: np.ndarray) -> mps_picks.PickBoundary:
    return mps_picks.PickBoundary(kind="polygon", shape="Polygon", source="test", vertices_nm=vertices_nm)


def write_pick_yaml(path: str, doc: Dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f)
    return path


class FakeLoc:
    """What open_session reads of a tools.mps_io.Localizations."""

    def __init__(self, x_nm: np.ndarray, y_nm: np.ndarray, *, group: Optional[np.ndarray] = None,
                 info: Optional[List[Dict[str, Any]]] = None, frame: Optional[np.ndarray] = None,
                 path: str = "axon_picked.hdf5") -> None:
        self.x_nm = np.asarray(x_nm, float)
        self.y_nm = np.asarray(y_nm, float)
        self.pixel_size_nm = PIXEL
        self.fileformat = 0
        self.path = path
        self.columns: Dict[str, Any] = {"x": (self.x_nm / PIXEL).astype(np.float32),
                                        "y": (self.y_nm / PIXEL).astype(np.float32)}
        if group is not None:
            self.columns["group"] = np.asarray(group)
        if frame is not None:
            self.columns["frame"] = np.asarray(frame)
        self.info = info or []

    def column(self, name: str) -> Any:
        return self.columns.get(name)

    @property
    def frame(self) -> Any:
        f = self.columns.get("frame")
        return None if f is None else np.asarray(f, np.int64)


# --------------------------------------------------------------------------- 1-3. the rule
def rule_on_geometry() -> str:
    labels = np.array([0] * 4 + [1] * 4 + [2] * 3)
    # cluster 0 far inside, cluster 1 with one localization 5 nm from the edge x = 100, cluster 2 with one ON it
    x = np.array([10, 12, 14, 16, 60, 70, 80, 95, 50, 60, 100.0])
    y = np.array([50, 52, 48, 50, 50, 50, 50, 50, 20, 20, 20.0])
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100.0]])
    ctx = cut_rule.CutContext(boundary=boundary(square), nena_nm=8.0, nena_source="test")
    rep = cut_rule.evaluate(x, y, labels, [0, 1, 2], ctx)
    assert rep.evaluated and rep.cut_labels == frozenset({1, 2}), rep.cut_labels
    assert rep.count_words() == "2 of 3 (67 %)", rep.count_words()
    assert abs(rep.min_distance_nm[0] - 10.0) < 1e-9 and rep.min_distance_nm[2] == 0.0
    # circle of radius 100 centred at (0, 0): a localization at r = 95 is 5 nm in
    circ = mps_picks.PickBoundary(kind="circle", shape="Circle", source="t", centre_nm=(0.0, 0.0), radius_nm=100.0)
    xc = np.array([0, 5, -5, 95.0, 90.0])
    yc = np.array([0, 0, 5, 0.0, 0.0])
    lc = np.array([0, 0, 0, 1, 1])
    rc = cut_rule.evaluate(xc, yc, lc, [0, 1], cut_rule.CutContext(boundary=circ, nena_nm=8.0))
    assert rc.cut_labels == frozenset({1}), rc.cut_labels
    # a localization 50 nm outside: the boundary is not this file's
    xo = np.append(x, 150.0)
    yo = np.append(y, 50.0)
    lo = np.append(labels, 0)
    ro = cut_rule.evaluate(xo, yo, lo, [0, 1, 2], ctx)
    assert ro.status == cut_rule.STATUS_NOT_HELD and not ro.evaluated and ro.k is None
    # only the clusters the curation kept are judged
    rk = cut_rule.evaluate(x, y, labels, [0, 1], ctx)
    assert rk.cut_labels == frozenset({1}) and rk.n_considered == 2
    return "polygon 2 of 3 (edge 5 nm and on the edge), circle 1 of 2, outside refused, curated left alone"


def nena_multiple() -> str:
    # cluster 1's closest localization is 12 nm (1.5 x 8 nm) from the edge
    labels = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([20, 22, 24, 70, 80, 88.0])
    y = np.array([50.0] * 6)
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100.0]])
    out = {}
    for m in cut_rule.SENSITIVITY_MULTIPLES:
        ctx = cut_rule.CutContext(boundary=boundary(square), nena_nm=8.0, multiple=m)
        out[m] = cut_rule.evaluate(x, y, labels, [0, 1], ctx).cut_labels
    assert out[0.5] == frozenset() and out[1.0] == frozenset() and out[2.0] == frozenset({1}), out
    assert cut_rule.DEFAULT_NENA_MULTIPLE == 1.0
    return "1.5 NeNA from the edge: cut at 2, not at 1 or 0.5; the default stays 1"


def not_evaluated() -> str:
    labels = np.array([0, 0, 1, 1])
    x = np.array([10, 12, 95, 96.0])
    y = np.array([50, 50, 50, 50.0])
    r0 = cut_rule.evaluate(x, y, labels, [0, 1], None)
    assert r0.status == cut_rule.STATUS_NO_BOUNDARY and r0.k is None and r0.percent is None
    r1 = cut_rule.evaluate(x, y, labels, [0, 1], cut_rule.CutContext(reason="no pick holds every localization"))
    assert r1.status == cut_rule.STATUS_NO_BOUNDARY and "no pick holds" in r1.reason
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100.0]])
    r2 = cut_rule.evaluate(x, y, labels, [0, 1], cut_rule.CutContext(boundary=boundary(square)))
    assert r2.status == cut_rule.STATUS_NO_NENA and r2.k is None
    cols = cut_rule.export_columns(None)
    assert cols["cut_status"].startswith("boundary unknown: not evaluated")
    assert cols["n_clusters_cut"] is None and cols["cut_percent"] is None and cols["cut_excluded"] is False
    assert cols["cut_rule"] is None and cols["cut_margin_nm"] is None
    cols2 = cut_rule.export_columns(r2)
    assert cols2["cut_status"].startswith("NeNA unknown: not evaluated")
    return "no context / no boundary / no NeNA: status and reason, no count, no rule"


# --------------------------------------------------------------------------- 4-5. the analysis and the export
def _analysis_inputs() -> Dict[str, Any]:
    r = ring(seed=3)
    poly = cutting_polygon_nm()
    keep = mps_picks.polygon_contains(r["x"], r["y"], poly)
    return dict(x=r["x"][keep], y=r["y"][keep], z=r["z"][keep], poly=poly, truth=r["truth"][keep])


def exclusion_switch() -> str:
    from tools.axon_export import analysis_id
    from tools.mps_analysis import analyze_axon

    d = _analysis_inputs()
    ctx = cut_rule.CutContext(boundary=boundary(d["poly"]), nena_nm=8.0, nena_source="test NeNA")
    kw = dict(run_randomization=False, source_name="ring.hdf5", pixel_size_nm=PIXEL, pixel_size_source="yaml")
    plain = analyze_axon(d["x"], d["y"], d["z"], **kw)
    flagged = analyze_axon(d["x"], d["y"], d["z"], cut_context=ctx, **kw)
    excluded = analyze_axon(d["x"], d["y"], d["z"], cut_context=ctx, exclude_cut_clusters=True, **kw)
    assert plain.cut is None
    c = flagged.cut
    assert c is not None and c.evaluated and c.k is not None and c.k >= 1, c
    # flagged only: the numbers are those of the analysis without the rule
    for attr in ("n_clusters_kept", "perimeter_um", "median_area_nm2", "median_1nn_nm", "occupancy_percent"):
        assert getattr(flagged, attr) == getattr(plain, attr), attr
    assert analysis_id(flagged) == analysis_id(plain)
    # excluded: the cut ones are gone and counted as removed
    assert excluded.cut is not None and excluded.cut.excluded
    assert excluded.n_clusters_kept == plain.n_clusters_kept - c.k
    assert excluded.bad_report.cut_by_pick == set(c.cut_labels)
    assert set(c.cut_labels) <= excluded.bad_report.bad_labels
    assert excluded.perimeter_um != plain.perimeter_um
    assert analysis_id(excluded) != analysis_id(plain)
    rows = {r[0]: r for r in excluded.summary_rows()}
    assert rows["Clusters cut by the pick"][1].startswith(f"{c.k} of {c.n_considered}")
    assert "left out of the analysis" in rows["Clusters cut by the pick"][3]
    assert any("left out of the analysis" in w for w in excluded.warnings)
    assert {r[0]: r for r in plain.summary_rows()}["Clusters cut by the pick"][1] == cut_rule.STATUS_NO_BOUNDARY
    # truth: the cut clusters are the ones at the right side of the ring
    lab = flagged.labels
    cut_x = [float(np.mean(flagged.x_slab[lab == k])) for k in c.cut_labels]
    assert min(cut_x) > 5000.0 + 600.0, cut_x
    return (f"{c.count_words()} cut; flagged: numbers unchanged; excluded: {excluded.n_clusters_kept} kept of "
            f"{plain.n_clusters_kept}, perimeter {plain.perimeter_um:.2f} -> {excluded.perimeter_um:.2f} um")


def export_columns() -> str:
    from tools import axon_export, column_dictionary
    from tools.mps_analysis import analyze_axon

    d = _analysis_inputs()
    ctx = cut_rule.CutContext(boundary=boundary(d["poly"]), nena_nm=8.0, nena_source="whole-field file w.hdf5")
    kw = dict(run_randomization=False, source_name="ring.hdf5", pixel_size_nm=PIXEL, pixel_size_source="yaml")
    a = analyze_axon(d["x"], d["y"], d["z"], cut_context=ctx, **kw)
    row = axon_export.axon_row(a, exported_at="2026-10-14T00:00:00Z")
    names = list(row)
    assert names[-len(axon_export.CUT_COLUMNS):] == list(axon_export.CUT_COLUMNS), names[-14:]
    assert row["table_version"] == "axon tables v6"
    assert row["cut_status"] == "evaluated" and row["cut_nena_nm"] == 8.0 and row["cut_margin_nm"] == 8.0
    assert row["cut_nena_multiple"] == 1.0 and row["cut_excluded"] is False
    assert row["n_clusters_cut"] == a.cut.k and row["n_clusters_cut_of"] == a.n_clusters_kept
    assert row["cut_boundary_shape"] == "Polygon" and row["cut_nena_source"] == "whole-field file w.hdf5"
    assert row["cut_rule"] == "cut if any localization lies within 1 NeNA of the pick boundary"
    assert abs(row["cut_percent"] - 100.0 * a.cut.k / a.n_clusters_kept) < 0.01
    clusters = axon_export.cluster_table(a)
    assert list(clusters[0])[-1] == "cut_by_pick"
    flagged = sorted(int(r["cluster_label"]) for r in clusters if r["cut_by_pick"])
    assert flagged == sorted(a.cut.cut_labels)
    assert column_dictionary.missing(list(row)) == [] and column_dictionary.missing(list(clusters[0])) == []
    plain = analyze_axon(d["x"], d["y"], d["z"], **kw)
    row0 = axon_export.axon_row(plain, exported_at="2026-10-14T00:00:00Z")
    assert row0["cut_status"].startswith("boundary unknown: not evaluated") and row0["n_clusters_cut"] is None
    assert all(r["cut_by_pick"] is None for r in axon_export.cluster_table(plain))
    ex = analyze_axon(d["x"], d["y"], d["z"], cut_context=ctx, exclude_cut_clusters=True, **kw)
    row_ex = axon_export.axon_row(ex, exported_at="2026-10-14T00:00:00Z")
    assert row_ex["cut_excluded"] is True and row_ex["n_clusters_removed"] >= a.cut.k
    assert not any(r["cut_by_pick"] for r in axon_export.cluster_table(ex))
    return (f"{len(axon_export.CUT_COLUMNS)} cut_ columns last, cut_by_pick last in the clusters table; "
            f"'{row0['cut_status'][:32]}...' without a boundary")


# --------------------------------------------------------------------------- 6. the reader
def pick_reader() -> str:
    work = tempfile.mkdtemp(prefix="test_cut_reader_")
    poly = write_pick_yaml(os.path.join(work, "poly.yaml"), {
        "Shape": "Polygon",
        "Vertices": [[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0]],
                     [[30.0, 30.0], [40.0, 30.0], [40.0, 40.0]],                       # unclosed
                     [[50.0, 50.0], [60.0, 50.0], [55.0, 60.0], [50.0, 50.0]]]})
    pf = mps_picks.read_pick_file(poly, PIXEL)
    assert [p.index for p in pf.picks] == [0, 1, 2] and [p.closed for p in pf.picks] == [True, False, True]
    assert pf.picks[0].closed_rank == 0 and pf.picks[2].closed_rank == 1 and pf.warnings
    assert abs(pf.picks[0].area_px2() - 100.0) < 1e-9 and abs(pf.picks[2].area_px2() - 50.0) < 1e-9
    b = pf.picks[0].boundary(PIXEL, "t")
    assert np.allclose(b.vertices_nm[1], [20 * PIXEL, 10 * PIXEL])
    circ = write_pick_yaml(os.path.join(work, "circ.yaml"), {"Shape": "Circle", "Diameter (nm)": 1300.0,
                                                             "Centers": [[5.0, 5.0], [15.0, 5.0]]})
    pc = mps_picks.read_pick_file(circ, PIXEL)
    assert len(pc.picks) == 2 and abs(pc.picks[0].radius_px - 5.0) < 1e-9
    try:
        mps_picks.read_pick_file(circ, None)
        raise AssertionError("a diameter in nm needs a pixel size")
    except mps_picks.PickFileError:
        pass
    old = write_pick_yaml(os.path.join(work, "old.yaml"), {"Diameter": 2.0, "Centers": [[5.0, 5.0]]})
    po = mps_picks.read_pick_file(old, None)
    assert po.shape == "Circle" and po.picks[0].radius_px == 1.0 and "camera pixels" in po.size_words
    rect = write_pick_yaml(os.path.join(work, "rect.yaml"), {"Shape": "Rectangle", "Width (nm)": 260.0,
                                                             "Center-Axis-Points": [[[0.0, 0.0], [10.0, 0.0]]]})
    pr = mps_picks.read_pick_file(rect, PIXEL)
    assert abs(pr.picks[0].area_px2() - 20.0) < 1e-9
    assert pr.picks[0].contains_px(np.array([5.0, 5.0]), np.array([0.5, 1.5])).tolist() == [True, False]
    sq = write_pick_yaml(os.path.join(work, "sq.yaml"), {"Shape": "Square", "Side Length (nm)": 520.0,
                                                         "Centers": [[10.0, 10.0]]})
    ps = mps_picks.read_pick_file(sq, PIXEL)
    assert abs(ps.picks[0].area_px2() - 16.0) < 1e-9
    sidecar = os.path.join(work, "axon.yaml")
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write("Frames: 100\n---\nGenerated by: Picasso Localize\nPixelsize: 130\n")
    single = write_pick_yaml(os.path.join(work, "meta.yaml"), {"Pixelsize": 130, "Generated by": "x"})
    bad = os.path.join(work, "bad.yaml")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("Shape: Polygon\nVertices: [[[1, 2, 3]]]\n")
    assert not mps_picks.looks_like_pick_file(sidecar) and not mps_picks.looks_like_pick_file(single)
    for path in (sidecar, single, bad):
        try:
            mps_picks.read_pick_file(path, PIXEL)
            raise AssertionError(f"{os.path.basename(path)} read as a pick file")
        except mps_picks.PickFileError:
            pass
    found = mps_picks.candidate_files(os.path.join(work, "axon.hdf5"))
    names = sorted(os.path.basename(p) for p in found)
    # bad.yaml says it is a pick file; reading it fails, and the match reports it
    assert names == ["bad.yaml", "circ.yaml", "old.yaml", "poly.yaml", "rect.yaml", "sq.yaml"], names
    m = mps_picks.match_pick(np.array([12.0]), np.array([12.0]), [bad], pixel_size_nm=PIXEL)
    assert m.status == "none" and "could not be used" in m.message, m.message
    return "polygon (unclosed skipped), circle in nm and in pixels, rectangle, square; sidecars refused"


# --------------------------------------------------------------------------- 7. the match
def _picked_setup(work: str) -> Dict[str, Any]:
    """The ring, picked in Picasso with polygon 1 of a file of three; polygon 0 elsewhere, polygon 2 holds the
    whole ring too (a larger pick drawn around the same axon)."""
    r = ring(seed=5)
    poly1_nm = cutting_polygon_nm()
    poly2_nm = poly1_nm + np.array([[-200, -200], [400, -200], [400, 200], [-200, 200]])
    poly0_nm = poly1_nm + 20000.0
    verts = []
    for v in (poly0_nm, poly1_nm, poly2_nm):
        px = (v / PIXEL).tolist()
        verts.append(px + [px[0]])
    path = write_pick_yaml(os.path.join(work, "picks.yaml"), {"Shape": "Polygon", "Vertices": verts})
    keep = mps_picks.polygon_contains(r["x"], r["y"], poly1_nm)
    areas = [mps_picks.polygon_area(v / PIXEL) * (PIXEL * 1e-3) ** 2 for v in (poly0_nm, poly1_nm, poly2_nm)]
    info = [{"Generated by": "Picasso Localize", "Pixelsize": PIXEL},
            {"Generated by": "Picasso v0.9.10 Render : Pick", "Pick Shape": "Polygon",
             "Pick Areas (um^2)": areas, "Area (um^2)": float(sum(areas)), "Number of picks": 3}]
    return dict(r=r, keep=keep, path=path, poly1=poly1_nm, info=info, x=r["x"][keep], y=r["y"][keep])


def matching() -> str:
    work = tempfile.mkdtemp(prefix="test_cut_match_")
    s = _picked_setup(work)
    xpx, ypx = (s["x"] / PIXEL).astype(np.float32), (s["y"] / PIXEL).astype(np.float32)
    # no record at all: two picks hold every localization -> ambiguous, never chosen
    m0 = mps_picks.match_pick(xpx, ypx, [s["path"]], pixel_size_nm=PIXEL)
    assert m0.status == "ambiguous" and m0.chosen is None and len(m0.candidates) == 2, m0.message
    assert m0.boundary(PIXEL) is None
    # the group column says pick 1 -> matched
    rec = mps_picks.picked_record(np.ones(xpx.size, int), s["info"])
    m1 = mps_picks.match_pick(xpx, ypx, [s["path"]], record=rec, pixel_size_nm=PIXEL)
    assert m1.status == "matched" and m1.chosen.pick.index == 1, m1.message
    assert m1.chosen.group_agrees and m1.chosen.shape_agrees and m1.chosen.area_agrees
    b = m1.boundary(PIXEL)
    assert b is not None and np.allclose(b.vertices_nm, s["poly1"], atol=1e-6) and "found automatically" in b.source
    # the list of every pick's area (no group column) checks the file, and does not tell them apart
    rec_area = mps_picks.picked_record(None, s["info"])
    m2 = mps_picks.match_pick(xpx, ypx, [s["path"]], record=rec_area, pixel_size_nm=PIXEL)
    assert m2.status == "ambiguous" and all(c.area_agrees for c in m2.candidates), m2.message
    # one pick per file ("Area (um^2)" of that pick only): the area tells them apart
    sep = [s["info"][0], {"Generated by": "Picasso v0.9.10 Render : Pick", "Pick Shape": "Polygon",
                          "Area (um^2)": s["info"][1]["Pick Areas (um^2)"][1]}]
    m2b = mps_picks.match_pick(xpx, ypx, [s["path"]], record=mps_picks.picked_record(None, sep),
                               pixel_size_nm=PIXEL)
    assert m2b.status == "matched" and m2b.chosen.pick.index == 1 and m2b.chosen.group_agrees is None
    # a copy of the same pick file: still one pick
    copy = os.path.join(work, "picks_copy.yaml")
    with open(s["path"], encoding="utf-8") as f, open(copy, "w", encoding="utf-8") as g:
        g.write(f.read())
    m3 = mps_picks.match_pick(xpx, ypx, [s["path"], copy], record=rec, pixel_size_nm=PIXEL)
    assert m3.status == "matched" and "2 files hold this same pick" in m3.message
    # the record contradicts every candidate (group 7): asked, not chosen
    rec_bad = mps_picks.picked_record(np.full(xpx.size, 7), [])
    m4 = mps_picks.match_pick(xpx, ypx, [s["path"]], record=rec_bad, pixel_size_nm=PIXEL)
    assert m4.status == "ambiguous" and m4.chosen is None and "contradicts" in m4.message
    # localizations no pick holds
    m5 = mps_picks.match_pick(xpx + 500, ypx, [s["path"]], record=rec, pixel_size_nm=PIXEL)
    assert m5.status == "none" and "no pick holds" in m5.message
    m6 = mps_picks.match_pick(xpx, ypx, [], record=rec, pixel_size_nm=PIXEL)
    assert m6.status == "none" and "no Picasso pick file found" in m6.message
    return "ambiguous without a record; group or area decides; copies are one pick; contradiction asked; none"


def session() -> str:
    work = tempfile.mkdtemp(prefix="test_cut_session_")
    s = _picked_setup(work)
    axon_path = os.path.join(work, "axon_picked.hdf5")
    loc = FakeLoc(s["x"], s["y"], path=axon_path)            # no group, no metadata: ambiguous
    ses = cut_rule.open_session(axon_path, loc, {})
    assert ses.match.status == "ambiguous" and ses.boundary is None and ses.worth_passing()
    ctx = ses.context()
    assert ctx is not None and ctx.boundary is None and "ambiguous" in ctx.reason
    cut_rule.choose_candidate(ses, [c.pick.index for c in ses.match.candidates].index(1))
    assert ses.boundary is not None and "chosen by hand" in ses.boundary.source
    # NeNA: the file has no frames; a value stored for the folder is used, and named
    ses.nena = cut_rule.resolve_nena(ses, loc, {})
    assert not ses.nena.measured and "no frame numbers" in ses.nena.reason
    c2 = ses.context()
    assert c2 is not None and c2.boundary is not None and c2.nena_nm is None and "whole-field" in c2.reason
    entry = cut_rule.nena_entry(cut_rule.NenaSource(nena_nm=9.5, words="whole-field file field.hdf5 (2,000 "
                                                                       "next-frame pairs)", n_pairs=2000,
                                                    source_file="field.hdf5"))
    ses.nena = cut_rule.resolve_nena(ses, loc, entry)
    assert ses.nena.measured and ses.nena.nena_nm == 9.5 and ses.nena.words.startswith("stored for this folder")
    c3 = ses.context()
    assert c3 is not None and c3.usable and c3.nena_nm == 9.5
    # a file with nothing beside it: nothing to pass
    lonely = tempfile.mkdtemp(prefix="test_cut_lonely_")
    inner = os.path.join(lonely, "a", "b")
    os.makedirs(inner)
    s0 = cut_rule.open_session(os.path.join(inner, "x.hdf5"), loc, {})
    assert s0.match.status == "none" and not s0.worth_passing() and s0.context() is None
    # by hand: the pick file pointed to
    cut_rule.choose_pick_file(s0, FakeLoc(s["x"], s["y"], group=np.ones(s["x"].size, int)), s["path"])
    assert s0.match.status == "matched" and s0.match.by_hand and s0.worth_passing()
    s0.set_aside = True
    c4 = s0.context()
    assert c4 is not None and c4.boundary is None and "set aside" in c4.reason
    # the store is cleaned
    cleaned = cut_rule.clean_store({"f": {"nena_nm": -3, "pick_files": ["a.yaml", 3]}, "g": "x",
                                    "h": dict(entry)})
    assert cleaned == {"f": {"pick_files": ["a.yaml"]}, "h": {**entry}}, cleaned
    return "ambiguous -> chosen; no frames -> stored folder NeNA; nothing beside -> None; by hand; set aside"


def derivation_number() -> str:
    p1 = cut_rule.sibling_outside_probability(1.0)
    p2 = cut_rule.sibling_outside_probability(2.0)
    p05 = cut_rule.sibling_outside_probability(0.5)
    assert abs(p1 - 0.2398) < 1e-3 and abs(p2 - 0.0786) < 1e-3 and abs(p05 - 0.3618) < 1e-3, (p05, p1, p2)
    return f"P(sibling beyond the edge) at 0.5 / 1 / 2 NeNA: {p05:.3f} / {p1:.3f} / {p2:.3f}"


def main() -> int:
    print("=" * 100)
    print("CLUSTERS CUT BY THE PICASSO PICK (P-R59), HEADLESS TESTS")
    print("=" * 100)
    check("the rule on geometry (polygon, circle, on the edge, outside)", rule_on_geometry)
    check("the NeNA multiple (0.5 / 1 / 2)", nena_multiple)
    check("no boundary, no NeNA: not evaluated", not_evaluated)
    check("the exclusion switch changes the analysis only when ticked", exclusion_switch)
    check("the export columns and the column dictionary", export_columns)
    check("the Picasso pick reader (v0.9.10 format)", pick_reader)
    check("matching the pick: unique, told apart, copies, ambiguous, contradicted, none", matching)
    check("the session: choose, by hand, stored NeNA, set aside", session)
    check("the derivation's numbers", derivation_number)
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
