"""
Scientific validation of the perimeter density analysis (package P11): shared helpers, the cross-package
tests and V1 (the automatic contour on the shared simulator).

    python validate_perimeter_cross.py cross --out DIR [--small]
    python validate_perimeter_cross.py v1 --bg 0.15 --out DIR [--seeds 20]
    python validate_perimeter_cross.py v1-gap --out DIR [--seeds 10]
    python validate_perimeter_cross.py v1-report --out DIR

Every validation ends in a PASS / FAIL table against the criterion of the implementation plan (sec. 4.4 and
5); a criterion that is not met is reported as FAIL with the number, never relaxed. Results (small CSV files and
a Markdown verdict table) go to ``--out``, which must be OUTSIDE the repository. Synthetic data only here; the
real axons are read by ``validate_perimeter_pilot.py``.

Cross tests (one function each, pure, returning rows):

* ``cross_bins_p4_p6``   the arc bin of ``mps_perimeter_density.density_profile`` (P4) is the bin of
                         ``mps_perimeter_bc.arc_bin_index`` and of the windows of question B (P6), on the real
                         ``density_profile`` / ``slice_profiles`` output, edges included.
* ``cross_leak_p1_p5``   the two independent derivations of the axial leak (P1 ``leak_matrix`` +
                         ``expected_slice_r``; P5 ``leak_expected_r``) against each other and against the
                         correlation measured on simulated axons.
* ``cross_halfbin_p1_p4`` P1's ``true_lambda`` lives on CELL STARTS of its arc table; P4 CENTRES its bins. The
                         expected counts per P4 bin are built both ways and compared with simulated counts.
* ``cross_empty_slice``  a lateral slice with 0 or 3 localizations: question C gives NaN (or a flagged value)
                         without an exception, end to end.
* ``cross_orientation``  the same contour given clockwise and counter-clockwise gives the same A, B, C.

Origin tags: [lit] literature, [own sim] this package's simulation, [plan] fixed by the implementation plan,
[ad hoc] chosen here.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tools import mps_perimeter_bc as BC  # noqa: E402
from tools import mps_perimeter_density as D  # noqa: E402
from tools import mps_perimeter_sim as S  # noqa: E402
from tools.mps_matching import SmoothPath  # noqa: E402
from tools.mps_perimeter_slices import cut_slices  # noqa: E402
from tools.mps_perimeter_types import SliceProfiles  # noqa: E402

ALPHA: float = 0.05                 # [lit] conventional level, as the analysis
FPR_BAND: Tuple[float, float] = (0.03, 0.07)   # [plan sec. 4.4/5] accepted false-positive band at alpha 0.05
POWER_MIN: float = 0.80             # [plan sec. 4.4/5]
BIN_NM: float = 100.0               # [director]
Row = Dict[str, Any]


# ----------------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson 95 % interval of a proportion k / n [lit: Wilson 1927]."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def rate_text(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.3f} (95% {lo:.3f}-{hi:.3f})" if n else "n = 0"


def ks_uniform(p: Sequence[float]) -> Tuple[float, float]:
    """One-sample Kolmogorov-Smirnov D of p against U(0, 1) and its asymptotic p-value [lit: Kolmogorov 1933;
    Stephens' small-n correction]. Monte Carlo p values are discrete (step 1 / (n_null + 1)), which makes D
    slightly larger than for a continuous variable; with >= 80 bins the step is <= 0.0125."""
    x = np.sort(np.asarray([v for v in p if math.isfinite(v)], dtype=np.float64))
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    d = float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    s = sum(((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return d, float(min(1.0, max(0.0, 2.0 * s)))


def write_csv(path: str, rows: Sequence[Mapping[str, Any]], append: bool = False) -> None:
    """Write rows (union of keys, in order of appearance); ``append`` merges with an existing file."""
    old: List[Dict[str, Any]] = []
    if append and os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            old = list(csv.DictReader(fh))
    allrows: List[Mapping[str, Any]] = list(old) + list(rows)
    keys: List[str] = []
    for r in allrows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in allrows:
            w.writerow({k: _cell(r.get(k)) for k in keys})


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cell(v: Any) -> Any:
    if isinstance(v, float):
        return "" if not math.isfinite(v) else round(v, 6)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return "" if not math.isfinite(f) else round(f, 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def fnum(v: Any) -> float:
    """A CSV cell back to float (empty = NaN)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def verdict(check: str, criterion: str, measured: str, ok: Optional[bool], note: str = "") -> Row:
    """One line of a verdict table; ``ok`` None = informative (no criterion in the plan)."""
    return {"check": check, "criterion": criterion, "measured": measured,
            "verdict": "INFO" if ok is None else ("PASS" if ok else "FAIL"), "note": note}


def write_verdicts(path: str, title: str, rows: Sequence[Row], lines: Iterable[str] = ()) -> None:
    """Markdown table of verdicts plus free lines (sizes, seeds, deviations)."""
    out = [f"# {title}", ""]
    out.extend(lines)
    out += ["", "| check | criterion | measured | verdict | note |", "|---|---|---|---|---|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")).replace("|", "/") for k in
                                     ("check", "criterion", "measured", "verdict", "note")) + " |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def radius_for_bins(n_bins: int, bin_nm: float = BIN_NM) -> float:
    """Radius of a circle whose perimeter holds ``n_bins`` nominal bins."""
    return n_bins * bin_nm / (2.0 * math.pi)


def profiles_of(sim: S.SimPerimeter, path: Optional[SmoothPath] = None, *, half_width_nm: float = 100.0,
                z0_nm: float = 0.0, guard_nm: float = 0.0, thickness_nm: float = 170.0
                ) -> Tuple[SliceProfiles, NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """The pipeline from projection to profiles (P2 slices, P4 projection / band / profiles) on ``path``
    (the TRUE curve when None). Returns (profiles, s, r, slice label)."""
    p = sim.true_path if path is None else path
    s, r = D.project(p, sim.x_nm, sim.y_nm)
    sl = cut_slices(sim.z_nm, z0_nm, thickness_nm=thickness_nm, guard_nm=guard_nm)
    sp = D.slice_profiles(s, r, sl.label, float(p.length_nm), half_width_nm=half_width_nm)
    return sp, s, r, sl.label


def ensure_out(out: str) -> str:
    out = os.path.abspath(out)
    if os.path.commonpath([out, REPO]) == REPO:
        raise SystemExit("--out must be outside the repository")
    os.makedirs(out, exist_ok=True)
    return out


# ----------------------------------------------------------------------------------------------------
# Cross test 1: P4 x P6 bin alignment on the real density_profile
# ----------------------------------------------------------------------------------------------------
def cross_bins_p4_p6(n_trials: int = 40, n_points: int = 3000, seed: int = 11) -> Tuple[List[Row], Row]:
    """Counts per bin of P4 versus ``arc_bin_index`` of P6, windows of P6 versus P4 bins, edges included."""
    rng = np.random.default_rng(seed)
    rows: List[Row] = []
    bad_counts = bad_windows = bad_profiles = 0
    for t in range(n_trials):
        L = float(rng.uniform(8000.0, 25000.0))
        n_bins, w = D.arc_grid(L)
        origin = float(rng.uniform(0.0, L))
        j_edge = rng.integers(0, n_bins, size=50)
        edges = np.mod(origin + (j_edge + 0.5) * w, L)                  # exactly on bin edges
        s = np.concatenate([rng.uniform(0.0, L, n_points), edges, np.mod(edges - 1e-9, L)])
        prof = D.density_profile(s, L, n_bins=n_bins, origin_arc_nm=origin)
        j6 = BC.arc_bin_index(s, length_nm=L, n_bins=n_bins, origin_arc_nm=prof.origin_arc_nm)
        mism = int(np.count_nonzero(np.bincount(j6, minlength=n_bins) != prof.counts))
        bad_counts += mism
        # windows: window i = bins i, i + 1 (window_counts) and spans [start_i, start_i + 2w) (tortuosity)
        starts = BC._window_starts(L, n_bins, prof.origin_arc_nm)
        sub = s[: min(400, s.size)]
        jj = BC.arc_bin_index(sub, length_nm=L, n_bins=n_bins, origin_arc_nm=prof.origin_arc_nm)
        rel = np.mod(sub[:, None] - starts[None, :], L)                  # (points, windows)
        inside = rel < 2.0 * w - 1e-7
        expect = np.zeros_like(inside)
        expect[np.arange(sub.size), jj] = True
        expect[np.arange(sub.size), (jj - 1) % n_bins] = True
        # points exactly on an edge may fall either side by 1e-9 of rounding: judge them away from edges only
        dist_edge = np.abs(((sub - prof.origin_arc_nm) / w + 0.5) % 1.0 - 0.5) * w
        far = np.abs(dist_edge - 0.5 * w) > 1e-6
        bad_w = int(np.count_nonzero((inside != expect)[far]))
        bad_windows += bad_w
        onehot = np.zeros(n_bins, dtype=np.int64)
        onehot[int(jj[0])] = 1
        wc = BC.window_counts(onehot, 2)
        bad_w2 = int(set(np.flatnonzero(wc).tolist()) != {int(jj[0]), int((jj[0] - 1) % n_bins)})
        bad_windows += bad_w2
        # the full chain: slice_profiles (origin at the max) and arc_bin_index on that origin
        lab = np.zeros(s.size, dtype=np.int8)
        sp = D.slice_profiles(s, np.zeros(s.size), lab, L, half_width_nm=100.0)
        p0 = sp.by_slice[0]
        j_sp = BC.arc_bin_index(s, length_nm=L, n_bins=n_bins, origin_arc_nm=p0.origin_arc_nm)
        bad_p = int(np.count_nonzero(np.bincount(j_sp, minlength=n_bins) != p0.counts))
        bad_profiles += bad_p
        rows.append({"trial": t, "length_nm": L, "n_bins": n_bins, "count_mismatch_bins": mism,
                     "window_mismatch": bad_w + bad_w2, "slice_profiles_mismatch_bins": bad_p})
    ok = bad_counts == 0 and bad_windows == 0 and bad_profiles == 0
    v = verdict("P4 x P6 bin alignment (real density_profile / slice_profiles)",
                "0 mismatched bins and windows, edges included",
                f"{n_trials} perimeters: {bad_counts} count bins, {bad_windows} windows, {bad_profiles} profile bins "
                "mismatched", ok)
    return rows, v


# ----------------------------------------------------------------------------------------------------
# Cross test 2: axial leak, P1 versus P5 versus simulation
# ----------------------------------------------------------------------------------------------------
def leak_theory(lpz_nm: float, guard_nm: float) -> Row:
    M, w = S.leak_matrix(lpz_nm=lpz_nm, guard_nm=guard_nm)
    R = S.expected_slice_r(M, w, 0.0)
    return {"lpz_nm": lpz_nm, "guard_nm": guard_nm, "p5_neighbour_r": BC.leak_expected_r(lpz_nm, 170.0, guard_nm),
            "p1_r_m1_0": float(R[0, 1]), "p1_r_0_p1": float(R[1, 2]), "p1_r_m1_p1": float(R[0, 2])}


def simulate_leak(n_axons: int, *, lpz_nm: float, n_locs: int, blink: bool, guard_nm: float,
                  seed0: int) -> List[Row]:
    """Independent random patterns per slice, axial error ``lpz_nm``; r of the three pairs on the true contour."""
    rows: List[Row] = []
    for i in range(n_axons):
        sim = S.simulate_perimeter("circle", {"kind": "random"}, n_locs=n_locs, radial_sd_nm=20.0,
                                   slices={"pattern": "independent"}, lpz_nm=lpz_nm, background_frac=0.0,
                                   interior_frac=0.0, blink=blink, seed=seed0 + i)
        sp, _, _, _ = profiles_of(sim, guard_nm=guard_nm)
        c = BC.slice_correlations(sp, n_null=199, seed=i)
        rows.append({"axon": i, "lpz_nm": lpz_nm, "blink": blink, "guard_nm": guard_nm, "n_locs": n_locs,
                     "r_m1_0": c[(-1, 0)].value, "r_0_p1": c[(0, 1)].value, "r_m1_p1": c[(-1, 1)].value})
    return rows


def cross_leak_p1_p5(n_axons: int = 60, n_locs: int = 60000) -> Tuple[List[Row], List[Row], List[Row]]:
    theory = [leak_theory(lpz, g) for lpz in (30.0, 47.0, 77.0) for g in (0.0, 42.5)]
    sims: List[Row] = []
    for blink in (False, True):
        for g in (0.0, 42.5):
            sims += simulate_leak(n_axons, lpz_nm=47.0, n_locs=n_locs, blink=blink, guard_nm=g,
                                  seed0=50000 + 1000 * int(blink) + int(g))
    verdicts: List[Row] = []
    t47 = [t for t in theory if t["lpz_nm"] == 47.0]
    for t in t47:
        g = t["guard_nm"]
        diff = abs(t["p5_neighbour_r"] - 0.5 * (t["p1_r_m1_0"] + t["p1_r_0_p1"]))
        verdicts.append(verdict(f"leak P1 vs P5, lpz 47, guard {g:g}", "|P1 - P5| <= 0.03 (P1's own tolerance)",
                                f"P5 {t['p5_neighbour_r']:.3f}, P1 {t['p1_r_m1_0']:.3f} / {t['p1_r_0_p1']:.3f}, "
                                f"P1 (-1,+1) {t['p1_r_m1_p1']:.3f}", diff <= 0.03))
    for blink in (False, True):
        for g in (0.0, 42.5):
            sel = [r for r in sims if r["blink"] == blink and r["guard_nm"] == g]
            nb = np.array([0.5 * (r["r_m1_0"] + r["r_0_p1"]) for r in sel], dtype=float)
            sk = np.array([r["r_m1_p1"] for r in sel], dtype=float)
            th = next(t for t in t47 if t["guard_nm"] == g)
            mean_nb = float(np.nanmean(nb))
            se = float(np.nanstd(nb) / math.sqrt(max(1, np.isfinite(nb).sum())))
            ok = abs(mean_nb - th["p5_neighbour_r"]) <= 0.03 if not blink else None
            verdicts.append(verdict(f"leak measured, blink {blink}, guard {g:g}",
                                    "neighbour r within 0.03 of P5 (no blink); blink = a lower bound (P1)",
                                    f"neighbour r {mean_nb:.3f} +- {se:.3f} (P5 {th['p5_neighbour_r']:.3f}); "
                                    f"(-1,+1) r {float(np.nanmean(sk)):.3f}", ok,
                                    f"{len(sel)} axons x {n_locs} locs"))
    g42 = next(t for t in t47 if t["guard_nm"] == 42.5)
    verdicts.append(verdict("plan: neighbour r < 0.10 with a 42.5 nm guard", "r < 0.10 [plan sec. 4.4 P5]",
                            f"P5 {g42['p5_neighbour_r']:.3f}, P1 {g42['p1_r_m1_0']:.3f}",
                            g42["p5_neighbour_r"] < 0.10,
                            "deviation of the plan: the count gives ~0.12; documented, not relaxed"))
    return theory, sims, verdicts


# ----------------------------------------------------------------------------------------------------
# Cross test 3: P1 true_lambda (cell starts) versus P4 bins (centred)
# ----------------------------------------------------------------------------------------------------
def _expected_counts(sim: S.SimPerimeter, n_bins: int, origin: float, centred: bool) -> NDArray[np.float64]:
    L = float(sim.truth["length_nm"])
    ds = L / sim.arc_grid_nm.size
    w = L / n_bins
    pos = sim.arc_grid_nm + (0.5 * ds if centred else 0.0)
    if centred:
        j = np.mod(np.floor((pos - origin) / w + 0.5).astype(np.int64), n_bins)
    else:   # the naive reading: P1 grid point = start of a P4 bin
        j = np.mod(np.floor((pos - origin) / w).astype(np.int64), n_bins)
    mass = np.bincount(j, weights=sim.true_lambda, minlength=n_bins)
    return np.asarray(mass / mass.sum(), dtype=np.float64)


def cross_halfbin_p1_p4(n_sims: int = 40, n_locs: int = 40000, seed0: int = 70000) -> Tuple[List[Row], Row]:
    """chi2 / dof of the observed P4 counts against the P1 truth binned on P4's centred bins and on cell starts."""
    rows: List[Row] = []
    for i in range(n_sims):
        sim = S.simulate_perimeter("circle", {"kind": "regular", "period_nm": 400.0}, n_locs=n_locs,
                                   radial_sd_nm=0.0, slices={"fracs": (0.0, 1.0, 0.0)}, lpz_nm=0.0,
                                   background_frac=0.0, interior_frac=0.0, blink=False, seed=seed0 + i,
                                   options={"precision_nm": 0.0})
        L = float(sim.true_path.length_nm)
        s, r = D.project(sim.true_path, sim.x_nm, sim.y_nm)
        n_bins, _ = D.arc_grid(L)
        origin = float(np.random.default_rng(seed0 + i).uniform(0, L))
        prof = D.density_profile(s, L, n_bins=n_bins, origin_arc_nm=origin)
        obs = prof.counts.astype(np.float64)
        out: Row = {"sim": i, "n_bins": n_bins}
        for name, centred in (("centred", True), ("cell_start", False)):
            e = _expected_counts(sim, n_bins, origin, centred) * obs.sum()
            ok = e > 0
            out[f"chi2_dof_{name}"] = float(np.sum((obs[ok] - e[ok]) ** 2 / e[ok]) / max(1, ok.sum() - 1))
        rows.append(out)
    c = float(np.median([r["chi2_dof_centred"] for r in rows]))
    n = float(np.median([r["chi2_dof_cell_start"] for r in rows]))
    v = verdict("P1 x P4 half-bin convention", "P1 truth binned on P4's CENTRED bins fits (chi2/dof ~ 1); the "
                "cell-start reading does not", f"median chi2/dof centred {c:.2f}, cell start {n:.2f} "
                f"({n_sims} sims, regular 400 nm, no blink)", bool(c < 1.3 and n > c))
    return rows, v


# ----------------------------------------------------------------------------------------------------
# Cross test 4: near-empty lateral slice
# ----------------------------------------------------------------------------------------------------
def cross_empty_slice() -> Tuple[List[Row], List[Row]]:
    from tools.mps_perimeter_analysis import analyze_perimeter
    rows: List[Row] = []
    verdicts: List[Row] = []
    for name, fracs, n in (("zero", (0.0, 0.5, 0.5), 10000), ("three", (0.0003, 0.5, 0.4997), 10000)):
        sim = S.simulate_perimeter("wavy", {"kind": "random"}, n_locs=n, radial_sd_nm=20.0,
                                   slices={"fracs": fracs, "pattern": "independent"}, lpz_nm=0.0,
                                   background_frac=0.0, interior_frac=0.0, blink=False, seed=4242)
        err = ""
        try:
            res = analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=None,
                                    source_path="sim_empty_slice.hdf5",
                                    params=_params(contour_method="drawn", n_null=199),
                                    drawn_contour=np.asarray(sim.true_path.points_nm)[::5])
        except Exception as exc:  # noqa: BLE001 - the point of the test
            err = f"{type(exc).__name__}: {exc}"
            res = None
        row: Row = {"case": name, "exception": err}
        if res is not None:
            sl = res.slices
            row.update(status=res.status.value, n_locs_m1=None if sl is None else sl.n_locs[0],
                       pct_m1_of_0=None if sl is None else sl.pct_of_central[0],
                       c_reading=None if res.diagnostics is None else res.diagnostics.c_reading)
            for pair, tag in (((-1, 0), "m1_0"), ((0, 1), "0_p1"), ((-1, 1), "m1_p1")):
                t = res.c.get(pair)
                row[f"r_{tag}"] = None if t is None else t.value
                row[f"p_{tag}"] = None if t is None else t.p
        rows.append(row)
        nan_m1 = res is not None and all(not math.isfinite(float(row.get(f"r_{t}") or float("nan")))
                                         for t in ("m1_0", "m1_p1"))
        ok = (err == "") and (nan_m1 if name == "zero" else True)
        verdicts.append(verdict(f"slice -1 with {name} localizations", "no exception; C pairs with -1 are NaN "
                                "(0 locs) or reported with the slice's percentage", f"status {row.get('status')}, "
                                f"r(-1,0) {row.get('r_m1_0')}, r(-1,+1) {row.get('r_m1_p1')}, "
                                f"reading '{row.get('c_reading')}'", ok, err))
    return rows, verdicts


def _params(**kw: Any) -> Any:
    from dataclasses import replace
    from tools.mps_perimeter_types import PerimeterParams
    return replace(PerimeterParams(), **kw)


# ----------------------------------------------------------------------------------------------------
# Cross test 5: clockwise versus counter-clockwise contour
# ----------------------------------------------------------------------------------------------------
def _reverse_keep_start(path: SmoothPath, start_shift: int = 0) -> SmoothPath:
    pts = np.asarray(path.points_nm, dtype=np.float64)
    n = pts.shape[0]
    order = np.mod(start_shift - np.arange(n), n)
    new = pts[order]
    nxt = np.roll(new, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - new[:, 0], nxt[:, 1] - new[:, 1]).astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(edges)]).astype(np.float64)
    rows = np.mod(start_shift - np.asarray(path.vertex_row, dtype=np.intp), n).astype(np.intp)
    return SmoothPath(points_nm=new, vertex_row=rows, edges_nm=edges, cum_nm=cum, length_nm=float(cum[-1]))


def cross_orientation() -> Tuple[List[Row], List[Row]]:
    from tools.mps_perimeter_analysis import analyze_perimeter
    from tools.mps_perimeter_types import AutoContour, Status
    sim = S.simulate_scenario("b_curvature", seed=7)
    true = sim.true_path

    def fn_for(p: SmoothPath) -> Any:
        def fn(x: Any, y: Any, **_kw: Any) -> AutoContour:
            return AutoContour(path=p, length_nm=float(p.length_nm), method="density_nodes", knot_nm=400.0,
                               residual_sd_nm=0.0, n_self_crossings=0, nodes_nm=np.zeros((0, 2)),
                               polar_distance_nm=float("nan"), star_shaped=True, status=Status.OK)
        return fn

    cases = {"ccw": true, "cw_same_start": _reverse_keep_start(true, 0),
             "cw_other_start": _reverse_keep_start(true, 1)}
    out: Dict[str, Row] = {}
    for name, p in cases.items():
        res = analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=None,
                                source_path="sim_orientation.hdf5", params=_params(n_null=499),
                                contour_fn=fn_for(p))
        assert res.profiles is not None and res.b is not None
        out[name] = {"case": name, "status": res.status.value,
                     "reversed": None if res.diagnostics is None else res.diagnostics.contour_reversed,
                     "a_contrast": res.a["contrast"].value, "a_period": res.a["period"].value,
                     "b_spearman": res.b.test.value, "c_m1_0": res.c[(-1, 0)].value,
                     "c_0_p1": res.c[(0, 1)].value, "c_m1_p1": res.c[(-1, 1)].value,
                     "max_count": int(res.profiles.by_slice[0].counts.max()),
                     "n_bins": res.profiles.by_slice[0].n_bins}
    keys = ("a_contrast", "a_period", "b_spearman", "c_m1_0", "c_0_p1", "c_m1_p1", "max_count")
    d_same = max(abs(float(out["ccw"][k]) - float(out["cw_same_start"][k])) for k in keys)
    d_other = max(abs(float(out["ccw"][k]) - float(out["cw_other_start"][k])) for k in keys)
    v = [verdict("contour clockwise vs counter-clockwise (same start)", "identical A, B, C (|diff| < 1e-9)",
                 f"max |diff| {d_same:.2e}; reversed flag {out['cw_same_start']['reversed']}", d_same < 1e-9),
         verdict("contour clockwise starting one table step away", "informative: binning moves by 2 nm",
                 f"max |diff| {d_other:.3g}", None)]
    return list(out.values()), v


# ----------------------------------------------------------------------------------------------------
# V1: P3 on the shared simulator (P1 x P3)
# ----------------------------------------------------------------------------------------------------
V1_SHAPES: Tuple[str, ...] = S.SHAPES
V1_ERR_MEAN_MAX: float = 25.0       # [plan sec. 4.4 P3]
V1_ERR_P95_MAX: float = 80.0        # [plan sec. 4.4 P3]
V1_LEN_TOL: float = 0.05            # [plan sec. 4.4 P3]
V1_SUPPORT_NM: float = 100.0        # [ad hoc, = P3's validate_auto_contour] a true point "has data" when its
V1_SUPPORT_K: int = 5               # fifth-nearest membrane localization is within 100 nm


def v1_one(shape: str, bg: float, seed: int) -> Row:
    from scipy.spatial import cKDTree
    from tools import mps_auto_contour as AC
    rng = np.random.default_rng([seed, V1_SHAPES.index(shape), int(round(bg * 100))])
    n_locs = int(rng.integers(4000, 15001))           # [plan sec. 4.4 P3] 4 000 - 15 000 localizations
    sim = S.simulate_perimeter(shape, {"kind": "flat"}, n_locs=n_locs, radial_sd_nm=20.0, slices={},
                               lpz_nm=47.0, background_frac=bg, interior_frac=0.0, blink=True,
                               seed=int(rng.integers(0, 2 ** 31)),
                               options={"rotation_deg": float(rng.uniform(0.0, 360.0))})
    t0 = time.time()
    det = AC.contour_details(sim.x_nm, sim.y_nm)
    secs = time.time() - t0
    c = det.contour
    row: Row = {"shape": shape, "bg": bg, "seed": seed, "n_locs": n_locs, "seconds": round(secs, 2),
                "status": c.status.value, "method": c.method, "knot_nm": c.knot_nm,
                "primary_reason": det.candidates[0].reason, "crossings_primary": det.candidates[0].n_self_crossings}
    if c.path is None:
        return row
    truth = np.asarray(sim.true_path.points_nm, dtype=np.float64)
    mem = np.column_stack([sim.x_nm, sim.y_nm])[sim.source == S.SOURCE_MEMBRANE]
    dk, _ = cKDTree(mem).query(truth, k=V1_SUPPORT_K)
    sup = dk[:, -1] <= V1_SUPPORT_NM
    fitted = np.asarray(c.path.points_nm, dtype=np.float64)
    e, _ = cKDTree(fitted).query(truth)
    area2 = float(np.dot(fitted[:, 0], np.roll(fitted[:, 1], -1)) - np.dot(fitted[:, 1], np.roll(fitted[:, 0], -1)))
    row.update(err_mean=float(np.mean(e[sup])), err_p95=float(np.percentile(e[sup], 95)),
               support_frac=float(np.mean(sup)), len_ratio=float(c.length_nm / sim.true_path.length_nm),
               crossings=int(c.n_self_crossings), ccw=bool(area2 > 0))
    return row


CURVE_KINDS: Tuple[str, ...] = ("auto", "knot600", "polar")   # review R1: the real path uses the fallbacks


def fit_curve(x: NDArray[np.float64], y: NDArray[np.float64], kind: str) -> Any:
    """P3's contour as the analysis gets it (``auto``: defaults and fallbacks) or with a fallback FORCED:
    ``knot600`` = density nodes with a 600 nm knot and no further fallback, ``polar`` = the polar curve."""
    from tools import mps_auto_contour as AC
    if kind == "auto":
        return AC.contour_details(x, y)
    if kind == "knot600":
        return AC.contour_details(x, y, knot_nm=600.0, fallbacks=False)
    if kind == "polar":
        return AC.contour_details(x, y, method="polar")
    raise ValueError(kind)


def curve_type(contour: Any) -> str:
    """'knot400' / 'knot600' / 'polar' / 'drawn' / 'failed': the curve actually used."""
    if contour.path is None:
        return "failed"
    if contour.method == "polar":
        return "polar"
    if contour.method == "density_nodes":
        return f"knot{int(round(contour.knot_nm))}"
    return str(contour.method)


def curve_error(sim: S.SimPerimeter, path: SmoothPath) -> Dict[str, float]:
    """Error of a fitted curve against the truth, on the true arcs with data (as P3's validation)."""
    from scipy.spatial import cKDTree
    truth = np.asarray(sim.true_path.points_nm, dtype=np.float64)
    mem = np.column_stack([sim.x_nm, sim.y_nm])[sim.source == S.SOURCE_MEMBRANE]
    dk, _ = cKDTree(mem).query(truth, k=V1_SUPPORT_K)
    sup = dk[:, -1] <= V1_SUPPORT_NM
    fitted = np.asarray(path.points_nm, dtype=np.float64)
    e, _ = cKDTree(fitted).query(truth)
    if not sup.any():
        sup = np.ones_like(sup)
    return {"err_mean": float(np.mean(e[sup])), "err_p95": float(np.percentile(e[sup], 95)),
            "len_ratio": float(path.length_nm / sim.true_path.length_nm)}


def v1_wide_one(shape: str, radial_sd: float, seed: int) -> List[Row]:
    """Review R1 (H1): a cloud as wide as the real one (radial sd 100-150 nm, 10 % interior spectrin, 15 %
    background) and the three curve types the real data use. One row per curve type."""
    rng = np.random.default_rng([seed, V1_SHAPES.index(shape), int(radial_sd), 7])
    n_locs = int(rng.integers(8000, 20001))
    sim = S.simulate_perimeter(shape, {"kind": "clumps"}, n_locs=n_locs, radial_sd_nm=radial_sd, slices={},
                               lpz_nm=47.0, background_frac=0.15, interior_frac=0.10, blink=True,
                               seed=int(rng.integers(0, 2 ** 31)),
                               options={"rotation_deg": float(rng.uniform(0.0, 360.0))})
    rows: List[Row] = []
    for kind in CURVE_KINDS:
        t0 = time.time()
        det = fit_curve(sim.x_nm, sim.y_nm, kind)
        c = det.contour
        row: Row = {"shape": shape, "radial_sd_nm": radial_sd, "seed": seed, "n_locs": n_locs, "asked": kind,
                    "used": curve_type(c), "status": c.status.value, "seconds": round(time.time() - t0, 2),
                    "primary_reason": det.candidates[0].reason}
        if c.path is not None:
            row.update(curve_error(sim, c.path))
            _, r = D.project(c.path, sim.x_nm, sim.y_nm)
            row["frac_within_100"] = float(np.mean(np.abs(r) <= 100.0))
        rows.append(row)
    return rows


def v1_wide_verdicts(rows: Sequence[Mapping[str, Any]]) -> List[Row]:
    out: List[Row] = []
    for sd in sorted({fnum(r["radial_sd_nm"]) for r in rows}):
        for kind in CURVE_KINDS:
            sel = [r for r in rows if fnum(r["radial_sd_nm"]) == sd and str(r["asked"]) == kind]
            if not sel:
                continue
            ok = [r for r in sel if str(r["status"]) == "ok"]
            em = np.array([fnum(r.get("err_mean")) for r in ok])
            ep = np.array([fnum(r.get("err_p95")) for r in ok])
            lr = np.array([fnum(r.get("len_ratio")) for r in ok])
            used: Dict[str, int] = {}
            for r in sel:
                used[str(r["used"])] = used.get(str(r["used"]), 0) + 1
            k = sum(v1_passes({**r, "crossings": 0}) for r in sel)
            fin = np.isfinite(em).any()
            out.append(verdict(f"V1-wide radial sd {sd:g} nm, curve asked '{kind}'",
                               "P3 thresholds (err <= 25, p95 <= 80, length +-5 %) per run",
                               f"{k}/{len(sel)} pass; accepted {len(ok)}; used {used}; median err "
                               f"{np.nanmedian(em) if fin else float('nan'):.1f} nm, median p95 "
                               f"{np.nanmedian(ep) if fin else float('nan'):.1f}, worst p95 "
                               f"{np.nanmax(ep) if fin else float('nan'):.1f}, length ratio "
                               f"{np.nanmedian(lr) if fin else float('nan'):.3f}", k == len(sel)))
    return out


def v1_run(bg: float, seeds: int, out: str, shapes: Sequence[str] = V1_SHAPES) -> List[Row]:
    rows: List[Row] = []
    t0 = time.time()
    for shape in shapes:
        for seed in range(seeds):
            rows.append(v1_one(shape, bg, seed))
        print(f"V1 bg {bg:g} {shape}: {time.time() - t0:.0f} s", flush=True)
    write_csv(os.path.join(out, f"v1_bg{int(round(bg * 100)):02d}.csv"), rows)
    return rows


def v1_passes(r: Mapping[str, Any]) -> bool:
    return (str(r.get("status")) == "ok" and fnum(r.get("err_mean")) <= V1_ERR_MEAN_MAX
            and fnum(r.get("err_p95")) <= V1_ERR_P95_MAX and abs(fnum(r.get("len_ratio")) - 1.0) <= V1_LEN_TOL
            and int(fnum(r.get("crossings"))) == 0)


def v1_verdicts(rows: Sequence[Mapping[str, Any]]) -> List[Row]:
    out: List[Row] = []
    cells = sorted({(str(r["shape"]), fnum(r["bg"])) for r in rows})
    for shape, bg in cells:
        sel = [r for r in rows if str(r["shape"]) == shape and fnum(r["bg"]) == bg]
        k = sum(v1_passes(r) for r in sel)
        em = np.array([fnum(r.get("err_mean")) for r in sel])
        ep = np.array([fnum(r.get("err_p95")) for r in sel])
        lr = np.array([fnum(r.get("len_ratio")) for r in sel])
        nfail = sum(str(r.get("status")) != "ok" for r in sel)
        fb = sum(str(r.get("status")) == "ok" and (str(r.get("method")) != "density_nodes"
                                                   or fnum(r.get("knot_nm")) != 400.0) for r in sel)
        ccw = all(str(r.get("ccw")) in ("True", "true", "1") for r in sel if str(r.get("status")) == "ok")
        out.append(verdict(f"V1 {shape}, background {bg:.0%}",
                           "every run: ok, err mean <= 25 nm, p95 <= 80 nm, length +-5 %, 0 crossings",
                           f"{k}/{len(sel)} runs pass; median err {np.nanmedian(em):.1f} nm (max "
                           f"{np.nanmax(em) if np.isfinite(em).any() else float('nan'):.1f}), median p95 "
                           f"{np.nanmedian(ep):.1f} (max {np.nanmax(ep) if np.isfinite(ep).any() else float('nan'):.1f}),"
                           f" length ratio {np.nanmin(lr) if np.isfinite(lr).any() else float('nan'):.3f}-"
                           f"{np.nanmax(lr) if np.isfinite(lr).any() else float('nan'):.3f}; rejected {nfail}; "
                           f"fallback used {fb}; all ccw {ccw}", k == len(sel)))
    return out


def v1_gap(seeds: int) -> Tuple[List[Row], Row]:
    """Contour with a 2 um gap ("C", P1 ``contour_c_gap``) through the whole orchestration: the curve closes and
    the gap's bins come out not measured."""
    from tools.mps_perimeter_analysis import analyze_perimeter
    rows: List[Row] = []
    for seed in range(seeds):
        sim = S.simulate_scenario("contour_c_gap", seed=900 + seed)
        res = analyze_perimeter(sim.x_nm, sim.y_nm, sim.z_nm, columns={}, info=[], identity=None,
                                source_path="sim_gap.hdf5", params=_params(n_null=99,
                                                                           band_sensitivity_nm=(),
                                                                           guard_sensitivity_nm=()))
        L = float(sim.truth["length_nm"])
        expected = (2000.0 - 400.0) / L     # bins fully inside a > 400 nm stretch with no data, roughly
        rows.append({"seed": seed, "status": res.status.value, "length_true_nm": L,
                     "unsupported_fraction": None if res.diagnostics is None else res.diagnostics.unsupported_fraction,
                     "expected_about": expected,
                     "contour_method": None if res.contour is None else res.contour.method,
                     "knot_nm": None if res.contour is None else res.contour.knot_nm})
    ok_status = sum(r["status"] == "ok" for r in rows)
    uf = np.array([fnum(r["unsupported_fraction"]) for r in rows])
    v = verdict("V1 2 um gap (C shape, whole orchestration)", "curve closes (status ok) and the gap's bins are "
                "not measured (unsupported fraction ~ 0.08-0.16)",
                f"ok {ok_status}/{len(rows)}; unsupported fraction median {np.nanmedian(uf):.3f} "
                f"(range {np.nanmin(uf):.3f}-{np.nanmax(uf):.3f})",
                bool(ok_status == len(rows) and np.nanmin(uf) >= 0.08 and np.nanmax(uf) <= 0.16))
    return rows, v


# ----------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------
def run_cross(out: str, small: bool) -> List[Row]:
    v: List[Row] = []
    lines = [f"Sizes: {'small (test)' if small else 'full'}; seeds fixed in the code."]
    r1, v1 = cross_bins_p4_p6(n_trials=5 if small else 40)
    write_csv(os.path.join(out, "cross_bins_p4_p6.csv"), r1)
    v.append(v1)
    th, sims, v2 = cross_leak_p1_p5(n_axons=5 if small else 60, n_locs=20000 if small else 60000)
    write_csv(os.path.join(out, "cross_leak_theory.csv"), th)
    write_csv(os.path.join(out, "cross_leak_sim.csv"), sims)
    v += v2
    r3, v3 = cross_halfbin_p1_p4(n_sims=4 if small else 40)
    write_csv(os.path.join(out, "cross_halfbin_p1_p4.csv"), r3)
    v.append(v3)
    r4, v4 = cross_empty_slice()
    write_csv(os.path.join(out, "cross_empty_slice.csv"), r4)
    v += v4
    r5, v5 = cross_orientation()
    write_csv(os.path.join(out, "cross_orientation.csv"), r5)
    v += v5
    write_verdicts(os.path.join(out, "VERDICT_cross.md"), "Cross-package tests (P11)", v, lines)
    return v


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="P11 cross tests and V1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cross")
    a.add_argument("--out", required=True)
    a.add_argument("--small", action="store_true")
    b = sub.add_parser("v1")
    b.add_argument("--bg", type=float, required=True)
    b.add_argument("--seeds", type=int, default=20)
    b.add_argument("--out", required=True)
    g = sub.add_parser("v1-gap")
    g.add_argument("--seeds", type=int, default=10)
    g.add_argument("--out", required=True)
    r = sub.add_parser("v1-report")
    r.add_argument("--out", required=True)
    wv = sub.add_parser("v1-wide")
    wv.add_argument("--sd", type=float, required=True)
    wv.add_argument("--seeds", type=int, default=10)
    wv.add_argument("--shapes", default=",".join(V1_SHAPES))
    wv.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = ensure_out(args.out)
    t0 = time.time()
    if args.cmd == "v1-wide":
        rows_w: List[Row] = []
        for shape in args.shapes.split(","):
            for seed in range(args.seeds):
                rows_w += v1_wide_one(shape, args.sd, seed)
            print(f"V1-wide sd {args.sd:g} {shape}: {time.time() - t0:.0f} s", flush=True)
        write_csv(os.path.join(out, "v1_wide.csv"), rows_w, append=True)
        allw = read_csv(os.path.join(out, "v1_wide.csv"))
        write_verdicts(os.path.join(out, "VERDICT_v1_wide.md"), "V1 on the real path (wide cloud, forced fallbacks)",
                       v1_wide_verdicts(allw), [f"Rows so far: {len(allw)}. P1 'clumps' density, radial sd as "
                                                "given, 10 % interior, 15 % background, 8 000-20 000 locs."])
    elif args.cmd == "cross":
        for x in run_cross(out, args.small):
            print(x["verdict"], x["check"], "|", x["measured"], flush=True)
    elif args.cmd == "v1":
        v1_run(args.bg, args.seeds, out)
    elif args.cmd == "v1-gap":
        rows, v = v1_gap(args.seeds)
        write_csv(os.path.join(out, "v1_gap.csv"), rows)
        write_verdicts(os.path.join(out, "VERDICT_v1_gap.md"), "V1 gap", [v])
        print(v)
    else:
        rows_all: List[Dict[str, str]] = []
        for f in sorted(os.listdir(out)):
            if f.startswith("v1_bg") and f.endswith(".csv"):
                rows_all += read_csv(os.path.join(out, f))
        vs = v1_verdicts(rows_all)
        gap = os.path.join(out, "v1_gap.csv")
        lines = [f"Runs: {len(rows_all)} (5 shapes x seeds x background levels), P1 simulator, P3 auto_contour "
                 "with its defaults and fallbacks; 4 000-15 000 localizations per run, flat density, blinking, "
                 "radial sd 20 nm, lpz 47 nm, random rotation."]
        if os.path.exists(gap):
            lines.append("Gap runs: see VERDICT_v1_gap.md.")
        write_verdicts(os.path.join(out, "VERDICT_v1.md"), "V1 automatic contour (P1 x P3)", vs, lines)
        for x in vs:
            print(x["verdict"], x["check"], "|", x["measured"], flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
