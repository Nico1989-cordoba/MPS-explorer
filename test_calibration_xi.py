"""xi calibration tool: recovery of a known xi from synthetic sparse labels.

Synthetic data only (no data file is read). The generator below is the
test's own and deliberately simple: labels placed at random, each with a
known number of switching cycles drawn from the model the tool fits
(rounded gamma, zero-truncated), cycles spread in time by exponential
off-times, a few frames per cycle, Gaussian localization error, optional
uniform background groups, and a movie that may end before every label
has bleached.

Run: py -3 -m pytest test_calibration_xi.py -q
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional, Tuple

import numpy as np
import pytest

from tools.calibration_xi import (
    Background, CalibrationParams, FieldData, analyse_calibration, derived_group_radius, fit_counts, main,
    n_min_from_background, single_linkage_labels, write_outputs)


# ----------------------------------------------------------------- generator
def draw_cycles(rng: np.random.Generator, n: int, mean: float, k: float) -> np.ndarray:
    """N = round(X), X ~ Gamma(k, mean/k), redrawn while N = 0 (the model the tool fits)."""
    out = np.zeros(n, dtype=np.int64)
    todo = np.arange(n)
    while todo.size:
        x = np.rint(rng.gamma(k, mean / k, todo.size)).astype(np.int64)
        out[todo] = x
        todo = todo[x == 0]
    return out


def simulate_field(rng: np.random.Generator, *, mean: float = 10.0, k: float = 1.0, field_um: float = 40.0,
                   density: float = 0.08, n_frames: int = 4000, off_mean: float = 150.0, on_mean: float = 1.5,
                   lp_nm: float = 10.0, label_sd_nm: float = 5.0, bg_density: float = 0.0,
                   bg_mean_events: float = 1.3, labels: bool = True, session: str = "S1",
                   name: str = "f1") -> Tuple[FieldData, dict]:
    side = field_um * 1000.0
    area = field_um ** 2
    frames, xs, ys = [], [], []
    truth = {"cycles": np.zeros(0, dtype=np.int64), "recorded": np.zeros(0, dtype=np.int64)}
    if labels:
        n_marks = rng.poisson(density * area)
        cx, cy = rng.uniform(0, side, n_marks), rng.uniform(0, side, n_marks)
        cycles = draw_cycles(rng, n_marks, mean, k)
        recorded = np.zeros(n_marks, dtype=np.int64)
        for m in range(n_marks):
            t = 0.0                       # all labels start on (glass, protocol F6)
            for _ in range(int(cycles[m])):
                if t >= n_frames:
                    break
                on = 1 + rng.geometric(1.0 / on_mean) - 1
                f0 = int(t)
                fr = np.arange(f0, min(f0 + on, n_frames))
                ox, oy = rng.normal(0, label_sd_nm, 2)
                frames.append(fr)
                xs.append(cx[m] + ox + rng.normal(0, lp_nm, fr.size))
                ys.append(cy[m] + oy + rng.normal(0, lp_nm, fr.size))
                recorded[m] += 1
                # Off-time of at least 3 frames, so one-frame dark tolerance never joins two cycles.
                t = f0 + on + 3 + rng.exponential(off_mean)
        truth = {"cycles": cycles, "recorded": recorded, "cx": cx, "cy": cy}
    n_bg = rng.poisson(bg_density * area)
    bx, by = rng.uniform(0, side, n_bg), rng.uniform(0, side, n_bg)
    for b in range(n_bg):
        ne = rng.geometric(1.0 / bg_mean_events)
        fr = np.sort(rng.choice(n_frames, ne, replace=False))
        frames.append(fr)
        xs.append(bx[b] + rng.normal(0, lp_nm, ne))
        ys.append(by[b] + rng.normal(0, lp_nm, ne))
    truth["n_background"] = n_bg
    if frames:
        f = np.concatenate(frames).astype(np.int64)
        x, y = np.concatenate(xs), np.concatenate(ys)
    else:
        f, x, y = np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
    order = np.argsort(f, kind="stable")
    fd = FieldData(frame=f[order], x_nm=x[order], y_nm=y[order], lp_nm=np.full(f.size, lp_nm),
                   photons=np.full(f.size, 2000.0), n_frames=n_frames, session=session, name=name,
                   bounds_nm=(0.0, side, 0.0, side))
    return fd, truth


def params(**kw) -> CalibrationParams:
    # 60 nm = 6 lp: consecutive frames of one cycle never split (at 3 lp about 10 % of links break,
    # which adds events per label -- real, and the reason the tissue radius must be used).
    base = dict(link_radius_nm=60.0, max_dark_frames=1, n_bootstrap=200, pump_frames_versions=(0,))
    base.update(kw)
    return CalibrationParams(**base)


# ----------------------------------------------------------------- pieces
def test_single_linkage_is_dbscan_min1():
    pts = np.array([[0, 0], [50, 0], [100, 0], [1000, 0], [1040, 0], [5000, 5000]], dtype=float)
    lab = single_linkage_labels(pts, 60.0)
    assert lab.tolist() == [0, 0, 0, 1, 1, 2]


def test_derived_group_radius():
    r = derived_group_radius(np.full(10, 10.0), 10.0)
    assert r == pytest.approx(5.257 * math.sqrt(200.0), rel=1e-3)
    with pytest.raises(ValueError):
        derived_group_radius(None, 0.0)


@pytest.mark.parametrize("mean, k", [(14.0, 1.0), (7.0, 1.0), (20.0, 2.5), (10.0, 0.6)])
def test_fit_recovers_model_parameters_from_counts(mean, k):
    rng = np.random.default_rng(1)
    c = draw_cycles(rng, 4000, mean, k)
    f = fit_counts(c, model="gamma")
    assert f.converged
    assert f.mean == pytest.approx(mean, rel=0.06)
    assert f.k == pytest.approx(k, rel=0.12)


def test_geometric_is_gamma_with_k1_and_aic_prefers_truth():
    rng = np.random.default_rng(2)
    c = draw_cycles(rng, 3000, 12.0, 1.0)
    g, a = fit_counts(c, model="geometric"), fit_counts(c, model="gamma")
    assert g.k == 1.0 and g.mean == pytest.approx(12.0, rel=0.06)
    assert a.loglik >= g.loglik - 1e-6
    c3 = draw_cycles(rng, 3000, 12.0, 3.0)
    assert fit_counts(c3, model="gamma").aic < fit_counts(c3, model="geometric").aic - 10


def test_censored_likelihood_undoes_end_of_movie_truncation():
    """Cap N at a random number of cycles; censored marks contribute P(N > n)."""
    rng = np.random.default_rng(3)
    n = draw_cycles(rng, 4000, 15.0, 1.0)
    cap = rng.poisson(12.0, n.size) + 1                         # cycles that fit in the movie
    obs, cen = np.minimum(n, cap), n > cap
    naive = fit_counts(obs, model="gamma")
    fixed = fit_counts(obs, cen, model="gamma")
    assert naive.mean < 12.0                                    # strongly biased low
    assert fixed.mean == pytest.approx(15.0, rel=0.08)


def test_background_mixture_and_n_min():
    rng = np.random.default_rng(4)
    sig = draw_cycles(rng, 600, 10.0, 1.0)
    noise = rng.geometric(1 / 1.2, 200)
    c = np.concatenate([sig, noise])
    pmf = np.bincount(rng.geometric(1 / 1.2, 20000)).astype(float)
    pmf /= pmf.sum()
    bg = Background(weight=200 / 800, pmf=pmf, expected_false=200.0)
    raw = fit_counts(c, model="gamma")
    mixed = fit_counts(c, model="gamma", background=bg)
    assert raw.mean < 8.5
    assert mixed.mean == pytest.approx(10.0, rel=0.12)
    nm = n_min_from_background(c, bg)
    assert nm >= 3
    trunc = fit_counts(c, model="gamma", n_min=nm)
    assert trunc.mean == pytest.approx(10.0, rel=0.2)


def test_bad_inputs():
    with pytest.raises(ValueError):
        fit_counts([0, 1, 2])
    with pytest.raises(ValueError):
        CalibrationParams(link_radius_nm=0, max_dark_frames=1).validate()
    with pytest.raises(ValueError):
        CalibrationParams(link_radius_nm=30, max_dark_frames=1, estimand="other").validate()


# ----------------------------------------------------------------- the whole pipeline
def _three_sessions(seed: int, mean: float, k: float, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for s in ("A1", "A2", "A3"):
        for f in ("c1", "c2"):
            fd, _ = simulate_field(rng, mean=mean, k=k, session=s, name=f, **kw)
            out.append(fd)
    return out


def test_pipeline_recovers_mean_and_k_within_ci_most_seeds():
    """Most seeds: the true mean and k fall inside the bootstrap 95 % interval."""
    hits_mean, hits_k, n = 0, 0, 12
    for seed in range(n):
        rng = np.random.default_rng(100 + seed)
        fields = [simulate_field(rng, mean=12.0, k=1.0, n_frames=16000, session=f"S{i // 2}", name=f"f{i}")[0]
                  for i in range(4)]
        res = analyse_calibration(fields, params(n_bootstrap=150, seed=seed))
        b = res.summary["result"]["bootstrap"]
        fit = res.fits["gamma"]
        lo, hi = b["mean_ci95"]
        hits_mean += lo <= 12.0 <= hi
        hits_k += b["gamma_k_ci95"][0] <= 1.0 <= b["gamma_k_ci95"][1]
        assert fit.n_marks > 150
    assert hits_mean >= 9, hits_mean
    assert hits_k >= 9, hits_k


def test_pipeline_gamma_shape_recovered():
    fields = _three_sessions(7, 15.0, 3.0, n_frames=16000)
    res = analyse_calibration(fields, params())
    assert res.summary["selected_model"] == "gamma"
    r = res.summary["result"]
    assert r["mean"] == pytest.approx(15.0, rel=0.1)
    assert r["k"] == pytest.approx(3.0, rel=0.3)
    lo, hi = r["bootstrap"]["gamma_k_ci95"]
    assert lo <= 3.0 <= hi
    assert "mean_of_session_means" in res.summary["sessions"]
    assert len(res.summary["sessions"]["per_session"]) == 3


def test_background_corrected_with_control():
    """As many background groups as labels: the raw fit is far off, the corrected ones are not."""
    rng = np.random.default_rng(11)
    out = [simulate_field(rng, mean=10.0, k=1.0, n_frames=16000, bg_density=0.08, name=f"f{i}") for i in range(3)]
    fields = [o[0] for o in out]
    ctrl = [simulate_field(rng, labels=False, n_frames=16000, bg_density=0.08, name=f"v0_{i}")[0]
            for i in range(2)]
    raw = analyse_calibration(fields, params(background="none", n_bootstrap=0))
    fixed = analyse_calibration(fields, params(n_bootstrap=100), controls=ctrl)
    nmin = analyse_calibration(fields, params(background="n_min", n_bootstrap=0), controls=ctrl)
    assert raw.summary["result"]["mean"] < 7.0                       # many 1-2 event false marks
    lo, hi = fixed.summary["result"]["bootstrap"]["mean_ci95"]
    assert lo <= 10.0 <= hi
    assert fixed.summary["result"]["mean"] == pytest.approx(10.0, rel=0.15)
    # Above n_min only the tail is seen: the geometric's mean is identified, the gamma's k is not.
    assert nmin.fits["geometric"].mean == pytest.approx(10.0, rel=0.2)
    assert any("n_min" in w for w in nmin.summary["warnings"])
    bgs = fixed.summary["background"]
    assert bgs["density_groups_per_um2"] == pytest.approx(0.08, rel=0.3)
    qc = {q["check"]: q for q in fixed.summary["quality"] if q["where"] == "all"}
    assert qc["background_share"]["ok"] is False                     # ~50 % false marks: flagged


def test_truncation_end_of_movie():
    """Short movie: 'recorded' gives the truncated count, 'censored' the count until bleaching."""
    rng = np.random.default_rng(21)
    fields, truths = [], []
    for i in range(4):
        fd, tr = simulate_field(rng, mean=20.0, k=1.0, n_frames=1500, off_mean=120.0, name=f"f{i}")
        fields.append(fd)
        truths.append(tr)
    rec = analyse_calibration(fields, params(n_bootstrap=100))
    cen = analyse_calibration(fields, params(n_bootstrap=100, estimand="censored"))
    recorded_truth = np.concatenate([t["recorded"] for t in truths])
    assert rec.summary["counts"]["empirical"]["mean"] == pytest.approx(recorded_truth.mean(), rel=0.1)
    assert rec.summary["result"]["mean"] < 15.0
    assert cen.summary["result"]["mean"] == pytest.approx(20.0, rel=0.2)
    act = [q for q in rec.summary["quality"] if q["check"] == "fraction_active_at_end"][0]
    assert act["value"] > 0.2 and act["ok"] is False


def test_isolation_and_quality_flags():
    rng = np.random.default_rng(31)
    dense = [simulate_field(rng, density=0.6, name="dense")[0]]
    res = analyse_calibration(dense, params(n_bootstrap=0))
    qc = {q["check"]: q for q in res.summary["quality"]}
    assert qc["density_marks_per_um2"]["ok"] is False
    excl = [m["exclusion"] for m in res.marks]
    assert excl.count("neighbour") > 0.3 * len(excl)


def test_drift_is_detected():
    rng = np.random.default_rng(41)
    fd, _ = simulate_field(rng, mean=15.0, density=0.08)
    fd.x_nm = fd.x_nm + fd.frame * (300.0 / fd.n_frames)          # 300 nm over the movie
    res = analyse_calibration([fd], params(n_bootstrap=0, group_radius_nm=150.0))
    dr = [q for q in res.summary["quality"] if q["check"] == "drift_nm_over_movie"][0]
    assert dr["value"] == pytest.approx(300.0, rel=0.25)
    assert dr["ok"] is False


def test_pump_frames_and_lost_fraction():
    rng = np.random.default_rng(51)
    fields = [simulate_field(rng, mean=8.0, name=f"f{i}")[0] for i in range(3)]
    res = analyse_calibration(fields, params(n_bootstrap=0, pump_frames_versions=(0, 200, 1000)))
    pv = {v["pump_frames"]: v for v in res.summary["pump_versions"]}
    assert pv[0]["fraction_lost"] == 0.0
    assert pv[1000]["fraction_lost"] > pv[200]["fraction_lost"] > 0.0
    assert pv[1000]["empirical_mean"] < pv[0]["empirical_mean"]


def test_irradiance_cut_and_bands():
    rng = np.random.default_rng(61)
    fd, _ = simulate_field(rng, mean=10.0, field_um=60.0)
    p = params(n_bootstrap=0, beam_centre_nm=(30000.0, 30000.0), beam_sigma_nm=20000.0)
    res = analyse_calibration([fd], p)
    assert len(res.summary["irradiance_bands"]) == 3
    assert any(m["exclusion"] == "irradiance" for m in res.marks)
    assert all(m["rel_irradiance"] >= 0.7 for m in res.marks if m["in_main"])


def test_calibrated_prior_args_are_usable():
    from tools.emitters_prior import SOURCE_CALIBRATED, calibrated_prior
    rng = np.random.default_rng(71)
    fields = [simulate_field(rng, mean=14.0, name=f"f{i}")[0] for i in range(2)]
    res = analyse_calibration(fields, params(n_bootstrap=50), dye="Alexa Fluor 647", buffer="GLOX + 10 mM MEA",
                              metadata={"lot": "TEST", "power_start_mw": "100", "power_end_mw": "120"})
    args = res.calibrated_prior_args()
    prior = calibrated_prior(**args)
    assert prior.source == SOURCE_CALIBRATED
    assert prior.mean == pytest.approx(args["mean"])
    assert "lot TEST" in args["note"] and "link 60 nm" in args["note"]
    pw = [q for q in res.summary["quality"] if q["check"] == "laser_power_change"][0]
    assert pw["ok"] is False                                          # 20 % change flagged


def test_deterministic():
    a = analyse_calibration(_three_sessions(81, 10.0, 1.0)[:2], params(n_bootstrap=60))
    b = analyse_calibration(_three_sessions(81, 10.0, 1.0)[:2], params(n_bootstrap=60))
    assert json.dumps(a.summary, sort_keys=True, default=str) == json.dumps(b.summary, sort_keys=True, default=str)
    assert json.dumps(a.marks, default=str) == json.dumps(b.marks, default=str)


def test_outputs_and_cli(tmp_path):
    pytest.importorskip("matplotlib")   # write_outputs draws the figure (CI headless job: no matplotlib)
    rng = np.random.default_rng(91)
    fd, _ = simulate_field(rng, mean=10.0)
    res = analyse_calibration([fd], params(n_bootstrap=20), dye="Alexa Fluor 647")
    paths = write_outputs(res, str(tmp_path / "cal"))
    for p in paths.values():
        assert os.path.getsize(p) > 0
    with open(paths["json"], encoding="utf-8") as fh:
        js = json.load(fh)
    assert set(js["calibrated_prior_args"]) == {"mean", "k", "dye", "buffer", "note"}
    with pytest.raises(FileExistsError):
        write_outputs(res, str(tmp_path / "cal"))
    # CLI on a synthetic ThunderSTORM-style CSV written here.
    csv_path = tmp_path / "synthetic_locs.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("id,frame,x [nm],y [nm],z [nm],uncertainty_xy [nm],intensity [photon]\n")
        for i, (f, x, y) in enumerate(zip(fd.frame, fd.x_nm, fd.y_nm)):
            fh.write(f"{i + 1},{f + 1},{x:.3f},{y:.3f},0.0,10.0,2000\n")
    code = main(["--session", "A1", str(csv_path), "--link-radius", "60", "--max-dark", "1", "--n-frames", str(fd.n_frames),
                 "--bootstrap", "20", "--out", str(tmp_path / "cli"), "--meta", "lot=TEST", "--no-figure"])
    assert code == 0
    with open(tmp_path / "cli_xi.json", encoding="utf-8") as fh:
        js2 = json.load(fh)
    assert js2["metadata"]["lot"] == "TEST"
    assert js2["result"]["mean"] == pytest.approx(js["result"]["mean"], rel=0.15)
