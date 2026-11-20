# -*- coding: utf-8 -*-
"""
Headless tests of the event layer, the "probable fusion" mark, the DBSCAN rule and the parameter sweep
(clustering plan A3, A4, A2, A1; docs/dbscan_rule.md).

Synthetic data only (R8): blinking movies and rings drawn here, with a known truth.

  1. localizations per cycle: recovered on a simulated movie with a known (geometric) event-length distribution;
     the 3 NeNA radius biases it low by the amount the Rayleigh split predicts, the split correction and the
     3 sqrt(2) NeNA radius recover it; a linked file's n / len columns; the labelled fallback; censored events;
  2. events per cluster: an event never spans two clusters; counts on a hand-made case;
  3. the "probable fusion" mark on synthetic merged pairs (4 labels in one cluster against 1), under c_label 14 and
     26; the mark changes no label;
  4. the rule: the worked check of docs/channel2_parameters.md section 5 (9.6 -> 10), the documented values
     (c_label 14 -> 5-6, 26 -> 10-13 with 2 localizations per cycle), the research's own formula, the floor, the
     eps cap, the interval of min_samples from the interval of the localizations per cycle, the double table;
  5. the sweep: 5 x 4 grid, the 25/10 cell equals the normal analysis exactly, DBSCAN's monotone noise, explicit
     cells (the double table), cancellation, a slow method refused, another method's grid;
  6. the plug-in function of the event layer on a duck-typed run.

Run:  py -3 -m pytest test_dbscan_events_rule.py -q      (about half a minute)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import dbscan_rule as rule  # noqa: E402
from tools import dbscan_sweep as sw  # noqa: E402
from tools import events_sweep_plugins as plug  # noqa: E402
from tools import mps_events as ev  # noqa: E402

SIGMA = 12.0      # [lit: ~NeNA of Gazal et al. 2026] per-axis precision of the synthetic movies
N_FRAMES = 20000  # [ad hoc] movie length, as the research's simulation


# --------------------------------------------------------------------------- synthetic movies
def blinking_movie(seed: int, *, n_emitters: int = 600, field_nm: float = 20000.0, cycles_mean: float = 14.0,
                   length_mean: float = 2.0, sigma: float = SIGMA, positions: Any = None,
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Emitters with geometric cycles (mean ``cycles_mean``), each cycle a run of consecutive frames with a geometric
    length (mean ``length_mean``), one localization per frame with N(0, sigma) per axis. Returns frame, xy,
    emitter of each localization, cycle of each localization, true cycle lengths."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, field_nm, (n_emitters, 2)) if positions is None else np.asarray(positions, float)
    n_emit = pos.shape[0]
    n_cyc = rng.geometric(1.0 / cycles_mean, n_emit)
    cyc_emitter = np.repeat(np.arange(n_emit), n_cyc)
    m = cyc_emitter.size
    start = rng.integers(1, N_FRAMES - 60, m)
    length = rng.geometric(1.0 / length_mean, m)
    loc_cyc = np.repeat(np.arange(m), length)
    frame = start[loc_cyc] + (np.arange(loc_cyc.size) - np.repeat(np.cumsum(length) - length, length))
    xy = pos[cyc_emitter[loc_cyc]] + rng.normal(size=(loc_cyc.size, 2)) * sigma
    order = np.argsort(frame, kind="stable")       # a localization file is in frame order
    return frame[order], xy[order], cyc_emitter[loc_cyc][order], loc_cyc[order], length


# --------------------------------------------------------------------------- 1. localizations per cycle
@pytest.mark.parametrize("length_mean", [1.5, 3.0])
def test_locs_per_cycle_recovered_on_a_known_movie(length_mean: float) -> None:
    frame, xy, _, _, length = blinking_movie(11, length_mean=length_mean)
    truth = float(length.mean())
    # D5 radius (3 NeNA): biased low by the predicted split, recovered by the correction
    r3 = ev.locs_per_cycle(frame, xy[:, 0], xy[:, 1], nena_nm=SIGMA, n_frames=N_FRAMES)
    assert r3.source == "linked here" and r3.measured and "[measured]" in r3.origin
    p = ev.expected_split_fraction(3.0)
    assert math.isclose(p, math.exp(-9.0 / 4.0))
    n_tot = float(length.sum())
    predicted = n_tot / (length.size + p * (n_tot - length.size))
    assert r3.value < truth
    assert abs(r3.value - predicted) / predicted < 0.03, (r3.value, predicted)
    # the correction ignores that a dark frame can bridge a broken step: within 5 % here (docs/dbscan_rule.md 4)
    assert r3.split_corrected is not None and abs(r3.split_corrected - truth) / truth < 0.05
    assert r3.ci95 is not None and r3.ci95[0] < r3.value < r3.ci95[1]
    # the research's radius (3 sqrt(2) NeNA): about 1 % of steps broken
    r4 = ev.locs_per_cycle(frame, xy[:, 0], xy[:, 1], nena_nm=SIGMA, radius_in_nena=3.0 * math.sqrt(2.0),
                           n_frames=N_FRAMES)
    assert abs(r4.value - truth) / truth < 0.03, (r4.value, truth)
    # the event-size histogram of the wide radius follows the geometric law it was drawn from
    assert r4.histogram is not None
    sizes, counts = r4.histogram
    share_one = counts[0] / counts.sum()
    assert abs(share_one - 1.0 / length_mean) < 0.04, share_one


def test_locs_per_cycle_from_a_linked_file_and_fallbacks() -> None:
    rng = np.random.default_rng(3)
    n = rng.geometric(0.5, 500)
    ln = n + rng.integers(0, 2, 500)
    got = ev.locs_per_cycle_from_columns(n_column=n, len_column=ln)
    assert got is not None and got.source == "file column n" and math.isclose(got.value, n.mean())
    upper = ev.locs_per_cycle_from_columns(len_column=ln)
    assert upper is not None and upper.source.startswith("file column len") and upper.value >= got.value
    assert ev.locs_per_cycle_from_columns() is None

    class Linked:          # a Picasso-linked file: n and len columns
        x_nm = y_nm = np.zeros(500)
        frame = np.arange(500)

        def column(self, name: str) -> Any:
            return {"n": n, "len": ln}.get(name)

    assert ev.locs_per_cycle_from_localizations(Linked(), nena_nm=SIGMA).source == "file column n"
    # fallbacks: labelled [ad hoc], never a measurement
    for got in (ev.locs_per_cycle(None, [0.0], [0.0], nena_nm=SIGMA),
                ev.locs_per_cycle(np.arange(3), np.zeros(3), np.zeros(3), nena_nm=None),
                ev.locs_per_cycle(np.arange(3), np.arange(3) * 1000.0, np.zeros(3), nena_nm=SIGMA),
                ev.locs_per_cycle_from_columns(n_column=n[:10])):
        assert got is not None and got.source == "fallback" and not got.measured
        assert got.value == ev.LOCS_PER_CYCLE_FALLBACK and "[ad hoc]" in got.origin and got.note


def test_censored_events_are_left_out_of_the_estimate() -> None:
    # one event on from frame 0 (censored), one in the middle, one still on in the last frame
    frame = np.array([0, 1, 2, 3, 100, 101, 198, 199])
    x = np.array([0, 1, 2, 1, 5000, 5001, 9000, 9001], float)
    layer = ev.link_events(frame, x, np.zeros(8), nena_nm=SIGMA, n_frames=200)
    assert layer.n_events == 3 and layer.censored.tolist() == [True, False, True]
    assert layer.n_locs.tolist() == [4, 2, 2] and layer.length.tolist() == [4, 2, 2]


# --------------------------------------------------------------------------- 2. events per cluster
def test_cluster_events_never_span_two_clusters() -> None:
    # one emitter on for 6 frames; the clustering put its first three localizations in cluster 0, the rest in 1
    frame = np.arange(10, 16)
    x = np.zeros(6)
    labels = np.array([0, 0, 0, 1, 1, 1])
    ce = ev.cluster_events(frame, x, x, labels, nena_nm=SIGMA)
    assert ce.cluster_ids.tolist() == [0, 1] and ce.n_events.tolist() == [1, 1] and ce.n_locs.tolist() == [3, 3]
    assert ce.locs_per_event_median.tolist() == [3.0, 3.0] and ce.event_on_frames_median.tolist() == [3.0, 3.0]
    # the same localizations unclustered make one event
    assert ev.link_events(frame, x, x, nena_nm=SIGMA).n_events == 1
    rows = ce.as_rows()
    assert rows[0] == {"cluster_id": 0, "n_events": 1, "n_locs": 3, "locs_per_event_median": 3.0,
                       "event_on_frames_median": 3.0}


# --------------------------------------------------------------------------- 3. the "probable fusion" mark
def _single_and_merged(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """60 isolated single labels and 60 merged pairs (two molecules 20 nm apart, two labels each: 4 labels in one
    DBSCAN cluster), 2 um apart. Returns frame, xy, true group (0..119) and whether the group is merged."""
    rng = np.random.default_rng(seed)
    centres = np.column_stack([np.arange(120) % 12 * 2000.0, np.arange(120) // 12 * 2000.0])
    pos, group = [], []
    for g, c in enumerate(centres):
        if g < 60:
            pos.append(c)
            group.append(g)
        else:
            for dx in (-10.0, 10.0):                       # two molecules 20 nm apart, two labels each
                for _ in range(2):
                    pos.append(c + [dx, 0.0] + rng.normal(0, 5.0, 2))
                    group.append(g)
    frame, xy, emitter, _, _ = blinking_movie(seed + 1, positions=np.array(pos), length_mean=1.5)
    return frame, xy, np.asarray(group)[emitter], np.arange(120) >= 60


def test_fusion_flag_marks_merged_pairs_and_changes_nothing() -> None:
    from sklearn.cluster import DBSCAN
    frame, xy, group, merged = _single_and_merged(21)
    labels = DBSCAN(eps=25.0, min_samples=10).fit_predict(xy)
    before = labels.copy()
    ce = ev.cluster_events(frame, xy[:, 0], xy[:, 1], labels, nena_nm=SIGMA)
    # each cluster's truth: the group most of its localizations come from
    truth = np.array([merged[np.bincount(group[labels == c]).argmax()] for c in ce.cluster_ids])
    est14, flag14 = ev.fusion_flags(ce.n_events, 14.0)
    est26, flag26 = ev.fusion_flags(ce.n_events, 26.0)
    assert np.allclose(est14 * 14.0, ce.n_events)
    recall = flag14[truth].mean()
    false_rate = flag14[~truth].mean()
    assert recall >= 0.85, recall                 # 4 labels x 14 cycles: about 4 labels estimated
    assert false_rate <= 0.30, false_rate         # one geometric label exceeds 28 events about 1 time in 8
    assert flag26.sum() <= flag14.sum()           # a larger c_label marks fewer clusters
    assert np.array_equal(labels, before)         # a mark only
    table = ev.fusion_table(ce.n_events)
    assert table["summary_c14"]["n_clusters_fusion_flag"] == int(flag14.sum())
    assert table["c_label_source"] == "literature_provisional" and "fusion_flag_c26" in table
    with pytest.raises(ValueError):
        ev.fusion_flags([1, 2], 0.0)


# --------------------------------------------------------------------------- 4. the rule
def test_rule_reproduces_the_worked_check_and_the_documented_values() -> None:
    ms, raw = rule.worked_check()
    assert ms == 10 and abs(raw - 9.6) < 0.01                     # docs/channel2_parameters.md section 5
    assert rule.C_LABEL_MEA == 14.0 and rule.C_LABEL_BME == 26.0 and rule.C_LABEL_DEFAULT == 14.0
    assert abs(rule.pair_probability(25.0, 12.0) - 0.662) < 0.001
    assert rule.cycles_quantile(14.0) == 4 and rule.cycles_quantile(26.0) == 8 and rule.cycles_quantile(1.0) == 1
    r14 = rule.derive_rule(12.0, c_label=14.0, locs_per_cycle=2.0)
    r26 = rule.derive_rule(12.0, c_label=26.0, locs_per_cycle=2.0)
    assert r14.eps_nm == 25.0 and r14.eps_capped                  # 2.5 x 12 = 30 -> capped at 25
    assert r14.min_samples in (5, 6) and r26.min_samples in range(10, 14)   # research plan A2
    assert (r14.min_samples, r26.min_samples) == (6, 11)
    # the research's own formula (n_q x lpc x P) gives 5 and 11
    assert rule.min_samples_rule(25.0, 12.0, 14.0, 2.0, count_self=False)[0] == 5
    assert rule.min_samples_rule(25.0, 12.0, 26.0, 2.0, count_self=False)[0] == 11
    # one localization per cycle: the floor of 3 (research plan D3)
    r1 = rule.derive_rule(12.0, c_label=14.0, locs_per_cycle=1.0)
    assert r1.min_samples == 3 and r1.inputs["loc_per_cycle"].origin == "[user] given"
    # eps below the cap with a good precision
    good = rule.derive_rule(8.0, locs_per_cycle=2.0)
    assert good.eps_nm == 20.0 and not good.eps_capped
    # every input carries its origin; the export columns of A2
    exp = r14.as_export()
    assert exp["dbscan_mode"] == "rule" and exp["eps_rule_nm"] == 25.0 and exp["min_samples_rule"] == 6
    assert "Dempsey" in exp["c_label_source"] and "provisional" in exp["c_label_source"]
    assert all(v.origin for v in r14.inputs.values())


def test_rule_with_a_measured_and_a_missing_localizations_per_cycle() -> None:
    frame, xy, _, _, length = blinking_movie(5, length_mean=2.0)
    lpc = ev.locs_per_cycle(frame, xy[:, 0], xy[:, 1], nena_nm=SIGMA, n_frames=N_FRAMES)
    r = rule.derive_rule(SIGMA, locs_per_cycle=lpc)
    assert r.inputs["loc_per_cycle"].value == lpc.value and "[measured]" in r.inputs["loc_per_cycle"].origin
    assert r.min_samples_range is not None and r.min_samples_range[0] <= r.min_samples <= r.min_samples_range[1]
    rc = rule.derive_rule(SIGMA, locs_per_cycle=lpc, use_split_corrected=True)
    assert rc.inputs["loc_per_cycle"].value == lpc.split_corrected and "corrected" in rc.inputs["loc_per_cycle"].origin
    missing = rule.derive_rule(SIGMA, locs_per_cycle=None)
    assert missing.inputs["loc_per_cycle"].value == ev.LOCS_PER_CYCLE_FALLBACK
    assert "[ad hoc]" in missing.inputs["loc_per_cycle"].origin and missing.notes
    custom = rule.derive_rule(SIGMA, c_label=20.0, locs_per_cycle=2.0)
    assert custom.inputs["c_label"].origin == "[user] typed"
    table = rule.rule_table(SIGMA, 2.0)
    assert sorted(table) == [14.0, 26.0] and table[26.0].min_samples > table[14.0].min_samples
    cells = rule.double_table_cells(r)
    assert [c["mode"] for c in cells] == ["compat_25_10", "rule"] and cells[0]["min_samples"] == 10
    assert rule.double_table_cells(None) == cells[:1]
    assert rule.method_specs(r) == [("DBSCAN", {"eps": 25.0, "min_samples": 10}),
                                    ("DBSCAN", {"eps": r.eps_nm, "min_samples": r.min_samples})]
    with pytest.raises(ValueError):
        rule.eps_rule(0.0)
    with pytest.raises(ValueError):
        rule.cycles_quantile(0.5)


# --------------------------------------------------------------------------- 5. the sweep
def _ring(seed: int = 0, n_clusters: int = 24, radius_nm: float = 400.0) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False) + rng.normal(0, 0.03, n_clusters)
    xs, ys, zs = [], [], []
    for a in ang:
        n = int(rng.integers(15, 40))
        xs.append(radius_nm * np.cos(a) + rng.normal(0, 9, n))
        ys.append(radius_nm * np.sin(a) + rng.normal(0, 9, n))
        zs.append(rng.normal(0, 20, n))
    nb = 150
    xs.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    ys.append(rng.uniform(-radius_nm * 1.3, radius_nm * 1.3, nb))
    zs.append(rng.normal(0, 60, nb))
    return np.concatenate(xs) + 2000.0, np.concatenate(ys) + 2000.0, np.concatenate(zs) + 300.0


_SWEEP: Dict[str, Any] = {}


def _dbscan_sweep() -> Tuple[Any, Any]:
    if not _SWEEP:
        from tools.mps_analysis import analyze_axon
        x, y, z = _ring()
        _SWEEP["normal"] = analyze_axon(x, y, z, run_randomization=False)
        _SWEEP["sweep"] = sw.sweep("DBSCAN", x_nm=x, y_nm=y, z_nm=z)
    return _SWEEP["normal"], _SWEEP["sweep"]


def test_sweep_grid_and_the_reference_cell_is_the_normal_analysis() -> None:
    normal, res = _dbscan_sweep()
    assert res.complete and res.n_planned == 20 and len(res.cells) == 20
    assert res.param_names == ("eps", "min_samples")
    assert res.axes == {"eps": (15.0, 20.0, 25.0, 30.0, 40.0), "min_samples": (5, 10, 15, 20)}
    assert res.grid("clusters_per_um").shape == (5, 4)
    ref = res.reference
    assert ref is not None and ref.params == {"eps": 25.0, "min_samples": 10}
    assert sum(c.is_reference for c in res.cells) == 1
    assert ref.readouts == sw.readouts_of(normal)                  # exactly the normal analysis
    assert ref.readouts["clusters_per_um"] == normal.clusters_per_um
    assert res.grid("clusters_per_um")[2, 1] == normal.clusters_per_um
    assert all(not c.error for c in res.cells)
    rows = res.rows("synthetic ring")
    assert len(rows) == 20 and rows[0]["param_eps"] == 15.0 and set(sw.READOUTS) <= set(rows[0])
    assert len(res.long_rows("synthetic ring")) == 20 * len(sw.READOUTS)


def test_sweep_noise_is_monotone_in_both_parameters() -> None:
    _, res = _dbscan_sweep()
    noise = res.grid("noise_fraction")
    # more min_samples -> fewer core points -> more noise; larger eps -> less noise (DBSCAN, Ester et al. 1996)
    assert np.all(np.diff(noise, axis=1) >= -1e-12) and np.all(np.diff(noise, axis=0) <= 1e-12)


def test_sweep_explicit_cells_cancellation_and_slow_methods() -> None:
    from tools.mps_analysis import analyze_axon
    x, y, z = _ring(1, n_clusters=12)
    r = rule.derive_rule(SIGMA, locs_per_cycle=2.0)
    double = sw.sweep("DBSCAN", x_nm=x, y_nm=y, z_nm=z, cells=rule.double_table_cells(r))
    assert [c.mode for c in double.cells] == ["compat_25_10", "rule"]
    assert double.cells[0].is_reference and not double.cells[1].is_reference
    assert double.cells[1].params == {"eps": r.eps_nm, "min_samples": r.min_samples}
    # cancelled after three cells: the three are kept
    calls = {"n": 0}

    def run_fn(method: str, params: Dict[str, Any]) -> Any:
        calls["n"] += 1
        return analyze_axon(x, y, z, run_randomization=False, eps_nm=params["eps"],
                            min_samples=params["min_samples"])

    # cancelled between cells (after the third): the three are kept
    seen: list = []
    res = sw.sweep("DBSCAN", run_fn=run_fn, cancel=lambda: len(seen) >= 3,
                   progress=lambda done, planned, left: seen.append((done, planned)))
    assert res.cancelled and not res.complete and len(res.cells) == 3 and seen[-1] == (3, 20)
    assert np.isnan(res.grid("clusters_per_um")).sum() == 17
    # cancelled inside a cell (the analysis' own checks, tools.mps_cancel): the cells before it are kept
    calls["n"] = 0
    inside = sw.sweep("DBSCAN", run_fn=run_fn, cancel=lambda: calls["n"] >= 3)
    assert inside.cancelled and len(inside.cells) == 2 and inside.notes
    # a slow method is refused without allow_slow, with the estimate
    with pytest.raises(sw.SlowMethodRefused, match="allow_slow"):
        sw.sweep("OPTICS", {"min_samples": (5, 10)}, x_nm=x, y_nm=y, z_nm=z)
    with pytest.raises(KeyError):
        sw.sweep("FOCAL", x_nm=x, y_nm=y, z_nm=z)
    # another method with its own preset grid: same readouts, its default cell marked
    hdb = sw.sweep("HDBSCAN", {"min_cluster_size": (5, 10), "min_samples": (5,)}, x_nm=x, y_nm=y, z_nm=z)
    assert hdb.complete and hdb.reference is not None and hdb.reference.params["min_cluster_size"] == 10
    assert set(hdb.reference.readouts) == set(sw.READOUTS)


# --------------------------------------------------------------------------- 6. the plug-in of the event layer
def test_events_plugin_on_a_duck_typed_run() -> None:
    from sklearn.cluster import DBSCAN
    frame, xy, _, _ = _single_and_merged(31)
    labels = DBSCAN(eps=25.0, min_samples=10).fit_predict(xy)
    an = SimpleNamespace(labels=labels, x_slab=xy[:, 0], y_slab=xy[:, 1], localization_frame_slab=frame,
                         bad_report=SimpleNamespace(bad_labels={0}))
    run = SimpleNamespace(analysis=an, method="DBSCAN", params={"eps": 25.0, "min_samples": 10})
    out = plug.events_plugin(run, SimpleNamespace(extras={"nena_nm": SIGMA}))
    assert set(out) == {r[0] for r in plug.EVENTS_READOUTS}
    assert out["events_n_fusion_c14"] >= out["events_n_fusion_c26"] > 0
    assert 0 < out["events_fraction_fusion_c14"] < 1 and out["events_locs_per_event_mean"] >= 1.0
    # without a NeNA (or frames) every readout is None, never 0
    none = plug.events_plugin(run, SimpleNamespace(extras={}))
    assert all(v is None for v in none.values())
    names = [r[0] for r in plug.EVENTS_READOUTS + plug.SWEEP_READOUTS]
    assert len(names) == len(set(names)) and all(n == n.lower() for n in names)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
