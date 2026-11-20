"""Tests of the automatic discard of unanchored clusters (package P14, ``tools.mps_unanchored``).

Simulated data only, with known truth: membrane clusters placed along a closed curve, interior lumps planted at
known depths. Run one file per process: ``python -m pytest test_unanchored.py``.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import axon_export, column_dictionary  # noqa: E402
from tools import mps_unanchored as mu  # noqa: E402
from tools import unanchored_export as ue  # noqa: E402

FAST = dict(noise_replicates=0, compare_references=False)


# ----------------------------------------------------------------------------------------------------
# Synthetic axons
# ----------------------------------------------------------------------------------------------------
def curve_points(radius_fn, n=4000, centre=(0.0, 0.0)):
    """Dense closed curve r(theta) around ``centre``."""
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = radius_fn(t)
    return np.column_stack([centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)])


def polygon_curve(vertices, n=4000):
    """Dense closed curve through polygon vertices (linear)."""
    v = np.asarray(vertices, dtype=float)
    nxt = np.roll(v, -1, axis=0)
    seg = np.hypot(*(nxt - v).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, cum[-1], n, endpoint=False)
    i = np.searchsorted(cum, s, side="right") - 1
    f = (s - cum[i]) / seg[i]
    return v[i] + f[:, None] * (nxt[i] - v[i])


def membrane_on(curve, rng, *, spacing_nm=250.0, radial_sd_nm=30.0, locs_per_cluster=80, cluster_sd_nm=15.0,
                diffuse=1000, diffuse_sd_nm=60.0):
    """Clusters every ``spacing_nm`` of arc along ``curve`` (centres scattered normal to it by ``radial_sd_nm``),
    plus diffuse localizations along the curve. Returns (x, y, true cluster centres)."""
    c = np.asarray(curve, dtype=float)
    nxt = np.roll(c, -1, axis=0)
    seg = np.hypot(*(nxt - c).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    tang = (nxt - np.roll(c, 1, axis=0))
    tang /= np.hypot(*tang.T)[:, None]
    normal = np.column_stack([tang[:, 1], -tang[:, 0]])
    n_cl = max(3, int(L // spacing_nm))
    s0 = rng.uniform(0, spacing_nm)
    xs, ys, cents = [], [], []
    for k in range(n_cl):
        s = (s0 + k * L / n_cl + rng.normal(0, 0.1 * spacing_nm)) % L
        i = int(np.searchsorted(cum, s, side="right") - 1) % len(c)
        centre = c[i] + normal[i] * rng.normal(0, radial_sd_nm)
        n = max(5, rng.poisson(locs_per_cluster))
        p = centre + rng.normal(0, cluster_sd_nm, (n, 2))
        xs.append(p[:, 0]); ys.append(p[:, 1]); cents.append(p.mean(axis=0))
    i = rng.integers(0, len(c), diffuse)
    p = c[i] + normal[i] * rng.normal(0, diffuse_sd_nm, diffuse)[:, None]
    xs.append(p[:, 0]); ys.append(p[:, 1])
    return np.concatenate(xs), np.concatenate(ys), np.asarray(cents)


def lump(at, rng, n=20, sd=15.0):
    p = np.asarray(at, dtype=float) + rng.normal(0, sd, (n, 2))
    return p[:, 0], p[:, 1], p.mean(axis=0)


def circle(r=1500.0):
    return curve_points(lambda t: np.full_like(t, r))


def with_lumps(x, y, cents, lumps, rng):
    xs, ys, cs = [x], [y], [cents]
    for at in lumps:
        lx, ly, lc = lump(at, rng)
        xs.append(lx); ys.append(ly); cs.append(lc[None, :])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def c_shape():
    """Boundary of a thick "C" (outer arc 1500 nm, inner arc 700 nm, 300 degrees): the membrane folds inward."""
    a = np.radians(np.linspace(-150, 150, 120))
    outer = np.column_stack([1500 * np.cos(a), 1500 * np.sin(a)])
    inner = np.column_stack([700 * np.cos(a[::-1]), 700 * np.sin(a[::-1])])
    return polygon_curve(np.concatenate([outer, inner]))


def two_lobes(rng):
    """Two separate discs of 900 nm, 200 nm apart (two axons in one pick): no single ring represents them."""
    xs, ys, cs = [], [], []
    for cx in (-1100.0, 1100.0):
        x, y, c = membrane_on(curve_points(lambda t: np.full_like(t, 900.0), centre=(cx, 0.0)), rng)
        xs.append(x); ys.append(y); cs.append(c)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def bay_curve(depth_nm, width_deg, r=1500.0):
    """A circle with a smooth concave bay of ``depth_nm`` and angular full width ``width_deg`` at theta = 0."""
    w = np.radians(width_deg) / 2.0

    def fn(t):
        d = np.where(np.abs(t) < w, 0.5 * (1 + np.cos(np.pi * t / w)), 0.0)
        return r - depth_nm * d
    return curve_points(fn)


# ----------------------------------------------------------------------------------------------------
# Geometry and classification
# ----------------------------------------------------------------------------------------------------
def test_signed_depth_of_a_square():
    sq = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], float)
    d = mu.signed_depth([[500, 500], [100, 500], [-50, 500], [500, 1200]], sq)
    assert np.allclose(d, [500, 100, -50, -200])


def test_classify_states_and_nan():
    s = mu.classify([0, 249.9, 250.1, 400.0, 400.1, np.nan, -300])
    assert list(s) == ["anchored", "anchored", "doubtful", "doubtful", "unanchored", "not evaluated", "anchored"]
    with pytest.raises(ValueError):
        mu.classify([1.0], exclude_nm=200, doubt_nm=300)
    with pytest.raises(ValueError):
        mu.classify([1.0], exclude_nm=float("nan"))


# ----------------------------------------------------------------------------------------------------
# Known interiors
# ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(1)
    x, y, cents = membrane_on(circle(), rng)
    n_mem = len(cents)
    depths = (150.0, 300.0, 500.0, 1000.0)
    lumps = [(np.cos(a) * (1500 - d), np.sin(a) * (1500 - d)) for a, d in zip((0.3, 1.9, 3.5, 5.0), depths)]
    x, y, cents = with_lumps(x, y, cents, lumps, rng)
    return x, y, cents, n_mem


@pytest.mark.parametrize("reference", mu.REFERENCES)
def test_lumps_planted_at_known_depths(planted, reference):
    x, y, cents, n_mem = planted
    res = mu.auto_unanchored(x, y, cents, reference=reference, noise_replicates=0)
    assert res.evaluated, res.failure
    assert list(res.state[n_mem:]) == ["anchored", "doubtful", "unanchored", "unanchored"]
    assert (res.state[:n_mem] == "anchored").all()
    assert np.allclose(res.depth_nm[n_mem:], [150, 300, 500, 1000], atol=60)
    assert res.n_changed_by_reference == 0


def test_noise_and_borderline_are_reported(planted):
    x, y, cents, n_mem = planted
    res = mu.auto_unanchored(x, y, cents, noise_replicates=10)
    assert math.isfinite(res.used.noise_nm) and res.used.noise_nm < 40
    assert res.used.depth_sd_nm.shape == (len(cents),)


def test_ring_without_interior_excludes_nothing_over_100_seeds():
    flagged = 0
    for seed in range(100):
        rng = np.random.default_rng(1000 + seed)
        x, y, cents = membrane_on(circle(1200 + 10 * seed), rng)
        res = mu.auto_unanchored(x, y, cents, **FAST)
        assert res.evaluated, (seed, res.failure)
        flagged += res.used.n_unanchored + res.used.n_doubtful
    assert flagged == 0


def test_rotation_and_translation_keep_the_verdicts(planted):
    x, y, cents, n_mem = planted
    base = mu.auto_unanchored(x, y, cents, **FAST)
    a = 0.7
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    p = np.column_stack([x, y]) @ rot.T + [12345.0, -6789.0]
    c = cents @ rot.T + [12345.0, -6789.0]
    moved = mu.auto_unanchored(p[:, 0], p[:, 1], c, **FAST)
    assert list(moved.state) == list(base.state)
    assert np.allclose(moved.depth_nm, base.depth_nm, atol=40)


def test_deterministic(planted):
    x, y, cents, _ = planted
    a = mu.auto_unanchored(x, y, cents, noise_replicates=5)
    b = mu.auto_unanchored(x, y, cents, noise_replicates=5)
    assert np.array_equal(a.depth_nm, b.depth_nm) and np.array_equal(a.used.depth_sd_nm, b.used.depth_sd_nm, equal_nan=True)
    assert list(a.state) == list(b.state)


def test_a_diametral_chain_of_lumps_is_excluded_not_mistaken_for_a_fold():
    """The pilot's worst axon: a chain of lumps across the middle. The ring must stay usable."""
    rng = np.random.default_rng(3)
    x, y, cents = membrane_on(circle(1600), rng)
    n_mem = len(cents)
    chain = [(r * np.cos(0.4), r * np.sin(0.4)) for r in np.arange(-900, 901, 150)]
    x, y, cents = with_lumps(x, y, cents, chain, rng)
    res = mu.auto_unanchored(x, y, cents, **FAST)
    assert res.evaluated, res.failure
    assert res.used.ring.inner_band_arc_nm < mu.INNER_BAND_MAX_ARC_NM
    assert (res.state[n_mem:] == "unanchored").all()
    assert (res.state[:n_mem] == "anchored").all()


@pytest.mark.parametrize("shape", ["c", "two_lobes"])
def test_folded_membranes_reject_the_ring_and_exclude_nothing(shape):
    rng = np.random.default_rng(4)
    x, y, cents = membrane_on(c_shape(), rng) if shape == "c" else two_lobes(rng)
    res = mu.auto_unanchored(x, y, cents, compare_references=False, noise_replicates=0)
    assert not res.evaluated
    assert res.status == mu.FAIL_NOT_STAR, (res.status, res.used.ring.inner_band_arc_nm)
    assert (res.state == mu.STATE_NOT_EVALUATED).all()
    flags = mu.final_flags(res)
    assert flags.n_excluded == 0 and "not evaluated" in flags.warnings[0]


@pytest.mark.parametrize("depth,width", [(300.0, 60.0), (300.0, 40.0)])
def test_a_concave_bay_of_300_nm_is_not_excluded(depth, width):
    rng = np.random.default_rng(5)
    x, y, cents = membrane_on(bay_curve(depth, width), rng)
    res = mu.auto_unanchored(x, y, cents, **FAST)
    assert res.evaluated, res.failure
    assert res.used.n_unanchored == 0


def test_failure_states_are_explicit():
    rng = np.random.default_rng(6)
    x, y, cents = membrane_on(circle(), rng, locs_per_cluster=2, diffuse=0)
    res = mu.auto_unanchored(x[:100], y[:100], cents, **FAST)
    assert res.status == mu.FAIL_TOO_FEW_LOCS and "nothing was excluded" in res.warnings[0]
    smooth = mu.auto_unanchored(x, y, cents[:5], reference=mu.REFERENCE_SMOOTH, **FAST)
    assert smooth.status == mu.FAIL_TOO_FEW_CENTRES
    with pytest.raises(ValueError):
        mu.auto_unanchored(x, y, cents, reference="made up")


def test_gate_warns_when_too_many_are_excluded():
    rng = np.random.default_rng(7)
    x, y, cents = membrane_on(circle(), rng)
    # 15 lumps scattered over the interior at different radii (not a ring of lumps, which would read as a fold)
    angles = np.linspace(0, 2 * np.pi, 15, endpoint=False)
    radii = np.linspace(100.0, 1000.0, 15)[np.random.default_rng(70).permutation(15)]
    inner = [(r * np.cos(a), r * np.sin(a)) for a, r in zip(angles, radii)]
    x, y, cents = with_lumps(x, y, cents, inner, rng)
    res = mu.auto_unanchored(x, y, cents, **FAST)
    assert res.evaluated
    assert any("check the axon and the ring" in w for w in res.warnings)
    assert mu.unanchored_block(mu.final_flags(res))["unanchored_gate_exceeded"] is True


# ----------------------------------------------------------------------------------------------------
# Decisions by hand
# ----------------------------------------------------------------------------------------------------
def test_decisions_round_trip_and_survive_renumbering(tmp_path, planted):
    x, y, cents, n_mem = planted
    res = mu.auto_unanchored(x, y, cents, **FAST)
    data = tmp_path / "axon7.hdf5"
    path = mu.decisions_path(str(data))
    assert path.endswith("axon7_unanchored.json")
    dec = mu.load_decisions(path)
    assert len(dec) == 0
    # restore the deepest lump, exclude membrane cluster 0
    deep = n_mem + 3
    dec = mu.add_decision(dec, *cents[deep], "restore", "looks like membrane", date="2026-11-17T10:00:00Z")
    dec = mu.add_decision(dec, *cents[0], "exclude", "glia", date="2026-11-17T10:01:00Z")
    mu.save_decisions(replace_source(dec, "axon7.hdf5"))
    back = mu.load_decisions(path)
    assert back.decisions == dec.decisions and back.source == "axon7.hdf5"
    flags = mu.final_flags(res, back, data_source="axon7.hdf5")
    assert not flags.excluded[deep] and flags.excluded[0] and flags.excluded[n_mem + 2]
    assert flags.source[deep] == mu.SOURCE_RESTORED and flags.source[0] == mu.SOURCE_MANUAL
    assert flags.discard_source == "automatic+manual"
    assert flags.n_excluded_manual == 1 and flags.n_restored_manual == 1
    # a re-analysis renumbers the clusters and moves their centres by a few nm: the decisions follow the positions
    rng = np.random.default_rng(9)
    order = rng.permutation(len(cents))
    c2 = cents[order] + rng.normal(0, 8, cents.shape)
    res2 = mu.auto_unanchored(x, y, c2, **FAST)
    flags2 = mu.final_flags(res2, back)
    where_deep = int(np.flatnonzero(order == deep)[0])
    where_0 = int(np.flatnonzero(order == 0)[0])
    assert not flags2.excluded[where_deep] and flags2.excluded[where_0]
    assert flags2.orphans == ()


def replace_source(dec, source):
    from dataclasses import replace
    return replace(dec, source=source)


def test_latest_decision_wins_and_remove_returns_to_automatic(planted):
    x, y, cents, n_mem = planted
    res = mu.auto_unanchored(x, y, cents, **FAST)
    dec = mu.ManualDecisions(path="")
    dec = mu.add_decision(dec, *cents[1], "exclude", "interior")
    dec = mu.add_decision(dec, cents[1][0] + 10, cents[1][1], "restore", "")
    assert len(dec) == 1 and dec.decisions[0].action == "restore"
    dec = mu.remove_decision(dec, *cents[1])
    assert len(dec) == 0
    flags = mu.final_flags(res, dec)
    assert np.array_equal(flags.excluded, res.unanchored)


def test_orphan_decisions_are_kept_and_reported(planted):
    x, y, cents, _ = planted
    res = mu.auto_unanchored(x, y, cents, **FAST)
    dec = mu.add_decision(mu.ManualDecisions(), 99999.0, 99999.0, "exclude", "artefact")
    flags = mu.final_flags(res, dec)
    assert len(flags.orphans) == 1 and any("match no cluster" in w for w in flags.warnings)
    assert mu.unanchored_block(flags)["unanchored_manual_n_orphan"] == 1


def test_bad_decision_files_are_refused_never_overwritten(tmp_path):
    other = tmp_path / "x_unanchored.json"
    other.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    with pytest.raises(mu.ManualDecisionsError):
        mu.load_decisions(str(other))
    with pytest.raises(mu.ManualDecisionsError):
        mu.save_decisions(mu.ManualDecisions(), str(other))
    assert json.loads(other.read_text(encoding="utf-8")) == {"something": "else"}
    bad = tmp_path / "y_unanchored.json"
    bad.write_text(json.dumps({"format": mu.DECISIONS_FORMAT, "decisions": [
        {"x_nm": 1, "y_nm": 2, "action": "exclude", "reason": ""}]}), encoding="utf-8")
    with pytest.raises(mu.ManualDecisionsError, match="without a reason"):
        mu.load_decisions(str(bad))
    with pytest.raises(mu.ManualDecisionsError):
        mu.add_decision(mu.ManualDecisions(), 0, 0, "delete", "x")


def test_source_mismatch_is_warned(planted):
    x, y, cents, _ = planted
    res = mu.auto_unanchored(x, y, cents, **FAST)
    dec = mu.ManualDecisions(source="axonA.hdf5")
    flags = mu.final_flags(res, dec, data_source="axonB.hdf5")
    assert any("were made on axonA.hdf5" in w for w in flags.warnings)


# ----------------------------------------------------------------------------------------------------
# Methods, preview, export
# ----------------------------------------------------------------------------------------------------
def test_widefield_and_off_methods(planted):
    x, y, cents, n_mem = planted
    res = mu.auto_unanchored(x, y, cents, **FAST)
    wf = np.zeros(len(cents), bool)
    wf[-1] = True
    flags = mu.final_flags(res, method="widefield", widefield=wf)
    assert flags.n_excluded == 1 and flags.discard_source == "widefield" and flags.n_widefield == 1
    off = mu.final_flags(None, mu.add_decision(mu.ManualDecisions(), *cents[2], "exclude", "glia"),
                         method="off", centroids=cents)
    assert off.n_excluded == 1 and off.discard_source == "manual"
    with pytest.raises(ValueError):
        mu.final_flags(res, method="widefield")
    none = mu.final_flags(None, method="off", centroids=cents)
    assert none.discard_source == "none"


def test_record_sources_reads_what_record_writes(planted):
    x, y, cents, _ = planted
    res = mu.auto_unanchored(x, y, cents, **FAST)
    hand = mu.add_decision(mu.ManualDecisions(), *cents[2], "exclude", "glia")
    wf = np.zeros(len(cents), bool)
    wf[-1] = True
    assert mu.record_sources(mu.final_flags(res).record()) == ("automatic", 0)
    assert mu.record_sources(mu.final_flags(res, hand).record()) == ("automatic", 1)
    assert mu.record_sources(mu.final_flags(res, method="widefield", widefield=wf).record()) == ("widefield", 0)
    assert mu.record_sources(mu.final_flags(None, hand, method="off", centroids=cents).record()) == ("off", 1)
    # not a record of this window: the Axoplasm panel's discard keeps "", anything else names no method
    for other in ("", "method", "method bogus; 2 decision(s) by hand", "widefield", None):
        assert mu.record_sources(other) == ("", 0), other


def test_the_discarded_row_names_who_discarded(axon_with_lumps):
    """Manual test of 2026-11-19: a cluster excluded by hand in the Unanchored window was drawn on the map as
    'Discarded by the axoplasm panel'. The row's label names the real source; the Axoplasm panel's stays."""
    from tools import mps_analysis as ma
    from tools import mps_axon_map_layers as L
    an = axon_with_lumps
    res = mu.auto_for_analysis(an, **FAST)
    flags = mu.final_flags(res)
    anchored = int(np.flatnonzero(~flags.excluded)[0])
    hand = mu.add_decision(mu.ManualDecisions(), *np.asarray(an.centroids)[anchored], "exclude", "glia")

    def row(cmp, shown):
        groups = L.build_map(L.MapInputs(analysis=an, comparison=cmp, shown=shown))
        return next(layer for g in groups for layer in g.layers if layer.key == "c_discarded")

    rule = mu.compare_unanchored(an, flags)
    both = mu.compare_unanchored(an, mu.final_flags(res, hand))
    by_hand = mu.compare_unanchored(an, mu.final_flags(None, hand, method="off", centroids=an.centroids,
                                                       labels=flags.labels))
    panel = ma.compare_discard(an, flags.excluded, margin_nm=400.0)
    for cmp, label, n in ((rule, "Discarded by the unanchored rule", 2),
                          (both, "Discarded (unanchored rule / by hand)", 3),
                          (by_hand, "Discarded by hand", 1),
                          (panel, "Discarded by the axoplasm panel", 2)):
        shown = row(cmp, "discard")
        assert shown.label == label and shown.enabled and shown.count == n, (label, shown.label, shown.count)
        measured = row(cmp, "measured")                  # the same name, greyed, while the measured one is shown
        assert measured.label == label and not measured.enabled, (label, measured.label)
        if cmp is panel:
            assert "Axoplasm panel" in shown.tip and "Inside both images" in shown.tip
        else:
            assert "Unanchored clusters window" in shown.tip and cmp.discard_applied.discard_source in shown.tip
            assert "axoplasm panel" not in shown.label.lower()
    assert "1 decision(s) by hand" in row(both, "discard").tip
    slab = next(layer for g in L.build_map(L.MapInputs(analysis=an, comparison=both, shown="discard"))
                for layer in g.layers if layer.key == "slab_discarded")
    assert "Unanchored clusters window" in slab.tip and "Axoplasm panel discarded" not in slab.tip
    # no comparison yet: a name that claims no source
    assert row(None, "measured").label == "Discarded"
    assert L.discard_by(None)[0] == L.discard_by(rule.all_clusters)[0] == "Discarded"


def test_preview_changes_nothing(planted):
    x, y, cents, n_mem = planted
    res = mu.auto_unanchored(x, y, cents, **FAST)
    before = res.depth_nm.copy()
    flags = mu.final_flags(res)
    pv = mu.preview(flags)
    assert pv.text.startswith("would exclude 2 of") and "1 doubtful" in pv.text
    assert len(pv.excluded_xy) == 2 and len(pv.doubtful_xy) == 1 and pv.ring is not None
    assert np.array_equal(before, res.depth_nm)


def test_not_run_block_and_columns():
    b = ue.not_run_block()
    assert list(b) == list(ue.UNANCHORED_COLUMNS) and b["unanchored_status"] == "not run"
    assert len(set(ue.UNANCHORED_COLUMNS)) == len(ue.UNANCHORED_COLUMNS)
    assert mu.unanchored_block(None) == b
    assert column_dictionary.missing(list(ue.UNANCHORED_COLUMNS) + list(ue.CLUSTER_UNANCHORED_COLUMNS)) == []
    assert column_dictionary.unit_of("unanchored_fraction") == "fraction (0-1)"


def test_block_values(planted):
    x, y, cents, n_mem = planted
    res = mu.auto_unanchored(x, y, cents, noise_replicates=4)
    flags = mu.final_flags(res)
    b = mu.unanchored_block(flags)
    assert list(b) == list(ue.UNANCHORED_COLUMNS)
    assert b["unanchored_method"] == "automatic" and b["unanchored_status"] == "ok"
    assert b["unanchored_n_auto"] == 2 and b["unanchored_n_doubtful"] == 1 and b["unanchored_n_final"] == 2
    assert b["unanchored_reference"] == mu.REFERENCE_POLAR and b["unanchored_exclude_nm"] == 400.0
    assert b["unanchored_other_reference"] == mu.REFERENCE_SMOOTH and b["unanchored_n_changed_by_reference"] == 0
    assert b["unanchored_labels_auto"] == f"{n_mem + 2} {n_mem + 3}"
    assert b["unanchored_discard_source"] == "automatic"
    assert all(";" not in str(v) for v in b.values())
    rows = mu.cluster_rows(flags)
    assert rows[n_mem + 3]["anchor_state"] == "unanchored" and rows[n_mem + 3]["anchor_source"] == "automatic"
    assert rows[n_mem + 1]["anchor_state"] == "doubtful" and rows[n_mem + 1]["anchor_source"] is None
    assert list(rows[0]) == list(ue.CLUSTER_UNANCHORED_COLUMNS)


def test_failed_reference_block_says_why():
    rng = np.random.default_rng(4)
    x, y, cents = membrane_on(c_shape(), rng)
    b = mu.unanchored_block(mu.final_flags(mu.auto_unanchored(x, y, cents, noise_replicates=0)))
    assert b["unanchored_status"] == mu.FAIL_NOT_STAR and "folds inward" in b["unanchored_failure"]
    assert b["unanchored_n_auto"] is None and b["unanchored_n_final"] == 0


# ----------------------------------------------------------------------------------------------------
# With and without the discard, on the real analysis
# ----------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def axon_with_lumps():
    from tools.mps_analysis import analyze_axon
    rng = np.random.default_rng(11)
    x, y, cents = membrane_on(circle(1500), rng, locs_per_cluster=40, cluster_sd_nm=12.0, diffuse=0)
    lumps = [(0.0, 0.0), (300.0, -200.0)]
    x, y, _ = with_lumps(x, y, cents, lumps, rng)
    z = rng.normal(0.0, 30.0, x.size)
    an = analyze_axon(x, y, z, source_name="sim_lumps", pixel_size_nm=130.0, pixel_size_source="override",
                      run_randomization=False)
    return an


def test_compare_unanchored_equals_the_same_flags_by_hand(axon_with_lumps):
    from tools import mps_analysis as ma
    an = axon_with_lumps
    res = mu.auto_for_analysis(an, **FAST)
    assert res.evaluated, res.failure
    flags = mu.final_flags(res)
    assert flags.n_excluded == 2, (flags.n_excluded, res.depth_nm.max())
    cmp_auto = mu.compare_unanchored(an, flags)
    cmp_hand = ma.compare_discard(an, flags.excluded, margin_nm=400.0)
    assert cmp_auto.discard_applied.perimeter_um == pytest.approx(cmp_hand.discard_applied.perimeter_um)
    assert cmp_auto.discard_applied.n_clusters_kept == an.n_clusters_kept - 2
    assert cmp_auto.discard_applied.discard_source.startswith("method automatic")
    assert any("Discard applied (method automatic" in w for w in cmp_auto.discard_applied.warnings)
    assert axon_export.analysis_id(cmp_auto.discard_applied) != axon_export.analysis_id(cmp_hand.discard_applied)
    # nothing excluded: the two columns are the same numbers
    none = mu.final_flags(res, method="off")
    cmp_none = mu.compare_unanchored(an, none)
    assert cmp_none.discard_applied.perimeter_um == pytest.approx(cmp_none.all_clusters.perimeter_um)


def test_axon_row_v18_block_is_last_and_cluster_columns(axon_with_lumps):
    # v18's block follows v17's; v19 (P10b, the perimeter contour assessment and pooling gate) is appended after it.
    from tools import perimeter_export as pe
    an = axon_with_lumps
    res = mu.auto_for_analysis(an, **FAST)
    flags = mu.final_flags(res)
    cmp = mu.compare_unanchored(an, flags)
    assert axon_export.TABLE_VERSION == "axon tables v22"
    bare = axon_export.axon_row(cmp.all_clusters, exported_at="t")
    row = axon_export.axon_row(cmp.all_clusters, discard=cmp.discard_applied, exported_at="t",
                               unanchored=mu.unanchored_block(flags))
    n = len(ue.UNANCHORED_COLUMNS)
    last = (list(pe.PERIMETER_V19_COLUMNS) + list(pe.PERIMETER_V20_COLUMNS)
            + list(pe.PERIMETER_V21_COLUMNS) + list(pe.PERIMETER_V22_COLUMNS))   # v19, v20, v21, v22, last
    n19 = len(last)
    assert list(row)[-n19:] == last == list(bare)[-n19:]
    names = list(row)[:-n19]
    assert names[-n:] == list(ue.UNANCHORED_COLUMNS) == list(bare)[:-n19][-n:]
    assert names[:-n][-len(pe.PERIMETER_COLUMNS):] == list(pe.PERIMETER_COLUMNS)
    assert bare["unanchored_status"] == "not run" and row["unanchored_n_final"] == 2
    assert column_dictionary.known_columns()[:-n19][-n:] == list(ue.UNANCHORED_COLUMNS)
    assert column_dictionary.missing(list(row)) == []
    bad = dict(mu.unanchored_block(flags), unanchored_made_up=1)
    with pytest.raises(ValueError, match="unanchored_made_up"):
        axon_export.axon_row(cmp.all_clusters, exported_at="t", unanchored=bad)
    tables = axon_export.build_tables(cmp.all_clusters, discard=cmp.discard_applied, with_clusters=True,
                                      unanchored=mu.unanchored_block(flags),
                                      unanchored_rows=mu.cluster_rows(flags))
    crow = tables.clusters[0]
    assert list(crow)[-len(ue.CLUSTER_UNANCHORED_COLUMNS):] == list(ue.CLUSTER_UNANCHORED_COLUMNS)
    assert sum(1 for r in tables.clusters if r["anchor_state"] == "unanchored") == 2
    assert column_dictionary.missing(list(crow)) == []
    plain = axon_export.build_tables(cmp.all_clusters, with_clusters=True)
    assert all(r["anchor_state"] is None for r in plain.clusters)


# ----------------------------------------------------------------------------------------------------
# Trim before the perimeter contour (D8, off by default)
# ----------------------------------------------------------------------------------------------------
def test_trim_interior_removes_lumps_and_spares_the_ring(planted):
    x, y, cents, n_mem = planted
    t = mu.trim_interior(x, y, depth_nm=250.0)
    assert t.status == "trimmed"
    assert 55 <= t.n_trimmed <= 70          # the lumps at 300, 500 and 1000 nm (20 each) and stray diffuse ones
    r = np.hypot(x, y)
    assert t.keep[r > 1400].mean() > 0.995


def test_trim_does_nothing_on_a_folded_membrane():
    rng = np.random.default_rng(4)
    x, y, _ = membrane_on(c_shape(), rng)
    t = mu.trim_interior(x, y)
    assert t.n_trimmed == 0 and t.status.startswith("not trimmed")


def test_trim_needs_the_strict_rules_of_the_polar_curve_a_c_shape_is_not_trimmed():
    """User's decision 2026-11-17 (D8): a "C" (band of 1200 nm on a 210 degree arc) whose cloud the weak rule accepts
    as star-shaped: against its polar ring the weak rule trims ~23 % of the TRUE membrane; the polar curve fails the
    strict rules of the automatic contour (inward notch), so the program's trim leaves it untouched."""
    import validate_auto_contour as vac
    sim = vac.simulate_ring("C", 4002, **vac.CONDITIONS["base_bg15"])
    memb = sim.kind == vac.KIND_MEMBRANE
    weak = mu.trim_interior(sim.x, sim.y, depth_nm=250.0, require_strict=False)
    assert weak.status == "trimmed" and np.mean(~weak.keep[memb]) > 0.05      # the danger the decision removes
    strict = mu.trim_interior(sim.x, sim.y, depth_nm=250.0)
    assert strict.n_trimmed == 0 and bool(strict.keep.all())
    assert strict.status.startswith(mu.TRIM_STRICT_PREFIX)
    valid, why = mu.strict_polar_reference(sim.x, sim.y)
    assert not valid and why
    # a shape whose polar curve passes the strict rules is still trimmed, with little membrane lost
    ell = vac.simulate_ring("ellipse", 4000, **vac.CONDITIONS["base_bg15"])
    t = mu.trim_interior(ell.x, ell.y, depth_nm=250.0)
    assert t.status == "trimmed" and t.n_trimmed > 0
    assert np.mean(~t.keep[ell.kind == vac.KIND_MEMBRANE]) < 0.01


def test_trim_is_off_by_default_in_the_perimeter_params():
    from tools.mps_perimeter_types import PerimeterParams
    assert PerimeterParams().contour_trim_interior_nm is None


# ----------------------------------------------------------------------------------------------------
# Hygiene
# ----------------------------------------------------------------------------------------------------
def test_the_forbidden_name_is_nowhere_in_the_files_of_this_package():
    root = Path(__file__).resolve().parent
    needle = "un" + "sain"
    for rel in ("tools/mps_unanchored.py", "tools/unanchored_export.py", "test_unanchored.py",
                "tools/column_dictionary.py", "tools/axon_export.py"):
        assert needle not in (root / rel).read_text(encoding="utf-8").lower(), rel
