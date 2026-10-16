# -*- coding: utf-8 -*-
"""
"eps = auto" opens the k-distance graph (decision P-R58; docs/eps_kdistance.md): the logic, without Qt.

  1. the convention: k = min samples - 1 other points; a k-distance <= eps is exactly a DBSCAN core point;
  2. the valley: found between clustered localizations and a background, with its bootstrap interval, the same on
     every call (seeded); NOT found on a background alone, on clusters without a background, on equal distances or
     on too few points -- each with its reason, and with no number to fall back on;
  3. the record of how eps was chosen: built, cleaned, applied only to the eps it describes (otherwise typed, or the
     published default), and written as the export's columns;
  4. the settings keep it, drop it when it no longer describes eps, and still load a file that says "auto" for eps
     (the rest of the file is kept) or that predates the record;
  5. the batch carries it; the column dictionary describes every new column; the algorithm labels: the adaptive
     entry under its new name and under "Auto".

Synthetic data only (R8). Run:  venv\\Scripts\\python.exe test_kdistance.py

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from typing import Callable, Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PASSED = 0
FAILED = 0


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


def blobs_and_background(n_clusters: int = 40, per: int = 30, sd: float = 12.0, f_bg: float = 0.5,
                         seed: int = 0) -> tuple:
    """Gaussian clusters on a 3 x 3 um field plus a uniform background that is ``f_bg`` of all points."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(300.0, 2700.0, size=(n_clusters, 2))
    pts = np.concatenate([c + rng.normal(0.0, sd, size=(per, 2)) for c in centres])
    n_bg = int(round(f_bg / (1.0 - f_bg) * pts.shape[0])) if f_bg > 0 else 0
    bg = rng.uniform(0.0, 3000.0, size=(n_bg, 2))
    allp = np.concatenate([pts, bg]) if n_bg else pts
    return allp[:, 0], allp[:, 1]


def main() -> int:
    from tools import mps_kdistance as kdm

    print("=" * 100)
    print("EPS = AUTO: THE K-DISTANCE GRAPH, ITS VALLEY AND THE RECORD OF THE CHOICE (synthetic, no Qt)")
    print("=" * 100)

    print("\n1. The convention")

    def convention() -> str:
        from sklearn.cluster import DBSCAN
        assert kdm.k_rank(10) == 9 and kdm.k_rank(2) == 1
        try:
            kdm.k_rank(1)
            raise AssertionError("min samples 1 has no k-distance")
        except ValueError:
            pass
        # four points on a line, 0 10 20 100: the 1st and 2nd other neighbours by hand
        x = np.array([0.0, 10.0, 20.0, 100.0])
        y = np.zeros(4)
        assert np.allclose(kdm.k_distances(x, y, 2), [10.0, 10.0, 10.0, 80.0])
        assert np.allclose(kdm.k_distances(x, y, 3), [20.0, 10.0, 20.0, 90.0])
        # exact duplicates are other points at distance 0, as DBSCAN counts them
        assert np.allclose(kdm.k_distances(np.zeros(3), np.zeros(3), 3), 0.0)
        # core points <=> k-distance <= eps, on a random cloud, for several eps
        xs, ys = blobs_and_background(seed=3)
        for ms in (5, 10):
            kd = kdm.k_distances(xs, ys, ms)
            for eps in (15.0, 25.0, 40.0):
                core = np.zeros(xs.size, bool)
                core[DBSCAN(eps=eps, min_samples=ms).fit(np.column_stack([xs, ys])).core_sample_indices_] = True
                assert np.array_equal(core, kd <= eps), (ms, eps)
                assert abs(kdm.noncore_fraction(kd, eps) - np.mean(~core)) < 1e-12
        u = kdm.unclustered_fraction(xs, ys, 25.0, 10)
        assert 0.0 <= u <= kdm.noncore_fraction(kdm.k_distances(xs, ys, 10), 25.0), "border points join clusters"
        try:
            kdm.k_distances(x[:2], y[:2], 10)
            raise AssertionError("fewer points than min samples")
        except ValueError:
            pass
        return "k = Min Pts - 1; k-distance <= eps is exactly sklearn's core set (2 Min Pts x 3 eps)"

    check("k = min samples - 1, and core points read off the curve", convention)

    print("\n2. The valley, and its absence")

    def valley_found() -> str:
        x, y = blobs_and_background(f_bg=0.5, seed=1)
        kd = kdm.k_distances(x, y, 10)
        v = kdm.find_valley(kd)
        assert v.found, v.reason
        assert v.left_mode_nm < v.eps_nm < v.right_mode_nm
        assert v.low_nm <= v.eps_nm <= v.high_nm and v.presence >= kdm.PRESENCE_LEVEL
        assert v.dip_z > kdm.DIP_Z and v.n_modes >= 2
        again = kdm.find_valley(kd)
        assert (again.eps_nm, again.low_nm, again.high_nm) == (v.eps_nm, v.low_nm, v.high_nm), "seeded"
        words = kdm.valley_words(v)
        assert "Valley at" in words and "bootstrap" in words
        return f"valley {v.eps_nm:.1f} nm [{v.low_nm:.1f}-{v.high_nm:.1f}], modes {v.left_mode_nm:.0f}/" \
               f"{v.right_mode_nm:.0f} nm, presence {v.presence:.2f}"

    def no_valley() -> str:
        rng = np.random.default_rng(7)
        reasons = []
        # a background alone (complete spatial randomness)
        x, y = rng.uniform(0, 3000, 5000), rng.uniform(0, 3000, 5000)
        v = kdm.find_valley(kdm.k_distances(x, y, 10))
        assert not v.found and v.reason and "no" in v.reason, v.reason
        reasons.append(v.reason.split(":")[0])
        # clusters without a background
        x, y = blobs_and_background(f_bg=0.0, seed=2)
        v = kdm.find_valley(kdm.k_distances(x, y, 10))
        assert not v.found, (v.eps_nm, v.reason)
        reasons.append(v.reason.split(":")[0])
        # equal distances, and too few points
        v = kdm.find_valley(np.full(100, 30.0))
        assert not v.found and "equal" in v.reason
        v = kdm.find_valley(np.array([10.0, 20.0, 30.0]))
        assert not v.found and "too few" in v.reason
        assert "No valley" in kdm.valley_words(v)
        # nothing to fall back on: no suggestion, whatever the reason
        rec = kdm.choice_record("kdist_click", 31.0, 10, valley=v, unclustered=0.2)
        assert rec["valley_found"] is False and "valley_nm" not in rec
        return "; ".join(reasons)

    def unstable_valley() -> str:
        # a valley that a bootstrap does not find again often enough is no valley
        x, y = blobs_and_background(f_bg=0.5, seed=1)
        kd = kdm.k_distances(x, y, 10)
        v = kdm.find_valley(kd, level=1.01)
        assert not v.found and "bootstrap" in v.reason, v.reason
        v = kdm.find_valley(kd, dip_z=1e9)
        assert not v.found and "noise" in v.reason, v.reason
        return "presence below the level, or a dip within the noise: no valley"

    check("a valley between clusters and background: suggested with its interval, reproducibly", valley_found)
    check("no valley on a background alone, on clusters alone, on equal or too few distances", no_valley)
    check("an unstable or shallow valley is no valley", unstable_valley)

    print("\n3. The record of how eps was chosen")

    def record() -> str:
        x, y = blobs_and_background(f_bg=0.5, seed=1)
        v = kdm.find_valley(kdm.k_distances(x, y, 10))
        rec = kdm.choice_record("kdist_valley", round(v.eps_nm, 1), 10, valley=v, unclustered=0.31,
                                file_name="sim.hdf5", n_locs=int(x.size))
        assert rec["k"] == 9 and rec["valley_found"] and rec["file"] == "sim.hdf5"
        assert kdm.clean_record(rec) == rec, "a record survives cleaning"
        cols = kdm.export_columns(rec, rec["eps_nm"])
        assert tuple(cols) == kdm.EPS_CHOICE_COLUMNS
        assert cols["eps_source"] == "kdist_valley" and cols["eps_kdist_k"] == 9
        assert cols["eps_kdist_valley_nm"] == v.eps_nm and cols["eps_kdist_unclustered_fraction"] == 0.31
        assert cols["eps_kdist_file"] == "sim.hdf5"
        # another eps since: typed, or the published default -- and nothing of the graph
        typed = kdm.export_columns(rec, 30.0)
        assert typed["eps_source"] == "typed" and typed["eps_kdist_k"] is None
        default = kdm.export_columns(rec, kdm.GAZAL_EPS_NM)
        assert default["eps_source"] == "gazal_default" and default["eps_kdist_valley_nm"] is None
        assert kdm.export_columns({}, 25.0)["eps_source"] == "gazal_default"
        assert kdm.export_columns(None, 12.5)["eps_source"] == "typed"
        # the published value chosen ON the graph keeps the graph's columns
        g = kdm.choice_record("gazal_default", 25.0, 10, valley=v, unclustered=0.4)
        gc = kdm.export_columns(g, 25.0)
        assert gc["eps_source"] == "gazal_default" and gc["eps_kdist_k"] == 9
        # junk is no record
        for junk in (None, "auto", {"source": "magic", "eps_nm": 25.0}, {"source": "typed"},
                     {"source": "kdist_click", "eps_nm": "x"}):
            assert kdm.clean_record(junk) == {}, junk
        try:
            kdm.choice_record("auto", 25.0, 10)
            raise AssertionError("an unknown source is refused")
        except ValueError:
            pass
        words = kdm.source_words(cols)
        assert "valley" in words and "k = 9" in words
        return words

    check("built, cleaned, applied only to the eps it describes, written as columns", record)

    print("\n4. The settings")

    def settings() -> str:
        from tools import mps_settings as ms
        with tempfile.TemporaryDirectory() as d:
            s = ms.MPSSettings(eps_nm=31.2)
            s.eps_choice = kdm.choice_record("kdist_click", 31.2, 10, unclustered=0.2, file_name="a.hdf5")
            assert ms.save_settings(s.validate(), d)
            back = ms.load_settings(d)
            assert back.eps_choice["source"] == "kdist_click" and back.eps_choice["eps_nm"] == 31.2
            # a record that no longer describes eps is dropped
            with open(ms.settings_path(d), encoding="utf-8") as f:
                data = json.load(f)
            data["eps_nm"] = 30.0
            with open(ms.settings_path(d), "w", encoding="utf-8") as f:
                json.dump(data, f)
            assert ms.load_settings(d).eps_choice == {}
            # a file from before the record loads, with an empty one
            data.pop("eps_choice")
            with open(ms.settings_path(d), "w", encoding="utf-8") as f:
                json.dump(data, f)
            old = ms.load_settings(d)
            assert old.eps_choice == {} and old.eps_nm == 30.0
            # a file that says "auto" for eps: the default for eps, the rest kept
            data.update(eps_nm="auto", min_samples=8, slab_half_width_nm=80.0)
            with open(ms.settings_path(d), "w", encoding="utf-8") as f:
                json.dump(data, f)
            auto = ms.load_settings(d)
            assert auto.eps_nm == ms.DEFAULT_EPS_NM and auto.min_samples == 8 and auto.slab_half_width_nm == 80.0
        return "kept; dropped when stale; old files load; eps 'auto' -> default, rest kept"

    check("the settings keep the record and still load old files", settings)

    print("\n5. Batch, dictionary, algorithm labels")

    def batch_and_dictionary() -> str:
        from tools import column_dictionary as cd
        from tools import mps_batch as mb
        from tools import mps_settings as ms
        from tools.axon_export import EPS_CHOICE_COLUMNS, TABLE_VERSION
        s = ms.MPSSettings(eps_nm=31.2)
        s.eps_choice = kdm.choice_record("kdist_click", 31.2, 10, unclustered=0.2, file_name="a.hdf5")
        b = mb.BatchSettings.from_settings(s)
        assert b.eps_choice == s.eps_choice and b.eps_choice is not s.eps_choice
        assert mb.BatchSettings.from_settings(ms.MPSSettings()).eps_choice == {}
        assert set(EPS_CHOICE_COLUMNS) <= set(cd.known_columns())
        assert not cd.missing(list(EPS_CHOICE_COLUMNS)), cd.missing(list(EPS_CHOICE_COLUMNS))
        assert cd.unit_of("eps_kdist_valley_nm") == "nm" and cd.unit_of("eps_kdist_unclustered_fraction") == ""
        assert TABLE_VERSION == "axon tables v9"
        return f"{len(EPS_CHOICE_COLUMNS)} columns described; {TABLE_VERSION}"

    def labels() -> str:
        from tools import clustering_strategies as cs
        assert cs.strategy_for_label("Auto") == "auto"
        assert cs.strategy_for_label(cs.ADAPTIVE_LABEL) == "auto"
        assert cs.strategy_for_label("DBSCAN") == "dbscan" and cs.strategy_for_label("HDBSCAN") == "hdbscan"
        assert cs.strategy_for_label("something else") == "dbscan"
        assert cs.label_for("Auto") == cs.ADAPTIVE_LABEL and cs.label_for("HDBSCAN") == "HDBSCAN"
        assert "exploratory" in cs.ADAPTIVE_LABEL.lower()
        from tools.mps_tooltips import ALGORITHM_ITEMS, MAIN_WINDOW
        assert set(ALGORITHM_ITEMS) == set(cs.ALGORITHM_LABELS)
        assert "exploratory" in ALGORITHM_ITEMS[cs.ADAPTIVE_LABEL].lower()
        assert "Auto" in ALGORITHM_ITEMS[cs.ADAPTIVE_LABEL] and "HDBSCAN" in ALGORITHM_ITEMS[cs.ADAPTIVE_LABEL]
        assert "auto" in MAIN_WINDOW["lineEdit_eps"] and "eps_kdistance.md" in MAIN_WINDOW["lineEdit_eps"]
        return f"'Auto' and '{cs.ADAPTIVE_LABEL}' -> adaptive; tooltips say exploratory"

    check("the batch carries the record; the dictionary describes its columns", batch_and_dictionary)
    check("the algorithm labels: the new name, and 'Auto' still understood", labels)

    print("\n" + "=" * 100)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
