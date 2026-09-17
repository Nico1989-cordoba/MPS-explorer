# -*- coding: utf-8 -*-
"""
Checks against the 2023 sciatic-nerve data (Guada Gazal and Gonzalo
Escalante): two colours split onto one camera, 3D by astigmatism, and the
axons picked one by one in Picasso.

The data is not in the repository (218 GB). Point MPS_DATA_2023 at it, or
leave it in "example data/2023"; without it these checks are skipped.

What is verified:
  1. the files are read as their own metadata describes them -- the pixel
     size from the YAML, z from the astigmatism fit, the pick columns --
     and the two channels of one axon cover the same area, as they must
     when both come from one movie;
  2. this software reproduces, value for value, the 1NN distances the lab
     measured in 2023 with its own script (DBSCAN eps 30 nm, 10 samples,
     x and y only) on the 18 axons it selected;
  3. how much the two loose ends of this data move that result: the pixel
     size, which the YAML gives as 135 nm and the camera metadata as
     133 nm, and the last cluster, which the 2023 script leaves out
     because it iterates over ``range(np.max(labels))``.

Nothing is written: the acquisition folder is only read.

Run:  python validate_dataset_2023.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.cluster_quality import good_cluster_centroids  # noqa: E402
from tools.mps_io import load_localizations, read_pixel_size  # noqa: E402
from tools.mps_registration import pixel_size_disagreement  # noqa: E402
from tools.mps_spatial import compute_nn_distances  # noqa: E402

DATA = os.environ.get("MPS_DATA_2023") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "example data", "2023")
DAY = os.path.join(DATA, "230911 - MPS", "230911-Analysis")
STAINING6 = os.path.join(DATA, "230915 - MPS", "Analisis - Tinción 6", "ROI3")

# The axons the lab kept for its first 1NN analysis, as
# load_selected_distances.py lists them: (ROI, axon).
SELECTED: List[Tuple[int, int]] = [
    (1, 4), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 3), (3, 4), (4, 1),
    (4, 2), (6, 1), (6, 3), (7, 2), (8, 1), (8, 3), (8, 4), (9, 1), (9, 2),
]
# open_hdf5_dbscan.py, the same in every ROI folder.
OLD_PIXEL_NM = 135.0
OLD_EPS_NM = 30.0
OLD_MIN_SAMPLES = 10
CAMERA_PIXEL_NM = 133.0

PASSED = 0
FAILED = 0
_LOADED: Dict[str, object] = {}


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def axon_file(roi: int, axon: int, protein: str = "spectrin") -> str:
    """The picked axon's file; the ROI folders are named two ways."""
    stem = f"ROI{roi}_{protein}_locs_drift_corrected_apicked_{axon}.hdf5"
    for folder in ("Seleccion de axones", "Seleccion axones"):
        path = os.path.join(DAY, f"ROI {roi}", folder, stem)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(stem)


def axon(roi: int, axon_number: int, protein: str = "spectrin"):
    """The picked axon, read once."""
    path = axon_file(roi, axon_number, protein)
    if path not in _LOADED:
        _LOADED[path] = load_localizations(path)
    return _LOADED[path]


def old_distances(roi: int, axon_number: int) -> np.ndarray:
    """The 1NN distances the 2023 script saved, in nm."""
    path = os.path.join(DAY, f"distances_{roi}_{axon_number}.txt")
    return np.loadtxt(path, ndmin=1).ravel()


def one_nn(loc, *, pixel_nm: float = OLD_PIXEL_NM,
           drop_last: bool = True) -> Tuple[np.ndarray, int]:
    """
    The 2023 analysis, run with this software's pieces: DBSCAN on x and y
    over the whole axial range, the centroids of the clusters, and their
    1NN distances in nm (the 2023 script saved them rounded).

    ``drop_last`` leaves the highest cluster label out, which is what
    ``for i in range(np.max(clustering.labels_))`` does.
    """
    scale = pixel_nm / float(loc.pixel_size_nm)
    x, y = loc.x_nm * scale, loc.y_nm * scale
    labels = DBSCAN(eps=OLD_EPS_NM,
                    min_samples=OLD_MIN_SAMPLES).fit(
        np.column_stack([x, y])).labels_
    highest = int(labels.max())
    left_out = {highest} if drop_last and highest >= 0 else set()
    centroids = good_cluster_centroids(x, y, labels, left_out)
    nn = compute_nn_distances(centroids, k=1)
    return nn.distances_nm[:, 0], len({int(v) for v in labels} - {-1})


# ======================================================== 1. the files
def section_reading() -> None:
    print("\n1. READING THE 2023 FILES")

    def one_axon():
        loc = axon(1, 4)
        assert loc.n > 1000, loc.n
        assert loc.pixel_size_nm == OLD_PIXEL_NM, loc.pixel_size_nm
        assert loc.pixel_size_source == "yaml", loc.pixel_size_source
        # 3D by astigmatism: z is fitted, in nm, and d_zcalib comes along.
        assert np.ptp(loc.z_nm) > 200, np.ptp(loc.z_nm)
        assert "d_zcalib" in loc.columns
        # Picked in Picasso with a rectangle, which leaves its own columns.
        for column in ("group", "x_pick_rot", "y_pick_rot"):
            assert column in loc.columns, loc.columns
        return (f"{loc.n:,} localizations, pixel {loc.pixel_size_nm:g} nm "
                f"from the yaml, z over {np.ptp(loc.z_nm):.0f} nm")

    def both_channels_of_one_axon():
        # Both channels come from one movie through the same affine
        # correction, so the same pick covers the same area.
        spectrin, tubulin = axon(1, 4), axon(1, 4, "tubulin")
        boxes = []
        for loc in (spectrin, tubulin):
            boxes.append((loc.x_nm.min(), loc.x_nm.max(),
                          loc.y_nm.min(), loc.y_nm.max()))
        worst = max(abs(a - b) for a, b in zip(*boxes))
        assert worst < OLD_PIXEL_NM, (boxes, worst)
        assert tubulin.pixel_size_nm == spectrin.pixel_size_nm
        return (f"the pick's corners agree within {worst:.0f} nm; "
                f"{spectrin.n:,} spectrin and {tubulin.n:,} tubulin "
                f"localizations")

    def the_channels_of_staining_6_disagree():
        # Localized with different pixel sizes, on one camera: what the
        # main window now warns about when both channels are loaded.
        spectrin = read_pixel_size(
            os.path.join(STAINING6, "ROI3-spectrin_locs_drift_corrected.hdf5"))
        adducin = read_pixel_size(
            os.path.join(STAINING6, "ROI3-adducin_locs_drift_corrected.hdf5"))
        assert spectrin == OLD_PIXEL_NM and adducin == CAMERA_PIXEL_NM, (
            spectrin, adducin)
        note = pixel_size_disagreement("spectrin", spectrin, "adducin",
                                       adducin, 35000.0)
        assert note and "different pixel sizes" in note
        return f"spectrin {spectrin:g} nm, adducin {adducin:g} nm; reported"

    check("a picked axon, as its metadata describes it", one_axon)
    check("the two channels of one axon cover the same area",
          both_channels_of_one_axon)
    check("staining 6: the channels give different pixel sizes",
          the_channels_of_staining_6_disagree)


# ============================================ 2. the 2023 1NN analysis
def section_reproduce() -> None:
    print("\n2. THE 2023 ANALYSIS, REPRODUCED")
    mine: Dict[Tuple[int, int], np.ndarray] = {}
    theirs: Dict[Tuple[int, int], np.ndarray] = {}
    raw: Dict[Tuple[int, int], int] = {}

    def every_selected_axon():
        for roi, number in SELECTED:
            loc = axon(roi, number)
            ours, n_raw = one_nn(loc)
            mine[(roi, number)] = ours
            theirs[(roi, number)] = old_distances(roi, number)
            raw[(roi, number)] = n_raw
        assert len(mine) == 18, len(mine)
        counts = [len(v) for v in mine.values()]
        return (f"{len(mine)} axons, {sum(counts):,} clusters "
                f"({min(counts)}-{max(counts)} per axon)")

    def same_number_of_clusters():
        wrong = {k: (len(mine[k]), len(theirs[k])) for k in mine
                 if len(mine[k]) != len(theirs[k])}
        assert not wrong, wrong
        return f"{sum(len(v) for v in mine.values()):,} distances in all"

    def same_distances() -> str:
        borderline = 0
        wrong: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        for key, ours in mine.items():
            for i in np.nonzero(np.round(ours) != theirs[key])[0]:
                # The centroids are rounded to 0.01 nm, as the old GUI
                # rounded them, so a distance sitting on a half nanometre
                # can round either way. Anything else is a real difference.
                on_a_half = abs(ours[i] % 1.0 - 0.5) < 0.01
                if on_a_half and abs(np.round(ours[i]) - theirs[key][i]) == 1:
                    borderline += 1
                else:
                    wrong.setdefault(key, []).append(
                        (float(ours[i]), float(theirs[key][i])))
        assert not wrong, wrong
        pooled = np.concatenate(list(mine.values()))
        return (f"{pooled.size - borderline:,} of {pooled.size:,} equal to "
                f"the nanometre, {borderline} on a half nanometre where the "
                f"centroid rounding decides; pooled median "
                f"{np.median(pooled):.1f} nm")

    check("the 18 selected axons run", every_selected_axon)
    check("the same clusters as in 2023", same_number_of_clusters)
    check("the same 1NN distances, value for value", same_distances)

    # ------------------------------------- 3. what the loose ends change
    print("\n3. WHAT THE LOOSE ENDS CHANGE")

    def the_last_cluster_the_2023_script_left_out():
        added = 0
        changed = 0
        for roi, number in SELECTED:
            whole, _ = one_nn(axon(roi, number), drop_last=False)
            added += whole.size - mine[(roi, number)].size
            common = min(whole.size, mine[(roi, number)].size)
            changed += int(np.count_nonzero(
                np.round(whole[:common])
                != np.round(mine[(roi, number)][:common])))
        assert added == 18, added
        return (f"one cluster per axon left out, {added} in all; putting "
                f"them back also moves {changed} of the other distances")

    def the_pixel_size():
        pooled_135 = np.concatenate(list(mine.values()))
        pooled_133 = np.concatenate(
            [one_nn(axon(roi, number), pixel_nm=CAMERA_PIXEL_NM)[0]
             for roi, number in SELECTED])
        # Everything scales with the pixel size, but eps stays 30 nm, so
        # the clusters change too: the difference is not a plain 1.5%.
        assert pooled_133.size and pooled_135.size
        plain = np.median(pooled_135) * CAMERA_PIXEL_NM / OLD_PIXEL_NM
        return (f"pooled median {np.median(pooled_135):.1f} nm over "
                f"{pooled_135.size:,} clusters with 135 nm pixels, "
                f"{np.median(pooled_133):.1f} nm over {pooled_133.size:,} "
                f"with 133 nm; a plain rescale would give {plain:.1f} nm, "
                f"so the clustering itself moves as well")

    check("the cluster the 2023 script left out",
          the_last_cluster_the_2023_script_left_out)
    check("135 nm or 133 nm pixels", the_pixel_size)


def main() -> int:
    print("=" * 72)
    print("2023 SCIATIC-NERVE DATA CHECKS")
    print("=" * 72)
    if not os.path.isdir(DAY):
        print(f"\nThe 2023 data is not here: {DATA}")
        print("Set MPS_DATA_2023 to the acquisition folder to run these "
              "checks.")
        print("=" * 72)
        return 0
    section_reading()
    section_reproduce()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
