# -*- coding: utf-8 -*-
"""
The axial profile of an axon's CLUSTERS (H6 clusters, exploratory): one
z' per cluster -- the median z' of its localizations -- instead of one per
localization, next to (never instead of) rule v2 (D-41), which reads the
localization profile.

Two cluster sets, because they answer different things (OPT_MAP_C traps
(i) and (ii), 2026-10-02):

* ``"groups_3d"`` -- slab-free 3D clusters: DBSCAN on the LAB coordinates
  (x, y, 0.5 z) with eps 25 nm and min_samples 10, the same clusters that
  set the cluster-robust SiZer variance of rule v2
  (``tools.mps_sizer.groups_3d``). They know nothing about the rings, so
  their medians can fall anywhere along z'.
* ``"ring_slab"`` -- the ring clusters the column test uses
  (``tools.mps_columns.build_rings``): DBSCAN INSIDE each ring's axial
  slab. A ring cluster's median can never cross its slab's cut, so a dip
  of their histogram at a cut is built in by the segmentation itself
  (moving the cut moves the dip).

For each set: the medians, a histogram (``HIST_BIN_CLUSTERS_NM``) and a
SiZer reading of the medians, iid with one point per cluster (each median
is one independent unit), at h = ``CLUSTER_SIZER_H_OVER_P`` * P.

The cluster sets above are descriptive. Rule v2-clusters
(``viability_v2c``, EXPLORATORY, D-42; at the end of the module) reads
the four criteria of rule v2 on these clusters; it is shown next to rule
v2, never instead of it; it never selects pairs under the pre-specified
rule, only when the user chooses it in the criteria switches, always as
an EXPLORATORY selection (Q-38 answered 2026-10-02, D-43:
``tools.mps_selection``). No column
statistic anywhere (no matching, no z_A, no p) -- the geometry of the z
profile, allowed on real axons (R8).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_axial_precision import P_REF_DEFAULT_NM

__all__ = [
    "CLUSTER_SET_LABELS",
    "CLUSTER_SET_ORDER",
    "CLUSTER_SIZER_H_OVER_P",
    "HIST_BIN_CLUSTERS_NM",
    "ClusterMedians",
    "ClusterZProfile",
    "cluster_median_rows",
    "cluster_z_profile",
    "groups_3d_labels",
    "ring_slab_medians",
    "groups_3d_medians",
    # rule v2-clusters (EXPLORATORY, D-42): selects pairs only as an exploratory selection (D-43)
    "V2C_ALPHA",
    "V2C_CUT_FLAG_NM",
    "V2C_H_OVER_P",
    "V2C_H_SENSITIVITY",
    "V2C_H_TABLE",
    "V2C_K_MIN",
    "V2C_REASON_AT_CUT",
    "V2C_RULE",
    "V2C_VARIANT_LABELS",
    "V2C_VARIANT_LETTERS",
    "V2C_VARIANT_ORDER",
    "AxonViabilityV2C",
    "V2CLeak",
    "V2CPair",
    "V2CRing",
    "V2CSizer",
    "V2CVariant",
    "exp_spur_frac_clusters",
    "tn_centre_fixed_sigma",
    "v2c_pair_rows",
    "v2c_sizer",
    "v2c_summary_lines",
    "valley_at_cut",
    "verdict_v2c",
    "viability_v2c",
]

# The SiZer bandwidth of the cluster medians as a fraction of P: the same 0.10 P as rule v2's localization profile so
# the two strips are comparable; a later stage may change it (the medians are ~10x fewer points than the locs).
CLUSTER_SIZER_H_OVER_P = 0.10
# a few hundred medians per axon against tens of thousands of localizations: 20 nm bars instead of the
# localizations' 10 nm.
HIST_BIN_CLUSTERS_NM = 20.0
CLUSTER_SET_ORDER: Tuple[str, ...] = ("groups_3d", "ring_slab")
CLUSTER_SET_LABELS: Dict[str, str] = {
    "groups_3d": "slab-free 3D clusters",
    "ring_slab": "ring clusters (as used by the column test)",
}


@dataclass
class ClusterMedians:
    """One cluster set of one axon: per cluster its median z' (nm), its localizations, the 25th / 75th percentiles
    of its z', and the ring whose slab holds the median (-1: outside every ring slab); the histogram of the medians
    and their SiZer reading (``tools.mps_sizer.SizerProfile``, iid; None with fewer than 2 clusters)."""

    source: str
    label: str
    median_z_nm: NDArray[np.float64]
    n_locs: NDArray[np.int64]
    q25_z_nm: NDArray[np.float64]
    q75_z_nm: NDArray[np.float64]
    ring_index: NDArray[np.int64]
    hist_edges: NDArray[np.float64]
    hist_counts: NDArray[np.float64]
    profile: Any
    params: str

    @property
    def n_clusters(self) -> int:
        return int(self.median_z_nm.size)

    def count_in_ring(self, ring: int) -> int:
        return int(np.count_nonzero(self.ring_index == int(ring)))


@dataclass
class ClusterZProfile:
    """Both cluster sets of one axon (``CLUSTER_SET_ORDER``), the ring slabs (index, z_lo, z_hi in z', nm), P and
    the bandwidth used, and the time it took."""

    sets: Dict[str, ClusterMedians]
    slabs: List[Tuple[int, float, float]]
    p_ref_nm: float
    h_over_p: float
    seconds: float
    warnings: List[str] = field(default_factory=list)


def _hist(values: NDArray[np.float64], bin_nm: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    fin = values[np.isfinite(values)]
    if not fin.size:
        return np.array([0.0, bin_nm]), np.array([0.0])
    lo = math.floor(float(fin.min()) / bin_nm) * bin_nm
    hi = math.ceil(float(fin.max()) / bin_nm) * bin_nm + bin_nm
    edges = np.arange(lo, hi + 0.5 * bin_nm, bin_nm)
    counts, edges = np.histogram(fin, bins=edges)
    return np.asarray(edges, dtype=np.float64), np.asarray(counts, dtype=np.float64)


def _slab_of(med: NDArray[np.float64], slabs: Sequence[Tuple[int, float, float]]) -> NDArray[np.int64]:
    out = np.full(med.size, -1, dtype=np.int64)
    for idx, lo, hi in slabs:
        m = (med >= lo) & (med < hi) & (out < 0)
        out[m] = int(idx)
    return out


def _profile(med: NDArray[np.float64], p_ref_nm: float, h_over_p: float) -> Any:
    from tools.mps_sizer import sizer_profile
    fin = med[np.isfinite(med)]
    if fin.size < 2:
        return None
    # iid: each cluster median is ONE independent unit (singleton groups would give the same variance)
    return sizer_profile(fin, None, p_ref_nm=p_ref_nm, c=h_over_p, robust=False)


def _make(source: str, med: List[float], n: List[int], q25: List[float], q75: List[float], ring: NDArray[np.int64],
          p_ref_nm: float, h_over_p: float, params: str) -> ClusterMedians:
    m = np.asarray(med, dtype=np.float64)
    edges, counts = _hist(m, HIST_BIN_CLUSTERS_NM)
    return ClusterMedians(
        source=source, label=CLUSTER_SET_LABELS[source], median_z_nm=m, n_locs=np.asarray(n, dtype=np.int64),
        q25_z_nm=np.asarray(q25, dtype=np.float64), q75_z_nm=np.asarray(q75, dtype=np.float64),
        ring_index=np.asarray(ring, dtype=np.int64), hist_edges=edges, hist_counts=counts,
        profile=_profile(m, p_ref_nm, h_over_p), params=params)


def ring_slab_medians(res: Any, *, p_ref_nm: float, h_over_p: float = CLUSTER_SIZER_H_OVER_P) -> ClusterMedians:
    """The ring clusters of ``res`` (``build_rings``: DBSCAN inside each ring's slab), one median z' each."""
    med: List[float] = []
    n: List[int] = []
    q25: List[float] = []
    q75: List[float] = []
    ring: List[int] = []
    for r in res.rings:
        for cl in r.clusters:
            z = np.asarray(cl.z_values_nm, dtype=np.float64)
            z = z[np.isfinite(z)]
            if not z.size:
                continue
            a, b, c = np.percentile(z, [25.0, 50.0, 75.0])
            med.append(float(b))
            q25.append(float(a))
            q75.append(float(c))
            n.append(int(cl.n_locs))
            ring.append(int(r.index))
    prm = res.params
    params = (f"build_rings DBSCAN inside each ring slab: eps {float(prm.eps_nm):g} nm, min_samples "
              f"{int(prm.min_samples)} (the clusters of the column test)")
    return _make("ring_slab", med, n, q25, q75, np.asarray(ring, dtype=np.int64), p_ref_nm, h_over_p, params)


def groups_3d_labels(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]) -> NDArray[np.int64]:
    """The DBSCAN labels of ``tools.mps_sizer.groups_3d`` (same eps, min_samples, z scale) with the noise kept as
    -1 (``groups_3d`` turns every noise point into a group of its own)."""
    from sklearn.cluster import DBSCAN

    from tools.mps_sizer import GROUP_EPS_NM, GROUP_MIN_SAMPLES, GROUP_Z_SCALE
    xyz = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
                           np.asarray(z, dtype=np.float64) * GROUP_Z_SCALE])
    if not xyz.shape[0]:
        return np.zeros(0, dtype=np.int64)
    lab = DBSCAN(eps=GROUP_EPS_NM, min_samples=GROUP_MIN_SAMPLES).fit_predict(xyz)
    return np.asarray(lab, dtype=np.int64)


def groups_3d_medians(res: Any, xyz_lab_nm: Sequence[Any], *, p_ref_nm: float,
                      h_over_p: float = CLUSTER_SIZER_H_OVER_P,
                      slabs: Optional[Sequence[Tuple[int, float, float]]] = None) -> ClusterMedians:
    """The slab-free 3D clusters (DBSCAN on the lab coordinates, ``groups_3d_labels``), one median of the z' of
    their localizations (``res.z_p``, input order) each; noise is left out."""
    from tools.mps_sizer import GROUP_EPS_NM, GROUP_MIN_SAMPLES, GROUP_Z_SCALE
    x, y, z = (np.asarray(v, dtype=np.float64).reshape(-1) for v in xyz_lab_nm)
    zp = np.asarray(res.z_p, dtype=np.float64)
    if x.size != zp.size:
        raise ValueError(f"{x.size} lab coordinates for {zp.size} localizations of the rings")
    lab = groups_3d_labels(x, y, z)
    med: List[float] = []
    n: List[int] = []
    q25: List[float] = []
    q75: List[float] = []
    keep = lab >= 0
    if keep.any():
        order = np.argsort(lab[keep], kind="stable")
        lk = lab[keep][order]
        zk = zp[keep][order]
        cuts = np.flatnonzero(np.diff(lk)) + 1
        for zz in np.split(zk, cuts):
            zz = zz[np.isfinite(zz)]
            if not zz.size:
                continue
            a, b, c = np.percentile(zz, [25.0, 50.0, 75.0])
            med.append(float(b))
            q25.append(float(a))
            q75.append(float(c))
            n.append(int(zz.size))
    ring = _slab_of(np.asarray(med, dtype=np.float64), list(slabs or ()))
    params = (f"DBSCAN on the lab coordinates (x, y, {GROUP_Z_SCALE:g} z): eps {GROUP_EPS_NM:g} nm, min_samples "
              f"{GROUP_MIN_SAMPLES} (the clusters of rule v2's cluster-robust SiZer variance), noise left out")
    return _make("groups_3d", med, n, q25, q75, ring, p_ref_nm, h_over_p, params)


def cluster_z_profile(res: Any, xyz_lab_nm: Sequence[Any], *, p_ref_nm: float,
                      h_over_p: float = CLUSTER_SIZER_H_OVER_P) -> ClusterZProfile:
    """Both cluster sets of one axon (module docstring). ``xyz_lab_nm``: the lab coordinates of every localization
    of ``res``, in its input order. Geometry only: no verdict, no column statistic."""
    t0 = time.perf_counter()
    slabs = [(int(r.index), float(r.z_lo_nm), float(r.z_hi_nm)) for r in res.rings]
    sets = {
        "groups_3d": groups_3d_medians(res, xyz_lab_nm, p_ref_nm=p_ref_nm, h_over_p=h_over_p, slabs=slabs),
        "ring_slab": ring_slab_medians(res, p_ref_nm=p_ref_nm, h_over_p=h_over_p),
    }
    warnings_: List[str] = []
    for s in sets.values():
        if s.n_clusters < 2:
            warnings_.append(f"{s.label}: {s.n_clusters} cluster(s), no SiZer reading of the medians")
    return ClusterZProfile(sets=sets, slabs=slabs, p_ref_nm=float(p_ref_nm), h_over_p=float(h_over_p),
                           seconds=float(time.perf_counter() - t0), warnings=warnings_)


def cluster_median_rows(prof: ClusterZProfile, head: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """One row per cluster of both sets (``CLUSTER_SET_ORDER``): the set, the cluster's number within it, its
    median z' and quartiles, its localizations and the ring slab that holds the median (-1: none)."""
    out: List[Dict[str, Any]] = []
    for key in CLUSTER_SET_ORDER:
        s = prof.sets.get(key)
        if s is None:
            continue
        for i in range(s.n_clusters):
            row: Dict[str, Any] = dict(head or {})
            row.update(cluster_set=key, cluster_set_label=s.label, cluster=i,
                       median_z_nm=float(s.median_z_nm[i]), q25_z_nm=float(s.q25_z_nm[i]),
                       q75_z_nm=float(s.q75_z_nm[i]), n_locs=int(s.n_locs[i]),
                       ring_slab_of_median=int(s.ring_index[i]), hist_bin_nm=HIST_BIN_CLUSTERS_NM,
                       sizer_h_nm=(float(s.profile.h_nm) if s.profile is not None else float("nan")),
                       params=s.params)
            out.append(row)
    return out


# ============================================================================
# Rule v2-clusters (EXPLORATORY; SPEC_v2c, H6 clusters 2026-10-02, D-42)
# ============================================================================
#
# The four criteria of rule v2 (D-41) read on the CLUSTERS -- one z' per cluster -- instead of the localizations, for
# three cluster sets ("variants"):
#   A "ring_slab"     the ring clusters of the column test (DBSCAN inside each ring's slab), point = median z';
#   B "groups_3d"     the slab-free 3D clusters (``groups_3d_medians``: every DBSCAN label >= 0, the ring = the slab
#                     that holds the median, -1 outside every slab), point = median z';
#   C "ring_slab_tn"  the A clusters with the centre corrected for the slab cut (truncated-normal MLE, sigma = the
#                     ring's sigma_z), point = that centre.
# (1c) a significant SiZer peak (iid: each cluster one point, h = 0.10 P) within P/3 of each ring centre; (2c) a
# significant valley between the two centres -- for A and C a valley inside the slab gap widened by 15 nm is built by
# the segmentation (moving the cut moves it) and never counts; (3c) K(ring) / K(central) >= 0.89 for both rings, always
# a fraction of the central ring (Q-35); (4c) the cluster-level expected leak copies "tnfix", computed once per pair
# from the ring-slab clusters and shared by every variant: <= 2 % VIABLE, <= 5 % MARGINAL.
#
# EXPLORATORY and NOT ADOPTED: it is shown next to rule v2, never instead of it. Under the pre-specified rule it never
# selects pairs; since the user's decision of 2026-10-02 (Q-38, D-43) the review, the batch and simnull can read it
# through tools.mps_selection, always labelled as an exploratory selection. Geometry only: no column
# statistic (R8).

V2C_RULE = "rule v2-clusters (exploratory)"
V2C_H_OVER_P = 0.10
V2C_H_SENSITIVITY: Tuple[float, ...] = (0.14, 0.20, 0.25)
# the sensitivity bandwidth shown next to the verdict (a few hundred medians need h ~ 0.20 P to have any power)
V2C_H_TABLE = 0.20
V2C_ALPHA = 0.05
V2C_PEAK_WINDOW_OVER_P = 1.0 / 3.0
# criterion 3c: K(ring) / K(central) >= 0.89 (zmin part B recipe at recall 0.90: 0.887 [0.855; 0.906]; re-derived in
# the review 0.878 [0.859; 0.902]; ~flat in x, so a constant)
V2C_K_MIN = 0.89
V2C_SPUR_VIABLE = 0.02
V2C_SPUR_MARGINAL = 0.05
# a valley strictly between the two centres inside [z_hi(lower) - 15, z_lo(upper) + 15] is "in the slab gap"
V2C_CUT_FLAG_NM = 15.0
V2C_MIN_POINTS = 5
# criterion 4c only: the sigma of F and of the slab's keep fraction is floored at 5 nm, the keep fraction at 5 %
V2C_SIGMA_FLOOR_NM = 5.0
V2C_KEEP_FLOOR = 0.05
V2C_VARIANT_ORDER: Tuple[str, ...] = ("ring_slab", "groups_3d", "ring_slab_tn")
V2C_VARIANT_LETTERS: Dict[str, str] = {"ring_slab": "A", "groups_3d": "B", "ring_slab_tn": "C"}
V2C_VARIANT_LABELS: Dict[str, str] = {
    "ring_slab": "A: ring clusters (median)",
    "groups_3d": "B: slab-free 3D clusters (median)",
    "ring_slab_tn": "C: ring clusters (centre corrected for the slab cut)",
}
# the variants whose clusters are cut by the slabs: a valley in the slab gap is built in and never satisfies (2c)
V2C_CUT_FORCED: Tuple[str, ...] = ("ring_slab", "ring_slab_tn")
V2C_REASON_AT_CUT = "valley lies in the slab gap (construction artefact)"
V2C_LEAK_ESTIMATOR = ("tnfix: per ring cluster, its truncated-normal centre (sigma = the ring's sigma_z) and "
                      "P[Bin(n0, F) >= min_samples] across the cut between the slabs")


@dataclass(frozen=True)
class V2CSizer:
    """SiZer of one variant's points at one bandwidth (iid, one point per cluster): the peaks and valleys and which
    ring centre has a peak within P/3. ``profile`` is the ``tools.mps_sizer.SizerProfile`` (None with fewer than
    ``V2C_MIN_POINTS`` points or no spread)."""

    h_over_p: float
    h_nm: float
    n_points: int
    peaks: Tuple[float, ...]
    valleys: Tuple[float, ...]
    ring_peak: Tuple[bool, ...]
    profile: Any


@dataclass(frozen=True)
class V2CRing:
    """One ring under one variant: its cluster count K and K over the central ring's, the median and the spread (sd,
    ddof 1, NaN with < 3 points) of the variant's points in it, the noise of one median (variant A only: the median
    over its clusters of 1.2533 sd / sqrt(n)), and the h = 0.10 P peak."""

    index: int
    centre_z_nm: float
    k_clusters: int
    k_ratio: float
    median_z_nm: float
    sd_within_nm: float
    median_noise_nm: float
    peak: bool
    peak_z_nm: Optional[float]


@dataclass(frozen=True)
class V2CPair:
    """One consecutive ring pair under one variant at one bandwidth: the four cluster criteria with their numbers,
    rule v2's D-39 leak beside the cluster-level one, the verdict and its plain reasons."""

    variant: str
    ring_a: int
    ring_b: int
    h_over_p: float
    peak_a: bool
    peak_b: bool
    valley_raw: bool
    valleys_between: Tuple[float, ...]
    valley_at_cut: bool
    valley: bool
    k_a: int
    k_b: int
    k_ratio_a: float
    k_ratio_b: float
    k_min: float
    count_ok: bool
    exp_spur_frac_clusters: float
    exp_spur_frac_d39: float
    verdict: str
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class V2CVariant:
    """One variant's points (z' per cluster, nm), the ring each point belongs to (-1: none), K per ring, its rings,
    and its SiZer at every bandwidth (``V2C_H_OVER_P`` and ``V2C_H_SENSITIVITY``)."""

    key: str
    letter: str
    label: str
    points_z_nm: NDArray[np.float64]
    ring_of_point: NDArray[np.int64]
    n_points: int
    n_outside: int
    k_central: int
    rings: Tuple[V2CRing, ...]
    sizer: Dict[float, V2CSizer]
    params: str


@dataclass(frozen=True)
class V2CLeak:
    """Criterion 4c of one pair (shared by every variant): the cut between the slabs, the cluster-level expected
    leak copies (tnfix) and rule v2's D-39 number."""

    ring_a: int
    ring_b: int
    cut_nm: float
    exp_spur_frac_clusters: float
    exp_spur_frac_d39: float


@dataclass(frozen=True)
class AxonViabilityV2C:
    """``viability_v2c``'s answer for one axon (EXPLORATORY: selects pairs only as an exploratory selection, D-43).
    ``pairs``: the verdicts at
    h = ``V2C_H_OVER_P`` P, pair by pair, every variant of ``V2C_VARIANT_ORDER``; ``sensitivity_pairs`` the same at
    h = ``V2C_H_TABLE`` P (information only)."""

    rule: str
    p_ref_nm: float
    h_over_p: float
    alpha: float
    k_min: float
    min_samples: int
    central_index: Optional[int]
    centres_z_nm: Tuple[float, ...]
    slabs: Tuple[Tuple[int, float, float], ...]
    variants: Dict[str, V2CVariant]
    leaks: Tuple[V2CLeak, ...]
    pairs: Tuple[V2CPair, ...]
    sensitivity_pairs: Tuple[V2CPair, ...]
    viable_pairs: Dict[str, Tuple[Tuple[int, int], ...]]
    warnings: Tuple[str, ...]
    seconds: float

    def pairs_of(self, variant: str, sensitivity: bool = False) -> List[V2CPair]:
        src = self.sensitivity_pairs if sensitivity else self.pairs
        return [p for p in src if p.variant == variant]


def _tn_nll(mu: float, sd: float, z: NDArray[np.float64], lo: float, hi: float) -> float:
    from scipy.stats import norm
    keep = float(norm.cdf((hi - mu) / sd) - norm.cdf((lo - mu) / sd))
    keep = max(keep, 1e-12)
    return float(0.5 * np.sum(((z - mu) / sd) ** 2) + z.size * (math.log(sd) + math.log(keep)))


def tn_centre_fixed_sigma(z: Any, sigma_nm: float, lo_nm: float, hi_nm: float) -> float:
    """
    The centre mu of N(mu, sigma) truncated to [lo, hi] (a ring slab), sigma known: the maximum-likelihood estimate
    over mu in [lo - 3 sigma, hi + 3 sigma] (scipy ``minimize_scalar``, bounded). A cluster whose localizations the
    slab cut is placed back towards the cut; an uncut one stays at its mean.
    """
    from scipy.optimize import minimize_scalar
    zz = np.asarray(z, dtype=np.float64)
    sd = float(sigma_nm)
    a, b = float(lo_nm) - 3.0 * sd, float(hi_nm) + 3.0 * sd
    r = minimize_scalar(lambda m: _tn_nll(m, sd, zz, float(lo_nm), float(hi_nm)), bounds=(a, b), method="bounded")
    return float(r.x)


def exp_spur_frac_clusters(n_a: Sequence[int], mu_a: Sequence[float], n_b: Sequence[int], mu_b: Sequence[float], *,
                           sigma_a_nm: float, sigma_b_nm: float, slab_a: Tuple[float, float],
                           slab_b: Tuple[float, float], cut_nm: float, min_samples: int) -> float:
    """
    Criterion 4c ("tnfix"): the expected fraction of the pair's ring clusters that are axial-leak COPIES, from each
    cluster's own centre. Ring a lies below the cut, ring b above. Per cluster c (n_c localizations, centre mu_c):
    sigma = max(the ring's sigma_z, 5 nm); F_c = the share of its localizations past the cut (a: above, b: below);
    keep_c = the share inside its slab (clipped to [0.05, 1]); n0 = round(n_c / keep_c); a copy needs min_samples
    of them across: P[Bin(n0, F_c) >= min_samples]. The sum over both rings over K_a + K_b (NaN without clusters).
    Validated on simulated axons against the true copies: Spearman 0.93, AUC 0.99, never below the truth,
    overestimates x2.9 (D-39: x1.6).
    """
    from scipy.stats import binom, norm
    e = 0.0
    for side, ns, mus, sig, (lo, hi) in ((+1, n_a, mu_a, sigma_a_nm, slab_a), (-1, n_b, mu_b, sigma_b_nm, slab_b)):
        sd = max(float(sig), V2C_SIGMA_FLOOR_NM)
        for n, mu in zip(ns, mus):
            f_to = float(norm.sf((cut_nm - mu) / sd)) if side > 0 else float(norm.cdf((cut_nm - mu) / sd))
            keep = float(norm.cdf((hi - mu) / sd) - norm.cdf((lo - mu) / sd))
            keep = min(max(keep, V2C_KEEP_FLOOR), 1.0)
            e += float(binom.sf(int(min_samples) - 1, int(round(int(n) / keep)), f_to))
    k = len(n_a) + len(n_b)
    return e / k if k else float("nan")


def valley_at_cut(valleys: Sequence[float], lo_centre: float, hi_centre: float, gap_lo_nm: float, gap_hi_nm: float,
                  flag_nm: float = V2C_CUT_FLAG_NM) -> Tuple[bool, Tuple[float, ...]]:
    """(flag, the valleys strictly between the two centres): a valley between them inside the slab gap widened by
    ``flag_nm``, [z_hi(lower ring) - flag, z_lo(upper ring) + flag], is flagged (the slab cut builds that dip)."""
    vals = tuple(float(v) for v in valleys if lo_centre < float(v) < hi_centre)
    flag = any(gap_lo_nm - flag_nm <= v <= gap_hi_nm + flag_nm for v in vals)
    return bool(flag), vals


def v2c_sizer(points: Any, centres: Sequence[float], *, p_ref_nm: float, h_over_p: float) -> V2CSizer:
    """SiZer on one variant's points (iid, alpha ``V2C_ALPHA``) and the peak-near-each-centre booleans; no profile
    with fewer than ``V2C_MIN_POINTS`` points or no spread."""
    from tools.mps_sizer import sizer_profile
    pts = np.asarray(points, dtype=np.float64)
    win = V2C_PEAK_WINDOW_OVER_P * float(p_ref_nm)
    if pts.size < V2C_MIN_POINTS or not np.ptp(pts) > 0:
        return V2CSizer(float(h_over_p), float(h_over_p) * float(p_ref_nm), int(pts.size), (), (),
                        tuple(False for _c in centres), None)
    prof = sizer_profile(pts, None, p_ref_nm=float(p_ref_nm), c=float(h_over_p), alpha=V2C_ALPHA, robust=False)
    peaks = tuple(float(p) for p in prof.peaks)
    return V2CSizer(float(h_over_p), float(prof.h_nm), int(pts.size), peaks, tuple(float(v) for v in prof.valleys),
                    tuple(any(abs(p - float(c)) <= win for p in peaks) for c in centres), prof)


def _pct0(f: float) -> str:
    return f"{100.0 * f:.0f} %" if math.isfinite(f) else "n/a"


def verdict_v2c(*, variant: str, rings: Tuple[int, int], peak_a: bool, peak_b: bool, valley_raw: bool,
                valley_at_cut_: bool, k_ratio_a: float, k_ratio_b: float, k_a: int, k_b: int, k_central: int,
                exp_spur_frac_clusters_: float, k_min: float = V2C_K_MIN,
                p_ref_nm: float = P_REF_DEFAULT_NM) -> Tuple[str, bool, bool, Tuple[str, ...]]:
    """
    The v2-clusters verdict of one pair under one variant: (verdict, valley counted for (2c), count_ok, reasons).
    VIABLE iff both peaks, the valley (for A and C not in the slab gap), both K ratios >= k_min and 4c <= 2 %;
    MARGINAL iff (1c)-(3c) and 2 % < 4c <= 5 %; else NOT VIABLE. Exploratory: never used to select pairs.
    """
    from tools.mps_axial_precision import VERDICT_MARGINAL, VERDICT_NOT_VIABLE, VERDICT_VIABLE
    a, b = rings
    reasons: List[str] = []
    win = V2C_PEAK_WINDOW_OVER_P * float(p_ref_nm)
    for k, pk in ((a, peak_a), (b, peak_b)):
        if not pk:
            reasons.append(f"ring {k}: no significant peak of the cluster points within {win:.0f} nm of its centre")
    forced = variant in V2C_CUT_FORCED
    valley = bool(valley_raw) and not (forced and valley_at_cut_)
    if not valley_raw:
        reasons.append(f"no significant valley of the cluster points between rings {a} and {b}")
    if valley_at_cut_:
        reasons.append(V2C_REASON_AT_CUT + ("" if forced else
                                            " -- flagged only: these 3D clusters are not cut by the slabs"))
    count_ok = all(math.isfinite(r) and r >= k_min for r in (k_ratio_a, k_ratio_b))
    if k_central <= 0:
        reasons.append("central ring without clusters: the cluster count criterion cannot be computed")
    else:
        for k, r, n in ((a, k_ratio_a, k_a), (b, k_ratio_b, k_b)):
            if not (math.isfinite(r) and r >= k_min):
                reasons.append(f"ring {k} has {n} clusters = {_pct0(r)} of the central ring's {k_central}; the "
                               f"minimum is {_pct0(k_min)}")
    crit = bool(peak_a and peak_b and valley and count_ok)
    spur = float(exp_spur_frac_clusters_)
    if not math.isfinite(spur):
        reasons.append("cluster-level leak copies unknown (no ring clusters)")
    elif spur > V2C_SPUR_MARGINAL:
        reasons.append(f"cluster-level leak copies {100 * spur:.1f} % > {100 * V2C_SPUR_MARGINAL:g} %")
    elif spur > V2C_SPUR_VIABLE:
        reasons.append(f"cluster-level leak copies {100 * spur:.1f} % > {100 * V2C_SPUR_VIABLE:g} %"
                       + (" (marginal)" if crit else f" (VIABLE needs <= {100 * V2C_SPUR_VIABLE:g} %)"))
    if crit and math.isfinite(spur) and spur <= V2C_SPUR_VIABLE:
        return VERDICT_VIABLE, valley, count_ok, tuple(reasons)
    if crit and math.isfinite(spur) and spur <= V2C_SPUR_MARGINAL:
        return VERDICT_MARGINAL, valley, count_ok, tuple(reasons)
    return VERDICT_NOT_VIABLE, valley, count_ok, tuple(reasons)


def _lab_of_frame(res: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rot = np.asarray(res.frame.rotation, dtype=np.float64)
    origin = np.asarray(res.frame.origin_nm, dtype=np.float64).reshape(3, 1)
    p = np.vstack([np.asarray(res.x_p, dtype=np.float64), np.asarray(res.y_p, dtype=np.float64),
                   np.asarray(res.z_p, dtype=np.float64)])
    lab = rot.T @ (p - origin) + origin
    return (np.asarray(lab[0], dtype=np.float64), np.asarray(lab[1], dtype=np.float64),
            np.asarray(lab[2], dtype=np.float64))


def viability_v2c(res: Any, v2: Any = None, *, xyz_lab_nm: Optional[Sequence[Any]] = None,
                  clusters: Optional[ClusterZProfile] = None, p_ref_nm: Optional[float] = None) -> AxonViabilityV2C:
    """
    Rule v2-clusters (EXPLORATORY, D-42) of a ``build_rings`` result: the four criteria of rule v2 read on the
    clusters (section comment above), for the variants A, B, C, next to -- never instead of -- ``viability_v2``.

    ``v2``: the axon's ``viability_v2`` (ring centres, central ring, D-39 leak shown beside 4c); None -> computed
    with ``xyz_lab_nm``. ``clusters``: the axon's ``cluster_z_profile`` (variant B is its "groups_3d" set); None ->
    computed from ``xyz_lab_nm`` (the lab coordinates of every localization, input order; rebuilt from the axon
    frame when not given). ``p_ref_nm``: P; None -> ``v2.p_ref_nm``.

    Geometry only (R8): no column statistic. Its verdicts never select pairs under the pre-specified rule; through
    ``tools.mps_selection`` they can, as an exploratory selection (Q-38, D-43).
    """
    t0 = time.perf_counter()
    warnings_: List[str] = []
    if v2 is None:
        from tools.mps_axial_precision import viability_v2, z_quality
        p0 = float(p_ref_nm) if p_ref_nm is not None else P_REF_DEFAULT_NM
        v2 = viability_v2(res, z_quality(res, p_ref_nm=p0), xyz_lab_nm=xyz_lab_nm, p_ref_nm=p0)
    p_ref = float(p_ref_nm) if p_ref_nm is not None else float(v2.p_ref_nm)
    rings = sorted(res.rings, key=lambda r: int(r.index))
    ms = int(res.params.min_samples)
    idx = [int(r.index) for r in rings]
    centres = [float(r.centre_z_nm) for r in v2.rings]
    if [int(r.index) for r in v2.rings] != idx:
        raise ValueError("viability_v2c: the rule-v2 rings do not match the rings of res")
    c_pos = idx.index(int(v2.central_index)) if v2.central_index is not None and int(v2.central_index) in idx else None
    slabs = tuple((int(r.index), float(r.z_lo_nm), float(r.z_hi_nm)) for r in rings)
    if clusters is None:
        xyz = _lab_of_frame(res) if xyz_lab_nm is None else tuple(xyz_lab_nm)
        if xyz_lab_nm is None:
            warnings_.append("3D clusters from the lab coordinates rebuilt from the axon frame")
        b_set = groups_3d_medians(res, xyz, p_ref_nm=p_ref, slabs=list(slabs))
    else:
        b_set = clusters.sets["groups_3d"]

    # ---- the ring clusters: median (A) and truncation-corrected centre (C); n and slab for 4c
    a_pts: List[float] = []
    c_pts: List[float] = []
    a_ring: List[int] = []
    a_noise: Dict[int, List[float]] = {k: [] for k in idx}
    per_ring_n: Dict[int, List[int]] = {k: [] for k in idx}
    per_ring_mu: Dict[int, List[float]] = {k: [] for k in idx}
    for r in rings:
        sig = float(r.sigma_z_nm)
        for cl in r.clusters:
            zv = np.asarray(cl.z_values_nm, dtype=np.float64)
            if not zv.size:
                continue
            mu = tn_centre_fixed_sigma(zv, sig, float(r.z_lo_nm), float(r.z_hi_nm))
            a_pts.append(float(np.median(zv)))
            c_pts.append(mu)
            a_ring.append(int(r.index))
            sd = float(zv.std(ddof=1)) if zv.size > 1 else 0.0
            a_noise[int(r.index)].append(1.2533 * sd / math.sqrt(zv.size))
            per_ring_n[int(r.index)].append(int(zv.size))
            per_ring_mu[int(r.index)].append(mu)

    # ---- criterion 4c, once per pair (shared by every variant)
    d39 = {(int(p.ring_a), int(p.ring_b)): float(p.exp_spur_frac) for p in v2.pairs}
    leaks: List[V2CLeak] = []
    for i in range(len(rings) - 1):
        ra, rb = rings[i], rings[i + 1]
        cut = 0.5 * (float(ra.z_hi_nm) + float(rb.z_lo_nm))
        e = exp_spur_frac_clusters(per_ring_n[int(ra.index)], per_ring_mu[int(ra.index)], per_ring_n[int(rb.index)],
                                   per_ring_mu[int(rb.index)], sigma_a_nm=float(ra.sigma_z_nm),
                                   sigma_b_nm=float(rb.sigma_z_nm), slab_a=(float(ra.z_lo_nm), float(ra.z_hi_nm)),
                                   slab_b=(float(rb.z_lo_nm), float(rb.z_hi_nm)), cut_nm=cut, min_samples=ms)
        leaks.append(V2CLeak(int(ra.index), int(rb.index), cut, float(e),
                             d39.get((int(ra.index), int(rb.index)), float("nan"))))

    # ---- the variants
    hs = (V2C_H_OVER_P,) + tuple(V2C_H_SENSITIVITY)
    raw = {
        "ring_slab": (np.asarray(a_pts, dtype=np.float64), np.asarray(a_ring, dtype=np.int64),
                      "build_rings DBSCAN inside each ring slab; point = median z' of the cluster"),
        "groups_3d": (np.asarray(b_set.median_z_nm, dtype=np.float64), np.asarray(b_set.ring_index, dtype=np.int64),
                      b_set.params + "; point = median z'; ring = the slab that holds the median"),
        "ring_slab_tn": (np.asarray(c_pts, dtype=np.float64), np.asarray(a_ring, dtype=np.int64),
                         "the ring clusters; point = truncated-normal centre given the slab (sigma = ring sigma_z)"),
    }
    variants: Dict[str, V2CVariant] = {}
    for key in V2C_VARIANT_ORDER:
        pts, ring_of, params = raw[key]
        ks = [int(np.count_nonzero(ring_of == k)) for k in idx]
        k_c = ks[c_pos] if c_pos is not None else 0
        sizer = {h: v2c_sizer(pts, centres, p_ref_nm=p_ref, h_over_p=h) for h in hs}
        s0 = sizer[V2C_H_OVER_P]
        win = V2C_PEAK_WINDOW_OVER_P * p_ref
        vr: List[V2CRing] = []
        for j, k in enumerate(idx):
            mine = pts[ring_of == k]
            near = [p for p in s0.peaks if abs(p - centres[j]) <= win]
            noise = (float(np.median(a_noise[k])) if key == "ring_slab" and a_noise[k] else float("nan"))
            vr.append(V2CRing(
                index=k, centre_z_nm=centres[j], k_clusters=ks[j],
                k_ratio=(ks[j] / k_c if k_c else float("nan")),
                median_z_nm=(float(np.median(mine)) if mine.size else float("nan")),
                sd_within_nm=(float(np.std(mine, ddof=1)) if mine.size > 2 else float("nan")),
                median_noise_nm=noise, peak=bool(near),
                peak_z_nm=(min(near, key=lambda p: abs(p - centres[j])) if near else None)))
        variants[key] = V2CVariant(
            key=key, letter=V2C_VARIANT_LETTERS[key], label=V2C_VARIANT_LABELS[key], points_z_nm=pts,
            ring_of_point=ring_of, n_points=int(pts.size), n_outside=int(np.count_nonzero(ring_of < 0)),
            k_central=int(k_c), rings=tuple(vr), sizer=sizer, params=params)

    # ---- the pairs (h 0.10 P: the verdict; h 0.20 P: sensitivity)
    def pairs_at(h: float) -> List[V2CPair]:
        from tools.mps_sizer import valley_between
        out: List[V2CPair] = []
        for i in range(len(rings) - 1):
            lo_c, hi_c = sorted((centres[i], centres[i + 1]))
            gap_lo, gap_hi = float(rings[i].z_hi_nm), float(rings[i + 1].z_lo_nm)
            for key in V2C_VARIANT_ORDER:
                var = variants[key]
                s = var.sizer[h]
                vraw = bool(s.profile is not None and valley_between(s.profile.grid, s.profile.sign, lo_c, hi_c))
                flag, vals = valley_at_cut(s.valleys, lo_c, hi_c, gap_lo, gap_hi)
                ra_, rb_ = var.rings[i], var.rings[i + 1]
                verdict_, valley, count_ok, reasons = verdict_v2c(
                    variant=key, rings=(idx[i], idx[i + 1]), peak_a=s.ring_peak[i], peak_b=s.ring_peak[i + 1],
                    valley_raw=vraw, valley_at_cut_=flag, k_ratio_a=ra_.k_ratio, k_ratio_b=rb_.k_ratio,
                    k_a=ra_.k_clusters, k_b=rb_.k_clusters, k_central=var.k_central,
                    exp_spur_frac_clusters_=leaks[i].exp_spur_frac_clusters, p_ref_nm=p_ref)
                out.append(V2CPair(
                    variant=key, ring_a=idx[i], ring_b=idx[i + 1], h_over_p=float(h), peak_a=bool(s.ring_peak[i]),
                    peak_b=bool(s.ring_peak[i + 1]), valley_raw=vraw, valleys_between=vals, valley_at_cut=flag,
                    valley=valley, k_a=ra_.k_clusters, k_b=rb_.k_clusters, k_ratio_a=ra_.k_ratio,
                    k_ratio_b=rb_.k_ratio, k_min=V2C_K_MIN, count_ok=count_ok,
                    exp_spur_frac_clusters=leaks[i].exp_spur_frac_clusters,
                    exp_spur_frac_d39=leaks[i].exp_spur_frac_d39, verdict=verdict_, reasons=reasons))
        return out

    pairs = pairs_at(V2C_H_OVER_P)
    sens = pairs_at(V2C_H_TABLE)
    if len(rings) < 2:
        warnings_.append("fewer than two rings: no pair")
    viable = {k: tuple((p.ring_a, p.ring_b) for p in pairs if p.variant == k and p.verdict == "viable")
              for k in V2C_VARIANT_ORDER}
    return AxonViabilityV2C(
        rule=V2C_RULE, p_ref_nm=p_ref, h_over_p=V2C_H_OVER_P, alpha=V2C_ALPHA, k_min=V2C_K_MIN, min_samples=ms,
        central_index=(idx[c_pos] if c_pos is not None else None), centres_z_nm=tuple(centres), slabs=slabs,
        variants=variants, leaks=tuple(leaks), pairs=tuple(pairs), sensitivity_pairs=tuple(sens),
        viable_pairs=viable, warnings=tuple(warnings_), seconds=float(time.perf_counter() - t0))


def v2c_pair_rows(v2c: AxonViabilityV2C, head: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """One row per pair x variant x bandwidth (h 0.10 P: the verdict; h 0.20 P: sensitivity only): the columns of
    ``<stem>_zquality_clusters_pairs.csv``."""
    out: List[Dict[str, Any]] = []
    for sens, src in ((False, v2c.pairs), (True, v2c.sensitivity_pairs)):
        for p in src:
            var = v2c.variants[p.variant]
            row: Dict[str, Any] = dict(head or {})
            reasons = list(p.reasons)
            if sens:
                reasons.insert(0, f"sensitivity only (h = {p.h_over_p:g} P); the verdict is the h = "
                                  f"{v2c.h_over_p:g} P row")
            row.update(rule=v2c.rule, variant=var.letter, variant_label=var.label, ring_a=p.ring_a, ring_b=p.ring_b,
                       h_over_p=p.h_over_p, peak_a=p.peak_a, peak_b=p.peak_b, valley=p.valley,
                       valley_at_cut=p.valley_at_cut, k_a=p.k_a, k_b=p.k_b, k_ratio_a=p.k_ratio_a,
                       k_ratio_b=p.k_ratio_b, k_min=p.k_min, count_ok=p.count_ok,
                       exp_spur_frac_clusters=p.exp_spur_frac_clusters, exp_spur_frac_d39=p.exp_spur_frac_d39,
                       n_points=var.n_points, n_outside=var.n_outside, verdict=p.verdict,
                       reasons=" | ".join(reasons))
            out.append(row)
    return out


def v2c_summary_lines(v2c: AxonViabilityV2C) -> List[str]:
    """The v2-clusters block of the view's plain-language summary (never part of rule v2's summary file)."""
    P = v2c.p_ref_nm
    lines = [f"RULE v2-CLUSTERS (EXPLORATORY, D-42): the same four criteria on the CLUSTERS -- NOT the rule, it never "
             "selects pairs under the pre-specified rule; only as an exploratory selection of the criteria switches "
             "(Q-38, D-43)",
             "   Rule v2 above reads the LOCALIZATIONS: (1)-(2) their density, (3) their count, (4) the D-39 leak. Here "
             "each cluster is ONE point (its median z', or for C its centre corrected for the slab cut): (1c)-(2c) "
             f"SiZer on those points (iid, h = {v2c.h_over_p:g} P = {v2c.h_over_p * P:.0f} nm), (3c) K(ring) / "
             f"K(central) >= {100 * v2c.k_min:.0f} %, (4c) cluster-level leak copies (tnfix) <= 2 % / 5 %."]
    by_pair: Dict[Tuple[int, int], List[V2CPair]] = {}
    for p in v2c.pairs:
        by_pair.setdefault((p.ring_a, p.ring_b), []).append(p)
    for key in V2C_VARIANT_ORDER:
        var = v2c.variants[key]
        s = var.sizer[v2c.h_over_p]
        pk = ", ".join(f"{v:.0f}" for v in s.peaks) or "none"
        vl = ", ".join(f"{v:.0f}" for v in s.valleys) or "none"
        ks = "/".join(str(r.k_clusters) for r in var.rings)
        lines.append(f"   {var.label}: {var.n_points} points (K by ring {ks}"
                     + (f", {var.n_outside} outside every slab" if var.n_outside else "")
                     + f"); peaks at {pk} nm, valleys at {vl} nm.")
    for (a, b), ps in by_pair.items():
        leak = ps[0].exp_spur_frac_clusters
        lines.append(f"   rings {a}-{b} (4c leak {100 * leak:.1f} %, D-39 {100 * ps[0].exp_spur_frac_d39:.1f} %):")
        for p in ps:
            var = v2c.variants[p.variant]
            lines.append(f"      {var.letter}: {p.verdict.upper()} (exploratory) -- "
                         + ("; ".join(p.reasons) if p.reasons else "all four criteria met"))
    lines.append("   A valley of the ring clusters (A, C) inside a slab gap is built by the segmentation (each cluster is "
                 "found inside its ring's slab, so no median crosses the cut; moving the cut moves the dip) and "
                 "never counts.")
    lines.append("   Each cluster's centre is still spread by the axial precision, so the axial resolution can "
                 "limit this rule as it limits rule v2 [H6 clusters research, 2026-10-02].")
    if v2c.warnings:
        lines.append("   Warnings: " + " | ".join(v2c.warnings))
    return lines
