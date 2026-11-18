# -*- coding: utf-8 -*-
"""
The batch side of the regularity reading (regularity plan sec. 6.4 and PR 4): pure functions over the per-axon rows
that ``tools.mps_gap_measures.GapRegularityResult.export_columns`` produces (one row per axon x method, plus the
``animal`` and ``group`` the batch table already carries). The integration branch wires them into the batch;
nothing here touches a file or a widget.

Why z-scores and animals
------------------------
Raw gap CVs, rho1 or Bragg peaks are not comparable between axons: each axon's null depends on its own K, contour,
precision and detection. The batch compares z = (observed - null mean) / null sd per measure and per axon, averages
them per ANIMAL (the unit of replication; axons within an animal are not independent: Lazic 2010; Aarts et al. 2014)
and compares groups with the animals as units. With fewer than 3 animals in a group a comparison is descriptive only
[regularity plan sec. 6.4; as tools.mps_constancy].

What it provides
----------------
* ``holm`` -- Holm's step-down adjustment (Holm 1979) across measures (P-R81: Holm by animal in the batch).
* ``aggregate_by_animal`` -- per animal: axons, mean z per measure, the one-sample t-test of the axons' z against 0
  and its Holm adjustment across measures (needs >= 2 axons).
* ``compare_groups`` -- per measure, the animals' mean z compared between two groups by Welch's t (animals as
  units; refused below 3 animals per group), Holm across measures.
* ``lattice_evidence`` -- P-R83 (both criteria): a group shows "evidence of a lattice" only when (a) at least one axon
  is read ``lattice`` (rho1 AND Bragg against the matched-exclusion null in the same axon) AND (b) at least 2
  animals have mean z of rho1 against the matched null < -2 AND mean z of S* > +2.
* ``readings_table`` -- per group, how many axons fell in each reading level and the median a* of the ``lattice`` /
  ``periodic_component`` axons.
* ``calibre_model`` -- P-R89: N ~ P with a random intercept per animal (REML, ``tools.mps_constancy``), the intercept
  with its 95 % interval, the mean spacing P/N per calibre bin, the same fit on the pipeline null's detected K
  (N_det(P)), and the conditional wording of the regularity plan sec. 4.6; refused with fewer than 3 animals.

Origins: z thresholds -2 / +2 and >= 2 animals [user decision P-R83, 2026-10-22]; alpha 0.05 [ad hoc]; < 3 animals
refused [regularity plan sec. 6.4 / P-R89]; intercept interval on n_animals - 2 degrees of freedom [ad hoc:
conservative, as tools.mps_constancy's slope]; calibre bins of 5 um [ad hoc].

References (APA 7)
------------------
Aarts, E., Verhage, M., Veenvliet, J. V., Dolan, C. V., & van der Sluis, S. (2014). A solution to dependency: using
    multilevel analysis to accommodate nested data. Nature Neuroscience, 17(4), 491-496. https://doi.org/10.1038/nn.3648
Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2),
    65-70.
Lazic, S. E. (2010). The problem of pseudoreplication in neuroscientific studies: is it affecting your analysis?
    BMC Neuroscience, 11, 5. https://doi.org/10.1186/1471-2202-11-5

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from tools.mps_constancy import fit_random_intercepts

ALPHA = 0.05                              # [ad hoc]
MIN_ANIMALS = 3                           # [regularity plan sec. 6.4; P-R89]
LATTICE_Z_RHO1 = -2.0                     # [user decision P-R83]
LATTICE_Z_BRAGG = 2.0                     # [user decision P-R83]
LATTICE_MIN_ANIMALS = 2                   # [user decision P-R83]
CALIBRE_BIN_UM = 5.0                      # [ad hoc]

# The z columns the batch compares (GapRegularityResult.export_columns names).
Z_COLUMNS: Tuple[str, ...] = ("gap_cv_z", "gap_excess_cv_z", "gap_rho1_z_matched", "bragg_s_z_matched")


class CalibreRefused(ValueError):
    """The calibre model needs at least 3 animals."""


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def z_score(observed: Optional[float], null_values: Sequence[Optional[float]]) -> Optional[float]:
    """(observed - mean) / sd of the null values (sd with ddof 1); None when not computable."""
    o = _num(observed)
    v = np.array([x for x in (_num(y) for y in null_values) if x is not None], dtype=np.float64)
    if o is None or v.size < 2:
        return None
    sd = float(np.std(v, ddof=1))
    return (o - float(np.mean(v))) / sd if sd > 0 else None


def holm(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm's step-down adjusted p values (Holm 1979), in the input order; None stays None and is not counted."""
    idx = [i for i, p in enumerate(p_values) if _num(p) is not None]
    m = len(idx)
    out: List[Optional[float]] = [None] * len(p_values)
    order = sorted(idx, key=lambda i: float(p_values[i]))
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * float(p_values[i]))   
        running = max(running, adj)
        out[i] = running
    return out


def _groupby(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        k = r.get(key)
        if k is None or str(k) == "":
            continue
        out.setdefault(str(k), []).append(r)
    return out


def aggregate_by_animal(rows: Sequence[Mapping[str, Any]], measures: Sequence[str] = Z_COLUMNS, *,
                        animal_key: str = "animal",
                        tested: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """One row per animal: ``n_axons``, ``mean_<m>``, ``sd_<m>``, ``n_<m>`` (axons with a value), the one-sample t
    test of the axons' z against 0 (``p_<m>``, needs >= 2 axons) and its Holm adjustment across the tested measures
    (``p_holm_<m>``). Rows without an animal are left out.

    ``tested`` (default: every measure) is the family that gets a p and enters the Holm adjustment; the other
    measures keep their descriptive numbers and get ``p_<m> = p_holm_<m> = None``."""
    family = list(measures) if tested is None else [m for m in measures if m in set(tested)]
    out: List[Dict[str, Any]] = []
    for animal, rr in sorted(_groupby(rows, animal_key).items()):
        row: Dict[str, Any] = {animal_key: animal, "n_axons": len(rr)}
        ps: List[Optional[float]] = []
        for m in measures:
            v = np.array([x for x in (_num(r.get(m)) for r in rr) if x is not None], dtype=np.float64)
            row[f"n_{m}"] = int(v.size)
            row[f"mean_{m}"] = float(np.mean(v)) if v.size else None
            row[f"sd_{m}"] = float(np.std(v, ddof=1)) if v.size > 1 else None
            p = None
            if m in family and v.size > 1 and np.std(v) > 0:
                p = float(stats.ttest_1samp(v, 0.0).pvalue)
            row[f"p_{m}"] = p
            if m in family:
                ps.append(p)
        for m in measures:
            row[f"p_holm_{m}"] = None
        for m, p in zip(family, holm(ps)):
            row[f"p_holm_{m}"] = p
        out.append(row)
    return out


def compare_groups(rows: Sequence[Mapping[str, Any]], measures: Sequence[str] = Z_COLUMNS, *,
                   group_key: str = "group", animal_key: str = "animal", alpha: float = ALPHA) -> Dict[str, Any]:
    """Two groups compared with the ANIMAL as the unit: per measure the animals' mean z, Welch's t between the two
    groups, Holm across measures. With fewer than 3 animals in either group: descriptive only (no p)."""
    groups = _groupby(rows, group_key)
    names = sorted(groups)
    result: Dict[str, Any] = {"groups": names, "measures": {}, "descriptive_only": False, "notes": []}
    if len(names) != 2:
        result["descriptive_only"] = True
        result["notes"].append(f"{len(names)} groups: the comparison needs exactly two.")
    per_group = {g: aggregate_by_animal(groups[g], measures, animal_key=animal_key) for g in names}
    n_animals = {g: len(per_group[g]) for g in names}
    result["n_animals"] = n_animals
    if any(n < MIN_ANIMALS for n in n_animals.values()):
        result["descriptive_only"] = True
        result["notes"].append(f"fewer than {MIN_ANIMALS} animals in a group: descriptive only.")
    ps: List[Optional[float]] = []
    for m in measures:
        entry: Dict[str, Any] = {}
        vals = {}
        for g in names:
            v = [a[f"mean_{m}"] for a in per_group[g] if a.get(f"mean_{m}") is not None]
            vals[g] = v
            entry[f"mean_{g}"] = float(np.mean(v)) if v else None
            entry[f"n_animals_{g}"] = len(v)
        p: Optional[float] = None
        if not result["descriptive_only"] and len(names) == 2 and all(len(vals[g]) >= MIN_ANIMALS for g in names):
            p = float(stats.ttest_ind(vals[names[0]], vals[names[1]], equal_var=False).pvalue)
        entry["p"] = p
        ps.append(p)
        result["measures"][m] = entry
    for m, p in zip(measures, holm(ps)):
        result["measures"][m]["p_holm"] = p
        result["measures"][m]["reject"] = None if p is None else bool(p <= alpha)
    return result


@dataclass
class LatticeEvidence:
    """P-R83: both criteria. ``evidence`` only when an axon is read ``lattice`` AND >= 2 animals pass both z tests."""
    n_axons: int
    n_axons_lattice: int
    animals_passing: List[str]
    animal_means: Dict[str, Tuple[Optional[float], Optional[float]]]
    evidence: bool
    sentence: str
    notes: List[str] = field(default_factory=list)


def lattice_evidence(rows: Sequence[Mapping[str, Any]], *, animal_key: str = "animal",
                     reading_key: str = "regularity_reading", z_rho1_key: str = "gap_rho1_z_matched",
                     z_bragg_key: str = "bragg_s_z_matched", z_rho1_max: float = LATTICE_Z_RHO1,
                     z_bragg_min: float = LATTICE_Z_BRAGG, min_animals: int = LATTICE_MIN_ANIMALS
                     ) -> LatticeEvidence:
    """The group-level lattice criterion (P-R83, user decision 2026-10-22): rho1 and Bragg significant against the
    matched-exclusion null in the same axon (reading ``lattice``) AND >= 2 animals whose mean z of rho1 against the
    matched null is < -2 and whose mean z of S* is > +2. Axons without stage 2 (no matched z) do not count towards an
    animal's mean."""
    n_lat = sum(1 for r in rows if r.get(reading_key) == "lattice")
    means: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    passing: List[str] = []
    for animal, rr in sorted(_groupby(rows, animal_key).items()):
        zr = [x for x in (_num(r.get(z_rho1_key)) for r in rr) if x is not None]
        zb = [x for x in (_num(r.get(z_bragg_key)) for r in rr) if x is not None]
        mr = float(np.mean(zr)) if zr else None
        mb = float(np.mean(zb)) if zb else None
        means[animal] = (mr, mb)
        if mr is not None and mb is not None and mr < z_rho1_max and mb > z_bragg_min:
            passing.append(animal)
    ev = n_lat >= 1 and len(passing) >= int(min_animals)
    if ev:
        s = (f"Evidence of a lattice in this group: {n_lat} axon(s) read 'lattice' against the matched-exclusion null, "
             f"and {len(passing)} animals ({', '.join(passing)}) with mean z of rho1 < {z_rho1_max:g} and of the Bragg "
             f"peak > {z_bragg_min:g}.")
    else:
        s = (f"No claim of a lattice for this group ({n_lat} axon(s) read 'lattice'; {len(passing)} animal(s) pass "
             f"both z criteria, {min_animals} needed). Never write 'the clusters form a lattice / a periodic pattern' "
             "without both.")
    return LatticeEvidence(len(rows), n_lat, passing, means, ev, s)


def readings_table(rows: Sequence[Mapping[str, Any]], levels: Sequence[str], *, group_key: str = "group",
                   reading_key: str = "regularity_reading", spacing_key: str = "bragg_spacing_nm"
                   ) -> List[Dict[str, Any]]:
    """Per group (all rows when no group): the number of axons in each reading level and the median a* of the
    ``lattice`` / ``periodic_component`` axons."""
    groups = _groupby(rows, group_key) or {"all": list(rows)}
    out = []
    for g, rr in sorted(groups.items()):
        row: Dict[str, Any] = {group_key: g, "n_axons": len(rr)}
        for lv in levels:
            row[f"n_{lv}"] = sum(1 for r in rr if r.get(reading_key) == lv)
        a = [x for x in (_num(r.get(spacing_key)) for r in rr
                         if r.get(reading_key) in ("lattice", "periodic_component")) if x is not None]
        row["median_spacing_periodic_nm"] = float(np.median(a)) if a else None
        out.append(row)
    return out


# ============================================================================
# Calibre (P-R89)
# ============================================================================

@dataclass
class CalibreFit:
    n_axons: int
    n_animals: int
    slope_per_um: float
    intercept: float
    intercept_ci: Tuple[float, float]
    df: int
    var_animal: float
    var_residual: float


@dataclass
class CalibreResult:
    """N ~ P with a random intercept per animal, for the observed K and (optionally) for the null's N_det(P)."""
    observed: CalibreFit
    null_detected: Optional[CalibreFit]
    spacing_by_bin: List[Dict[str, Any]]
    sentence: str
    warnings: List[str] = field(default_factory=list)


def _fit(n: Sequence[float], p_um: Sequence[float], animals: Sequence[str]) -> CalibreFit:
    beta, cov, s2, var_g = fit_random_intercepts(list(n), list(p_um), [list(animals)])
    n_an = len(set(animals))
    df = max(n_an - 2, 1)
    se = math.sqrt(max(float(cov[0, 0]), 0.0))
    t = float(stats.t.ppf(0.975, df))
    b0 = float(beta[0])
    return CalibreFit(len(n), n_an, float(beta[1]), b0, (b0 - t * se, b0 + t * se), df, float(var_g[0]), float(s2))


def calibre_model(n_clusters: Sequence[float], perimeter_um: Sequence[float], animals: Sequence[str], *,
                  null_k_detected: Optional[Sequence[Optional[float]]] = None,
                  bin_um: float = CALIBRE_BIN_UM) -> CalibreResult:
    """P-R89: N = b0 + b1 P + animal + error (REML; ``tools.mps_constancy.fit_random_intercepts``), the intercept with
    a 95 % interval on n_animals - 2 df, the mean spacing P/N per calibre bin, the same fit on the pipeline null's
    mean detected K per axon (N_det(P)) when given, and the wording of the regularity plan sec. 4.6. Raises
    ``CalibreRefused`` with fewer than 3 animals."""
    n = [float(x) for x in n_clusters]
    p = [float(x) for x in perimeter_um]
    a = [str(x) for x in animals]
    if not (len(n) == len(p) == len(a)):
        raise ValueError("n_clusters, perimeter_um and animals must have the same length")
    n_an = len(set(a))
    if n_an < MIN_ANIMALS:
        raise CalibreRefused(f"{n_an} animal(s): the calibre model needs at least {MIN_ANIMALS} animals (the animal is "
                             "the unit of replication); report the axons descriptively.")
    warnings: List[str] = []
    if n_an < 6:
        warnings.append(f"{n_an} animals: the intercept test has little power below 6 animals [simulation: regularity "
                        "research sim6, seed 6: 0.67 with 3 x 6 axons under the published line].")
    obs = _fit(n, p, a)
    nul: Optional[CalibreFit] = None
    if null_k_detected is not None:
        keep = [i for i, v in enumerate(null_k_detected) if _num(v) is not None]
        if len({a[i] for i in keep}) >= MIN_ANIMALS:
            nul = _fit([float(null_k_detected[i]) for i in keep], [p[i] for i in keep],
                       [a[i] for i in keep])
        else:
            warnings.append("the null's detected K is missing for too many axons: N_det(P) not fitted.")
    bins: List[Dict[str, Any]] = []
    if p:
        edges = np.arange(0.0, max(p) + bin_um, bin_um)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = [i for i in range(len(p)) if lo <= p[i] < hi and n[i] > 0]
            if sel:
                bins.append({"perimeter_lo_um": float(lo), "perimeter_hi_um": float(hi), "n_axons": len(sel),
                             "mean_spacing_nm": float(np.mean([1000.0 * p[i] / n[i] for i in sel]))})
    lo, hi = obs.intercept_ci
    if lo > 0 or hi < 0:
        core = (f"The intercept of N on P is {obs.intercept:+.1f} clusters (95 % CI {lo:+.1f} to {hi:+.1f}; animals as "
                f"units, {obs.n_animals} animals): the cluster number is not proportional to the perimeter. A lattice of "
                "fixed spacing predicts proportionality; it would need a detection loss that depends on calibre")
        if nul is not None:
            core += (f" -- the pipeline null's detected K has intercept {nul.intercept:+.1f} "
                     f"(95 % CI {nul.intercept_ci[0]:+.1f} to {nul.intercept_ci[1]:+.1f}), which is how much of it "
                     "detection alone produces")
        core += "."
    else:
        core = (f"The intercept of N on P is {obs.intercept:+.1f} clusters (95 % CI {lo:+.1f} to {hi:+.1f}; animals as "
                f"units, {obs.n_animals} animals): not distinguishable from 0, so the data are compatible with a "
                "cluster number proportional to the perimeter (and with other arrangements: this is absence of "
                "evidence, not a test of a lattice).")
    return CalibreResult(obs, nul, bins, core, warnings)
