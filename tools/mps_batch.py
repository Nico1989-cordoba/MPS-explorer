# -*- coding: utf-8 -*-
"""
Batch analysis across axons, and comparison between genotypes.

For objectives 2 and 3 of the thesis, which compare betaII-spectrin
organization between alpha-adducin or 4.1B knockouts and their wild-type
controls. Many axons are measured per animal, so the observations are
NESTED: axon within ROI within animal within genotype.

Why this module refuses to run a test for you
---------------------------------------------
Treating every axon as an independent observation is pseudoreplication,
and here it is not a small effect. Measured on the 18 real test axons,
grouping by ROI:

    parameter          ICC     18 axons behave like
    occupancy          0.75    2.6 independent observations
    cluster area       0.98    2.0
    1NN median         0.52    3.5

An unpaired test over 18 "independent" axons would therefore claim
roughly six times more evidence than the data holds. (With only two
groups that ICC estimate is itself unstable -- see
``intraclass_correlation`` -- but the direction is unambiguous.)

Which remedy to apply is a statistical design decision that belongs to
the experimenter and their supervisor, not to analysis software: a linear
mixed model with animal as a random effect, or aggregating to one value
per animal before comparing, are both defensible and give different
answers. This module therefore reports the data at BOTH levels, reports
the ICC and the design effect so the cost of ignoring the nesting is
visible, and stops there.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import functools
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_analysis import AxonAnalysis

# Parameters worth comparing between groups, and how to pull them off an
# AxonAnalysis. Kept explicit so the batch table has a stable schema.
COMPARABLE_PARAMETERS: Dict[str, str] = {
    "perimeter_um": "perimeter_um",
    "n_clusters_kept": "n_clusters_kept",
    "clusters_per_um": "clusters_per_um",
    "median_area_nm2": "median_area_nm2",
    "median_r_eff_nm": "median_r_eff_nm",
    "median_1nn_nm": "median_1nn_nm",
    "occupancy_percent": "occupancy_percent",
    "mean_delta_z_nm": "mean_delta_z_nm",
    # Not a property of the axon but of its reconstruction: a group whose
    # contours are more inflated than the other's would show a difference
    # in every perimeter-derived parameter above without any biology.
    "contour_tour_over_hull": "contour_tour_over_hull",
}


# ============================================================================
# Metadata
# ============================================================================

@dataclass
class AxonMetadata:
    """Where one axon came from. Everything the nesting depends on."""

    source_path: str
    animal_id: Optional[str] = None
    genotype: Optional[str] = None
    roi: Optional[str] = None
    axon_id: Optional[str] = None
    notes: str = ""

    @property
    def is_complete(self) -> bool:
        """True when the fields the group comparison needs are present."""
        return self.animal_id is not None and self.genotype is not None


def parse_metadata(
    path: str,
    animal_pattern: Optional[str] = None,
    genotype_pattern: Optional[str] = None,
    roi_pattern: str = r"(ROI\s*\d+)",
    axon_pattern: str = r"[Aa]xon\s*_?(\d+)",
) -> AxonMetadata:
    """
    Pull metadata out of a file path with regular expressions.

    Nothing is guessed. ``animal_id`` and ``genotype`` stay None unless a
    pattern is supplied and matches, because inventing an animal identity
    would silently determine the statistics downstream. The batch summary
    reports how many files lack them.

    Parameters
    ----------
    animal_pattern, genotype_pattern : regexes with one capturing group,
        applied to the full path. For example, if animals are folders like
        ``.../M12_WT/...`` then ``animal_pattern=r"(M\\d+)_"`` and
        ``genotype_pattern=r"M\\d+_(WT|KO)"``.
    """
    def grab(pat: Optional[str]) -> Optional[str]:
        if not pat:
            return None
        m = re.search(pat, path)
        return m.group(1) if m else None

    return AxonMetadata(
        source_path=path,
        animal_id=grab(animal_pattern),
        genotype=grab(genotype_pattern),
        roi=grab(roi_pattern),
        axon_id=grab(axon_pattern),
    )


@dataclass
class AxonRecord:
    """One analysed axon plus where it came from."""

    metadata: AxonMetadata
    analysis: Optional[AxonAnalysis]
    error: Optional[str] = None

    def value(self, parameter: str) -> Optional[float]:
        if self.analysis is None:
            return None
        v = getattr(self.analysis, COMPARABLE_PARAMETERS.get(parameter, parameter), None)
        return None if v is None else float(v)

    def export_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "source_path": self.metadata.source_path,
            "animal_id": self.metadata.animal_id,
            "genotype": self.metadata.genotype,
            "roi": self.metadata.roi,
            "axon_id": self.metadata.axon_id,
            "error": self.error or "",
        }
        if self.analysis is not None:
            row.update(self.analysis.export_dict())
        return row


# ============================================================================
# Nesting diagnostics
# ============================================================================

@dataclass
class NestingDiagnostic:
    """How much the grouping structure matters for one parameter."""

    parameter: str
    icc: Optional[float]
    design_effect: Optional[float]
    effective_n: Optional[float]
    n_observations: int
    n_groups: int
    mean_group_size: float
    null_p95: Optional[float] = None      # ICC this design reaches by chance
    warnings: List[str] = field(default_factory=list)

    @property
    def exceeds_chance(self) -> Optional[bool]:
        """True when the ICC is above what this design produces by chance."""
        if self.icc is None or self.null_p95 is None:
            return None
        return self.icc > self.null_p95

    @property
    def severity(self) -> str:
        if self.icc is None:
            return "undetermined"
        # A raw ICC is not interpretable on its own: the estimator is
        # clamped at zero and so biased upward, and with few groups a true
        # ICC of zero routinely estimates well above it (simulated: with
        # two groups of nine, the 95th percentile under the null is 0.31).
        if self.null_p95 is not None and self.icc <= self.null_p95:
            return "within chance for this design"
        if self.icc > 0.5:
            return "severe"
        if self.icc > 0.2:
            return "substantial"
        return "mild"


@functools.lru_cache(maxsize=256)
def icc_null_threshold(
    n_groups: int,
    mean_group_size: float,
    percentile: float = 95.0,
    n_sim: int = 400,
    random_seed: int = 0,
) -> Optional[float]:
    """
    ICC this design reaches by chance alone, at the given percentile.

    Simulates groups drawn from one identical distribution -- true ICC
    zero -- and returns the requested percentile of the estimates. An
    observed ICC below this is not evidence of nesting.

    Needed because the estimator's null distribution depends strongly on
    the design: simulated at 400 draws, the 95th percentile is 0.31 for
    two groups of nine, 0.13 for six groups of nine, and 0.02 for six
    groups of fifty. Studies with three to five animals per genotype sit
    squarely in the noisy regime.

    Cached on the design, since the simulation is the expensive part of
    a diagnostic that is otherwise instant and gets called per parameter.
    """
    k = int(n_groups)
    m = max(int(round(mean_group_size)), 2)
    if k < 2:
        return None
    rng = np.random.default_rng(random_seed)
    out = np.empty(n_sim)
    for i in range(n_sim):
        groups = [rng.normal(0.0, 1.0, m) for _ in range(k)]
        d = _icc_core(groups)
        out[i] = np.nan if d is None else d
    out = out[np.isfinite(out)]
    return float(np.percentile(out, percentile)) if out.size else None


def _icc_core(groups: Sequence[NDArray[np.float64]]) -> Optional[float]:
    """Bare ICC computation, shared by the estimator and its null."""
    clean = [np.asarray(g, dtype=float) for g in groups]
    clean = [g[np.isfinite(g)] for g in clean]
    clean = [g for g in clean if g.size > 0]
    k = len(clean)
    n_obs = int(sum(g.size for g in clean))
    if k < 2 or n_obs <= k:
        return None
    sizes = np.array([g.size for g in clean], dtype=float)
    all_v = np.concatenate(clean)
    means = np.array([g.mean() for g in clean])
    ms_b = float(np.sum(sizes * (means - all_v.mean()) ** 2) / (k - 1))
    ms_w = float(sum(((g - m) ** 2).sum() for g, m in zip(clean, means))
                 / (n_obs - k))
    n0 = float((sizes.sum() - (sizes ** 2).sum() / sizes.sum()) / (k - 1))
    if n0 <= 0 or ms_w <= 0:
        return None
    var_b = (ms_b - ms_w) / n0
    return max(0.0, var_b / (var_b + ms_w))


def intraclass_correlation(
    groups: Sequence[Sequence[float]], parameter: str = "",
    observation_label: str = "axon", group_label: str = "group",
) -> NestingDiagnostic:
    """
    One-way random-effects intraclass correlation: the share of total
    variance sitting BETWEEN groups rather than within them.

    ICC near 0 means axons within a group are no more alike than axons
    from different groups, so treating them as independent is roughly
    fair. ICC near 1 means they are near-duplicates and the effective
    sample size is far below the number of axons.

    The design effect ``1 + (m - 1) * ICC`` converts that into the number
    of independent observations the data is actually worth.

    Note that with very few groups the estimate is unstable -- with two
    groups it is really just restating that those two groups differ -- so
    a warning is attached below five.

    Parameters
    ----------
    observation_label, group_label : nouns for the warning text. The same
        arithmetic describes axons nested in animals and segments nested in
        axons, but a message naming the wrong level is worse than none.
    """
    clean = [np.asarray([v for v in g if v is not None and np.isfinite(v)],
                        dtype=float) for g in groups]
    clean = [g for g in clean if g.size > 0]
    warnings_: List[str] = []

    n_obs = int(sum(g.size for g in clean))
    k = len(clean)
    if k < 2 or n_obs <= k:
        # Keyword arguments on purpose: passed positionally, the warning
        # list lands in null_p95 (the 8th field) and the warning itself is
        # lost, so the caller is told nothing about why there is no ICC.
        return NestingDiagnostic(
            parameter=parameter, icc=None, design_effect=None,
            effective_n=None, n_observations=n_obs, n_groups=k,
            mean_group_size=0.0,
            warnings=[f"Not enough {group_label}s or observations for an "
                      f"ICC."])

    sizes = np.array([g.size for g in clean], dtype=float)
    m = float(sizes.mean())
    icc_val = _icc_core(clean)
    if icc_val is None:
        return NestingDiagnostic(
            parameter=parameter, icc=None, design_effect=None,
            effective_n=None, n_observations=n_obs, n_groups=k,
            mean_group_size=m, null_p95=None,
            warnings=["Degenerate variance; ICC undefined."])

    icc = float(icc_val)
    deff = 1.0 + (m - 1.0) * icc
    eff_n = n_obs / deff if deff > 0 else None
    null_p95 = icc_null_threshold(k, round(m))

    if k < 5:
        warnings_.append(
            f"Only {k} {group_label}s: this ICC is unstable and, with two, "
            f"amounts to saying the {group_label}s differ. Treat it as "
            f"indicative."
        )
    if null_p95 is not None and icc <= null_p95:
        warnings_.append(
            f"ICC = {icc:.2f} is within what this design "
            f"({k} {group_label}s of about {m:.0f}) reaches by chance "
            f"(null 95th percentile {null_p95:.2f}). It is not evidence "
            f"of nesting."
        )
    elif icc > 0.5:
        warnings_.append(
            f"ICC = {icc:.2f}: {observation_label}s of the same "
            f"{group_label} behave close to replicates. Treating the {n_obs} "
            f"{observation_label}s as independent claims about {deff:.1f}x "
            f"more evidence than the data holds."
        )

    return NestingDiagnostic(
        parameter=parameter, icc=icc, design_effect=float(deff),
        effective_n=None if eff_n is None else float(eff_n),
        n_observations=n_obs, n_groups=k, mean_group_size=m,
        null_p95=null_p95, warnings=warnings_,
    )


# ============================================================================
# Group comparison, reported at both levels
# ============================================================================

@dataclass
class LevelSummary:
    """Descriptive statistics for one group at one level of aggregation."""

    group: str
    n: int
    mean: Optional[float]
    median: Optional[float]
    sd: Optional[float]
    values: NDArray[np.float64]


@dataclass
class GroupComparison:
    """One parameter compared between genotypes, at both nesting levels."""

    parameter: str
    per_axon: List[LevelSummary]
    per_animal: List[LevelSummary]
    nesting: Optional[NestingDiagnostic]
    warnings: List[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"{self.parameter}"]
        for label, level in (("per axon", self.per_axon),
                             ("per animal", self.per_animal)):
            if not level:
                lines.append(f"  {label:<11} (no data)")
                continue
            parts = [f"{s.group}: n={s.n}"
                     f" mean={s.mean:.3g}" if s.mean is not None else f"{s.group}: n={s.n}"
                     for s in level]
            lines.append(f"  {label:<11} " + " | ".join(parts))
        if self.nesting and self.nesting.icc is not None:
            n = self.nesting
            lines.append(f"  nesting     ICC={n.icc:.2f} ({n.severity}), "
                         f"design effect {n.design_effect:.1f}, "
                         f"effective n {n.effective_n:.1f} of {n.n_observations}")
        return "\n".join(lines)


def _summarise(group: str, values: List[float]) -> LevelSummary:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    return LevelSummary(
        group=group, n=int(v.size),
        mean=float(v.mean()) if v.size else None,
        median=float(np.median(v)) if v.size else None,
        sd=float(v.std(ddof=1)) if v.size > 1 else None,
        values=v,
    )


def compare_groups(
    records: Sequence[AxonRecord],
    parameter: str,
    group_by: str = "genotype",
    nest_by: str = "animal_id",
) -> GroupComparison:
    """
    Summarise one parameter between groups, at the axon level and again
    after collapsing to one value per nesting unit.

    Deliberately returns no p-value. The two levels routinely disagree,
    and choosing between them (or fitting a mixed model instead) is a
    design decision for the experimenter -- see the module docstring.

    Parameters
    ----------
    group_by : metadata field defining the comparison, normally "genotype".
    nest_by : metadata field defining the nesting unit, normally
        "animal_id".
    """
    warnings_: List[str] = []
    usable = [r for r in records if r.analysis is not None]
    if not usable:
        return GroupComparison(parameter, [], [], None,
                               ["No successfully analysed axons."])

    missing_group = sum(1 for r in usable
                        if getattr(r.metadata, group_by, None) is None)
    missing_nest = sum(1 for r in usable
                       if getattr(r.metadata, nest_by, None) is None)
    if missing_group:
        warnings_.append(
            f"{missing_group}/{len(usable)} axons have no '{group_by}'; they "
            f"are excluded from the comparison. Supply a pattern to "
            f"parse_metadata so this is not silently dropped.")
    if missing_nest:
        warnings_.append(
            f"{missing_nest}/{len(usable)} axons have no '{nest_by}', so the "
            f"per-{nest_by} level cannot be built for them. Without it the "
            f"nesting cannot be accounted for at all.")

    # --- per axon ---
    by_group: Dict[str, List[float]] = {}
    for r in usable:
        g = getattr(r.metadata, group_by, None)
        v = r.value(parameter)
        if g is None or v is None:
            continue
        by_group.setdefault(str(g), []).append(v)
    per_axon = [_summarise(g, vs) for g, vs in sorted(by_group.items())]

    # --- collapsed to one value per nesting unit ---
    per_unit: Dict[Tuple[str, str], List[float]] = {}
    for r in usable:
        g = getattr(r.metadata, group_by, None)
        u = getattr(r.metadata, nest_by, None)
        v = r.value(parameter)
        if g is None or u is None or v is None:
            continue
        per_unit.setdefault((str(g), str(u)), []).append(v)

    collapsed: Dict[str, List[float]] = {}
    for (g, _u), vs in per_unit.items():
        collapsed.setdefault(g, []).append(float(np.mean(vs)))
    per_animal = [_summarise(g, vs) for g, vs in sorted(collapsed.items())]

    if per_animal:
        n_units = sum(s.n for s in per_animal)
        n_axons = sum(s.n for s in per_axon)
        if n_units < n_axons:
            warnings_.append(
                f"{n_axons} axons collapse to {n_units} {nest_by} values. "
                f"Any test run on the {n_axons} axons treats them as "
                f"independent, which they are not.")

    # --- nesting diagnostic, across all nesting units ---
    groups_for_icc = [vs for vs in per_unit.values() if len(vs) > 0]
    nesting = (intraclass_correlation(groups_for_icc, parameter)
               if len(groups_for_icc) >= 2 else None)
    if nesting:
        warnings_.extend(nesting.warnings)

    return GroupComparison(parameter, per_axon, per_animal, nesting, warnings_)


# ============================================================================
# Export
# ============================================================================

def export_batch_csv(records: Sequence[AxonRecord], path: str) -> int:
    """
    Write one row per axon, metadata first.

    Returns the number of rows written. Failed axons are written too,
    with their error, so a batch never silently loses a file.
    """
    rows = [r.export_row() for r in records]
    if not rows:
        return 0
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    return len(rows)


# ============================================================================
# Multi-segment (ring) batches
# ============================================================================

@dataclass
class RingRecord:
    """One axon's every-segment analysis plus where it came from.

    Kept separate from ``AxonRecord`` because the observations have
    different shapes: an axon contributes one row there, and here it
    contributes one row per segment plus one per segment pair. Writing
    both into one table is what makes a CSV unloadable later.
    """

    metadata: AxonMetadata
    ms: Optional[Any] = None            # MultiSegmentAnalysis
    rings: Optional[Any] = None         # RingAnalysis
    error: Optional[str] = None

    def _meta_row(self) -> Dict[str, Any]:
        return {
            "source_path": self.metadata.source_path,
            "animal_id": self.metadata.animal_id,
            "genotype": self.metadata.genotype,
            "roi": self.metadata.roi,
            "axon_id": self.metadata.axon_id,
        }

    def segment_rows(self) -> List[Dict[str, Any]]:
        """One row per analysed segment: metadata, the per-segment
        parameters, and that segment's gaps and patches."""
        if self.ms is None:
            row = self._meta_row()
            row["error"] = self.error or "not analysed"
            return [row]

        rows = self.ms.export_rows()
        runs = [] if self.rings is None else self.rings.runs
        # export_rows() skips segments whose analysis failed, so it cannot
        # be indexed against runs (which keeps a None per segment). Pair
        # them through segment_index instead.
        run_by_index = {
            seg.index: run
            for seg, run in zip(self.ms.segments, runs) if run is not None
        }
        out: List[Dict[str, Any]] = []
        for row in rows:
            merged = self._meta_row()
            merged.update(row)
            run = run_by_index.get(row.get("segment_index"))
            if run is not None:
                merged.update(run.export_dict())
            merged["error"] = ""
            out.append(merged)
        return out

    def pair_rows(self) -> List[Dict[str, Any]]:
        """One row per consecutive segment pair: metadata, the axial
        boundary between them, and their gap/patch correlation."""
        if self.rings is None:
            return []
        out: List[Dict[str, Any]] = []
        for pair in self.rings.pairs:
            row = self._meta_row()
            row.update(pair.export_dict())
            out.append(row)
        return out


def _write_rows(rows: Sequence[Dict[str, Any]], path: str) -> int:
    if not rows:
        return 0
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    return len(rows)


def export_ring_csvs(
    records: Sequence[RingRecord], segment_path: str, pair_path: str
) -> Tuple[int, int]:
    """
    Write the per-segment and per-pair tables.

    Returns (segment rows, pair rows). Axons that failed appear in the
    segment table with their error, so a batch never silently loses a file.
    """
    seg_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    for r in records:
        seg_rows.extend(r.segment_rows())
        pair_rows.extend(r.pair_rows())
    return (_write_rows(seg_rows, segment_path),
            _write_rows(pair_rows, pair_path))


def segment_nesting(
    records: Sequence[RingRecord], parameter: str = "occupancy_percent"
) -> NestingDiagnostic:
    """
    How much the segments of one axon resemble each other, for one
    per-segment parameter.

    A level of nesting the per-axon path never had to consider: several
    segments now come from the SAME axon, imaged in the same acquisition,
    and they are the observations. If they behave like replicates, a
    comparison over segments claims more evidence than the data holds --
    the same trap ``compare_groups`` guards against for axons within an
    animal, one level down. Groups here are axons.

    Parameters
    ----------
    parameter : a column of the per-segment export (for example
        "occupancy_percent", "median_gap_nm", "n_patches").
    """
    groups: List[List[float]] = []
    for r in records:
        values: List[float] = []
        for row in r.segment_rows():
            v = row.get(parameter)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                values.append(fv)
        if len(values) >= 2:
            groups.append(values)

    diag = intraclass_correlation(
        groups, parameter=parameter,
        observation_label="segment", group_label="axon")
    if not groups:
        diag.warnings.append(
            "No axon contributed two or more segments, so there is nothing "
            "to nest: every segment is its own axon here.")
    return diag


def ring_batch_summary(records: Sequence[RingRecord]) -> str:
    """Short account of what a ring batch produced."""
    total = len(records)
    ok = [r for r in records if r.ms is not None]
    n_seg = sum(len(r.segment_rows()) for r in ok)
    n_pair = sum(len(r.pair_rows()) for r in ok)

    resolved = unresolved = 0
    for r in ok:
        for row in r.pair_rows():
            if row.get("boundary_is_true_valley") is True:
                resolved += 1
            elif row.get("boundary_is_true_valley") is False:
                unresolved += 1

    # Provenance lives on each segment's AxonAnalysis, not on the
    # multi-segment wrapper.
    guessed = [
        r.metadata.source_path for r in ok
        if any(a is not None and a.pixel_size_source == "override"
               for a in r.ms.analyses)
    ]

    lines = [
        f"axons analysed    : {len(ok)}/{total}"
        + (f"  ({total - len(ok)} failed)" if total - len(ok) else ""),
        f"segments          : {n_seg}",
        f"consecutive pairs : {n_pair}",
        f"axially resolved  : {resolved} of {resolved + unresolved} pair(s) "
        f"have a real density valley between them",
    ]
    if unresolved:
        lines.append(
            f"  ! {unresolved} pair(s) sit either side of a boundary with no "
            f"density minimum: those two segments are two halves of one axial "
            f"distribution, not two rings. Read their correlation with that "
            f"in mind.")
    if guessed:
        lines.append(
            f"  ! {len(guessed)} file(s) used a pixel size supplied by hand "
            f"rather than from a Picasso YAML sidecar; every lateral distance "
            f"scales with it.")
    return "\n".join(lines)


def batch_summary(records: Sequence[AxonRecord]) -> str:
    """Short human-readable account of what a batch produced."""
    total = len(records)
    ok = sum(1 for r in records if r.analysis is not None)
    failed = total - ok
    animals = {r.metadata.animal_id for r in records
               if r.metadata.animal_id is not None}
    genos = {r.metadata.genotype for r in records
             if r.metadata.genotype is not None}
    rois = {r.metadata.roi for r in records if r.metadata.roi is not None}

    lines = [
        f"axons analysed : {ok}/{total}" + (f"  ({failed} failed)" if failed else ""),
        f"animals        : {len(animals) if animals else 'UNKNOWN'}",
        f"genotypes      : {sorted(genos) if genos else 'UNKNOWN'}",
        f"ROIs           : {len(rois) if rois else 'unknown'}",
    ]
    if not animals:
        lines.append(
            "  ! No animal_id on any file. Without it the nesting cannot be "
            "accounted for and every comparison risks pseudoreplication.")
    return "\n".join(lines)
