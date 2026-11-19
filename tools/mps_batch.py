# -*- coding: utf-8 -*-
"""
Many axons at once, and a comparison between groups that does not
pretend the axons are independent.

For objectives 2 and 3 of the thesis, which compare betaII-spectrin
organization between alpha-adducin or 4.1B knockouts and their wild-type
controls. Many axons are measured per animal, so the observations are
NESTED: axon within ROI within slide within animal within genotype.

What a batch writes
-------------------
The same row the axon window exports, built by the same function
(``axon_export.axon_row``), so that a batch's rows and the ones exported
one by one fit in one table. They differ in one thing, and say so: a
batch has no ROI drawn around the axon, so the automatic curation
measures edge-touching against the convex hull of the localizations
(``edge_reference``), which keeps other clusters than an ROI does (a
few fewer on an axon of unpublished pilot data). That axon then has a different
``analysis_id`` in the two, and a table must not hold it twice; the export
leaves out the files the table already holds from the axon window.

Why nothing here runs a test for you
------------------------------------
Treating every axon as an independent observation is pseudoreplication.
Which remedy to apply is a statistical design decision that belongs to
the experimenter and their supervisor, not to analysis software: a linear
mixed model with animal as a random effect, or aggregating to one value
per animal before comparing, are both defensible and give different
answers. This module therefore reports the data at BOTH levels -- per
axon, and one value per nesting unit -- with the intraclass correlation
that says how much the difference between them matters, and stops there.

With axons grouped in only two units (two ROIs of one animal, say), a
high intraclass correlation says that those two units differ, and not how
much evidence an axon is worth: an ICC measured over two groups is a
comparison of those two groups. How much the axons of one animal resemble
each other needs several animals. ``validate_batch.py`` recomputes the ICC
of a real test folder against values kept in a private file.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import csv
import functools
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import f as f_distribution

from tools.mps_identity import FIELD_LABELS, AxonIdentity
from tools.mps_identity import FIELDS as IDENTITY_FIELDS
from tools.mps_occupancy import SIGMA_CAP_FRACTION
from tools.mps_randomization import DEFAULT_N_RANDOMIZATIONS, RandomizationParams


# ============================================================================
# Metadata for the ring batch (batch_rings.py)
# ============================================================================

@dataclass
class AxonMetadata:
    """Where one axon came from, for the ring batch's rows."""

    source_path: str
    animal_id: Optional[str] = None
    genotype: Optional[str] = None
    roi: Optional[str] = None
    axon_id: Optional[str] = None
    notes: str = ""

    @property
    def is_complete(self) -> bool:
        """True when the fields a group comparison needs are present."""
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
    would silently determine the statistics downstream.

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


# ============================================================================
# One axon, as the axon table holds it
# ============================================================================

# The axon table's columns worth comparing between groups, and what each
# is. The same names with "_discard" are the analysis without the
# clusters the axoplasm panel discarded, where that was done.
#
# Never a p-value (ks_p_mc_median, ks_p_mc_d): they are within-axon
# p-values (are this axon's clusters spaced unlike random ones?), and a
# p-value is not a quantity whose difference between genotypes means
# anything. Not the raw KS distance D either: its size under the null
# falls as about 0.87/sqrt(K), and K grows with the perimeter, so a
# difference in D between groups confounds organisation with cluster
# number. z_D, D standardised by its own null for that axon's K, is
# comparable (SCI-3).
COMPARABLE_COLUMNS: Dict[str, str] = {
    "occupancy_percent": "Occupancy of the perimeter (%)",
    "median_1nn_nm": "Median distance to the nearest cluster (nm)",
    "clusters_per_um": "Clusters per um of perimeter",
    "n_clusters_kept": "Clusters kept",
    "perimeter_um": "Perimeter (um)",
    "median_area_nm2": "Median cluster area (nm^2)",
    "median_r_eff_nm": "Median cluster effective radius (nm)",
    "delta_z_mean_nm": "Axial period, mean Delta-Z (nm)",
    "ks_z": "KS distance against its own null, z_D (clusters against randomized ones)",
    "nn_regularity_index": "1NN regularity index r = median 1NN x K / P",
    # Not a property of the axon but of its reconstruction: a group whose
    # contours are more inflated than the other's would show a difference
    # in every perimeter-derived column above without any biology.
    "contour_tour_over_hull": "Contour length over its convex hull",
}

# Columns that say how a row was measured. A table that pools rows where
# one of these takes two values is pooling two measurements, and a
# difference between groups can then be a difference in method.
GUARD_COLUMNS: Tuple[str, ...] = (
    "table_version", "pixel_size_source", "eps_nm", "min_samples",
    "cluster_method", "cluster_scale_nm",
    "dbcv_threshold", "edge_reference", "slab_half_width_nm", "slab_source",
    "mahalanobis_threshold", "ellipse_mode", "contour_2opt",
    "randomization_requested",
)


@dataclass
class AxonRecord:
    """One axon: which it is, and its row of the axon table."""

    source: str
    identity: AxonIdentity = field(default_factory=AxonIdentity)
    # The axon table's row: built by axon_export.axon_row for an axon
    # analysed here, or read back from a table (then every cell is text).
    row: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    # C3 / C4: the per-axon null once per closure the batch ran (tools.clustering_integration.null_closure_rows),
    # written to <table>_null_by_closure.csv; empty without a null.
    null_rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Analysed, with a row to show for it."""
        return self.error is None and bool(self.row)

    @property
    def axon_id(self) -> str:
        return str(self.row.get("axon_id") or "")

    def label(self, name: str) -> Optional[str]:
        """An identity field, or None when it is empty.

        Raises for a name that is not an identity field: grouping by a
        field that does not exist used to give every axon None and an
        empty comparison with nothing but a warning to show for it.
        """
        if name not in IDENTITY_FIELDS:
            raise ValueError(
                f"{name!r} is not one of the identity fields "
                f"({', '.join(IDENTITY_FIELDS)}).")
        value = str(getattr(self.identity, name) or "")
        return value or None

    def value(self, column: str) -> Optional[float]:
        """A number of the row, or None when the cell is empty or not one."""
        cell = self.row.get(column)
        if cell is None or isinstance(cell, bool):
            return None
        if isinstance(cell, str):
            cell = cell.strip()
            if not cell or cell.lower() in ("true", "false", "nan"):
                return None
        try:
            number = float(cell)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None


def column_for(name: str, discard: bool) -> str:
    """The table's column for ``name``, measured or with the discard.

    What the discard cannot change -- the axial period -- has one column
    only, and is the same either way.
    """
    from tools.axon_export import DISCARD_SUFFIX, MEASURED_COLUMNS

    if discard and name in MEASURED_COLUMNS:
        return name + DISCARD_SUFFIX
    return name


# ============================================================================
# Running a batch
# ============================================================================

@dataclass
class BatchSettings:
    """What every axon of a batch is analysed with.

    The main window's own settings, read when the batch starts: a batch
    measures the way the axon window does, save for the ROI it does not
    have.
    """

    eps_nm: float
    min_samples: int
    slab_half_width_nm: float
    dbcv_threshold: float
    mahalanobis_threshold: float
    # The randomization and the occupancy cap, from the same settings the
    # axon window reads (SCI-3).
    n_randomizations: int = DEFAULT_N_RANDOMIZATIONS
    random_seed: int = 0
    randomization_params: RandomizationParams = field(default_factory=RandomizationParams)
    sigma_cap_fraction: float = SIGMA_CAP_FRACTION
    # How eps was chosen on the k-distance graph (P-R58), when it was: the
    # settings' record, written beside every row (on another file's graph).
    eps_choice: Dict[str, Any] = field(default_factory=dict)
    # Phase D: the method saved in the settings (D12: only the chosen one
    # runs; the comparison is a separate box in the batch window).
    cluster_method: str = "DBSCAN"
    cluster_params: Dict[str, Any] = field(default_factory=dict)
    cluster_methods_compared: List[str] = field(default_factory=list)
    # The Working scale panel's c_label, from the settings (2026-10-29: the batch used 14 whatever the panel said).
    # It reaches the DBSCAN rule, the events / eclipse / size plug-ins and the per-axon null of every file, and each
    # row records it (c_label_used, c_label_source; null_c_label when a null ran).
    c_label: float = 14.0
    # C3 / C4: what the per-axon null is closed on ("n_locs", "k" or "both"; settings key batch_null_closure). With
    # "both" the axon table's row carries the n_locs null (the default) and <table>_null_by_closure.csv both.
    null_closure: str = "n_locs"

    @classmethod
    def from_settings(cls, settings: Any) -> "BatchSettings":
        stored = dict(getattr(settings, "randomization_params", {}) or {})
        try:
            n_rand = int(stored.get("n_iterations", DEFAULT_N_RANDOMIZATIONS))
        except (TypeError, ValueError):
            n_rand = DEFAULT_N_RANDOMIZATIONS
        try:
            seed = int(stored.get("seed", 0))
        except (TypeError, ValueError):
            seed = 0
        record = getattr(settings, "eps_choice", None)
        return cls(eps_nm=float(settings.eps_nm),
                   min_samples=int(settings.min_samples),
                   slab_half_width_nm=float(settings.slab_half_width_nm),
                   dbcv_threshold=float(settings.dbcv_threshold),
                   mahalanobis_threshold=float(
                       settings.mahalanobis_threshold),
                   n_randomizations=n_rand if n_rand >= 1 else DEFAULT_N_RANDOMIZATIONS,
                   random_seed=seed,
                   randomization_params=RandomizationParams.from_dict(stored),
                   sigma_cap_fraction=float(getattr(settings, "occupancy_sigma_cap_fraction",
                                                    SIGMA_CAP_FRACTION)),
                   eps_choice=dict(record) if isinstance(record, dict)
                   else {},
                   cluster_method=str(getattr(settings, "cluster_method", "DBSCAN") or "DBSCAN"),
                   cluster_params=dict(getattr(settings, "cluster_params", {}) or {}),
                   cluster_methods_compared=list(getattr(settings, "cluster_methods_compared", []) or []),
                   c_label=_settings_c_label(settings),
                   null_closure=_settings_null_closure(settings))


    def describe(self) -> str:
        method = ("" if self.cluster_method == "DBSCAN"
                  else f"clustering method {self.cluster_method} {self.cluster_params}; ")
        return (f"{method}DBSCAN eps {self.eps_nm:g} nm, min samples "
                f"{self.min_samples}; c_label {self.c_label:g} cycles per label; axial slab +/-"
                f"{self.slab_half_width_nm:g} nm around the main peak, "
                f"found per axon; Mahalanobis threshold "
                f"{self.mahalanobis_threshold:g}; the clusters compared "
                f"with {self.n_randomizations} randomized placements "
                f"(band +/-{self.randomization_params.band_half_width_nm:g} nm, "
                f"seed {self.random_seed}), as in the axon window")


def _settings_null_closure(settings: Any) -> str:
    """The settings' batch_null_closure ("n_locs" when missing or not a closure)."""
    from tools.mps_settings import BATCH_NULL_CLOSURES
    value = str(getattr(settings, "batch_null_closure", "n_locs") or "n_locs")
    return value if value in BATCH_NULL_CLOSURES else "n_locs"


def _settings_c_label(settings: Any) -> float:
    """The c_label of the settings, validated as ``tools.mps_settings.clean_c_label`` does (14 when missing)."""
    from tools.mps_settings import clean_c_label
    return clean_c_label(getattr(settings, "c_label", 14.0))


@dataclass
class BatchFile:
    """One file a batch is about to analyse, and who it is."""

    path: str
    identity: AxonIdentity
    # Where each identity field came from (mps_identity.propose).
    origin: Dict[str, str] = field(default_factory=dict)
    # The file's own pixel size; None for a Picasso file that lost its
    # metadata, which then needs one settled before it can be read.
    own_pixel_size_nm: Optional[float] = None
    needs_pixel_size: bool = False
    # Why the file cannot be opened at all, when it cannot: then it is
    # not a file that lacks a pixel size, and must not be asked one.
    unreadable: str = ""
    # Settled for a file that needed it: the value, and where it came
    # from (a PIXEL_SIZE_SOURCES token).
    pixel_size_nm: Optional[float] = None
    pixel_size_source: str = ""


@dataclass
class BatchPlan:
    """What a folder holds for a batch, before anything is analysed."""

    root: str
    files: List[BatchFile]
    # Left out, and why: the program's own outputs by name, the tables it
    # wrote by their columns.
    skipped_derived: List[str] = field(default_factory=list)
    skipped_tables: List[str] = field(default_factory=list)
    # Files that look like one acquisition twice (mps_io.duplicate_sources).
    duplicates: Dict[str, List[str]] = field(default_factory=dict)

    def describe(self) -> str:
        parts = [f"{len(self.files)} file(s) to analyse"]
        if self.skipped_derived:
            parts.append(f"{len(self.skipped_derived)} left out as files "
                         f"this program derived from others")
        if self.skipped_tables:
            parts.append(f"{len(self.skipped_tables)} left out as tables "
                         f"this program wrote")
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} acquisition(s) appear "
                         f"more than once")
        return "; ".join(parts) + "."


def plan_batch(root: str, pattern: str = "",
               patterns: Optional[Dict[str, str]] = None) -> BatchPlan:
    """
    The localization files under ``root``, each with the identity its path
    proposes and whether its pixel size is known.

    Nothing is read but the files' names, the first line of each CSV and
    the pixel size of each Picasso file, so this is fast enough to show
    before the user decides anything.
    """
    from tools import mps_io
    from tools.mps_identity import propose

    found, skipped = mps_io.find_localization_files(root, pattern=pattern)
    derived = [p for p in skipped
               if mps_io.is_derived_output(os.path.basename(p))]
    tables = [p for p in skipped if p not in set(derived)]
    files: List[BatchFile] = []
    for path in found:
        proposal = propose(path, patterns=patterns)
        own: Optional[float] = None
        needs = False
        unreadable = ""
        if mps_io.detect_format(path) == mps_io.FORMAT_PICASSO_HDF5:
            # A file that is not HDF5 at all reads as one without a pixel
            # size -- the metadata chain finds nothing in it -- and asking
            # the user for one would be asking the wrong question.
            try:
                import h5py

                if not h5py.is_hdf5(path):
                    unreadable = "could not be opened: it is not an HDF5 file"
                else:
                    own = mps_io.read_pixel_size(path)
                    needs = own is None
            except Exception as error:                # noqa: BLE001
                unreadable = f"could not be opened: {error}"
        files.append(BatchFile(path=path, identity=proposal.identity,
                               origin=dict(proposal.origin),
                               own_pixel_size_nm=own,
                               needs_pixel_size=needs,
                               unreadable=unreadable))
    return BatchPlan(root=root, files=files, skipped_derived=derived,
                     skipped_tables=tables,
                     duplicates=mps_io.duplicate_sources(found))


# Methods the batch never runs on its own (BaGoL phase F5): they take
# minutes per axon. They run only with the batch window's 'Slow methods
# (Emitters)' box ticked (``allow_slow``); otherwise every file is reported
# as not analysed, with the reason.
BATCH_SLOW_METHODS = frozenset({"emitters_bagol_like"})
SLOW_REFUSAL = ("not analysed: the saved clustering method, Emitters (BaGoL-like "
                "reimplementation), is slow (minutes per axon); tick 'Allow slow methods' "
                "in the batch window to run it")
# A2 (user decision 2026-10-23): every slow method (tools.method_costs.is_slow: Emitters, OPTICS, voronoi_csr) is
# refused by the Batch unless 'Allow slow methods' is ticked -- not only Emitters (BATCH_SLOW_METHODS keeps naming
# the methods that need the imaging buffer and the precisions).
SLOW_METHOD_REFUSAL = ("not analysed: the saved clustering method, {label}, is slow (about {per} per axon); tick "
                       "'Allow slow methods' in the batch window to run it")


def slow_refusal_for(method: str) -> str:
    """Why the Batch did not analyse a file saved with the slow ``method`` (A2)."""
    if method in BATCH_SLOW_METHODS:
        return SLOW_REFUSAL
    from tools.method_costs import estimate_seconds, method_label, words_seconds
    return SLOW_METHOD_REFUSAL.format(label=method_label(method), per=words_seconds(estimate_seconds([method], 4000)))


def needs_emitters_buffer(settings: BatchSettings, allow_slow: bool,
                          compared: Sequence[str] = ()) -> bool:
    """Whether this run will run Emitters, and so needs the imaging buffer asked once for the run (F5c: no
    default buffer). Only with 'Slow methods (Emitters)' ticked: otherwise Emitters is refused anyway."""
    if not allow_slow:
        return False
    return (settings.cluster_method in BATCH_SLOW_METHODS
            or any(str(m) in BATCH_SLOW_METHODS for m in compared))


def with_emitters_buffer(settings: BatchSettings, buffer: str) -> BatchSettings:
    """The settings with the run's imaging buffer in the Emitters parameters (only when Emitters is the saved
    method; the comparison passes it per method). Every row records it (emitters_buffer)."""
    import dataclasses
    from tools.emitters_prior import normalise_buffer
    if settings.cluster_method not in BATCH_SLOW_METHODS:
        return settings
    params = dict(settings.cluster_params)
    params["buffer"] = normalise_buffer(str(buffer))
    return dataclasses.replace(settings, cluster_params=params)


def _precisions(loc: Any, method: str) -> Optional[Any]:
    """(n, 2) = (lpx, lpy) in nm for a method that models them, else None (nothing changes for the others)."""
    if method not in BATCH_SLOW_METHODS:
        return None
    lpx, lpy = getattr(loc, "lpx_nm", None), getattr(loc, "lpy_nm", None)
    if lpx is None or lpy is None:
        return None
    import numpy as np
    return np.column_stack([np.asarray(lpx, dtype=float), np.asarray(lpy, dtype=float)])


def _load(item: BatchFile) -> Tuple[Any, Optional[str]]:
    """The file's localizations, or None and why it could not be read (never raises)."""
    from tools import mps_io

    name = os.path.basename(item.path)
    if item.unreadable:
        return None, item.unreadable
    if item.needs_pixel_size and item.pixel_size_nm is None:
        return None, (f"{name} carries no pixel size and none was given, so it "
                      f"was not read")
    try:
        loc = mps_io.load_localizations(
            item.path,
            pixel_size_nm=(item.pixel_size_nm if item.needs_pixel_size
                           else None))
        if item.needs_pixel_size:
            loc.pixel_size_source = item.pixel_size_source or "manual"
    except Exception as error:                        # noqa: BLE001
        return None, f"could not be read: {error}"
    return loc, None


def _analyze_loaded(loc: Any, item: BatchFile, settings: BatchSettings) -> Any:
    """``analyze_axon`` on a loaded file with the batch's settings (raises on failure)."""
    from tools.mps_analysis import analyze_axon

    extra: Dict[str, Any] = {}
    sigma = _precisions(loc, settings.cluster_method)
    if sigma is not None:
        extra["localization_sigma_nm"] = sigma
    if getattr(loc, "frame", None) is not None:
        # The Emitters layer links them into events; the event layer, the eclipse tolerance and the size's content
        # count events with them (the clustering integration), for every method.
        extra["localization_frame"] = loc.frame
    return analyze_axon(
        loc.x_nm, loc.y_nm, loc.z_nm, source_name=item.path, **extra,
        pixel_size_nm=loc.pixel_size_nm,
        pixel_size_source=loc.pixel_size_source,
        eps_nm=settings.eps_nm, min_samples=settings.min_samples,
        cluster_method=settings.cluster_method,
        cluster_params=settings.cluster_params,
        cluster_methods_compared=settings.cluster_methods_compared,
        slab_half_width_nm=settings.slab_half_width_nm,
        dbcv_threshold=settings.dbcv_threshold,
        mahalanobis_threshold=settings.mahalanobis_threshold,
        n_randomizations=settings.n_randomizations,
        random_seed=settings.random_seed,
        randomization_params=settings.randomization_params,
        sigma_cap_fraction=settings.sigma_cap_fraction)


def _file_nena(loc: Any) -> Tuple[Optional[float], str]:
    """The file's NeNA (tools.mps_cut_clusters.nena_from_localizations) and where it came from, or (None, why)."""
    from tools.mps_cut_clusters import nena_from_localizations
    try:
        src = nena_from_localizations(loc, "this file")
    except Exception as error:                        # noqa: BLE001 - a NeNA that fails leaves its columns empty
        return None, f"NeNA failed: {error}"
    return (src.nena_nm, src.words) if src.measured else (None, src.reason)


# The per-axon null in the batch (A9 sec. 5, regularity PR 4): off unless the batch window's box is ticked; a slow
# method's null is refused unless 'Slow methods' is ticked too (open question A2); the regularity's stage 2 at the
# pipeline level (P-R87 c).
NULL_REFUSED_SLOW = "null not run: {label} is slow; tick 'Allow slow methods' to run its null too"


def analyse_file(item: BatchFile, settings: BatchSettings,
                 z_store: Optional[Dict[str, Any]] = None,
                 allow_slow: bool = False, null_replicates: int = 0,
                 null_seed: int = 0) -> AxonRecord:
    """
    Load one file, analyse it as the axon window would, and build its row.

    ``allow_slow``: run a method of ``BATCH_SLOW_METHODS`` (the Emitters
    layer); without it such a file comes back not analysed, with the reason.

    ``z_store`` is the settings' z-calibration store
    (``MPSSettings.z_calibration_by_folder``): the row carries the file's
    z calibration as the axon window's export does (read, never applied).

    ``null_replicates`` > 0 also runs the per-axon pipeline null of the
    file's own method with that many replicates (and the regularity
    reading, stage 2 at the pipeline level); a slow method's null is
    refused without ``allow_slow``. The row's v13 block carries the file's
    NeNA, the DBSCAN rule and the null (tools.clustering_integration).

    Never raises: a file that cannot be read or analysed comes back with
    the reason, so a batch reports it instead of losing it.
    """
    from tools.axon_export import axon_row

    from tools.method_costs import is_slow as _is_slow
    if (settings.cluster_method in BATCH_SLOW_METHODS or _is_slow(settings.cluster_method)) and not allow_slow:
        return AxonRecord(source=item.path, identity=item.identity, error=slow_refusal_for(settings.cluster_method))
    loc, why = _load(item)
    if loc is None:
        return AxonRecord(source=item.path, identity=item.identity, error=why)
    try:
        analysis = _analyze_loaded(loc, item, settings)
        from tools import clustering_integration as ci
        from tools import mps_zcalibration as zcal

        nena, nena_words = _file_nena(loc)
        lpc = ci.locs_per_cycle_of(loc, nena)
        # B3 (c): localizations per cycle by depth, the slab against the whole file (a diagnostic; the rule keeps
        # the whole file's value)
        from tools import lpc_depth
        lpc_prof, lpc_why = None, ""
        try:
            lpc_prof = lpc_depth.lpc_depth_profile_of(loc, analysis, nena)
        except ValueError as error:
            lpc_why = f"not computed: {error}"
        null = None
        by_closure: Dict[str, Any] = {}
        if int(null_replicates) > 0:
            run = ci.run_of(analysis, axon_id=item.path)
            if ci.is_slow(run.method) and not allow_slow:
                null = ci.NullOutcome(key=run.key, method=run.method, params=dict(run.params), status="refused",
                                      message=NULL_REFUSED_SLOW.format(label=ci.method_label(run.method)))
            else:
                # C3 / C4: one null per closure asked for (the settings' batch_null_closure); the row carries the
                # first (n_locs when it is among them), <table>_null_by_closure.csv every one
                by_closure = ci.run_nulls_by_closure(
                    run, closure=settings.null_closure, n_replicates=int(null_replicates), nena_nm=nena,
                    seed=int(null_seed), allow_slow=allow_slow, gap_stage2=ci.GAP_STAGE2_BATCH,
                    c_label=float(settings.c_label))
                null = next(iter(by_closure.values()))
        integration = ci.axon_columns(analysis, nena_nm=nena, nena_source=nena_words, locs_per_cycle=lpc,
                                      null=null, c_label=float(settings.c_label), lpc_profile=lpc_prof,
                                      lpc_profile_note=lpc_why)
        if null is not None and not null.ok:
            integration["null_type"] = f"none ({null.status}: {null.message})"
        # finding (n), design review 2026-10-29: the methods block says why it is empty in a batch row; when the
        # batch's own comparison runs, fill_methods_columns fills it at export (the comparison runs after the rows)
        for name in METHODS_BLOCK:
            if integration.get(name) == ci.METHODS_NOT_RUN:
                integration[name] = BATCH_METHODS_NOT_RUN
        z_state = zcal.state_for(loc.info, zcal.store_entry(z_store,
                                                            item.path))
        measured, discard, unanchored = unanchored_for_file(analysis, item.path)
        row = axon_row(measured, identity=item.identity, discard=discard,
                       z_calibration=zcal.export_columns(z_state),
                       eps_choice=settings.eps_choice, integration=integration,
                       unanchored=unanchored)
        null_rows = ci.null_closure_rows(row, by_closure) if by_closure else []
    except Exception as error:                        # noqa: BLE001
        return AxonRecord(source=item.path, identity=item.identity,
                          error=f"could not be analysed: {error}")
    return AxonRecord(source=item.path, identity=item.identity, row=row, null_rows=null_rows)


def unanchored_for_file(analysis: Any, data_path: str) -> Tuple[Any, Optional[Any], Optional[Dict[str, Any]]]:
    """(measured, discard, v18 block) of one file (package P15, decision D4: always with and without the discard).

    The automatic rule of ``tools.mps_unanchored`` on the file's analysis, the decisions by hand saved beside the
    file (``<data>_unanchored.json``, D11) when there are any, and the analysis without the excluded clusters
    (``compare_unanchored``; as cheap as a copy when nothing is excluded). ``MPS_UNANCHORED_AUTO=0`` turns it off
    (the row then says "not run", as before P15). Never raises: a failure keeps the analysis as it is and the block
    says why."""
    from tools import mps_unanchored as mu
    if os.environ.get("MPS_UNANCHORED_AUTO", "").strip() == "0":
        return analysis, None, None
    block: Dict[str, Any]
    try:
        result = mu.auto_for_analysis(analysis)
        note = ""
        try:
            decisions = mu.load_decisions(mu.decisions_path(data_path))
        except mu.ManualDecisionsError as error:
            decisions, note = mu.ManualDecisions(), f"decisions by hand not read: {error}"
        flags = mu.final_flags(result, decisions, data_source=os.path.basename(data_path))
        block = mu.unanchored_block(flags)
        if getattr(analysis, "contour_2opt", None) == "one start":
            # 'one start' was chosen for this file (a long perimeter): the discard column would rebuild the
            # contour from every start, the cost that answer avoided; the flags are still written
            note = "; ".join(w for w in (note, "discard column not computed: the perimeter of this file was built "
                                         "from one start") if w)
            block["unanchored_warnings"] = " | ".join(w for w in (block.get("unanchored_warnings"), note) if w)
            return analysis, None, block
        comparison = mu.compare_unanchored(analysis, flags)
        if note:
            block["unanchored_warnings"] = " | ".join(w for w in (block.get("unanchored_warnings"), note) if w)
        return comparison.all_clusters, comparison.discard_applied, block
    except Exception as error:                        # noqa: BLE001 - the row is written without the discard
        block = mu.unanchored_block(None)
        block["unanchored_status"] = "failed"
        block["unanchored_failure"] = f"{type(error).__name__}: {error}"
        return analysis, None, block


NULL_BY_CLOSURE_SUFFIX = "_null_by_closure.csv"


def write_null_by_closure(records: Sequence[AxonRecord], table_path: str) -> str:
    """C3 / C4: ``<table>_null_by_closure.csv`` beside the axon table -- one row per axon x closure the batch's null
    ran (column null_closure), the same columns for every closure; '' when no null ran. Never overwrites."""
    from tools import clustering_integration as ci
    from tools import multimethod as mm
    rows = [r for rec in records if rec.ok for r in rec.null_rows]
    if not rows:
        return ""
    base = os.path.splitext(table_path)[0]
    return mm.write_rows_csv(base + NULL_BY_CLOSURE_SUFFIX, rows, ci.null_by_closure_columns())


def null_cost_per_file(method: str, n_locs_slab: int, n_replicates: int, closure: str = "n_locs") -> float:
    """Seconds of the per-axon null of one file before Run (C3 / C4: "both" is about twice "n_locs", and the K
    closure adds its pipeline pilots), with stage 2 of the regularity at the pipeline level as the Batch runs it
    (an upper bound: it runs only when stage 1 says "more regular"). A first estimate: the cost table's per-method
    figure for a slab of ``n_locs_slab`` localizations (tools.mps_pipeline_null.estimated_seconds_per_replicate)."""
    from types import SimpleNamespace
    from tools import clustering_integration as ci
    run: Any = SimpleNamespace(method=str(method), analysis=SimpleNamespace(n_locs_slab=int(n_locs_slab)))
    return ci.null_estimate_seconds(run, int(n_replicates), closure, ci.GAP_STAGE2_BATCH)


def run_batch(
    items: Sequence[BatchFile],
    settings: BatchSettings,
    *,
    on_done: Optional[Callable[[int, AxonRecord], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    z_store: Optional[Dict[str, Any]] = None,
    allow_slow: bool = False,
    null_replicates: int = 0,
) -> List[AxonRecord]:
    """
    Analyse ``items`` in order, one record each.

    ``on_done(index, record)`` is called after each file, and
    ``cancelled()`` before each: when it returns True the batch stops
    there and returns what it has. A file already started is finished.
    ``null_replicates`` > 0: the per-axon null of every file too
    (:func:`analyse_file`).
    """
    records: List[AxonRecord] = []
    for index, item in enumerate(items):
        if cancelled is not None and cancelled():
            break
        # allow_slow and the null are passed only when given, so the call stays the one of before F5 for every
        # batch that does not ask for them.
        kw: Dict[str, Any] = {}
        if allow_slow:
            kw["allow_slow"] = True
        if null_replicates:
            kw["null_replicates"] = int(null_replicates)
        record = analyse_file(item, settings, z_store, **kw)
        records.append(record)
        if on_done is not None:
            on_done(index, record)
    return records


# ---------------------------------------------------------------------------
# F1 (manual test of 2026-10-22): the perimeter's time, estimated before a batch
# ---------------------------------------------------------------------------
# A file with ~1,100 kept clusters builds its perimeter by 2-opt from every start for about 9 minutes on the fit of
# tools/mps_perimeter_cost (1.4 x that on a messy contour), and the batch never asks (no one to ask in its worker
# thread). The batch window now clusters each file first, counts the clusters the perimeter would join, and asks
# once for the whole batch when any file reaches the warning threshold (the same 60 s [ad hoc] as the axon window).

@dataclass
class PerimeterEstimate:
    """What the scan found for one file: the kept clusters, when they reach the warning threshold."""

    source: str
    # The kept clusters the perimeter would join; None below the threshold (the scan stops at the perimeter and
    # does not need the number) or when the file could not be clustered.
    n_clusters: Optional[int] = None
    error: str = ""

    @property
    def over(self) -> bool:
        """The file reaches the warning threshold (tools.mps_perimeter_cost.needs_warning)."""
        return self.n_clusters is not None

    @property
    def seconds_every_start(self) -> float:
        from tools import mps_perimeter_cost as cost
        return cost.estimate_perimeter_seconds(self.n_clusters or 0)

    @property
    def seconds_one_start(self) -> float:
        from tools import mps_perimeter_cost as cost
        return cost.estimate_one_start_seconds(self.n_clusters or 0)


class _BelowThreshold(BaseException):
    """The scan reached the perimeter without a warning: the file is below the threshold (internal)."""


def scan_perimeter_costs(
    items: Sequence[BatchFile],
    settings: BatchSettings,
    *,
    z_store: Optional[Dict[str, Any]] = None,
    allow_slow: bool = False,
    cancelled: Optional[Callable[[], bool]] = None,
    on_done: Optional[Callable[[int, PerimeterEstimate], None]] = None,
    observer: Optional[Callable[[str, Optional[int], Optional[int]], None]] = None,
) -> List[PerimeterEstimate]:
    """For each file, whether its perimeter would reach the warning threshold, and with how many clusters.

    Each file is read and clustered exactly as ``analyse_file`` does (same settings, same curation), and the
    analysis is stopped where it would build the perimeter: at the question of ``tools.mps_perimeter_cost`` when
    the clusters reach the threshold (the number is recorded), or at the check point "before the perimeter"
    when they do not. So the scan costs one reading and one clustering per file, and nothing it computes is kept.
    ``cancelled()`` stops the scan at the next check point; the list then holds the files scanned so far.
    ``observer`` gets the check points (``tools.mps_progress``), for a progress bar.
    """
    from tools import mps_cancel, mps_progress
    from tools import mps_perimeter_cost as cost

    out: List[PerimeterEstimate] = []
    for index, item in enumerate(items):
        if cancelled is not None and cancelled():
            break
        seen: Dict[str, int] = {}
        below = [False]

        def gate(n: int) -> str:
            seen["n"] = int(n)
            return cost.CANCEL

        def watch(where: str, done: Optional[int], total: Optional[int]) -> None:
            if observer is not None:
                observer(where, done, total)
            if where == "before the perimeter" and "n" not in seen:
                raise _BelowThreshold()

        estimate = PerimeterEstimate(source=item.path)
        try:
            with cost.asking(gate), mps_progress.observing(watch), \
                    mps_cancel.cancellable(cancelled):
                if observer is not None:
                    observer(mps_progress.LOADING, None, None)
                record = (analyse_file(item, settings, z_store, allow_slow=True) if allow_slow
                          else analyse_file(item, settings, z_store))
            if "n" in seen:
                estimate.n_clusters = seen["n"]
            elif record.error:
                estimate.error = record.error
        except _BelowThreshold:
            pass
        except mps_cancel.AnalysisCancelled:
            break
        out.append(estimate)
        if on_done is not None:
            on_done(index, estimate)
    return out


def perimeter_policy_gate(choice: str) -> Callable[[int], str]:
    """The gate a batch installs (``tools.mps_perimeter_cost.asking``) once the user chose for the whole batch.

    EVERY_START is what the batch always did (the analysis' note is the same); ONE_START is recorded in each
    such file's notes and in its contour_2opt column, as in the axon window.
    """
    from tools import mps_perimeter_cost as cost
    if choice not in (cost.EVERY_START, cost.ONE_START):
        raise ValueError(f"a batch runs with {cost.EVERY_START!r} or {cost.ONE_START!r}, not {choice!r}")
    return lambda _n: choice


# ---------------------------------------------------------------------------
# Phase D (D0): the optional comparison of methods, a table apart
# ---------------------------------------------------------------------------
COMPARISON_COLUMNS: Tuple[str, ...] = (
    "source", "axon_id", "cluster_method", "cluster_params", "cluster_scale_nm",
    "clusters_per_um", "median_1nn_nm", "median_r_eff_nm", "occupancy_percent",
    "n_clusters_kept", "time_s", "error")
COMPARISON_FILE = "batch_methods_comparison.csv"
COMPARISON_SUFFIX = "_methods_comparison.csv"


def methods_to_compare(settings: BatchSettings) -> List[str]:
    """The methods the batch's 'Compare methods (slow)' box runs (D12, review of D0 M4).

    Exactly the ones the user compared and chose in the axon window's
    'Compare methods...' (``cluster_methods_compared`` of the settings), in
    that order, with DBSCAN first as the reference when it is not among
    them; names the registry no longer has, and 'External labels', are left
    out. Nothing chosen yet: an empty list, and the box stays disabled. It
    never adds a method on its own, so with 'Experimental clustering
    methods' off no experimental method runs unless the user chose it.
    """
    from tools.clustering_strategies import COMPAT_METHOD, strategy_registry
    reg = strategy_registry()
    chosen: List[str] = []
    for m in settings.cluster_methods_compared or []:
        name = str(m).strip()
        if name and name in reg and name != "External labels" and name not in chosen:
            chosen.append(name)
    if not chosen:
        return []
    return chosen if COMPAT_METHOD in chosen else [COMPAT_METHOD] + chosen


def compare_methods(
    items: Sequence[BatchFile],
    settings: BatchSettings,
    methods: Sequence[str],
    *,
    cancelled: Optional[Callable[[], bool]] = None,
    z_store: Optional[Dict[str, Any]] = None,
    on_done: Optional[Callable[[int, int], None]] = None,
    allow_slow: bool = False,
    emitters_buffer: Optional[str] = None,
    analyses: Optional[Sequence[str]] = None,
    results_out: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Every file by every method in ``methods``, one row each (D12: only
    when the batch window's 'Compare methods (slow)' box is ticked), through
    ``tools.multimethod`` (the clustering integration): each file is loaded
    once, DBSCAN 25/10 is always run first as the reference, every method
    runs on the same localizations, and the analyses (``analyses``, the
    integration's defaults when None) run on every method's result.

    The saved method runs with the saved parameters (DBSCAN with the saved
    eps / min samples), the others with the registry defaults. Slow methods
    are refused without ``allow_slow`` (the 'Slow methods' box: open
    question A2). ``on_done(file_index, method_index)`` after each method.
    Never raises: a failed run is a row with ``error``. ``emitters_buffer``:
    the imaging buffer asked for the run (F5c), put in the parameters of a
    slow method (Emitters) when they have none. ``results_out``: when a list,
    each file's ``tools.clustering_integration.ComparisonResult`` is
    appended to it (the long table, the agreement and the summary).
    """
    import dataclasses
    import json as _json
    from tools import clustering_integration as ci
    from tools import multimethod as mm
    from tools.axon_export import axon_id as _axon_id
    from tools.cluster_quality import describe_roi
    from tools.clustering_strategies import strategy_registry
    rows: List[Dict[str, Any]] = []
    reg = strategy_registry()
    chosen = list(ci.DEFAULT_ANALYSES if analyses is None else analyses)

    def params_for(method: str) -> Dict[str, Any]:
        if method == mm.REFERENCE_METHOD:
            p: Dict[str, Any] = {"eps": float(settings.eps_nm), "min_samples": int(settings.min_samples)}
        elif method == settings.cluster_method:
            p = dict(settings.cluster_params)
        else:
            info = reg.get(method)
            p = {} if info is None else {k: info.defaults[k]["value"] for k in info.numeric_params()}
        if method in BATCH_SLOW_METHODS and emitters_buffer and not p.get("buffer"):
            p["buffer"] = str(emitters_buffer)
        return p

    for i, item in enumerate(items):
        if cancelled is not None and cancelled():
            return rows
        loc, why = _load(item)
        n_locs = 0 if loc is None else int(np.asarray(loc.x_nm).size)
        plan = mm.plan_runs(list(methods), n_locs, params_for=params_for, allow_slow=allow_slow)

        def run_fn(method: str, params: Dict[str, Any], _loc: Any = loc, _item: BatchFile = item,
                   _why: Optional[str] = why) -> Any:
            if _loc is None:
                raise RuntimeError(_why or "not read")
            p = dict(params)
            per = dataclasses.replace(settings, cluster_method=method, cluster_params=p,
                                      cluster_methods_compared=list(methods))
            if method == mm.REFERENCE_METHOD:
                per = dataclasses.replace(per, eps_nm=float(p.get("eps", settings.eps_nm)),
                                          min_samples=int(p.get("min_samples", settings.min_samples)),
                                          cluster_params={})
            return _analyze_loaded(_loc, _item, per)

        def done(j: int, n: int, run: Any, _i: int = i) -> None:
            if on_done is not None:
                on_done(_i, j)

        nena = None if loc is None else _file_nena(loc)[0]
        # finding (m), design review 2026-10-29: the comparison's axon_id held the file's full path; it is the axon
        # table's own identity (tools.mps_identity.axon_id of the source's name and the selection), so the
        # batch_methods_* rows join the axon table's rows
        aid = _axon_id(item.path, describe_roi(None))
        runs = mm.run_methods(run_fn, list(methods), n_locs=n_locs, axon_id=aid, allow_slow=allow_slow,
                              cancelled=cancelled, on_done=done, plan=plan)
        first = next((r.analysis for r in runs if r.ok), None)
        if first is not None:
            own = _axon_id(first.source_name, describe_roi(first.roi))
            if own != aid:                      # a selection the batch did not expect: the analysis' own wins
                aid = own
                for r in runs:
                    r.axon_id = aid
        extras = ci.comparison_extras(nena_nm=nena, analysis=first, c_label=float(settings.c_label))
        names = [n for n in chosen if n in ci.register_all_plugins()]
        try:
            long_rows = mm.apply_analyses(runs, names, allow_slow=allow_slow, cancelled=cancelled, extras=extras)
        except BaseException:                         # noqa: BLE001 - a cancellation: keep the runs' rows
            long_rows = []
        if results_out is not None:
            results_out.append(ci.ComparisonResult(
                axon_id=aid, runs=runs, analyses=names, long_rows=long_rows,
                wide_rows=mm.wide_view(long_rows), agreement_rows=mm.pairwise_agreement(runs),
                summary_rows=mm.conclusion_summary(long_rows)))
        for run in runs:
            an = run.analysis
            row: Dict[str, Any] = {c: None for c in COMPARISON_COLUMNS}
            row["source"] = item.path
            row["axon_id"] = aid
            row["cluster_method"] = run.method if run.key in (run.method, mm.REFERENCE_KEY) else run.key
            row["cluster_params"] = _json.dumps(run.params, sort_keys=True)
            if an is not None:
                row["axon_id"] = _axon_id(an.source_name, describe_roi(an.roi))
                row["cluster_scale_nm"] = an.cluster_scale_nm
                for c in ("clusters_per_um", "median_1nn_nm", "median_r_eff_nm", "occupancy_percent",
                          "n_clusters_kept"):
                    row[c] = getattr(an, c, None)
            row["time_s"] = round(float(run.seconds), 3)
            if run.status == mm.STATUS_REFUSED:
                row["error"] = (f"not analysed: {ci.method_label(run.method)} is slow; tick 'Slow methods "
                                "(Emitters)' in the batch window to run it")
            else:
                row["error"] = run.error or ""
            rows.append(row)
    return rows


METHODS_BLOCK: Tuple[str, ...] = ("methods_compared", "methods_long_file", "methods_depends_on_method")
BATCH_METHODS_NOT_RUN = "not run (the Batch's 'Compare methods (slow)' box was not ticked)"
BATCH_METHODS_NOT_REACHED = "not run (the Batch's comparison stopped before this file: cancelled or failed)"


def fill_methods_columns(rows: Sequence[Dict[str, Any]], results: Sequence[Any], long_file: str = "") -> int:
    """Fill the methods block (methods_compared, methods_depends_on_method, methods_long_file) of the batch's rows
    from its own comparison (``compare_methods(..., results_out=...)``), as the axon window fills it from its last
    'Compare methods...': the runs by the names the user knows (DBSCAN 25/10 first), the readouts whose verdict
    changed with the method ('none'), and the long table's file name (``long_file``, '' = not written). Rows are
    matched on axon_id (the comparison's is the axon table's own: finding m). A row whose file the comparison did
    not reach says so. Returns the rows filled."""
    from tools import clustering_integration as ci
    if not results:
        return 0
    by_axon = {str(r.axon_id): r for r in results}
    n = 0
    for row in rows:
        res = by_axon.get(str(row.get("axon_id") or ""))
        if res is None:
            for name in METHODS_BLOCK:
                row[name] = BATCH_METHODS_NOT_REACHED
            continue
        row["methods_compared"] = "; ".join(res.labels.get(r.key, r.key) for r in res.runs)
        row["methods_depends_on_method"] = "; ".join(res.depends_on_method()) or "none"
        row["methods_long_file"] = os.path.basename(long_file) if long_file else ci.METHODS_TABLES_NOT_WRITTEN
        n += 1
    return n


def write_methods_comparison_long(results: Sequence[Any], table_path: str) -> Dict[str, str]:
    """The batch's comparison as ``<table stem>_methods_long.csv`` (+ agreement, summary, wide) beside the axon table
    (open question A4; ``<table stem>_methods_comparison.csv`` stays as it was). Never overwrites."""
    from tools import clustering_integration as ci
    if not results:
        return {}
    merged = ci.ComparisonResult(
        axon_id="batch", runs=[r for res in results for r in res.runs],
        analyses=list(results[0].analyses), long_rows=[x for res in results for x in res.long_rows],
        wide_rows=[x for res in results for x in res.wide_rows],
        agreement_rows=[x for res in results for x in res.agreement_rows],
        summary_rows=[x for res in results for x in res.summary_rows])
    stem = os.path.splitext(os.path.abspath(table_path))[0]       # the table's own name, not a fixed "batch"
    return ci.write_comparison(stem, merged)


def comparison_path(table_path: str) -> str:
    """``<table stem>_methods_comparison.csv`` beside the axon table: it takes the name the user chose for the table
    (``COMPARISON_FILE`` is the old fixed name, kept for the files already written)."""
    return os.path.splitext(os.path.abspath(table_path))[0] + COMPARISON_SUFFIX


def write_methods_comparison_file(rows: Sequence[Dict[str, Any]], path: str) -> str:
    """Write the comparison rows (COMPARISON_COLUMNS) to ``path`` -- or, when a file is there already, to the first
    free ``<stem>_2.csv``, ``<stem>_3.csv``, ... (``tools.results_table.free_path``: a second batch's comparison
    never overwrites the first's; design review 2026-10-29). Returns the path written."""
    import csv
    from tools.results_table import free_path
    target = free_path(path)
    with open(target, "x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COMPARISON_COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in COMPARISON_COLUMNS})
    return target


def write_methods_comparison(rows: Sequence[Dict[str, Any]], path: str) -> int:
    """:func:`write_methods_comparison_file` (never overwrites); returns the count."""
    write_methods_comparison_file(rows, path)
    return len(rows)


# ============================================================================
# Writing a batch to a table
# ============================================================================

@dataclass
class ExportCheck:
    """What writing a batch's rows to one table would do."""

    path: str
    rows: List[Dict[str, Any]]
    # Line numbers of the table that already hold one of these axons with
    # this same selection: a previous batch over the same files.
    already: List[int] = field(default_factory=list)
    # Files the table already holds with another selection -- exported
    # from the axon window with an ROI. The batch's row would be the same
    # axon a second time, so it is left out.
    in_table_otherwise: Dict[str, List[str]] = field(default_factory=dict)

    def rows_to_write(self) -> List[Dict[str, Any]]:
        """The rows, without those of files the table holds otherwise."""
        return [r for r in self.rows
                if _base(r.get("source")) not in self.in_table_otherwise]


def _base(path: Any) -> str:
    return os.path.basename(str(path or "")).lower()


def check_export(records: Sequence[AxonRecord], path: str) -> ExportCheck:
    """
    Look at the table at ``path`` before these records' rows go into it.

    Raises TableMismatch when the table has other columns (another version
    of the table, or not a table of axons): nothing is written then.
    """
    from tools.axon_export import AXON_KEY_COLUMNS
    from tools.results_table import (
        check_appendable, column_values, duplicate_rows,
    )

    rows = [r.row for r in records if r.ok]
    if not rows:
        return ExportCheck(path=path, rows=[])
    check_appendable(path, rows)
    already = duplicate_rows(path, rows, AXON_KEY_COLUMNS)

    # The same file under another selection. Read row by row rather than
    # through column_values, which cannot say which ROI went with which
    # file.
    otherwise: Dict[str, List[str]] = {}
    if column_values(path, "source"):
        ours = {_base(r.get("source")): str(r.get("roi") or "")
                for r in rows}
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for existing in csv.DictReader(handle):
                base = _base(existing.get("source"))
                roi = existing.get("roi") or ""
                if base in ours and roi != ours[base]:
                    otherwise.setdefault(base, []).append(roi)
    return ExportCheck(path=path, rows=rows, already=already,
                       in_table_otherwise=otherwise)


def log_path(table_path: str) -> str:
    """Where the batch's log goes, beside its table."""
    base, ext = os.path.splitext(table_path)
    return f"{base}_batch_log{ext or '.csv'}"


def write_log(records: Sequence[AxonRecord], table_path: str,
              left_out: Optional[Dict[str, str]] = None,
              skipped: Optional[Dict[str, str]] = None) -> str:
    """
    One row per file the batch saw, and what became of it, beside the
    table: analysed and written, left out and why, or failed and why. A
    batch never loses a file without saying so.

    ``left_out`` maps a file to why its row was not written; ``skipped``
    a file to why it was never analysed.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: List[Dict[str, Any]] = []
    for path, why in (skipped or {}).items():
        rows.append({"file": path, "status": "not analysed", "reason": why,
                     "axon_id": "", "analysis_id": "", "n_warnings": "",
                     "logged_at": stamp})
    for record in records:
        why = (left_out or {}).get(record.source)
        if not record.ok:
            status, reason = "failed", record.error or ""
        elif why:
            status, reason = "not written", why
        else:
            status, reason = "written", ""
        rows.append({
            "file": record.source, "status": status, "reason": reason,
            "axon_id": record.row.get("axon_id", ""),
            "analysis_id": record.row.get("analysis_id", ""),
            "n_warnings": record.row.get("n_warnings", ""),
            "logged_at": stamp})
    # never overwrite (design review 2026-10-29): a second batch exported to the same table used to replace the first
    # batch's log, and with it the record of what became of those files; it now goes to <table>_batch_log_2.csv
    from tools.results_table import free_path
    out = free_path(log_path(table_path))
    with open(out, "x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "file", "status", "reason", "axon_id", "analysis_id",
            "n_warnings", "logged_at"])
        writer.writeheader()
        writer.writerows(rows)
    return out


# ============================================================================
# Reading a table back
# ============================================================================

def read_axon_table(path: str) -> Tuple[List[AxonRecord], List[str]]:
    """
    The axons of an exported axon table, and what to know about them.

    Accepts any table the axon window or a batch wrote. Refuses the copy
    made for Excel -- its decimals are commas -- and anything that is not
    a table of axons.
    """
    from tools.results_table import read_header

    header = read_header(path)
    if header is None:
        raise ValueError(f"{os.path.basename(path)} is empty.")
    if len(header) == 1 and ";" in header[0]:
        raise ValueError(
            f"{os.path.basename(path)} is the copy made for Excel, with "
            f"decimal commas. Open the table it was copied from.")
    if "axon_id" not in header or "analysis_id" not in header:
        raise ValueError(
            f"{os.path.basename(path)} is not a table of axons: it has no "
            f"axon_id and analysis_id columns.")
    notes: List[str] = []
    absent = [n for n in IDENTITY_FIELDS if n not in header]
    if absent:
        notes.append(
            f"This table has no {', '.join(absent)} column: it was written "
            f"before {'it was' if len(absent) == 1 else 'they were'} part "
            f"of the table, so those are empty for every axon.")
    records: List[AxonRecord] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = AxonIdentity(**{n: row.get(n) or ""
                                       for n in IDENTITY_FIELDS})
            records.append(AxonRecord(source=row.get("source") or "",
                                      identity=identity, row=dict(row)))
    return records, notes


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
    null_p95: Optional[float] = None      # ICC this design reaches by chance (at null_percentile)
    warnings: List[str] = field(default_factory=list)
    null_percentile: float = 95.0
    # Optional plain-language labels (off by default): the clustered-data
    # convention small / medium / large = 0.05 / 0.10 / 0.15.
    labels: bool = False

    @property
    def exceeds_chance(self) -> Optional[bool]:
        """True when the ICC is above what this design produces by chance."""
        if self.icc is None or self.null_p95 is None:
            return None
        return self.icc > self.null_p95

    @property
    def severity(self) -> str:
        """How to read this ICC, without adjectives (SCI-3): within chance,
        or the design effect and the effective number of observations. The
        cut-offs 0.2 / 0.5 used before had no source."""
        if self.icc is None:
            return "undetermined"
        # A raw ICC is not interpretable on its own: the estimator is
        # clamped at zero and so biased upward, and with few groups a true
        # ICC of zero routinely estimates well above it -- by how much is
        # the F quantile of icc_chance_level.
        if self.null_p95 is not None and self.icc <= self.null_p95:
            return "within chance for this design"
        words = ("" if self.design_effect is None
                 else f"design effect {self.design_effect:.1f}")
        if self.effective_n is not None:
            words += (f": {self.n_observations} observations are worth "
                      f"{self.effective_n:.1f}")
        if self.labels:
            label = icc_label(self.icc)
            words += f" ({label}, Zyzanski et al. 2004 convention)"
        return words or "undetermined"


# The clustered-data convention for small / medium / large intraclass
# correlations (Zyzanski et al. 2004, after Hox 2002). Optional, off by
# default: the design effect is what should be read.
ICC_LABEL_CUTOFFS: Tuple[float, float, float] = (0.05, 0.10, 0.15)
ICC_PERCENTILE_RANGE: Tuple[float, float] = (90.0, 99.0)


def icc_label(icc: float, cutoffs: Tuple[float, float, float] = ICC_LABEL_CUTOFFS) -> str:
    """'below small', 'small', 'medium' or 'large' by the cited convention."""
    small, medium, large = cutoffs
    if icc >= large:
        return "large"
    if icc >= medium:
        return "medium"
    if icc >= small:
        return "small"
    return "below small"


@functools.lru_cache(maxsize=256)
def icc_chance_level(sizes: Tuple[int, ...], percentile: float = 95.0) -> Optional[float]:
    """
    ICC a design with these group sizes reaches by chance alone, at the
    given percentile, in closed form.

    The one-way ANOVA estimator is (F - 1) / (F + n0 - 1) with F = MS_b /
    MS_w and n0 = (N - sum n_i^2 / N) / (k - 1); under a true ICC of zero
    and normal data F follows F(k - 1, N - k) exactly for balanced designs
    and approximately, with n0, for ragged ones (McGraw & Wong 1996; Donner
    1986). The estimator is monotone in F, so its percentile is that of F:
    q = (F_p(k - 1, N - k) - 1) / (F_p + n0 - 1). An observed ICC below it
    is not evidence of nesting.

    The sizes are the real ones, not their mean: an ROI holds six axons or
    more and a slide several ROIs, so the groups of a real design are
    ragged. ``icc_null_for_sizes`` (Monte Carlo) is kept as the cross-check.
    """
    groups_sizes = [int(v) for v in sizes if int(v) > 0]
    k = len(groups_sizes)
    n = sum(groups_sizes)
    if k < 2 or n <= k:
        return None
    n0 = (n - sum(v * v for v in groups_sizes) / n) / (k - 1)
    if n0 <= 0:
        return None
    f_q = float(f_distribution.ppf(float(percentile) / 100.0, k - 1, n - k))
    return max(0.0, (f_q - 1.0) / (f_q + n0 - 1.0))


@functools.lru_cache(maxsize=256)
def icc_null_for_sizes(
    sizes: Tuple[int, ...],
    percentile: float = 95.0,
    n_sim: int = 400,
    random_seed: int = 0,
) -> Optional[float]:
    """
    The same chance level by Monte Carlo: groups of exactly these sizes
    drawn from one normal distribution (true ICC zero), the requested
    percentile of the clamped estimates. Kept as the cross-check of
    ``icc_chance_level`` (its Monte Carlo sd at 400 draws is 0.01-0.03).
    """
    groups_sizes = [int(s) for s in sizes if int(s) > 0]
    if len(groups_sizes) < 2 or sum(groups_sizes) <= len(groups_sizes):
        return None
    rng = np.random.default_rng(random_seed)
    out = np.empty(n_sim)
    for i in range(n_sim):
        groups = [rng.normal(0.0, 1.0, s) for s in groups_sizes]
        d = _icc_core(groups)
        out[i] = np.nan if d is None else d
    out = out[np.isfinite(out)]
    return float(np.percentile(out, percentile)) if out.size else None


def icc_null_threshold(
    n_groups: int,
    mean_group_size: float,
    percentile: float = 95.0,
) -> Optional[float]:
    """
    The same for a balanced design of ``n_groups`` groups of
    ``mean_group_size``, in closed form: (F_0.95(k - 1, N - k) - 1) /
    (F_0.95 + m - 1). Few groups give a high chance level, so studies with
    three to five animals per genotype sit in the noisy regime.
    """
    k = int(n_groups)
    m = max(int(round(mean_group_size)), 2)
    if k < 2:
        return None
    return icc_chance_level(tuple([m] * k), percentile)


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
    percentile: float = 95.0, labels: bool = False,
) -> NestingDiagnostic:
    """
    One-way random-effects intraclass correlation: the share of total
    variance sitting BETWEEN groups rather than within them.

    ICC near 0 means axons within a group are no more alike than axons
    from different groups, so treating them as independent is roughly
    fair. ICC near 1 means they are near-duplicates and the effective
    sample size is far below the number of axons.

    The design effect ``1 + (m - 1) * ICC`` converts that into the number
    of independent observations the data is actually worth (Killip et al.
    2004): that number, not an adjective, is what to read. The chance
    level is the closed form of ``icc_chance_level`` at ``percentile``
    (documented range 90-99); ``labels`` adds the optional small / medium /
    large words of the clustered-data convention.

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
            warnings=["Degenerate variance; ICC undefined."],
            null_percentile=float(percentile), labels=bool(labels))

    icc = float(icc_val)
    deff = 1.0 + (m - 1.0) * icc
    eff_n = n_obs / deff if deff > 0 else None
    null_p95 = icc_chance_level(tuple(sorted(int(s) for s in sizes)), float(percentile))

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
            f"(null {percentile:g}th percentile {null_p95:.2f}). It is not "
            f"evidence of nesting."
        )
    elif deff >= 2.0 and eff_n is not None:
        warnings_.append(
            f"ICC = {icc:.2f}, design effect {deff:.1f}: treating the "
            f"{n_obs} {observation_label}s of {k} {group_label}s as "
            f"independent claims about {deff:.1f}x more evidence than the "
            f"data holds (worth about {eff_n:.1f})."
        )

    return NestingDiagnostic(
        parameter=parameter, icc=icc, design_effect=float(deff),
        effective_n=None if eff_n is None else float(eff_n),
        n_observations=n_obs, n_groups=k, mean_group_size=m,
        null_p95=null_p95, warnings=warnings_,
        null_percentile=float(percentile), labels=bool(labels),
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

    def describe(self) -> str:
        if self.mean is None:
            return f"{self.group}: n={self.n}"
        sd = "" if self.sd is None else f" sd={self.sd:.3g}"
        median = "" if self.median is None else f" median={self.median:.3g}"
        return f"{self.group}: n={self.n} mean={self.mean:.3g}{sd}{median}"


# The group every axon is in when the comparison is not split by anything.
ALL_AXONS = "all axons"

# What one unit of each identity field is called in a sentence ("per
# animal", "3 slides"): the labels of the identity panel are written for a
# form, "ROI (the measurement)", and read badly there.
UNIT_NOUNS: Dict[str, str] = {
    "genotype": "genotype", "protein": "protein", "animal": "animal",
    "sample": "slide", "roi_name": "ROI", "axon_name": "axon",
}


def unit_noun(name: str) -> str:
    """The noun for one unit of the identity field ``name``."""
    return UNIT_NOUNS.get(name, FIELD_LABELS.get(name, name).lower())


@dataclass
class GroupComparison:
    """One column compared between groups, at both nesting levels."""

    parameter: str
    group_by: Optional[str]
    nest_by: str
    per_axon: List[LevelSummary]
    per_unit: List[LevelSummary]
    # Within each group: do the axons of one nesting unit resemble each
    # other? Computed per group, never across them -- see compare_groups.
    nesting: Dict[str, NestingDiagnostic] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def unit_label(self) -> str:
        return unit_noun(self.nest_by)

    def describe(self) -> str:
        lines = [self.parameter]
        for label, level in (("per axon", self.per_axon),
                             (f"per {self.unit_label}", self.per_unit)):
            if not level:
                lines.append(f"  {label:<16} (no data)")
                continue
            lines.append(f"  {label:<16} "
                         + " | ".join(s.describe() for s in level))
        for group, n in self.nesting.items():
            if n.icc is None:
                continue
            chance = ("" if n.null_p95 is None
                      else f"; chance reaches {n.null_p95:.2f}")
            lines.append(
                f"  nesting in {group}: ICC={n.icc:.2f} ({n.severity}"
                f"{chance})")
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


def pooled_methods(records: Sequence[AxonRecord]) -> Dict[str, List[str]]:
    """The GUARD_COLUMNS that take more than one value among ``records``,
    with the values. Empty when every row was measured the same way."""
    out: Dict[str, List[str]] = {}
    for column in GUARD_COLUMNS:
        values: Set[str] = set()
        for record in records:
            cell = record.row.get(column)
            if cell is None or cell == "":
                continue
            values.add(str(cell))
        if len(values) > 1:
            out[column] = sorted(values)
    return out


def compare_groups(
    records: Sequence[AxonRecord],
    column: str,
    group_by: Optional[str] = "genotype",
    nest_by: str = "animal",
) -> GroupComparison:
    """
    Summarise one column between groups, at the axon level and again
    after collapsing to one value per nesting unit.

    Deliberately returns no p-value. The two levels routinely disagree,
    and choosing between them (or fitting a mixed model instead) is a
    design decision for the experimenter -- see the module docstring.

    The nesting diagnostic is computed WITHIN each group. Computed across
    groups, a real difference between KO and WT lands between the nesting
    units, inflates the ICC, and makes the diagnostic more alarming
    exactly when the experiment works.

    Parameters
    ----------
    column : a column of the axon table (see COMPARABLE_COLUMNS).
    group_by : an identity field defining the comparison, normally
        "genotype"; None for all the axons in one group.
    nest_by : the identity field of the nesting unit, normally "animal".
    """
    for name in ([group_by] if group_by else []) + [nest_by]:
        if name not in IDENTITY_FIELDS:
            raise ValueError(
                f"{name!r} is not one of the identity fields "
                f"({', '.join(IDENTITY_FIELDS)}).")
    if group_by is not None and group_by == nest_by:
        raise ValueError("The groups and the nesting unit are the same "
                         "field.")
    nest_label = unit_noun(nest_by)
    group_label = unit_noun(group_by) if group_by else ""
    warnings_: List[str] = []

    usable: List[AxonRecord] = []
    seen: Set[str] = set()
    repeated = 0
    for record in records:
        if not record.ok:
            continue
        key = record.axon_id or record.source
        if key in seen:
            repeated += 1
            continue
        seen.add(key)
        usable.append(record)
    if repeated:
        warnings_.append(
            f"{repeated} row(s) repeat an axon already counted (same "
            f"axon_id): each axon is counted once, from its first row.")
    # One file under two selections is either two axons of a whole-field
    # file or one axon measured twice -- a batch row beside the one
    # exported with an ROI. Only the person knows which.
    selections: Dict[str, Set[str]] = {}
    for record in usable:
        selections.setdefault(_base(record.source), set()).add(
            str(record.row.get("roi") or ""))
    twice = sorted(name for name, rois in selections.items()
                   if len(rois) > 1)
    if twice:
        warnings_.append(
            f"{len(twice)} file(s) appear under more than one selection "
            f"({', '.join(twice[:3])}{' ...' if len(twice) > 3 else ''}): "
            f"two axons of one field, or one axon measured twice. If the "
            f"second, it is counted twice here.")
    if not usable:
        return GroupComparison(column, group_by, nest_by, [], [], {},
                               ["No analysed axons."])

    methods = pooled_methods(usable)
    for name, values in methods.items():
        warnings_.append(
            f"These axons were not all measured the same way: {name} takes "
            f"{', '.join(values)}. A difference between groups can be a "
            f"difference in method.")

    def group_of(record: AxonRecord) -> Optional[str]:
        return ALL_AXONS if group_by is None else record.label(group_by)

    with_value = [r for r in usable if r.value(column) is not None]
    if len(with_value) < len(usable):
        warnings_.append(
            f"{len(usable) - len(with_value)} of {len(usable)} axons have no "
            f"value in {column}; they are left out.")
    missing_group = [r for r in with_value if group_of(r) is None]
    missing_unit = [r for r in with_value if r.label(nest_by) is None]
    if group_by is not None and missing_group:
        warnings_.append(
            f"{len(missing_group)} of {len(with_value)} axons have no "
            f"{group_label}; they are left out of the comparison. Fill it "
            f"in from the identity, or in the batch's table of files.")
    if missing_unit:
        if len(missing_unit) == len(with_value):
            warnings_.append(
                f"No axon has its {nest_label}, so there is one level only: "
                f"per axon. Any test over these axons counts the axons of "
                f"one {nest_label} as independent observations, and they "
                f"are not.")
        else:
            warnings_.append(
                f"{len(missing_unit)} of {len(with_value)} axons have no "
                f"{nest_label}; the per-{nest_label} level is built without "
                f"them.")

    # --- per axon ---
    by_group: Dict[str, List[float]] = {}
    for r in with_value:
        g = group_of(r)
        if g is None:
            continue
        by_group.setdefault(g, []).append(float(r.value(column) or 0.0))
    per_axon = [_summarise(g, vs) for g, vs in sorted(by_group.items())]

    # --- one value per nesting unit, within each group ---
    per_unit_values: Dict[Tuple[str, str], List[float]] = {}
    for r in with_value:
        g, u = group_of(r), r.label(nest_by)
        if g is None or u is None:
            continue
        per_unit_values.setdefault((g, u), []).append(
            float(r.value(column) or 0.0))
    collapsed: Dict[str, List[float]] = {}
    for (g, _u), vs in per_unit_values.items():
        collapsed.setdefault(g, []).append(float(np.mean(vs)))
    per_unit = [_summarise(g, vs) for g, vs in sorted(collapsed.items())]

    if per_unit:
        n_units = sum(s.n for s in per_unit)
        n_axons = sum(len(vs) for vs in per_unit_values.values())
        if n_units < n_axons:
            warnings_.append(
                f"{n_axons} axons are {n_units} {nest_label}(s). A test on "
                f"the {n_axons} axons treats them as independent, which "
                f"they are not.")

    # --- nesting, within each group ---
    nesting: Dict[str, NestingDiagnostic] = {}
    for g in sorted(collapsed):
        units = [vs for (gg, _u), vs in per_unit_values.items() if gg == g]
        if len(units) < 2:
            continue
        diag = intraclass_correlation(
            units, parameter=column, observation_label="axon",
            group_label=nest_label)
        nesting[g] = diag
        warnings_.extend(f"{g}: {w}" for w in diag.warnings)

    return GroupComparison(column, group_by, nest_by, per_axon, per_unit,
                           nesting, warnings_)


def batch_summary(records: Sequence[AxonRecord]) -> str:
    """Short human-readable account of what a batch produced."""
    total = len(records)
    ok = [r for r in records if r.ok]
    failed = total - len(ok)
    lines = [f"axons analysed : {len(ok)}/{total}"
             + (f"  ({failed} failed)" if failed else "")]
    for name in IDENTITY_FIELDS:
        values = {r.label(name) for r in ok} - {None}
        lines.append(f"{FIELD_LABELS[name].lower():<15}: "
                     + (f"{len(values)} ({', '.join(sorted(values)[:6])}"
                        f"{' ...' if len(values) > 6 else ''})"
                        if values else "not filled in"))
    if ok and not any(r.label("animal") for r in ok):
        lines.append(
            "  ! No axon has its animal. Without it the nesting cannot be "
            "accounted for, and a comparison between genotypes risks "
            "pseudoreplication.")
    guessed = [r for r in ok if str(r.row.get("pixel_size_source") or "")
               in ("override", "manual", "neighbour", "remembered",
                   "unknown")]
    if guessed:
        lines.append(
            f"  ! {len(guessed)} file(s) used a pixel size that was not "
            f"their own record; every lateral distance scales with it.")
    return "\n".join(lines)


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
            # SCI-6's columns after every column of before (no earlier column moves).
            from tools.mps_multisegment import SCI6_SEGMENT_COLUMNS, columns_last
            out.append(columns_last(merged, SCI6_SEGMENT_COLUMNS))
        return out

    def pair_rows(self) -> List[Dict[str, Any]]:
        """One row per consecutive segment pair: metadata, the axial
        boundary between them, and their gap/patch correlation."""
        if self.rings is None:
            return []
        out: List[Dict[str, Any]] = []
        from tools.mps_multisegment import SCI6_PAIR_COLUMNS, columns_last
        for pair in self.rings.pairs:
            row = self._meta_row()
            row.update(pair.export_dict())
            out.append(columns_last(row, SCI6_PAIR_COLUMNS))
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
