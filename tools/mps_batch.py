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
(``edge_reference``), which keeps other clusters than an ROI does -- on
the April axon 7, 90 against 94. That axon then has a different
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

On the 18 April test axons grouped by ROI -- two ROIs of nine -- the
intraclass correlation is 0.86 for occupancy, 0.96 for the median
cluster area and 0.44 for the median 1NN, where two groups of nine reach
0.31 by chance alone (95th percentile). With two groups that says the two
ROIs differ, and not how much evidence an axon is worth: an ICC measured
over two groups is a comparison of those two groups. How much the axons of
one animal resemble each other needs several animals, which the test data
does not have. ``validate_batch.py`` recomputes these numbers every run
(``ICC_ON_THE_TEST_AXONS``).

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

from tools.mps_identity import FIELD_LABELS, AxonIdentity
from tools.mps_identity import FIELDS as IDENTITY_FIELDS
from tools.mps_randomization import DEFAULT_N_RANDOMIZATIONS

# What validate_batch.py measured on the 18 April axons, grouped by ROI
# (two ROIs of nine), on 2026-09-22. Kept here so that the numbers in the
# docstring are ones someone can check, and rechecked by that script on
# every run.
ICC_ON_THE_TEST_AXONS: Dict[str, float] = {
    "occupancy_percent": 0.86,
    "median_area_nm2": 0.96,
    "median_1nn_nm": 0.44,
}


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
# Never ks_pvalue: it is a within-axon p-value (are this axon's clusters
# spaced unlike random ones?), and a p-value is not a quantity whose
# difference between genotypes means anything. The KS statistic itself is
# a size of effect and can be compared.
COMPARABLE_COLUMNS: Dict[str, str] = {
    "occupancy_percent": "Occupancy of the perimeter (%)",
    "median_1nn_nm": "Median distance to the nearest cluster (nm)",
    "clusters_per_um": "Clusters per um of perimeter",
    "n_clusters_kept": "Clusters kept",
    "perimeter_um": "Perimeter (um)",
    "median_area_nm2": "Median cluster area (nm^2)",
    "median_r_eff_nm": "Median cluster effective radius (nm)",
    "delta_z_mean_nm": "Axial period, mean Delta-Z (nm)",
    "ks_statistic": "KS statistic, clusters against randomized ones",
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

    @classmethod
    def from_settings(cls, settings: Any) -> "BatchSettings":
        return cls(eps_nm=float(settings.eps_nm),
                   min_samples=int(settings.min_samples),
                   slab_half_width_nm=float(settings.slab_half_width_nm),
                   dbcv_threshold=float(settings.dbcv_threshold),
                   mahalanobis_threshold=float(
                       settings.mahalanobis_threshold))

    def describe(self) -> str:
        return (f"DBSCAN eps {self.eps_nm:g} nm, min samples "
                f"{self.min_samples}; axial slab +/-"
                f"{self.slab_half_width_nm:g} nm around the main peak, "
                f"found per axon; Mahalanobis threshold "
                f"{self.mahalanobis_threshold:g}; the clusters compared "
                f"with {DEFAULT_N_RANDOMIZATIONS} randomized placements, as "
                f"in the axon window")


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


def analyse_file(item: BatchFile, settings: BatchSettings) -> AxonRecord:
    """
    Load one file, analyse it as the axon window would, and build its row.

    Never raises: a file that cannot be read or analysed comes back with
    the reason, so a batch reports it instead of losing it.
    """
    from tools import mps_io
    from tools.axon_export import axon_row
    from tools.mps_analysis import analyze_axon

    name = os.path.basename(item.path)
    if item.unreadable:
        return AxonRecord(source=item.path, identity=item.identity,
                          error=item.unreadable)
    if item.needs_pixel_size and item.pixel_size_nm is None:
        return AxonRecord(
            source=item.path, identity=item.identity,
            error=f"{name} carries no pixel size and none was given, so it "
                  f"was not read")
    try:
        loc = mps_io.load_localizations(
            item.path,
            pixel_size_nm=(item.pixel_size_nm if item.needs_pixel_size
                           else None))
        if item.needs_pixel_size:
            loc.pixel_size_source = item.pixel_size_source or "manual"
    except Exception as error:                        # noqa: BLE001
        return AxonRecord(source=item.path, identity=item.identity,
                          error=f"could not be read: {error}")
    try:
        analysis = analyze_axon(
            loc.x_nm, loc.y_nm, loc.z_nm, source_name=item.path,
            pixel_size_nm=loc.pixel_size_nm,
            pixel_size_source=loc.pixel_size_source,
            eps_nm=settings.eps_nm, min_samples=settings.min_samples,
            slab_half_width_nm=settings.slab_half_width_nm,
            dbcv_threshold=settings.dbcv_threshold,
            mahalanobis_threshold=settings.mahalanobis_threshold)
        row = axon_row(analysis, identity=item.identity)
    except Exception as error:                        # noqa: BLE001
        return AxonRecord(source=item.path, identity=item.identity,
                          error=f"could not be analysed: {error}")
    return AxonRecord(source=item.path, identity=item.identity, row=row)


def run_batch(
    items: Sequence[BatchFile],
    settings: BatchSettings,
    *,
    on_done: Optional[Callable[[int, AxonRecord], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> List[AxonRecord]:
    """
    Analyse ``items`` in order, one record each.

    ``on_done(index, record)`` is called after each file, and
    ``cancelled()`` before each: when it returns True the batch stops
    there and returns what it has. A file already started is finished.
    """
    records: List[AxonRecord] = []
    for index, item in enumerate(items):
        if cancelled is not None and cancelled():
            break
        record = analyse_file(item, settings)
        records.append(record)
        if on_done is not None:
            on_done(index, record)
    return records


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
    out = log_path(table_path)
    with open(out, "w", encoding="utf-8", newline="") as handle:
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
def icc_null_for_sizes(
    sizes: Tuple[int, ...],
    percentile: float = 95.0,
    n_sim: int = 400,
    random_seed: int = 0,
) -> Optional[float]:
    """
    ICC a design with these group sizes reaches by chance alone, at the
    given percentile.

    Simulates groups of exactly these sizes drawn from one distribution --
    true ICC zero -- and returns the requested percentile of the
    estimates. An observed ICC below it is not evidence of nesting.

    The sizes are the real ones, not their mean: an ROI holds six axons or
    more and a slide several ROIs, so the groups of a real design are
    ragged, and a threshold simulated for a balanced design is the
    threshold of a design that does not exist.
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
    n_sim: int = 400,
    random_seed: int = 0,
) -> Optional[float]:
    """
    The same for a balanced design of ``n_groups`` groups of
    ``mean_group_size``: simulated at 400 draws, the 95th percentile is
    0.31 for two groups of nine, 0.13 for six groups of nine, and 0.02 for
    six groups of fifty. Studies with three to five animals per genotype
    sit squarely in the noisy regime.
    """
    k = int(n_groups)
    m = max(int(round(mean_group_size)), 2)
    if k < 2:
        return None
    return icc_null_for_sizes(tuple([m] * k), percentile, n_sim, random_seed)


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
    null_p95 = icc_null_for_sizes(tuple(sorted(int(s) for s in sizes)))

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
                      else f", chance reaches {n.null_p95:.2f}")
            lines.append(
                f"  nesting in {group}: ICC={n.icc:.2f} ({n.severity}"
                f"{chance}), {n.n_observations} axons worth about "
                f"{n.effective_n:.1f}")
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
