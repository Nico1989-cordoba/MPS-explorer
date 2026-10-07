# -*- coding: utf-8 -*-
"""
H6 batch runner of the H-ECL intra-axon column test (BUILD stage): do
betaII-spectrin clusters of consecutive MPS rings stack in columns more than
chance, restricted to the ring pairs whose axial separation lets the
question be asked?

Per axon (one picked-localization file each), exactly as the review window
(``tools/mps_columns_window.py``) does it, no Qt:

1. the file through ``power_columns.load_simnull_input`` (``mps_io.
   load_localizations``: the loader the GUI and the Batch button use);
2. ``build_rings`` with ``roi=None`` and the pre-registered parameters
   (``rings_params_from(default_columns_params())``), as ``prepare_review``;
3. D-39 ``z_quality`` of the built rings, the rule-v2 analysis of D-41
   (``tools.mps_axial_precision.viability_v2``: SiZer peak + valley, the
   localization minimum vs the central ring, the leak tiers; its numbers
   are always written) and the VIABILITY of every consecutive ring pair
   through a pluggable function (below; default rule v2);
4. the automatic lumen cleaning, REMOVE only (``observed_lumen_cleaning``:
   ``classify_lumen`` + ``LumenDecisions.apply``); with the widefield masks
   when ``<stem>_lumen_widefield.json`` exists (beside the file or in
   ``--widefield-dir``), else isolation only, and the table says which;
5. the arc test on the CENTROID membrane (the primary curve) of the cleaned
   rings, read on the VIABLE pairs only (the "primary" tier); the MARGINAL
   pairs are read as a separate "sensitivity" tier (viable + marginal);
   NOT VIABLE pairs are reported (z-quality metrics, verdict, reasons) and
   NOT analysed (their statistic cells are empty). An axon with no viable or
   marginal pair does not run the arc test at all.

Every p in the output is labelled UNCALIBRATED (columns ``*_uncalibrated``
and ``calibration``): the H5-E calibration of the simulated null was closed
as NOT ACCEPTED (D-41, Q-33; reopened when data with viable pairs exist), so
the p-values are Monte Carlo p-values against the plain rotation null. Inputs of an acquisition outside the
calibrated range (a site pattern: ``MPS_UNCALIBRATED_LABELS`` or ``uncalibrated_labels`` in
mps_analysis_settings.json; none by default) carry
``calibration_range_flags`` (``roi2_flag``, ``calibration_flags``). A
calibrated mode does not exist yet: ``ColumnBatchSettings.calibrated=True``
raises.

R8 (binding): the column statistic is NEVER run on REAL axons here. An input
counts as simulated only when it is an NPZ written by
``write_simulated_input`` (key ``simulated`` true); everything else is real
and the run is REFUSED (exit 4, before anything is read) unless
``--allow-real`` is given, in which case a reminder is printed and every row
says "exploratory". ``process_axon`` repeats the check, so a direct call
cannot skip it.

Pluggable viability
-------------------
``--viability module:callable`` (default ``batch_columns:v2_viability``, the
D-41 rule v2; ``batch_columns:d39_viability`` keeps the D-39 verdict)::

    viability(res: RingsResult, zq: ZQuality) -> ViabilityResult

(a callable with ``accepts_lab_xyz = True`` also receives ``xyz_lab_nm=(x, y,
z)``, the localizations' lab coordinates, and ``precomputed=`` the rule-v2
analysis the runner already made).

``res`` are the rings as ``build_rings`` returned them (before the lumen
cleaning; ring indices are the ones the cleaned rings keep), ``zq`` the D-39
analysis of those rings (``zq.boundaries``: one entry per consecutive pair,
with d_sep, valley depth, localizations... ``zq.rings``: n_locs per ring).
It returns ``ViabilityResult(rule, pairs)``: ``rule`` a text naming the rule
AND its version (written in every row and checked on restart), ``pairs`` one
``PairVerdict(ring_a, ring_b, verdict, reasons)`` per ``zq.boundaries`` pair,
``verdict`` one of "viable" / "marginal" / "not viable". Missing, extra or
unknown verdicts are an error (a rule must not skip a pair silently). The
D-39 numbers and the rule-v2 numbers are always written next to the verdict,
whatever the rule.

Output (``--out``)
------------------
``column_pairs.csv``   one row per consecutive ring pair
``column_axons.csv``   one row per axon (summary of the tiers, lumen counts,
                       z-quality, provenance)
Both are rebuilt from the journals at every exit, sorted by input order, with
no timing column, so an interrupted and resumed run ends with the bytes an
uninterrupted run writes. ``journal_pairs.csv`` / ``journal_axons.csv`` are
the append-only journals (flush + fsync per row group; a truncated last line
is dropped on start; the pair rows of an axon whose summary row is missing
are dropped and the axon re-runs), ``batch_meta.json`` the settings that
determine the numbers, with the viability function's source hash (a restart
with other ones is refused; so is a rule text that differs from the
journal's, or an input whose ``input_sha256`` changed),
``progress.json`` (rewritten atomically at least every 60 s) and
``errors.log``. Adapted from the H5-E calibration harness (Rows / progress /
wall limit; a private research validator), which is deliberately not imported.

Exploratory selections (D-43)
-----------------------------
``--selection LABEL`` (``tools.mps_selection``: e.g. ``v2[peak,valley,count]``,
``v2c-B[peak,valley,count,leak:tnfix]``, ``v2[...] AND v2c-B[...]``) selects
the pairs with the criteria of rule v2 and / or rule v2-clusters checked or
unchecked. A label equivalent to the pre-specified rule
(``v2[peak,valley,count,leak:D-39]``) is the default run, byte for byte.
Any other is an EXPLORATORY selection: ``viability_rule`` (meaning
unchanged: the rule that selected) carries "exploratory selection <label>
#<hash> | <rule v2 text>", the run writes ``selection.json`` and
``selection_pairs.csv`` (every criterion of every pair, journal
``journal_selection.csv``), ``batch_meta.json`` records the label and hash,
and a restart with another selection is refused. It cannot be combined with
a custom ``--viability`` (one rule source per run). ``--exploration-log
PATH`` appends one row per finished axon to the session's exploration log
(``tools.mps_selection.ExplorationLog``).

Exit codes: 0 done, 1 job error(s) or usage, 2 wall limit / ``--max-jobs``
reached (resumable), 4 R8 refusal.

Usage
-----
    venv\\Scripts\\python.exe batch_columns.py --out OUT --inputs SIM_DIR --workers 4
    venv\\Scripts\\python.exe batch_columns.py --out OUT --inputs a.npz b.npz --viability mymod:my_rule
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import dataclasses
import hashlib
import importlib
import json
import math
import os
import sys
import time
import traceback

# one BLAS / OpenMP thread per worker (the validators do the same)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT",
           "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

TABLE_VERSION_PAIRS = "column batch pairs v2"
TABLE_VERSION_AXONS = "column batch axons v2"
PROGRAM = "MPS Explorer (batch_columns.py, H6)"
V2_VIABILITY = "batch_columns:v2_viability"
D39_VIABILITY = "batch_columns:d39_viability"
DEFAULT_VIABILITY = V2_VIABILITY
MAX_WORKERS = 12
DEFAULT_WORKERS = 4
MAX_ERRORS = 20
PROGRESS_EVERY_S = 60.0
PRIOR_SECONDS = 20.0
SIM_KEY = "simulated"
SIM_PROVENANCE_KEY = "sim_provenance"
EXIT_DONE, EXIT_ERRORS, EXIT_RESUMABLE, EXIT_R8 = 0, 1, 2, 4

VERDICT_VIABLE = "viable"
VERDICT_MARGINAL = "marginal"
VERDICT_NOT_VIABLE = "not viable"
VERDICTS = (VERDICT_VIABLE, VERDICT_MARGINAL, VERDICT_NOT_VIABLE)
TIER_PRIMARY, TIER_SENSITIVITY, TIER_NONE = "primary", "sensitivity", "not analysed"
TIER_OF = {VERDICT_VIABLE: TIER_PRIMARY, VERDICT_MARGINAL: TIER_SENSITIVITY, VERDICT_NOT_VIABLE: TIER_NONE}

CALIBRATION_NOTE = "uncalibrated"
CALIBRATION_DETAIL = ("p-values are Monte Carlo p-values against the plain rotation null, NOT calibrated: the H5-E "
                      "calibration was closed as not accepted (D-41, Q-33)")
EXPLORATORY_REAL = ("EXPLORATORY: real axon run with --allow-real (R8); results are uncalibrated and are not evidence "
                    "about the hypothesis")
SIMULATED_NOTE = "simulated input (R8-safe); uncalibrated"
R8_REMINDER = ("R8 REMINDER: --allow-real was passed. The column statistic ran on REAL axons: every result is "
               "EXPLORATORY and UNCALIBRATED (H5-E not accepted). Do not read it as a test of the hypothesis.")

IDENTITY_FIELDS: Tuple[str, ...] = ("genotype", "protein", "animal", "sample", "roi_name", "axon_name")
HEAD_COLUMNS: Tuple[str, ...] = ("axon_id",) + IDENTITY_FIELDS + ("source", "roi", "program")
PROVENANCE_COLUMNS: Tuple[str, ...] = ("viability_rule", "lumen_rule_version", "arcc_recipe", "cluster_set_sha",
                                       "columns_params_sha", "git_head")

ZQ_METRIC_COLUMNS: Tuple[str, ...] = (
    "n_locs_a", "n_locs_b", "k_a_before", "k_b_before", "d_sep", "ashman_d", "valley_depth", "gap_nm", "f_lo", "f_hi",
    "misassign", "ambiguous_frac", "exp_spur_frac", "kl_nats", "n_req_events", "n_events", "info_ratio", "d39_verdict",
    "d39_reasons")
# D-41 rule v2, always written (whatever --viability says): peaks, valley, fractions of the central ring, f_min(x)
V2_PAIR_COLUMNS: Tuple[str, ...] = (
    "v2_peak_a", "v2_peak_b", "v2_valley", "v2_f_a", "v2_f_b", "v2_f_min", "v2_x", "v2_x_in_range", "v2_verdict",
    "v2_reasons")
V2_AXON_COLUMNS: Tuple[str, ...] = (
    "v2_rule", "v2_central_ring", "v2_in_focus_ring", "v2_x", "v2_f_min", "v2_x_in_range", "v2_x_note",
    "v2_pct_of_central", "v2_ring_peaks", "v2_groups_source", "v2_z_edges", "v2_n_viable", "v2_n_marginal",
    "v2_limiting_factors", "v2_warnings")
PAIR_STAT_COLUMNS: Tuple[str, ...] = (
    "n_matched_obs", "k_a_usable", "k_b_usable", "E_dir", "E_star", "zeta", "p_excess_uncalibrated",
    "p_deficit_uncalibrated", "p_two_sided_uncalibrated")

PAIR_COLUMNS: Tuple[str, ...] = (
    ("pair_key", "input_key", "table_version") + HEAD_COLUMNS
    + ("input_kind", "ring_a", "ring_b") + ZQ_METRIC_COLUMNS + V2_PAIR_COLUMNS
    + ("viability_verdict", "viability_reasons", "analysis_tier", "pair_status", "k_a_after", "k_b_after")
    + PAIR_STAT_COLUMNS
    + ("calibration", "calibration_flags", "roi2_flag", "n_null") + PROVENANCE_COLUMNS)

TIER_STAT_FIELDS: Tuple[str, ...] = ("pairs", "n_pairs", "T_obs", "null_mean", "z_A", "p_excess_uncalibrated",
                                     "p_two_sided_uncalibrated")
LUMEN_COLUMNS: Tuple[str, ...] = (
    "lumen_mode", "lumen_widefield_file", "lumen_widefield_sha256", "lumen_widefield", "lumen_interior_usable",
    "lumen_n_clusters",
    "lumen_n_removed_auto", "lumen_n_doubtful", "lumen_n_vetoed", "lumen_n_removed_final", "lumen_n_manual",
    "lumen_removed_keys")
AXON_COLUMNS: Tuple[str, ...] = (
    ("input_key", "table_version") + HEAD_COLUMNS
    + ("input_kind", "sim_provenance", "input_sha256", "exploratory_note", "analysis_status", "analysis_detail", "n_locs", "n_rings",
       "n_locs_per_ring", "k_per_ring_before", "k_per_ring_after")
    + LUMEN_COLUMNS
    + ("zq_axon_verdict", "zq_p_ref_nm", "zq_thresholds", "zq_n_pairs", "zq_n_viable", "zq_n_marginal",
       "zq_n_not_viable", "zq_min_d_sep", "zq_max_exp_spur_frac", "zq_warnings", "viability_pairs", "not_viable_pairs")
    + V2_AXON_COLUMNS
    + tuple(f"primary_{f}" for f in TIER_STAT_FIELDS)
    + tuple(f"sensitivity_{f}" for f in TIER_STAT_FIELDS)
    + ("arcc_membrane_length_nm", "arcc_knot_spacing_nm", "arcc_n_self_crossings", "n_null", "tau0_nm", "random_seed",
       "calibration", "calibration_detail", "calibration_flags", "roi2_flag", "warnings") + PROVENANCE_COLUMNS)

JOURNAL_PAIRS = "journal_pairs.csv"
JOURNAL_AXONS = "journal_axons.csv"
TABLE_PAIRS = "column_pairs.csv"
TABLE_AXONS = "column_axons.csv"
META_FILE = "batch_meta.json"
# D-43: written only under an exploratory selection
JOURNAL_SELECTION = "journal_selection.csv"
TABLE_SELECTION = "selection_pairs.csv"
SELECTION_FILE = "selection.json"
# what decides the numbers: a restart that changes one of these must start a fresh --out ("selection" and
# "selection_hash" are absent from a default run's meta, so None == None there and in folders made before D-43)
RESTART_KEYS: Tuple[str, ...] = ("table_version_pairs", "table_version_axons", "n_null", "viability_spec",
                                 "viability_source_sha", "widefield_dir", "patterns", "pixel_size_nm", "selection",
                                 "selection_hash")


# ============================================================================ viability (the plug-in point)

@dataclasses.dataclass(frozen=True)
class PairVerdict:
    """The viability verdict of one consecutive ring pair (``ring_a`` below ``ring_b``)."""

    ring_a: int
    ring_b: int
    verdict: str
    reasons: Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ViabilityResult:
    """What a viability function returns: ``rule`` names the rule and its version, ``pairs`` has one verdict per
    consecutive pair of ``zq.boundaries``."""

    rule: str
    pairs: Tuple[PairVerdict, ...]


D39_RULE = "D-39 first filter (tools.mps_axial_precision.z_quality, frozen thresholds)"


def d39_viability(res: Any, zq: Any) -> ViabilityResult:
    """The default viability rule: the D-39 verdict of ``z_quality`` as it is (viable / marginal / not viable)."""
    return ViabilityResult(rule=D39_RULE, pairs=tuple(
        PairVerdict(int(b.ring_a), int(b.ring_b), str(b.verdict), tuple(str(r) for r in b.reasons))
        for b in zq.boundaries))


def v2_viability(res: Any, zq: Any, *, xyz_lab_nm: Optional[Sequence[Any]] = None,
                 precomputed: Optional[Any] = None) -> ViabilityResult:
    """The default viability rule: the D-41 rule v2 (``tools.mps_axial_precision.viability_v2``; SiZer peak and
    valley with 3D DBSCAN groups on the LAB coordinates, localizations >= f_min(x) of the central ring, D-39 leak
    tiers). ``precomputed`` is that analysis when the caller already made it (same res, zq, coordinates)."""
    from tools.mps_axial_precision import viability_v2
    v = precomputed if precomputed is not None else viability_v2(res, zq, xyz_lab_nm=xyz_lab_nm)
    return ViabilityResult(rule=str(v.rule), pairs=tuple(
        PairVerdict(int(p.ring_a), int(p.ring_b), str(p.verdict), tuple(str(r) for r in p.reasons)) for p in v.pairs))


v2_viability.accepts_lab_xyz = True  # type: ignore[attr-defined]
BUILTIN_VIABILITY: Dict[str, Callable[..., Any]] = {V2_VIABILITY: v2_viability, D39_VIABILITY: d39_viability}


def resolve_viability(spec: str) -> Callable[..., Any]:
    """``"module:callable"`` -> the function (the two built-in specs return ``v2_viability`` / ``d39_viability``
    without importing again)."""
    if spec in BUILTIN_VIABILITY:
        return BUILTIN_VIABILITY[spec]
    module, sep, name = str(spec).partition(":")
    if not sep or not module or not name:
        raise ValueError(f"--viability must be 'module:callable', got {spec!r}")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise ValueError(f"--viability {spec!r} is not callable")
    return fn  # type: ignore[no-any-return]


def viability_source_sha(spec: str) -> str:
    """sha256 (12 hex) of the viability function's source: a rule edited without bumping its version text still
    changes it, so a restart on an existing --out is refused ("unavailable" when the source cannot be read)."""
    import inspect
    try:
        src = inspect.getsource(resolve_viability(spec))
        if spec == V2_VIABILITY:
            # the rule lives in mps_axial_precision / mps_sizer: their code is what decides the verdicts
            from tools import mps_axial_precision as ap
            from tools import mps_sizer
            src += "".join(inspect.getsource(f) for f in (ap.viability_v2, ap.verdict_v2, ap.f_min_of_x,
                                                          ap.central_ring_position, ap.z_quality))
            src += inspect.getsource(mps_sizer) + repr((ap.VIABILITY_RULE_V2, ap.FMIN_INTERCEPT, ap.FMIN_SLOPE_PER_LN_X,
                                                         ap.FMIN_X_RANGE, ap.FMIN_CAP, ap.V2_PEAK_WINDOW_OVER_P,
                                                         ap.V2_SPUR_VIABLE, ap.V2_SPUR_MARGINAL))
    except (OSError, TypeError):
        return "unavailable"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_viability(result: Any, zq: Any) -> Tuple[str, Dict[Tuple[int, int], Tuple[str, str]]]:
    """Validate a viability result against ``zq.boundaries`` (duck-typed): (rule, {(a, b): (verdict, reasons text)}).
    ValueError on a missing or extra pair, a repeated pair or a verdict outside ``VERDICTS``."""
    rule = str(getattr(result, "rule", "")).strip()
    if not rule:
        raise ValueError("the viability result carries no rule name/version")
    wanted = [(int(b.ring_a), int(b.ring_b)) for b in zq.boundaries]
    out: Dict[Tuple[int, int], Tuple[str, str]] = {}
    for pv in getattr(result, "pairs", ()):
        key = (int(pv.ring_a), int(pv.ring_b))
        if key in out:
            raise ValueError(f"viability rule {rule!r}: pair {key} given twice")
        if str(pv.verdict) not in VERDICTS:
            raise ValueError(f"viability rule {rule!r}: verdict {pv.verdict!r} of pair {key} is not one of {VERDICTS}")
        out[key] = (str(pv.verdict), "; ".join(str(r) for r in pv.reasons))
    if set(out) != set(wanted):
        raise ValueError(f"viability rule {rule!r} must give exactly one verdict per consecutive pair {wanted}; "
                         f"it gave {sorted(out)}")
    return rule, out


# ============================================================================ inputs and the R8 guard

class RealInputRefused(RuntimeError):
    """A real (not simulated) input without ``allow_real`` (R8)."""


def input_kind(path: str) -> Tuple[str, str]:
    """("simulated", provenance text) for an NPZ written by ``write_simulated_input``, else ("real", ""). Fail closed:
    an unreadable or unmarked file is real."""
    if str(path).lower().endswith(".npz"):
        try:
            with np.load(path, allow_pickle=False) as store:
                if SIM_KEY in store.files and bool(np.asarray(store[SIM_KEY]).reshape(-1)[0]):
                    prov = str(np.asarray(store[SIM_PROVENANCE_KEY]).reshape(-1)[0]) if SIM_PROVENANCE_KEY in store.files else ""
                    return "simulated", prov
        except Exception:  # noqa: BLE001 - fail closed
            return "real", ""
    return "real", ""


def write_simulated_input(path: str, contour: str = "circle", seed: int = 1, s: float = 0.3) -> str:
    """Simulate one axon (``tools.mps_sim_harness``: the synthetic leak configuration of the case ``contour``, the
    axial leak scaled by ``s``, the pick background) and save it as the NPZ the runner reads, marked simulated.
    Returns the path written."""
    from tools import mps_sim_harness as vl
    from tools.mps_simulate_axon import scale_axial_leak, simulate_axon
    spec = vl.case_spec(contour)
    cfg = scale_axial_leak(vl.sim_config(spec["contour"], spec["overrides"], spec["n_rings"], True, spec["box"]), float(s))
    axon = simulate_axon(cfg, int(seed))
    keep = vl.pick_keep(cfg, axon, spec["pick"])
    prov = json.dumps(dict(contour=contour, seed=int(seed), leak_scale=float(s)), sort_keys=True)
    np.savez(path, x=np.array(axon.x_nm, float)[keep], y=np.array(axon.y_nm, float)[keep], z=np.array(axon.z_nm, float)[keep],
             frame=np.array(axon.frame, np.int64)[keep], lp=np.array(axon.lp_lateral_nm, float)[keep],
             lpz=np.array(axon.lpz_nm, float)[keep], n_frames=np.array(int(cfg.n_frames)), simulated=np.array(True),
             sim_provenance=np.array(prov))
    return path if str(path).lower().endswith(".npz") else str(path) + ".npz"


def input_key(path: str) -> str:
    return hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:12]


def collect_inputs(paths: Sequence[str]) -> List[str]:
    """The input files, absolute, sorted, without repeats: a file as given; a folder through
    ``mps_io.find_localization_files`` plus its ``*.npz`` files."""
    found: List[str] = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            from tools import mps_io
            files, _skipped = mps_io.find_localization_files(p)
            found.extend(os.path.abspath(f) for f in files)
            for dirpath, _dirs, names in os.walk(p):
                found.extend(os.path.join(dirpath, n) for n in names if n.lower().endswith(".npz"))
        elif os.path.isfile(p):
            found.append(p)
        else:
            raise FileNotFoundError(f"input not found: {p}")
    return sorted(set(found), key=os.path.normcase)


def real_inputs(paths: Sequence[str]) -> List[str]:
    return [p for p in paths if input_kind(p)[0] == "real"]


# ============================================================================ the axon (worker side)

@dataclasses.dataclass
class ColumnBatchSettings:
    """What every axon of a run is processed with (picklable; the determinants are in ``batch_meta.json``)."""

    n_null: Optional[int] = None          # None: the pre-registered null size (columns_params.yaml)
    viability_spec: str = DEFAULT_VIABILITY
    allow_real: bool = False
    widefield_dir: Optional[str] = None
    patterns: Dict[str, str] = dataclasses.field(default_factory=dict)
    pixel_size_nm: Optional[float] = None
    git_head: str = ""
    calibrated: bool = False
    # D-43: an exploratory selection's canonical label ("" = the pre-specified rule, the --viability function)
    selection: str = ""

    def selection_spec(self) -> Optional[Any]:
        """The ``SelectionSpec`` of ``selection`` (None for the default run)."""
        if not self.selection:
            return None
        from tools.mps_selection import parse_selection
        return parse_selection(self.selection)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if bool(v) else "False"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "" if math.isnan(f) else repr(f)
    return str(v)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pair_text(pairs: Sequence[Tuple[int, int]]) -> str:
    return " ".join(f"{a}-{b}" for a, b in pairs)


def find_widefield(path: str, widefield_dir: Optional[str]) -> Optional[str]:
    """``<stem>_lumen_widefield.json`` in ``widefield_dir`` or beside the file (None when there is none)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    name = f"{stem}_lumen_widefield.json"
    for folder in ([widefield_dir] if widefield_dir else []) + [os.path.dirname(os.path.abspath(path))]:
        cand = os.path.join(folder, name)
        if os.path.isfile(cand):
            return cand
    return None


def restricted_joint(arc: Any, pairs: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    """The arc result's JOINT null restricted to ``pairs`` (the observed matched count and every null draw's count
    summed over those pairs): T_obs, the null mean, z_A (``power_columns.z_over_pairs``: the same standardisation as
    the joint test) and the Phipson-Smyth p-values of the sum. All NaN / empty when no pair is in the joint."""
    import power_columns as pc
    from tools.mps_matching import _phipson_smyth
    nan = float("nan")
    out: Dict[str, Any] = dict(pairs="", n_pairs=0, T_obs=None, null_mean=nan, z_A=nan, p_excess_uncalibrated=nan,
                               p_two_sided_uncalibrated=nan, p_deficit_uncalibrated=nan)
    joint = getattr(arc, "joint", None)
    if joint is None or not pairs:
        return out
    have: List[Tuple[int, int]] = [(int(p[0]), int(p[1])) for p in joint.pairs]
    wanted = {(int(a), int(b)) for a, b in pairs}
    sel = [q for q, p in enumerate(have) if p in wanted]
    if not sel:
        return out
    t_obs = int(np.asarray(joint.n_matched_obs)[sel].sum())
    t_null = np.asarray(joint.null_n_matched)[:, sel].sum(axis=1)
    p_ex, p_def, p_two = _phipson_smyth(t_null, t_obs)
    out.update(pairs=_pair_text([have[q] for q in sel]), n_pairs=len(sel), T_obs=t_obs, null_mean=float(t_null.mean()),
               z_A=float(pc.z_over_pairs(arc, [have[q] for q in sel])), p_excess_uncalibrated=float(p_ex),
               p_deficit_uncalibrated=float(p_def), p_two_sided_uncalibrated=float(p_two))
    return out


def _load_input(path: str, kind: str, settings: ColumnBatchSettings) -> Any:
    import power_columns as pc
    if path.lower().endswith(".npz"):
        return pc.load_simnull_input(None, path)
    return pc.load_simnull_input(path, None, pixel_size_nm=settings.pixel_size_nm)


def _identity_of(path: str, kind: str, settings: ColumnBatchSettings) -> Dict[str, Optional[str]]:
    """The identity columns as the exports carry them: proposed from the path with the user's patterns for a real
    file (nothing invented), empty for a simulated one."""
    from tools.mps_identity import AxonIdentity, propose
    ident = AxonIdentity() if kind == "simulated" else propose(path, patterns=settings.patterns or None).identity
    return dict(ident.columns())


def process_axon(path: str, settings: ColumnBatchSettings) -> Dict[str, Any]:
    """One axon -> ``{"pairs": [pair rows], "axon": summary row}`` (dicts keyed by ``PAIR_COLUMNS`` /
    ``AXON_COLUMNS``). Deterministic: nothing here depends on time, process or order. ``RealInputRefused`` for a real
    input without ``settings.allow_real``; a data failure that is a result (no rings, a cleaning that cannot apply, an
    arc test that cannot be fitted) is a row with its status, any other exception propagates."""
    import power_columns as pc
    from tools.mps_axial_precision import calibration_range_flags, z_quality
    from tools.mps_columns import build_rings, rings_params_from, RingsParams
    from tools.mps_identity import axon_id
    from tools.mps_lumen import default_columns_params
    from tools.mps_unroll import CENTROID_MEMBRANE_RECIPE, UnrollParams, analyze_arc_columns

    if settings.calibrated:
        raise NotImplementedError("calibrated mode does not exist: the H5-E calibration is not accepted yet")
    kind, sim_prov = input_kind(path)
    if kind == "real" and not settings.allow_real:
        raise RealInputRefused(f"{path}: a real axon; R8 forbids running the column statistic on it (pass --allow-real "
                               "for an exploratory run)")
    viability = resolve_viability(settings.viability_spec)
    cp = default_columns_params()
    n_null = int(cp.n_null if settings.n_null is None else settings.n_null)
    inp = _load_input(path, kind, settings)
    key = input_key(path)
    aid = axon_id(os.path.basename(path), "")
    head: Dict[str, Any] = {"axon_id": aid}
    head.update(_identity_of(path, kind, settings))
    head.update(source=path, roi="", program=PROGRAM)
    with open(pc.PARAMS_YAML, "rb") as fh:
        params_sha = hashlib.sha256(fh.read()).hexdigest()[:12]
    prov: Dict[str, Any] = dict(viability_rule="", lumen_rule_version="", arcc_recipe=CENTROID_MEMBRANE_RECIPE,
                                cluster_set_sha="", columns_params_sha=params_sha, git_head=settings.git_head)
    note = EXPLORATORY_REAL if kind == "real" else SIMULATED_NOTE
    axon: Dict[str, Any] = {"input_key": key, "table_version": TABLE_VERSION_AXONS, **head, "input_kind": kind,
                            "sim_provenance": sim_prov, "input_sha256": file_sha256(path), "exploratory_note": note, "n_locs": int(inp.x_nm.size),
                            "n_null": n_null, "tau0_nm": float(cp.tau0_nm), "random_seed": int(cp.random_seed),
                            "calibration": CALIBRATION_NOTE, "calibration_detail": CALIBRATION_DETAIL, **prov}
    warnings_: List[str] = []

    selection_out: List[Dict[str, Any]] = []

    def done(status: str, detail: str = "", pairs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        axon["analysis_status"] = status
        axon["analysis_detail"] = detail
        axon["warnings"] = " | ".join(dict.fromkeys(warnings_))
        out: Dict[str, Any] = {"pairs": pairs or [], "axon": axon}
        if settings.selection:
            out["selection_pairs"] = selection_out
        return out

    try:
        rp = rings_params_from(cp)
    except (OSError, ValueError) as exc:
        rp = RingsParams()
        warnings_.append(f"the pre-registered column parameters could not be applied ({exc}); RingsParams defaults used")
    res = build_rings(inp.x_nm.copy(), inp.y_nm.copy(), inp.z_nm.copy(), frame=inp.frame.copy(),
                      lp_lateral_nm=inp.lp_lateral_nm.copy(), lpz_nm=inp.lpz_nm.copy(), params=rp, source_name=str(path),
                      pixel_size_nm=inp.pixel_size_nm, pixel_size_source=str(inp.pixel_size_source), n_frames=inp.n_frames,
                      roi=None)
    rings = sorted(res.rings, key=lambda r: int(r.index))
    n_clusters = sum(len(r.clusters) for r in rings)
    axon.update(n_rings=len(rings), k_per_ring_before="/".join(str(len(r.clusters)) for r in rings),
                n_locs_per_ring="/".join(str(int(np.asarray(r.loc_index).size)) for r in rings))
    warnings_.extend(f"rings: {w}" for w in res.warnings[:5])
    if not rings or n_clusters == 0:
        return done("no ring with clusters", "; ".join(str(w) for w in res.warnings[:3]))

    zq = z_quality(res)
    flags = calibration_range_flags(f"{inp.name} {inp.source}", zq)
    from tools.mps_axial_precision import ROI2_FLAG
    axon.update(calibration_flags=" | ".join(flags), roi2_flag=bool(ROI2_FLAG in flags),
                zq_axon_verdict=str(zq.axon_verdict), zq_p_ref_nm=float(zq.p_ref_nm),
                zq_thresholds=json.dumps(dataclasses.asdict(zq.thresholds), sort_keys=True), zq_n_pairs=len(zq.boundaries))
    warnings_.extend(f"z_quality: {w}" for w in zq.warnings)
    # D-41 rule v2 on the rings as built, with the 3D groups on the lab coordinates: always written
    from tools.mps_axial_precision import viability_v2
    xyz_lab = (np.asarray(inp.x_nm, dtype=np.float64), np.asarray(inp.y_nm, dtype=np.float64),
               np.asarray(inp.z_nm, dtype=np.float64))
    v2 = viability_v2(res, zq, xyz_lab_nm=xyz_lab)
    axon.update(_v2_axon_values(v2))
    v2_of = {(int(p.ring_a), int(p.ring_b)): p for p in v2.pairs}
    sel_spec = settings.selection_spec()
    selection_rows: List[Dict[str, Any]] = []
    if sel_spec is not None:
        # D-43: an exploratory selection over the criteria of rule v2 (the analysis above) and rule v2c
        verdicts, selection_rows = _selection_verdicts(sel_spec, res, v2, xyz_lab, aid, key)
    elif getattr(viability, "accepts_lab_xyz", False):
        verdicts = viability(res, zq, xyz_lab_nm=xyz_lab, precomputed=v2 if viability is v2_viability else None)
    else:
        verdicts = viability(res, zq)
    rule, verdict_of = check_viability(verdicts, zq)
    selection_out.extend(selection_rows)
    axon["viability_rule"] = rule
    prov["viability_rule"] = rule
    d_all = [float(b.d_sep) for b in zq.boundaries if math.isfinite(float(b.d_sep))]
    sp_all = [float(b.exp_spur_frac) for b in zq.boundaries if math.isfinite(float(b.exp_spur_frac))]
    n_v = sum(1 for v, _r in verdict_of.values() if v == VERDICT_VIABLE)
    n_m = sum(1 for v, _r in verdict_of.values() if v == VERDICT_MARGINAL)
    n_n = sum(1 for v, _r in verdict_of.values() if v == VERDICT_NOT_VIABLE)
    axon.update(zq_n_viable=n_v, zq_n_marginal=n_m, zq_n_not_viable=n_n, zq_min_d_sep=min(d_all) if d_all else None,
                zq_max_exp_spur_frac=max(sp_all) if sp_all else None, zq_warnings=" | ".join(str(w) for w in zq.warnings),
                viability_pairs=" ".join(f"{a}-{b}:{verdict_of[(a, b)][0]}" for a, b in sorted(verdict_of)),
                not_viable_pairs=_pair_text([p for p in sorted(verdict_of) if verdict_of[p][0] == VERDICT_NOT_VIABLE]))
    ring_q = {int(q.index): q for q in zq.rings}

    # the lumen cleaning (REMOVE only), with the widefield masks when the file is there
    wf_path = find_widefield(path, settings.widefield_dir)
    axon["lumen_widefield_sha256"] = file_sha256(wf_path) if wf_path else ""
    try:
        lum = pc.observed_lumen_cleaning(res, inp, widefield_path=wf_path)
    except ValueError as exc:
        axon.update(lumen_mode="failed", lumen_widefield_file=wf_path or "")
        return done("lumen cleaning failed", str(exc),
                    _pair_rows(head, kind, zq, verdict_of, {}, f"lumen cleaning failed: {exc}", flags, n_null, prov, key,
                               v2_of=v2_of))
    lv = pc.lumen_row_values(lum.classification, lum.decisions)
    c, d = lum.classification, lum.decisions
    prov.update(lumen_rule_version=str(c.rule_version), cluster_set_sha=str(d.cluster_set_sha))
    axon.update(lumen_mode=("widefield + isolation" if c.has_widefield else "isolation only (no widefield masks)"),
                lumen_widefield_file=wf_path or "", lumen_widefield=bool(lv["lumen_widefield"]),
                lumen_interior_usable=lv["lumen_interior_usable"], lumen_n_clusters=lv["lumen_n_clusters"],
                lumen_n_removed_auto=lv["lumen_n_removed_auto"], lumen_n_doubtful=lv["lumen_n_doubtful"],
                lumen_n_vetoed=lv["lumen_n_vetoed"], lumen_n_removed_final=lv["lumen_n_removed_final"],
                lumen_n_manual=lv["lumen_n_manual"], lumen_removed_keys=" ".join(d.final_removed_keys()),
                lumen_rule_version=prov["lumen_rule_version"], cluster_set_sha=prov["cluster_set_sha"])
    warnings_.extend(f"lumen: {w}" for w in lum.warnings)
    cleaned = lum.cleaned
    axon["k_per_ring_after"] = "/".join(str(len(r.clusters)) for r in sorted(cleaned.rings, key=lambda r: int(r.index)))

    if len(rings) < 2:
        return done("fewer than two rings", "", _pair_rows(head, kind, zq, verdict_of, {}, None, flags, n_null, prov, key,
                                                           v2_of=v2_of))
    primary = [p for p in sorted(verdict_of) if verdict_of[p][0] == VERDICT_VIABLE]
    marginal = [p for p in sorted(verdict_of) if verdict_of[p][0] == VERDICT_MARGINAL]
    if not primary and not marginal:
        return done("no analysable pair", "every consecutive pair is NOT VIABLE under the rule: the arc test was not run",
                    _pair_rows(head, kind, zq, verdict_of, {}, None, flags, n_null, prov, key, v2_of=v2_of))

    cols_stub: Any = SimpleNamespace(include_suspect=True, tau0_nm=float(cp.tau0_nm), random_seed=int(cp.random_seed))
    try:
        arc = analyze_arc_columns(cleaned, cols_stub, params=UnrollParams(n_null=n_null),
                                  reference_curve=pc.CENTROID_REFERENCE_CURVE)
    except Exception as exc:  # noqa: BLE001 - a membrane that cannot be fitted is a result of this axon
        if isinstance(exc, (ValueError, np.linalg.LinAlgError)):
            return done("arc test failed", f"{type(exc).__name__}: {exc}",
                        _pair_rows(head, kind, zq, verdict_of, {}, f"arc test failed: {exc}", flags, n_null, prov, key,
                                   v2_of=v2_of))
        raise
    warnings_.extend(f"arc test: {w}" for w in (arc.warnings or []))
    cm = getattr(arc, "centroid_membrane", None)
    axon.update(arcc_membrane_length_nm=_num(arc.length_nm), arcc_knot_spacing_nm=_num(arc.membrane_knot_spacing_nm),
                arcc_n_self_crossings=None if cm is None else int(cm.n_self_crossings))
    by_pair = {(int(m.ring_a), int(m.ring_b)): m for m in arc.adjacent}
    tier_p = restricted_joint(arc, primary)
    tier_s = restricted_joint(arc, primary + marginal) if marginal else None
    for name, tier in (("primary", tier_p), ("sensitivity", tier_s)):
        if tier is not None:
            for f in TIER_STAT_FIELDS:
                axon[f"{name}_{f}"] = tier[f]
    after = {int(r.index): len(r.clusters) for r in cleaned.rings}
    pairs = _pair_rows(head, kind, zq, verdict_of, by_pair, None, flags, n_null, prov, key, after=after, ring_q=ring_q,
                       v2_of=v2_of)
    if primary and tier_p["n_pairs"] == len(primary):
        status, detail = "ok", ""
    elif primary:
        status, detail = "ok (some viable pairs not in the arc test)", "a viable pair lost a ring or its contour in the cleaning"
    elif tier_s is not None and tier_s["n_pairs"]:
        status, detail = "ok (sensitivity only)", "no viable pair: only the marginal tier was analysed"
    else:
        status, detail = "arc test without the selected pairs", "the selected pairs are not in the arc test"
    return done(status, detail, pairs)


def _pair_rows(head: Mapping[str, Any], kind: str, zq: Any, verdict_of: Mapping[Tuple[int, int], Tuple[str, str]],
               by_pair: Mapping[Tuple[int, int], Any], failure: Optional[str], flags: Sequence[str], n_null: int,
               prov: Mapping[str, Any], key: str, *, after: Optional[Mapping[int, int]] = None,
               ring_q: Optional[Mapping[int, Any]] = None,
               v2_of: Optional[Mapping[Tuple[int, int], Any]] = None) -> List[Dict[str, Any]]:
    """The rows of the pairs table: the z-quality numbers (D-39 and rule v2) and the verdict of every pair, and the arc
    statistics of the ones the tiers analysed (primary: viable; sensitivity: marginal)."""
    from tools.mps_axial_precision import ROI2_FLAG
    rq = ring_q if ring_q is not None else {int(q.index): q for q in zq.rings}
    rows: List[Dict[str, Any]] = []
    for b in zq.boundaries:
        a, c = int(b.ring_a), int(b.ring_b)
        verdict, reasons = verdict_of[(a, c)]
        tier = TIER_OF[verdict]
        p2 = (v2_of or {}).get((a, c))
        v2_cells: Dict[str, Any] = {} if p2 is None else {
            "v2_peak_a": bool(p2.peak_a), "v2_peak_b": bool(p2.peak_b), "v2_valley": bool(p2.valley),
            "v2_f_a": _num(p2.f_a), "v2_f_b": _num(p2.f_b), "v2_f_min": _num(p2.f_min), "v2_x": _num(p2.x),
            "v2_x_in_range": bool(p2.x_in_range), "v2_verdict": str(p2.verdict), "v2_reasons": "; ".join(p2.reasons)}
        row: Dict[str, Any] = {"pair_key": f"{key}|{a}-{c}", "input_key": key, "table_version": TABLE_VERSION_PAIRS, **head,
                               "input_kind": kind, "ring_a": a, "ring_b": c,
                               "n_locs_a": int(rq[a].n_locs), "n_locs_b": int(rq[c].n_locs), "k_a_before": int(b.k_a),
                               "k_b_before": int(b.k_b), "d_sep": b.d_sep, "ashman_d": b.ashman_d,
                               "valley_depth": b.valley_depth, "gap_nm": b.gap_nm, "f_lo": b.f_lo, "f_hi": b.f_hi,
                               "misassign": b.misassign, "ambiguous_frac": b.ambiguous_frac,
                               "exp_spur_frac": b.exp_spur_frac, "kl_nats": b.kl_nats, "n_req_events": b.n_req_events,
                               "n_events": b.n_events, "info_ratio": b.info_ratio, "d39_verdict": b.verdict,
                               "d39_reasons": "; ".join(str(r) for r in b.reasons), "viability_verdict": verdict,
                               "viability_reasons": reasons, "analysis_tier": tier,
                               "k_a_after": None if after is None else after.get(a),
                               "k_b_after": None if after is None else after.get(c),
                               "calibration": CALIBRATION_NOTE, "calibration_flags": " | ".join(flags),
                               "roi2_flag": bool(ROI2_FLAG in flags), "n_null": n_null, **dict(prov), **v2_cells}
        if tier == TIER_NONE:
            row["pair_status"] = "not analysed (NOT VIABLE under the rule)"
        elif failure is not None:
            row["pair_status"] = failure
        elif (a, c) in by_pair:
            m = by_pair[(a, c)]
            row.update(pair_status="ok", n_matched_obs=int(m.n_matched), k_a_usable=int(m.K_a), k_b_usable=int(m.K_b),
                       E_dir=_num(m.E_dir), E_star=_num(m.E_star), zeta=_num(m.zeta),
                       p_excess_uncalibrated=_num(m.p_excess), p_deficit_uncalibrated=_num(m.p_deficit),
                       p_two_sided_uncalibrated=_num(m.p_two_sided))
        else:
            row["pair_status"] = "pair not in the arc test (a ring lost its contour or its clusters in the cleaning)"
        rows.append(row)
    return rows


def _selection_verdicts(spec: Any, res: Any, v2: Any, xyz_lab: Sequence[Any], aid: str,
                        key: str) -> Tuple[ViabilityResult, List[Dict[str, Any]]]:
    """D-43: the verdicts of an exploratory selection (``tools.mps_selection``) from the rule-v2 analysis already
    made and, when the selection reads it, rule v2c on the same rings (a v2c failure is recorded and every pair that
    needs it is NOT VIABLE, never a silent fall-back to v2); plus the rows of ``selection_pairs.csv``."""
    from tools.mps_selection import compute_axon_criteria, evaluate_selection, selection_pair_rows
    crit = compute_axon_criteria(res, xyz_lab, v2=v2, with_v2c=bool(spec.needs_v2c), axon_id=aid)
    result = evaluate_selection(crit, spec)
    verdicts = ViabilityResult(rule=result.rule_text, pairs=tuple(
        PairVerdict(int(p.ring_a), int(p.ring_b), str(p.verdict), tuple(str(r) for r in p.reasons))
        for p in result.pairs))
    return verdicts, selection_pair_rows(crit, result, input_key=key)


def _v2_axon_values(v2: Any) -> Dict[str, Any]:
    """The v2_* cells of the axons table from a ``viability_v2`` result (D-41)."""
    return {
        "v2_rule": str(v2.rule), "v2_central_ring": v2.central_index, "v2_in_focus_ring": v2.in_focus_index,
        "v2_x": _num(v2.x), "v2_f_min": _num(v2.f_min), "v2_x_in_range": bool(v2.x_in_range), "v2_x_note": str(v2.x_note),
        "v2_pct_of_central": "/".join("" if not math.isfinite(r.f_of_central) else f"{100 * r.f_of_central:.1f}"
                                      for r in v2.rings),
        "v2_ring_peaks": "/".join("1" if r.peak else "0" for r in v2.rings),
        "v2_groups_source": str(v2.groups_source),
        "v2_z_edges": " ".join(f"{e.side}:{e.z_nm:.0f}:{e.edge_frac:.3f}:{'cut' if e.truncated else 'tail'}"
                               for e in v2.z_edges),
        "v2_n_viable": len(v2.viable_pairs), "v2_n_marginal": len(v2.marginal_pairs),
        "v2_limiting_factors": " | ".join(f"{f.key}={f.status}: {f.value}" for f in v2.limiting_factors),
        "v2_warnings": " | ".join(str(w) for w in v2.warnings)}


def run_unit(job: Dict[str, Any]) -> Dict[str, Any]:
    """The pool's task: one axon -> its rows (or the traceback; nothing escapes)."""
    t = time.perf_counter()
    try:
        out = process_axon(job["path"], ColumnBatchSettings(**job["settings"]))
    except Exception:  # noqa: BLE001 - reported by the parent
        return dict(job=job, error=traceback.format_exc())
    res = dict(job=job, pairs=out["pairs"], axon=out["axon"], wall_s=time.perf_counter() - t)
    if "selection_pairs" in out:
        res["selection_pairs"] = out["selection_pairs"]
    return res


# ============================================================================ parent side: journals, tables, schedule

class Rows:
    """An append-only CSV journal: fixed header, one complete row per key, flush + fsync per group, a truncated last
    line dropped on start (adapted from the H5-E calibration harness's ``Rows``)."""

    def __init__(self, path: str, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = tuple(columns)
        self.rows: Dict[str, Dict[str, str]] = {}
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", newline="") as fh:
                text = fh.read()
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                lines = lines[:-1]                        # truncated last line: dropped, the axon re-runs
            reader = csv.reader(lines)
            header = next(reader, None)
            if header is not None and tuple(header) != self.columns:
                raise SystemExit(f"{path}: header differs from this runner's; use a fresh --out")
            for rec in reader:
                if len(rec) == len(self.columns):
                    self.rows[rec[0]] = dict(zip(self.columns, rec))
            self._rewrite()
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow(self.columns)

    def _rewrite(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.columns)
            for r in self.rows.values():
                w.writerow([r[c] for c in self.columns])
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def keep_only(self, predicate: Callable[[Dict[str, str]], bool]) -> int:
        """Drop the rows ``predicate`` rejects (rewrites the file); the number dropped."""
        keep = {k: r for k, r in self.rows.items() if predicate(r)}
        n = len(self.rows) - len(keep)
        if n:
            self.rows = keep
            self._rewrite()
        return n

    def append_many(self, rows: Sequence[Mapping[str, Any]]) -> None:
        recs = [[_fmt(r.get(c)) for c in self.columns] for r in rows]
        if not recs:
            return
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            for rec in recs:
                w.writerow(rec)
            fh.flush()
            os.fsync(fh.fileno())
        for rec in recs:
            self.rows[rec[0]] = dict(zip(self.columns, rec))


def _atomic_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def _write_csv(path: str, columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_tables(out: str, axons: Rows, pairs: Rows, order: Sequence[str],
                 selection: Optional[Rows] = None) -> Tuple[str, str]:
    """``column_pairs.csv`` and ``column_axons.csv`` from the journals, in input order (pairs by ring), atomically
    (and ``selection_pairs.csv`` from ``selection``, the journal of an exploratory selection)."""
    rank = {k: i for i, k in enumerate(order)}
    ax = sorted(axons.rows.values(), key=lambda r: rank.get(r["input_key"], len(rank)))
    pr = sorted(pairs.rows.values(), key=lambda r: (rank.get(r["input_key"], len(rank)), int(r["ring_a"]), int(r["ring_b"])))
    pp, pa = os.path.join(out, TABLE_PAIRS), os.path.join(out, TABLE_AXONS)
    _write_csv(pp, PAIR_COLUMNS, pr)
    _write_csv(pa, AXON_COLUMNS, ax)
    if selection is not None:
        sr = sorted(selection.rows.values(),
                    key=lambda r: (rank.get(r["input_key"], len(rank)), int(r["ring_a"]), int(r["ring_b"])))
        _write_csv(os.path.join(out, TABLE_SELECTION), selection.columns, sr)
    return pp, pa


def _meta_of(settings: ColumnBatchSettings, n_null_effective: int) -> Dict[str, Any]:
    meta = dict(table_version_pairs=TABLE_VERSION_PAIRS, table_version_axons=TABLE_VERSION_AXONS, n_null=int(n_null_effective),
                viability_spec=settings.viability_spec, viability_source_sha=viability_source_sha(settings.viability_spec),
                widefield_dir=None if not settings.widefield_dir else os.path.abspath(settings.widefield_dir),
                patterns=dict(settings.patterns), pixel_size_nm=settings.pixel_size_nm, git_head=settings.git_head)
    spec = settings.selection_spec()
    if spec is not None:
        # D-43: only under an exploratory selection (a default run's meta stays byte-identical)
        from tools.mps_selection import selection_source_sha
        meta.update(viability_source_sha=selection_source_sha(str(meta["viability_source_sha"])), selection=spec.label,
                    selection_hash=spec.hash)
    return meta


def _warning_texts(spec: Any) -> List[str]:
    from tools.mps_selection import selector_warnings
    return [f"{level}: {text}" for level, text in selector_warnings(spec)]


def check_meta(out: str, meta: Dict[str, Any]) -> None:
    """Write ``batch_meta.json`` on the first start; refuse a restart whose result-determining settings differ."""
    path = os.path.join(out, META_FILE)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        for k in RESTART_KEYS:
            if old.get(k) != meta.get(k):
                raise SystemExit(f"{path}: this folder was started with {k} = {old.get(k)!r}, this run has {meta.get(k)!r}; "
                                 "use a fresh --out")
        if old.get("git_head") != meta.get("git_head"):
            print(f"warning: this folder was started at git HEAD {old.get('git_head')}, now {meta.get('git_head')}; "
                  "the rows keep the HEAD they were computed with")
        return
    _atomic_json(path, meta)


class Runner:
    def __init__(self, out: str, inputs: Sequence[str], settings: ColumnBatchSettings, *, workers: int, wall_hours: float,
                 max_jobs: Optional[int] = None) -> None:
        self.out = os.path.abspath(out)
        os.makedirs(self.out, exist_ok=True)
        self.inputs = list(inputs)
        self.keys = [input_key(p) for p in self.inputs]
        if len(set(self.keys)) != len(self.keys):
            raise SystemExit("two inputs have the same path key")
        self.settings = settings
        self.workers, self.wall_s, self.max_jobs = int(workers), float(wall_hours) * 3600.0, max_jobs
        if settings.calibrated:
            raise NotImplementedError("calibrated mode does not exist: the H5-E calibration is not accepted yet")
        spec = settings.selection_spec()
        if spec is not None and settings.viability_spec != DEFAULT_VIABILITY:
            raise SystemExit(f"--selection and --viability {settings.viability_spec}: one rule source per run")
        if spec is not None:
            settings = dataclasses.replace(settings, selection=spec.label)   # canonical
            self.settings = settings
        elif settings.selection:
            settings = dataclasses.replace(settings, selection="")           # a default-equivalent label
            self.settings = settings
        self.selection_spec = spec
        from tools.mps_lumen import default_columns_params
        n_null = int(default_columns_params().n_null if settings.n_null is None else settings.n_null)
        meta = _meta_of(settings, n_null)
        check_meta(self.out, meta)
        self.axons = Rows(os.path.join(self.out, JOURNAL_AXONS), AXON_COLUMNS)
        self.pairs = Rows(os.path.join(self.out, JOURNAL_PAIRS), PAIR_COLUMNS)
        self.selection_rows: Optional[Rows] = None
        if spec is not None:
            from tools.mps_selection import SELECTION_PAIR_COLUMNS
            self.selection_rows = Rows(os.path.join(self.out, JOURNAL_SELECTION), SELECTION_PAIR_COLUMNS)
            sel_json = os.path.join(self.out, SELECTION_FILE)
            if not os.path.isfile(sel_json):
                _atomic_json(sel_json, dict(spec.to_json(), selection_source_sha=meta["viability_source_sha"],
                                            git_head=settings.git_head, warnings=_warning_texts(spec)))
        self.exploration_log: Optional[Any] = None
        # final review F3: a log that cannot be written (locked, another header) never stops the run; said loudly
        self.log_errors: List[str] = []
        # an axon is finished when its summary row exists: the pair rows of any other axon are leftovers of a crash
        finished = set(self.axons.rows)
        self.dropped_pairs = self.pairs.keep_only(lambda r: r["input_key"] in finished)
        if self.selection_rows is not None:
            self.selection_rows.keep_only(lambda r: r["input_key"] in finished)
        # one rule per --out (its text names rule and version) and rows only of the files as they are now
        rules = {r["viability_rule"] for r in self.axons.rows.values() if r["viability_rule"]}
        if len(rules) > 1:
            raise SystemExit(f"{self.out}: the journal holds rows of {len(rules)} viability rules {sorted(rules)}; "
                             "use a fresh --out")
        self.rule = next(iter(rules), "")
        for k, p in zip(self.keys, self.inputs):
            row = self.axons.rows.get(k)
            if row is not None and row.get("input_sha256") != file_sha256(p):
                raise SystemExit(f"{p}: the file changed since its row in {self.out} was computed; use a fresh --out")
        self.errors = 0
        self.failed: Set[str] = set()
        self.walls: List[float] = []
        self.t0 = time.perf_counter()
        self.skipped = len([k for k in self.keys if k in finished])

    def pending(self) -> List[Dict[str, Any]]:
        sdict = dataclasses.asdict(self.settings)
        return [dict(key=k, path=p, settings=sdict) for k, p in zip(self.keys, self.inputs)
                if k not in self.axons.rows and k not in self.failed]

    def expected(self) -> float:
        return float(np.mean(self.walls[-20:])) if self.walls else PRIOR_SECONDS

    def progress(self, in_flight: int, status: str) -> None:
        todo = len([k for k in self.keys if k not in self.axons.rows])
        _atomic_json(os.path.join(self.out, "progress.json"), dict(
            status=status, axons_total=len(self.keys), axons_done=len(self.keys) - todo, axons_skipped_finished=self.skipped,
            pair_rows=len(self.pairs.rows), errors=self.errors, dropped_unfinished_pair_rows=self.dropped_pairs,
            wall_elapsed_h=(time.perf_counter() - self.t0) / 3600.0,
            eta_h=todo * self.expected() / max(1, self.workers) / 3600.0, in_flight=in_flight,
            calibration=CALIBRATION_NOTE, allow_real=self.settings.allow_real,
            reminder=R8_REMINDER if self.settings.allow_real else "", updated=time.strftime("%Y-%m-%d %H:%M:%S")))

    def accept(self, res: Dict[str, Any]) -> bool:
        """Journal one finished unit (its pair rows, then the summary row last); False for an error."""
        job = res["job"]
        if "error" in res:
            self.errors += 1
            self.failed.add(job["key"])
            with open(os.path.join(self.out, "errors.log"), "a", encoding="utf-8") as fh:
                fh.write(f"==== {time.strftime('%Y-%m-%d %H:%M:%S')} {job['path']}\n{res['error']}\n")
            return False
        rule = str(res["axon"].get("viability_rule") or "")
        if rule and self.rule and rule != self.rule:
            raise SystemExit(f"{job['path']}: viability rule {rule!r} differs from {self.rule!r} of the rows already in "
                             f"{self.out} (the rule changed during or between runs); use a fresh --out")
        self.rule = rule or self.rule
        if self.selection_rows is not None:
            self.selection_rows.append_many(res.get("selection_pairs") or [])
        self.pairs.append_many(res["pairs"])
        self.axons.append_many([res["axon"]])
        self.walls.append(float(res.get("wall_s") or 0.0))
        if self.exploration_log is not None:
            try:
                self._log_axon(res)
            except Exception as exc:  # noqa: BLE001 - the log counts the selections tried; it never stops the batch
                self.log_errors.append(f"{type(exc).__name__}: {exc}")
                if len(self.log_errors) == 1:
                    print(f"warning: the exploration log could not be written ({self.log_errors[0]}): the selections "
                          "tried in this batch are NOT counted", file=sys.stderr, flush=True)
        return True

    def _log_axon(self, res: Dict[str, Any]) -> None:
        """D-43: one exploration-log row for a finished axon (its selection, tiers and p values)."""
        from tools.mps_selection import DEFAULT_SELECTION, log_row, pair_p_text
        ax = res["axon"]
        spec = self.selection_spec or DEFAULT_SELECTION
        pairs = sorted(res["pairs"], key=lambda r: (int(r["ring_a"]), int(r["ring_b"])))
        viable = [(int(r["ring_a"]), int(r["ring_b"])) for r in pairs if r.get("viability_verdict") == VERDICT_VIABLE]
        marginal = [(int(r["ring_a"]), int(r["ring_b"])) for r in pairs if r.get("viability_verdict") == VERDICT_MARGINAL]
        tier = {name: {f: ax.get(f"{name}_{f}") for f in TIER_STAT_FIELDS} for name in ("primary", "sensitivity")}
        pp = pair_p_text((int(r["ring_a"]), int(r["ring_b"]), r.get("p_excess_uncalibrated")) for r in pairs
                         if r.get("p_excess_uncalibrated") is not None)
        lumen = f"{ax.get('cluster_set_sha', '')}:{ax.get('lumen_removed_keys', '')}"
        assert self.exploration_log is not None
        self.exploration_log.append(log_row(
            source="batch", spec=spec, axon_id=str(ax.get("axon_id") or ""), input_sha256=str(ax.get("input_sha256") or ""),
            lumen=lumen, viable=viable, marginal=marginal, primary=tier["primary"], sensitivity=tier["sensitivity"],
            pair_p=pp, n_null=ax.get("n_null"), git_head=self.settings.git_head))

    def run(self) -> int:
        code = EXIT_DONE
        jobs = self.pending()
        issued = 0
        last = 0.0
        stop = False
        # fresh worker processes now and then (Python >= 3.11; the typeshed of older mypy lacks the argument)
        pool = (cf.ProcessPoolExecutor(max_workers=self.workers, max_tasks_per_child=25)  # type: ignore[call-overload,unused-ignore]
                if self.workers > 1 else None)
        inflight: Dict[Any, Dict[str, Any]] = {}
        try:
            while True:
                while jobs and not stop and (pool is None or len(inflight) < 2 * self.workers):
                    if time.perf_counter() - self.t0 + self.expected() > self.wall_s or (
                            self.max_jobs is not None and issued >= self.max_jobs):
                        stop, code = True, EXIT_RESUMABLE
                        break
                    job = jobs.pop(0)
                    issued += 1
                    if pool is None:
                        self.accept(run_unit(job))
                        if time.perf_counter() - last >= PROGRESS_EVERY_S:
                            self.progress(0, "running")
                            last = time.perf_counter()
                        if self.errors > MAX_ERRORS:
                            self.progress(0, "aborted: too many errors")
                            return EXIT_ERRORS
                    else:
                        inflight[pool.submit(run_unit, job)] = job
                if not inflight:
                    break
                done, _ = cf.wait(list(inflight), timeout=30.0, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    job = inflight.pop(fut)
                    try:
                        res = fut.result()
                    except Exception:  # noqa: BLE001 - a dead worker
                        res = dict(job=job, error=traceback.format_exc())
                    self.accept(res)
                    if self.errors > MAX_ERRORS:
                        self.progress(len(inflight), "aborted: too many errors")
                        pool.shutdown(wait=False, cancel_futures=True)
                        return EXIT_ERRORS
                if time.perf_counter() - last >= PROGRESS_EVERY_S:
                    self.progress(len(inflight), "running")
                    last = time.perf_counter()
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        if self.errors:
            code = EXIT_ERRORS
        self.progress(0, {EXIT_DONE: "done", EXIT_RESUMABLE: "wall limit / max jobs: resumable",
                          EXIT_ERRORS: "errors"}[code])
        return code


# ============================================================================ command line

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--out", required=True, help="output folder (outside the repo)")
    p.add_argument("--inputs", nargs="*", default=[], help="picked-localization files or folders (simulated NPZ by default)")
    p.add_argument("--list", default=None, help="text file with one input path per line")
    p.add_argument("--n-null", type=int, default=None, help="null size (default: the pre-registered 1999)")
    p.add_argument("--viability", default=DEFAULT_VIABILITY,
                   help=f"module:callable of the viability rule (default {V2_VIABILITY}, the D-41 rule v2; "
                        f"{D39_VIABILITY} for the D-39 verdict)")
    p.add_argument("--widefield-dir", default=None, help="folder with <stem>_lumen_widefield.json files (else beside the input)")
    p.add_argument("--patterns", default=None, help="JSON file {identity field: regex} for the identity of real files")
    p.add_argument("--pixel-size", type=float, default=None, help="pixel size (nm) for a Picasso file without metadata")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"processes (default {DEFAULT_WORKERS}, maximum {MAX_WORKERS}; "
                                                                      "1 runs in-process)")
    p.add_argument("--wall-hours", type=float, default=8.0, help="no new axon once elapsed + expected time exceeds it (exit 2)")
    p.add_argument("--max-jobs", type=int, default=None, help="stop issuing after this many axons (restart tests; exit 2)")
    p.add_argument("--allow-real", action="store_true", help="R8: process REAL axons (exploratory, uncalibrated)")
    p.add_argument("--selection", default=None,
                   help="D-43: an EXPLORATORY selection of the pairs, e.g. 'v2[peak,valley,count]' or "
                        "'v2[peak,valley,count,leak:D-39] AND v2c-B[peak,valley,count,leak:tnfix]' (tools.mps_selection); "
                        "a label equal to the pre-specified rule v2[peak,valley,count,leak:D-39] is the default run")
    p.add_argument("--exploration-log", default=None,
                   help="D-43: append one row per finished axon to this exploration log (selection_exploration_log.csv)")
    return p.parse_args(list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.workers < 1 or args.workers > MAX_WORKERS:
        print(f"--workers must be 1..{MAX_WORKERS}", file=sys.stderr)
        return EXIT_ERRORS
    paths = list(args.inputs)
    if args.list:
        with open(args.list, "r", encoding="utf-8") as fh:
            paths.extend(ln.strip() for ln in fh if ln.strip())
    if not paths:
        print("no inputs: pass --inputs and/or --list", file=sys.stderr)
        return EXIT_ERRORS
    try:
        inputs = collect_inputs(paths)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_ERRORS
    if not inputs:
        print("no input files found", file=sys.stderr)
        return EXIT_ERRORS
    real = real_inputs(inputs)
    if real and not args.allow_real:
        print(f"R8: {len(real)} of {len(inputs)} input(s) are REAL axons (not NPZ marked simulated), e.g. {real[0]}\n"
              "The column statistic is never run on real axons without --allow-real; refused, nothing was read.",
              file=sys.stderr)
        return EXIT_R8
    if args.allow_real and real:
        print(R8_REMINDER)
    patterns: Dict[str, str] = {}
    if args.patterns:
        with open(args.patterns, "r", encoding="utf-8") as fh:
            patterns = {str(k): str(v) for k, v in json.load(fh).items()}
    import power_columns as pc
    selection = ""
    if args.selection is not None:
        from tools.mps_selection import parse_selection
        try:
            sel = parse_selection(args.selection)
        except ValueError as exc:
            print(f"--selection: {exc}", file=sys.stderr)
            return EXIT_ERRORS
        if sel is not None and args.viability != DEFAULT_VIABILITY:
            print(f"--selection and --viability {args.viability}: one rule source per run", file=sys.stderr)
            return EXIT_ERRORS
        selection = "" if sel is None else sel.label
    settings = ColumnBatchSettings(n_null=args.n_null, viability_spec=args.viability, allow_real=bool(args.allow_real),
                                   widefield_dir=args.widefield_dir, patterns=patterns, pixel_size_nm=args.pixel_size,
                                   git_head=pc.git_commit(), selection=selection)
    runner = Runner(args.out, inputs, settings, workers=args.workers, wall_hours=args.wall_hours, max_jobs=args.max_jobs)
    if args.exploration_log:
        from tools.mps_selection import ExplorationLog
        runner.exploration_log = ExplorationLog(args.exploration_log)
    print(f"batch_columns: out {runner.out}, {len(inputs)} axon(s) ({runner.skipped} already finished), "
          f"{args.workers} worker(s), wall {args.wall_hours} h, all p-values {CALIBRATION_NOTE}")
    if runner.selection_spec is not None:
        print(f"batch_columns: {runner.selection_spec.display()}")
        for line in _warning_texts(runner.selection_spec):
            print(f"  {line}")
    code = runner.run()
    tp, ta = write_tables(runner.out, runner.axons, runner.pairs, runner.keys, runner.selection_rows)
    print(f"batch_columns: {len(runner.axons.rows)}/{len(inputs)} axon(s), {len(runner.pairs.rows)} pair row(s), "
          f"{runner.errors} error(s), exit {code}\n  {ta}\n  {tp}")
    if runner.selection_spec is not None:
        print(f"batch_columns: {runner.selection_spec.display()} -- every p is UNCALIBRATED and not corrected for trying "
              "several selections")
    if runner.exploration_log is not None and runner.log_errors:
        print(f"batch_columns: exploration log {runner.exploration_log.path} NOT written for {len(runner.log_errors)} "
              f"axon(s) ({runner.log_errors[0]}): the selections tried are NOT counted - p values are not corrected "
              "for trying several selections")
    elif runner.exploration_log is not None:
        from tools.mps_selection import counter_text
        tried = {r.get("selection_hash") for r in runner.exploration_log.rows()
                 if r.get("session_id") == runner.exploration_log.session_id and r.get("source") == "batch"}
        print(f"batch_columns: exploration log {runner.exploration_log.path}: "
              + counter_text(len(tried)).replace("on this axon", "in batch runs"))
    if args.allow_real and real:
        print(R8_REMINDER)
    return code


if __name__ == "__main__":
    sys.exit(main())
