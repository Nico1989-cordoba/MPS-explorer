# -*- coding: utf-8 -*-
"""
Channel 2's own DBSCAN parameters, derived from its measured precision and its dye.

The rule (decided by the user's research group on 2026-10-14; public methods page ``docs/channel2_parameters.md``)
-----------------------------------------------------------------------------------------------------------------
DBSCAN's radius and minimum count are density parameters of ONE dataset (Ester et al. 1996, section 4.2), so a second
channel is never clustered with channel 1's numbers. What is kept equal between the channels is the criterion -- one
labelled molecule with its expected number of localizations forms a cluster -- and the numbers that implement it are
computed from each channel's own measurements:

    eps_2         = eps_1 x NeNA_2 / NeNA_1
    min_samples_2 = round(1 + (n_loc,mol - 1) x (1 - exp(-eps_2^2 / (4 NeNA_2^2))))

* NeNA (Endesfelder et al. 2014) is the precision, measured on each channel's WHOLE-FIELD, UNLINKED localization
  file: a picked axon rarely holds the ``MIN_NENA_PAIRS`` next-frame pairs the fit needs, and a linked file has
  merged exactly the consecutive-frame pairs NeNA measures.
* Two localizations of one emitter differ by a Rayleigh-distributed distance of scale sqrt(2) x sigma (the law NeNA
  fits), so the chance that they fall within eps of each other is 1 - exp(-eps^2 / (4 sigma^2)). DBSCAN's core
  condition counts the localizations within eps of a localization, itself included; for a molecule that leaves
  n_loc,mol localizations that count is 1 + (n_loc,mol - 1) x that chance.
* n_loc,mol comes from the dye table (Dempsey et al. 2011, Table 1: mean switching cycles per molecule, by buffer).
  Those cycles are a LOWER bound of the localizations per labelled molecule (a label can carry several dyes, an
  event can span several frames) until the label calibration exists.
* eps_1 is channel 1's radius (25 nm in Gazal et al. 2026); NeNA_1 is measured the same way. When channel 1's NeNA
  cannot be measured, the published 12 nm (Gazal et al. 2026, NeNA) is used for the ratio, and the origin says so.
* With eps 25 nm, NeNA 12 nm and Alexa Fluor 647 in MEA (14 cycles) the formula gives 9.6 -> 10: Gazal's MinPoints.

The values are PRE-FILLED and editable; each carries its origin, and an edited one becomes "user". When the precision
cannot be measured (too few pairs, a linked file, no whole-field file) the value stays empty with the reason: nothing
is invented, and channel 1's numbers are never copied. The ratio of the two channels' sampling densities is a warning
only (it contains the abundance of the partner protein) and never scales min_samples.

No Qt here: the dialog (``tools.channel2_ui``) and the tests use the same functions.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "GAZAL_EPS_NM", "GAZAL_NENA_NM", "GAZAL_MIN_SAMPLES", "DYE_TABLE_SOURCE", "BUFFERS", "BUFFER_WORDS",
    "DYE_CYCLES", "dyes", "cycles", "pair_within_probability", "derive_eps2", "expected_core_count",
    "derive_min_samples", "ChannelPrecision", "is_linked", "precision_from_file", "find_whole_field",
    "Channel2Derivation", "derive_channel2", "provenance", "reconcile", "EXPORT_KEYS", "export_fields",
    "ORIGIN_DERIVED", "ORIGIN_USER", "DOC_PATH", "DOC_URL",
]

# --------------------------------------------------------------------------- published reference values
# Gazal et al. 2026 (bioRxiv 10.64898/2026.02.19.705149; eLife RP 10.7554/eLife.111108.1): "the DBSCAN searching
# radius epsilon was set to 25 nm"; "the global localization precision (12nm) estimated by the NeNA algorithm";
# "MinPoints parameter was set to a value of 10, following the mean number of switching cycles".
GAZAL_EPS_NM = 25.0
GAZAL_NENA_NM = 12.0
GAZAL_MIN_SAMPLES = 10

# --------------------------------------------------------------------------- the dye table
DYE_TABLE_SOURCE = ("Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011) Nat Methods 8:1027-1036, Table 1: "
                    "mean number of switching cycles per molecule")
BUFFERS: Tuple[str, ...] = ("MEA", "bME")
BUFFER_WORDS: Dict[str, str] = {"MEA": "MEA", "bME": "βME"}
# Dempsey et al. 2011, Table 1, "# Switching Cycles (Mean)", MEA / betaME columns, every row of the table (read from
# the Europe PMC full text, PMC3272503). Mean switching cycles per molecule: a lower bound of n_loc,mol.
DYE_CYCLES: Dict[str, Dict[str, float]] = {
    # blue-absorbing
    "Atto 488": {"MEA": 11, "bME": 49},
    "Alexa Fluor 488": {"MEA": 16, "bME": 139},
    "Atto 520": {"MEA": 9, "bME": 17},
    "Fluorescein": {"MEA": 4, "bME": 15},
    "FITC": {"MEA": 17, "bME": 16},
    "Cy2": {"MEA": 0.4, "bME": 0.7},
    # yellow-absorbing
    "Cy3B": {"MEA": 8, "bME": 5},
    "Alexa Fluor 568": {"MEA": 7, "bME": 52},
    "TAMRA": {"MEA": 10, "bME": 59},
    "Cy3": {"MEA": 0.5, "bME": 1.6},
    "Cy3.5": {"MEA": 5.7, "bME": 3.3},
    "Atto 565": {"MEA": 4, "bME": 5},
    # red-absorbing
    "Alexa Fluor 647": {"MEA": 14, "bME": 26},
    "Cy5": {"MEA": 10, "bME": 17},
    "Atto 647": {"MEA": 10, "bME": 24},
    "Atto 647N": {"MEA": 9, "bME": 39},
    "Dyomics 654": {"MEA": 20, "bME": 19},
    "Atto 655": {"MEA": 17, "bME": 22},
    "Atto 680": {"MEA": 8, "bME": 27},
    "Cy5.5": {"MEA": 16, "bME": 25},
    # near-infrared-absorbing
    "DyLight 750": {"MEA": 5, "bME": 6},
    "Cy7": {"MEA": 5, "bME": 2.6},
    "Alexa Fluor 750": {"MEA": 1.5, "bME": 6},
    "Atto 740": {"MEA": 3, "bME": 14},
    "Alexa Fluor 790": {"MEA": 5, "bME": 2.7},
    "IRDye 800 CW": {"MEA": 3, "bME": 127},
}

ORIGIN_DERIVED = "derived"
ORIGIN_USER = "user"
DOC_PATH = "docs/channel2_parameters.md"
DOC_URL = "https://github.com/Nico1989-cordoba/MPS-explorer/blob/main/docs/channel2_parameters.md"

# The plausibility limits the dialog and the comparison already apply (no scientific content).
EPS_LIMITS_NM = (1e-9, 1000.0)
MIN_SAMPLES_LIMITS = (1, 10000)


def dyes() -> List[str]:
    """The dyes of the table, in the table's order."""
    return list(DYE_CYCLES)


def cycles(dye: Optional[str], buffer: Optional[str]) -> Optional[float]:
    """Mean switching cycles of ``dye`` in ``buffer`` (Dempsey et al. 2011), or None when either is not in the
    table."""
    if not dye or not buffer:
        return None
    return DYE_CYCLES.get(dye, {}).get(buffer)


# --------------------------------------------------------------------------- the two formulas
def pair_within_probability(eps_nm: float, sigma_nm: float) -> float:
    """P(d <= eps) for two localizations of one emitter, each with precision sigma per axis: d is Rayleigh with
    scale sqrt(2) sigma, so P = 1 - exp(-eps^2 / (4 sigma^2))."""
    return 1.0 - math.exp(-(float(eps_nm) ** 2) / (4.0 * float(sigma_nm) ** 2))


def derive_eps2(eps1_nm: float, nena1_nm: float, nena2_nm: float) -> float:
    """eps_2 = eps_1 x NeNA_2 / NeNA_1: the same radius in units of each channel's precision."""
    return float(eps1_nm) * float(nena2_nm) / float(nena1_nm)


def expected_core_count(eps_nm: float, sigma_nm: float, n_loc_mol: float) -> float:
    """Expected number of one molecule's localizations within eps of one of them, itself included (DBSCAN's
    core-point count): 1 + (n - 1) x P(d <= eps)."""
    return 1.0 + (float(n_loc_mol) - 1.0) * pair_within_probability(eps_nm, sigma_nm)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def derive_min_samples(eps_nm: float, sigma_nm: float, n_loc_mol: float) -> Tuple[int, float]:
    """min_samples_2 = round(1 + (n - 1)(1 - exp(-eps^2 / 4 sigma^2))), half up and never below 1; returns the
    rounded value and the unrounded one."""
    raw = expected_core_count(eps_nm, sigma_nm, n_loc_mol)
    return max(1, _round_half_up(raw)), raw


# --------------------------------------------------------------------------- the precision of one channel
@dataclass
class ChannelPrecision:
    """NeNA of one channel's whole-field file, or the reason there is none."""

    nena_nm: Optional[float] = None
    n_pairs: int = 0
    source_path: str = ""          # never exported: the file name is (``source_file``)
    found_by: str = ""             # how the file was chosen, in words
    reason: str = ""               # why there is no value ("" when there is one)

    @property
    def source_file(self) -> str:
        return os.path.basename(self.source_path) if self.source_path else ""

    @property
    def measured(self) -> bool:
        return self.nena_nm is not None and math.isfinite(self.nena_nm) and self.nena_nm > 0

    def describe(self) -> str:
        if self.measured:
            return f"{self.nena_nm:.1f} nm (whole field {self.source_file}, {self.n_pairs:,} next-frame pairs)"
        where = f" ({self.source_file})" if self.source_file else ""
        return f"not measured{where}: {self.reason}"


def is_linked(loc: Any) -> bool:
    """Whether a loaded file is a linked one (Picasso Link): it carries Link's ``len`` column, or its processing
    chain names Link. A linked file has merged the consecutive-frame pairs NeNA measures."""
    columns = getattr(loc, "columns", {}) or {}
    if "len" in columns:
        return True
    for step in getattr(loc, "info", []) or []:
        if isinstance(step, dict):
            made_by = str(step.get("Generated by", ""))
            if "link" in made_by.lower():
                return True
    return False


def precision_from_file(path: str, *, found_by: str = "",
                        loader: Optional[Callable[[str], Any]] = None) -> ChannelPrecision:
    """NeNA of the localization file ``path``, which must be a whole-field, unlinked file with frame numbers.

    Every way it can fail gives an empty value and the reason: no file, a file that does not load, no frames, a
    linked file, fewer than ``tools.mps_quality.MIN_NENA_PAIRS`` next-frame pairs, or a fit that does not converge.
    """
    out = ChannelPrecision(source_path=str(path or ""), found_by=found_by)
    if not path:
        out.reason = "no whole-field file: choose one"
        return out
    if not os.path.isfile(path):
        out.reason = "the file does not exist"
        return out
    if loader is None:
        from tools.mps_io import load_localizations as loader
    try:
        loc = loader(path)
    except Exception as error:                                       # noqa: BLE001 - shown to the user
        out.reason = f"the file could not be loaded ({error})"
        return out
    if is_linked(loc):
        out.reason = ("a linked file (Picasso Link): its consecutive-frame pairs are merged, so NeNA cannot be "
                      "measured on it; choose the unlinked whole-field file")
        return out
    frame = loc.column("frame") if hasattr(loc, "column") else None
    if frame is None:
        out.reason = "the file has no frame numbers, which NeNA needs"
        return out
    import numpy as np

    from tools.mps_quality import MIN_NENA_PAIRS, nena
    result = nena(np.asarray(frame, dtype=np.int64), loc.x_nm, loc.y_nm)
    out.n_pairs = int(result.n_pairs)
    if not result.converged or not math.isfinite(result.precision_nm) or result.precision_nm <= 0:
        if result.n_pairs < MIN_NENA_PAIRS:
            out.reason = (f"only {result.n_pairs} next-frame neighbour pairs (NeNA needs {MIN_NENA_PAIRS}); is "
                          f"this a picked region rather than the whole field?")
        else:
            out.reason = "the NeNA fit did not converge on the next-frame distances"
        return out
    out.nena_nm = float(result.precision_nm)
    return out


_PICK_TOKENS = ("picked", "apicked", "_pick")


def find_whole_field(loaded_path: str) -> Tuple[str, str]:
    """The whole-field file of a loaded channel, by names only: (path, how it was found), or ("", why not).

    The name matcher of the Axoplasm panel (``tools.mps_file_match``) first; when it proposes nothing and the loaded
    file is not a picked one, the loaded file itself is the whole field.
    """
    if not loaded_path:
        return "", "no file loaded"
    from tools.mps_file_match import WHOLE_FIELD, match_files
    try:
        match = match_files(loaded_path)
    except OSError as error:
        match = None
        why = f"the folder could not be read ({error})"
    else:
        why = ""
    if match is not None:
        best = match.best(WHOLE_FIELD)
        if best is not None:
            return best.path, f"matched by name: {best.reason}"
        if match.is_ambiguous(WHOLE_FIELD):
            why = "several equally good whole-field files: choose one"
    name = os.path.basename(loaded_path).lower()
    if not why and not any(token in name for token in _PICK_TOKENS):
        return loaded_path, "the loaded file itself (its name is not a picked region's)"
    return "", why or "no whole-field file found by name: choose one"


# --------------------------------------------------------------------------- the derivation
@dataclass
class Channel2Derivation:
    """What the rule gives for channel 2, each value with its origin or the reason it is empty."""

    eps_nm: Optional[float] = None
    min_samples: Optional[int] = None
    min_samples_raw: Optional[float] = None
    eps1_nm: float = GAZAL_EPS_NM
    nena1_nm: Optional[float] = None        # the value used for the ratio
    nena1_origin: str = ""                  # "measured" | "reference (Gazal et al. 2026)" | ""
    nena2_nm: Optional[float] = None
    dye: str = ""
    buffer: str = ""
    n_loc_mol: Optional[float] = None
    n_loc_mol_origin: str = ""
    eps_origin: str = ""                    # text shown beside the field
    min_samples_origin: str = ""
    eps_reason: str = ""                    # why empty
    min_samples_reason: str = ""
    notes: List[str] = field(default_factory=list)


def derive_channel2(precision1: Optional[ChannelPrecision], precision2: Optional[ChannelPrecision],
                    dye: Optional[str], buffer: Optional[str], *, eps1_nm: float = GAZAL_EPS_NM
                    ) -> Channel2Derivation:
    """eps_2 and min_samples_2 from the two channels' precisions and channel 2's dye and buffer."""
    out = Channel2Derivation(eps1_nm=float(eps1_nm), dye=dye or "", buffer=buffer or "")
    p2 = precision2 if precision2 is not None else ChannelPrecision(reason="no whole-field file: choose one")
    if not p2.measured:
        out.eps_reason = out.min_samples_reason = f"channel 2's precision is {p2.describe()}"
        return out
    nena2 = float(p2.nena_nm or 0.0)                            # measured: a positive number
    out.nena2_nm = nena2
    p1 = precision1
    if p1 is not None and p1.measured:
        out.nena1_nm, out.nena1_origin = float(p1.nena_nm or 0.0), "measured"
        ch1_words = f"channel 1's NeNA {out.nena1_nm:.1f} nm (whole field {p1.source_file})"
    else:
        out.nena1_nm, out.nena1_origin = GAZAL_NENA_NM, "reference (Gazal et al. 2026)"
        why = p1.describe() if p1 is not None else "not measured"
        out.notes.append(f"Channel 1's NeNA is {why}; the published {GAZAL_NENA_NM:g} nm (Gazal et al. 2026) is "
                         f"used for the ratio instead.")
        ch1_words = f"the published channel-1 NeNA {GAZAL_NENA_NM:g} nm (Gazal et al. 2026; channel 1 not measured)"
    out.eps_nm = derive_eps2(eps1_nm, out.nena1_nm, nena2)
    ch2_words = f"NeNA ({nena2:.1f} nm, whole field {p2.source_file})"
    out.eps_origin = (f"derived from {ch2_words}: eps_1 {eps1_nm:g} nm x {nena2:.1f} / {out.nena1_nm:.1f}, "
                      f"with {ch1_words}")
    n = cycles(dye, buffer)
    if n is None:
        out.min_samples_reason = "choose channel 2's dye and buffer (the table of Dempsey et al. 2011)"
        return out
    out.n_loc_mol = float(n)
    out.n_loc_mol_origin = "dye table (Dempsey et al. 2011, Table 1), lower bound"
    words = f"{dye}/{BUFFER_WORDS.get(str(buffer), buffer)}"
    if n < 2:
        out.min_samples_reason = (f"{words} has {n:g} switching cycles per molecule in the table: fewer than two "
                                  f"localizations per molecule cannot form a density cluster")
        return out
    out.min_samples, out.min_samples_raw = derive_min_samples(out.eps_nm, nena2, n)
    out.min_samples_origin = (f"derived from {ch2_words} and {words} ({n:g} switching cycles, a lower bound): "
                              f"1 + {n - 1:g} x {pair_within_probability(out.eps_nm, nena2):.3f} = "
                              f"{out.min_samples_raw:.1f}")
    return out


# --------------------------------------------------------------------------- what is stored and exported
# The keys of the record kept per acquisition folder (settings) and written to the two-channel export, in order.
EXPORT_KEYS: Tuple[str, ...] = (
    "eps_b_origin", "min_samples_b_origin",
    "nena_a_nm", "nena_a_source_file", "nena_a_origin", "eps_a_reference_nm",
    "nena_b_nm", "nena_b_source_file",
    "dye_b", "buffer_b", "n_loc_mol_b", "n_loc_mol_b_origin",
)


def provenance(derivation: Optional[Channel2Derivation], precision1: Optional[ChannelPrecision],
               precision2: Optional[ChannelPrecision], eps_nm: float, min_samples: int) -> Dict[str, Any]:
    """The record of where the pair (eps_nm, min_samples) in use comes from. A value equal to the derived one is
    "derived"; any other is "user"."""
    d = derivation or Channel2Derivation()
    rec: Dict[str, Any] = {
        "eps_nm": float(eps_nm), "min_samples": float(int(min_samples)),
        "derived_eps_nm": d.eps_nm, "derived_min_samples": None if d.min_samples is None else float(d.min_samples),
        "nena_a_nm": d.nena1_nm if d.nena1_nm is not None else (
            precision1.nena_nm if precision1 is not None and precision1.measured else None),
        "nena_a_source_file": precision1.source_file if precision1 is not None else "",
        "nena_a_origin": d.nena1_origin,
        "eps_a_reference_nm": d.eps1_nm if d.eps_nm is not None else None,
        "nena_b_nm": precision2.nena_nm if precision2 is not None and precision2.measured else None,
        "nena_b_source_file": precision2.source_file if precision2 is not None else "",
        "dye_b": d.dye, "buffer_b": d.buffer,
        "n_loc_mol_b": d.n_loc_mol, "n_loc_mol_b_origin": d.n_loc_mol_origin,
    }
    return reconcile(rec, eps_nm, min_samples)


def _same(a: Any, b: Any, tol: float) -> bool:
    try:
        return a is not None and b is not None and abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def reconcile(record: Optional[Dict[str, Any]], eps_nm: float, min_samples: int) -> Dict[str, Any]:
    """The record for the pair now in use: each origin is "derived" only while the value in use is the derived
    one (eps within half of the 0.1 nm the dialog writes it to); otherwise "user"."""
    rec = dict(record or {})
    rec["eps_nm"], rec["min_samples"] = float(eps_nm), float(int(min_samples))
    rec["eps_b_origin"] = ORIGIN_DERIVED if _same(rec.get("derived_eps_nm"), eps_nm, 0.0501) else ORIGIN_USER
    rec["min_samples_b_origin"] = (ORIGIN_DERIVED if _same(rec.get("derived_min_samples"), int(min_samples), 0.0)
                                   else ORIGIN_USER)
    return rec


def export_fields(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The export columns of a record (``EXPORT_KEYS``); empty cells where nothing is known. Never a path: the
    source files are written by name only."""
    rec = record or {}
    out: Dict[str, Any] = {}
    for key in EXPORT_KEYS:
        value = rec.get(key)
        if key.endswith("_source_file") and value:
            value = os.path.basename(str(value))
        if isinstance(value, float):
            value = round(value, 3)
        out[key] = "" if value is None else value
    return out
