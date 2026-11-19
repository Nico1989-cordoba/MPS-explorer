# -*- coding: utf-8 -*-
"""
The explicit rule for DBSCAN's eps and min_samples, with an editable c_label (clustering plan A2).

What it computes
----------------
* **eps = min(2.5 x NeNA, 25 nm)** [simulation: clustering research sections 8.3-8.5 - the reference 25 nm is
  2.2-2.3 NeNA there, and 2-3 NeNA rendered the same; the 25 nm cap is the published radius (Gazal et al. 2026) and
  keeps a poor precision (sigma 20 nm) from inflating the fusion of neighbours].
* **min_samples = round(1 + (n_q x lpc - 1) x P(d <= eps)), never below 3**, where
  - n_q is the q-quantile (q = 0.25) of the switching cycles of ONE label, modelled as geometric with mean c_label
    [lit: Karathanasis et al. 2017 and Hummer et al. 2016 - blink counts of Alexa Fluor 647 follow a geometric law;
    q: simulation, clustering research: the label at the lower quartile of cycles still forms a cluster],
  - lpc is the localizations per cycle of the file (measured: ``tools.mps_events``; decision D3),
  - P(d <= eps) = 1 - exp(-eps^2 / (4 NeNA^2)) is the probability that two localizations of one emitter are within
    eps (Rayleigh law of scale sqrt(2) sigma) [derivation; docs/channel2_parameters.md section 4],
  - "1 + ... - 1": DBSCAN counts the point itself [lit: Ester et al. 1996; scikit-learn's min_samples includes it].
  The floor 3 is minPts >= dimension + 1 for 2D data [ad hoc floor; the heuristic is attributed to Ester et al. 1996
  by Mazouchi & Milstein 2016 (pending full-text check); Schubert et al. 2017 give 4 in 2D, docs/dbscan_rule.md 2.1].

The same formula with the MEAN cycles instead of the quantile and one localization per cycle is the worked check of
docs/channel2_parameters.md section 5: 1 + 13 x 0.662 = 9.6 -> 10, the published min_samples
(``expected_core_count``, reused from ``tools.mps_channel_params``).

c_label (decision D1, 2026-10-22): 14 (Alexa Fluor 647 in MEA, provisional default) and 26 (in beta-mercaptoethanol,
sensitivity column), both from Dempsey et al. (2011, Table 1), read from the dye table of
``tools.mps_channel_params``. Neither is measured in the laboratory's buffer.

What it does NOT change
-----------------------
The analysis default stays the compatibility mode 25/10 (decision D2). The rule is computed and shown beside it: every
table can come out double, "compat_25_10" and "rule" (``double_table_cells``).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools import mps_channel_params as ch
from tools.mps_events import LOCS_PER_CYCLE_FALLBACK, LocsPerCycle

__all__ = [
    "EPS_RULE_FACTOR", "EPS_RULE_CAP_NM", "C_LABEL_MEA", "C_LABEL_BME", "C_LABEL_DEFAULT", "C_LABELS_REPORTED",
    "CYCLE_QUANTILE", "MIN_SAMPLES_FLOOR", "COMPAT_EPS_NM", "COMPAT_MIN_SAMPLES", "DBSCAN_MODES", "RuleInput",
    "DBSCANRule", "eps_rule", "cycles_quantile", "pair_probability", "min_samples_rule", "derive_rule",
    "rule_table", "double_table_cells", "method_specs", "worked_check",
]

# --------------------------------------------------------------------------- defaults (origin in brackets)
EPS_RULE_FACTOR: float = 2.5         # [simulation: clustering research 8.3-8.5; 25 nm = 2.2-2.3 NeNA there]
EPS_RULE_CAP_NM: float = ch.GAZAL_EPS_NM   # [lit: Gazal et al. 2026, eps 25 nm; simulation: sigma 20 nm row]
C_LABEL_MEA: float = float(ch.DYE_CYCLES["Alexa Fluor 647"]["MEA"])   # [lit: Dempsey et al. 2011, Table 1]
C_LABEL_BME: float = float(ch.DYE_CYCLES["Alexa Fluor 647"]["bME"])   # [lit: Dempsey et al. 2011, Table 1]
C_LABEL_DEFAULT: float = C_LABEL_MEA  # provisional (D1): the laboratory images in MEA; not measured in its buffer
C_LABELS_REPORTED: Tuple[float, ...] = (C_LABEL_MEA, C_LABEL_BME)   # D1: both, 26 as the sensitivity column
CYCLE_QUANTILE: float = 0.25         # [simulation: clustering research, the "cycles rule" strategy]
MIN_SAMPLES_FLOOR: int = 3           # [ad hoc: minPts >= dimension + 1; not in Schubert et al. 2017 (they give 4
                                     # in 2D); attributed to Ester et al. 1996, pending full-text check]
COMPAT_EPS_NM: float = ch.GAZAL_EPS_NM            # [lit: Gazal et al. 2026] the compatibility mode
COMPAT_MIN_SAMPLES: int = ch.GAZAL_MIN_SAMPLES    # [lit: Gazal et al. 2026] the compatibility mode
DBSCAN_MODES: Tuple[str, ...] = ("compat_25_10", "rule")


@dataclass(frozen=True)
class RuleInput:
    """One input of the rule: its value and where it comes from ([lit: ...], [measured], [ad hoc], ...)."""

    value: Any
    origin: str


@dataclass
class DBSCANRule:
    """The rule's result: eps and min_samples with every input and its origin.

    ``min_samples_raw`` is the value before rounding and the floor; ``min_samples_range`` the rule at the two ends
    of the localizations-per-cycle 95 % interval (None without one).
    """

    eps_nm: float
    min_samples: int
    min_samples_raw: float
    p_neighbour: float
    n_cycles_quantile: int
    inputs: Dict[str, RuleInput]
    min_samples_range: Optional[Tuple[int, int]] = None
    eps_capped: bool = False
    notes: List[str] = field(default_factory=list)

    def as_export(self) -> Dict[str, Any]:
        """The export columns of plan A2."""
        lpc = self.inputs["loc_per_cycle"]
        return {"dbscan_mode": "rule", "eps_rule_nm": float(self.eps_nm), "min_samples_rule": int(self.min_samples),
                "c_label_used": float(self.inputs["c_label"].value), "c_label_source": self.inputs["c_label"].origin,
                "loc_per_cycle_used": float(lpc.value), "loc_per_cycle_source": lpc.origin,
                "p_neighbour": float(self.p_neighbour), "nena_nm_used": float(self.inputs["nena_nm"].value),
                "cycle_quantile": float(self.inputs["cycle_quantile"].value)}

    def describe(self) -> str:
        lines = [f"eps = {self.eps_nm:.1f} nm, min_samples = {self.min_samples} (raw {self.min_samples_raw:.2f})"]
        for k, v in self.inputs.items():
            lines.append(f"  {k} = {v.value} {v.origin}")
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


# --------------------------------------------------------------------------- the pieces
def eps_rule(nena_nm: float, factor: float = EPS_RULE_FACTOR, cap_nm: float = EPS_RULE_CAP_NM) -> Tuple[float, bool]:
    """eps = min(factor x NeNA, cap): the value and whether the cap decided it."""
    s = float(nena_nm)
    if not (math.isfinite(s) and s > 0):
        raise ValueError(f"NeNA must be positive, got {nena_nm}")
    raw = float(factor) * s
    return (float(cap_nm), True) if raw > float(cap_nm) else (raw, False)


def cycles_quantile(c_label: float, q: float = CYCLE_QUANTILE) -> int:
    """Smallest n with P(N <= n) >= q for N ~ Geometric(p = 1 / c_label) on 1, 2, ... (mean c_label)."""
    c = float(c_label)
    if not (math.isfinite(c) and c >= 1.0):
        raise ValueError(f"a geometric cycle count has mean >= 1, got {c_label}")
    if not 0.0 < float(q) < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    if c == 1.0:
        return 1
    p = 1.0 / c
    return max(1, int(math.ceil(math.log(1.0 - float(q)) / math.log(1.0 - p) - 1e-12)))


def pair_probability(eps_nm: float, nena_nm: float) -> float:
    """P(d <= eps) for two localizations of one emitter (reused from tools.mps_channel_params)."""
    return ch.pair_within_probability(eps_nm, nena_nm)


def min_samples_rule(eps_nm: float, nena_nm: float, c_label: float, locs_per_cycle: float,
                     q: float = CYCLE_QUANTILE, floor: int = MIN_SAMPLES_FLOOR,
                     count_self: bool = True) -> Tuple[int, float, int, float]:
    """(min_samples, raw value, n_q, P) of the rule.

    ``count_self=False`` gives the clustering research's own formula, n_q x lpc x P (it multiplies the point itself by
    P too); the default counts the point itself once, as DBSCAN does and as docs/channel2_parameters.md does. The two
    differ by 1 - P (0.34 at eps / NeNA = 25 / 12).
    """
    lpc = float(locs_per_cycle)
    if not (math.isfinite(lpc) and lpc > 0):
        raise ValueError(f"localizations per cycle must be positive, got {locs_per_cycle}")
    n_q = cycles_quantile(c_label, q)
    p = pair_probability(eps_nm, nena_nm)
    n_loc = n_q * lpc
    raw = (1.0 + max(n_loc - 1.0, 0.0) * p) if count_self else n_loc * p
    return max(int(floor), int(math.floor(raw + 0.5))), raw, n_q, p   # half up, as channel 2


# --------------------------------------------------------------------------- the rule
def derive_rule(nena_nm: float, *, nena_origin: str = "[measured] NeNA of the file",
                c_label: float = C_LABEL_DEFAULT, c_label_origin: Optional[str] = None,
                locs_per_cycle: Optional[Any] = None, q: float = CYCLE_QUANTILE,
                eps_factor: float = EPS_RULE_FACTOR, eps_cap_nm: float = EPS_RULE_CAP_NM,
                floor: int = MIN_SAMPLES_FLOOR, count_self: bool = True,
                use_split_corrected: bool = False) -> DBSCANRule:
    """The rule for one file.

    ``locs_per_cycle``: a ``tools.mps_events.LocsPerCycle`` (measured, with its interval), a number (taken as given,
    labelled [user]), or None (the [ad hoc] fallback of 2, labelled as such). ``c_label`` is editable; its origin is
    written for 14 and 26 and must be given for any other value. ``use_split_corrected`` takes the measured value
    corrected for the events the 3 NeNA radius splits (``LocsPerCycle.split_corrected``) instead of the raw mean;
    off by default (decision D5 as approved; see docs/dbscan_rule.md section 4 for the size of the difference).
    """
    eps, capped = eps_rule(nena_nm, eps_factor, eps_cap_nm)
    notes: List[str] = []
    if c_label_origin is None:
        if math.isclose(float(c_label), C_LABEL_MEA):
            c_label_origin = "[lit: Dempsey et al. 2011, Alexa Fluor 647 in MEA] provisional, not measured here"
        elif math.isclose(float(c_label), C_LABEL_BME):
            c_label_origin = "[lit: Dempsey et al. 2011, Alexa Fluor 647 in beta-ME] sensitivity, not measured here"
        else:
            c_label_origin = "[user] typed"
    lpc_range: Optional[Tuple[float, float]] = None
    if isinstance(locs_per_cycle, LocsPerCycle):
        lpc_value = float(locs_per_cycle.value)
        lpc_origin = f"{locs_per_cycle.origin} ({locs_per_cycle.source}, {locs_per_cycle.n_events} events)"
        if locs_per_cycle.ci95 is not None and all(math.isfinite(v) for v in locs_per_cycle.ci95):
            lpc_range = (max(locs_per_cycle.ci95[0], 1e-9), locs_per_cycle.ci95[1])
        if use_split_corrected and locs_per_cycle.split_corrected is not None:
            shift = float(locs_per_cycle.split_corrected) - lpc_value
            lpc_value = float(locs_per_cycle.split_corrected)
            lpc_origin += ", corrected for the expected split of events by the linking radius [derivation]"
            if lpc_range is not None:
                lpc_range = (max(lpc_range[0] + shift, 1e-9), lpc_range[1] + shift)
        if not locs_per_cycle.measured:
            notes.append(f"localizations per cycle not measured: {locs_per_cycle.note}")
    elif locs_per_cycle is None:
        lpc_value = float(LOCS_PER_CYCLE_FALLBACK)
        lpc_origin = "[ad hoc] fallback: assumed by the clustering research's simulation, not measured"
        notes.append("localizations per cycle not measured: the fallback is used")
    else:
        lpc_value = float(locs_per_cycle)
        lpc_origin = "[user] given"
    ms, raw, n_q, p = min_samples_rule(eps, nena_nm, c_label, lpc_value, q, floor, count_self)
    ms_range: Optional[Tuple[int, int]] = None
    if lpc_range is not None:
        lo = min_samples_rule(eps, nena_nm, c_label, lpc_range[0], q, floor, count_self)[0]
        hi = min_samples_rule(eps, nena_nm, c_label, lpc_range[1], q, floor, count_self)[0]
        ms_range = (lo, hi)
    if capped:
        notes.append(f"eps capped at {eps_cap_nm:g} nm: {eps_factor:g} x NeNA = {eps_factor * float(nena_nm):.1f} nm")
    if raw < floor:
        notes.append(f"min_samples raised to the floor {floor} (raw {raw:.2f})")
    inputs = {
        "nena_nm": RuleInput(float(nena_nm), nena_origin),
        "eps_factor": RuleInput(float(eps_factor), "[simulation: clustering research 8.3-8.5]"),
        "eps_cap_nm": RuleInput(float(eps_cap_nm), "[lit: Gazal et al. 2026] the published eps"),
        "c_label": RuleInput(float(c_label), c_label_origin),
        "cycle_quantile": RuleInput(float(q), "[simulation: clustering research, cycles rule]"),
        "loc_per_cycle": RuleInput(lpc_value, lpc_origin),
        "floor": RuleInput(int(floor), "[lit: Ester et al. 1996; Schubert et al. 2017] minPts >= dimension + 1"),
        "count_self": RuleInput(bool(count_self), "[lit: Ester et al. 1996] the point counts itself"),
    }
    return DBSCANRule(eps_nm=float(eps), min_samples=int(ms), min_samples_raw=float(raw), p_neighbour=float(p),
                      n_cycles_quantile=int(n_q), inputs=inputs, min_samples_range=ms_range, eps_capped=capped,
                      notes=notes)


def rule_table(nena_nm: float, locs_per_cycle: Optional[Any] = None,
               c_labels: Sequence[float] = C_LABELS_REPORTED, **kwargs: Any) -> Dict[float, DBSCANRule]:
    """D1: the rule under every c_label reported (14 provisional default, 26 sensitivity)."""
    return {float(c): derive_rule(nena_nm, c_label=float(c), locs_per_cycle=locs_per_cycle, **kwargs)
            for c in c_labels}


def double_table_cells(rule: Optional[DBSCANRule]) -> List[Dict[str, Any]]:
    """D2, the double-table convention: the parameter sets every table is computed with, compatibility first.

    Each item is ``{"mode", "eps", "min_samples"}``; feed them to ``tools.dbscan_sweep.sweep(..., cells=...)`` (the
    params of a cell are ``eps`` and ``min_samples``). Without a rule (no NeNA) only the compatibility cell is
    returned, and the table says the rule column is empty.
    """
    cells: List[Dict[str, Any]] = [{"mode": "compat_25_10", "eps": float(COMPAT_EPS_NM),
                                    "min_samples": int(COMPAT_MIN_SAMPLES)}]
    if rule is not None:
        cells.append({"mode": "rule", "eps": float(rule.eps_nm), "min_samples": int(rule.min_samples)})
    return cells


def method_specs(rule: Optional[DBSCANRule]) -> List[Tuple[str, Dict[str, Any]]]:
    """The double table as method specs of ``tools.multimethod.run_methods``: DBSCAN 25/10 (its reference) and DBSCAN
    with the rule's parameters, so every multi-method table carries both columns."""
    return [("DBSCAN", {"eps": c["eps"], "min_samples": c["min_samples"]}) for c in double_table_cells(rule)]


def worked_check(eps_nm: float = COMPAT_EPS_NM, nena_nm: float = ch.GAZAL_NENA_NM,
                 cycles: float = C_LABEL_MEA) -> Tuple[int, float]:
    """docs/channel2_parameters.md section 5: mean cycles, one localization per cycle -> (10, 9.6)."""
    return ch.derive_min_samples(eps_nm, nena_nm, cycles)
