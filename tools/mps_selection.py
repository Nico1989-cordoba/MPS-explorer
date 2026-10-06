# -*- coding: utf-8 -*-
"""
Which ring pairs the column test reads: the criteria of the viability rules as SWITCHES (D-43, user decisions of
2026-10-02). No Qt; geometry only (R8: nothing here computes a column statistic).

Rule v2 (D-41, ``tools.mps_axial_precision.viability_v2``) reads four criteria on the LOCALIZATIONS: (1) a SiZer peak
of each ring, (2) a SiZer valley between them, (3) the localizations of each ring >= f_min(x) of the central ring,
(4) the D-39 leak copies <= 2 % VIABLE / <= 5 % MARGINAL. Rule v2-clusters (D-42, ``tools.mps_axial_clusters.
viability_v2c``) reads the same four on the CLUSTERS, variants A (ring-slab medians), B (slab-free 3D groups),
C (ring-slab truncated-normal centres), with the per-cluster leak "tnfix" as (4c).

The user asked (2026-10-02) that every criterion can be checked and unchecked, so the pairs can be seen under
stricter or more permissive combinations, and (Q-38) that BOTH rules can select the pairs of the column tests. This
module is the single place that turns the per-pair criteria (already computed by the two rules, so toggling never
re-runs SiZer or DBSCAN) into a selection:

- ``SelectionSpec``: rule ("v2", "v2c", "both" = AND, "either" = OR), the v2c variant, one ``CriteriaSet`` per rule
  (the four switches + the leak estimate: "D-39", "tnfix" or "max", Q-39 still open), a canonical text label
  (``v2[peak,valley,count,leak:D-39]``) and an 8-hex hash that also covers the thresholds.
- ``DEFAULT_SELECTION`` = the pre-specified rule v2 with its four criteria and the D-39 leak. Under it every output is
  byte-identical to the program without this module (``rule_text`` returns rule v2's own text, the pairs and reasons
  are rule v2's verbatim). ANY other selection is an "exploratory selection" and says so wherever a result appears.
- ``axon_criteria`` / ``compute_axon_criteria``: the per-pair criteria of one axon (``AxonCriteria``).
- ``evaluate_selection``: the pure evaluation (microseconds), ``SelectionResult`` with the verdict, tier and reasons
  of every pair, the key a simulated replicate must share (simnull), the rule text written in the outputs.
- ``ExplorationLog``: the append-only CSV of every selection a column statistic was shown or written under, and the
  count of selection variants tried this session (p values are NOT corrected for trying several selections).

MARGINAL tier: with the leak criterion checked, the 2 % / 5 % tiers apply as in D-41 / D-42; with it unchecked, a
pair that meets the other checked criteria is VIABLE (there is no MARGINAL tier then).
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import itertools
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from tools.mps_axial_precision import P_REF_DEFAULT_NM

__all__ = [
    "CRITERIA", "LEAK_ESTIMATES", "RULES", "VARIANT_KEYS", "SELECTION_SCHEMA", "EXPLORATORY", "CriteriaSet",
    "V2_DEFAULT", "V2C_DEFAULT", "SelectionSpec", "DEFAULT_SELECTION", "SideCriteria", "PairCriteria",
    "AxonCriteria", "axon_criteria", "compute_axon_criteria", "leak_value", "SideEvaluation", "evaluate_side",
    "PairSelection", "SelectionResult", "evaluate_selection", "combination_table", "CombinationCell",
    "selector_warnings", "selection_thresholds", "selection_source_sha", "selection_viability", "selection_tiers",
    "lumen_fp", "lumen_fp_text", "ExplorationLog", "LOG_COLUMNS", "SESSION_ID", "default_log_path", "counter_text",
    "log_row", "pair_p_text", "geometry_for_file", "SELECTION_PAIR_COLUMNS", "selection_pair_rows",
]

# ============================================================================ the vocabulary
CRITERIA: Tuple[str, ...] = ("peak", "valley", "count", "leak")
CRITERION_LABELS: Dict[str, str] = {"peak": "(1) Peak", "valley": "(2) Valley", "count": "(3) Count", "leak": "(4) Leak"}
CRITERION_LABELS_V2C: Dict[str, str] = {"peak": "(1c) Peak", "valley": "(2c) Valley", "count": "(3c) Count",
                                        "leak": "(4c) Leak"}
LEAK_ESTIMATES: Tuple[str, ...] = ("D-39", "tnfix", "max")
LEAK_ESTIMATE_LABELS: Dict[str, str] = {"D-39": "D-39 (per ring, rule v2's)", "tnfix": "per cluster (tnfix, D-42)",
                                        "max": "the larger of the two"}
RULES: Tuple[str, ...] = ("v2", "v2c", "both", "either")
# the letters of the v2c variants and the keys of ``tools.mps_axial_clusters.V2C_VARIANT_ORDER``
VARIANT_KEYS: Dict[str, str] = {"A": "ring_slab", "B": "groups_3d", "C": "ring_slab_tn"}
VARIANTS: Tuple[str, ...] = ("A", "B", "C")
SIDES: Tuple[str, ...] = ("v2",) + VARIANTS
SELECTION_SCHEMA = "selection v1"
EXPLORATORY = "exploratory selection"
VERDICT_VIABLE = "viable"
VERDICT_MARGINAL = "marginal"
VERDICT_NOT_VIABLE = "not viable"
_TIER_RANK = {VERDICT_NOT_VIABLE: 0, VERDICT_MARGINAL: 1, VERDICT_VIABLE: 2}
TIER_PRIMARY, TIER_SENSITIVITY, TIER_NONE = "primary", "sensitivity", "not analysed"
TIER_OF = {VERDICT_VIABLE: TIER_PRIMARY, VERDICT_MARGINAL: TIER_SENSITIVITY, VERDICT_NOT_VIABLE: TIER_NONE}
V2C_NOT_COMPUTED = "v2c not computed"


@dataclass(frozen=True)
class CriteriaSet:
    """The four switches of one rule and the leak estimate criterion 4 reads ("D-39", "tnfix", "max"). The estimate
    is kept when the leak is unchecked (re-checking restores it) but is then not part of the label."""

    peak: bool = True
    valley: bool = True
    count: bool = True
    leak: bool = True
    leak_estimate: str = "D-39"

    def __post_init__(self) -> None:
        if self.leak_estimate not in LEAK_ESTIMATES:
            raise ValueError(f"leak_estimate must be one of {LEAK_ESTIMATES}, got {self.leak_estimate!r}")
        for name in CRITERIA:
            v = getattr(self, name)
            if not isinstance(v, (bool, np.bool_)):
                raise ValueError(f"criterion {name} must be a bool, got {v!r}")

    def checked(self, name: str) -> bool:
        if name not in CRITERIA:
            raise ValueError(f"unknown criterion {name!r} (one of {CRITERIA})")
        return bool(getattr(self, name))

    @property
    def enabled(self) -> Tuple[str, ...]:
        return tuple(c for c in CRITERIA if self.checked(c))

    def items_text(self) -> str:
        return ",".join(f"leak:{self.leak_estimate}" if c == "leak" else c for c in self.enabled)

    def with_(self, **changes: Any) -> "CriteriaSet":
        return dataclasses.replace(self, **changes)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {c: bool(getattr(self, c)) for c in CRITERIA}
        out["leak_estimate"] = self.leak_estimate
        return out

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "CriteriaSet":
        return cls(**{c: bool(d[c]) for c in CRITERIA}, leak_estimate=str(d["leak_estimate"]))


# the pre-specified sets: rule v2 with D-39 (D-41; Q-39 open), rule v2c with its own tnfix (D-42)
V2_DEFAULT = CriteriaSet(True, True, True, True, "D-39")
V2C_DEFAULT = CriteriaSet(True, True, True, True, "tnfix")
_SIDE_RE = re.compile(r"^(v2|v2c-([ABC]))\[(.*)\]$")


def _side_label(name: str, cs: CriteriaSet) -> str:
    return f"{name}[{cs.items_text()}]"


def _parse_side(text: str, default: CriteriaSet) -> Tuple[str, Optional[str], CriteriaSet]:
    m = _SIDE_RE.match(text.strip())
    if not m:
        raise ValueError(f"selection side {text!r}: expected v2[...] or v2c-A/B/C[...]")
    seen: Set[str] = set()
    est = default.leak_estimate
    for raw in (m.group(3).split(",") if m.group(3).strip() else []):
        item = raw.strip()
        name, sep, value = item.partition(":")
        name = name.strip()
        if name not in CRITERIA:
            raise ValueError(f"selection side {text!r}: unknown criterion {name!r} (one of {CRITERIA})")
        if name in seen:
            raise ValueError(f"selection side {text!r}: criterion {name!r} given twice")
        if name == "leak":
            if not sep:
                raise ValueError(f"selection side {text!r}: the leak item is written leak:<estimate> "
                                 f"(one of {LEAK_ESTIMATES})")
            est = value.strip()
            if est not in LEAK_ESTIMATES:
                raise ValueError(f"selection side {text!r}: leak estimate {est!r} is not one of {LEAK_ESTIMATES}")
        elif sep:
            raise ValueError(f"selection side {text!r}: only the leak item takes a value")
        seen.add(name)
    cs = CriteriaSet(**{c: (c in seen) for c in CRITERIA}, leak_estimate=est)
    return ("v2" if m.group(2) is None else "v2c"), m.group(2), cs


@dataclass(frozen=True, eq=False)
class SelectionSpec:
    """
    One selection of the pairs the column tests read: ``rule`` "v2" (localizations), "v2c" (clusters, variant
    ``variant``), "both" (a pair must pass both: the worse tier) or "either" (one is enough: the better tier); one
    ``CriteriaSet`` per rule. Two specs are equal when their canonical labels are (the variant and the unused side
    are normalised away for rule "v2", the unused v2 side for rule "v2c"; an unchecked leak's estimate is not part of
    the label).
    """

    rule: str = "v2"
    variant: str = "B"
    v2: CriteriaSet = V2_DEFAULT
    v2c: CriteriaSet = V2C_DEFAULT

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {self.rule!r}")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {self.variant!r}")
        if not isinstance(self.v2, CriteriaSet) or not isinstance(self.v2c, CriteriaSet):
            raise ValueError("v2 and v2c must be CriteriaSet")
        if self.rule == "v2":
            object.__setattr__(self, "variant", "B")
            object.__setattr__(self, "v2c", V2C_DEFAULT)
        elif self.rule == "v2c":
            object.__setattr__(self, "v2", V2_DEFAULT)

    # ---- identity
    @property
    def label(self) -> str:
        left = _side_label("v2", self.v2)
        right = _side_label(f"v2c-{self.variant}", self.v2c)
        if self.rule == "v2":
            return left
        if self.rule == "v2c":
            return right
        return f"{left} {'AND' if self.rule == 'both' else 'OR'} {right}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SelectionSpec) and other.label == self.label

    def __hash__(self) -> int:
        return hash(self.label)

    def __repr__(self) -> str:
        return f"SelectionSpec({self.label!r})"

    @property
    def hash(self) -> str:
        text = self.label + "\n" + SELECTION_SCHEMA + "\n" + repr(selection_thresholds())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    @property
    def is_default(self) -> bool:
        return self.label == DEFAULT_SELECTION.label

    @property
    def exploratory(self) -> bool:
        return not self.is_default

    @property
    def uses_v2(self) -> bool:
        return self.rule in ("v2", "both", "either")

    @property
    def uses_v2c(self) -> bool:
        return self.rule in ("v2c", "both", "either")

    @property
    def needs_v2c(self) -> bool:
        """v2c must be computed: a v2c side, or a leak estimate that reads tnfix."""
        sides = ([self.v2] if self.uses_v2 else []) + ([self.v2c] if self.uses_v2c else [])
        return self.uses_v2c or any(cs.leak and cs.leak_estimate in ("tnfix", "max") for cs in sides)

    @property
    def side_names(self) -> Tuple[str, ...]:
        """The sides read, as ``evaluate_side`` names them: "v2" and / or the variant letter."""
        return (("v2",) if self.uses_v2 else ()) + ((self.variant,) if self.uses_v2c else ())

    def display(self) -> str:
        """The text shown next to a result: "EXPLORATORY SELECTION <label> #hash" or the pre-specified rule."""
        if self.is_default:
            return f"pre-specified rule {self.label}"
        return f"EXPLORATORY SELECTION {self.label} #{self.hash}"

    def rule_text(self, v2_rule: str, v2c_rule: str = "") -> str:
        """The rule text written in the outputs (``viability_rule`` and the like): the default selection gives rule
        v2's own text EXACTLY (outputs byte-identical); any other "exploratory selection <label> #<hash> | <v2 rule>"
        (+ " | <v2c rule>" when v2c is read)."""
        if self.is_default:
            return str(v2_rule)
        out = f"{EXPLORATORY} {self.label} #{self.hash} | {v2_rule}"
        if self.needs_v2c and v2c_rule:
            out += f" | {v2c_rule}"
        return out

    # ---- parsing and JSON
    @classmethod
    def from_label(cls, text: str) -> "SelectionSpec":
        """Parse a label (canonical or with any item order / spaces around AND / OR); ValueError otherwise."""
        t = str(text).strip()
        parts = re.split(r"\s+(AND|OR)\s+", t)
        if len(parts) == 1:
            kind, letter, cs = _parse_side(parts[0], V2_DEFAULT)
            if kind == "v2":
                return cls("v2", "B", cs, V2C_DEFAULT)
            # the default of the unchecked leak estimate is the side's own
            if not cs.leak:
                cs = cs.with_(leak_estimate=V2C_DEFAULT.leak_estimate)
            return cls("v2c", str(letter), V2_DEFAULT, cs)
        if len(parts) != 3:
            raise ValueError(f"selection {text!r}: at most two sides joined by AND or OR")
        sides = [_parse_side(parts[0], V2_DEFAULT), _parse_side(parts[2], V2C_DEFAULT)]
        kinds = sorted(s[0] for s in sides)
        if kinds != ["v2", "v2c"]:
            raise ValueError(f"selection {text!r}: AND / OR join one v2[...] side and one v2c-X[...] side")
        v2s = next(s for s in sides if s[0] == "v2")
        v2cs = next(s for s in sides if s[0] == "v2c")
        v2_cs = v2s[2] if v2s[2].leak else v2s[2].with_(leak_estimate=V2_DEFAULT.leak_estimate)
        v2c_cs = v2cs[2] if v2cs[2].leak else v2cs[2].with_(leak_estimate=V2C_DEFAULT.leak_estimate)
        return cls("both" if parts[1] == "AND" else "either", str(v2cs[1]), v2_cs, v2c_cs)

    def to_json(self) -> Dict[str, Any]:
        return {"schema": SELECTION_SCHEMA, "label": self.label, "hash": self.hash, "rule": self.rule,
                "variant": self.variant, "v2": self.v2.to_json(), "v2c": self.v2c.to_json(),
                "exploratory": self.exploratory}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "SelectionSpec":
        if str(d.get("schema", "")) != SELECTION_SCHEMA:
            raise ValueError(f"selection JSON of schema {d.get('schema')!r}, this program reads {SELECTION_SCHEMA!r}")
        spec = cls(str(d["rule"]), str(d["variant"]), CriteriaSet.from_json(d["v2"]), CriteriaSet.from_json(d["v2c"]))
        if "label" in d and str(d["label"]) != spec.label:
            raise ValueError(f"selection JSON label {d['label']!r} does not match its fields ({spec.label!r})")
        return spec

    def with_side(self, side: str, **changes: Any) -> "SelectionSpec":
        """A copy with one rule's switches changed (``side`` "v2" or "v2c")."""
        if side == "v2":
            return SelectionSpec(self.rule, self.variant, self.v2.with_(**changes), self.v2c)
        if side == "v2c":
            return SelectionSpec(self.rule, self.variant, self.v2, self.v2c.with_(**changes))
        raise ValueError(f"side must be 'v2' or 'v2c', got {side!r}")


DEFAULT_SELECTION = SelectionSpec("v2", "B", V2_DEFAULT, V2C_DEFAULT)


def parse_selection(text: Optional[str]) -> Optional[SelectionSpec]:
    """A command-line selection: None for "" / None / a label equivalent to the default (the run is then the
    pre-specified one, byte-identical); the spec otherwise. ValueError on a bad label."""
    if text is None or not str(text).strip():
        return None
    spec = SelectionSpec.from_label(str(text))
    return None if spec.is_default else spec


def selection_thresholds() -> Tuple[Any, ...]:
    """Every threshold a selection's verdict depends on, read from the two rule modules (a threshold edit changes
    every hash)."""
    from tools import mps_axial_clusters as ac
    from tools import mps_axial_precision as ap
    return (ap.V2_SPUR_VIABLE, ap.V2_SPUR_MARGINAL, ac.V2C_SPUR_VIABLE, ac.V2C_SPUR_MARGINAL, ac.V2C_K_MIN,
            ac.V2C_H_OVER_P, ap.VIABILITY_RULE_V2, ac.V2C_RULE)


def selection_source_sha(base: str = "") -> str:
    """sha256 (12 hex) of what decides an exploratory selection's verdicts: ``base`` (the batch's rule-v2 source
    hash), this module and ``tools.mps_axial_clusters`` (rule v2c), and the thresholds."""
    import inspect
    from tools import mps_axial_clusters as ac
    from tools import mps_sizer
    import sys
    try:
        src = str(base) + inspect.getsource(sys.modules[__name__]) + inspect.getsource(ac) + inspect.getsource(mps_sizer)
    except (OSError, TypeError):
        return "unavailable"
    src += repr(selection_thresholds())
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


# ============================================================================ the per-pair criteria (already computed)
_NAN = float("nan")


def _f(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _NAN
    return f


@dataclass(frozen=True)
class SideCriteria:
    """
    The criteria of one consecutive pair under one rule side: "v2" (localizations) or the v2c variant "A" / "B" /
    "C" (clusters). The booleans are the rule's own (for A and C ``valley`` is already False when the valley lies in
    the slab gap: never re-derived), ``verdict`` / ``reasons`` the rule's verbatim; the numbers are for display and
    for the reasons of an exploratory selection.
    """

    side: str
    peak_a: bool
    peak_b: bool
    valley: bool
    count_ok: bool
    verdict: str
    reasons: Tuple[str, ...]
    # rule v2 (localization fractions of the central ring)
    f_a: float = _NAN
    f_b: float = _NAN
    f_min: float = _NAN
    x: float = _NAN
    x_note: str = ""
    # rule v2c (cluster counts)
    valley_raw: bool = False
    valley_at_cut: bool = False
    k_a: int = 0
    k_b: int = 0
    k_ratio_a: float = _NAN
    k_ratio_b: float = _NAN
    k_min: float = _NAN
    k_central: int = 0

    @property
    def peak(self) -> bool:
        return bool(self.peak_a and self.peak_b)

    def passes(self, criterion: str) -> bool:
        """peak / valley / count (the leak lives on the pair: ``leak_value``)."""
        if criterion == "peak":
            return self.peak
        if criterion == "valley":
            return bool(self.valley)
        if criterion == "count":
            return bool(self.count_ok)
        raise ValueError(f"passes: criterion {criterion!r} is not peak / valley / count")


@dataclass(frozen=True)
class PairCriteria:
    """One consecutive ring pair: its v2 side, its three v2c sides (None when v2c was not computed or failed:
    ``v2c_error``), and the two leak estimates (D-39 per ring, tnfix per cluster)."""

    ring_a: int
    ring_b: int
    p_ref_nm: float
    d39: float
    tnfix: float
    v2: SideCriteria
    v2c: Optional[Tuple[SideCriteria, ...]] = None
    v2c_error: str = ""

    @property
    def key(self) -> Tuple[int, int]:
        return (int(self.ring_a), int(self.ring_b))

    def side(self, name: str) -> Optional[SideCriteria]:
        if name == "v2":
            return self.v2
        if name not in VARIANTS:
            raise ValueError(f"side must be one of {SIDES}, got {name!r}")
        if self.v2c is None:
            return None
        for s in self.v2c:
            if s.side == name:
                return s
        return None


@dataclass(frozen=True)
class AxonCriteria:
    """The criteria of every consecutive pair of one axon, the two rules' texts and why v2c is missing (if it is)."""

    axon_id: str
    n_rings: int
    pairs: Tuple[PairCriteria, ...]
    v2_rule: str
    v2c_rule: str = ""
    v2c_error: str = ""

    @property
    def has_v2c(self) -> bool:
        return not self.v2c_error and all(p.v2c is not None for p in self.pairs)

    # ---- rows (the explorer cache, the batch sidecar)
    def to_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for p in self.pairs:
            r: Dict[str, Any] = {"axon_id": self.axon_id, "n_rings": self.n_rings, "v2_rule": self.v2_rule,
                                 "v2c_rule": self.v2c_rule, "v2c_error": self.v2c_error or p.v2c_error,
                                 "ring_a": p.ring_a, "ring_b": p.ring_b, "p_ref_nm": p.p_ref_nm, "d39": p.d39,
                                 "tnfix": p.tnfix}
            for s in (p.v2,) + tuple(p.v2c or ()):
                for f in dataclasses.fields(SideCriteria):
                    if f.name == "side":
                        continue
                    v = getattr(s, f.name)
                    r[f"{s.side}_{f.name}"] = list(v) if isinstance(v, tuple) else v
            rows.append(r)
        return rows

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]], *, axon_id: str = "", n_rings: int = 0,
                  v2_rule: str = "", v2c_rule: str = "", v2c_error: str = "") -> "AxonCriteria":
        pairs: List[PairCriteria] = []
        for r in rows:
            sides: Dict[str, SideCriteria] = {}
            for name in SIDES:
                if f"{name}_verdict" not in r or r.get(f"{name}_verdict") in (None, ""):
                    continue
                kw: Dict[str, Any] = {}
                for f in dataclasses.fields(SideCriteria):
                    if f.name == "side":
                        continue
                    v = r.get(f"{name}_{f.name}")
                    if f.name == "reasons":
                        kw[f.name] = tuple(str(x) for x in (v or ()))
                    elif f.type in ("bool",):
                        kw[f.name] = _bool_of(v)
                    elif f.type in ("int",):
                        kw[f.name] = int(_f(v)) if math.isfinite(_f(v)) else 0
                    elif f.type in ("float",):
                        kw[f.name] = _f(v)
                    else:
                        kw[f.name] = "" if v is None else str(v)
                sides[name] = SideCriteria(side=name, **kw)
            v2c = tuple(sides[k] for k in VARIANTS if k in sides) or None
            pairs.append(PairCriteria(int(r["ring_a"]), int(r["ring_b"]), _f(r.get("p_ref_nm")), _f(r.get("d39")),
                                      _f(r.get("tnfix")), sides["v2"], v2c if v2c and len(v2c) == 3 else None,
                                      str(r.get("v2c_error") or "")))
            axon_id = axon_id or str(r.get("axon_id") or "")
            n_rings = n_rings or int(_f(r.get("n_rings")) if math.isfinite(_f(r.get("n_rings"))) else 0)
            v2_rule = v2_rule or str(r.get("v2_rule") or "")
            v2c_rule = v2c_rule or str(r.get("v2c_rule") or "")
            v2c_error = v2c_error or str(r.get("v2c_error") or "")
        return cls(axon_id, n_rings, tuple(pairs), v2_rule, v2c_rule, v2c_error)

    def to_json(self) -> Dict[str, Any]:
        def clean(v: Any) -> Any:
            if isinstance(v, float) and not math.isfinite(v):
                return None
            if isinstance(v, (np.floating,)):
                return clean(float(v))
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v
        return {"schema": SELECTION_SCHEMA, "axon_id": self.axon_id, "n_rings": self.n_rings,
                "v2_rule": self.v2_rule, "v2c_rule": self.v2c_rule, "v2c_error": self.v2c_error,
                "rows": [{k: clean(v) for k, v in r.items()} for r in self.to_rows()]}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "AxonCriteria":
        return cls.from_rows(d.get("rows", []), axon_id=str(d.get("axon_id", "")), n_rings=int(d.get("n_rings", 0)),
                             v2_rule=str(d.get("v2_rule", "")), v2c_rule=str(d.get("v2c_rule", "")),
                             v2c_error=str(d.get("v2c_error", "")))


def _bool_of(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


def _same(a: float, b: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return True
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def axon_criteria(v2: Any, v2c: Any = None, v2c_error: str = "", *, axon_id: str = "") -> AxonCriteria:
    """
    The per-pair criteria of one axon from objects already computed: ``v2`` an ``AxonViabilityV2``, ``v2c`` its
    ``AxonViabilityV2C`` (None with ``v2c_error`` saying why). v2c is read at h = ``V2C_H_OVER_P`` P only (its
    verdict rows, never ``sensitivity_pairs``). Pairs are matched by (ring_a, ring_b); a key-set mismatch, a
    repeated pair or two different D-39 numbers for one pair raise ValueError.
    """
    p_ref = _f(getattr(v2, "p_ref_nm", _NAN))
    by_v2c: Dict[Tuple[int, int], Dict[str, Any]] = {}
    tnfix: Dict[Tuple[int, int], float] = {}
    if v2c is not None:
        from tools.mps_axial_clusters import V2C_VARIANT_LETTERS
        for vp in v2c.pairs:
            key = (int(vp.ring_a), int(vp.ring_b))
            letter = V2C_VARIANT_LETTERS[str(vp.variant)]
            slot = by_v2c.setdefault(key, {})
            if letter in slot:
                raise ValueError(f"axon_criteria: v2c pair {key} variant {letter} given twice")
            slot[letter] = vp
        for lk in v2c.leaks:
            tnfix[(int(lk.ring_a), int(lk.ring_b))] = _f(lk.exp_spur_frac_clusters)
    v2_keys = [(int(p.ring_a), int(p.ring_b)) for p in v2.pairs]
    if len(set(v2_keys)) != len(v2_keys):
        raise ValueError(f"axon_criteria: rule v2 gives a pair twice ({v2_keys})")
    if v2c is not None and set(by_v2c) != set(v2_keys):
        raise ValueError(f"axon_criteria: rule v2 pairs {sorted(v2_keys)} and rule v2c pairs {sorted(by_v2c)} differ")
    pairs: List[PairCriteria] = []
    for p in v2.pairs:
        key = (int(p.ring_a), int(p.ring_b))
        d39 = _f(p.exp_spur_frac)
        s2 = SideCriteria(side="v2", peak_a=bool(p.peak_a), peak_b=bool(p.peak_b), valley=bool(p.valley),
                          count_ok=bool(p.count_ok), verdict=str(p.verdict), reasons=tuple(str(r) for r in p.reasons),
                          f_a=_f(p.f_a), f_b=_f(p.f_b), f_min=_f(p.f_min), x=_f(p.x),
                          x_note=str(getattr(v2, "x_note", "") or ""))
        sides: Optional[Tuple[SideCriteria, ...]] = None
        if v2c is not None:
            slot = by_v2c[key]
            if set(slot) != set(VARIANTS):
                raise ValueError(f"axon_criteria: v2c pair {key} has variants {sorted(slot)}, expected {VARIANTS}")
            out = []
            for letter in VARIANTS:
                vp = slot[letter]
                if not _same(_f(vp.exp_spur_frac_d39), d39):
                    raise ValueError(f"axon_criteria: pair {key}: v2c carries D-39 {vp.exp_spur_frac_d39!r}, rule v2 "
                                     f"{d39!r} (computed from another v2?)")
                var = v2c.variants[VARIANT_KEYS[letter]]
                out.append(SideCriteria(
                    side=letter, peak_a=bool(vp.peak_a), peak_b=bool(vp.peak_b), valley=bool(vp.valley),
                    count_ok=bool(vp.count_ok), verdict=str(vp.verdict), reasons=tuple(str(r) for r in vp.reasons),
                    valley_raw=bool(vp.valley_raw), valley_at_cut=bool(vp.valley_at_cut), k_a=int(vp.k_a),
                    k_b=int(vp.k_b), k_ratio_a=_f(vp.k_ratio_a), k_ratio_b=_f(vp.k_ratio_b), k_min=_f(vp.k_min),
                    k_central=int(var.k_central)))
            sides = tuple(out)
        pairs.append(PairCriteria(key[0], key[1], p_ref, d39, tnfix.get(key, _NAN), s2, sides,
                                  "" if v2c is not None else str(v2c_error)))
    return AxonCriteria(axon_id=str(axon_id), n_rings=len(getattr(v2, "rings", ())), pairs=tuple(pairs),
                        v2_rule=str(v2.rule), v2c_rule=str(getattr(v2c, "rule", "") or ""),
                        v2c_error="" if v2c is not None else str(v2c_error))


def compute_axon_criteria(res: Any, xyz_lab_nm: Optional[Sequence[Any]], *, zq: Any = None, v2: Any = None,
                          v2c: Any = None, with_v2c: bool = True, axon_id: str = "",
                          p_ref_nm: Optional[float] = None) -> AxonCriteria:
    """
    The per-pair criteria of a ``build_rings`` result, computing what is not given: ``z_quality``, ``viability_v2``
    (lab coordinates ``xyz_lab_nm``) and, when ``with_v2c``, ``viability_v2c(res, v2, xyz_lab_nm)`` -- whose failure
    (e.g. a ring with a non-finite sigma_z, review N2) is recorded in ``v2c_error`` and never falls back to v2.
    Geometry only (R8).
    """
    from tools.mps_axial_precision import viability_v2, z_quality
    xyz = None if xyz_lab_nm is None else tuple(np.asarray(v, dtype=np.float64).reshape(-1) for v in xyz_lab_nm)
    if v2 is None:
        if zq is None:
            zq = z_quality(res) if p_ref_nm is None else z_quality(res, p_ref_nm=p_ref_nm)
        v2 = viability_v2(res, zq, xyz_lab_nm=xyz, p_ref_nm=p_ref_nm)
    err = ""
    if v2c is None and with_v2c:
        try:
            from tools.mps_axial_clusters import viability_v2c
            v2c = viability_v2c(res, v2, xyz_lab_nm=xyz)
        except Exception as exc:  # noqa: BLE001 - recorded; a selection that needs v2c then selects nothing
            v2c, err = None, f"{type(exc).__name__}: {exc}"
    elif v2c is None:
        err = "not requested"
    return axon_criteria(v2, v2c, err, axon_id=axon_id)


# ============================================================================ the pure evaluation
def leak_value(pc: PairCriteria, estimate: str) -> float:
    """Criterion 4's number under one estimate: D-39 (per ring), tnfix (per cluster) or the larger of the two (NaN
    when either is NaN: unknown fails)."""
    if estimate == "D-39":
        return _f(pc.d39)
    if estimate == "tnfix":
        return _f(pc.tnfix)
    if estimate == "max":
        a, b = _f(pc.d39), _f(pc.tnfix)
        return max(a, b) if math.isfinite(a) and math.isfinite(b) else _NAN
    raise ValueError(f"leak estimate must be one of {LEAK_ESTIMATES}, got {estimate!r}")


def _spur_thresholds(side: str) -> Tuple[float, float]:
    if side == "v2":
        from tools.mps_axial_precision import V2_SPUR_MARGINAL, V2_SPUR_VIABLE
        return float(V2_SPUR_VIABLE), float(V2_SPUR_MARGINAL)
    from tools.mps_axial_clusters import V2C_SPUR_MARGINAL, V2C_SPUR_VIABLE
    return float(V2C_SPUR_VIABLE), float(V2C_SPUR_MARGINAL)


@dataclass(frozen=True)
class SideEvaluation:
    """One pair under one side of a selection: the tier, the CHECKED criteria that fail, the UNCHECKED ones that
    would fail (shown, never used) and the plain reasons."""

    side: str
    verdict: str
    failing: Tuple[str, ...]
    unchecked_failing: Tuple[str, ...]
    reasons: Tuple[str, ...]


_LEAK_NAME = {"D-39": "expected leak copies", "tnfix": "cluster-level leak copies",
              "max": "leak copies (the larger of D-39 and the cluster-level tnfix)"}


def _leak_reason(side: str, est: str, spur: float, crit: bool) -> Optional[str]:
    """The leak reason in the rules' own words (verdict_v2 for D-39 on v2, verdict_v2c for tnfix on v2c)."""
    viable, marginal = _spur_thresholds(side)
    name = _LEAK_NAME[est]
    if not math.isfinite(spur):
        if est == "D-39":
            return "expected leak copies unknown (no axial mixture component)"
        if est == "tnfix":
            return "cluster-level leak copies unknown (no ring clusters)"
        return "leak copies unknown (D-39 or the cluster-level tnfix not computable)"
    if spur > marginal:
        return f"{name} {100 * spur:.1f} % > {100 * marginal:g} %"
    if spur > viable:
        if side == "v2" and est == "D-39":
            tail = "(marginal: sensitivity only, D-39 d)" if crit else f"(VIABLE needs <= {100 * viable:g} %)"
            return f"{name} {100 * spur:.1f} % > {100 * viable:g} % " + tail
        tail = " (marginal)" if crit else f" (VIABLE needs <= {100 * viable:g} %)"
        return f"{name} {100 * spur:.1f} % > {100 * viable:g} %" + tail
    return None


def _criteria_reasons(pc: PairCriteria, s: SideCriteria, cs: CriteriaSet) -> List[str]:
    """The reasons of the CHECKED peak / valley / count criteria, in the rules' own words (the rule's verdict
    function with the unchecked criteria set to pass and no leak)."""
    if s.side == "v2":
        from tools.mps_axial_precision import verdict_v2
        _v, reasons = verdict_v2(
            rings=pc.key, peak_a=(s.peak_a if cs.peak else True), peak_b=(s.peak_b if cs.peak else True),
            valley=(s.valley if cs.valley else True), f_a=(s.f_a if cs.count else 1.0),
            f_b=(s.f_b if cs.count else 1.0), f_min=(s.f_min if cs.count else 0.0), exp_spur_frac=0.0,
            x=s.x, x_note=s.x_note, p_ref_nm=(pc.p_ref_nm if math.isfinite(pc.p_ref_nm) else P_REF_DEFAULT_NM))
        return list(reasons)
    from tools.mps_axial_clusters import verdict_v2c
    _v, _valley, _count, reasons = verdict_v2c(
        variant=VARIANT_KEYS[s.side], rings=pc.key, peak_a=(s.peak_a if cs.peak else True),
        peak_b=(s.peak_b if cs.peak else True), valley_raw=(s.valley_raw if cs.valley else True),
        valley_at_cut_=(s.valley_at_cut if cs.valley else False), k_ratio_a=(s.k_ratio_a if cs.count else 1.0),
        k_ratio_b=(s.k_ratio_b if cs.count else 1.0), k_a=s.k_a, k_b=s.k_b,
        k_central=(s.k_central if cs.count else max(1, s.k_central)), exp_spur_frac_clusters_=0.0,
        k_min=(s.k_min if math.isfinite(s.k_min) else 0.89),
        p_ref_nm=(pc.p_ref_nm if math.isfinite(pc.p_ref_nm) else P_REF_DEFAULT_NM))
    return list(reasons)


def evaluate_side(pc: PairCriteria, side: str, cs: CriteriaSet) -> SideEvaluation:
    """
    One pair under one side: a checked criterion fails when its boolean is False (the leak: its value is not finite
    or above the MARGINAL threshold). When every checked one of peak / valley / count passes: leak checked -> VIABLE
    (<= 2 %), MARGINAL (2-5 %), else NOT VIABLE; leak unchecked -> VIABLE. Otherwise NOT VIABLE.
    """
    s = pc.side(side)
    if s is None:
        err = pc.v2c_error or "no v2c result for this pair"
        return SideEvaluation(side, VERDICT_NOT_VIABLE, ("v2c",), (), (f"{V2C_NOT_COMPUTED}: {err}",))
    viable_t, marginal_t = _spur_thresholds(side)
    failing: List[str] = []
    unchecked: List[str] = []
    for c in ("peak", "valley", "count"):
        ok = s.passes(c)
        if not ok:
            (failing if cs.checked(c) else unchecked).append(c)
    spur = leak_value(pc, cs.leak_estimate)
    leak_fails = not (math.isfinite(spur) and spur <= marginal_t)
    if cs.leak:
        if leak_fails:
            failing.append("leak")
    elif not (math.isfinite(spur) and spur <= viable_t):
        unchecked.append("leak")
    crit = not any(c in failing for c in ("peak", "valley", "count"))
    if not crit:
        verdict = VERDICT_NOT_VIABLE
    elif not cs.leak:
        verdict = VERDICT_VIABLE
    elif math.isfinite(spur) and spur <= viable_t:
        verdict = VERDICT_VIABLE
    elif math.isfinite(spur) and spur <= marginal_t:
        verdict = VERDICT_MARGINAL
    else:
        verdict = VERDICT_NOT_VIABLE
    reasons = _criteria_reasons(pc, s, cs)
    if cs.leak:
        lr = _leak_reason(side, cs.leak_estimate, spur, crit)
        if lr:
            reasons.append(lr)
    if unchecked:
        parts = []
        for c in unchecked:
            if c == "leak":
                parts.append(f"leak ({cs.leak_estimate} {100 * spur:.1f} %)" if math.isfinite(spur)
                             else f"leak ({cs.leak_estimate} unknown)")
            else:
                parts.append(f"{c} (would fail)")
        reasons.append("not checked (exploratory): " + ", ".join(parts))
    return SideEvaluation(side, verdict, tuple(failing), tuple(unchecked), tuple(reasons))


@dataclass(frozen=True)
class PairSelection:
    """One pair under a selection (duck-compatible with ``batch_columns.PairVerdict``: ring_a, ring_b, verdict,
    reasons): its tier ("primary" / "sensitivity" / "not analysed"), whether it is read at all, the failing and
    the unchecked-but-failing criteria (prefixed "v2:" / "v2c-B:" under AND / OR) and the reasons."""

    ring_a: int
    ring_b: int
    verdict: str
    tier: str
    selected: bool
    failing: Tuple[str, ...]
    unchecked_failing: Tuple[str, ...]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class SelectionResult:
    """A selection on one axon: the spec, its label / hash, the rule text the outputs carry, every pair."""

    spec: SelectionSpec
    axon_id: str
    n_rings: int
    pairs: Tuple[PairSelection, ...]
    rule_text: str

    @property
    def label(self) -> str:
        return self.spec.label

    @property
    def hash(self) -> str:
        return self.spec.hash

    @property
    def exploratory(self) -> bool:
        return self.spec.exploratory

    @property
    def rule(self) -> str:
        return self.rule_text

    def of(self, verdict: str) -> List[Tuple[int, int]]:
        return sorted((p.ring_a, p.ring_b) for p in self.pairs if p.verdict == verdict)

    @property
    def viable(self) -> List[Tuple[int, int]]:
        return self.of(VERDICT_VIABLE)

    @property
    def marginal(self) -> List[Tuple[int, int]]:
        return self.of(VERDICT_MARGINAL)

    @property
    def selected(self) -> List[Tuple[int, int]]:
        return sorted(self.viable + self.marginal)

    def key(self, base_mode: str) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
        """(number of rings, selected pairs sorted): exactly ``viability_v2_key``'s form; ``base_mode`` "viable" or
        "viable+marginal"."""
        if base_mode == "viable":
            return self.n_rings, tuple(self.viable)
        if base_mode == "viable+marginal":
            return self.n_rings, tuple(sorted(self.viable + self.marginal))
        raise ValueError(f"base_mode must be 'viable' or 'viable+marginal', got {base_mode!r}")

    def summary(self) -> str:
        """"k of n pairs selected (0-1 VIABLE, 1-2 MARGINAL)"."""
        v, m = self.viable, self.marginal
        parts = ([f"{' '.join(f'{a}-{b}' for a, b in v)} VIABLE"] if v else []) + \
                ([f"{' '.join(f'{a}-{b}' for a, b in m)} MARGINAL"] if m else [])
        return f"{len(v) + len(m)} of {len(self.pairs)} pairs selected" + (f" ({'; '.join(parts)})" if parts else "")


def _combine(rule: str, evs: Sequence[SideEvaluation]) -> str:
    ranks = [_TIER_RANK[e.verdict] for e in evs]
    r = min(ranks) if rule == "both" else max(ranks)
    return next(k for k, v in _TIER_RANK.items() if v == r)


def evaluate_selection(crit: AxonCriteria, spec: SelectionSpec) -> SelectionResult:
    """
    Every pair of one axon under ``spec``: "v2" / "v2c" read one side, "both" the worse tier of the two sides,
    "either" the better. Reasons: the default selection gives rule v2's reasons verbatim; a v2c side alone with its
    four D-42 criteria and tnfix gives v2c's verbatim; otherwise they are written per failing criterion in the
    rules' words, then "not checked (exploratory): ..." (prefixed "v2: " / "v2c-B: " under AND / OR).
    """
    full_v2c = spec.rule == "v2c" and spec.v2c == V2C_DEFAULT
    out: List[PairSelection] = []
    for pc in crit.pairs:
        evs: List[SideEvaluation] = []
        if spec.uses_v2:
            evs.append(evaluate_side(pc, "v2", spec.v2))
        if spec.uses_v2c:
            evs.append(evaluate_side(pc, spec.variant, spec.v2c))
        verdict = evs[0].verdict if len(evs) == 1 else _combine(spec.rule, evs)
        if spec.is_default:
            reasons = pc.v2.reasons
        elif full_v2c and pc.side(spec.variant) is not None:
            side = pc.side(spec.variant)
            assert side is not None
            reasons = side.reasons
        elif len(evs) == 1:
            reasons = evs[0].reasons
        else:
            reasons = tuple(f"{'v2' if e.side == 'v2' else 'v2c-' + e.side}: {r}" for e in evs for r in e.reasons)
        pre = (lambda e: "") if len(evs) == 1 else (lambda e: ("v2:" if e.side == "v2" else f"v2c-{e.side}:"))
        failing = tuple(pre(e) + f for e in evs for f in e.failing)
        unchecked = tuple(pre(e) + f for e in evs for f in e.unchecked_failing)
        out.append(PairSelection(pc.ring_a, pc.ring_b, verdict, TIER_OF[verdict], verdict != VERDICT_NOT_VIABLE,
                                 failing, unchecked, tuple(reasons)))
    return SelectionResult(spec, crit.axon_id, crit.n_rings, tuple(out), spec.rule_text(crit.v2_rule, crit.v2c_rule))


# ============================================================================ the combination table (explorer)
@dataclass(frozen=True)
class CombinationCell:
    """One cell of the explorer's table: a subset of the four criteria under one rule column."""

    criteria: Tuple[str, ...]
    column: str
    n_viable: int
    n_viable_marginal: int
    n_axons_viable: int
    n_pairs: int
    n_axons: int

    @property
    def text(self) -> str:
        return f"{self.n_viable} ({self.n_viable_marginal}) | {self.n_axons_viable}"


def criteria_subsets() -> List[Tuple[str, ...]]:
    """The 16 subsets of the four criteria, strictest first (4, then 3, 2, 1, 0 criteria; CRITERIA order)."""
    out: List[Tuple[str, ...]] = []
    for k in range(len(CRITERIA), -1, -1):
        out.extend(itertools.combinations(CRITERIA, k))
    return out


COMBINATION_COLUMNS: Tuple[str, ...] = ("v2", "v2c-A", "v2c-B", "v2c-C")


def spec_of_cell(criteria: Sequence[str], column: str, leak_v2: str = "D-39", leak_v2c: str = "tnfix") -> SelectionSpec:
    """The selection a cell of the combination table stands for."""
    on = {c: (c in criteria) for c in CRITERIA}
    if column == "v2":
        return SelectionSpec("v2", "B", CriteriaSet(**on, leak_estimate=leak_v2), V2C_DEFAULT)
    if column.startswith("v2c-") and column[4:] in VARIANTS:
        return SelectionSpec("v2c", column[4:], V2_DEFAULT, CriteriaSet(**on, leak_estimate=leak_v2c))
    raise ValueError(f"column must be one of {COMBINATION_COLUMNS}, got {column!r}")


def combination_table(crits: Sequence[AxonCriteria], leak_v2: str = "D-39",
                      leak_v2c: str = "tnfix") -> List[CombinationCell]:
    """For each of the 16 criteria subsets and each rule column (v2, v2c-A, v2c-B, v2c-C): pairs VIABLE, pairs
    VIABLE + MARGINAL, axons with >= 1 VIABLE pair. Pure loops (geometry only)."""
    cells: List[CombinationCell] = []
    n_pairs = sum(len(c.pairs) for c in crits)
    for sub in criteria_subsets():
        for col in COMBINATION_COLUMNS:
            spec = spec_of_cell(sub, col, leak_v2, leak_v2c)
            nv = nvm = na = 0
            for c in crits:
                r = evaluate_selection(c, spec)
                v = len(r.viable)
                nv += v
                nvm += v + len(r.marginal)
                na += int(v > 0)
            cells.append(CombinationCell(tuple(sub), col, nv, nvm, na, n_pairs, len(crits)))
    return cells


# ============================================================================ the selector's plain warnings
WARN, INFO, BAD = "warn", "info", "bad"
WARNING_AC_VALLEY = ("v2c-A / v2c-C cannot pass the cluster valley criterion for similar rings (slab cut at the "
                     "valley): uncheck (2c) to use them")
WARNING_B = "v2c-B: selection depends on whether columns exist: lower power, exploratory"
WARNING_NO_LEAK = ("leak not checked: leak copies sit in register with their source cluster and inflate column "
                   "matches (false columns)")
WARNING_D39 = "D-39 assumes flat rings and can under-estimate copies for non-flat rings (Q-39 open)"
WARNING_TNFIX = "tnfix never under-estimated in simulation, over-estimates ~3x"
WARNING_EXPLORATORY = ("Exploratory selection: not the pre-specified rule v2[peak,valley,count,leak:D-39]; p values are "
                       "not corrected for trying several selections")


def selector_warnings(spec: SelectionSpec) -> List[Tuple[str, str]]:
    """The plain warnings shown next to the selector, (level, text) -- the only place these texts live."""
    out: List[Tuple[str, str]] = []
    if spec.uses_v2c and spec.variant in ("A", "C") and spec.v2c.valley:
        out.append((BAD, WARNING_AC_VALLEY))
    if spec.uses_v2c and spec.variant == "B":
        out.append((WARN, WARNING_B))
    sides = ([spec.v2] if spec.uses_v2 else []) + ([spec.v2c] if spec.uses_v2c else [])
    if any(not cs.leak for cs in sides):
        out.append((BAD, WARNING_NO_LEAK))
    ests = {cs.leak_estimate for cs in sides if cs.leak}
    if ests & {"D-39"}:
        out.append((INFO, WARNING_D39))
    if ests & {"tnfix", "max"}:
        out.append((INFO, WARNING_TNFIX))
    if spec.exploratory:
        out.append((WARN, WARNING_EXPLORATORY))
    return out


# ============================================================================ plumbing helpers (review window, batch)
def selection_viability(base: Any, result: SelectionResult) -> Any:
    """
    The review window's viability under a selection: ``base`` (a ``ReviewViability``-like dataclass with ``rule``
    and ``pairs``) with the selection's rule text and one ``batch_columns.PairVerdict`` per pair; its ``selection`` /
    ``spec`` fields are set when it has them. Under the default selection it equals ``base`` field by field.
    """
    import batch_columns as bc
    pairs = tuple(bc.PairVerdict(int(p.ring_a), int(p.ring_b), str(p.verdict), tuple(p.reasons)) for p in result.pairs)
    changes: Dict[str, Any] = {"rule": result.rule_text, "pairs": pairs}
    names = {f.name for f in dataclasses.fields(base)} if dataclasses.is_dataclass(base) else set()
    if "selection" in names:
        changes["selection"] = None if result.spec.is_default else result
    if "spec" in names:
        changes["spec"] = None if result.spec.is_default else result.spec
    return dataclasses.replace(base, **changes)


def selection_tiers(arc: Any, result: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(primary, sensitivity) of an arc result under a selection (or any object with ``viable`` / ``marginal``),
    exactly as the review window and the batch compute them: the joint restricted to the VIABLE pairs, and to
    VIABLE + MARGINAL when there are marginal pairs (``batch_columns.restricted_joint``; sub-millisecond, so a toggle
    re-renders without re-running the analyses)."""
    if arc is None:
        return None, None
    from batch_columns import restricted_joint
    viable, marginal = list(result.viable), list(result.marginal)
    primary = restricted_joint(arc, viable) if viable else None
    sensitivity = restricted_joint(arc, viable + marginal) if marginal else None
    return primary, sensitivity


def lumen_fp(snapshot: Any) -> Tuple[Tuple[str, ...], str]:
    """The fingerprint of a lumen snapshot the review's analyses are cached under: (removed keys, cluster set)."""
    return tuple(str(k) for k in snapshot.final_removed_keys()), str(snapshot.cluster_set_sha)


def lumen_fp_text(fp: Tuple[Tuple[str, ...], str]) -> str:
    """A short text of a lumen fingerprint for the log: sha256 (12 hex) of its parts."""
    keys, sha = fp
    return hashlib.sha256((sha + "\n" + "\n".join(keys)).encode("utf-8")).hexdigest()[:12]


SELECTION_PAIR_COLUMNS: Tuple[str, ...] = (
    "pair_key", "input_key", "axon_id", "selection_label", "selection_hash", "ring_a", "ring_b", "verdict", "tier",
    "failing", "unchecked_failing", "reasons", "p_ref_nm", "d39", "tnfix", "v2c_error") + tuple(
    f"{s}_{f.name}" for s in SIDES for f in dataclasses.fields(SideCriteria) if f.name not in ("side",))


def selection_pair_rows(crit: AxonCriteria, result: SelectionResult, *, input_key: str = "") -> List[Dict[str, Any]]:
    """One row per pair (``SELECTION_PAIR_COLUMNS``): the selection, the verdict, and every criterion of every side
    (the batch sidecar ``selection_pairs.csv``)."""
    by = {(int(p.ring_a), int(p.ring_b)): p for p in result.pairs}
    rows: List[Dict[str, Any]] = []
    for base in crit.to_rows():
        key = (int(base["ring_a"]), int(base["ring_b"]))
        ps = by[key]
        row = {k: v for k, v in base.items() if k in SELECTION_PAIR_COLUMNS}
        for k, v in list(row.items()):
            if isinstance(v, list):
                row[k] = "; ".join(str(x) for x in v)
        row.update(pair_key=f"{input_key}|{key[0]}-{key[1]}", input_key=input_key, axon_id=crit.axon_id,
                   selection_label=result.label, selection_hash=result.hash, verdict=ps.verdict, tier=ps.tier,
                   failing=" ".join(ps.failing), unchecked_failing=" ".join(ps.unchecked_failing),
                   reasons="; ".join(ps.reasons))
        rows.append(row)
    return rows


# ============================================================================ the exploration log
LOG_FILE = "selection_exploration_log.csv"
LOG_COLUMNS: Tuple[str, ...] = (
    "row_id", "time_iso", "session_id", "source", "axon_id", "input_sha256", "lumen_fp", "selection_label",
    "selection_hash", "is_default", "n_selected", "viable_pairs", "marginal_pairs", "primary_z_A", "primary_p_excess",
    "primary_p_two_sided", "sensitivity_z_A", "sensitivity_p_excess", "pair_p", "simnull_p", "n_null", "git_head")
SESSION_ENV = "MPS_SELECTION_SESSION_ID"
LOG_DIR_ENV = "MPS_SELECTION_LOG_DIR"
# one id per program session; a dialog's subprocess inherits it through the environment (``ensure_session_env``),
# so "this session" covers the window and the batch / simnull runs it starts
SESSION_ID: str = os.environ.get(SESSION_ENV) or uuid.uuid4().hex


def ensure_session_env() -> str:
    """Put this session's id in the environment (children started afterwards log under it); returns it."""
    os.environ[SESSION_ENV] = SESSION_ID
    return SESSION_ID


def default_log_path(store_dir: Optional[str] = None) -> str:
    """``$MPS_SELECTION_LOG_DIR/selection_exploration_log.csv`` when set (tests), else ``<store_dir>/selection_log/``
    (the review store of the GUI), else ``~/.mps_explorer/selection_log/``."""
    env = os.environ.get(LOG_DIR_ENV)
    if env:
        folder = env
    elif store_dir:
        folder = os.path.join(store_dir, "selection_log")
    else:
        folder = os.path.join(os.path.expanduser("~"), ".mps_explorer", "selection_log")
    return os.path.join(folder, LOG_FILE)


def counter_text(n: int) -> str:
    """The note shown next to every p."""
    return (f"{int(n)} selection variant{'s' if int(n) != 1 else ''} tried this session on this axon - p values are "
            "not corrected for trying several selections")


def pair_p_text(pairs: Iterable[Tuple[int, int, float]]) -> str:
    """"0-1:0.0123;1-2:0.4" (NaN written "nan")."""
    out = []
    for a, b, p in pairs:
        f = _f(p)
        out.append(f"{int(a)}-{int(b)}:{f:.6g}" if math.isfinite(f) else f"{int(a)}-{int(b)}:nan")
    return ";".join(out)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "True" if bool(v) else "False"
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return "" if math.isnan(f) else repr(f)
    return str(v)


def log_row(*, source: str, spec: SelectionSpec, axon_id: str = "", input_sha256: str = "", lumen: str = "",
            viable: Sequence[Tuple[int, int]] = (), marginal: Sequence[Tuple[int, int]] = (),
            primary: Optional[Mapping[str, Any]] = None, sensitivity: Optional[Mapping[str, Any]] = None,
            pair_p: str = "", simnull_p: Any = None, n_null: Any = None, git_head: str = "") -> Dict[str, Any]:
    """One row of the exploration log (``LOG_COLUMNS``) from what a result shows."""
    primary = primary or {}
    sensitivity = sensitivity or {}
    return {"source": source, "axon_id": axon_id, "input_sha256": input_sha256, "lumen_fp": lumen,
            "selection_label": spec.label, "selection_hash": spec.hash, "is_default": spec.is_default,
            "n_selected": len(viable) + len(marginal), "viable_pairs": " ".join(f"{a}-{b}" for a, b in viable),
            "marginal_pairs": " ".join(f"{a}-{b}" for a, b in marginal), "primary_z_A": primary.get("z_A"),
            "primary_p_excess": primary.get("p_excess_uncalibrated"),
            "primary_p_two_sided": primary.get("p_two_sided_uncalibrated"),
            "sensitivity_z_A": sensitivity.get("z_A"), "sensitivity_p_excess": sensitivity.get("p_excess_uncalibrated"),
            "pair_p": pair_p, "simnull_p": simnull_p, "n_null": n_null, "git_head": git_head}


class ExplorationLog:
    """
    The append-only CSV of every selection a column statistic was shown or written under (``LOG_COLUMNS``): opened
    in append mode, the header written only when the file is empty, one write + flush per row, never rewritten (a
    file with another header is refused). ``variants_tried`` counts the distinct selection hashes of one axon in one
    session (the window shows ``counter_text`` next to every p).
    """

    def __init__(self, path: Optional[str] = None, *, session_id: Optional[str] = None) -> None:
        self.path = os.path.abspath(path or default_log_path())
        self.session_id = session_id or SESSION_ID
        self._logged: Set[Tuple[str, ...]] = set()

    def _check_header(self) -> None:
        if os.path.isfile(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "r", encoding="utf-8", newline="") as fh:
                header = next(csv.reader([fh.readline()]), [])
            if tuple(header) != LOG_COLUMNS:
                raise ValueError(f"{self.path}: not an exploration log of this program (header differs); it is never "
                                 "rewritten: move it away or pass another path")

    def append(self, row: Mapping[str, Any]) -> Dict[str, str]:
        """Append one row (row_id, time_iso and session_id filled in when missing); returns what was written."""
        self._check_header()
        rec: Dict[str, Any] = {c: row.get(c) for c in LOG_COLUMNS}
        rec["row_id"] = rec.get("row_id") or uuid.uuid4().hex
        rec["time_iso"] = rec.get("time_iso") or time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["session_id"] = rec.get("session_id") or self.session_id
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        cells = [_cell(rec[c]) for c in LOG_COLUMNS]
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        with open(self.path, "a", encoding="utf-8", newline="") as fh:
            if fh.tell() == 0:
                w.writerow(LOG_COLUMNS)
            w.writerow(cells)
            fh.write(buf.getvalue())
            fh.flush()
        return dict(zip(LOG_COLUMNS, cells))

    def log_once(self, key: Sequence[Any], row: Mapping[str, Any]) -> Optional[Dict[str, str]]:
        """Append unless ``key`` (e.g. (source, axon, lumen fp, selection hash)) was logged by this object already;
        None when skipped."""
        k = tuple(str(v) for v in key)
        if k in self._logged:
            return None
        out = self.append(row)
        self._logged.add(k)
        return out

    def rows(self) -> List[Dict[str, str]]:
        if not os.path.isfile(self.path):
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def variants_tried(self, axon_id: str = "", *, input_sha256: str = "", session_id: Optional[str] = None) -> int:
        """The distinct selection hashes logged for this axon (matched on ``input_sha256`` when given, else on
        ``axon_id``) in this session."""
        sid = session_id or self.session_id
        hashes = set()
        for r in self.rows():
            if r.get("session_id") != sid:
                continue
            if input_sha256 and r.get("input_sha256"):
                if r.get("input_sha256") != input_sha256:
                    continue
            elif r.get("axon_id") != axon_id:
                continue
            hashes.add(r.get("selection_hash", ""))
        return len(hashes)


# ============================================================================ geometry of one file (the explorer)
def geometry_for_file(path: str, *, pixel_size_nm: Optional[float] = None, with_v2c: bool = True) -> AxonCriteria:
    """
    The per-pair criteria of one localization file exactly as the batch builds its rings (``batch_columns``'
    loader, ``build_rings`` with ``rings_params_from(default_columns_params())`` and ``roi=None``, z_quality, rule v2
    on the lab coordinates, rule v2c). Geometry only (R8): no column statistic, so it may run on real axons.
    """
    import batch_columns as bc
    from tools.mps_columns import RingsParams, build_rings, rings_params_from
    from tools.mps_identity import axon_id as axon_id_of
    from tools.mps_lumen import default_columns_params
    kind, _prov = bc.input_kind(path)
    inp = bc._load_input(path, kind, bc.ColumnBatchSettings(pixel_size_nm=pixel_size_nm))
    try:
        rp = rings_params_from(default_columns_params())
    except (OSError, ValueError):
        rp = RingsParams()
    res = build_rings(inp.x_nm.copy(), inp.y_nm.copy(), inp.z_nm.copy(), frame=inp.frame.copy(),
                      lp_lateral_nm=inp.lp_lateral_nm.copy(), lpz_nm=inp.lpz_nm.copy(), params=rp,
                      source_name=str(path), pixel_size_nm=inp.pixel_size_nm,
                      pixel_size_source=str(inp.pixel_size_source), n_frames=inp.n_frames, roi=None)
    xyz = (np.asarray(inp.x_nm, dtype=np.float64), np.asarray(inp.y_nm, dtype=np.float64),
           np.asarray(inp.z_nm, dtype=np.float64))
    return compute_axon_criteria(res, xyz, with_v2c=with_v2c, axon_id=axon_id_of(os.path.basename(path), ""))

