# -*- coding: utf-8 -*-
"""
The "working scale" of one analysed file (plan A9, sec. 1.1; A5 merged into A9).

What it is
----------
The distances below which the program, with THIS file's parameters, cannot tell two objects apart, and above
which the readouts start to be interpretable. Five marks (six with the slab), each with its origin:

==========================  =========================================  ==========================================
mark                        value                                      origin
==========================  =========================================  ==========================================
eps                         the clustering's eps (DBSCAN: 25 nm)       [lit: Gazal, Escalante, et al. 2026]
min_samples                 DBSCAN: 10                                 [lit: Gazal, Escalante, et al. 2026]
sigma NeNA                  measured on the file                       [lit: Endesfelder et al. 2014] + measured
d50                         2 eps + 2.5 sigma_NeNA  (~80 nm)           [simulation: clustering research sec. 8.6,
                                                                       sim3_pairs_resolution; an empirical fit,
                                                                       inference, not literature]
near-total fusion           d50 - 20 nm  (~60 nm)                      [simulation: same table; <= 10 % of the
                                                                       pairs resolved; A5's definition]
axial slab                  +/-90 nm                                   [lit: Gazal, Escalante, et al. 2026]
==========================  =========================================  ==========================================

d50 is the separation at which half of the isolated pairs of two labelled tetramers come out as two clusters.
In the project's simulation (two epitopes, linkage, geometric blinking; 25/10, sigma_loc 12 nm) 10 % of the
pairs were resolved at 60 nm, 49 % at 80 nm and 79 % at 100 nm; the rule d50 ~ 2 eps + 2.5 sigma fits that table
to +/-8 nm except at sigma_loc >= 20 nm, where it UNDER-estimates (predicts 100 nm; the table gives 100 to more
than 120 nm). Below the fusion distance what the readouts see is the algorithm merging molecules, not biology:
the per-axon null (tools.mps_pipeline_null) excludes radii below it from its global test, and a 1NN between
clusters below d50 is an upper bound of the distance between molecules.

Other clustering methods
------------------------
The d50 rule was fitted for DBSCAN. For any other method the program applies the same rule with the reference
pipeline's eps, 25 nm, and labels the result [ad hoc]: every method is then tested on the same radii, which is what
a comparison across methods needs (a method's edge margin is not a resolution scale: HDBSCAN's is its largest core
distance, 221 nm on one synthetic axon, which emptied the Ripley test); the physical limit of every strategy tried in the simulation was
~2.5-3 sigma_eff with sigma_eff = sqrt(sigma_loc^2 + sigma_link^2) [simulation: same section], which is reported
next to it (``physical_limit_nm``). So the working scale stays defined for every method, and says where it is a
guess.

Pure functions, no Qt.

References (APA 7)
------------------
Endesfelder, U., Malkusch, S., Fricke, F., & Heilemann, M. (2014). A simple method to estimate the average
    localization precision of a single-molecule localization microscopy experiment. Histochemistry and Cell
    Biology, 141(6), 629-638. https://doi.org/10.1007/s00418-014-1192-3
Gazal, N. G., Escalante, ..., et al. (2026). eLife Reviewed Preprint (the reference pipeline of MPS Explorer).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- the rule and its constants ---------------------------------------------------------------------------------
# [simulation: clustering research, sec. 8.6 / Table 8.5, sim3_pairs_resolution.py, seed 20261016] d50 ~ 2 eps +
# 2.5 sigma_loc for DBSCAN; an empirical fit (inference, not literature).
D50_EPS_FACTOR = 2.0
D50_SIGMA_FACTOR = 2.5
# [simulation: same table; A9 sec. 1.1 defines the fusion mark as d50 - 20 nm] at 25/10 and sigma 12 nm, 10 % of
# the pairs were resolved at 60 nm against 49 % at 80 nm.
FUSION_BELOW_D50_NM = 20.0
# [simulation: same section] above this NeNA the rule under-estimates d50 (100 predicted, 100 to > 120 simulated).
D50_RULE_SIGMA_LIMIT_NM = 20.0
# [simulation: same section] the physical limit of every strategy tried: ~2.5-3 sigma_eff.
PHYSICAL_LIMIT_SIGMA_EFF = (2.5, 3.0)
# [ad hoc: the simulator's value, of the order of the linkage errors measured by Frueh et al. 2021] linkage sd.
DEFAULT_SIGMA_LINK_NM = 12.0
# [lit: Gazal, Escalante, et al. 2026] the slab and the reference pipeline's parameters.
DEFAULT_SLAB_HALF_WIDTH_NM = 90.0
COMPAT_EPS_NM = 25.0
COMPAT_MIN_SAMPLES = 10
# [lit: Gazal, Escalante, et al. 2026, Methods] the paper's lateral precision, used only when no NeNA is given
# (the mark is then labelled "paper, not this file").
FALLBACK_NENA_NM = 12.0

ORIGIN_LIT = "literature"
ORIGIN_MEASURED = "measured"
ORIGIN_SIM = "simulation"
ORIGIN_AD_HOC = "ad hoc"
ORIGIN_USER = "user"


@dataclass(frozen=True)
class ScaleMark:
    """One mark of the working scale: its value, unit, a one-word origin and the sentence that says why."""
    key: str
    label: str
    value: Optional[float]
    unit: str
    origin: str
    why: str


@dataclass
class WorkingScale:
    """The working scale of one file and one clustering method (A9 sec. 1.1).

    ``eps_nm`` is the scale the d50 rule is evaluated with: DBSCAN's eps, or the method's separation scale for
    any other method (``eps_source`` says which). ``min_samples`` is None for a method without one."""
    method: str
    eps_nm: float
    eps_source: str
    min_samples: Optional[int]
    nena_nm: float
    nena_source: str
    slab_half_width_nm: float
    d50_nm: float
    fusion_nm: float
    physical_limit_nm: Tuple[float, float]
    sigma_link_nm: float = DEFAULT_SIGMA_LINK_NM
    rule_applies: bool = True
    method_scale_nm: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def marks(self) -> List[ScaleMark]:
        """The marks in display order, each with its origin (A9 sec. 4.1, the 'Working scale' block)."""
        dbscan = self.eps_source == "DBSCAN eps"
        eps_origin = ORIGIN_LIT if (dbscan and self.eps_nm == COMPAT_EPS_NM) else (
            ORIGIN_USER if dbscan else ORIGIN_AD_HOC)
        rule_origin = ORIGIN_SIM if (self.rule_applies and dbscan) else ORIGIN_AD_HOC
        out = [
            ScaleMark("eps_nm", "DBSCAN eps" if dbscan else "reference eps (for the marks)", self.eps_nm, "nm",
                      eps_origin,
                      "Gazal, Escalante, et al. (2026): 25 nm" if eps_origin == ORIGIN_LIT else
                      ("typed by the user" if dbscan else
                       "the reference pipeline's 25 nm, used in DBSCAN's d50 rule for every method (same tested "
                       "radii across methods)")),
            ScaleMark("min_samples", "min samples", None if self.min_samples is None else float(self.min_samples), "",
                      ORIGIN_LIT if self.min_samples == COMPAT_MIN_SAMPLES else ORIGIN_USER,
                      "the reference pipeline's value (Gazal, Escalante, et al., 2026)"),
            ScaleMark("nena_nm", "NeNA precision (lateral)", self.nena_nm, "nm",
                      ORIGIN_MEASURED if self.nena_source == "measured" else ORIGIN_LIT,
                      "NeNA of this file (Endesfelder et al., 2014)" if self.nena_source == "measured" else
                      "no NeNA measured: the paper's 12 nm (Gazal, Escalante, et al., 2026), NOT this file's"),
            ScaleMark("fusion_nm", "near-total fusion", self.fusion_nm, "nm", rule_origin,
                      "d50 - 20 nm: two molecules closer than this come out as one cluster (<= 10 % of pairs resolved; "
                      "project simulation)"),
            ScaleMark("d50_nm", "d50 (half of pairs resolved)", self.d50_nm, "nm", rule_origin,
                      "2 eps + 2.5 NeNA (project simulation, an empirical fit); a 1NN below it is an upper bound"),
            ScaleMark("slab_half_width_nm", "axial slab (half width)", self.slab_half_width_nm, "nm",
                      ORIGIN_LIT if self.slab_half_width_nm == DEFAULT_SLAB_HALF_WIDTH_NM else ORIGIN_USER,
                      "Gazal, Escalante, et al. (2026): 180 nm"),
        ]
        return out

    def export_columns(self) -> Dict[str, Any]:
        """The ``scale_*`` columns of A9 sec. 5 (the integration adds them to the axon row)."""
        return {
            "scale_eps_nm": round(self.eps_nm, 2),
            "scale_min_samples": self.min_samples,
            "scale_nena_nm": round(self.nena_nm, 2),
            "scale_fusion_nm": round(self.fusion_nm, 1),
            "scale_d50_nm": round(self.d50_nm, 1),
            "scale_slab_nm": round(self.slab_half_width_nm, 1),
        }


def d50_rule_nm(eps_nm: float, nena_nm: float) -> float:
    """d50 ~ 2 eps + 2.5 sigma_NeNA [simulation: clustering research sec. 8.6; empirical fit, +/-8 nm except at
    sigma >= 20 nm, where it under-estimates]."""
    return float(D50_EPS_FACTOR * float(eps_nm) + D50_SIGMA_FACTOR * float(nena_nm))


def fusion_nm_from_d50(d50_nm: float) -> float:
    """Near-total fusion = d50 - 20 nm [simulation; A5's definition in A9 sec. 1.1], never below 0."""
    return float(max(0.0, float(d50_nm) - FUSION_BELOW_D50_NM))


def physical_limit_nm(nena_nm: float, sigma_link_nm: float = DEFAULT_SIGMA_LINK_NM) -> Tuple[float, float]:
    """(2.5, 3) x sigma_eff, sigma_eff = sqrt(sigma_loc^2 + sigma_link^2) [simulation: clustering research sec. 8.6,
    the limit no strategy beat]."""
    s_eff = math.hypot(float(nena_nm), float(sigma_link_nm))
    return (PHYSICAL_LIMIT_SIGMA_EFF[0] * s_eff, PHYSICAL_LIMIT_SIGMA_EFF[1] * s_eff)


def working_scale(eps_nm: float = COMPAT_EPS_NM, min_samples: Optional[int] = COMPAT_MIN_SAMPLES,
                  nena_nm: Optional[float] = None, *, method: str = "DBSCAN",
                  eps_source: str = "DBSCAN eps", slab_half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
                  sigma_link_nm: float = DEFAULT_SIGMA_LINK_NM) -> WorkingScale:
    """The working scale from eps (or the method's separation scale), min_samples and the file's NeNA.

    ``nena_nm`` None or not positive falls back to the paper's 12 nm, labelled as such (not this file's)."""
    notes: List[str] = []
    if nena_nm is None or not (math.isfinite(float(nena_nm)) and float(nena_nm) > 0):
        nena, source = FALLBACK_NENA_NM, "paper"
        notes.append("No NeNA for this file: the marks use the paper's 12 nm, not this file's precision.")
    else:
        nena, source = float(nena_nm), "measured"
    if not (math.isfinite(float(eps_nm)) and float(eps_nm) > 0):
        raise ValueError(f"working_scale: eps_nm must be a positive number, got {eps_nm!r}")
    d50 = d50_rule_nm(eps_nm, nena)
    rule_applies = nena < D50_RULE_SIGMA_LIMIT_NM
    if not rule_applies:
        notes.append(f"NeNA {nena:.1f} nm >= {D50_RULE_SIGMA_LIMIT_NM:.0f} nm: the d50 rule under-estimates here "
                     "(the simulation gave 100 to more than 120 nm at 20 nm); read d50 as a lower bound.")
    if eps_source != "DBSCAN eps":
        notes.append(f"d50 and the fusion mark come from DBSCAN 25/10's rule, not from {method}: ad hoc for this "
                     "method (the same tested radii for every method); the physical limit of every strategy "
                     "(2.5-3 sigma_eff) is reported beside it.")
    return WorkingScale(method=str(method), eps_nm=float(eps_nm), eps_source=str(eps_source),
                        min_samples=None if min_samples is None else int(min_samples), nena_nm=nena,
                        nena_source=source, slab_half_width_nm=float(slab_half_width_nm), d50_nm=d50,
                        fusion_nm=fusion_nm_from_d50(d50), physical_limit_nm=physical_limit_nm(nena, sigma_link_nm),
                        sigma_link_nm=float(sigma_link_nm), rule_applies=rule_applies, notes=notes)


def working_scale_for_method(method: str, params: Optional[Dict[str, Any]] = None, nena_nm: Optional[float] = None,
                             *, separation_scale_nm: Optional[float] = None,
                             slab_half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
                             sigma_link_nm: float = DEFAULT_SIGMA_LINK_NM) -> WorkingScale:
    """The working scale for any clustering method of ``tools.clustering_strategies``.

    DBSCAN: eps and min_samples from ``params`` (defaults 25 / 10). Any other method: the rule is evaluated with
    the REFERENCE pipeline's eps, 25 nm [ad hoc], so that every method is tested on the same radii and a comparison
    across methods compares like with like; ``separation_scale_nm`` (the method's ``edge_margin_nm`` after a fit,
    ``AxonAnalysis.cluster_edge_margin_nm``) is only recorded. It is not used because it is not a resolution
    scale: HDBSCAN's is its LARGEST core distance (221 nm on a synthetic axon of the calibration, which put the
    fusion mark beyond the 400 nm of the Ripley grid and emptied the test) [simulation: docs/sims/pipeline_null]."""
    p = dict(params or {})
    if method == "DBSCAN":
        return working_scale(float(p.get("eps", COMPAT_EPS_NM)), int(p.get("min_samples", COMPAT_MIN_SAMPLES)),
                             nena_nm, method=method, eps_source="DBSCAN eps", slab_half_width_nm=slab_half_width_nm,
                             sigma_link_nm=sigma_link_nm)
    ms = p.get("min_samples")
    ws = working_scale(COMPAT_EPS_NM, None if ms is None else int(ms), nena_nm, method=method,
                       eps_source="reference pipeline eps (same radii for every method)",
                       slab_half_width_nm=slab_half_width_nm, sigma_link_nm=sigma_link_nm)
    if separation_scale_nm is not None and math.isfinite(float(separation_scale_nm)):
        ws.method_scale_nm = float(separation_scale_nm)
        ws.notes.append(f"{method}'s own separation scale (edge margin) is {float(separation_scale_nm):.1f} nm; "
                        "not used for the marks.")
    return ws


def working_scale_of_analysis(analysis: Any, nena_nm: Optional[float] = None) -> WorkingScale:
    """The working scale of a ``tools.mps_analysis.AxonAnalysis`` (its method, params, edge margin and slab)."""
    method = str(getattr(analysis, "cluster_method", "DBSCAN"))
    if method == "DBSCAN":
        params: Dict[str, Any] = {"eps": float(analysis.eps_nm), "min_samples": int(analysis.min_samples)}
    else:
        params = dict(getattr(analysis, "cluster_params", {}) or {})
    return working_scale_for_method(method, params, nena_nm,
                                    separation_scale_nm=getattr(analysis, "cluster_edge_margin_nm", None),
                                    slab_half_width_nm=float(getattr(analysis, "slab_half_width_nm",
                                                                     DEFAULT_SLAB_HALF_WIDTH_NM)))
