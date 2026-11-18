# -*- coding: utf-8 -*-
"""
Localizations per cycle (lpc) by depth: a diagnostic beside the DBSCAN rule (user decision B3 (c), 2026-10-23).

The question
------------
The DBSCAN rule (``tools.dbscan_rule``) reads the localizations per switching cycle measured on the WHOLE loaded
file (B3: like the NeNA, the photophysics is the movie's, not the axon's) -- and keeps doing so. The user also wants
to know whether the localizations per event change with z: the axon is analysed in a slab of +/-90 nm around its
main peak, and an event far from the focal plane is dimmer and may break into fewer frames. This module says, for
one file, without changing the rule:

* ``lpc_file``: mean localizations per uncensored event over the whole file;
* ``lpc_slab``: the same over the events whose centre lies in the analysis' slab, with its bootstrap 95 % interval;
* the slab against the rest of the file: the difference (slab - outside) with its bootstrap 95 % interval, and the
  flag ``lpc_slab_differs`` when that interval excludes 0 (the slab and the file differ beyond chance);
* a profile of lpc against z: bands as wide as the slab, stacked above and below it (the slab is one band), each
  with its events, mean and bootstrap 95 % interval.

How
---
The file's localizations are linked once into events exactly as the rule's measurement does (``tools.mps_events
.link_events``: radius 3 sqrt(2) NeNA (decision B1), one dark frame; censored events -- on in the first frame or
still on in the last -- left out). Each event goes to the band of its CENTRE z (the mean z of its localizations),
so one event is counted once. A file Picasso already linked (a ``len`` column) is read as events: its rows, its
``n`` (localizations per event; ``len`` as an upper bound without it) and its z.

Numbers and their origin: band width = the slab's width [ad hoc: so that the slab is one band and its neighbours
are read at the same resolution]; 999 bootstrap draws [ad hoc: Davison & Hinkley 1997's order for a 95 % percentile
interval]; a band is summarised from 30 events on [ad hoc: below it a mean of skewed event sizes is not a number to
report]; alpha 0.05 [ad hoc: conventional]. The bootstrap is seeded (deterministic).

What it is not: no correction is applied anywhere, the rule's lpc is unchanged, and the flag says only that the
slab's events are not a random sample of the file's in their size -- with tens of thousands of events a difference
of a few per cent is "beyond chance"; read it with the difference itself.

No Qt.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["N_BOOTSTRAP", "MIN_EVENTS_PER_BAND", "ALPHA", "LPC_DEPTH_COLUMNS", "DepthBand", "LpcDepthProfile",
           "lpc_depth_profile", "lpc_depth_profile_of", "export_columns", "COLUMN_DESCRIPTIONS"]

N_BOOTSTRAP = 999              # [ad hoc: Davison & Hinkley 1997's order for a 95 % percentile interval]
MIN_EVENTS_PER_BAND = 30       # [ad hoc: a mean of skewed event sizes over fewer events is not reported]
ALPHA = 0.05                   # [ad hoc: conventional]
SEED = 20261029                # [ad hoc] the bootstrap's seed (deterministic)

FloatArray = NDArray[np.float64]


@dataclass
class DepthBand:
    """One z band: [z_lo, z_hi) in nm, its events, mean localizations per event and bootstrap 95 % interval (None
    below ``MIN_EVENTS_PER_BAND``), and whether it is the slab."""
    z_lo_nm: float
    z_hi_nm: float
    n_events: int
    mean: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    is_slab: bool


@dataclass
class LpcDepthProfile:
    """The diagnostic of one file (module docstring)."""
    lpc_file: Optional[float]
    n_file: int
    lpc_slab: Optional[float]
    slab_ci: Tuple[Optional[float], Optional[float]]
    n_slab: int
    lpc_outside: Optional[float]
    diff: Optional[float]
    diff_ci: Tuple[Optional[float], Optional[float]]
    differs: Optional[bool]
    band_nm: float
    bands: List[DepthBand]
    source: str
    note: str = ""
    warnings: List[str] = field(default_factory=list)

    def profile_text(self) -> str:
        """The bands in one cell: 'z_lo..z_hi: mean [lo, hi] n=..' separated by ' | ' (the slab marked *)."""
        parts = []
        for b in self.bands:
            star = "*" if b.is_slab else ""
            if b.mean is None:
                parts.append(f"{star}{b.z_lo_nm:.0f}..{b.z_hi_nm:.0f} nm: n={b.n_events} (too few)")
            else:
                parts.append(f"{star}{b.z_lo_nm:.0f}..{b.z_hi_nm:.0f} nm: {b.mean:.3f} [{b.ci_low:.3f}, "
                             f"{b.ci_high:.3f}] n={b.n_events}")
        return " | ".join(parts)


def _boot_means(v: FloatArray, rng: np.random.Generator, n_boot: int) -> FloatArray:
    n = int(v.size)
    out = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        out[b] = float(v[rng.integers(0, n, n)].mean())
    return out


def _ci(draws: FloatArray, alpha: float = ALPHA) -> Tuple[float, float]:
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _summary(sizes: FloatArray, rng: np.random.Generator, n_boot: int
             ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[FloatArray]]:
    if sizes.size < MIN_EVENTS_PER_BAND:
        return None, None, None, None
    draws = _boot_means(sizes, rng, n_boot)
    lo, hi = _ci(draws)
    return float(sizes.mean()), lo, hi, draws


def _profile(sizes: FloatArray, centre_z: FloatArray, slab: Tuple[float, float], source: str, n_boot: int,
             seed: int) -> LpcDepthProfile:
    zmin, zmax = float(slab[0]), float(slab[1])
    width = zmax - zmin
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 61, int(sizes.size)]))
    ok = np.isfinite(sizes) & np.isfinite(centre_z)
    sizes, centre_z = sizes[ok], centre_z[ok]
    in_slab = (centre_z >= zmin) & (centre_z <= zmax)          # the slab's own rule: both ends in
    file_mean = float(sizes.mean()) if sizes.size else None
    s_mean, s_lo, s_hi, s_draws = _summary(sizes[in_slab], rng, n_boot)
    o_mean, _o_lo, _o_hi, o_draws = _summary(sizes[~in_slab], rng, n_boot)
    diff = None if s_mean is None or o_mean is None else s_mean - o_mean
    d_lo = d_hi = None
    differs: Optional[bool] = None
    if s_draws is not None and o_draws is not None:
        d_lo, d_hi = _ci(s_draws - o_draws)
        differs = not (d_lo <= 0.0 <= d_hi)
    bands: List[DepthBand] = []
    if width > 0 and centre_z.size:
        k_lo = int(math.floor((float(centre_z.min()) - zmin) / width))
        k_hi = int(math.floor((float(centre_z.max()) - zmin) / width))
        for k in range(k_lo, k_hi + 1):
            lo_z, hi_z = zmin + k * width, zmin + (k + 1) * width
            sel = in_slab if k == 0 else ((centre_z >= lo_z) & (centre_z < hi_z) & ~in_slab)
            m, lo, hi, _d = _summary(sizes[sel], rng, n_boot)
            bands.append(DepthBand(lo_z, hi_z, int(np.count_nonzero(sel)), m, lo, hi, k == 0))
    warnings: List[str] = []
    if differs:
        warnings.append(f"localizations per event in the slab ({s_mean:.3f}) differ from the rest of the file "
                        f"({o_mean:.3f}) beyond chance (95 % interval of the difference {d_lo:+.3f} .. {d_hi:+.3f}); "
                        "the DBSCAN rule keeps the whole file's value (B3)")
    return LpcDepthProfile(lpc_file=file_mean, n_file=int(sizes.size), lpc_slab=s_mean, slab_ci=(s_lo, s_hi),
                           n_slab=int(np.count_nonzero(in_slab)), lpc_outside=o_mean, diff=diff, diff_ci=(d_lo, d_hi),
                           differs=differs, band_nm=float(width), bands=bands, source=source, warnings=warnings)


def lpc_depth_profile(frame: Any, x_nm: Any, y_nm: Any, z_nm: Any, *, nena_nm: Optional[float],
                      slab: Tuple[float, float], radius_in_nena: Optional[float] = None,
                      n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> LpcDepthProfile:
    """The diagnostic of an UNLINKED file: link it (``tools.mps_events.link_events`` at ``radius_in_nena`` x NeNA,
    default 3 sqrt(2) as the rule's measurement, decision B1; one dark frame), drop censored events, put each event
    in the band of its centre z. Raises ValueError without frames or a NeNA."""
    from tools.mps_events import link_events
    if frame is None:
        raise ValueError("no frame numbers: events cannot be linked")
    if nena_nm is None or not (math.isfinite(float(nena_nm)) and float(nena_nm) > 0):
        raise ValueError("no NeNA: the linking radius cannot be set")
    k = 3.0 * math.sqrt(2.0) if radius_in_nena is None else float(radius_in_nena)
    z = np.asarray(z_nm, dtype=np.float64).ravel()
    layer = link_events(frame, x_nm, y_nm, nena_nm=float(nena_nm), radius_in_nena=k)
    g = np.asarray(layer.link_group, dtype=np.int64)
    valid = (g >= 0) & np.isfinite(z)
    n_ev = int(layer.n_locs.size)
    counts = np.bincount(g[valid], minlength=n_ev).astype(np.float64)
    zsum = np.bincount(g[valid], weights=z[valid], minlength=n_ev)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = np.where(counts > 0, zsum / counts, np.nan)
    keep = ~np.asarray(layer.censored, dtype=bool)
    sizes = np.asarray(layer.n_locs, dtype=np.float64)[keep]
    prof = _profile(sizes, centre[keep], slab, f"linked here at {k:.3g} NeNA, 1 dark frame", n_boot, seed)
    prof.note = f"{int(np.count_nonzero(~keep))} censored events left out"
    return prof


def lpc_depth_profile_of(loc: Any, analysis: Any, nena_nm: Optional[float], **kw: Any) -> LpcDepthProfile:
    """The diagnostic of a loaded file (``tools.mps_io.Localizations``) for the slab of ``analysis``
    (``slab_zmin_nm`` / ``slab_zmax_nm``): a Picasso-linked file (``len`` column) is read as events (its rows, its
    ``n`` -- or ``len``, an upper bound -- and z); otherwise linked here."""
    slab = (float(analysis.slab_zmin_nm), float(analysis.slab_zmax_nm))
    column = getattr(loc, "column", None)
    len_col = column("len") if callable(column) else None
    if len_col is not None:
        n_col = column("n") if callable(column) else None
        sizes = np.asarray(n_col if n_col is not None else len_col, dtype=np.float64).ravel()
        z = np.asarray(loc.z_nm, dtype=np.float64).ravel()
        src = "file column n" if n_col is not None else "file column len (upper bound)"
        return _profile(sizes, z, slab, src, int(kw.get("n_boot", N_BOOTSTRAP)), int(kw.get("seed", SEED)))
    return lpc_depth_profile(getattr(loc, "frame", None), loc.x_nm, loc.y_nm, loc.z_nm, nena_nm=nena_nm, slab=slab,
                             **kw)


LPC_DEPTH_COLUMNS: Tuple[str, ...] = (
    "lpc_file", "lpc_file_n_events", "lpc_slab", "lpc_slab_ci95_low", "lpc_slab_ci95_high", "lpc_slab_n_events",
    "lpc_slab_minus_outside", "lpc_slab_minus_outside_ci95_low", "lpc_slab_minus_outside_ci95_high",
    "lpc_slab_differs", "lpc_z_band_nm", "lpc_z_profile", "lpc_depth_note")

COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "lpc_file": "Diagnostic (B3 c): mean localizations per uncensored event over the whole file, linked at 3 sqrt(2) "
                "NeNA and one dark frame as the DBSCAN rule's measurement (the rule's own value is "
                "loc_per_cycle_used, unchanged).",
    "lpc_file_n_events": "Uncensored events of the whole file behind lpc_file.",
    "lpc_slab": "Mean localizations per event over the events whose centre z (mean z of its localizations) lies in "
                "the analysis' slab.",
    "lpc_slab_ci95_low": "Lower end of the bootstrap 95 % interval of lpc_slab (999 draws over events).",
    "lpc_slab_ci95_high": "Upper end of that interval.",
    "lpc_slab_n_events": "Events whose centre lies in the slab.",
    "lpc_slab_minus_outside": "lpc of the slab's events minus lpc of the file's other events.",
    "lpc_slab_minus_outside_ci95_low": "Lower end of the bootstrap 95 % interval of that difference.",
    "lpc_slab_minus_outside_ci95_high": "Upper end of that interval.",
    "lpc_slab_differs": "True when that interval excludes 0: the slab's events differ from the rest of the file "
                        "beyond chance (a warning only; the rule keeps the whole file's value, B3). With many events "
                        "a small difference is 'beyond chance': read it with lpc_slab_minus_outside.",
    "lpc_z_band_nm": "Width of the z bands of lpc_z_profile: the slab's width (the slab is one band).",
    "lpc_z_profile": "lpc against z: per band 'z_lo..z_hi nm: mean [95 % interval] n=events', the slab marked *; "
                     "a band with fewer than 30 events says 'too few'. Each event in the band of its centre z.",
    "lpc_depth_note": "How the events were obtained (linked here, or a Picasso-linked file's own n / len column), or "
                      "why the diagnostic is empty (no frames, no NeNA, not computed in the axon window).",
}


def export_columns(profile: Optional[LpcDepthProfile], why_empty: str = "") -> Dict[str, Any]:
    """``LPC_DEPTH_COLUMNS`` of one file (all empty but the note when ``profile`` is None)."""
    out: Dict[str, Any] = {c: None for c in LPC_DEPTH_COLUMNS}
    if profile is None:
        out["lpc_depth_note"] = why_empty or None
        return out

    def r(v: Optional[float], nd: int = 4) -> Optional[float]:
        return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)

    out.update({
        "lpc_file": r(profile.lpc_file), "lpc_file_n_events": int(profile.n_file), "lpc_slab": r(profile.lpc_slab),
        "lpc_slab_ci95_low": r(profile.slab_ci[0]), "lpc_slab_ci95_high": r(profile.slab_ci[1]),
        "lpc_slab_n_events": int(profile.n_slab), "lpc_slab_minus_outside": r(profile.diff),
        "lpc_slab_minus_outside_ci95_low": r(profile.diff_ci[0]),
        "lpc_slab_minus_outside_ci95_high": r(profile.diff_ci[1]),
        "lpc_slab_differs": profile.differs, "lpc_z_band_nm": r(profile.band_nm, 1),
        "lpc_z_profile": profile.profile_text() or None,
        "lpc_depth_note": "; ".join(x for x in (profile.source, profile.note) if x) or None,
    })
    return out
