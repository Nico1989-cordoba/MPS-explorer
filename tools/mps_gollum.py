# -*- coding: utf-8 -*-
"""
Adapter for Gollum / ringfinder: periodicity of the MPS in 2D images.

Gollum quantifies how well a region of a fluorescence image matches a
periodic reference pattern, by computing the 2D Pearson correlation
between each sub-region and a synthetic pattern swept over orientation
and phase, and keeping the maximum. It is the published method of this
laboratory:

    Barabas, F. M., Masullo, L. A., Bordenave, M. D., A. Giusti, S.,
    Unsain, N., Refojo, D., Caceres, A., & Stefani, F. D. (2017).
    Automated quantification of protein periodic nanostructures in
    fluorescence nanoscopy images: abundance and regularity of neuronal
    spectrin membrane-associated skeleton.
    Scientific Reports 7, 16029.  https://doi.org/10.1038/s41598-017-16280-x

    Source: https://github.com/cibion-conicet/Gollum

This module does NOT reimplement any of that. It calls Gollum, supplies
the parameters the paper publishes, and adapts the call for current
library versions.

LICENSING -- read before distributing
-------------------------------------
Gollum is GPL-3.0. MPS Explorer's README declares MIT (though the repo
currently ships no LICENSE file). GPL-3.0 is copyleft, so Gollum's source
is deliberately NOT vendored here: this module only imports it if the
user has installed it separately, which keeps the two licences apart.
If Gollum is ever to be shipped inside MPS Explorer, the combined work
would have to be GPL-3.0 as well -- a decision for the author, not for
this file.

Installation (once, next to MPS Explorer or anywhere on the path):

    git clone https://github.com/cibion-conicet/Gollum
    # then either add it to PYTHONPATH or pass gollum_path= below

Compatibility
-------------
Gollum was last updated in 2018. Its algorithmic core imports and runs
unchanged on numpy 2.x / scipy 1.17 / scikit-image 0.26, with one
adaptation applied here: ``probabilistic_hough_line`` now requires an
integer ``line_length``, and Gollum passes a float derived from the
pixel size. This module casts it, which is why ``min_line_length_nm`` is
converted with ``int()`` before the call.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------
# Published parameters (Barabas et al. 2017 and the Gollum README)
# ---------------------------------------------------------------------

# Sub-region side, in nm. 1000 nm for a ~190 nm periodicity.
DEFAULT_ROI_SIZE_NM = 1000.0
# Gaussian blur used to decide which pixels hold neuronal material.
# The paper reports 100-150 nm working well for both STED and STORM.
DEFAULT_SIGMA_FILTER_NM = 100.0
# Intensity threshold for that same decision, in standard deviations
# above the mean image intensity. The paper reports 0.5-0.8.
DEFAULT_SIGMA_THRESHOLD = 0.5
# MPS period to look for.
DEFAULT_PERIODICITY_NM = 180.0
# Minimum length of the edge lines used to estimate the axon direction.
DEFAULT_MIN_LINE_LENGTH_NM = 300.0
# Exponent of the reference pattern function; P = 6 in the paper, which
# is `sinPow` in the source.
DEFAULT_SIN_POW = 6
# Angular sweep around the estimated direction.
DEFAULT_THETA_STEP_DEG = 3.0
DEFAULT_DELTA_THETA_DEG = 20.0

# Correlation above which a sub-region is called periodic. These are the
# values calibrated on the authors' own images and MUST be recalibrated
# per setup -- the Gollum README says so explicitly, and they depend on
# resolution, SNR and labelling density.
DEFAULT_THRESHOLD_STED = 0.17
DEFAULT_THRESHOLD_STORM = 0.20


class GollumNotAvailable(ImportError):
    """Raised when Gollum is not importable."""


@dataclass
class GollumParams:
    """Parameters for one Gollum run, in physical units."""

    pixel_size_nm: float
    roi_size_nm: float = DEFAULT_ROI_SIZE_NM
    sigma_filter_nm: float = DEFAULT_SIGMA_FILTER_NM
    sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD
    periodicity_nm: float = DEFAULT_PERIODICITY_NM
    min_line_length_nm: float = DEFAULT_MIN_LINE_LENGTH_NM
    sin_pow: int = DEFAULT_SIN_POW
    theta_step_deg: float = DEFAULT_THETA_STEP_DEG
    delta_theta_deg: float = DEFAULT_DELTA_THETA_DEG
    discrimination_threshold: float = DEFAULT_THRESHOLD_STED

    # ---- derived, in pixels ----
    @property
    def roi_size_px(self) -> int:
        return int(round(self.roi_size_nm / self.pixel_size_nm))

    @property
    def periodicity_px(self) -> float:
        return self.periodicity_nm / self.pixel_size_nm

    @property
    def min_line_length_px(self) -> int:
        # int() is required: current scikit-image rejects a float
        # line_length, which is what Gollum would otherwise pass.
        return int(round(self.min_line_length_nm / self.pixel_size_nm))

    @property
    def sigma_filter_px(self) -> float:
        return self.sigma_filter_nm / self.pixel_size_nm


@dataclass
class SubregionResult:
    """Gollum's verdict on one sub-region."""

    correlation: Optional[float]      # max Pearson over angle and phase
    theta_max_deg: Optional[float]
    phase_max: Optional[float]
    direction_deg: Optional[float]    # estimated axon direction (th0)
    is_periodic: Optional[bool]       # correlation >= threshold
    has_direction: bool               # False when no edge lines were found

    @property
    def usable(self) -> bool:
        return self.correlation is not None and np.isfinite(self.correlation)


@dataclass
class ImageResult:
    """Gollum applied over a whole image, sub-region by sub-region."""

    subregions: List[SubregionResult]
    correlations: NDArray[np.float64]
    ring_fraction: Optional[float]       # fraction of usable subregions periodic
    n_subregions: int
    n_usable: int
    params: GollumParams
    warnings: List[str] = field(default_factory=list)

    @property
    def median_correlation(self) -> Optional[float]:
        c = self.correlations[np.isfinite(self.correlations)]
        return float(np.median(c)) if c.size else None


# ---------------------------------------------------------------------
# Import plumbing
# ---------------------------------------------------------------------

_TOOLS_CACHE: Dict[str, Any] = {}


def _load_gollum(gollum_path: Optional[str] = None):
    """Import ``ringfinder.tools``, optionally from an explicit path."""
    key = gollum_path or "<default>"
    if key in _TOOLS_CACHE:
        return _TOOLS_CACHE[key]

    if gollum_path:
        p = os.path.abspath(gollum_path)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        tools = importlib.import_module("ringfinder.tools")
    except ImportError as exc:
        raise GollumNotAvailable(
            "Gollum (ringfinder) is not importable. It is GPL-3.0 and is not "
            "bundled with MPS Explorer, so install it separately:\n"
            "    git clone https://github.com/cibion-conicet/Gollum\n"
            "then put that directory on PYTHONPATH or pass gollum_path=.\n"
            f"Original error: {exc}"
        ) from exc
    _TOOLS_CACHE[key] = tools
    return tools


def is_available(gollum_path: Optional[str] = None) -> bool:
    """True when Gollum can be imported."""
    try:
        _load_gollum(gollum_path)
        return True
    except GollumNotAvailable:
        return False


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

def neuron_mask(
    image: NDArray[np.float64], params: GollumParams
) -> NDArray[np.bool_]:
    """
    Pixels holding neuronal material, per the paper's two-step rule:
    blur with a Gaussian of sigma_filter, then keep everything more than
    ``sigma_threshold`` standard deviations above the mean intensity.

    Returned in Gollum's convention: True means MASKED OUT (background).
    """
    from scipy.ndimage import gaussian_filter

    blurred = gaussian_filter(np.asarray(image, dtype=float),
                              params.sigma_filter_px)
    cut = blurred.mean() + params.sigma_threshold * blurred.std()
    return ~(blurred > cut)


def analyze_subregion(
    image: NDArray[np.float64],
    params: GollumParams,
    mask: Optional[NDArray[np.bool_]] = None,
    gollum_path: Optional[str] = None,
) -> SubregionResult:
    """
    Run Gollum's correlation method on a single sub-region.

    Parameters
    ----------
    image : 2D sub-region, ideally ``params.roi_size_px`` on a side.
    mask : True where the pixel is background. Computed with
        ``neuron_mask`` when omitted.

    Returns
    -------
    SubregionResult. ``correlation`` is None when Gollum could not
    estimate a direction (no edge lines long enough), which is its way of
    saying there is no identifiable structure here -- not a failure.
    """
    tools = _load_gollum(gollum_path)
    img = np.asarray(image, dtype=float)
    if mask is None:
        mask = neuron_mask(img, params)

    th0, _corr_theta, corr_max, theta_max, phase_max = tools.corrMethod(
        img, mask,
        minLen=params.min_line_length_px,      # int, see module docstring
        thStep=params.theta_step_deg,
        deltaTh=params.delta_theta_deg,
        wvlen=params.periodicity_px,
        sinPow=params.sin_pow,
        developer=False,
    )

    def clean(v):
        if v is None:
            return None
        v = float(v)
        return None if not np.isfinite(v) else v

    corr = clean(corr_max)
    return SubregionResult(
        correlation=corr,
        theta_max_deg=clean(theta_max),
        phase_max=clean(phase_max),
        direction_deg=clean(th0),
        is_periodic=(None if corr is None
                     else corr >= params.discrimination_threshold),
        has_direction=th0 is not None,
    )


def analyze_image(
    image: NDArray[np.float64],
    params: GollumParams,
    gollum_path: Optional[str] = None,
) -> ImageResult:
    """
    Tile an image into sub-regions and run Gollum on each.

    Returns
    -------
    ImageResult, whose ``ring_fraction`` is the fraction of USABLE
    sub-regions called periodic. Sub-regions where no direction could be
    estimated are excluded from that denominator rather than counted as
    non-periodic, since they carry no evidence either way; ``n_usable``
    reports how many were kept.
    """
    tools = _load_gollum(gollum_path)          # fail early if missing
    img = np.asarray(image, dtype=float)
    warnings_: List[str] = []

    n = params.roi_size_px
    if n < 4:
        raise ValueError(
            f"ROI of {params.roi_size_nm} nm is only {n} px at "
            f"{params.pixel_size_nm} nm/px; too small to analyse.")
    if img.shape[0] < n or img.shape[1] < n:
        raise ValueError(
            f"Image {img.shape} is smaller than one {n}x{n} px sub-region.")

    mask_full = neuron_mask(img, params)

    ny, nx = img.shape[0] // n, img.shape[1] // n
    if ny * nx == 0:
        raise ValueError("Image yields no complete sub-regions.")
    if img.shape[0] % n or img.shape[1] % n:
        warnings_.append(
            f"Image {img.shape} is not an exact multiple of the "
            f"{n} px sub-region; the remainder on the right/bottom edges "
            f"is not analysed."
        )

    results: List[SubregionResult] = []
    for iy in range(ny):
        for ix in range(nx):
            sub = img[iy * n:(iy + 1) * n, ix * n:(ix + 1) * n]
            sub_mask = mask_full[iy * n:(iy + 1) * n, ix * n:(ix + 1) * n]
            if sub_mask.all():
                # entirely background: no structure to interrogate
                results.append(SubregionResult(None, None, None, None,
                                               None, False))
                continue
            results.append(analyze_subregion(sub, params, sub_mask,
                                             gollum_path))

    corrs = np.array([r.correlation if r.correlation is not None else np.nan
                      for r in results], dtype=float)
    usable = [r for r in results if r.usable]
    ring_fraction = (float(np.mean([r.is_periodic for r in usable]))
                     if usable else None)

    if not usable:
        warnings_.append(
            "No sub-region yielded a direction estimate. Either the image "
            "holds no resolvable structure, or the discrimination "
            "parameters (sigma_filter_nm, sigma_threshold, "
            "min_line_length_nm) do not suit this data."
        )
    elif len(usable) < 0.25 * len(results):
        warnings_.append(
            f"Only {len(usable)} of {len(results)} sub-regions were usable. "
            f"The ring fraction rests on a small part of the image."
        )

    return ImageResult(
        subregions=results, correlations=corrs,
        ring_fraction=ring_fraction, n_subregions=len(results),
        n_usable=len(usable), params=params, warnings=warnings_,
    )


# ---------------------------------------------------------------------
# Autocorrelation amplitude (Zhong et al. 2014)
# ---------------------------------------------------------------------

def autocorrelation_amplitude(
    profile: NDArray[np.float64],
    pixel_size_nm: float,
    expected_period_nm: float = 190.0,
    search_window: float = 0.5,
) -> Tuple[Optional[float], Optional[float], NDArray[np.float64]]:
    """
    Periodicity of a 1D intensity profile, by the definition in

        Zhong, G., He, J., Zhou, R., Lorenzo, D., Babcock, H. P.,
        Bennett, V., & Zhuang, X. (2014). Developmental mechanism of the
        periodic membrane skeleton in axons. eLife 3, e04581.

    whose Figure 1E legend states it exactly: "The amplitude was measured
    as the difference between the first peak and the average of the two
    first valleys of the autocorrelation curve."

    Note that this is NOT "the mean amplitude of the first two peaks", a
    paraphrase that appears elsewhere in the literature; the two differ.

    Parameters
    ----------
    profile : intensity along the axon.
    expected_period_nm : where to look for the first peak.
    search_window : fractional tolerance around that period.

    Returns
    -------
    (amplitude, period_nm, autocorrelation) -- amplitude and period are
    None when no peak is found inside the search window.
    """
    p = np.asarray(profile, dtype=float).ravel()
    p = p[np.isfinite(p)]
    if p.size < 8:
        return None, None, np.array([])

    x = p - p.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return None, None, np.array([])
    full = np.correlate(x, x, mode="full") / denom
    ac = full[full.size // 2:]            # non-negative lags

    lags_nm = np.arange(ac.size) * pixel_size_nm
    lo = expected_period_nm * (1 - search_window)
    hi = expected_period_nm * (1 + search_window)
    band = np.where((lags_nm >= lo) & (lags_nm <= hi))[0]
    if band.size == 0:
        return None, None, ac

    peak_i = int(band[np.argmax(ac[band])])
    if peak_i <= 0 or peak_i >= ac.size - 1:
        return None, None, ac

    # The two valleys flanking that peak: minima before and after it.
    left = ac[:peak_i]
    right = ac[peak_i + 1:]
    if left.size == 0 or right.size == 0:
        return None, None, ac
    valley_mean = (float(np.min(left)) + float(np.min(right))) / 2.0

    return float(ac[peak_i] - valley_mean), float(lags_nm[peak_i]), ac
