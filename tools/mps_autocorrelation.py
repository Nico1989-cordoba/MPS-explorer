# -*- coding: utf-8 -*-
"""
Periodicity of a 1D intensity profile along the axon.

The amplitude of the autocorrelation, as defined in

    Zhong, G., He, J., Zhou, R., Lorenzo, D., Babcock, H. P., Bennett, V.,
    & Zhuang, X. (2014). Developmental mechanism of the periodic membrane
    skeleton in axons. eLife 3, e04581.

whose Figure 1E legend states it exactly: "The amplitude was measured as
the difference between the first peak and the average of the two first
valleys of the autocorrelation curve."

This is NOT "the mean amplitude of the first two peaks", a paraphrase that
appears elsewhere in the literature; the two differ.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Where to look for the first peak: the literature reference period P_ref
# (C1: 170 nm, the axial period of mouse sciatic nerve, Gazal et al. 2026;
# the published periods of cultured neurons, 182 +/- 16 nm (Xu et al. 2013)
# and ~190 nm (Zhong et al. 2014), lie inside the search window below).
DEFAULT_PERIOD_NM = 170.0
# The first peak is searched within +/- this fraction of P_ref
# (acr-search-window-050; P-R30): +/-35 % is 110-230 nm at 170 nm, which
# covers the published periods of cultured neurons and sciatic nerve to
# about three standard deviations while staying above the first valley at
# P/2 and below the second at 3P/2 (Han et al. 2017). Editable 0.10-0.50.
DEFAULT_SEARCH_WINDOW = 0.35
# The shortest profile the amplitude is computed on, in expected periods
# (acr-min-profile-length-8): the first peak (lag ~P) and the second valley
# (lag up to ~2P) need one period of overlap at lag 2P, so N >= 3P / pixel.
# Editable 2-5 periods.
DEFAULT_MIN_PERIODS = 3.0
# A floor that only keeps the arithmetic defined for a degenerate call.
_MIN_SAMPLES = 8


def min_profile_samples(pixel_size_nm: float, expected_period_nm: float = DEFAULT_PERIOD_NM,
                        min_periods: float = DEFAULT_MIN_PERIODS) -> int:
    """ceil(min_periods x P / pixel): 31 at 170 nm and 20 nm pixels (3 periods)."""
    return int(max(_MIN_SAMPLES, int(np.ceil(float(min_periods) * float(expected_period_nm)
                                             / float(pixel_size_nm)))))


def autocorrelation_amplitude(
    profile: NDArray[np.float64],
    pixel_size_nm: float,
    expected_period_nm: float = DEFAULT_PERIOD_NM,
    search_window: float = DEFAULT_SEARCH_WINDOW,
    min_periods: float = DEFAULT_MIN_PERIODS,
) -> Tuple[Optional[float], Optional[float], NDArray[np.float64]]:
    """
    Amplitude and period of a profile's autocorrelation (Zhong et al. 2014).

    The first peak is the highest LOCAL MAXIMUM within ``search_window`` (a
    fraction) of ``expected_period_nm``: a profile without a repeat there
    (an unstructured one, whose autocorrelation only decays) returns no
    period instead of a spurious one at the edge of the window. Its two
    valleys are the lowest points before it and between it and the second
    peak, which lies near twice its lag. The lowest point anywhere after
    the first peak is not the second valley: a slow trend along the
    profile (bleaching, uneven labelling) drives the far lags of the
    autocorrelation down, and on a sine with such a trend it overstated the
    amplitude by a third.

    Parameters
    ----------
    profile : intensity along the axon, one sample per ``pixel_size_nm``.
    expected_period_nm : the reference period P_ref the peak is searched
        around (170 nm by default).
    search_window : fractional tolerance around that period (0.35).
    min_periods : profiles shorter than this many expected periods return
        no amplitude (3: the shortest on which the first peak and the
        second valley can be estimated with one period of overlap).

    Returns
    -------
    (amplitude, period_nm, autocorrelation). Amplitude and period are None
    when the profile is shorter than ``min_periods`` periods or no local
    maximum lies inside the search window. The period is a multiple of
    ``pixel_size_nm``: with 20 nm pixels, 190 nm reads 180 nm.
    """
    p = np.asarray(profile, dtype=float).ravel()
    p = p[np.isfinite(p)]
    if p.size < min_profile_samples(pixel_size_nm, expected_period_nm, min_periods):
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
    band = band[(band > 0) & (band < ac.size - 1)]
    if band.size == 0:
        return None, None, ac
    local = band[(ac[band] > ac[band - 1]) & (ac[band] > ac[band + 1])]
    if local.size == 0:
        return None, None, ac

    peak_i = int(local[np.argmax(ac[local])])

    # The first valley lies between lag 0 and the first peak, the second
    # between the first peak and the second, near twice the first's lag.
    first = ac[:peak_i]
    second = ac[peak_i + 1:min(2 * peak_i + 1, ac.size)]
    if first.size == 0 or second.size == 0:
        return None, None, ac
    valley_mean = (float(np.min(first)) + float(np.min(second))) / 2.0

    return float(ac[peak_i] - valley_mean), float(lags_nm[peak_i]), ac
