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

# Where to look for the first peak: the MPS period.
DEFAULT_PERIOD_NM = 190.0


def autocorrelation_amplitude(
    profile: NDArray[np.float64],
    pixel_size_nm: float,
    expected_period_nm: float = DEFAULT_PERIOD_NM,
    search_window: float = 0.5,
) -> Tuple[Optional[float], Optional[float], NDArray[np.float64]]:
    """
    Amplitude and period of a profile's autocorrelation (Zhong et al. 2014).

    The first peak is the highest point within ``search_window`` (a
    fraction) of ``expected_period_nm``. Its two valleys are the lowest
    points before it and between it and the second peak, which lies near
    twice its lag. The lowest point anywhere after the first peak is not
    the second valley: a slow trend along the profile (bleaching, uneven
    labelling) drives the far lags of the autocorrelation down, and on a
    sine with such a trend it overstated the amplitude by a third.

    Parameters
    ----------
    profile : intensity along the axon, one sample per ``pixel_size_nm``.
    expected_period_nm : where to look for the first peak.
    search_window : fractional tolerance around that period.

    Returns
    -------
    (amplitude, period_nm, autocorrelation). Amplitude and period are None
    when no peak is found inside the search window. The period is a
    multiple of ``pixel_size_nm``: with 20 nm pixels, 190 nm reads 180 nm.
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

    # The first valley lies between lag 0 and the first peak, the second
    # between the first peak and the second, near twice the first's lag.
    first = ac[:peak_i]
    second = ac[peak_i + 1:min(2 * peak_i + 1, ac.size)]
    if first.size == 0 or second.size == 0:
        return None, None, ac
    valley_mean = (float(np.min(first)) + float(np.min(second))) / 2.0

    return float(ac[peak_i] - valley_mean), float(lags_nm[peak_i]), ac
