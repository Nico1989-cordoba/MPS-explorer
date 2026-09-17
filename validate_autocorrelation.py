# -*- coding: utf-8 -*-
"""
Checks for tools/mps_autocorrelation.py: the autocorrelation amplitude of
Zhong et al. (2014), "the difference between the first peak and the
average of the two first valleys of the autocorrelation curve".

The valleys are located independently here, as the first local minima
before and after the first peak, and the module's amplitude has to match.

Run:  python validate_autocorrelation.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from scipy.signal import argrelmin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_autocorrelation import autocorrelation_amplitude  # noqa: E402

PASSED = 0
FAILED = 0

PIXEL_NM = 10.0
PERIOD_NM = 190.0


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def profile(n: int = 400, period_nm: float = PERIOD_NM,
            trend: float = 0.0) -> np.ndarray:
    """A sine along the axon, optionally on a linear trend."""
    i = np.arange(n)
    return np.asarray(np.sin(2 * np.pi * i * PIXEL_NM / period_nm)
                      + trend * i / n, dtype=float)


def by_definition(ac: np.ndarray, period_nm: float) -> float:
    """The amplitude from the first local minima around the peak."""
    peak = int(round(period_nm / PIXEL_NM))
    minima = argrelmin(ac)[0]
    first = minima[minima < peak][0]
    second = minima[minima > peak][0]
    return float(ac[peak] - (ac[first] + ac[second]) / 2)


def main() -> int:
    print("=" * 72)
    print("AUTOCORRELATION AMPLITUDE CHECKS")
    print("=" * 72)

    def clean_sine():
        amp, period, ac = autocorrelation_amplitude(profile(), PIXEL_NM)
        assert period == PERIOD_NM, period
        want = by_definition(ac, period)
        assert abs(amp - want) < 1e-12, (amp, want)
        return f"period {period:.0f} nm, amplitude {amp:.3f}"

    def slow_trend():
        # A trend drives the far lags down. Taking the lowest point after
        # the peak as the second valley gave 0.989 here.
        amp, period, ac = autocorrelation_amplitude(profile(trend=3.0),
                                                    PIXEL_NM)
        assert period == PERIOD_NM, period
        want = by_definition(ac, period)
        assert abs(amp - want) < 1e-12, (amp, want)
        assert float(np.min(ac[int(period / PIXEL_NM) + 1:])) < \
            float(np.min(ac[int(period / PIXEL_NM) + 1:
                            2 * int(period / PIXEL_NM) + 1])), \
            "the trend no longer puts a deeper point at far lags"
        return f"amplitude {amp:.3f}, the definition gives {want:.3f}"

    def another_period_in_the_window():
        amp, period, ac = autocorrelation_amplitude(
            profile(period_nm=170.0), PIXEL_NM)
        assert period == 170.0, period
        assert abs(amp - by_definition(ac, period)) < 1e-12
        return "170 nm found while looking around 190 nm"

    def no_structure():
        flat = autocorrelation_amplitude(np.ones(400), PIXEL_NM)
        short = autocorrelation_amplitude(np.arange(5.0), PIXEL_NM)
        assert flat[0] is None and flat[1] is None
        assert short[0] is None and short[1] is None
        return "flat and too-short profiles give no amplitude"

    def missing_samples():
        p = profile()
        with_gaps = p.copy()
        with_gaps[[3, 50]] = np.nan
        amp, period, _ = autocorrelation_amplitude(with_gaps, PIXEL_NM)
        assert amp is not None and period == PERIOD_NM
        return "NaN samples are dropped"

    def sampling_step():
        coarse = np.sin(2 * np.pi * np.arange(200) * 20.0 / PERIOD_NM)
        amp, period, _ = autocorrelation_amplitude(coarse, 20.0)
        assert amp is not None and period is not None
        assert period % 20.0 == 0, period
        return f"190 nm at 20 nm per sample reads {period:.0f} nm"

    check("a clean sine: period, and amplitude by the definition",
          clean_sine)
    check("a slow trend does not move the second valley", slow_trend)
    check("a period away from the expected one, within the window",
          another_period_in_the_window)
    check("no structure, no amplitude", no_structure)
    check("missing samples", missing_samples)
    check("the period is a multiple of the sampling step", sampling_step)

    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
