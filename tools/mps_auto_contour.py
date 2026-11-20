"""
Automatic contour of the axon for the perimeter density analysis (package P3).

What it does
------------
One closed membrane curve per axon, fitted to the LOCALIZATIONS without any cluster label:

1. Nodes (``density_nodes``): local maxima of the localization density smoothed by a Gaussian of
   ``sigma_nm``, at least ``min_separation_nm`` apart and above ``threshold_frac`` of
   ``reference_level`` (the MIN_NODES-th strongest maximum). Nodes are NOT clusters: they only order
   the initial curve.
2. Tour: ``tools.mps_geometry.reconstruct_perimeter(all_starts=True)`` (polar angle + 2-opt from every
   start, the tour the main analysis already uses) joins the nodes into a closed polygon. Vertices the
   band bypasses (``BYPASS_SUPPORT``: the band runs from their neighbour to their neighbour without them)
   are dropped and the tour rebuilt.
3. Curve: ``tools.mps_membrane.fit_membrane_from_localizations`` (the periodic P-spline fitted to the
   localizations, robust to background) refines that polygon, with ``knot_nm`` between knots.
4. Nodes farther than ``node_drop_nm`` from the fitted curve are dropped and steps 2-3 repeat,
   ``rounds`` times at most.
5. Notch rule (``NOTCH_MIN_RADIUS_NM``): a node at the bottom of an inward bend tighter than that radius
   (an interior lump the curve went through) is dropped and the curve refitted.

Every result is checked before it is returned. A curve that crosses itself, whose length is
implausible for a closed membrane around the cloud (``MAX_HULL_RATIO``), that encloses no lumen
(``MAX_LUMEN_RATIO``: a filled disk is not an axon) or whose fit aborted is REJECTED: ``status`` is
``Status.CONTOUR_FAILED``, ``path`` is None and the first warning reads
``"contour rejected: <reason>: <detail>"`` (``failure_reason`` returns the reason). Nothing downstream
should be computed on a rejected contour.

When the density-node curve is rejected, two fallbacks are tried in this order, and the warnings say
which one produced the curve (``AutoContour.method`` and ``knot_nm`` are the ones actually used):

* the same nodes with ``FALLBACK_KNOT_NM`` between knots (a stiffer spline cannot fold into the
  narrow loops that a 400 nm basis makes across a hairpin of the tour);
* the polar contour, when it exists and the cloud is star-shaped around its centre.

The polar contour (``polar_contour``) is also computed as a cross-check: ``polar_distance_nm`` is the
mean distance between the two curves (NaN when the polar curve itself won), and a warning is added when
the two differ by more than ``POLAR_WARN_NM`` (worded differently when the curve is not star-shaped,
where the polar curve is only a rough reference).

Quality gate and support (review R1; decisions D7, D9). Passing the rejection rules does not make a curve
good: the accepted curve also gets ``contour_quality`` -- radial robust sd, fraction within +-50/100/150 nm,
the local offsets of the band from the curve (largest single window, largest PERSISTENT offset over
``QUALITY_PERSIST_WINDOWS`` windows, fraction of the arc where the band misses the +-100 nm band) and, when
the z-slice labels are given, the misfit between the slices (``slice_misfit``) -- and ``contour_support``
(stretches > 400 nm with no density under the curve). ``quality_reasons`` is the gate: a general rule on
those numbers, the same for every axon. A curve that fails it keeps ``status = ok`` (the contract has no
other state) but is NOT RELIABLE: its first warning starts with ``LOW_QUALITY_PREFIX`` and
``ContourDetails.assessment`` says ``reliable = False`` with the reasons, the curve kind (``CURVE_KINDS``:
primary / stiff fallback / polar fallback / polar / drawn) and the failure reason as fields
(``assessment_row`` flattens them into columns). A curve the rules accept but the band leaves along a
stretch makes the fallbacks run (``QUALITY_FALLBACK``). Downstream, pooled analyses leave unreliable
contours out by default and report results stratified by curve kind: the three kinds are different
estimators (the 600 nm one smooths concavities tighter than the 400 nm one follows; the polar one fails on
shapes that are not star-shaped).

Interior trimming (decision D8, OFF by default): ``exclude`` drops localizations from the fit, e.g. those
deeper than ``INTERIOR_TRIM_DEPTH_NM`` inside a reference ring (``interior_exclusion`` turns depths given by
the unanchored-cluster package into the mask). This module never computes that reference.

Selection bias (review R1, H4), declared: the notch rule (250 nm) keeps only curves without tight inward
bends, and the stiff fallback cannot make one, so the accepted set under-represents tight concavities;
real bays and valleys of radius < 250 nm are smoothed over rather than rejected (the local offset then
flags the smoothing). The margin of that rule is narrow: lumps gave inward bends of 15-242 nm, correct
curves >= 297 nm, the limit is 250 nm (one real pilot axon fell back by 2 nm).

Independent of clustering: no cluster labels in or out, nothing from the clustering modules is
imported (a test inspects the imports). Pure and deterministic: no random draw anywhere. Every
curve returned (accepted, polar, rejected) runs COUNTER-CLOCKWISE seen from +z (``counter_clockwise``):
a clockwise fit is reversed keeping its arc origin.

Knot spacing: the contract's ``CONTOUR_KNOT_NM`` (400 nm = ``mps_curvature.CURVATURE_KNOT_SPACING_NM``,
[code]). Swept 300 / 400 / 600 nm [own sim]: mean error 5.0 / 6.4 / 14.1 nm, worst p95 44 / 49 / 107 nm.
300 nm is slightly closer in simulation; 400 nm is kept so that the curvature of question B is read on
the basis the programme already validated for curvature.

Origin of every number: [own sim] swept in this package's simulation (``validate_auto_contour.py``,
2026-11-16; the table is in its report), [code] taken from an existing constant of the programme,
[plan] fixed by the implementation plan, [ad hoc] chosen here without further support.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

from tools.mps_curvature import curvature_of_closed_curve
from tools.mps_geometry import reconstruct_perimeter
from tools.mps_matching import SmoothPath
from tools.mps_membrane import (
    MEMBRANE_KNOT_SPACING_NM,
    LocalizationMembrane,
    _linear_closed_path,
    _outward_sign,
    fit_membrane_from_localizations,
)
from tools.mps_perimeter_types import CONTOUR_KNOT_NM, CONTOUR_METHODS, UNSUPPORTED_GAP_NM, AutoContour, Status
from tools.mps_unroll import project_on_path

# ----------------------------------------------------------------------------------------------------
# Parameters of the density nodes (defaults of ``density_nodes`` and ``auto_contour``)
# ----------------------------------------------------------------------------------------------------
NODE_PIXEL_NM: float = 25.0
"""[ad hoc] Grid of the density histogram: half the smoothing sigma, so the Gaussian is sampled by two
pixels per sigma. Not swept."""

# The sweep (validate_auto_contour.py sweep, 2026-11-16; final defaults): one parameter at a time,
# 5 shapes x 12 seeds, each seed in one of 6 conditions (15 % background; radial sd 50 nm; a 2 um gap; a
# dominant cluster with 20 % of the membrane localizations; an interior lump with 10 % at 400 nm; 2000
# membrane localizations). Error = distance from the true curve (where it has data) to the fitted one.
# Rows in the sweep_<parameter>.csv files of the validation output (simulated, outside the repository).

NODE_SIGMA_NM: float = 50.0
"""[own sim] Smoothing of the density. Swept 25 / 50 / 75 / 100 nm: mean error 6.3 / 6.4 / 8.0 / 10.1 nm,
median p95 16 / 16 / 19 / 29 nm, worst p95 49 / 49 / 386 / 199 nm, accepted 100 / 100 / 100 / 97 %. 75 and
100 nm merge the clusters of tight bends and let lumps through. Flat from 25 to 50 nm; 50 nm (~2x the
cluster radius, r_eff ~21-25 nm in the programme's real data) is the smoother of the two."""

NODE_THRESHOLD_FRAC: float = 0.05
"""[own sim] A node must exceed this fraction of ``reference_level`` (the MIN_NODES-th strongest density
maximum). Swept 0.05 / 0.15 / 0.30: mean error 6.4 / 6.6 / 8.2 nm, worst p95 49 / 49 / 409 nm, accepted
100 / 98 / 95 %. Higher thresholds leave weak stretches of membrane without nodes and the tour cuts across
them. Background peaks that pass 0.05 are removed by the bypass test (V1 with 30 % background: all
accepted, worst p95 48 nm)."""

NODE_MIN_SEPARATION_NM: float = 200.0
"""[own sim] Minimum distance between two nodes (size of the maximum filter). Swept 100 / 200 / 300 /
400 nm: mean error 6.3 / 6.4 / 7.2 / 11.4 nm, worst p95 50 / 49 / 318 / 567 nm, accepted 100 / 100 / 100 /
97 %; 300-400 nm lose the corners of the "C" and of the star. Flat from 100 to 200 nm; 200 nm keeps the
tour shorter (fewer nodes, faster 2-opt)."""

NODE_DROP_NM: float = 150.0
"""[own sim, no effect in the range] A node farther than this from the fitted curve is dropped and the
tour rebuilt. Swept 100 / 150 / 250 nm: identical results, because the bypass and notch rules below
already remove the off-membrane nodes before this one sees them. Kept as a last guard at ~the real
radial FWHM (148 nm, pilot) and well below the depth of an interior lump (400 nm in the simulation)."""


BYPASS_MIN_DETOUR: float = 1.05
"""[own sim] Bypass test, who is looked at: a tour vertex whose detour (d(prev, v) + d(v, next)) /
d(prev, next) exceeds this. On a smooth ring the detour is ~1 + (s/R)^2/8 (spacing s, curvature radius R),
<= 1.03 for s <= R/2; a lump 400 nm inside a ring with 400 nm node spacing gave 1.44."""

BYPASS_SUPPORT: float = 0.7
"""[own sim] Bypass test: the vertex is dropped when at least this fraction of the middle 60 % of the chord
prev -> next lies on the band (smoothed density >= ``BYPASS_LEVEL_FRAC`` x ``reference_level``): the
membrane runs from prev to next without passing by v, so v is off it (an interior lump, a background
peak) or redundant (on a straight stretch, where dropping it costs nothing). A corner node keeps its place
(its chord crosses the empty lumen) and so does a node at a gap (its chord crosses the gap).

Why not the simpler rules: (1) the plain drop rule (distance to the curve fitted THROUGH the node) missed
an interior lump of 8 % of the localizations 400 nm deep in every simulated shape -- the P-spline
initialised through the lump keeps its localizations, because its robust step rejects only points far from
the INITIAL curve; (2) a leave-one-out refit dropped legitimate corner nodes -- the same robust step keeps
the refit on the chord; (3) support measured on a 100 nm smoothing was saturated by the lump's own halo.
Limitation: a lump in line with a gap (next to its end) is not caught; its chord crosses the gap."""

BYPASS_LEVEL_FRAC: float = 0.05
"""[ad hoc] Band level of the bypass test, as a fraction of ``reference_level`` of the node map. Between
two clusters 250 nm apart (4 clusters/um) the density smoothed at 50 nm falls to ~0.09 of a cluster's
peak; uniform background (30 %) and a lump's halo 400 nm away are below 0.01."""

BYPASS_MAX_PASSES: int = 8
"""[ad hoc] Passes of the bypass test; each drops every hit vertex that is not next to another one taken in
the same pass, then rebuilds the tour (one drop per rebuild took 80 s on a filled disk of 9000 points)."""

NOTCH_MIN_RADIUS_NM: float = 250.0
"""[own sim] Notch rule: when the fitted curve bends INWARD with a radius below this and a node sits at the
bottom of the bend (within ``NOTCH_NODE_RADIUS_NM``), that node is dropped and the curve refitted (at most
``NOTCH_MAX_DROPS`` times); a notch that survives REJECTS the curve (``FAIL_NOTCH``; the fallbacks are tried). Simulation (5 shapes x 4 seeds x 6 clean
conditions: 0-30 % background, radial sd 20 and 50 nm, 2000 localizations, a 2 um gap, a dominant
cluster): the tightest inward bend of a correct curve was >= 297 nm (the true star valleys are 356 nm);
with an interior lump inside the tour it was 15-242 nm. The plain drop rule cannot see the lump: the
P-spline initialised through it keeps its localizations (its robust step rejects only points far from the
INITIAL curve), and opposite a sparse stretch of membrane the lump is the densest thing around, so no
density rule on the tour separates it either; only the geometry does."""

POLAR_NOTCH_MIN_RADIUS_NM: float = 150.0
"""[own sim] Notch limit of the POLAR curve, used together with ``POLAR_REJECT_IF_LEAVES_CLOUD``. The polar
curve is not built from tour nodes, and the 250 nm rule of the node curve rejected 24 of 130 correct polar
curves (p95 <= 40 nm: real bays and valleys) while catching 79 of 86 polar curves through a lump. "Notch < 150
nm OR local offset > 75 nm": 17 of 130 correct curves lost, 77 of 86 lump curves caught, 224 of 235 bad curves
(p95 > 170 nm) caught, against 209 with the old rule (first calibration, polar curves only, with the SINGLE-window
offset). Since v2 the offset is the persistent one (``QUALITY_PERSIST_WINDOWS``): on the 18 real pilot axons the polar
curve is then rejected in 14 (inward notch, or a stretch off the band of 86-388 nm), two axons with offsets of
120 and 96 nm among them, which the first version accepted."""

POLAR_REJECT_IF_LEAVES_CLOUD: bool = True
"""[own sim] Reject a polar curve the cloud leaves (persistent offset above ``QUALITY_MAX_LOCAL_OFFSET_NM``)."""

NOTCH_NODE_RADIUS_NM: float = 150.0
"""[ad hoc] A node this close to the bottom of a notch is the one the notch rule drops (= ``NODE_DROP_NM``)."""

NOTCH_MAX_DROPS: int = 3
"""[own sim] Nodes the notch rule may drop per curve (one per refit, each at the bottom of a notch). 8 was
tried: on the 18 real pilot axons it changed nothing, and in V1 it turned two correctly rejected sparse "C"
curves into ACCEPTED curves with a p95 error of ~1190 nm. More drops let the rule carve a wrong curve
until it looks smooth; 3 covers the simulated single lump."""

NODE_ROUNDS: int = 2
"""[ad hoc] Drop-and-refit rounds after the first fit; a second round never dropped anything in the
simulation once the first had removed an interior lump."""

MIN_NODES: int = 8
"""[plan] Fewer nodes than this: no contour (the tour of 7 points is a heptagon, not a membrane)."""

FALLBACK_KNOT_NM: float = float(MEMBRANE_KNOT_SPACING_NM)
"""[code] 600 nm, the membrane fit's own knot spacing (``tools.mps_membrane``): the stiffer basis the
density-node curve retries with when the 400 nm one is rejected."""

MAX_HULL_RATIO: float = 1.5
"""[own sim + ad hoc margin] Rejection when the curve's length exceeds this multiple of the perimeter
of its own convex hull. In the validation (V1, every condition) the accepted correct curves stayed <= 1.25
("C", the least convex shape); wrong curves of the 1000-localization "C" reached 1.59;
a curve that snakes back and forth inside the ring (the failure the pilot saw on real axons: 32 um of
contour where the polar curve gave 10) is far above it."""

WARN_HULL_RATIO: float = 1.3
"""[own sim] A warning (not a rejection) above this hull ratio: just past the largest correct value (1.25)."""

LUMEN_DEPTH_NM: float = 250.0
"""[ad hoc] Depth inside the curve beyond which the axoplasm starts for the lumen test: 2.5x the band
half-width of the analysis (100 nm), so the band's own scatter does not reach it."""

MAX_LUMEN_RATIO: float = 0.5
"""[own sim] Rejection when the areal density of localizations deeper than ``LUMEN_DEPTH_NM`` inside
the curve exceeds this fraction of the density in the +-100 nm band. V1: rings with up to 30 % uniform
background and an interior lump gave <= 0.10; the filled disk (negative control) gave 0.93-1.04."""

LUMEN_MIN_AREA_NM2: float = math.pi * 300.0 ** 2
"""[ad hoc] Below this inner area the lumen test is not possible and is skipped with a warning."""

LUMEN_BAND_NM: float = 100.0
"""[plan] The band of the lumen test = the analysis's band half-width (BAND_HALF_WIDTH_NM)."""

POLAR_N_SECTORS: int = 72
"""[ad hoc, pilot] 5 degree sectors of the polar contour."""

POLAR_RADIAL_BIN_NM: float = 40.0
"""[ad hoc, pilot] Radial histogram bin of the polar contour (~ the cluster diameter)."""

POLAR_MIN_PER_SECTOR: int = 15
"""[ad hoc, pilot] A sector with fewer localizations is interpolated from its neighbours."""

POLAR_WARN_NM: float = 120.0
"""[own sim] Warning when the curve is star-shaped and its mean distance to the polar curve exceeds this.
V1, star-shaped correct curves: the 95th percentile of that distance was <= 6 nm without background and
46-94 nm with 15-30 % background (the polar mode is pulled by the background; the density-node curve is
not, its error stayed <= 8 nm there). 120 nm sits above that; a gap or a lump pushes it to 130-280 nm."""

QUALITY_NEAR_NM: float = 400.0
"""[ad hoc] Localizations within this distance of the curve enter the radial sd and the local offset: wide
enough to see a curve running 200-500 nm inside the cloud, narrow enough to leave the far background out."""

QUALITY_WINDOW_NM: float = 500.0
"""[ad hoc] Arc window of the local offset (= ``CONTOUR_KNOT_NM`` + one bin: the scale a 400 nm spline can
follow). Step ``QUALITY_STEP_NM``."""

QUALITY_STEP_NM: float = 100.0
"""[plan] Step of the local-offset windows = the analysis bin (``BIN_NM``)."""

QUALITY_MIN_WINDOW_COUNT: int = 30
"""[ad hoc] A window with fewer localizations has no median worth reading (its sd would be ~sd/5)."""

QUALITY_BIN_NM: float = 25.0
"""[ad hoc] Radial bin of the local histogram whose peak is the local offset (~ the cluster radius)."""

QUALITY_PEAK_CONTRAST: float = 3.0
"""[ad hoc] A window counts only when its radial peak reaches this multiple of the median bin (the band
stands out of the background). The first version used the median offset of every window: in sparse
stretches the background and interior spectrin decided it and 70-80 % of the CORRECT simulated curves were
flagged."""

QUALITY_PEAK_MIN_COUNT: float = 5.0
"""[ad hoc] ... and at least this many (smoothed) localizations in the peak bin: a few background points in
one bin of a sparse window are not a band."""

# Calibration of the quality gate (validate_auto_contour.py calib, 2026-11-16): 13 conditions (15 % background,
# radial sd 50, a cloud like the real one with radial sd 75 / 100 / 150 nm and 10 % interior spectrin, the 2 um gap
# at 0 / 15 / 30 % background and in the real-like cloud, the interior lump in both clouds, 1000 localizations, the
# combined hazards) x 8 shapes (the 5 base shapes plus bays of 500 / 700 nm and five valleys of radius ~110 nm) x 6
# seeds x each curve type ON ITS OWN (density nodes at 400 nm, at 600 nm, polar): 1872 curves.

QUALITY_MIN_FRAC_100: float = 0.55
"""[own sim; start value from review R1] Low quality below this fraction of the localizations within +-100 nm.
Correct curves (p95 error <= 40 nm) in clouds of radial sd up to 100 nm kept >= 0.57 (real-like sd 100 nm:
0.57-0.59; sd 75 nm: >= 0.66; sd 20-50 nm: >= 0.77); no correct curve was flagged. A cloud of radial sd 150 nm
is flagged whatever the curve: the band of +-100 nm does not hold it, which is what the flag says."""

QUALITY_MAX_SD_NM: float = 130.0
"""[own sim; start value from review R1] Low quality above this radial robust sd. Correct curves stayed
<= 115 nm with radial sd 100 nm clouds; no correct curve was flagged."""

QUALITY_MAX_LOCAL_OFFSET_NM: float = 75.0
"""[own sim] Low quality when the PERSISTENT offset (``QUALITY_PERSIST_WINDOWS`` consecutive windows) exceeds
this; the same number makes the fallbacks run and rejects a polar curve. 75 nm is where a local shift costs
~30 % of the band's localizations (review R1: with a radial sd of 73 nm, a 50 nm shift loses 12 %, 100 nm
loses 40 %). The first version read the largest SINGLE window: in simulation it flagged 8.7 % of the correct
curves, and on the 18 real pilot axons it flagged all 12 accepted curves, the three the review judged good
included (single-window offsets 78-131 nm on clouds of radial sd 51-69 nm): one window's radial peak is
noisy. Calibration (``validate_auto_contour.py calib``, 2026-11-17: 13 conditions x 8 shapes x 6 seeds x 3
curve types alone = 1872 curves, 1168 accepted by the rules; "correct" = p95 error <= 40 nm in clouds of radial
sd <= 100 nm, n = 489; "bad" = p95 error or curve-to-truth p95 > 170 nm, n = 166): persistent offset over
2 / 3 / 4 / 5 windows above 75 nm flags 2.7 / 1.0 / 0.4 / 0.0 % of the correct curves and 61 / 39 / 29 / 13 %
of the bad ones; the largest single window flagged 9.8 % / 80 %. Three windows keep the false alarms at 1 %
(at 60 nm: 1.2 % / 50 %). The bad curves it misses are wrong where the cloud has no data (gap ends:
``contour_support``), go through a lump, or are the stiff fallback smoothing a bay (review R1, H3)."""

QUALITY_BAND_NM: float = 100.0
"""[plan] Half-width of the band of the analysis (``BAND_HALF_WIDTH_NM``): a window whose band lies farther
than this from the curve counts in ``off_band_fraction``."""

QUALITY_PERSIST_WINDOWS: int = 3
"""[ad hoc] Consecutive windows (step 100 nm, width 500 nm: 700 nm of arc) an offset must hold to count as a
stretch where the curve leaves the cloud (the scale a 400 nm spline can follow, and a lump or bay is wider)."""

QUALITY_MAX_OFF_BAND_FRAC: float = 0.15
"""[own sim] Low quality when the band lies more than ``QUALITY_BAND_NM`` from the curve on more than this
fraction of the windows with a band. Calibration (same 1872 curves): the correct curves reached at most
0.049 (p99 0.026); 0.10 / 0.15 / 0.25 flag 0 % of them and 12 / 4.8 / 2.4 % of the bad ones. 0.15 = three
times the largest correct value, because real clouds are wider than the simulated ones. In simulation it adds
almost nothing to the persistent offset; it is the read-out of a curve that runs beside the band on a large
part of the arc (the anchoring report measured 0.29-0.37 of the arc on the three pilot axons that motivated
decision D9, with a different window rule)."""

SLICE_FIT_NEAR_NM: float = 300.0
"""[ad hoc] Localizations within this distance of the curve enter the per-slice fit of ``slice_misfit``
(3x the band half-width: wide enough for a ring displaced 100-200 nm, narrow enough to leave most interior
spectrin and background out; the Huber weights handle the rest)."""

SLICE_MIN_LOCS: int = 50
"""[ad hoc] A slice with fewer localizations near the curve gets NaN (three coefficients need far more)."""

SLICE_HUBER_K: float = 1.345
"""[literature] Huber's constant: 95 % efficiency under Gaussian errors (Huber 1981, Robust Statistics)."""

SLICE_HUBER_ITER: int = 10
"""[ad hoc] Reweighting iterations of the Huber fit (converged to < 1 nm in 5 on the pilot axons)."""

SLICE_MAX_RMS_NM: float = 50.0
"""[derivation, review R1] Low quality when the ring of slice -1 or +1 is displaced a RMS of more than this
from the central slice's (decision D7). With the band of +-100 nm and a radial sd of 73 nm (the real median), a
50 nm displacement costs 12 % of a slice's band localizations and 100 nm costs 40 %: beyond 50 nm one curve for
the three slices no longer describes the outer ones."""

INTERIOR_TRIM_DEPTH_NM: float = 250.0
"""[pilot, decision D2 of the anchoring report] ``interior_exclusion``: a localization deeper than this inside
the reference ring is excluded from the contour fit when the trimming is on (off by default, decision D8).
= ``LUMEN_DEPTH_NM``, the start of the "doubtful" strip of the unanchored-cluster criterion (250-400 nm)."""

SUPPORT_LEVEL_FRAC: float = 0.05
"""[own sim] ``contour_support``: a curve sample is supported when the density under it (smoothed at 50 nm)
reaches this fraction of ``reference_level``. Calibrated on the 2 um gap (6 seeds x 8 shapes): k = 0.01 /
0.02 / 0.03 / 0.05 / 0.08 found 96-98 % of the gap with 0 % background but only 23 / 84 / 95 / 97 / 98 %
with 30 % background and 5 / 33 / 58 / 86 / 90 % in a cloud like the real one (radial sd 100 nm, interior
spectrin); flagged arcs facing membrane that HAS data: 0-1 / 1 / 1-2 / 2-3 / 4-6 %. Without a gap the
simulation still leaves 15-20 % of the curve unsupported: stretches between Poisson clusters (4 per um)
longer than 400 nm that truly hold no localization."""

NODE_WARN_COUNT: int = 150
"""[review R1 timing] Above this many density nodes a warning: one tour of every start takes ~N^2.5
(60 nodes 0.2 s, 120 nodes 1.1 s, 180 nodes 3.1 s, 300 nodes 12 s) and a contour builds 10-15 tours."""

NODE_MAX_COUNT: int = 250
"""[ad hoc, timing] Above this many nodes only the strongest ``NODE_MAX_COUNT`` are kept (warning). A
contour with 250 nodes takes ~1-2 min; real pilot axons had 24-123 nodes."""

MIN_LOCALIZATIONS: int = 200
"""[ad hoc] Below this the contour is not attempted (the P-spline needs >= 3 per knot after the robust
step, and a 10 um axon at 400 nm knots has 25 knots)."""

STAR_SAMPLES: int = 2000
"""[ad hoc] Rows of the curve the star-shape test looks at (the dense table subsampled)."""

FAIL_TOO_FEW_LOCS = "too few localizations"
FAIL_TOO_FEW_NODES = "too few nodes"
FAIL_FIT = "fit aborted"
FAIL_SELF_CROSSING = "self-crossing"
FAIL_IMPLAUSIBLE_LENGTH = "implausible length"
FAIL_NO_LUMEN = "no lumen"
FAIL_NO_DRAWN = "no drawn contour"
FAIL_NO_POLAR = "no polar contour"
FAIL_NOTCH = "inward notch"
FAIL_LEAVES_CLOUD = "leaves the cloud"
FAILURE_REASONS: Tuple[str, ...] = (FAIL_TOO_FEW_LOCS, FAIL_TOO_FEW_NODES, FAIL_FIT, FAIL_SELF_CROSSING, FAIL_NOTCH, FAIL_LEAVES_CLOUD,
                                    FAIL_IMPLAUSIBLE_LENGTH, FAIL_NO_LUMEN, FAIL_NO_DRAWN, FAIL_NO_POLAR)
REJECT_PREFIX = "contour rejected: "


# ----------------------------------------------------------------------------------------------------
# Small geometry helpers
# ----------------------------------------------------------------------------------------------------
def _as_xy(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y have different lengths ({x.size} and {y.size})")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("non-finite coordinate")
    return x, y


def _signed_area(poly: NDArray[np.float64]) -> float:
    nxt = np.roll(poly, -1, axis=0)
    return 0.5 * float(np.sum(poly[:, 0] * nxt[:, 1] - nxt[:, 0] * poly[:, 1]))


def _area_centroid(poly: NDArray[np.float64]) -> Tuple[float, float]:
    """Area centroid of a closed polygon (falls back to the vertex mean when the area is ~0)."""
    nxt = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * nxt[:, 1] - nxt[:, 0] * poly[:, 1]
    a = 0.5 * float(np.sum(cross))
    if abs(a) < 1e-9:
        return float(poly[:, 0].mean()), float(poly[:, 1].mean())
    cx = float(np.sum((poly[:, 0] + nxt[:, 0]) * cross)) / (6.0 * a)
    cy = float(np.sum((poly[:, 1] + nxt[:, 1]) * cross)) / (6.0 * a)
    return cx, cy


def _subsample(points: NDArray[np.float64], n_max: int) -> NDArray[np.float64]:
    step = max(1, int(math.ceil(points.shape[0] / float(n_max))))
    return points[::step]


def _self_crossings(points: NDArray[np.float64], max_vertices: int = 600) -> int:
    """Proper crossings of a closed polyline with itself (same predicate as ``mps_membrane.self_crossings``,
    on a finer subsample: 600 vertices instead of 400, so loops of a few hundred nm on a 30 um curve count)."""
    a = _subsample(np.asarray(points, dtype=np.float64), max_vertices)
    n = int(a.shape[0])
    if n < 4:
        return 0
    b = np.roll(a, -1, axis=0)
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]

    def orient(p1: NDArray[np.float64], p2: NDArray[np.float64], p3: NDArray[np.float64]) -> NDArray[np.float64]:
        return (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])

    d1 = orient(a[i], b[i], a[j])
    d2 = orient(a[i], b[i], b[j])
    d3 = orient(a[j], b[j], a[i])
    d4 = orient(a[j], b[j], b[i])
    return int(np.count_nonzero((d1 * d2 < 0.0) & (d3 * d4 < 0.0)))


def hull_ratio(path: SmoothPath) -> float:
    """Length of the closed curve over the perimeter of its convex hull (1 for a convex curve)."""
    pts = _subsample(np.asarray(path.points_nm, dtype=np.float64), 4000)
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return float("nan")
    v = pts[hull.vertices]
    per = float(np.hypot(*(np.roll(v, -1, axis=0) - v).T).sum())
    return float(path.length_nm) / per if per > 0.0 else float("nan")


def counter_clockwise(path: SmoothPath) -> SmoothPath:
    """
    The same closed curve traversed counter-clockwise (positive signed area, seen from +z). A clockwise
    table is reversed keeping row 0 (the arc origin) in place: rows become 0, N-1, ..., 1 and the edges and
    cumulative arcs are recomputed. A counter-clockwise curve is returned unchanged. Every curve this module
    returns goes through here, so the arc runs counter-clockwise and the sign of the curvature means the
    same thing for every axon (the plan's common frame).
    """
    pts = np.asarray(path.points_nm, dtype=np.float64)
    if _signed_area(pts) >= 0.0:
        return path
    rev = np.concatenate([pts[:1], pts[:0:-1]], axis=0)
    edges = np.hypot(*(np.roll(rev, -1, axis=0) - rev).T)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return SmoothPath(points_nm=rev, vertex_row=np.zeros(0, dtype=np.intp), edges_nm=edges, cum_nm=cum,
                      length_nm=float(cum[-1]))


def _points_in_polygon(px: NDArray[np.float64], py: NDArray[np.float64], poly: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Even-odd rule, vectorised over the points, looped over the polygon's edges."""
    inside = np.zeros(px.shape, dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for k in range(poly.shape[0]):
        cond = (y1[k] > py) != (y2[k] > py)
        if not cond.any():
            continue
        dy = y2[k] - y1[k]
        xint = x1[k] + (py - y1[k]) * (x2[k] - x1[k]) / (dy if dy != 0.0 else 1e-300)
        inside ^= cond & (px < xint)
    return inside


def project(path: SmoothPath, x: NDArray[np.float64], y: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(arc, signed offset + outward) of the points on ``path`` (``tools.mps_unroll.project_on_path``)."""
    return project_on_path(path, np.column_stack([x, y]), outward_sign=_outward_sign(path.points_nm))


def lumen_ratio(path: SmoothPath, x: NDArray[np.float64], y: NDArray[np.float64], *,
                depth_nm: float = LUMEN_DEPTH_NM, band_nm: float = LUMEN_BAND_NM,
                pixel_nm: float = 25.0) -> Tuple[float, float]:
    """
    (areal density deeper than ``depth_nm`` inside the curve) / (areal density within +-``band_nm``),
    and the inner area in nm^2. The inner area is measured on a grid (pixels inside the curve whose
    distance to it exceeds ``depth_nm``); the band's area is 2 band_nm L. nan when the inner area is below
    ``LUMEN_MIN_AREA_NM2`` or the band holds no localization.
    """
    poly = _subsample(np.asarray(path.points_nm, dtype=np.float64), 600)
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    gx = np.arange(lo[0] + 0.5 * pixel_nm, hi[0], pixel_nm)
    gy = np.arange(lo[1] + 0.5 * pixel_nm, hi[1], pixel_nm)
    if gx.size == 0 or gy.size == 0:
        return float("nan"), 0.0
    grid_x, grid_y = np.meshgrid(gx, gy)
    mx, my = grid_x.reshape(-1), grid_y.reshape(-1)
    inside = _points_in_polygon(mx, my, poly)
    d, _ = cKDTree(np.asarray(path.points_nm, dtype=np.float64)).query(np.column_stack([mx[inside], my[inside]]))
    inner_area = float(np.count_nonzero(np.asarray(d) > depth_nm)) * pixel_nm * pixel_nm
    if inner_area < LUMEN_MIN_AREA_NM2:
        return float("nan"), inner_area
    _s, r = project(path, x, y)
    n_band = int(np.count_nonzero(np.abs(r) <= band_nm))
    if n_band == 0:
        return float("nan"), inner_area
    n_inner = int(np.count_nonzero(r < -depth_nm))
    band_area = 2.0 * band_nm * float(path.length_nm)
    return (n_inner / inner_area) / (n_band / band_area), inner_area


def curve_distance_nm(a: SmoothPath, b: SmoothPath) -> float:
    """Symmetric mean distance between two closed curves (nearest row of the other dense table)."""
    pa = _subsample(np.asarray(a.points_nm, dtype=np.float64), 3000)
    pb = _subsample(np.asarray(b.points_nm, dtype=np.float64), 3000)
    da, _ = cKDTree(np.asarray(b.points_nm, dtype=np.float64)).query(pa)
    db, _ = cKDTree(np.asarray(a.points_nm, dtype=np.float64)).query(pb)
    return 0.5 * (float(np.mean(da)) + float(np.mean(db)))


# ----------------------------------------------------------------------------------------------------
# Public functions of the contract
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class DensityMap:
    """Localization histogram smoothed by a Gaussian, with its grid (pixel j spans x0 + [j, j + 1) px)."""

    x0: float
    y0: float
    pixel_nm: float
    values: NDArray[np.float64]                 # (ny, nx)

    def sample(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        j = np.floor((pts[:, 0] - self.x0) / self.pixel_nm).astype(np.int64)
        i = np.floor((pts[:, 1] - self.y0) / self.pixel_nm).astype(np.int64)
        ny, nx = self.values.shape
        ok = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
        out = np.zeros(pts.shape[0], dtype=np.float64)
        out[ok] = self.values[i[ok], j[ok]]
        return out


def density_map(x: NDArray[np.float64], y: NDArray[np.float64], *, pixel_nm: float, sigma_nm: float) -> DensityMap:
    """The smoothed localization density on a grid that covers the cloud plus 4 sigma."""
    pad = 4.0 * sigma_nm + pixel_nm
    x0, y0 = float(x.min()) - pad, float(y.min()) - pad
    nx = int(math.ceil((float(x.max()) + pad - x0) / pixel_nm))
    ny = int(math.ceil((float(y.max()) + pad - y0) / pixel_nm))
    h, _, _ = np.histogram2d(y, x, bins=[ny, nx], range=[[y0, y0 + ny * pixel_nm], [x0, x0 + nx * pixel_nm]])
    sm = ndimage.gaussian_filter(h, sigma_nm / pixel_nm, mode="constant")
    return DensityMap(x0=x0, y0=y0, pixel_nm=float(pixel_nm), values=np.asarray(sm, dtype=np.float64))


def _local_maxima(dm: DensityMap, min_separation_nm: float) -> Tuple[NDArray[np.intp], NDArray[np.float64]]:
    sm = dm.values
    size = int(round(min_separation_nm / dm.pixel_nm)) | 1
    peak = (sm == ndimage.maximum_filter(sm, size=size, mode="constant")) & (sm > 0.0)
    return np.argwhere(peak), np.asarray(sm[peak], dtype=np.float64)


def reference_level(weights: NDArray[np.float64]) -> float:
    """The ``MIN_NODES``-th strongest weight (the smallest when there are fewer). A ring holds at least
    MIN_NODES clusters, so this is a membrane cluster's level whatever the area of background around it (a
    pixel percentile is not: in a wide, sparse "C" the 95th percentile of the pixels fell on the
    background and background peaks became nodes), and one dominant cluster cannot raise it."""
    if weights.size == 0:
        return 0.0
    srt = np.sort(weights)[::-1]
    return float(srt[min(MIN_NODES, srt.size) - 1])


def density_nodes(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], *, pixel_nm: float = NODE_PIXEL_NM,
                  sigma_nm: float = NODE_SIGMA_NM, threshold_frac: float = NODE_THRESHOLD_FRAC,
                  min_separation_nm: float = NODE_MIN_SEPARATION_NM) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Ring nodes from the smoothed localization density; returns (nodes (N, 2) in nm, their smoothed
    density). A node is a pixel equal to the maximum of its ``min_separation_nm`` neighbourhood and above
    ``threshold_frac`` x ``reference_level`` of all such maxima. Deterministic; nodes are sorted by
    decreasing density (ties by row, then column).
    """
    x, y = _as_xy(x_nm, y_nm)
    if not (pixel_nm > 0.0 and sigma_nm > 0.0 and min_separation_nm > 0.0 and threshold_frac >= 0.0):
        raise ValueError("density_nodes: pixel, sigma and separation must be positive, threshold >= 0")
    if x.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    dm = density_map(x, y, pixel_nm=pixel_nm, sigma_nm=sigma_nm)
    rc, w = _local_maxima(dm, min_separation_nm)
    keep = w > float(threshold_frac) * reference_level(w)
    rc, w = rc[keep], w[keep]
    order = np.lexsort((rc[:, 1], rc[:, 0], -w))
    rc, w = rc[order], w[order]
    nodes = np.column_stack([dm.x0 + (rc[:, 1] + 0.5) * pixel_nm, dm.y0 + (rc[:, 0] + 0.5) * pixel_nm]).astype(np.float64)
    return nodes, np.asarray(w, dtype=np.float64)


def polar_contour(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], *, n_sectors: int = POLAR_N_SECTORS,
                  radial_bin_nm: float = POLAR_RADIAL_BIN_NM) -> Optional[NDArray[np.float64]]:
    """
    Polar comparison contour (n_sectors, 2): from a centre (the median of the cloud, then twice the area
    centroid of the previous contour), the mode of the radial histogram in each sector, interpolated
    over empty sectors, median- and Gaussian-smoothed periodically. None when fewer than a third of the
    sectors hold ``POLAR_MIN_PER_SECTOR`` localizations. Valid only for clouds that are star-shaped
    around their centre: ``is_star_shaped`` of the fitted curve says whether it was.
    """
    x, y = _as_xy(x_nm, y_nm)
    if x.size < POLAR_MIN_PER_SECTOR or n_sectors < 8 or not (radial_bin_nm > 0.0):
        return None
    cx, cy = float(np.median(x)), float(np.median(y))
    poly: Optional[NDArray[np.float64]] = None
    idx = np.arange(n_sectors)
    for _ in range(2):
        dx, dy = x - cx, y - cy
        th = np.arctan2(dy, dx)
        r = np.hypot(dx, dy)
        sect = np.floor((th + np.pi) / (2.0 * np.pi) * n_sectors).astype(np.int64) % n_sectors
        rm = np.full(n_sectors, np.nan)
        edges = np.arange(0.0, float(np.percentile(r, 99.5)) + radial_bin_nm, radial_bin_nm)
        if edges.size < 3:
            return None
        for s in range(n_sectors):
            rr = r[sect == s]
            if rr.size < POLAR_MIN_PER_SECTOR:
                continue
            hcount, _ = np.histogram(rr, bins=edges)
            hs = ndimage.gaussian_filter1d(hcount.astype(np.float64), 1.0, mode="constant")
            k = int(np.argmax(hs))
            rm[s] = 0.5 * (edges[k] + edges[k + 1])
        ok = np.isfinite(rm)
        if int(ok.sum()) < n_sectors // 3:
            return None
        rm = np.interp(idx, np.concatenate([idx[ok] - n_sectors, idx[ok], idx[ok] + n_sectors]), np.tile(rm[ok], 3))
        rm = ndimage.median_filter(rm, size=5, mode="wrap")
        rm = ndimage.gaussian_filter1d(rm, 1.0, mode="wrap")
        tc = -np.pi + (idx + 0.5) * 2.0 * np.pi / n_sectors
        poly = np.column_stack([cx + rm * np.cos(tc), cy + rm * np.sin(tc)]).astype(np.float64)
        cx, cy = _area_centroid(poly)
    return poly


def is_star_shaped(path: SmoothPath, centre: Tuple[float, float]) -> bool:
    """True when every ray from ``centre`` crosses the closed curve once: the polar angle of the curve
    around the centre is monotonic and winds exactly once."""
    p = _subsample(np.asarray(path.points_nm, dtype=np.float64), STAR_SAMPLES)
    if p.shape[0] < 3:
        return False
    th = np.arctan2(p[:, 1] - float(centre[1]), p[:, 0] - float(centre[0]))
    d = np.diff(np.concatenate([th, th[:1]]))
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    total = float(d.sum())
    if abs(abs(total) - 2.0 * np.pi) > 1e-6:
        return False
    sign = 1.0 if total > 0.0 else -1.0
    return bool(np.all(d * sign >= -1e-12))


# ----------------------------------------------------------------------------------------------------
# The contour and its diagnostics
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Candidate:
    """One curve tried (accepted or not), for the diagnostics."""

    label: str                                  # e.g. "density_nodes knot 400"
    method: str
    knot_nm: float
    path: Optional[SmoothPath]
    nodes_nm: NDArray[np.float64]
    residual_sd_nm: float
    n_self_crossings: int
    hull_ratio: float
    lumen_ratio: float
    reason: str                                 # "" when accepted, else one of FAILURE_REASONS
    detail: str
    fit_warnings: Tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class ContourDetails:
    """``auto_contour``'s result plus everything tried on the way (not part of the contract: diagnostics,
    the validation script and the figures read it)."""

    contour: AutoContour
    candidates: Tuple[Candidate, ...]
    polar_path: Optional[SmoothPath]
    n_nodes_initial: int
    rejected_path: Optional[SmoothPath] = None  # the primary curve when it was rejected (for figures only)
    notes: Tuple[str, ...] = field(default_factory=tuple)
    assessment: Optional["ContourAssessment"] = None


# ----------------------------------------------------------------------------------------------------
# Quality of the fit and support of the curve (review R1, 2026-11-16)
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SliceMisfit:
    """How far the three z slices' rings fall from one another (decision D7: measured and flagged, never
    corrected; a contour per slice is a separate package). Index 0 / 1 / 2 = slices -1 / 0 / +1.

    Per slice, the signed offsets r (outward) of its localizations within ``SLICE_FIT_NEAR_NM`` of the curve
    are fitted by r = a + b n_x + c n_y (n = outward normal of the curve where the localization projects;
    Huber-weighted least squares): ``a`` is the change of radius of that slice's ring (nm, + = larger), (b, c)
    its lateral shift. ``rms_displacement_nm`` = RMS over the curve of the displacement of slice -1 / +1
    relative to slice 0 ((a_s - a_0) + (b_s - b_0) n_x + (c_s - c_0) n_y). ``frac100_central`` /
    ``frac100_all``: fraction of the localizations within +-100 nm of the SAME curve, central slice only and
    the three slices together. NaN where a slice holds fewer than ``SLICE_MIN_LOCS``."""

    n_locs: Tuple[int, int, int]
    radius_change_nm: Tuple[float, float, float]
    shift_xy_nm: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    lateral_shift_nm: Tuple[float, float, float]   # |(b, c)| of each slice against the curve
    lateral_vs_central_nm: Tuple[float, float]     # |(b_s - b_0, c_s - c_0)| for slices -1 and +1
    radius_change_per_slice_nm: float              # (a_{+1} - a_{-1}) / 2
    rms_displacement_nm: Tuple[float, float]       # slices -1 and +1 against slice 0
    frac100_by_slice: Tuple[float, float, float]
    frac100_central: float
    frac100_all: float
    low: bool
    reason: str


@dataclass(frozen=True, eq=False)
class ContourQuality:
    """How well the curve sits in the cloud. ``low_quality`` is True when any of ``reasons`` holds; it does
    not reject the curve (the status stays ok) but the contour is NOT RELIABLE and must stay out of pooled
    analyses (decision D9); the per-axon table keeps it with its reasons.

    ``frac_within`` (keys 50 / 100 / 150 nm) is over ALL the localizations given; ``robust_sd_nm`` is the
    1.4826 x MAD of the signed offsets within ``QUALITY_NEAR_NM`` of the curve.

    Local offset: in arc windows of ``QUALITY_WINDOW_NM`` every ``QUALITY_STEP_NM`` (localizations within
    ``QUALITY_NEAR_NM``), the position of the significant radial peak nearest the curve
    (``window_offset_nm``, NaN where no band stands out of the background). ``max_local_offset_nm`` is the
    largest |offset| of any single window (diagnostic: on real clouds one window's peak is noisy);
    ``persistent_offset_nm`` the largest offset held by ``QUALITY_PERSIST_WINDOWS`` consecutive windows (a
    stretch, not a spike: the number the gate and the fallbacks read); ``off_band_fraction`` the fraction
    of the windows with a band whose band lies more than ``QUALITY_BAND_NM`` from the curve (the band of
    the analysis misses the membrane there). ``slices``: ``SliceMisfit`` when the slice labels were given."""

    robust_sd_nm: float
    frac_within: Dict[float, float]
    max_local_offset_nm: float
    local_offset_arc_nm: float
    n_windows: int
    low_quality: bool
    reasons: Tuple[str, ...]
    persistent_offset_nm: float = float("nan")
    persistent_offset_arc_nm: float = float("nan")
    off_band_fraction: float = float("nan")
    window_arc_nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    window_offset_nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    slices: Optional[SliceMisfit] = None


@dataclass(frozen=True, eq=False)
class ContourSupport:
    """Stretches of the curve with no localization density under them (``contour_support``)."""

    level: float                                  # density level (counts per pixel, smoothed) below which = no support
    arcs_nm: Tuple[Tuple[float, float], ...]      # (start, end) arcs of the unsupported runs; end < start wraps
    unsupported_fraction: float
    sample_arc_nm: NDArray[np.float64]            # the samples along the curve
    supported: NDArray[np.bool_]                  # per sample: density >= level


@dataclass(frozen=True, eq=False)
class ContourAssessment:
    """Structured read-out of one contour (not part of the P0 contract, which stays untouched; read it from
    ``ContourDetails.assessment``, or flat with ``assessment_row``).

    ``failure_reason`` is the rejection reason as a field ("" when accepted, else one of ``FAILURE_REASONS``),
    so nothing downstream has to parse the warning text. ``reliable`` is False when the contour was rejected
    or its quality gate failed (``quality.reasons``): such an axon stays in the per-axon table with its
    reasons and is left out of pooled analyses by default. ``n_excluded``: localizations left out of the fit
    by the optional interior trimming (``exclude``; 0 when off)."""

    kind: str                                     # one of CURVE_KINDS
    failure_reason: str                           # "" when accepted, else one of FAILURE_REASONS
    quality: Optional[ContourQuality]
    support: Optional[ContourSupport]
    reliable: bool = False
    n_excluded: int = 0


CURVE_KINDS: Tuple[str, ...] = ("primary", "stiff fallback", "polar fallback", "polar", "drawn", "none")
LOW_QUALITY_PREFIX = "CONTOUR NOT RELIABLE (left out of pooled analyses): "
QUALITY_FALLBACK: bool = True
"""[own sim] A curve the rules accept but the cloud leaves (local offset above the limit) makes the next
fallbacks run, and the first of them that the cloud does not leave is used instead (see the calibration)."""


def _robust_sd(v: NDArray[np.float64]) -> float:
    if v.size == 0:
        return float("nan")
    med = float(np.median(v))
    return 1.4826 * float(np.median(np.abs(v - med)))


def window_offsets(path: SmoothPath, s: NDArray[np.float64], r: NDArray[np.float64], *,
                   window_nm: float = QUALITY_WINDOW_NM, step_nm: float = QUALITY_STEP_NM) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(window centre arcs, signed offset of the significant radial peak nearest the curve in each window; NaN
    where the window holds fewer than ``QUALITY_MIN_WINDOW_COUNT`` localizations or no band stands out).
    ``s``, ``r``: arc and signed outward offset of the localizations (``project``)."""
    length = float(path.length_nm)
    arcs: NDArray[np.float64] = np.asarray(np.arange(0.0, length, float(step_nm)) if length > 0 else np.zeros(0),
                                     dtype=np.float64)
    offs: NDArray[np.float64] = np.full(arcs.size, np.nan)
    near = np.abs(r) <= QUALITY_NEAR_NM
    sn, rn = s[near], r[near]
    if sn.size == 0:
        return arcs, offs
    order = np.argsort(sn, kind="stable")
    sn, rn = sn[order], rn[order]
    ss = np.concatenate([sn - length, sn, sn + length])
    rr = np.concatenate([rn, rn, rn])
    edges = np.arange(-QUALITY_NEAR_NM, QUALITY_NEAR_NM + QUALITY_BIN_NM, QUALITY_BIN_NM)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for i, c in enumerate(arcs):
        lo = int(np.searchsorted(ss, c - 0.5 * window_nm))
        hi = int(np.searchsorted(ss, c + 0.5 * window_nm))
        if hi - lo < QUALITY_MIN_WINDOW_COUNT:
            continue
        hcount, _ = np.histogram(rr[lo:hi], bins=edges)
        hs = ndimage.gaussian_filter1d(hcount.astype(np.float64), 1.0, mode="constant")
        floor = float(np.median(hs))
        level_ok = max(QUALITY_PEAK_CONTRAST * max(floor, 0.5), QUALITY_PEAK_MIN_COUNT)
        left = np.concatenate([[-np.inf], hs[:-1]])
        right = np.concatenate([hs[1:], [-np.inf]])
        peaks = np.flatnonzero((hs >= left) & (hs >= right) & (hs >= level_ok))
        # only a window where a band stands out of the flat background says where the membrane is
        if peaks.size == 0:
            continue
        # the significant peak NEAREST the curve: a lump or a neighbouring wall next to a correct curve adds a
        # second peak that must not count against it; a curve off the band has no peak near 0
        k_peak = int(peaks[np.argmin(np.abs(centres[peaks]))])
        pos = float(centres[k_peak])
        if 0 < k_peak < hs.size - 1:          # parabolic refinement between bins
            den = hs[k_peak - 1] - 2.0 * hs[k_peak] + hs[k_peak + 1]
            if den < 0:
                pos += 0.5 * QUALITY_BIN_NM * float(hs[k_peak - 1] - hs[k_peak + 1]) / float(den)
        offs[i] = pos
    return arcs, offs


def persistent_offset(window_offset_nm: NDArray[np.float64], n_windows: int = QUALITY_PERSIST_WINDOWS) -> Tuple[float, int]:
    """(largest offset held by ``n_windows`` consecutive windows, index of the first of them; circular). A run
    with a window without a band does not count. (nan, -1) when no run qualifies."""
    a = np.abs(np.asarray(window_offset_nm, dtype=np.float64))
    m = int(a.size)
    k = max(1, int(n_windows))
    if m == 0:
        return float("nan"), -1
    k = min(k, m)
    ext = np.concatenate([a, a[:k - 1]])
    best, where = float("nan"), -1
    for i in range(m):
        run = ext[i:i + k]
        if np.isnan(run).any():
            continue
        v = float(run.min())
        if not (v <= best):               # also true while best is nan
            best, where = v, i
    return best, where


def _outward_normals(path: SmoothPath) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(points, unit outward normals) of the dense table of the counter-clockwise curve."""
    pts = np.asarray(counter_clockwise(path).points_nm, dtype=np.float64)
    t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    norm = np.hypot(*t.T)
    norm[norm == 0.0] = 1.0
    t = t / norm[:, None]
    return pts, np.column_stack([t[:, 1], -t[:, 0]])


def _huber_fit(design: NDArray[np.float64], r: NDArray[np.float64]) -> NDArray[np.float64]:
    """Huber-weighted least squares (k = ``SLICE_HUBER_K`` robust sds), ``SLICE_HUBER_ITER`` reweightings."""
    w: NDArray[np.float64] = np.ones(r.size)
    coef: NDArray[np.float64] = np.zeros(design.shape[1])
    for _ in range(SLICE_HUBER_ITER):
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(design * sw[:, None], r * sw, rcond=None)
        coef = np.asarray(sol, dtype=np.float64)
        res = r - design @ coef
        scale = _robust_sd(res)
        if not (scale > 0.0):
            break
        a = np.abs(res) / (SLICE_HUBER_K * scale)
        w = np.where(a <= 1.0, 1.0, 1.0 / np.maximum(a, 1e-12))
    return np.asarray(coef, dtype=np.float64)


def slice_misfit(path: SmoothPath, x_nm: NDArray[np.float64], y_nm: NDArray[np.float64],
                 slice_label: NDArray[np.integer]) -> SliceMisfit:
    """How far the rings of the z slices -1 / 0 / +1 fall from one another about ONE curve (``SliceMisfit``).
    ``slice_label``: -1 / 0 / +1 per localization (anything else is ignored). Pure; measures, never corrects."""
    x, y = _as_xy(x_nm, y_nm)
    lab = np.asarray(slice_label).reshape(-1)
    if lab.shape != x.shape:
        raise ValueError(f"slice_label has {lab.size} entries for {x.size} localizations")
    pts, nrm = _outward_normals(path)
    ccw = counter_clockwise(path)
    s, r = project(ccw, x, y)
    _d, row = cKDTree(pts).query(np.column_stack([x, y]))
    row = np.asarray(row, dtype=np.int64)
    n_by: List[int] = []
    coefs: List[NDArray[np.float64]] = []
    frac_by: List[float] = []
    for sl in (-1, 0, 1):
        m = lab == sl
        n_by.append(int(np.count_nonzero(m)))
        frac_by.append(float(np.mean(np.abs(r[m]) <= QUALITY_BAND_NM)) if m.any() else float("nan"))
        near = m & (np.abs(r) <= SLICE_FIT_NEAR_NM)
        if int(np.count_nonzero(near)) < SLICE_MIN_LOCS:
            coefs.append(np.full(3, np.nan))
            continue
        design = np.column_stack([np.ones(int(near.sum())), nrm[row[near], 0], nrm[row[near], 1]])
        coefs.append(_huber_fit(design, r[near]))
    a = tuple(float(c[0]) for c in coefs)
    shift = tuple((float(c[1]), float(c[2])) for c in coefs)
    lateral = tuple(float(math.hypot(c[1], c[2])) if np.isfinite(c).all() else float("nan") for c in coefs)
    edges = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)
    wsum = float(edges.sum()) if edges.sum() > 0 else 1.0
    rms: List[float] = []
    for k in (0, 2):
        dlt = coefs[k] - coefs[1]
        if not np.isfinite(dlt).all():
            rms.append(float("nan"))
            continue
        disp = dlt[0] + dlt[1] * nrm[:, 0] + dlt[2] * nrm[:, 1]
        rms.append(float(math.sqrt(float(np.sum(edges * disp * disp)) / wsum)))
    in3 = np.isin(lab, (-1, 0, 1))
    frac_all = float(np.mean(np.abs(r[in3]) <= QUALITY_BAND_NM)) if in3.any() else float("nan")
    worst = max((v for v in rms if math.isfinite(v)), default=float("nan"))
    low = bool(worst > SLICE_MAX_RMS_NM)
    reason = ""
    if low:
        reason = (f"the three z slices do not fall on one another: the ring of an outer slice is displaced a RMS "
                  f"{worst:.0f} nm from the central one (limit {SLICE_MAX_RMS_NM:g}); one curve for the three slices "
                  "widens the band")
    return SliceMisfit(n_locs=(n_by[0], n_by[1], n_by[2]), radius_change_nm=(a[0], a[1], a[2]),
                       shift_xy_nm=(shift[0], shift[1], shift[2]), lateral_shift_nm=(lateral[0], lateral[1], lateral[2]),
                       lateral_vs_central_nm=(float(math.hypot(shift[0][0] - shift[1][0], shift[0][1] - shift[1][1])),
                                              float(math.hypot(shift[2][0] - shift[1][0], shift[2][1] - shift[1][1]))),
                       radius_change_per_slice_nm=0.5 * (a[2] - a[0]), rms_displacement_nm=(rms[0], rms[1]),
                       frac100_by_slice=(frac_by[0], frac_by[1], frac_by[2]), frac100_central=frac_by[1],
                       frac100_all=frac_all, low=low, reason=reason)


def quality_reasons(frac100: float, robust_sd_nm: float, persistent_nm: float, persistent_arc_nm: float,
                    off_band_fraction: float, slices: Optional[SliceMisfit] = None) -> Tuple[str, ...]:
    """The quality gate (decision D9): every reason that makes the contour not reliable. A general rule on
    the numbers, the same for every axon; nothing in it names or was tuned on a particular axon."""
    reasons: List[str] = []
    if frac100 < QUALITY_MIN_FRAC_100:
        reasons.append(f"only {frac100:.0%} of the localizations within +-100 nm (limit {QUALITY_MIN_FRAC_100:.0%}): "
                       "the band of the analysis does not hold this cloud")
    if robust_sd_nm > QUALITY_MAX_SD_NM:
        reasons.append(f"radial robust sd {robust_sd_nm:.0f} nm (limit {QUALITY_MAX_SD_NM:g}): the cloud is wider "
                       "than a membrane band")
    if persistent_nm > QUALITY_MAX_LOCAL_OFFSET_NM:
        reasons.append(f"along a stretch around arc {persistent_arc_nm:.0f} nm the band sits {persistent_nm:.0f} nm to "
                       f"one side of the curve (limit {QUALITY_MAX_LOCAL_OFFSET_NM:g}): the curve leaves the cloud there")
    if off_band_fraction > QUALITY_MAX_OFF_BAND_FRAC:
        reasons.append(f"the band lies more than {QUALITY_BAND_NM:g} nm from the curve on {off_band_fraction:.0%} of the "
                       f"arc (limit {QUALITY_MAX_OFF_BAND_FRAC:.0%})")
    if slices is not None and slices.low:
        reasons.append(slices.reason)
    return tuple(reasons)


def contour_quality(path: SmoothPath, x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], *,
                    window_nm: float = 0.0, step_nm: float = 0.0,
                    slice_label: Optional[NDArray[np.integer]] = None) -> ContourQuality:
    """Radial spread of the localizations about the curve, the fraction within +-50/100/150 nm, the local
    offsets of the band from the curve and, with ``slice_label``, the misfit between the z slices (see
    ``ContourQuality``). Pure; thresholds are the module constants ``QUALITY_*`` / ``SLICE_*``."""
    x, y = _as_xy(x_nm, y_nm)
    win = float(window_nm) if window_nm > 0 else QUALITY_WINDOW_NM
    step = float(step_nm) if step_nm > 0 else QUALITY_STEP_NM
    s, r = project(path, x, y)
    n = max(int(r.size), 1)
    frac = {w: float(np.count_nonzero(np.abs(r) <= w)) / n for w in (50.0, 100.0, 150.0)}
    near = np.abs(r) <= QUALITY_NEAR_NM
    sd = _robust_sd(r[near])
    arcs, offs = window_offsets(path, s, r, window_nm=win, step_nm=step)
    has = np.isfinite(offs)
    n_win = int(np.count_nonzero(has))
    if n_win:
        i_w = int(np.nanargmax(np.abs(offs)))
        worst, worst_arc = float(abs(offs[i_w])), float(arcs[i_w])
        off_band = float(np.mean(np.abs(offs[has]) > QUALITY_BAND_NM))
    else:
        worst, worst_arc, off_band = 0.0, float("nan"), float("nan")
    pers, i_p = persistent_offset(offs)
    pers_arc = float(arcs[(i_p + QUALITY_PERSIST_WINDOWS // 2) % arcs.size]) if i_p >= 0 else float("nan")
    sl = slice_misfit(path, x, y, slice_label) if slice_label is not None else None
    reasons = quality_reasons(frac[100.0], sd, pers, pers_arc, off_band, sl)
    return ContourQuality(robust_sd_nm=sd, frac_within=frac, max_local_offset_nm=worst, local_offset_arc_nm=worst_arc,
                          n_windows=n_win, low_quality=bool(reasons), reasons=reasons, persistent_offset_nm=pers,
                          persistent_offset_arc_nm=pers_arc, off_band_fraction=off_band, window_arc_nm=arcs,
                          window_offset_nm=offs, slices=sl)


def contour_support(path: SmoothPath, x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], *,
                    k: float = SUPPORT_LEVEL_FRAC, min_run_nm: float = UNSUPPORTED_GAP_NM,
                    pixel_nm: float = NODE_PIXEL_NM, sigma_nm: float = NODE_SIGMA_NM,
                    min_separation_nm: float = NODE_MIN_SEPARATION_NM, step_nm: float = 10.0) -> ContourSupport:
    """The localization density (smoothed at ``sigma_nm``) sampled every ``step_nm`` along the curve; a
    sample is supported when it reaches ``k`` x ``reference_level`` of the density maxima. Unsupported runs
    longer than ``min_run_nm`` are returned as arcs (the curve there is an interpolation: nothing holds it),
    with the fraction of the length they cover. Pure."""
    x, y = _as_xy(x_nm, y_nm)
    length = float(path.length_nm)
    m = max(16, int(math.ceil(length / float(step_nm))))
    s_samp = np.asarray(np.arange(m) * (length / m), dtype=np.float64)
    pts = np.asarray(path.points_nm, dtype=np.float64)
    cum = np.asarray(path.cum_nm, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    q = np.column_stack([np.interp(s_samp, cum, closed[:, 0]), np.interp(s_samp, cum, closed[:, 1])])
    dm = density_map(np.concatenate([x, q[:, 0]]), np.concatenate([y, q[:, 1]]), pixel_nm=pixel_nm, sigma_nm=sigma_nm) \
        if x.size else None
    if dm is None:
        return ContourSupport(level=float("nan"), arcs_nm=((0.0, length),), unsupported_fraction=1.0,
                              sample_arc_nm=s_samp, supported=np.zeros(m, dtype=bool))
    # the grid covers the curve too, but the density is that of the localizations only
    h, _, _ = np.histogram2d(y, x, bins=list(dm.values.shape),
                             range=[[dm.y0, dm.y0 + dm.values.shape[0] * pixel_nm], [dm.x0, dm.x0 + dm.values.shape[1] * pixel_nm]])
    dm = DensityMap(x0=dm.x0, y0=dm.y0, pixel_nm=dm.pixel_nm,
                    values=np.asarray(ndimage.gaussian_filter(h, sigma_nm / pixel_nm, mode="constant"), dtype=np.float64))
    level = float(k) * reference_level(_local_maxima(dm, min_separation_nm)[1])
    sup = dm.sample(q) >= level
    arcs: List[Tuple[float, float]] = []
    total = 0.0
    if not sup.all():
        if not sup.any():
            arcs, total = [(0.0, length)], length
        else:
            start = int(np.flatnonzero(sup)[0])               # roll so the array starts on a supported sample
            rolled = np.roll(sup, -start)
            i = 0
            while i < m:
                if rolled[i]:
                    i += 1
                    continue
                j = i
                while j < m and not rolled[j]:
                    j += 1
                run = (j - i) * (length / m)
                if run > float(min_run_nm):
                    a0 = s_samp[(i + start) % m]
                    a1 = s_samp[(j - 1 + start) % m]
                    arcs.append((float(a0), float(a1)))
                    total += run
                i = j
    return ContourSupport(level=level, arcs_nm=tuple(arcs), unsupported_fraction=float(total / length) if length > 0 else 1.0,
                          sample_arc_nm=s_samp, supported=np.asarray(sup, dtype=bool))


def failure_reason(contour: AutoContour) -> str:
    """The rejection reason of a failed contour ("" when the status is ok). For a structured value without
    parsing text, use ``contour_details(...).assessment.failure_reason``."""
    if contour.status is Status.OK:
        return ""
    for w in contour.warnings:
        if w.startswith(REJECT_PREFIX):
            return w[len(REJECT_PREFIX):].split(":", 1)[0]
    return FAIL_FIT


def _fit(x: NDArray[np.float64], y: NDArray[np.float64], polygon: NDArray[np.float64], knot_nm: float) -> LocalizationMembrane:
    """The repo's localization P-spline on the polygon; ValueError passes through."""
    return fit_membrane_from_localizations(x, y, initial_path=_linear_closed_path(polygon), knot_spacing_nm=float(knot_nm))


def _leaves_cloud_nm(path: SmoothPath, x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """The persistent local offset of the band from the curve (0 when no stretch qualifies)."""
    s, r = project(path, x, y)
    _arcs, offs = window_offsets(path, s, r)
    v, _i = persistent_offset(offs)
    return v if math.isfinite(v) else 0.0


def _check(label: str, method: str, knot_nm: float, path: SmoothPath, nodes: NDArray[np.float64], residual: float,
           x: NDArray[np.float64], y: NDArray[np.float64], fit_warnings: Tuple[str, ...]) -> Candidate:
    """Orient the fitted curve counter-clockwise, then apply the rejection rules. The polar curve has its own
    notch limit (``POLAR_NOTCH_MIN_RADIUS_NM``) and, when ``POLAR_REJECT_IF_LEAVES_CLOUD``, is also rejected
    when the cloud leaves it (``FAIL_LEAVES_CLOUD``): it is not built from tour nodes, so the notch rule's
    argument (a lump node the spline went through) does not carry over to it."""
    path = counter_clockwise(path)
    n_cross = _self_crossings(np.asarray(path.points_nm, dtype=np.float64))
    hr = hull_ratio(path)
    reason, detail = "", ""
    lr = float("nan")
    notch = tightest_concave(path)[0] if n_cross == 0 else float("nan")
    notch_limit = POLAR_NOTCH_MIN_RADIUS_NM if method == "polar" else NOTCH_MIN_RADIUS_NM
    if n_cross > 0:
        reason, detail = FAIL_SELF_CROSSING, f"the curve crosses itself {n_cross} time(s)"
    elif not (hr <= MAX_HULL_RATIO):
        reason, detail = FAIL_IMPLAUSIBLE_LENGTH, (f"length {path.length_nm:.0f} nm is {hr:.2f}x the perimeter of its convex "
                                                   f"hull (limit {MAX_HULL_RATIO:g}): the curve snakes inside the ring")
    elif notch < notch_limit:
        reason, detail = FAIL_NOTCH, (f"the curve bends inward with a radius of {notch:.0f} nm (limit {notch_limit:g}): it "
                                      "went through an interior lump, background, or across the lumen where the membrane is sparse")
    elif method == "polar" and POLAR_REJECT_IF_LEAVES_CLOUD and \
            _leaves_cloud_nm(path, x, y) > QUALITY_MAX_LOCAL_OFFSET_NM:
        off = _leaves_cloud_nm(path, x, y)
        reason, detail = FAIL_LEAVES_CLOUD, (f"along a stretch the nearest band sits {off:.0f} nm from the polar curve "
                                             f"(limit {QUALITY_MAX_LOCAL_OFFSET_NM:g})")
    else:
        lr, _area = lumen_ratio(path, x, y)
        if lr > MAX_LUMEN_RATIO:
            reason, detail = FAIL_NO_LUMEN, (f"the density deeper than {LUMEN_DEPTH_NM:g} nm inside is {lr:.2f}x the band's "
                                             f"(limit {MAX_LUMEN_RATIO:g}): no ring around a lumen")
    return Candidate(label=label, method=method, knot_nm=float(knot_nm), path=path, nodes_nm=nodes, residual_sd_nm=float(residual),
                     n_self_crossings=n_cross, hull_ratio=hr, lumen_ratio=lr, reason=reason, detail=detail,
                     fit_warnings=fit_warnings)


def _failed(label: str, method: str, knot_nm: float, nodes: NDArray[np.float64], reason: str, detail: str) -> Candidate:
    return Candidate(label=label, method=method, knot_nm=float(knot_nm), path=None, nodes_nm=nodes, residual_sd_nm=float("nan"),
                     n_self_crossings=0, hull_ratio=float("nan"), lumen_ratio=float("nan"), reason=reason, detail=detail)


def detour_ratios(polygon: NDArray[np.float64]) -> NDArray[np.float64]:
    """(d(prev, v) + d(v, next)) / d(prev, next) for every vertex v of a closed polygon (inf when prev = next)."""
    p = np.asarray(polygon, dtype=np.float64)
    prv, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    base = np.hypot(*(nxt - prv).T)
    via = np.hypot(*(p - prv).T) + np.hypot(*(nxt - p).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(base > 0.0, via / np.maximum(base, 1e-300), np.inf)


def bypass_support(polygon: NDArray[np.float64], dm: DensityMap, level: float) -> NDArray[np.float64]:
    """Per vertex: the fraction of the middle 60 % of the chord prev -> next on pixels >= ``level``."""
    p = np.asarray(polygon, dtype=np.float64)
    prv, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    out = np.zeros(p.shape[0], dtype=np.float64)
    step = 0.5 * dm.pixel_nm
    for k in range(p.shape[0]):
        n = max(3, int(math.ceil(0.6 * float(np.hypot(*(nxt[k] - prv[k]))) / step)))
        t = 0.2 + 0.6 * (np.arange(n) + 0.5) / n
        pts = prv[k] + t[:, None] * (nxt[k] - prv[k])
        out[k] = float(np.mean(dm.sample(pts) >= level))
    return out


def _bypassed_tour(nodes: NDArray[np.float64], dm: DensityMap, level: float, label: str,
                   notes: List[str]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The 2-opt tour of every start, after dropping the vertices the membrane bypasses (``BYPASS_SUPPORT``):
    each pass drops every hit vertex not adjacent to another taken in the same pass (largest detour first),
    then rebuilds the tour; at most ``BYPASS_MAX_PASSES`` passes. Returns (tour polygon, nodes kept)."""
    removed = 0
    for _ in range(BYPASS_MAX_PASSES):
        tour = reconstruct_perimeter(nodes, all_starts=True)
        poly = np.asarray(tour.contour, dtype=np.float64)
        det = detour_ratios(poly)
        hit = (det > BYPASS_MIN_DETOUR) & (bypass_support(poly, dm, level) >= BYPASS_SUPPORT)
        if not hit.any() or nodes.shape[0] <= MIN_NODES:
            break
        # largest detour first; a vertex next to one already taken waits for the next pass (its chord changes)
        taken: List[int] = []
        n_v = poly.shape[0]
        for v in np.flatnonzero(hit)[np.argsort(-det[hit], kind="stable")]:
            if nodes.shape[0] - len(taken) <= MIN_NODES:
                break
            if any(min(abs(int(v) - t), n_v - abs(int(v) - t)) <= 1 for t in taken):
                continue
            taken.append(int(v))
        nodes = np.delete(nodes, [int(tour.order[v]) for v in taken], axis=0)
        removed += len(taken)
    else:
        tour = reconstruct_perimeter(nodes, all_starts=True)
        poly = np.asarray(tour.contour, dtype=np.float64)
    if removed:
        notes.append(f"{label}: {removed} node(s) bypassed (the band runs from their neighbour to their neighbour "
                     "without passing by them)")
    return poly, nodes


def _nodes_candidate(x: NDArray[np.float64], y: NDArray[np.float64], nodes0: NDArray[np.float64], *, knot_nm: float,
                     node_drop_nm: float, rounds: int, dm: DensityMap, level: float, notes: List[str],
                     first_tour: Optional[Tuple[NDArray[np.float64], NDArray[np.float64]]] = None) -> Candidate:
    """Tour of the nodes after the bypass test, P-spline, drop of the nodes off the curve, repeated
    ``rounds`` times. ``first_tour`` = the first (tour polygon, nodes) already built for ``nodes0`` (it does
    not depend on the knots, so the stiff fallback reuses the primary's)."""
    label = f"density_nodes knot {knot_nm:g}"
    nodes = nodes0
    if nodes.shape[0] < MIN_NODES:
        return _failed(label, "density_nodes", knot_nm, nodes, FAIL_TOO_FEW_NODES,
                       f"{nodes.shape[0]} density nodes (minimum {MIN_NODES})")
    fit = None
    for it in range(int(rounds) + 1):
        if it == 0 and first_tour is not None:
            polygon, nodes = first_tour
        else:
            polygon, nodes = _bypassed_tour(nodes, dm, level, label, notes)
        try:
            fit = _fit(x, y, polygon, knot_nm)
        except ValueError as exc:
            return _failed(label, "density_nodes", knot_nm, nodes, FAIL_FIT, str(exc))
        if it == int(rounds):
            break
        _s, r = project(fit.path, nodes[:, 0], nodes[:, 1])
        keep = np.abs(r) <= float(node_drop_nm)
        if bool(keep.all()):
            break
        if int(keep.sum()) < MIN_NODES:
            notes.append(f"{label}: dropping {int((~keep).sum())} nodes would leave fewer than {MIN_NODES}; kept the previous fit")
            break
        notes.append(f"{label}: round {it + 1} dropped {int((~keep).sum())} of {nodes.shape[0]} nodes farther than "
                     f"{node_drop_nm:g} nm from the curve")
        nodes = nodes[keep]
    assert fit is not None
    for _ in range(NOTCH_MAX_DROPS):
        radius, where = tightest_concave(fit.path)
        if not (radius < NOTCH_MIN_RADIUS_NM) or nodes.shape[0] <= MIN_NODES:
            break
        dn = np.hypot(*(nodes - where).T)
        k = int(np.argmin(dn))
        if dn[k] > NOTCH_NODE_RADIUS_NM:
            break
        trial_nodes = np.delete(nodes, k, axis=0)
        polygon, trial_nodes = _bypassed_tour(trial_nodes, dm, level, label, notes)
        try:
            trial = _fit(x, y, polygon, knot_nm)
        except ValueError:
            break
        notes.append(f"{label}: node at ({nodes[k, 0]:.0f}, {nodes[k, 1]:.0f}) nm dropped: the curve bent inward through it "
                     f"with a radius of {radius:.0f} nm (below {NOTCH_MIN_RADIUS_NM:g})")
        fit, nodes = trial, trial_nodes
    return _check(label, "density_nodes", knot_nm, fit.path, nodes, fit.residual_sd_nm, x, y, tuple(fit.warnings))


def tightest_concave(path: SmoothPath) -> Tuple[float, NDArray[np.float64]]:
    """(radius in nm, position) of the tightest INWARD bend of the closed curve (inf when it is convex),
    from ``tools.mps_curvature.curvature_of_closed_curve`` (signed, + convex) on the counter-clockwise table."""
    q, _s, kappa, _length = curvature_of_closed_curve(np.asarray(counter_clockwise(path).points_nm, dtype=np.float64))
    i = int(np.argmin(kappa))
    if not (kappa[i] < 0.0):
        return float("inf"), q[i]
    return float(-1.0 / kappa[i]), q[i]


def contour_details(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], *, method: str = "density_nodes",
                    knot_nm: float = CONTOUR_KNOT_NM, node_drop_nm: float = NODE_DROP_NM, rounds: int = NODE_ROUNDS,
                    drawn: Optional[NDArray[np.float64]] = None, pixel_nm: float = NODE_PIXEL_NM,
                    sigma_nm: float = NODE_SIGMA_NM, threshold_frac: float = NODE_THRESHOLD_FRAC,
                    min_separation_nm: float = NODE_MIN_SEPARATION_NM,
                    compute_polar: bool = True, fallbacks: bool = True,
                    exclude: Optional[NDArray[np.bool_]] = None,
                    slice_label: Optional[NDArray[np.integer]] = None) -> ContourDetails:
    """``auto_contour`` with every candidate kept (see the module docstring for the procedure).

    ``exclude`` (optional, decision D8; None = off, the default): boolean per localization, True = left out of
    the contour fit (interior spectrin trimmed by the unanchored-cluster package; ``interior_exclusion`` turns
    depths into this mask). The quality is still measured on ALL the localizations given (the band of the
    analysis counts them all): a trimming that removed real membrane shows up in the gate instead of hiding.
    Validation (report v2): against a valid reference the trimming removed <= 0.5 % (p95) of the true
    membrane; against a polar ring of a cloud that is not star-shaped ("C") it removed 5-22 %.
    ``slice_label`` (optional, decision D7): -1 / 0 / +1 per localization; the quality then includes the misfit
    between the z slices (``slice_misfit``) and flags the contour when the slices do not fall on one another."""
    if method not in CONTOUR_METHODS:
        raise ValueError(f"auto_contour: method must be one of {CONTOUR_METHODS}, got {method!r}")
    x, y = _as_xy(x_nm, y_nm)
    notes: List[str] = []
    n_excluded = 0
    lab: Optional[NDArray[np.integer]] = None
    if slice_label is not None:
        lab = np.asarray(slice_label).reshape(-1)
        if lab.shape != x.shape:
            raise ValueError(f"slice_label has {lab.size} entries for {x.size} localizations")
    if exclude is not None:
        ex = np.asarray(exclude, dtype=bool).reshape(-1)
        if ex.shape != x.shape:
            raise ValueError(f"exclude has {ex.size} entries for {x.size} localizations")
        n_excluded = int(np.count_nonzero(ex))
        if n_excluded:
            x, y = x[~ex], y[~ex]
            notes.append(f"interior trimming on: {n_excluded} of {ex.size} localizations ({n_excluded / ex.size:.1%}) left "
                         "out of the contour fit (decision D8; the quality is measured with all of them)")
    xq, yq = _as_xy(x_nm, y_nm)              # the quality reads every localization given
    no_nodes = np.zeros((0, 2), dtype=np.float64)
    qcache: Dict[int, ContourQuality] = {}

    def quality_of(c: Candidate) -> ContourQuality:
        assert c.path is not None
        if id(c) not in qcache:
            qcache[id(c)] = contour_quality(c.path, xq, yq, slice_label=lab)
        return qcache[id(c)]

    def leaves_cloud(c: Candidate) -> bool:
        """The curve-specific part of the quality gate: along a stretch the band sits to one side of the curve."""
        return c.path is not None and quality_of(c).persistent_offset_nm > QUALITY_MAX_LOCAL_OFFSET_NM

    def kind_of(c: Candidate) -> str:
        if c.method == "drawn":
            return "drawn"
        if c.method == "polar":
            return "polar" if method == "polar" else "polar fallback"
        return "primary" if c.knot_nm == float(knot_nm) else "stiff fallback"

    def finish(cands: List[Candidate], polar_path: Optional[SmoothPath], n_nodes0: int, polar_dist: float,
               rejected: Optional[SmoothPath]) -> ContourDetails:
        ok_cands = [c for c in cands if c.reason == "" and c.path is not None]
        winner = ok_cands[0] if ok_cands else None
        if winner is not None and QUALITY_FALLBACK and leaves_cloud(winner):
            better = next((c for c in ok_cands if not leaves_cloud(c)), None)
            if better is not None:
                notes.append(f"{winner.label} accepted by the rules but along a stretch the band sits "
                             f"{quality_of(winner).persistent_offset_nm:.0f} nm to one side of it; {better.label} follows "
                             f"the cloud ({quality_of(better).persistent_offset_nm:.0f} nm) and is used")
                winner = better
        warnings: List[str] = []
        if winner is None:
            first = cands[0]
            warnings.append(f"{REJECT_PREFIX}{first.reason}: {first.detail}")
            for c in cands[1:]:
                warnings.append(f"fallback {c.label} also rejected: {c.reason}: {c.detail}")
            warnings.extend(notes)
            ac = AutoContour(path=None, length_nm=float("nan"), method=method, knot_nm=float(knot_nm),
                             residual_sd_nm=float("nan"), n_self_crossings=int(first.n_self_crossings),
                             nodes_nm=first.nodes_nm, polar_distance_nm=polar_dist, star_shaped=False,
                             status=Status.CONTOUR_FAILED, warnings=tuple(warnings))
            return ContourDetails(contour=ac, candidates=tuple(cands), polar_path=polar_path, n_nodes_initial=n_nodes0,
                                  rejected_path=rejected, notes=tuple(notes),
                                  assessment=ContourAssessment(kind="none", failure_reason=first.reason, quality=None,
                                                               support=None, reliable=False, n_excluded=n_excluded))
        assert winner.path is not None
        quality = quality_of(winner)
        for c in cands:
            if c is winner:
                break
            why = f"{c.reason}: {c.detail}" if c.reason else "the cloud leaves it (local offset)"
            warnings.append(f"{c.label} not used ({why}); the curve below is the fallback {winner.label}")
        if winner.method != method:
            warnings.append(f"method asked: {method}; method used: {winner.method}")
        if kind_of(winner) == "stiff fallback":
            warnings.append(f"stiff fallback ({winner.knot_nm:g} nm knots): concavities tighter than the {knot_nm:g} nm curve "
                            "could follow are smoothed over; read the local offset in the quality")
        pts = np.asarray(winner.path.points_nm, dtype=np.float64)
        star = is_star_shaped(winner.path, _area_centroid(_subsample(pts, 2000)))
        dist = polar_dist
        if polar_path is not None and winner.method != "polar":
            dist = curve_distance_nm(winner.path, polar_path)
            if dist > POLAR_WARN_NM:
                if star:
                    warnings.append(f"the curve is star-shaped and differs from the polar curve by a mean {dist:.0f} nm "
                                    f"(warning above {POLAR_WARN_NM:g} nm)")
                else:
                    warnings.append(f"the curve is not star-shaped and differs from the polar curve by a mean {dist:.0f} nm "
                                    f"(warning above {POLAR_WARN_NM:g} nm; the polar curve is only a rough reference for a "
                                    "shape that is not star-shaped)")
        elif winner.method == "polar":
            dist = float("nan")         # the polar curve compared with itself is not an agreement
        support = contour_support(winner.path, x, y)
        if support.arcs_nm:
            warnings.append(f"{len(support.arcs_nm)} stretch(es) of the curve, {support.unsupported_fraction:.0%} of its length, "
                            f"have no localization density under them (> {UNSUPPORTED_GAP_NM:g} nm): the curve is an "
                            "interpolation there")
        if quality.low_quality:
            warnings.insert(0, LOW_QUALITY_PREFIX + "; ".join(quality.reasons))
        if winner.hull_ratio > WARN_HULL_RATIO:
            warnings.append(f"length is {winner.hull_ratio:.2f}x the convex hull's perimeter (warning above {WARN_HULL_RATIO:g})")
        if math.isnan(winner.lumen_ratio):
            warnings.append("lumen test not possible (inner area too small)")
        warnings.extend(winner.fit_warnings)
        warnings.extend(notes)
        ac = AutoContour(path=winner.path, length_nm=float(winner.path.length_nm), method=winner.method,
                         knot_nm=winner.knot_nm, residual_sd_nm=winner.residual_sd_nm, n_self_crossings=0,
                         nodes_nm=winner.nodes_nm, polar_distance_nm=float(dist), star_shaped=bool(star),
                         status=Status.OK, warnings=tuple(warnings))
        return ContourDetails(contour=ac, candidates=tuple(cands), polar_path=polar_path, n_nodes_initial=n_nodes0,
                              rejected_path=rejected, notes=tuple(notes),
                              assessment=ContourAssessment(kind=kind_of(winner), failure_reason="", quality=quality,
                                                           support=support, reliable=not quality.low_quality,
                                                           n_excluded=n_excluded))

    if x.size < MIN_LOCALIZATIONS:
        c = _failed(f"{method} knot {knot_nm:g}", method, knot_nm, no_nodes, FAIL_TOO_FEW_LOCS,
                    f"{x.size} localizations (minimum {MIN_LOCALIZATIONS})")
        return finish([c], None, 0, float("nan"), None)

    # The polar curve: the cross-check, the polar method itself, and the last fallback.
    polar_cand: Optional[Candidate] = None
    polar_path: Optional[SmoothPath] = None
    if compute_polar or method == "polar" or fallbacks:
        poly = polar_contour(x, y)
        label = f"polar knot {knot_nm:g}"
        if poly is None:
            polar_cand = _failed(label, "polar", knot_nm, no_nodes, FAIL_NO_POLAR, "too few sectors hold localizations")
        else:
            try:
                pf = _fit(x, y, poly, knot_nm)
                polar_cand = _check(label, "polar", knot_nm, pf.path, no_nodes, pf.residual_sd_nm, x, y, tuple(pf.warnings))
                polar_path = polar_cand.path
            except ValueError as exc:
                polar_cand = _failed(label, "polar", knot_nm, no_nodes, FAIL_FIT, str(exc))

    if method == "polar":
        assert polar_cand is not None
        return finish([polar_cand], polar_path, 0, float("nan"), polar_cand.path if polar_cand.reason else None)

    if method == "drawn":
        label = f"drawn knot {knot_nm:g}"
        if drawn is None:
            c = _failed(label, "drawn", knot_nm, no_nodes, FAIL_NO_DRAWN, "method 'drawn' needs the drawn polygon")
        else:
            poly_d = np.asarray(drawn, dtype=np.float64)
            if poly_d.ndim != 2 or poly_d.shape[1] != 2 or poly_d.shape[0] < 3 or not np.isfinite(poly_d).all():
                c = _failed(label, "drawn", knot_nm, no_nodes, FAIL_NO_DRAWN, f"the drawn polygon must be finite (M >= 3, 2), got {poly_d.shape}")
            else:
                try:
                    df = _fit(x, y, poly_d, knot_nm)
                    c = _check(label, "drawn", knot_nm, df.path, poly_d, df.residual_sd_nm, x, y, tuple(df.warnings))
                except ValueError as exc:
                    c = _failed(label, "drawn", knot_nm, poly_d, FAIL_FIT, str(exc))
        dist = curve_distance_nm(c.path, polar_path) if (c.path is not None and polar_path is not None) else float("nan")
        return finish([c], polar_path if compute_polar else None, 0, dist, c.path if c.reason else None)

    # density nodes
    nodes0, _w = density_nodes(x, y, pixel_nm=pixel_nm, sigma_nm=sigma_nm, threshold_frac=threshold_frac,
                               min_separation_nm=min_separation_nm)
    if nodes0.shape[0] > NODE_WARN_COUNT:
        notes.append(f"{nodes0.shape[0]} density nodes (warning above {NODE_WARN_COUNT}): the tour of every start grows as "
                     "~N^2.5; expect a slow contour")
    if nodes0.shape[0] > NODE_MAX_COUNT:
        notes.append(f"only the {NODE_MAX_COUNT} strongest of {nodes0.shape[0]} density nodes kept (time cap)")
        nodes0 = nodes0[:NODE_MAX_COUNT]           # density_nodes sorts by decreasing density
    dm = density_map(x, y, pixel_nm=pixel_nm, sigma_nm=sigma_nm)
    level = BYPASS_LEVEL_FRAC * reference_level(_local_maxima(dm, min_separation_nm)[1])
    first_tour = _bypassed_tour(nodes0, dm, level, "density_nodes", notes) if nodes0.shape[0] >= MIN_NODES else None
    cands: List[Candidate] = [_nodes_candidate(x, y, nodes0, knot_nm=knot_nm, node_drop_nm=node_drop_nm, rounds=rounds,
                                               dm=dm, level=level, notes=notes, first_tour=first_tour)]
    rejected = cands[0].path if cands[0].reason else None
    need_more = bool(cands[0].reason) or (QUALITY_FALLBACK and leaves_cloud(cands[0]))
    if need_more and fallbacks and cands[0].reason != FAIL_TOO_FEW_LOCS:
        if cands[0].reason != FAIL_TOO_FEW_NODES and float(knot_nm) < FALLBACK_KNOT_NM:
            cands.append(_nodes_candidate(x, y, nodes0, knot_nm=FALLBACK_KNOT_NM, node_drop_nm=node_drop_nm,
                                          rounds=rounds, dm=dm, level=level, notes=notes, first_tour=first_tour))
        if (cands[-1].reason or (QUALITY_FALLBACK and leaves_cloud(cands[-1]))) and polar_cand is not None:
            if polar_cand.path is not None and polar_cand.reason == "":
                pts = np.asarray(polar_cand.path.points_nm, dtype=np.float64)
                if not is_star_shaped(polar_cand.path, _area_centroid(_subsample(pts, 2000))):
                    polar_cand = replace(polar_cand, reason=FAIL_NO_POLAR,
                                         detail="the polar curve is not star-shaped around its centre")
            cands.append(polar_cand)
    return finish(cands, polar_path if compute_polar else None, int(nodes0.shape[0]), float("nan"), rejected)


def auto_contour(x_nm: NDArray[np.float64], y_nm: NDArray[np.float64], *, method: str = "density_nodes",
                 knot_nm: float = CONTOUR_KNOT_NM, node_drop_nm: float = NODE_DROP_NM, rounds: int = NODE_ROUNDS,
                 drawn: Optional[NDArray[np.float64]] = None, pixel_nm: float = NODE_PIXEL_NM,
                 sigma_nm: float = NODE_SIGMA_NM, threshold_frac: float = NODE_THRESHOLD_FRAC,
                 min_separation_nm: float = NODE_MIN_SEPARATION_NM,
                 compute_polar: bool = True, fallbacks: bool = True,
                 exclude: Optional[NDArray[np.bool_]] = None,
                 slice_label: Optional[NDArray[np.integer]] = None) -> AutoContour:
    """
    Fit the closed membrane curve to the localizations (no cluster labels). ``status`` is
    ``Status.CONTOUR_FAILED`` (``path`` None, reason in the first warning, ``failure_reason``) on a
    self-crossing, an implausible length, no lumen, an aborted fit, too few localizations or fewer than
    ``MIN_NODES`` nodes. ``method="drawn"`` refits the person's polygon ``drawn`` with the same P-spline
    and the same checks (comparison only). The keyword defaults are the module constants; the node
    parameters are exposed for the sensitivity sweep. ``exclude`` / ``slice_label``: see ``contour_details``
    (both off by default). An accepted contour whose quality gate fails keeps ``status = ok`` and its first
    warning starts with ``LOW_QUALITY_PREFIX``; ``contour_details(...).assessment.reliable`` says it as a field.
    """
    return contour_details(x_nm, y_nm, method=method, knot_nm=knot_nm, node_drop_nm=node_drop_nm, rounds=rounds,
                           drawn=drawn, pixel_nm=pixel_nm, sigma_nm=sigma_nm, threshold_frac=threshold_frac,
                           min_separation_nm=min_separation_nm, compute_polar=compute_polar,
                           fallbacks=fallbacks, exclude=exclude, slice_label=slice_label).contour


def interior_exclusion(depth_nm: NDArray[np.float64], *, depth_limit_nm: float = INTERIOR_TRIM_DEPTH_NM) -> NDArray[np.bool_]:
    """The ``exclude`` mask of the optional interior trimming (decision D8) from depths INSIDE a reference ring
    (nm, positive inward; negative or NaN = on or outside the ring), as the unanchored-cluster package gives
    them: True where the depth exceeds ``depth_limit_nm``. Nothing here computes the reference ring: whoever
    calls decides it (and must not trim when the reference is not valid, e.g. a polar ring of a cloud that is
    not star-shaped)."""
    d = np.asarray(depth_nm, dtype=np.float64).reshape(-1)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.isfinite(d) & (d > float(depth_limit_nm)), dtype=bool)


def assessment_row(details: ContourDetails) -> Dict[str, object]:
    """The assessment as flat columns (names without the ``pd_`` prefix the export adds): curve kind,
    reliability and its reasons, failure reason, fit quality, support, slice misfit and trimming. NaN / ""
    where not computed."""
    a = details.assessment
    nan = float("nan")
    row: Dict[str, object] = {
        "contour_kind": a.kind if a is not None else "none",
        "contour_reliable": bool(a.reliable) if a is not None else False,
        "contour_failure_reason": a.failure_reason if a is not None else FAIL_FIT,
        "contour_quality_reasons": " | ".join(a.quality.reasons) if (a is not None and a.quality is not None) else "",
        "contour_frac100": nan, "contour_robust_sd_nm": nan, "contour_persistent_offset_nm": nan,
        "contour_off_band_fraction": nan, "contour_max_window_offset_nm": nan,
        "contour_unsupported_fraction": nan, "contour_n_unsupported_arcs": 0,
        "contour_slice_rms_minus_nm": nan, "contour_slice_rms_plus_nm": nan,
        "contour_slice_radius_change_nm": nan, "contour_slice_lateral_minus_nm": nan,
        "contour_slice_lateral_plus_nm": nan, "contour_slice_frac100_central": nan, "contour_slice_frac100_all": nan,
        "contour_n_excluded": int(a.n_excluded) if a is not None else 0,
    }
    if a is not None and a.quality is not None:
        q = a.quality
        row.update(contour_frac100=q.frac_within[100.0], contour_robust_sd_nm=q.robust_sd_nm,
                   contour_persistent_offset_nm=q.persistent_offset_nm, contour_off_band_fraction=q.off_band_fraction,
                   contour_max_window_offset_nm=q.max_local_offset_nm)
        if q.slices is not None:
            sm = q.slices
            row.update(contour_slice_rms_minus_nm=sm.rms_displacement_nm[0], contour_slice_rms_plus_nm=sm.rms_displacement_nm[1],
                       contour_slice_radius_change_nm=sm.radius_change_per_slice_nm,
                       contour_slice_lateral_minus_nm=sm.lateral_vs_central_nm[0], contour_slice_lateral_plus_nm=sm.lateral_vs_central_nm[1],
                       contour_slice_frac100_central=sm.frac100_central, contour_slice_frac100_all=sm.frac100_all)
    if a is not None and a.support is not None:
        row.update(contour_unsupported_fraction=a.support.unsupported_fraction, contour_n_unsupported_arcs=len(a.support.arcs_nm))
    return row
