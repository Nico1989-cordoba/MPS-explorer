# NeNA: the localization precision measured from the data

> **Resumen (español).** NeNA (Endesfelder et al. 2014) mide la precisión de localización con los propios datos: una
> molécula que sigue encendida en el cuadro siguiente aparece dos veces, y la distancia entre esas dos posiciones sólo
> refleja el error de medición. MPS Explorer ajusta el mismo modelo que Picasso (un término de "misma molécula" más uno
> de "moléculas cercanas"), pero hasta el 2026-10-14 lo hacía por mínimos cuadrados sobre el histograma, y ese ajuste
> se quedaba a veces con un pico demasiado angosto: en 20 archivos sintéticos de precisión conocida (σ = 12 nm) dio
> 0,81-0,95 σ en 5. Ahora el modelo se ajusta por **máxima verosimilitud** sobre las distancias, con el término de
> moléculas cercanas nunca más angosto que √2·s y **usado sólo cuando los datos lo justifican** (criterio BIC); en los
> mismos 20 archivos queda dentro de ±3 % en todos. Junto al valor se informan un **intervalo del 95 %** (perfil de
> verosimilitud), una **estimación independiente** (el mismo ajuste con un término de vecinos al azar) y un aviso cuando
> el campo está tan poblado que la regla del vecino más cercano elige otra molécula (entonces NeNA da **bajo**, sea cual
> sea el ajuste). Un aviso nombra cada problema; el valor nunca se reemplaza por otro. El valor anterior (Picasso,
> mínimos cuadrados) se sigue mostrando para comparar. Validación en simulaciones propias (sección 5).

This page explains what NeNA measures in MPS Explorer, the model and how it is fitted, why the fit was changed on
2026-10-14, the interval and the warnings reported next to the value, the validation on synthetic data, and the
limitations. The code is `nena()` and `precision_check()` in `tools/mps_quality.py`; the Data quality window
(`tools/mps_quality_window.py`, tab Precision) shows it. Every number below is labelled with where it comes from:

- **literature**: a published value or method, with its reference;
- **derivation**: shown here;
- **own simulation**: synthetic data only (no measured file), with the script and seed that reproduce it. The scripts
  are in `docs/sims/nena/`: `nena_sim.py` (the generator), `nena_diagnosis.py` (section 2, writes
  `nena_diagnosis.json` and `.png`) and `nena_grid.py` (section 5, writes `nena_grid.json`). The JSON committed is
  the run quoted here.

## 1. What NeNA measures

A fluorophore that stays on for several camera frames is localized once per frame. Two localizations of the same
molecule in consecutive frames differ only by their two measurement errors; with an isotropic Gaussian error of
standard deviation σ per axis, the difference has σ√2 per axis and its length follows a Rayleigh law of scale √2·s:

    p_same(d) = d / (2 s²) · exp(−d² / 4 s²),        s = σ                         (derivation)

NeNA (nearest neighbour in adjacent frames; Endesfelder et al. 2014, literature) takes, for every localization, the
distance to the nearest localization in the next frame, and reads s off the peak of that distribution. It needs no
photon calibration, which is what makes it the check on the precision the localization fit reports (lpx, lpy), and
the input of the channel-2 DBSCAN parameters (eps₂ = eps₁ · NeNA₂ / NeNA₁).

The nearest neighbour in the next frame is not always the same molecule: when the molecule switched off, or when
another localization is closer, the pair is a "nearby" pair, and its distance follows the spatial arrangement of the
other molecules. Picasso (`picasso.postprocess.nena`; Schnitzbauer et al. 2017, literature) models those pairs with a
Gaussian term and fits

    a · p_same(d; s)  +  ac / (sc √(2π)) · exp(−(d − dc)² / 2 sc²),    a ≥ ac,          (Picasso's model)

by least squares on the histogram of distances (Picasso's code, read 2026-10-14: `scipy.optimize.curve_fit` from three
starting points, the smallest residual kept, histogram bins of 0.001 camera pixel up to 1 pixel). MPS Explorer used
the same fit on 0.5 nm bins up to `max_distance_nm` = 120 nm. Endesfelder et al. fit their own distribution function;
its exact form is not reproduced here (the article is paywalled and was not re-read for this page); comparability is
kept with Picasso, whose code was read.

## 2. Why the previous fit failed

Own simulation (`docs/sims/nena/nena_diagnosis.py`; the generator of the channel-2 tests: 600 molecules over a
25 µm field, 4000 frames, 1 + Poisson(3) switching events of Geometric(0.4) frames, σ = 12 nm per axis, seeds 0-19;
and 20 files of `nena_sim.py` at σ = 12 nm with about 2000 pairs and no other molecules nearby, seeds
900000-900019). In these files every next-frame pair within 120 nm is the same molecule, so the right answer is a
pure Rayleigh peak with s = 12 nm.

| Estimator (cumulative changes) | channel-2 files: off by > 5 % | min / max (× σ) | nena_sim files: off by > 5 % | min / max (× σ) |
|---|---|---|---|---|
| A. Picasso least squares, 3 starts (previous) | **5 / 20** | 0.805 / 1.028 | **3 / 20** | 0.677 / 1.033 |
| B. the same least squares from 36 starts | 2 / 20 | 0.938 / 1.028 | 3 / 20 | 0.677 / 1.067 |
| C. maximum likelihood, nearby width free | 1 / 20 | 0.872 / 1.021 | 7 / 20 | 0.787 / 1.020 |
| D. C with nearby width ≥ √2·s | 0 / 20 | 0.974 / 1.020 | 2 / 20 | 0.755 / 1.019 |
| **E. D with the nearby term kept only when BIC supports it (new)** | **0 / 20** | 0.982 / 1.026 | **0 / 20** | 0.965 / 1.018 |
| oracle: √(⟨d²⟩/4) of the same-molecule pairs | 0 / 20 | 0.984 / 1.026 | 0 / 20 | 0.965 / 1.018 |

The five low channel-2 files are seeds 2, 3, 7, 8 and 11 (0.937, 0.805, 0.946, 0.938, 0.938 σ). What happened in them:
the Gaussian "nearby" term settled on the right shoulder of the same-molecule peak (dc ≈ 28-33 nm, sc ≈ 11-15 nm) and
took part of it, so the Rayleigh term that is read as the precision came out narrower. There is nothing there for a
nearby term to describe. Two mechanisms make this happen, and the table separates them:

- **Local minima** (A → B). All three of Picasso's starting points put 10 % of the area in the nearby term, and the
  least-squares landscape is rugged; more starts recover most channel-2 files. `nena_diagnosis.png` (right) profiles
  the residual over s for files 3 and 7: even with 32 starts at each fixed s the low-residual solution is found only
  near s = 12 nm.
- **Over-fitting** (B, C → D → E). With three free parameters, the nearby term can always lower the residual or raise
  the likelihood a little by carving the peak; least squares from many starts still fails on 3 of the nena_sim files,
  and maximum likelihood with a free width fails on 7 (the Gaussian narrows onto the peak). Bounding its width (D) and
  then asking whether it is needed at all (E) removes the failures; E coincides with the oracle.

A direct moment estimate √(⟨d²⟩/4) would also have been right on these files, but it is not a replacement: any
nearby pair inflates it, because its distances add to ⟨d²⟩ in full.

## 3. The estimator

`nena()` keeps Picasso's model and changes how it is fitted.

| Choice | Value | Origin |
|---|---|---|
| Fit | maximum likelihood on the distances: multinomial likelihood of 0.05 nm bins with exact bin probabilities (differences of the two components' CDFs), each component normalized on [0, `max_distance_nm`) | derivation: binning loses only the position of a distance inside 0.05 nm, 1/60 of the smallest precision expected (3 nm); the likelihood weights every pair equally, where unweighted least squares on 0.5 nm bins over-weights the noisy, sparsely filled bins |
| Same-molecule share w (Picasso's a ≥ ac) | w ≥ 0.5 | literature (Picasso's constraint), restated for normalized components |
| Nearby width | sc ≥ √2 · s | derivation: two DIFFERENT molecules a fixed distance D apart give localization distances spread radially by both errors, √2·s; a spread of D only widens it. A narrower term is not a population of neighbours; it is the fit carving the peak |
| Nearby centre | 0 ≤ dc ≤ 3 · `max_distance_nm` | convention: a centre beyond the window lets the term act as a rising ramp, the shape of randomly placed neighbours |
| Is the nearby term used? | only if 2 ΔlogL > 3 · ln(n) | literature: Bayesian information criterion (Schwarz 1978) for 3 extra parameters (w, dc, sc); otherwise the model is Picasso's with ac = 0, inside Picasso's own bounds |
| Starting points | 24: s from the median distance (median of a Rayleigh law = 1.665 s) × 0.6, 1, 1.4, plus the reported lpx/lpy when given; w 0.95 and 0.7; the nearby term at (3 s, 2 s), (D/2, D/4), (D, D/3) | convention; the profile (section 4) re-checks the optimum |
| Window `max_distance_nm` | 120 nm (unchanged) | convention, as before (Picasso uses 1 camera pixel) |
| Minimum pairs `MIN_NENA_PAIRS` | 500 (unchanged) | ad hoc when first set; a derivation agrees with it: for same-molecule pairs the ML Rayleigh scale has standard error s / (2√n), so its 95 % interval is narrower than ±5 % once n ≥ (1.96 / (2 × 0.05))² = 384 |

The value is `NeNAResult.precision_nm` as before. `lsq_nm` keeps Picasso's least-squares value from the same distances
for comparison (Data quality shows it as "Picasso least squares (before 2026-10-14)").

## 4. The interval and the warnings

**95 % interval** (`ci_low_nm`, `ci_high_nm`). Profile likelihood (Wilks 1938, literature): the values of s whose
log-likelihood, maximized over the other parameters, lies within 3.841 / 2 of the maximum (χ²₁ at 0.95). Without the
nearby term the likelihood has one parameter and the interval is read off it directly. The profile also re-checks the
global optimum: if it ever finds a higher likelihood, the fit restarts from there.

**Independent estimate** (`random_term_nm`). The same fit with a third term, the nearest-neighbour law of randomly
placed localizations (2πρd·exp(−πρd²), a Rayleigh law of its own, scale ≥ 2 s; derivation). It is not reported as
NeNA; it is what the agreement check compares against. Its likelihood has several local optima, so it is started
from a fixed grid of 20 points around the two-component fit and the best is kept: the check then does not depend on
the starting point (an earlier version with 4 starts changed its verdict when the reported lpx was added as a start).

**Warnings** (`flags`, `warnings`, in Data quality's findings; `well_identified` is False for the first three):

| Flag | When | Meaning |
|---|---|---|
| `wide_interval` | the interval is wider than ±10 % of the value, or does not close inside the fitted range | the distances cannot pin NeNA down (ad hoc limit: twice the 5 % tolerance) |
| `nearby_dominates` | the same-molecule share is at Picasso's limit (w ≤ 0.51) | the peak is not separated from the background; NeNA may be too large |
| `disagreement` | the independent estimate is outside the interval AND more than 5 % away | the background is not the shape the model assumes; check the fitted curve before using either value |
| `crowded` | more than 15 % of the close pairs (d < 2√2 s) have a second next-frame localization just as close | the nearest-neighbour rule picks another molecule; NeNA reads **low** whatever the fit (section 6) |
| `window_short` | more than 10 % of the same-molecule law lies beyond `max_distance_nm` | the fit accounts for the cut; a longer window measures better |
| `no_peak` | the fit ran to its ceiling (s = `max_distance_nm` / 2); `converged` is False | no same-molecule peak: a linked file, or random neighbours swamp it |
| `too_few_pairs` | fewer than 500 pairs; `converged` is False | normal for a picked region; use the whole field |

The value is never replaced by another estimate: a warning names the problem and the numbers.

## 5. Validation

Own simulation (`docs/sims/nena/nena_grid.py`, `nena_sim.py`; seeds 1000 × cell index + replicate). σ ∈ {6, 10, 12,
20, 30} nm; contamination: none; uniform single-frame background tuned to about 10 / 30 / 50 % of the next-frame pairs
(closed form in `background_for`; the share reached is reported); molecules in clusters of 10 (Gaussian spread 20 nm)
with on average 0.1 (moderate) or 0.4 (heavy) cluster mates on in the next frame; target same-molecule pairs 500,
2000, 10000 (all σ) and 50000 (σ = 12 nm only, for the compute budget); 10 files per cell; plus linked files (Picasso
Link without dark frames) of the "none" and "clustered, heavy" kinds. Window 120 nm. "Old" is Picasso's least squares
as the channel-2 code calls it (no seed); "> 5 %" counts files more than 5 % from σ; "unflagged" counts those that also
carry no warning; the oracle is √(⟨d²⟩/4) of the pairs the nearest-neighbour rule matched to the same molecule, i.e.
the best any fit of these distances could do.

| contamination | files | old >5 % | new >5 % | new >5 % without a flag | oracle >5 % | median abs(old − 1) | median abs(new − 1) | CI covers σ | CI covers oracle |
|---|---|---|---|---|---|---|---|---|---|
| none | 160 | 15 % (136) | 0 % (136) | 0 % | 5 % | 1.2 % | 0.7 % | 90 % | 85 % |
| uniform 10 % | 160 | 14 % (157) | 15 % (157) | 8 % | 5 % | 1.4 % | 1.2 % | 81 % | 83 % |
| uniform 30 % | 160 | 19 % (160) | 19 % (160) | 7 % | 8 % | 1.7 % | 1.7 % | 74 % | 75 % |
| uniform 50 % | 160 | 74 % (160) | 34 % (160) | 6 % | 4 % | 7.2 % | 3.1 % | 52 % | 55 % |
| clustered, moderate | 160 | 23 % (145) | 3 % (145) | 2 % | 8 % | 2.2 % | 1.1 % | 79 % | 69 % |
| clustered, heavy | 160 | 64 % (160) | 40 % (160) | 1 % | 36 % | 6.3 % | 3.7 % | 37 % | 61 % |
| linked, none | 100 | — (0) | — (0) | 0 % | 43 % | — | — | — | — |
| linked, clustered heavy | 100 | 92 % (50) | 82 % (50) | 4 % | 99 % | 21.5 % | 34.7 % | 18 % | 2 % |

Per cell (bias = median of estimate/σ − 1; sd = spread of estimate/σ):

<details><summary>All cells</summary>

| contamination | σ (nm) | pairs (target) | pairs (median) | non-self pairs | old bias % | old sd % | old >5 % | new bias % | new sd % | new >5 % | new >5 % unflagged | oracle bias % | random-term bias % | CI covers σ | CI covers oracle | flagged | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform 10 % | 6 | 500 | 547 | 11 % | -0.3 | 3 | 0 % | -0.3 | 3 | 0 % | 0 % | -0.2 | -0.5 | 100 % | 100 % | 10 % | too_few_pairs 1 |
| uniform 10 % | 6 | 2000 | 2262 | 10 % | -0.9 | 1 | 0 % | -0.1 | 1 | 0 % | 0 % | -0.4 | -0.2 | 100 % | 100 % | 0 % |  |
| uniform 10 % | 6 | 10000 | 11210 | 10 % | +0.1 | 0 | 0 % | -0.4 | 0 | 0 % | 0 % | -0.4 | -0.3 | 100 % | 100 % | 0 % |  |
| uniform 10 % | 10 | 500 | 556 | 9 % | -3.1 | 8 | 22 % | -2.1 | 3 | 11 % | 10 % | +0.3 | -2.4 | 89 % | 100 % | 20 % | disagreement 1, too_few_pairs 1 |
| uniform 10 % | 10 | 2000 | 2206 | 10 % | +0.3 | 1 | 0 % | -0.3 | 1 | 0 % | 0 % | -0.4 | -0.3 | 90 % | 100 % | 0 % |  |
| uniform 10 % | 10 | 10000 | 11274 | 10 % | -0.1 | 1 | 0 % | -0.1 | 1 | 0 % | 0 % | -0.6 | -0.5 | 100 % | 100 % | 0 % |  |
| uniform 10 % | 12 | 500 | 580 | 10 % | +1.7 | 4 | 22 % | +2.2 | 3 | 0 % | 0 % | +0.1 | +0.7 | 89 % | 100 % | 10 % | too_few_pairs 1 |
| uniform 10 % | 12 | 2000 | 2198 | 10 % | -0.6 | 2 | 0 % | -0.0 | 1 | 0 % | 0 % | -0.4 | -1.0 | 100 % | 100 % | 0 % |  |
| uniform 10 % | 12 | 10000 | 11205 | 10 % | +0.3 | 1 | 0 % | +0.2 | 1 | 0 % | 0 % | -0.3 | -0.1 | 100 % | 100 % | 0 % |  |
| uniform 10 % | 12 | 50000 | 55873 | 11 % | +0.2 | 0 | 0 % | +0.1 | 0 | 0 % | 0 % | -0.2 | -0.2 | 90 % | 100 % | 0 % |  |
| uniform 10 % | 20 | 500 | 534 | 10 % | +0.3 | 17 | 40 % | -0.9 | 6 | 30 % | 20 % | -0.3 | -1.7 | 80 % | 90 % | 30 % | wide_interval 3 |
| uniform 10 % | 20 | 2000 | 2192 | 10 % | +1.3 | 3 | 0 % | +1.5 | 2 | 10 % | 10 % | -0.3 | +0.2 | 90 % | 100 % | 0 % |  |
| uniform 10 % | 20 | 10000 | 11152 | 10 % | +0.5 | 1 | 0 % | +1.3 | 1 | 0 % | 0 % | +0.1 | +0.8 | 60 % | 80 % | 0 % |  |
| uniform 10 % | 30 | 500 | 570 | 11 % | -2.5 | 13 | 60 % | +9.3 | 4 | 80 % | 50 % | -2.7 | +0.7 | 20 % | 0 % | 50 % | disagreement 5 |
| uniform 10 % | 30 | 2000 | 2226 | 10 % | -3.9 | 7 | 50 % | +8.8 | 5 | 100 % | 30 % | -3.8 | -5.6 | 10 % | 10 % | 70 % | disagreement 6, wide_interval 1 |
| uniform 10 % | 30 | 10000 | 10883 | 11 % | +1.1 | 4 | 30 % | +1.0 | 3 | 10 % | 10 % | -3.7 | -1.0 | 80 % | 50 % | 0 % |  |
| uniform 30 % | 6 | 500 | 683 | 31 % | -0.4 | 2 | 0 % | -1.1 | 2 | 10 % | 10 % | -1.1 | -1.1 | 90 % | 100 % | 0 % |  |
| uniform 30 % | 6 | 2000 | 2854 | 30 % | +0.6 | 1 | 0 % | +0.6 | 1 | 0 % | 0 % | +0.3 | +0.2 | 100 % | 100 % | 0 % |  |
| uniform 30 % | 6 | 10000 | 14306 | 30 % | -0.2 | 1 | 0 % | -0.6 | 0 | 0 % | 0 % | -0.5 | -0.5 | 90 % | 100 % | 0 % |  |
| uniform 30 % | 10 | 500 | 726 | 31 % | +2.7 | 3 | 30 % | +2.7 | 2 | 10 % | 10 % | +0.8 | +1.6 | 100 % | 100 % | 0 % |  |
| uniform 30 % | 10 | 2000 | 2904 | 29 % | -0.5 | 2 | 0 % | +0.4 | 2 | 0 % | 0 % | -0.9 | -1.0 | 90 % | 100 % | 10 % | disagreement 1 |
| uniform 30 % | 10 | 10000 | 14241 | 30 % | +0.2 | 1 | 0 % | +0.4 | 1 | 0 % | 0 % | -0.0 | -0.4 | 90 % | 100 % | 0 % |  |
| uniform 30 % | 12 | 500 | 681 | 31 % | +1.4 | 4 | 30 % | +2.7 | 2 | 30 % | 30 % | -0.0 | +0.3 | 80 % | 90 % | 10 % | disagreement 1 |
| uniform 30 % | 12 | 2000 | 2839 | 30 % | +2.0 | 2 | 0 % | +1.5 | 2 | 0 % | 0 % | +0.0 | +0.7 | 80 % | 100 % | 10 % | disagreement 1 |
| uniform 30 % | 12 | 10000 | 14326 | 30 % | +0.9 | 1 | 0 % | +0.9 | 1 | 0 % | 0 % | -0.0 | -0.3 | 70 % | 80 % | 0 % |  |
| uniform 30 % | 12 | 50000 | 72184 | 31 % | +0.9 | 0 | 0 % | +1.2 | 0 | 0 % | 0 % | +0.1 | +0.1 | 20 % | 0 % | 0 % |  |
| uniform 30 % | 20 | 500 | 726 | 31 % | +1.4 | 5 | 20 % | +4.2 | 4 | 40 % | 20 % | -0.7 | -3.2 | 90 % | 80 % | 70 % | disagreement 2, wide_interval 6 |
| uniform 30 % | 20 | 2000 | 2863 | 30 % | +1.5 | 2 | 0 % | +2.0 | 2 | 0 % | 0 % | +0.1 | -0.6 | 100 % | 100 % | 10 % | disagreement 1 |
| uniform 30 % | 20 | 10000 | 14266 | 30 % | +2.4 | 1 | 0 % | +2.7 | 1 | 0 % | 0 % | +0.1 | +0.5 | 30 % | 30 % | 0 % |  |
| uniform 30 % | 30 | 500 | 696 | 30 % | -3.9 | 13 | 80 % | +25.6 | 14 | 100 % | 0 % | -5.6 | -11.4 | 10 % | 20 % | 100 % | disagreement 8, wide_interval 2, window_short 1 |
| uniform 30 % | 30 | 2000 | 2784 | 30 % | +6.6 | 11 | 90 % | -0.4 | 12 | 60 % | 0 % | -3.8 | -3.1 | 90 % | 80 % | 100 % | disagreement 2, wide_interval 9 |
| uniform 30 % | 30 | 10000 | 14102 | 31 % | +5.2 | 3 | 50 % | +5.3 | 3 | 50 % | 40 % | -3.8 | +1.5 | 60 % | 20 % | 20 % | disagreement 2 |
| uniform 50 % | 6 | 500 | 996 | 50 % | +3.4 | 1 | 10 % | -0.1 | 2 | 0 % | 0 % | +0.5 | -0.4 | 100 % | 100 % | 90 % | nearby_dominates 9 |
| uniform 50 % | 6 | 2000 | 3939 | 50 % | +1.6 | 2 | 10 % | -0.1 | 2 | 0 % | 0 % | -0.6 | -1.2 | 100 % | 100 % | 100 % | nearby_dominates 10 |
| uniform 50 % | 6 | 10000 | 20002 | 50 % | +3.4 | 1 | 0 % | +0.9 | 0 | 0 % | 0 % | +0.1 | +0.1 | 80 % | 100 % | 100 % | nearby_dominates 10 |
| uniform 50 % | 10 | 500 | 1016 | 50 % | +8.0 | 3 | 90 % | +4.2 | 3 | 30 % | 10 % | +0.8 | +1.9 | 80 % | 90 % | 60 % | nearby_dominates 6 |
| uniform 50 % | 10 | 2000 | 4007 | 50 % | +5.8 | 2 | 70 % | +2.1 | 2 | 0 % | 0 % | +0.1 | -0.1 | 60 % | 90 % | 80 % | nearby_dominates 8 |
| uniform 50 % | 10 | 10000 | 20136 | 50 % | +5.9 | 1 | 80 % | +2.1 | 1 | 0 % | 0 % | +0.1 | +0.1 | 10 % | 10 % | 100 % | nearby_dominates 10 |
| uniform 50 % | 12 | 500 | 985 | 49 % | +7.6 | 3 | 70 % | +3.0 | 3 | 40 % | 10 % | +0.2 | +0.2 | 70 % | 100 % | 50 % | disagreement 2, nearby_dominates 5 |
| uniform 50 % | 12 | 2000 | 3986 | 51 % | +6.8 | 1 | 100 % | +3.1 | 1 | 0 % | 0 % | -0.5 | +0.4 | 60 % | 60 % | 50 % | disagreement 1, nearby_dominates 5 |
| uniform 50 % | 12 | 10000 | 19940 | 50 % | +6.6 | 1 | 100 % | +2.7 | 1 | 0 % | 0 % | +0.4 | -0.3 | 10 % | 0 % | 70 % | nearby_dominates 7 |
| uniform 50 % | 12 | 50000 | 100799 | 51 % | +7.3 | 0 | 100 % | +2.6 | 0 | 0 % | 0 % | -0.3 | -0.1 | 0 % | 0 % | 100 % | nearby_dominates 10 |
| uniform 50 % | 20 | 500 | 1009 | 49 % | +12.5 | 4 | 90 % | +9.7 | 4 | 80 % | 0 % | -0.5 | +7.8 | 90 % | 80 % | 90 % | wide_interval 9 |
| uniform 50 % | 20 | 2000 | 4078 | 49 % | +7.2 | 121 | 100 % | +3.2 | 2 | 40 % | 20 % | -0.6 | -4.1 | 60 % | 60 % | 70 % | disagreement 7 |
| uniform 50 % | 20 | 10000 | 20071 | 50 % | +8.7 | 1 | 100 % | +6.0 | 1 | 70 % | 20 % | -0.1 | -1.7 | 0 % | 0 % | 80 % | disagreement 8 |
| uniform 50 % | 30 | 500 | 973 | 51 % | +9.6 | 14 | 60 % | +53.8 | 28 | 100 % | 0 % | -4.0 | -16.0 | 20 % | 20 % | 100 % | disagreement 9, nearby_dominates 1, wide_interval 2, window_short 8 |
| uniform 50 % | 30 | 2000 | 3952 | 51 % | +16.5 | 6 | 100 % | +12.4 | 6 | 100 % | 0 % | -3.8 | +4.3 | 90 % | 60 % | 100 % | disagreement 1, wide_interval 10 |
| uniform 50 % | 30 | 10000 | 19912 | 51 % | +13.8 | 5 | 100 % | +10.7 | 4 | 90 % | 40 % | -3.7 | +3.4 | 10 % | 10 % | 60 % | disagreement 6 |
| clustered, heavy | 6 | 500 | 628 | 22 % | +0.6 | 5 | 40 % | +1.1 | 4 | 20 % | 20 % | -0.6 | +2.1 | 100 % | 100 % | 0 % |  |
| clustered, heavy | 6 | 2000 | 2540 | 21 % | -2.8 | 2 | 0 % | -1.9 | 1 | 0 % | 0 % | -2.9 | -2.5 | 80 % | 100 % | 0 % |  |
| clustered, heavy | 6 | 10000 | 12600 | 21 % | -2.1 | 1 | 0 % | -1.8 | 1 | 0 % | 0 % | -2.3 | -2.8 | 30 % | 90 % | 0 % |  |
| clustered, heavy | 10 | 500 | 619 | 23 % | -1.6 | 12 | 40 % | -0.1 | 4 | 20 % | 0 % | -3.8 | -3.7 | 90 % | 90 % | 70 % | crowded 7, disagreement 2 |
| clustered, heavy | 10 | 2000 | 2631 | 24 % | -4.2 | 3 | 30 % | -2.0 | 2 | 10 % | 0 % | -3.8 | -5.1 | 70 % | 90 % | 80 % | crowded 8, disagreement 1 |
| clustered, heavy | 10 | 10000 | 12640 | 24 % | -4.4 | 1 | 30 % | -2.4 | 1 | 0 % | 0 % | -4.0 | -5.5 | 0 % | 40 % | 100 % | crowded 10 |
| clustered, heavy | 12 | 500 | 622 | 25 % | -4.9 | 5 | 60 % | -5.6 | 4 | 70 % | 0 % | -2.9 | -6.4 | 70 % | 100 % | 100 % | crowded 10, disagreement 2, wide_interval 3 |
| clustered, heavy | 12 | 2000 | 2556 | 25 % | -7.3 | 3 | 90 % | -4.7 | 2 | 50 % | 0 % | -4.2 | -8.3 | 40 % | 90 % | 100 % | crowded 10, disagreement 5 |
| clustered, heavy | 12 | 10000 | 12718 | 25 % | -6.4 | 1 | 80 % | -3.5 | 1 | 10 % | 0 % | -4.2 | -6.7 | 0 % | 90 % | 100 % | crowded 10, disagreement 3 |
| clustered, heavy | 12 | 50000 | 63848 | 26 % | -5.9 | 0 | 100 % | -2.1 | 0 | 0 % | 0 % | -4.2 | -5.9 | 0 % | 0 % | 100 % | crowded 10, disagreement 1 |
| clustered, heavy | 20 | 500 | 626 | 27 % | -12.5 | 9 | 80 % | +0.9 | 2 | 20 % | 0 % | -5.1 | -11.5 | 70 % | 0 % | 100 % | crowded 10, disagreement 9 |
| clustered, heavy | 20 | 2000 | 2566 | 29 % | -17.9 | 5 | 100 % | -11.0 | 5 | 90 % | 0 % | -5.8 | -11.1 | 10 % | 60 % | 100 % | crowded 10, wide_interval 2 |
| clustered, heavy | 20 | 10000 | 12578 | 29 % | -13.6 | 2 | 100 % | -10.3 | 2 | 100 % | 0 % | -5.3 | -11.9 | 0 % | 0 % | 100 % | crowded 10 |
| clustered, heavy | 30 | 500 | 590 | 30 % | -15.0 | 9 | 70 % | -5.8 | 3 | 70 % | 0 % | -8.4 | -10.9 | 30 % | 100 % | 100 % | crowded 10, disagreement 5 |
| clustered, heavy | 30 | 2000 | 2463 | 31 % | -18.7 | 6 | 100 % | -5.7 | 1 | 80 % | 0 % | -8.7 | -15.5 | 0 % | 20 % | 100 % | crowded 10, disagreement 9 |
| clustered, heavy | 30 | 10000 | 12540 | 31 % | -21.2 | 4 | 100 % | -15.6 | 2 | 100 % | 0 % | -8.7 | -15.6 | 0 % | 0 % | 100 % | crowded 10, wide_interval 3 |
| clustered, moderate | 6 | 500 | 530 | 7 % | +2.2 | 8 | 25 % | +1.5 | 3 | 0 % | 0 % | +0.2 | +1.0 | 100 % | 100 % | 20 % | too_few_pairs 2 |
| clustered, moderate | 6 | 2000 | 2228 | 7 % | +0.7 | 2 | 0 % | +0.2 | 1 | 0 % | 0 % | -0.6 | -0.2 | 100 % | 100 % | 10 % | disagreement 1 |
| clustered, moderate | 6 | 10000 | 10682 | 7 % | -0.6 | 1 | 0 % | -0.4 | 1 | 0 % | 0 % | -0.2 | -0.7 | 90 % | 100 % | 0 % |  |
| clustered, moderate | 10 | 500 | 500 | 7 % | -0.8 | 5 | 60 % | +0.9 | 4 | 20 % | 0 % | -1.4 | +0.9 | 100 % | 80 % | 70 % | disagreement 1, too_few_pairs 5, wide_interval 1 |
| clustered, moderate | 10 | 2000 | 2128 | 8 % | -3.4 | 9 | 60 % | -1.0 | 2 | 0 % | 0 % | -1.1 | -2.4 | 80 % | 100 % | 10 % | disagreement 1 |
| clustered, moderate | 10 | 10000 | 10735 | 8 % | -0.8 | 3 | 10 % | +0.1 | 1 | 0 % | 0 % | -1.0 | -0.3 | 90 % | 90 % | 0 % |  |
| clustered, moderate | 12 | 500 | 531 | 8 % | -1.4 | 4 | 25 % | +2.5 | 4 | 25 % | 20 % | -0.4 | +0.4 | 75 % | 75 % | 20 % | too_few_pairs 2 |
| clustered, moderate | 12 | 2000 | 2128 | 8 % | -1.2 | 4 | 30 % | -0.4 | 2 | 0 % | 0 % | -0.1 | -0.4 | 90 % | 100 % | 0 % |  |
| clustered, moderate | 12 | 10000 | 10828 | 8 % | -0.6 | 2 | 10 % | -0.7 | 1 | 0 % | 0 % | -1.0 | -1.0 | 70 % | 90 % | 0 % |  |
| clustered, moderate | 12 | 50000 | 53478 | 8 % | -1.1 | 0 | 0 % | -0.0 | 0 | 0 % | 0 % | -1.1 | -1.3 | 80 % | 10 % | 0 % |  |
| clustered, moderate | 20 | 500 | 521 | 9 % | -2.4 | 5 | 25 % | +1.3 | 3 | 25 % | 20 % | -1.3 | -2.1 | 75 % | 100 % | 40 % | disagreement 2, too_few_pairs 2 |
| clustered, moderate | 20 | 2000 | 2148 | 9 % | -2.4 | 8 | 40 % | +0.5 | 1 | 0 % | 0 % | -1.2 | -2.7 | 70 % | 80 % | 30 % | disagreement 3 |
| clustered, moderate | 20 | 10000 | 10668 | 9 % | -2.1 | 2 | 20 % | +0.5 | 2 | 0 % | 0 % | -1.2 | -1.9 | 40 % | 10 % | 0 % |  |
| clustered, moderate | 30 | 500 | 515 | 10 % | -2.8 | 12 | 33 % | -2.6 | 1 | 0 % | 0 % | -4.4 | -3.5 | 100 % | 100 % | 60 % | disagreement 2, too_few_pairs 4 |
| clustered, moderate | 30 | 2000 | 2101 | 10 % | -1.7 | 6 | 30 % | -0.7 | 1 | 0 % | 0 % | -4.3 | -2.5 | 80 % | 0 % | 20 % | disagreement 2 |
| clustered, moderate | 30 | 10000 | 10510 | 10 % | -2.6 | 6 | 20 % | -1.1 | 1 | 0 % | 0 % | -4.9 | -1.9 | 50 % | 0 % | 0 % |  |
| linked, clustered heavy | 6 | 2000 | 241 | 89 % | — | — | — | — | — | — | 0 % | -23.7 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, clustered heavy | 6 | 10000 | 1229 | 90 % | +239.8 | 19 | 100 % | +219.2 | 15 | 100 % | 0 % | -22.8 | +241.6 | 0 % | 0 % | 100 % | disagreement 4, nearby_dominates 1, wide_interval 9 |
| linked, clustered heavy | 10 | 2000 | 244 | 89 % | — | — | — | — | — | — | 0 % | -23.6 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, clustered heavy | 10 | 10000 | 1234 | 90 % | +56.0 | 41 | 100 % | +66.2 | 28 | 100 % | 10 % | -25.3 | +12.1 | 0 % | 0 % | 90 % | disagreement 7, nearby_dominates 5, wide_interval 8 |
| linked, clustered heavy | 12 | 2000 | 242 | 90 % | — | — | — | — | — | — | 0 % | -25.1 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, clustered heavy | 12 | 10000 | 1244 | 91 % | +13.4 | 19 | 90 % | +55.2 | 15 | 100 % | 10 % | -20.3 | +5.7 | 0 % | 0 % | 90 % | disagreement 8, wide_interval 8 |
| linked, clustered heavy | 20 | 2000 | 246 | 90 % | — | — | — | — | — | — | 0 % | -24.2 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, clustered heavy | 20 | 10000 | 1239 | 90 % | -1.1 | 12 | 70 % | +19.3 | 7 | 90 % | 20 % | -26.0 | +0.7 | 30 % | 0 % | 80 % | disagreement 6, wide_interval 4 |
| linked, clustered heavy | 30 | 2000 | 240 | 90 % | — | — | — | — | — | — | 0 % | -26.5 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, clustered heavy | 30 | 10000 | 1197 | 90 % | -14.6 | 7 | 100 % | -3.0 | 6 | 20 % | 0 % | -22.0 | -14.8 | 60 % | 10 % | 90 % | disagreement 8, wide_interval 1 |
| linked, none | 6 | 2000 | 0 | 0 % | — | — | — | — | — | — | 0 % | -59.4 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 6 | 10000 | 2 | 19 % | — | — | — | — | — | — | 0 % | -19.2 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 10 | 2000 | 0 | 0 % | — | — | — | — | — | — | 0 % | +11.2 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 10 | 10000 | 2 | 21 % | — | — | — | — | — | — | 0 % | -34.6 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 12 | 2000 | 0 | — % | — | — | — | — | — | — | 0 % | — | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 12 | 10000 | 2 | 19 % | — | — | — | — | — | — | 0 % | -29.3 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 20 | 2000 | 0 | 0 % | — | — | — | — | — | — | 0 % | -40.3 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 20 | 10000 | 0 | 7 % | — | — | — | — | — | — | 0 % | -17.6 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 30 | 2000 | 0 | 0 % | — | — | — | — | — | — | 0 % | -31.7 | — | — % | — % | 100 % | too_few_pairs 10 |
| linked, none | 30 | 10000 | 1 | 12 % | — | — | — | — | — | — | 0 % | -22.8 | — | — % | — % | 100 % | too_few_pairs 10 |
| none | 6 | 500 | 467 | 0 % | -1.4 | 2 | 0 % | -0.3 | 2 | 0 % | 0 % | +1.3 | -0.3 | 100 % | 100 % | 60 % | too_few_pairs 6 |
| none | 6 | 2000 | 2042 | 0 % | -0.7 | 2 | 10 % | -1.3 | 2 | 0 % | 0 % | -1.3 | -1.1 | 90 % | 100 % | 0 % |  |
| none | 6 | 10000 | 9944 | 0 % | -0.0 | 2 | 10 % | +0.1 | 1 | 0 % | 0 % | +0.1 | +0.1 | 80 % | 100 % | 0 % |  |
| none | 10 | 500 | 503 | 0 % | -0.1 | 2 | 0 % | +3.0 | 2 | 0 % | 0 % | +2.7 | +2.7 | 60 % | 100 % | 50 % | too_few_pairs 5 |
| none | 10 | 2000 | 1999 | 0 % | -0.7 | 5 | 20 % | +0.9 | 1 | 0 % | 0 % | +0.9 | +0.9 | 100 % | 100 % | 0 % |  |
| none | 10 | 10000 | 9998 | 0 % | -0.5 | 2 | 0 % | +0.1 | 1 | 0 % | 0 % | +0.0 | -0.0 | 100 % | 100 % | 0 % |  |
| none | 12 | 500 | 499 | 0 % | -0.9 | 9 | 20 % | -2.7 | 2 | 0 % | 0 % | -0.2 | -2.7 | 100 % | 100 % | 50 % | too_few_pairs 5 |
| none | 12 | 2000 | 1930 | 0 % | -2.7 | 8 | 30 % | -0.2 | 1 | 0 % | 0 % | -0.2 | -0.3 | 80 % | 100 % | 20 % | disagreement 2 |
| none | 12 | 10000 | 10053 | 0 % | -1.0 | 1 | 0 % | +0.1 | 1 | 0 % | 0 % | +0.1 | +0.1 | 90 % | 100 % | 0 % |  |
| none | 12 | 50000 | 49904 | 0 % | -0.3 | 2 | 10 % | -0.2 | 0 | 0 % | 0 % | -0.2 | -0.2 | 90 % | 100 % | 0 % |  |
| none | 20 | 500 | 502 | 0 % | -3.0 | 9 | 60 % | +0.9 | 1 | 0 % | 0 % | +1.2 | +0.9 | 100 % | 100 % | 60 % | disagreement 1, too_few_pairs 5 |
| none | 20 | 2000 | 2002 | 0 % | -0.4 | 6 | 20 % | -0.7 | 1 | 0 % | 0 % | -0.8 | -0.7 | 90 % | 100 % | 0 % |  |
| none | 20 | 10000 | 9945 | 0 % | -0.9 | 2 | 10 % | +0.0 | 1 | 0 % | 0 % | -0.1 | -0.2 | 90 % | 100 % | 0 % |  |
| none | 30 | 500 | 510 | 0 % | -4.1 | 12 | 29 % | -0.2 | 2 | 0 % | 0 % | -4.0 | -0.2 | 100 % | 100 % | 30 % | too_few_pairs 3 |
| none | 30 | 2000 | 1918 | 0 % | -0.9 | 3 | 20 % | -1.0 | 1 | 0 % | 0 % | -4.5 | -1.7 | 90 % | 0 % | 0 % |  |
| none | 30 | 10000 | 9802 | 0 % | -0.6 | 5 | 20 % | +0.2 | 0 | 0 % | 0 % | -3.6 | +0.3 | 90 % | 0 % | 0 % |  |

</details>

How to read it (own simulation; numbers from `nena_grid.json`):

- **No other molecules nearby.** The previous fit was more than 5 % off in 15 % of the files (and up to 30 % of a
  cell); the new estimate in none, and it equals the oracle. This is the failure that started this change.
- **Uniform background at 10-30 % of the pairs.** Overall the two fit about equally often within 5 % (old 14-19 %
  off, new 15-19 %). At σ ≤ 12 nm the new estimate fails only at ~500 pairs; nearly all other failures of both are at
  σ = 20-30 nm, where the 120 nm window holds only 3-4 s and the background ramp overlaps the whole peak. There the
  new estimate reads high and most failing files are flagged, but not all (unflagged up to 50 % of a σ = 30 nm cell).
  A longer window is not a general cure (section 6).
- **Uniform background at half the pairs.** Old: 74 % of the files off by more than 5 % (median +7 %); new: 34 %
  (median +3 %), flagged as `nearby_dominates` or `disagreement`; the random-neighbour estimate is within ±2 % of σ in
  every cell at σ ≤ 12 nm.
- **Clustered molecules.** Moderate crowding: old 23 % off, new 3 %. Heavy crowding: old 64 %, new 40 % — and the
  oracle itself 36 %: the nearest-neighbour rule, not the fit, loses the same-molecule pairs. All but 1 % of
  the files there are either within 5 % or carry a warning (`crowded`, calibrated on these files: below a crowding of 0.15 at most 2 % of the
  files read more than 5 % low; from 0.15 to 0.20, 20 %; above 0.20, 73 %).
- **Linked files.** Linking left 0-2 pairs in the uncrowded files ("not measured"). In crowded linked files about
  1200 pairs remain, 90 % of them between different molecules; the value is meaningless and at least 90 % of those files are flagged.
  NeNA must not be run on linked files (section 6).
- **Interval.** Without contamination it covered σ in 90 % of the files (nominal 95 %): pairs that share a
  localization are not independent (section 6). With contamination coverage drops further, because the interval
  measures the noise of the fit, not the error of the model; that is what the agreement check is for.
- **False alarms.** In the uncontaminated files 3 of 136 measured files carried `disagreement` (the random-neighbour
  model over-fitted) and none was off by more than 5 %.
- The oracle is a second-moment estimate and reads ~4 % low at σ = 30 nm because the 120 nm window cuts the tail it
  is most sensitive to; the fitted estimators account for the cut.

## 6. Limitations

- **Linked files.** Picasso Link merges a molecule's consecutive-frame localizations into one, which removes exactly
  the pairs NeNA measures. NeNA must be run on the whole-field, unlinked file; on linked files it returns "not
  measured" (too few pairs) or `no_peak`.
- **Crowding reads low.** When other molecules are often on within a few s of a molecule in the next frame, the
  nearest-neighbour rule picks them instead of the molecule's own next localization, and the same-molecule distances
  it keeps are truncated: the oracle itself reads low (section 5). No fit of these distances can undo that; the
  `crowded` warning says so, and a sparser region or file measures better.
- **Model shape.** Picasso's nearby term is a Gaussian; randomly placed neighbours follow 2πρd·exp(−πρd²). With
  uniform background at half the pairs the Picasso-model value reads a few percent high and the random-neighbour model
  is closer (section 5); the disagreement check flags the larger cases.
- **Large precision with background.** At σ = 30 nm the 120 nm window holds only 4 s, and the background overlaps
  the whole peak; with uniform background both fits often miss by more than 5 % (section 5). Own simulation
  (`docs/sims/nena/nena_window.py`, `nena_window.json`; the grid's files at 2000 pairs, 10 per cell): doubling the
  window to 240 nm brought the median error at σ = 30 nm from +8.8 % to +1.9 % with background at 10 % of the pairs,
  but made it worse with background at 30 % (from −0.4 % to +16 %, all flagged), because the background pairs grow
  with the window's area. No single window suits every file; the warnings are the guard, and a sparser file is the
  cure.
- **Dependent pairs.** An event of L frames gives L − 1 pairs and neighbouring pairs share a localization (the
  correlation of their squared distances is 1/4; derivation). The interval treats pairs as independent, so it is
  somewhat too narrow on large files (coverage in section 5).
- **Lateral only.** NeNA here is the lateral (x, y) precision. The axial analogue per z stratum is `axial_nena` in
  `tools/mps_axial_precision.py`, a different estimator.
- **Synthetic validation.** The grid uses isotropic Gaussian errors and a known σ shared by every localization. Real
  files mix precisions (photon counts vary), and NeNA then reports an average precision (Endesfelder et al. 2014).

## 7. Decision record

- **2026-10-14**: approved by the user: the NeNA fit is made robust, keeping Picasso's model and reporting an
  interval and agreement checks; no value is silently replaced. The previous value stays visible (`lsq_nm`).

## References

- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochemistry and Cell Biology*
  141(6):629-638. doi:10.1007/s00418-014-1192-3
- Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R (2017). Super-resolution microscopy with
  DNA-PAINT. *Nature Protocols* 12(6):1198-1228. doi:10.1038/nprot.2017.024 (Picasso; the NeNA code read is
  `picasso/postprocess.py` on github.com/jungmannlab/picasso, 2026-10-14)
- Schwarz G (1978). Estimating the dimension of a model. *The Annals of Statistics* 6(2):461-464.
  doi:10.1214/aos/1176344136
- Wilks SS (1938). The large-sample distribution of the likelihood ratio for testing composite hypotheses. *The
  Annals of Mathematical Statistics* 9(1):60-62. doi:10.1214/aoms/1177732360
