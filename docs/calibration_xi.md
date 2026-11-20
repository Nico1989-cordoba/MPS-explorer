# Calibrating ξ: events per isolated label, from a sparse control

> **Resumen (español).** `tools/calibration_xi.py` analiza la sesión de calibración de ξ: marcas fluorescentes ralas
> (secundarios adsorbidos en vidrio, o marcas aisladas en un corte control) y, opcionalmente, una muestra sin
> anticuerpo para el fondo. Enlaza cuadros con la misma función y los mismos parámetros que los datos de tejido,
> agrupa los eventos de cada marca, se queda con las marcas aisladas, descuenta el fondo, ajusta la distribución de
> eventos por marca (gamma y geométrica por máxima verosimilitud, con censura opcional de las marcas que siguen
> activas al final) y da un intervalo por remuestreo por sesión y campo. Escribe un CSV por marca, un JSON con los
> argumentos listos para `calibrated_prior(...)` y una figura. Cada número lleva su rótulo de origen.

This page describes the analysis of the ξ calibration session. The code is `tools/calibration_xi.py`; the tests are
`test_calibration_xi.py` (synthetic data only). It implements the analysis section of the lab's calibration protocol
(sections F6, F7, F8, F11 and F12 of that protocol; the protocol itself lives with the project notes, not in this
repository). Origin labels, as everywhere in this project:

- **[lit: …]** a published value or statement, with its reference;
- **[deriv]** an argument or calculation shown here;
- **[ad hoc]** a choice of this project with no measurement behind it, and what would replace it;
- **[simulation]** a property checked on the project's own synthetic data.

Method name: **Emitters (BaGoL-like reimplementation; Fazel et al. 2022)**.

## 1. What it measures

The Emitters layer needs ξ, the distribution of the number of localizations each emitter leaves
[lit: Fazel et al. 2022, Results]. Fazel et al. foresee obtaining it from "a dedicated control experiment where the
sample is known to be at low labeling fraction" [lit: Fazel et al. 2022, Results]. This tool is the analysis of that
control. An emitter here is **one isolated fluorescent label** — on glass, one secondary IgG with its dyes — not a
protein [lit: Fazel et al. 2022, Introduction: the model assumes one emitter per label; deriv: the label of the
control is what is counted].

The default literature values it replaces are the switching cycles of Dempsey et al. (2011, Table 1, p. 1030), which
were measured on secondary antibodies adsorbed to glass, in TIRF, at lower irradiance, and are called a lower bound by
the authors because some molecules had not bleached at the end of the movie [lit: Dempsey et al. 2011, p. 1030 and
Online Methods]. See `docs/emitters_xi.md`.

What it counts is **events**: localizations linked across consecutive frames, with the same linking radius and the
same dark-frame tolerance as the tissue data. That is what `calibrated_prior` expects: "localizations per emitter
after the same linking and filtering as the data" [lit: docstring of `tools.emitters_prior.calibrated_prior`].

## 2. Input

- One or more localization files per **session** (one file = one field of view): Picasso HDF5 or ThunderSTORM CSV,
  read by `tools.mps_io.load_localizations` (x, y in nm after the pixel-size conversion; frame; lpx/lpy; photons).
  Localize, filter and undrift them exactly as the tissue data.
- Optionally, one or more **controls** without antibody (glass + buffer, or an unstained section): the background.
- The tissue **linking radius** and **max dark frames** (required: the tool does not choose them, because the
  precision on glass is better than in tissue and a radius derived from the glass file would differ) [deriv].
  Decisions of the user (2026-10-17): both linking parameters are required arguments (`--link-radius`,
  `--max-dark`), the background correction is the mixture with a known weight (step 4) and the grouping radius
  is derived from the precision (step 2) by default.
- Optionally: pump frames (frames skipped at the start, protocol F7), beam centre and σ for the irradiance cut,
  movie length if the metadata lacks it, and free session metadata (`--meta key=value`: dye lot, laser power at start
  and end, buffer pH at start and end, buffer age, date…). Metadata are stored in the JSON; `ph_start`, `ph_end`,
  `power_start_mw` and `power_end_mw` are also checked (§5).

## 3. Steps

1. **Link** with `tools.mps_paint.build_events` (Picasso's link rule) at the given radius and dark frames. Events
   bound in the first or last frame are kept: their length would be censored, but they are still one event of their
   label [deriv].
2. **Group events into marks** by single linkage (the same partition as DBSCAN with a minimum of 1). Default radius
   `5.26 × sqrt(lp_median² + 10²)` nm: 5.26 σ is the 99.9 % quantile of the distance between two positions of one
   label with per-axis sd σ [deriv]; the 10 nm for antibody size and residual drift is [ad hoc]. With lp = 10 nm this
   gives 74 nm. The protocol's fixed 150 nm [ad hoc, protocol F11.3] can be given with `--group-radius 150`, and when the
   radius is derived the JSON also reports the mean at 150 nm as a sensitivity check.
3. **Isolation**: a mark counts only if no other mark (background groups included) is within 0.5 µm, it lies at
   least 1 µm from the field edge, and its extent is at most twice the grouping radius (an elongated group is two
   labels merged) [ad hoc, protocol F6 and F11.4]. At 0.05-0.15 labels/µm² placed at random, 4-11 % of labels have a
   neighbour within 0.5 µm: 1 − exp(−πρr²) [deriv]. With `--use-z` the neighbour distance is 3D (tissue).
   With a beam profile, the main estimate keeps marks at ≥ 70 % of the peak irradiance and the others are reported by
   band (≥ 85 %, 70-85 %, < 70 %) [ad hoc, after the flat-field criterion of Lin et al. 2015].
4. **Background.** The controls go through steps 1-3 unchanged. Their density of groups per µm² (inner area) times the
   analysed area times the share of groups that pass the cuts gives the expected number of false marks among the
   counted ones, and the controls give their distribution of events per group [deriv]. This enters the fit as a
   mixture with a known weight (default). The protocol's alternative — fit only counts ≥ n_min, the smallest count
   above which the expected background is < 5 % of the marks at every count [ad hoc, protocol F11.5] — is available
   (`--background n_min`) and always reported; above n_min only the tail is seen, so the geometric mean is
   identified but the gamma's shape is not, and the tool says so [simulation].
5. **Fit** by maximum likelihood (Nelder-Mead on log parameters):
   - **gamma**: N = round(X), X ~ Gamma(k, θ), conditioned on N ≥ 1. The Emitters layer evaluates the continuous
     gamma density at integer N (`gampdf` in `BaGoL_RJMCMC.m`), which is ≈ F(n + ½) − F(n − ½); this discretization
     makes the fitted (k, θ) the ones that layer needs [deriv]. The reported mean is **k·θ**, the mean of that
     continuous gamma; the empirical mean of the counts is about ½ higher for k = 1 [deriv].
   - **geometric**: the same with k = 1 (a discretized exponential is geometric). Fazel et al. used an exponential
     (gamma with shape 1) prior for simulated dSTORM [lit: Fazel et al. 2022, Methods], and at long times compared with
     bleaching the count approaches a geometric [lit: Nieuwenhuizen et al. 2015, Results].
   - **negative binomial**, zero-truncated, as the discrete check the protocol asks for (F11.6).
   - The selected model is the one with the lower AIC between gamma and geometric [ad hoc, protocol F11.6].
6. **End of the movie.** Two estimands:
   - **recorded** (default): the count in a movie of this length, the end ignored. It is the right one when the
     tissue movies have the same length (protocol F7: same frames, same truncation) [deriv].
   - **censored**: the count until bleaching, for movies of another length or a comparison with Dempsey's lower
     bound. A label whose last event is g frames before the end either bleached, or is still active and stayed dark
     for g frames; so each mark contributes P(N = n) + s(g)·P(N > n), with s(g) the empirical share of the dark gaps
     inside isolated marks that are longer than g [deriv: right censoring with an unknown censoring status]. A hard
     rule (P(N > n) for marks within a fixed window of the end) was tried first and overestimated the mean by 21 % on
     a too-short movie, because labels that had just bleached were counted as still active [simulation].
   Both estimands are always in the JSON. For the quality check, a mark is *active at the end* when its last event is
   closer to the end than the 95th percentile of those dark gaps [deriv].
7. **Pump frames.** ξ is reported with all frames and without the first N frames (default versions 0, 200, 1000;
   [ad hoc, protocol F7]), with the fraction of labels seen at the start that leave no event afterwards (protocol M6).
8. **Uncertainty.** Bootstrap (default 1000 replicates, fixed seed): sessions with replacement, then fields within
   each session, then marks within each field [protocol F11.6]. With two or more sessions, the mean of the session
   means, their SD and a t interval are reported too: the spread between sessions is the main uncertainty
   [protocol F8; lit: Thevathasan et al. 2019, Table 1, report 3.9 ± 0.9 and 5.4 ± 0.5 localizations per AF647 over 4
   and 8 independent experiments]. In each replicate the control is redrawn too: its number of groups as Poisson
   and their events per group as multinomial [deriv: a uniform background is a Poisson process].

## 4. Output

`python -m tools.calibration_xi --session A1 f1.hdf5 f2.hdf5 --session A2 f3.hdf5 --control v0.hdf5
--link-radius <tissue nm> --max-dark <tissue frames> --dye "Alexa Fluor 647" --buffer "GLOX + 10 mM MEA"
--meta lot=... --meta power_start_mw=... --out <prefix>` writes, never overwriting:

- `<prefix>_marks.csv`: one row per mark (session, field, position, events, localizations, first/last frame,
  extent, nearest neighbour, distance to the edge, relative irradiance, frames between the last event and the end,
  s(g), active at the end, isolated, used, reason for
  exclusion);
- `<prefix>_xi.json`: parameters with their origin, counts, background, the three fits with AIC, the selected model
  with mean, k, θ, P(N ≤ 1), P(N ≤ 3) and the bootstrap interval, per-session fits, the alternative correction and
  estimand, pump-frame versions, irradiance bands, the 150 nm sensitivity check, the survival of the marks over the
  movie, the quality checks, warnings, and **`calibrated_prior_args`**;
- `<prefix>_xi.png`: histogram of events per isolated mark with the fitted pmfs and the expected background, and the
  survival of the marks.

## 5. Quality checks

All thresholds are [ad hoc], from table F12 of the protocol; each check is in the JSON with its rule and an `ok` flag.

| Check | Rule |
|---|---|
| density of marks | 0.02-0.20 per µm² |
| aggregates | share of neighbours closer than 0.5 µm ≤ 2 × the Poisson expectation 1 − exp(−πρr²) |
| background | expected false marks ≤ 10 % of the counted marks |
| drift | median per-mark slope of position against frame × movie length ≤ 100 nm |
| marks per session | ≥ 100 isolated marks |
| active at the end | ≤ 20 % (above it the recorded count is mostly a lower bound, as in Dempsey et al. 2011, p. 1030) |
| buffer pH (if given) | start 7.8-8.2; drop ≤ 0.3; never below 7.5 |
| laser power (if given) | start and end within 10 % |

## 6. Loading the result into the program

The JSON's `calibrated_prior_args` are the arguments of
`tools.emitters_prior.calibrated_prior(mean, k=..., dye=..., buffer=..., note=...)`; the note already states what an
emitter is in this calibration, the sessions, n, the linking parameters, the grouping radius, the pump frames, the
estimand, the background correction and the bootstrap interval, plus lot and date when given. The prior is built
once inside the tool to validate the arguments (`calibrated_prior` rejects a non-positive mean). The source label
becomes `calibrated`.

Which ξ goes to the tissue analysis — glass or tissue control, all frames or without the pump frames — is decided
with the user (protocol F11, "which ξ goes to tissue") [ad hoc].

## 7. Checks on synthetic data

The tests generate labels at random positions with a known number of switching cycles (rounded gamma), exponential
off-times, a few frames per cycle, Gaussian localization error, uniform background groups and a finite movie, and
check [simulation]:

- the fit recovers mean and k from counts (4000 draws: mean within 6 %, k within 12 %);
- the pipeline's bootstrap interval contains the true mean and the true k in most seeds (≥ 9 of 12 in the test);
- the background mixture brings the estimate back when half of the counted marks are background;
- the censored estimand recovers the count until bleaching in a movie that is too short, while the recorded one
  reproduces the truncated count;
- density, aggregate, drift, power and active-at-the-end flags fire when they should; pump-frame versions lose
  labels as expected; output is deterministic for a fixed seed.

Coverage over 30 seeds, 3 sessions × 2 fields of 40 × 40 µm at 0.08 labels/µm² (about 640 isolated labels per
seed), 200 bootstrap replicates [simulation]:

| Case | True mean, k | Mean of estimates (SD) | 95 % interval contains the mean | contains k |
|---|---|---|---|---|
| geometric, 16 000 frames | 12, 1 | 12.03 (0.46) | 29/30 | 30/30 |
| gamma, 16 000 frames | 15, 3 | 15.02 (0.38) | 30/30 | 30/30 |
| geometric + as many background groups as labels, with control | 10, 1 | 10.11 (0.58) | 30/30 | 30/30 |
| geometric, 1 500 frames, censored estimand | 20, 1 | 19.16 (1.20) | 29/30 | 30/30 |
| geometric, 4 000 frames, censored estimand | 12, 1 | 12.05 (0.49) | 30/30 | 30/30 |
| geometric, link radius 3 lp | 12, 1 | 12.63 (0.48) | 27/30 | 30/30 |

The intervals are conservative: their width corresponds to 1.3-1.5 times the SD of the estimates across seeds,
because resampling at every level of a hierarchy adds the lower levels' variance again [simulation]. The last row
is the expected excess from split events (next paragraph), not a failure of the fit.

One property worth knowing [simulation]: at a linking radius of 3 lp, about 10 % of consecutive-frame pairs of one
switching cycle are farther apart than the radius (two positions with per-axis sd lp differ by more than 3 lp with
probability exp(−9/4) ≈ 0.105 [deriv]), so the events per label exceed the switching cycles by a few percent. That is
intended: the tissue data are linked the same way, so the prior must count the same split events.

## 8. Limitations

- Cross-talk between channels (protocol M4), short dark times inside a label (M7), brightness against events and the
  per-buffer-age version of ξ are not computed yet; the metadata to compute them is stored.
- The background is assumed spatially uniform and independent of the labels.
- s(g) comes from the dark gaps observed inside the movie, which under-represents gaps longer than the movie; with
  very long off-times the censored estimand is biased low (−4 % on the synthetic 1500-frame case of §7)
  [simulation].
- Isolation in tissue (T-dil) depends on 3D coordinates and on the section geometry; it has only been tested on 2D
  synthetic glass.
- The censored estimand assumes that when a label bleaches does not depend on how many cycles fit in the movie (the
  usual right-censoring assumption) [deriv].

## References

- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768
- Fazel M, Wester MJ, Schodt DJ, … Jungmann R, Lidke KA (2022). *Nature Communications* 13:7152.
  doi:10.1038/s41467-022-34894-2
- Lin Y, Long JJ, Huang F, … Bewersdorf J (2015). *PLOS ONE* 10(5):e0128135. doi:10.1371/journal.pone.0128135
- Nieuwenhuizen RPJ, Bates M, Szymborska A, Lidke KA, Rieger B, Stallinga S (2015). *PLOS ONE* 10(5):e0127989.
  doi:10.1371/journal.pone.0127989
- Thevathasan JV, Kahnwald M, Cieśliński K, … Ellenberg J, Ries J (2019). *Nature Methods* 16(10):1045-1053.
  doi:10.1038/s41592-019-0574-9
