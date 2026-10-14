# The randomization control, the KS distance and the 1NN regularity index

> **Resumen (español).** El control de aleatorización pregunta si el espaciado de los centros de cluster observados
> difiere de una colocación al azar en la misma banda del contorno, con el mismo número de clusters y la misma
> separación mínima (Gazal et al. 2026). MPS Explorer construye ese nulo con B = 999 iteraciones (banda ±50 nm
> alrededor del contorno cerrado y suavizado, grilla de 5 nm, separación mínima = el 1NN observado más chico, regla
> del paper, o su corrección opcional), re-sortea y excluye las iteraciones que no pueden ubicar todos los clusters y
> rechaza el control cuando el empaquetamiento lo hace imposible. La distancia KS **D** queda como descriptiva, con su
> propio nulo (media nula, z_D) y p-valores Monte Carlo de rango, rotulados aproximados. El **p del KS de scipy se
> eliminó**: sus supuestos no se cumplen y en simulaciones con nulo verdadero rechazaba 9-22 % a un 5 % nominal. El
> cruce de las CDF se informa sólo fuera de la banda que dan las propias iteraciones, y es significativo por una
> prueba Monte Carlo de su fuerza. El índice de regularidad **r = mediana 1NN · K / P** separa regularidad de
> densidad de clusters detectados; coincidir con los 260 nm publicados no es un criterio de validación. El lote
> compara z_D, nunca D crudo, y el nivel de azar del ICC sale de una fórmula cerrada. Decisiones del usuario del
> 2026-10-08 (P-R23 a P-R27, P-R29, P-R31, P-R52); implementación del 2026-10-14, con dos desvíos que esperan su
> aprobación (sección 9).

This page explains what the randomization control of MPS Explorer computes, which numbers it reports and why, and
which numbers it no longer reports. The code is in `tools/mps_randomization.py` (the null, D and its null, the
Monte Carlo p-values, the CDF band), `tools/mps_analysis.py` (the results table rows and the exported columns, the
regularity index), `tools/mps_spatial.py` (the pooled 1NN summaries), `tools/mps_batch.py` (the batch statistics) and
`tools/mps_randomization_ui.py` (the editors). Every number below is labelled with where it comes from:

- **literature**: a published value, with its reference;
- **derivation**: shown here;
- **own simulation**: synthetic data only (no measured axon), with its setup and the script that reproduces it.
  The scripts are in this repository: `docs/sims/randomization/` (the research simulations sim1-sim4, each writes
  its JSON beside itself; the JSON committed is the run quoted here) and the validators `validate_randomization.py`
  and `validate_nn1_controls.py` at the repository root.

## 1. What the control asks, and how the null is built

Gazal et al. (2026, parameter 8) ask whether the observed arrangement of βII-spectrin cluster centres along an
axon's perimeter differs from random placement: *is the spacing of the observed centres different from random
placement on the same contour band, with the same number of clusters and the same minimum separation?* They smooth
the contour by B-spline interpolation, take the region between offset curves ±50 nm from it, place the cluster
centres at random in that region under a minimum separation equal to the smallest observed nearest-neighbour (1NN)
distance, repeat 1000 times, and compare the 1NN distance distributions with a Kolmogorov-Smirnov (KS) test
(reported D = 0.054, p < 10⁻³, and a crossing of the two CDFs near 0.8).

MPS Explorer builds the null as follows. Each choice is editable in the Randomization group of the MPS analysis
window's parameter strip (More → Edit...), recorded in the settings file and written to every exported row.

| Choice | Default | Range | Origin |
|---|---|---|---|
| Iterations B | **999** | 199 / 399 / 999 / 1999 | derivation: a Monte Carlo p-value (1 + b)/(B + 1) is exact on its grid at level α when α(B + 1) is an integer (α = 0.05 and 0.01 at B = 999; Besag & Diggle 1977; Phipson & Smyth 2010); the paper used 1000 |
| Band half-width | **±50 nm** | 20-150 nm | literature (Gazal et al. 2026, "approximates the accessible membrane surface"); derivation below |
| Contour | **closed**, periodic cubic B-spline | — | derivation below (a bug of the open fit) |
| Smoothing | **RMS residual 1 nm** per vertex (s = K·rms²) | 0-50 nm | derivation below |
| Candidate grid | **5 nm** | 1-20 nm | derivation below; own simulation (section 4) |
| Minimum separation (hard core) | **smallest observed 1NN** (the paper's rule) | or the bias-corrected value | literature / derivation below |
| Placement | simple sequential inhibition, **20 attempts per cluster** | 5-100 | standard method (Diggle, Besag & Gleaves 1976; random sequential adsorption, Evans 1993) |
| Incomplete iterations | **re-drawn up to 10 times, then excluded** | or kept | derivation below |
| Seed | **0** | any integer | convention: a fixed seed makes the control repeatable; the Monte Carlo error is set by B, SE(p) = √(p(1 − p)/B) ≈ 0.007 at p = 0.05 (Marriott 1979) |

**The band.** The region is the set of grid points within the half-width of the smoothed curve. This equals the
region between the paper's two offset curves wherever they do not self-intersect, and stays well defined on concave
contours, where explicit offsets do. Why ±50 nm is defensible beyond the paper's word (derivation): the radial
scatter of a membrane-anchored cluster centre combines the centroid's precision (~20 nm / √10 ≈ 6 nm for the smallest
DBSCAN cluster), the antibody linkage (10-20 nm for a primary plus secondary IgG; Früh et al. 2021) and half the
radial extent of a cluster of effective radius ~25 nm (Gazal et al. 2026), ≈ 12 nm; together σ_r ≈ √(36 + 225 +
144) ≈ 20 nm, so ±50 nm is about ±2.5 σ_r and holds about 99 % of membrane-anchored centres. A narrower band makes
the null more one-dimensional (longer random 1NN); a wider one shortens random 1NN. Own simulation
(`validate_randomization.py`, section C; simulated axons with radial scatter 10 / 20 / 30 nm, B = 199): between half-
widths 25 and 100 nm, z_D (section 4) changed by at most 0.24 on rings with 10 nm scatter, below the 0.5 adequacy
limit set beforehand.

**The contour is closed before the fit.** scipy's `splprep(per=1)` (FITPACK; Dierckx 1993) requires a periodic curve
to close exactly and overwrites the last point with the first. Given the open contour, the last cluster was left out
of the fit and could sit tens of nanometres off the curve — outside the band the null draws from. The control now
appends the first vertex before fitting (`close_contour`); the test `test_closed_contour_keeps_the_last_vertex`
(`test_randomization_sci3.py`) builds a ring whose last vertex the open fit misses by more than 20 nm, while the
closed fit passes within 5 nm of every vertex.

**Smoothing.** FITPACK's s bounds the summed squared residuals under unit weights, Σ r_i² ≤ s. With s = K·rms², the
RMS residual per vertex is at most rms; at 1 nm (s = K, the previous value) the curve nearly interpolates the centres,
consistent with the paper's "B-spline interpolation" (which states no degree or smoothing factor). scipy's suggested
range m ± √(2m) applies to weights 1/σ_i, not to unit weights, so it is not a smoothing rule here. A genuine smoothing
scale (the centres' radial scatter, ~20 nm) can be set in the editor; the value is this project's choice, not a
published rule.

**The grid.** A 5 nm grid displaces a continuous position by at most half the cell diagonal, 3.5 nm (per-axis sd
5/√12 = 1.4 nm). Own simulation (`validate_randomization.py`, section D; K = 50 centres placed off-grid on a ring of
12.5 µm, B = 999, 4 seeds per grid): going from 2.5 to 5 nm moved the mean null median 1NN by 0.5 nm and the mean z_D
by 0.05 (within Monte Carlo error); going from 5 to 10 nm moved them by +1.5 nm and +0.21, so 10 nm is no longer
negligible.

**Placement and incomplete iterations.** Candidates are visited in random order and accepted when no placed centre
lies closer than the minimum separation; an iteration stops after 20 candidates per cluster. Sequential placement
jams: in a band much wider than the separation near a coverage of 0.547 (Wang 1994), in the narrow default band only
when the separation approaches the mean spacing (car-parking jamming; Evans 1993). Pooling iterations that placed
fewer clusters makes the null sparser than the data and biases it toward longer distances. So an incomplete iteration
is drawn again, up to 10 times, and if it is still incomplete it is counted and left out of the null; the number used
(B_eff) is reported. When **no** iteration can place every cluster, the control is **refused** with a warning that
gives the packing fraction φ = K π d² / (4 × band area), band area = 2 × half-width × contour length (a perfectly
regular ring, whose smallest 1NN is its spacing, gives φ ≈ 2 and is refused). The previous behaviour — pooling with a
warning — remains available in the editor.

**The minimum separation.** The paper's rule (the smallest observed 1NN) is the default, for comparability. It is a
parameter estimated from the same data (the Lilliefors problem; section 3), and it builds in an excess of short
observed distances (section 3). The optional **bias-corrected** hard core uses the first-order correction for a 1-D
hard core, E[d_min] = d₀ + (P − K d₀)/K², hence d̂₀ = (K² d_min − P)/(K(K − 1)) (derivation; P the smoothed contour
length); when d̂₀ ≤ 0 there is no evidence of a hard core and the null places the centres without one. A minimum of
0 nm (coincident centroids) is replaced by the smallest non-zero separation, with a warning; if every 1NN is 0 nm the
control is refused (a hard-core model has no coincident points; spatstat's `Hardcore` likewise presupposes a positive
minimum; Baddeley et al. 2015).

**Can the null produce the observed positions?** A Monte Carlo comparison is only meaningful when the observed
pattern could be one more draw of the null (Besag & Diggle 1977; Davison & Hinkley 1997). The control counts the
observed centres farther than the half-width from the smoothed contour, where the null never places one ("centres
outside the band" in the results table, `randomization_n_outside_band` in the export). For the single-channel control
the centres are the contour's own vertices, so the count is normally 0.

## 2. D: a descriptive distance, not comparable across axons

D = sup_x |F_obs(x) − F_pool(x)|, the largest gap between the empirical CDF of the K observed 1NN distances and the
CDF pooled over the iterations (the same number as `scipy.stats.ks_2samp(...).statistic`; checked in
`test_ks_distance_is_scipys`). Because the pool holds B × K distances, F_pool is an accurate Monte Carlo estimate of
the null CDF and D behaves like a **one-sample** statistic with n = K.

**Its size under the null falls with K.** For n independent observations, E₀[√n D] → √(π/2) ln 2 = 0.8687 (the mean of
Kolmogorov's limiting distribution; Marsaglia et al. 2003; own check: 0.852 at n = 50 and 0.862 at n = 200,
`docs/sims/randomization/sim1_ks_mc.py`), so E₀[D] ≈ 0.869/√K: **0.194 at K = 20, 0.123 at K = 50, 0.087 at K = 100**.
1NN distances of one pattern are not independent (mutual nearest-neighbour pairs share one value; Diggle 2013), which
inflates D. Own simulation (**sim1**, `docs/sims/randomization/sim1_ks_mc.py`: K centres on a ring of radius 2000 nm,
band ±50 nm uniform in the radius, true hard core 60 nm, sequential inhibition; the "observed" pattern is one more
draw of the null; the null uses the paper's rule; B = 1000; 120 / 120 / 80 replicates at K = 20 / 50 / 100; seed 1):
the mean observed D under a true null was **0.242, 0.150 and 0.104**, i.e. 20-25 % above the independent-data value.

Consequences:

- a per-axon D cannot be compared with the paper's D = 0.054, whose aggregation level is not stated (presumably
  pooled over its 34 axons: thousands of distances);
- D decreases with K, and K grows with the perimeter, so a difference in raw D between groups of different calibre
  confounds organisation with cluster number.

D is therefore kept as a **descriptive** column (`ks_statistic`) and reported **with its own null** (section 4).

## 3. Why the two-sample KS p-value was removed

Until this version the program reported `ks_pvalue = scipy.stats.ks_2samp(observed, pooled).pvalue`. Four
assumptions of that p-value fail here:

1. **Independence within the sample.** The K observed 1NN distances are dependent (mutual pairs share one value;
   Diggle 2013), so the effective sample size is below K.
2. **A second independent sample.** The "random sample" is B replicates of one null for the same axon, not an
   independent sample; scipy treats its B × K values as independent draws and, with more than 10,000 of them, uses the
   asymptotic formula (SciPy developers, `ks_2samp` documentation): the test is effectively one-sample with n = K while
   being computed as two-sample.
3. **A null parameter estimated from the data.** The hard core is the smallest observed 1NN, so the null is
   conditioned on a statistic of the same data; the tabulated KS distribution is then wrong (Lilliefors 1967;
   Baddeley et al. 2014; the standard remedy re-estimates the parameter inside each simulation, Dao & Genton 2014).
4. **Ties.** Randomized positions sit on a 5 nm grid, so the pooled distances are discrete; the KS distribution
   assumes continuity (Conover 1972).

**Measured miscalibration** (own simulation, true null, nominal 5 %): in **sim1** (paper's rule, B = 1000) the scipy
p rejected **22.5 %, 20.0 % and 13.8 %** of true nulls at K = 20, 50, 100; in **sim1b**
(`docs/sims/randomization/sim1b_exchangeable.py`: the same generator, B = 199, 150 replicates, K = 20 and 50, seed 11)
it rejected **18.0 % and 16.7 %** with an exchangeable null (the true hard core) and **20.7 % and 8.7 %** with the
paper's rule. Across these runs: 9-22 % rejections at a nominal 5 %, up to about four times the nominal level. The
program's own validator reproduces this as a negative control (`validate_randomization.py`, section A: 7.5-23.3 %).

**The built-in head start of the paper's rule.** With d_min = the smallest observed 1NN, the null never produces a
distance below the observed minimum, while the observed sample always has the two members of the minimum pair there.
So F_obs exceeds F_null by **2/K** at the shortest distances **by construction**: sim1b measured a mean fraction of
observed 1NN below the null's smallest distance of 0.100 at K = 20 and 0.040 at K = 50 — exactly 2/K. With K ≈ 50-100
this head start is 0.02-0.04 in CDF units, the order of the D = 0.054 reported by the source paper. (This point, the
aggregation level of D = 0.054 and how its p < 10⁻³ was computed are questions for the authors.)

The p-value is gone from the results table, the export (`ks_pvalue`) and the column dictionary.

## 4. What replaces it: D's own null and Monte Carlo rank p-values

Each iteration's own distance against the pool, **D_i = sup|F_i − F_pool|**, gives the null distribution of D for
**this axon's** number of clusters, band and minimum separation, at no extra cost (the 1/B self-contribution of
iteration i to the pool is negligible). From it (`tools/mps_randomization.py`, `randomize_cluster_positions`):

- **null mean** E₀[D] = mean of D_i (`ks_null_mean`);
- **z_D** = (D − mean D_i)/sd D_i (`ks_z`), comparable between axons;
- **p_D** = (1 + #{D_i ≥ D})/(B_eff + 1) (`ks_p_mc_d`), one-sided;
- **p_med** = the two-sided rank of the observed median 1NN among the per-iteration medians, 2 × min of the two tails
  of (1 + b)/(B_eff + 1), capped at 1 (`ks_p_mc_median`).

These are Monte Carlo rank tests (Besag & Diggle 1977; Hope 1968; the "+1" so a p-value is never zero, Phipson &
Smyth 2010). They are exact only when the observed pattern is exchangeable with the simulations (Davison & Hinkley
1997); with the data-conditioned hard core they are **approximate**, and are labelled so in the table and the export
(Dao & Genton 2014). The batch never compares p-values between groups (section 7).

**Calibration** (own simulation, `validate_randomization.py`, section A, seed 20261014: K centres placed by sequential
inhibition in a ±50 nm band around a circle of perimeter K × 250 nm with a true hard core of 80 nm; B = 199;
120 replicates per cell; rejection rate at 0.05; the Wilson 95 % interval of 0.05 at n = 120 is 0.023-0.105):

| K | null | p_med | p_D | scipy KS p | mean z_D |
|---|---|---|---|---|---|
| 20 | exchangeable (true hard core) | 0.067 | 0.058 | 0.142 | +0.02 |
| 20 | paper's rule | 0.108 | 0.083 | 0.233 | +0.28 |
| 20 | bias-corrected hard core | 0.025 | 0.017 | 0.092 | −0.21 |
| 50 | exchangeable | 0.025 | 0.033 | 0.125 | +0.03 |
| 50 | paper's rule | 0.050 | 0.050 | 0.125 | −0.03 |
| 50 | bias-corrected | 0.042 | 0.042 | 0.108 | −0.12 |
| 100 | exchangeable | 0.017 | 0.017 | 0.075 | −0.19 |
| 100 | paper's rule | 0.083 | 0.100 | 0.225 | +0.23 |
| 100 | bias-corrected | 0.042 | 0.025 | 0.117 | −0.06 |

Reading: with an exchangeable null the Monte Carlo p-values hold their level (inside the Wilson band, or below it:
conservative); with the paper's rule they are somewhat liberal at K = 20 and K = 100 (0.08-0.11), as sim1 predicted
(0.108 / 0.058 / 0.063); the bias-corrected hard core brings them to or below the nominal level. The mean z_D of the
exchangeable null at K = 100 (−0.19) is about two Monte Carlo standard errors (≈ 1/√120 ≈ 0.09) from 0; the
validator's acceptance is |mean z_D| ≤ 0.15 + 2/√n. A run with 300 replicates (`validate_randomization.py --full`)
would settle it; it is listed as an open point. The scipy p over-rejects in every cell (negative control).

## 5. The CDF crossing

Gazal et al. report that the observed and randomized CDFs cross near 0.8: more short distances than random below it,
fewer long ones above it. Per axon, the empirical CDF of K distances has a sampling noise bounded by the
Dvoretzky-Kiefer-Wolfowitz inequality, √(ln(2/α)/(2K)) at level α (Massart 1990): **0.30 at K = 20, 0.19 at K = 50,
0.14 at K = 100** (α = 0.05; independent data, so a lower bound for 1NN distances). A sign change of a difference
smaller than that is noise, and "the last of several crossings" is arbitrary. The earlier detector (a sign change
beyond 15 % of the largest gap) scaled with the signal, not with the noise, and has been removed.

What the program does now (`cdf_band_from_samples`, `cdf_band_crossing`, `crossing_strength`):

1. **The null band.** On 256 distances spanning both samples, the central 1 − α (default 95 %; editable 0.01-0.10) of
   F_i − F_pool over the iterations. It is drawn on the CDF plot as two dotted grey curves around the randomized CDF.
   This is a pointwise band; a global rank envelope would need at least 2500 iterations (Myllymäki et al. 2017), and
   pointwise envelopes read at many distances do not hold an overall error rate (Loosmore & Ford 2006).
2. **The crossing.** Reported when the observed difference rises above the band at some distance and falls below it
   at a longer one; its level is F_null where the difference turns non-positive after the last excess point. An
   excess counts **only where the null has support** (F_null > 0): below the null's shortest distance the observed
   CDF is ahead by construction (the 2/K of section 3). Without this restriction almost every axon "crossed" at
   F ≈ 0.07.
3. **Significance.** The crossing's **strength** is the largest, over split points, of the smaller of the largest
   excess before the split and the largest deficit after it. Its Monte Carlo rank among the iterations' own strengths,
   (1 + b)/(B + 1) ≤ α, makes it significant (`ks_cdf_crossing_significant`). The DKW bound is shown beside it.

**Why not the DKW bound for significance** (own simulation, `validate_randomization.py`, section B: distances drawn
from a gamma(9, 25 nm) null, K = 50, B = 199, 200 replicates; a "planted crossing" draws a share of the distances from
0.5 × and the same share from 1.4 × the null — an excess of short and a deficit of long distances, as a fixed number
of clusters on a fixed perimeter produces; 40 % + 40 % "strong", 25 % + 25 % "moderate"):

| case | Monte Carlo strength (the program's rule) | DKW bound on both branches |
|---|---|---|
| null | 0.000 | 0.000 |
| planted crossing, strong | **0.815** | 0.070 |
| planted crossing, moderate | **0.585** | 0.005 |
| pure shift (no crossing) | 0.000 | 0.000 |

Under the true nulls of section A, significant crossings occurred in 0-3 % of draws. A single axon with tens of
clusters often cannot resolve a crossing: the table then says "not detectable at this K". (A mixture with only a
shorter component lies above the null everywhere and has no crossing at all; an earlier version of the validator's
planted case did that and could not detect anything by construction.)

## 6. The regularity index r = median 1NN × K / P

**Derivation.** For K cluster centres on a closed contour of length P the mean spacing is s = P/K. The Euclidean 1NN
of a centre is at most its arc distance to its nearer neighbour, so the **mean** 1NN cannot exceed s (the nearer gaps
sum to at most P); the **median** equals s for evenly spaced centres and can exceed it only slightly (e.g. by K/(K − 1)
with one coincident pair and the rest evenly spaced). So the median 1NN confounds two things: the **density** of
detected clusters (K/P) and the **regularity** of their arrangement. r = median 1NN / (P/K) removes the density
(the planar analogue is Clark & Evans 1954).

**Values** (own simulation, **sim3**, `docs/sims/randomization/sim3_nn_factors.py`: K = 70 centres on a circle of
perimeter 25 µm, seed 3): r = **1.00** evenly spaced, **0.92 / 0.84 / 0.75** with arc jitter sd 30 / 60 / 100 nm,
**0.47** for a random arrangement with a 60 nm hard core (the 1-D Poisson value is ln2/2 = 0.35 for the median).
`validate_nn1_controls.py` (section 4; simulated axons through the whole pipeline) gives r = 0.98-0.99 for a full
lattice, 0.82-0.90 / 0.66-0.79 / 0.69-0.75 with jitter 30 / 60 / 100 nm and 0.66-0.72 for the simulator's random
hard-core rings.

**Why agreement with the published 1NN is not a validation criterion.** From the paper's own regression
N = 4.08 P − 12.62 (P in µm) the mean spacing is 355 nm at P = 10 µm and 273 nm at 30 µm; with its median of per-axon
medians, 260 nm, r = 0.73-0.95 (derivation from Gazal et al. 2026). A pipeline that detects λ times more clusters per
µm at the same regularity reports a median 1NN λ times smaller. SMLM cluster detection depends on localization and
label density, blinking and the clustering parameters (Baumgart et al. 2016; Burgert et al. 2015; Annibale et al.
2011; Nieves et al. 2023; the eLife public review of the source preprint makes the same point), and an axial slab can
collect clusters of a neighbouring ring. sim3 measured each factor on synthetic rings: splitting a fraction f of
clusters into twins 40 nm apart lowers the median 1NN by 3 % (f = 0.1), 8 % (0.2), 17 % (0.3) and collapses it to the
twin distance at f = 0.5; foreign centres of a neighbouring ring lower it by 3 % (0.1), 11 % (0.25), 62 % (0.5);
missed clusters (detection efficiency 0.4) raise it by 33 %; interior clusters at 0.5-0.9 of the radius change it by
less than 2 % at 20 % of K; chord versus arc differ by less than 1 % on a folded contour. The pipeline version of the
controls (`validate_nn1_controls.py`; 2 simulated axons per cell, today's eps 25 nm / min samples 10) shows the
signature directly: DBSCAN eps 15-25 nm detects 0.92-1.04 of the true clusters (except eps 15 nm with min samples
20: 0.39) with median 1NN ≈ 117-135 nm, eps 40 nm detects 0.76-0.87 with ≈ 147-177 nm, and r stays at 0.59-0.67 in
every setting but that one (0.41): **the detected density moves the 1NN, r does not**. (In that run interior clusters changed the median by up to 11 % on one axon at 20 % of K, more than
sim3's < 2 %; the slab-leak control found no foreign clusters at any axial precision with the simulator's 190 nm
period.)

The results table therefore shows "Clusters per um" (the first thing to compare) and "regularity r" next to the 1NN;
the reference column is labelled "Reference (Gazal et al. 2026, preprint v1, pooled)" — another dataset and another
pipeline. The validation helper (`pool_nn_across_axons`, printed by `validate_full_pipeline.py`) reports the pooled
median and quartiles first and the pooled mode by a Gaussian KDE (Scott's rule; Parzen 1962; Scott 1992) and by a
Freedman-Diaconis histogram (Freedman & Diaconis 1981; 50 bins as the literal reading of the published figure), each
with a 95 % interval from resampling **axons**: the mode is smoothing-dependent (Silverman 1986), and the paper reads
its ~200 nm peak without stating bin width or smoothing.

## 7. The batch

- **Compare z_D, never raw D** (and never a p-value): `COMPARABLE_COLUMNS` holds `ks_z` and `nn_regularity_index`;
  `ks_statistic` and the p-values are not offered for group comparisons.
- **ICC chance level in closed form.** The one-way ANOVA estimator is ICC = (F − 1)/(F + n₀ − 1), F = MS_b/MS_w; under a
  true ICC of 0 and normal data F ~ F(k − 1, N − k) exactly for balanced designs and approximately, with
  n₀ = (N − Σn_i²/N)/(k − 1), for ragged ones (McGraw & Wong 1996; Donner 1986). The chance level at percentile p is
  (F_p(k − 1, N − k) − 1)/(F_p + n₀ − 1). Own simulation (**sim2**, `docs/sims/randomization/sim2_icc_null.py`):
  closed form vs 20,000 Monte Carlo draws, 2 × 9: 0.280 vs 0.282; 6 × 9: 0.135 vs 0.137; ragged (2, 3, 12, 20): 0.205
  vs 0.207; the 400-draw Monte Carlo used before had an sd of 0.013-0.027 across seeds (it was the 0.31 once quoted
  for 2 × 9). `validate_randomization.py`, section E, checks 8 designs within 0.01; with heavy-tailed (t₃) data the
  closed form differs by 0.01-0.03.
- **Design effect, not adjectives.** An ICC above its chance level is reported as "design effect 1 + (m − 1) ICC: N
  observations are worth N / design effect" (Killip et al. 2004). The earlier labels (mild / substantial / severe at
  0.2 / 0.5) had no source and are gone; optional labels small / medium / large at 0.05 / 0.10 / 0.15 follow the
  clustered-data convention (Zyzanski et al. 2004, after Hox 2002), off by default. Reliability cut-offs (Koo & Li
  2016) answer another question and are not used.

## 8. Related changes of the same release

- **Occupancy's sigma cap** (σ_max ≤ d_max/3; Gazal et al. 2026) is editable (0.2-1.0) and exported
  (`occupancy_sigma_cap_fraction`). When it binds (own simulation, **sim4**, `docs/sims/randomization/sim4_sigma_cap.py`,
  and `validate_randomization.py`, section F; Gaussian clusters of sd 20 nm and uniform discs of radius 40 nm, 2000
  each): it binds in 100 / 82 / 54 / 21 / 1 % of Gaussian clusters of n = 5-10 / 50 / 100 / 200 / 500 localizations,
  and then the claimed half-width is 36 / 44 / 56 nm at n = 5 / 10 / 50 against a 3σ of 60 nm; for discs of n ≥ 15 it
  returns about the radius (41-42 nm). Occupancy therefore depends on localizations per cluster.
- **Occupancy against calibre.** The claim of constancy is examined only with animals as the unit of replication
  (pooled axon-level correlations are pseudoreplication; Lazic 2010; Aarts et al. 2014): `tools/mps_constancy.py`
  (Pearson r with a Fisher interval, Fisher 1915; a bootstrap over animals; a random-intercept model for animal and
  ROI within animal, REML), refused with fewer than 3 animals. Coverage of its slope interval in
  `validate_randomization.py`, section G: 0.985-1.00 (3 and 6 animals × 6 axons).

## 9. Limitations and open points

- The Monte Carlo p-values are **approximate** with the paper's rule (somewhat liberal at small and large K in our
  synthetic runs); the bias-corrected hard core is an option, not the default (comparability with the paper).
- The mean z_D of −0.19 at K = 100 under an exchangeable null is within two Monte Carlo errors of 0 but should be
  re-checked with 300 replicates.
- The CDF band is pointwise; a global envelope (≥ 2500 iterations) is not implemented.
- A grid of 10 nm is not negligible (section 1); keep 5 nm or finer.
- Two-channel nulls place channel B in channel A's band; a shift-along-contour null (Lotwick & Silverman 1982) is the
  band-free alternative and belongs to the two-channel work, not to this control.
- The 1NN controls run with today's clustering parameters and are re-run when the clustering batch settles eps and
  min samples.

## 10. Decision record

- **2026-10-08** — the user accepted the recommendations of the research proposal: the KS p is removed and replaced by
  Monte Carlo p-values of the median and of D, with z_D, raw D kept one version as descriptive (P-R23); the paper's
  minimum rule stays, with the corrected hard core as an option, all labelled approximate (P-R24); the regularity
  index and the thinning control are added, and the comparison with the published values moves to the documentation
  (P-R25); the heuristic CDF-crossing detector is replaced (P-R26); the contour is closed before the spline,
  incomplete iterations are excluded/re-drawn by default, and the 25/50/100 nm band sweep is published (P-R27); only
  the 1/3 sigma cap, with its binding curve documented (P-R29); no ICC adjectives, design effect and effective n,
  optional cited labels off (P-R31); one Monte Carlo rule, α(B + 1) an integer, the 1NN randomization at 999 (P-R52).
  Records: `rand-ks-pvalue`, `rand-pooled-sample-and-ks-statistic`, `rand-cdf-crossing-level`,
  `rand-cdf-crossing-floor-015`, `rand-annulus-half-width-50nm`, `rand-bspline-smoothing-s-equals-K`,
  `rand-grid-spacing-5nm`, `rand-placement-sequential-rejection`, `rand-null-excludes-clusters-outside-annulus`,
  `rand-zero-min-distance-fallback`, `rand-max-attempts-factor-20`, `rand-random-seed-0`, `nn1-discrepancy-vs-paper`,
  `nn-pooled-mode-estimator`, `nn1-pooled-peak-histogram-bins`, `occupancy-sigma-cap-one-third`,
  `occupancy-calibrated-equivalent-thresholds`, `occupancy-constancy-vs-calibre`, `hecl-batch-icc-null`,
  `hecl-batch-icc-severity`.
- **2026-10-14** — implemented (axon tables v6). Two choices depart from the written design and **await the user's
  approval**: (a) the crossing's significance uses the Monte Carlo rank of its strength instead of the DKW bound on
  both branches (section 5: 81 % vs 7 % power for a strong crossing at K = 50, both 0 % under the null); (b) the
  column `ks_cdf_crossing` now holds the band-based level (empty when none) instead of keeping the old heuristic
  value for one more version as P-R26 suggested; the table version (v6) and the column dictionary say so. The same
  day the user required every analysis choice to be documented publicly for peer review; this page is that record for
  the randomization control.

## References

- Aarts E, Verhage M, Veenvliet JV, Dolan CV, van der Sluis S (2014). A solution to dependency: using multilevel
  analysis to accommodate nested data. *Nature Neuroscience* 17(4):491-496. doi:10.1038/nn.3648
- Annibale P, Vanni S, Scarselli M, Rothlisberger U, Radenovic A (2011). Quantitative photo activated localization
  microscopy: unraveling the effects of photoblinking. *PLoS ONE* 6(7):e22678. doi:10.1371/journal.pone.0022678
- Anonymous (Reviewer #3) (2026). Public review of Gazal et al. *eLife* Reviewed Preprint 111108 v1.
  doi:10.7554/eLife.111108.1.sa0
- Baddeley A, Diggle PJ, Hardegen A, Lawrence T, Milne RK, Nair G (2014). On tests of spatial pattern based on
  simulation envelopes. *Ecological Monographs* 84(3):477-489. doi:10.1890/13-2042.1
- Baddeley A, Rubak E, Turner R (2015). *Spatial Point Patterns: Methodology and Applications with R.* Chapman &
  Hall/CRC. doi:10.1201/b19708
- Baumgart F, Arnold AM, Leskovar K, Staszek K, Fölser M, Weghuber J, Stockinger H, Schütz GJ (2016). Varying label
  density allows artifact-free analysis of membrane-protein nanoclusters. *Nature Methods* 13(8):661-664.
  doi:10.1038/nmeth.3897
- Besag J, Diggle PJ (1977). Simple Monte Carlo tests for spatial pattern. *Applied Statistics* 26(3):327-333.
  doi:10.2307/2346974
- Burgert A, Letschert S, Doose S, Sauer M (2015). Artifacts in single-molecule localization microscopy.
  *Histochemistry and Cell Biology* 144(2):123-131. doi:10.1007/s00418-015-1340-4
- Clark PJ, Evans FC (1954). Distance to nearest neighbor as a measure of spatial relationships in populations.
  *Ecology* 35(4):445-453. doi:10.2307/1931034
- Conover WJ (1972). A Kolmogorov goodness-of-fit test for discontinuous distributions. *Journal of the American
  Statistical Association* 67(339):591-596. doi:10.1080/01621459.1972.10481254
- Dao NA, Genton MG (2014). A Monte Carlo-adjusted goodness-of-fit test for parametric models describing spatial point
  patterns. *Journal of Computational and Graphical Statistics* 23(2):497-517. doi:10.1080/10618600.2012.760459
- Davison AC, Hinkley DV (1997). *Bootstrap Methods and their Application.* Cambridge University Press.
  doi:10.1017/CBO9780511802843
- Dierckx P (1993). *Curve and Surface Fitting with Splines.* Oxford University Press.
  doi:10.1093/oso/9780198534419.001.0001
- Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal Point Patterns*, 3rd ed. Chapman and
  Hall/CRC. doi:10.1201/b15326
- Diggle PJ, Besag J, Gleaves JT (1976). Statistical analysis of spatial point patterns by means of distance methods.
  *Biometrics* 32(3):659-667. doi:10.2307/2529754
- Donner A (1986). A review of inference procedures for the intraclass correlation coefficient in the one-way random
  effects model. *International Statistical Review* 54(1):67-82. doi:10.2307/1403259
- Evans JW (1993). Random and cooperative sequential adsorption. *Reviews of Modern Physics* 65(4):1281-1329.
  doi:10.1103/RevModPhys.65.1281
- Fisher RA (1915). Frequency distribution of the values of the correlation coefficient in samples from an
  indefinitely large population. *Biometrika* 10(4):507-521. doi:10.1093/biomet/10.4.507
- Freedman D, Diaconis P (1981). On the histogram as a density estimator: L2 theory. *Zeitschrift für
  Wahrscheinlichkeitstheorie und verwandte Gebiete* 57(4):453-476. doi:10.1007/BF01025868
- Früh SM, Matti U, Spycher PR, Rubini M, Lickert S, Schlichthaerle T, Jungmann R, Vogel V, Ries J, Schoen I (2021).
  Site-specifically-labeled antibodies for super-resolution microscopy reveal in situ linkage errors. *ACS Nano*
  15(7):12161-12170. doi:10.1021/acsnano.1c03677
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of βII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic
  nerve axons. bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1
- Hope ACA (1968). A simplified Monte Carlo significance test procedure. *Journal of the Royal Statistical Society B*
  30(3):582-598. doi:10.1111/j.2517-6161.1968.tb00759.x
- Killip S, Mahfoud Z, Pearce K (2004). What is an intracluster correlation coefficient? *Annals of Family Medicine*
  2(3):204-208. doi:10.1370/afm.141
- Koo TK, Li MY (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability
  research. *Journal of Chiropractic Medicine* 15(2):155-163. doi:10.1016/j.jcm.2016.02.012
- Lazic SE (2010). The problem of pseudoreplication in neuroscientific studies. *BMC Neuroscience* 11:5.
  doi:10.1186/1471-2202-11-5
- Lilliefors HW (1967). On the Kolmogorov-Smirnov test for normality with mean and variance unknown. *Journal of the
  American Statistical Association* 62(318):399-402. doi:10.1080/01621459.1967.10482916
- Loosmore NB, Ford ED (2006). Statistical inference using the G or K point pattern spatial statistics. *Ecology*
  87(8):1925-1931. doi:10.1890/0012-9658(2006)87[1925:SIUTGO]2.0.CO;2
- Lotwick HW, Silverman BW (1982). Methods for analysing spatial processes of several types of points. *Journal of the
  Royal Statistical Society B* 44(3):406-413. doi:10.1111/j.2517-6161.1982.tb01221.x
- Marriott FHC (1979). Barnard's Monte Carlo tests: how many simulations? *Applied Statistics* 28(1):75-77.
  doi:10.2307/2346816
- Marsaglia G, Tsang WW, Wang J (2003). Evaluating Kolmogorov's distribution. *Journal of Statistical Software*
  8(18):1-4. doi:10.18637/jss.v008.i18
- Massart P (1990). The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality. *Annals of Probability*
  18(3):1269-1283. doi:10.1214/aop/1176990746
- McGraw KO, Wong SP (1996). Forming inferences about some intraclass correlation coefficients. *Psychological
  Methods* 1(1):30-46. doi:10.1037/1082-989X.1.1.30
- Myllymäki M, Mrkvička T, Grabarnik P, Seijo H, Hahn U (2017). Global envelope tests for spatial processes. *Journal
  of the Royal Statistical Society B* 79(2):381-404. doi:10.1111/rssb.12172
- Nieves DJ, Pike JA, Levet F, et al., Owen DM (2023). A framework for evaluating the performance of SMLM cluster
  analysis algorithms. *Nature Methods* 20:259-267. doi:10.1038/s41592-022-01750-6
- Parzen E (1962). On estimation of a probability density function and mode. *Annals of Mathematical Statistics*
  33(3):1065-1076. doi:10.1214/aoms/1177704472
- Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating exact p-values when permutations
  are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1):39. doi:10.2202/1544-6115.1585
- SciPy developers. `scipy.stats.ks_2samp` documentation.
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- Scott DW (1992). *Multivariate Density Estimation.* John Wiley & Sons. doi:10.1002/9780470316849
- Silverman BW (1986). *Density Estimation for Statistics and Data Analysis.* Chapman and Hall.
  doi:10.1201/9781315140919
- Wang JS (1994). A fast algorithm for random sequential adsorption of discs. *International Journal of Modern Physics
  C* 5(4):707-715. doi:10.1142/S0129183194000817
- Zyzanski SJ, Flocke SA, Dickinson LM (2004). On the nature and analysis of clustered data. *Annals of Family
  Medicine* 2(3):199-200. doi:10.1370/afm.197
