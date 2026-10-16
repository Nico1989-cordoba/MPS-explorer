# Ring-pair viability and the reference period

> **Resumen (español).** Antes de comparar los clusters de dos anillos consecutivos, MPS Explorer decide si el
> perfil axial del axón separa esos anillos (regla v2: pico y valle de SiZer, piso de localizaciones, copias por
> fuga axial). El **período de referencia** es 170 nm (Gazal et al. 2026, ciático, eje z) y sólo sirve de vara
> externa; lo que escala (ancho de SiZer, ventana de pico, σ/P) usa el **período medido en cada axón**. Las copias
> por fuga se toleran **por par**, c = δ·sd*/(2(1−p0)), con δ = 0,25 sd (viable) y 0,5 sd (marginal), en vez de
> 2 % / 5 % fijos. De las reglas de fuga preregistradas se quitan (i), (iii) y (iv) porque no separan fuga de
> columnas; (ii) va por par con Bonferroni y (v) es un test de permutación. El piso de conteo se calibra en cada axón
> por raleo. Cada número dice de dónde sale (literatura / derivación / simulación propia, con su script en
> `docs/sims/viability/`). Decidido por el grupo de investigación del usuario (registro al final).

This page explains how MPS Explorer decides which pairs of consecutive rings of one axon can be compared by the
column test, and why each number it uses has the value it has. The code is in `tools/mps_axial_precision.py`
(`ViabilityParams`, `z_quality`, `viability_v2`, `copy_tolerance`, `calibrate_count_floor`, `silverman_reading`),
`tools/mps_sizer.py` (SiZer), `tools/mps_axial_clusters.py` (the cluster-level variant), `tools/mps_leak.py` (the leak
diagnostics) and `tools/mps_viability_params_ui.py` (the **Criteria...** dialog of the Z quality window). Every value
below is editable in that dialog, shown with its origin badge and its documented range; a value outside the range is
kept and flagged, never clamped. The values are stored once (`viability_params` in the settings file) and read by Z
quality, the viability explorer, the column review and the batch (`batch_columns.py --viability-params`).

The origin of each number is written in brackets:

- **[literature]** — taken from a publication, cited;
- **[derivation]** — computed by a formula written on this page (first principles, a standard method, or a convention
  with its reasoning);
- **[own simulation]** — measured on this project's synthetic data; the setup is described and the script is in
  `docs/sims/viability/` with its output next to it (no real data enters any of them).

## 1. What "viable" means

The column test compares the lateral positions of the clusters of two consecutive rings. It can only be read when the
two rings are two **resolved** structures along the axon's axis (z'), when neither ring has lost its clusters to the
detection floor, and when axial measurement error has not copied clusters from one ring into the other. Rule v2 asks,
for each consecutive pair:

1. each ring has a significant SiZer **peak** within the peak window of its fitted centre;
2. a significant SiZer **valley** separates the two ring centres;
3. each ring holds at least **f_min** of the reference ring's localizations (the **count floor**);
4. the expected fraction of axial-leak **copies** is within the pair's tolerance (VIABLE), or within the looser
   marginal tolerance (MARGINAL: a sensitivity analysis only); otherwise NOT VIABLE.

The rule is a conjunction of necessary conditions built from published components (SiZer, Chaudhuri & Marron 1999;
DBSCAN, Ester et al. 1996; Gaussian tails); it has no published counterpart as a whole, and every threshold is an
editable suggestion. **The viability never reads the column statistic**: it is the geometry of the z profile only, so
restricting the column test's Monte Carlo null to the selected pairs does not bias its (b + 1)/(B + 1) p-value
(Phipson & Smyth 2010). The column statistic's p-values remain labelled *not calibrated*.

A first filter (`z_quality`) is shown beside rule v2: the separation index d, the same copy tolerance, and an
information ratio; its verdict is not used by the column test.

## 2. The reference period and the axon's own period

**Reference period P_ref = 170 nm [literature].** Gazal et al. (2026) report a periodicity of 170 ± 15 nm of
βII-spectrin along the axonal axis (z) of mouse sciatic nerve, the preparation this program was built for. P_ref is
used only as an **external yardstick** of the merge flags of the first filter:

- a ring component wider than 0.5 P_ref cannot be one ring of a bimodal pair (the equal-weight bimodality condition,
  section 4) **[derivation]**;
- a gap between two ring centres larger than 1.3 P_ref suggests a lost ring in between **[derivation]**: with the
  published spread 170 ± 15 nm, 1.3 P lies 3.4 period standard deviations above P and still 0.7 P below the 2 P of a
  skipped ring, so it separates a normal spacing from a missed ring with margin on both sides.

These flags ask whether a spacing is anomalous **for the biology**; an axon with a lost ring would raise its own
median spacing, which is why they must not use the axon's own period.

Presets in the dialog (and a free value, documented range 150-210 nm):

| Preset | Value | Source |
|---|---|---|
| sciatic nerve, axial (default) | 170 nm | Gazal et al. 2026 (170 ± 15 nm along z) |
| sciatic nerve, lateral STED, mean | 186 nm | D'Este et al. 2016 (as quoted by Gazal et al. 2026; the median there is ~175 nm) |
| cultured neurons | 182 nm | Xu et al. 2013 (βII-spectrin C-terminus, 182 ± 18 nm) |
| cultured neurons | 190 nm | Zhong et al. 2014 (~190 nm) |

The value used is written in every report with its source words ("literature: Gazal et al. 2026, sciatic nerve axial
period 170 nm", or "user: 175 nm").

**The axon's own period P [derivation].** Every **scale-type** quantity uses the axon's measured period: the median,
over consecutive listed rings, of the spacing between ring centres divided by their index step (`axon_period_nm`),
so a missing ring (index step 2) counts as one period, not two; the median is robust to one mis-fitted ring. P sets
the SiZer bandwidth h = 0.10 P, the peak window P/3 and the width-over-period summary σ_z / P. **With fewer than two
rings P_ref is the fallback**, and the report says "fallback: the reference period (fewer than two rings)".

What changed: the program's earlier constant was 170.8 nm. Its comment named a dataset; an audit traced the number to
the pooled median of the example regions distributed with the program, so it was replaced by the published value
with its source, and the scale-type quantities were moved to the axon's own period.

## 3. SiZer: the peak and the valley, and the map over bandwidths

SiZer (Chaudhuri & Marron 1999) smooths the z' profile with a Gaussian kernel of width h and asks, at every point,
whether the slope of the smoothed density is significantly positive, negative or neither (simultaneous quantile by
the independent-blocks device, α = 0.05). A **peak** is a significant rise followed by a significant fall; a
**valley** the opposite. The variance of the slope is **cluster-robust**: the localizations of one 3D cluster share
their z and are not independent, so each cluster is one unit of the sandwich variance (Liang & Zeger 1986; Cameron &
Miller 2015). An optional small-sample factor G/(G − 1) (G the number of groups; under 3 % for G ≥ 40) is off by
default.

**h = 0.10 P [derivation + own simulation]; documented range 0.05-0.25 P.** The scale is bounded above by the
resolution condition of an equal pair: smoothing a ring of width s gives width √(s² + h²), and the valley survives
only while √(s² + h²) < P/2 (Helguerro's condition, Schilling et al. 2002), i.e. h < 0.38 P at s/P = 0.32. It is
bounded below by the slope variance, proportional to 1/(n h³). Own simulation (`sim_sizer_valley.py`: two equal
Gaussian rings at 0 and P, n iid points or 4000 points in 80 clusters with the cluster-robust variance, 100
replicates per cell (60 at 16000 points); output `sim_sizer_valley.txt`):

| σ/P | points | P(valley) at h = 0.05 / 0.10 / 0.15 / 0.20 P |
|---|---|---|
| 0.30 | 1000 iid | 0.02 / 0.83 / 1.00 / 1.00 |
| 0.30 | 4000 iid | 0.73 / 1.00 / 1.00 / 1.00 |
| 0.35 | 4000 iid | 0.07 / 0.67 / 0.96 / 0.99 |
| 0.25 | 4000 in 80 clusters | — / 0.95 / — / 0.89 |
| 0.30 | 4000 in 80 clusters | — / 0.47 / — / 0.25 |
| 0.35 | 4000 in 80 clusters | — / 0.05 / — / 0.03 |

With iid points a wider kernel helps; with **clustered** points (the real case) 0.10 P beats 0.20 P because a wider
kernel mixes more clusters into each slope and inflates the cluster-robust variance. 0.10 P is therefore a defensible
default for clustered data, not a unique optimum, and the report shows the **SiZer map**: the valley of every pair at
h = 0.05, 0.10, 0.15, 0.20 and 0.25 P (the verdict reads the chosen h).

**Peak window P/3 [derivation]; documented range P/5-P/2.** The windows of two rings one period apart must not
overlap (w < P/2), and the window must exceed a few standard deviations of the peak position. For a kernel-density
mode the position sd is √((√2/4)(s² + h²)^(5/2)/(n h³)) (Parzen 1962); the simulation above checks it (measured /
analytic 0.0300 / 0.0316 P at n = 1000, s = 0.25 P, h = 0.10 P). For a ring of tens of clusters it is about 0.05 P,
so P/3 is roughly four such deviations.

## 4. The separation index d (first filter)

d = |μ_b − μ_a| / √(σ_a² + σ_b²) compares the gap between the fitted ring centres with their combined width. Two equal
Gaussian rings are bimodal only above **d = √2** (mean difference > 2σ; Schilling et al. 2002; Eisenberger 1964)
**[derivation]**. For equal widths and a gap of one period, σ/P = 1/(d√2) and the valley depth relative to the density
at a ring centre is 1 − φ(P/2σ) / (½[φ(0) + φ(P/σ)]) **[derivation]**:

| d | σ/P | valley depth |
|---|---|---|
| 2.2 (viable, default) | 0.32 | 41 % |
| 2.0 (marginal, default) | 0.35 | 28 % |
| √2 (floor) | 0.50 | none |

The thresholds 2.2 / 2.0 **[own simulation]** were chosen from a simulation study whose generator was parameterised
on a pilot acquisition; no column statistic of a real axon entered their choice. They are suggestions (documented
range 1.41-3.0, d marginal ≤ d viable), and the dialog shows the equivalents above for any typed value.

## 5. The tolerated fraction of leak copies, per pair

**Mechanism.** A localization of ring k measured with an axial error larger than its distance to the boundary lands
in ring k ± 1, where DBSCAN can promote the leaked localizations of one cluster to a cluster of its own: a **copy** at
the parent's lateral position. A copy is matched to its parent within any tolerance, so it inflates the column count.
The expected copy fraction of a pair is computed as the sum, over the clusters of both rings, of P[Bin(n0, F) ≥
min_samples] (the DBSCAN core count, Ester et al. 1996), n0 the cluster's size corrected for the slab's Gaussian
retention (an inclusion-probability correction, Horvitz & Thompson 1952; floored at 5 % retention, a numerical
safeguard) and F the Gaussian tail beyond the cut, divided by K_a + K_b. It evaluates every cluster at its ring's
centre, so it can under-estimate copies when the rings are not flat.

**Derivation of the tolerance [derivation].** Each copy adds one match unless its parent was already matched by
chance, with probability p0 (the null match rate). With copies = c (K_a + K_b), the expected extra matches are
c (K_a + K_b)(1 − p0), and the standardised column excess shifts by

  Δζ = c (K_a + K_b)(1 − p0) / (sd* · min K) = 2 c (1 − p0) / sd*   (equal K),

where sd* is the null standard deviation of the matched fraction (toroidal-shift null; Lotwick & Silverman 1982;
Besag & Diggle 1977). Tolerating a bias of δ null standard deviations gives

  **c_max = δ · sd* / (2 (1 − p0))**,

with **δ_viable = 0.25** and **δ_marginal = 0.5** (documented ranges 0.1-0.5 and 0.25-1.0). A bias of 0.25 sd inflates a
one-sided 5 % test to about 8 %, 0.5 sd to about 13 %. Before the null is run, p0 and sd* are approximated per pair
from its own counts:

  p0 = 1 − exp(−2 τ K_other / L_other) (mean of the two directions),  sd* = √(p0 (1 − p0) / min K),

with τ the eclipse tolerance the column test runs at (today the pre-specified τ_0 of `config/columns_params.yaml`),
K the cluster counts and L the ring lengths. The same tolerance is used by rule v2, the first filter and the
cluster-level variant (one parametrisation). When the counts, the lengths or τ are unknown, the fixed **2 % / 5 %**
are used as the displayed suggestion, said in words.

**Check [own simulation]** (`sim_leak_bias.py`: two rings of K clusters on a closed perimeter, τ = 80 nm, mean
spacing 200 nm so p0 ≈ 0.55, copies placed in register, 300 replicates per cell; output `sim_leak_bias.txt`): the
measured shift per 1 % of copies was 0.062 / 0.108 / 0.135 sd at K = 20 / 40 / 80, against 0.083 / 0.115 / 0.163
predicted (6-26 % below the first-order value, i.e. the formula is slightly conservative). With p0 ≈ 0.55 the fixed
2 % equals the 0.25 sd tolerance at about 50-60 clusters per ring and the fixed 5 % the 0.5 sd one at about 30-45; a
fixed fraction corresponds to a fixed bias for one ring size only, which is why the tolerance is computed per pair.

**On synthetic axons** (section 11): with τ_0 = 80.6 nm and 30-45 clusters per ring the tolerance came out
1.44-1.68 % (viable) and 2.88-3.36 % (marginal).

## 6. The leak diagnostics: what was removed, and why

The pre-registered analysis listed five leak diagnostics as rules. On synthetic axons with known truth three of them
were found to have no valid reading as a decision between "leak" and "columns", and are no longer evaluated as rules
(their descriptive fields stay in the tables):

- **(i) shared emission events against a random re-pairing null — removed.** The linking of localizations into
  emission events (as in Picasso; Schnitzbauer et al. 2017) is local, so random cross-ring pairs essentially never
  share an event, and the test rejects whenever **any** matched pair shares one — under leak and under genuine
  columns alike (a true partner can absorb its neighbour's leaked localizations). The per-pair shared-event
  fractions are reported; no highlight threshold is set by default (it depends on the linking radius and the dark
  time, Annibale et al. 2011), the user can set one.
- **(iii) the excess persists with a dead zone at the boundaries — removed.** A positive excess at a guard narrower
  than the width that suppresses the copies is as likely under leak as under columns. The guard is reported as a
  **sensitivity table** in units of the axon's effective axial width s_eff (median over rings of √(σ_struct² +
  lpz²)): total dead-zone widths {0, 0.5, 1, 1.5} × s_eff plus the derived guard **g\* = 2 (1.91 s_eff − P/2)**
  **[derivation]**, the width beyond which the residual leak of a 200-localization cluster no longer makes a copy (at
  least 10 leaked) with 5 % probability (P[Bin(200, F\*) ≥ 10] = 0.05 gives F\* = 0.028, z\* = 1.91): g\* = 25 nm at
  s_eff/P = 0.30 and 90 nm at 0.40, so the device is informative only for narrow rings. (The implemented zone was a
  60 nm total width, half the pre-registered 60 nm per side; recorded as a deviation.)
- **(iv) axial profiles bimodal more often than under simulated leak — removed.** The bootstrap likelihood-ratio test
  (McLachlan 1987) rejects a single Gaussian for leak tails and for genuine second clusters alike, and the
  simulated-leak reference it needs does not exist. The fraction of profiles rejecting one Gaussian is reported
  descriptively, over the eligible profiles (below).
- **(ii) partner vs axial precision — kept, per pair with Bonferroni.** Leak predicts that clusters with poorer median
  axial precision have a partner more often: a one-sided permutation test of the Spearman correlation (Spearman 1904;
  p = (b + 1)/(B + 1), Phipson & Smyth 2010). A conjunction over n pairs at 5 % has a family-wise false-alarm rate of
  1 − 0.95ⁿ (9.8 % with two pairs), so each defined pair is read at **α / n_defined** (Bonferroni; Dunn 1961). It is
  computed per ring pair across clusters, a change of unit from the pre-registered across-pair form (a deviation).
- **(v) size ratio — replaced by a permutation test.** The pre-registered constant (median matched size ratio ≥ 0.5)
  passes or fails genuine columns according to the ring's own size dispersion: for lognormal sizes the median
  smaller-to-larger ratio of two independent clusters is exp(−0.674 √2 σ_log), about 0.55 at a coefficient of
  variation of 0.7 and 0.45 at 1.0 **[derivation]**. The statistic is kept (the median of the matched pairs'
  smaller-to-larger ratios) and compared with its **within-ring permutation null**: sizes are permuted among the
  usable clusters of each ring with the matching held fixed, B = 999, p = (b + 1)/(B + 1) with b the null medians at
  or below the observed one (Besag & Diggle 1977; Phipson & Smyth 2010); per ring pair and for the axon (each ring
  permuted once per replicate). It is exact under within-ring exchangeability with a scalar eclipse tolerance and
  labelled approximate with a per-pair tolerance. A Mann-Whitney test against all unmatched cross pairs is **not**
  used: those K_a × K_b ratios share their clusters, so its tabulated p rests on an independence that does not hold.
  Own simulation (`sim_rule_v_mc_null.py`, lognormal sizes cv 0.4-1.5, K 10-60, 1000 ring pairs per cell, B = 199;
  output `sim_rule_v_mc_null_out.json`): under genuine columns the Mann-Whitney p rejected 1.5-4.3 % at a nominal 5 %,
  the permutation p 3.7-6.2 % (5 % within Monte Carlo error); with 6 / 10 source-copy pairs among 12 matches at K = 40
  the permutation p detected leak in 59 % / 97 % of ring pairs, the Mann-Whitney p in 47 % / 91 %.

Other values of the diagnostics, all editable in `LeakParams`:

- **"explained by leak"** = the child's z′ is more likely under the parent's Gaussian tail than under its own ring
  component (likelihood ratio; Neyman & Pearson 1933) **and** the child is no larger than binomial leak would make
  it. The likelihood-ratio cut is the log prior odds, 0 by default (equal odds; range −3 to 3).
- **size α = 0.01** (range 0.001-0.1): P[Bin(n0, F) ≥ n_child] ≥ α accepts a child as consistent with pure leak; a
  smaller α attributes more children to leak, the conservative direction for the column hypothesis (a conventional
  level; Cowles & Davis 1982).
- **descriptive boundary band** at 0.5, 1 and 2 σ_z side by side (1 σ is a convention; with σ/P = 0.3 and the
  boundary at one slab edge the centred expectations are 0.08 / 0.23 / 0.64 [derivation]); the likelihood ratio is
  the decision statistic.
- **profile eligibility in events**: a column enters the axial-profile test when its emission events reach
  n_80(π_min) = (z_0.975 + z_0.8)² / (π_min² I(π_min)), the Wald sample size for 80 % power to detect a
  period-offset component of weight π_min = 0.15 (range 0.05-0.5), with I the Fisher information of the fitted
  mixture (Wald 1943; Lachin 1981; Cohen 1992). With the nuisance parameters treated as known this is an information
  bound, a lower bound on the events needed. The earlier 100-localization rule is shown only as a suggestion: a
  cluster of 100 localizations may hold 10 events. The closed form with I(0) = exp((P/σ)²) − 1 is labelled
  "information lower bound", not a sample size.
- **profile bootstrap B = 199** (range 99-1999): the test is exact at any B (Hope 1968) with p on a grid of 0.005 and a
  Monte Carlo standard error of about 0.015 near p = 0.05.
- **cleaned excess**: the toroidal-shift test rerun with the leak-attributed children removed from the observed
  configuration **and** from every null replicate. The earlier form that removed them from the observed count only is
  biased low by about the number removed over the null's scale; it is kept only as "a lower bound" and not used for
  inference.

## 7. The count floor, calibrated on each axon by thinning

Criterion 3 protects the comparison against clusters lost to the DBSCAN detection limit. A cluster survives the random
deletion of localizations while some kept localization still has at least min_samples kept neighbours within eps
(the core condition; Ester et al. 1996). **Calibrate count floor** (Z quality) thins the detected clusters of the
reference ring on the axon itself: each localization kept with probability f = 1.00, 0.95, …, 0.30, 20 replicates
each, the run's eps and min_samples in the axon-frame plane; the recall at f is the share of clusters that survive,
and **f_min** is the f at which the mean recall falls to the **recall tolerance 0.90** (linear interpolation).

**Recall tolerance 0.90 [derivation]; range 0.80-0.95.** Random loss of clusters does not bias the toroidal-shift
test (the null is computed on the detected clusters) but costs power: a 10 % loss raises sd* by about 5 %. The
tolerance is a power convention (compare the 0.8 convention, Cohen 1992); 0.90 is cautious because the only validity
concern, non-random loss of dim clusters, cannot be seen in the count fraction.

**Until the calibration is run**, the floor is read from a line fitted to an earlier simulation study,
f_min(x) = 1.46 − 0.336 ln x, x = localizations per cluster of the reference ring / min_samples, valid for x in
5.8-13.3 (frozen above, extrapolated and capped at 100 % below, both flagged) **[own simulation]**; it is labelled
"suggestion (simulation)". Why the per-axon calibration replaces it (`sim_thinning_recall.py`: strips of clusters with
lognormal sizes cv 0.7, 2D sd 20 nm, eps 25 nm, min_samples 10, 25 replicates; output `sim_thinning_recall.txt`):

| x | f_90 true (sd) | f_90 by in-silico thinning | fitted line |
|---|---|---|---|
| 4 | 0.79 (0.06) | 0.80 (0.06) | 0.99 |
| 8 | 0.53 (0.06) | 0.55 (0.07) | 0.76 |
| 16 | 0.34 (0.03) | 0.34 (0.04) | 0.53 |

The floor depends on the cluster-size distribution and on eps, not on x alone; thinning the detected clusters
recovers the truth, the line does not.

The **reference ring** (the denominator) is the central ring of the stack by default (the middle ring; with two
rings, or two middle rings, the more populated), expected nearest focus where astigmatic precision and localization
density are best (Huang et al. 2008); the most populated ring or a picked ring can be chosen. x uses the mean
localizations per cluster by default (the median is offered). The ring with the smallest median lpz is marked "best
axial precision" (information only).

## 8. Reference multimodality reading: Silverman's test (and why not the dip test)

Beside the SiZer verdict each pair shows **Silverman's critical bandwidth** h_crit (the smallest Gaussian bandwidth at
which the kernel density of the pair's z′ has one mode) and its smoothed-bootstrap p (Silverman 1981; Efron's
variance-corrected resampling at h_crit, B = 199, p = (b + 1)/(B + 1); at most 4000 values, a deterministic
subsample). It treats the localizations as independent, which they are not, so it is a **reading, never a
criterion**. Hartigan's dip test (Hartigan & Hartigan 1985) is not included: SciPy does not provide it and its exact
algorithm needs a verified implementation; a reading that cannot be checked against the published algorithm is not
shown.

## 9. The 3D variance groups: the axial scale from the file's precision

The units of the cluster-robust variance are 3D DBSCAN clusters with eps = 25 nm and min_samples = 10, the lab's
published settings (Gazal et al. 2026: eps from the ~20 nm lateral precision, min_samples from the mean switching
cycles of the dyes, Dempsey et al. 2011) **[literature]**. In 3D the axial coordinate must be in the same precision
units, so z is scaled by **σ_xy / σ_z of the file** (median lateral over median axial reported precision) before the
clustering **[derivation]**. Astigmatic axial precision is typically 2-3× the lateral (Huang et al. 2008: 9-11 nm in
x, y and 22 nm in z), so **0.5** is the suggestion when the file carries no precision. A typed value overrides both
(documented range 0.25-1.0). **A measured ratio outside that range is used as measured and flagged** in the report
("outside the documented range 0.25-1.0 (used as measured)") with a warning to check the precisions (Data quality)
or to type a scale. On this project's simulated axons the ratio was 0.18 (section 11).

## 10. The cluster-level variant (exploratory)

Rule v2-clusters reads the same four criteria on one axial position per cluster (ring clusters by their median or a
truncation-corrected centre, or slab-free 3D clusters). With about a hundred-fold fewer points than the localizations
its slope variance is about a hundred times larger at the same h (variance ∝ 1/(n h³)), so it has little power; two
of its cluster sets carry a construction gap at every slab cut, and the third depends on the axial structure itself.
It is an **exploratory** reading displayed beside rule v2 and selects pairs only as an exploratory selection.

- **k_min = 0.90 [derivation]** (was 0.89, a fitted constant): under Bernoulli thinning the ratio of detected cluster
  counts K(ring)/K(reference) **is** the cluster recall, so the threshold is the recall tolerance itself (the fit
  that gave 0.887 [0.855; 0.906] recovered this identity up to sampling noise).
- **Bandwidth 0.10 P** for comparability with rule v2 (range 0.10-0.30 P), with the map at 0.15-0.30 P and 0.20 P as
  the sensitivity row.
- **Slab-gap flag = 1.0 h [derivation]** (was a fixed 15 nm; range 0.5-2 h): a valley strictly between the centres
  within one bandwidth of the slab gap is a construction artefact (ring clusters cannot cross the cut, and the kernel
  spreads the gap edges by about h, about twice the location noise of a valley among a hundred cluster centres).
- **Cluster-level leak estimate ("tnfix")**: each cluster at its own truncated-normal centre (Cohen 1950) with the
  **ring's** width, which includes the between-cluster dispersion of the centres, so it over-estimates copies (about
  three-fold in simulation, never below the truth): reported as an **upper bound**. An **own-width** estimate (each
  cluster's own centre and spread, both fitted by truncated-normal maximum likelihood on its own z values; the fitted
  spread already contains the localization precision, so lpz is not added again) is computed and shown but not used
  until its synthetic calibration is run. Floors (5 nm width, 5 % retention) are numerical safeguards that do not act
  on realistic data.

## 11. What changed in numbers (synthetic axons only)

Six simulated axons (`batch_columns.write_simulated_input`: circle seeds 39 / 22 / 30 with leak scales 0.3 / 0.6 / 0.7
— the three demonstration cases —, ellipse seed 7 leak 0.3, circle seeds 21 and 5), analysed by the program before
and after this change with `docs/sims/viability/viability_numbers.py` (outputs `viability_numbers_before.json` /
`_after.json`):

- P_ref 170.8 → 170 nm; the axon's own period 151.9-174.0 nm, so h = 15.2-17.4 nm instead of 17.08 nm everywhere;
- copy tolerance 2 % / 5 % → 1.44-1.68 % / 2.88-3.36 % per pair; **no verdict of rule v2 or of the first filter
  changed** on these six axons (the marginal pair of the marginal case, 2.69 % copies, stays marginal);
- 3D groups' z scale 0.5 → 0.18 (the simulator's precision ratio, flagged as outside the range); no SiZer peak or
  valley changed;
- cluster-level variant: one pair (circle seed 21, slab-free 3D clusters, rings 1-2) moved from viable to marginal
  (cluster-level copies 1.97 % against a tolerance of 1.48 %).

## 12. Limitations and open items

- **The axial precision is the reported lpz.** The calibrated axial precision (lpz scaled by the axial NeNA ratio of
  each depth stratum; Endesfelder et al. 2014) enters the leak fraction in a later version; until then a single
  global factor (`lpz_scale`, 1.0, range 0.5-3) is a sensitivity knob. Averaging the tail over each localization's
  own width matters where the boundary is more than √2 widths away (there the mean-width shortcut under-estimates the
  leak, Jensen 1906); nearer the boundary the shortcut over-estimates it.
- **The tolerance uses today's τ.** p0 is computed from the eclipse tolerance the column test uses now (τ_0 = 80.6
  nm, a pilot-derived value). When the derived tolerance replaces it, the per-pair tolerance follows automatically;
  the bias simulation of section 5 will be re-run at the new τ.
- **Which leak estimate is THE criterion** stays the user's choice until the synthetic calibration of the ring-level,
  cluster-level and own-width estimates is run; the own-width estimate becomes a selectable criterion then.
- The detection rates of "explained by leak" on synthetic axons depend on the simulated regime and on the axial
  precision calibration; they are not properties of the rule and are not quoted as such.
- The conditional simulated null (a replicate is accepted only when it reproduces the observed rings and selected
  pairs) is labelled *uncalibrated* until its false-alarm rate is verified on synthetic axons.
- The thinning calibration models random loss of localizations; it does not see a non-random loss of dim clusters.
- The p-values computed under exploratory selections are not adjusted for the number of selections tried; they are
  labelled exploratory and the count of selections tried (this session and across every session of the log) is shown
  and written beside the batch outputs (`selection_counter.json`), following the position that adjustment belongs to
  confirmatory conclusions (Bender & Lange 2001) and that analytic flexibility must be disclosed (Simmons et al. 2011;
  Steegen et al. 2016; Simonsohn et al. 2020).

## 13. Decision record

- **2026-10-02**: rule v2 adopted as the declared default (SiZer peak and valley, count floor, leak copies). It was
  fixed before any column statistic was computed on real data, after inspection of pilot z profiles; the earlier
  claim that its thresholds were fixed "before any real axon was measured" was withdrawn as not supported by the
  record.
- **2026-10-08**: accepted by the user's research group (proposal items on the reference period, the copy tolerance,
  the count floor, the 3D groups, the leak rules, the viability window, the provenance wording):
  - reference period 170 nm (Gazal et al. 2026) with presets and range 150-210 nm; the axon's own period scales h,
    the peak window and σ/P;
  - per-pair copy tolerance in null standard deviations (0.25 / 0.5) instead of fixed 2 % / 5 %;
  - count floor calibrated per axon by thinning, recall tolerance 0.90, k_min = that tolerance;
  - 3D groups' z scale from the file's precision ratio (0.5 as fallback);
  - removal of rules (i), (iii), (iv); (ii) per pair with Bonferroni; (v) replaced (finally, 2026-10-09, by the
    within-ring permutation test after the Mann-Whitney version was found to lack a guaranteed level); eligibility in
    events with π_min = 0.15; the shared-event highlight blank; the boundary band at 0.5 / 1 / 2 σ;
  - SiZer map over 0.05-0.25 P; Silverman's test as a reference reading; cluster-level slab-gap flag 1.0 h.
- **2026-10-14** (implementation; provisional answers of the user's research group): the per-pair tolerance is used
  with the current τ and recomputed when the derived τ lands; the own-width estimate uses the fitted dispersion
  without adding lpz again; a measured precision ratio outside its documented range is used and flagged; only
  Silverman's test is shown (no dip test).

## References

- Annibale P, Vanni S, Scarselli M, Rothlisberger U, Radenovic A (2011). Quantitative photo activated localization
  microscopy: unraveling the effects of photoblinking. *PLOS ONE* 6:e22678. doi:10.1371/journal.pone.0022678
- Bender R, Lange S (2001). Adjusting for multiple testing—when and how? *J Clin Epidemiol* 54:343-349.
  doi:10.1016/S0895-4356(00)00314-0
- Besag J, Diggle PJ (1977). Simple Monte Carlo tests for spatial pattern. *Applied Statistics* 26:327-333.
  doi:10.2307/2346974
- Cameron AC, Miller DL (2015). A practitioner's guide to cluster-robust inference. *J Human Resources* 50:317-372.
  doi:10.3368/jhr.50.2.317
- Chaudhuri P, Marron JS (1999). SiZer for exploration of structures in curves. *JASA* 94:807-823.
  doi:10.1080/01621459.1999.10474186
- Cohen AC (1950). Estimating the mean and variance of normal populations from singly truncated and doubly truncated
  samples. *Ann Math Stat* 21:557-569. doi:10.1214/aoms/1177729751
- Cohen J (1992). A power primer. *Psychological Bulletin* 112:155-159. doi:10.1037/0033-2909.112.1.155
- Cowles M, Davis C (1982). On the origins of the .05 level of statistical significance. *American Psychologist*
  37:553-558. doi:10.1037/0003-066X.37.5.553
- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8:1027-1036. doi:10.1038/nmeth.1768
- D'Este E, Kamin D, Velte C, Göttfert F, Simons M, Hell SW (2016). Subcortical cytoskeleton periodicity throughout
  the nervous system. *Scientific Reports* 6:22741. doi:10.1038/srep22741
- Dunn OJ (1961). Multiple comparisons among means. *JASA* 56:52-64. doi:10.1080/01621459.1961.10482090
- Eisenberger I (1964). Genesis of bimodal distributions. *Technometrics* 6:357-363.
  doi:10.1080/00401706.1964.10490199
- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochem Cell Biol* 141:629-638.
  doi:10.1007/s00418-014-1192-3
- Ester M, Kriegel H-P, Sander J, Xu X (1996). A density-based algorithm for discovering clusters in large spatial
  databases with noise. *Proc. KDD-96*, AAAI Press, pp. 226-231.
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of βII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic
  nerve axons. bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1
- Hartigan JA, Hartigan PM (1985). The dip test of unimodality. *Ann Stat* 13:70-84. doi:10.1214/aos/1176346577
- Hope ACA (1968). A simplified Monte Carlo significance test procedure. *JRSS B* 30:582-598.
  doi:10.1111/j.2517-6161.1968.tb00759.x
- Horvitz DG, Thompson DJ (1952). A generalization of sampling without replacement from a finite universe. *JASA*
  47:663-685. doi:10.1080/01621459.1952.10483446
- Huang B, Wang W, Bates M, Zhuang X (2008). Three-dimensional super-resolution imaging by stochastic optical
  reconstruction microscopy. *Science* 319:810-813. doi:10.1126/science.1153529
- Jensen JLWV (1906). Sur les fonctions convexes et les inégalités entre les valeurs moyennes. *Acta Math* 30:175-193.
  doi:10.1007/BF02418571
- Lachin JM (1981). Introduction to sample size determination and power analysis for clinical trials. *Controlled
  Clinical Trials* 2:93-113. doi:10.1016/0197-2456(81)90001-5
- Liang K-Y, Zeger SL (1986). Longitudinal data analysis using generalized linear models. *Biometrika* 73:13-22.
  doi:10.1093/biomet/73.1.13
- Lotwick HW, Silverman BW (1982). Methods for analysing spatial processes of several types of points. *JRSS B*
  44:406-413. doi:10.1111/j.2517-6161.1982.tb01221.x
- McLachlan GJ (1987). On bootstrapping the likelihood ratio test statistic for the number of components in a normal
  mixture. *Applied Statistics* 36:318-324. doi:10.2307/2347790
- Neyman J, Pearson ES (1933). On the problem of the most efficient tests of statistical hypotheses. *Phil Trans R Soc
  A* 231:289-337. doi:10.1098/rsta.1933.0009
- Parzen E (1962). On estimation of a probability density function and mode. *Ann Math Stat* 33:1065-1076.
  doi:10.1214/aoms/1177704472
- Phipson B, Smyth GK (2010). Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 9:Article 39.
  doi:10.2202/1544-6115.1585
- Schilling MF, Watkins AE, Watkins W (2002). Is human height bimodal? *The American Statistician* 56:223-229.
  doi:10.1198/00031300265
- Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R (2017). Super-resolution microscopy with
  DNA-PAINT. *Nature Protocols* 12:1198-1228. doi:10.1038/nprot.2017.024
- Silverman BW (1981). Using kernel density estimates to investigate multimodality. *JRSS B* 43:97-99.
  doi:10.1111/j.2517-6161.1981.tb01155.x
- Simmons JP, Nelson LD, Simonsohn U (2011). False-positive psychology. *Psychological Science* 22:1359-1366.
  doi:10.1177/0956797611417632
- Simonsohn U, Simmons JP, Nelson LD (2020). Specification curve analysis. *Nature Human Behaviour* 4:1208-1214.
  doi:10.1038/s41562-020-0912-z
- Spearman C (1904). The proof and measurement of association between two things. *Am J Psychol* 15:72-101.
  doi:10.2307/1412159
- Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W (2016). Increasing transparency through a multiverse analysis.
  *Perspectives on Psychological Science* 11:702-712. doi:10.1177/1745691616658637
- Wald A (1943). Tests of statistical hypotheses concerning several parameters when the number of observations is
  large. *Trans Am Math Soc* 54:426-482. doi:10.1090/S0002-9947-1943-0012401-3
- Xu K, Zhong G, Zhuang X (2013). Actin, spectrin, and associated proteins form a periodic cytoskeletal structure in
  axons. *Science* 339:452-456. doi:10.1126/science.1232251
- Zhong G, He J, Zhou R, Lorenzo D, Babcock HP, Bennett V, Zhuang X (2014). Developmental mechanism of the periodic
  membrane skeleton in axons. *eLife* 3:e04581. doi:10.7554/eLife.04581
