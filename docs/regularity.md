# Regularity of the cluster arrangement around the perimeter

> **Resumen (español).** Esta página explica cómo MPS Explorer mide la regularidad del arreglo de los clústeres de
> βII-espectrina alrededor del perímetro del axón, por qué un solo índice no alcanza, y cómo separa «más regular que
> el azar» de «hay un componente periódico». Las medidas (CV de los huecos, ρ₁, CV del exceso, pico de Bragg, g(s))
> se calculan sobre las posiciones de arco de los centroides en la membrana P-spline de 600 nm. La lectura tiene **dos
> etapas**: (1) el CV contra el **nulo de pipeline** (moléculas al azar con contacto, el mismo método de clusterizado,
> la misma curación y el mismo contorno), bilateral; (2) sólo si (1) dice «más regular», ρ₁ y el pico de Bragg contra
> un **nulo de exclusión apareado** (un núcleo duro elegido para que su CV iguale al observado). «Red» exige que las
> dos firmas rechacen en la etapa 2. No hay etiqueta de «paracristal» ni la palabra «vacantes». Todo funciona con
> cualquier método de clusterizado. Los números salen de simulaciones sintéticas (scripts en
> `docs/sims/gap_measures/`) o de la literatura citada; ningún axón real.

This page is the public methods record of the gap measures and of the two-stage reading
(`tools/mps_gap_measures.py`), of their batch helpers (`tools/mps_gap_batch.py`), and of how they use the per-axon
pipeline null ([null_and_scale.md](null_and_scale.md), `tools/mps_pipeline_null.py`). Every number is labelled:
**literature** (with its reference), **derivation** (shown here), **own simulation** (synthetic data only, never a
measured axon; the script and seed are named; the output committed beside the script is the run quoted), **user
decision** (a choice the user made among documented options) or **ad hoc** (a convention of this program). The
research behind the page (simulations sim1-sim7 of the regularity research) was audited independently before
implementation; the audit replaced the reading rule and several wordings (section 8).

## 1. The question, and why one index is not enough

Per axon the program has K cluster centres (K ≈ 20-150) on a closed contour of length P (3-30 µm), and the regularity
index r = median 1NN × K / P. Three different arrangements raise r:

- **Exclusion**: centres never closer than some distance d, otherwise placed at random. Anything with a size does
  this. It is "they do not overlap", not a pattern.
- **Long-range order (a lattice)**: centres near the sites of a grid of spacing a that repeats around the whole
  perimeter. That would be a pattern.
- **Detection**: a clustering method merges clusters whose localization clouds touch, so the closest pairs disappear
  and the remaining spacings look more even.

The measures below separate the first two as far as the data allow; the pipeline null removes the third.

Nothing is published about the circumferential arrangement of spectrin within a ring except the source preprint
(Gazal et al. 2026: discrete clusters, cluster number N = 4.08 P − 12.62 with Pearson r = 0.81 (Fig. 3D legend), a 1NN
peak near 200 nm with a median of 260 nm (Fig. 4B), DBSCAN with ε = 25 nm and minPts = 10, a randomization control with
the minimum separation set to the smallest observed 1NN, and the wording "neither fully regular nor random"). The
authors do not claim a lattice, and the eLife public review objects that a peak in a distribution does not imply a
periodic arrangement. The longitudinal periodicity metrics of the field (Xu et al. 2013; Zhong et al. 2014; Barabas
et al. 2017) answer a different question. The ~39 evenly spaced tetramers per ring of Zhang et al. (2017) are a
modelling assumption, not a measurement.

## 2. The measures

All measures use **arc positions** s_i of the kept centroids projected on the 600 nm P-spline membrane through them
(the same curve the pipeline null places its molecules on; never polar angles, which are invalid on star-shaped
contours) and the K cyclic **gaps** g_i between consecutive centres (Σ g_i = P; mean spacing a = P/K). A closed curve
has no ends, so no edge correction is needed (derivation).

| measure | definition | random (no exclusion) | exclusion (hard core d) | lattice, jitter σ | origin |
|---|---|---|---|---|---|
| gap CV | sd(g)/mean(g) | √((K−1)/(K+1)) ≈ 1 | falls with d/a | ≈ σ√2 / a | derivation (Moran 1947) |
| ρ₁ | circular lag-1 correlation of consecutive gaps | 0 | 0 | **−½** | derivation: g_i = a + ε_{i+1} − ε_i |
| excess-gap CV | sd/mean of (g − min g) | ≈ 1 | ≈ 1 (shifted exponential) | 0.4-0.6 | own simulation (below) |
| S(m) | \|Σ_j exp(i 2π m s_j / P)\|² / K | ≈ 1 | ≈ CV² at low m; **a peak near m ≈ K that grows with d/a** | **Bragg peak** E S(n) = K e^{−q} + 1 − e^{−q}, q = (2πσ/a)² | Bartlett 1963; Salsburg et al. 1953; derivation |
| S*, m*, a* | the largest S(m) for m in [K/2, 3K/2], its mode, a* = P/m* | — | the hard-rod peak (a* ≈ a) | the lattice constant | regularity plan (glossary) |
| S_low | mean S(m) for m = 1..3 | ≈ 1 | ≈ CV² | ≈ 0 | Cox & Lewis 1966 (renewal plateau); ad hoc (3 modes) |
| g(s) | pair correlation along the arc, bins of 25 nm to 2 µm | 1 | 0 below d, damped oscillations | peaks at a, 2a, … | Illian et al. 2008 |
| elongation | per cluster: sd along the membrane tangent / sd along the normal | — | — | — | regularity plan sec. 6.1 |

**Values** (own simulation, `docs/sims/gap_measures/centres_calibration.py`, part 1, seed 1; K = 50, a = 250 nm, 400
patterns per cell; means; centre level, no detection):

| arrangement | CV | ρ₁ | excess CV | S* | S_low | r (arc) |
|---|---|---|---|---|---|---|
| random (Poisson) | 0.98 | −0.02 | 1.00 | 4.4 | 1.03 | 0.35 |
| hard core 80 nm | 0.64 | −0.03 | 0.97 | 3.2 | 0.44 | 0.57 |
| hard core 120 nm | 0.46 | −0.02 | 0.91 | **4.6** | 0.22 | 0.69 |
| hard core 160 nm | 0.29 | −0.01 | 0.82 | **5.7** | 0.09 | 0.79 |
| M8: iid Gamma gaps, σ/a = 0.25 | 0.24 | −0.02 | 0.52 | 5.9 | 0.06 | 0.85 |
| lattice, jitter 30 nm | 0.17 | **−0.50** | 0.45 | **28.7** | 0.00 | 0.89 |
| lattice, jitter 60 nm | 0.33 | **−0.49** | 0.45 | 6.6 | 0.00 | 0.78 |
| lattice, jitter 30 nm, 15 % empty sites | 0.42 | −0.10 | 0.81 | **23.3** | 0.15 | 0.76 |

Identities checked in the same run (`checks` in the JSON): random CV² 0.926 / 0.970 / 0.986 against (K−1)/(K+1) =
0.905 / 0.961 / 0.980 at K = 20 / 50 / 100; lattice ρ₁ −0.48 / −0.50 / −0.50; lattice Bragg height 11.9 / 28.7 / 56.7
against 11.8 / 28.8 / 57.1 from the formula; renewal ρ₁ within 0.07 of 0 (hard core and M8); S_low within 0.03 of CV²
for renewal processes; arc r of random positions 0.36 / 0.35 / 0.35 against ln 2 / 2 = 0.347. They reproduce the
regularity research's sim1 (seed 1, its own code) within Monte Carlo error. *Correction*: an earlier summary table of
the research quoted S* = 3.3 for random positions; its own output (sim1) and this run give 4.4-4.5 — the maximum of
~51 periodogram ordinates of mean 1.

**What the table shows.** CV says *how much* regularity there is but not which kind. ρ₁ ≈ −½ is the signature of a
full lattice and is diluted by empty sites (−0.10 with 15 % empty). The excess-gap CV separates "minimum + exponential
tail" (exclusion, ≈ 1) from bell-shaped gaps (lattice, M8). **The Bragg peak is not specific to a lattice**: a random
arrangement with a strong exclusion has a structure-factor peak near m ≈ K too (S* 4.6-5.7 against 3.2 at d = 80 nm;
the oscillating structure factor of a dense one-dimensional hard-rod fluid, Salsburg, Zwanzig & Kirkwood 1953). A
peak therefore only means "periodic beyond random *with contact*", not "periodic beyond exclusion". Section 3 draws
the consequence.

**Measures not used, and why** (regularity research, audited): Ripley's K/L on the arc (the same information as g(s),
cumulative; it is tested globally by the pipeline null as H₁, [null_and_scale.md](null_and_scale.md)); the number
variance (equivalent to S at low m); Greenwood, Rao, Kuiper, Rayleigh (tests of uniformity, not of the kind of
non-uniformity); Fourier of polar angles (invalid on star-shaped contours); Gibbs-model fitting by pseudo-likelihood
(Besag 1975; Baddeley & Turner 2000; the fitted parameter mixes biology and detection at K = 20-150; postponed by the
user, P-R88 a; the matched hard core d_m of section 3.2 is a minimum-contrast "effective exclusion" instead, Diggle
1979); CV2 (Holt et al. 1996); the τ order metric.

**On the polygon.** The same measures are also computed on the polygon through the centroids (`*_polygon` columns),
**descriptive only, without p**: the polygon passes through the centres it measures and its length grows with their
radial scatter (regularity research, sim4: polygon bias in the CV −0.01 to −0.02 at 15-30 nm of radial scatter, none
on the P-spline membrane). r keeps the polygon for continuity with the published definition.

## 3. The two nulls and the reading rule

### 3.1 Stage 1: the pipeline null ("is the arrangement more regular than random with contact?")

The per-axon pipeline null ([null_and_scale.md](null_and_scale.md)) simulates B axons whose molecules sit at random on
this axon's membrane with a contact hard core d_c = 2 × median r_eff (read; P-R77 a, editable), with blinking,
precision, background and two neighbouring rings, and passes each through **the same clustering method, parameters,
curation and contour** as the observed axon. The gap measures are registered as readouts of that null
(`register_readout`), so their Monte Carlo p come from the same replicates as every other readout (one null, decision
D7). The gap CV is tested **two-sided** (P-R91 a): below the null → stage 2; above → `less_regular_than_random`
(missed clusters thin the arrangement and raise the CV: regularity research, sim2); neither →
`random_with_contact`. B = 199 (user decision P-R78); a two-sided Monte Carlo p is at least 2/(B + 1) = 0.01 (Besag &
Diggle 1977; Phipson & Smyth 2010) — with B = 49 it could not go below 0.04, which is why the calibration below uses
the default B. The p are **approximate**: the null reads K, size, precision and contour off the same axon.

### 3.2 Stage 2: the matched-exclusion null ("a periodic component beyond an exclusion of the same strength?")

For an axon whose CV lies below the pipeline null, a second null is built: sequential inhibition (Diggle, Besag &
Gleaves 1976) with the hard core d_m chosen by bisection so that the null's mean gap CV equals the observed CV. ρ₁ is
tested on its lower tail and S* on its upper tail. A lattice has ρ₁ ≈ −½ and a Bragg peak that no exclusion reproduces
at the same CV; a strongly excluded random arrangement matches its own CV and shows neither. d_m is reported as the
axon's **effective exclusion** (a minimum-contrast estimate, Diggle 1979). Near packing (d ≳ 0.65 a) the null cannot
be drawn; d_m is capped there and the cap is reported (regularity plan sec. 5).

Where stage 2 runs (P-R87 c, both available):

- **pipeline** (exact; for the batch): the stage-1 null re-run with the molecules' hard core raised to d_m — bisection
  over 6 steps of 16 replicates with common random numbers, then B fresh replicates [ad hoc: cost]. It contains the
  clustering method's merging and missing.
- **centres, "approx., centres"** (seconds; the GUI preview): sequential inhibition of the observed K on a circle of
  the membrane's length, bisection over 8 steps of 40 patterns (regularity plan sec. 5), then B patterns. It does not
  contain detection, which moves ρ₁ of a lattice from −0.49 to −0.38 and raises S* by ~0.5 (regularity research,
  sim7). The sampler draws each new point uniformly on the free arcs, which is exactly the law of the "redraw until
  far enough" sampler without its rejections (checked in `test_gap_measures.py`).

### 3.3 The reading rule (pre-written; P-R81 b)

0. K < 8 → `not_testable` (also when the pipeline null is NOT INTERPRETABLE: its p cannot be read). If the smallest
   observed 1NN is below d_c, the row first says "minimum below contact: split clusters or clusters of another ring"
   (a signature no null repairs); the verdict follows, labelled.
1. Gap CV against the pipeline null, two-sided, α = 0.05: below → stage 2; above → `less_regular_than_random`;
   neither → `random_with_contact`.
2. Against the matched-exclusion null: ρ₁ rejects (p ≤ 0.05 **and** ρ₁ ≤ −0.25) **and** S* rejects (p ≤ 0.05) →
   `lattice` ("lattice signature, spacing a* = P/m*"); only ρ₁ → `lattice_rho1_only` ("anticorrelated consecutive gaps:
   a lattice with large jitter, no resolvable peak"); only S* → `periodic_component` ("a periodic component at a*: a
   lattice with empty sites, undetected clusters or residual structure of a strong exclusion — indistinguishable from
   the arrangement alone"); neither → `more_regular_kind_not_identified` (with the excess-gap CV as a descriptive
   pointer: ≈ 1 suggests exclusion, < 1 bell-shaped gaps).

The threshold −0.25 is half the lattice value (regularity audit: a written threshold). α = 0.05 per test without a
Bonferroni correction per axon: the tests answer different questions and the rule is hierarchical (stage 1 is the
gate: under the null, any label other than `random_with_contact` has the CV test's level). `lattice` needs two
rejections. In the batch, Holm across measures per animal (section 5). **No "short-range order" (paracrystal) label**
is issued: the generator has the M8 model (iid Gamma gaps, P-R92 a) but the label was never validated through the
pipeline; M8 arrangements are read `more_regular_kind_not_identified` (section 4).

Why not the simpler hierarchy "CV → ρ₁ → Bragg" against the contact null: applied to synthetic axons with a strong
exclusion (d = 150 nm), the Bragg peak rejects the contact null in 60 % of them (table below; 72 % through the
pipeline in the regularity research, sim3), so that hierarchy would call a strongly excluded random arrangement
"periodic" most of the time.

## 4. Calibration

### 4.1 Centre level (the two-stage rule alone)

Own simulation, `docs/sims/gap_measures/centres_calibration.py`, part 2 (seed 1011; K = 50, a = 250 nm, 150 patterns
per truth; stage 1 against sequential inhibition with the contact hard core 50 nm, a centre-level proxy of the
pipeline null, B = 199, two-sided; stage 2 matched centres, B = 199). Label rates (95 % Wilson intervals in the JSON):

| truth | stage 1 rejects (CV) | Bragg vs contact | Bragg vs matched | ρ₁ vs matched | d_m found | labels |
|---|---|---|---|---|---|---|
| contact 50 nm (the null is true) | 0.06 | 0.02 | — | — | — | random 94 %, less regular 5 %, other 1 % |
| exclusion 120 nm | 0.96 | **0.29** | 0.03 | 0.02 | 121 | kind not identified 91 %, periodic 3 %, lattice 0 % |
| exclusion 150 nm | 1.00 | **0.60** | 0.08 | 0.05 | 149 | kind not identified 91 %, periodic 8 %, lattice 0 % |
| exclusion 170 nm | 1.00 | **0.75** | 0.05 | 0.04 | 162 | kind not identified 93 %, periodic 5 %, lattice 0 % |
| lattice, jitter 30 | 1.00 | 1.00 | 1.00 | 0.99 | 163 (cap) | **lattice 99 %** |
| lattice, jitter 60 | 1.00 | 0.69 | 0.17 | 1.00 | 148 | lattice ρ₁ only 83 %, lattice 17 % |
| lattice, jitter 30, 15 % empty | 0.99 | 1.00 | 0.99 | 0.13 | 132 | periodic 91 %, lattice 9 % |
| lattice, jitter 30, 30 % empty | 0.80 | 1.00 | 0.80 | 0.08 | 108 | periodic 74 %, lattice 6 %, random 20 % |
| M8, σ/a = 0.25 | 1.00 | 0.73 | 0.03 | 0.09 | 162 | kind not identified 91 %, periodic 3 %, ρ₁ only 6 %, lattice 0 % |
| random (Poisson) | 0.39 | 0.35 | — | — | — | random 61 %, less regular 39 % |

Reading: against the matched null the false "periodic" rate on pure exclusion falls from 29-75 % to 3-8 %, while
lattices with empty sites still reject (99 % and 80 %); d_m recovers the true exclusion (121 / 149 / 162 for 120 / 150 /
170); the level of stage 1 is **consistent with 5 %** (9 of 150 axons got a label other than `random_with_contact`:
6 %, 95 % interval about 3-11 %). These reproduce the regularity audit's own check (seed 1011, its own code: Bragg
against the matched null 11 / 5 / 8 % for d = 120 / 150 / 170).

### 4.2 Through the pipeline (the acceptance test)

Own simulation, `docs/sims/gap_measures/pipeline_confusion.py` (seed 2810). Synthetic axons (ellipse of semi-axes
2.0 × 1.4 µm, membrane 10.8 µm, three rings, 4 molecules per µm = 43 per ring [order of Gazal et al. 2026], NeNA 12 nm)
with a known arrangement of molecules go through the localization simulator of the pipeline null (labels, blinking,
precision, background), DBSCAN 25/10, curation and contour; then stage 1 against the pipeline null (B = 199, the
default) and stage 2 against the matched-exclusion null at the pipeline level (5 bisection steps × 16 replicates, then
B = 49) and at the centres level (B = 49). 12 axons per truth; label rates with 95 % Wilson intervals. Two runs:
**with a drawn ROI** (the membrane's box + 500 nm, as a user draws it; `pipeline_confusion_roi.json`) and **without an
ROI** (`pipeline_confusion.json`), where the curation's edge reference is the slab's own outline.

| truth (per ring) | K detected (of 43) | stage 1 rejects | Bragg vs contact | labels with ROI (pipeline stage 2) | labels without ROI |
|---|---|---|---|---|---|
| random, contact 50 nm (the null is true) | 36.3 | 0 / 12 | 1 / 12 | **random 12 / 12** (76-100 %) | random 12 / 12 |
| exclusion 150 nm | 40.3 | 12 / 12 | 6 / 12 | kind not identified 11, **periodic 1 (8 %; 1-35 %), lattice 0** | kind not identified 8, less regular 3, random 1; lattice 0, periodic 0 |
| full lattice, jitter 30 nm | 40.6 | 12 / 12 | 12 / 12 | **lattice 4 (33 %; 14-61 %)**, periodic 8 | lattice 3, periodic 7, random 2 |
| lattice, 15 % empty sites, jitter 30 | 39.8 | 10 / 12 | 12 / 12 | periodic 10 (83 %; 55-95 %), random 2 | periodic 6, random 6 |
| M8, iid Gamma gaps σ/a = 0.25 | 40.5 | 12 / 12 | 4 / 12 | kind not identified 10, lattice ρ₁ only 2 (17 %; 5-45 %) | kind not identified 5, random 5, less regular 2 |

The centre-level stage 2 ("approx., centres") gave the same label in 56 of 60 axons with the ROI; of the four
differences, three are lattices read `lattice` by one stage 2 and `periodic_component` by the other (both ways), and
one excluded axon is `periodic_component` at the pipeline level and `more_regular_kind_not_identified` at the centres.
The axon window runs stage 2 at the centres level and the Batch at the pipeline level (P-R87 c; the default is not
changed), so one axon can carry two different readings depending on where it was run. Since table v15 every row
says which: `regularity_stage2_level` is `centres` (window) or `pipeline` (Batch), whether or not stage 1 asked for
stage 2. Compare or pool readings only at the same level. Stage 1 (the gap CV against the pipeline null, its p and
z) is the same at both levels for the same null (same method, B and seed): only the stage-2 columns
(`gap_rho1_p_mc_matched`, `bragg_p_mc_matched`, `matched_null_*`) and the reading that depends on them can differ.

Reading against the regularity plan's acceptance criteria (PR 2):

- **Met — the false labels**: a strong exclusion is never `lattice` (0 / 24 over both runs; criterion ≤ 5 %) and
  `periodic_component` in 1 / 24 (criterion ≤ 15 %); the true null is `random_with_contact` in 24 / 24 (criterion ≥ 85 %).
  The Bragg peak against the contact null alone would have called half of the excluded axons periodic (6 / 12): the
  blocking finding of the audit, reproduced through the pipeline.
- **Not met — the power for a lattice**: a full lattice is `lattice` in 4 / 12 (criterion ≥ 80 %) and
  `periodic_component` in the other 8; a lattice with 15 % empty sites is `periodic_component` in 10 / 12 (criterion ≥ 90 %
  for periodic + lattice). The cause is detection: through the pipeline the observed ρ₁ of a full lattice is −0.17 on
  average instead of −0.50 at the centres (0-6 of 43 molecules are not detected and every missing site breaks the
  anticorrelation, as 15 % empty sites do at the centres: −0.10), while the Bragg peak survives (S* 20.9 against 28.7).
  At the centre level the same rule labels 99 % of full lattices `lattice` (section 4.1). The per-axon label `lattice` is
  therefore conservative through the pipeline; the claim for a group rests on the animals' z (section 5).
- **Without an ROI** the curation removes runs of edge-touching clusters on the outer rim (up to 11 of 41 in one axon):
  a stretch of missing clusters makes an excluded or M8 arrangement read `less_regular_than_random` (5 / 24) and costs
  power. A drawn ROI with margin avoids it (0 clusters removed in 60 axons); the same failure appears when an ROI is
  drawn tight around the ring ([the bad-cluster criterion](cut_clusters.md) explains the edge rule).
- M8 (iid gamma gaps, a "paracrystal") is read `more_regular_kind_not_identified` (10 / 12) and never `lattice`; 2 / 12
  `lattice_rho1_only` (5-45 %): the reason the "short-range order" label stays out (P-R92 a).
- In 22 of the 60 ROI axons the final matched pipeline null lost one of its 49 replicates to a jammed sampler near
  packing. At the time the pipeline null's strict rule (every replicate asked for must be analysed) marked those
  axons "NOT INTERPRETABLE" in their notes. The user's decision G1 (c) (2026-10-23) replaced it, for stage 1 and
  stage 2 alike: up to 5 % of the replicates may fail (2 of 49, 9 of 199), and each failed replicate is counted as at
  least as extreme as the observed value in the tail(s) of each test, so the p stays conservative instead of being
  computed over the replicates that happened not to jam ([null_and_scale.md](null_and_scale.md), section 6.4).
  `matched_null_failed_replicates` and `null_failed_replicates` say how many failed, and every p has a companion
  `..._ignoring_failed` column (descriptive only). More than 5 % failed is still NOT INTERPRETABLE. The 60-axon
  figures of this section were computed before the change (failures dropped); with one failure in 49 a stage-2 p
  rises by at most 1/50 = 0.02 [derivation].
- The merged-cluster flag under the true null: 25 clusters flagged in 12 axons against 21.8 expected (5 % of the
  measured clusters): calibrated.

Cost (this laptop, one process): 1.8-4.5 min per axon (stage 1 B = 199 ≈ 2 min; the pipeline-level stage 2 ≈ 1-2 min).

### 4.3 Power per axon is limited

At the centre level the regularity research (sim5, seed 5; B = 999) found that the CV detects an exclusion 40 nm
stronger than the null's in 42 / 80 / 95 % of axons with K = 20 / 50 / 100, ρ₁ identifies a full lattice from K = 20
and is diluted by empty sites, and pooling 6 axons brings almost every contrast to 1.00. ρ₁ per axon has a standard
deviation of 0.10-0.16 even for a true lattice (derivation + sim1). **Conclusions about the kind of regularity will
come from axons pooled per animal, not from single axons.**

## 5. The batch (functions for the integration)

`tools/mps_gap_batch.py` (pure functions; the batch window wires them):

- z = (observed − null mean) / null sd per measure and per axon (`gap_cv_z`, `gap_excess_cv_z`, `gap_rho1_z_matched`,
  `bragg_s_z_matched`); never raw values or p across axons.
- **Per animal** (`aggregate_by_animal`): mean z per measure, the one-sample t-test of the axons' z, Holm (1979) across
  measures.
- **Between groups** (`compare_groups`): the animals' mean z, Welch's t with animals as units (Lazic 2010; Aarts et al.
  2014), Holm across measures; descriptive only with fewer than 3 animals in a group.
- **Evidence of a lattice for a group** (`lattice_evidence`; user decision P-R83, both criteria): at least one axon read
  `lattice` **and** at least 2 animals whose mean z of ρ₁ against the matched null is < −2 and whose mean z of S* is
  > +2. Otherwise the sentence says not to write "the clusters form a lattice / a periodic pattern".
- **Reading table** (`readings_table`): per group, the axons in each level and the median a* of the `lattice` and
  `periodic_component` axons.
- **Calibre** (`calibre_model`; P-R89): N ~ P with a random intercept per animal (REML, `tools/mps_constancy.py`), the
  intercept with a 95 % interval on n_animals − 2 df [ad hoc: conservative, as `mps_constancy`'s slope], the mean
  spacing P/N per 5 µm bin of perimeter [ad hoc], and the same fit on the pipeline null's mean detected K
  (N_det(P)); refused with fewer than 3 animals. A lattice of fixed spacing predicts N = P/a, a line through the origin
  (derivation). The published line (Gazal et al. 2026, Fig. 3D) has a negative intercept without a published interval,
  so the wording is conditional: **if** the intercept is really negative, a fixed spacing would need a detection loss
  that depends on calibre, which N_det(P) measures. The regularity research (sim6, seed 6, 500 replicates) found the
  power to reject a zero intercept under the published line 0.67 with 3 animals × 6 axons and 0.92 with 6 × 6 (REML);
  a coverage check of the interval is in `test_gap_measures.py`.
- No global envelope of g(s) in the batch (P-R90 a: cost); the plots draw pointwise bands, labelled so.

## 6. How to write the conclusion (templates; the numbers are INVENTED for illustration)

- *Per axon, typical case:* "At the level of this axon the K = 52 clusters are more regularly spaced than random
  placement with contact, once detection is accounted for (gap CV 0.41 against 0.55 [0.47-0.63] of the pipeline null,
  p ≈ 0.01, B = 199). Against an exclusion of the same strength there is no evidence of a periodic component (Bragg
  p = 0.31; ρ₁ = −0.03): the kind of regularity is not identified with this K; the excess gap is exponential-like
  (excess-gap CV 0.96), compatible with exclusion." *(invented numbers)*
- *Per axon, lattice:* "… against an exclusion of the same strength, consecutive gaps are anticorrelated (ρ₁ = −0.41,
  p = 0.01) and there is a peak at m* = 51 (a* = 245 nm, S* = 19 against 4.2 [2.6-7.1], p = 0.01): lattice signature at
  the level of this axon." *(invented numbers)*
- *Per group:* "In 11 of 18 axons (3 animals) the reading was 'more regular, kind not identified'; in 4 'periodic
  component (a* ≈ 240 nm)'; in 1 'lattice'; in 2 'random with contact'. The mean z of the CV per animal was −2.1, −1.8
  and −2.6." *(invented numbers)*
- *Always:* "p approximate: the null reads K, size, precision and contour off the same axon; the clustering method's
  merging is in the null; the biology of the labelling (epitope density) is not."
- *Never:* "the clusters form a lattice / a periodic pattern" without ρ₁ **and** Bragg against the matched null in
  ≥ 2 animals (P-R83).

The program writes the per-axon sentence itself (`Reading.sentence`), always with "at the level of this axon" and the p.

## 7. What is exported, and the multi-method plug-in

Per axon (regularity plan sec. 6.2; P-R86 names, `_matched` for stage-2 p): `gap_cv` with `_null_mean/_lo/_hi`,
`gap_cv_p_mc`, `gap_cv_z`; `gap_rho1` with its null-1 band and `gap_rho1_p_mc_matched`, `gap_rho1_z_matched`;
`gap_excess_cv` with band and `gap_excess_cv_p_mc`; `bragg_s` with band, `bragg_m`, `bragg_spacing_nm`,
`bragg_p_mc_matched`, `bragg_s_z_matched`; `s_low`; `gap_cv_polygon`, `gap_rho1_polygon`, `gap_excess_cv_polygon`,
`bragg_s_polygon` (descriptive); `regularity_reading`, `regularity_flag_min_below_contact`; `matched_null_kind`
(`pipeline` / `centres_approx` / `not_run`), `matched_null_d_nm`, `matched_null_B`, `matched_null_capped`;
`pipeline_null_B`, `_K_sim`, `_d_contact_nm`, `_epitope_R_nm`, `_lp_nm`, `_K_det_mean`, `_closure_ratio`, `_seed`;
`n_clusters_elongated`, `elongation_threshold` (the median over the null's replicates of their 95th percentile of
elongation), `n_clusters_elongated_null_expected` (5 % of the measured clusters; P-R84 a: merged clusters are marked
and counted, never split).

For "Compare methods…" and the batch, the analysis registers itself with `tools.multimethod` as `gap_measures`
(readouts prefixed `gap_`; p-values `gap_cv_p`, `gap_excess_cv_p`, `gap_rho1_matched_p`, `gap_bragg_matched_p`;
verdicts `gap_regularity_verdict` with the seven levels above, `gap_min_contact_verdict`, `gap_matched_null_verdict`).
It is a pure function of one method's clustered result: the pipeline null re-runs **that** method with its own
parameters and curation, so the readings of DBSCAN, HDBSCAN, ToMATo, … sit side by side and the summary says whether
the conclusion depends on the method. Slow methods are refused unless asked for.

## 8. Limitations

The p are approximate (the null is conditioned on the same axon). The null shares its measurement model with the
simulations that calibrated it; a real feature outside that model (two cluster populations, biological merging) is not
reproduced — the count of elongated clusters against its null's expectation is the visible check. A lattice with empty
sites, a lattice with undetected clusters and the residual structure of a strong exclusion are indistinguishable from
the arrangement alone at K ≈ 50, and `periodic_component` says so. Split or foreign clusters are not repaired by any
null; the minimum-below-contact flag is the safeguard. Single axons have little power for the kind of regularity.
Estimating a biological effective exclusion with Gibbs models and deciding whether merging is biological need other
labelling (DNA-PAINT, an endogenous tag), as the eLife reviewers asked.

## 9. Decision record

- 2026-10-14 — research (synthetic sim1-sim7) and independent audit. The audit's re-analysis showed that the proposed
  hierarchy against the contact null labels a strongly excluded random arrangement "lattice" / "characteristic spacing"
  in 73 % of synthetic axons; the rule was replaced by the two-stage test with a matched-exclusion null, the paracrystal
  label was withdrawn, the detection bias in r is reported as a range (+0.01 to +0.05 at a true hard core of 80 nm, −0.02
  to +0.16 per axon), the level is stated with its n ("consistent with 5 %"), the CV test is two-sided, and a pilot-data
  number was removed from the public text.
- 2026-10-22 — the user approved P-R76 to P-R92 (pipeline null by default, the centres null kept as "paper's rule";
  199 replicates; the two-stage rule; both criteria for a group lattice claim; merged clusters marked and counted;
  stage 2 at the pipeline level for the batch and at the centres level for the GUI, both available; the random-intercept
  calibre model refused below 3 animals; no global g(s) envelope; M8 in the generator only). Implemented in branch
  `feat/gap-measures`.

## The interface batch (2026-11-07): bands, labels, the ROI edge

* **Bands (G2, user decision 2026-11-07).** The regularity plots of the "Perimeter arrangement" tab draw their 95 %
  band from the pipeline null's replicates once the null ran (`keep_arcs`: each replicate's centre arcs on its own
  membrane) -- the same null as stage 2's p. Before it, the band is the centres-level one (the K centres at random
  with the contact hard core 2 × median r_eff), labelled "approx., centres". The legend, the plots' titles and an
  exported image say which null the band comes from.
* **The per-axon label is descriptive (G4).** The type of a group's regularity is read from the per-animal z (P-R83).
  The label of one axon under-reads "lattice": a full lattice (jitter 30 nm) through the pipeline reads lattice in 4
  of 12 simulated axons and periodic in 8 [own sim, the table above]. The window and the Batch summary say so.
* **The ROI edge (G5).** When the edge curation removed clusters, the regularity tab shows "contour stretches removed
  at the ROI edge: regularity may be biased" (from `edge_note`; the column `regularity_edge_note`). The ROI tooltips
  recommend drawing the ROI with a margin of at least 500 nm around the membrane [own sim: 0 clusters removed in 60
  axons with that margin].

## References

- Aarts E, Verhage M, Veenvliet JV, Dolan CV, van der Sluis S (2014). *Nature Neuroscience* 17(4):491-496. doi:10.1038/nn.3648
- Baddeley A, Turner R (2000). *Australian & New Zealand Journal of Statistics* 42(3):283-322. doi:10.1111/1467-842X.00128
- Barabas FM, Masullo LA, Bordenave MD, Giusti SA, Unsain N, Refojo D, Cáceres A, Stefani FD (2017). *Scientific Reports* 7:16029. doi:10.1038/s41598-017-16280-x
- Bartlett MS (1963). *Journal of the Royal Statistical Society B* 25(2):264-296. doi:10.1111/j.2517-6161.1963.tb00508.x
- Besag J (1975). *The Statistician* 24(3):179-195. doi:10.2307/2987782
- Besag J, Diggle PJ (1977). *Applied Statistics* 26(3):327-333. doi:10.2307/2346974
- Cox DR, Lewis PAW (1966). *The Statistical Analysis of Series of Events*. Methuen. doi:10.1007/978-94-011-7801-3
- Diggle PJ (1979). *Biometrics* 35(1):87-101. doi:10.2307/2529938
- Diggle PJ, Besag J, Gleaves JT (1976). *Biometrics* 32(3):659-667. doi:10.2307/2529754
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026). bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1 (with its public reviews)
- Holm S (1979). *Scandinavian Journal of Statistics* 6(2):65-70.
- Holt GR, Softky WR, Koch C, Douglas RJ (1996). *Journal of Neurophysiology* 75(5):1806-1814. doi:10.1152/jn.1996.75.5.1806
- Illian J, Penttinen A, Stoyan H, Stoyan D (2008). *Statistical Analysis and Modelling of Spatial Point Patterns*. Wiley. doi:10.1002/9780470725160
- Lazic SE (2010). *BMC Neuroscience* 11:5. doi:10.1186/1471-2202-11-5
- Moran PAP (1947). *Journal of the Royal Statistical Society B* 9(1):92-98. doi:10.2307/2983572
- Phipson B, Smyth GK (2010). *Statistical Applications in Genetics and Molecular Biology* 9(1):39. doi:10.2202/1544-6115.1585
- Salsburg ZW, Zwanzig RW, Kirkwood JG (1953). *Journal of Chemical Physics* 21(6):1098-1107. doi:10.1063/1.1699116
- Xu K, Zhong G, Zhuang X (2013). *Science* 339(6118):452-456. doi:10.1126/science.1232251
- Zhang Y, Abiraman K, Li H, Pierce DM, Tzingounis AV, Lykotrafitis G (2017). *PLoS Computational Biology* 13(2):e1005407. doi:10.1371/journal.pcbi.1005407
- Zhong G, He J, Zhou R, Lorenzo D, Babcock HP, Bennett V, Zhuang X (2014). *eLife* 3:e04581. doi:10.7554/eLife.04581
