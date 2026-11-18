# The per-axon pipeline null and the working scale

> **Resumen (español).** Un clúster es una desviación respecto del azar a una escala dada, y sólo se puede afirmar
> contra un nulo. En SMLM el nulo correcto pone al azar las **moléculas**, no las localizaciones: cada marca deja una
> nube de decenas de localizaciones, y un nulo de localizaciones al azar no forma ningún clúster, de modo que «hay
> más clústeres que al azar» saldría siempre. El nulo por axón de MPS Explorer simula, para cada axón, B = 199 axones
> (999 bajo pedido) con moléculas al azar sobre su propia membrana (con un núcleo duro de contacto igual a 2·r_eff),
> dos marcas por molécula, c_label = 14 ciclos por marca (provisional, de la literatura; 26 como sensibilidad), las
> localizaciones por ciclo medidas en el archivo, su NeNA, su fondo y dos anillos vecinos, y los pasa por **el mismo
> pipeline**: el mismo método de clusterizado con los mismos parámetros (DBSCAN, HDBSCAN, ToMATo, Voronoi, …), la
> misma losa, la misma curación y el mismo contorno. El nulo se cierra en el número de localizaciones de la losa
> (por defecto) o en el número de clústeres (sensibilidad). Da un p de Monte Carlo por lectura (K/µm, 1NN, r_eff,
> ocupancia, regularidad r, y las que agregue la rama de regularidad) y compara H(r) de Ripley sobre el arco (H₁) y en
> la banda (H₂) con su envolvente **global** por encima de la marca de fusión; la envolvente puntual sólo se dibuja.
> Los p son aproximados y el nulo lleva el rótulo «provisional» hasta calibrar c_label en el buffer propio. Hasta un 5 %
> de réplicas fallidas se tolera (2 de 49, 9 de 199) y cada una cuenta como tan extrema como el observado, así que el
> p es conservador; el p que las ignora va al lado, sólo como dato. Cuando hay menos de 19 réplicas, fallan más del
> 5 %, el cierre no se alcanza o los axones simulados no se parecen al observado, el resultado dice
> **NOT INTERPRETABLE** con sus razones. Los métodos lentos (Emitters) se rechazan con una estimación de costo salvo
> que se los pida explícitamente. Tasa de rechazo bajo el nulo y potencia medidas en sintético: sección 6.

This page explains what the per-axon null of MPS Explorer simulates, what it reads from each axon, how it decides,
and what its measured error rates are. The code is `tools/mps_pipeline_null.py` (the generator, the closure, the
run, the result), `tools/mps_working_scale.py` (the working scale), `tools/spatial_stats.py` (Ripley's K and the
envelopes) and `tools/mps_analysis.py` (the pipeline the null repeats). The resolution marks are explained in
[resolution_limit.md](resolution_limit.md). Every number below is labelled: **literature** (with its reference),
**derivation**, **own simulation** (synthetic data only, never a measured axon; the scripts are in
`docs/sims/pipeline_null/` and their output tables are committed next to them), **read** (measured on the axon
being tested), or **ad hoc**.

## 1. What the page is for

A cluster of localizations is a statement about scale: more neighbours at some distance r than a random arrangement
would give (Ripley 1977; Kiskowski et al. 2009; Nieves et al. 2023). It can only be made against a null, and the null
must contain everything that produces clusters by itself: blinking, localization precision, background and the
clustering algorithm. The per-axon null answers one question for one axon: **what would this pipeline -- this
clustering method with these parameters, this curation, this contour -- report if the molecules sat at random on this
axon's membrane?** Each readout of the axon is then placed among B such answers.

It does **not** answer: how many molecules there are (that needs a calibrated count; the readouts are identifiable
only up to c_label); whether a particular cluster is "real"; whether the arrangement is a lattice (the regularity
plan's two-stage rule, which consumes this null: [regularity.md](regularity.md)); anything about channel 2.

## 2. The working scale

Five marks, recomputed for every file and method (`tools/mps_working_scale.py`; details and the simulation behind
them in [resolution_limit.md](resolution_limit.md)):

| Mark | Value with the reference pipeline | Origin |
|---|---|---|
| ε / min samples | 25 nm / 10 | literature (Gazal et al. 2026) |
| NeNA (lateral precision) | measured per file | literature (Endesfelder et al. 2014) + read |
| near-total fusion | d50 − 20 nm ≈ 60 nm | own simulation (clustering research, Table 8.5) |
| d50 | 2·ε + 2.5·σ_NeNA ≈ 80 nm | own simulation; an empirical fit, not literature |
| axial slab | ±90 nm | literature (Gazal et al. 2026) |

The fusion mark is the null's `r_min`: Ripley radii below it are drawn shaded and never enter the test.

## 3. Why a null of random localizations is wrong in SMLM

Each labelled molecule leaves tens of localizations within a few precisions of its label (c_label switching cycles
of one or a few frames each; Dempsey et al. 2011). If a null scatters *localizations* independently, none of them has
ten neighbours within 25 nm at the densities of an axon's slab, DBSCAN finds no cluster in the null, and "the axon has
more clusters than random" is significant for any biology. Platzer et al. (2020) show that a random distribution of
blinking fluorophores could not be discriminated from a clustered one by Ripley's K; Nieves et al. (2023, scenario 1)
note that molecules at random with blinking "would look clustered"; Schneider & Schütz (2022) count 22 DBSCAN clusters
in CSR without blinking and 116 with realistic blinking; Veatch et al. (2012) decompose the apparent clustering
into over-counting and true structure. The null therefore randomises what biology could randomise -- the
**molecules** -- and keeps what the instrument and the algorithm do anyway.

## 4. The per-axon null with blinking

### 4.1 What one simulated axon is

1. **Molecules**: positions of arc uniform on the axon's own membrane (`NullModel.geometry = "ring"`), one after
   the other with a **contact hard core** d_c along the arc (sequential inhibition), each displaced along the
   outward normal by N(0, σ_radial). Two neighbouring rings at ± the axial period carry molecules at the same
   density, so that the axial leak into the ±90 nm slab is in the null too.
2. **Labels**: Binomial(2, p_label) per molecule, each at a uniform point of the epitope disc (radius R, in the
   membrane plane: tangent × axial) plus a 3D linkage offset N(0, σ_link) drawn **once per label**.
3. **Cycles**: c_label switching cycles per label (geometric with mean c_label; Poisson on request), **all at the
   label's position**: the cycles of one label share its linkage. This is the step that makes a molecule a cloud
   (`test_cycles_of_one_label_share_its_position` checks it: the localizations of one label scatter by the NeNA only).
4. **Localizations**: per cycle, a geometric number (mean loc/cycle) in consecutive frames from a uniform start, with
   lateral error N(0, NeNA) and axial error N(0, σ_z).
5. **Background**: uniform in the region (the drawn ROI, or the membrane's box with a 500 nm margin when there is
   none) and in z over the three rings, at the density read on the axon.
6. **The same pipeline**: `analyze_axon` with the observed axon's method and parameters, slab half-width, DBCV
   threshold, ROI (edge curation), occupancy settings and contour mode (2-opt from every start or one start), the slab
   centred on the middle ring. The randomization control of SCI-3 is off in the replicates (it is the paper's
   separate centres null, section 8).
7. **Readouts** of the simulated axon, computed by the same functions as for the observed one.

### 4.2 What is read from the axon, and what is fixed

| Input | Default | Origin |
|---|---|---|
| Membrane | 600 nm P-spline through the observed centroids (`tools.mps_membrane.radial_scatter_nm`) | read; regularity plan P-R80 (a) |
| σ_radial of the molecules | the centroids' scatter about that membrane minus the centroid noise of one simulated molecule, in quadrature | read + own simulation (`cloud_statistics`) |
| Hard core d_c | 2 × median r_eff (contact), editable | read; regularity plan P-R77 (a) + (e) |
| NeNA (lateral error) | the file's NeNA; 12 nm when none is given, labelled | read / literature (Gazal et al. 2026, Methods) |
| σ_z | the slab ring's GMM component sd minus the label offsets in quadrature | read + derivation |
| Axial period | the axon's mean ΔZ; 170 nm when not measured | read / literature (Gazal et al. 2026) |
| loc/cycle | mean localizations per linked event (radius 3·NeNA, 1 dark frame, Picasso's rule; `tools.mps_paint.link_localizations`) on the kept clusters, corrected for the links the radius breaks (below); 2 without frames | read + derivation; regularity plan P-R79 (b) with (a) meanwhile |
| c_label | **14**; sensitivity 26 | literature (Dempsey et al. 2011, Table 1, Alexa Fluor 647 in MEA / βME), provisional until measured in the lab's buffer (plan D1) |
| Labels per molecule, p_label | 2, 1 | ad hoc (two epitopes per tetramer, every one labelled) |
| σ_link | 12 nm | ad hoc (the simulator's value, of the order of Früh et al. 2021) |
| Epitope disc radius R | 25 nm | ad hoc (the simulator's value); **not read**, see 4.5 |
| Background density | slab localizations beyond max(50 + 3·NeNA, 200) nm of the membrane AND beyond 100 nm of every kept centroid, per µm² of the area meeting the same conditions inside the region and the slab's convex hull (10 nm grid) | read; A9 section 3.1, made robust to the method (6.2); the 100 nm discs are the simulator's, the 200 nm floor ad hoc (above the worst centroid-membrane error seen, 133 nm) |
| Frames | last frame + 1; 60 000 without frames | read / ad hoc |
| Molecules per ring | solved by the closure (4.3) | — |

The label of every value travels with the result (`NullModel.origins`).

**The loc/cycle correction (derivation).** Two localizations of one emitter differ by a vector of sd √2·σ per axis,
so their distance exceeds a link radius ρ with probability q = exp(−ρ²/4σ²) = 0.105 at ρ = 3σ. An event of true mean
length μ is then cut into pieces of mean m = μ / (1 + q(μ − 1)), hence μ = m(1 − q)/(1 − mq). Uncorrected, synthetic
axons with a true 2.0 read 1.80; the null then needed 11 % more molecules to match the slab's localizations,
produced too many clusters, and rejected K per µm in 14 of 60 axons under a true null, 12 of them on the low side
(own simulation, `calibration_null_run1_lpc_uncorrected.json`). Corrected, they read 1.99 ± 0.04.

### 4.3 Closure: what the null is matched to

**On the slab's localizations (default).** Every replicate is **conditioned** on the observed number of slab
localizations: it draws a pool of molecules per ring (sequential inhibition, so any first n of the pool is itself a
hard-core sample) and keeps the first n per ring whose slab localizations, background included, come closest to the
observed count. 4 pilots of 10 simulated axons each (no clustering; cheap) first estimate the mean number of
molecules, which sizes the pool (1.6× + 6, never more than the hard core can pack). K per µm is then a free readout
of the null, and its p is meaningful -- with the label "provisional (c_label)", because 14 or 26 cycles change the
molecules by 1.9×.

Why per replicate and not one molecule count for all replicates (the first implementation): with geometric cycles
the localizations one molecule leaves are heavy-tailed, so the slab count of an axon varies by ~10-15 % at a fixed
number of molecules. A single count solved from the observed slab inherits that error, and every replicate carries
it: under a true null the closure on a fixed count rejected K per µm in 16 of 60 synthetic axons and the occupancy in
15 of 60, half of them on each side (own simulation, `calibration_null_run2_closure_on_mean.json`). Conditioning
each replicate on the observed count is the Monte Carlo version of conditioning the test on N_loc, which is what
"closure on N_loc" means. **On the detected clusters
(sensitivity).** N ← N·(K_obs / K_sim)^0.8 over 4 pilots of 2 full pipeline runs [ad hoc damping: K is a compressive
function of N because of fusion]; it needs no c_label but hides the fusion, and K per µm is then conditioned: its p
is left empty. The result says which closure was used (`null_closure`) and how close it came (`closure_ratio`,
reached within 5 %).

When the file holds far more localizations per cluster than labels × c_label × loc/cycle predicts, the closure on
localizations needs so many molecules that the contact hard core cannot be honoured (random sequential packing jams
near 75 % of the ring: Rényi 1958); the null then refuses with that explanation and suggests the closure on clusters.

**Both closures in the Batch (user decisions C3 / C4, 2026-10-23; headless part, 2026-10-29).** The Batch reads the
settings key `batch_null_closure`: `n_locs` (default), `k` or `both` (no selector in the window yet). With `both` it
runs one null per closure on each axon, with the same seed, so the two differ only by the closure; the axon table's
row carries the `n_locs` null (the default) and `<table>_null_by_closure.csv`, beside it and never overwritten, holds
one row per axon × closure (column `null_closure`, the same null, failure and regularity columns for both, and
`null_status`). Under the K closure `p_mc_k_per_um` stays empty (K is conditioned). The extra cost, measured on two
synthetic axons (own simulation, 1,095 and 1,647 slab localizations, DBSCAN 25/10, B = 39, this laptop, one
process): the N_loc null took 18.8 s and 21.2 s, the K null 25.0 s and 25.8 s (+33 % and +22 %: its 4 × 4 pilots run
the whole pipeline, the N_loc pilots only simulate), so `both` costs 2.2-2.3× the default; at B = 199 the pilots
weigh less and `both` is about 2×. When stage 1 of the regularity says "more regular", stage 2 at the pipeline level
runs once per closure too. The Batch's estimate before Run now counts the closure(s), the K pilots and that stage 2
(`tools.mps_batch.null_cost_per_file`); before, it counted B + 8 replicates and no stage 2. On those two axons the
K closure was NOT INTERPRETABLE with 4 pilots of 4 axons (the known C3 caveat: the closure on K misses its 5 %
mark on a share of axons).

### 4.4 Method-agnostic by construction

The null receives the method as a name and its parameters (or a `ClusteringStrategy` template, deep-copied for each
replicate) together with the curation settings, and re-runs exactly that pipeline on every simulated axon
(`PipelineSpec.from_analysis` reads them back from the observed `AxonAnalysis`). The same code is therefore the null of
DBSCAN, HDBSCAN, FOCAL, ToMATo, GraphCommunity, the Voronoi methods, and any method added later; a null of one method
is never used for another (the observed axon must have been analysed with the method being tested).
`tools/multimethod.py` (the multi-method runner) calls it through the `pipeline_null` plug-in, once per method.

Two exceptions: **External labels** is always refused (its labels come from a file computed on the observed
localizations; a simulated axon has none). **Slow methods** (`tools.cluster_method_ui.is_slow`: Emitters, OPTICS,
voronoi_csr) are refused unless the caller passes `allow_slow=True`, with the expected duration in the message, so a
199-replicate null never starts an hours-long run silently. The measured cost per replicate is in section 6.1.

### 4.5 Two estimates the null does not make (deviations from the plans, with the evidence)

- **The epitope radius is not read from the clusters.** The regularity plan (section 4.4) read R from the clusters'
  median tangential spread. Under the blinking model the spread hardly depends on R: for isolated simulated molecules
  (NeNA 12 nm, σ_link 12 nm, c_label 14) the median tangential sd is 13.0 / 13.8 / 15.0 nm at R = 0 / 25 / 40 nm
  (own simulation, `cloud_statistics`), because one label usually dominates a molecule's localizations, while merged
  clusters add more than that: a synthetic axon with R = 25 nm read 54 nm. R stays at the simulator's 25 nm (ad hoc),
  editable, and the observed spread is reported as a diagnostic.
- **K is not a fidelity check under the closure on localizations.** A9 section 3.4 listed "K per ring within 15 %"
  among the fidelity checks. Under the closure on localizations K per µm is one of the readouts being tested;
  requiring it to match would declare NOT INTERPRETABLE exactly the axons where it differs. K stays a fidelity check
  under the closure on K.

## 5. Ripley's K on the arc and in the slab

The points are the **centroids of the curated clusters** -- the objects every readout is computed on (A9 section 2.1).

**H₁ along the arc.** The centroids are projected on the 600 nm membrane through them (arc positions s₁ … s_n on a
closed curve of length L); with the cyclic distance d(i, j) = min(|sᵢ − sⱼ|, L − |sᵢ − sⱼ|),

    K₁(r) = L / (n (n − 1)) · Σ_{i≠j} 1[d(i, j) ≤ r],    E[K₁(r)] = 2r under CSR on a closed curve,    H₁ = K₁ − 2r

(derivation: a closed curve has no boundary and every point has 2r of curve within r). Each simulated axon is
projected on ITS OWN membrane, built the same way.

**H₂ in the band.** The same centroids in the plane, in the ±50 nm band around the membrane (Gazal et al. 2026), of
area A = 2·50 nm·L (the tube formula; exact while 50 nm is below the smallest radius of curvature, derivation):

    K₂(r) = A / (n (n − 1)) · Σ_{i≠j} 1[‖xᵢ − xⱼ‖ ≤ r],    H₂(r) = √(K₂(r)/π) − r      (Kiskowski et al. 2009)

**No edge correction, and the null in the same window.** The band is 100 nm wide against radii up to 400 nm: Ripley's
isotropic weights exceed 10 for most pairs beyond 100 nm and the translation correction is unstable in a thin,
non-convex window. The test is a Monte Carlo test and every replicate is computed in its own band exactly as the
observed, so the edge bias is in both and cancels in the comparison (Besag & Diggle 1977; Baddeley et al. 2015). H₂
is therefore compared **with the null, never with 0**.

**What their disagreement means.** Same sign and r\* → the organisation is along the perimeter (H₁ has one degree of
freedom and less noise). H₂ shows aggregation below ~100 nm and H₁ does not → centres stacked across the band (another
ring leaking into the slab, or split clusters). H₁ shows structure and H₂ does not → the projection may be inventing
order (a badly traced contour, centres outside the band). Neither departs above the fusion mark → consistent with the
null at every working scale. Both are always reported.

**Radii.** 20, 30, …, 400 nm (39 values; ad hoc, decided 2026-10-22: below the fusion mark at the bottom, about two
mean spacings at the top), the upper limit editable (`r_max_nm`).

**Axons without a ring (2D fallback, A9 section 3.5).** With `no_ring=True`, or when the analysis has no contour,
the molecules are placed uniformly in the ROI (or the slab's convex hull), H₁, occupancy and regularity are not
tested, and H₂ uses the ROI as its window (`ripley2d_window = "roi"`, correction "none").

## 6. Envelopes, p-values and the measured error rates

**Pointwise envelope** (2.5-97.5 % of the replicates at each r): **drawn only**, as a shade. With 39 radii looked at
together, "leaving the band somewhere" happens under the null far more often than 5 % (measured below).

**Global envelope and p (the decision).** One statistic per curve, the maximum studentised deviation over the radii
at or above the fusion mark, T = max_r |H(r) − median(H)| / sd(H), with the median and sd of the B + 1 curves
(observed included, so that the B + 1 values are exchangeable under the null), and
p_global = (1 + #{T_b ≥ T_obs}) / (B + 1): the maximum-absolute-deviation (MAD) test (Besag & Diggle 1977; Baddeley
et al. 2014; `tools.spatial_stats.envelope_from_curves`). r\* is the radius of the largest studentised deviation of
the observed curve and h_max = H_obs(r\*) − median, signed (+ aggregation, − inhibition). r\* is "the scale at which
this axon departs most from the null", not a domain radius (Kiskowski et al. 2009). The extreme-rank envelope of
Myllymäki et al. (2017) is computed as well.

**Monte Carlo p per readout** (Besag & Diggle 1977; `tools.mps_randomization.monte_carlo_p`): two-sided,
p = min(1, 2·min(p_high, p_low)) with p_high = (1 + #{Y_b ≥ Y_obs}) / (B + 1). With B = 199 the smallest p is 0.005.

**The p are approximate.** The null reads the contour, density and precision off the same axon it tests (Dao &
Genton 2014, conservative in their examples). Section 6.2 measures how far that matters here.

### 6.1 Cost per replicate, per method (own simulation)

One synthetic axon (perimeter ~10.8 µm, 7 402 localizations, 2 446 in the slab), one process, 2 replicates timed
per method (generate + the whole `analyze_axon` with that method + Ripley curves), on the project's laptop
(`docs/sims/pipeline_null/cost_per_method.py`, seed 20261022; `cost_per_method.json`, Emitters in
`cost_per_method_emitters.json`, one replicate timed). The table estimate is the
first guess shown before a run (the Compare-methods cost table, clustering + pipeline per 1000 slab localizations).

| method | slow? | K observed | s per replicate | 199 replicates | 999 replicates | table estimate (s) |
|---|---|---|---|---|---|---|
| DBSCAN | no | 35 | 0.73 | 2.4 min | 12 min | 0.74 |
| HDBSCAN | no | 40 | 0.78 | 2.6 min | 13 min | 0.76 |
| HDBSCAN (scikit-learn) | no | 38 | 0.90 | 3.0 min | 15 min | 0.76 |
| FOCAL | no | 40 | 0.68 | 2.2 min | 11 min | 0.74 |
| ToMATo | no | 51 | 1.02 | 3.4 min | 17 min | 0.81 |
| GraphCommunity | no | 39 | 0.78 | 2.6 min | 13 min | 0.88 |
| voronoi_factor | no | 31 | 0.86 | 2.8 min | 14 min | 0.86 |
| OPTICS | **yes** | 43 | 2.5 | 8.3 min | 42 min | 2.9 |
| voronoi_csr | **yes** | 35 | 9.2 | 30 min | 2.5 h | 23 |
| Emitters (BaGoL-like, 1 worker, buffer MEA) | **yes** | 92 | 35 | 1.9 h | 9.7 h | 47 |

The cost grows with the slab's localizations (DBSCAN, curation and occupancy roughly linearly) and with the number
of clusters (2-opt from every start, about K³; docs/perimeter_time.md). A real axon with ten times the slab
localizations of this one should be expected to cost ten times more per replicate or more; the run measures its own
figure (`seconds_per_replicate`) and `measure_cost` probes it before a long run.

### 6.2 Rejection rate under the null (own simulation)

60 synthetic axons drawn from a **true** null (molecules at random with a contact hard core of 50 nm, 4 per µm,
perimeter ~10.8 µm, NeNA 12 nm, c_label 14, 2 loc/cycle, background 2 per µm²; median K = 34 detected clusters),
each tested with B = 99 replicates of the null read off it, DBSCAN 25/10, closure on the slab's localizations
(script `docs/sims/pipeline_null/calibration_null.py`, seeds 20261022 + axon; table `calibration_null.json`, rows
`calibration_null.jsonl`; code of commit e05eeee). A9's acceptance criterion is ≤ 7 rejections of 60 for the
global 1D test and for each readout (P(fail | true level 5 %) = 0.03).

| statistic | rejections at 0.05 | rate | Wilson 95 % |
|---|---|---|---|
| H₁ global (MAD) | 7 / 60 | 0.117 | 0.058-0.222 |
| H₂ global (MAD) | 5 / 60 | 0.083 | 0.036-0.181 |
| K per µm | 7 / 60 | 0.117 | 0.058-0.222 |
| median 1NN | 2 / 60 | 0.033 | 0.009-0.114 |
| median r_eff | 5 / 60 | 0.083 | 0.036-0.181 |
| occupancy | 4 / 60 | 0.067 | 0.026-0.159 |
| regularity r | 1 / 60 | 0.017 | 0.003-0.089 |
| H₁ **pointwise** band left somewhere above the fusion mark | 31 / 60 | 0.52 | 0.39-0.64 |
| H₂ **pointwise** band left somewhere above the fusion mark | 29 / 60 | 0.48 | 0.36-0.61 |

Reading. Every statistic meets A9's criterion; H₁ and K per µm sit exactly at its limit, on the liberal side (their
intervals include 5 % but are centred near 12 %). The pointwise band, read as a test, would call half of these
structureless axons "clustered somewhere": it is drawn, never used to decide. 58 of 60 results were interpretable
(2 failed the radial-scatter check); the closure matched the slab's localizations within 0.3 %; the read inputs were
close to the truth (molecules per ring 43.4 ± 4.8 for 43.2, loc/cycle 1.99 ± 0.04 for 2, background 2.31 ± 0.46 for
2), except the hard core (below); 0.63 s per replicate (median).

**Where the H₁ excess comes from (diagnostic).** Four of the seven H₁ rejections have r\* at 60-80 nm with the
observed curve ABOVE the null: more close pairs than the null makes. The null's hard core is read as 2 × median r_eff
(P-R77 a): 55.8 ± 2.9 nm on these axons against a true contact of 50 nm, so the null forbids pairs the truth allows.
The same 60 axons with the hard core set to the truth (B = 49, background estimator of the previous version,
`calibration_null_hardcore50_background_v1.json`, against `calibration_null_background_v1.json` with the read hard
core): H₁ 4 / 60 instead of 7 / 60, H₂ 7 / 60 instead of 8 / 60, K per µm 4 / 60 instead of 7 / 60. The read hard core
is conservative for "more regular than random" (as the regularity plan says) but slightly liberal for "aggregated
at short range"; open question in the decision record.

**The method-agnostic null with HDBSCAN** (30 axons from the same truth, analysed and nulled with HDBSCAN
min_cluster_size 10 / min_samples 5, B = 49; `calibration_null_HDBSCAN.json`): H₁ 6 / 30, H₂ 5 / 30, K per µm 4 / 30, 1NN 3 / 30, r_eff 2 / 30,
occupancy 4 / 30, regularity 3 / 30 (A9's limit scaled to 30 axons: 4); pointwise band 15 / 30. The readouts meet the
criterion; the global tests are liberal (20 % and 17 %), with the hard core read even higher than with DBSCAN
(59.4 ± 4.1 nm against the true 50: HDBSCAN's clusters are larger). Background read 2.44 ± 0.57 per µm², 28 of 30
interpretable, 0.65 s per replicate.

**Earlier runs, kept as evidence** (same axons, same seeds): `calibration_null_run1_lpc_uncorrected.json`
(loc/cycle read without the link correction: K per µm 14 / 60, 1NN 13 / 60), `calibration_null_run2_closure_on_mean.json`
(one molecule count for all replicates: K per µm 16 / 60, occupancy 15 / 60) and the `*_background_v1.json` files
(background read only beyond ±(50 + 3·NeNA) nm of the centroid membrane: with HDBSCAN it read 11.6 ± 15.4 per µm²
for a true 2, and the HDBSCAN null rejected the occupancy in 10 of 30 axons) and `*_background_v2.json` (100 nm discs around the KEPT centroids only: HDBSCAN still read 11.0 ± 19.1, from the clusters its curation removed, and rejected the occupancy in 9 of 30). The causes are fixed in the code
(sections 4.2, 4.3 and the background row of 4.2).

### 6.3 Power on planted structure (own simulation)

20 synthetic axons per scenario, same generator and settings as 6.2 but with the molecules PLANTED (all three rings),
B = 99 (`calibration_pairs.json`, `calibration_lattice.json`; commit e05eeee). A9 asked for these rates to be
reported, not required.

| truth | H₁ global | H₂ global | K per µm | 1NN | r_eff | occupancy | regularity r |
|---|---|---|---|---|---|---|---|
| pairs of molecules 150 nm apart (pair centres ≥ 200 nm) | 3 / 20 | 4 / 20 | 1 / 20 | 0 / 20 | 1 / 20 | 1 / 20 | 1 / 20 |
| lattice of 190 nm, jitter 30 nm | 3 / 20 | 13 / 20 | 5 / 20 | 0 / 20 | 0 / 20 | 1 / 20 | **15 / 20** |

Reading. At K ≈ 37-51 per axon the global tests have little power against these two structures; the sign of the
deviation is right most of the time (h_max of H₁ positive in 15 of 20 pair axons, negative in 17 of 20 lattice
axons) but the maximum over 35 radii must clear a high bar. The scale is located only roughly: r\* within ±30 nm of
the planted 150 nm in 1 of 3 H₁ and 1 of 4 H₂ rejections, and of 190 nm in 0 of 3 H₁ and 1 of 13 H₂ rejections (H₂
puts r\* at 100-160 nm on the lattice: the first trough, not the period). The regularity index detects the lattice
in 15 of 20 axons. Conclusions about the type of arrangement are expected to come from the batch over axons and
animals (the regularity plan's section 4.4 reached the same conclusion), not from one axon's global test.

**Closure on the detected clusters (sensitivity; 20 axons, B = 19, `calibration_null_closure_k_clusters.json`).** The
null's median K matched the observed within 4.3 % on average (A9 test 5: ≤ 5 %), but with 2 axons per pilot the
median missed the 5 % mark in 9 of 20 axons (NOT INTERPRETABLE by "closure not reached"), and the slab localizations
missed the 15 % fidelity check in 6 of 20: closed on K, the null leaves N_loc free. No readout rejected at 0.05;
H₁ 2 / 20, H₂ 0 / 20.

### 6.4 NOT INTERPRETABLE

The result says NOT INTERPRETABLE above its p values, with its reasons, when (A9 section 4.4; the same criteria and
wording as the simulated null of the column analysis):

- fewer than 19 simulated axons were analysed (the p can never reach 0.05; derivation: 1/(19 + 1));
- the closure was not reached (the null's median of the closed quantity more than 5 % from the observed);
- the simulated axons do not resemble the observed one: slab localizations not within 15 %, the centroids' radial
  scatter not within max(15 %, 3 nm) (the 3 nm floor is ad hoc), K not within 15 % under the closure on K;
- more than 5 % of the requested replicates failed (said, with the first reason; see below).

An axon with fewer than 8 curated clusters is refused outright (regularity plan, rule 1). The numbers are kept with
`null_interpretable = false`.

**Failed replicates (user decision G1 (c), 2026-10-23).** A replicate fails when the hard-core sampler jams near
packing or the pipeline raises on the simulated axon. Up to floor(0.05 B) failures are tolerated -- 2 of 49, 9 of
199, none of 19 [derivation] -- and every one of them is counted **in the worst case**: as a replicate at least as
extreme as the observed value, in the tail (one-sided) or in both tails (two-sided) of each test. For a readout with
b of the B_ok analysed replicates at least as extreme and f failures,

    one-sided:  p = (1 + b + f) / (B_ok + f + 1)        two-sided:  2 min(p_low, p_high), each tail with + f

and for the global H₁ / H₂ tests p = (1 + b + f) / (n + f + 1) over the n replicate curves. So a failure can only
raise a p: the p read is conservative whatever the failed replicates would have given. Before this decision the
failures were dropped from the p (stage 1 refused the whole null for a single one; stage 2 tolerated 5 % by
dropping them), which made the p of an axon depend on which replicates happened to jam -- typically the most
regular ones, near packing [simulation: docs/regularity.md, 22 of 60 synthetic axons lost one of 49 replicates of
the matched null to a jammed sampler]. The export carries `null_failed_replicates` (and
`matched_null_failed_replicates` for stage 2) and, beside every p, the same p computed without the failures
(`p_mc_<readout>_ignoring_failed`, `ripley1d_p_global_ignoring_failed`, `gap_cv_p_mc_ignoring_failed`, ...): a
descriptive number, never the one to read. Above 5 % the null is NOT INTERPRETABLE, as before. A readout that is not
computable on an analysed replicate (fewer than three centres, for instance) is still left out of that readout's
p: the replicate itself did not fail.

## 7. What the synthetic axon shows and does not show

The result keeps one replicate (the first) with its slab localizations, labels, centroids, contour and its seed key,
for the "Show synthetic axon" view of the integration: the same contour, scale and colours as the observed axon. It
shows what blinking plus the clustering make of molecules at random **on this axon** -- often clusters that look
real. It is a visual control, not a test: one replicate says nothing about the p.

## 8. Relation to the randomization control (SCI-3) and to the regularity plan

- **SCI-3's randomization control** (`tools/mps_randomization.py`, [randomization_and_ks.md](randomization_and_ks.md))
  takes the *detected* clusters and moves them rigidly to random positions in the band, with the smallest observed 1NN
  as hard core: given that there are K clusters like these, are their positions random? It cannot say whether the
  clusters themselves are a blinking artefact. It stays as it is ("paper's rule").
- **This null** randomises the molecules and runs everything after them: it answers the question SCI-3 cannot, and
  it is the regularity plan's "pipeline null" (decision D7: one null shared by the cluster readouts and the regularity
  readouts). The regularity measures (gap CV, ρ₁, excess CV, Bragg peak) plug into it with `register_readout`, and
  get their Monte Carlo p from the same replicates.

## 9. Limitations

- **Identifiability.** The readouts depend on how many localizations a molecule leaves; with c_label from the
  literature the null is "provisional" (`null_type`), and K per µm moves with c_label (the column with 26 is the
  sensitivity).
- **The p are approximate** (section 6).
- **The model.** No dye-per-antibody distribution, no multiple secondaries, no overlapping emitters, a uniform
  background, an epitope disc of a fixed radius; a feature of real data outside the model (two populations of
  molecule size, biological fusion) is not in the null.
- **Thin slab and 3 rings.** The axial leak is modelled with a Gaussian axial error of one width.
- **Hand-made steps are not reproduced**: a contour ordered by hand, the axoplasm discard (widefield images) and the
  pick-edge exclusion; the result warns when the observed analysis used them.
- **Axons without a ring**: only H₂ in the ROI.
- **Nothing about channel 2.**

## 10. Decision record

- 2026-10-16: A9 drafted (null per axon with blinking, working scale; absorbs A7 and the regularity plan's PR 1).
- 2026-10-22: the user approves A9 and the regularity plan with their recommendations: closure on the slab's
  localizations with K as sensitivity; c_label 14 provisional + 26 as sensitivity; decision by the global (MAD) p,
  the pointwise envelope only as a shade; r from 20 to 400 nm in steps of 10, upper limit editable; A5 merged into
  A9; 199 replicates by default and 999 on demand (P-R78 unified at 199); contact hard core 2·r_eff read and editable
  (P-R77 a + e); loc/cycle measured by event linking, 2 meanwhile (P-R79 b + a); membrane P-spline 600 nm (P-R80 a);
  one pipeline null for clusters and regularity (D7). Central request: the null must be method-agnostic.
- 2026-10-22 (implementation, `feat/pipeline-null`), deviations awaiting approval: R not read (4.5); K not a
  fidelity check under the closure on localizations (4.5); cycles geometric by default (A9 wrote "Poisson, or better
  the buffer's geometric distribution": the clustering research's generator, whose d50 and fusion marks the working
  scale uses, is geometric); the radial-scatter check with a 3 nm floor; replicate seeds from
  `SeedSequence(seed, spawn_key=(0, b))` instead of `seed + b` (so that the runs of seed 0 and seed 1 do not share
  replicates); the closure on N_loc applied replicate by replicate (4.3, with the evidence of run 2); loc/cycle
  corrected for the links the 3·NeNA radius breaks (4.2, run 1); for methods other than DBSCAN the working-scale
  marks use the reference ε 25 nm, not the method's edge margin (resolution_limit.md, section 3).
- 2026-10-23 (user decision G1 c), implemented 2026-10-29 (`fix/core-batch-tanda1`): up to 5 % failed replicates
  tolerated in stage 1 and stage 2 alike, counted in the worst case (6.4); `null_failed_replicates` and the
  `*_ignoring_failed` p values exported (axon table v15).
- Open (for the user): the read hard core 2·r_eff is ~6 nm above the true contact on the synthetic axons and makes
  the global tests slightly liberal at 60-80 nm (6.2). Options: keep it (approved P-R77 a, conservative for
  regularity); subtract a bias read from the null itself; or start the global test at d50 (~80 nm) instead of the
  fusion mark.

## The membrane and the closures in the window (2026-11-07)

* **Membrane (user decision C2).** Every method's null puts its molecules on ONE membrane: by default the 600 nm
  P-spline through the centroids of DBSCAN 25/10 (the reference of every comparison [lit: Gazal et al. 2026]); the
  DBSCAN 25/10 run's own null is exactly the null of before. "Membrane: localizations" in the "Perimeter arrangement"
  tab fits it to the slab's localizations instead (robust periodic P-spline, `tools.mps_membrane`), to compare where
  the centroids deform what the localizations show. The tab shows how far apart the two membranes are -- mean and
  largest distance, and a warning for each 600 nm stretch [derivation: the knot spacing, the shortest piece the curve
  can bend on] where they are more than 50 nm apart [ad hoc: the H2 band's half-width; a membrane displaced by more
  moves the simulated molecules out of the band H2 counts them in]. The radial scatter of the molecules stays the
  run's own. The axon table carries `null_membrane` and the indicator (v16). Code: `tools/null_membrane.py`.
* **Closures (C3 / C4).** The null panel offers "Closure: N_loc (default) / K / both side by side". "both" runs the
  two nulls with the same seed (about twice the time, said in the estimate; K's pilots run the whole pipeline) and
  "Compare closures..." puts their p values (H1, H2, the per-axon readouts, the gap CV) and H1 / H2 envelopes side by
  side, marking the tests whose verdict at 0.05 changes with the closure. "Export curves..." writes
  `<stem>_null_by_closure.csv` (one row per method and closure, `null_closure`) and `<stem>_closures_compared.csv`.
  N_loc stays the default [user decisions A9, C4]; the K closure's limit (C3: its pilots can leave the closure off)
  still applies.

## References

- Baddeley A, Diggle PJ, Hardegen A, Lawrence T, Milne RK, Nair G (2014). On tests of spatial pattern based on
  simulation envelopes. *Ecological Monographs* 84(3):477-489. doi:10.1890/13-2042.1
- Baddeley A, Rubak E, Turner R (2015). *Spatial Point Patterns: Methodology and Applications with R.* Chapman &
  Hall/CRC. doi:10.1201/b19708 [the passage on Monte Carlo tests in the same window: to verify]
- Besag J, Diggle PJ (1977). Simple Monte Carlo tests for spatial pattern. *Applied Statistics* 26(3):327-333.
  doi:10.2307/2346974
- Dao NA, Genton MG (2014). A Monte Carlo-adjusted goodness-of-fit test for parametric models describing spatial point
  patterns. *Journal of Computational and Graphical Statistics* 23(2):497-517. doi:10.1080/10618600.2012.760459
- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768
- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochemistry and Cell Biology*
  141(6):629-638. doi:10.1007/s00418-014-1192-3
- Früh SM, Matti U, Spycher PR, Rubini M, Lickert S, Schlichthaerle T, Jungmann R, Vogel V, Ries J, Schoen I (2021).
  Site-specifically-labeled antibodies for super-resolution microscopy reveal in situ linkage errors. *ACS Nano*
  15(7):12161-12170. doi:10.1021/acsnano.1c03677
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of βII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic
  nerve axons. bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1
- Kiskowski MA, Hancock JF, Kenworthy AK (2009). On the use of Ripley's K-function and its derivatives to analyze
  domain size. *Biophysical Journal* 97(4):1095-1103. doi:10.1016/j.bpj.2009.05.039
- Myllymäki M, Mrkvička T, Grabarnik P, Seijo H, Hahn U (2017). Global envelope tests for spatial processes. *Journal
  of the Royal Statistical Society B* 79(2):381-404. doi:10.1111/rssb.12172
- Nieves DJ, Pike JA, Levet F, Williamson DJ, Baragilly M, Oloketuyi S, de Marco A, Griffié J, Sage D, Cohen EAK,
  Sibarita J-B, Heilemann M, Owen DM (2023). A framework for evaluating the performance of SMLM cluster analysis
  algorithms. *Nature Methods* 20(2):259-267. doi:10.1038/s41592-022-01750-6
- Platzer R, Rossboth BK, Schneider MC, Sevcsik E, Baumgart F, Stockinger H, Schütz GJ, Huppa JB, Brameshuber M
  (2020). Unscrambling fluorophore blinking for comprehensive cluster detection via photoactivated localization
  microscopy. *Nature Communications* 11:4993. doi:10.1038/s41467-020-18726-9
- Rényi A (1958). On a one-dimensional problem concerning random space filling. *Publications of the Mathematical
  Institute of the Hungarian Academy of Sciences* 3:109-127 [the parking constant 0.7476; the text was not checked].
- Ripley BD (1977). Modelling spatial patterns. *Journal of the Royal Statistical Society B* 39(2):172-192.
  doi:10.1111/j.2517-6161.1977.tb01615.x
- Schneider MC, Schütz GJ (2022). Don't be fooled by randomness: valid p-values for single molecule microscopy.
  *Frontiers in Bioinformatics* 2:811053. doi:10.3389/fbinf.2022.811053
- Veatch SL, Machta BB, Shelby SA, Chiang EN, Holowka DA, Baird BA (2012). Correlation functions quantify
  super-resolution images and estimate apparent clustering due to over-counting. *PLoS ONE* 7(2):e31457.
  doi:10.1371/journal.pone.0031457
