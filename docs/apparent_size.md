# Apparent cluster size: content, footprint, and estimators that do not grow with N

> **Resumen (español).** El "tamaño" de un cluster en dSTORM son dos cosas distintas: cuántas moléculas marcadas
> tiene (**contenido**: eventos o localizaciones) y cuántos nanómetros de membrana ocupa (**huella**). El contenido se
> mide bien cluster por cluster y es la medida **primaria**; la huella sólo se mide **por axón** (ρ̂_pool), con ≥ 1,5
> etiquetas por cluster y la NeNA del archivo, y el programa dice cuándo no es identificable. La columna «Cluster
> area» (envolvente convexa) y su r_eff **quedan exactamente como estaban** (decisión del usuario); como la envolvente
> crece con el número de localizaciones, se **agregan al lado** dos medidas que no crecen con N: el desvío de la nube
> (`size_cloud_sd_nm`) y la envolvente a N fijo = 15 (`size_hull_area_n15_nm2`, `size_r_rare_nm`, indefinida con
> < 15 localizaciones). Para la pregunta (a), tamaño vs distancia al vecino, el clusterizado fabrica solo una
> asociación positiva: el nulo es el **nulo de pipeline**; la permutación de marcas queda como **diagnóstico** rotulado.
> Para la pregunta (b), tamaño vs curvatura, se usa κ·R (nunca κ cruda) de una P-spline con nudos cada 400 nm, los
> tramos cóncavos se excluyen del análisis primario y se informan aparte, y con R < 600 nm la curvatura se rotula «no
> resuelta». El axón es la unidad para (b) y el animal para comparar grupos. Todo funciona con las etiquetas de
> cualquier método de clusterizado. Decisiones del usuario del 2026-10-22 (D6 y respuestas 1a-9a).

This page explains what MPS Explorer reports about the size of a cluster, why, and what it does not report. The code
is in `tools/mps_apparent_size.py` (per-cluster estimators, the footprint pool, the statistics within an axon),
`tools/mps_curvature.py` (the 400 nm membrane and κ·R), `tools/mps_size_stats.py` (combining axons and animals),
`tools/size_plugins.py` (the multi-method plug-ins, the per-cluster export, the pipeline-null hook) and
`tools/mps_size_synth.py` (the synthetic generator the tests and the probes use). Every number below is labelled:

- **literature**: a published value, with its reference;
- **derivation**: shown here;
- **simulation (feasibility study)**: the apparent-size feasibility simulation of 2026-10-15 (synthetic data only;
  scenarios A-C: group difference, size vs neighbour, size vs curvature; its generator is ported to
  `tools/mps_size_synth.py`);
- **simulation (probe)**: `docs/sims/apparent_size/size_probes.py`, which re-measures what the shipped estimators
  do (synthetic data only; the committed `size_probes.json` is the run quoted here; ~20 s, one process);
- **ad hoc**: a choice without a measurement behind it, said so.

No measured axon was used for any number on this page.

## 1. What the image shows of a cluster

A cluster in a localization image is not the cluster: it is a cloud of localizations seen through four layers:

1. the **footprint** ρ: the stretch of membrane the epitopes occupy (what one would like to call its size);
2. the **labels**: 1-5 antibody stacks that landed at random points of the footprint;
3. the **linkage**: the dye sits ~10 nm from its epitope in a direction fixed for the whole movie (σ_link ≈ 10 nm per
   axis; literature: Früh et al. 2021);
4. the **precision**: every localization is off by σ_loc ≈ 8-14 nm, which NeNA measures per file (literature:
   Endesfelder et al. 2014; Mortensen et al. 2010).

On top of that, each label blinks a geometric number of times (mean 14 in MEA, 26 in β-ME; literature: Dempsey et
al. 2011), so one label can give 3 localizations and another 90. The variance of the localizations along the
membrane is (derivation, P-R61; verified to 0.3 nm in the feasibility simulation):

    E[s_t²] = σ_loc² + (ρ²/4 + σ_link²) · (1 − Σw²) · N/(N − 1)

where w_m is the share of the localizations that label m gave. With **one label** Σw² = 1 and the footprint
disappears from the equation: 30 localizations of one label are 30 copies of one point plus the precision, whatever
the footprint. With several labels the footprint appears, estimated as a variance with N_labels − 1 degrees of
freedom (≈ 100 % error per cluster). Hence two notions:

- **content**: how many labelled molecules the cluster holds, seen as more labels → more events → more
  localizations. It is what is measured per cluster;
- **footprint**: nm of membrane. Not measurable per cluster; measurable **per axon** by pooling (section 5).

Everything per cluster is therefore an **apparent** size (footprint ⊕ linkage ⊕ precision), and every column says so.

## 2. What the program reports

| Column / readout | What | Role | Origin |
|---|---|---|---|
| `size_n_events`, `size_n_events_median` | events (blinks linked in time) per cluster | **content, primary** when frames and NeNA exist | decision 1a; event layer: radius 3·NeNA, 1 dark frame (decision D5, `docs/dbscan_rule.md`) |
| `size_n_locs`, `size_n_locs_median` | localizations per cluster | content (the fallback; the statistic the pipeline null tests) | decision 1a |
| `size_rho_pool_nm` ± `size_rho_pool_se_nm` | footprint of the axon (section 5) | **footprint, beside the content**, only when identifiable | P-R61 §3.2; decision 1a |
| `size_cloud_sd_nm` | SD of the localization cloud | apparent, **does not grow with N** | decision D6; section 3 |
| `size_hull_area_n15_nm2`, `size_r_rare_nm` | convex hull of 15 localizations, 20 draws | apparent, **does not grow with N**; empty below 15 | decision D6; answer 8 (n = 15) |
| `size_radius_gyration_nm`, `size_sd_major_nm`, `size_sd_minor_nm`, `size_sd_tangential_nm`, `size_sd_normal_nm`, `size_elongation` | shape of the cloud | apparent, descriptive | feasibility simulation §8.2 step 1 |
| `area_nm2` / «Cluster area», `r_eff_nm` | convex hull of all localizations | **legacy, unchanged**; grows with N | the user's decision (no rename); Gazal et al. 2026 |

The user decided on 2026-10-22 that «Cluster area», its column name and every existing label stay exactly as they
are; the new measures are added beside them with the `size_` prefix. No new readout uses r_eff as the size of a
cluster.

## 3. Why the hull grows with N and the SD does not

The convex hull wraps the outermost points of the cloud, and the extremes of a sample grow with its size: the
expected hull of n random points grows with n (literature: Rényi & Sulanke 1963; Efron 1965, who links the expected
hull area to the expected number of vertices). A standard deviation does not: it is estimated from all the points
and its bias at small n is a few percent (E[s]/σ = 0.94 at n = 5, 0.97 at 10, 0.99 at 20; literature: Gurland &
Tripathi 1971).

Probe "same_cloud" (simulation, probe): one Gaussian cloud, σ = 10 nm, measured with 20 and with 200
localizations, 500 clouds each (medians):

| | N = 20 | N = 200 |
|---|---|---|
| `size_cloud_sd_nm` | 9.93 | 9.97 |
| `size_hull_area_n15_nm2` | 705 | 720 |
| `size_r_rare_nm` | 14.96 | 15.01 |
| hull of all localizations (= «Cluster area») | 846 | 2088 (× 2.47) |
| its r_eff | 16.4 | 25.8 |

Probe "population" (simulation, probe): 2000 clusters of the P-R61 model (footprint 25 nm, σ_loc 10 nm with a 25 %
spread between localizations, linkage 10 nm), Spearman correlation of each estimator with N_loc:

| estimator | 1 label | 3 labels |
|---|---|---|
| hull of all localizations (legacy) | **0.89** | **0.61** |
| `size_cloud_sd_nm` (shipped: √(trace S / 2)) | 0.11 | −0.05 |
| geometric mean of the principal SDs, √(sd_major · sd_minor) | 0.36 | −0.01 |
| `size_r_rare_nm` (n = 15) | 0.08 | −0.05 |

The feasibility simulation found the same order: Spearman(r_hull, log N_loc) 0.65-0.80, r_rare −0.01 to 0.23, and
the r_eff of the same clusters measured complete (≥ 200 localizations) and with 20 drawn: 37.8 vs 22.9 nm.

**Which SD.** `size_cloud_sd_nm` is √(trace(S)/2), with S the sample covariance (ddof 1): the root mean square of
the SDs along two perpendicular axes. It is rotation-invariant, needs no membrane frame, equals σ for an isotropic
cloud, and its variance part is unbiased at every n. The geometric mean of the principal SDs (the √(σ_major σ_minor)
of the occupancy ellipse, and the simulation's r_dispersion, which used the membrane frame) is biased low in small
clusters, because det(S) subtracts the chance correlation of a small sample: 0.36 against 0.11 in the table above.
The principal and the tangential / normal SDs are exported anyway.

**What the cloud SD measures.** With one label per cluster it measures the precision (σ_loc), not the footprint
(the cloud is 10.1 nm wide with σ_loc 10 nm). The NeNA enters every apparent estimator at 1-2.3 nm per nm of NeNA
(simulation, feasibility §3.1), so apparent sizes are compared across files only with the NeNA as a covariate
(section 8).

## 4. The hull at fixed N

`size_hull_area_n15_nm2` is the mean hull area of **15** localizations drawn without replacement, over **20** draws;
`size_r_rare_nm` the mean radius √(area/π) over the same draws. Fifteen is the user's approved answer 8 (the plan's
first proposal, 20, was ad hoc); it keeps more one-label clusters defined: 34 % of the one-label clusters of the
probe have fewer than 15 localizations (41 % in the feasibility simulation). Those get `size_r_rare_undefined` =
True and empty values, never a number from fewer points. Twenty draws [simulation, feasibility §3]; each cluster
draws from its own seeded generator (seed, label), so a cluster's value does not depend on the other clusters or on
the other methods compared.

## 5. The footprint per axon

Pooling the K clusters of an axon recovers the footprint by the method of moments (P-R61 §3.2):

    σ_L² = Σ_i (s_t,i² − σ_loc²) · (N_i − 1)/N_i  /  Σ_i (1 − E[Σw² | events_i])
    ρ̂²   = 4 · (σ_L² − σ_link²)

with s_t the SD of each cluster **along** the membrane (the 400 nm membrane of section 7), σ_loc the file's NeNA,
σ_link = 10 nm (literature: Früh et al. 2021; adopted, not measured) and E[Σw² | events] tabulated by Monte Carlo
for the label prior that the mean events per cluster imply (c_label = 14, literature: Dempsey et al. 2011,
provisional, decision D1; 2 localizations per event, ad hoc, the project simulator's default). It estimates the
root-mean-square footprint of the axon's clusters. The standard error is the delete-one jackknife over clusters
(literature: Efron & Tibshirani 1993).

`size_label_regime_verdict` says when it is a number:

| verdict | when | origin |
|---|---|---|
| footprint measurable | mean events / c_label ≥ 1.5 labels per cluster, ≥ 5 usable clusters | simulation, feasibility §8.2; ≥ 5 ad hoc |
| content only | mean labels < 1.5: the footprint is not in the data | same |
| not evaluated | no frames (no events), no NeNA, no membrane frame, or < 5 clusters | — |

The pool leaves out clusters with elongation > **2.5** (a probable fusion). The feasibility simulation excluded the
TRUE fusions, which the program cannot know; probe "footprint_pool" (simulation, probe; one ring per axon, circle
R 1400 nm, K = 60, 12 axons, mean ± SD of ρ̂ in nm, true footprint in the first column):

| labels | ρ | true clusters | DBSCAN 25/10, nothing left out | DBSCAN, elongation > 1.8 out | DBSCAN, > 2.5 out |
|---|---|---|---|---|---|
| 2 | 0 | 5.4 ± 5.0 | 15.6 ± 8.3 | 3.4 ± 5.4 | 6.7 ± 5.9 |
| 2 | 25 | 28.1 ± 6.7 | 34.3 ± 7.6 | 16.3 ± 6.8 | 23.3 ± 6.4 |
| 3 | 0 | 5.4 ± 5.3 | 17.1 ± 9.1 | 2.2 ± 3.5 | 6.2 ± 4.8 |
| 3 | 25 | 24.9 ± 3.7 | 39.2 ± 8.6 | 16.1 ± 7.1 | 23.6 ± 3.9 |

Keeping every cluster lets the fusions inflate the footprint; leaving out the "elongated" flag's 1.8 truncates the
tangential spread; 2.5 lands within ~1-2 nm of the pool of the true clusters. The cut is therefore its own constant
(`POOL_EXCLUDE_ELONGATION`), separate from the flag.

**The NeNA must be right.** With 3 labels and ρ = 25 nm, a NeNA 2 nm too low gives 31.8 ± 4.5 nm and 2 nm too high
10.5 ± 6.8 nm (probe, DBSCAN, cut 2.5); the feasibility simulation found 6-15 nm per 2 nm. A difference of footprint
between groups is indistinguishable from a difference of NeNA unless each file's NeNA is measured (`docs/nena.md`).
The pool takes no covariate: it already deconvolves the NeNA (decision 6a).

**Known limit.** With one label per cluster DBSCAN's fusions raise the events per cluster, and 3 of 12 one-label
axons of the probe passed the 1.5 guard (with a footprint of 0 nm). The guard reads events, which fusions inflate;
a measured c_label (calibration B1) and the fusion mark of the event layer are the remedy.

## 6. Flags

| flag | rule | origin |
|---|---|---|
| `size_flag_elongated` | tangential / normal SD > 1.8 (candidate fusion) | simulation, feasibility §5.1: 92 % of DBSCAN's fusions, 5 % false alarms with 3 labels |
| `size_flag_close` | nearest cluster centre < 50 nm (candidate split) | simulation, feasibility §5 |
| `size_flag_few_locs` | < 20 localizations (detection margin) | simulation, feasibility §5 |
| `size_r_rare_undefined` | < 15 localizations | answer 8 |

Flags mark clusters for the reader; they do not remove anything. In the feasibility simulation all three together
reduced the fabricated size-neighbour association (z 6.5 → 2.6) but did not remove it.

## 7. Question (a): size against the distance to the nearest cluster

`size_nloc_nn_spearman` is Spearman's correlation, within the axon, of the localizations per cluster with the
distance to the nearest kept cluster centre (`size_d_nn_nm`, 2D; on a ring the chord, within 1 % of the arc for
neighbours closer than 200 nm on a ring of R ≥ 600 nm, derivation). `size_events_nn_spearman` is the same with
events; `size_cloud_sd_nn_partial_spearman` the apparent cloud SD with log N_loc partialled out (decision 6a).

**The clustering fabricates a positive value.** When two clusters are close, DBSCAN tends to fuse the large ones
(the fused cluster is bigger and its nearest remaining neighbour is farther) and to rescue small ones that have a
neighbour nearby. Under a true null (sizes independent of position), the feasibility simulation (scenario B, 3
labels, DBSCAN 25/10) found Spearman 0.17 and a combined z of 6.5 over 18 axons; probe "permutation_diagnostic"
(simulation, probe; 20 axons, one ring of R 2000 nm, 3 labels, 5 % background):

| detection | mean Spearman | mean z per axon | combined (Stouffer) z | axons with \|z\| > 1.96 |
|---|---|---|---|---|
| true clusters | 0.007 | 0.10 | 0.43 | 0 % |
| DBSCAN eps 20 | 0.137 | 1.05 | 4.69 | 15 % |
| DBSCAN eps 25 | 0.136 | 1.08 | 4.83 | 20 % |
| DBSCAN eps 30 | 0.175 | 1.32 | 5.92 | 25 % |

**The null is the pipeline null** (decision 2a): the axon simulated with its own measured parameters, clustered by
the SAME method with the same parameters and curation, 199 replicates (`feat/pipeline-null`, `docs/null_and_scale.md`
on that branch). `size_nloc_nn_null_p` is its Monte Carlo p; it is empty when the null did not run, and it is never
replaced by anything else. In the feasibility simulation the pipeline null held 5 % in every null cell (9-15 % when
the simulated axon did not reproduce a heterogeneous footprint, so the simulator must take the axon's own
parameters).

**The mark permutation is a diagnostic only.** `size_nloc_nn_markperm_diagnostic_z` is the z of the observed
Spearman against 199 reassignments of the sizes among the detected positions (the null first proposed). It asks
whether sizes are assigned at random to these positions, and the answer is "no, because the clustering assigned
them": in the feasibility simulation it rejected 100 % of the null experiments (18 axons combined); in the probe
above its combined z is 4.7-5.9 with DBSCAN and 0.4 with the true clusters. It is labelled DIAGNOSTIC everywhere
and declared as a value, not as a p-value, so the multi-method summary never turns it into a verdict.

**eps 20 / 25 / 30.** The fabricated association grows with eps (feasibility: z 5.3 / 6.5 / 8.1; probe above), so
the statistic is always reported at the three eps (decision 9a): plug-in `size_eps`, readouts
`size_nloc_nn_spearman_eps20`, `_eps25`, `_eps30` (DBSCAN only; the same slab, min_samples and curation).

**What can be detected.** With the pipeline null, the feasibility simulation detected a CONTENT relation (clusters
with more room hold more molecules) 94-100 % of the time, and a FOOTPRINT relation (clusters with more room are
wider, same molecules) only 10-35 % with DBSCAN. Hence the content is the primary readout.

## 8. Question (b): size against local curvature

**Curvature.** `tools/mps_curvature.py` fits the periodic cubic P-spline of `tools.mps_membrane.
fit_membrane_from_localizations` to the clustered localizations with knots every **400 nm** (decision 4a), started
from the P-spline through the centroids at the same spacing; the pipeline null keeps its 600 nm membrane. The
curve is resampled every 5 nm and differentiated by central differences 10 nm apart (derivation: the spline is
smooth on the knot scale, the truncation error is far below 1 % of κ). The sign is + convex whatever the
orientation. Feasibility simulation (§6.1), Spearman of the estimated with the true κ·R at 400 / 600 / 800 nm
knots: thin axons (R 400-600 nm) 0.88 / 0.78 / 0.70, thick (R 1200-1600 nm) 0.90 / 0.95 / 0.96; notched thin
0.75 / 0.66 / 0.60. Probe "curvature_400nm" (simulation, probe; 5 rings each, 3 labels):

| contour | Spearman (mean, min) | RMSE of κ·R | concave clusters found (κ·R < −0.3) |
|---|---|---|---|
| ellipse, R 1400 nm, aspect 1.6 | 0.89, 0.85 | 0.18 | — |
| notched ellipse, R 1400 nm, 3 notches | 0.96, 0.93 | 0.19 | 100 % |
| ellipse, R 500 nm | 0.92, 0.89 | 0.24 | — |
| notched ellipse, R 500 nm | 0.80, 0.63 | 0.40 | 0 % |

**κ·R, never κ.** R = L/2π with L the length of the fitted membrane (κ·R = 1 on a circle). Thin axons have a large
κ and may differ in everything else (NeNA, labelling, fusions): with raw κ and clusters as units the feasibility
simulation gave 95-100 % false positives, with κ·R and the axon as unit 2-8 %. The raw κ is exported
(`size_curvature_per_um`) with a description that says not to analyse it.

**R < 600 nm: "curvature not resolved".** Notches shorter than ~500 nm are lost in thin axons (table above: the
thin notched contour finds none of its concave clusters). The values are kept; `size_curvature_verdict` and
`size_curvature_resolved` carry the label (decision 4a; feasibility §6.1, §7.3).

**Concave stretches apart.** A sectioning artefact can change density and shape in concave stretches without
changing size; untreated, it gave 15-77 % false positives in the feasibility simulation. The primary readout
`size_content_kappa_r_spearman` uses only the clusters with κ·R ≥ 0; `size_content_kappa_r_spearman_concave` and
`size_n_concave` report the others apart (decision 3a). When the pipeline null includes the measured density and
elongation per stretch, the concave clusters can come back (the decision's option b).

`size_cloud_sd_kappa_r_partial_spearman` is the apparent cloud SD against κ·R with log N_loc partialled out.

## 9. Units and covariates across axons

| analysis | unit | how | origin |
|---|---|---|---|
| (b) size vs curvature | **axon** | per-axon Fisher z (`*_fisher_z`) → one-sample t over axons; animal as second level (mean z per animal, t over animals), or a mixed model with a random intercept per axon on the clusters | decision 5a; feasibility §6.3 |
| group comparison (genotype) | **animal** | Welch t on animal means, or a mixed model with a random intercept per animal | decision 5a; feasibility §4.2 |
| (a) size vs neighbour | axon | pipeline-null p per axon | decision 2a |

With clusters as units the feasibility simulation rejected the null 71-92 % of the time, with axons as units 28-44 %,
with animals or the mixed model 2-8 % (literature: Lazic 2010; Aarts et al. 2014). The t over 3 animals has 2 df
and loses about half the power in (b): the animal is the second level there, not the only one.

**Covariates (decision 6a)** (`tools.mps_size_stats.COVARIATES_BY_KIND`): apparent estimators (cloud SD, r_rare,
gyration) always with the file's NeNA and log N_loc; the footprint pool with none (it deconvolves the NeNA, and two
covariates on 6 animals leave 2 df: power 87 % → 31 % in the feasibility simulation); the content with none (ad
hoc: log N_loc cannot be its own covariate). The mixed model is REML with a random intercept (literature: Patterson
& Thompson 1971), implemented in numpy/scipy; its t uses G − p degrees of freedom (conservative). Within one axon
the NeNA is a single number, so the within-axon partial correlations control log N_loc only.

**Design (answer 7: decide after a pilot).** `tools.mps_size_stats.design_guidance()` lists the feasibility
simulation's power for a 5 nm footprint difference with 3 animals per group: ~1 label per cluster — not measurable;
~2 labels — 44-53 % (about 5-6 animals per group, an extrapolation, not simulated); ≥ 3 labels — 87-89 %; 5 labels
97-98 %. No default is set: the label regime (events per cluster / c_label, P-R75) is measured on a pilot first.

## 10. Across clustering methods

Every function here takes the labels of whatever method produced them. `tools.size_plugins.register_with_multimethod()`
registers two plug-ins with `tools.multimethod`:

- `size`: every per-axon `size_*` readout of `SIZE_READOUTS` (contents, footprint, flags, question (a), question
  (b)); reads `context.extras["nena_nm"]` and, optionally, `size_c_label`, `size_locs_per_event`,
  `pipeline_null_results` and `size_pipeline_null`;
- `size_eps`: question (a) at DBSCAN eps 20 / 25 / 30 (empty for other methods; its cost is two extra analyses).

`tools.size_plugins.cluster_rows(size_analysis(an, nena_nm=...))` gives the per-cluster `size_*` columns, joined
to the cluster table by `cluster_label`; the clusters are those «Cluster area» describes (not bad, not discarded).
`register_with_pipeline_null()` adds the question-(a) statistic to `tools.mps_pipeline_null`'s readouts so that the
null re-computes it on every replicate with the same method.

## 11. Limits

- Every number on this page is synthetic and uses the P-R61 measurement model (disc footprint, fixed Gaussian
  linkage, geometric blinking, log-normal precision). Different linkage or blinking change the numbers, not the
  order of the findings, which follow from structural properties: extremes of the hull, selection by density in the
  clustering, confounding by calibre.
- The simulated alternatives were strong (true Spearman 0.89-0.99); the reported power is a ceiling.
- Rings were clustered one at a time (as if resolved in z); in 3D the clustering is joint and fusions between
  neighbouring rings can make the artefact of question (a) larger.
- c_label (14) is provisional until calibration B1; without it the content is better counted in localizations and
  the footprint guard can be fooled by fusions (section 5).

## References

- Aarts E., Verhage M., Veenvliet J.V., Dolan C.V., van der Sluis S. (2014) A solution to dependency: using
  multilevel analysis to accommodate nested data. *Nat. Neurosci.* 17:491-496. doi:10.1038/nn.3648
- Besag J., Diggle P.J. (1977) Simple Monte Carlo tests for spatial pattern. *Appl. Stat.* 26:327-333.
  doi:10.2307/2346974
- Dempsey G.T., Vaughan J.C., Chen K.H., Bates M., Zhuang X. (2011) Evaluation of fluorophores for optimal
  performance in localization-based super-resolution imaging. *Nat. Methods* 8:1027-1036. doi:10.1038/nmeth.1768
- Efron B. (1965) The convex hull of a random set of points. *Biometrika* 52:331-343. doi:10.1093/biomet/52.3-4.331
- Efron B., Tibshirani R.J. (1993) *An Introduction to the Bootstrap*. Chapman & Hall (ch. 11, the jackknife).
- Eilers P.H.C., Marx B.D. (1996) Flexible smoothing with B-splines and penalties. *Stat. Sci.* 11:89-121.
  doi:10.1214/ss/1038425655
- Endesfelder U., Malkusch S., Fricke F., Heilemann M. (2014) A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochem. Cell Biol.* 141:629-638.
  doi:10.1007/s00418-014-1192-3
- Fisher R.A. (1915) Frequency distribution of the values of the correlation coefficient in samples from an
  indefinitely large population. *Biometrika* 10:507-521. doi:10.2307/2331838
- Früh S.M. et al. (2021) Site-specifically-labeled antibodies for super-resolution microscopy reveal in situ
  linkage errors. *ACS Nano* 15:12161-12170. doi:10.1021/acsnano.1c03677
- Gazal N.G. et al. (2026) The reference pipeline of MPS Explorer (DBSCAN 25 nm / 10; convex-hull cluster area).
- Gurland J., Tripathi R.C. (1971) A simple approximation for unbiased estimation of the standard deviation.
  *Am. Stat.* 25(4):30-32. doi:10.1080/00031305.1971.10477279
- Huang B., Wang W., Bates M., Zhuang X. (2008) Three-dimensional super-resolution imaging by stochastic optical
  reconstruction microscopy. *Science* 319:810-813. doi:10.1126/science.1153529
- Lazic S.E. (2010) The problem of pseudoreplication in neuroscientific studies: is it affecting your analysis?
  *BMC Neurosci.* 11:5. doi:10.1186/1471-2202-11-5
- Mortensen K.I., Churchman L.S., Spudich J.A., Flyvbjerg H. (2010) Optimized localization analysis for
  single-molecule tracking and super-resolution microscopy. *Nat. Methods* 7:377-381. doi:10.1038/nmeth.1447
- Patterson H.D., Thompson R. (1971) Recovery of inter-block information when block sizes are unequal.
  *Biometrika* 58:545-554. doi:10.1093/biomet/58.3.545
- Phipson B., Smyth G.K. (2010) Permutation p-values should never be zero. *Stat. Appl. Genet. Mol. Biol.* 9(1):39.
  doi:10.2202/1544-6115.1585
- Rényi A. (1958) On a one-dimensional problem concerning random space filling. *Publ. Math. Inst. Hung. Acad.
  Sci.* 3:109-127 (the hard-core cap of the synthetic generator).
- Rényi A., Sulanke R. (1963) Über die konvexe Hülle von n zufällig gewählten Punkten. *Z.
  Wahrscheinlichkeitstheorie verw. Gebiete* 2:75-84. doi:10.1007/BF00535300

(The DOIs of Aarts, Dempsey, Eilers & Marx, Endesfelder, Früh, Lazic, Mortensen, Phipson & Smyth and Rényi &
Sulanke were checked against Crossref for the feasibility study; the others were not re-checked for this page.)
