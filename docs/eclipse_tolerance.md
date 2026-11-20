# The eclipse tolerance: when two clusters of consecutive rings are "in the same column"

> **Resumen (español).** Dos clusters de anillos consecutivos están "eclipsados" (en la misma columna) cuando sus
> centros coinciden a lo largo de la membrana dentro del error con que se conoce cada centro. El programa calcula
> dos reglas y las muestra siempre juntas: la **escalar** (|Δs| ≤ τ = 31 nm, derivada, PRIMARIA hoy) y la **por
> par** (|Δs_ij| ≤ 1,96·√(σ_c,i² + σ_c,j²), con el error de centro de CADA cluster calculado de sus localizaciones
> y eventos; EXPLORATORIA hasta calibrar cuántos eventos da cada anticuerpo, c_label). El valor piloto anterior se
> eliminó. El tamaño de un cluster no se puede medir para esto (con 1-5 anticuerpos la huella es invisible o tiene
> ±100 % de error); la huella se mide una vez por AXÓN, juntando todos los clusters, sólo si hay en promedio dos o
> más anticuerpos por cluster; si no, se usan 25 nm como cota superior (y 0 en la sensibilidad), y eso es una
> **decisión, no una medición**. El área de la envolvente (r_eff) crece con el número de localizaciones y no entra
> en ninguna tolerancia. Todo funciona con cualquier método de clusterizado. Decisiones del usuario del 2026-10-22
> (P-R61, respuestas 1a, 2 a+c, 3a, 4 a+b, 5a, 6, 7).

This page explains the eclipse tolerance of the column test (H-ECL: do the βII-spectrin clusters of consecutive MPS
rings stack in columns more often than chance?), which numbers it uses and where each one comes from. The code is
in `tools/mps_eclipse_tolerance.py` (centre errors, footprint, merged clusters, exports, the multi-method plug-in),
`tools/mps_matching.py` and `tools/mps_unroll.py` (the matching and the rotation null with a per-pair tolerance), and
`config/columns_params.yaml` (the pre-specified scalar τ). Every number is labelled with its origin:

- **literature**: a published value, with its reference;
- **derivation**: shown here;
- **own simulation**: synthetic data only, with the script that reproduces it. The P-R61 research simulations
  (2026-10-15) and the validation of this implementation, `docs/sims/eclipse_tolerance/eclipse_sim.py` (its JSON,
  `eclipse_sim.json`, is the run quoted here; seed 20261022; 94 s, one process);
- **ad hoc**: a choice without a better basis, said so.

## 1. The question and the two rules

Two clusters i (ring k) and j (ring k + 1) are a column candidate when the difference of their positions along the
membrane, Δs_ij, is compatible with zero given the error of each centre. The matching itself is unchanged: one-to-one,
maximum cardinality, then minimum total distance (D-24b), against a rotation null of each ring on the reference curve
(the evidence is the excess of matched pairs over that null, its ζ and its Monte Carlo p, never the raw count). Only
the admissibility of a pair changes between the two rules:

| Rule | A pair is admissible when | Role today | Origin |
|---|---|---|---|
| **Scalar** | \|Δs_ij\| ≤ τ, τ = **31 nm** (`tau0_nm`) | **PRIMARY** | derivation (section 2) |
| **Per-pair** | \|Δs_ij\| ≤ t_ij = **1.96** · √(σ_c,i² + σ_c,j²) | **EXPLORATORY** (labelled) | derivation (section 3) |

Both are computed on the same rings, the same curve and the **same rotation draws** (the per-pair tolerances travel
with their clusters in every replicate: a rotation moves a cluster, never its tolerance), and are shown in the same
table, always. The per-pair rule becomes the primary one only when the events per antibody stack (c_label) are
calibrated on the user's own antibodies (P-R75), with the date of that calibration recorded
(`ToleranceInputs.primary_rule()`); nothing is replaced silently.

The arc (1D, along the membrane) is the primary test and the plane (2D) a sensitivity [user's decision 7; own
simulation P-R61: power 0.56 on the arc against 0.48 in the plane, 30 % of columns].

## 2. The scalar τ = 31 nm (derived; the earlier pilot value was removed)

τ = 1.96 · √2 · σ_c = **2.77 × 11.3 nm = 31 nm**:

- **1.96 · √2** [derivation]: the difference of two independent centres, each with per-axis error σ_c, has standard
  deviation √2 σ_c; 1.96 of those hold 95 % of true coincidences along a line.
- **σ_c = 11.3 nm** [own simulation, P-R61 / methods draft 2.3]: the per-axis centre error of a typical cluster —
  three antibody stacks on a 25 nm footprint [literature: Gazal et al. 2026, median r_eff], linkage 10 nm per axis
  [literature: Früh et al. 2021], localization precision 10 nm [literature: Huang et al. 2008], geometric event
  counts with a mean of 10 events of two localizations each.

The tolerance used before 2026-10-22 had been measured once on one exploratory pilot axon with a recipe that adds
the clusters' hull radii to their bootstrap errors (a "contact" rule). The P-R61 simulation showed that this kind of
rule matches 63-65 % of clusters by chance and has about a quarter of the power of 31 nm (0.08-0.16 against
0.53-0.67), because the hull radius measures how many times a cluster blinked, not its size (section 4). The value is
gone from the configuration, the code, the tests and the documentation; the scalar rule stays reproducible at any
other tolerance through the review's tolerance control ("Free"). The sensitivity presets are **62 nm** (2τ, the
structural column jitter sensitivity, P-R3) and **100 nm** [ad hoc: a wide value kept from the earlier list].

## 3. The per-pair rule: each cluster's own centre error

The per-axis error of a cluster's centre [derivation, P-R61 2.4; verified to 0.3 nm in 29 simulated scenarios]:

    σ_c² = σ_loc² / N_loc + (σ_link² + ρ²/4) · E[Σw² | events] + σ_reg²

| Term | Default | Origin |
|---|---|---|
| σ_loc, localization precision | the file's **NeNA**; **10 nm** if it is not given | measurement per file; literature (Huang et al. 2008) |
| N_loc | the cluster's localizations | measured, from ANY clustering method's labels |
| σ_link, linkage error per axis | **10 nm** (editable) | literature (Früh et al. 2021: offsets up to ~19 nm → l/√3); an assumption until single-molecule clusters on glass calibrate it (Phase B) |
| Σw², sum of squared label shares | E[Σw² \| events], from a simulated table | own simulation (P-R61 generator; 100 000 clusters per label prior; seed 20261015); 1 (one label) without events |
| c_label, events per antibody stack | **14** (AF647 in MEA) or 26 (βME), labelled "preset" | literature (Dempsey et al. 2011), a lower bound; the calibration (sparse primary on glass with the same secondary, P-R75) is Phase B |
| Label prior of the table | zero-truncated Poisson, λ from the axon's mean events per cluster / c_label | own simulation (grid 0.05-6, the research's) |
| ρ, footprint | the axon's pooled footprint, or 25 nm (section 4) | measurement per axon; literature (Gazal et al. 2026) for the bound |
| σ_reg, registration per ring | **2 nm** | literature (Wang et al. 2014, RCC ~1.6 nm) |
| k (arc) / k (plane) | **1.96** / **2.448** | derivation (95 % of a normal / of a Rayleigh: √(−2 ln 0.05)) |
| Locs per event (table only) | 2 | ad hoc (the project simulator's default) |

What the program can measure per cluster, and what enters the tolerance legitimately, is its number of **events**
(through Σw²) and of localizations — not its size. Each cluster also gets `centroid_r95_nm` = 2.448 σ_c, the radius
of the circle that holds its true centre 95 % of the time (approved export).

Why the null still holds with per-pair tolerances [own simulation, P-R61 5.1]: a rule of the form √(σ_i² + σ_j²) is
concave in the pairing of large with large, so the observed configuration has, if anything, fewer chance matches
than the rotated ones; under extreme position-shared heterogeneity (0.2-8 labels and 6-25 nm precision on the same
side of the axon in both rings) the rejection rate stayed at 0.04-0.05.

## 4. The footprint: per axon, not per cluster

**The size of one cluster cannot be measured for this purpose** [own simulation, P-R61 3; user's decision 1a]. The
localizations of one antibody are copies of one point plus photon noise, so with one label the footprint is invisible
(the cloud measures σ_loc whatever the footprint), and with three it is estimated with ±100 % error (a variance with
two degrees of freedom). The convex-hull radius r_eff measures mostly how many localizations the cluster has (the
logarithm of N_loc explains 69-75 % of its variance; Rényi & Sulanke 1963): it is exported for the record, never used
as the size of a cluster, and "Cluster area" keeps its name and meaning.

**The footprint is measured once per axon** by pooling every cluster (method of moments) [own simulation, P-R61
3.2]:

    σ_L² = Σ_i (s_i² − σ_loc²) (N_i − 1)/N_i / Σ_i (1 − Σw²_i),   ρ² = 4 (σ_L² − σ_link²)

with s_i the standard deviation of cluster i's localizations along the membrane (the tangent at the cluster,
perpendicular to the line from the axon's centre [ad hoc approximation, exact on a circle]). In a population of mixed
footprints it returns the root-mean-square footprint, which is what σ_c needs. Its standard error is a bootstrap over
clusters (200 draws [ad hoc]).

| Situation | ρ used (primary) | Sensitivity | What the table says |
|---|---|---|---|
| ≥ 2 labels per cluster on average and ≥ 10 clusters | the pool of every cluster | 25 nm | "pool of every cluster (upper bound: merged clusters not excluded...)" |
| < 2 labels per cluster on average | **25 nm** | **0 nm** | "default 25 nm upper bound (not identifiable...)" |
| no frames, so no events | 25 nm (Σw² = 1, one label) | 0 nm | "default 25 nm upper bound (no events...)" |
| fewer than 10 usable clusters | 25 nm | 0 nm | "... (too few clusters to pool)" |

- **2 labels per cluster** [user's decision 1a; own simulation: with one label the pool gives 29 ± 27 nm on a truth of
  0]. The labels per cluster are the mean events per cluster / c_label, so this guard uses the preset c_label until it
  is calibrated.
- **25 nm as primary, 0 nm as sensitivity** when the footprint cannot be measured [user's decision 3a]. This is a
  **decision, not a measurement**: 25 nm is an upper bound (the reference paper's median r_eff, which is inflated by
  the precision), the only choice that cannot call a pair "not eclipsed" by underestimating its error; in the
  one-label regime it costs power (0.42 against 0.67 in the simulation).
- **10 clusters** [ad hoc]: below that the moment estimate is too noisy to call a measurement.

## 5. Merged clusters and the pool (decision 4)

A clustering method sometimes links two real clusters into one. Such a cluster has a huge spread along the membrane
and inflates the pool: under DBSCAN 25/10 the pool of every cluster reads **+10 to +12 nm** above the true footprint
(table below). Until the user accepts a criterion that excludes them, the pool of every cluster is used as an **upper
bound**, labelled so [decision 4b].

**The criterion** (`merged_cluster_flags`) [decision 4a]: the cluster's emission events are sorted along the
membrane and split in two contiguous groups so that the between-group sum of squares is largest (the exact
one-dimensional two-means split; each group ≥ 2 events [ad hoc]); the cluster is flagged when the distance between
the two groups' means exceeds

    3.29 · √2 · √(σ_link² + ρ_ref²/4) = 74.5 nm   (σ_link 10 nm, ρ_ref 25 nm)

the separation that two labels of ONE cluster with the upper-bound footprint exceed 0.1 % of the time [derivation:
3.29 is the two-sided 99.9 % normal quantile]. This is the "internal spread incompatible with one cluster" written
on the modes: the bound comes from the footprint model, not from ε, because ε is a linking distance, not a size.

**Validation** [own simulation, `eclipse_sim.py`, experiment `merged`: 40 synthetic axons of two rings per scenario,
62 clusters per axon, DBSCAN 25/10 on the localizations, a detected cluster "merged" when two true clusters each give
it ≥ 25 % of their localizations]:

| Scenario | true RMS ρ | flagged singles (true / DBSCAN) | merges detected | pool, every cluster | pool, merged excluded |
|---|---|---|---|---|---|
| ρ 25 nm, ~3 labels | 25.0 | 0.0 % / 0.0 % | 62 % of 164 | 37.2 ± 6.8 | **26.6 ± 3.7** |
| mixed 0/25/40 nm, ~3 labels | 22.6 | 0.3 % / 0.0 % | 60 % of 144 | 32.3 ± 9.1 | **22.8 ± 4.4** |
| ρ 40 nm, ~3 labels | 40.0 | 0.8 % / 0.2 % | 56 % of 269 | 49.9 ± 10.0 | **36.2 ± 4.4** |
| ρ 0, one label | 0 | 0 % / 0 % | — (11 merges) | not identifiable | not identifiable |

(mean ± sd over axons; the pool of the true clusters, without any clustering, reads 26.2, 23.8 and 39.5 nm.)

Why 3.29 and not another level [own simulation, the `merged_z_sweep` of the same script]: at 2.576 (99 %) the
criterion catches 77-91 % of the merges but also the widest genuine clusters, and the pool falls **below** the truth
(23.8 / 17.6 / 32.7 nm: −1.2 to −7.3 nm); at 3.89 (99.99 %) it misses most merges and the pool stays high (31.4 /
27.1 / 41.0 nm: +1.0 to +6.4 nm); at 3.29 the residual is −3.8 to +1.6 nm, inside the pool's own ±4 nm. A simple trim of the largest
contributions had failed the other way in the research (P-R61 5.3).

The pool without the flagged clusters is computed and exported (`rho_pooled_excluding_merged_nm`) and switched on by
`ToleranceInputs(exclude_merged_from_pool=True)` once the user accepts it; the switch is recorded in `rho_source`.

## 6. The rules on synthetic rings

Through the program's own `arc_eclipse_test`, the same rotation draws for every rule [own simulation,
`eclipse_sim.py` experiment `rules`: 100 synthetic axons per cell, B = 199, nominal level 0.05; with 100 axons the
standard error of a rate is 0.02 near 5 % and 0.05 near 50 %]:

| Cell | scalar 31 nm | per-pair | wide scalar 80 nm [ad hoc comparator] |
|---|---|---|---|
| no columns, mixed footprints (true clusters) | 0.02 | 0.03 | 0.03 |
| no columns, one label, ρ 0 | 0.06 | 0.01 | 0.01 |
| no columns, ρ 40 nm | 0.03 | 0.03 | 0.03 |
| no columns, mixed, DBSCAN 25/10 | 0.06 | 0.02 | 0.05 |
| 30 % columns, mixed (true clusters) | **0.63** | **0.55** | 0.09 |
| 30 % columns, mixed, DBSCAN 25/10 | **0.33** | **0.38** | 0.12 |

The false-positive rate stays at the nominal level for every rule; the two precision rules are equivalent in power
(the research: 0.56 / 0.53 with true clusters, 0.38 / 0.39 with DBSCAN) and a wide tolerance loses most of it. The
benefit of the per-pair rule is not power: each pair is judged with its own error, and the tolerance adapts by itself
when clusters are heterogeneous. Effect size: with 31 nm the excess of the matched fraction over its null is about
0.55 · f (P-R61 5.4), so a fraction of columns needs a calibration curve, not E alone.

## 7. Any clustering method

Everything takes a cluster as the indices of its localizations, from any method's labels
(`clusters_from_labels`), with the events of those localizations when frames exist. `build_rings(...,
cluster_method=..., cluster_params=...)` builds the rings with any method of `tools/clustering_strategies.py` (None:
DBSCAN with the pre-specified ε / min_samples, exactly as before). The plug-in `"eclipse"` of `tools/multimethod.py`
gives, per method, the readouts `eclipse_*`: the clusters' median σ_c and `centroid_r95_nm`, the τ derived from that
method's clusters (2.77 × median σ_c, the P-R4 comparator), the footprint and its source, the labels per cluster, the
merged-looking clusters, and — when the integration passes the whole ROI — the arc test under both rules on rings
built with that method (slow methods refused unless allowed).

## 8. What is exported (decision 5a)

- **Per cluster** (`CLUSTER_COLUMNS`): ring, cluster, label, n_locs, n_events, sum_w2_hat, n_label_hat, sigma_c_nm,
  sigma_c_sensitivity_nm, centroid_r95_nm, r_eff_nm (labelled: grows with N_loc, not the footprint, in no tolerance),
  sigma_c_source (calibrated / preset / no events), s_t_nm, mode_separation_nm, merged_suspect.
- **Per matched pair** (`PAIR_COLUMNS`), under each rule: ring_a, ring_b, rule, role (primary / exploratory),
  cluster_a, cluster_b, d_nm and **t_ij_nm** (the tolerance it was admitted with).
- **Per axon** (`AXON_COLUMNS`): the primary rule; rho_pooled_nm with its bootstrap SE, rho_pooled_excluding_merged_nm,
  rho_source, rho_used_nm, rho_sensitivity_nm, the clusters pooled and flagged; n_label_mean_hat, λ; c_label and its
  source and date; σ_loc and its source; σ_link and its source; σ_reg; tau_scalar_nm and tau_derived_axon_nm; the
  median σ_c; and for each rule the joint test: pairs, T, E, the chance rate E*, the excess, z_A and the Monte Carlo
  p (not calibrated), plus the mean t_ij of the per-pair rule.

Every column has a one-line description (`describe()`).

## 9. Limits

- The measurement model is the project simulator's (a disc of epitopes, a fixed Gaussian linkage, geometric
  blinking); stacks with several secondaries per primary would underestimate σ_c by 7-30 % in this formula (P-R75).
- The event linking is assumed correct; linking errors change "events" and with them Σw².
- Until c_label and σ_link are calibrated, σ_c is "by preset" and the per-pair rule stays exploratory.
- The test is somewhat conservative (≈ 3 % at a nominal 5 %), as already known for the rotation null.
- Everything above is synthetic. The first real number worth looking at is the distribution of events per cluster:
  it says which label regime a sample is in.

## References

- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nat Methods* 8:1027-1036.
- Früh SM, Matti U, Spycher PR, Rubini M, Lickert S, Schlichthaerle T, Jungmann R, Vogel V, Ries J, Schoen I (2021).
  Site-Specifically-Labeled Antibodies for Super-Resolution Microscopy Reveal In Situ Linkage Errors. *ACS Nano*
  15(7):12161-12170. doi:10.1021/acsnano.1c03677
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of βII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic
  nerve axons. bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1
- Huang B, Wang W, Bates M, Zhuang X (2008). Three-dimensional super-resolution imaging by stochastic optical
  reconstruction microscopy. *Science* 319:810-813.
- Phipson B, Smyth GK (2010). Permutation P-values should never be zero. *Stat Appl Genet Mol Biol* 9:39.
- Rényi A, Sulanke R (1963). Über die konvexe Hülle von n zufällig gewählten Punkten. *Z Wahrscheinlichkeitstheorie*
  2:75-84.
- Wang Y, Schnitzbauer J, Hu Z, et al. (2014). Localization events-based sample drift correction for localization
  microscopy with redundant cross-correlation algorithm. *Opt Express* 22:15982-15991.
