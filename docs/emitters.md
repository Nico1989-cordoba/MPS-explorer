# Emitters (BaGoL-like reimplementation): from localizations to emitters, and from emitters to clusters

> **Resumen (español).** Capa experimental «Events → Emitters» del análisis por axón. Agrupa las localizaciones
> repetidas de cada marca fluorescente en un *emisor* con el modelo bayesiano de Fazel et al. (2022), y después une
> los emisores a menos de `r_mol` (50 nm por defecto, derivado) en un clúster. Lecturas nuevas: emisores por clúster
> (mediana, q25-q75), emisores por µm, emisores con localizaciones, precisión antes/después, y la ξ usada con su
> origen. Es lenta (segundos por cada 1000 localizaciones), está apagada por defecto y nunca corre sola. **Los
> emisores son pilas de marcas fluorescentes, no proteínas.** Desde F5b: asignación Dirichlet por defecto (no
> parte los emisores brillantes), lecturas sobre emisores *no singletes* (los singletes, casi siempre fondo
> aislado, se informan aparte), ξ aprendida sobre no singletes, chequeos de §3.7 (calibrar / ξ mezcla / precisión
> sub-reportada con NeNA) y criterios de aceptación C1-C6 (sección 8).

Method name, everywhere it appears (interface, tables, files): **Emitters (BaGoL-like reimplementation)**. It is a
Python reimplementation of the published model (Fazel et al. 2022) in `tools/cluster_emitters.py`, written from the
paper, its Supplementary Notes and the authors' MATLAB code read as a specification (LidkeLab/BaGoL, MIT licence,
copyright the LidkeLab authors). **It is not the authors' code**, and its results are not those of BaGoL itself; the
known deviations are listed in the module docstring and in section 6 below.

## 1. What an emitter is — and what it is not

**Emitters are fluorescent label stacks, not proteins.** In dSTORM one emitter is whatever blinks as one point: a
primary antibody with one or more secondary antibodies, each carrying ~1-2 dyes. Several dyes on one antibody blink
as one emitter in photoswitching buffer (literature: Helmerich et al. 2020). One protein can carry zero, one or
several such stacks (two epitopes per spectrin tetramer, a labelling efficiency below 1, a variable number of
secondaries), so the number of emitters is *not* a count of molecules. The model groups localizations; it never
says which protein they came from.

ξ, the distribution of localizations per emitter, is the model's prior; its values, sources and the derivation of
its default are in [`emitters_xi.md`](emitters_xi.md).

## 2. Where the layer sits, and when it runs

```
localizations of the axial slab
   -> events: localizations linked across frames, as the DNA-PAINT panel  [this layer]
   -> Events -> Emitters (RJMCMC, per pre-cluster of 200-nm subregions)   [this layer]
   -> emitters -> clusters: single linkage within r_mol                     [D-B5 (a)]
   -> the usual pipeline: curation, contour, perimeter, occupancy, 1NN
```

* It appears in the clustering method list (**Clustering method: emitters_bagol_like**) only with *Experimental
  clustering methods* ticked; DBSCAN stays the default (D10).
* Nothing runs when a file is loaded (D12). Choosing the method shows its box and runs
  nothing (it is the only method that does not re-run on choice): the layer runs when the user presses *Run with these
  values*, or ticks it in *Compare methods…* and presses *Run*.
* It is slow (section 5). The batch never runs it on its own: with the method saved and the batch window's
  **Slow methods (Emitters)** box off, every file is reported as *not analysed* with the reason; ticking the box
  runs it. *Compare methods (slow)* in the batch skips it the same way. A batch that runs Emitters asks the
  **imaging buffer** once for the run, with no default (F5c): without it the batch does not start, and every row
  records it (`emitters_buffer`).
* **The model groups events, not raw localizations.** BaGoL takes "frame connected" localizations (Fazel et al.
  2022; `BaGoL.m`), and ξ (`emitters_xi.md`) counts events: one localization per switching cycle after linking, by
  construction. So the slab's localizations are first linked across frames exactly as the DNA-PAINT panel links them
  (`tools.mps_paint.build_events`, Picasso's chain rule; the radius is `LINK_RADIUS_IN_PRECISIONS` = 3 × the median
  lateral precision of the slab and one dark frame is allowed, as in `paint_report`; both editable in the panel and
  saved in the settings and in `cluster_params`). Each event enters the model as its precision-weighted mean
  position (weights 1/lp², lp = (σx + σy)/2, as `build_events`) with the precision of that mean,
  √(Σ w² σ²)/Σ w per axis [derivation], which is 1/√(Σ 1/lp²) for isotropic precisions. Every localization then
  takes the emitter of its event. Unlike the DNA-PAINT kinetics, events bound in the first or last frame are kept:
  that censoring matters for event lengths, not for positions, and dropping them would drop localizations. A file
  without frame numbers runs on raw localizations: the results table and *Analysis notes* say so and the three
  event columns stay empty.
* The slab's own precisions are used: the file's `lpx`, `lpy` (Picasso, converted to nm) are passed to the model
  for every localization (`EmittersBagolLikeStrategy.set_localization_precision`). Failed values (0, NaN) are
  replaced by the median of that axis [ad hoc] and counted. A file without precision columns falls back to
  `sigma_nm` (12 nm [ad hoc: the benchmark's σ_loc]) for every localization, and the results table says so.

## 3. Parameters shown in the panel

| Parameter | Default | Origin |
|---|---|---|
| imaging buffer | **none**: chosen for each file | user, per file (F5c, 2026-10-19: the lab images in both buffers). MEA = GLOX + 10 mM MEA, βME = GLOX + 140 mM β-mercaptoethanol, the two columns of Dempsey et al. (2011) Table 1, p. 1030 [lit]. It sets the literature ξ and is recorded with every source; loading another file empties it, and while it is empty *Run with these values* is disabled with «Choose the imaging buffer (MEA or βME) for this file» |
| ξ source | `literature` | decision D-B3 (c): literature, learned or calibrated. Default literature since F5b [simulation, this project]: with the Dirichlet allocation the positions do not depend on the source (F4), while the learned ξ depends on what one event is in the file (section 7); learned is an option, checked by section 5. The source no longer names a buffer (F5c); the source may be remembered between files, the buffer never is |
| dye (channel 1) | Alexa Fluor 647 | the channel's dye; Dempsey et al. (2011) Table 1 transcribes AF647 and AF568 only |
| k (gamma shape of ξ) | 1 | literature: Fazel et al. (2022), Methods, η = 1 for dSTORM |
| calibrated mean | — | user: localizations per emitter measured on a control; only with source `calibrated` |
| r_mol [nm] | 50 | derivation (section 4); 0 = no linking (every emitter is a cluster) |
| event link radius [nm] | auto = 3 × median lateral precision | derivation, as the DNA-PAINT panel (`paint_report`) |
| max dark frames | 1 | the DNA-PAINT panel's default (`DEFAULT_MAX_DARK_TIME`, Picasso's *max dark time*) |
| has_locs_min | 0.5 | simulation, this project (F3 session 3); changes only the "with localizations" flag |
| se_adjust_nm | 0 | code: BaGoL `SE_Adjust`, added to every precision; literature: Fazel et al. (2022) raised the precisions of their real data by 0.75-2 nm. The ξ check suggests a value when NeNA / precision > 1.2 (section 5) |
| n_burnin / n_trials | 2000 / 3000 | code: LidkeLab/BaGoL, `BaGoL.m` defaults |
| roi_size_nm, overlap_nm, overlap_sigma_factor | 200, 20, 4 | code (`BaGoL.m`) / derivation: overlap ≥ 4 σ₉₅ so an emitter on a subregion border is seen whole by both |
| processes, seed | 2, 0 | ad hoc: this laptop's budget; fixed seed for reproducibility |

Not shown in the panel (code and command line only): `allocation` = `dirichlet` (default; `uniform` = the original
code, also implied by `matlab_compat`; section 6) and `xi_counts` = `nonsingleton` (default; `occupied` = the
original code's count; section 5).

The three ξ sources (each needs the buffer, and records it):

| Source | ξ | Fixed or learned |
|---|---|---|
| `literature` | Gamma(k, mean/k), mean = switching cycles of the dye in the buffer chosen (Dempsey et al. 2011, Table 1, p. 1030 [lit]: MEA AF647 14, AF568 7; βME AF647 26, AF568 52) | fixed |
| `learned` | hierarchical: starts at the dye's literature value **in the buffer chosen** (before F5c it always started at MEA), global Gamma(α = 1, β = 50) hyperprior updated every 10 steps (Fazel et al. 2022, Methods; `BaGoL.m`) | learned; the column reports k × θ of the posterior means |
| `calibrated` | Gamma(k, mean/k) with the mean typed by the user; the buffer is recorded only | fixed |

The sources of F5b, `literature_MEA` and `literature_bME`, are still read (settings files, the `cluster_params` of
older tables) as `literature` plus that buffer. When a new file is loaded the program drops that implied buffer too:
the buffer of each file is chosen again.

Dempsey's numbers are a lower bound measured under other conditions (see `emitters_xi.md`, section 6): read the
literature sources as a sensitivity range, not as the truth for this sample.

## 4. From emitters to clusters: r_mol

Decision D-B5 (a): emitters within `r_mol` of each other are linked (single linkage, connected components of the
graph "distance ≤ r_mol") into one cluster. The default, 50 nm, is a derivation: the epitopes of one spectrin
tetramer lie within ~10-15 nm, and a primary + secondary antibody stack displaces the dye by up to ~17.5 nm
(literature: Zwettler et al. 2020) — two stacks pointing in opposite directions put two emitters of one molecule at
most ≈ 15 + 2 × 17.5 ≈ 50 nm apart, while neighbouring tetramers are 100-260 nm apart (literature: Gazal et al.
2026). It is editable; 0 turns the linking off.

The number of emitters per cluster is therefore a **readout**, never a constant such as "three emitters = one
molecule": the degree of labelling of the secondaries is not published and the labelling efficiency is below one.

## 5. Readouts and the axon table (v11, v12)

Shown under the results table when the layer ran, and written **at the end** of every axon row (axon tables v11,
and three more columns at the end in v12, F5c; no earlier column moves). With any other method the twenty cells
are empty.

**Singletons** (decision of the user 2026-10-17, F4 (d)). An emitter that explains fewer than 1.5 events
(posterior mean; rounded, one event) is a *singleton*. With an isolated background of single localizations — on
the synthetic grid of F4 as many background events as true label stacks — most singletons are that background:
legitimate one-event emitters of the model (P(N = 1) = 7 % for an exponential ξ of mean 14), but not label stacks
of the structure. So the per-µm and per-cluster readouts count **non-singleton** emitters only, singletons are
reported apart (their own row and column), and the learned ξ is learned on non-singletons
(`xi_counts = 'nonsingleton'`: only the kept emitters with ≥ 2 events in each state, and no one-event pre-cluster,
enter the hierarchical update). Cost [derivation]: for an exponential (geometric) ξ the mean of the counts ≥ 2 is
the mean + 1 (memorylessness), so the learned mean reads about +1 event high (+7 % at 14). Origin: with all
emitters counted, the learned ξ on the F4 grid was the mixture of ~90 background singletons and ~70 real stacks of
~40 events, shape k 0.34-0.55 [simulation, this project, F4 part B].

| Column | Meaning |
|---|---|
| `emitters_method` | "Emitters (BaGoL-like reimplementation)" when the layer ran, else empty |
| `emitters_n` | emitters of the model in the slab (K of the MAPN estimate, emitters without localizations included) |
| `emitters_n_with_locs` | emitters whose posterior mean of explained events is ≥ has_locs_min (0.5) |
| `emitters_n_nonsingleton` | emitters that explain ≥ 1.5 events: the ones the readouts below count |
| `emitters_n_singletons` | emitters with localizations that explain < 1.5 events (one event), reported apart |
| `emitters_per_um` | distinct non-singleton emitters in the kept clusters / the perimeter (the denominator of K/µm) |
| `emitters_per_cluster_median`, `_q25`, `_q75` | distinct non-singleton emitters per kept cluster (clusters of singletons only left out and counted in the results row): median and quartiles |
| `emitters_xi_mean` | mean events per emitter of the ξ used (learned: k × θ of the posterior means, learned on non-singletons) |
| `emitters_xi_source` | `literature`, `learned` or `calibrated` (tables before v12: `literature_MEA`, `literature_bME`) |
| `emitters_precision_nm_before` | median over the slab of (σx + σy)/2 of the localizations |
| `emitters_precision_nm_after` | median over the emitters with localizations of (σx + σy)/2 of their posterior position (sd of the chain) |
| `emitters_r_mol_nm` | the radius emitters were linked into clusters with |
| `emitters_n_events` | events the model grouped (frame-linked localizations); empty without frame numbers |
| `emitters_link_radius_nm` | the radius the events were linked with |
| `emitters_max_dark_frames` | dark frames allowed inside one event |
| `emitters_buffer` | v12: the imaging buffer chosen for the file (`MEA` or `bME`) |
| `emitters_nena_over_sigma` | v12: check T4, NeNA of the slab / (median reported precision + `se_adjust_nm`); empty when not measured |
| `emitters_precision_flag` | v12: `ok`, `under-reported` (ratio > 1.2) or `not measured` |

**ξ check (section 3.7 of the plan, thresholds of F4; decision of the user 2026-10-17, F4 (c)).** After every run
the results table shows a row *xi check* and, when a check fires, *Analysis notes* repeats its sentence
(`tools.emitters_prior.xi_checks`):

| Check | Fires when | What it means | Origin |
|---|---|---|---|
| T1 *calibrate* | learned mean (on non-singletons) outside [½ × the dye's smallest tabulated cycles, 2 × 4.5 × the largest] × events per cycle × p: [7, 234] for AF647 | not even the stretched literature, times the dyes of one label stack (DOL 1.5 × 3 secondaries), explains ξ: calibrate (`docs/calibration_xi.md`) | literature: Dempsey et al. 2011; the stack factor from the project's generator; the bounds ad hoc |
| T1' *xi mixture* | learned shape k outside [0.5, 2] | the learned ξ is not one population (background singletons give k < 0.5, split emitters k ≈ 2): report it, keep a literature ξ for positions | simulation, this project (F4) |
| T4 *precision under-reported* | NeNA of the slab / median reported precision > 1.2 | the model splits emitters when precisions are too small (+53 % with 20 % under-reported, F4; +28 to +63 % with the Dirichlet allocation, F5b): set `se_adjust_nm` ≈ NeNA − median precision (suggested) and run again | simulation, this project; literature: Fazel et al. 2022 Methods; 1.2 ad hoc |

**T4 before the run, and after it (F5c).** T4 is no longer only a sentence after a run of minutes:

* (a) *before* the Emitters run, the slab's NeNA is compared with its median reported precision (plus any
  `se_adjust_nm` already set). When the ratio is above 1.2, the interactive program shows the measurement and the
  suggested correction, `se_adjust_nm` = NeNA − median precision, with **Apply and run** (the value is set before
  the run and recorded in `cluster_params`), **Run as is** and **Cancel**. The batch, the tests and any offscreen
  session never show it: the run goes on as reported and *Analysis notes* carries the measurement and the
  suggestion. With fewer than 500 next-frame pairs nothing is asked;
* (b) *after* a run where T4 fired, **Apply suggested correction and re-run** (beside *Analysis notes*) sets
  `se_adjust_nm` to the suggestion and re-runs;
* (c) the axon table carries the ratio and the flag (`emitters_nena_over_sigma`, `emitters_precision_flag`).

Origins: the 1.2 threshold [ad hoc: the 20 % under-reported case of F4], the 500 next-frame pairs NeNA needs and the
effect on the emitters (+53 %, F4; +28 to +63 % with the Dirichlet allocation, F5b) [simulation, this project, F4
section 3.7], the correction as the difference [derivation: `se_adjust_nm` is added to every precision, so the
median then equals NeNA].

T1 and T1' apply only to the `learned` source. NeNA is measured on the slab's raw localizations
(`tools.mps_quality.nena`, next-frame neighbours) and needs ≥ 500 next-frame pairs; a slab often has fewer, and
then the row says so: measure the whole-field NeNA (*Pick boundary* dialog) and compare by hand. Any absolute count
(emitters per molecule, molecules per µm) still needs a calibration (T3), and T2 (the per-cluster readout changing
category between ξ sources) is read by the user from two runs.

The method's parameters (including ξ source, dye, k, r_mol) are also in `cluster_params`, and `cluster_scale_nm`
is r_mol (the separation the clusters were built with; the edge-curation margin is max(eps, r_mol)).

**Files.** *Export axon* writes, beside the axon table, `<table>_emitters_<axon_id>.csv` (one row per emitter:
position, σ, n_locs, has_locs, P(occupied), pre-cluster, posterior K, ξ, method) and
`<table>_emitter_labels_<axon_id>.csv` (`x, y, label` for every localization of the slab, in nm, with the cluster of
each). The second file is in the format of the *External labels* method: re-analysing the same slab with it, and its
scale declared as r_mol, gives the same clusters and the same K/µm (tested in `test_emitters_ui.py`).

**Time.** A lower bound: about 19 s per 1000 slab localizations with 2 processes and the default chains
(simulation, F5b: the golden's simulated axon, 4,314 slab localizations → 2,821 events in 602 pre-clusters, ξ
literature MEA, 2000 + 3000 steps, 140 s with 1 process = 32.5 s per 1000; the 1.7 × speed-up of 2 processes is the
one measured in F3). The cost goes with the number of pre-clusters × steps (one Python loop per pre-cluster) more
than with the localizations: a dense synthetic field of 9,842 localizations in 122 pre-clusters ran at 2.8 s per
1000 with 2 processes (F3). On top of the layer comes the rest of the analysis, which grows with the number of
clusters: when the emitters leave many small clusters it dominates (same axon, learned ξ, 300 + 300 steps: 20 s in
the layer, 680-730 s for the whole analysis, about 1,000 kept clusters, of which 646 s were the perimeter's 2-opt
from every start, `tools.mps_geometry.reconstruct_perimeter`). The interface's estimate is a lower bound. Above about 500
kept clusters the analysis asks before building that perimeter ([perimeter_time.md](perimeter_time.md)).

**In the background (F5d).** In the MPS analysis window, *Run with these values*, *Apply suggested correction and
re-run* and any re-run with a slow method (Emitters, OPTICS, voronoi_csr: `tools.cluster_method_ui.is_slow`) run the
analysis in a worker thread: a line under the window's banner says what runs and for how long, the controls that
would start another analysis are greyed out, and the window keeps responding. The dialogs before a run (imaging
buffer missing) show before it starts; the two questions in the middle of a run (check T4 before the layer, the
perimeter with many clusters) are relayed to the window and asked there. *Cancel* is cooperative
(`tools/mps_cancel.py`): the analysis stops at its next check — between pre-clusters with one process, within
0.5 s with two (the worker processes are then terminated), at every start of the perimeter's 2-opt, at every
randomization iteration; one clustering fit or the occupancy finishes first. Whatever a cancelled run returns is
not used: the analysis shown stays. Closing the window cancels the run the same way. With a slow method saved, the
main window's *cluster Ch1* and *MPS analysis* hand the run to that window too (interactive sessions only: the
batch and the golden reference stay synchronous, and DBSCAN is unchanged).

## 6. Known deviations from the original code, and the original's Death rule

The full list is in the docstring of `tools/cluster_emitters.py`. Two change results.

**The allocation prior (decision of the user 2026-10-17, F4 (a)).** The original code allocates every localization
to one of the K emitters with equal probability, p(Z | K) = K^−N (Fazel et al. 2022, Supplementary Note 1, eq. 8).
That is the exact conditional only for a Poisson ξ. For the gamma ξ used for dSTORM (shape k = 1, exponential
blinking counts) the exact conditional is the Dirichlet-multinomial (weights w ~ Dirichlet(k)) [derivation, F4 and
its independent review]. With equal weights and heavy-tailed counts, a second emitter on a bright one pays off
(those localizations go from weight 1/K to 2/K), so every emitter with more than ≈ 2 × mean(ξ) events is split into
≈ N/mean(ξ) pieces in dense data: +17 % emitters, 24 % of the true emitters split, on a dense synthetic field
(720 emitters, σ 12 nm, exponential counts of mean 14); with the Dirichlet allocation −2 % and 7 %, and the
learned ξ recovers mean and shape [simulation, this project, F4]. `allocation = 'dirichlet'` is the default for a
gamma ξ; `'uniform'` (the original) only when asked for or with `matlab_compat`. The sampled target is a hybrid:
the Dirichlet-multinomial allocation with the paper's continuous p(N | K) = Gamma(N; Kk, θ); its residual against
the exact negative-binomial model is small (total variation 0.009-0.058 in the number of occupied emitters, against
0.048-0.234 for the uniform allocation, by enumeration; independent review of F4). `prune_empty` with the Dirichlet
allocation warns: the pruning deletes the empty emitters this prior makes frequent and biases K down.

**The Death rule.** The original code's **Death move
of the original code is not the inverse of its Birth move.** The code accepts a death with
`Pk(K−1)/Pk(K) · L'/L`, without the factor `Area · q(μ_j)` that its own Birth ratio divides by
(`1 / (Area · q(μ*))`). The paper defines the Death acceptance as the inverse of the Birth ratio (Fazel et al.
2022, Supplementary Note 1, eq. 27, the inverse of eq. 26), which is what detailed balance requires; no prior on
the positions makes both of the code's rules valid at once. Where a surplus emitter sits on the localizations,
`Area · q` is about 3-6, so surplus emitters die that many times too rarely and K is biased upwards. Measured
against the exact posterior (simulation, this project, F3 session 2: two groups 10 nm apart, K ≤ 2, Poisson
λ = 3, exact P(K = 2) = 0.728): this implementation 0.725; the code's Death rule alone 0.823; the code complete
(its Death rule and its pruning of empty emitters) 0.804. Here the Death ratio follows eq. 27; the original
behaviour is available only for comparisons (`fit_emitters(..., matlab_compat=True)`).

## 7. Limitations

* 2-D in the slab (decision D-B6): z of an emitter is the weighted mean of its localizations, not inferred.
* With the original's uniform allocation the model over-split dense regions (≈ 850 emitters for 720 true ones,
  learned ξ biased low; F3). F4 traced it to the allocation prior; the Dirichlet default removes it (section 6).
  What remains in dense data: emitters with ≤ 3 events next to a brighter one are absorbed (41-49 % of the true
  emitters with ≤ 7 events were missed on the dense F4 field, with either prior): a limit of the data.
* Under-reported precisions still make the model split emitters (C6 below is not met on that side): check NeNA.
* ξ is not identifiable from a single isolated structure; learn it on a whole axon, not on one cluster.
* The learned ξ is "events per emitter that the model can resolve", and it follows what one event is in the file.
  On the golden's simulated axon every fluorophore blinks once (geometric run of mean 2.5 consecutive frames,
  ~20 fluorophores per cluster within ~20 nm): the original's configuration (uniform allocation, all occupied
  emitters) learned 1.7 events per emitter (0.9 in F5) and the default one (Dirichlet, non-singletons) 3.9 (k 1.6),
  with ~1,000 singletons out of ~1,450 emitters either way, and check T1 fires (< 7). Both are honest readings of a
  file in which one event is one fluorophore, not of a dSTORM label stack; the number of non-singleton emitters
  hardly moves (441, 444, 446 with learned Dirichlet, learned uniform, literature MEA). That is why the default
  source is the literature (simulation, F5b).
* The results are stochastic: same seed, same result (independent of the number of processes); another seed,
  slightly different emitters.

## 8. Acceptance criteria (C1-C6)

Decision of the user 2026-10-17, F4 (b): the three criteria of the plan (ad hoc) are replaced by these. Fast
synthetic versions run as regression tests in `test_emitters_acceptance.py` (≈ 35 s, one process); the values
below were measured on larger fields in F4 and on the test fields in F5b.

| # | Criterion | Measured | Origin | Test |
|---|---|---|---|---|
| C1a | Poisson regime (the paper's): Jaccard (Hungarian matching, cost ≤ 3 × the mean returned precision) ≥ 0.80 with emitters ≥ 3 σ apart and λ ≥ 14 | 0.85 (dense F4 field); 0.91-0.95 (test fields) | literature: Fazel et al. 2022, Fig. 4 and Suppl. Fig. 16 | yes |
| C1b | dSTORM regime (exponential counts): recall ≥ 0.85 for emitters with 8-14 events and ≥ 0.95 for ≥ 15; Jaccard ≥ 0.70 for emitters with ≥ 8 events; those with ≤ 7 reported apart as recall, without a threshold | recall 0.95-1.0 / 0.97-1.0; Jaccard (all emitters) 0.71-0.81 | simulation, this project; 0.70 ad hoc (15 % under the paper) | recall yes |
| C2 | counting: \|emitters / truth − 1\| ≤ 10 % and P(bright emitter split \| N ≥ 2 × mean ξ) ≤ 20 %, ξ fixed at the truth | −2 % / 7 % (F4); +3, −11, +1 % (mean −2 %) / 0-5 % (test fields) | literature: Bohrer et al. 2021 counting error; Fazel Suppl. Fig. 5; simulation, this project | yes (mean of 3 fields) |
| C3 | learned ξ: mean within ±25 % and shape k within [0.5, 2] × the true k, on isolated clusters (or the whole field when the model is consistent) | mean −9 %, k 1.08 (test field); +0.3 %, k 0.97 (dense F4 field) | literature: Fazel et al. 2022, Methods (learn on sparse regions); simulation, this project | yes |
| C4 | ≤ 30 min per 10⁵ events with 2 processes; 24 h as the hard cap; the interface marks the method slow | section 5 (*Time*) | measured | no |
| C5 | against the MATLAB code (if ever run): same mode of K in ≥ 90 % of subregions, compared with `matlab_compat=True` | not evaluated | derivation | no (`matlab_compat` tests only) |
| C6 | robustness: with the model's σ 0.8-1.2 × the real one, \|emitters / truth − 1\| ≤ 25 %; the panel shows NeNA against the median reported σ, and `se_adjust_nm` is editable | 1.2 ×: −13 to −23 % (met); **0.8 ×: +28 to +63 % (not met)**, also with the Dirichlet allocation | literature: Fazel et al. 2022 (precisions raised 0.75-2 nm); simulation, this project | 1.2 × yes; 0.8 × covered by the T4 check |

## References

- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768 (Table 1,
  p. 1030).
- Fazel M, Wester MJ, Schodt DJ, Cruz SR, Strauss S, Schueder F, Schlichthaerle T, Gillette JM, Lidke DS, Rieger B,
  Jungmann R, Lidke KA (2022). High-precision estimation of emitter positions using Bayesian grouping of
  localizations. *Nature Communications* 13:7152. doi:10.1038/s41467-022-34894-2 (Methods; Supplementary Note 1,
  eqs. 26-27). Code: https://github.com/LidkeLab/BaGoL, MIT licence.
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Reference study of this project (spacing of neighbouring spectrin tetramers).
- Helmerich DA, Beliu G, Sauer M (2020). Multiple-labeled antibodies behave like single emitters in photoswitching
  buffer. *ACS Nano* 14(10):12629-12641. doi:10.1021/acsnano.0c06099.
- Zwettler FU, Reinhard S, Gambarotto D, Bell TDM, Hamel V, Guichard P, Sauer M (2020). Molecular resolution imaging
  by post-labeling expansion single-molecule localization microscopy (Ex-SMLM). *Nature Communications* 11:3388.
  doi:10.1038/s41467-020-17086-8 (linkage error of primary + secondary antibodies, ~17.5 nm).
