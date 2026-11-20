# Clustering benchmark harness (synthetic data only)

`tools/cluster_benchmark.py` compares clustering strategies on synthetic SMLM
scenarios with known ground truth, in the spirit of the multi-algorithm
framework of Nieves et al. (2023) but with scenarios built for this project
(transverse sections of myelinated axons, βII-spectrin, dSTORM blinking), because
no standard benchmark covers all five families (density, tessellation,
graph/topological, model-based, machine learning) plus the non-assignative
spatial statistics on this kind of data.

Everything here is synthetic (rule R8): no real axon file enters the benchmark or
its tests. Results from it are cited as "own simulation".

## Scenarios

The generator (`generate(Scenario, rng)`) ports the ground-truth model of the
clustering study (`SIM_CLUSTERING_REPORT.md`, section 2.1, own simulation
2026-10-15):

| element | model | origin |
|---|---|---|
| geometry | `ring`: closed ellipse of perimeter L ~ U(10, 30) µm, axis ratio U(1, 1.6); `window`: flat square of side `window_um` | ad hoc (perimeter range of the 18-axon set) |
| molecules | arc positions with hard core 0.4 × mean spacing; mean spacing 200 / 100 / 40 nm (sparse / medium / dense) | sparse: literature, preprint (≈ 5 clusters/µm, Gazal et al. 2026); medium / dense: ad hoc stress levels; hard core ad hoc (as the project simulator) |
| pairs | fraction `f_pair` (0.15) gets a partner at U(30, 60) nm | ad hoc |
| epitopes | 2 per molecule, U(5, 10) nm apart, random orientation | inference from the tetramer structure (two β C-termini); no published distance |
| labelling | each epitope labelled with probability `p_label`; linkage offset N(0, 10 nm) per axis, fixed for the movie | literature + derivation: Früh et al. (2021) report linkage errors up to ~19 nm; 19 / √3 ≈ 11 → ~10 nm per axis |
| blinking | cycles per label ~ Geometric(mean `c_label`), 14 (MEA) or 26 (βME); each cycle starts at a uniform frame of 20 000 and lasts 1-3 frames (one localization per frame) | Dempsey et al. (2011) for 14 / 26; on-time ad hoc |
| precision | N(0, σ) per axis, σ log-normal (CV 0.25) around `sigma_loc` (12 or 20 nm) | as the P-R61 generator; 12 nm ≈ NeNA of Gazal et al. (2026) |
| background | non-specific labels (blinking like a label) at `bg_labels_per_um` per µm of contour in a ±150 nm band, plus isolated localizations at `bg_locs_per_um2` | ad hoc (no laboratory measurement) |

Each localization records `frame`, `molecule_id` (≥ 0 real; −1 isolated
background; ≤ −2 non-specific label), `label_id`, `event_id` and the precision
actually used. `SyntheticSample.oracle_labels` is the perfect labelling
(molecule id, background = noise).

Presets (`python -m tools.cluster_benchmark --list`): `sparse_lab1`,
`sparse_lab2`, `medium_lab1`, `medium_lab2` (spacing 200 / 100 nm × p_label 0.5
/ 1.0, i.e. about 1 or 2 labels per molecule), `sparse_lab2_sig20`,
`medium_lab2_sig20` (σ 20 nm), `sparse_lab2_c26`, `medium_lab2_c26` (βME
blinking), `sparse_lab2_bghigh` (2 labels/µm, 120 locs/µm²),
`sparse_lab2_bgnone`, and `window_sparse_lab2` (flat window, no contour). The
default background is the study's "low" level (0.5 labels/µm, 30 locs/µm²).

Not simulated: axial mixing between rings (SCI-6), PSF overlap of simultaneous
emitters, drift, more than one fluorophore per secondary (absorbed in
`c_label`), 3D membrane curvature.

## Metrics

Molecule level, by majority matching (report section 8.1), computed by
`molecule_metrics`:

| metric | definition |
|---|---|
| `exact` | fraction of clusters whose majority molecule holds ≥ 80 % of the cluster's localizations **and** the cluster holds ≥ 80 % of that molecule's localizations ("one cluster = one molecule") |
| `merge` | fraction of non-false clusters in which ≥ 2 molecules are present (a molecule is present when ≥ 20 % of its localizations are in the cluster) |
| `split` | fraction of labelled molecules present in ≥ 2 clusters |
| `missed` | fraction of labelled molecules with < 50 % of their localizations in any cluster |
| `false` | fraction of clusters with > 50 % background localizations |
| `ari` | adjusted Rand index (scikit-learn) between `molecule_id` and the labels, on clustered non-background localizations |
| `iou_mean` | mean over clusters of \|cluster ∩ majority molecule\| / \|cluster ∪ majority molecule\| |

Pipeline readouts (`readouts`) and their bias against the oracle, paired by
replicate (report section 8.4): clusters per µm of arc (`clusters_per_um`),
median nearest-neighbour distance between cluster centres (`median_1nn_nm`),
median effective radius from the convex hull (`median_r_eff_nm`, scipy), with
`bias_<readout>_pct = 100 × (strategy − oracle) / oracle`. The 80 % / 20 % / 50 %
thresholds (`TAU_EXACT`, `PRESENT_SHARE`, `FALSE_BG_SHARE`, `MISSED_SHARE`) are
ad hoc; they are the ones the project's clustering study used.

## Running it

```
python -m tools.cluster_benchmark --scenario sparse_lab1 --strategies dbscan_25_10,hdbscan_10_5 --replicates 3 --out out_dir
python -m tools.cluster_benchmark --all-scenarios --replicates 5 --seed 1 --out out_dir
python -m tools.cluster_benchmark --scenario sparse_lab1 --strategies "dbscan_25_10,reg:ToMATo,reg:FOCAL" --jobs 3 --out out_dir
```

Outputs in `out_dir`: `benchmark_long.csv` (one row per scenario × replicate ×
strategy × parameter set, with the metrics and the fit time),
`benchmark_summary.csv` (mean and standard error over replicates),
`benchmark_summary.md` (one table per scenario) and `benchmark_config.json`
(seed, scenarios, strategy parameters and the generator constants: the run is
reproducible from it; replicate `r` of scenario `s` uses
`default_rng([seed, crc32(s), r])`, so two scenarios never share molecules or
background — before 2026-10-16 it was `default_rng([seed, r])`, which made
`sparse_lab1` and `sparse_lab2` share positions, REVIEW_D0 M2) and
`verdicts.json` (decision D11, see `docs/clustering_benchmark.md` section 5).

Strategy specs on the command line: `dbscan_<eps>_<min_samples>` (the repo's
`DBSCANStrategy`), `hdbscan_<min_cluster_size>_<min_samples>` (the repo's
`HDBSCANStrategy` when the `hdbscan` package is installed, scikit-learn's HDBSCAN
otherwise), `reg:<name>` or `reg:<name>:k=v,k=v` (a method of the registry of
`tools/clustering_strategies.py` with overrides; values are read as JSON, bare
words as strings, e.g. `reg:OPTICS:cluster_method=dbscan,eps=25`) and
`external:<csv>` (below). The oracle is always added. Specs that contain commas
go in a JSON list given with `--strategies-json`. `--jobs N` runs one process
per spec (N at a time): the labels are identical to the serial run, the times are
measured under the load of the other processes. Set `LOKY_MAX_CPU_COUNT=1`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1` so that no
method uses more cores than DBSCAN when the D11 time ratio matters.

All 11 presets × 2 replicates × 2 strategies take about 15 s on a laptop
(3 000-8 000 localizations per sample).

## Adding a strategy

Any object with `fit(xy) -> labels` (the `ClusteringStrategy` interface of
`tools/clustering_strategies.py`; labels are integers, −1 = noise, one per row
of `xy`) can be passed to `run_benchmark`:

```python
from tools import cluster_benchmark as cb
from tools.clustering_strategies import DBSCANStrategy

strategies = {
    "dbscan_25_10": DBSCANStrategy(eps=25.0, min_samples=10),
    "my_method": cb.CallableStrategy(lambda xy: my_labels(xy), "my_method", {"k": 3}),
}
rows = cb.run_benchmark([cb.PRESETS["sparse_lab2"]], strategies, n_replicates=3, seed=0,
                        grid={"dbscan_25_10": [{"eps": 20.0}, {"eps": 30.0}]})
summary = cb.summarize(rows)
cb.write_csv(rows, "long.csv"); cb.write_markdown(summary, "summary.md")
```

`grid` applies attribute overrides to a fresh deep copy of the strategy before
each `fit` (one parameter set never leaks into the next) and records every
parameter set as its own row and its own D11 entry. Strategies that need more than `xy` (frames for
temporal linking, precisions for model-based methods) are wrapped in a closure
over the sample: generate the sample with `generate`, build the strategy with
what it needs, then call `evaluate(labels, sample, oracle_readouts)` directly.

## Importing labels from another program (BaGoL, R packages, ...)

For methods that live outside Python, export the localizations, run the method,
and write a CSV with columns `x, y, label` (same coordinates in nm; label −1 =
noise). `ExternalLabelsStrategy(csv_path, tol_nm=...)` matches rows to the data
by nearest position (default tolerance 1e-3 nm, for exact round trips; set it to
the rounding of the file, e.g. 0.01 nm for 2 decimals) and returns the labels;
unmatched localizations become noise. If fewer than 99 % of the localizations
match [ad hoc], `fit` raises `ValueError` with the matched fraction instead of
returning (almost) only noise; two localizations matching the same row give a
warning. The row order does not matter. The sample to export is reproducible
from the seed:

```python
sample = cb.generate(cb.PRESETS["sparse_lab2"],
                     np.random.default_rng([seed, cb.scenario_seed("sparse_lab2"), replicate]))
# write sample.xy (and sample.frame, sample.sigma if the method needs them) -> run the method -> labels.csv
labels = cb.ExternalLabelsStrategy("labels.csv").fit(sample.xy)
metrics = cb.evaluate(labels, sample, cb.readouts(sample.xy, sample.oracle_labels, sample.arc_length_nm))
```

## Emitter-level truth (antibody stacks)

Methods such as BaGoL (Fazel et al., 2022) return **emitters**, not clusters.
An emitter is a fluorescent label stack, not a protein: one secondary antibody
with its dyes, whose dyes sit closer than the localization precision. To score
such methods, a scenario can carry an `EmitterModel` (`Scenario.emitters`):

| Step | Parameter | Default | Origin |
|---|---|---|---|
| epitope carries a primary | `ele` | 0.65 | [literature] Thevathasan et al. (2019), effective labelling efficiency of indirect immunofluorescence |
| secondaries per primary (each one a stack) | `n_secondary` | P(1, 2, 3) = 0.554, 0.327, 0.119 (mean 1.57) | [derivation + ad hoc] 1 + Poisson(0.59) truncated at 3, near the 1.59 of Ehmann et al. (2014) |
| dyes per secondary | `dol`, `dol_dist` | Poisson, mean 1.5 (0 = dark secondary) | [literature] Dempsey et al. (2011) Online Methods: ~1.5 Alexa 647 per secondary; Poisson shape [ad hoc] |
| offset per antibody layer (primary, secondary) | `sigma_primary_nm`, `sigma_secondary_nm` | 6.5 nm per axis each | [literature] Jimenez et al. (2020): s.d. ~6.5 nm per layer (read as per axis [assumption]); 9.2 nm in total per axis, close to the legacy `SIGMA_LINK_NM` = 10 nm |
| dye position on its secondary | `dye_spread_nm` | 3 nm per axis | [derivation] uniform in an IgG of radius ~6.5 nm: r / sqrt(5) |
| switching cycles per dye | `cycles_dist`, `cycles_mean`, `cycles_shape` | geometric, mean 14 | [literature] Dempsey et al. (2011) Table 1, AF647 in MEA (a lower bound); "gamma" (Poisson-gamma, shape k) and "poisson" also available; `EmitterModel.from_prior(prior)` takes the cycles from a xi prior of `tools/emitters_prior.py` |

Every sample then carries `stack_id` per localization (-1 for isolated
background localizations), `stack_xy` (centroid of each stack's dyes),
`stack_molecule` (<= -2 for non-specific stacks), `stack_epitope`, `label_xy`
and `label_stack`. Without an `EmitterModel` the legacy labelling runs: one
label per labelled epitope, exactly the samples of commit 26a1cd4 (bit for bit,
tested), and every label is its own stack. The official benchmark and its
verdicts use only the legacy presets; the emitter presets (`EMITTER_PRESETS`:
`sparse_emitters`, `_bme`, `_af568`, `_dol1`) are listed by `--list` but are
not part of `--all-scenarios`.

`emitter_metrics(est_xy, sample)` pairs estimated and true emitters (stacks
with at least one localization) with the Hungarian algorithm, a pair being
valid within 3 sigma (Fazel et al., 2022, Fig. 4 and Methods). Sigma is the
precision of each estimate when given (`est_sigma`, as in the paper), else the
nominal localization precision of the scenario [ad hoc, lenient]. It reports
JAC = TP / (TP + FP + FN), RMSE, over-grouping (true stacks absorbed by an
emitter matched to a neighbour), under-grouping (extra emitters within the
radius of a found stack) and emitters per molecule (true vs recovered, exact /
under / over). `oracle_emitters(sample)` scores JAC = 1.

`export_smd_csv(sample, path, level="localization" | "event")` writes the SMD
columns BaGoL reads (`X, Y, X_SE, Y_SE, FrameNum`, nm; frames from 1 [ad hoc,
to verify against the MATLAB run]); `write_stack_truth_csv` writes the truth
next to it. From the command line:
`python -m tools.cluster_benchmark --export-smd --scenario sparse_emitters --replicates 1 --out smd_out`.

## Tests

`test_cluster_benchmark.py`: generator consistency, oracle = 100 % exact and ARI
1, DBSCAN 25/10 on the sparse scenario within the band of the study (77-83 %),
a fully merged labelling, CSV / Markdown / JSON round trip and reproducibility,
`ExternalLabelsStrategy` round trip and the rounded-CSV error, `CallableStrategy`
and the grid (no leak between parameter sets), independent seeds per scenario,
`reg:` specs with words and null, the OPTICS diagnostic verdict, presets and
`--list`. Run: `py -3 -m pytest test_cluster_benchmark.py -q`.

`test_cluster_benchmark_emitters.py`: the legacy generator against a frozen copy
of the one of 26a1cd4, byte for byte on every preset; stack truth consistency;
exact counts of a fixed hierarchy; sampling statistics of ELE, secondaries and
DOL; the cycle distributions; the oracle at JAC = 1; hand-made matching cases
(shift, misses, false positives, split, merge, a pairing where greedy matching
fails); the SMD export and the `--export-smd` command.

## References

- Nieves, D. J., et al. (2023). A framework for evaluating the performance of SMLM cluster analysis algorithms. *Nature Methods*, 20, 259-267.
- Gazal, N. G., Escalante, G., Lopez, L. F., Goutebroze, L., Bisbal, M., Gorostiza, E. A., Szalai, A. M., Stefani, F. D., & Unsain, N. (2026). Nanoscale organization of betaII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic nerve axons [Reviewed preprint, version 1]. *eLife*. https://doi.org/10.7554/elife.111108.1
- Dempsey, G. T., et al. (2011). Evaluation of fluorophores for optimal performance in localization-based super-resolution imaging. *Nature Methods*, 8, 1027-1036.
- Fazel, M., Wester, M. J., Schodt, D. J., et al. (2022). High-precision estimation of emitter positions using Bayesian grouping of localizations. *Nature Communications*, 13, 7152.
- Thevathasan, J. V., et al. (2019). Nuclear pores as versatile reference standards for quantitative superresolution microscopy. *Nature Methods*, 16, 1045-1053.
- Ehmann, N., et al. (2014). Quantitative super-resolution imaging of Bruchpilot distinguishes active zone states. *Nature Communications*, 5, 4650.
- Jimenez, A., Friedl, K., & Leterrier, C. (2020). About samples, giving examples: optimized single molecule localization microscopy. *Methods*, 174, 100-114.
- Früh, S. M., et al. (2021). Site-specifically-labeled antibodies for super-resolution microscopy reveal in situ linkage errors. *ACS Nano*, 15, 12161-12170.
