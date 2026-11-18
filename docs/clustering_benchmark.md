# The clustering benchmark

> **Resumen (español).** Benchmark propio de MPS Explorer para comparar estrategias de clusterización sobre axones
> simulados con verdad de terreno conocida. Desde el 2026-10-16 (Fase D, D0) el módulo `tools/cluster_benchmark.py`
> y sus escenarios están en el programa, y la corrida oficial con los valores por defecto verificados (10 réplicas
> apareadas, semillas independientes por escenario, tiempos con un núcleo) está en `docs/sims/clustering_benchmark/`
> (sección 5): HDBSCAN, FOCAL y ToMATo quedan «preferible» frente a DBSCAN 25/10 en los dos regímenes (pierden menos
> moléculas); GraphCommunity, OPTICS (diagnóstico) y los dos Voronoi, «no recomendable». Las etiquetas valen sólo
> para el benchmark sintético; el valor por defecto sigue siendo DBSCAN 25/10. Los escenarios vienen del
> diseño de la simulación del proyecto (espaciado entre moléculas, pares cercanos, dos epítopos, fracción de marcado,
> error de enlace, ciclos por buffer, precisión, fondo); las métricas, de Nieves et al. (2023) (ARI, IoU) y de esa
> simulación (pureza, completitud, «1 clúster = 1 molécula», fusión, partición, pérdida, falsos, sesgo contra el
> oráculo). El catálogo de métodos está en `docs/clustering_methods.md`.

## 1. Objective

To run every clustering strategy of MPS Explorer (the compatibility mode DBSCAN eps 25 nm / MinPts 10 of Gazal et al.
2026, the new density, tessellation and topological strategies, and labels imported from external tools) on the
**same** simulated axons whose molecules are known, and to report how often each one returns one cluster per
molecule, how it fails (merges, splits, losses, false clusters) and how much each pipeline readout is biased. The
published benchmarks do not reproduce localizations on an annular envelope with antibody-labelled tetramers
*(inference)*, and there is no standardized multi-algorithm benchmark in 3D (see `docs/clustering_methods.md`,
section 3.6). No strategy other than DBSCAN 25/10 becomes a default because of this benchmark: its labels guide the
choice in 'Compare methods' (decision D11) and the default stays DBSCAN 25/10 until the user decides otherwise.

## 2. Scenarios

Taken from the design of the project's clustering simulation; every value is labelled with its origin.

| Factor | Levels | Origin |
|---|---|---|
| Geometry | 2D transverse plane within the axial window; molecules on an elliptical contour | ad hoc (axial mixing, 3D curvature and drift are not simulated in the first version) |
| Mean spacing between molecules | 200 nm (sparse), 100 nm (medium), 40 nm (dense) | sparse: regime suggested by Gazal et al. (2026); medium and dense: ad hoc stress levels |
| Hard core between molecules | 0.4 × mean spacing | ad hoc |
| Close pairs or oligomers | 15 % of molecules with a neighbour at U(30, 60) nm of arc | ad hoc |
| Epitopes per molecule | 2, separated U(5, 10) nm | inference from the tetramer structure (review §3.1) |
| Labelling probability per epitope | 0.5 and 1.0 in the presets (`Scenario` default 0.6) | ad hoc |
| Linkage error (primary + secondary) | N(0, 10 nm) per axis, fixed during the movie | literature + derivation (Früh et al. 2021 report up to ~19 nm; 19/√3 ≈ 11 → ~10 nm per axis) |
| Cycles per label | Geometric, mean 14 (MEA) or 26 (βME) | literature (Dempsey et al. 2011) |
| Frames | one cycle per uniform frame among 20 000, 1-3 frames long, one localization per frame | ad hoc |
| Localization precision | log-normal (CV 0.25) around 12 or 20 nm | 12 nm: literature (NeNA of Gazal et al. 2026); 20 nm: ad hoc; CV 0.25: simulation (project's P-R61 generator) |
| Background | non-specific blinking labels at 0 / 0.5 / 2 per µm of perimeter (±150 nm band), loose localizations at 0 / 30 / 120 per µm² | ad hoc (no laboratory measurement) |

Planned additions, not in the project simulation yet: the 3D slab with σ_z larger than σ_xy, axial mixing between
rings, and a CSR control without any clustering (Nieves et al. 2023 recommend testing that a method does not create
clusters where there are none).

## 3. Metrics

From Nieves et al. (2023), the benchmark reference for SMLM clustering:

- **ARI**, adjusted Rand index between ground-truth identity and the strategy's labels (Hubert & Arabie 1985);
- **IoU**, intersection over union between true and recovered clusters.

From the project's simulation design (per molecule and per cluster):

- purity and completeness of each cluster;
- **one cluster = one molecule** when purity ≥ 0.8, completeness ≥ 0.8 and the cluster is not false (ad hoc
  thresholds);
- **merge**: a cluster holds ≥ 20 % of the localizations of ≥ 2 molecules; **split**: a molecule is spread over ≥ 2
  clusters with ≥ 20 % each; **lost**: < 50 % of its localizations are in clusters; **false**: > 50 % of the cluster's
  localizations are background (all four thresholds ad hoc);
- pipeline readouts against the oracle (perfect grouping by molecule identity): clusters per µm, median 1NN between
  centroids, median area and r_eff from the convex hull, Spearman ρ (r_eff, N_loc), occupancy; bias =
  100·(strategy − oracle)/oracle, paired by replicate;
- internal validation without ground truth, for comparison: DBCV (Moulavi et al. 2014; as proposed by Hammer et al.
  2025).

Summary per cell: mean and standard error (sd/√n) over replicates.

## 4. How it is run

- `tools/cluster_benchmark.py` generates the scenarios with a seed per scenario and replicate
  (`default_rng([seed, crc32(scenario), replicate])`), calls every strategy through the common interface of
  `tools/clustering_strategies.py` (one label per localization, −1 for noise), and writes the long table, the summary,
  the config (parameters and generator constants) and the D11 verdicts.
- Strategies that live outside Python are compared through `ExternalLabelsStrategy`: the benchmark exports each
  simulated file, the external tool clusters it, and its per-localization labels are read back and scored in the same
  way.
- A strategy can be run over a parameter sweep (`grid`; each parameter set on its own copy, with its own row and
  verdict), as Nieves et al. (2023) recommend; the official run of section 5 judges the registry defaults.
- Planned, not implemented: event linking before clustering as a strategy variant (radius 3·NeNA and one dark frame
  [ad hoc], with the project's Picasso-compatible linking, Schnitzbauer et al. 2017, and NeNA, Endesfelder et al. 2014).
- The runs quoted in the documentation are committed in `docs/sims/clustering_benchmark/` (CSV, Markdown, config,
  verdicts); the verification runs A-F of 2026-10-16 that chose the defaults are summarised in section 5.
- Budget: runs are sized to finish on a laptop (≤ 4 processes, one core each); grids are sampled rather than
  exhausted (ad hoc).

## 5. The D0 run (2026-10-16) and the D11 verdicts

Command (synthetic data only; seed 0; 10 paired replicates; scenarios `sparse_lab1`, `sparse_lab2`, `medium_lab1`,
`medium_lab2`; every method at its registry defaults as verified on 2026-10-16, `voronoi_csr` with its default
`csr_iterations` = 50):

```
python -m tools.cluster_benchmark --scenario sparse_lab1 --scenario sparse_lab2 --scenario medium_lab1 \
  --scenario medium_lab2 --replicates 10 --seed 0 --jobs 4 --strategies "dbscan_25_10,reg:voronoi_csr,reg:OPTICS,\
reg:GraphCommunity,reg:HDBSCAN,reg:FOCAL,reg:voronoi_factor,reg:ToMATo" --out docs/sims/clustering_benchmark
```

1131 s with 4 processes. The times of every method except OPTICS and `voronoi_csr` were then re-measured serially
with one core on the same samples (`RESULTS_D0.md` explains why and how; only ToMATo's label depended on it: 10-12×
under load, 8× with one core). Outputs in `docs/sims/clustering_benchmark/`: `benchmark_long.csv` (one row per
scenario × replicate × method), `benchmark_summary.csv` / `.md` (mean ± s.e.), `benchmark_config.json` (parameters,
seeds, constants, how the times were measured) and `verdicts.json` (decision D11: paired mean difference and 95 % t
interval of `exact`, `merge`, `missed` against `dbscan_25_10`, Δ = 0.05, time ≤ 10×, one label per regime;
`criterion.status` = "defaults verificados 2026-10-16; benchmark sintético"). `--verdicts-from benchmark_long.csv`
rewrites the verdicts without re-running. The interface reads `verdicts.json` in 'Compare methods'
(`docs/clustering_methods.md` section 7). `RESULTS_D0.md` in that folder has every interval.

**Before and after.** The first run of the day judged the initial defaults (status "provisional"); the defaults were
then verified against the literature and against the benchmark (verification runs A-F, synthetic, summarised in
`docs/clustering_methods.md` sections 3 and 5), several implementations were corrected (FOCAL without the 3 × 3 map
and border cells; ToMATo on a k-NN graph that never produced noise; Voronoi and GraphCommunity references drawn in the
convex hull or the bounding box of the ring), and the replicate seeds were made independent across scenarios.

| method | sparse (the regime of the axons) | medium (stress test) | mean exact / merge / missed, sparse |
|---|---|---|---|
| DBSCAN 25/10 (reference) | — | — | 0.81 / 0.14 / 0.11 |
| HDBSCAN (10 / 5) | no recomendable → **preferible** | preferible → **preferible** | 0.79 / 0.16 / 0.03 |
| FOCAL (15 nm / 6 / 3, as in Nino et al. 2020) | no recomendable → **preferible** | no recomendable → **preferible** | 0.78 / 0.13 / 0.07 |
| ToMATo (radius graph 20 nm, KDE 10 nm) | no recomendable → **preferible** | no recomendable → **preferible** | 0.78 / 0.07 / 0.02 |
| GraphCommunity (25 nm, null in the occupied region) | no recomendable → no recomendable | no recomendable → no recomendable | 0.75 / 0.14 / 0.05 |
| OPTICS-ξ (ξ 0.1; diagnostic) | no recomendable → no recomendable | no recomendable → no recomendable | 0.59 / 0.07 / 0.30 |
| voronoi_csr (50 sets, first crossing) | no recomendable → no recomendable | no recomendable → no recomendable | 0.37 / 0.15 / 0.16 |
| voronoi_factor (rank 1, 2 × mean) | no recomendable → no recomendable | no recomendable → no recomendable | 0.14 / 0.21 / 0.32 |

What the labels say and do not say:

- "Preferible" means non-inferior to DBSCAN 25/10 within 5 points on the three metrics, at most 10× its time, and
  better on at least one (here, fewer losses for all three; fewer merges for ToMATo). In the sparse regime the three
  "preferible" methods still recover slightly fewer molecules one-to-one than DBSCAN (−0.013 to −0.027, inside the
  margin); the gain is in losses. None changes the default: DBSCAN 25/10 stays the compatibility mode.
- GraphCommunity was "preferible" in the sparse regime in verification run B with its random clouds in the bounding
  box; with the null in the occupied region (REVIEW_D0 M3, a correction of the null, not of the radius) its degree
  threshold prunes more and `exact` falls 5.5 points below DBSCAN (interval [−0.068, −0.042]): just outside the
  margin.
- Both Voronoi methods got worse after their corrections (first-rank density as in Levet et al. 2015; reference region
  = the occupied region; first crossing as in Andronov et al. 2016): on clusters of 28-56 localizations the rim cells
  border large background cells, so the cut erodes molecules, and edge-connected chains merge neighbours. They are
  documented as not recommended for molecule-scale clusters (`docs/clustering_methods.md` section 3.2), not tuned
  further.
- OPTICS carries "no recomendable / diagnóstico" whatever its numbers (`DIAGNOSTIC_METHODS`): no setting of run F
  reached DBSCAN and it takes ~200× its time; its use is the reachability plot.
- The labels are about two synthetic regimes, one geometry (2D ring) and one blinking model (AF647 in MEA, 14
  cycles); the 10 replicates, Δ = 5 points and 10× are ad hoc (decision D11).

## References

- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768
- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochemistry and Cell Biology* 141(6):629-638.
  doi:10.1007/s00418-014-1192-3
- Früh SM, Matti U, Spycher PR, Rubini M, Lickert S, Schlichthaerle T, Jungmann R, Vogel V, Ries J, Schoen I (2021).
  Site-Specifically-Labeled Antibodies for Super-Resolution Microscopy Reveal In Situ Linkage Errors. *ACS Nano*
  15(7):12161-12170. doi:10.1021/acsnano.1c03677
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of betaII-spectrin within segments of the membrane-associated periodic skeleton in mouse
  sciatic nerve axons [Reviewed Preprint, version 1]. eLife. doi:10.7554/elife.111108.1 [preprint with eLife public
  review; no version of record]
- Hammer JL, Devanny AJ, Kaufman LJ (2025). Bayesian optimized parameter selection for density-based clustering
  applied to single molecule localization microscopy. *Communications Biology* 8(1):902.
  doi:10.1038/s42003-025-08332-0
- Hubert L, Arabie P (1985). Comparing partitions. *Journal of Classification* 2(1):193-218. doi:10.1007/bf01908075
- Moulavi D, Jaskowiak PA, Campello RJGB, Zimek A, Sander J (2014). Density-Based Clustering Validation. In
  *Proceedings of the 2014 SIAM International Conference on Data Mining* (pp. 839-847). Society for Industrial and
  Applied Mathematics. doi:10.1137/1.9781611973440.96
- Nieves DJ, Pike JA, Levet F, Williamson DJ, Baragilly M, Oloketuyi S, de Marco A, Griffié J, Sage D, Cohen EAK,
  Sibarita J-B, Heilemann M, Owen DM (2023). A framework for evaluating the performance of SMLM cluster analysis
  algorithms. *Nature Methods* 20(2):259-267. doi:10.1038/s41592-022-01750-6
- Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R (2017). Super-resolution microscopy with
  DNA-PAINT. *Nature Protocols* 12(6):1198-1228. doi:10.1038/nprot.2017.024
