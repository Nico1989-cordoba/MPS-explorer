# Benchmark run D0, verified defaults (2026-10-16), and the D11 verdicts

Synthetic data only (R8). Seed 0, 10 paired replicates, scenarios sparse_lab1, sparse_lab2, medium_lab1, medium_lab2; 360 rows in 1131 s with 4 processes (one per method spec). Every method at its registry defaults as verified on 2026-10-16 (`benchmark_config.json` lists every parameter); `voronoi_csr` with its default 50 random sets. Replicate seeds: `numpy default_rng([seed, crc32(scenario name), replicate])` (independent scenarios).

Criterion (D11): reference `dbscan_25_10`; paired differences of exact, merge, missed; Δ = 0.05 (5 points, ad hoc: decision D11); time ≤ 10.0× the reference; 95 % t, paired by scenario and replicate; scenarios use independent seeds (crc32 of the name).

**Status: defaults verificados 2026-10-16; benchmark sintético.** The project's axons are in the sparse regime; medium is a stress test. See docs/clustering_benchmark.md section 5.

**How the times were measured.** The run used 4 processes (one per method spec); one of them (the `hdbscan`
package, `core_dist_n_jobs` = 4) opened its own pool, so times measured under that load are not comparable with
DBSCAN's. The times of DBSCAN 25/10, HDBSCAN, FOCAL, GraphCommunity, voronoi_factor and ToMATo were therefore
re-measured serially, one process and one core (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`LOKY_MAX_CPU_COUNT` = 1), on the same samples (the script also checked that `exact`, `merge` and `missed` came out
identical), and `verdicts.json` was rewritten from them. ToMATo was the only label that depended on time: 10.0× /
11.7× in the loaded run, 8.0× / 7.9× with one core. OPTICS and voronoi_csr keep the times of the loaded run; against
the one-core DBSCAN their ratios are overstated by about 2×, and they stay above 100× either way.

## Labels before and after the verification of the defaults

Before: the run of 2026-10-16 at the initial defaults (status «provisional»; scenario seeds shared). After: this run.

| method | sparse before → after | medium before → after |
|---|---|---|
| voronoi_csr | no recomendable → **no recomendable** | no recomendable → **no recomendable** |
| OPTICS | no recomendable → **no recomendable** | no recomendable → **no recomendable** |
| GraphCommunity | no recomendable → **no recomendable** | no recomendable → **no recomendable** |
| HDBSCAN | no recomendable → **preferible** | preferible → **preferible** |
| FOCAL | no recomendable → **preferible** | no recomendable → **preferible** |
| voronoi_factor | no recomendable → **no recomendable** | no recomendable → **no recomendable** |
| ToMATo | no recomendable → **preferible** | no recomendable → **preferible** |

## Verdicts per regime

| method | regime | n pairs | exact Δ [CI95] | merge Δ [CI95] | missed Δ [CI95] | median time / ref [s] (ratio) | label |
|---|---|---|---|---|---|---|---|
| voronoi_csr (`reg:voronoi_csr`) | medium | 20 | -0.348 [-0.386, -0.309] | -0.089 [-0.116, -0.062] | +0.087 [+0.077, +0.096] | 29.363 / 0.028 (1039.0×) | **no recomendable** |
| voronoi_csr (`reg:voronoi_csr`) | sparse | 20 | -0.443 [-0.472, -0.413] | +0.000 [-0.010, +0.011] | +0.041 [+0.030, +0.052] | 22.492 / 0.015 (1502.9×) | **no recomendable** |
| OPTICS (`reg:OPTICS`) | medium | 20 | -0.078 [-0.106, -0.050] | -0.285 [-0.333, -0.238] | +0.296 [+0.243, +0.350] | 6.101 / 0.028 (215.9×) | **no recomendable** (no recomendable / diagnóstico: usar el reachability plot, no la segmentación xi (corrida F de verificación, 2026-10-16: ningún xi / min_samples / max_eps alcanza a DBSCAN 25/10 y tarda ~200x)) |
| OPTICS (`reg:OPTICS`) | sparse | 20 | -0.220 [-0.238, -0.202] | -0.077 [-0.107, -0.047] | +0.181 [+0.140, +0.223] | 3.962 / 0.015 (264.7×) | **no recomendable** (no recomendable / diagnóstico: usar el reachability plot, no la segmentación xi (corrida F de verificación, 2026-10-16: ningún xi / min_samples / max_eps alcanza a DBSCAN 25/10 y tarda ~200x)) |
| GraphCommunity (`reg:GraphCommunity`) | medium | 20 | -0.067 [-0.082, -0.051] | -0.021 [-0.032, -0.009] | -0.035 [-0.052, -0.017] | 0.043 / 0.028 (1.5×) | **no recomendable** |
| GraphCommunity (`reg:GraphCommunity`) | sparse | 20 | -0.055 [-0.068, -0.042] | -0.001 [-0.006, +0.004] | -0.068 [-0.093, -0.042] | 0.023 / 0.015 (1.5×) | **no recomendable** |
| HDBSCAN (`reg:HDBSCAN`) | medium | 20 | +0.059 [+0.032, +0.086] | -0.129 [-0.173, -0.085] | -0.043 [-0.063, -0.023] | 0.078 / 0.028 (2.8×) | **preferible** |
| HDBSCAN (`reg:HDBSCAN`) | sparse | 20 | -0.013 [-0.037, +0.011] | +0.017 [-0.014, +0.047] | -0.084 [-0.113, -0.055] | 0.045 / 0.015 (3.0×) | **preferible** |
| FOCAL (`reg:FOCAL`) | medium | 20 | +0.002 [-0.009, +0.012] | -0.035 [-0.045, -0.024] | -0.033 [-0.045, -0.021] | 0.006 / 0.028 (0.2×) | **preferible** |
| FOCAL (`reg:FOCAL`) | sparse | 20 | -0.027 [-0.040, -0.014] | -0.011 [-0.018, -0.003] | -0.048 [-0.066, -0.030] | 0.005 / 0.015 (0.4×) | **preferible** |
| voronoi_factor (`reg:voronoi_factor`) | medium | 20 | -0.504 [-0.546, -0.462] | -0.009 [-0.034, +0.016] | +0.212 [+0.199, +0.225] | 0.516 / 0.028 (18.2×) | **no recomendable** |
| voronoi_factor (`reg:voronoi_factor`) | sparse | 20 | -0.669 [-0.699, -0.640] | +0.066 [+0.043, +0.088] | +0.210 [+0.193, +0.227] | 0.287 / 0.015 (19.2×) | **no recomendable** |
| ToMATo (`reg:ToMATo`) | medium | 20 | +0.130 [+0.103, +0.157] | -0.240 [-0.271, -0.210] | -0.067 [-0.089, -0.045] | 0.227 / 0.028 (8.0×) | **preferible** |
| ToMATo (`reg:ToMATo`) | sparse | 20 | -0.026 [-0.047, -0.006] | -0.075 [-0.092, -0.058] | -0.090 [-0.120, -0.060] | 0.119 / 0.015 (7.9×) | **preferible** |

## Means over replicates (from benchmark_summary.csv)

| scenario | strategy | exact | merge | split | missed | false | ari | n_clusters | time_s |
|---|---|---|---|---|---|---|---|---|---|
| sparse_lab1 | oracle | 1 | 0 | 0 | 0 | 0 | 1 | 83.4 | 0 |
| sparse_lab1 | dbscan_25_10 | 0.83 | 0.123 | 0 | 0.19 | 0.0805 | 0.911 | 63 | 0.00972 |
| sparse_lab1 | reg:voronoi_csr | 0.358 | 0.135 | 0 | 0.221 | 0.0832 | 0.897 | 62.1 | 18.7 |
| sparse_lab1 | reg:OPTICS | 0.628 | 0.0925 | 0.00676 | 0.304 | 0.0875 | 0.966 | 70.3 | 3.68 |
| sparse_lab1 | reg:GraphCommunity | 0.765 | 0.12 | 0 | 0.0755 | 0.0903 | 0.897 | 73.3 | 0.0138 |
| sparse_lab1 | reg:HDBSCAN | 0.781 | 0.191 | 0.00202 | 0.0495 | 0.0837 | 0.943 | 70.3 | 0.0286 |
| sparse_lab1 | reg:FOCAL | 0.802 | 0.11 | 0.00229 | 0.113 | 0.0855 | 0.911 | 71.1 | 0.00557 |
| sparse_lab1 | reg:voronoi_factor | 0.139 | 0.207 | 0 | 0.401 | 0.0816 | 0.836 | 52.5 | 0.193 |
| sparse_lab1 | reg:ToMATo | 0.786 | 0.0643 | 0.0315 | 0.0441 | 0.09 | 0.953 | 84.1 | 0.0691 |
| sparse_lab2 | oracle | 1 | 0 | 0 | 0 | 0 | 1 | 128 | 0 |
| sparse_lab2 | dbscan_25_10 | 0.786 | 0.167 | 0.000885 | 0.0388 | 0.0667 | 0.846 | 107 | 0.0231 |
| sparse_lab2 | reg:voronoi_csr | 0.372 | 0.156 | 0.00183 | 0.089 | 0.0665 | 0.857 | 108 | 27.5 |
| sparse_lab2 | reg:OPTICS | 0.547 | 0.0438 | 0.0158 | 0.287 | 0.0667 | 0.938 | 124 | 6.53 |
| sparse_lab2 | reg:GraphCommunity | 0.741 | 0.168 | 0.000885 | 0.018 | 0.0747 | 0.843 | 111 | 0.0357 |
| sparse_lab2 | reg:HDBSCAN | 0.808 | 0.132 | 0.00615 | 0.0106 | 0.0674 | 0.906 | 118 | 0.0675 |
| sparse_lab2 | reg:FOCAL | 0.76 | 0.158 | 0.0039 | 0.0199 | 0.0737 | 0.857 | 112 | 0.00721 |
| sparse_lab2 | reg:voronoi_factor | 0.137 | 0.214 | 0 | 0.247 | 0.0598 | 0.804 | 94.5 | 0.414 |
| sparse_lab2 | reg:ToMATo | 0.777 | 0.0752 | 0.0547 | 0.00442 | 0.0734 | 0.941 | 136 | 0.194 |
| medium_lab1 | oracle | 1 | 0 | 0 | 0 | 0 | 1 | 185 | 0 |
| medium_lab1 | dbscan_25_10 | 0.681 | 0.369 | 0.00304 | 0.153 | 0.0618 | 0.758 | 109 | 0.0204 |
| medium_lab1 | reg:voronoi_csr | 0.276 | 0.323 | 0 | 0.228 | 0.0583 | 0.783 | 110 | 24.2 |
| medium_lab1 | reg:OPTICS | 0.574 | 0.17 | 0.00887 | 0.342 | 0.0553 | 0.928 | 134 | 4.65 |
| medium_lab1 | reg:GraphCommunity | 0.592 | 0.336 | 0.000422 | 0.084 | 0.0633 | 0.746 | 118 | 0.0309 |
| medium_lab1 | reg:HDBSCAN | 0.709 | 0.323 | 0.00772 | 0.0712 | 0.0534 | 0.88 | 130 | 0.0625 |
| medium_lab1 | reg:FOCAL | 0.678 | 0.326 | 0.0043 | 0.0967 | 0.0608 | 0.775 | 120 | 0.00699 |
| medium_lab1 | reg:voronoi_factor | 0.109 | 0.383 | 0 | 0.376 | 0.0591 | 0.718 | 90 | 0.404 |
| medium_lab1 | reg:ToMATo | 0.773 | 0.167 | 0.0466 | 0.0406 | 0.0561 | 0.917 | 166 | 0.157 |
| medium_lab2 | oracle | 1 | 0 | 0 | 0 | 0 | 1 | 216 | 0 |
| medium_lab2 | dbscan_25_10 | 0.502 | 0.491 | 0.00513 | 0.0253 | 0.0502 | 0.542 | 110 | 0.0427 |
| medium_lab2 | reg:voronoi_csr | 0.212 | 0.359 | 0.00264 | 0.123 | 0.0461 | 0.653 | 132 | 41.9 |
| medium_lab2 | reg:OPTICS | 0.453 | 0.119 | 0.0111 | 0.428 | 0.0456 | 0.848 | 161 | 8.77 |
| medium_lab2 | reg:GraphCommunity | 0.457 | 0.483 | 0.00251 | 0.0249 | 0.0519 | 0.546 | 113 | 0.0681 |
| medium_lab2 | reg:HDBSCAN | 0.593 | 0.279 | 0.0126 | 0.0211 | 0.0448 | 0.765 | 162 | 0.342 |
| medium_lab2 | reg:FOCAL | 0.508 | 0.465 | 0.00811 | 0.0146 | 0.0497 | 0.571 | 117 | 0.0064 |
| medium_lab2 | reg:voronoi_factor | 0.0662 | 0.459 | 0.000685 | 0.225 | 0.0418 | 0.59 | 110 | 0.766 |
| medium_lab2 | reg:ToMATo | 0.67 | 0.212 | 0.0831 | 0.00318 | 0.045 | 0.836 | 194 | 0.356 |

Labels are read by the interface ('Compare methods...') from `verdicts.json`; nothing here comes from a real axon. A label is about the synthetic regimes, not about any particular measurement.
