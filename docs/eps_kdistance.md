# eps = auto: the k-distance graph and its valley

> **Resumen.** Escribir `auto` en Epsilon del **canal 1** (o el botón **k-distance...**) ya no elige un número: abre el gráfico de
> k-distancias de la selección (k = Min Pts − 1). El **valle** de la densidad de log(k-distancia) se marca como
> **sugerencia**, con su intervalo bootstrap y la fracción de localizaciones que DBSCAN dejaría sin agrupar; 25 nm
> (Gazal et al. 2026) aparece como línea de referencia; un clic fija eps. Si no hay valle, se dice y no se sugiere nada.
> Cómo se eligió eps queda en la configuración y en cada exportación (`eps_source` y columnas `eps_kdist_*`). En
> simulaciones propias el valle existió en el 70 % de los anillos, nunca en fondo puro, y cayó **por encima** del eps
> que mejor recupera los clusters en 9 de cada 10 corridas (mediana 1,8×): es una referencia para mirar, no una
> respuesta. El **canal 2** se rige por P-R57 (parámetros derivados de la NeNA de cada canal, editables): su `auto`
> no abre este gráfico; sólo su botón **k-distance... (exploratory)** lo muestra, como chequeo que nunca escribe en
> sus casillas. La entrada "Auto" del selector de algoritmo pasa a llamarse **"Adaptive (exploratory)"**.

Decision P-R58 (2026-10-14). Labels used below: **[literature]** a published source; **[derivation]** follows from a
definition or a formula written here; **[own simulation]** this repository's script and seed; **[ad hoc]** a convention
chosen here without a derivation.

## 1. What changed, and why

Before this change, `auto` in the eps box read the 90th percentile of the distance to the 4th other neighbour. That
is a fixed criterion with the confirmation step removed: Ester et al. (1996) propose reading eps off the sorted
k-distance graph *with* the user's estimate of the noise share. When more than 10 % of the localizations are
unclustered the 90th percentile lies in the background part of the curve, and in our simulations it returned
2.5-4 times the best eps whenever the background was 30 % or more (table in section 5) **[own simulation]**.

The decision was: never a fixed criterion. `auto` now opens a graph and the person decides, with the valley as a
marked, uncertain suggestion and the published value as a reference.

## 2. The convention: k = Min Pts − 1

Ester et al. (1996) define the k-distance of a point as its distance to its k-th nearest neighbour. scikit-learn's
`min_samples` counts the point itself (a core point has at least `min_samples` points within eps, itself included),
so the k-distance that matches it is the distance to the (Min Pts − 1)-th **other** localization, the convention of
the R package dbscan's `kNNdist` ("will set k = minPts - 1"; Hahsler et al. 2019). With it, a localization is a core
point exactly when its k-distance is at most eps **[derivation]**; `test_kdistance.py` checks this against
scikit-learn's own core set. For the analysis' Min Pts 10, k = 9. The old estimator used the 4th other neighbour
(k = 5 including the point), which matches Min Pts 5, not 10.

Consequences shown in the dialog:

* the share of k-distances above eps is the share of localizations that are **not core points** at that eps (shown
  live under the pointer) **[derivation]**;
* the share DBSCAN leaves **unclustered** is smaller (border points join a cluster) and is computed by running
  DBSCAN(eps, Min Pts) on the same localizations, before the analysis removes any cluster (shown for the valley, for
  25 nm, for the value in use and for a clicked value).

## 3. The valley

On u = log10(k-distance):

1. **Density.** A Gaussian kernel density estimate with Silverman's rule-of-thumb bandwidth,
   h = 0.9 min(sd, IQR / 1.34) n^(−1/5) (Silverman 1986, eq. 3.31) **[literature]**, evaluated on 512 grid points by
   linear binning (Wand 1994) **[literature]**. The log axis is used because, for a homogeneous Poisson background, the
   k-th neighbour distance scales as density^(−1/2): on a log axis the background's mode keeps its shape whatever its
   density and only moves **[derivation]**. The rule-of-thumb bandwidth is tuned to a single Gaussian and
   over-smooths a two-mode density, so it tends to hide a weak valley rather than invent one **[literature]**.
2. **Modes and valley.** Modes are the local maxima of the estimate; the valley is the lowest point between the two
   modes of largest topographic prominence. When there are more than two modes, the dialog says so and lists the
   other minima.
3. **Is it a valley?** It is reported only when
   * the lower of its two peaks exceeds it by more than **1.96 standard errors** of the estimate, with
     var f(u) ≈ f(u) R(K) / (n h), R(K) = 1 / (2√π) for the Gaussian kernel (Silverman 1986, section 3.3)
     **[literature + derivation]** (1.96: two-sided 95 % of a normal **[ad hoc convention]**); and
   * the same procedure finds it again in at least **95 % of 200 bootstrap resamples** of the k-distances
     (multinomial on the binned counts; Efron 1979) **[ad hoc convention]**. With 200 resamples the Monte Carlo
     standard error of a share near 0.95 is √(0.95 × 0.05 / 200) ≈ 0.015 **[derivation]**.
4. **Uncertainty.** The 2.5-97.5 % percentiles of the valley's position over the resamples that have one. This is
   the uncertainty of **where the valley is**, not of the right eps: it covered the best eps in 4 % of the
   simulated runs (section 5) **[own simulation]**.
5. **No valley, no number.** Otherwise the dialog says why (single mode; dip within the noise; not stable under the
   bootstrap; too few or all-equal distances) and suggests nothing. There is no fallback value.

The bootstrap is seeded (seed 0), so the same localizations always give the same suggestion and interval.

### Why a valley and not the knee of the sorted curve

Both were run on the same simulated rings (section 5). The basic offline Kneedle (Satopää et al. 2011: the point of
the normalised curve farthest below its chord) **[literature]** was closer to the best eps when the background was
at most half of the localizations (median ratio 0.98-1.47), but

* it always returns a number: on background alone (no clusters at all) it returned 30-154 nm in 9 of 9 runs, where
  the valley correctly reported nothing in 9 of 9 **[own simulation]**;
* it failed when the background was 70 % (median ratio 3.8, adjusted Rand index about 0, i.e. no recovery)
  **[own simulation]**: with that much noise the sorted curve has no knee at the clusters' scale.

Requirement 5 of the decision (no valley: say so, invent nothing) can only be met by a criterion that can fail
visibly, so the valley was kept; its bias is documented and shown in the dialog instead.

## 4. What is recorded

The choice is a record in the settings (`eps_choice`) and seven columns at the end of the per-axon table
(table version **v6**), described in the column dictionary written beside every export:

| column | meaning |
|---|---|
| `eps_source` | `gazal_default` (25 nm, kept or chosen on the graph), `typed`, `kdist_valley` (suggestion accepted), `kdist_click` |
| `eps_kdist_k` | k of the graph (Min Pts − 1) |
| `eps_kdist_valley_nm` | the valley, empty when there was none |
| `eps_kdist_valley_low_nm`, `eps_kdist_valley_high_nm` | its 95 % bootstrap interval |
| `eps_kdist_unclustered_fraction` | share DBSCAN left unclustered at the chosen eps on the graph's localizations |
| `eps_kdist_file` | the file whose selection the graph was drawn on (name only; another file's in a batch) |

A record describes one value of eps. When eps is later typed, reset or changed in the parameter strip, the record is
dropped, and the export says `typed`, or `gazal_default` when eps equals 25 nm. The batch writes the record of the
settings beside every row. The parameter registry lists the rank, the bandwidth rule, the depth (1.96), the bootstrap
(200, 95 %) and the reference line, each with its origin.

**Channel 2.** Channel 2's eps and Min Pts are governed by decision P-R57: derived from each channel's NeNA in the
Two channels box, pre-filled and editable (docs/channel2_parameters.md, on the channel-2 branch). This change does
not alter what `auto` does in channel 2's boxes (the program's behaviour before P-R58). The k-distance graph of
channel 2's own localizations is available **only** through its **k-distance... (exploratory)** button, as an
exploratory check: its Min Pts is channel 2's box, or typed in the dialog, and is never taken from channel 1; the
pair chosen clusters channel 2 in the main window only, and is never written to channel 2's boxes, its remembered
values or the Two channels panel's (derived from NeNA where available).

**The algorithm selector.** "Auto" is renamed **"Adaptive (exploratory)"**: it runs DBSCAN below 100,000
localizations and HDBSCAN above, a switch for speed with no basis in the data, used by the main window's own
clustering preview only (the MPS analysis always runs DBSCAN). "Auto" is still read as that entry
(`tools.clustering_strategies.strategy_for_label`).

## 5. Validation on simulated rings **[own simulation]**

Script `docs/sims/kdist/sim_kdist_valley.py`, results `docs/sims/kdist/sim_kdist_valley_results.json` (seeds: the
simulator 1000 + s, the background 2000 + s, controls 3000 + s, s = 0, 1, 2; bootstrap seed 0; 249 runs, about
2 minutes on 4 processes).

* **Data.** One transverse ring from `tools.mps_simulate_axon` (ellipse 1.5 × 1.0 µm, 5 clusters per µm, hard core
  50 nm, radial offset sd 30 nm, epitope disc 10 nm, linkage 12 nm, 2.5 localizations per fluorophore), plus a
  uniform background in the ring's box widened by 300 nm. Grid: lateral precision 6 / 10 / 15 / 20 nm × mean
  localizations per cluster 10 / 25 / 60 / 150 × background share f = 0 / 0.1 / 0.3 / 0.5 / 0.7 × 3 seeds; plus
  background alone at 200 / 1,000 / 5,000 per µm² × 3 seeds. Min Pts 10 (k = 9).
* **Truth.** DBSCAN on a grid of 60 eps (5-150 nm, log-spaced); eps* maximises the adjusted Rand index (Hubert and
  Arabie 1985) against the true clusters (background and noise as one label). "Relative ARI" = ARI(eps) / ARI(eps*).

| background share f | valley found | valley / eps*, median [10-90 %] | valley rel. ARI, median | Kneedle / eps*, median | old 90th pct / eps*, median | 25 nm / eps*, median |
|---|---|---|---|---|---|---|
| 0 | 1 of 48 | 0.68 (one run) | 0.76 | 0.98 | 0.72 | 0.89 |
| 0.1 | 35 of 48 | 2.74 [2.23-3.47] | 0.71 | 1.47 | 1.29 | 0.79 |
| 0.3 | 39 of 48 | 2.13 [1.80-2.44] | 0.87 | 1.34 | 3.99 | 0.73 |
| 0.5 | 48 of 48 | 1.74 [0.92-2.01] | 0.93 | 1.26 | 3.34 | 0.84 |
| 0.7 | 45 of 48 | 1.51 [0.87-1.76] | 0.96 | 3.83 | 2.55 | 0.89 |
| background alone | 0 of 9 | - | - | 30-154 nm returned | - | - |

What this says:

* **When there is a valley.** It existed in 168 of 240 rings (70 %). Less often when the clusters are too small to
  hold Min Pts 10 core points (10 localizations per cluster: found in 42 % of those runs), and almost never without
  a background (1 of 48), where there is nothing to separate. Never on background alone (0 of 9).
* **Where it lands.** Above eps* in 90 % of the runs where it existed (median 1.8×; more than 1.25× in 86 %). The
  same holds against the eps that best separates clustered from background localizations alone (median 1.8×), so
  the bias is not only about neighbouring clusters merging: the minimum of the mixture density lies in the right
  tail of the clustered mode. It fell inside the range of eps within 90 % of the best ARI in 51 % of the runs.
* **Precision and sampling.** Found in 65-75 % of the runs at every precision; relative ARI 0.93 at 6-10 nm and
  0.86 at 20 nm. More localizations per cluster (heavier oversampling) move the valley and eps* down together
  (median ratio 2.0 at 25 and 1.8 at 150 localizations per cluster): the valley does not track the overestimation
  that oversampling causes in a fixed Min Pts, which is why channel 2 keeps its own parameters.

The project's earlier finding that a k-distance elbow can be off by 2.3-11.6× on a second channel whose answer was
known agrees with this: the suggestion is shown with its interval and its bias, and is never applied by itself.

## 6. Tests

* `test_kdistance.py` (no Qt): the convention against scikit-learn's core set; the valley with and without; its
  reasons; the record, its cleaning and its export columns; settings that say "auto" for eps or predate the record;
  the batch; the column dictionary; the algorithm labels.
* `test_kdistance_gui.py` (offscreen, simulated axons): a real click on the graph sets eps; no valley, no suggestion;
  `auto` in channel 1's box (cancel, accept, re-run, saved, exported); the record follows eps; `auto` in channel 2
  does not open the graph; channel 2's button opens it and leaves both channels' boxes and stored values alone
  (with 'auto' or with values in them); no selection, no graph.

## 7. Decision record

* 2026-10-14, user (P-R58): `auto` for eps never picks a fixed reading of a curve; it opens the k-distance graph
  with the valley as a suggestion, a reference line at 25 nm (Gazal et al. 2026), eps set by click, no number when
  there is no valley, and the choice recorded in settings and export; the algorithm selector's "Auto" renamed.
* 2026-10-14, user: **P-R57 governs channel 2; k-distance in channel 2 is exploratory only, via its button.**
  Channel 2's parameters are the NeNA-derived, editable ones of the Two channels box; `auto` in channel 2's eps
  does not open the k-distance dialog; the graph, through its own button, never writes to channel 2's boxes nor
  overrides the NeNA prefill. Channel 1's behaviour is unchanged.

## References

* Efron B. (1979). Bootstrap methods: another look at the jackknife. *Ann. Stat.* 7(1):1-26.
  doi:10.1214/aos/1176344552
* Ester M., Kriegel H.-P., Sander J., Xu X. (1996). A density-based algorithm for discovering clusters in large
  spatial databases with noise. *Proc. KDD-96*, 226-231. AAAI Press (no DOI).
* Gazal N.G. et al. (2026). Preprint v1: the replicated pipeline and its DBSCAN parameters (eps 25 nm, Min Pts 10).
* Hahsler M., Piekenbrock M., Doran D. (2019). dbscan: fast density-based clustering with R. *J. Stat. Softw.*
  91(1). doi:10.18637/jss.v091.i01
* Hubert L., Arabie P. (1985). Comparing partitions. *J. Classif.* 2:193-218. doi:10.1007/BF01908075
* Satopää V., Albrecht J., Irwin D., Raghavan B. (2011). Finding a "Kneedle" in a haystack: detecting knee points in
  system behavior. *ICDCS Workshops 2011*, 166-171. doi:10.1109/ICDCSW.2011.20
* Schubert E., Sander J., Ester M., Kriegel H.-P., Xu X. (2017). DBSCAN revisited, revisited: why and how you should
  (still) use DBSCAN. *ACM Trans. Database Syst.* 42(3):19. doi:10.1145/3068335
* Silverman B.W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman & Hall (reprint Routledge
  2018, doi:10.1201/9781315140919).
* Wand M.P. (1994). Fast computation of multivariate kernel estimators. *J. Comput. Graph. Stat.* 3(4):433-445.
  doi:10.1080/10618600.1994.10474656
