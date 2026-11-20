# The resolution limit of the cluster readouts: fusion and d50

> **Resumen (español).** Con los parámetros de un archivo (ε y min_samples de DBSCAN y la precisión NeNA), el
> programa no puede separar dos moléculas que estén más cerca que cierta distancia: DBSCAN las une en un solo
> clúster. Esta página declara dos marcas de esa «escala de trabajo»: **d50**, la separación a la que la mitad de los
> pares de moléculas sale como dos clústeres (≈ 2·ε + 2,5·σ_NeNA, unos 80 nm con 25 nm y 12 nm), y la **fusión casi
> total**, d50 − 20 nm (≈ 60 nm), por debajo de la cual se resuelve como mucho uno de cada diez pares. Las dos salen
> de una simulación propia del proyecto (no de la literatura) y la regla es un ajuste empírico. Por debajo de la
> fusión, lo que ven K/µm, el 1NN y la K de Ripley es el algoritmo, no la biología: el nulo por axón deja esos radios
> fuera de su test y el 1NN entre clústeres se lee como cota superior de la distancia entre moléculas. Para otros
> métodos la regla es *ad hoc* y se informa junto al límite físico de todas las estrategias, 2,5-3 σ_eff.

This page explains the two distance marks of the "working scale" block (`tools/mps_working_scale.py`), where they
come from and how the rest of the program uses them. It is the A5 part of plan A9, merged into A9 by the user's
decision of 2026-10-22; the per-axon null that uses the marks is described in [null_and_scale.md](null_and_scale.md).
Every number is labelled: **literature** (with its reference), **own simulation** (synthetic data only, never a
measured axon), **derivation**, or **ad hoc**.

## 1. What "resolved" means here

Two labelled molecules at a distance d produce two clouds of localizations. DBSCAN with radius ε connects the two
clouds as soon as a chain of core points bridges the gap; then the pipeline reports ONE cluster where there were two
molecules (a *fusion*). The pair is *resolved* when the pipeline reports two clusters, one per molecule. The
fraction of pairs resolved grows with d; **d50** is the separation at which it reaches one half.

## 2. The numbers and their origin

| Mark | Formula | With ε 25 nm, NeNA 12 nm | Origin |
|---|---|---|---|
| d50 | 2·ε + 2.5·σ_NeNA | 80 nm | own simulation: clustering research, section 8.6 and Table 8.5 (script `sim3_pairs_resolution.py`, seed 20261016, ~150 pairs per point); the formula is an empirical fit to that table, an inference, not literature |
| near-total fusion | d50 − 20 nm | 60 nm | own simulation, same table: with 25/10 and σ_loc 12 nm, 10 % of the pairs were resolved at 60 nm, 49 % at 80 nm, 79 % at 100 nm; A9 section 1.1 defines the mark as d50 − 20 nm |
| physical limit (every strategy) | 2.5-3 · σ_eff, σ_eff = √(σ_loc² + σ_link²) | 42-51 nm (σ_link 12 nm) | own simulation, same section: no strategy tried (DBSCAN grids, HDBSCAN on localizations and on events, OPTICS, Voronoi) separated pairs below it |

The simulation behind the table: two tetramers (two epitopes each, 5-10 nm apart), indirect immunolabelling with a
linkage error, geometric switching cycles (mean 14 per label, Dempsey et al. 2011, AF647 in MEA), 1-3 localizations
per cycle, the localization precision of the row, and the clustering of the column; 2D, no axial mixing.

**Where the rule fails.** The fit is within ±8 nm of the simulated d50 except at σ_loc = 20 nm, where it
*under*-estimates: it predicts 100 nm and the simulation gave 100 to more than 120 nm. The program therefore marks the
rule as not applicable when the file's NeNA is 20 nm or more and says that d50 is then a lower bound
(`WorkingScale.rule_applies`).

**Without a NeNA.** When no NeNA was measured for the file, the marks use 12 nm, the lateral precision in the Methods
of Gazal et al. (2026), and say so ("paper, not this file").

## 3. Other clustering methods

The rule was fitted for DBSCAN. For any other method of `tools.clustering_strategies` the program applies the same
formula with the reference pipeline's ε, 25 nm, and labels both marks **ad hoc** (`working_scale_for_method`). Every
method is then tested on the same radii, which is what a comparison across methods needs. The method's own
separation scale (`ClusteringStrategy.edge_margin_nm`) is recorded but not used: it is not a resolution scale.
HDBSCAN's is its *largest* core distance, 221 nm on one synthetic axon of the calibration, and a first version that
used it put the fusion mark beyond the 400 nm of the Ripley grid, which emptied the test (own simulation,
`docs/sims/pipeline_null/`). The physical limit of all strategies (2.5-3 σ_eff) is shown beside the marks, because
that one does not depend on the method.

## 4. How the program uses the marks

- **The per-axon null** ([null_and_scale.md](null_and_scale.md)) leaves every radius below the fusion mark out of
  its global test of Ripley's H(r): below it the curve shows the algorithm merging molecules. Those radii are drawn
  (shaded) but never decide.
- **The 1NN between clusters** is an upper bound of the distance between molecules whenever pairs closer than d50
  can exist: a fused pair removes its short distance and replaces it with the distance to the next cluster.
- **K per µm** is a lower bound of the number of labelled molecules per µm for the same reason.

## 5. Export

The axon row carries `scale_eps_nm`, `scale_min_samples`, `scale_nena_nm`, `scale_fusion_nm`, `scale_d50_nm` and
`scale_slab_nm` (A9 section 5), recomputed from the parameters and the NeNA the row was analysed with
(`WorkingScale.export_columns`). The integration branch adds them to the table and the dictionary.

## 6. Limitations

- The table is 2D: it ignores the axial mixing of two rings inside the ±90 nm slab.
- Blinking is idealised (geometric cycles, no overlapping emitters), and the background is ad hoc.
- d50 depends on how many localizations a molecule leaves (labels, cycles, localizations per cycle): with many more
  localizations per molecule than the simulation assumed, density bridges form further out and the true d50 is larger.

## Decision record

- 2026-10-16: A9 proposes to merge A5 into A9 (same formula, same lines, same export).
- 2026-10-22: the user approves A9 with its recommendations; A5 is merged; this page is the A5 part.

## References

- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768
- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochemistry and Cell Biology*
  141(6):629-638. doi:10.1007/s00418-014-1192-3
- Ester M, Kriegel H-P, Sander J, Xu X (1996). A density-based algorithm for discovering clusters in large spatial
  databases with noise. *Proceedings of KDD-96*, 226-231.
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of βII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic
  nerve axons. bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1
