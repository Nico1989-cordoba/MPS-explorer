# DBSCAN by rule: eps, min_samples, events per cluster, and the parameter sweep

> **Resumen (español).** MPS Explorer sigue agrupando por defecto con DBSCAN eps 25 nm / min_samples 10 (modo
> compatibilidad con Gazal et al. 2026). Al lado calcula una **regla explícita**: eps = mín(2,5·NeNA, 25 nm) y
> min_samples = redondeo(1 + (n_q·lpc − 1)·P), con piso 3, donde n_q es el cuartil inferior de los ciclos de una marca
> (geométrico de media c_label), lpc las localizaciones por ciclo **medidas en el archivo** (largo de los eventos
> enlazados a 3·NeNA con 1 cuadro oscuro) y P = 1 − exp(−eps²/4·NeNA²). c_label es editable: 14 (Alexa 647 en MEA,
> provisorio) y 26 (βME, columna de sensibilidad), ambos de Dempsey et al. (2011). Con la media de ciclos y una
> localización por ciclo la misma fórmula da 9,6 → 10, el valor publicado. Toda tabla puede salir doble (25/10 y regla).
> Los eventos se usan para contar y marcar «probable fusión» (eventos/c_label > 2), nunca para reemplazar
> localizaciones antes de agrupar. El barrido eps 15-40 × min_samples 5-20 repite el análisis normal en cada celda y
> marca la 25/10; sirve para cualquier método con su propia grilla. Decisiones D1-D5 del usuario, 2026-10-22.

This page explains the numbers behind four pieces of the clustering plan: the rule for DBSCAN's two parameters
(`tools/dbscan_rule.py`), the event layer and the localizations per cycle it measures (`tools/mps_events.py`), the
"probable fusion" mark (`tools/mps_events.py`, `fusion_flags`) and the parameter sweep (`tools/dbscan_sweep.py`). The
plug-ins of the multi-method comparison are in `tools/events_sweep_plugins.py`. Every number is labelled:

- **literature**: a published value, with its reference;
- **derivation**: shown here;
- **own simulation**: synthetic data only, with the script that reproduces it. The clustering research's simulation
  (outside this repository) is quoted by section; the simulation of this page is
  `docs/sims/events/events_rule_sim.py`, whose output is `docs/sims/events/events_rule_sim.json`.

## 1. Three layers: localization, event, label

DBSCAN (Ester et al. 1996) counts **localizations**: a localization is a core point when at least `min_samples`
localizations, itself included, lie within `eps` of it. What a labelled molecule leaves is a chain: one labelled
antibody (a **label**) switches on c_label times (its **switching cycles** or **events**), and each event that stays
on for several camera frames leaves one localization per frame. So

  localizations of one label = cycles × localizations per cycle (lpc).

Gazal et al. (2026) set min_samples = 10 "following the mean number of switching cycles expected", the cycles
measured by Dempsey et al. (2011, Table 1): 14 for Alexa Fluor 647 in MEA and 26 in β-mercaptoethanol, on glass, as a
lower bound. Cycles are events, min_samples counts localizations: lpc is the missing link, and it depends on the
illumination intensity, the exposure and the buffer (Lin et al. 2015; Olivier et al. 2013). With τ_on ≈ 10 ms
(Helmerich et al. 2020) and 50 ms frames it would be about 1; the clustering research's simulation assumed 1-3 frames
(mean 2). The program therefore **measures** it (section 4; decision D3).

The events are a layer read **after** the clustering. Clustering linked events instead of raw localizations did not
raise the share of clusters that hold exactly one molecule (69-74 % against 77-83 %) and added false clusters
(13-16 %) in the research's simulation (own simulation: clustering research, section 8.6). The event layer is used for
counting (events per cluster, the fusion mark) and for lpc.

## 2. eps = min(2.5 × NeNA, 25 nm)

The localization precision σ is NeNA (Endesfelder et al. 2014), measured on the whole-field, unlinked file
(`docs/nena.md`, `docs/channel2_parameters.md` section 3). Two localizations of one emitter, each displaced by
N(0, σ²) per axis, are a Rayleigh distance of scale √2·σ apart:

  P(d ≤ eps) = 1 − exp(−eps² / (4 σ²))   (derivation; `docs/channel2_parameters.md` section 4).

- **2.5** (own simulation, clustering research sections 8.3-8.5): eps from 2, 2.5 or 3 × NeNA (NeNA measured on the
  synthetic files, 10.9-11.2 nm) rendered the same one-cluster-one-molecule rate as 25/10; the published 25 nm is
  2.2-2.3 × NeNA there. The rule justifies the published radius; it does not move it.
- **cap 25 nm** (literature: Gazal et al. 2026; own simulation: the σ = 20 nm row of the research's section 8.5): with a
  poor precision 2.5 × NeNA would grow to 50 nm and merge neighbours; the cap keeps the published radius.

With NeNA 12 nm, 2.5 × 12 = 30 nm, so eps = 25 nm (capped). With NeNA 8 nm, eps = 20 nm.

## 3. min_samples from the cycles of the weakest labels

**The worked check.** For one localization of a label that leaves n localizations, DBSCAN's count within eps is the
point itself plus each of the other n − 1 with probability P (derivation):

  expected count = 1 + (n − 1) × P.

With eps 25 nm, NeNA 12 nm, P = 1 − exp(−625/576) = 0.662, and the **mean** 14 cycles at **one** localization per
cycle: 1 + 13 × 0.662 = **9.6 → 10**, the published min_samples (`docs/channel2_parameters.md` section 5; the function
`dbscan_rule.worked_check`). The published value is therefore the expected core count of an *average* label whose
events last one frame.

**The rule.** An average label is not the one that gets lost: cycles per label are geometric, with a long tail and many
small counts (Karathanasis et al. 2017; Hummer et al. 2016 for the geometric law; Dempsey et al. 2011 report only the
mean). The rule asks that the label at the **lower quartile** of the cycles still forms a cluster:

  n_q = the 25 % quantile of Geometric(mean c_label) = ⌈ln(1 − 0.25) / ln(1 − 1/c_label)⌉,
  **min_samples = round(1 + (n_q × lpc − 1) × P)**, never below **3**.

- q = 0.25: own simulation (clustering research, the "cycles rule" strategy): with one label per molecule it halved
  the molecules lost by 25/10 (20-24 % → 9-12 %) without changing the fusion of neighbours, at the cost of about one
  point more of false clusters (8 → 9 %).
- the floor 3: min_samples of at least the dimension + 1 (Ester et al. 1996; Schubert et al. 2017), here 2D.
- "1 + … − 1": DBSCAN counts the point itself once. The research's own simulation used n_q × lpc × P, which multiplies
  the point itself by P too; the two differ by 1 − P (0.34 here) and cross a rounding boundary in one cell of the table
  below. The program uses the self-counting form, consistent with section 5 of `docs/channel2_parameters.md`;
  `min_samples_rule(..., count_self=False)` reproduces the research's.

| c_label | lpc | n_q | raw | min_samples (rule) | research's form |
|---|---|---|---|---|---|
| 14 (MEA) | 1 | 4 | 2.99 | 3 (floor) | 3 (floor) |
| 14 (MEA) | 2 | 4 | 5.63 | **6** | 5 |
| 26 (βME) | 1 | 8 | 5.63 | 6 | 5 |
| 26 (βME) | 2 | 8 | 10.93 | **11** | 11 |

(derivation; eps 25 nm, NeNA 12 nm, P = 0.662; unit test `test_dbscan_events_rule.py`). The clustering plan's
expected ranges (5-6 with c_label 14, 10-13 with 26, at lpc 2) hold for both forms.

**What happens when c_label is wrong.** 14 against 26 changes n_q from 4 to 8: min_samples roughly doubles, and every
count of labels (events / c_label) changes 1.9× (own simulation, clustering research section 8.5). Neither value is
measured in the laboratory's buffer: 14 is the provisional default because the laboratory images in MEA, and 26 is
always reported beside it (decision D1). The rule result carries every input with its origin
(`DBSCANRule.inputs`), and its `min_samples_range` shows the rule at both ends of lpc's 95 % interval.

## 4. Localizations per cycle, measured (D3)

**From a linked file.** A file linked by Picasso carries `n` (localizations per event) and `len` (the event's span in
frames; Picasso documentation, `Link`). The mean of `n` is lpc; with only `len` the value is an upper bound (a bridged
dark frame adds a frame without a localization) and is labelled so.

**From an unlinked file.** Consecutive localizations within a radius of **3 × NeNA**, bridging at most **one dark
frame**, are linked into events (`tools.mps_paint.link_localizations`, Picasso's rule, reused without change); events
already on in the first frame or still on in the last are censored and left out; lpc is the mean localizations per
event, with its standard error (sd / √n) and a 95 % interval (± 1.96 SE). 3 × NeNA and one dark frame are decision D5:
what the research's strategies linked with (own simulation) and the one dark frame of Thevathasan et al. (2019). At
least **100** uncensored events are required (derivation: the standard error is then a tenth of the event-size
spread); otherwise, and without frames or NeNA, the fallback **2** is used and labelled **[ad hoc]** (the research's
simulation assumed it), never presented as a measurement.

**What the 3 × NeNA radius does to lpc.** Two consecutive localizations of one event are a Rayleigh distance of scale
√2·σ apart, so a radius of k·σ breaks a fraction **exp(−k²/4)** of an event's genuine steps (derivation): 10.5 % at
k = 3, 1.1 % at k = 3√2 (the radius the research recommends but did not simulate). A broken step splits an event, so
3 × NeNA counts more events than cycles and **under-estimates lpc**, the more so the longer the events (own
simulation, `docs/sims/events/events_rule_sim.py`, part 1: 600 isolated emitters, σ = 12 nm, geometric event lengths):

| mean event length (frames) | 1 | 1.5 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| lpc at 3 × NeNA (D5) | +0.1 % | −4.5 % | −8.7 % | −15.1 % | −25.6 % |
| the same, corrected for the expected split | +0.2 % | +0.7 % | +1.1 % | +3.7 % | +9.7 % |
| lpc at 3√2 × NeNA | +0.1 % | −0.4 % | −0.8 % | −1.6 % | −3.4 % |

The correction (`LocsPerCycle.split_corrected`) inverts E_obs = E + p (N − E) for the true events E, with N the
localizations and p = exp(−k²/4) (derivation); it over-corrects for long events because a bridged dark frame can
re-join a broken step. On the clustering benchmark's generator (part 2: molecules on an axon contour, 1-3 frames per
cycle, a log-normal spread of the precision, four regimes × 5 replicates), 3 × NeNA split 10-11 % of the true events
and 3√2 × NeNA 2.2-2.4 %, while the share of linked events that join two true events stayed at 0.4-0.6 % and
0.5-0.7 % respectively; lpc came out 1.74-1.79 against a truth of 1.92-1.98 at 3 × NeNA and 1.88-1.95 at 3√2 × NeNA.
In those regimes the wider radius bought accuracy at no visible cost in false links. With lpc at 3 × NeNA the rule
gave min_samples 5 instead of 6 (c_label 14).

The default stays the approved D5 (3 × NeNA, the measured value used as is); `derive_rule(...,
use_split_corrected=True)` and `locs_per_cycle(..., radius_in_nena=3 * sqrt(2))` are one argument away, and the choice
is an open question for the user (the report of this branch).

## 5. The rule against the compatibility mode (own simulation)

`docs/sims/events/events_rule_sim.py`, part 3: the clustering benchmark's generator (σ 12 nm, c_label 14, background
at the research's medium level), 5 replicates per regime, DBSCAN 25/10 against the rule (eps 25 nm, min_samples 5 from
lpc measured at 3 × NeNA); molecule-level outcomes of `tools.cluster_benchmark.molecule_metrics`:

| regime | molecules missed 25/10 → rule | merged clusters 25/10 → rule | false clusters 25/10 → rule |
|---|---|---|---|
| sparse, ~1 label per molecule | 19.1 % → 9.6 % | 14.2 % → 14.2 % | 8.7 % → 9.5 % |
| medium, ~1 label per molecule | 14.6 % → 5.3 % | 35.3 % → 37.2 % | 5.0 % → 5.7 % |
| sparse, 2 labels | 3.4 % → 0.1 % | 17.8 % → 18.4 % | 6.0 % → 6.6 % |
| medium, 2 labels | 1.9 % → 0.4 % | 49.6 % → 53.8 % | 5.0 % → 6.1 % |

This reproduces the research's result (clustering research section 8.7): the rule halves the molecules lost when a
molecule carries one label, leaves the fusion of neighbours to eps, and adds about one point of false clusters. It
does not resolve neighbours closer than ~60 nm; no min_samples does.

## 6. The "probable fusion" mark

Per cluster, labels_est = events / c_label, and the cluster is marked when labels_est > **2** (own simulation:
clustering research section 8.6, thresholds 1.5, 2 and 3 compared). The mark changes nothing: no cluster is removed,
merged or split. It is a control, not a cluster criterion. Its performance depends on the regime (own simulation,
part 3 above, against "two or more molecules each with ≥ 20 % of its localizations in the cluster"):

| regime | fusions found (recall) | marked clusters that are fusions (precision) |
|---|---|---|
| medium, 2 labels | 95 % | 69 % |
| medium, ~1 label | 81 % | 57 % |
| sparse, 2 labels | 97 % | 30 % |
| sparse, ~1 label | 75 % | 26 % |

In a sparse arrangement most marked clusters hold one molecule (two epitopes, or a label with more cycles than the
mean); the research reported 73-79 % precision (medium) and 20-27 % (sparse), in line with this. Because c_label is not
measured, the mark is computed under 14 and 26 (`fusion_table`), and the export says `c_label_source =
"literature_provisional"`.

## 7. The parameter sweep

`dbscan_sweep.sweep(method, grid)` runs the program's own per-axon analysis (`analyze_axon`, unchanged: same slab,
same ROI, same curation; the randomization control off, it does not change these readouts) once per cell and keeps
K/µm, the median 1NN, the median r_eff, the occupancy, the clusters kept and found, the median DBCV of the kept
clusters and the share of the slab left as noise.

- **DBSCAN grid**: eps {15, 20, 25, 30, 40} nm × min_samples {5, 10, 15, 20}, 20 cells (decision D4: the range the
  research recommends after Nieves et al. 2023, Hammer et al. 2025 and Bíró et al. 2025, and the grid of its own
  simulation). In that simulation eps governed the fusion (11 → 26 % from 15 to 40 nm, sparse regime) and min_samples
  the losses (2 → 33 % from 5 to 20) (clustering research, Table 8.2).
- **The reference cell** 25/10 is marked; it is the normal analysis exactly (unit test: every readout equal).
- **Other methods**: the same function with the method's own grid; presets for HDBSCAN (min_cluster_size {5, 10, 20} ×
  min_samples {3, 5, 10}: the research's strategy 4) and ToMATo (radius {15, 20, 25, 30} nm × min_cluster_size
  {3, 5, 10}: **ad hoc**, around Pike et al. 2020's 20 nm and StormGraph's minimum of 3, Scurll et al. 2019). Each
  method's default cell is marked.
- **Cost**: a cell is one analysis; the estimate before running is the per-analysis estimate of the Compare-methods
  dialog × the cells, replaced after the first cell by its measured time. A slow method (Emitters, OPTICS,
  voronoi_csr) is refused unless asked for. Cancel stops between cells, and inside a cell through the analysis' own
  checks; the cells done are kept.

## 8. The double table (D2)

The default analysis stays **compatibility 25/10**. The rule is computed beside it, and every table can come out in
two columns: `compat_25_10` and `rule` (`double_table_cells(rule)` for the sweep, `method_specs(rule)` for the
multi-method comparison, where the rule is one more "method": DBSCAN with the rule's parameters). Each exported row
names its mode (`dbscan_mode`) and the rule's inputs (`eps_rule_nm`, `min_samples_rule`, `c_label_used`,
`c_label_source`, `loc_per_cycle_used`, `loc_per_cycle_source`, `p_neighbour`). Without a NeNA there is no rule, and
the rule column is empty, with the reason.

## 9. Limitations

- c_label is a literature value on glass, not measured in the laboratory's buffer or tissue (a calibration is plan
  B1). The rule moves with it: 14 → 26 roughly doubles min_samples.
- lpc is a property of the movie, not of the axon: it should be measured on the whole field, and on the same
  localizations that are clustered (a filtered file and its parent can differ).
- The geometric law of cycles and the Gaussian precision are models; the real cycle distribution is broader (several
  dark states; Lin et al. 2015).
- The rule is 2D. The analysis clusters x, y inside the axial slab; σ_z is not part of P.
- The simulations here are 2D, without axial mixing, overlapping PSFs or drift (as the research's).

## 10. Decision record

- **2026-10-15**: the clustering research proposes the rule, the event layer, the fusion mark and the sweep.
- **2026-10-22**: the user approves D1 (c_label 14 and 26, 14 provisional with a sensitivity column), D2 (default
  25/10, rule available, double table), D3 (lpc measured from the data, 2 only as a labelled fallback), D4 (the grid),
  D5 (link at 3 × NeNA, one dark frame). This branch implements them as modules; the window wiring is the integration
  branch's.

## References

- Bíró P, H. Kovács BB, Novák T, Erdélyi M (2025). Cluster parameter-based DBSCAN maps for image characterization.
  *Computational and Structural Biotechnology Journal* 27:920-927. doi:10.1016/j.csbj.2025.02.037
- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768
- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochemistry and Cell Biology*
  141(6):629-638. doi:10.1007/s00418-014-1192-3
- Ester M, Kriegel H-P, Sander J, Xu X (1996). A density-based algorithm for discovering clusters in large spatial
  databases with noise. *Proc. KDD-96*, AAAI Press, pp. 226-231.
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of βII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic
  nerve axons. bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1
- Hammer JL, Devanny AJ, Kaufman LJ (2025). Bayesian optimized parameter selection for density-based clustering
  applied to single molecule localization microscopy. *Communications Biology* 8:902.
  doi:10.1038/s42003-025-08332-0
- Helmerich DA, Beliu G, Sauer M (2020). Multiple-labeled antibodies behave like single emitters in photoswitching
  buffer. *ACS Nano* 14(10):12629-12641. doi:10.1021/acsnano.0c06099
- Hummer G, Fricke F, Heilemann M (2016). Model-independent counting of molecules in single-molecule localization
  microscopy. *Molecular Biology of the Cell* 27(22):3637-3644. doi:10.1091/mbc.e16-07-0525
- Karathanasis C, Fricke F, Hummer G, Heilemann M (2017). Molecule counts in localization microscopy with organic
  fluorophores. *ChemPhysChem* 18(8):942-948. doi:10.1002/cphc.201601425
- Lin Y, Long JJ, Huang F, Duim WC, Kirschbaum S, Zhang Y, Schroeder LK, Rebane AA, Velasco MGM, Virrueta A,
  Moonan DW, Jiao J, Hernandez SY, Zhang Y, Bewersdorf J (2015). Quantifying and optimizing single-molecule switching
  nanoscopy at high speeds. *PLOS ONE* 10(5):e0128135. doi:10.1371/journal.pone.0128135
- Nieves DJ, Pike JA, Levet F, Williamson DJ, Baragilly M, Oloketuyi S, de Marco A, Griffié J, Sage D, Cohen EAK,
  Sibarita J-B, Heilemann M, Owen DM (2023). A framework for evaluating the performance of SMLM cluster analysis
  algorithms. *Nature Methods* 20(2):259-267. doi:10.1038/s41592-022-01750-6
- Olivier N, Keller D, Gönczy P, Manley S (2013). Resolution doubling in 3D-STORM imaging through improved buffers.
  *PLOS ONE* 8(7):e69004. doi:10.1371/journal.pone.0069004
- Pike JA, Khan AO, Pallini C, Thomas SG, Mund M, Ries J, Poulter NS, Styles IB (2020). Topological data analysis
  quantifies biological nano-structure from single molecule localization microscopy. *Bioinformatics*
  36(5):1614-1621. doi:10.1093/bioinformatics/btz788
- Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R (2017). Super-resolution microscopy with
  DNA-PAINT. *Nature Protocols* 12(6):1198-1228. doi:10.1038/nprot.2017.024 (Picasso, `Link`)
- Schubert E, Sander J, Ester M, Kriegel HP, Xu X (2017). DBSCAN revisited, revisited: why and how you should (still)
  use DBSCAN. *ACM Transactions on Database Systems* 42(3):19. doi:10.1145/3068335
- Scurll JM, Abraham L, Zheng DW, Tafteh R, Chou KC, Gold MR, Coombs D (2019). StormGraph: a graph-based algorithm
  for quantitative clustering analysis of diverse single-molecule localization microscopy data [Preprint]. bioRxiv.
  doi:10.1101/515627 [preprint, not peer reviewed]
- Thevathasan JV, Kahnwald M, Cieśliński K, Hoess P, Peneti SK, Reitberger M, Heid D, Kasuba KC, Hoerner SJ, Li Y,
  Wu Y-L, Mund M, Matti U, Pereira PM, Henriques R, Nijmeijer B, Kueblbeck M, Sabinina VJ, Ellenberg J, Ries J (2019).
  Nuclear pores as versatile reference standards for quantitative superresolution microscopy. *Nature Methods*
  16(10):1045-1053. doi:10.1038/s41592-019-0574-9
