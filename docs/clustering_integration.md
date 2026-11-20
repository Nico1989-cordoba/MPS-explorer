# Every analysis, for every clustering method: the clustering integration

*MPS Explorer, 2026-10-22. Code: `tools/clustering_integration.py` (no Qt), `tools/compare_analyses_ui.py`,
`tools/arrangement_ui.py`; the analyses themselves are documented on their own pages, linked below.*

## 1. What this is for

The question asked of the clustering is not "which method is right" but **would the conclusion change with another
reasonable method?** Every analysis added in the clustering plan (the per-axon null and the working scale, the event
layer and the DBSCAN rule, the eclipse tolerance, the apparent size, the regularity reading) is therefore a pure
function of ONE method's result, and the program runs it once per method chosen, on the same localizations, and shows
the results side by side. DBSCAN with eps 25 nm and min samples 10, the published pipeline [lit: Gazal et al. 2026],
is always run first as the reference against which every other method is read. A null re-runs **the same method,
parameters and curation** as the result it tests: otherwise it would not be the null of that pipeline.

Nothing here renames "Cluster area", and no analysis uses r_eff as a size (decisions of 2026-10-22).

## 2. Where each piece is

| In the program | What it does | Methods page |
|---|---|---|
| Tab **Perimeter arrangement**, top: **Working scale** | the marks below which this file's pipeline cannot tell two objects apart (eps, min samples, NeNA, near-total fusion, d50, slab), each with its origin; the editable **c_label**; the DBSCAN rule of this file beside 25/10 and, after a comparison with the rule, the double table "25/10 vs rule" | `docs/null_and_scale.md` §2, `docs/dbscan_rule.md` |
| Strip, Randomization group: **Null (per axon)...** | opens the Perimeter arrangement tab (the estimate is read there before anything runs) | `docs/null_and_scale.md` |
| Tab **Perimeter arrangement** | **Null (per axon)** for the method shown and every method compared, each re-running its own method; B 199 (999 on demand); Cancel; slow methods refused unless ticked; H1 on the arc and H2 in the band against the null; the regularity reading with its plots; **Show synthetic axon**; **Export curves...** | `docs/null_and_scale.md`, `docs/regularity.md` |
| **Compare methods...** | the methods, then the analyses on each result; the DBSCAN rule as two more runs; tabs "Readouts by method", "Does the conclusion depend on the method?", "Agreement between methods", "25/10 vs rule"; **Export tables...** | `docs/clustering_methods.md` §9 |
| Axon map, group **Events: probable fusions** | kept clusters whose events / c_label exceed 2 (c_label of the panel, solid; 26, dotted); off by default | `docs/dbscan_rule.md` |
| Column review (H-ECL), under the arc test | the axon's footprint, the c_label warning, the scalar tau and the per-pair tolerance with their roles | `docs/eclipse_tolerance.md` |
| Export axon / Batch | the axon table's block (§4; v13, v14); Batch: **Null per axon (B = 199)**, the comparison's long tables, the regularity summary per animal | this page; what to report: `docs/reporting_checklist.md` |

## 3. Running the analyses on every method

`run_comparison` runs the methods through `tools.multimethod` (each on the slab, ROI, curation, contour guide and
pick-boundary rule of the analysis shown; randomization off for speed [ad hoc]) and then the chosen analyses on every
run that worked. Defaults of the dialog [ad hoc: open questions B5 and E4]:

| Analysis | Default | Cost |
|---|---|---|
| Per-axon readouts | always | — |
| Events and probable fusions | on | a few per cent of an analysis [ad hoc] |
| Eclipse tolerance | on | a few per cent |
| Apparent size | on | a few per cent |
| Parameter sweep | **off** | 20 analyses per method with a preset grid (DBSCAN: eps 15-40 × min samples 5-20 [lit + own simulation]) |
| Size at eps 20 / 25 / 30 | **off** | 2 more analyses, DBSCAN only |

A slow method ticked in the dialog is the permission to run it (open question A2). The export writes
`<name>_methods_long.csv` (one row per axon × method × readout), `_methods_agreement.csv`, `_methods_summary.csv` and
`_methods_wide.csv`, never over an existing file (A4). The flag "large spread" (range over the reference's value
above 10 %) is for the eye only [ad hoc], not a test (A3).

## 4. The axon table, v13 and v14

One block at the very end of the row, after the emitters' columns; no earlier column moved. Every column is in the
dictionary written beside each export. What a thesis or a paper should report from these columns, item by item, is
in [reporting_checklist.md](reporting_checklist.md).

* **Working scale** `scale_*`: eps (DBSCAN's, or the reference 25 nm for other methods [ad hoc: same tested radii for
  every method]), min samples, NeNA and where it came from, near-total fusion (d50 − 20 nm) and d50 (2 eps + 2.5 NeNA)
  [own simulation, an empirical fit], slab half-width [lit: Gazal et al. 2026].
* **Methods block** (plan A8): `dbscan_mode` (compatibility 25/10, rule, typed, not DBSCAN); the rule's eps and min
  samples with c_label 14 and 26 [lit: Dempsey et al. 2011, Table 1, Alexa Fluor 647 in MEA / βME; provisional];
  localizations per cycle (measured on the whole file at 3√2 NeNA, or the labelled fallback 2 [ad hoc]); the linking
  radius and dark frames of the event layer (3 NeNA, 1 frame [own simulation; lit: Thevathasan et al. 2019]); the
  runs of the last comparison and the readouts whose verdict changed with the method.
* **Events** `events_*`, **eclipse** `eclipse_*`, **size** `size_*`: the plug-ins' per-axon readouts for the analysis
  exported.
* **Regularity** `gap_*`, `bragg_*`, `regularity_*`, `matched_null_*`, `pipeline_null_*`: the two-stage reading
  (values alone without a null), and `regularity_edge_note` when the edge curation removed clusters (G5).
* **Null** `null_*`, `ripley*`, `p_mc_*`, `null_median_*`: the per-axon null of the analysis exported when it was
  run (`null_type` "none" otherwise). `null_global_test_from` says where the global test started.
* **v14, the reporting checklist** (plan A8 completed, 2026-10-23; three columns after the null, so no v13 column
  moved): `nena_method` (how the NeNA was estimated, or that none was), `link_radius_nm` (the event layer's linking
  radius in nm) and `occupancy_arc_fraction_within_25nm` (P-R103: the fraction of the contour of `perimeter_um`
  within 25 nm of a kept cluster's centre, exact; `occupancy_percent` unchanged). Definitions and the analytical
  check 50 K / P: [reporting_checklist.md](reporting_checklist.md) sec. 3.

## 5. The open questions and the values used until they are answered

Each is one named value in `tools/clustering_integration.py` (`PROVISIONAL`), with its origin; changing an answer is
one line there (or in the module named).

| # | Question | Value used | Origin |
|---|---|---|---|
| A1 | edge-margin floor of methods that are not DBSCAN | the eps of the analysis shown | [ad hoc] |
| A2 | slow methods | ticking one in Compare = permission; refused in Batch and in the null unless their box is ticked | [ad hoc] |
| A3 | "large spread" between methods | visual flag (> 10 % of the reference) | [ad hoc] |
| A4 | export | long + agreement + summary; wide kept | [ad hoc] |
| B1 | radius of the loc-per-cycle measurement of the rule | 3√2 NeNA (events and fusion keep 3 NeNA) | [own simulation: 3 NeNA under-reads by 9-26 %] |
| B2 | min samples counts the point itself | yes (6 with c_label 14, 2 loc/cycle) | [lit: Ester et al. 1996] |
| B3 | where loc per cycle is measured | the whole loaded file | [ad hoc] |
| B4 | ToMATo grid of the sweep | kept, labelled | [ad hoc] |
| B5 | sweep | off, cost shown | [ad hoc] |
| C1 | start of the global H1 / H2 test | d50 (~80 nm) instead of the fusion mark | [own simulation: the contact core reads ~6 nm high] |
| C2 | null membrane for methods that leave gaps | centroids, unchanged (localizations later, after a calibration) | not applied |
| C3 | pilots of the K closure | 4 | [own simulation: 2 left 9 of 20 axons > 5 % off] |
| C4 | closure | localizations in the slab | [user decision A9] |
| C5 | power per axon | conclude per animal; said in the tab | [ad hoc] |
| D1 | τ₀ of the 2D test | 40 nm recommended, **not applied** | the H-ECL parameters are a frozen pre-specification (`config/columns_params.yaml`) |
| D2 | merged-cluster criterion as a measurement | yes: the footprint pool leaves them out | [derivation + own simulation] |
| D3, E5 | per-pair tolerance in 2D; null for the size question (b) | later | — |
| D4 | "Free" τ at 31 nm | accepted | [ad hoc] |
| E1 | elongation cut of the size pool | 2.5 | [own simulation] |
| E2 | fusions passing the 1.5-labels filter | the pool also leaves out the event layer's probable fusions | [own simulation: 3 of 12 axons] |
| E4 | size at eps 20/25/30 | off, cost shown | [ad hoc] |
| G1 | failed replicates of the pipeline-level matched null | tolerated up to 5 % (stage 2 only) | [own simulation: 22 of 60 axons lost 1 of 49] |
| G2 | pipeline-level bands of the regularity plots | the null keeps every replicate's arcs | [ad hoc] |
| G3 | one null for every readout | yes (decision D7) | [user decision] |
| G4 | type of regularity of a group | from the per-animal z (P-R83) | [user decision] |
| G5 | stretches lost at the ROI edge | a note beside the reading | [own simulation] |

## 6. The batch

**Null per axon (B = 199)** runs the null of every file's own method (about B times the batch's time; the estimate is
shown when ticked; slow methods refused unless "Slow methods" is ticked) and the regularity reading with stage 2 at
the pipeline level [regularity plan P-R87 c]. Writing the table then adds `<table>_regularity_by_animal.csv` (mean z
per measure and animal, one-sample t and Holm across measures [lit: Holm 1979]), `<table>_regularity_readings.csv`
and `<table>_regularity.txt` (the P-R83 lattice criterion per genotype; the calibre model N ~ P with a random
intercept per animal, refused below 3 animals [regularity plan sec. 6.4]). **Compare methods (slow)** goes through the
same comparison as the window and writes `batch_methods_long.csv` and its companions beside the
`batch_methods_comparison.csv` it always wrote.

## References

Dempsey, G. T., Vaughan, J. C., Chen, K. H., Bates, M., & Zhuang, X. (2011). Evaluation of fluorophores for optimal
performance in localization-based super-resolution imaging. *Nature Methods, 8*(12), 1027-1036.
https://doi.org/10.1038/nmeth.1768

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large
spatial databases with noise. *Proceedings of KDD-96*, 226-231.

Gazal, N. G., Escalante, et al. (2026). eLife Reviewed Preprint (the reference pipeline of MPS Explorer).

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2),
65-70.

Thevathasan, J. V., et al. (2019). Nuclear pores as versatile reference standards for quantitative superresolution
microscopy. *Nature Methods, 16*, 1045-1053. https://doi.org/10.1038/s41592-019-0574-9
