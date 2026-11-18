# Clustering methods for SMLM: a catalogue, and how MPS Explorer uses them

> **Resumen (español).** «Clúster» no es un objeto único en SMLM: para la familia de **densidad** (DBSCAN, OPTICS,
> HDBSCAN, FOCAL) es un conjunto de puntos conectados por densidad; para la de **teselación** (SR-Tesseler,
> ClusterViSu, 3DClusterViSu) es un grupo de celdas de Voronoi más chicas que un umbral; para la **topológica y de
> grafos** (ToMATo, StormGraph) es un modo persistente de la densidad o una comunidad de un grafo; para la **basada en
> modelo** (bayesiano de Rubin-Delanchy/Griffié, BaGoL, G5M) es un componente de un modelo generativo, y algunas de
> ellas agrupan localizaciones en **emisores**; para el **aprendizaje automático** (CAML, MIRO) es lo que la red
> aprendió de sus simulaciones de entrenamiento. Ninguna responde cuántas moléculas hay en el objeto, y casi todas
> confunden el parpadeo múltiple de un fluoróforo con un clúster. La **estadística no asignativa** (K/L/H de Ripley,
> correlación de pares) no reparte puntos: prueba si hay agrupamiento, a qué escala y contra qué nulo, y es el primer
> paso recomendado (Nieves et al. 2023). Esta página cataloga cada familia (supuestos, parámetros y cómo se eligen,
> manejo del parpadeo, 2D/3D, software, fortalezas, modos de falla, evidencia de benchmarks) con su cita, y explica cómo
> las incorpora MPS Explorer: **DBSCAN eps 25 nm / MinPts 10 queda como modo de compatibilidad** con Gazal et al.
> (2026); las demás entraron el 2026-10-16 (Fase D) como **métodos experimentales** detrás de una casilla, con sus
> valores por defecto verificados ese día contra la literatura y contra el benchmark sintético propio (sección 5), y
> con una etiqueta D11 que sale SÓLO de ese benchmark (sección 7); las que viven fuera de Python (MATLAB, R, Fiji) se
> **importan como etiquetas externas**. El régimen de los axones reales es el disperso; el medio es una prueba de
> estrés.

This page is the reference for the clustering families that exist for single-molecule localization microscopy
(SMLM), what each one assumes, how its parameters are chosen, and how MPS Explorer is going to offer them beside its
current DBSCAN. Its source is the project's literature review on what a cluster is in SMLM (October 2026); every
quotation below is from the cited work. Every number is labelled with where it comes from:

- **literature**: a published value, with its reference;
- **derivation**: shown here;
- **simulation (project)**: synthetic data only, no measured axon. Two sources, kept apart: the benchmark harness of
  Phase D (`tools/cluster_benchmark.py`; runs, config and verdicts committed in `docs/sims/clustering_benchmark/`,
  reproducible from the repository), and the earlier clustering study of the project (October 2026), whose report
  and scripts are private: numbers quoted from it (marked "clustering study") cannot be reproduced from this
  repository;
- **ad hoc**: a convenient value with no published or derived basis, said as such.

Statements marked *(inference)* are the review's reading of the evidence, not a published result.

## 1. What a cluster is for each family

The common denominator is stated by Pike et al. (2020): the commonly used strategies estimate the local density of
detections and build clusters from the detections whose density is above a specified threshold;
Wu et al. (2020) make it a definition, any subset of close, high-density localizations. Beyond that, each family
groups a mathematically different object:

- **Density**: a maximal set of points connected by density. DBSCAN uses "the simple uniform kernel, with bandwidth
  h = ε and a density threshold of minPts/n" (Schubert et al. 2017).
- **Tessellation**: a collection of neighbouring Voronoi cells with areas below a threshold, the inverse cell area
  being the local density (Andronov et al. 2016); SR-Tesseler merges all selected polygons that touch (Levet et al.
  2015).
- **Topological / graph**: a persistent mode of the density, candidates with persistence below τ being merged into a
  neighbour (Pike et al. 2020), or a community of an uncertainty-weighted neighbourhood graph (Scurll et al. 2019,
  preprint).
- **Model-based**: a component of a fully specified generative model, "including the localization precisions",
  scored by its posterior probability (Rubin-Delanchy et al. 2015), with Gaussian clusters on a CSR background
  (Griffié et al. 2016). Emitter-grouping methods (BaGoL, G5M) instead make a group equal to one emitter (Fazel et al.
  2022; Kowalewski et al. 2026).
- **Machine learning**: a per-point "clustered / not clustered" label learned from simulated training sets
  (Williamson et al. 2020), or a point cloud compacted by a graph neural network before DBSCAN (Pineda et al. 2025).

None of these definitions says how many molecules the object contains. What is localized is each switching event of
a fluorophore, and the several points of one fluorophore self-cluster at short length scales, so only clustering
beyond that scale can be considered significant (Nicovich et al. 2017). The same localization pattern can therefore come
from a clustered protein or from a dispersed protein whose fluorophores blink many times: with simulated multiple
blinking, a completely random ground truth "would appear clustered" and pass Ripley's K quality control (Nieves et al.
2023), and the apparent T-cell receptor nanoclustering of a much-cited case was attributed to overcounting (Rossboth et
al. 2018). Complete clustering methods "are not designed to test for the presence of clustering" and can give wrong
results on data without spatial structure (Nieves et al. 2023); DBSCAN found clusters in a pattern that was CSR by
construction, and more once realistic blinking was added (Schneider & Schütz 2022).

## 2. Table of families

Columns: principle; parameters and how they are chosen; multiple-blinking handling; 2D/3D and software; strengths;
failure modes; benchmark evidence. All entries are **literature** unless marked otherwise.

| Method (original source) | Principle | Parameters and how they are chosen | Blinking | 2D/3D; software | Strengths | Failure modes | Benchmark evidence |
|---|---|---|---|---|---|---|---|
| DBSCAN (Ester et al. 1996; Schubert et al. 2017) | ≥ minPts points within ε → core point; density connectivity | ε, minPts; valley of the sorted k-dist plot (k = 4 in 2D), minPts = 2·dim, ε "as small as possible" (Schubert et al. 2017); in SMLM, sweeps against simulations (Nieves et al. 2023), FINDER (Verzelli et al. 2022), DBOpt (Hammer et al. 2025), DBSCAN maps (Bíró et al. 2025) | Not modelled; the most robust to blinking in benchmark, but creates pseudoclusters (Khater et al. 2020) | 2D/3D; scikit-learn (Pedregosa et al. 2011), Picasso and others | Arbitrary shape; separates noise; adaptable | Merges two clusters closer than ε (Ester et al. 1996); varying density (Paupiah et al. 2023); parameter sensitivity (Mazouchi & Milstein 2016; Bíró et al. 2025); nanodomains closer than 50 nm needed ~11 nm precision at ε 20 / minPts 3 (Mollazade et al. 2017); underestimates radius (Staszowska et al. 2018) | DBSCAN ≈ ToMATo ≈ KDE, most robust to blinking (Nieves et al. 2023); above HDBSCAN and OPTICS (Hammer et al. 2025) |
| OPTICS (Ankerst et al. 1999) | Ordered reachability; hierarchy | minPts, maximum ε; cut of the reachability plot | Not modelled | 2D/3D; scikit-learn | Varying density | Slow; below DBSCAN (Hammer et al. 2025); lost 42-48 % of molecules (simulation (project)) | Hammer et al. (2025) |
| HDBSCAN (Campello et al. 2013, 2015; McInnes et al. 2017) | Mutual-reachability tree; no border points | min_cluster_size (+ min_samples) | Not modelled | 2D/3D; `hdbscan` (Python) | Varying density; less merging in the dense regime (simulation (project)) | Below DBSCAN except at low noise (Hammer et al. 2025); larger hull, size decoupled from N_loc (simulation (project)) | Hammer et al. (2025) |
| FOCAL / FOCAL3D (Mazouchi & Milstein 2016; Nino et al. 2020) | Density on a grid (voxel ~ precision), connection of cells | minL (against a randomized version), minC (plateau), Δ | Simulated, not corrected | 2D; 3D (Python) | One parameter; O(n); filters out-of-focus | Small, fine-grid and overlapping clusters (Khater et al. 2020); drops on sparse data (Nieves et al. 2023) | Nieves et al. (2023); Nino et al. (2020) |
| SR-Tesseler (Levet et al. 2015) | Voronoi; normalized first-rank density | Threshold 1× or 2× the mean density; minimum localizations | Not in dSTORM (Levet et al. 2015) | 2D (3D with Voro3D; Levet 2023) | No search radius; multiscale | Underestimates radius (Staszowska et al. 2018); does not separate adjacent clusters; drops on sparse data (Nieves et al. 2023) | Nieves et al. (2023); Staszowska et al. (2018) |
| ClusterViSu / 3DClusterViSu (Andronov et al. 2016, 2018) | Cell area or volume; threshold by Monte Carlo (first intersection with CSR) | Automatic threshold; 50 rounds (Andronov et al. 2016, 2018); 20 nm voxel (source passage not re-checked) | Consecutive events within 50 nm averaged | 2D/3D (MATLAB) | Objective threshold; edge effects controlled | Hours in 3D (Levet 2023); does not separate close clusters; drops on sparse data (Nieves et al. 2023) | Nieves et al. (2023) |
| ToMATo (Chazal et al. 2013; Pike et al. 2020) | Density modes and persistence; merge if P < τ | Density radius; τ on the persistence diagram | Not modelled | 2D/3D (R; RSMLM) | Better separation of close clusters (Paupiah et al. 2023) | τ set by eye; the radius is an ε *(inference)* | DBSCAN ≈ ToMATo ≈ KDE (Nieves et al. 2023); most accurate overall in Kutz et al. (2021) |
| StormGraph (Scurll et al. 2019; preprint, not peer reviewed) | r₀ graph weighted by localization uncertainty; multilevel Infomap | α (0.05), k (10-20), minimum size; r₀ automatic | Explicit filter for a molecule counted several times | 2D/3D (MATLAB); z rescaled | Uses per-localization uncertainties | Preprint; computational cost | Advantage over DBSCAN and ClusterViSu in its authors' simulations (preprint) |
| Bayesian cluster analysis (Rubin-Delanchy et al. 2015; Griffié et al. 2016, 2017) | Scores proposals (L(r) sweeping r and T) by posterior under Gaussian clusters + CSR background + precision | p_B = 0.5, α = 20; r 5-300 nm, T 5-500 | It "does not attempt" to be robust to it (Rubin-Delanchy et al. 2015) | 2D; 3D (Griffié et al. 2017); R | Uses precision and background; no scale parameter | ~30 min per 1000 points; sensitive to priors and artefacts (Khater et al. 2020); not for fibres (Griffié et al. 2016) | Kutz et al. (2021) |
| BaGoL (Fazel et al. 2022); G5M (Kowalewski et al. 2026); DDC (Bohrer et al. 2021); MBC (Jensen et al. 2022) | Group localizations into emitters with the precision and a blinking model | Distribution of events per emitter (learned, or from a control); σ bounds | Yes (it is their object) | 2D/3D (G5M in Picasso) | Return emitters with high-precision positions | Return fluorophores or docking sites, not proteins *(inference)*; need the buffer's event distribution | DNA origami, NPC (Fazel et al. 2022; Kowalewski et al. 2026) |
| CAML (Williamson et al. 2020); MIRO (Pineda et al. 2025) | Per-point classification from neighbour distances; GNN that compacts clouds before DBSCAN | Trained model; DBSCAN afterwards (MIRO) | CAML makes no correction for re-blinking; MIRO trains with 4-5 loc./molecule but uses no temporal information | 2D (CAML 3D) | High accuracy within the trained distribution | Training bias inherent to simulated sets (Williamson et al. 2020); ARI with blinking 0.59-0.66 (Pineda et al. 2025) | Pineda et al. (2025, Table 1) |
| Ripley K / pair correlation / 1NN (Ripley 1977; Sengupta et al. 2011; Veatch et al. 2012) | Second moment against a null; no assignment | Range of r; null (CSR, surface, with blinking) | PC-PALM separates the overcounting term; K does not (Platzer et al. 2020) | 2D (3D and surfaces: Griffié et al. 2017; Ward et al. 2021) | Tests whether there is clustering and at which scale; cross-correlation immune to overcounting (Veatch et al. 2012) | No partition; apparent domain size biased (Shivanandan et al. 2015, 2016) | — |
| Parameter selection: FINDER, DBOpt, DBSCAN maps, similarity to simulations | Stability, internal validation (DBCV; Moulavi et al. 2014) or statistical similarity | Sweep ranges | No | 2D (DBOpt also HDBSCAN/OPTICS) | Remove arbitrariness; reportable | Do not define a molecule | Simulations (Verzelli et al. 2022; Hammer et al. 2025; Bíró et al. 2025; Nieves et al. 2023) |

## 3. The families

### 3.1 Density: DBSCAN, OPTICS, HDBSCAN, FOCAL

**DBSCAN.** For every localization it counts the neighbours within ε, itself included; ≥ minPts makes a core point,
core points chained within ε form a cluster, points within ε of a core point are border points, the rest is noise
(Ester et al. 1996; Schubert et al. 2017). The authors warned from the start that, with global Eps and MinPts,
DBSCAN may merge two clusters of different density that are "close" to each other (Ester et al. 1996, §4.1). The
original heuristic is the sorted k-dist plot with k = 4 in 2D, because larger k did not differ significantly
(Ester et al. 1996, §4.2). Twenty years later the same group reformulated: minPts is the easiest parameter to set,
its purpose is to smooth the density estimate, 4 is a reasonable default in 2D, it should be raised for noisy or
duplicate-rich data, and "ε should be chosen as small as possible" (Schubert et al. 2017). In its source literature
minPts is therefore a density smoother, not a number of fluorophore cycles; once ε is fixed by the expected scale,
minPts is the parameter left to vary *(inference: our summary, not a quotation)*.

In SMLM, parameter sensitivity is the constant objection (Mazouchi & Milstein 2016; Khater et al. 2020; Bíró et al.
2025), and the minPts = D + 1 heuristic and the search for elbows in k-dist plots often fail in practice with SMLM
data (Nino et al. 2020). DBSCAN performs weakly when detection density varies between clusters (Paupiah et al.
2023) and overcounts at large ε by merging indirectly related emitters (Marenda et al. 2021). With RGD nanodomains of
known spacing and ε 20 nm / minPts 3, domains closer than 50 nm needed a localization precision of ~11 nm for the
modal nearest-neighbour distance to be estimated correctly (Mollazade et al. 2017). In astigmatic 3D, an anisotropic
search volume whose eps_z/eps_xy ratio follows the z/xy precision ratio (Spearman 0.9) identifies clusters better
(Lörzing et al. 2024). On size, with DNA origami of known 52 nm radius, DBSCAN measured 37 ± 11 nm and SR-Tesseler
22 ± 8 nm (Staszowska et al. 2018). Two variants put photophysics into the threshold: DBOpt sets MinPts from the
fluorophore:antibody ratio (Hammer et al. 2025) and Diinamic from the expected number of detections per molecule
(Paupiah et al. 2023); both also link events in time first.

**OPTICS and HDBSCAN.** Hierarchical variants that relax the single density: OPTICS orders points by reachability
and the clusters are cuts of the reachability plot (Ankerst et al. 1999); HDBSCAN builds a hierarchy over the
mutual-reachability distance and extracts stable clusters with a minimum cluster size, without border points
(Campello et al. 2013, 2015; McInnes et al. 2017). In the most recent benchmark DBSCAN scored significantly higher than
both for every cluster type, though at low noise HDBSCAN occasionally returned a slightly higher V-measure (Hammer et
al. 2025).

**FOCAL / FOCAL3D.** Counts localizations on a grid whose voxel is of the order of the precision and connects the
dense cells; minL is set against a randomized version of the data and minC on a plateau (Mazouchi & Milstein 2016;
Nino et al. 2020). It is fast and has one main parameter, but struggles with small, overlapping clusters (Khater et
al. 2020) and its performance drops on sparse data (Nieves et al. 2023).

*In MPS Explorer (2026-10-16).* The implementation follows FOCAL3D as Nino et al. (2020, Fig 1B-D) describe it, in
2D: each occupied cell is replaced by the sum of its 3 × 3 neighbourhood, cells at or above minL are core, cells that
share an edge with a core cell are border cells, and connected groups of at least minC cells are clusters. The first
version counted raw cells without border cells and lost 60-83 % of the molecules (benchmark run of 2026-10-16,
before the fix). Defaults: Δ = 15 nm [literature: Δ "based on the localization precision" (Nino et al. 2020); ≈ 1.25-1.5 ×
the 10-12 nm NeNA of the project; simulation (project): verification run E], minL = 6 on the enhanced map
[simulation (project): run E; `suggest_min_l` gives the minL* of Nino et al. 2020, the lowest value with no cluster in
randomized data, drawn in the region the data occupy], minC = 3 [ad hoc; Nino et al. choose it on a plateau].

*OPTICS and HDBSCAN in MPS Explorer.* HDBSCAN: min_cluster_size 10 [derivation: about a third of the ~28 localizations
of one labelled molecule in the benchmark] and min_samples 5 [simulation (project): verification run A]. OPTICS-ξ:
ξ = 0.1 [simulation (project): run F; sklearn's 0.05 left 45-58 % of the molecules unclustered], max_eps = 100 nm
[simulation (project): a finite, editable value; it does not shorten the run, which stays ~200× DBSCAN's because the
cost is scikit-learn's per-point loop], min_samples 10 [ad hoc: as DBSCAN]. No OPTICS setting reached DBSCAN 25/10 on
the benchmark (run F): it is kept as a **diagnostic** (the reachability plot), labelled "no recomendable" for
segmentation.

### 3.2 Tessellation: SR-Tesseler, ClusterViSu, 3DClusterViSu

Voronoi tessellation gives every localization a cell whose inverse area is the local density, a direct measure that,
unlike Ripley's analysis, does not depend on a chosen search radius (Andronov et al. 2016). SR-Tesseler selects and merges the polygons whose first-rank density exceeds α times the mean density
of a uniform reference; all its biological examples used α = 2 (Levet et al. 2015). ClusterViSu sets the area
threshold at the first intersection between the experimental distribution and that of Monte Carlo CSR simulations at
the same mean density (Andronov et al. 2016); in 3D the threshold falls as background density rises and ~10⁶
localizations took 6 hours (Andronov et al. 2018), which Voro3D reduces to minutes with an analytical approximation of
the random cell-size distribution (Levet 2023).

*In MPS Explorer (2026-10-16).* Both thresholds refer to the region the localizations occupy (a dilated 100-nm
occupancy grid) instead of the convex hull of the ring, which on a 20-µm axon is ~5× the band and made "2 × mean
density" select nearly the whole band [derivation; grid and dilation ad hoc]. `voronoi_factor` uses the first-rank
density and α = 2 (Levet et al. 2015); `voronoi_csr` uses rank 0, 50 random sets and the first crossing of the
observed and mean random distributions after the excess leaves the ±1.96σ envelope (Andronov et al. 2016), with the
0.99 quantile of the random pool only as a fallback [ad hoc]. Coincident localizations share one cell. Even so,
neither reaches DBSCAN on molecule-scale clusters (28-56 localizations): the first-rank average mixes the rim cells
of a cluster with the large background cells around it (30 % of a molecule's localizations fall below 2 × mean, 14 %
with rank 0), and with rank 0 no factor between 2 and 6 avoids both merges (edge-connected chains between neighbours)
and losses (exact ≤ 0.47 against 0.88 for DBSCAN on one replicate) [simulation (project), 2026-10-16]. The second,
object-relative threshold and the minimum distance of SR-Tesseler (50 nm for GlyR, Levet et al. 2015, Fig 3h) are
not implemented. Both Voronoi methods are **not recommended** for this purpose (section 7).

Limits: in dSTORM the stochastic blinking of organic dyes prevents blinking correction (Levet et al. 2015); the
multiple localizations of a blinking fluorophore are indistinguishable from a true molecular cluster because the
method detects tight points by cell area (Hyun & Kim 2023); tessellation methods did not handle multiple blinking in
the review of Khater et al. (2020); and in the benchmark of Nieves et al. (2023) ClusterViSu, SR-Tesseler and FOCAL
often scored lower, especially on sparse data, probably because points at the cluster edge have a density similar to
the background. A threshold of 1× or 2× the random mean density is a criterion of departure from randomness, not of
a molecule: with several localizations per fluorophore, every isolated fluorophore is already a Voronoi cluster
*(inference)*.

#### The Voronoi layer of the axon map

When the analysis shown was made with `voronoi_factor` or `voronoi_csr`, the axon map of the MPS analysis window has
a group **Voronoi cells (method)** with two rows, both off by default:

* **Cells in clusters**: the cell of every localization the method put in a cluster, filled by its density class,
  opaque, with a thin outline in the plot's neutral colour.
* **Cells outside clusters**: the cells of the other localizations, on the same scale, faint and without an outline.
  A dense cell here belongs to a group of dense cells smaller than `min_cluster_points`.

Under the rows, a scale shows the classes from sparse (left) to dense (right) with a white tick at the threshold the
method applied, and the densities at its ends and at the tick in localizations per µm²; the group's caption repeats
the threshold, its rule (`factor`, `first crossing` or `quantile fallback`) and its ratio to the mean density of the
occupied region.

What it shows, exactly:

* **The method's own cells.** The analysis keeps the method's diagnostics (density per localization, clip window,
  threshold); the polygons are those the fit kept (`keep_polygons=True`) or are rebuilt from the same unique points and
  the same clip window with the same function the fit called, and their areas are checked against the fit's own
  before anything is drawn. Nothing is clustered again and no number of the analysis changes (the labels are identical
  with and without `keep_polygons`; `test_voronoi_layer.py`).
* **The density the method thresholded**: first-rank density for `voronoi_factor` (its default `density_rank = 1`),
  1/area for `voronoi_csr` (rank 0), whatever `density_rank` was set to.
* **Eight classes of log density** [ad hoc]: four equal steps from the 1 % quantile of the cells up to the threshold
  and four from the threshold up to the 99 % quantile, so that above or below the cut is read from the colour alone;
  cells beyond the quantiles take the end colours. If the threshold lies outside that range (no finite threshold,
  or none inside the data), the eight classes span the range and the scale says so.
* **In a cluster** means the method's own label, *before* the analysis' edge curation: the layer shows what the
  tessellation did; the localization rows show what the curation kept.
* **Colours**: viridis sampled at 0.25-1.0 (its darkest quarter is near-black on the map's black), a sequential map
  whose lightness rises with density under normal vision and the three dichromacies (checked by
  `validate_plot_colours.py`, which also lists the marks drawn over the cells and what tells each from the fill).
  Cells of a cluster and the others are told apart by opacity and the outline, never by hue.

Cost [simulation (project), 2026-10-20, synthetic rings, the user's notebook]: the cells are built the first time a
row is ticked, one path per class rather than one item per cell: 0.3 s for 5,128 localizations and 1.3-1.5 s for
20,290; drawing a frame takes 8 / 25 ms. Above 50,000 cells the rows are disabled and say why [ad hoc limit, ~3-4 s
by the same measurement].

### 3.3 Graph and topological: ToMATo, StormGraph

ToMATo replaces the global threshold by the persistence of density modes: every local maximum of the density is a
candidate, persistence is the difference between birth and death densities, and in the SMLM implementation the two
free parameters are the search radius and the persistence threshold τ (Chazal et al. 2013; Pike et al. 2020).
Persistence clustering improves the separation of clusters that are close to each other (Paupiah et al. 2023); its
cost is that τ is read by eye from the persistence diagram and the density radius plays the role of an ε
*(inference)*. In Nieves et al. (2023) ToMATo performed like DBSCAN and KDE; Kutz et al. (2021) found it the most
accurate overall in their comparison.

StormGraph (Scurll et al. 2019; preprint, not peer reviewed) builds a neighbourhood graph weighted by Monte Carlo over
each localization's uncertainty, detects communities with Infomap, and, among the methods reviewed on this page, is
the one that describes an explicit filter meant to reclassify clusters that may be one molecule counted several
times (as read in the preprint; not verified in a peer-reviewed version). Its authors report advantages over DBSCAN
and ClusterViSu in their own simulations (preprint result, not confirmed by peer review).

*In MPS Explorer (2026-10-16).* **ToMATo** runs on a radius graph of 20 nm with a Gaussian kernel density of σ = 10 nm
[literature: Pike et al. 2020 build the graph and count the density within a search radius, 20 nm on integrin
dSTORM; σ = radius / 2 is a derivation; simulation (project): verification run D]. The first version used a k-NN
graph, in which every point has neighbours, so nothing was ever noise (missed 0, split 0.22-0.44); with the radius
graph a localization that cannot be linked stays alone and `min_cluster_size` (3, ad hoc, taken from StormGraph) makes
it noise, as in Pike et al. There is no density floor by default (it is not part of ToMATo, and in run D a floor of
min_samples / πε² lost 17 % of the molecules). τ is proposed at the largest relative gap of the persistence diagram
[ad hoc; Pike et al. read it by eye]. **GraphCommunity** is NOT StormGraph (no uncertainty Monte Carlo, no Infomap):
a radius graph with Gaussian weights, a node-degree threshold at the (1 − α) quantile of random clouds and connected
components. Its radius is fixed at 25 nm [ad hoc: the compatibility ε; the plateau scan of the first version chose
radii that merged clusters, merge 0.24-0.74, and stays available with radius None] and its random clouds are drawn in
the region the data occupy, not in the bounding box of the ring (REVIEW_D0 M3).

### 3.4 Model-based: Bayesian cluster analysis, BaGoL, G5M

**Bayesian cluster analysis.** Rubin-Delanchy et al. (2015) assume a single coordinate per molecule, model positions
as Gaussian clusters on a CSR background perturbed by Gaussian localization errors, and score thousands of proposals
(from L(r) thresholds over a grid of r and T) by their posterior; α = 20 and p_B = 0.5 (literature). The authors
delimit the scope: the algorithm does not attempt to correct for, or be robust to, multiple blinking, so it is
quantitatively reliable only when blinking has been corrected; dimers and trimers were not identified accurately,
and six localizations per cluster is suggested as the lower detection limit (Rubin-Delanchy et al. 2015). The protocol
adds that the default prior makes CSR extremely unlikely a priori, that a 1000-point region takes about 30 min, and
that for distributions not approximable by circular clusters, fibres for example, DBSCAN or Voronoi segmentation are
more appropriate (Griffié et al. 2016). The 3D extension adds a Beta prior in z (Griffié et al. 2017). The review of
Khater et al. (2020) rates it very slow and sensitive to priors and imaging artefacts. For an antibody-labelled
protein with many localizations per label, a single labelled molecule would be a legitimate Gaussian cluster for the
model *(inference)*.

**Emitter grouping.** BaGoL explores groupings of localizations from multiple blinking or binding events and uses
the localization precisions to learn the distribution of events per emitter and infer emitter number and positions;
that distribution can come from a low-labelling control or be learned from the data (Fazel et al. 2022). G5M is a
modified Gaussian mixture that incorporates the localization precision, spatial constraints and DNA hybridization
kinetics, with DBSCAN as a pre-split, and is integrated in Picasso (Kowalewski et al. 2026). DDC corrects the pair
distance distribution using pairs separated by much more than the fluorescence survival time (Bohrer et al. 2021);
MBC estimates blinking dynamics without calibration for PALM (Jensen et al. 2022). All return fluorophores or docking
sites, not proteins: the step emitter → protein still depends on the labelling stoichiometry *(inference)*.

### 3.5 Machine learning: CAML, MIRO

CAML describes each point by the distances to its 100 or 1000 nearest neighbours and classifies it as clustered or
not with a network trained on simulated scenarios (92-94 % accuracy); its authors state that it makes no assumption
about, nor correction for, fluorophore re-blinking, and that bias is inherent to any simulated training set
(Williamson et al. 2020). MIRO, a recurrent graph neural network, transforms point clouds so that DBSCAN performs
better; it is the only proposal trained explicitly with 4-5 localizations per molecule, but uses no temporal
information, and its ARI with blinking is 0.59-0.66 against 0.56-0.65 for DBSCAN (Pineda et al. 2025, Table 1). In
2020 little work had applied ML to SMLM clustering (Khater et al. 2020) and in 2023 no extraordinary results had yet
been achieved (Hyun & Kim 2023). Applied to a ring-shaped skeleton, the risk is that the network imposes compact
clusters at the scale it was trained on *(inference)*.

### 3.6 What the published benchmarks say

The only multi-algorithm benchmark with standardized scenarios is Nieves et al. (2023): seven algorithms (DBSCAN,
ToMATo, KDE, FOCAL, CAML, ClusterViSu, SR-Tesseler), ten 2D scenarios of 2 × 2 µm with 50 realizations, ARI and IoU,
DBSCAN swept over ε 5-100 nm × minPts 2-50 (literature). Its conclusions: algorithms with more user inputs (DBSCAN,
ToMATo, KDE) are the most adaptable and keep good performance even with multiple blinking, at a cost in
reproducibility between laboratories; there are areas of relative stability at high scale and low density for those
three, so it may be safer to err to those sides; sparse data were particularly hard for single-parameter algorithms;
DBSCAN was overall the most robust to multiple blinking; and because optimal parameters shift upwards, erring towards
higher density thresholds is recommended (Nieves et al. 2023). Hammer et al. (2025) note that the method is limited to
the simulated sets analysed and propose internal validation by DBCV (Moulavi et al. 2014). FINDER picks the most stable
(ε, minPts) pair in parameter space, the radial parameter playing the predominant role (Verzelli et al. 2022). **There
is no standardized multi-algorithm benchmark in 3D**; only single-method evaluations exist (Nino et al. 2020; Lörzing
et al. 2024; Andronov et al. 2018; Griffié et al. 2017). None of the published benchmarks reproduces localizations on
an annular envelope with σ_z larger than σ_xy and antibody-labelled tetramers *(inference)*, which is why MPS Explorer
plans its own (section 5).

## 4. Non-assignative statistics

**What they test.** A localization set is a point pattern. The reference null is complete spatial randomness (CSR),
a homogeneous Poisson process; "clustering" is any departure towards more closeness than expected at some scale.
Ripley's K counts the mean number of neighbours within r of a typical point and compares it with πr²λ under CSR;
L(r) = √(K(r)/π) (Besag 1977, in the discussion of Ripley 1977) and L(r) − r (H(r)) linearize the comparison. Kiskowski et al. (2009) phrase it for SMLM and show that
the r of the maximum of L(r) − r approximates the domain radius only under idealizations. The pair-correlation
function g(r) and the nearest-neighbour (1NN) distribution answer the same question at a single distance band. These
are global methods that return a result for the whole set; the methods of sections 3.1-3.5 are complete methods
that assign every localization to a cluster or to the unclustered population (Nieves et al. 2023).

**Scale.** The answer is relative to a scale r: a pattern can be clustered at 30 nm and regular at 200 nm. K was the
first statistic applied to PALM (Owen et al. 2010); its reading as a domain size is biased by incomplete detection and
precision (Shivanandan et al. 2015) and by a factor that can be arbitrarily large (Shivanandan et al. 2016).

**Null.** PC-PALM separates one protein appearing several times from protein clusters (Sengupta et al. 2011), because
the overcounting term of the autocorrelation has the shape of the precision and an amplitude inversely proportional to
the density, and overcounting produces no apparent co-clustering in cross-correlation (Veatch et al. 2012). The
correct null includes the measured blinking (Platzer et al. 2020) or uses two colours on the same molecule (Arnold et
al. 2020); on curved or bounded 3D shapes CSR must be defined on the shape (Ward et al. 2021; Griffié et al. 2017).
p-values are built by Monte Carlo (Besag & Diggle 1977) with α fixed in advance on a hypothesis about biomolecules,
not about the localization map (Schneider & Schütz 2022). The standard texts on point processes and cluster processes
are Diggle (2013) and Baddeley et al. (2015). The recommended first step of an analysis is to test for clustering
with Ripley's K and 95 % simulation envelopes, after correcting blinking (Nieves et al. 2023).

## 5. How MPS Explorer uses them

**Compatibility mode: DBSCAN eps 25 nm / MinPts 10 (existing).** The current pipeline clusters channel 1 with
DBSCAN at the published parameters of the reference study (literature: Gazal et al. 2026), through scikit-learn
(Pedregosa et al. 2011). It stays as the default and as the **compatibility mode**, so that every result can be
compared with that study. In the project's synthetic benchmark (simulation (project); 2D transverse plane, Alexa 647
blinking statistics from Dempsey et al. 2011, Picasso-style linking from Schnitzbauer et al. 2017, NeNA from
Endesfelder et al. 2014, ARI from Hubert & Arabie 1985):

- in the sparse regime (mean spacing ~200 nm between labelled molecules), DBSCAN 25/10 turned 77-83 % of its clusters
  into exactly one molecule, 10-17 % were merges and 7-8 % false clusters from non-specific labels;
- at ~100 nm spacing it fell to 50-67 % one-to-one and merged 37-51 %; clusters/µm were −43 to −49 % against the
  oracle, while occupancy barely moved (−2 %);
- in the ε × min_samples grid, ε governs merging and min_samples governs losses; 25/10 lies on the ridge of the
  sparse regime;
- deriving ε = 2-3·NeNA and min_samples from the expected cycles per label gives ε 22-33 nm and min_samples 5-7,
  which performs like 25/10: the rule *justifies* the parameters, it does not improve them;
- HDBSCAN (min_cluster_size 10, min_samples 5) was the only strategy that won in the medium regime; OPTICS-ξ and
  Voronoi (SR-Tesseler 2×, Monte Carlo threshold) were worse than DBSCAN in every regime tested (clustering study;
  the Phase D benchmark of 2026-10-16, with verified defaults, is in section 7);
- for isolated pairs at σ_loc 12 nm, DBSCAN 25/10 separates half of them only at 81-88 nm; empirical rule
  d50 ≈ 2·ε + 2.5·σ_loc (simulation (project), fitted, not a law).

**The strategy interface (existing).** `tools/clustering_strategies.py` defines an abstract `ClusteringStrategy` with
`fit(data) -> labels`, and the concrete `DBSCANStrategy`, `HDBSCANStrategy` and `AutoClusteringStrategy` (which
today chooses between them by dataset size). New methods will be added as further strategies with the same contract:
one label per localization, −1 for noise, plus a record of the parameters and of where each came from.

**Modules (integrated 2026-10-16, Phase D / D0; see sections 7 and 8 for how they are reached):**

| Module | Content | Status |
|---|---|---|
| `tools/cluster_density.py` | OPTICS (ξ or DBSCAN cut), FOCAL-style grid density, HDBSCAN (scikit-learn port when the `hdbscan` package is missing) | available (2026-10-16); HDBSCAN on linked events and anisotropic 3D DBSCAN (Lörzing et al. 2024) still planned |
| `tools/cluster_tessellation.py` | SR-Tesseler-style (α × mean density, `voronoi_factor`) and ClusterViSu-style (Monte Carlo CSR threshold, `voronoi_csr`) Voronoi segmentation, 2D | available (2026-10-16); 3D planned |
| `tools/cluster_topological.py` | ToMATo (density modes + persistence τ) and `GraphCommunity` (a StormGraph-inspired substitute, not the published algorithm) | available (2026-10-16) |
| `tools/spatial_stats.py` | Ripley K/L/H, pair correlation (2D and along the arc) with Monte Carlo CSR envelopes | available as a library (2026-10-16); not in the interface yet; the null with blinking planned |
| `tools/cluster_benchmark.py` | The project benchmark that runs every strategy on the same simulated axons and writes the D11 verdicts (see `docs/clustering_benchmark.md`) | available (2026-10-16) |
| `ExternalLabelsStrategy` | Imports per-localization labels produced outside Python (CSV with x, y, label) and treats them as one more strategy, with a scale the user declares | available in the registry and the benchmark CLI (2026-10-16); no file picker in the interface yet |

**Defaults, verified 2026-10-16.** Each default was checked against its source and against the synthetic benchmark
(verification runs A-F, synthetic data only); the origin label is the one the parameter panel shows.

| Method | Parameter = default | Origin |
|---|---|---|
| DBSCAN (compatibility) | ε = 25 nm; min_samples = 10 | literature: Gazal et al. (2026) for ε; min_samples is the reference pipeline's value (not among the verified sentences of the preprint) |
| HDBSCAN | min_cluster_size = 10; min_samples = 5 | derivation (≈ 28 / 3 localizations per molecule); simulation (project): run A |
| OPTICS (diagnostic) | min_samples = 10; ξ = 0.1; max_eps = 100 nm | ad hoc (= DBSCAN); simulation (project): run F; simulation (project), does not bound the time |
| FOCAL | Δ = 15 nm; minL = 6 (3 × 3 map); minC = 3 | literature (Δ ~ precision, Nino et al. 2020) + simulation (project): run E; simulation: run E; ad hoc |
| voronoi_factor | first-rank density, α = 2, occupied region | literature: Levet et al. (2015); derivation (reference region) |
| voronoi_csr | rank 0, 50 random sets, first crossing, occupied region; fallback quantile 0.99 | literature: Andronov et al. (2016); ad hoc (fallback) |
| ToMATo | radius graph 20 nm; KDE σ 10 nm; τ by largest gap; min size 3; no floor | literature: Pike et al. (2020) + simulation: run D; derivation (r/2); ad hoc; ad hoc (StormGraph's value) |
| GraphCommunity | radius 25 nm; α = 0.05; 5 random clouds in the occupied region; min size 3 | ad hoc (= ε) + simulation: run B; literature, preprint (Scurll et al. 2019); ad hoc; ad hoc |

Still ad hoc after the verification: GraphCommunity's 25 nm, ToMATo's σ = r/2 and automatic τ, ξ of OPTICS, α = 0.05
and the k range of StormGraph (only the preprint was read), FOCAL's minC and 4-connectivity, the 100-nm grid of the
occupied region, D11's Δ = 5 points, 10× and 10 replicates, and HDBSCAN's 5 / 10 (benchmark-specific).

Methods that need a trained model (CAML, MIRO) or a calibrated event distribution per emitter (G5M, DDC) are
only offered through external labels: the calibration they require does not exist yet for the buffers and labels in
use *(inference)*. The exception is BaGoL, reimplemented in Python as the experimental, slow method
**Emitters (BaGoL-like reimplementation)** (`emitters_bagol_like`, not the authors' code): its event distribution
per emitter, ξ, is taken from the literature (Dempsey et al. 2011), learned from the axon or typed from a control,
and the emitters are linked into clusters within r_mol; see [`emitters.md`](emitters.md) and
[`emitters_xi.md`](emitters_xi.md). No strategy other than DBSCAN 25/10 changes a default result: the benchmark labels (section 7) guide
the choice in 'Compare methods', and the default stays DBSCAN 25/10 until the user decides otherwise.

## 6. How to choose and report

- **Parameter sweep, not a single value.** Report the result over a sweep of each strategy's parameters (for DBSCAN,
  ε × minPts), the region of stability, and where the chosen value lies in it (Nieves et al. 2023; Verzelli et al.
  2022; Bíró et al. 2025); internal validation by DBCV where it applies (Moulavi et al. 2014; Hammer et al. 2025).
- **Scale declared.** State the scale the clusters live at (ε, radius, persistence or density threshold) against the
  localization precision in xy and z; clustering below the single-fluorophore scale is not significant (Nicovich et
  al. 2017).
- **Null per axon.** Test whether there is clustering at all before partitioning, with Monte Carlo envelopes against a
  null built on the same axon's geometry and, when measured, its blinking (Nieves et al. 2023; Platzer et al. 2020;
  Schneider & Schütz 2022).
- **Minimum report** *(inference, no consensus checklist exists for SMLM cluster analysis)*: algorithm and version;
  parameters and how they were chosen; blinking correction before clustering (method and thresholds); global
  clustering test with envelopes; sensitivity to ± parameters; precision in xy and z; localization density; fraction
  of localizations left unassigned. General guidance: Schmied et al. (2024), Aaron & Chew (2021), Jost & Waters (2019).

## 7. Choosing a method (D11)

The label a method carries in the 'Compare methods' dialog ("recomendable", "preferible", "no recomendable") is
computed **only on the synthetic benchmark** (`docs/sims/clustering_benchmark/verdicts.json`, written by
`tools/cluster_benchmark.py`), never on the user's axon:

- reference: DBSCAN 25/10 (the compatibility mode);
- three primary molecule metrics: 1:1 (`exact`), fusions (`merge`), losses (`missed`); paired differences
  (same scenario and replicate, 10 replicates) with a 95 % t interval;
- **non-inferiority margin Δ = 5 points** [ad hoc: decision D11 of the user, as are the 10× time factor and the 10
  replicates]: a method is *recomendable* in a regime when the interval
  of `exact` − reference stays above −Δ, the intervals of `merge` and `missed` − reference stay below +Δ, and its
  median time is ≤ 10× DBSCAN's; *preferible* when, besides, at least one interval excludes 0 on the favourable
  side [ad hoc: D11 names the label; the rule for "superior" is this PR's]; *no recomendable* otherwise;
- one label per regime (sparse ≈ 200 nm spacing, medium ≈ 100 nm), from the scenarios `*_lab1` / `*_lab2`.
- **Seeds.** Since 2026-10-16 replicate r of scenario s is generated with `default_rng([seed, crc32(s), r])`. Before,
  `default_rng([seed, r])` made `sparse_lab1` and `sparse_lab2` (and the two medium scenarios) share molecule
  positions and background, so the "20 pairs" of a regime were 10 correlated blocks and the intervals were
  mis-calibrated (REVIEW_D0 M2). Now the 20 pairs are independent. The flat-window scenario is its own regime.
- **Which regime matters.** The project's axons are in the sparse regime (~5 clusters/µm, Gazal et al. 2026); the
  medium regime is a stress test. A method "preferible" only in the medium regime is not a reason to change the
  default for these data.
- **Labels of the run of 2026-10-16** (verified defaults; `docs/clustering_benchmark.md` section 5,
  `docs/sims/clustering_benchmark/RESULTS_D0.md`): HDBSCAN, FOCAL and ToMATo *preferible* in both regimes (fewer
  losses than DBSCAN; in the sparse regime 1.3-2.7 points fewer one-to-one molecules, inside the margin);
  GraphCommunity, OPTICS (diagnostic), voronoi_csr and voronoi_factor *no recomendable*. Times were measured with one
  core per method.

"No benchmark" means the method is not in the verdicts file (or the file is missing): the method may still be
used; nothing is said about it. Since the run of 2026-10-16 the labels judge defaults that were verified
(`criterion.status` = "defaults verificados 2026-10-16; benchmark sintético"), so the dialog no longer appends
"(provisional)"; OPTICS carries the note "no recomendable / diagnóstico" (`role` in the file).

Every method also reports its own length scale, `cluster_scale_nm` (exported; informative): eps for DBSCAN, the grid
for FOCAL, the square root of the mean area of the selected cells for Voronoi [derivation], the radius for ToMATo /
GraphCommunity, the median core distance for HDBSCAN / OPTICS-ξ [derivation], the declared value for external labels.
The **edge-curation margin** is a different number (`ClusteringStrategy.edge_margin_nm`, not exported): ε for DBSCAN,
exactly as before Phase D; for the other methods the larger of the compatibility ε and the method's separation scale
(HDBSCAN / OPTICS: the largest core distance of the clustered points, the ε that would reproduce them) [derivation; the
floor at ε is ad hoc]. The median core distance and the Voronoi spacing are intra-cluster scales of 10-15 nm and, used
as a margin, curated less than DBSCAN does.

## 8. On-demand execution (D12)

- Loading an axon runs **nothing**. 'Analyse' runs the one saved method (DBSCAN 25/10 on a fresh install).
- In the parameter strip, every method but DBSCAN is listed only after ticking **Experimental clustering
  methods** (D10); **Compatibility 25/10** puts DBSCAN eps 25 nm / min samples 10 back (bit for bit the analysis
  before Phase D: the golden of the interface stays identical in that mode).
- **Compare methods...** opens a dialog; only the methods ticked there run, only when Run is pressed, in a worker
  thread with progress and cancel; the table shows K/µm, median 1NN, median r_eff, occupancy, number of clusters,
  time, `scale_nm` and the D11 verdict per regime. **Use this method** saves the choice (settings:
  `cluster_method`, `cluster_params`, `cluster_methods_compared`; export columns of the same names, axon tables
  v10) and only that method runs in the analyses that follow. Several rows selected: the first becomes the
  method in use and every method that **ran** in the dialog is recorded as compared [ad hoc: running several
  methods per analysis is not implemented in D0].
- **Cancel and closing.** Cancel stops before the next method; the method running finishes first (the clustering
  functions cannot be interrupted). Closing the dialog (Close, Esc or the window's X) while methods run cancels them
  and the dialog closes when the running method ends; Run, Use and Close are disabled meanwhile. A method that fails
  shows its traceback on its row. The time estimate scales OPTICS and voronoi_csr faster than linearly
  (simulation: OPTICS 1.1 / 3.2 / 7.4 s at 2k / 4k / 8k localizations); read it as a lower bound.
- **Parameters.** Every numeric parameter of the registry gets a field, with no per-method code. A parameter whose
  default is None is offered as the lowest step of its field: «no limit» for `max_*`, «auto» for the rest (the
  method derives the value); None is never written into `cluster_params`. Each edit is checked by building the
  method: values it rejects are flagged under the panel and not applied.
- **Batch.** «Compare methods (slow)» is off by default and runs exactly the methods recorded in
  `cluster_methods_compared` (the methods that ran in the axon window's «Compare methods...»), with DBSCAN first as
  the reference when it is not among them; saved parameters for the saved method, registry defaults for the others,
  into `batch_methods_comparison.csv` beside the exported table. With nothing compared the box is disabled («Choose
  methods in Compare methods... first»). It never adds a method by itself, so with «Experimental clustering
  methods» off no experimental method runs unless the user compared it.
- **Same contour as the analysis shown.** Each method is run on the slab of the analysis shown with its ROI, eps /
  min samples (for DBSCAN), occupancy threshold, and its contour construction: 2-opt from every start (the default
  since 2026-09-19) and the same contour guide, so the DBSCAN row is the analysis shown. This is slower when a method
  finds many clusters. Randomization is off for speed [ad hoc], and the clusters cut by the Picasso pick are kept even
  when the analysis shown leaves them out. The analysis that follows 'Use this method' runs the full pipeline.
- **A method that is not installed.** A settings file naming a method whose optional package is missing on this
  machine (HDBSCAN without `hdbscan`) falls back to DBSCAN with a warning in the log; that method's parameters are
  dropped, and the list of methods compared is kept.
- **Scope in D0.** The chosen method makes the clusters of the per-axon analysis ('Analyse', the strip's re-runs,
  Batch, 'Compare methods'). Everything else that clusters still runs DBSCAN with its own parameters whatever
  method is chosen [not implemented in D0]: the ring analysis (`tools/mps_multisegment.py`, the rings panel) and
  the columns window (`tools/mps_columns.py`) with the strip's eps / min samples; Two channels
  (`tools/mps_crosschannel.py`; channel 2 with its own eps / min samples); and the tools that are DBSCAN by
  construction (k-distance graph, axial groups, sizer, bead finder). Their outputs do not carry `cluster_method`.
- `analysis_id` gains the method and its parameters only for a method other than DBSCAN; it changes from v9 to v10
  for every row anyway, because the table version is part of it (as at every version).

### Overlaying a second method on the map

'Compare methods...' fills a table; it does not change the plot. To SEE where two methods part ways, select one
row of the table (a method other than the one of the analysis shown) and press **Overlay on the map**. The dialog
closes and the axon map gets a group **Second method: B over A** at the top of its layers, with two rows, both
ticked:

* **A clusters (solid)**: every cluster the analysis shown kept (after its edge curation), outlined by the convex
  hull of its localizations, thin and solid in the sky blue of its own localizations;
* **B clusters (dotted)**: the same for the method of the selected row, run on the same slab, dotted in reddish
  purple (the map's only dotted line).

A cluster of A cut by two dotted loops is a split; one dotted loop around two solid ones is a fusion; a loop with
no partner is a cluster the other method does not have. Clusters of fewer than three distinct localizations have no
outline. The group's caption counts these with the synthetic benchmark's matching rules
(`tools.cluster_benchmark.molecule_metrics`, with A's clusters in the role of the molecules and A's unclustered
localizations in the role of the background; `tools/method_overlay.py`):

| count | rule (all thresholds the benchmark's, ad hoc) |
|---|---|
| one-to-one | B clusters whose majority A cluster holds ≥ 80 % of the B cluster, and the B cluster ≥ 80 % of it |
| split | A clusters with ≥ 20 % of their localizations in each of two or more B clusters |
| merge | B clusters in which two or more A clusters are present (≥ 20 % each) |
| A without a partner | A clusters with < 50 % of their localizations in any B cluster |
| B mostly on A's unclustered localizations | B clusters with > 50 % of their localizations in no A cluster |
| ARI | adjusted Rand index of the two labellings on the localizations both put in a cluster |

The roles are not symmetric ("split" and "merge" are read from A's side). It is display only: neither analysis
changes, the method in use stays A, and nothing is exported. The overlay belongs to the analysis it was drawn on:
any re-run, another selection or another file drops it, and **Clear the overlay** under its rows removes it. Its
rows are disabled while the centres are coloured by the two widefield images (the spectrin rows use the same
purple) and with the segments' source. Two methods at most: the analysis shown and one other (to compare B with C,
make B the method in use first).

`B` runs as every comparison does (same slab, ROI and contour; randomization off; the clusters cut by the pick kept,
see above), so its outline set is the one its table row describes.

### Every method on the map

The overlay above draws one other method, and only through its button. Since 2026-10-22 every method that ran in
'Compare methods...' stays on the map when the dialog is closed (with **Close**, or with **Overlay on the map**; not
with **Use this method**, which re-runs the analysis): a group **Methods on the map** at the top of the layers has
one row per method, the analysis shown first, each with its own colour AND line pattern, so that any of them can be
ticked or unticked and two or more seen together or apart (`tools.method_overlay.MethodLayers`):

| slot | line | colour |
|---|---|---|
| 1 (the analysis shown) | thin solid | sky blue |
| 2 | dotted | reddish purple |
| 3 | dashed | orange |
| 4 | dash-dot | bluish green |
| 5 | dash-dot-dot | vermillion |
| 6 | thin dashed | grey |
| 7 | thick dashed | sky blue |
| 8 | thick dash-dot | orange |

Eight slots (a ninth method takes slot 1's style again; the row's label names its line). The map already uses
every chromatic Okabe-Ito colour, so colour alone cannot tell eight methods apart under colour-vision deficiency:
two slots whose colours come within 25 CIELAB units under a simulated deuteranopia, protanopia or tritanopia
differ in line pattern or width (checked by `validate_plot_colours.py`, "every compared method's outlines differ
by colour or line"). The first three rows start ticked [ad hoc: three loops per cluster are still readable].

The caption counts each method against one reference with the rules of the table above: DBSCAN when it ran (the
published pipeline, Gazal et al. 2026), else the analysis shown. A compared voronoi_factor or voronoi_csr also
brings its own **Voronoi cells** group (the same layer as above, rebuilt from that run's own fit), even when the
analysis shown is another method. **Method diagnostics...** under the rows opens one tab per method with its own
diagnostic plot where it costs no new clustering or one cheap fit (`tools/method_diagnostics.py`): the k-distance
graph of DBSCAN with eps [Ester et al. 1996], the histogram of the Voronoi cells' log density with the threshold
applied [Levet et al. 2015], the persistence diagram of ToMATo with the line birth - death = tau [Chazal et al.
2013; Pike et al. 2020]. Not offered, and why: OPTICS (its reachability plot needs the slow method to run again),
both HDBSCAN (the condensed tree is not kept, and scikit-learn's does not expose one), FOCAL (no standard plot),
GraphCommunity (a stand-in for StormGraph), External labels (labels from outside), Emitters (its own map layer and
the xi checks are its diagnostics). Each plot exports as a figure. Display only, as the overlay: nothing of the
analysis shown changes, and a re-run drops the rows (**Clear the methods** removes them).

## 9. Comparing analyses across methods

The question the user asked for (plan of 2026-10-22) is not "which method is right" but **"would the conclusion
change if another method had made the clusters?"**. `tools/multimethod.py` answers it for every analysis of an axon,
not only for the four numbers of 'Compare methods...'. It is a computing module (no window); the interface uses it.

**What runs.** Each method chosen runs the whole per-axon analysis on the **same localizations**: the same axial
slab, ROI, edge-curation settings (DBCV threshold; the compatibility eps as the floor of a method's edge margin, as
in section 8), pick-boundary rule, occupancy settings and contour construction (2-opt from every start, same contour
guide). Only the clustering changes, so every difference in the table is the method's. A run whose slab differs from
the reference's is refused as an error: the comparison is meaningless on other localizations. The randomization
control is off unless asked for [ad hoc: speed, as in section 8]. A method that fails gives a row with its error; the
others still run. Slow methods (`tools.cluster_method_ui.is_slow`: Emitters, OPTICS, voronoi_csr) are refused
unless the user allows them, and the estimated time is shown before anything runs (rough per-method constants
measured on a synthetic ring, a lower bound with many localizations). Cancel stops before the next method, or inside
the running one at its next check (`tools/mps_cancel.py`).

**The reference is always there.** DBSCAN eps 25 nm / min samples 10, the published pipeline (Gazal et al. 2026), is
run first in every comparison, whether or not it was ticked, under the name `DBSCAN 25/10`. Two reasons: every
published number this program reproduces comes from it, so it is the only method whose readouts can be set beside the
literature; and a difference between two experimental methods says nothing about the published conclusion unless
one can also see where each stands against the reference. A DBSCAN with other parameters is a separate run with its
parameters in its name.

**What is compared.** Three tables, all one row per axon and something:

| table | one row per | what it says |
|---|---|---|
| long | axon × method × readout | every readout of every analysis (the four of 'Compare methods...' - K/µm, median 1NN, median r_eff, occupancy - plus clusters kept, the regularity index, and whatever the later analyses add: pipeline null, gap measures, eclipse tolerance, apparent size); the wide view puts one method per row and one readout per column |
| agreement | axon × pair of methods | how the two partitions correspond: adjusted Rand index (Hubert & Arabie 1985) on the localizations both put in a cluster, and the one-to-one / split / merged / unmatched counts of section 8's overlay (the synthetic benchmark's majority rules; thresholds 80 / 20 / 50 %, ad hoc). The reference is method A of every pair it is in |
| summary | axon × readout | the smallest and largest value over the methods and which methods give them, the range, and the range relative to the reference's value; for a p-value or a verdict, each method's verdict and whether it agrees with the reference's |

**How to read disagreement.**

* A **number** (K/µm, a median distance, a size) always differs between methods: each method draws the cluster
  boundary somewhere else, and that is what a method is. Its spread is reported, not judged: `large_spread` flags a
  range above 10 % of the reference's value [ad hoc: a flag for the eye; no measurement error of these readouts
  across methods is known yet]. `depends_on_method` says "spread only" for numbers.
* A **p-value or a verdict** is a conclusion. Each method's p gives "reject" at p ≤ α (0.05 by default [ad hoc:
  conventional level]; each readout declares its own) or "no evidence". `depends_on_method` is **yes** when at least
  one method's verdict differs from the reference's, **no** when the reference and at least one other method have a
  verdict and all agree, **undetermined** otherwise (the reference has none, or no other method does). A "yes" is
  the finding to report: the conclusion is not robust to the choice of clustering for that axon. A "no" with only
  methods of the same family (DBSCAN with two eps, say) is weaker than a "no" across families.
* A low **ARI** with similar readouts means the methods draw different clusters that happen to give the same summary
  number; a high ARI with different readouts means the curation or the contour amplified a small difference. Read
  the agreement table before trusting the summary.
* Over many axons, the summary over axons counts on how many the verdict depends on the method. Pooling verdicts
  over axons is not a test; the batch statistics (by animal) remain the tests.

**For developers: adding an analysis.** An analysis is a plug-in, `register_analysis(name, fn, readouts)`, where
`fn(run, context)` is a pure function of one method's result and returns its readouts; a null re-clusters with
`context.rerun(...)` (same method, same parameters, same curation), draws `context.n_replicates` replicates (199 by
default, the user's decision of 2026-10-22, 999 on demand; with 199, α(B + 1) is an integer at α = 0.05, Davison &
Hinkley 1997) and refuses slow methods unless the user allowed them. The contract (readout naming, how a p-value or a verdict declares itself, randomness, cancellation) is the
module docstring of `tools/multimethod.py`.

**Files.** The long table is written as CSV (comma, decimal point, UTF-8, like every table of the program; the
"for Excel" copy of `tools.results_table.excel_copy` applies), and a file is never overwritten: when the name is
taken the next free `name_2.csv`, `name_3.csv`, ... is used. Every column is described in
`tools.multimethod.MULTIMETHOD_COLUMNS`.

## References

- Aaron J, Chew T-L (2021). A guide to accurate reporting in digital image processing - can anyone reproduce your
  quantitative analysis? *Journal of Cell Science* 134(6):jcs254151. doi:10.1242/jcs.254151
- Andronov L, Michalon J, Ouararhni K, Orlov I, Hamiche A, Vonesch J-L, Klaholz BP (2018). 3DClusterViSu: 3D
  clustering analysis of super-resolution microscopy data by 3D Voronoi tessellations. *Bioinformatics*
  34(17):3004-3012. doi:10.1093/bioinformatics/bty200
- Andronov L, Orlov I, Lutz Y, Vonesch J-L, Klaholz BP (2016). ClusterViSu, a method for clustering of protein
  complexes by Voronoi tessellation in super-resolution microscopy. *Scientific Reports* 6(1):24084.
  doi:10.1038/srep24084
- Ankerst M, Breunig MM, Kriegel H-P, Sander J (1999). OPTICS: Ordering points to identify the clustering structure.
  *ACM SIGMOD Record* 28(2):49-60. doi:10.1145/304181.304187
- Arnold AM, Schneider MC, Hüsson C, Sablatnig R, Brameshuber M, Baumgart F, Schütz GJ (2020). Verifying molecular
  clusters by 2-color localization microscopy and significance testing. *Scientific Reports* 10(1):4230.
  doi:10.1038/s41598-020-60976-6
- Baddeley A, Rubak E, Turner R (2015). *Spatial Point Patterns: Methodology and Applications with R*. Chapman and
  Hall/CRC. doi:10.1201/b19708
- Besag J (1977). Comments on Ripley's paper. *Journal of the Royal Statistical Society: Series B (Methodological)*
  39(2):193-195 [pages to confirm]. doi:10.1111/j.2517-6161.1977.tb01616.x
- Besag J, Diggle PJ (1977). Simple Monte Carlo Tests for Spatial Pattern. *Applied Statistics* 26(3):327-333.
  doi:10.2307/2346974
- Bohrer CH, Yang X, Thakur S, Weng X, Tenner B, McQuillen R, Ross B, Wooten M, Chen X, Zhang J, Roberts E,
  Lakadamyali M, Xiao J (2021). A pairwise distance distribution correction (DDC) algorithm to eliminate
  blinking-caused artifacts in SMLM. *Nature Methods* 18(6):669-677. doi:10.1038/s41592-021-01154-y
- Bíró P, H. Kovács BB, Novák T, Erdélyi M (2025). Cluster parameter-based DBSCAN maps for image characterization.
  *Computational and Structural Biotechnology Journal* 27:920-927. doi:10.1016/j.csbj.2025.02.037
- Campello RJGB, Moulavi D, Sander J (2013). Density-based clustering based on hierarchical density estimates. In J.
  Pei, V. S. Tseng, L. Cao, H. Motoda and G. Xu (Eds.), *Advances in Knowledge Discovery and Data Mining: PAKDD 2013*
  (Lecture Notes in Computer Science, vol. 7819, Part II, pp. 160-172). Springer. doi:10.1007/978-3-642-37456-2_14
- Campello RJGB, Moulavi D, Zimek A, Sander J (2015). Hierarchical density estimates for data clustering,
  visualization, and outlier detection. *ACM Transactions on Knowledge Discovery from Data* 10(1):1-51.
  doi:10.1145/2733381
- Chazal F, Guibas LJ, Oudot SY, Skraba P (2013). Persistence-Based Clustering in Riemannian Manifolds. *Journal of
  the ACM* 60(6), Article 41, 1-38. doi:10.1145/2535927
- Davison AC, Hinkley DV (1997). *Bootstrap Methods and their Application.* Cambridge University Press.
- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768
- Diggle PJ (2013). *Statistical Analysis of Spatial and Spatio-Temporal Point Patterns*. Chapman and Hall/CRC.
  doi:10.1201/b15326
- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochemistry and Cell Biology* 141(6):629-638.
  doi:10.1007/s00418-014-1192-3
- Ester M, Kriegel H-P, Sander J, Xu X (1996). A density-based algorithm for discovering clusters in large spatial
  databases with noise. In E. Simoudis, J. Han and U. Fayyad (Eds.), *Proceedings of the Second International
  Conference on Knowledge Discovery and Data Mining (KDD-96)* (pp. 226-231). AAAI Press.
- Fazel M, Wester MJ, Schodt DJ, Cruz SR, Strauss S, Schueder F, Schlichthaerle T, Gillette JM, Lidke DS, Rieger B,
  Jungmann R, Lidke KA (2022). High-precision estimation of emitter positions using Bayesian grouping of
  localizations. *Nature Communications* 13(1):7152. doi:10.1038/s41467-022-34894-2
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of betaII-spectrin within segments of the membrane-associated periodic skeleton in mouse
  sciatic nerve axons [Reviewed Preprint, version 1]. eLife. doi:10.7554/elife.111108.1 [preprint with eLife public
  review; no version of record]
- Griffié J, Shannon M, Bromley CL, Boelen L, Burn GL, Williamson DJ, Heard NA, Cope AP, Owen DM, Rubin-Delanchy P
  (2016). A Bayesian cluster analysis method for single-molecule localization microscopy data. *Nature Protocols*
  11(12):2499-2514. doi:10.1038/nprot.2016.149
- Griffié J, Shlomovich L, Williamson DJ, Shannon M, Aaron J, Khuon S, Burn GL, Boelen L, Peters R, Cope AP, Cohen
  EAK, Rubin-Delanchy P, Owen DM (2017). 3D Bayesian cluster analysis of super-resolution data reveals LAT recruitment
  to the T cell synapse. *Scientific Reports* 7(1):4077. doi:10.1038/s41598-017-04450-w
- Hammer JL, Devanny AJ, Kaufman LJ (2025). Bayesian optimized parameter selection for density-based clustering
  applied to single molecule localization microscopy. *Communications Biology* 8(1):902.
  doi:10.1038/s42003-025-08332-0
- Hubert L, Arabie P (1985). Comparing partitions. *Journal of Classification* 2(1):193-218. doi:10.1007/bf01908075
- Hyun Y, Kim D (2023). Recent development of computational cluster analysis methods for single-molecule localization
  microscopy images. *Computational and Structural Biotechnology Journal* 21:879-888. doi:10.1016/j.csbj.2023.01.006
- Jensen LG, Hoh TY, Williamson DJ, Griffié J, Sage D, Rubin-Delanchy P, Owen DM (2022). Correction of
  multiple-blinking artifacts in photoactivated localization microscopy. *Nature Methods* 19(5):594-602.
  doi:10.1038/s41592-022-01463-w
- Jost AP-T, Waters JC (2019). Designing a rigorous microscopy experiment: Validating methods and avoiding bias.
  *Journal of Cell Biology* 218(5):1452-1466. doi:10.1083/jcb.201812109
- Khater IM, Nabi IR, Hamarneh G (2020). A review of super-resolution single-molecule localization microscopy cluster
  analysis and quantification methods. *Patterns* 1(3):100038. doi:10.1016/j.patter.2020.100038
- Kiskowski MA, Hancock JF, Kenworthy AK (2009). On the Use of Ripley's K-Function and Its Derivatives to Analyze
  Domain Size. *Biophysical Journal* 97(4):1095-1103. doi:10.1016/j.bpj.2009.05.039
- Kowalewski R, Reinhardt SCM, Pachmayr I, Xu S, Masullo LA, Jungmann R (2026). Molecular mapping in DNA-PAINT via
  modified Gaussian Mixture Modeling. *Nature Communications* 17(1):2315. doi:10.1038/s41467-026-70198-5
- Kutz S, Zehrer AC, Svetlitckii R, Gülcüler Balta GS, Galli L, Kleber S, Rentsch J, Martin-Villalba A, Ewers H
  (2021). An Efficient GUI-Based Clustering Software for Simulation and Bayesian Cluster Analysis of Single-Molecule
  Localization Microscopy Data. *Frontiers in Bioinformatics* 1:723915. doi:10.3389/fbinf.2021.723915
- Levet F (2023). Optimizing Voronoi-based quantifications for reaching interactive analysis of 3D localizations in
  the million range. *Frontiers in Bioinformatics* 3:1249291. doi:10.3389/fbinf.2023.1249291
- Levet F, Hosy E, Kechkar A, Butler C, Beghin A, Choquet D, Sibarita J-B (2015). SR-Tesseler: A method to segment and
  quantify localization-based super-resolution microscopy data. *Nature Methods* 12(11):1065-1071.
  doi:10.1038/nmeth.3579
- Lörzing P, Schake P, Schlierf M (2024). Anisotropic DBSCAN for 3D SMLM data clustering. *The Journal of Physical
  Chemistry B* 128(33):7934-7940. doi:10.1021/acs.jpcb.4c02030
- Marenda M, Lazarova E, van de Linde S, Gilbert N, Michieletto D (2021). Parameter-free molecular super-structures
  quantification in single-molecule localization microscopy. *Journal of Cell Biology* 220(5):e202010003.
  doi:10.1083/jcb.202010003
- Mazouchi A, Milstein JN (2016). Fast Optimized Cluster Algorithm for Localizations (FOCAL): A spatial cluster
  analysis for super-resolved microscopy. *Bioinformatics* 32(5):747-754. doi:10.1093/bioinformatics/btv630
- McInnes L, Healy J, Astels S (2017). Hdbscan: Hierarchical density based clustering. *The Journal of Open Source
  Software* 2(11):205. doi:10.21105/joss.00205
- Mollazade M, Tabarin T, Nicovich PR, Soeriyadi A, Nieves DJ, Gooding JJ, Gaus K (2017). Can single molecule
  localization microscopy be used to map closely spaced RGD nanodomains? *PLOS ONE* 12(7):e0180871.
  doi:10.1371/journal.pone.0180871
- Moulavi D, Jaskowiak PA, Campello RJGB, Zimek A, Sander J (2014). Density-Based Clustering Validation. In
  *Proceedings of the 2014 SIAM International Conference on Data Mining* (pp. 839-847). Society for Industrial and
  Applied Mathematics. doi:10.1137/1.9781611973440.96
- Nicovich PR, Owen DM, Gaus K (2017). Turning single-molecule localization microscopy into a quantitative
  bioanalytical tool. *Nature Protocols* 12(3):453-460. doi:10.1038/nprot.2016.166
- Nieves DJ, Pike JA, Levet F, Williamson DJ, Baragilly M, Oloketuyi S, de Marco A, Griffié J, Sage D, Cohen EAK,
  Sibarita J-B, Heilemann M, Owen DM (2023). A framework for evaluating the performance of SMLM cluster analysis
  algorithms. *Nature Methods* 20(2):259-267. doi:10.1038/s41592-022-01750-6
- Nino DF, Djayakarsana D, Milstein JN (2020). FOCAL3D: A 3-dimensional clustering package for single-molecule
  localization microscopy. *PLOS Computational Biology* 16(12):e1008479. doi:10.1371/journal.pcbi.1008479
- Owen DM, Rentero C, Rossy J, Magenau A, Williamson D, Rodriguez M, Gaus K (2010). PALM imaging and cluster analysis
  of protein heterogeneity at the cell surface. *Journal of Biophotonics* 3(7):446-454. doi:10.1002/jbio.200900089
- Paupiah A-L, Marques X, Merlaud Z, Russeau M, Levi S, Renner M (2023). Introducing Diinamic, a flexible and robust
  method for clustering analysis in single-molecule localization microscopy. *Biological Imaging* 3:e14.
  doi:10.1017/s2633903x23000156
- Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, Blondel M, Prettenhofer P, Weiss R, Dubourg V,
  Vanderplas J, Passos A, Cournapeau D, Brucher M, Perrot M, Duchesnay É (2011). Scikit-learn: Machine learning in
  Python. *Journal of Machine Learning Research* 12:2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Pike JA, Khan AO, Pallini C, Thomas SG, Mund M, Ries J, Poulter NS, Styles IB (2020). Topological data analysis
  quantifies biological nano-structure from single molecule localization microscopy. *Bioinformatics* 36(5):1614-1621.
  doi:10.1093/bioinformatics/btz788
- Pineda J, Masó-Orriols S, Masoliver M, Bertran J, Goksör M, Volpe G, Manzo C (2025). Enhanced spatial clustering of
  single-molecule localizations with graph neural networks. *Nature Communications* 16(1):9693.
  doi:10.1038/s41467-025-65557-7
- Platzer R, Rossboth BK, Schneider MC, Sevcsik E, Baumgart F, Stockinger H, Schütz GJ, Huppa JB, Brameshuber M
  (2020). Unscrambling fluorophore blinking for comprehensive cluster detection via photoactivated localization
  microscopy. *Nature Communications* 11(1):4993. doi:10.1038/s41467-020-18726-9
- Ripley BD (1977). Modelling spatial patterns. *Journal of the Royal Statistical Society Series B: Statistical
  Methodology* 39(2):172-192. doi:10.1111/j.2517-6161.1977.tb01615.x
- Rossboth B, Arnold AM, Ta H, Platzer R, Kellner F, Huppa JB, Brameshuber M, Baumgart F, Schütz GJ (2018). TCRs are
  randomly distributed on the plasma membrane of resting antigen-experienced T cells. *Nature Immunology*
  19(8):821-827. doi:10.1038/s41590-018-0162-7
- Rubin-Delanchy P, Burn GL, Griffié J, Williamson DJ, Heard NA, Cope AP, Owen DM (2015). Bayesian cluster
  identification in single-molecule localization microscopy data. *Nature Methods* 12(11):1072-1076.
  doi:10.1038/nmeth.3612
- Schmied C, Nelson MS, Avilov S, Bakker G-J, Bertocchi C, Bischof J, Boehm U, Brocher J, Carvalho MT, Chiritescu C,
  Christopher J, Cimini BA, Conde-Sousa E, Ebner M, Ecker R, Eliceiri K, Fernandez-Rodriguez J, Gaudreault N, Gelman
  L, ..., Jambor HK (2024). Community-developed checklists for publishing images and image analyses. *Nature Methods*
  21(2):170-181. doi:10.1038/s41592-023-01987-9
- Schneider MC, Schütz GJ (2022). Don’t Be Fooled by Randomness: Valid p-Values for Single Molecule Microscopy.
  *Frontiers in Bioinformatics* 2:811053. doi:10.3389/fbinf.2022.811053
- Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R (2017). Super-resolution microscopy with
  DNA-PAINT. *Nature Protocols* 12(6):1198-1228. doi:10.1038/nprot.2017.024
- Schubert E, Sander J, Ester M, Kriegel HP, Xu X (2017). DBSCAN Revisited, Revisited: Why and How You Should (Still)
  Use DBSCAN. *ACM Transactions on Database Systems* 42(3):1-21. doi:10.1145/3068335
- Scurll JM, Abraham L, Zheng DW, Tafteh R, Chou KC, Gold MR, Coombs D (2019). StormGraph: A graph-based algorithm for
  quantitative clustering analysis of diverse single-molecule localization microscopy data [Preprint]. bioRxiv.
  doi:10.1101/515627 [preprint, not peer reviewed]
- Sengupta P, Jovanovic-Talisman T, Skoko D, Renz M, Veatch SL, Lippincott-Schwartz J (2011). Probing protein
  heterogeneity in the plasma membrane using PALM and pair correlation analysis. *Nature Methods* 8(11):969-975.
  doi:10.1038/nmeth.1704
- Shivanandan A, Unnikrishnan J, Radenovic A (2015). Accounting for Limited Detection Efficiency and Localization
  Precision in Cluster Analysis in Single Molecule Localization Microscopy. *PLOS ONE* 10(3):e0118767.
  doi:10.1371/journal.pone.0118767
- Shivanandan A, Unnikrishnan J, Radenovic A (2016). On characterizing protein spatial clusters with correlation
  approaches. *Scientific Reports* 6(1):31164. doi:10.1038/srep31164
- Staszowska AD, Fox-Roberts P, Hirvonen LM, Peddie CJ, Collinson LM, Jones GE, Cox S (2018). The Rényi divergence
  enables accurate and precise cluster analysis for localization microscopy. *Bioinformatics* 34(23):4102-4111.
  doi:10.1093/bioinformatics/bty403
- Veatch SL, Machta BB, Shelby SA, Chiang EN, Holowka DA, Baird BA (2012). Correlation Functions Quantify
  Super-Resolution Images and Estimate Apparent Clustering Due to Over-Counting. *PLoS ONE* 7(2):e31457.
  doi:10.1371/journal.pone.0031457
- Verzelli P, Nold A, Sun C, Heilemann M, Schuman EM, Tchumatchenko T (2022). Unbiased choice of global clustering
  parameters for single-molecule localization microscopy. *Scientific Reports* 12(1):22561.
  doi:10.1038/s41598-022-27074-1
- Ward S, Cohen EAK, Adams N (2021). Testing for complete spatial randomness on three dimensional bounded convex
  shapes. *Spatial Statistics* 41:100489. doi:10.1016/j.spasta.2020.100489
- Williamson DJ, Burn GL, Simoncelli S, Griffié J, Peters R, Davis DM, Owen DM (2020). Machine learning for cluster
  analysis of localization microscopy data. *Nature Communications* 11(1):1493. doi:10.1038/s41467-020-15293-x
- Wu Y-L, Tschanz A, Krupnik L, Ries J (2020). Quantitative data analysis in single-molecule localization microscopy.
  *Trends in Cell Biology* 30(11):837-851. doi:10.1016/j.tcb.2020.07.005
