# Clusters cut by the Picasso pick

> **Resumen (español).** Cada axón se analiza desde un archivo que es la selección de un *pick* dibujado a mano en
> Picasso sobre el campo entero. Donde el borde del pick pasa por un cluster, parte de sus localizaciones queda afuera
> del archivo y el cluster se mide más chico, con el centro corrido. MPS Explorer marca como **cortado** a todo
> cluster conservado que tenga **alguna localización a menos de 1 NeNA** (la precisión de localización, medida en el
> archivo del campo entero sin *link*) del borde del pick. No se usa ninguna regla de porcentaje de clusters. La
> razón de 1 NeNA es una derivación: una localización a una precisión del borde tiene un 24 % de probabilidad de que
> otra localización de la misma molécula haya caído del otro lado (8 % a dos precisiones). En simulaciones propias
> (800 campos, σ de 5 a 16 nm) la regla encuentra el 91-97 % de los clusters cortados y el 97,5-100 % de los que
> perdieron al menos el 10 % de sus localizaciones; marca además el 5-11 % de los clusters intactos (los que quedaron
> pegados al borde sin perder nada). Los clusters cortados se cuentan ("k de K (x %)"), se dibujan en el mapa en una
> capa que se puede apagar, y se escriben en las tablas; **sólo se excluyen del análisis si el usuario tilda
> "Exclude cut clusters"** (destildado por defecto). Sin el borde del pick o sin NeNA la regla **no se aplica** y la
> tabla lo dice ("boundary unknown: not evaluated"); nada se adivina. La alternativa "selección por centroide"
> (agrupar el campo entero y quedarse con los clusters cuyo centroide cae dentro del pick) se informa al lado
> (sección 5).

This page explains the cut-cluster rule of MPS Explorer: the problem, the rule and its derivation, where the pick
boundary and the NeNA come from, the validation on simulated axons (with the alternative "centroid selection"), what
the program shows and writes, and the limitations. The code is `tools/mps_cut_clusters.py` (the rule),
`tools/mps_picks.py` (Picasso's pick files) and `tools/mps_cut_ui.py` (the widgets). Every number below is labelled:

- **literature**: a published value or method, with its reference;
- **derivation**: shown here;
- **own simulation**: synthetic data only (no measured file), with the script and seed that reproduce it. The script
  is `docs/sims/cut_clusters/cut_sim.py`; the JSON committed beside it (`cut_sim.json`) is the run quoted here;
- **ad hoc**: a choice with no derivation, said so.

## 1. The problem

A per-axon file is what a Picasso Render pick kept of the whole field (Schnitzbauer et al. 2017): every localization
inside the pick, nothing outside. The pick is drawn by hand, usually as a polygon around one axon. Where its edge
runs through a spectrin cluster, the localizations of that cluster beyond the edge never reach the file. The cluster
is then measured with fewer localizations, a smaller convex-hull area and a centre pulled inwards, and its
nearest-neighbour distance, its share of the perimeter occupancy and the contour through it move with it. Without
handling, 16-22 % of the clusters analysed in the simulated fields of section 4 had lost at least a tenth of their
localizations (own simulation).

The automatic curation's edge criterion (`tools/cluster_quality.py`) is a different thing: it removes clusters at
the edge of the ROI drawn **inside MPS Explorer**, not at the edge of the Picasso pick, which MPS Explorer did not know
until now.

## 2. The rule and why 1 NeNA

**Rule.** A cluster the curation kept is *cut* when **any** of its localizations lies within `m x NeNA` of the pick
boundary, with `m = 1` (decided with the user). The distance is the shortest distance in x, y to the boundary (the
pick has no z). Each cluster is judged on its own localizations; no share of clusters decides anything (the user
rejected rules such as "discard if more than half of the clusters ..." as arbitrary).

**Derivation.** NeNA estimates the localization precision σ, the standard deviation per axis of a localization around
its molecule (Endesfelder et al. 2014; literature). Take a straight edge, and a localization observed at distance d
inside it. With no prior knowledge of where the molecule is (molecules on both sides of the edge equally likely, as
when the pick cuts through a cluster), the molecule lies at d + N(0, σ²) along the normal, and a second localization
of the same molecule at d + N(0, 2σ²). The chance that this sibling fell beyond the edge, and so is missing from the
file, is

    P(sibling beyond the edge | d) = Φ(-d / (√2 σ))

which is **0.36 at d = 0.5 σ, 0.24 at d = σ and 0.08 at d = 2 σ** (derivation). A molecule localized n times has
n - 1 such siblings, so the chance that at least one of them is missing grows with n (not as for independent draws:
all of them share the molecule's position), and a cluster holds several molecules. So a localization within one precision of the edge is the trace a cut leaves: the chance
that something of its molecule is missing is about one in four per sibling; at two precisions it falls to one in
twelve, and the rule would start flagging clusters only because they are near the edge.

Checked by Monte Carlo (own simulation, `derivation_check` in `cut_sim.py`, seed 59, 400,000 molecules, σ = 10 nm):
0.355 / 0.241 / 0.070 against 0.362 / 0.240 / 0.079 from the formula. When every molecule lies inside the edge (no
structure beyond it, only the localization error can carry a sibling out), the share is lower: 0.229 / 0.166 / 0.064.

**Sensitivity to m.** 0.5 and 2 NeNA are reported in section 4 beside the default; the default does not change with
them. A smaller m misses more cut clusters, a larger one flags more intact ones.

## 3. The boundary and the NeNA

**The boundary** is read from the YAML file Picasso Render's *Save picks* writes (`tools/mps_picks.py`; format read
from Picasso v0.9.10's own source, `picasso/gui/render.py`, `save_picks` and `load_picks`): `Shape` Polygon
(`Vertices`, one closed list of corners per pick), Circle (`Centers`, `Diameter (nm)`, or `Diameter` in camera pixels
in older files), Rectangle (`Center-Axis-Points`, `Width (nm)`) or Square (`Centers`, `Side Length (nm)`), all in
camera pixels. *Save picked localizations* does not write the regions: it writes a `group` column (the index of the
pick) and a metadata step with `Pick Shape`, `Number of picks` and `Pick Areas (um^2)` (or `Area (um^2)` of the one
pick, when one file per pick is saved).

**Which pick.** The pick files beside the per-axon file and one folder up are read (and the ones the user pointed to
in that folder). A pick is a candidate when it holds every localization of the file (Picasso's even-odd test, with
0.001 camera pixel for the float32 storage of x and y; derivation). It is kept when nothing the file records
contradicts it: its index is the `group` value, its shape the recorded shape, its area the recorded area within 1 %
(ad hoc tolerance; the only differences expected are rounding and a retyped display pixel size). One such pick, or
several copies of the same pick, is a match. Two or more different picks, or only contradicted ones, are
**ambiguous**: the rule is not applied and the user is asked to choose (*Pick boundary...*). Nothing is chosen by a
score. The user can also point to the pick file by hand, or set the boundary aside for a file.

**The NeNA** is the precision of channel 1 measured on the **whole-field, unlinked** file (`tools.mps_quality.nena`,
docs/nena.md): a picked file seldom has the 500 next-frame pairs NeNA needs, and a linked file has merged them. The
user measures it once per acquisition (*Measure on the whole-field file...*); it is stored for the folder of the
per-axon files (one folder = one acquisition, the convention of the stored pixel sizes) and used for every axon
there. With no stored value, the per-axon file's own NeNA is used when it has enough pairs and is not linked; the
table names the source (`cut_nena_source`). With neither, the rule is not applied ("NeNA unknown: not evaluated").

**Not applied, and said so.** No pick file found, no pick holding the file, an ambiguous match, no NeNA, a boundary
that leaves analysed localizations outside it (another pick, another pixel size): the status and the reason go to the
results table, the map and the exported row. Nothing is guessed in their place.

## 4. Validation on simulated axons

**Setup** (own simulation, `cut_sim.py`; seeds 59000 + 1000 x index(σ) + field). Each field is one axon of the
project's simulator (`tools.mps_simulate_axon`): an ellipse membrane (semi-axes 600-1500 nm), clusters on it at 4 per
µm with an 80 nm hard core, 15 fluorophores per cluster on average, 3 localizations per fluorophore in consecutive
frames, a 25 nm epitope disc and 12 nm linkage (the simulator's defaults), a lateral error of sd σ. One ring is
analysed. The pick is drawn like a hand-drawn polygon: 12 corners at jittered angles, each at the membrane's radius
plus a margin uniform in [-150, +250] nm (ad hoc: wide enough for the edge to run through the ring in places).
DBSCAN eps 25 nm, min samples 10 (Gazal et al. 2026). NeNA is measured on the simulated whole field; its median was
1.00 σ at every σ. 200 fields per σ, 800 in all (169 s on 4 processes).

**Truth.** A true cluster is *cut* when some of its localizations are inside the pick and some outside, and
*materially cut* when it lost at least 10 % of them (ad hoc threshold, for reading only). Each DBSCAN cluster of the
per-axon file is named by the true cluster most of its localizations come from.

**The rule** (sensitivity = cut clusters flagged / cut clusters; specificity = intact clusters not flagged / intact
clusters; PPV = flagged clusters that were cut / flagged clusters):

| σ (nm) | m (NeNA) | sensitivity | sensitivity, materially cut | specificity | PPV |
|---|---|---|---|---|---|
| 5 | 0.5 | 0.785 | 0.875 | 0.974 | 0.893 |
| 5 | **1** | **0.913** | **0.975** | **0.952** | **0.838** |
| 5 | 2 | 0.967 | 0.990 | 0.918 | 0.762 |
| 8 | 0.5 | 0.884 | 0.964 | 0.949 | 0.860 |
| 8 | **1** | **0.971** | **1.000** | **0.915** | **0.803** |
| 8 | 2 | 0.996 | 1.000 | 0.854 | 0.708 |
| 12 | 0.5 | 0.879 | 0.977 | 0.957 | 0.898 |
| 12 | **1** | **0.972** | **0.996** | **0.907** | **0.818** |
| 12 | 2 | 0.995 | 1.000 | 0.810 | 0.691 |
| 16 | 0.5 | 0.906 | 0.978 | 0.941 | 0.893 |
| 16 | **1** | **0.966** | **0.996** | **0.891** | **0.828** |
| 16 | 2 | 0.998 | 1.000 | 0.755 | 0.690 |

(own simulation; the counts behind each line are in `cut_sim.json`, `summary`.)

At 1 NeNA the rule finds 91-97 % of the cut clusters and 97.5-100 % of the materially cut ones. The cut clusters it
misses lost one or a few localizations. The intact clusters it flags (5-11 %) are clusters the edge passed just
outside of: whole in this simulation, but in contact with the edge, the clusters whose loss a slightly different pick
would have caused. 2 NeNA buys little sensitivity for many more such flags; 0.5 NeNA misses up to a fifth of the
cut clusters.

## 5. The rule against centroid selection

*Centroid selection* clusters the whole field first and keeps a cluster when its centroid lies inside the pick. Its
clusters are never cut, but the decision of which clusters straddling the edge belong to the axon is made by where
their centre falls. Per method, the clusters analysed per field, the share of them missing at least 10 % of their
localizations (against the same cluster found by DBSCAN on the whole field, so DBSCAN's own losses are not counted)
and the true clusters with localizations in the pick that the method leaves out entirely (own simulation):

| σ (nm) | method | clusters per field | missing ≥ 10 % | left out per field |
|---|---|---|---|---|
| 5 | no handling | 12.9 | 15.6 % | 1.45 |
| 5 | rule, cut clusters excluded | 9.9 | 0.8 % | 1.45 + the excluded |
| 5 | centroid selection | 12.2 | 0.3 % | 2.13 |
| 8 | no handling | 13.3 | 17.6 % | 1.76 |
| 8 | rule, cut clusters excluded | 9.1 | 0.3 % | 1.76 + the excluded |
| 8 | centroid selection | 12.4 | 0.2 % | 2.54 |
| 12 | no handling | 12.7 | 19.4 % | 2.15 |
| 12 | rule, cut clusters excluded | 8.2 | 0.2 % | 2.15 + the excluded |
| 12 | centroid selection | 11.8 | 0.2 % | 2.88 |
| 16 | no handling | 11.6 | 22.0 % | 2.75 |
| 16 | rule, cut clusters excluded | 6.8 | 0.4 % | 2.75 + the excluded |
| 16 | centroid selection | 10.7 | 0.1 % | 3.41 |

Of the clusters the pick cut, centroid selection kept about half and dropped the other half at every σ (405 / 430,
527 / 507, 595 / 576, 648 / 684 kept / dropped). Both ways remove the damage; they differ in what they cost. The rule
with exclusion analyses fewer clusters (it also excludes the intact clusters touching the edge); centroid selection
keeps more, all whole, but needs the whole field clustered with the same parameters and the pick applied to the
clusters instead of to the localizations. That is not how the per-axon files are made today (Picasso picks
localizations), which is why the rule is the default and centroid selection is reported, not implemented.

## 6. What the program shows and writes

- **Results table** (MPS analysis window): the row *Clusters cut by the pick*, "k of K (x %)" with the rule, the
  NeNA and its source, and whether the cut clusters were left out; or the status ("boundary unknown: not evaluated",
  "NeNA unknown: not evaluated") and the reason.
- **Under the table**: the same count, the switch **Exclude cut clusters** (off by default; for the session) and
  **Pick boundary...**, which opens the dialog to choose an ambiguous pick, point to a pick file, set the boundary
  aside, or measure the NeNA on the whole-field file.
- **Axon map**: the group *Picasso pick boundary*, two rows ticked in the MPS analysis view: the boundary (a cased
  dashed line) and the cut clusters (cased hollow hexagons around their centres), in reddish purple, the one Okabe-Ito
  hue that view does not use; both are told apart by shape as well (`validate_plot_colours.py`), and disabled while
  the centres are coloured by the widefield images, whose spectrin rows use the same purple. The figure title names
  them.
- **Axon table** (v6; renumbered at merge if another table version lands first), at the end of the row:
  `cut_status`, `cut_rule`, `cut_nena_multiple`, `cut_nena_nm`, `cut_nena_source`, `cut_margin_nm`,
  `cut_boundary_source`, `cut_boundary_shape`, `n_clusters_cut` (k), `n_clusters_cut_of` (K), `cut_percent`,
  `cut_excluded`. When excluded, the cut clusters are counted in `n_clusters_removed` and every measured column is
  without them. The clusters table ends with `cut_by_pick`. The column dictionary describes each.
- **Parameter registry**: `cut.nena_multiple` (1, derived), `cut.exclude` (off, derived) and `cut.nena_nm` (blank:
  measured per acquisition).

## 7. Limitations

- The pick margins of the simulation are ad hoc; a real hand-drawn pick may run along the ring for longer stretches,
  which raises the share of cut clusters but not, by construction, the rule's per-cluster behaviour.
- The rule looks at x and y only; the analysis' axial slab is not the pick's concern (the pick has no z).
- The NeNA stored for a folder assumes one folder holds one acquisition; the source is written to every row.
- Clusters near the edge whose localizations all made it into the file are flagged (5-11 % of intact clusters in
  the simulation); they are flagged, not removed, unless the switch is ticked.
- The batch window does not read pick files: its rows say "boundary unknown: not evaluated".
- Only clusters the curation kept are judged; a cluster the edge criterion or DBCV removed is not counted in K.

## 8. Decision record

- **2026-10-14**: decided with the user: a cluster is cut when any localization lies within 1 NeNA of the pick
  boundary; the cut clusters are flagged and counted, excluded only on request; without a boundary or a NeNA the
  rule is not applied and the tables say so; percentage rules are rejected as arbitrary.

## References

- Endesfelder U, Malkusch S, Fricke F, Heilemann M (2014). A simple method to estimate the average localization
  precision of a single-molecule localization microscopy experiment. *Histochemistry and Cell Biology*
  141(6):629-638. doi:10.1007/s00418-014-1192-3
- Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R (2017). Super-resolution microscopy with
  DNA-PAINT. *Nature Protocols* 12(6):1198-1228. doi:10.1038/nprot.2017.024 (Picasso; the pick format was read in
  Picasso v0.9.10's `picasso/gui/render.py`, `picasso/postprocess.py` and `picasso/lib.py`)
- Ester M, Kriegel H-P, Sander J, Xu X (1996). A density-based algorithm for discovering clusters in large spatial
  databases with noise. *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining
  (KDD-96)*, 226-231 (no DOI).
- Gazal et al. (2026), preprint v1: the DBSCAN parameters (eps 25 nm, min samples 10) used here.
- Okabe M, Ito K (2008). *Color Universal Design (CUD): how to make figures and presentations that are friendly to
  colorblind people* (web page, no DOI): the palette.
