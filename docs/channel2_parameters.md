# DBSCAN parameters of a second channel

> **Resumen (español).** El canal 2 (aducina, 4.1B u otra proteína) nunca se agrupa con el eps y el MinPoints del
> canal de espectrina: los parámetros de DBSCAN pertenecen a un conjunto de datos. El programa los **calcula y los
> precarga** (editables) desde la precisión de cada canal, medida por NeNA sobre su archivo de **campo completo y sin
> enlazar**, y desde el colorante y el buffer del canal 2: eps₂ = eps₁ · NeNA₂ / NeNA₁ y
> MinPts₂ = redondeo(1 + (n_loc,mol − 1) · (1 − exp(−eps₂² / 4 NeNA₂²))). Con eps 25 nm, NeNA 12 nm y Alexa 647 en
> MEA (14 ciclos) da 9,6 → 10, el MinPoints publicado. Si la NeNA no se puede medir, los campos quedan vacíos con el
> motivo; nunca se copian los del canal 1. Un valor editado queda registrado como "user". El cociente de densidades
> entre canales es solo un aviso. Decidido por el grupo de investigación del usuario el 2026-10-14.

This page explains why the two-colour comparison of MPS Explorer pre-fills channel 2's DBSCAN radius (*eps*) and
minimum count (*Min Pts*, `min_samples`) with the values it does, and not others. The code is in
`tools/mps_channel_params.py` (the formulas, the dye table, the NeNA measurement) and `tools/channel2_ui.py` (the
box that shows them).

## 1. What DBSCAN counts

DBSCAN (Ester et al. 1996) has two parameters, a radius *eps* and a count *MinPts*. For every localization it counts
the localizations within *eps* of it, **itself included**:

- if that count is at least *MinPts*, the localization is a **core point**;
- if not, but it lies within *eps* of a core point, it is a **border point** and joins that core point's cluster;
- otherwise it is **noise**.

A cluster is a set of core points chained within *eps* of each other, with their border points. *MinPts* therefore
counts **localizations**, not molecules.

## 2. Why each channel needs its own parameters

Ester et al. (1996, section 4.2) choose *Eps* and *MinPts* for the "thinnest" cluster of **one** database: they are
density parameters of a dataset. Two channels of a two-colour experiment are not sampled alike:

- each localization of a molecule is displaced by that channel's **localization precision**, which depends on the
  dye's photon yield, the background and the fit;
- each labelled molecule leaves a number of localizations that depends on the **dye and the imaging buffer** (its
  switching cycles; Dempsey et al. 2011).

So the same two numbers do not implement the same criterion in both channels, and cluster counts and inter-channel
distances change with the parameters (Khater et al. 2020). What is kept equal between the channels is the
**criterion** of the reference pipeline (Gazal et al. 2026): *one labelled molecule, with its expected number of
localizations, forms a cluster*. The numbers that implement it are computed from each channel's own measurements.

## 3. Precision = NeNA on the whole-field, unlinked file

The precision used is **NeNA** (nearest-neighbour based analysis; Endesfelder et al. 2014): the same molecule
detected in two consecutive frames appears twice, separated only by measurement error, and the distribution of
next-frame neighbour distances has a peak whose width is the experimental localization precision. It needs no photon
calibration. MPS Explorer uses Picasso's model (`tools/mps_quality.py`, `nena`), the same one the Data quality
window shows, and does not report a value from fewer than 500 next-frame pairs.

It is measured on each channel's **whole-field** localization file, **unlinked**:

- a picked axon rarely holds the 500 next-frame pairs the fit needs; the whole field of the same movie has the same
  precision and many more pairs;
- a **linked** file (Picasso *Link*) has merged exactly the consecutive-frame pairs NeNA measures, so it cannot be
  used. A file is treated as linked when it carries Link's `len` column or its processing record names Link.

The whole-field file is found **by name** from the loaded file with the same matcher the Axoplasm panel uses
(`tools/mps_file_match.py`); when the loaded file is itself not a picked region, it is its own whole field; and the
user can always choose the file. If the precision cannot be measured (no whole-field file, too few pairs, a linked
file, a fit that does not converge) the fields stay **empty, with the reason shown**: no value is invented and
channel 1's values are never copied.

## 4. The two formulas, step by step

**Distance between two localizations of one emitter.** Each localization of an emitter at μ is μ + ε, with
ε ~ N(0, σ²) independently on each axis. The difference of two of them is N(0, 2σ²) per axis, so their distance *d*
follows a **Rayleigh law of scale √2·σ**:

  p(d) = d / (2σ²) · exp(−d² / 4σ²),  P(d ≤ eps) = 1 − exp(−eps² / 4σ²).

This is the single-molecule term NeNA fits, so σ here is exactly the NeNA value.

**eps.** Keeping *eps* the same multiple of each channel's precision keeps that pair probability the same in both
channels:

  **eps₂ = eps₁ × NeNA₂ / NeNA₁**,

with eps₁ the radius of channel 1 (25 nm in the reference pipeline) and NeNA₁ measured the same way on channel 1's
whole-field file. When channel 1's NeNA cannot be measured, the published 12 nm (Gazal et al. 2026, NeNA) is used for
the ratio, and the box and the export say so.

**Min Pts.** For one localization of a molecule that leaves n_loc,mol localizations, DBSCAN's core count is the point
itself plus each of the other n_loc,mol − 1 that falls within *eps*, each with the probability above. Its expectation
is

  **Min Pts₂ = round(1 + (n_loc,mol − 1) × (1 − exp(−eps₂² / (4 NeNA₂²))))**

(rounded half up, never below 1). n_loc,mol is taken from the dye table below for the dye and buffer chosen.

## 5. The worked check: the reference pipeline's 10

With the reference pipeline's eps = 25 nm, NeNA = 12 nm and Alexa Fluor 647 in MEA (14 switching cycles):

- eps² / 4σ² = 625 / 576 = 1.085;
- 1 − exp(−1.085) = 0.662;
- 1 + 13 × 0.662 = **9.6 → 10**, the MinPoints = 10 that Gazal et al. (2026) set "following the mean number of
  switching cycles".

Alexa Fluor 568 in MEA (7 cycles) gives 1 + 6 × 0.662 = 5.0 → **5**. A channel 2 with NeNA 15 nm gets
eps₂ = 25 × 15 / 12 = 31 nm; eps/σ is unchanged, so the pair probability is again 0.662 and Alexa Fluor 568 in MEA
again gives 5. In a simulation of molecules of 14 localizations with σ = 12 nm (this project's own), the mean count
within 25 nm is 9.63 against 9.61 from the formula, and about 60 % of a molecule's localizations are core points at
Min Pts = 10: one labelled molecule forms a cluster. This check is a unit test (`test_channel2_params.py`).

## 6. The dye table

Mean number of switching cycles per molecule, from **Dempsey et al. (2011), Nature Methods 8:1027, Table 1**
("# Switching Cycles (Mean)"), in the two buffers of that table: MEA (cysteamine) and βME (β-mercaptoethanol). The
user picks the dye and the buffer of channel 2.

| Dye | MEA | βME | | Dye | MEA | βME |
|---|---|---|---|---|---|---|
| Atto 488 | 11 | 49 | | Alexa Fluor 647 | 14 | 26 |
| Alexa Fluor 488 | 16 | 139 | | Cy5 | 10 | 17 |
| Atto 520 | 9 | 17 | | Atto 647 | 10 | 24 |
| Fluorescein | 4 | 15 | | Atto 647N | 9 | 39 |
| FITC | 17 | 16 | | Dyomics 654 | 20 | 19 |
| Cy2 | 0.4 | 0.7 | | Atto 655 | 17 | 22 |
| Cy3B | 8 | 5 | | Atto 680 | 8 | 27 |
| Alexa Fluor 568 | 7 | 52 | | Cy5.5 | 16 | 25 |
| TAMRA | 10 | 59 | | DyLight 750 | 5 | 6 |
| Cy3 | 0.5 | 1.6 | | Cy7 | 5 | 2.6 |
| Cy3.5 | 5.7 | 3.3 | | Alexa Fluor 750 | 1.5 | 6 |
| Atto 565 | 4 | 5 | | Atto 740 | 3 | 14 |
| | | | | Alexa Fluor 790 | 5 | 2.7 |
| | | | | IRDye 800 CW | 3 | 127 |

A dye with fewer than two cycles in the chosen buffer leaves Min Pts empty: one localization per molecule cannot form
a density cluster.

## 7. Why not the density ratio between the channels

The box and the two-channel export show how densely each channel is sampled: the median number of localizations
within the channel's **own eps** of a localization, the count DBSCAN's core condition tests. A ratio of 3 or more
between the channels, either way, is **warned about**: with the same parameters, the detected boundary of a Gaussian
cluster sits at r = σ √(2 ln(λ₀/λ_min)), and a three-fold denser channel moves it from 2.15 σ to 2.61 σ (about a fifth
farther out) at λ₀/λ_min = 10. Three is an editable convention with that meaning.

The ratio is **never used to scale Min Pts**. It mixes the labelling density with the blinking multiplicity
(Dempsey et al. 2011; Nieuwenhuizen et al. 2015) and, above all, it contains **how abundant the partner protein is**,
which is what the two-colour comparison measures. Scaling Min Pts by it would make the cluster criterion depend on the
answer.

## 8. Pre-filled, editable, and recorded

The two values are **pre-filled and editable**. Under each field the box writes its origin, for example
"derived from NeNA (14.2 nm, whole field channel2_locs.hdf5) and Alexa Fluor 568/MEA"; a value typed or edited by
hand becomes **"user"** (the derived value stays shown beside it, and *Use derived values* puts it back). The choice is
remembered per acquisition folder with its provenance, and the two-colour export writes, per row:
`nena_a_nm`, `nena_a_source_file`, `nena_a_origin`, `eps_a_reference_nm`, `nena_b_nm`, `nena_b_source_file`,
`dye_b`, `buffer_b`, `n_loc_mol_b`, `n_loc_mol_b_origin`, `eps_b_origin`, `min_samples_b_origin` (file names only, never
a folder), next to `eps_b_nm`, `min_samples_b`, `locs_within_eps_a` and `locs_within_eps_b`. The column dictionary
written beside every export describes each of them.

## 9. Limitations

- **The dye cycles are a lower bound** of the localizations per labelled molecule: a secondary antibody can carry
  several dyes and a switching event can span several frames. Until the label is calibrated (emission events per
  label, c_label, measured on isolated antibodies; a later version), the table value is used and labelled
  "lower bound". A larger n_loc,mol gives a larger Min Pts.
- The formula counts only the molecule's own localizations; background and neighbouring molecules add to the count,
  so the expected count of a real localization is at least this.
- The cycles were measured by Dempsey et al. in their conditions (laser power, buffer concentration); another set-up
  can differ.
- NeNA's two-component fit can settle on a narrower peak: on this project's own synthetic whole-field files of known
  precision it came out 5-20 % low in 5 of 20 files (a direct Rayleigh estimate of the same distances was within
  3 % in all 20). The box shows the NeNA value and its file; the fitted curve can be checked in the Data quality
  window.
- The parameters can also be calibrated on simulations statistically similar to the data (Nieves et al. 2023); that
  scan is not part of this box.

## 10. Decision record

- **2026-09-21**: channel 2 no longer inherits channel 1's eps and Min Pts; the box asked for them with the fields
  empty ("never pre-filled").
- **2026-10-14**: decided by the user's research group, replacing the 2026-09-21 never-prefill rule: eps and Min Pts
  of each channel are computed from that channel's NeNA (whole-field, unlinked file) and, for Min Pts, from its dye
  and buffer (Dempsey et al. 2011), **pre-filled and editable**, each with its origin; channel 1's values are never
  copied; when the precision cannot be measured the fields stay empty with the reason; the density ratio between the
  channels stays a warning only.

## References

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
- Khater IM, Nabi IR, Hamarneh G (2020). A review of super-resolution single-molecule localization microscopy cluster
  analysis and quantification methods. *Patterns* 1(3):100038. doi:10.1016/j.patter.2020.100038
- Nieuwenhuizen RPJ, Bates M, Szymborska A, Lidke KA, Rieger B, Stallinga S (2015). Quantitative localization
  microscopy: effects of photophysics and labeling stoichiometry. *PLOS ONE* 10:e0127989.
  doi:10.1371/journal.pone.0127989
- Nieves DJ, Pike JA, Levet F, Williamson DJ, Baragilly M, Oloketuyi S, de Marco A, Griffié J, Sage D, Cohen EAK,
  Sibarita J-B, Heilemann M, Owen DM (2023). A framework for evaluating the performance of SMLM cluster analysis
  algorithms. *Nature Methods* 20(2):259-267. doi:10.1038/s41592-022-01750-6
