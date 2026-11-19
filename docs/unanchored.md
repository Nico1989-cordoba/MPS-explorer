# Unanchored clusters: spectrin clusters that lie under the membrane, found without tubulin

> **Resumen (español).** Algunos clusters de βII-espectrina no están en el anillo de la membrana sino por dentro del
> axón (espectrina en tránsito, de una célula vecina, o artefactos). Si se dejan, inflan el perímetro que recorre los
> centros y la ocupancia. Esta página explica cómo el programa los detecta **sin βIII-tubulina**: mide cuán hondo
> está el centro de cada cluster por debajo del anillo que dibuja la propia espectrina (contorno polar de las
> localizaciones, o curva suave de los centros) y le pone uno de tres estados: **anclado**, **dudoso** (250–400 nm,
> solo se marca) o **no anclado** (más de 400 nm, se excluye del análisis «con descarte»). El análisis se informa
> **siempre con y sin descarte**. Las decisiones a mano se guardan por posición en `<datos>_unanchored.json`
> (tolerancia 50 nm). **Ningún umbral sale de la literatura**: el 400 nm es **provisorio** (datos piloto de un solo
> animal + criterio ad hoc) hasta etiquetar a mano axones; el 250 nm es el margen que ya usaba el método viejo
> (tubulina + widefield), que sigue disponible como opción. Excluir espectrina interior es una decisión de análisis,
> no un hecho biológico. Los números rotulados [pilot] son datos piloto de un solo animal (18 axones de un
> experimento): sirven para elegir un criterio, no para validarlo.

This page is the public methods record of the **automatic discard of unanchored clusters** (`tools/mps_unanchored.py`,
`tools/unanchored_export.py`, the window `tools/mps_unanchored_window.py`). It is the counterpart, for data without a
tubulin channel, of the widefield method of [the axoplasm panel](clustering_integration.md). Every number is labelled:

- **[lit]**: literature (reference at the end).
- **[own sim]**: a simulation of this program's own simulator, synthetic data only.
- **[pilot]**: exploratory pilot, **one experiment, one animal, 18 axons**; read to choose a criterion, never a
  validation.
- **[code]**: inherited from another module of this program.
- **[ad hoc]**: a convention of this program without further support yet.

**None of the parameters below comes from the literature.** In the papers read for this page nobody separates
peripheral spectrin from interior spectrin automatically: they select by hand, use a co-label, optics or genetics
(Han 2017, D'Este 2016, Huang 2017, Zhong 2014; see References). An automatic criterion is therefore a new
contribution and has to be validated as such.

## 1. What question it answers, in plain words

**What question does it answer?** Which of the clusters of an axon sit under the membrane ring instead of on it, so that
the analysis can be shown with and without them.

**Plain explanation.** Seen in cross-section, the spectrin of the membrane skeleton makes a ring. A cluster whose
centre is far *inside* that ring is probably not part of it. For each cluster the program measures how deep its centre
is below the ring that the localizations themselves draw, and labels it. Nothing about the cluster's size or shape is
used: in the pilot those did not separate interior from membrane clusters.

**Example.** A ring of 1.5 µm radius with lumps planted at 150, 300, 500 and 1000 nm inside: the program labels them
anchored, doubtful, unanchored and unanchored; every membrane cluster stays anchored, with either reference ring
[own sim].

### Why not the old method alone

The widefield method (tubulin mask AND spectrin widefield, 250 nm margin) only discards a cluster when two images put it
deeper than the margin. It is prudent: when it flags, it is almost surely right; when it does not flag, that proves
nothing. On the pilot it flagged 9 of 1102 clusters, in 4 of 18 axons, while a clearly visible chain of interior
clusters of another axon went unflagged because the dark interior of the widefield image did not close [pilot]. It is
therefore a floor, not a truth. It stays available (`widefield` method) and its result is exported with its origin.

## 2. The rule

1. **Reference ring.** Two are available and both are always computed:
   - `polar_localizations` (default): the polar contour of the slab's localizations (`mps_auto_contour.polar_contour`,
     72 sectors, 40 nm radial bins, radial mode per sector). It uses thousands of localizations rather than tens of
     centres; its noise is 21 nm median (90th percentile 54 nm) when 70 % of the localizations are subsampled [pilot].
   - `smooth_centres`: a closed curve r(θ) with harmonics up to order 4 fitted to the cluster centres with a robust
     Tukey biweight (c = 4.685, Beaton & Tukey 1974) [ad hoc]. Needs at least 12 centres.
   Why polar by default: on simulated bays (below) the smooth ring excludes 2–8 % of real membrane in bays of 500–800
   nm, the polar ring 0–4 % [own sim]; in the pilot the smooth ring was also less stable (removing 10 % of the centres
   changed 18 of 108 verdicts, against 5 of 51 for the polar ring at 400 nm).
2. **Depth.** Signed perpendicular distance from the cluster centre to the ring, positive inside (the same definition
   for both rings, so that 400 nm means the same with either).
3. **State.** Depth > `exclude_nm` → `unanchored` (excluded in the "with discard" column); `doubt_nm` < depth ≤
   `exclude_nm` → `doubtful` (only marked, it stays); otherwise `anchored`. If the ring is not usable →
   `not evaluated`, nothing is excluded and the row says why.
4. **Noise of the reference.** 20 subsamples (70 % of the localizations, 90 % of the centres, fixed seed); a cluster
   closer than 2 of its own depth SDs to the exclusion threshold is marked `borderline`.
5. **Gate.** A warning (not a rejection) when more than 20 % of the clusters would be excluded: the problem is then the
   axon or the reference, not the clusters (the idea of rejecting a whole structure when too much falls inside is
   Thevathasan 2019).
6. **Star-shape test of the cloud.** The polar ring is only meaningful for a cloud that is star-shaped around its
   centre. From the centre, 72 rays are cast; on each, density peaks deeper than the exclusion threshold are looked for
   and chained from ray to ray (radii within 150 nm, skipping up to 400 nm of arc between clusters, because a ray only
   sees a band near its clusters). If the chain is longer than 1500 nm of arc it is a folded membrane (a "C", two
   lobes), not lumps: the ring is rejected and nothing is excluded. Simulated: a thick "C" gives a 3992 nm band
   (rejected); two 900 nm discs 200 nm apart, 1837 nm (rejected, the closest to the limit); a diametral row of 13 lumps,
   0 nm (accepted, all 13 excluded) [own sim]. With a first version that did not skip gaps the "C" gave only 469 nm
   and was not rejected: that is why the skip exists.
7. **Failure states, explicit.** Fewer than 200 localizations, no polar contour, fewer than 12 centres (smooth ring),
   invalid smooth fit, cloud not star-shaped: nothing is excluded, the status and the reason are exported.

### Parameters and where each number comes from

| parameter | value | origin | why this number |
|---|---|---|---|
| `exclude_nm` | **400 nm, PROVISIONAL** | [pilot + ad hoc] | Above it, in the field where tubulin is a usable cross-check, tubulin puts 71 % of the flagged clusters inside (20 of 28) against a 24 % background for membrane clusters; only 5 of 51 verdicts change with a 70 % subsample; the 8 deepest widefield positives are at ≥ 737 nm. One experiment, one animal. It stays provisional until axons are labelled by hand (decision D6) |
| `doubt_nm` | 250 nm | [code + ad hoc] | The margin the widefield method already used (`DEFAULT_MARGIN_NM`). Between 250 and 400 nm tubulin does not decide (36 % inside, 32 % at the edge) and most of the instability sits there (24 verdicts change at 250 nm, 5 at 400) |
| reference ring | polar, 72 sectors, 40 nm bins | [own sim] (contour package) | Uses localizations, not centres; least noisy; independent of whether the final contour comes out well |
| noise replicates | 20 at 70 % (polar) / 90 % (smooth) | [ad hoc] | Enough to see whether a verdict is stable; not tuned |
| borderline | 2 depth SDs from the threshold | [ad hoc] | Convention, as 2–3 MAD in outlier rules (Leys 2013); no derivation |
| gate | warning above 20 % excluded | [ad hoc, pilot] | The maximum at 400 nm was 19 % of the clusters of an axon |
| smooth ring | harmonics ≤ 4, Tukey c 4.685 | [ad hoc] / [lit] for c | Enough for ovals and one lobe; 4.685 is the usual 95 %-efficiency constant |
| star-shape test | 72 rays, 400 nm gap, 150 nm link, 1500 nm arc | [own sim + ad hoc] | Chosen so a "C" and two lobes are rejected and lumps are not; not tested on real axons |
| manual tolerance | 50 nm | [ad hoc] | A decision matches the nearest centre within it (see section 4) |

### What the pilot says about the 400 nm (aggregate, [pilot], exploratory)

- The 9 widefield positives are separated from the other 1093 clusters by any depth measure (AUC 0.96–0.99, interval
  over whole axons 0.82–1.00); they lie at a median 934 nm below the ring (the others: 3 nm). Cluster size, shape,
  number of localizations and z do **not** separate them (AUC 0.3–0.5): position is what counts.
- With 9 positives (8 in the same field, 5 in one axon, all among the deepest) **no threshold can be learned**: leaving
  one axon out moves the "optimal" threshold between 257 and 1063 nm. Hence the threshold is a decision, not a fit.
- At 400 nm the rule flags 51 of 1102 clusters (95 at 250 nm); 6 of 18 axons are left untouched at 400 nm (1 at 250 nm).
- A rule "radius below median minus k MAD" finds 6 of 9 and flags nothing in the axon where it is most needed (axons are
  not round and interior clusters inflate the MAD); nearest-neighbour distance and an RANSAC ellipse were discarded
  (they flag many clusters that tubulin puts at the edge).
- There is **no independent ground truth** for the clusters the rule flags and the widefield method does not. The
  corroboration is the tubulin of one field plus visual inspection.

### What the simulations say about excluding real membrane (own simulator, no interior clusters at all)

Clusters every 250 nm of arc, 80 localizations per cluster plus 1000 diffuse; 40 seeds per wide ring, 20 per bay.
Anything flagged is membrane wrongly flagged.

| scenario | polar ring: excluded at 400 nm | smooth ring: excluded |
|---|---|---|
| wide ring, radial SD 30–100 nm | 0 % | 0–0.06 % |
| wide ring, radial SD 150 nm | 0.27–0.32 % | 0.5 % |
| bay 300 nm deep, 20–60° wide | 0 % | 0 % |
| bay 500 nm deep, 20° | 0.5 % | 2.0 % |
| bay 800 nm deep, 20° (narrow and deep) | **4.4 %** (18 of 20 axons) | 7.9 % |
| bay 800 nm deep, 40–60° | 0–0.1 % | 6–8 % |

For comparison, the radial robust SD of the real localizations in one slice was 95 nm (median) and 131 nm (worst axon)
in the pilot. The weak point is a **narrow, deep bay**: the polar ring smooths by sectors and cuts it, and the bay's
floor is excluded. Such axons have to be restored by hand.

## 3. What it measures, and what it does NOT

It measures **how deep below the spectrin's own ring a cluster centre lies**, as a proxy for "not part of the
membrane-associated skeleton". It does **not**:

- prove the cluster is "not anchored" in any biological sense: nobody has measured how much βII-spectrin is inside an
  adult peripheral axon (axonal transport, Lorenzo 2019 and Glomb 2023, only says there can be some);
- separate **Schwann cell** spectrin hugging the axon (periaxonal collar, Susuki 2018; Trapp 1989) from the axolema:
  at ~10 nm lateral precision (SD; ~20 nm FWHM) with spectrin alone they are indistinguishable. This is a limit to declare;
- find interior clusters in a folded or non-star-shaped cloud (it refuses instead);
- judge a cluster in a real concavity of the axon: a cluster inside a bay at 250–400 nm can be real membrane, which is
  why that band is only "doubtful";
- replace looking at the figures: the preview is there to be looked at.

Excluding interior spectrin is an analysis decision, not a biological fact. That is why the analysis is **always shown
with and without the discard** (two columns, the measured one first) and why the origin of every exclusion is exported.

How sensitive the perimeter is: removing two interior clusters lowered one pilot axon's centre-to-centre perimeter by
23 %, because the path no longer zigzags to reach them, not because membrane was cut [pilot]. At 400 nm the median
change of the perimeter was −7.4 % (range −26 to 0 %); of the occupancy, +1.4 points.

## 4. Decisions by hand

Clicking a cluster in the window toggles it excluded / restored, on any cluster (also to restore one the rule
excluded). Excluding asks for a reason (interior, neighbouring axon, glia, artefact, other: free text). The decisions
are saved **by position, not by cluster number**, in `<data file>_unanchored.json` beside the data: each decision is the
cluster centre (x, y in nm, to 0.1 nm), the action, the reason (compulsory to exclude), the date and a note. A decision
is applied to the nearest centre within **50 nm** [ad hoc]; the last decision on a cluster wins; decisions that find
no cluster stay in the file and are counted (`unanchored_manual_n_orphan`). If the eps, MinPts or the slab change, the
cluster numbers change but the decision still lands on the same place (tested by permuting and moving the centres 8 nm).
The decisions enter the `analysis_id` fingerprint. A file in that path that is not a decisions file is never
overwritten.

## 5. Trimming before the contour fit (D8)

An optional step of the perimeter density analysis ([perimeter_density.md](perimeter_density.md), section 4): the
localizations deeper than 250 nm below the polar ring are left out of the contour fit **only**. It is **off by
default** and, since 2026-11-17, **strict**: it is applied only when the P-spline fitted on the polar contour passes
every rule of the automatic contour (no self-crossing, length at most 1.5 × its convex hull, no inward notch tighter
than 150 nm, the cloud never more than 75 nm to one side, a lumen). Otherwise nothing is trimmed and the status says so.
Why: with the weak rule, a "C" lost 23 % of the true membrane in the test; with the strict rule it loses none [own sim].
On the pilot the strict rule acts on 4 of 18 axons, all already reliable, so no reliability changes.

## 6. Windows and exports

The window is described in [perimeter_density.md](perimeter_density.md), section 11 ("The Perimeter... and
Unanchored... windows"). In short: the button **Unanchored...** of the MPS analysis window opens a map with the
reference ring (dashed), excluded centres (red cross), doubtful ones (orange ring) and restored ones (open square);
nothing changes in the numbers until **Apply**. In the same results window the analysis appears with and without the
discard. The export (axon table, per-cluster table and Batch) carries the v18 block (35 columns at the end of the
axon row, `unanchored_*` and `pd_contour_trim_*`) and five per-cluster columns (`depth_below_ring_nm`, `anchor_state`,
`anchor_borderline`, `anchor_source`, `anchor_reason`); each has its entry and origin in
`tools/column_dictionary.py`. The environment variable `MPS_UNANCHORED_AUTO=0` switches the automatic run off (for
scripted references).

## 7. Limits

- One experiment, one animal: every [pilot] number is exploratory.
- The 400 nm is **provisional**; the band 250–400 nm cannot be resolved with the pilot's data.
- The 400 nm was simulated only for membrane loss (section 2), not for a real axon with bays.
- The polar ring is valid only for star-shaped clouds; the star-shape test was calibrated in simulation only.
- Cause of a ring that changes radius or shifts between the three z slices (tilt, depth-dependent lateral distortion of
  the optics, Carlini 2015, or different clusters at different heights): not established.
- Where the ring is wide in itself (radial robust SD above 100 nm in a single slice) neither this rule nor removing the
  interior fixes the contour; that is a property of the cloud.

## References

- Gazal N.G., Escalante G., et al. (2026), bioRxiv doi:10.64898/2026.02.19.705149: the reference pipeline. It removes
  low-quality clusters by hand without a numerical criterion [read in full, web version].
- Han et al. (2017). *PNAS*. doi:10.1073/pnas.1705043114: clusters of at most 10
  localizations dropped; a 10–20 % background treated as noise [read in full].
- Zhong et al. (2014). *eLife*. doi:10.7554/eLife.04581: a non-incorporated, mobile βII-spectrin pool [read
  in full].
- D'Este et al. (2016). *Sci Rep*. doi:10.1038/srep22741: βII-spectrin along the adult sciatic axon and in
  the outer myelin layer; manual line profiles [read in full].
- Susuki et al. (2018). *J Neurosci*. doi:10.1523/JNEUROSCI.3647-17.2018: glial βII-spectrin in Cajal bands and
  paranodes; the axonal one along the axolema [read in full].
- Thevathasan et al. (2019). *Nat Methods*. doi:10.1038/s41592-019-0574-9: reject a whole structure when
  too large a fraction of localizations falls inside a radius [partly read].
- Wu et al. (2023). *Nat Methods* doi:10.1038/s41592-022-01676-z (LocMoFit): maximum-likelihood fit of a
  geometric model with a background component [partly read].
- Carlini et al. (2015). *PLoS ONE*. doi:10.1371/journal.pone.0142949: lateral shift of out-of-focus
  molecules in 3D astigmatic SMLM, ~20 nm per 125 nm of z [read in full].
- Lorenzo et al. (2019). *PNAS* doi:10.1073/pnas.1820649116 and Glomb et al. (2023). *Dev Cell*
  doi:10.1016/j.devcel.2023.08.031: spectrin in axonal transport [abstract only].
- Trapp et al. (1989). *J Neurocytol* 18:47-60: spectrin in the periaxonal membrane and incisures of Schwann cells
  [abstract only; the antibody does not tell the isoform].
- Leys et al. (2013). *J Exp Soc Psychol*. doi:10.1016/j.jesp.2013.03.013: median ± 2.5 MAD for outliers;
  state and justify the threshold [abstract and secondary sources].
- Beaton AE, Tukey JW (1974). *Technometrics*: the biweight [cited from memory, not re-read].
- Rousseeuw PJ, Croux C (1993). *JASA*. doi:10.1080/01621459.1993.10476408 and Fischler MA, Bolles RC (1981).
  doi:10.1145/358669.358692 (RANSAC): [cited from memory, not re-read].
