# ξ: the number of localizations per emitter, and where its values come from

> **Resumen (español).** La capa «Emitters (BaGoL-like reimplementation)» agrupa las localizaciones repetidas de una
> misma marca fluorescente en un *emisor* y necesita saber cuántas localizaciones deja, en promedio, una marca: esa
> distribución es **ξ**. Un emisor es una **pila de marcas fluorescentes** (anticuerpos secundarios con sus
> colorantes), **no una proteína**. Esta página dice qué es ξ, de dónde sale cada número (Dempsey et al. 2011, Tabla 1,
> p. 1030: 14 ciclos para Alexa Fluor 647 en MEA y 26 en βME; 7 y 52 para Alexa Fluor 568), por qué esos números son
> un **rango de sensibilidad** y no una constante (fueron medidos en vidrio, a menor intensidad, y el propio paper los
> llama «cota inferior»; el buffer solo los mueve ×2 a ×7), con qué forma se usan (gamma-exponencial para dSTORM,
> Poisson para DNA-PAINT), cuándo hace falta una calibración propia (criterio ad hoc, a confirmar en la fase F4) y cómo
> se miden en un archivo propio las localizaciones por ciclo y la fracción que sobrevive a los filtros. Cada número
> lleva su rótulo de origen: literatura / derivación / ad hoc.

This page documents the prior ξ of the Emitters layer of MPS Explorer: what it is, the values it takes from the
literature with citation and page for each number, why those values are a sensitivity bracket rather than a
constant, the shape of the distribution, the criterion for when a dedicated calibration becomes necessary, and how
the two factors that are not in the literature are measured on the user's own file. The code is
`tools/emitters_prior.py`; the tests are `test_emitters_prior.py`. Every number is labelled with where it comes from:

- **literature**: a published value, with its reference and page;
- **derivation**: shown here;
- **ad hoc**: a choice of this project, with no measurement behind it yet, and what would replace it.

Method name, everywhere it appears: **Emitters (BaGoL-like reimplementation; Fazel et al. 2022)**. It is a
reimplementation of the published model in Python, not the authors' MATLAB code (MIT licence, LidkeLab/BaGoL), and
it is never called "BaGoL" alone.

## 1. What an emitter is, and what ξ is

Fazel et al. (2022) write: "we refer to a 'localization' as the estimate of the position of the fluorophore from a
single blinking/binding event and an 'emitter' as the fluorescently tagged molecule", and the model assumes that
"the emitter is labeled by a single dye or docking strand" (literature: Fazel et al. 2022, Introduction and
Discussion). In an immunostained dSTORM sample the fluorescently tagged thing is a stack of antibodies (primary,
one or more secondaries, each secondary carrying its dyes), and a multiply labelled IgG behaves as one emitter in
photoswitching buffer (literature: Helmerich et al. 2020, on Alexa Fluor 647 IgGs with 1.1-8.3 dyes each, in 100 mM
MEA). Therefore:

**Emitters are fluorescent label stacks, not proteins.** The Emitters layer returns one position per label stack
that the data can resolve. It does not count proteins: how many stacks a protein carries (epitopes labelled, secondaries
per primary, dyes per secondary) is not in the model and not in any number on this page.

ξ is "the distribution of the number of localizations per emitter" (literature: Fazel et al. 2022, Results). The
model takes it as a Poisson(λ) or a Gamma(k, θ) distribution (literature: Fazel et al. 2022, Supplementary Note 1;
code `BaGoL_RJMCMC.m`), and it can either be given ("If there is a priori knowledge of the probability distribution
for the number of blinking/binding events of a single label, ξ, such as from a dedicated control experiment where the
sample is known to be at low labeling fraction, this can be given to the algorithm") or learned from the data
("Otherwise, this distribution can be directly learned from the data"; literature: Fazel et al. 2022, Results). For
their own dSTORM data the authors did the latter on sparse regions and then fixed the result: "we first ran BaGoL to
learn ξ ... using less dense regions of the same data sets. The returned ξ values were then used in second runs"
(literature: Fazel et al. 2022, Methods).

## 2. Three priors side by side

MPS Explorer runs the Emitters layer with up to three ξ, each result carrying the label of the prior that produced it
(column `emitters_xi_source`, decision of 2026-10-16):

| Label | What it is | Role |
|---|---|---|
| `literature_MEA` | Gamma(1, mean) with the mean from Dempsey et al. (2011) in GLOX + 10 mM MEA (section 3) | sensitivity bracket, lower |
| `literature_bME` | same, in GLOX + 140 mM βME | sensitivity bracket, upper |
| `learned` | the hierarchical scheme of Fazel et al. (2022) on the isolated clusters of the same file, then fixed | main prior for positions |
| `calibrated` | measured on a dedicated control (isolated labelled antibodies, same buffer and session) | required for any absolute count (section 6) |

This page fixes the first two and the shape of all of them; the learned prior is implemented in a later phase and
this module only receives its values.

## 3. The literature values

Dempsey et al. (2011), *Nature Methods* 8:1027, Table 1 (p. 1030), "Summary of switching properties of the 26 dyes
tested in this study". The footnote of the table: "Dye-switching properties are reported in the presence of GLOX and
10 mM MEA as well as GLOX and 140 mM βME." Rows transcribed for the two dyes of this project (literature, every cell):

| Dye | Buffer | Detected photons per switching event | Equilibrium on-off duty cycle (400-600 s) | Survival fraction after 400 s | **Number of switching cycles (mean)** |
|---|---|---|---|---|---|
| Alexa Fluor 647 | GLOX + 10 mM MEA | 3,823 | 0.0005 | 0.83 | **14** |
| Alexa Fluor 647 | GLOX + 140 mM βME | 5,202 | 0.0012 | 0.73 | **26** |
| Alexa Fluor 568 | GLOX + 10 mM MEA | 2,826 | 0.00058 | 0.58 | **7** |
| Alexa Fluor 568 | GLOX + 140 mM βME | 1,686 | 0.0027 | 0.99 | **52** |

Two sentences of the same paper qualify the last column:

- it is a **lower bound**: "we determined for each molecule the number of switching cycles during the data
  acquisition time, which represents a lower bound value, given that some molecules did not photobleach before the
  end of the observation period" (literature: Dempsey et al. 2011, p. 1030);
- it was measured on **isolated antibodies on glass**, not in a stained sample: "dye-labeled donkey anti-rat secondary
  antibodies" with "dye-labeling ratios 0.2-0.8 dyes per antibody such that the majority of dye-labeled antibody
  molecules were labeled with only one dye molecule", "adsorbed to the coverslip at a low density (typically 0.04-0.12
  dye µm⁻²)" (literature: Dempsey et al. 2011, Online Methods).

The code keeps each cell with its citation (`DEMPSEY_TABLE1` in `tools/emitters_prior.py`) and refuses any dye or
buffer that is not in this table with a message that says so; other rows of Table 1 can be added from the paper (the
Channel 2 box of the program already uses the full "switching cycles" column, see `docs/channel2_parameters.md`).

**Dyes of the reference protocol.** In the reference protocol of this project, channel 1 uses Alexa Fluor 647 on an
anti-mouse secondary, so its literature prior is the AF647–MEA row; channel 2 uses Alexa Fluor 568 on an anti-rabbit
secondary, so its literature prior is the AF568–MEA row. For now these are the only two dyes in use, which is why
only their rows are transcribed; a protocol with another dye or buffer adds its row of Table 1 or goes through
`calibrated_prior()`.

## 4. From switching cycles to localizations per emitter

What the model needs is localizations per emitter in the list the layer receives, which is not the number of
switching cycles. The three factors (derivation):

    mean(ξ) = switching_cycles × loc_per_cycle × p_filters

- **switching_cycles**: the table above (literature).
- **loc_per_cycle**: localizations left by one switching cycle. A cycle that stays on for several camera frames is
  localized once per frame. After the frame linking of the DNA-PAINT module of MPS Explorer (`build_events` in
  `tools/mps_paint.py`, the same chain as Picasso's *link*), one cycle becomes one event, so **loc_per_cycle = 1 by
  construction** (derivation). Without linking it is the mean on-time in frames: Alexa Fluor 647 on IgGs has an
  on-state lifetime of ~10 ms (literature: Helmerich et al. 2020, Table 1, 100 mM MEA, 2 kW/cm², 125 Hz: 10.2-11.1 ms
  for every DOL), so at 50 ms per frame almost every cycle falls in one frame, sometimes two when it straddles a
  frame boundary (derivation); the project's synthetic generator assumes 1-3 frames, mean 2 (ad hoc:
  `ON_FRAMES` in `tools/cluster_benchmark.py`). Measured on the user's own file in section 7.
- **p_filters**: the fraction of cycles that survive the filtering applied to the file before it reaches the layer
  (photon, precision or Picasso filter steps). Default 1 (ad hoc: nothing is known until it is measured; section 7).

Default values in the code: `loc_per_cycle = 1` (derivation) and `p_filters = 1` (ad hoc). Both are explicit
arguments and both are written into the origin text of the prior, so a result never hides which one was assumed.

For Alexa Fluor 647 at these defaults: mean(ξ) = 14 (`literature_MEA`) and 26 (`literature_bME`); for Alexa Fluor
568: 7 and 52.

## 5. The shape: gamma-exponential for dSTORM, Poisson for DNA-PAINT

In dSTORM a dye switches until it bleaches. If every cycle ends in bleaching with the same probability, the number
of cycles before bleaching is geometric, whose continuous counterpart is the exponential, i.e. Gamma with shape
k = 1 (derivation). Fazel et al. (2022) simulated exactly this (on, off and bleached states with fixed rates) and
report that "the resulting number of blinking events per fluorophore have exponential distributions with means of
~5. We used the same exponential distribution as the prior on number of blinking events per emitter by taking a gamma
prior with η = 1 and γ = 5" (literature: Fazel et al. 2022, Methods). The default shape here is therefore
**Gamma(k = 1, θ = mean(ξ))** (literature + derivation), with k editable.

What the exponential implies (derivation, from the continuous CDF the model evaluates):

| Mean(ξ) | P(≤ 1 localization) | P(≤ 3 localizations) |
|---|---|---|
| 14 (AF647, MEA) | 6.9 % | 19.3 % |
| 26 (AF647, βME) | 3.8 % | 10.9 % |
| 7 (AF568, MEA) | 13.3 % | 34.9 % |
| 52 (AF568, βME) | 1.9 % | 5.6 % |

(The discrete geometric gives 7.1 % and 20.1 % for mean 14.) An emitter that leaves one localization is
indistinguishable from background; the Emitters layer therefore never removes them and only labels them, and the
"NND filter must not be used for dSTORM data where it may be possible to have only one localization per emitter"
(literature: Fazel et al. 2022, Methods).

A **Poisson** shape describes DNA-PAINT, where binding events arrive at a constant rate for the whole movie and
nothing bleaches (literature: Fazel et al. 2022 use λ = 50 for their DNA origami); the code builds it only for that
modality (`poisson_prior`) and never from the Dempsey table.

## 6. Why the literature is a sensitivity range, not a constant

The two literature priors bracket the plausible mean; they are not the value for this laboratory's data. The
arguments, from strongest to weakest:

**(a) Conditions.** Dempsey et al. measured isolated IgGs on glass in TIRF, at 0.8 kW/cm² (647 nm) and 2.2 kW/cm²
(561 nm), 33 Hz, movies of 1,500-2,000 s, no 405 nm activation (literature: Dempsey et al. 2011, Online Methods). The
reference study of this project images 5 µm tissue sections in 3D at 10-20 kW/cm², 50 ms per frame, with increasing
405 nm light and ~16,000 frames, in GLOX + 10 mM MEA (literature: Gazal et al. 2026, Methods; the MEA concentration
is the same as Dempsey's MEA condition). Dempsey et al. found "the number of photons per switching cycle to be largely
independent of laser intensity", but the intensity sets the on/off rates and the bleaching, and the number of cycles
is a lower bound that depends on how long the movie is (literature: Dempsey et al. 2011, p. 1030 and "dependence of
switching properties on light intensity"). At 12-25 times the irradiance bleaching can come earlier (fewer cycles),
and a longer movie can observe more; the sign of the net change cannot be known without measuring it (derivation).

**(b) The buffer alone moves ξ by ×2 to ×7.** Same paper, same instrument, same dye: Alexa Fluor 647 goes from 14
(MEA) to 26 (βME) and Alexa Fluor 568 from 7 to 52 (literature: Table 1). Dempsey et al. also report that the
switching "was rather sensitive to the thiol concentration" for several dyes including Alexa Fluor 647, and recommend
"that buffer conditions be optimized for any given probe" (literature: Dempsey et al. 2011, "Different thiols"). A
number that a change of thiol multiplies by seven is not transferable as a constant to another session of a buffer
of the same family; it is transferable as a bracket.

**(c) Truncation and survival.** The survival fraction after 400 s is 0.83 / 0.73 for Alexa Fluor 647 and **0.58**
for Alexa Fluor 568 in MEA (literature: Table 1): on glass, at 2.2 kW/cm², more than half of the Alexa Fluor 568
molecules are bleached before 400 s. At a higher irradiance it will be earlier. If a channel uses Alexa Fluor 568 in
MEA, ξ can be so low that an emitter is one or two localizations and the layer adds no precision, which the authors
state for dSTORM in general: "it does not exhibit much improvement in precisions due to small number of
localizations per emitter" (literature: Fazel et al. 2022, Results).

**(d) 3D and depth.** With astigmatic 3D the detection efficiency depends on z (a wider PSF gives fewer photons per
localization, and fewer pass a photon filter); Dempsey et al. is 2D TIRF. The effective ξ in an axial slab is cycles
× the fraction detected at that depth (derivation; not measured).

**(e) ξ is per emitter; the question of the project is per molecule.** A spectrin tetramer has two epitopes for
the antibody used; each may carry 0 or 1 primary, each primary one or more secondaries, each secondary a number of
dyes that the manufacturer does not publish for the antibodies of this laboratory. None of that is in Dempsey et al.
or inferred by the model. "N emitters = 1 molecule" therefore has no supporting number today, and the layer reports
*emitters per cluster* as a reading, never a conversion to molecules (derivation; see `docs/emitters.md` when it
exists).

**(f) What learning ξ can and cannot do.** The hierarchical scheme learns ξ from the numbers of emitters and
localizations and is what the authors used on their dSTORM data (literature: Fazel et al. 2022, Methods). Two limits
(derivation): emitters closer than about one localization precision with few localizations are not separated, and a
learned ξ absorbs that fusion (two stacks with 14 cycles each look like one emitter with ξ ≈ 28); and in dense regions
the number of emitters and ξ compete (more emitters with fewer localizations, or fewer with more), which is why the
authors learn it in sparse regions and then fix it.

**Conclusion.** For obtaining emitters, grouping them and comparing with DBSCAN, the literature bracket plus the
learned prior are enough and no new experiment is required. For the statement "one cluster = one molecule" with a
number of emitters, a dedicated calibration is required (section 7 of the BaGoL plan; argument (e) has no
substitute).

## 7. When a dedicated calibration becomes necessary

Criterion of the project [**ad hoc, to be confirmed in phase F4 on synthetic data with known ξ**]. A calibration
on isolated labelled antibodies in the same buffer and the same microscope session (the B1 experiment of the work
plan) becomes necessary when any of these holds:

1. the mean of the learned ξ falls outside [½ × smallest, 2 × largest] of the tabulated means of the dye, after the
   same `loc_per_cycle` and `p_filters` (for Alexa Fluor 647 at 1 and 1: outside [7, 52]), i.e. outside what Dempsey
   et al. allow even when stretched (`calibration_needed` in the code reports this one with its sentence);
2. the reading "emitters per cluster" changes category (median ≤ 2 against ≥ 3) between `literature_MEA`,
   `literature_bME` and `learned`;
3. an **absolute count** is to be published (emitters per molecule, molecules per ring).

The factors ½ and 2 are a choice; F4 measures on synthetic data how far a learned ξ drifts from the true one when
labels are isolated, and will replace them.

## 8. Measuring loc_per_cycle and p_filters on your own data

Both factors are measured with the DNA-PAINT panel of MPS Explorer (toolbar button **DNA-PAINT**), which links
localizations into events with Picasso's algorithm (`tools/mps_paint.py`) and works on any file with a `frame`
column; nothing in it is specific to DNA-PAINT except the names. The panel is opened on the loaded file (whole file
when no ROI is drawn, the ROI and axial slab otherwise).

- **loc_per_cycle**: tab *Events*, first line: "*N* localizations → *M* binding events kept (*c* localizations per
  kept event, link radius *r* nm, max dark *d* frames)". *c* is the mean number of localizations per event, i.e. per
  switching cycle, **before** linking; the histogram "Binding event length" on the same tab shows its distribution
  in frames. After linking, the Emitters layer receives events and loc_per_cycle = 1 by construction; *c* is what
  the same file would have without linking, and it is the value to use when the layer is run on unlinked
  localizations. Use the default link radius (3 × the median lateral precision of the file) and 1 dark frame; a
  longer dark tolerance merges cycles and lowers the count of events, which the kinetics tab reports as a longer
  tau_dark.
- **p_filters**: events in the filtered file divided by events in the unfiltered file of the same acquisition, both
  counted on the same tab with the same link radius and dark frames, with no ROI (whole file). This is an
  approximation: a filter that removes one frame in the middle of a cycle splits the event, which the one dark frame
  partly repairs.

Note the two numbers with the file name, the link radius and the dark frames used; the prior stores them in its
origin text.

## 9. Limitations

- Only Alexa Fluor 647 and Alexa Fluor 568 are transcribed; the other 24 dyes of Table 1 are not (the Channel 2 box
  has the cycles column for all of them).
- The literature values are lower bounds measured in other conditions (section 6 (a)); the code labels them
  `literature_*` and never `calibrated`.
- `p_filters` has no literature value and defaults to 1 (ad hoc) until measured.
- The gamma shape with k = 1 is the published choice for simulated dSTORM; a measured cycle distribution on this
  laboratory's antibodies may prefer another k (the shape parameter is editable).
- The criterion of section 7 is ad hoc until F4.

## References

- Dempsey GT, Vaughan JC, Chen KH, Bates M, Zhuang X (2011). Evaluation of fluorophores for optimal performance in
  localization-based super-resolution imaging. *Nature Methods* 8(12):1027-1036. doi:10.1038/nmeth.1768 (Table 1,
  p. 1030; Online Methods).
- Fazel M, Wester MJ, Schodt DJ, Cruz SR, Strauss S, Schueder F, Schlichthaerle T, Gillette JM, Lidke DS, Rieger B,
  Jungmann R, Lidke KA (2022). High-precision estimation of emitter positions using Bayesian grouping of
  localizations. *Nature Communications* 13:7152. doi:10.1038/s41467-022-34894-2. Code: LidkeLab/BaGoL (MIT).
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Reference study of this project (Methods: buffer, irradiance, exposure, frames).
- Helmerich DA, Beliu G, Sauer M (2020). Multiple-labeled antibodies behave like single emitters in photoswitching
  buffer. *ACS Nano* 14(10):12629-12641. doi:10.1021/acsnano.0c06099 (Table 1: on-state lifetime ~10 ms).
- Schnitzbauer J, Strauss MT, Schlichthaerle T, Schueder F, Jungmann R (2017). Super-resolution microscopy with
  DNA-PAINT. *Nature Protocols* 12:1198-1228. doi:10.1038/nprot.2017.024 (the software whose code, Picasso 0.9.10, defines the frame linking; the article does not describe it).
