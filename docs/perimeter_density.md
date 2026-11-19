# Perimeter density analysis: how the beta-II spectrin is laid out along the membrane, without clusters

> **Resumen (español).** Esta página explica el análisis de densidad sobre el perímetro (`tools/mps_perimeter_*.py`,
> `tools/mps_auto_contour.py`, `tools/mps_trace_id.py`, `tools/perimeter_export.py`, `batch_perimeter.py`). Para cada
> axón de un corte transversal se cortan tres rodajas de 170 nm en z (z−1, z0, z+1), se traza **automáticamente** un
> contorno de membrana ajustado a las localizaciones (sin etiquetas de clúster), se cuentan las localizaciones de una
> banda de ±100 nm alrededor del contorno en bins de 100 nm de arco, y con esos perfiles se contestan tres preguntas:
> **A** (¿la densidad se agrupa, se aleja o se repite alrededor de su máximo?), **B** (¿la densidad depende de cuán
> curvo es el tramo?) y **C** (¿se parecen los perfiles de rodajas distintas?). Cada pregunta tiene su estadístico y su
> nulo de Monte Carlo, y se explica qué se puede y qué no se puede leer. z0 es por defecto el **pico** de
> localizaciones en z; el **foco** (nominal o empírico) es elegible y las dos distancias pico−foco se exportan siempre.
> La unidad entre grupos es el animal; con un solo animal o sin animal todo es descriptivo. La exportación es la tabla
> por axón v21 (columnas `pd_*`) más tablas largas, cada una con su diccionario. Los números rotulados [pilot] son del
> piloto exploratorio (18 axones de un solo experimento, un solo animal); las secciones «Validation» traen los falsos
> positivos y la potencia medidos en simulación (sección 13).
>
> **En el piloto (18 axones, un experimento, un animal) solo 3 contornos pasan la compuerta de calidad (4 si el
> desajuste entre rodajas solo informa). Ninguna lectura A, B o C de ese piloto es un hallazgo: son 3 o 4 axones de un
> animal, sin comparación posible entre animales ni entre genotipos.** Con 4 axones la potencia de A es 0,05–0,24:
> «no detectado» no es «ausente»; con menos de 9 axones el agrupado de A dice «n too small» y no se interpreta
> (mínimo fijado simulando de 5 a 11 axones, con el techo de 7,5 % de Bradley, y confirmado con 2000 conjuntos por escenario en 9, 10 y 11). C se lee solo sobre contornos confiables y, para la lectura principal, solo sobre rodajas que coinciden bien: la rodaja
> lateral mejor alineada a ≤ 32 nm RMS de la central (calibrado en simulación, sección 8; en el piloto pasan 3 axones).
> La fuga axial del par −1/+1 con las fracciones reales de cada axón es mucho mayor que la de rodajas iguales. El
> ancho axial se midió en el piloto con una «NeNA en z»: la cifra principal es **47 nm**, con **36–51 nm** como
> incertidumbre; 77 nm queda solo como cota superior (incluye el grosor físico del anillo). En datos nuevos hay que
> medirlo por archivo. Las lecturas de vecinas («leakage only» / «neighbours above leakage») comparan cada par con su
> propia fuga con las fracciones reales del axón, a 47 nm. El recorte opcional de localizaciones interiores solo actúa
> si la curva polar pasa las reglas estrictas del contorno automático.

This page is the public methods record of the **perimeter density analysis**. It is independent of how a cluster is
defined: nothing here uses cluster labels, DBSCAN parameters or cluster centres. Every number is labelled:

- **[spec]**: fixed by the specification of the analysis (not derived here).
- **[lit]**: literature (reference at the end).
- **[derivation]**: a calculation shown here or in the code docstring.
- **[own sim]**: a simulation of this program's own simulator (`tools/mps_perimeter_sim.py`), synthetic data only.
- **[pilot]**: the exploratory pilot of 18 axons of one experiment (one animal), read once, never used to tune a
  test.
- **[ad hoc]**: a convention of this program without further support yet.

A number whose origin cannot be checked in the code or the references is not stated. Two band half-widths exist in the
program and must not be confused: the **100 nm** band around the perimeter contour of this analysis
(`pd_band_half_width_nm`) is not the **50 nm** band of the pipeline null H2 (`tools/mps_pipeline_null.py`).

## 1. What is measured and why

**What question does it answer?**
How is the beta-II spectrin organised *along the perimeter* of an axon, as read straight from the localizations: is
there a place where the density piles up, does it follow the curvature of the membrane, and does the pattern repeat
from one thin slice of the cut to the next?

**Plain explanation.**
A transverse cut of an axon shows a ring of localizations. Walk once around that ring and count how many
localizations fall in each 100 nm stretch: that is a density profile. Everything below is a question asked of such
profiles, drawn on three neighbouring 170 nm slices of the cut. No cluster is ever identified, so the answer cannot
depend on which clustering method or which DBSCAN parameters were used.

**Example.**
An axon of 8 µm perimeter gives about 80 bins. If the highest bin sits at 3 µm of arc, question A asks whether the
neighbouring density is higher, lower or periodic compared with the same profile shifted at random; question B asks
whether bins on tight bends have more or fewer localizations; question C asks whether the profile of the slice above
looks like the profile of the slice below.

Technical frame. One axon is a pick of localizations (x, y in nm, z in nm) with no cut in z and no clustering. The
analysis returns one row per axon with a status; a failure keeps what was computed and says why (section 10).

## 2. The pipeline per axon

The order is fixed (`tools/mps_perimeter_analysis.py`, `analyze_perimeter`):

1. identity and trace label (section 9);
2. z0 and the two foci (section 3);
3. three contiguous slices (section 3);
4. one automatic contour (section 4);
5. projection, band and the three profiles on one grid (section 5);
6. questions A, B, C (sections 6-8);
7. sensitivity: profiles are **recomputed** at every band half-width of the sensitivity list (50 and 150 nm) and A, B,
   C are read again; C is repeated with an axial guard (42.5 nm);
8. diagnostics (unsupported fraction of the contour, radial shift of the lateral slices, tortuosity and curvature per
   window).

Fewest finite localizations analysed: 100 (`MIN_LOCALIZATIONS`, [ad hoc]: below that an 8-bin profile has about 12
counts per bin, nothing A, B or C can test).

## 3. z0, the foci and the three slices

**What question does it answer?**
At which height of the cut do we look, and how far is that height from the focal plane?

**Plain explanation.**
In a transverse cut the axon runs along z, so there is no "middle of the axon" in z. What exists is the plane where
the microscope is in focus and the height where most localizations were detected; the two nearly coincide because the
detection efficiency is highest in focus. The program lets you choose either as the centre z0 of the central slice and
always reports both distances so the choice can be judged.

**Example.**
If the peak of the z histogram is at -20 nm and the nominal focus (z = 0 of the file) is at 0, the peak minus the
nominal focus is -20 nm. That number goes to the table whichever z0 you chose.

**Technical.**

| item | definition | origin |
|---|---|---|
| peak | maximum of a Gaussian kernel density of z, kernel sigma 25 nm, evaluated on a 1 nm grid | [ad hoc] sigma, see below |
| nominal focus | wx = wy crossing of the file's 3D calibration record times its magnification (z = 0 of the file when the record is the one Picasso used) | the record, `tools/mps_zcalibration.py` |
| empirical focus | z where the per-bin median of sx − sy crosses zero (bins of 40 nm with at least 200 localizations; zero crossing nearest z = 0; no interpolation across empty bins) | [ad hoc] bin 40 nm and 200 per bin |
| z0 criterion | `peak` (default), `focus_nominal`, `focus_empirical` (the option "focus" defaults to nominal) | [spec] |

- **Both foci are always estimated** and `pd_peak_minus_focus_*` is exported signed and absolute against both
  (`pd_peak_minus_focus_nominal_nm`, `pd_peak_minus_focus_empirical_nm` and their `pd_abs_*`). `pd_peak_minus_focus_nm`
  is the distance to the focus of the selected kind (nominal when z0 is the peak).
- **No silent fallback.** A focus criterion with no such focus gives `pd_status = "no focus"` and no analysis; it does
  not fall back to the peak. The empirical focus is empty without sx/sy columns, with sx = sy constant (simulated
  files) or without a zero crossing; the nominal focus is empty without a calibration record with curves and
  magnification.
- **The peak is not the program's mixture-model peak.** The plan was to read `fit_z_periodicity(z).main_peak_nm`; the
  module uses its own kernel density instead (deterministic, numpy only). The two differ by a median of 7.7 nm and at
  most 36 nm on the 18 pilot axons [pilot]. 25 nm is an ad hoc width: under a seventh of the 170 nm period, so the
  peak is not smeared across rings. This deviation from the plan is recorded for decision.
- **Slices.** Slice 0 is [z0 − 85, z0 + 85) nm; slices −1 and +1 follow contiguously, 170 nm each, so the three cover
  [z0 − 255, z0 + 255). Intervals are half-open: a localization on a shared edge belongs to exactly one slice. 170 nm
  is fixed by the specification and equals the program's reference period of the ring (see
  [viability_and_reference_period.md](viability_and_reference_period.md)) [spec].
- **Axial guard** (sensitivity only): trimming 42.5 nm (a quarter of the thickness, [ad hoc]) from both sides of every
  slice leaves 85 nm cores; the localizations in the gaps are labelled outside. Default 0 (contiguous slices).
- **No slice is dropped for having few localizations.** The count and the percentage of each lateral slice relative to
  the central one (`pd_pct_m1_of_0`, `pd_pct_p1_of_0`) are reported instead, together with the fraction of the axon's
  localizations inside the three slices (`pd_frac_locs_in_slices`).
- **Truncation.** `pd_z_coverage_m1/0/p1` is the fraction of each slice's z interval that lies inside the file's z
  range; below 1, the slice is incomplete (a file cut in z). Truncated slices are also named in `pd_z_truncated_slices`
  and the warnings. The analysis goes on.
- **Exploratory pilot, one animal** [pilot]: the empirical focus lay at about −43 nm of the nominal one in the 18
  axons (range −47 to −38 nm), so the two foci are not interchangeable; the peak minus the nominal focus had a median
  of −1.7 nm and an absolute median of 19.9 nm (range 0.3-62.7 nm). These are exploratory numbers from one experiment;
  the cause of the 43 nm offset was not verified.

## 4. The automatic contour

**What question does it answer?**
Where is the membrane, so that localizations can be placed along it?

**Plain explanation.**
The program draws a closed smooth curve through the ring of localizations by itself. It first marks the dense spots
(nodes), joins them into a closed polygon, then fits a smooth spline to the *localizations* (not to the nodes), and
finally checks the curve for the usual failures. If the curve is not acceptable the axon is rejected with a reason;
nothing downstream is computed on a bad curve.

**Example.**
A ring with an interior lump 400 nm inside the membrane: the lump creates a node, but the rule that drops nodes the
curve can bypass removes it, and the spline follows the ring.

**Technical** (`tools/mps_auto_contour.py`; deterministic, no random draw; no cluster label in or out, and a test
inspects the imports).

1. Nodes: local maxima of the localization density smoothed with a Gaussian of 50 nm, at least 200 nm apart, above 5 %
   of a reference level (the 8th strongest maximum). Nodes are not clusters: they only order the initial curve.
2. Tour: the program's own `reconstruct_perimeter(all_starts=True)` (polar angle plus 2-opt from every start) joins
   the nodes. Vertices the density band bypasses are dropped and the tour rebuilt (at most 8 passes).
3. Curve: the periodic P-spline fitted to the localizations (`tools/mps_membrane.py`), knots 400 nm apart (the
   curvature module's spacing).
4. Nodes farther than 150 nm from the fitted curve are dropped and steps 2-3 repeat (at most 2 rounds). A node at the
   bottom of an inward bend tighter than 250 nm of radius is dropped too (at most 3 drops).
5. Every curve runs counter-clockwise seen from +z (a clockwise one is reversed keeping its arc origin; exported as
   `pd_contour_reversed`).

Origin of each constant: the sweep (one parameter at a time; 5 shapes x 12 seeds, six conditions) is in the docstrings
of `tools/mps_auto_contour.py`. Smoothing sigma 50 nm (swept 25/50/75/100 nm: mean error 6.3/6.4/8.0/10.1 nm), node
threshold 0.05 (swept 0.05/0.15/0.30), minimum node separation 200 nm (swept 100-400 nm), knot spacing 400 nm (swept
300/400/600 nm: mean error 5.0/6.4/14.1 nm) [own sim]. The 150 nm node-drop distance had no effect in the swept range
[own sim] and is kept as a last guard [ad hoc]. The grid pixel (25 nm) and the minimum number of nodes (8) are
[ad hoc] and [plan] respectively.

**Curve types.** What the table calls the method and the knot spacing identifies how the curve was obtained
(`pd_contour_method`, `pd_contour_knot_nm`):

| type | when | note |
|---|---|---|
| primary: density nodes, knots 400 nm | first attempt | the curve validated in simulation |
| density nodes, knots 600 nm | the primary is rejected | a stiffer spline, 600 nm = the membrane fit's own spacing |
| polar | the nodes curve is rejected and the cloud is star-shaped about its centre | 72 sectors of 5 degrees, radial bin 40 nm |
| drawn | explicit, a manual trace | comparison only, never the default |

These are three different estimators with different rigidity; a fallback curve is not equivalent to the primary one.
The curve used is exported in one column, `pd_contour_kind` (`primary`, `stiff fallback`, `polar fallback`, `polar`,
`drawn`; `external` for a curve given by a caller-supplied function, `none` when rejected), and folded into four strata
in `pd_curve_stratum` (primary / stiff / polar / drawn). In simulation their errors differ (95th percentile of the
error, median over reliable curves with clouds up to 100 nm: primary 36 nm, n = 404; stiff 81 nm, n = 85; polar 74 nm,
n = 78) [own sim], so every pooled result is also given per stratum (section 6).

**Acceptance checks** (a failed check gives `pd_status = "contour failed"` and the reason in `pd_failure`): the curve
crosses itself; its length exceeds 1.5 times the perimeter of its convex hull ([own sim] plus an ad hoc margin: correct
curves stayed at or below 1.25 in simulation); an inward notch tighter than 250 nm of radius survives the drop rule
([own sim]: the tightest bend of a correct curve was at least 297 nm, interior lumps gave 15-242 nm); the density
deeper than 250 nm inside the curve exceeds half the density in the band (no lumen: a filled disk is not an axon; [own
sim]); too few nodes (fewer than 8); too few localizations (fewer than 200); the spline fit aborts. A warning
(not a rejection) is added when the hull ratio exceeds 1.3, or when a star-shaped curve is more than 120 nm from the
polar one. The polar curve, used as the last fallback, is also rejected when it has an inward notch tighter than
150 nm of radius or when the band runs persistently more than 75 nm to one side of it ([own sim]). More than 150
density nodes gives a warning (slow contour) and only the 250 strongest are kept ([ad hoc] time cap).

**Fit-quality gate** (what question does it answer? whether an accepted curve really sits on the cloud, so that the
±100 nm band counts the membrane and not something else). An accepted contour is not automatically a reliable one.
After the acceptance checks the program looks at where the band of localizations lies relative to the curve, over the
whole cloud and in windows of 500 nm of arc every 100 nm, and marks the contour **not reliable** when any of these
holds (`tools/mps_auto_contour.py`, `quality_reasons`):

| rule | limit | origin |
|---|---|---|
| fraction of the localizations of the three slices within ±100 nm (`pd_contour_frac100`) | below 0.55 | [own sim + review]: no correct simulated curve with a cloud up to 100 nm falls below it (correct curves: ≥ 0.80 for 20-50 nm clouds, ≥ 0.68 at 75 nm, 0.58-0.59 at 100 nm) |
| robust radial SD within 400 nm (`pd_contour_robust_sd_nm`) | above 130 nm | [own sim + review]: correct curves stayed at or below 111 nm |
| band offset that persists over 3 consecutive windows (`pd_contour_persistent_offset_nm`) | above 75 nm | [own sim]: 1.0 % false alarms on correct curves, 39 % detection of bad ones, calibrated on 1872 simulated curves; 75 nm because, with the median real radial SD of 73 nm, moving the band 50 nm costs 12 % of its localizations and 100 nm costs 40 % [derivation] |
| share of the arc whose band lies more than 100 nm from the curve (`pd_contour_off_band_fraction`) | above 0.15 | [own sim]: three times the largest share of correct curves (0.049) |
| misfit between the z slices: RMS of the displacement of slice −1 or +1's ring from the central one about the same curve (`pd_contour_slice_rms_minus_nm`, `_plus_nm`; mark `pd_contour_slice_misfit`) | above 50 nm | [derivation]: 50 nm costs about 12 % of the band; not calibrated (the contour simulator has no z) |

The largest offset of a single window (`pd_contour_max_window_offset_nm`) is exported as a diagnostic only: as a rule it
flagged 12 of the 12 accepted real contours (earlier count, contour fitted on the whole pick) and 9.8 % of the correct simulated ones. The quality is always measured on
all the localizations of the three slices, also when the optional interior trim is on (below).

**What the gate does with a contour.** The contour keeps `pd_status = "ok"` and stays in the per-axon table with its
reasons (`pd_contour_quality_reasons`); `pd_contour_reliable` is the contour's own verdict (false on any rule). Whether
the axon enters the pooled analyses is a separate column, `pd_pool_included`, decided by the pooling: **every rule but
the slice misfit keeps the axon out** (`pd_contour_gate_reasons`, `pd_pool_exclusion_reason`); **the slice misfit only
informs** (decision D12): the axon stays in the main pooled set with the mark, and every pooled result is given a
second time without the axons the misfit marked (section 6). The cause of the misfit (tilt, optical distortion with z,
composition by height) is not established.

**Support.** The density under the curve is sampled every 10 nm; stretches longer than the gap length with no density
under them are exported (`pd_contour_support_unsupported_fraction`, `pd_contour_n_unsupported_arcs`): the curve is an
interpolation there. (Not the same as `pd_contour_unsupported_fraction`, the share of bins not measured, section 5.)

**Optional interior trim** (decision D8; **off by default**). With `PerimeterParams.contour_trim_interior_nm` (batch
`--trim-interior 250`) the localizations deeper than that below the polar ring of the three slices
(`tools/mps_unanchored.trim_interior`, the unanchored-cluster package) are left out of the contour fit only. **The
trim is applied only when the polar curve passes the strict rules of the automatic contour** (user's decision
2026-11-17; `mps_unanchored.strict_polar_reference`): the P-spline fitted on the polar contour must come out with no
rejection reason (no self-crossing, length at most 1.5 times its convex hull, no inward bend tighter than 150 nm, the
band never more than 75 nm to one side of it along a stretch, a lumen). Otherwise nothing is trimmed and
`pd_contour_trim_status` says `not trimmed: the polar reference fails the strict rules of the automatic contour
(<rule>)`; nothing is trimmed either when P14's polar ring itself is not valid (a cloud that is not star-shaped, too few
localizations). In simulation, against a valid reference the trim removed at most 0.5 % (95th percentile) of
the true membrane; against the polar ring of a cloud that is not star-shaped ("C") it removed 5-22 % [own sim]. On the
exploratory pilot two counts were reported: "with a valid reference only 4 axons are trimmed and no reliability
changes" (contour report v2) and "16 accepted, reliable from 4 to 8" (validation run). **Both are right, for two
different validity rules of the reference** (reproduced 2026-11-17 on the integrated code with the pilot's files, read
only). The report v2 called a reference valid when the polar CURVE passes the contour's own rules (no notch, not leaving
the cloud): 4 of 18, all already reliable, so nothing changes. Up to v20 the
program implemented a weaker rule: `trim_interior` refused only a cloud that is not star-shaped (2 of 18) and trimmed
the other 16 (31-1972 localizations); the gate then accepted 16 contours and 8 were reliable under D12 (4 more than
without the trim; 5 without slice misfit). The 4 added axons are exactly those whose polar
reference the stricter rule rejects, the case in which simulation lost 5-22 % of the membrane and the gate did not
always see it. **Since 2026-11-17 the trim requires the strict rule** (above), so switched on it acts on the same 4
axons as the report v2 and changes no reliability. The trim stays off by default (D8); the weak rule is kept only
behind `trim_interior(..., require_strict=False)`, for validation.

**What the gate does not see (measured).** Curves that go wrong where there are no data (the ends of a 2 µm gap: see
the support columns); curves that pass through an interior lump (in simulation 11 of 77 accepted curves of the
lump scenario went through the lump and still came out reliable: the gate reads stretches of at least 700 nm and a
lump's notch is 300-400 nm); a stiff curve that smooths over a 700 nm bay.

**Limits.**
- The contour is not "the membrane" in a verified sense: with real data there is no ground truth. In the exploratory
  pilot (18 axons, one experiment, one animal) two reasonable automatic methods differed by a mean of 45 nm (95th
  percentile 144 nm) and in perimeter length (median 15.5 against 9.6 µm) [pilot].
- The smooth curve flattens real concavities narrower than its knot spacing; a bay can then fall outside the band.
- Selection bias on shape: only curves without closed concavities are accepted, so question B is read on axons that
  were filtered by their own concavity.
- Time: the tour grows roughly as N^2.5 in the number of nodes; hundreds of nodes take minutes.
- The nodes parameters were not re-swept on a real-width cloud: the real radial spread (robust SD 47-165 nm in the
  pilot) is two to six times the simulated one (20 and 50 nm).
- **On the exploratory pilot only 3 of 18 contours are reliable (4 under D12).** In the analysis the contour is fitted
  to the localizations of the **three slices**, not to the whole pick; on that path (validation run P11, section 13)
  11 contours are accepted (4 primary, 5 stiff, 2 polar fallback) and 7 rejected (inward notch in 5, one implausible
  length, one aborted fit). Of the 11 accepted, 7 fail at least one rule other than the slice misfit: all 7 have a
  persistent band offset above 75 nm (93-302 nm), some also a fraction within ±100 nm below 0.55 (range of the 7:
  0.43-0.62) or a robust radial SD above 130 nm (range 101-156 nm); one (a polar fallback)
  fails only the slice misfit. The slice misfit marks 7 of the 11 accepted (larger lateral-slice RMS
  67-146 nm, against 27-46 nm in the three reliable ones). The 5 stiff curves are all not reliable. The three reliable
  contours (two primary, one polar fallback) are exactly the three a reviewer judged good by
  looking at the figures, but that agreement is a hint, not a proof: there is no ground truth on real data. With the
  slice misfit only reported (D12), the main pooled set of the pilot holds 4 axons (those three and one more) and the
  set without slice misfit 3. (An earlier count, made by fitting the contour on the whole pick, gave 12 accepted with 3
  primary / 9 stiff / 0 polar; the reliable axons were the same, the curve kinds were not. The validation run computed
  the gate with the same functions the integrated analysis uses, on a base just before the integration; the integrated
  v20 code reproduces the same 11 accepted, 4 reliable and 3 without slice misfit, run on the pilot's files on
  2026-11-17.) **Any pooled result of questions A, B or C on the pilot
  therefore rests on 3 or 4 axons of one animal.** This is a finding about the contour on real clouds, not something
  to fix by lowering the limits.
- In simulation the gate's false alarms were 1.0 %, but the simulator does not reproduce the real regime of the
  persistent offset (correct curves stay at or below 55 nm in the real-width scenarios; almost every real contour
  exceeds 75 nm). Whether the real curves are truly off the membrane in places or the real cloud has structure the
  simulator lacks (patches, lumps near the membrane) cannot be told apart.

## 5. Band, profile and common frame

**What question does it answer?**
Which localizations count as "on the membrane", and how are they counted along the arc?

**Plain explanation.**
Each localization is projected onto the contour, giving its position along the arc and its distance from the curve.
Those within ±100 nm are kept and counted in bins of about 100 nm along the arc. The same grid and the same origin are
used for the three slices, so profiles of different slices can be compared bin by bin.

**Example.**
A 10 µm contour has 100 bins of 100 nm; a localization 40 nm from the curve at 3.52 µm of arc adds one to bin 35
(bins are centred on the grid points).

**Technical.**
- **Band**: half-width 100 nm by default, symmetric around the curve [ad hoc, pilot]. The band is symmetric because the
  number of localizations per unit of arc carries a factor (1 + κ·r) with κ the curvature and r the signed offset;
  integrating over a symmetric band cancels the first-order term, whereas an asymmetric band would put a spurious
  curvature-density relation into question B [derivation]. In the exploratory pilot the ±100 nm band held a median of
  77 % of the localizations (range 67-85 %) and the radial FWHM of the cloud was a median of 148 nm against 19 nm
  expected from the precision [pilot]; these figures justify the width only as a pilot choice. Alternative mode `fwhm`:
  half-width = the axon's own Gaussian-equivalent radial FWHM (2.3548 × 1.4826 × MAD of the offsets) clipped to
  [50, 150] nm.
- **Sensitivity (mandatory)**: A, B and C are recomputed with half-widths of 50 and 150 nm [ad hoc]; the column
  `pd_band_sensitivity` is `consistent` or says which decision changes.
- **Always reported**: radial FWHM and robust SD, the fraction of localizations within 50/100/150 nm, and the fractions
  beyond the band inside and outside.
- **Grid**: n_bins = max(8, round(L / 100 nm)); effective bin = L / n_bins (nominal 100 nm [spec]; exported as
  `pd_bin_nm_effective`). A closed uniform grid is what the circular rotation and the Fourier computation need; a
  partial last bin would break both. The 8-bin floor is [ad hoc].
- **Relative density**: counts divided by the slice's own maximum, for display and for question A. Pearson and
  Spearman correlations are scale-free, so C and B do not depend on this normalisation.
- **Maximum**: the highest bin; a tie goes to the bin whose three-bin sum is larger, then to the lower index; the tie is
  flagged (`pd_max_tie`). No smoothing before the maximum.
- **Not measured bins**: a bin lying entirely inside an arc stretch longer than 400 nm without a localization is not
  measured ([ad hoc]); the contour there is an interpolation. Those bins are left out of every test and the fraction is
  exported (`pd_contour_unsupported_fraction`).
- **Common frame ("z0_max")**: the origin of the arc for the three slices is the centre of the maximum bin of the
  *central* slice. The three profiles are rotated by the same amount, so their correlations do not change. Aligning each
  slice on its own maximum is allowed only for question A and for display; the code refuses to correlate profiles in
  that frame.

## 6. Question A: does the density pile up, stay away, or repeat around its maximum?

**What question does it answer?**
Around the highest bin of a slice (the "large cluster", defined by the profile, not by DBSCAN), is the rest of the
density attracted to it (an excess near it), repelled by it (an empty zone), or laid out at regular intervals counted
from it?

**Plain explanation.**
If every profile is lined up on its own maximum, the bin at zero is the maximum by construction and its neighbours are
high just because a clump spans more than one bin. Pooling many aligned profiles shows a peak at zero even when
nothing is there. So the test never looks at the maximum or at its flanks, and its null *repeats the alignment*: it
shifts the rest of the profile at random and asks how often a shifted copy looks as structured around the maximum as
the real one.

**Example.**
A flat profile with Poisson noise, aligned on its maximum and averaged over 18 axons, shows a peak at zero (the
artefact, documented in the tests). The two statistics below do not flag it, because the zone and the maximum are
excluded and the null is the shifted remainder.

**Technical** (`tools/mps_perimeter_a.py`).

- **Exclusion zone** (rule `baseline`, default [own sim]): from the maximum to each side down to the first bin below
  `median + max(0.10 × height, 2 × noise)` (height = maximum above the median, noise = 1.4826 MAD of the profile),
  then 1.5 times as far, never less than 200 nm per side. A Gaussian cluster reaches the 10 % threshold at 2.15 sd, so
  the zone ends at about 3.2 sd, where less than 0.1 % of the cluster is left. The previous rule (stop at half height,
  `half_max`) leaves the flank of every resolved cluster in the remainder and gave an "attractor" reading in 20 of 20
  sets of 18 pooled axons without any attraction (a cluster with sd 150 nm) [own sim, review of the first design]; it
  is kept only for the sensitivity table and must not be used for a reading. The 200 nm minimum is [ad hoc]. The zone
  width (nm, not bins) is exported (`pd_a_exclusion_nm`).
- **Null**: the remainder (the arc outside the zone) is shifted circularly inside its own arc by a random number of
  bins (toroidal shift; Lotwick & Silverman 1982, cited from memory, not re-read). It keeps the number of localizations,
  the perimeter, the maximum, its shape and the autocorrelation of the remainder (blinking clumps included) and breaks
  only the relative position. No remainder value exceeds the maximum, so the maximum stays the maximum in every
  replicate.
- **Both statistics are computed on the remainder divided by its own mean** (a constant under the shift): without it, a
  pooled statistic follows whichever axons hold each distance and its null is not centred.
  - `contrast`: mean density within 1000 nm of the zone ([ad hoc], to be swept) minus the mean of the rest, in units of
    the remainder's mean (+0.2 = 20 % more density near than far). Two-sided. Its sign is read against the null
    (`value − null mean`), never from the raw value. Positive: "attractor". Negative: "inhibition zone".
  - `period`: the largest *phase-locked* periodogram power of the folded remainder over periods from 300 nm
    ([ad hoc]: three nominal bins) to half the span: the variance carried by a cosine whose crests sit at multiples of
    the period counted from the maximum. One-sided; the maximum over periods is taken in every replicate, so there is
    no hidden multiple comparison. A peak at the longest scanned period is a trend, not a repeat (flagged in
    `pd_a_period_at_edge`).
  - Why phase-locked: the modulus of the Fourier transform is invariant under a circular shift, so the plain
    periodogram peak equals itself in every replicate and p is 1 whatever the data (a test shows it). What the shift
    null can test is whether modes are placed relative to the maximum. Whether the remainder is periodic regardless of
    the maximum is a different question with a different null and is not answered here.
- **Per axon**: z (signed, against its own null) and a Monte Carlo p (1 + b) / (1 + n) [lit: Phipson & Smyth 2010]. They
  are descriptive and carry no correction. For one axon a remainder of period T looks almost the same shifted by T, so
  the per-axon `period` test has little power and its null is skewed: do not feed its z to a t-test unseen. The
  per-animal table therefore averages it but never gives it a p (section 9).
- **Pooled** (`mps_perimeter_a.pooled_test`): mean of the folded profiles, one axon one weight, on an absolute-nm axis
  (alternative: fraction of the perimeter); the number of axons supporting each distance is exported (perimeters run
  from 6 to 32 µm, so long distances are held by few axons). The pooled null shifts the remainder of *each* axon
  separately and pools identically. The two pooled statistics are corrected between them with Holm (1979).
- **Which axons are pooled** (`tools/mps_perimeter_pool.py`, `pool_gate`; decisions D9 and D12). Only axons whose
  analysis finished and whose contour passed the quality gate of section 4, the slice misfit aside. Each pooled result
  (pooled A, per animal, between genotypes) is reported for several sets, one after the other with a `scope` column:
  "reliable contours (slice misfit included)" (the main set), "reliable contours without slice misfit", and "reliable
  contours, curve: primary / stiff / polar / drawn" (the main set restricted to one kind of curve: they are different
  estimators). The `gate` table says, axon by axon, whether it enters each set and why not; `gate_summary` counts the
  axons of every set and the reasons of the others. On the exploratory pilot the main set holds 4 axons and the set
  without slice misfit 3, and the curve strata hold 2 (primary), 2 (polar) and 0 (stiff) (sections 4 and 13): a pooled
  reading there is a reading of three or four axons of one animal, and a stratum of two axons is no pooled reading at
  all.
- **Minimum number of axons for a pooled reading of A** (`mps_perimeter_pool.MIN_POOLED_A_AXONS` = **9**; 11 in v20,
  10 on 2026-11-17 after simulating every size from 5 to 10, **9 on 2026-11-19** after 2000 sets per scenario at 9, 10
  and 11 axons, user decision). Below it the label is "n too small (n = k axons
  < 9): not interpreted; computed label '...'" and the numbers stay in the table, not to be read. **Criterion**
  [own sim; Bradley 1978 robustness band 7.5 %, the same ceiling as the slice criterion of C], fixed before reading
  the table: the smallest size that meets, together with every larger simulated size, (S) every null emits each
  reading ("attractor", "regular", "inhibition zone") with an upper 95 % Wilson bound at most 0.075, and (I) every
  effect the reading can recognise (attractor, inhibition zone, regular) appears with a lower 95 % Wilson bound above
  0.075, i.e. more often than a calibrated null may produce it, so "no structure detectable" carries information
  against it. The halo is left out of (I): about 0.01 at every size up to 18 (the robustness rule takes it for a
  cluster flank), and no number of axons fixes it. `validate_perimeter_a.py`, P1's simulator and the program's pooled
  A; 8-25 µm axons, 30 localizations per bin, blinking, true arc; 300 sets per null (flat, clumps, resolved cluster
  sd 150 nm, t-tailed cluster 300 nm) and 500 per effect at sizes 5-10 (sizes 4, 11 and 18 from the validation, 200
  per effect):

  | axons | worst null reading (rate, Wilson 95 %) | attractor | inhibition zone | regular | halo | S | I |
  |---|---|---|---|---|---|---|---|
  | 4 | 0.027 (0.014-0.052) | 0.24 (0.19-0.30) | 0.045 (0.024-0.083) | 1.00 | not run | yes | no |
  | 5 | 0.030 (0.016-0.056) | 0.22 (0.19-0.26) | 0.050 (0.034-0.073) | 1.00 | 0.004 | yes | no |
  | 6 | 0.023 (0.011-0.047) | 0.26 (0.22-0.30) | 0.070 (0.051-0.096) | 1.00 | 0.010 | yes | no |
  | 7 | 0.023 (0.011-0.047) | 0.30 (0.27-0.35) | 0.076 (0.056-0.103) | 1.00 | 0.004 | yes | no |
  | 8 | 0.023 (0.011-0.047) | 0.36 (0.32-0.40) | 0.074 (0.054-0.100) | 1.00 | 0.006 | yes | no |
  | 9 | 0.017 (0.007-0.038) | 0.35 (0.31-0.39) | 0.092 (0.070-0.121) | 1.00 | 0.006 | yes | no |
  | **10** | 0.027 (0.014-0.052) | 0.39 (0.35-0.44) | **0.104 (0.080-0.134)** | 1.00 | 0.010 | yes | **yes** |
  | 11 | 0.017 (0.007-0.038) | 0.42 (0.35-0.48) | 0.125 (0.086-0.178) | 1.00 | not run | yes | yes |
  | 18 | 0.026 (0.015-0.044) | 0.62 (0.55-0.68) | 0.275 (0.218-0.341) | 1.00 | 0.010 | yes | yes |

  With 500 sets the inhibition zone decided the minimum, and **the margin at 10 was thin** (lower bound 0.080 against 0.075; 0.070
  with 9). If only the attractor counted, 4 axons would already pass (I); the criterion asks it of every effect the
  reading can name. The minimum makes a reading informative, not powerful: the conventional 0.8 is not reached at any
  size up to 18 (attractor 0.62, inhibition 0.28 with 18 axons). Report: INTEGRACION_FINAL2.md of the validation
  (outside the repository).

  **Confirmation with 2000 sets (2026-11-19).** The sizes at the edge were simulated again with 2000 sets per
  scenario (4 nulls and 4 effects) at 9, 10 and 11 axons, with new seeds and the same criterion:

  | axons | worst null reading (rate, Wilson 95 %) | attractor | inhibition zone | regular | halo | S | I |
  |---|---|---|---|---|---|---|---|
  | 9 | 0.016 (0.011-0.022) | 0.38 (0.36-0.40) | 0.106 (0.094-0.121) | 1.00 | 0.005 | yes | yes |
  | 10 | 0.021 (0.016-0.028) | 0.42 (0.40-0.44) | 0.112 (0.099-0.127) | 1.00 | 0.006 | yes | yes |
  | 11 | 0.025 (0.019-0.032) | 0.43 (0.41-0.45) | 0.119 (0.106-0.134) | 1.00 | 0.009 | yes | yes |

  With 2000 sets the inhibition zone at 9 axons clears the ceiling with margin (lower bound 0.094), so the minimum
  is **9** [own sim; user decision 2026-11-19]. Eight does not pass: its 500-set estimate of the inhibition zone
  (0.074) is already below the ceiling. The power against the attractor stays near 0.4 and against the halo near
  0.01: "no structure detectable" with 9-11 axons says little against those two patterns.
- **Per-axon reading** (`pd_a_reading` in the per-axon table): the same labels from the axon's own two tests, Holm
  between them, without the zone-widening check (not run per axon) and therefore never "robust"; descriptive only.
- **Reading** (descriptive labels; the numbers are what is reported):
  - "attractor": pooled contrast above its null, significant after Holm **and** still above and significant (raw p)
    with the zone widened 2 and 3 times;
  - "inhibition zone": pooled contrast below its null, significant after Holm;
  - "regular": pooled period significant after Holm, not at the longest scanned period, and still so with the widened
    zones;
  - "... (not robust to the exclusion zone: cluster flank)": significant with the default zone only. These readings are
    **not robust** and should not be reported as findings;
  - "no structure detectable", always with the number of axons.
  Without an animal or with one animal the reading starts with "EXPLORATORY, within-sample".
- **What "attractor" means and what it does not.** It means only: more density near the edge of the exclusion zone than
  far from it, compared with shifting the rest of the profile. A cluster wider than the zone gives the same reading with
  no attraction; that is why the reading must survive widening the zone beyond the width of the cluster. A halo and the
  shape of the cluster differ only by scale: what the program calls an attractor is excess beyond the edge of the
  cluster as the zone rule defines it.
- **The global band** drawn with the pooled profile (a simultaneous band by directional quantiles, drawn only where at
  least 3 axons contribute) is for display. "The profile leaves the band" would be a third test outside the Holm family;
  the decision is the two statistics.
- **Scope**: a randomization test *within these axons* (each axon is its own stratum); it does not generalise to the
  population of animals. The unit between groups is the animal (section 9).
- **Analyst's choices to freeze with a date before new data**: the 1000 nm "near" distance, the 300 nm minimum period,
  the zone rule and its three numbers, the widening factors (2, 3), the axis and the weights.

**Validation** (`validate_perimeter_a.py`; pooled sets of 18 axons, 30 localizations per bin, true arc, 499 pooled
replicates; criterion: false readings at most 0.07, power at least 0.8) [own sim]:

| condition | sets | "attractor" robust | "regular" | "inhibition zone" | other | verdict |
|---|---|---|---|---|---|---|
| flat | 500 | 0.004 | 0.020 | 0.002 | Holm contrast 0.008, period 0.022 | readings pass; contrast and period conservative (below 0.03) |
| blinking clumps | 500 | 0.008 | 0.026 | 0.008 | contrast 0.028, period 0.032 | pass |
| resolved Gaussian cluster, sd 100 / 150 / 300 nm | 200 each | 0 / 0 / 0 | 0.015 / 0.010 / 0.010 | 0.020 / 0.010 / 0.015 | | pass |
| heavy-tailed cluster (t3), 100 / 150 / 300 nm | 200 each | 0 / 0 / 0.010 | 0.020 / 0 / 0.010 | 0 | "attractor (not robust)" 36 / 66 / 174 of 200 | pass (the heavy tail comes out "not robust", which is correct) |
| attractor, 600 nm | 200 | **0.62** (0.55-0.68) | | | 76 more as "not robust" | **fails** (power below 0.8) |
| halo around a resolved cluster | 200 | **0.010** | | | 178 "not robust", 20 no structure | **fails** |
| inhibition zone, 800 nm | 200 | | | **0.275** (0.22-0.34) | 125 no structure | **fails** |
| regular modes, 800 nm | 200 | | **1.000**, period 800 ± 100 nm | 200 of 200 also "inhibition zone" | | passes, with an extra label |

(Wilson 95 % intervals in brackets.) Per axon, under the flat null, the contrast and the period tests gave p < 0.05 in
4.6 % each of 9000 axons. The artefact at zero is reproduced: the raw pooled profile is 1 at lag 0 with no structure,
and the 100 nm lag falls inside its null band in 500 of 500 sets.

Reading. **A does not invent structure**: under the eight nulls no reading exceeds 2.6 %. **A has little power**
with 18 axons: an attractor of the simulated size comes out robust in 62 % of the sets, an inhibition zone in 28 %,
and a halo around a wide cluster almost never (the robustness rule cannot tell the halo from the cluster's flank: the
price of never reading a wide cluster as an attractor). A regular pattern always gets "inhibition zone" as well (the
valleys between modes look like an empty zone): that combined label is one finding, not two. With the fitted contour
on a narrow cloud (radial SD 40 nm, 20 sets per condition): flat 0 of 20 false readings, attractor robust in 13 of 20
(0.65; 0.43-0.82), the same as with the true arc; 20 sets only say that it does not get grossly worse.

**Power with few axons** (true arc; 300 sets per null, 200 per positive):

| axons per set | worst false reading (flat / clumps / t3 300 nm) | attractor robust | inhibition zone | regular |
|---|---|---|---|---|
| 18 | 0.026 | 0.62 (0.55-0.68) | 0.28 (0.22-0.34) | 1.00 |
| 11 (accepted contours of the pilot) | 0.017 | 0.42 (0.35-0.48) | 0.13 (0.09-0.18) | 1.00 |
| **4 (reliable contours of the pilot, D12)** | 0.027 | **0.24** (0.19-0.30) | **0.045** (0.02-0.08) | 1.00 |

Sizes 5 to 10 were simulated afterwards (500 sets per positive) to fix the minimum of a pooled reading: see the table
in section 6 (minimum 10). With 4 axons A still invents nothing, but an attractor of the simulated size is seen once
in four sets and an inhibition zone practically never. **"No structure detectable" on 4 axons says nothing about the absence of
structure.** The sensitivity of the readings to the zone rule (beyond the widening check built into the reading) is
not reported by this validation.

## 7. Question B: does the density follow the curvature?

**What question does it answer?**
Are the stretches of membrane that are more curved the ones with more (or fewer) localizations?

**Plain explanation.**
For every 200 nm window of the perimeter the program measures how straight it is (the straight-line distance between
its two ends divided by the arc between them: 1 is straight, less than 1 is curved) and how many localizations it
holds, and asks whether the two go together. Because tortuosity falls as a window curves more, *more density on more
convex or more curved stretches gives a negative Spearman correlation*.

**Example.**
A window that is a straight stretch has tortuosity 1.00; one on a bend of 600 nm radius has about 0.995. If the bends
systematically hold more localizations, the Spearman correlation is negative and, if it exceeds what rotating the
profile gives, significant.

**Technical** (`tools/mps_perimeter_bc.py`, group B).

- **Windows**: 2 contiguous bins (200 nm nominal, [spec]), step 1 bin, so there are as many windows as bins and the
  rotation has resolution (an 8 µm axon has 80 windows; with non-overlapping windows the smallest p would be 0.025).
  The window crossing the closure of the curve is handled modulo the length; tortuosity does not depend on where the
  curve's table starts.
- **Statistic**: Spearman correlation between tortuosity and the window's count (the sum of its bins), central slice.
  Windows with a bin not measured are excluded; windows with few localizations are not (a low count is the signal).
  At least 8 usable windows are needed ([ad hoc]).
- **Null** ("randomised version"): circular rotation of the density profile against the tortuosity profile. It keeps
  the autocorrelation of both (the window overlap included) and breaks only their alignment. Two-sided p
  (1 + #{|r_k| ≥ |r_0|}) / (1 + n) [lit: Phipson & Smyth 2010]; no direction is predicted.
- **What the null does not control (the contour-band bias).** Where the fitted contour cuts a corner it is shorter than
  the membrane there, the localizations bunch up per unit of contour arc and the band loses a different share of them:
  a real tortuosity-count association that is not biology. Controls: `pd_b_tau_vs_offset` (Spearman of tortuosity with
  the median radial offset per window; if it is not near zero the contour is biased where the membrane bends), the
  sensitivity to the band half-width, and simulation with a noisy contour and density independent of curvature.
- **Resolution limit.** For an arc of radius R, chord / arc = sin(x)/x with x = w / 2R, about 1 − (w/R)²/24 [derivation,
  checked numerically in the tests]; with w = 200 nm and R = 600 nm this is 0.9954. Tortuosity at 200 nm is therefore a
  reading of curvature at scales of 400 nm and above, not at 200 nm. `pd_b_tau_min` is the lowest window tortuosity and
  `pd_b_curvature_resolved` is true when the equivalent radius L / 2π is at least 600 nm (the program's resolved
  radius, `tools/mps_curvature.py`).
- **Tortuosity loses the sign** (convex or concave) because ranking by it is ranking by the mean squared curvature of the
  window. The signed mean curvature per window is exported in the profiles table (`kappa_mean_per_nm`).
- **Limited power on symmetric shapes**: a near circle or a smooth ellipse has almost no tortuosity range and the ranks
  are mostly ties; the test is then conservative and a null result is weak evidence of no relation.
- **B depends on the contour more than A or C.** The curvature is the contour's, so a stiff fallback curve (knots
  600 nm) has less tortuosity range than a primary one, and a curve that cuts a bay in a reliable contour can still
  pass the gate (section 4). Per-axon B is exported for every axon that finished; the pooled and per-animal summaries
  take only axons whose contour passed the quality gate, and are also given per curve stratum. With a noisy contour
  the false positives of B were 7-11 % in the first simulations (flagged by `pd_b_tau_vs_offset`).
- The convention is the same as the profile's: bin j is centred on origin + j·bin, so window i spans bins i and i + 1
  starting half a bin before the grid point. (A half-bin mismatch between the profile and the windows was found and
  fixed before integration; the cross-check is in the validation tests.)

**Validation** (`validate_perimeter_b.py`; 1999 rotations; criterion: false positives 0.03-0.07 with the true contour,
at most 0.08 per band with a fitted contour, power at least 0.8) [own sim]:

| case | n | measured | verdict |
|---|---|---|---|
| null, true contour (wavy shape, β = 0) | 500 | false positives 0.036 (0.023-0.056); p uniform (KS p 0.55); `tau_vs_offset` median +0.003 | passes |
| wavy shape, β = 0.5, true contour | 200 | power 0.900 (0.851-0.934); Spearman median −0.60 | passes |
| symmetric ellipse, β = 0.5, true contour | 200 | power **0.225** (0.173-0.288) | **fails** |
| null, fitted contour, narrow cloud (radial SD 20 nm), automatic curve | 300 | band 50 / 100 / 150 nm: 0.037 / 0.047 / 0.047; error of the curve 2 nm | passes |
| same, stiff curve forced / polar curve forced | 300 each | 0.037 / 0.063 / 0.063 and 0.040 / 0.043 / 0.043; `tau_vs_offset` median −0.20 with the stiff curve | passes (the stiff curve already shows the contour-band bias in the control, not yet in the false positives) |
| wavy shape, β = 0.5, automatic curve, narrow cloud | 200 | power 0.925 (0.880-0.954) | passes |
| null, automatic curve, real-width cloud (radial SD 100 nm, 15 % background, 10 % interior) | 40 | band 50 / 100 / 150 nm: 0.050 / 0.050 / **0.100** (Wilson 0.04-0.23); 39 of 40 curves fell back to the polar one, only 2 of 40 reliable | undecided: compatible with 0.05 and with twice that |

(Wilson 95 % intervals in brackets.) Reading: with the true contour B is calibrated and has power on asymmetric
shapes. On a shape with two-fold symmetry (an ellipse) rotating the profile reproduces almost the same
curvature-density relation and the power drops to 0.23 (an earlier run of the module had measured 0.60; the difference
was not investigated): **on symmetric shapes "no relation detectable" is weak evidence**. With a fitted contour on a
narrow cloud nothing changes. On a cloud as wide as the real one the size of the test with the automatic curve is
**not established** (40 axons; the reliable stratum is almost empty because the simulated cloud is wider than the
band), and in that stratum it was not measured at all. Until it is, a per-axon `pd_b_p` with a fitted contour is a
nominal level. Not run: a thin axon (R = 400 nm) where `pd_b_curvature_resolved` is false.

## 8. Question C: do neighbouring slices share a pattern?

**What question does it answer?**
Does the density profile of one slice resemble that of another slice of the same axon, in the common frame?

**Plain explanation.**
The three slices are 170 nm thick but the axial precision is about 47 nm, so a molecule near a slice boundary shows up in
the neighbour. That leak alone makes adjacent slices correlated even if their true patterns are independent. The pair
**−1 / +1**, which do not touch, is the clean reading; the neighbouring pairs are reported next to the correlation the
leak alone would produce.

**Example.**
If r(−1, 0) = 0.45 and the leak expected for that pair with this axon's slice fractions is 0.50, that is "leakage
only". If r(−1, +1) is positive and significant, that
is registry between slices.

**Technical.**
- Three pairs (−1, 0), (0, +1), (−1, +1), in the common frame `z0_max`, never aligned on each slice's own maximum.
  Pearson r at zero shift, the one-sided p (the hypothesis predicts r > 0) against the circular rotation null
  (`tools/mps_gaps.profile_correlation`, unchanged; all shifts are used when n_bins − 1 ≤ 1999), bins not measured in
  either profile masked, z = r in null standard deviations. Number of replicates 1999 and seed 0 [lit: Phipson & Smyth
  2010 style, as the rest of the program].
- **Floor of the p**: with all shifts used, the smallest p is 1 / (n_bins + 1) (for example about 0.01 for 100 bins). A
  p at the floor means "no rotation reached it", not p = 0.
- **Expected axial leak** (`leak_expected_r`) [derivation]. Assumptions: every molecule at a depth uniform inside its
  slice; Gaussian z error of sd lpz (47 nm by default, or the median `lpz` of the file); independent true patterns of
  equal mean and variance in different slices; noise-free counts (Poisson noise lowers the observed r, so this is the
  largest r the leak produces); only the nearest neighbour leaks. With a = fraction of a slice's molecules observed in
  its own core and e = fraction observed in one neighbour's core, an observed profile is a·P_k + e·(P_(k−1) + P_(k+1)), so
  r(k, k+1) = 2ae / (a² + 2e²). For 170 nm and 47 nm: a = 0.78, e = 0.11, r ≈ 0.27; the pair (−1, +1) shares only the
  central pattern and gets e² / (a² + 2e²) ≈ 0.02. **These two numbers assume three slices with the same number of
  localizations**, which the pilot does not have (laterals at 13-78 % of the central slice).
- **Expected leak with the axon's own slice fractions** (v20, `leak_expected_r_pairs`, columns
  `pd_c_leak_r_<pair>_sz47nm` and `_sz77nm`) [derivation]. Same assumptions, but each slice weighs what it holds:
  the observed localizations of the three slices are taken as the molecule masses, the leak to the slice after next is
  kept, and the expected r of every pair follows the simulator's formula (`mps_perimeter_sim.expected_slice_r`, ρ = 0;
  a test checks the two agree). A lateral slice that holds less than the central one receives, from the central slice,
  a larger share of its own profile, so the pair (−1, +1) gets far more than 0.02: on the pilot 0.05-0.36 with
  σ_z = 47 nm and 0.17-0.67 with 77 nm (median about 0.12 and 0.32). Inverting the leak to recover the masses was
  tried and rejected: under the uniform-depth model it calls a sparse lateral almost all leak (neighbour leak 1.00
  for the axon with the best contour), far from the observed neighbour r of about 0.3. Taking the observed fractions as masses can
  underestimate the leak when a lateral is mostly leak. Counts are noise-free, so these are upper bounds of what
  the leak produces with ideal counting.
- **Which axial width? Measured on the pilot, decided 2026-11-17 (v21).** A "NeNA in z" was measured on the pilot
  (85 729 pairs of z of the same molecule in consecutive frames, whose spread is √2 σ_z without structural width;
  validation report NENA_Z.md) [real data, pilot]. Frame to frame the equivalent σ_z for the leak is 36 nm; adding a
  slow error component (~27 nm) that is most probably real error, it is **47 nm** (median 47.0 across axons, 43-51).
  The user's decision: **47 nm is the main width** (it also equals the median `lpz` Picasso reports), the measured
  range **36-51 nm is its uncertainty** (36 = frame-to-frame only, 51 = the upper end across axons), and **77 nm**
  (the standard deviation of the rings' components in z of the mixture fitted to the pilot, 1.76 × lpz) is kept only
  as an **upper bound**: it also contains the physical thickness of the ring (~61 nm implied), and for the leak only
  the localization error counts (a physically thick ring crossing two slices is a true shared structure, not leak).
  Columns: `pd_c_leak_r_<pair>_sz47nm` (main) and `_sz77nm` (upper bound) since v20, `_sz36nm` and `_sz51nm` (the
  range) since v21. These are the pilot's values: **for new data σ_z is measured per file** (next bullet; v22) and
  the pilot's 47 nm is the fallback when a file cannot be measured. In simulation with laterals at 45 % and no misfit, the false positives of
  C(−1, +1) from the leak alone were 0.068-0.080 with 47 nm (0.15-0.26 with the 77 nm upper bound) [own sim,
  calibration study]: no slice criterion removes this.
- **σ_z measured in the file itself (v22; `tools.mps_axial_sigma`).** The leak depends on how wrong the z of a
  localization is, and Picasso's `lpz` does not say it (on the pilot the real error ranged from 0.45 × `lpz` close to
  the focus to 1.0-1.7 × `lpz` above z = +100 nm), so each file is measured with the method of the pilot study
  [real data, pilot; derivation]:
  1. *Frame to frame.* Localizations in consecutive frames closer than 25 nm in x, y, with no other partner within
     40 nm, are the same molecule; the difference of their z is only error (NeNA, Endesfelder et al. 2014, turned to
     z). The error is not Gaussian, so the difference is fitted as the mixture that follows from "each localization is
     good (sd s_c) with probability 1 − w and bad (sd s_b) with probability w", by maximum likelihood, with BIC against
     a single Gaussian. The leak is linear in the mean absolute error (uniform depth, error smaller than the slice),
     so the equivalent sd is (1 − w) s_c + w s_b, not a root-mean-square. This is `pd_sz_frame_nm` (36 nm on the
     pilot).
  2. *Slow component.* The same molecule switched on again 1001 frames or more later shows an extra error that
     consecutive frames cannot see (focus drift, a z bias that changes between switch-ons). Pairs closer than 10 nm are
     a mixture of the same molecule (narrow) and its neighbours (wide); the narrow component at lag ≥ 1001 against the
     one at lag 1 gives δ = sqrt(s_long² − s_1²), `pd_sz_slow_nm` (27 nm on the pilot). The total adds δ in
     quadrature to both components: `pd_sz_total_nm` = (1 − w) √(s_c² + δ²) + w √(s_b² + δ²) (47 nm on the pilot).
  3. *Intervals.* 95 %, resampling blocks of frames (or the groups the caller gives, when there are six or more) and
     refitting everything.
  When the measurement is reliable (`pd_sz_status` = "measured") its total is the main width and its interval the
  range, and `pd_sz_source` is "measured"; the 77 nm upper bound does not change. Otherwise the pilot's 47 nm
  (36-51 nm) is used, `pd_sz_source` is "pilot 47" and the reason is a warning and `pd_sz_note`. It is **not**
  measurable, and the pilot width is used, when: there is no `frame` column (nothing identifies the same molecule);
  there are too few pairs (fewer than 1500 unique frame pairs, 500 at lag 1 or 1500 at lag ≥ 1001; the frame error
  alone is then reported but not used, because it leaves out the slow component: 36 against 47 nm on the pilot); the
  acquisition is shorter than 2002 frames; the density is high; or the interval is wider than 60 % of the value. Two
  controls from the data catch a dense acquisition, in which localizations of different molecules would be taken for
  the same one and widen the pairs: the share of localizations with another one in the SAME frame within 300 nm
  (`pd_sz_same_frame_fraction`; warning above 1 %, no measurement above 5 %) and the pairs of consecutive frames at
  100-300 nm scaled to the 25 nm disc along a line, the pessimistic scale (`pd_sz_contamination`; warning above 10 %,
  no measurement above 30 %). `pd_c_m1_p1_vs_leak` and `pd_c_reading` are read against the width used, and the
  numbers inside their labels are that width's; `pd_c_leak_r_<pair>_szused` (and `_lo`, `_hi`) hold the leak at it,
  while the `_sz47nm`/`_sz77nm`/`_sz36nm`/`_sz51nm` columns stay at the pilot's fixed widths for comparison. What the
  measurement does not see: a fixed bias per molecule (dipole orientation, a local aberration that does not change
  between switch-ons), and whether the slow component is error or neighbours of the same ring at < 10 nm (36 nm if it
  is contamination, 47 nm if it is error; fixed beads or fiducials would settle it). The thresholds (pair counts,
  density controls, interval width) are [ad hoc]; the radii and lag are [real data, pilot]; `lp` and `lpz` are
  reported for comparison and never enter the estimate. Tested on simulated files of known error
  (`test_axial_sigma.py`).
- **r(−1, +1) against its own leak** (`pd_c_m1_p1_vs_leak`, v21 vocabulary): "above the leak (sigma_z 47 nm, range
  36-51 nm) and above its upper bound (77 nm)"; "above the leak (sigma_z 47 nm, range 36-51 nm); within the upper
  bound (77 nm)"; "above the leak at sigma_z 47 nm, within its range (not above 51 nm)"; "at the limit: above the
  leak at sigma_z 36 nm, not at 47 nm"; "not above the leak (not even at sigma_z 36 nm)". A "registry between slices"
  reading gets " (r(−1, +1) above the axial leak expected with sigma_z 47 nm but within its measured range 36-51 nm)",
  " (r(−1, +1) at the limit: above the axial leak expected with sigma_z 36 nm, not with 47 nm)" or " (r(−1, +1) not
  above the axial leak expected even with sigma_z 36 nm)"; above 51 nm it stays bare (the 77 nm upper bound adds no
  suffix). The comparison is not a test: the leak is a noise-free upper bound and the observed r is attenuated by counting
  noise, so "above the leak" is the conservative side.
- **Axial guard** (sensitivity, 42.5 nm): keeping only the 85 nm core of each slice brings the expected neighbour leak to
  about **0.12**. The first design asked for less than 0.10; the calculation does not give that (two independent
  implementations, the one in `mps_perimeter_bc` and the simulator's, agree), so the guard reduces the leak by about a
  half but does not remove it. This deviation from the plan is documented, not hidden: the validation measured the
  same value (theory 0.122 in both implementations; 0.109 ± 0.014 in simulation without blinking), so the criterion
  "below 0.10" **fails** and was not relaxed. With blinking the measured leak is lower (0.093 ± 0.011 with the guard,
  0.25 without; 0.070 with the guard in the leak-alone test of question C), so 0.27 and 0.12 are upper bounds, not the
  values expected in data with blinking. Whether the plan's criterion is corrected is left for decision.
- **Reading** (`pd_c_reading`, `mps_perimeter_bc.slice_reading_real_fractions`; rule and 0.10 tolerance [ad hoc]):
  "registry between slices" (r(−1, +1) positive and significant); "leakage only" (every significant neighbour pair
  with r within 0.10 of **its own expected leak with the axon's slice fractions** or below, and r(−1, +1) not
  significant); "neighbours above leakage" (a significant neighbour pair more than 0.10 above its own leak:
  suggestive, not conclusive); "no correlation between slices"; "not computable" when a pair has an empty or constant
  profile. Since 2026-11-17 (user's decision) the neighbours are compared with `pd_c_leak_r_m1_0_*` and
  `pd_c_leak_r_0_p1_*`, not with the leak of three equal slices (`pd_c_leak_expected_r`, 0.27 at 47 nm, still
  reported): with the pilot's fractions the neighbour leak is 0.29-0.76 at 47 nm (11 finished
  axons), so the old comparison called the leak itself "neighbours above leakage" (3 axons change, section 13). The reading uses the **main width (47 nm)** and is repeated at the ends of
  the measured range (36 and 51 nm): when an end gives another reading the label is the 47 nm reading followed by
  " (within the measured range of sigma_z 36-51 nm: <reading> at 36 nm; <reading> at 51 nm)" (only the ends that
  differ are listed). The 77 nm upper bound does not enter the reading. The leak is a noise-free upper bound that may
  be underestimated when a lateral is mostly leak (above).
- **When the per-axon reading cannot be read as it stands** (`tools/mps_perimeter_pool.py`, `axon_c_reading`). The
  rule above looks only at the three correlations. The cell of `pd_c_reading` adds the reason when the axon's contour
  or slices do not support it; the numbers (r, z, p) do not change:
  - " (contour not reliable: not interpretable)" (or " (contour not assessed: not interpretable)") when the quality
    gate leaves the axon out of the pooled analyses (`pd_pool_included` false). It is added to every computable
    reading, "no correlation between slices" included, because a contour error both creates a common correlation and
    blurs a real one;
  - " (slices do not coincide: outside the main reading of C, descriptive)" when the contour passed the gate but the
    slices do not coincide by the calibrated criterion below (`pd_c_slices_coincide` false, which includes "not
    measurable"). It is added to every computable reading. (Before v20 a " (slice misfit: ...)" suffix followed the
    uncalibrated 50 nm mark; the criterion replaces it for C, because one shifted lateral, which the mark flags, does
    not inflate C);
  - the three leak suffixes above, on "registry between slices" only.
  Only a bare label (no parenthesis) is a reading of an axon in the main reading of C whose r(−1, +1), if it claims
  registry, is above the leak for the whole measured range of σ_z (36-51 nm).
- **One contour for the three slices.** A local contour error lowers the counts in the same bins in all three slices
  and creates a common positive correlation, r(−1, +1) included, which is the signature used to tell registry from
  leakage. **This risk is real and large, and the quality gate stops it** (`validate_perimeter_c.py`, common-error
  test, [own sim]): the true contour was deformed by smooth Gaussian bumps (one every ~3 µm, random sign, sd 300 nm, so
  about 700 nm wide) of 0 to 200 nm amplitude, on independent slices with no leak, so every correlation is false; 300
  axons per row, 1999 rotations.

  | radial SD of the cloud | bump | false positives r(−1, +1), all curves | mean r(−1, +1) | curves passing the gate | false positives among those |
  |---|---|---|---|---|---|
  | 20 nm | 0 nm | 0.030 | +0.002 | 300 / 300 | 0.030 |
  | 20 nm | 50 nm | 0.047 | +0.010 | 224 / 300 | 0.031 |
  | 20 nm | 75 nm | **0.107** | +0.041 | 177 / 300 | 0.040 |
  | 20 nm | 100 nm | **0.273** | +0.098 | 0 / 300 | — |
  | 20 nm | 150 nm | **0.767** | +0.272 | 0 / 300 | — |
  | 20 nm | 200 nm | **0.370** | +0.150 | 0 / 300 | — |
  | 73 nm (real median), 15 % background, 10 % interior | 0 nm | 0.057 | +0.008 | 300 / 300 | 0.057 |
  | 73 nm | 50 nm | 0.067 | +0.011 | 236 / 300 | 0.059 |
  | 73 nm | 75 nm | **0.103** | +0.034 | 88 / 300 | 0.045 |
  | 73 nm | 100 nm | **0.187** | +0.071 | 16 / 300 | 1 of 16 |
  | 73 nm | 150 nm | **0.473** | +0.175 | 2 / 300 | 0 of 2 |
  | 73 nm | 200 nm | **0.783** | +0.274 | 2 / 300 | 1 of 2 |

  A common contour error of 75 nm amplitude already doubles the false positives of C(−1, +1); 150 nm takes them to
  0.47-0.77. The false correlation (r +0.1 to +0.3) is of the same order as the "high" r(−1, +1) of the pilot. Among the
  curves that pass the gate the false positives stay at 0.03-0.06 wherever n is large enough. (The drop at 200 nm on the
  narrow cloud is not a protection: the shifted stretches leave the band altogether.) Limits: only smooth bumps were
  tested; a long chord (a bay smoothed over by the stiff curve) and an error that differs between slices are not
  covered, and the gate was calibrated without z. Exported controls: the radial shift of each lateral slice relative to the central one
  (`pd_radial_shift_m1_nm`, `pd_radial_shift_p1_nm`) and the misfit between the slices measured by the contour's
  quality gate (`pd_contour_slice_rms_minus_nm`, `pd_contour_slice_rms_plus_nm`, `pd_contour_slice_radius_change_nm`,
  mark `pd_contour_slice_misfit` above 50 nm RMS). **For C the misfit matters directly**: a lateral displacement of
  the rings of slices −1 and +1 from the contour (a tilted axon, a lateral shift of the optics with z) removes
  localizations from the band at the same angles in both slices, whatever its sign. This alone produces a positive
  r(−1, +1) between independent slices, which is why C has its own slice criterion (below). The misfit mark does
  not exclude the axon from A and B (D12). On the exploratory
  pilot the misfit marks 7 of the 11 accepted contours (larger lateral-slice RMS 67-146 nm; 27-46 nm in the three
  reliable ones). An independent check by the review (one circle, 400 axons per row, perfect central contour,
  independent slices, no leak) [own sim, review]: lateral shifts of the slices ±1 with misfit RMS 35 / 50 / 71 / 99 nm
  gave a mean r(−1, +1) of +0.01 / +0.07 / +0.23 / +0.49 and false positives 0.045 / 0.105 / 0.152 / 0.215; a pure
  change of radius gave no correlation. The value of r grows much faster than the p: the r and z of a misfit axon are
  inflated even when its p is not significant.
- **Reading rule for C** (what may be read and where):
  1. the pair −1/+1 is the clean reading; the neighbouring pairs measure mostly the axial leak and are not read as
     registry;
  2. **C is read only on reliable contours** (`pd_pool_included` true). On the others a common contour error of 75 nm or
     more gives false registry in 10-78 % of the axons instead of 5 %, and the per-axon cell says "not interpretable";
  3. **for the main reading of C, only on axons whose slices coincide well** (decision P-R3-1;
     `mps_perimeter_bc.slices_coincide_for_c`, column `pd_c_slices_coincide`; `pd_c_main_reading` = reliable contour
     AND coinciding slices). **Criterion: the slices coincide when the better-fitting lateral slice is at most 32 nm
     RMS from the central ring, `min(pd_contour_slice_rms_minus_nm, pd_contour_slice_rms_plus_nm) <= 32`; when either
     misfit could not be measured, they do not coincide.** Where 32 nm comes from [own sim; Bradley 1978 robustness
     band 7.5 %]: about 27 500 simulated axons with no registry, six geometries of misfit (both laterals tilted to
     opposite sides, both to the same side, a change of radius, radius plus tilt, one lateral only, one at half the
     other), lateral densities of 8 / 20 / 50 localizations per bin and radial SD 40 / 73 nm, with the misfit measured
     by the program's own slice-misfit function. For each stratum the local false-positive rate of C(−1, +1) was fitted
     as a monotone (isotonic) function of the statistic; the threshold is the largest misfit at which it stays inside
     Bradley's liberal robustness band (at most 1.5 α = 0.075; Bradley 1978), taken as the 5th percentile of 2000
     bootstrap crossings in the worst stratum (both laterals to the same side, 50 per bin: point 33.6 nm, 5th
     percentile 32.0 nm). Among the axons that pass, the false positives were 0.011-0.064 in 29 of 30 cells (the
     other 0.089 with n = 79, 0.044-0.172), and the power with identical patterns was 0.68 against 0.43 among those
     that fail: what is lost is axons, not sensitivity per axon. **Why the smaller of the two misfits**: the spurious
     correlation needs both laterals to lose band at the same angles; with one lateral shifted up to 90 nm RMS the
     false positives never left 0.075, and a change of radius alone (up to 120 nm) neither, but radius combined with a
     lateral shift does inflate C, so the lateral part alone is not a safe statistic (its threshold falls to 18 nm).
     **The threshold falls with density** (63 → 56 → 34 nm from 8 to 50 localizations per bin in the worst
     geometry): it was calibrated with lateral densities up to 50 per bin (the pilot has 7 to about 45); **with
     denser data it must be recalibrated** by re-running the calibration simulation (`crit_sim.py`, about 10 minutes)
     at the new density. Axons between 32 and about 45 nm are in the uncertain zone of the bootstrap. One cloud model
     only (Gaussian radial + background + interior). The axons that do not pass are reported apart, descriptive:
     per axon by the suffix of `pd_c_reading`, per animal by the `scope` "reliable contours with coinciding slices
     (main reading of C)", the only scope where `pd_c_z_m1_p1` gets a p (A and B get theirs only in the other scopes;
     with two or more animals the Holm adjustment runs over the family of three across the two scopes). The rule
     for A and B does not change (D12). On the pilot the main reading of C holds 3 axons;
     a fourth (in the main set of A and B by D12) does not pass (63 / 67 nm);
  4. a per-axon "no correlation between slices" does not exclude a weak registry (power below).
- A lateral slice with zero or three localizations is kept: its pairs are NaN (empty or constant profile) and the reading
  is "not computable", with no exception.

**Validation** (`validate_perimeter_c.py`, true contour; circular axons of 8-25 µm perimeter; 1999 rotations;
criterion: false positives 0.03-0.07, power at least 0.8) [own sim]:

| case | n | (−1, 0) | (0, +1) | (−1, +1) | verdict |
|---|---|---|---|---|---|
| null: independent slices, no leak (false positives) | 500 | 0.046 (0.031-0.068) | 0.050 (0.034-0.073) | 0.056 (0.039-0.080) | passes; p uniform (KS p 0.22 / 0.53 / 0.75) |
| identical patterns (power) | 200 | — | — | 0.980 (0.950-0.992); median r 0.42 | passes |
| dose-response, true correlation 0.2 / 0.3 / 0.5 | 200 each | — | — | 0.26 / 0.38 / 0.73 (r 0.09 / 0.12 / 0.22) | informative: **little power for a weak registry** |
| leak alone: neighbour r | 200 | mean r 0.203 | significant 0.675 / 0.700 | — | **fails** the planned 0.27 ± 0.05 (blinking lowers the leak to 0.20) |
| leak alone: (−1, +1) must stay silent | 200 | — | — | 0.060 (0.035-0.102) | passes |
| leak alone: reading "leakage only" | 200 | "leakage only" 0.775; "registry" 0.060 | | | **fails narrowly** (criterion 0.8) |
| guard 42.5 nm: neighbour r, with blinking | 200 | 0.070 | | | passes (0.12 without blinking) |

With the fitted contour (`validate_perimeter_b.py fitted`): on a narrow cloud (radial SD 20 nm, 300 nulls) the false
positives of (−1, 0) / (0, +1) / (−1, +1) were 0.040 / 0.037 / 0.047 with the automatic curve (0.033-0.047 with the
stiff or polar curve forced), and the power with identical patterns 0.995 (r 0.42), the same as with the true contour.
On a real-width cloud (radial SD 100 nm, 15 % background, 10 % interior; 40 nulls, automatic curve, polar fallback in
39) the false positives of (−1, +1) were 3 of 40 (0.075; Wilson 0.03-0.20), and the power with identical patterns 8 of
10 with the true and with the automatic curve (Wilson 0.49-0.94): an indication, not a measurement. In that run the
only axon with a primary curve had an error of 39 nm and gave r(−1, +1) = 0.55, p = 0.006 with independent slices: a
concrete case of the common-error risk above.

Reading: **the pair (−1, +1) is calibrated under the null, is barely contaminated by the leak with equal slices (6 %;
with laterals at 45 % of the central slice it is 0.07-0.08 at 47 nm and 0.15-0.26 at 77 nm, see above), and has high power
when the slices share the pattern, but little when they share it weakly** (0.26 at a true correlation of 0.2). The
neighbouring pairs are dominated by the leak. In this validation the "leakage only" label compared against 0.27, the
leak of equal slices without blinking; with blinking the leak is about 0.20 and the label was right in 78 % of the
cases. This changes no decision about (−1, +1). Since 2026-11-17 the program's label compares each neighbour pair with
its own leak from the axon's slice fractions (section 8, Reading), still noise-free and without blinking, so the leak
remains an upper bound and the label leans to "leakage only".

## 9. Identity, animals and groups

**What question does it answer?**
Which axon is this, and what may be compared with what?

**Plain explanation.**
Each axon gets a readable label `year.ROI.axon` (for example `26.1.2`). The label is for reading only: the key stays the
program's `axon_id`. Statistics between groups use the *animal* as the unit, never the axon: axons of one animal are not
independent replicates.

**Example.**
Two acquisitions of the same year with the same ROI number share a label; the batch warns about duplicate labels and
exports the acquisition date to tell them apart.

**Technical.**
- **Label** (`tools/mps_trace_id.py`): year = two digits of the acquisition date at the start of the file name (`yy.m.d_`
  or `yy.mm.dd_`; a date in a folder name is not read; a typed year wins), ROI and axon = the digit runs of the
  identity's ROI and axon names (leading zeros dropped). A part that cannot be read is `NA`, never guessed, with
  `trace_complete = false` and a warning that says what is missing (`trace_warnings`). The year pattern is [ad hoc], taken
  from the file names of the exploratory pilot.
- **Unit of inference between groups: the animal.** Per-animal means of the signed z readouts (`pd_a_contrast_z`,
  `pd_a_period_z`, `pd_b_z`, `pd_c_z_m1_p1`, `pd_c_z_m1_0`, `pd_c_z_0_p1`) via `tools/mps_gap_batch.py`; a comparison between
  genotypes needs at least 3 animals per group, otherwise it is descriptive only (rule of the program's regularity
  plan).
- **The p of the `by_animal` table is not a randomization test.** For each animal, `p_<readout>` is a one-sample
  t test of its axons' z against 0 (`mps_gap_batch.aggregate_by_animal`), Holm-adjusted across the tested readouts
  (`p_holm_<readout>`). It treats the axons of one animal as replicates, so it says nothing about other animals and
  does not generalise. It is computed only for `pd_a_contrast_z`, `pd_b_z` and `pd_c_z_m1_p1` (the Holm family is
  these three). `pd_c_z_m1_0` and `pd_c_z_0_p1` are averaged but never tested, because axial leakage alone makes the
  neighbouring pairs positive (section 8) and a p would measure the leak; `pd_a_period_z` is averaged but never
  tested, because its per-axon null is skewed (section 6). With one animal, or with the provisional label, every p of
  the row is empty and the note says "one animal: no p" (or "provisional animal: no p"); the means, SDs and counts
  stay.
- **Rows without an animal** go to a group labelled `SIN ANIMAL`, with descriptive numbers per ROI and no p between
  genotypes.
- **One animal of unknown label** (the batch option `--single-animal`) is given the explicit placeholder
  `provisional-single-animal`; it is not an animal's name. With n_animals = 1 no comparison between animals is possible and
  every output says so.
- **What follows with one animal or none**: every p of this analysis is then a within-axon (per axon) or
  within-sample (pooled over axons) randomization test, since the per-animal t test is left empty; none supports a
  statement about animals or genotypes. With two or more animals the per-animal t test above appears; it is the only
  p of the analysis that is not a randomization test, and it is still a statement within each animal.
- **Peak minus focus per ROI**: axons of one field share one focus, so the summary of the distances is given per ROI (and
  over all axons) in the `peak_focus` table, not as independent axons.

## 10. Failure states

Every failure produces a complete row with a status and a one-sentence reason (`pd_status`, `pd_failure`); nothing raises.

| status | meaning |
|---|---|
| `ok` | the analysis ran |
| `not run` | the block of the table is empty because the analysis was not run |
| `no focus` | a focus criterion was asked and that focus is not available; it never falls back to the peak |
| `too few localizations` | fewer than 100 finite localizations, or no peak in z |
| `contour failed` | the automatic contour was rejected (reason in `pd_failure`, section 4; includes fewer than 200 localizations, which the contour needs) |
| `empty profile` | the central slice holds no localization, or none within the band |
| `invalid input` | an unreadable file or arrays of different lengths |
| `internal error` | an unexpected failure, caught and reported with its text |

## 11. Export (axon tables v17 to v21), the windows, and how to reproduce

- **Per-axon table, v17.** A block at the very end of the row, after the six columns of v16; no earlier column moves.
  The block is empty with `pd_status = "not run"` when the analysis was not run, and keeps every column (empty where a
  value does not exist) when it failed. The columns are `trace_label`, `trace_year`, `trace_complete`,
  `acquisition_date`, `trace_warnings` and the `pd_*` block: status; z0 and both foci with the signed and absolute
  peak−focus distances; the slices (counts, percentages, coverage, truncation, lpz); the contour (method, status, knots,
  length, residual SD, crossings, distance to the polar curve, unsupported fraction, star-shape, nodes, reversal, radial
  shift of the lateral slices); the band; the grid; questions A, B and C; the sensitivity flag, the number of replicates,
  the seed and the warnings. Every column has an entry in the column dictionary (`tools/column_dictionary.py`).
- **Per-axon table, v19.** v18 (the unanchored-cluster discard) follows v17; v19 is a second perimeter block at the very
  end, after v18: `trace_roi`; the contour's assessment (`pd_contour_kind`, `pd_curve_stratum`, `pd_contour_reliable`,
  `pd_contour_failure_reason`, `pd_contour_quality_reasons`, `pd_contour_gate_reasons`, the fit numbers of the gate, the
  support, the slice misfit and its mark, the trim); the pooling gate (`pd_pool_included`,
  `pd_pool_included_without_misfit`, `pd_pool_exclusion_reason`); and the per-axon reading `pd_a_reading`. No earlier
  column moved.
- **Per-axon table, v20.** A third perimeter block after v19, the very last: `pd_c_slices_coincide`,
  `pd_c_main_reading`, the leak expected for the three pairs with the axon's own fractions at 47 and 77 nm
  (`pd_c_leak_r_m1_0_sz47nm` ... `pd_c_leak_r_m1_p1_sz77nm`) and `pd_c_m1_p1_vs_leak`. No earlier column moved. A
  failed axon carries every column of the three blocks, empty where nothing was computed (`pd_a_*`, `pd_b_*`,
  `pd_c_*` included). `gate` gains `pd_c_slices_coincide` and `pd_c_main_reading`, `gate_summary` the scope of C and
  `n_excluded_slices_not_coinciding`.
- **Per-axon table, v21.** A fourth perimeter block after v20, the very last: the leak of the three pairs at the
  ends of the measured range of σ_z, `pd_c_leak_r_m1_0_sz36nm` ... `pd_c_leak_r_m1_p1_sz51nm` (NeNA in z, decision of
  2026-11-17: 47 nm main, 36-51 nm range, 77 nm upper bound). No earlier column moved. The table goes from 656 to
  **662 columns**; the 47 and 77 nm columns of v20 stay (the 77 nm one is labelled "upper bound"), and every new column
  has its entry, with its origin, in `tools/column_dictionary.py` (the columns of v18, the unanchored-cluster block, are
  described in [unanchored.md](unanchored.md)).
- **Per-axon table, v22.** A fifth perimeter block after v21, the very last (`pd_sz_source` ... `pd_c_leak_r_m1_p1_szused_hi`):
  where the width of the leak of C came from ("measured" or "pilot 47"), the width used with its range, the
  frame-to-frame, slow and total σ_z with their intervals, the pairs and density controls they rest on, a note, and the
  leak of the three pairs at the width used (main, low and high end). No earlier column moved. The table goes from 662 to **691 columns** (29 new, all at the end; the v21 `_sz36nm`, `_sz47nm`,
  `_sz51nm` and `_sz77nm` columns stay as the pilot reference).

**The "Perimeter..." and "Unanchored..." windows.** Two buttons of the MPS analysis window open these windows on the
axon it shows; neither computes anything of its own (the analysis lives in `tools/mps_perimeter_analysis.py` and
`tools/mps_unanchored.py`). Both run in the background with a real progress bar and a Cancel that stops at the next step
(`tools/mps_cancel.check`); a result that arrives after Cancel is not used. The windows open with the defaults every time.

*"Perimeter..."* (`tools/mps_perimeter_window.py`). Bar: **z0 from** (peak of the z density curve, the default; focus,
nominal; focus, empirical), **Band ± (nm)** (default 100, range 25-300), **Contour** (Automatic, or Drawn: the trace
drawn on that axon), **Run** (analyse the axon shown), **Cancel** (enabled only during a run; nothing the run computed is
shown), **Save...** (one save dialog; the name chosen is the name used; if companion tables with that name exist it asks
before replacing). Three tabs. *Slices and contour*: the cloud of the three slices with the contour and its band, the z
density curve with the peak, both foci and the three slices, a table with the n and the percentage of each lateral slice
against the central one (100 %), and one line with the state of the contour, read from the quality gate of the contour
module and not judged by the window ("Contour reliable" or "Contour NOT reliable" with the reasons; an unreliable
contour is drawn red and dashed). *Profile: A and C* and *Curvature: B*: for each question a table, a one-line
descriptive reading and grey warnings (A is exploratory and its pooled reading comes from the pooled profile; C: the
pair z−1/z+1 is the clean one, the expected axial leak and the unmeasured risk of a common contour error; B: a negative
sign means denser where more curved, and the tortuosity-versus-radial-displacement control). The Batch button of the
analysis window has a "Perimeter density" check box that runs the analysis in its thread; the tables are written at
Export beside the chosen table, named `<name>_perimeter_*.csv` (`_run2` if they exist; never overwritten). **What is
exported** by Save...: the axon's row (`<name>.csv`) and the long tables (`<name>_perimeter_*.csv`), each with its
`_columns.csv` dictionary.

*"Unanchored..."* (`tools/mps_unanchored_window.py`). Bar: **method** (Automatic, own ring; Widefield images, only when
the axoplasm panel has flags; Off, by hand only), **reference ring** (polar ring of the localizations, the default;
smooth ring of the centres), **Exclude beyond (nm)** (400, provisional), **Doubtful from (nm)** (250), **Apply**
(analyse the axon again without the excluded clusters; both columns are kept), **Cancel**, **Load...** (decisions by
hand saved under another name), **Save...** (the decisions; beside the data file as `<data>_unanchored.json` they load
by themselves; another name is respected but only the default name is read automatically). The map draws the slab's
localizations, the reference ring (dashed), and the cluster centres: red cross = excluded, orange ring = doubtful,
open square = restored by hand. A click on a centre excludes it (a reason is asked) or restores it; a click on a
cluster that already has a decision takes it back. One line says what the discard would do ("would exclude N of K; M
doubtful") with the rule's warnings, and a table records every decision (cluster, x, y, depth, automatic state,
decision, reason, date) with **Undo** and **Back to automatic**; decisions that find no cluster within 50 nm are listed
as orphans. Nothing changes in the numbers until Apply. The program keeps one hidden instance that, the first time an
analysis is shown, runs the rule and the analysis without the excluded clusters, so the results window always shows
**with and without the discard** and the exports carry the v18 block. **Exported:** the axon row and the Batch carry the
two columns and the v18 block; the per-cluster table carries `depth_below_ring_nm`, `anchor_state`, `anchor_borderline`,
`anchor_source` and `anchor_reason`. If the perimeter was computed "one start" (a long perimeter), the column with
discard is not computed (it would rebuild the contour from every start, the cost that choice avoids); the v18 block is
still written and says so. With the Widefield method chosen and the axoplasm panel open, the panel's flags rule in the
export (a known limitation of the agreement check). The long explanation of every number is in the guided tours
("perimeter" and "unanchored"), which are bilingual; the windows themselves are English only for now.
- **Long tables** next to the per-axon table, `<stem>_perimeter_<name>.csv`, each with a `_columns.csv` dictionary:
  `profiles` (axon x slice x bin: lag, count, relative density, measured, tortuosity, signed mean curvature, radial
  medians), `pooled` (the pooled A profile per group with its bands and reading), `by_animal`, `contour` (x, y and arc of
  the contour used, to reproduce and inspect it), `sensitivity` (band and axial guard), `peak_focus` (per ROI), `gate`
  (per axon: pooled or not, with and without the slice misfit, and why not) and `gate_summary` (per pooled set: how
  many axons enter and why the others do not). `pooled` and `by_animal` hold every pooled set (`scope`). The
  localization table gets one column, `pd_slice` (−1, 0, +1 or empty), so every localization is traceable to its slice.
- **Never overwrites.** A table is created exclusively; when a table of the stem exists the stem gets `_run2`, `_run3`, …
- **Command line** (`batch_perimeter.py`; one file at a time, one process):

```bash
python batch_perimeter.py FOLDER --out STEM [--pattern axon] [--z0 peak|focus_nominal|focus_empirical]
    [--band 100] [--contour density_nodes|polar] [--single-animal | --animal LABEL] [--year 26]
    [--n-null 1999] [--seed 0] [--trim-interior NM]
```

  `--trim-interior` is off unless given (section 4, decision D8). It also writes the `axons`, `genotypes` (per pooled
  set) and `log` tables (one row per file with status, reason and duplicate labels).
  The same seed gives the same p. The parameters are not saved between sessions.

## 12. Limitations, and what cannot be concluded

- **Only the analysis of a thin transverse cut.** The three slices are not three independent samples of the axon: the
  neighbouring ones leak into each other (section 8).
- **The contour is not verified against the membrane** on real data; a fallback contour is a different estimator from the
  primary one. The fit-quality gate (section 4) leaves out of the pooled analyses the contours that do not hold the
  cloud; on the exploratory pilot it keeps **only 3 of 18** (4 counting the one marked only for slice misfit), so the
  pooled readings of that pilot rest on three or four axons of one animal. The gate's limits come from simulation and
  a derivation; whether real contours are wrong where the gate says so cannot be verified without ground truth.
- **The band is a pilot choice.** The real radial spread of the cloud is wider than the simulated one; the 50 and 150 nm
  sensitivity is mandatory because A, B or C can change with the width. If the decision changes, the axon carries the
  flag and the conclusion does not stand.
- **A is a within-sample test** and its "attractor" reading is only the excess beyond the cluster's edge as the zone rule
  defines it; it must survive widening the zone.
- **B reads curvature at scales of 400 nm and above**, with a contour-band bias the null does not remove and little power on
  symmetric shapes.
- **C cannot tell registry from leakage on neighbouring slices**; the pair −1/+1 can, if the contour is right and the
  slices coincide. A common contour error of 75 nm or more, or a lateral shift of the slices ±1, produces false
  registry; C is therefore read only on reliable contours (section 8), and its power for a weak registry is low.
- **What 3-4 reliable axons of one animal allow.** Describing those axons, with their per-axon rotation p values and
  the label "exploratory, one animal". Not: any p between animals or genotypes (n_animals = 1; nor a p "of the pilot"
  as if axons were animals); any generalisation ("the rings are in registry", "they are not", "there is no
  attractor"). With 4 axons the power of pooled A is 0.05-0.24 (section 6): **"not detected" is not "absent"**. One
  significant axon in four is compatible with chance (about 0.2 false positives expected from 4 tests at 0.05) and
  with a real effect; the two cannot be told apart.
- **One animal, or none, supports no comparison between animals or genotypes.** The exploratory pilot (18 axons of one
  experiment) is a training set: parameters chosen by looking at it must be frozen with a date before new data.
- **Not verified**: why the empirical focus lies 43 nm from the nominal one in the pilot; which z filter cut the files of one
  ROI at about −190 nm; which of two automatic contours is closer to the membrane; why the real radial spread is 2-3 times
  the simulated one.
- A statement such as "the rings are aligned" or "the large cluster attracts" is not supported by this analysis alone.

## 13. Validation (contour and questions A, B, C: measured in simulation; pilot: exploratory)

The validation uses the program's own simulator (`tools/mps_perimeter_sim.py`: circle, ellipse, wavy, non-star-shaped
"C" and two-lobed shapes; flat, attractor, inhibition, regular and curvature-modulated intensities; identical,
independent and correlated slices; axial leak; blinking; a 2 µm gap; synthetic sx/sy with a focus at a given z). The five
stages and the criteria they were planned with are below. Stage V1 (the contour) was measured by the report of the
automatic contour (version 2); stages V2-V5 and the cross-checks by the validation run P11 (scripts
`validate_perimeter_cross.py`, `validate_perimeter_a.py`, `_b.py`, `_c.py` and the pilot script; fast tests in
`test_perimeter_validation.py`). The tables are in sections 6 (A), 7 (B) and 8 (C).

**Summary.** (1) **None of the three tests invents signals**: with the true contour C gives 4.6-5.6 % false
positives, B 3.6 %, pooled A at most 2.6 % under eight different nulls; with the automatic curve on a narrow cloud the
same (B 4.7 %, C 4.7 %, A 0 of 20). (2) **Power**: C(−1, +1) 0.98 with identical patterns, but 0.26-0.73 for a true
correlation of 0.2-0.5; B 0.90 on asymmetric shapes, **0.23 on a symmetric ellipse**; pooled A over 18 axons:
attractor 0.62, inhibition zone 0.28, regular 1.00, halo around a wide cluster 0.01; **with 4 axons: attractor 0.24,
inhibition zone 0.045**. (3) **A contour error common to the three slices** doubles the false positives of C(−1, +1) at
75 nm and takes them to 0.47-0.77 at 150 nm; the quality gate brings them back to 0.03-0.06 (section 8). (4) The
axial guard leaves a leak of 0.12, not below 0.10 as planned (section 8). (5) On a cloud as wide as the real one,
40 nulls and 10 positives are too few to fix the size of B and C with the automatic curve (B 0.10 at ±150 nm, C 0.075,
both compatible with 0.05 and with twice that).

**Cross-checks between packages** [own sim]:

| check | criterion | measured | verdict |
|---|---|---|---|
| bins of the profile vs the arc bins and windows of B, edges included | 0 mismatches | 40 perimeters: 0 | passes |
| expected leak (module) vs simulator, no guard | difference at most 0.03 | 0.272 vs 0.275; pair (−1, +1) 0.020 | passes |
| same with the 42.5 nm guard | same | 0.122 vs 0.122; pair (−1, +1) 0.004 | passes |
| leak measured in simulation without blinking, no guard / guard | within 0.03 of the module | 0.247 ± 0.014 / 0.109 ± 0.014 (60 axons x 60 000 localizations) | passes |
| leak measured with blinking, no guard / guard | informative | 0.252 ± 0.012 / 0.093 ± 0.011 | info |
| plan: neighbour r below 0.10 with the 42.5 nm guard | below 0.10 | 0.122 by calculation | **fails** (a deviation of the plan, documented, not relaxed) |
| half-bin convention, simulator truth vs profile | χ²/df about 1 with centred bins | median 1.04 centred vs 100.5 at cell starts | passes |
| slice −1 with 0 or 3 localizations, end to end | no exception; its pairs NaN | ok; reading "not computable" | passes |
| contour clockwise vs counter-clockwise, same start | A, B, C identical | largest difference 0 | passes |
| clockwise contour starting one table step later | informative | largest difference 0.033 (binning moves 2 nm) | info |

Only the 2 µm gap of V1 was re-run on the full path with the current contour: 8 of 10 contours close (2 rejected) and
the unsupported fraction has a median of 0.095 (0 to 0.141; expected 0.08-0.16): **fails**, as with the earlier
contour; in one of the 8 the gap is not marked at all.

**V1, the automatic contour on its full path** (10 seeds x 8 shapes per condition; "reliable" = passes the quality
gate; error = 95th percentile of the distance to the true curve, median over the reliable curves; "bad reliable" =
reliable curves whose error over the data exceeds 170 nm) [own sim]:

| condition | accepted | reliable | primary / stiff / polar | error of reliable (median; 95th pct) | bad reliable |
|---|---|---|---|---|---|
| base, 15 % background | 79 / 80 | 76 | 68 / 5 / 6 | 23; 93 nm | 0 |
| 2 µm gap | 79 | 75 | 70 / 6 / 3 | 30; 129 nm | 0 (13 counting the gap) |
| 1000 localizations | 79 | 79 | 52 / 20 / 7 | 37; 164 nm | 4 |
| hard shapes | 80 | 71 | 71 / 7 / 2 | 41; 227 nm | 6 |
| interior lump (10 %) | 77 | 70 | 63 / 6 / 8 | 35; 355 nm | 9 |
| real-width cloud, 75 nm | 71 | 69 | 45 / 16 / 10 | 53; 95 nm | 0 |
| real-width cloud, 100 nm | 52 | 47 | 15 / 17 / 20 | 72; 113 nm | 1 |
| gap + 100 nm cloud | 65 | 62 | 34 / 18 / 13 | 77; 131 nm | 0 |
| lump + 100 nm cloud | 48 | 18 | 11 / 27 / 10 | 124; 351 nm | 5 |
| 150 nm cloud | 20 | **0** | 2 / 14 / 4 | — | 0 |
| filled disk (negative control) | 0 / 10 | — | — | — | — |

Calibration of the gate (13 conditions x 8 shapes x 6 seeds x 3 curve kinds, 1872 curves): the full gate flags 1.0 %
of the correct curves and 47 % of the bad ones. The slice misfit was validated with known truth only in the unit tests
(a ring 60 nm larger in slice +1 and a lateral shift of 80 nm in slice −1 are recovered as 60 ± 6 and 80 ± 8 nm;
coinciding slices give an RMS below 8 nm).

**V5, the exploratory pilot** (18 axons, one experiment, one animal under a provisional label; no ground truth).
**EXPLORATORY: a training set. No p between animals or genotypes; nothing below is a test between groups.** The full
analysis (`analyze_perimeter`, default parameters, 1999 rotations) ran on the 18 files without exception (11 "ok",
7 "contour failed" with their reason), 0.5-8.0 s per axon (median 1.8; criterion below 30 s: passes); the 7 batch
tables were written and not empty. One criterion **failed** then: the per-axon rows of the 7 failed contours lacked
32 keys (`pd_a_*`, `pd_b_*`, `pd_c_*`, ...) instead of carrying them empty; fixed in v20 (`perimeter_row` fills every
column of the layout, a test checks it). The contour fitted on the three slices gives 11 accepted (4 primary, 5 stiff,
2 polar fallback), 7 rejected; **4 reliable under D12** and **3 without slice
misfit** (the same without the one whose slices do not coincide); 7 of the 11 accepted carry the slice misfit mark (section 4 for the details).
The validation run computed the gate with the functions the integrated analysis uses, on a base just before the
integration; the integrated v20 code, run on the pilot's files on 2026-11-17, gives the same counts.

**C on the pilot with v21** (one animal, descriptive; rerun 2026-11-17 on the integrated code, files read only): the
main reading of C holds 3 axons. In one of them r(−1, +1) is +0.51 (p 0.011), "registry between slices" with no suffix:
it is above the leak of its pair over the whole measured range and above the 77 nm upper bound (leak 0.11 / 0.18 / 0.21
at 36 / 47 / 51 nm, 0.41 at 77 nm; its lateral −1 holds only 13 % of the central slice). In another r(−1, +1) is −0.09,
not significant, "leakage only", not above the leak even at 36 nm. **The third is a limit case**: r(−1, +1) +0.08,
not significant, "leakage only"; against its pair's leak it is "at the limit: above the leak at sigma_z 36 nm, not at
47 nm" (leak 0.06 / 0.11 / 0.14). Every other accepted axon is read apart. One axon in three with a registry reading is
still compatible with chance and is not a finding.

Neighbour reading with the axon's own fractions (since 2026-11-17): of the 11 axons whose analysis finished, 3 change
their base label from "neighbours above leakage" (against the equal-slice 0.27) to "leakage only" (all three with a
contour that is not reliable, so not interpretable anyway); the neighbour leak at 47 nm is 0.29-0.76. The accepted axon
pooled under D12 whose slices do not coincide stays "neighbours above leakage" (r(−1, 0) 0.53 against a leak of 0.35).
In the main reading of C nothing changes: the two "leakage only" axons already were.

Contour quality of the pilot, by group (no axon is named: [pilot], one animal):

| group | n | curve kinds | without slice misfit | D12 | frac ±100 | robust SD (nm) | persistent offset (nm) | slice RMS −1 / +1 (nm) |
|---|---|---|---|---|---|---|---|---|
| reliable, slices coincide | 3 | 2 primary, 1 polar fallback | yes | yes | 0.78-0.85 | 52-68 | 51-62 | 24-46 / 27-38 |
| reliable, slices do not coincide | 1 | polar fallback | no (slices) | yes | 0.73 | 80 | 58 | 63 / 67 |
| accepted, not reliable | 7 | 2 primary, 5 stiff | no | no | 0.43-0.62 | 101-156 | 93-302 | 43-146 / 35-122 |

**A, B and C on the pilot by subset** (descriptive, one animal; with profiles of 52-97 bins the smallest p of C is
about 0.01):

| subset | n | A contrast p < 0.05 | A period p < 0.05 | B p < 0.05 | C(−1, 0) | C(0, +1) | C(−1, +1) p < 0.05 | r(−1, +1) per axon | pooled A (within-sample) |
|---|---|---|---|---|---|---|---|---|---|
| without slice misfit | 3 | 0/3 | 0/3 | 0/3 | 3/3 | 3/3 | 1/3 | +0.51; −0.09; +0.08 | no structure (Holm p 0.69 / 0.28) |
| **reliable, D12 (main reading)** | 4 | 0/4 | 0/4 | 0/4 | 4/4 | 3/4 | 1/4 | +0.51; +0.01; −0.09; +0.08 | no structure (Holm p 0.84 / 0.15) |
| all accepted (not protected by the gate) | 11 | 0/11 | 1/11 | 1/11 | 8/11 | 6/11 | 1/11 | median +0.05 (−0.15 to +0.51) | no structure as a reading; period Holm p 0.015 at the edge of the range (4.8 µm: "trend, not repeat") |

- **C**: the only significant (−1, +1) is one axon (r 0.51, p 0.011), the axon with the best contour (persistent
  offset 51 nm, SD 52 nm). With a common contour error of that size passing the gate, the simulated false-positive
  rate is 0.03-0.06, so the simulated contour error does not explain it; but it is **one** axon of **one** animal,
  and one in four is compatible with chance. The neighbouring pairs are significant in almost every axon (r 0.26-0.65),
  as expected from the leak plus any common contour error; they are not read as registry.
- **B**: none of the 4 reliable axons; 1 of 11 accepted (not reliable, persistent offset 302 nm).
  `pd_b_tau_vs_offset` is negative in most axons (median −0.12): part of any B signal on these contours comes from the
  band shifting, as in simulation with a noisy contour.
- **A**: no axon with a significant contrast. The pooled period of the 11 accepted (Holm p 0.015) lies at the edge of
  the range and disappears with the reliable contours: **not a finding**.
- Band sensitivity 50/150 nm: 9 of 11 consistent; 2 change one decision at 50 nm. Peak minus nominal focus: median
  −2 nm (absolute 21 nm). Lateral slices: −1 holds 36 % of the central slice (13-50 %), +1 holds 53 % (25-78 %): the
  plane −1 is systematically emptier.
- Contour variants: stiff curve forced, 11 accepted and 0 reliable; polar curve forced, 4 accepted, the same 4
  reliable; no decision at 0.05 changes against the automatic curve in either. Interior trim (D8, 250 nm): with the
  weak validity rule of the trim's reference used up to v20, 16 accepted and 8 reliable under D12, not used
  (reproduced on the integrated code); the program now requires the strict rule (section 4), under which only the 4
  axons whose polar curve passes are trimmed and no reliability changes.

**What the validation leaves open.** The quality gate has since been wired into the program (sections 4 and 6), but
the "leakage only" label uses the leak without blinking (with the axon's own slice fractions since 2026-11-17, which
may underestimate it); "regular" always brings "inhibition zone"; the
power of B on symmetric shapes differs from the module's earlier run (0.23 vs 0.60); the 2 µm gap fails; the common
contour error was tested only with smooth bumps (no long chord, no per-slice error); the real-width cloud needs more
axons and a width that passes the gate (for example 75 nm) to measure B and C in the reliable stratum; the axial width
that sets the leak was measured on the pilot (47 nm, range 36-51 nm; section 8) but must be measured per file in new
data, and whether it is 36 or 47 nm (the slow error component) needs fixed beads or fiducials; the slice criterion of C was calibrated up to 50
localizations per lateral bin and must be recalibrated for denser data.

| stage | what is run | criterion planned |
|---|---|---|
| V1 contour | five shapes x seeds x three background levels | error at arcs with data and crossings within the limits of section 4 |
| V2 question C | independent slices, positive slices, leak alone | false positives about 0.03-0.07; power at least 0.8; the leak recognised |
| V3 question B | null with the true contour, null with the fitted contour, positive | same; table by band half-width |
| V4 question A | pooled sets of 18 axons: flat, blinking clumps, resolved clusters; attractor, inhibition, regular | same; the artefact at zero documented |
| V5 real files | `batch_perimeter.py` on the exploratory pilot, one file at a time | runs without exception; tables complete |

Cross-checks planned: bin alignment between the profile and the windows; the expected leak against the simulator's leak;
the contour orientation; a slice with almost no localizations; the peak against the mixture-model peak on the pilot.

## 14. Where every number comes from

| number | value | origin |
|---|---|---|
| slice thickness | 170 nm | [spec]; equals the program's reference period |
| arc bin / tortuosity window | 100 nm / 200 nm | [spec] |
| band half-width | 100 nm; sensitivity 50, 150 nm | [ad hoc, pilot] |
| z peak kernel / focus bin / min per bin | 25 nm / 40 nm / 200 | [ad hoc] |
| axial guard sensitivity | 42.5 nm | [ad hoc]: a quarter of the thickness |
| contour: smoothing / separation / threshold / knots | 50 nm / 200 nm / 0.05 / 400 nm | [own sim] sweeps (docstrings of `mps_auto_contour.py`) |
| contour: fallback knots | 600 nm | [code]: the membrane fit's spacing |
| contour: hull ratio reject / warn | 1.5 / 1.3 | [own sim] plus ad hoc margin / [own sim] |
| contour: inward notch limit | 250 nm | [own sim] |
| contour: lumen test depth / ratio | 250 nm / 0.5 | [ad hoc] / [own sim] |
| contour: minimum nodes / localizations | 8 / 200 | [plan] / [ad hoc] |
| not-measured stretch | 400 nm without data | [ad hoc] |
| minimum bins of a profile | 8 | [ad hoc] |
| A zone: height fraction / noise factor / widening / minimum | 0.10 / 2 / 1.5 / 200 nm | [own sim] / [own sim] / [own sim] / [ad hoc] |
| A robustness widening | 2 and 3 times | [own sim] |
| A near distance / minimum period | 1000 nm / 300 nm | [ad hoc] |
| minimum windows of B | 8 | [ad hoc] |
| resolved curvature radius | 600 nm | [code] (`mps_curvature`) |
| default axial precision | 47 nm | [pilot] |
| expected neighbour leak | 0.27 (no guard), 0.12 (guard 42.5 nm); measured with blinking 0.25 / 0.09 | [derivation]; [own sim] |
| slice coincidence criterion for the main reading of C | min(RMS −1, RMS +1) ≤ 32 nm; not measurable = does not coincide | [own sim; Bradley 1978 robustness band 7.5 %]; calibrated up to 50 loc./bin, recalibrate for denser data |
| axial width for the leak with real fractions | measured in the file when reliable (v22); else 47 nm main, 36-51 nm range; 77 nm upper bound only | [pilot]: NeNA in z, user's decision 2026-11-17; per-file measurement [derivation + real data, pilot]; thresholds [ad hoc] |
| interior trim (D8) applied only when | the polar curve passes the strict rules of the automatic contour | user's decision 2026-11-17; [own sim] contour report v2: weak rule lost 5-22 % of the membrane |
| minimum axons for a pooled reading of A | 10 | [own sim; Bradley 1978]: smallest simulated size (4-11, 18) where every recognisable effect has a lower Wilson bound above 0.075 and every null stays at most 0.075 |
| leak tolerance in the reading | 0.10 | [ad hoc] |
| replicates / seed | 1999 / 0 | [lit] Phipson & Smyth 2010 style / convention |
| significance level / Holm | 0.05 / between the two pooled A statistics | [lit] conventional / Holm 1979 |
| animals needed to test between genotypes | 3 per group | rule of the program's regularity plan |
| fewest finite localizations analysed | 100 | [ad hoc] |

## 15. References

- Gazal N.G. et al. (2026): the reference pipeline of MPS Explorer and the 170 nm period of the ring (as cited in
  [viability_and_reference_period.md](viability_and_reference_period.md)).
- Bradley JV (1978). Robustness? *British Journal of Mathematical and Statistical Psychology* 31:144-152. **Cited
  as the calibration study cites it; not re-read for this page.**
- Holm S (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2).
- Phipson B, Smyth GK (2010). Permutation p-values should never be zero. *Stat Appl Genet Mol Biol* 9:Article 39.
  doi:10.2202/1544-6115.1585
- Lotwick HW, Silverman BW (1982). Methods for analysing spatial processes of several types of points. *JRSS B*
  44:406-413. **Cited from memory, not re-read for this analysis.**
- Myllymäki M, Mrkvička T, Grabarnik P, Seijo H, Hahn U (2017). Global envelope tests for spatial processes. *JRSS B*
  79:381-404. **Cited from memory, not re-read for this analysis** (the global band of the pooled A profile follows its
  idea; the module uses a simultaneous band of directional quantiles).
