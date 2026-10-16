# The axial mixture, the ring boundaries, the axon axis and the ring correlation

> **Resumen (español).** Esta página documenta, para revisión de pares, cada elección del análisis axial de MPS
> Explorer: la mezcla gaussiana de z (candidatos de K, semilla, el pico principal leído como moda de la densidad y
> la fracción ajena F de la losa, con aviso por encima de 5 %), ΔZ con su error estándar y su distancia a los
> períodos publicados (nunca un veredicto), las fronteras entre anillos (valle = antimodo con profundidad referida
> a los picos, frontera de Bayes, punto medio; un nivel 0,10; la guarda en unidades de s_eff), el eje del axón (planos
> por anillo, piso de ruido de δβ, chequeos de inclinación), la correlación entre anillos (longitud de arco a lo
> largo de la membrana, 1999 rotaciones, p unilateral, **bins sin medir enmascarados por defecto** porque interpolarlos
> dio ~45 % de falsos positivos con 10 % de bins compartidos) y la ventana de la autocorrelación. Cada número dice de
> dónde sale: literatura (con cita), derivación (a la vista) o simulación propia (con su configuración y su script).
> Dos desvíos respecto de la propuesta de investigación están aceptados provisoriamente con su evidencia (sección 10).

This page explains, for peer review, every choice the axial analysis of MPS Explorer makes and where each number
comes from. Each number carries its origin in brackets:

- **[literature]** taken from a publication (cited);
- **[derived]** computed by the formula shown (first principles or a standard method);
- **[simulation]** measured on this project's own synthetic data, with the set-up and the script that reproduces it;
- **[convention]** a working value with its reasoning written, editable.

The code is in `tools/mps_periodicity.py` (the mixture, the valleys, the trade-offs), `tools/mps_multisegment.py`
(the segments and the boundary table), `tools/mps_axis.py` and `tools/mps_columns.py` (the axon axis),
`tools/mps_gaps.py` (the ring correlation), `tools/mps_autocorrelation.py` and the windows that show them
(`tools/mps_results_window.py`, `tools/mps_axon_map_layers.py`, `tools/mps_rings_window.py`). Every value also has
an entry in the parameter registry (`tools/mps_param_registry.py`), whose tooltip names this page.

Only simulated data were used to set or check any number on this page.

## 1. The axial (z) mixture of a region

**Model.** The z of every localization of the loaded region is modelled as a one-dimensional Gaussian mixture fitted
by expectation-maximisation (Dempster et al. 1977), each component read as one spectrin ring and the distance
between consecutive means as the axial period ΔZ, as in Gazal et al. (2026). A ring's width is its axial thickness,
the axial localization precision and the smear of any residual tilt in quadrature; the Gaussian shape is an
assumption.

| Value | Default | Origin |
|---|---|---|
| Candidate numbers of components | 2, 3 (offered 1-4) | [literature] chosen by BIC among 2 and 3 in Gazal et al. (2026); Schwarz (1978); Fraley & Raftery (2002). BIC approximates the density rather than counting rings (Baudry et al. 2010), so 1 and 4 can be offered (a keyword; their BIC is fitted only when asked, because it doubles the fit time). |
| Dominant component | weight > 0.05 | [literature] Gazal et al. (2026). |
| EM restarts, seed | 10, 0 | [convention] the best of 10 starts is kept; if each start finds the best optimum with probability p ≥ 0.3, all ten miss it with probability (1 − p)^10 < 3 % [derived]. A fixed seed makes a region give the same mixture every run. |
| Centre of the single-ring slab | mode of the fitted density | [convention] the program's reading of "the main peak of the axial localization distribution" (Gazal et al. 2026, who do not say how it was located); a mixture's modes are its critical points (Ray & Lindsay 2005). The mean of the heaviest component is always computed beside it and can be chosen; both are exported (`z_peak_density_nm`, `z_peak_weight_nm`, `main_peak_mode`). |
| Slab half-width | 90 nm | [literature] the published 180 nm window (Gazal et al. 2026). Half the measured period is shown beside it as the alternative. |
| Disagreement of the two peak readings | warned when > the half-width in use | [derived] two windows of width 2h centred d apart share (2h − d)/(2h) of their extent; d > h means they share less than half. The warning prints that share. |

**The slab's foreign fraction F** (replaces a warning on near-equal weights). F is the share of the slab's
localizations the fitted mixture expects from components other than the slab's own ring:

F = Σ_{k ≠ ring} w_k [Φ((hi − μ_k)/σ_k) − Φ((lo − μ_k)/σ_k)] / Σ_k w_k [Φ((hi − μ_k)/σ_k) − Φ((lo − μ_k)/σ_k)],

summed over every fitted component; the slab's ring is the component with the most mass inside [lo, hi]. It is
always shown (results table, Axial details, column `slab_foreign_fraction`) and warned above **5 %**.

- Why not the weights [derived]: equal weights are the expected state of two equally labelled rings, and the
  standard error of a difference of two weights at n = 10^4 localizations is ~0.01, so a 0.10 difference is 10 SE,
  not a test of anything.
- Magnitudes [derived, closed form, three equal rings, window ±P/2]: at P = 170 nm, F = 1.5 / 8.9 / 19.1 / 26.9 % for
  ring widths σ = 35 / 50 / 65 / 77 nm; at P = 190 nm, 0.7 / 5.7 / 14.4 / 21.7 %.
- The 5 % level [convention with derivation]: a neighbouring ring's clusters start to reach `min_samples`
  localizations of their own inside the slab once F × n_cluster ≥ min_samples, i.e. F ≥ 3-10 % for clusters of
  100-300 localizations at min_samples = 10. Editable 1-20 % (a keyword).
- Beside F the note gives the number of dominant components (no threshold).

## 2. ΔZ, its standard error and the published periods

The region's mean ΔZ is reported with its own **standard error**, SE = sd(spacings)/√m with m ≥ 2 spacings [derived],
and, for each published period, the **signed distance in units of the published spread**, (ΔZ − P_ref)/s_ref
[derived]. No pass/fail label is attached: if a published ± is the SD of individual spacings, about a third of
healthy spacings lie outside one SD, and a ±1-SD band around a mean of m spacings is left by 16 % (m = 2) of healthy
regions [derived, 2(1 − Φ(√m))]. The axial scale of the z calibration alone (Huang et al. 2008) can move a
measurement by ~10 %.

| Reference (editable table `REFERENCE_PERIODS`) | Period | Spread | What the ± is |
|---|---|---|---|
| Gazal et al. 2026, mouse sciatic nerve, along z | 170 nm | 15 nm | not stated |
| D'Este et al. 2016, mouse sciatic nerve, lateral (STED, actin) | 186 nm | 23 nm | SD |
| Xu et al. 2013, cultured hippocampal neurons, lateral (actin) | 182 nm | 16 nm | SD |
| Unsain et al. 2018, cultured neurons, lateral (spectrin) | 191 nm | 28 nm | not stated |
| Zhong et al. 2014, cultured neurons, autocorrelation | ~190 nm | none (the plain difference in nm is reported) | - |

ΔZ is in the length units of the measurement's z calibration; when the calibration record gives its magnification
factor m, it is printed beside ΔZ ("per z calibration, m = …"). `validate_periodicity.py` is a consistency check on
the five example regions shipped with the repository (pooled mean 175.2 nm, SE 5.0 nm, +0.34 spreads from Gazal
et al. 2026), not a calibration; it reads the pixel size from the files' metadata and prints when it assumes one.

## 3. Boundaries between consecutive rings

**The valley is the antimode.** Between two consecutive dominant components the program finds the local minimum
(the sign change of the slope) of the complete fitted density on a 0.5 nm grid [convention: 1-2 orders of magnitude
below any axial precision]. A two-component normal mixture has at most three critical points (Behboodian 1970;
Amendola et al. 2020), so a boundary has a valley exactly when the two components are two modes; for equal
components that needs a spacing above 2 widths. Where there is no antimode the boundary falls back to the midpoint,
its depth is 0 and a warning says that the two segments are two halves of one axial distribution.

**Depth** = 1 − f(antimode)/min(f(mode_a), f(mode_b)), referred to the two **peaks** adjacent to the valley (0 = no
dip, 1 = the density vanishes between the rings) [convention]. Referred to the density at the two means (the
program's index before 2026-10-14), shallow valleys read lower and a bimodal pair between 2.0 and ~2.2 widths read 0.
For two equal Gaussians [derived, closed form]: depth 0 at 2.0 widths apart, 0.007 at 2.1, 0.054 at 2.3, **0.10 at
~2.43**, 0.20 at 2.67, 0.36 at 3.0, 0.53 at 3.4. The separation index d = |μ_b − μ_a|/√(σ_a² + σ_b²) is reported per
boundary (the equal-width separation of Ashman et al. 1994 divided by √2).

**One depth level, 0.10** [convention], for the analysis warning ("barely separated") and the rings window's colour:
rings about 2.4 widths apart, the first level at which two Gaussian rings are visibly two (Schilling et al. 2002).
Editable range 0.02-0.30.

**Three cut positions per boundary**, all reported (rings window, pairs CSV `boundary_*`):

- the **valley** (antimode);
- the **Bayes boundary**, where the two components are equally probable, w_a f_a(z) = w_b f_b(z): the
  minimum-misassignment cut of a fitted mixture (Kittler & Illingworth 1986; McLachlan & Peel 2000). For equal widths
  z_B = m + σ² ln(w_a/w_b)/(μ_b − μ_a) [derived]: 97.5 nm for weights 0.7/0.3, σ = 50 nm, P = 170 nm, where the
  valley lies at 104.2 nm and the midpoint at 85 nm — the valley over-corrects toward the lighter ring. It exists even
  where there is no valley; where it has no root between the means the midpoint is used, with a warning;
- the **midpoint**.

The segmentation mode chooses which one cuts: `valley` (the recorded primary choice of this project), `bayes`,
`partition` (midpoint) or `paper` (overlapping ±90 nm windows, as in Gazal et al. 2026). No published MPS analysis
defines a boundary between consecutive rings.

**A flat antimode keeps the midpoint cut** (decision of 2026-10-14, section 10): the antimode is the cut only where
the dip lies below the density at both component means; otherwise it is reported with its (small) depth but the cut
stays at the midpoint and a warning says why. Evidence [simulation]: on the re-simulated leak axons of
`validate_simulate_axon.py` (test "simulate from the measured config", ring widths ~71-80 nm at P ≈ 170 nm, depth
~0.01) the antimode moved from 67.5 to 103.1 nm and from 95.4 to 61.6 nm between replicates (35 nm, in either
direction), the middle ring's median cluster size fell from 139.5 to 107 localizations and the check failed. With
the rule, no cut of any segment moves with respect to the program before this change.

**The shift** of the valley from the midpoint and from the Bayes boundary (positive toward the lighter ring) is
reported per boundary, never judged: a shift is the normal state of rings that differ in weight or width. An optional
warning threshold is left blank (off).

**Posterior membership** is that of the **complete** fitted mixture (the same model the valleys are located in); a
localization whose most probable component is a discarded low-weight one belongs to no ring (McLachlan & Peel 2000).
The optional posterior control (off in the primary analysis; 0.85 when switched on) is a reject option (Chow 1970).
For two equal rings at spacing P the rejected zone is (σ²/P) ln(p/(1 − p)) on each side of a boundary [derived]; at
P = 170 nm and p = 0.85:

| σ (nm) | half-width rejected (nm) | rejected share of a two-ring stack | error among accepted |
|---|---|---|---|
| 35 | 12.5 | 1.6 % | 0.3 % |
| 50 | 25.5 | 10.4 % | 1.5 % |
| 65 | 43.1 | 23.5 % | 3.2 % |
| 77 | 60.5 | 34.6 % | 4.5 % |

The pre-registered sensitivity {0.7, 0.85, 0.95} [simulation: `validate_rings_h1.make_synthetic_axon(3,
a_nm=800, b_nm=550, n_clusters=24, tilt_deg=2, azimuth_deg=30, n_fluor=8, axial_noise_nm=σ)`, `build_rings` with
`posterior_min = p`] rejected 0.3 / 1.0 / 1.9 % at σ_z = 35 nm, 4.6 / 9.1 / 17.2 % at 50 nm and 10.0 / 22.6 / 41.8 %
at 65 nm, as the closed form predicts; the complete and the dominant-only posterior agreed on these axons.

**The guard band** (a control, not the primary analysis): a dead zone of **total** width g centred on each boundary,
half on each side; 0 in the primary analysis and 60 nm as the recorded sensitivity. For rings of effective axial
width s_eff (shown as the median width of the fitted components) at a boundary h = ΔZ/2 from the ring centre
[derived]: lost = Φ((h + g/2)/s_eff) − Φ((h − g/2)/s_eff), leak = 1 − Φ((h + g/2)/s_eff). At P = 170 nm:

| s_eff | g = 0 | 30 nm | 60 nm | 90 nm | retention cap g* |
|---|---|---|---|---|---|
| 35 nm | leak 0.76 % | lost 2.1 %, leak 0.21 % | 5.8 %, 0.05 % | 12.6 %, 0.01 % | 123 nm |
| 50 nm | 4.5 % | 5.8 %, 2.3 % | 12.5 %, 1.1 % | 20.7 %, 0.47 % | 103 nm |
| 77 nm | 13.5 % | 8.5 %, 9.7 % | 17.0 %, 6.8 % | 25.6 %, 4.6 % | 67 nm |

The cap g* = 2(h − 0.67 s_eff) is the widest guard that keeps half of a ring guarded at both boundaries
(Φ(0.67) − Φ(−0.67) = 0.50) [derived]. The rings window shows the guard as a multiple of s_eff with its leak;
`validate_gap_controls.py` uses the grid {0, 0.5, 1, 1.5} × s_eff plus g*, prints the leak and the measured fraction
of localizations removed, and calls the control **partial** where the cap leaves a leak ≥ 1 %; 0/30/60/90 nm total
is the fallback without an s_eff. A guard also removes genuine ring structure, so a drop under it is ambiguous; no
published guard-band control for axial slabs was found.

The earlier viability verdict (D-39) required a valley with depth > 0; that minimum is now an editable threshold
(default 0, unchanged). The rule used for selection (v2) tests the valley's significance with SiZer (Chaudhuri &
Marron 1999) instead; the earlier verdict is shown for comparison.

## 4. The axon axis and the tilt

| Value | Default | Origin |
|---|---|---|
| Plane per ring | least squares refined with SciPy's soft-L1 loss | [derived] regressing z on (x, y) is right because the lateral error is negligible against a ring's lateral extent; slope error σ_z/(√n · sd(x)) (Draper & Smith 1998). The robust loss (Huber 1964; Triggs et al. 2000) is a safeguard against heavy-tailed axial outliers, not a correction. Each ring reports its predicted slope error σ/(√n · s_min). |
| Localizations per ring plane | 200 | [convention] ~0.2° slope error at σ_z = 50 nm over a 0.9 µm spread [derived]; an optional gate by predicted slope error (suggested 0.3°) exists, off by default. A plane on all rings is the fallback and is flagged "frame less trusted". |
| Noise floor of δβ | √(2(k−1)/k) σ_z/(√n s) per ring, n-weighted | [derived] Draper & Smith (1998); checked on simulated untilted rings (`test_sci6_axial_rings.py`: the RMS δβ of 60 replicates of 3 rings × 500 localizations, σ_z = 50 nm, R = 1.5 µm, is within ±30 % of it). The noise-corrected √max(δβ² − floor², 0) is reported. |
| δβ warning | max(1°, 3 × floor) | [convention] 1° shifts consecutive rings laterally by P tan δβ ≈ 3 nm at P = 170-190 nm [derived], small against the ~25 nm cluster radius (Gazal et al. 2026): the warning concerns the rigid-axis model, not the pairing. A δβ that cannot be computed makes the shift "unknown" in the pair tolerances. |
| Profile-sharpness start | ±8°, coarse 0.5°, fine 0.1°, 5 nm bins | [derived] the collision probability Σp_i² (Simpson 1949; Rényi 1961) halves when D tan δ ≈ 5.7 σ_z; the bin biases it by −(b/σ)²/24; the range is at least the tilt warning + 3°. |
| Stage tolerance | 0.05° | [convention] the per-slope error at σ_z = 50 nm, N = 5000, s = 900 nm. The per-axon alternative max(0.05°, noise floor) exists but is **off** (section 10). |
| Maximum stages | 12 | [convention, to be re-measured] assumes a contraction of ~30 % of the residual per stage; a frame that runs out of stages is labelled "noise-limited" (last increment below 3 × the floor) or "not converged". |
| Stage posterior | 0.85 | [convention] only localizations assigned to a ring with posterior ≥ 0.85 vote for the axis (Chow 1970). Alternative 0.95 measured (below). |
| Judge between the two starts | sum of valley depths | [convention, validated on synthetic axons only] stacking half rings onto a third removes a boundary and lowers the sum; the mean depth and the mixture BIC at a fixed K are selectable for comparison. Both candidate tilts and depths are recorded. |
| Start disagreement warning | 0.5° | [convention] 3-10 × the slope noise of one plane; a multiple of the measured floor is the per-axon alternative. |
| Tilt warning | 5° | [convention] no published threshold (the reference pipeline discards visibly oblique fields by eye, Gazal et al. 2026). |

**Stage posterior, the 0.95 alternative** [simulation: the half-ring axon of `validate_rings_h1.py`, section 7c,
σ_z = 50 nm, true tilt 2°, `build_rings` with `axis_stage_posterior_min = p`]: fitted tilt 0.77 / 1.16 / **1.34** /
1.51° and ring agreement 89.0 / 90.0 / **90.3** / 90.6 % for p = 0.5 / 0.7 / **0.85** / 0.95. Some cut helps, and
0.95 helped more here; 0.85 is kept until a broader validation.

**Two derived tilt checks per axon** (effective diameter D = median ring perimeter / π) [derived]:

- D tan β > P: one ring spans a full period in laboratory z, so the uncorrected profile cannot show rings and the
  frame correction is mandatory (β_P = atan(P/D) = 3.2° for a 3 µm axon at P = 170 nm);
- D tan β > Z_usable − P: a ring and its neighbours leave the usable depth of the z calibration and no rotation
  recovers them (β_Z = atan((Z_usable − P)/D) = 11.9° for a 3 µm axon at Z_usable = 800 nm). Z_usable defaults to
  800 nm [literature: the accessible part of the ±1000 nm bead stack of Gazal et al. 2026; Huang et al. 2008 worked
  within ~600 nm], editable 300-1500 nm.

The rotation into the axon frame is Rodrigues' formula (Rodrigues 1840; Dai 2015), exact and distance-preserving.

## 5. The correlation between consecutive rings

**Frame: arc length along the membrane (default).** Every segment's perimeter is projected on one common curve, the
pooled centroid membrane of all segments (the curve the column analysis uses; cf. elliptic Fourier descriptors,
Kuhl & Giardina 1982), and binned by its arc s ∈ [0, L) with s = 0 the same point for all segments. It does not need
a star-shaped contour. The **angle** about the pooled centre is kept as an exploratory frame: it is a one-to-one
coordinate only on contours star-shaped about that centre; the pairs CSV reports the share of perimeter samples whose
angle runs backwards (`angle_nonmonotonic_fraction_a/_b`; 0 for a star-shaped contour). When the common curve cannot
be fitted the angle frame is used, with a warning.

| Value | Default | Origin |
|---|---|---|
| Bins | 3600 | [convention] 1-11 nm of arc on perimeters of 5-40 µm, below the lateral precision and far below a patch (tens of nm; ~25 nm effective cluster radius, Gazal et al. 2026). |
| Perimeter samples per bin | 8 | [convention] equally spaced samples, so per-bin coverage is resolved in steps of 1/8; it never changes which perimeter is occupied. |
| Rotations of the null | 1999, drawn without replacement, seed 0 | [literature] the toroidal-shift test on the circle (Lotwick & Silverman 1982; Harms et al. 2001; Alexander-Bloch et al. 2018), decoupled from the bins. |
| p-value | (1 + b)/(1 + n_rot), one-sided | [literature] counting the unrotated configuration (Hope 1968; Phipson & Smyth 2010); one-sided because aligned patches predict positive correlation. |
| Effective rotations | perimeter / mean patch length | [derived] neighbouring rotations are correlated over a patch width; exported as `n_effective_rotations`. |
| Unmeasured bins | **masked** (interpolation is an option, warned) | [simulation] below. |

**Masking against interpolation.** A bin no perimeter sample reaches is either left out of the correlation (masked:
the Pearson correlation at every rotation is taken over the bins both profiles measured, computed by FFT) or filled
from its neighbours (interpolated). A stretch interpolated in **both** profiles is shared structure the rotation null
cannot remove (two linear ramps over the same bins correlate perfectly there). The public calibration of the null,
`validate_ring_correlation_null.py` [simulation: 400 pairs of independent stationary patch profiles per case, 3600
bins on a 20 µm perimeter, Poisson patches at 20 % occupancy with exponential lengths (mean 60 nm, +10 nm), seeds
1-3], measured at a nominal 5 %:

| Case | False-positive rate (Wilson 95 %) |
|---|---|
| independent profiles, nothing unmeasured | 0.055 (0.037-0.082) |
| 10 % of the bins unmeasured in both, **masked** | 0.060 (0.041-0.088) |
| 10 % of the bins interpolated identically in both | **0.455 (0.407-0.504)** |

and detected aligned patches (80 % kept in each profile, 10 nm jitter) in 100 of 100 pairs. Masking is therefore the
default (decision of 2026-10-14). When interpolation is chosen and a pair shares at least 2 % of interpolated bins, a
warning cites this rate. On the program's own simulated axons with three clusters placed inside the axon, the common
curve detours through them and about 30 % of the bins of every segment are not reached; the warning also asks to look
for such clusters.

**Aggregates over many pairs** (`validate_gaps.py`, exploratory, real data only from outside the repository): the
Fisher-z mean of r(0), the binomial count of significant pairs and the KS test of the folded offsets are labelled
**indicative**, because pairs nested in one axon are not independent (Lazic 2010); the dependence-aware aggregate is a
permutation over axons (every pair's second segment re-paired with a segment of another axon). The angular
nearest-neighbour cross-check is compared with the value for N independent uniform clusters, median 180(1 − 2^(−1/N))
≈ 125/N degrees (mean 180/(N + 1)) [derived].

## 6. The autocorrelation amplitude of a profile along the axon

The amplitude is the first peak minus the mean of the two first valleys (Zhong et al. 2014, Fig. 1E legend).

| Value | Default | Origin |
|---|---|---|
| Reference period P_ref | 170 nm | [literature] the axial period of mouse sciatic nerve (Gazal et al. 2026). |
| First-peak window | ±35 % of P_ref (110-230 nm), a local maximum required | [derived] covers 182 ± 16 nm (Xu et al. 2013) and 170 ± 15 nm (Gazal et al. 2026) to about 3 SD while staying above the first valley at P/2 and below the second at 3P/2 (Han et al. 2017). An unstructured profile returns no period instead of the window's edge. Editable 0.10-0.50. |
| Shortest profile | 3 expected periods (⌈3P/pixel⌉ samples) | [derived] the first peak (lag P) and the second valley (lag up to 2P) need one period of overlap at lag 2P. Editable 2-5. |

## 7. What is exported

- Axon table (version 6), after every earlier column: `main_peak_mode`, `z_peak_density_nm`, `z_peak_weight_nm`,
  `slab_foreign_fraction`, `delta_z_se_nm`.
- Rings segments CSV, at the end: the same five columns. Rings pairs CSV, at the end: `boundary_midpoint_nm`,
  `boundary_bayes_nm`, `boundary_valley_nm`, `boundary_valley_shift_from_midpoint_nm`,
  `boundary_valley_shift_from_bayes_nm`, `boundary_separation_d`, `correlation_frame`, `n_bins`, `empty_bins`,
  `n_empty_bins_a`, `n_empty_bins_b`, `n_effective_rotations`, `angle_nonmonotonic_fraction_a`,
  `angle_nonmonotonic_fraction_b`; `n_rotations` is 1999.
- The column dictionary written beside every axon export describes each column.

## 8. Limitations

- The Gaussian ring shape is an assumption; non-Gaussian tails (depth-dependent precision, residual tilt, partial arcs)
  earn extra components under BIC.
- F and the posterior trade-offs use the fitted mixture as if it were the truth; their own sampling uncertainty is
  not propagated.
- s_eff is the fitted component width, not a measured structural width; the guard's leak figures inherit that.
- The arc-length frame depends on the common curve: clusters off the membrane (inside the axon) pull it inward and
  leave stretches no segment reaches (masked by default, warned).
- The rotation null assumes stationarity around the ring; its calibration above is on synthetic stationary profiles.
- The judge between the axis starts and the stage count rest on synthetic axons only; the judge-accuracy protocol
  (at least 95 % right) has not been run.
- The z-calibration's usable depth is not yet passed from the bead stack to the tilt check (800 nm is used).

## 9. Open points (deferred)

- **Editors** for the advanced values (candidate K, restarts and seed, main-peak reading, F level, axis thresholds,
  judge, Z_usable, depth reference, posterior model): today keywords / `RingsParams` fields with registry entries;
  their editors belong to the parameter-home work.
- **Long calibration runs** (deferred to SCI-9): the Latin-hypercube study of the mixture and ΔZ (300 cells × 50
  replicates), the judge-accuracy protocol, the re-measured contraction per axis stage, and a broader validation of
  the stage posterior (0.85 against 0.95).
- The live posterior / guard tables in the interface (the closed forms exist: `posterior_tradeoff`, `guard_tradeoff`).

## 10. Decision record

- **2026-10-08**: the research proposal's recommendations accepted (valley primary with the Bayes boundary reported;
  posterior on the complete mixture; depth referred to the peaks; one level 0.10; the guard as a total width in
  units of s_eff; arc-length frame for the ring correlation with rotations decoupled from the bins; autocorrelation
  window 0.35).
- **2026-10-14, accepted provisionally on synthetic evidence** (the user reviews):
  1. *Stage tolerance from the noise floor is off by default.* On the half-ring axon of `validate_rings_h1.py`
     (σ_z = 50 nm, 2°) max(0.05°, floor) stopped a still-converging refinement (increments 0.38, 0.48, 0.29, 0.13° …)
     at the floor (0.21°): fitted tilt 1.00° instead of 1.34° (truth 2°), ring agreement 89.7 % instead of 90.3 %.
  2. *A flat antimode keeps the midpoint cut* (section 3): the antimode moved 35 nm between simulated replicates and
     broke the re-simulation check of `validate_simulate_axon.py`.
- **2026-10-14**: unmeasured bins of the ring correlation are **masked by default**; interpolation stays an option with
  a warning citing the measured false-positive rate (0.455 at a nominal 0.05 with 10 % of bins interpolated in both).
- **2026-10-14**: the stage posterior stays at 0.85; 0.95 is documented as the alternative (section 4).

## References

- Alexander-Bloch AF et al. (2018). On testing for spatial correspondence between maps of human brain structure and
  function. *NeuroImage* 178:540-551. doi:10.1016/j.neuroimage.2018.05.070
- Amendola C, Engström A, Haase C (2020). Maximum number of modes of Gaussian mixtures. *Information and Inference*
  9(3):587-600. doi:10.1093/imaiai/iaz013
- Ashman KA, Bird CM, Zepf SE (1994). Detecting bimodality in astronomical datasets. *Astronomical Journal*
  108:2348. doi:10.1086/117248
- Baudry J-P, Raftery AE, Celeux G, Lo K, Gottardo R (2010). Combining mixture components for clustering. *J. Comput.
  Graph. Stat.* 19(2):332-353. doi:10.1198/jcgs.2010.08111
- Behboodian J (1970). On the modes of a mixture of two normal distributions. *Technometrics* 12(1):131-139.
  doi:10.1080/00401706.1970.10488640
- Chaudhuri P, Marron JS (1999). SiZer for exploration of structures in curves. *JASA* 94(447):807-823.
  doi:10.1080/01621459.1999.10474186
- Chow CK (1970). On optimum recognition error and reject tradeoff. *IEEE Trans. Inf. Theory* 16(1):41-46.
  doi:10.1109/TIT.1970.1054406
- Dai JS (2015). Euler-Rodrigues formula variations, quaternion conjugation and intrinsic connections. *Mechanism and
  Machine Theory* 92:144-152. doi:10.1016/j.mechmachtheory.2015.03.004
- Dempster AP, Laird NM, Rubin DB (1977). Maximum likelihood from incomplete data via the EM algorithm. *J. R. Stat.
  Soc. B* 39(1):1-38. doi:10.1111/j.2517-6161.1977.tb01600.x
- D'Este E, Kamin D, Velte C, Göttfert F, Simons M, Hell SW (2016). Subcortical cytoskeleton periodicity throughout the
  nervous system. *Scientific Reports* 6:22741. doi:10.1038/srep22741
- Draper NR, Smith H (1998). *Applied Regression Analysis*, 3rd ed. Wiley. doi:10.1002/9781118625590
- Fraley C, Raftery AE (2002). Model-based clustering, discriminant analysis, and density estimation. *JASA*
  97(458):611-631. doi:10.1198/016214502760047131
- Gazal NG, Escalante G, Lopez LF, Goutebroze L, Bisbal M, Gorostiza EA, Szalai AM, Stefani FD, Unsain N (2026).
  Nanoscale organization of βII-spectrin within segments of the membrane-associated periodic skeleton in mouse sciatic
  nerve axons. bioRxiv doi:10.64898/2026.02.19.705149; eLife Reviewed Preprint doi:10.7554/eLife.111108.1
- Han B, Zhou R, Xia C, Zhuang X (2017). Structural organization of the actin-spectrin-based membrane skeleton in
  dendrites and soma of neurons. *PNAS* 114(32):E6678-E6685. doi:10.1073/pnas.1705043114
- Harms KE, Condit R, Hubbell SP, Foster RB (2001). Habitat associations of trees and shrubs in a 50-ha neotropical
  forest plot. *Journal of Ecology* 89(6):947-959. doi:10.1111/j.1365-2745.2001.00615.x
- Hope ACA (1968). A simplified Monte Carlo significance test procedure. *J. R. Stat. Soc. B* 30(3):582-598.
  doi:10.1111/j.2517-6161.1968.tb00759.x
- Huang B, Wang W, Bates M, Zhuang X (2008). Three-dimensional super-resolution imaging by stochastic optical
  reconstruction microscopy. *Science* 319(5864):810-813. doi:10.1126/science.1153529
- Huber PJ (1964). Robust estimation of a location parameter. *Ann. Math. Stat.* 35(1):73-101.
  doi:10.1214/aoms/1177703732
- Kittler J, Illingworth J (1986). Minimum error thresholding. *Pattern Recognition* 19(1):41-47.
  doi:10.1016/0031-3203(86)90030-0
- Kuhl FP, Giardina CR (1982). Elliptic Fourier features of a closed contour. *Computer Graphics and Image Processing*
  18(3):236-258. doi:10.1016/0146-664X(82)90034-X
- Lazic SE (2010). The problem of pseudoreplication in neuroscientific studies. *BMC Neuroscience* 11:5.
  doi:10.1186/1471-2202-11-5
- Lotwick HW, Silverman BW (1982). Methods for analysing spatial processes of several types of points. *J. R. Stat.
  Soc. B* 44(3):406-413. doi:10.1111/j.2517-6161.1982.tb01221.x
- McLachlan GJ, Peel D (2000). *Finite Mixture Models*. Wiley. doi:10.1002/0471721182
- Phipson B, Smyth GK (2010). Permutation p-values should never be zero. *Stat. Appl. Genet. Mol. Biol.* 9(1):39.
  doi:10.2202/1544-6115.1585
- Ray S, Lindsay BG (2005). The topography of multivariate normal mixtures. *Ann. Stat.* 33(5):2042-2065.
  doi:10.1214/009053605000000417
- Rényi A (1961). On measures of entropy and information. *Proc. 4th Berkeley Symp.* 1:547-561.
- Rodrigues O (1840). Des lois géométriques qui régissent les déplacements d'un système solide dans l'espace. *J. Math.
  Pures Appl.* 5:380-440.
- Schilling MF, Watkins AE, Watkins W (2002). Is human height bimodal? *Am. Stat.* 56(3):223-229.
  doi:10.1198/00031300265
- Schwarz G (1978). Estimating the dimension of a model. *Ann. Stat.* 6(2):461-464. doi:10.1214/aos/1176344136
- Simpson EH (1949). Measurement of diversity. *Nature* 163:688. doi:10.1038/163688a0
- Triggs B, McLauchlan PF, Hartley RI, Fitzgibbon AW (2000). Bundle adjustment — a modern synthesis. *Vision
  Algorithms: Theory and Practice*, LNCS 1883:298-372. doi:10.1007/3-540-44480-7_21
- Unsain N, Bordenave MD, Martinez GF, Jalil S, von Bilderling C, Barabas FM, Masullo LA, Johnstone AD, Barker PA,
  Bisbal M, Stefani FD, Cáceres AO (2018). Remodeling of the actin/spectrin membrane-associated periodic skeleton,
  growth cone collapse and F-actin decrease during axonal degeneration. *Scientific Reports* 8:3007.
  doi:10.1038/s41598-018-21232-0
- Xu K, Zhong G, Zhuang X (2013). Actin, spectrin, and associated proteins form a periodic cytoskeletal structure in
  axons. *Science* 339(6118):452-456. doi:10.1126/science.1232251
- Zhong G, He J, Zhou R, Lorenzo D, Babcock HP, Bennett V, Zhuang X (2014). Developmental mechanism of the periodic
  membrane skeleton in axons. *eLife* 3:e04581. doi:10.7554/eLife.04581
