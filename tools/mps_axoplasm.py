# -*- coding: utf-8 -*-
"""
Where is the axoplasm? A widefield betaIII-tubulin image as a mask for the
betaII-spectrin localizations.

betaIII-tubulin fills the axoplasm, while betaII-spectrin anchored to the
membrane forms the periodic lattice at its rim. A widefield tubulin image
taken with the super-resolution acquisition therefore tells which spectrin
localizations lie inside the axon and which lie at its edge. That holds
only if the image is placed where the localizations are, and if its
blurred edge is treated as blurred. The module works in three steps, each
reported so it can be checked:

1. PLACING THE IMAGE. Picasso stores a localization at x = c for an
   emitter centred on camera pixel c (``gausslq.fit_spot`` fits on a grid
   centred on the box's central pixel). Pixel c of an image taken with the
   same camera therefore spans [c - 0.5, c + 0.5) in localization
   coordinates. Different camera regions are reconciled from the
   Micro-Manager metadata that both files carry. The sample still moves
   between acquisitions: on 30/4/26 the widefield images sat (-2.1, +5.6)
   px away from the drift-corrected STORM data. The shift is measured by
   cross-correlating a widefield image of the super-resolved protein with a
   rendering of its localizations. The tubulin image cannot do this: it
   shows another structure. It takes the shift of a widefield image
   acquired with it.

2. THE MASK. The tubulin image is smoothed and thresholded around the
   selected axon, and the connected region under the axon's centre is
   kept. The threshold proposed is Otsu's (Otsu 1979, IEEE Trans. Syst.
   Man Cybern. 9:62), computed on that region only, because the contrast
   varies from axon to axon. The user adjusts it and the value used is
   recorded.

3. THE CLASSIFICATION. Every localization gets its signed distance to the
   mask's edge, positive inside. It is "interior" when it lies deeper than
   a margin the user sets, and "membrane" otherwise.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from tools import mps_metadata, mps_pixel_size

# (x, y, width, height) of the camera region, as Micro-Manager writes it.
CameraRegion = Tuple[int, int, int, int]

# Registration: the widefield point-spread function the rendering is
# blurred with, and the band kept on both images, in camera pixels.
DEFAULT_PSF_SIGMA_PX = 1.2
DEFAULT_BAND_LOW_PX = 1.0
DEFAULT_BAND_HIGH_PX = 8.0
DEFAULT_MAX_SHIFT_PX = 20
# How a registration is judged. The runner-up is the highest separate
# local maximum of the correlation (farther than RUNNER_UP_EXCLUSION_PX
# from the best shift). The score is the best shift's height above the
# median of the search, in robust standard deviations (1.4826 * MAD).
# Measured while building this: correct registrations had a runner-up of
# 0.09-0.11 of the peak and a score of 13-21, both on synthetic fields and
# on the April 2026 ROI 1 data. Wrong ones scored 2.6-6.5 with a runner-up
# of 0.58-0.88: the true shift outside the search, a periodic field, one
# picked axon alone, or the tubulin image as reference.
RUNNER_UP_EXCLUSION_PX = 3
RUNNER_UP_FRACTION = 0.5
MIN_REGISTRATION_SCORE = 10.0

# Mask: smoothing of the tubulin image, the fine grid the edge is traced
# on, and the size of the region around the axon.
DEFAULT_SMOOTH_SIGMA_PX = 1.0
DEFAULT_UPSAMPLE = 4
REGION_RADIUS_FACTOR = 2.0
REGION_MIN_MARGIN_PX = 5.0
# Warn when the mask's area is outside this range of the area the spectrin
# ring encloses. Only a prompt to look at the overlay, not a criterion.
AREA_RATIO_WARN = (0.5, 2.0)
# An image whose recorded pixel size differs from the localizations' by
# this much or more is refused: that is another scale altogether (another
# binning, camera, or a unit written wrong), and placing it would be
# meaningless. A few percent is reported instead and the image is placed,
# because which of the two values is right is often unsettled -- the 2023
# data records 133 nm in its widefield images and is analysed with 135.
PIXEL_SIZE_REFUSE = 0.10

LABEL_INTERIOR = "interior"
LABEL_MEMBRANE = "membrane"
LABEL_OUTSIDE = "outside region"


# ===================================================================
#  The widefield image
# ===================================================================
@dataclass
class WidefieldImage:
    """A widefield image, averaged over its planes."""

    path: str
    image: NDArray[np.float64]            # (rows, cols)
    n_planes: int
    camera_region: Optional[CameraRegion]
    binning: Optional[int]
    pixel_size_nm: Optional[float]        # None when the file records none
    acquired: Optional[str]
    # Where the pixel size was read from, for the messages.
    pixel_size_source: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def shape(self) -> Tuple[int, int]:
        rows, cols = self.image.shape
        return int(rows), int(cols)


def _parse_region(value: Any) -> Optional[CameraRegion]:
    """Micro-Manager's "x-y-width-height", or None."""
    if value is None:
        return None
    parts = str(value).strip().split("-")
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError:
        return None
    return (x, y, w, h)


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower().split("x")[0]
    try:
        return int(float(text))
    except ValueError:
        return None


def _page_metadata(tif: Any) -> Dict[str, Any]:
    """The Micro-Manager metadata of the first plane, or {}."""
    tag = tif.pages[0].tags.get("MicroManagerMetadata")
    if tag is None:
        return {}
    value = tag.value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def load_widefield(path: str) -> WidefieldImage:
    """
    Read a widefield TIFF and average it over its planes.

    Frames and z planes are averaged. An image with several channels is
    refused, because averaging would mix them. The camera region, binning,
    pixel size and acquisition time come from the Micro-Manager metadata
    when the file carries it.
    """
    import tifffile

    name = os.path.basename(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = str(series.axes)
        data = np.asarray(series.asarray())
        meta = _page_metadata(tif)
    for axis in ("C", "S"):
        if axis in axes and data.shape[axes.index(axis)] > 1:
            raise ValueError(
                f"{name} holds {data.shape[axes.index(axis)]} channels "
                f"(axes {axes}). Averaging them would mix them: save the "
                f"tubulin channel on its own first.")
    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"{name} has no image plane (axes {axes}).")
    order = [i for i, a in enumerate(axes) if a not in "YX"]
    order += [axes.index("Y"), axes.index("X")]
    planes = np.transpose(data, order).reshape(-1, data.shape[axes.index("Y")],
                                               data.shape[axes.index("X")])
    image = planes.astype(np.float64).mean(axis=0)

    # Micro-Manager's own field, ImageJ's unit and resolution, or the
    # plain TIFF tags -- whichever the file carries.
    recorded, notes = mps_pixel_size.from_tiff(path)
    pixel = recorded.nm if recorded is not None else None
    region = _parse_region(meta.get("ROI"))
    if not meta:
        notes.append(
            f"{name} carries no Micro-Manager metadata: its camera region "
            f"is unknown, and it is assumed to cover the same pixels as the "
            f"localization movie.")
    return WidefieldImage(
        path=path, image=image, n_planes=int(planes.shape[0]),
        camera_region=region, binning=_parse_int(meta.get("Binning")),
        pixel_size_nm=pixel, acquired=meta.get("ReceivedTime"),
        pixel_size_source="" if recorded is None else recorded.source,
        notes=notes,
    )


def movie_geometry(
    info: Sequence[Dict[str, Any]],
) -> Tuple[Optional[CameraRegion], Optional[int], Optional[str],
           Optional[str]]:
    """
    Camera region, binning, start time and source movie of a Picasso file.

    Picasso Localize copies the movie's Micro-Manager metadata into its
    first document, and the movie's path into ``File``.
    """
    mm = mps_metadata.get_value(info, "Micro-Manager Metadata")
    movie = mps_metadata.get_value(info, "File")
    source = None if movie is None else str(movie)
    if not isinstance(mm, dict):
        return None, None, None, source
    return (_parse_region(mm.get("ROI")), _parse_int(mm.get("Binning")),
            mm.get("ReceivedTime"), source)


def camera_offset(
    image: WidefieldImage,
    info: Sequence[Dict[str, Any]],
    pixel_size_nm: Optional[float],
) -> Tuple[Tuple[float, float], List[str]]:
    """
    The offset, in camera pixels, that takes a localization's coordinates
    to the image's pixel indices: ``column = x_px + dx``.

    Both files must share the camera binning and the pixel size. When
    either camera region is unknown the offset is zero, and a note says
    that this was assumed.
    """
    notes: List[str] = []
    name = os.path.basename(image.path)
    if (image.pixel_size_nm is not None and pixel_size_nm
            and abs(image.pixel_size_nm - pixel_size_nm) > 1e-6):
        ratio = abs(image.pixel_size_nm - pixel_size_nm) / pixel_size_nm
        where = image.pixel_size_source or f"{name}"
        if ratio >= PIXEL_SIZE_REFUSE:
            raise ValueError(
                f"{name} gives a pixel size of {image.pixel_size_nm:g} nm "
                f"and the localizations {pixel_size_nm:g} nm, "
                f"{ratio:.0%} apart. That is another scale -- a different "
                f"binning, camera or unit -- and the image cannot be placed "
                f"on them.")
        # A few percent still places the image, stretched: the 2023 data
        # records 133 nm in its widefield images and is analysed with 135.
        notes.append(
            f"{where} records {image.pixel_size_nm:g} nm per pixel and the "
            f"localizations are analysed with {pixel_size_nm:g} nm "
            f"({ratio:.1%} apart). The image is placed anyway, stretched by "
            f"that much against the localizations: check the overlay, and "
            f"correct whichever value is wrong.")
    region, binning, _time, _movie = movie_geometry(info)
    if (image.binning is not None and binning is not None
            and image.binning != binning):
        raise ValueError(
            f"{name} was taken with binning {image.binning} and the "
            f"localization movie with binning {binning}: their pixels do "
            f"not match.")
    if image.camera_region is None or region is None:
        notes.append(
            f"The camera region of "
            f"{'the localization movie' if region is None else name} is not "
            f"recorded: the image is assumed to start at the movie's first "
            f"pixel.")
        return (0.0, 0.0), notes
    return ((float(region[0] - image.camera_region[0]),
             float(region[1] - image.camera_region[1])), notes)


# ===================================================================
#  Registration
# ===================================================================
@dataclass
class ImageRegistration:
    """
    Shift added to the localizations' pixel coordinates (after the camera
    offset) to land them on a widefield image acquired at another time.
    """

    shift_px: Tuple[float, float] = (0.0, 0.0)
    source: str = "none"              # "none" | "measured" | "manual"
    peak: Optional[float] = None      # correlation at the shift
    zero: Optional[float] = None      # correlation without shifting
    runner_up: Optional[float] = None  # best separate local maximum
    score: Optional[float] = None     # peak over the search, robust SDs
    n_localizations: int = 0
    warnings: List[str] = field(default_factory=list)

    def shift_nm(self, pixel_size_nm: float) -> Tuple[float, float]:
        return (self.shift_px[0] * pixel_size_nm,
                self.shift_px[1] * pixel_size_nm)


def render_counts(x_px: NDArray[np.float64], y_px: NDArray[np.float64],
                  shape: Tuple[int, int]) -> NDArray[np.float64]:
    """Localizations per image pixel; pixel c spans [c - 0.5, c + 0.5)."""
    rows, cols = shape
    counts, _, _ = np.histogram2d(
        np.asarray(y_px, float), np.asarray(x_px, float),
        bins=(rows, cols), range=((-0.5, rows - 0.5), (-0.5, cols - 0.5)))
    return np.asarray(counts, dtype=np.float64)


def _band(image: NDArray[np.float64], low: float, high: float
          ) -> NDArray[np.float64]:
    out = (ndimage.gaussian_filter(image, low)
           - ndimage.gaussian_filter(image, high))
    spread = float(out.std())
    return np.asarray((out - out.mean()) / spread if spread > 0 else out * 0,
                      dtype=np.float64)


def _parabola(minus: float, centre: float, plus: float) -> float:
    denom = minus - 2.0 * centre + plus
    return 0.0 if denom == 0 else 0.5 * (minus - plus) / denom


def measure_shift(
    reference: NDArray[np.float64],
    x_px: NDArray[np.float64],
    y_px: NDArray[np.float64],
    *,
    max_shift_px: int = DEFAULT_MAX_SHIFT_PX,
    psf_sigma_px: float = DEFAULT_PSF_SIGMA_PX,
    band_low_px: float = DEFAULT_BAND_LOW_PX,
    band_high_px: float = DEFAULT_BAND_HIGH_PX,
) -> ImageRegistration:
    """
    The shift that lands the localizations on a widefield image of the same
    protein, from the cross-correlation of the two after a band-pass.

    ``x_px``/``y_px`` must already be in the image's pixel indices (camera
    offset applied). The result is refined to sub-pixel precision with a
    parabola through the peak's neighbours. The correlation at zero shift
    and the best one away from the peak are reported alongside. A peak at
    the edge of the search, or a second shift nearly as good, raises a
    warning.
    """
    ref = np.asarray(reference, dtype=np.float64)
    x = np.asarray(x_px, float)
    y = np.asarray(y_px, float)
    rows, cols = ref.shape
    inside = (x > -0.5) & (x < cols - 0.5) & (y > -0.5) & (y < rows - 0.5)
    warnings: List[str] = []
    if int(inside.sum()) < 2:
        raise ValueError(
            "The localizations do not fall on the widefield image: they "
            "cannot be registered to it.")
    rendering = ndimage.gaussian_filter(
        render_counts(x[inside], y[inside], ref.shape), psf_sigma_px)
    a = _band(rendering, band_low_px, band_high_px)
    b = _band(ref, band_low_px, band_high_px)
    corr = np.fft.irfft2(np.fft.rfft2(b) * np.conj(np.fft.rfft2(a)),
                         s=ref.shape) / ref.size
    corr = np.fft.fftshift(corr)
    cy, cx = rows // 2, cols // 2
    win = int(min(max_shift_px, cy - 1, cx - 1))
    sub = corr[cy - win:cy + win + 1, cx - win:cx + win + 1]
    i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    peak = float(sub[i, j])
    fy = (_parabola(sub[i - 1, j], peak, sub[i + 1, j])
          if 0 < i < 2 * win else 0.0)
    fx = (_parabola(sub[i, j - 1], peak, sub[i, j + 1])
          if 0 < j < 2 * win else 0.0)
    dy, dx = i - win + fy, j - win + fx

    e = RUNNER_UP_EXCLUSION_PX
    local = sub == ndimage.maximum_filter(sub, size=2 * e + 1)
    li, lj = np.nonzero(local)
    separate = np.hypot(li - i, lj - j) > e
    runner = (float(np.max(sub[li[separate], lj[separate]]))
              if np.any(separate) else None)
    median = float(np.median(sub))
    spread = 1.4826 * float(np.median(np.abs(sub - median)))
    score = (peak - median) / spread if spread > 0 else None

    if i in (0, 2 * win) or j in (0, 2 * win):
        warnings.append(
            f"The best shift lies at the limit of the search (±{win} px): "
            f"the images may be further apart, or may not show the same "
            f"field.")
    if peak <= 0:
        warnings.append(
            "The localizations do not resemble the widefield image at any "
            "shift. Is it an image of the same protein and field?")
    elif runner is not None and runner >= RUNNER_UP_FRACTION * peak:
        warnings.append(
            f"Another shift fits nearly as well (correlation {runner:.2f} "
            f"against {peak:.2f}): the registration is ambiguous. Check it "
            f"on the overlay.")
    elif score is not None and score < MIN_REGISTRATION_SCORE:
        warnings.append(
            f"The best shift barely stands out from the rest of the search "
            f"({score:.1f} robust standard deviations; correct "
            f"registrations score above {MIN_REGISTRATION_SCORE:.0f}). Use "
            f"all the localizations of the movie, and a widefield image of "
            f"the same protein; check the overlay.")
    return ImageRegistration(
        shift_px=(float(dx), float(dy)), source="measured", peak=peak,
        zero=float(corr[cy, cx]), runner_up=runner, score=score,
        n_localizations=int(inside.sum()), warnings=warnings,
    )


# ===================================================================
#  The mask
# ===================================================================
def otsu_threshold(values: NDArray[np.float64], bins: int = 256) -> float:
    """
    Otsu's threshold (Otsu 1979): the cut that maximises the between-class
    variance of the histogram. Values above it are the foreground.

    Where the variance is flat over a run of empty bins -- two levels with
    nothing between them, as in a sharp-edged image -- the middle of the
    run is taken, not its first bin, which would sit on the lower level.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    if v.min() == v.max():
        return float(v.min())
    hist, edges = np.histogram(v, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centres)
    total = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = m0 / w0
        mu1 = (total - m0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between[:-1], nan=-1.0)
    best = float(np.max(between))
    ties = np.nonzero(between >= best * (1 - 1e-9))[0]
    # The first run of maxima, as indices into the bin edges.
    run = ties[: int(np.argmax(np.diff(np.append(ties, ties[-1] + 2)) > 1)) + 1]
    return float(0.5 * (edges[run[0] + 1] + edges[run[-1] + 1]))


@dataclass
class AxoplasmMask:
    """The axoplasm around one axon, and the signed distance to its edge."""

    threshold: float
    threshold_source: str               # "otsu" | "manual"
    otsu: float
    smooth_sigma_px: float
    region: Tuple[int, int, int, int]   # row0, row1, col0, col1 (image px)
    upsample: int
    pixel_size_nm: float
    ring_radius_px: float
    mask: NDArray[np.bool_]             # on the fine grid
    signed_distance_nm: NDArray[np.float64]
    smoothed: NDArray[np.float64]       # the region, smoothed, image px
    warnings: List[str] = field(default_factory=list)

    @property
    def area_um2(self) -> float:
        step = self.pixel_size_nm / self.upsample
        return float(self.mask.sum()) * (step / 1000.0) ** 2

    @property
    def ring_area_um2(self) -> float:
        """Area inside the spectrin ring, from its median radius."""
        return math.pi * (self.ring_radius_px * self.pixel_size_nm
                          / 1000.0) ** 2

    @property
    def area_ratio(self) -> Optional[float]:
        ring = self.ring_area_um2
        return None if ring <= 0 else self.area_um2 / ring

    @property
    def empty(self) -> bool:
        return not bool(self.mask.any())

    def distance_at(self, col: NDArray[np.float64], row: NDArray[np.float64]
                    ) -> NDArray[np.float64]:
        """Signed distance, nm, at image coordinates; NaN off the region."""
        col = np.asarray(col, float)
        row = np.asarray(row, float)
        r0, r1, c0, c1 = self.region
        up = self.upsample
        fr = (row - r0 + 0.5) * up - 0.5
        fc = (col - c0 + 0.5) * up - 0.5
        rows, cols = self.signed_distance_nm.shape
        ok = (fr >= -0.5) & (fr <= rows - 0.5) & (fc >= -0.5) & (fc <= cols - 0.5)
        out = np.full(col.shape, np.nan)
        if self.empty:
            # No axoplasm: everything on the region is outside it, by an
            # unbounded distance.
            out[ok] = -np.inf
        elif np.any(ok):
            out[ok] = ndimage.map_coordinates(
                self.signed_distance_nm, [fr[ok], fc[ok]], order=1,
                mode="nearest")
        return out


def build_mask(
    image: NDArray[np.float64],
    centre: Tuple[float, float],
    radius_px: float,
    pixel_size_nm: float,
    *,
    reach_px: Optional[float] = None,
    threshold: Optional[float] = None,
    smooth_sigma_px: float = DEFAULT_SMOOTH_SIGMA_PX,
    upsample: int = DEFAULT_UPSAMPLE,
) -> AxoplasmMask:
    """
    Threshold the tubulin image around one axon.

    ``centre`` is (column, row) in image pixels and ``radius_px`` the
    axon's radius. The region reaches ``REGION_RADIUS_FACTOR`` radii from
    the centre, and at least ``reach_px`` plus a margin, so that it holds
    every localization of the selection. Its smoothed intensities are traced on a grid
    ``upsample`` times finer than the camera's, so the edge is not
    staircased at 113 nm. The connected area over the threshold that
    contains the centre is kept, with its holes filled. Without a
    ``threshold``, Otsu's value for the region is used.
    """
    img = np.asarray(image, dtype=np.float64)
    rows, cols = img.shape
    ccol, crow = float(centre[0]), float(centre[1])
    half = max(REGION_RADIUS_FACTOR * radius_px,
               radius_px + REGION_MIN_MARGIN_PX,
               (reach_px or 0.0) + REGION_MIN_MARGIN_PX)
    r0 = max(0, int(math.floor(crow - half)))
    r1 = min(rows, int(math.ceil(crow + half)) + 1)
    c0 = max(0, int(math.floor(ccol - half)))
    c1 = min(cols, int(math.ceil(ccol + half)) + 1)
    if r1 - r0 < 3 or c1 - c0 < 3:
        raise ValueError(
            "The axon lies outside the tubulin image: check the alignment.")
    sub = img[r0:r1, c0:c1]
    smoothed = (ndimage.gaussian_filter(sub, smooth_sigma_px)
                if smooth_sigma_px > 0 else sub.copy())
    otsu = otsu_threshold(smoothed)
    cut = otsu if threshold is None else float(threshold)
    warnings: List[str] = []

    nr, nc = smoothed.shape
    fine_r = (np.arange(nr * upsample) + 0.5) / upsample - 0.5
    fine_c = (np.arange(nc * upsample) + 0.5) / upsample - 0.5
    grid_r, grid_c = np.meshgrid(fine_r, fine_c, indexing="ij")
    fine = ndimage.map_coordinates(smoothed, [grid_r, grid_c], order=1,
                                   mode="nearest")
    labels, _count = ndimage.label(fine > cut)
    seed_r = int(np.clip(round((crow - r0 + 0.5) * upsample - 0.5), 0,
                         nr * upsample - 1))
    seed_c = int(np.clip(round((ccol - c0 + 0.5) * upsample - 0.5), 0,
                         nc * upsample - 1))
    seed = int(labels[seed_r, seed_c])
    if seed == 0 and labels.max() > 0:
        # The centre itself is below the cut: take the largest region
        # overlapping the axon, and say so.
        near = np.hypot((grid_r - (crow - r0)), (grid_c - (ccol - c0))) \
            <= radius_px
        counts = np.bincount(labels[near].ravel(),
                             minlength=int(labels.max()) + 1)
        counts[0] = 0
        if counts.max() > 0:
            seed = int(np.argmax(counts))
            warnings.append(
                "The axon's centre is below the threshold: the largest "
                "region over it inside the axon was taken instead. Check "
                "the mask.")
    mask = ndimage.binary_fill_holes(labels == seed) if seed else \
        np.zeros(fine.shape, dtype=bool)
    if not mask.any():
        warnings.append(
            "No axoplasm was found over the threshold: every localization "
            "is classified as membrane. Lower the threshold, or check the "
            "alignment.")
    elif (mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any()
          or mask[:, -1].any()):
        warnings.append(
            "The mask reaches the edge of the analysed region, where it is "
            "cut: it probably includes a neighbouring axon or background. "
            "Raise the threshold.")

    step = pixel_size_nm / upsample
    ring_area = math.pi * (radius_px * pixel_size_nm / 1000.0) ** 2
    if mask.any() and ring_area > 0:
        ratio = float(mask.sum()) * (step / 1000.0) ** 2 / ring_area
        if ratio > AREA_RATIO_WARN[1]:
            warnings.append(
                f"The mask covers {ratio:.1f} times the area inside the "
                f"spectrin ring: it probably takes in a neighbouring axon. "
                f"Raise the threshold.")
        elif ratio < AREA_RATIO_WARN[0]:
            warnings.append(
                f"The mask covers only {ratio:.0%} of the area inside the "
                f"spectrin ring. Lower the threshold, or check the "
                f"alignment.")
    if mask.any():
        # The padding makes the region's border count as outside.
        padded = np.pad(mask, 1, constant_values=False)
        inside = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
        outside = ndimage.distance_transform_edt(~padded)[1:-1, 1:-1]
        # Distances run between fine-pixel centres; the edge lies half a
        # step from the last pixel on either side.
        signed = np.where(mask, inside - 0.5, -(outside - 0.5)) * step
    else:
        signed = np.full(mask.shape, -np.inf)
    return AxoplasmMask(
        threshold=cut, threshold_source="otsu" if threshold is None else "manual",
        otsu=otsu, smooth_sigma_px=float(smooth_sigma_px),
        region=(r0, r1, c0, c1), upsample=int(upsample),
        pixel_size_nm=float(pixel_size_nm), ring_radius_px=float(radius_px),
        mask=mask,
        signed_distance_nm=np.asarray(signed, dtype=np.float64),
        smoothed=smoothed, warnings=warnings,
    )


# ===================================================================
#  Classification
# ===================================================================
@dataclass
class Classification:
    """Each localization's place relative to the axoplasm."""

    distance_nm: NDArray[np.float64]     # signed, positive inside; NaN off
    labels: NDArray[np.object_]
    margin_nm: float

    def count(self, label: str) -> int:
        return int(np.sum(self.labels == label))

    @property
    def n(self) -> int:
        return int(self.labels.size)

    @property
    def fraction_interior(self) -> Optional[float]:
        classified = self.n - self.count(LABEL_OUTSIDE)
        if classified == 0:
            return None
        return self.count(LABEL_INTERIOR) / classified


def classify(mask: AxoplasmMask, col: NDArray[np.float64],
             row: NDArray[np.float64], margin_nm: float) -> Classification:
    """
    Interior when deeper inside the axoplasm than ``margin_nm``, membrane
    otherwise; localizations off the analysed region are labelled apart.
    """
    distance = mask.distance_at(col, row)
    labels = np.full(distance.shape, LABEL_MEMBRANE, dtype=object)
    labels[np.isnan(distance)] = LABEL_OUTSIDE
    with np.errstate(invalid="ignore"):
        labels[distance > float(margin_nm)] = LABEL_INTERIOR
    return Classification(distance_nm=distance, labels=labels,
                          margin_nm=float(margin_nm))


def axon_centre(col: NDArray[np.float64], row: NDArray[np.float64]
                ) -> Tuple[Tuple[float, float], float, float]:
    """
    Median centre of the selection, the median distance to it (the
    radius of a ring) and the farthest distance, all in pixels.
    """
    col = np.asarray(col, float)
    row = np.asarray(row, float)
    ccol, crow = float(np.median(col)), float(np.median(row))
    distance = np.hypot(col - ccol, row - crow)
    return ((ccol, crow), float(np.median(distance)),
            float(distance.max()) if distance.size else 0.0)


# ===================================================================
#  Export
# ===================================================================
def summary_row(
    *,
    localizations: str,
    tubulin: str,
    reference: str,
    registration_file: str,
    roi: str,
    pixel_size_nm: float,
    offset_px: Tuple[float, float],
    registration: ImageRegistration,
    mask: AxoplasmMask,
    result: Classification,
    cluster_result: Optional[Classification],
) -> Dict[str, Any]:
    """One row per axon; the same columns whatever was computed."""
    sx, sy = registration.shift_nm(pixel_size_nm)

    def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
        return None if value is None else round(float(value), digits)

    return {
        "source_localizations": localizations,
        "tubulin_image": tubulin,
        "registration_reference_image": reference,
        "registration_localizations": registration_file,
        "roi": roi,
        "pixel_size_nm": pixel_size_nm,
        "camera_offset_x_px": offset_px[0],
        "camera_offset_y_px": offset_px[1],
        "registration_source": registration.source,
        "shift_x_nm": round(sx, 1),
        "shift_y_nm": round(sy, 1),
        "registration_peak": rounded(registration.peak),
        "registration_zero_shift": rounded(registration.zero),
        "registration_runner_up": rounded(registration.runner_up),
        "registration_score": rounded(registration.score, 2),
        "threshold": round(mask.threshold, 3),
        "threshold_source": mask.threshold_source,
        "otsu_threshold": round(mask.otsu, 3),
        "smooth_sigma_nm": round(mask.smooth_sigma_px * pixel_size_nm, 1),
        "margin_nm": result.margin_nm,
        "mask_area_um2": round(mask.area_um2, 4),
        "ring_area_um2": round(mask.ring_area_um2, 4),
        "n_localizations": result.n,
        "n_interior": result.count(LABEL_INTERIOR),
        "n_membrane": result.count(LABEL_MEMBRANE),
        "n_outside_region": result.count(LABEL_OUTSIDE),
        "fraction_interior": rounded(result.fraction_interior),
        "n_clusters": None if cluster_result is None else cluster_result.n,
        "n_clusters_interior": (None if cluster_result is None
                                else cluster_result.count(LABEL_INTERIOR)),
        "n_clusters_membrane": (None if cluster_result is None
                                else cluster_result.count(LABEL_MEMBRANE)),
        "n_warnings": len(registration.warnings) + len(mask.warnings),
    }
