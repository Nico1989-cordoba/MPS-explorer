# -*- coding: utf-8 -*-
"""
Where is the axoplasm? A widefield betaIII-tubulin image as a mask for the
betaII-spectrin localizations.

betaIII-tubulin fills the axoplasm, while betaII-spectrin anchored to the
membrane forms the periodic lattice at its rim. A widefield tubulin image
taken with the super-resolution acquisition therefore tells which spectrin
localizations lie inside the axon and which lie at its edge. That holds
only if the image is placed where the localizations are, and if its
blurred edge is treated as blurred. The module works in four steps, each
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

3. THE DISTANCE TO THE MASK. Every localization gets its signed distance
   to the mask's edge, positive inside (``classify``). The tubulin alone
   does not decide what is inside: where the mask drifts off the spectrin
   ring it puts membrane localizations inside, which on axon 7 of April
   was most of what it put there. A localization is counted inside only
   through its cluster, in step 4 (``localization_labels``).

4. THE CLUSTERS NOT ANCHORED TO THE MEMBRANE. The widefield betaII-spectrin
   image shows the membrane as a blurred bright ring around a dark inside
   (``build_ring_interior``). A cluster of the MPS analysis is discarded
   only when both images put it inside the axon by more than the margin,
   and the contour is rebuilt from the rest (``anchored_clusters``).
   Either image alone misplaces clusters in its own way -- the tubulin
   mask merges with its neighbours and drifts off the ring, the spectrin
   ring opens where it is dim -- which is why the two must agree. Where
   the spectrin interior stops short of the axon's middle (at the spill
   point of a ring that is open), a cluster there cannot be discarded
   whatever the tubulin says. tools.mps_analysis.without_clusters then
   repeats every parameter of the MPS analysis without them.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from tools import mps_metadata, mps_pixel_size
from tools.mps_geometry import PerimeterResult, reconstruct_perimeter
from tools.results_table import cell_text

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
# The margin the panel opens with: how far inside a widefield edge a point
# must be before the image is trusted to put it inside. The edge is blurred
# by the diffraction limit, so without a margin a ring on the membrane falls
# half inside. 250 nm is the value the Axoplasm panel was first used with on
# the April data; the user sets it, and it is exported with every result.
DEFAULT_MARGIN_NM = 250.0
# An image whose recorded pixel size differs from the localizations' by
# this much or more is refused: that is another scale altogether (another
# binning, camera, or a unit written wrong), and placing it would be
# meaningless. A few percent is reported instead and the image is placed,
# because which of the two values is right is often unsettled -- the 2023
# data records 133 nm in its widefield images and is analysed with 135.
PIXEL_SIZE_REFUSE = 0.10

# Where the tubulin mask alone puts a localization (classify): only the
# distance to its edge is reported, never these labels as "inside".
LABEL_INTERIOR = "interior"
LABEL_MEMBRANE = "membrane"
LABEL_OUTSIDE = "outside region"

# Where a localization is, through its cluster (localization_labels): inside
# when its cluster is one both widefield images put inside the axon (a
# discarded one), at the membrane when its cluster is kept, and in no
# cluster when DBSCAN left it out or its cluster was curated away.
LOC_INSIDE = "inside"
LOC_MEMBRANE = "membrane"
LOC_NO_CLUSTER = "no cluster"


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
    # Drawn from localizations acquired together with the movie, on the
    # movie's own camera-pixel grid, rather than read from a widefield
    # image taken at another moment. Such an image is placed on the
    # localizations by construction: no camera offset, no shift.
    from_localizations: bool = False
    n_localizations: int = 0

    @property
    def shape(self) -> Tuple[int, int]:
        rows, cols = self.image.shape
        return int(rows), int(cols)

    @property
    def source(self) -> str:
        """What the image is, in the words the table uses."""
        return "localizations" if self.from_localizations else "widefield image"


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


def image_from_localizations(
    path: str,
    movie_pixel_nm: float,
    *,
    min_extent_px: Tuple[float, float] = (0.0, 0.0),
) -> WidefieldImage:
    """
    An image of localizations acquired together with the movie.

    Each localization is counted in the camera pixel it was localized in
    -- its own pixel coordinates, the file's nm divided by the file's own
    pixel size -- so the image sits on the movie's camera grid exactly as
    a pixel of the movie does, and a position of the movie in nm maps to
    it by the movie's pixel size alone. Counting by the file's own pixel
    and not by nm matters: two channels of one dual-view camera are
    registered to each other in camera pixels, and the 2023 data carries
    files of one acquisition localized with 133 and with 135 nm. Placing
    them by nm would move one against the other by 1.5 % of the distance
    from the origin -- 225 nm at 15 um, the size of the default margin.

    Every localization is counted, whatever its z, as a widefield image
    would: a microtubule runs along the axon, and the cross-section it
    draws is the same through the section.

    ``min_extent_px`` makes the image at least that large (columns,
    rows), so that every position of the movie falls inside it.
    """
    from tools import mps_io

    loc = mps_io.load_localizations(path)
    own = loc.pixel_size_nm
    notes: List[str] = []
    if own is None:
        # Already in nm (a CSV): the movie's pixel is the only grid there
        # is, and the file is taken to share its frame.
        own = float(movie_pixel_nm)
        notes.append(
            f"{os.path.basename(path)} is in nanometres with no pixel size "
            f"of its own; it is counted on the movie's {own:g} nm grid.")
    elif abs(float(own) - float(movie_pixel_nm)) > 1e-6:
        notes.append(
            f"{os.path.basename(path)} was localized with a {float(own):g} "
            f"nm pixel and the movie with {float(movie_pixel_nm):g} nm. The "
            f"mask is placed by camera pixel, which is how the two halves of "
            f"one camera are registered, so this does not move it; but one "
            f"of the two pixel sizes is wrong, and every distance in nm of "
            f"that channel with it.")
    x_px = np.asarray(loc.x_nm, dtype=float) / float(own)
    y_px = np.asarray(loc.y_nm, dtype=float) / float(own)
    if x_px.size == 0:
        raise ValueError(f"{os.path.basename(path)} holds no localizations.")
    cols = int(np.ceil(max(float(np.max(x_px)), min_extent_px[0]))) + 2
    rows = int(np.ceil(max(float(np.max(y_px)), min_extent_px[1]))) + 2
    image = render_counts(x_px, y_px, (rows, cols))
    return WidefieldImage(
        path=str(path), image=image, n_planes=1, camera_region=None,
        binning=None, pixel_size_nm=float(movie_pixel_nm), acquired=None,
        pixel_size_source="the movie's", notes=notes,
        from_localizations=True, n_localizations=int(x_px.size))


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
    # The intensity levels a spectrin interior was cut at (see
    # build_ring_interior); empty for a tubulin mask.
    levels: Dict[str, float] = field(default_factory=dict)

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
    ccol, crow = float(centre[0]), float(centre[1])
    r0, r1, c0, c1 = _region_bounds(img.shape, centre, radius_px, reach_px,
                                    "tubulin")
    sub = img[r0:r1, c0:c1]
    smoothed = (ndimage.gaussian_filter(sub, smooth_sigma_px)
                if smooth_sigma_px > 0 else sub.copy())
    otsu = otsu_threshold(smoothed)
    cut = otsu if threshold is None else float(threshold)
    warnings: List[str] = []

    nr, nc = smoothed.shape
    grid_r, grid_c, fine = _fine_grid(smoothed, upsample)
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
    return AxoplasmMask(
        threshold=cut, threshold_source="otsu" if threshold is None else "manual",
        otsu=otsu, smooth_sigma_px=float(smooth_sigma_px),
        region=(r0, r1, c0, c1), upsample=int(upsample),
        pixel_size_nm=float(pixel_size_nm), ring_radius_px=float(radius_px),
        mask=mask,
        signed_distance_nm=_signed_distance(mask, step),
        smoothed=smoothed, warnings=warnings,
    )


def _region_bounds(shape: Tuple[int, ...], centre: Tuple[float, float],
                   radius_px: float, reach_px: Optional[float],
                   what: str) -> Tuple[int, int, int, int]:
    """
    Rows and columns of the region analysed around one axon: it reaches
    REGION_RADIUS_FACTOR radii from the centre, and at least ``reach_px``
    plus a margin, so it holds every localization of the selection.
    """
    rows, cols = shape[0], shape[1]
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
            f"The axon lies outside the {what} image: check the alignment.")
    return r0, r1, c0, c1


def _fine_grid(smoothed: NDArray[np.float64], upsample: int
               ) -> Tuple[NDArray[np.float64], NDArray[np.float64],
                          NDArray[np.float64]]:
    """The region on a grid ``upsample`` times finer than the camera's."""
    nr, nc = smoothed.shape
    fine_r = (np.arange(nr * upsample) + 0.5) / upsample - 0.5
    fine_c = (np.arange(nc * upsample) + 0.5) / upsample - 0.5
    grid_r, grid_c = np.meshgrid(fine_r, fine_c, indexing="ij")
    fine = ndimage.map_coordinates(smoothed, [grid_r, grid_c], order=1,
                                   mode="nearest")
    return grid_r, grid_c, fine


def _signed_distance(mask: NDArray[np.bool_], step: float
                     ) -> NDArray[np.float64]:
    """Distance to the mask's edge, nm, positive inside."""
    if not mask.any():
        return np.full(mask.shape, -np.inf)
    # The padding makes the region's border count as outside.
    padded = np.pad(mask, 1, constant_values=False)
    inside = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    outside = ndimage.distance_transform_edt(~padded)[1:-1, 1:-1]
    # Distances run between fine-pixel centres; the edge lies half a step
    # from the last pixel on either side.
    signed = np.where(mask, inside - 0.5, -(outside - 0.5)) * step
    return np.asarray(signed, dtype=np.float64)


# ===================================================================
#  The interior the spectrin ring encloses
# ===================================================================
# Bisection steps for the spill level: 2**-40 of the ring's contrast.
SPILL_BISECTIONS = 40


def build_ring_interior(
    image: NDArray[np.float64],
    centre: Tuple[float, float],
    radius_px: float,
    pixel_size_nm: float,
    *,
    ring_col: NDArray[np.float64],
    ring_row: NDArray[np.float64],
    reach_px: Optional[float] = None,
    smooth_sigma_px: float = DEFAULT_SMOOTH_SIGMA_PX,
    upsample: int = DEFAULT_UPSAMPLE,
) -> AxoplasmMask:
    """
    The dark interior the betaII-spectrin ring encloses in a widefield image.

    Spectrin at the membrane shows in widefield as a blurred bright ring,
    and the axon's inside as the dark area it encloses. No threshold for
    the whole region can find it: the neighbouring axons' rings are often
    brighter than this one's, and a cut above this ring opens it. So the
    levels come from the axon itself:

    ring level      the median of the smoothed image under the axon's own
                    cluster centres (``ring_col``, ``ring_row``, image
                    pixels): where the super-resolved spectrin actually is;
    interior level  the darkest point inside the convex hull of those
                    centres;
    edge            half-way between the two, the half maximum of a blurred
                    step, where its edge is for a symmetric blur.

    The interior is the dark area around that darkest point, below the
    edge level -- but never outside the convex hull of the centres. Flooded
    any higher it would leak through a gap in the ring into the myelin,
    which is as dark as the inside of an axon, so it is flooded only up to
    the level at which it would leave the hull (its spill point) when that
    comes first, and says so. With the hull as its bound, a cluster on the
    ring can never be deep inside the interior.

    Returns an ``AxoplasmMask`` on the same region as ``build_mask`` with
    ``levels`` = interior, ring, half_max, spill, cut. An empty mask comes
    back, with the reason in ``warnings``, when there are fewer than three
    cluster centres or they are collinear, or when they sit no brighter
    than the darkest point inside them.
    """
    img = np.asarray(image, dtype=np.float64)
    r0, r1, c0, c1 = _region_bounds(img.shape, centre, radius_px, reach_px,
                                    "spectrin widefield")
    sub = img[r0:r1, c0:c1]
    smoothed = (ndimage.gaussian_filter(sub, smooth_sigma_px)
                if smooth_sigma_px > 0 else sub.copy())
    grid_r, grid_c, fine = _fine_grid(smoothed, upsample)
    step = pixel_size_nm / upsample
    warnings: List[str] = []
    levels: Dict[str, float] = {}

    def result(mask: NDArray[np.bool_], cut: float, source: str
               ) -> AxoplasmMask:
        return AxoplasmMask(
            threshold=cut, threshold_source=source, otsu=float("nan"),
            smooth_sigma_px=float(smooth_sigma_px), region=(r0, r1, c0, c1),
            upsample=int(upsample), pixel_size_nm=float(pixel_size_nm),
            ring_radius_px=float(radius_px), mask=mask,
            signed_distance_nm=_signed_distance(mask, step),
            smoothed=smoothed, warnings=warnings, levels=levels)

    empty = np.zeros(fine.shape, dtype=bool)
    col = np.asarray(ring_col, dtype=float) - c0
    row = np.asarray(ring_row, dtype=float) - r0
    ok = np.isfinite(col) & np.isfinite(row)
    col, row = col[ok], row[ok]
    try:
        if col.size < 3:
            raise QhullError("fewer than three points")
        hull = ConvexHull(np.column_stack([col, row]))
    except (QhullError, ValueError):
        warnings.append(
            "The spectrin interior needs at least three cluster centres "
            "that are not on one line; there are not.")
        return result(empty, float("nan"), "none")

    # The hull's facets: normal . (col, row) + offset <= 0 inside.
    normals, offsets = hull.equations[:, :2], hull.equations[:, 2]
    points = np.stack([grid_c, grid_r], axis=-1)
    in_hull = np.all(points @ normals.T + offsets <= 1e-9, axis=-1)
    if not in_hull.any():
        warnings.append("The cluster centres enclose no area on the image.")
        return result(empty, float("nan"), "none")

    ring = float(np.median(ndimage.map_coordinates(
        smoothed, [row, col], order=1, mode="nearest")))
    seed_r, seed_c = np.unravel_index(
        int(np.argmin(np.where(in_hull, fine, np.inf))), fine.shape)
    interior = float(fine[seed_r, seed_c])
    half_max = 0.5 * (interior + ring)
    levels.update(interior=interior, ring=ring, half_max=half_max)
    if ring <= interior:
        warnings.append(
            "The axon's cluster centres sit no brighter than the darkest "
            "point inside them: the spectrin image shows no ring here.")
        levels.update(spill=float("nan"), cut=float("nan"))
        return result(empty, float("nan"), "none")

    def basin(cut: float) -> NDArray[np.bool_]:
        labels, _count = ndimage.label(fine < cut)
        seed = labels[seed_r, seed_c]
        return (labels == seed) if seed else empty

    def spills(area: NDArray[np.bool_]) -> bool:
        return bool((area & ~in_hull).any() or area[0, :].any()
                    or area[-1, :].any() or area[:, 0].any()
                    or area[:, -1].any())

    if not spills(basin(ring)):
        spill = ring
    else:
        low, high = interior, ring
        for _ in range(SPILL_BISECTIONS):
            middle = 0.5 * (low + high)
            if spills(basin(middle)):
                high = middle
            else:
                low = middle
        spill = low
    cut = min(half_max, spill)
    levels.update(spill=spill, cut=cut)
    source = "half maximum"
    if spill < half_max:
        source = "spill point"
        reached = (spill - interior) / (half_max - interior)
        warnings.append(
            f"The spectrin ring is open below its half maximum: the "
            f"interior was flooded only {reached:.0%} of the way there, up "
            f"to where it would have left the area the clusters enclose. "
            f"It is smaller than the axon's inside, so fewer clusters can "
            f"be found inside it.")
    mask = ndimage.binary_fill_holes(basin(cut))
    return result(mask, cut, source)


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
    Each localization's signed distance to the mask's edge, and where the
    mask alone would put it: interior when deeper inside the axoplasm than
    ``margin_nm``, membrane otherwise, and apart when off the analysed
    region. The panel reports the distances but counts a localization
    inside only through its cluster (localization_labels).
    """
    distance = mask.distance_at(col, row)
    labels = np.full(distance.shape, LABEL_MEMBRANE, dtype=object)
    labels[np.isnan(distance)] = LABEL_OUTSIDE
    with np.errstate(invalid="ignore"):
        labels[distance > float(margin_nm)] = LABEL_INTERIOR
    return Classification(distance_nm=distance, labels=labels,
                          margin_nm=float(margin_nm))


def cluster_of_points(
    x_nm: NDArray[np.float64], y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    ref_x_nm: NDArray[np.float64], ref_y_nm: NDArray[np.float64],
    ref_z_nm: NDArray[np.float64], ref_cluster: NDArray[np.intp],
) -> NDArray[np.intp]:
    """
    For each point, the cluster of the same point among the reference
    ones -- the localizations the MPS analysis clustered -- or -1 when it
    is not among them. Points are matched by their exact coordinates:
    both sets come from the same file, so a localization has the same
    three numbers in each.
    """
    def rows(x: Any, y: Any, z: Any) -> NDArray[Any]:
        table = np.ascontiguousarray(np.column_stack(
            [np.asarray(x, float).ravel(), np.asarray(y, float).ravel(),
             np.asarray(z, float).ravel()]))
        return table.view(np.dtype((np.void, table.dtype.itemsize * 3))
                          ).ravel()

    ref = rows(ref_x_nm, ref_y_nm, ref_z_nm)
    ref_cluster = np.asarray(ref_cluster, dtype=np.intp).ravel()
    if ref.size != ref_cluster.size:
        raise ValueError("One cluster per reference point is needed.")
    points = rows(x_nm, y_nm, z_nm)
    out = np.full(points.size, -1, dtype=np.intp)
    if ref.size == 0 or points.size == 0:
        return out
    order = np.argsort(ref, kind="stable")
    ordered = ref[order]
    at = np.searchsorted(ordered, points)
    found = at < ordered.size
    found[found] = ordered[at[found]] == points[found]
    out[found] = ref_cluster[order[at[found]]]
    return out


def localization_labels(cluster_of: NDArray[np.intp],
                        discarded: NDArray[np.bool_]) -> NDArray[np.object_]:
    """
    Where each localization is, through its cluster: LOC_INSIDE when its
    cluster is one both images put inside the axon (``discarded``, one
    flag per kept cluster), LOC_MEMBRANE when its cluster is kept, and
    LOC_NO_CLUSTER when it belongs to none (``cluster_of`` -1).
    """
    index = np.asarray(cluster_of, dtype=np.intp).ravel()
    flags = np.asarray(discarded, dtype=bool).ravel()
    labels = np.full(index.shape, LOC_NO_CLUSTER, dtype=object)
    member = (index >= 0) & (index < flags.size)
    inside = np.zeros(index.shape, dtype=bool)
    inside[member] = flags[index[member]]
    labels[member & ~inside] = LOC_MEMBRANE
    labels[inside] = LOC_INSIDE
    return labels


@dataclass
class AnchoredClusters:
    """
    Which betaII-spectrin clusters are not anchored to the membrane, and the
    contour of the ones that are.

    A cluster is discarded only when BOTH widefield images put its centre
    inside the axon by more than the margin: deeper than ``margin_nm``
    inside the tubulin mask AND inside the spectrin ring's dark interior.
    Either image alone fails in its own way -- the tubulin mask merges with
    neighbours and drifts off the ring, the spectrin ring opens where it is
    dim -- and a cluster is kept whenever they disagree. A cluster off
    either analysed region is kept.
    """

    depth_tubulin_nm: NDArray[np.float64]    # signed, positive inside
    depth_spectrin_nm: NDArray[np.float64]
    margin_nm: float
    discarded: NDArray[np.bool_]
    # All clusters, connected as the MPS analysis connects them; all
    # clusters with 2-opt from every start -- the same contour, unless the
    # analysis was refined from one start, as it was before 2026-09-19; and
    # the anchored ones with 2-opt from every start. The last two differ
    # ONLY by the discarded clusters, so they measure what discarding them
    # does.
    contour_all: Optional[PerimeterResult]
    contour_all_starts: Optional[PerimeterResult]
    contour_anchored: Optional[PerimeterResult]
    warnings: List[str] = field(default_factory=list)
    # How the widefield images were placed on the localizations when this
    # was found ("measured, score 13.5", "set by hand", ...). Which
    # clusters are inside depends on it entirely -- on axon 7, 5 of 94
    # after the shift was measured against 4 of 94 before -- so it travels
    # with the discard into every table that reports it.
    registration: str = ""

    @property
    def n(self) -> int:
        return int(self.discarded.size)

    @property
    def n_discarded(self) -> int:
        return int(self.discarded.sum())

    def _inside(self, depth: NDArray[np.float64]) -> NDArray[np.bool_]:
        with np.errstate(invalid="ignore"):
            return np.asarray(depth > self.margin_nm)

    # The clusters fall in four groups, one of each per cluster: inside
    # both images (discarded), inside the tubulin mask only, inside the
    # spectrin interior only, and inside neither (a cluster off an image's
    # analysed region counts as not inside it).
    @property
    def tubulin_only(self) -> NDArray[np.bool_]:
        """Inside by the tubulin mask only: kept."""
        return (self._inside(self.depth_tubulin_nm)
                & ~self._inside(self.depth_spectrin_nm))

    @property
    def spectrin_only(self) -> NDArray[np.bool_]:
        """Inside the spectrin interior only: kept."""
        return (self._inside(self.depth_spectrin_nm)
                & ~self._inside(self.depth_tubulin_nm))

    @property
    def inside_neither(self) -> NDArray[np.bool_]:
        """Inside neither image: kept, on the membrane."""
        return (~self._inside(self.depth_tubulin_nm)
                & ~self._inside(self.depth_spectrin_nm))

    @property
    def n_tubulin_only(self) -> int:
        return int(self.tubulin_only.sum())

    @property
    def n_spectrin_only(self) -> int:
        return int(self.spectrin_only.sum())


def anchored_clusters(
    centroids_nm: NDArray[np.float64],
    tubulin: AxoplasmMask,
    tubulin_col: NDArray[np.float64],
    tubulin_row: NDArray[np.float64],
    spectrin: AxoplasmMask,
    spectrin_col: NDArray[np.float64],
    spectrin_row: NDArray[np.float64],
    margin_nm: float,
    contour_all: Optional[PerimeterResult] = None,
    contour_cache: Optional[Dict[bytes, PerimeterResult]] = None,
    registration: str = "",
) -> AnchoredClusters:
    """
    Discard the clusters both images put inside the axon, and rebuild the
    contour with 2-opt on the rest.

    ``centroids_nm`` are the kept cluster centres of the MPS analysis; the
    column/row pairs are the same centres in each image's pixels (the two
    images can sit at different camera offsets). ``contour_all`` is the
    contour the MPS analysis built from all of them; without it, it is
    rebuilt the way the analysis builds it, with 2-opt from every start.

    The contour of the anchored clusters is built the same way, keeping
    the shortest tour over every starting point, so that it does not
    depend on the start. When ``contour_all`` came from one start, the
    contour of all the clusters is rebuilt from every start too, for the
    comparison; otherwise it is ``contour_all`` itself. ``contour_cache``
    (keyed by the centres' bytes) saves rebuilding them when only the
    margin moved and the same clusters stay.
    """
    centroids = np.asarray(centroids_nm, dtype=float).reshape(-1, 2)
    depth_t = tubulin.distance_at(tubulin_col, tubulin_row)
    depth_s = spectrin.distance_at(spectrin_col, spectrin_row)
    margin = float(margin_nm)
    with np.errstate(invalid="ignore"):
        discarded = np.asarray((depth_t > margin) & (depth_s > margin))
    warnings: List[str] = []
    cache: Dict[bytes, PerimeterResult] = (
        {} if contour_cache is None else contour_cache)

    # A contour drawn by hand travels with the analysis' contour; the
    # clusters this panel keeps are joined along the same path, so the
    # contour it draws and the one the "_discard" columns are measured on
    # are the one the person drew, not one the program built instead.
    guide = None if contour_all is None else contour_all.guide

    def shortest(points: NDArray[np.float64]) -> Optional[PerimeterResult]:
        if len(points) < 3:
            return None
        key = np.ascontiguousarray(points).tobytes() + (
            b"" if guide is None else np.ascontiguousarray(guide).tobytes())
        if key not in cache:
            cache[key] = reconstruct_perimeter(points, all_starts=True,
                                               guide=guide)
        return cache[key]

    if (contour_all is not None
            # A hand-set order runs no 2-opt, so it has no starts
            # to count; it is still the contour to reuse.
            and (contour_all.n_starts > 1
                 or contour_all.order_source != "automatic")
            and contour_all.n_clusters == len(centroids)
            and np.array_equal(contour_all.contour,
                               centroids[contour_all.order])):
        # Already the shortest over every start, of these same centres:
        # the analysis' own contour stands for all the clusters.
        cache[np.ascontiguousarray(centroids).tobytes() + (
            b"" if guide is None
            else np.ascontiguousarray(guide).tobytes())] = contour_all
    if contour_all is None:
        contour_all = shortest(centroids)
    kept = centroids[~discarded]
    if len(kept) < 3 and discarded.any():
        warnings.append(
            f"Only {len(kept)} cluster(s) are left after discarding "
            f"{int(discarded.sum())}: no contour can be closed.")
    return AnchoredClusters(
        depth_tubulin_nm=depth_t, depth_spectrin_nm=depth_s,
        margin_nm=margin, discarded=discarded, contour_all=contour_all,
        contour_all_starts=shortest(centroids),
        contour_anchored=shortest(kept), warnings=warnings,
        registration=registration)


def axon_centre(col: NDArray[np.float64], row: NDArray[np.float64]
                ) -> Tuple[Tuple[float, float], float, float]:
    """
    Median centre of the selection, the median distance to it (the
    radius of a ring) and the farthest distance, all in pixels.

    Only where to look for the axon in the widefield images. The centre
    the MPS analysis reports is the area centroid of its contour
    (tools.mps_geometry.contour_centre).
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

# What says that two rows -- of any of the three tables -- describe the
# same selection: the localization file and the ROI drawn on it. Rows of
# two axons picked from one whole-field file agree on the first alone.
AXON_KEY_COLUMNS = ("source_localizations", "roi")


# The columns of ``summary_row``, in its order. They are the same whatever
# the panel could measure -- what it could not is an empty cell -- and the
# axon table writes all of them for every axon, the panel's or not, so that
# rows with and without the panel fit in one table. validate_axoplasm
# checks that a real row has exactly these.
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "source_localizations", "tubulin_image", "tubulin_source",
    "spectrin_interior_source", "registration_reference_image",
    "registration_localizations", "roi", "pixel_size_nm",
    "camera_offset_x_px", "camera_offset_y_px", "registration_source",
    "shift_x_nm", "shift_y_nm", "registration_peak",
    "registration_zero_shift", "registration_runner_up",
    "registration_score", "threshold", "threshold_source",
    "otsu_threshold", "smooth_sigma_nm", "margin_nm", "mask_status",
    "mask_area_um2", "ring_area_um2", "selection_zmin_nm",
    "selection_zmax_nm", "selection_z_source", "n_localizations",
    "n_outside_tubulin_region", "n_localizations_inside",
    "n_localizations_membrane", "n_localizations_no_cluster",
    "fraction_inside", "n_clusters", "spectrin_interior_image",
    "spectrin_interior_status", "discard_registration",
    "spectrin_interior_cut", "spectrin_interior_cut_source",
    "spectrin_level_interior", "spectrin_level_ring",
    "spectrin_level_half_max", "spectrin_level_spill",
    "spectrin_interior_area_um2", "n_clusters_discarded",
    "n_clusters_inside_tubulin_only", "n_clusters_inside_spectrin_only",
    "perimeter_all_clusters_um", "perimeter_all_clusters_all_starts_um",
    "perimeter_anchored_um", "perimeter_anchored_2opt_starts",
    "perimeter_anchored_start_spread_um", "anchored_contour_deep_vertices",
    "n_warnings", "warnings",
)


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
    spectrin: Optional[AxoplasmMask] = None,
    anchored: Optional[AnchoredClusters] = None,
    spectrin_image: str = "",
    tubulin_source: str = "widefield image",
    spectrin_source: str = "",
    located: Optional[NDArray[np.object_]] = None,
    n_clusters: Optional[int] = None,
    z_range: Optional[Tuple[float, float]] = None,
    z_range_source: str = "none",
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    One row per axon; the same columns whatever was computed.

    ``z_range`` is the axial cut the main window applied to the
    selection these localizations come from, and ``z_range_source`` where
    it came from: every count below is a count over that cut, and the
    same axon with the Z fields cleared reports quite different ones.

    ``spectrin_image`` is the widefield image the ring interior (and so
    the discard) was found in. It can differ from ``reference``, the image
    the shift was measured against. ``located`` is localization_labels
    for the localizations of ``result``: the localizations are counted
    inside only through their clusters, never by the tubulin mask alone.
    """
    sx, sy = registration.shift_nm(pixel_size_nm)
    # The warnings as the panel lists them, so that the table says the
    # same as the screen. Counting only the four objects below left out
    # the image notes -- a pixel size stretched by 2.7 %, a camera offset
    # assumed to be zero -- and exported n_warnings 0 with three on
    # screen. Their text was never exported at all, so filtering a table
    # on "no warnings" kept axons with a scale problem.
    shown = (list(warnings) if warnings is not None else
             list(registration.warnings) + list(mask.warnings)
             + ([] if spectrin is None else list(spectrin.warnings))
             + ([] if anchored is None else list(anchored.warnings)))

    # What could be measured at all. An empty mask, or a spectrin image
    # with no dark interior, used to export 0 discarded, 0 % inside and
    # areas of 0.0: numbers indistinguishable from a real axon where
    # nothing is inside. They are left empty now, and these columns say
    # why.
    mask_ok = not mask.empty
    interior_ok = spectrin is not None and not spectrin.empty
    # Could the images measure anything at all, and did they place the
    # clusters? With an empty mask, or a spectrin image with no dark
    # inside, the old row read 0 discarded, 0 % inside and an area of
    # 0.0: a failure written exactly like an axon with nothing inside.
    measured = mask_ok and (spectrin is None or interior_ok)
    placed = anchored is not None and mask_ok and interior_ok

    def count(label: str) -> Optional[int]:
        return (None if located is None or not measured
                else int(np.sum(located == label)))

    def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
        return None if value is None else round(float(value), digits)

    def finite(value: Optional[float], digits: int = 4) -> Optional[float]:
        if value is None or not np.isfinite(value):
            return None
        return round(float(value), digits)

    def level(name: str) -> Optional[float]:
        return None if spectrin is None else finite(
            spectrin.levels.get(name), 2)

    all_ = None if anchored is None else anchored.contour_all
    all_starts = None if anchored is None else anchored.contour_all_starts
    new = None if anchored is None else anchored.contour_anchored
    spectrin_columns = {
        "spectrin_interior_image": None if spectrin is None else spectrin_image,
        "spectrin_interior_status": (
            "not loaded" if spectrin is None
            else "ok" if interior_ok
            else "no dark interior found in this image"),
        # How the widefield images were placed when the clusters were
        # sorted. Loading the spectrin image is enough to sort them, so
        # this says whether that happened before or after the shift was
        # measured -- on axon 7, 4 clusters discarded against 5.
        "discard_registration": (None if anchored is None
                                 else anchored.registration or None),
        "spectrin_interior_cut": None if spectrin is None else finite(
            spectrin.threshold, 2),
        "spectrin_interior_cut_source": (None if spectrin is None
                                         else spectrin.threshold_source),
        "spectrin_level_interior": level("interior"),
        "spectrin_level_ring": level("ring"),
        "spectrin_level_half_max": level("half_max"),
        "spectrin_level_spill": level("spill"),
        "spectrin_interior_area_um2": (round(spectrin.area_um2, 4)
                                       if interior_ok else None),
        "n_clusters_discarded": (None if anchored is None or not placed
                                 else anchored.n_discarded),
        "n_clusters_inside_tubulin_only": (
            None if anchored is None or not placed
            else anchored.n_tubulin_only),
        "n_clusters_inside_spectrin_only": (
            None if anchored is None or not placed
            else anchored.n_spectrin_only),
        "perimeter_all_clusters_um": (None if all_ is None
                                      else finite(all_.perimeter_um)),
        "perimeter_all_clusters_all_starts_um": (
            None if all_starts is None else finite(all_starts.perimeter_um)),
        # The contour without the discarded clusters is only a result
        # when the discard could be evaluated; otherwise it is the contour
        # of all of them under another name.
        "perimeter_anchored_um": (None if new is None or not placed
                                  else finite(new.perimeter_um)),
        "perimeter_anchored_2opt_starts": (None if new is None or not placed
                                           else new.n_starts),
        "perimeter_anchored_start_spread_um": (
            None if new is None or not placed
            else finite(new.start_spread_um)),
        "anchored_contour_deep_vertices": (
            None if new is None or not placed or new.health is None
            else new.health.n_deep_vertices),
    }

    return {
        "source_localizations": localizations,
        "tubulin_image": tubulin,
        # A widefield image placed by a measured shift, or localizations
        # acquired with the movie and placed by construction.
        "tubulin_source": tubulin_source,
        "spectrin_interior_source": spectrin_source or None,
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
        # Empty means the threshold left no axoplasm at all, not an
        # axon of zero area.
        "mask_status": "ok" if mask_ok else "empty at this threshold",
        "mask_area_um2": round(mask.area_um2, 4) if mask_ok else None,
        "ring_area_um2": round(mask.ring_area_um2, 4),
        # The axial cut these localizations survived, which the panel
        # takes from the main window and cannot recompute.
        "selection_zmin_nm": (None if z_range is None
                              else round(float(z_range[0]), 2)),
        "selection_zmax_nm": (None if z_range is None
                              else round(float(z_range[1]), 2)),
        "selection_z_source": z_range_source,
        "n_localizations": result.n,
        # Off the region the tubulin mask was built in: no distance.
        "n_outside_tubulin_region": (result.count(LABEL_OUTSIDE)
                                     if mask_ok else None),
        # Through their clusters, as both images place those
        # (localization_labels); empty until both have been compared.
        "n_localizations_inside": count(LOC_INSIDE),
        "n_localizations_membrane": count(LOC_MEMBRANE),
        "n_localizations_no_cluster": count(LOC_NO_CLUSTER),
        "fraction_inside": (
            None if located is None or result.n == 0 or not measured
            else round(int(np.sum(located == LOC_INSIDE)) / result.n, 4)),
        # The clusters the MPS analysis kept for this selection.
        "n_clusters": (n_clusters if n_clusters is not None
                       else None if anchored is None else anchored.n),
        **spectrin_columns,
        "n_warnings": len(shown),
        "warnings": " | ".join(cell_text(w) for w in shown),
    }


def cluster_rows(
    *,
    localizations: str,
    roi: str,
    centroids_nm: NDArray[np.float64],
    anchored: AnchoredClusters,
    spectrin_image: str = "",
    labels: Optional[Union[Sequence[int], NDArray[np.int64]]] = None,
) -> List[Dict[str, Any]]:
    """
    One row per cluster: where each image puts it, and whether it went.

    ``spectrin_image`` is the widefield image of the ring interior.
    ``labels`` are the clusters' DBSCAN labels, in the order of
    ``centroids_nm``: that is the number every table of this program uses
    for a cluster, including the main window's per-localization export.
    Numbering the rows instead only matched it while the automatic
    curation removed nothing -- with one cluster removed, 48 of 88
    clusters were numbered differently in the two files.

    Which side of each image a cluster is on is written out, rather than
    left to be recomputed from the depths: the depths are rounded to
    0.1 nm and the decision is not, so a cluster 250.05 nm inside at a
    margin of 250 came back on the other side (13 "tubulin only" against
    the 14 the panel counts).
    """
    def depth(value: float) -> Any:
        """The depth, or an empty cell.

        Not -inf (no mask at all) and not nan (off the analysed region):
        both used to be written as words a reader takes for numbers, and
        pandas turns -inf into a mean of -inf.
        """
        return round(float(value), 1) if np.isfinite(value) else None

    gone = np.asarray(anchored.discarded, dtype=bool)
    inside_t = np.asarray(anchored.tubulin_only, dtype=bool) | gone
    inside_s = np.asarray(anchored.spectrin_only, dtype=bool) | gone
    names = np.asarray(labels).ravel() if labels is not None else None

    def group(t: bool, s: bool) -> str:
        if t and s:
            return "discarded"
        if t:
            return "tubulin only"
        if s:
            return "spectrin only"
        return "membrane"

    return [
        {"source_localizations": localizations, "roi": roi,
         "spectrin_interior_image": spectrin_image,
         "cluster_label": i if names is None else int(names[i]),
         "x_nm": round(float(x), 2), "y_nm": round(float(y), 2),
         "depth_in_tubulin_mask_nm": depth(dt),
         "depth_in_spectrin_interior_nm": depth(ds),
         "margin_nm": anchored.margin_nm,
         "inside_tubulin_mask": bool(inside_t[i]),
         "inside_spectrin_interior": bool(inside_s[i]),
         "group": group(bool(inside_t[i]), bool(inside_s[i])),
         "discarded": bool(left_out)}
        for i, ((x, y), dt, ds, left_out) in enumerate(zip(
            np.asarray(centroids_nm, dtype=float).reshape(-1, 2),
            anchored.depth_tubulin_nm, anchored.depth_spectrin_nm,
            anchored.discarded))
    ]
