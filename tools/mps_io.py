# -*- coding: utf-8 -*-
"""
Reading localization files without a GUI.

``MPS_explorer.import_file`` is a method on the Qt window, so every
headless path so far has re-implemented its own loader. They differ, and
the difference that matters is the pixel size: Picasso stores x and y in
CAMERA PIXELS and z in nanometres, so a wrong pixel size rescales every
lateral distance and squares into every cluster area without ever raising
an error.

Read the whole file, not three columns
--------------------------------------
Until now this loader returned x, y and z and discarded everything else.
A Picasso file from this project also carries ``frame, photons, sx, sy,
bg, lpx, lpy, lpz, ellipticity, net_gradient, d_zcalib`` and sometimes
``group``. Those are not decoration:

  * ``frame`` is the whole of DNA-PAINT. Binding events, dark times,
    qPAINT counting and the sticking filter are all functions of it, and
    none of them can be reconstructed afterwards.
  * ``lpx``/``lpy``/``lpz`` are what separate "this structure is wide"
    from "this measurement is imprecise". Without them a fitted spread
    cannot be compared against the spread the data could possibly
    resolve.

So ``columns`` now holds every column of the file in its NATIVE units,
and the named accessors convert the ones whose unit is known.

The second unit trap
--------------------
x and y in pixels against z in nm is the trap everyone knows about. The
precision columns repeat it exactly: ``lpx`` and ``lpy`` are in CAMERA
PIXELS while ``lpz`` is already in NANOMETRES (Picasso's file-format
table, "Localization HDF5 Files"). Multiplying lpz by the pixel size
would inflate it by ~113x and silently make every ring look perfectly
resolved. Use ``lpx_nm`` / ``lpy_nm`` / ``lpz_nm``, which handle it.

Policy difference from the GUI, on purpose
------------------------------------------
When the metadata is missing, the GUI asks the user. A batch run has
nobody to ask, so this module REFUSES to load rather than falling back to
a default. The original software hardcoded 133 nm; silently applying that
to data acquired at 113 nm would rescale every distance by 18 % and every
area by 39 %, and nothing downstream could detect it. Pass
``pixel_size_nm`` explicitly to override, and the provenance travels with
the data so the export can record which files were guessed.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tools import mps_metadata

# Format codes match MPS_explorer's fileformat combo box.
FORMAT_PICASSO_HDF5 = 0
FORMAT_THUNDERSTORM_CSV = 1
FORMAT_CUSTOM_CSV = 2

# Columns Picasso stores in CAMERA PIXELS. Everything else is either
# already in nanometres (z, lpz), in photons, or dimensionless. Keeping
# this list explicit is the only defence against the lpz trap described
# in the module docstring.
PIXEL_COLUMNS: Tuple[str, ...] = (
    "x",
    "y",
    "lpx",
    "lpy",
    "sx",
    "sy",
    "sx_unc",
    "sy_unc",
    "x_pick_rot",
    "y_pick_rot",
)

# A fitted precision this far below the file's median, on a localization
# that is NOT unusually bright, is a failed fit. On the 15.07.26 DNA-PAINT
# files every such localization had its width collapsed to the fitter's
# floor (about 0.06 px) and a sixth of the usual photons; the nearest
# genuine ones sat above 0.2x. Precision scales as 1/sqrt(photons), so a
# real 0.1x takes about a hundred times the typical photon count -- which
# is exactly what a linked or combined localization has, hence the
# exemptions below.
FAILED_FIT_PRECISION_FRACTION = 0.1


def plausible_precision(
    values: NDArray[np.float64],
    exempt: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.bool_]:
    """
    True where a precision is finite, positive and not a failed fit.

    ``exempt`` marks localizations the relative floor must not apply to
    (bright ones); they still need a finite, positive precision.
    """
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values) & (values > 0)
    if good.any():
        floor = FAILED_FIT_PRECISION_FRACTION * float(np.median(values[good]))
        above = values >= floor
        if exempt is not None:
            above |= np.asarray(exempt, dtype=bool)
        good &= above
    return good


# ThunderSTORM spells the same quantities differently, and in nm. Mapping
# them to the Picasso names lets everything downstream speak one
# vocabulary instead of branching on the file format.
_THUNDERSTORM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "x": ("x [nm]",),
    "y": ("y [nm]",),
    "z": ("z [nm]",),
    "frame": ("frame", "frame number"),
    "photons": ("intensity [photon]", "intensity[photon]"),
    "bg": ("offset [photon]", "background [photon]"),
    "lpx": ("uncertainty_xy [nm]", "uncertainty [nm]", "uncertainty_x [nm]"),
    "lpy": ("uncertainty_xy [nm]", "uncertainty [nm]", "uncertainty_y [nm]"),
    "lpz": ("uncertainty_z [nm]",),
    "sx": ("sigma1 [nm]", "sigma [nm]"),
    "sy": ("sigma2 [nm]", "sigma [nm]"),
}


@dataclass
class Localizations:
    """
    One file's localizations.

    ``x_nm``/``y_nm``/``z_nm`` are always nanometres -- that contract has
    not changed. ``columns`` holds every column of the source file under
    its own name and in its own units; use the ``*_nm`` accessors rather
    than reading ``columns["lpz"]`` and guessing.
    """

    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    path: str
    fileformat: int
    pixel_size_nm: Optional[float]
    # "yaml" | "hdf5" | "yaml_scan" | "override" | "not_applicable"
    # -- never a guess, see the module docstring.
    pixel_size_source: str
    # Every column of the source file, native units, native dtype.
    columns: Dict[str, NDArray[Any]] = field(default_factory=dict)
    # Picasso's processing chain, oldest step first.
    info: List[Dict[str, Any]] = field(default_factory=list)
    # "yaml" | "hdf5" | "yaml_scan" | "none"
    metadata_source: str = "none"
    # Subtracted from the file's frame numbers so they count from 0: 1 for
    # a ThunderSTORM export, 0 otherwise. Decided once, at load time.
    frame_offset: int = 0

    # ---------------------------------------------------------- basics
    @property
    def n(self) -> int:
        return int(self.x_nm.size)

    def has(self, name: str) -> bool:
        """True when ``name`` is available, under its own or an alias."""
        return self._resolve(name) is not None

    def column(self, name: str) -> Optional[NDArray[Any]]:
        """
        A column in its NATIVE units, or None when the file lacks it.

        Accepts Picasso names; for a ThunderSTORM CSV the equivalent
        column is found through ``_THUNDERSTORM_ALIASES``.
        """
        key = self._resolve(name)
        return None if key is None else self.columns[key]

    def _resolve(self, name: str) -> Optional[str]:
        if name in self.columns:
            return name
        lowered = {str(k).strip().lower(): k for k in self.columns}
        if name.lower() in lowered:
            return lowered[name.lower()]
        for alias in _THUNDERSTORM_ALIASES.get(name, ()):
            if alias.lower() in lowered:
                return lowered[alias.lower()]
        return None

    def column_nm(self, name: str) -> Optional[NDArray[np.float64]]:
        """
        A column converted to nanometres, or None when it is absent.

        Only meaningful for spatial columns. A column in PIXEL_COLUMNS is
        multiplied by the pixel size -- but only when the value really is
        in pixels: a ThunderSTORM CSV stores the same quantities in nm
        already, and ``pixel_size_nm`` is None there, so no conversion is
        applied and none is invented.
        """
        values = self.column(name)
        if values is None:
            return None
        out = np.asarray(values, dtype=float)
        if name in PIXEL_COLUMNS and self.pixel_size_nm is not None:
            if self.fileformat == FORMAT_PICASSO_HDF5:
                return out * float(self.pixel_size_nm)
        return out

    # ------------------------------------------------- named accessors
    @property
    def frame(self) -> Optional[NDArray[np.int64]]:
        """
        Frame index of each localization, counted from 0, or None.

        Picasso counts from 0. ThunderSTORM counts from 1 (ImageJ's slice
        numbering), so its frames are shifted by ``frame_offset``: the
        DNA-PAINT code decides which binding events were already under way
        when imaging started by looking for frame 0.
        """
        values = self.column("frame")
        if values is None:
            return None
        return np.asarray(values, dtype=np.int64) - int(self.frame_offset)

    @property
    def photons(self) -> Optional[NDArray[np.float64]]:
        values = self.column("photons")
        return None if values is None else np.asarray(values, dtype=float)

    @property
    def lpx_nm(self) -> Optional[NDArray[np.float64]]:
        """Lateral precision in x, nanometres (source is camera pixels)."""
        return self.column_nm("lpx")

    @property
    def lpy_nm(self) -> Optional[NDArray[np.float64]]:
        """Lateral precision in y, nanometres (source is camera pixels)."""
        return self.column_nm("lpy")

    @property
    def lpz_nm(self) -> Optional[NDArray[np.float64]]:
        """
        Axial precision, nanometres.

        Picasso already stores lpz in nm, so this is deliberately NOT
        scaled by the pixel size -- see the module docstring.
        """
        values = self.column("lpz")
        return None if values is None else np.asarray(values, dtype=float)

    @property
    def lp_lateral_nm(self) -> Optional[NDArray[np.float64]]:
        """
        Per-localization lateral precision, the mean of lpx and lpy.

        The single number Picasso's own guidance is phrased in ("2*NeNA",
        "3*LP"), so it is worth having one definition of it.

        NaN where EITHER component is a failed fit: zero, not finite, or
        below FAILED_FIT_PRECISION_FRACTION of that axis's median. A failed
        fit often reports a collapsed precision on one axis while the
        other stays plausible, and averaging hides it: the mean comes out
        ordinary and the localization passes as a good one. So each axis
        is tested before averaging. (Picasso's ``lib.ensure_sanity`` only
        rejects negative values, so it keeps all of these.)

        The relative floor is not applied to localizations brighter than
        the median, nor to a file whose rows are already
        merged events (an ``n`` or ``len`` column): a very small precision
        is genuine there.
        """
        lpx, lpy = self.lpx_nm, self.lpy_nm
        if lpx is None and lpy is None:
            return None
        if self.has("n") or self.has("len"):
            exempt: Optional[NDArray[np.bool_]] = np.ones(self.n, dtype=bool)
        else:
            photons = self.photons
            exempt = None
            if photons is not None and np.isfinite(photons).any():
                exempt = photons > float(np.nanmedian(photons))
        if lpx is None or lpy is None:
            single = lpx if lpy is None else lpy
            assert single is not None
            return np.where(plausible_precision(single, exempt), single,
                            np.nan)
        good = (plausible_precision(lpx, exempt)
                & plausible_precision(lpy, exempt))
        return np.where(good, 0.5 * (lpx + lpy), np.nan)

    # -------------------------------------------------- from metadata
    @property
    def n_frames(self) -> Optional[int]:
        """Length of the acquisition in frames, from the metadata."""
        value = mps_metadata.get_value(self.info, mps_metadata.KEY_FRAMES)
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    @property
    def box_size_px(self) -> Optional[int]:
        """The fitting box Picasso: Localize used, in camera pixels."""
        value = mps_metadata.get_value(self.info, mps_metadata.KEY_BOX_SIZE)
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    @property
    def fit_method(self) -> Optional[str]:
        value = mps_metadata.get_value(self.info, mps_metadata.KEY_FIT_METHOD)
        return None if value is None else str(value)

    @property
    def z_calibration(self) -> Optional[Dict[str, Any]]:
        """The astigmatism calibration, when the file was fitted in 3D."""
        value = mps_metadata.get_value(
            self.info, mps_metadata.KEY_Z_CALIBRATION
        )
        return value if isinstance(value, dict) else None

    @property
    def is_3d(self) -> bool:
        """True when the file carries a z coordinate that is not all zero."""
        return bool(self.z_nm.size) and bool(np.any(self.z_nm != 0.0))

    @property
    def processing_steps(self) -> List[str]:
        """Picasso's ``Generated by`` chain, oldest step first."""
        return mps_metadata.steps(self.info)

    def subset(self, indices: NDArray[Any]) -> "Localizations":
        """
        The same file restricted to ``indices``, columns and all.

        Needed because the analysis works on a ROI and an axial slab while
        the file holds the whole field of view: a cluster label indexes the
        SUBSET, so lining a label up with the frame or precision of its
        localization means carrying the same selection through every
        column. Metadata and pixel size are properties of the acquisition
        and travel unchanged.
        """
        index = np.asarray(indices)
        return Localizations(
            x_nm=self.x_nm[index],
            y_nm=self.y_nm[index],
            z_nm=self.z_nm[index],
            path=self.path,
            fileformat=self.fileformat,
            pixel_size_nm=self.pixel_size_nm,
            pixel_size_source=self.pixel_size_source,
            columns={name: values[index]
                     for name, values in self.columns.items()},
            info=self.info,
            metadata_source=self.metadata_source,
            frame_offset=self.frame_offset,
        )


def read_pixel_size(hdf5_path: str) -> Optional[float]:
    """
    Pixel size in nm for a Picasso file, or None when it is unknown.

    Reads the full metadata chain (YAML sidecar, then the ``/metadata``
    dataset embedded since Picasso v0.11), so it no longer fails on a
    current Picasso file written without a sidecar.
    """
    info, _source = mps_metadata.load_metadata(hdf5_path)
    return mps_metadata.pixel_size_nm(info)


def detect_format(path: str) -> int:
    """Format code from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".hdf5", ".h5"):
        return FORMAT_PICASSO_HDF5
    if ext == ".csv":
        return FORMAT_THUNDERSTORM_CSV
    raise ValueError(
        f"Unsupported extension {ext!r} for {os.path.basename(path)}; "
        f"expected .hdf5, .h5 or .csv."
    )


def _load_picasso_hdf5(
    path: str, pixel_size_nm: Optional[float]
) -> Localizations:
    import h5py as h5

    with h5.File(path, "r") as handle:
        if "locs" not in handle:
            raise ValueError(
                f"{os.path.basename(path)} has no '/locs' dataset, so it is "
                f"not a Picasso localization file. Files of pick properties "
                f"store their table under '/groups' instead."
            )
        table = handle["locs"][:]

    names = list(table.dtype.names or ())
    for required in ("x", "y"):
        if required not in names:
            raise ValueError(
                f"{os.path.basename(path)} has no {required!r} column; its "
                f"columns are {names}."
            )
    # A copy per column rather than a strided view into the compound
    # buffer: views keep the whole record array alive and are
    # non-contiguous, which surprises anything that later hands one to a
    # C extension.
    columns: Dict[str, NDArray[Any]] = {
        name: np.array(table[name]) for name in names
    }

    info, metadata_source = mps_metadata.load_metadata(path)

    if pixel_size_nm is not None:
        pixel, source = float(pixel_size_nm), "override"
    else:
        from_metadata = mps_metadata.pixel_size_nm(info)
        if from_metadata is None:
            raise ValueError(
                f"{os.path.basename(path)} has no pixel size: no Picasso "
                f"YAML sidecar and no embedded '/metadata'. Picasso stores "
                f"x and y in camera pixels, so loading it without one would "
                f"rescale every lateral distance and square the error into "
                f"every cluster area. Pass pixel_size_nm explicitly if you "
                f"know the value for this acquisition."
            )
        pixel, source = float(from_metadata), metadata_source

    x = np.asarray(columns["x"], dtype=float) * pixel
    y = np.asarray(columns["y"], dtype=float) * pixel
    if "z" in columns:
        z = np.asarray(columns["z"], dtype=float)
    else:
        # A 2D acquisition. Zeros keep every downstream array shape valid;
        # `is_3d` is how a caller tells this apart from a real flat slab.
        z = np.zeros_like(x)

    return Localizations(
        x_nm=x,
        y_nm=y,
        z_nm=z,
        path=path,
        fileformat=FORMAT_PICASSO_HDF5,
        pixel_size_nm=pixel,
        pixel_size_source=source,
        columns=columns,
        info=info,
        metadata_source=metadata_source,
    )


def _load_csv(path: str, fmt: int) -> Localizations:
    import pandas as pd

    frame = pd.read_csv(path)
    columns: Dict[str, NDArray[Any]] = {
        str(c): np.asarray(frame[c].to_numpy()) for c in frame.columns
    }
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    named = [lookup.get(f"{a} [nm]") for a in ("x", "y", "z")]

    if all(c is not None for c in named):
        x, y, z = (
            np.asarray(frame[c], dtype=float).ravel() for c in named
        )
    elif fmt == FORMAT_CUSTOM_CSV and frame.shape[1] >= 3:
        # Only when the caller asked for this format by name. Auto-detection
        # must not take the first three numeric columns of any CSV it finds:
        # plenty of files in a working folder have three numeric columns and
        # are not localizations at all.
        values = frame.iloc[:, :3].to_numpy(dtype=float)
        x, y, z = values[:, 0], values[:, 1], values[:, 2]
    else:
        raise ValueError(
            f"{os.path.basename(path)} has no 'x [nm]'/'y [nm]'/'z [nm]' "
            f"columns. If its first three columns are x, y and z in "
            f"nanometres, load it with fileformat=FORMAT_CUSTOM_CSV."
        )

    # ThunderSTORM numbers frames from 1. Picasso's "Export for
    # ThunderSTORM" writes the same layout with its own 0-based frames, and
    # gives itself away with a background-noise column of zeros, which a
    # real ThunderSTORM fit never produces.
    frame_offset = 0
    frame_column = lookup.get("frame")
    if fmt == FORMAT_THUNDERSTORM_CSV and frame_column is not None:
        frames = np.asarray(frame[frame_column])
        bkgstd = lookup.get("bkgstd [photon]")
        from_picasso = (bkgstd is not None and
                        np.all(np.asarray(frame[bkgstd], dtype=float) == 0))
        if frames.size and frames.min() >= 1 and not from_picasso:
            frame_offset = 1

    return Localizations(
        x_nm=x,
        y_nm=y,
        z_nm=z,
        path=path,
        fileformat=fmt,
        pixel_size_nm=None,
        pixel_size_source="not_applicable",
        columns=columns,
        info=[],
        metadata_source="none",
        frame_offset=frame_offset,
    )


def load_localizations(
    path: str,
    fileformat: Optional[int] = None,
    pixel_size_nm: Optional[float] = None,
) -> Localizations:
    """
    Load one localization file, returning nanometres on every axis.

    Parameters
    ----------
    path : the file.
    fileformat : one of the FORMAT_* codes; inferred from the extension
        when None. A .csv is tried as ThunderSTORM first and falls back to
        three bare columns.
    pixel_size_nm : overrides the metadata. Required for a Picasso HDF5
        that carries no metadata at all, since there is nothing to ask
        here.

    Returns
    -------
    Localizations, with ``columns`` holding every column of the source
    file and ``info`` holding Picasso's processing chain.

    Raises
    ------
    ValueError when a Picasso HDF5 has no pixel size from any source and
    no override, when it has no '/locs' dataset, or when a CSV lacks
    usable coordinate columns.
    """
    fmt = detect_format(path) if fileformat is None else int(fileformat)
    if fmt == FORMAT_PICASSO_HDF5:
        return _load_picasso_hdf5(path, pixel_size_nm)
    return _load_csv(path, fmt)


# Suffixes of the files MPS Explorer itself writes next to the data. They
# are DERIVED -- already clustered, filtered or reduced -- and some carry
# the same "x [nm],y [nm],z [nm]" header as a real ThunderSTORM export
# (``_all_clusters.csv`` is exactly that plus a cluster_label column). Content
# sniffing therefore cannot tell them apart from input data, and analysing
# one as raw localizations would silently re-process processed data into
# plausible-looking numbers. Excluding them by name is the only defence.
DERIVED_SUFFIXES: Tuple[str, ...] = (
    "_cluster_centers",
    "_all_clusters",
    "neighbor_distances",
    "_filtered_clusters_thunderstorm",
    "_two_channels",
    # The tables the panels write beside the data. A batch over the folder
    # of the April axon 7 took two of them for axons and reported them as
    # files that failed to load.
    "_axoplasm",
    "_axoplasm_localizations",
    "_axoplasm_clusters",
    "_mps_parameters",
    "_mps_parameters_discard",
    "_mps_parameters_every_start",
    "_mps_rings",
    "_mps_rings_pairs",
)

# Also written by the Picasso tools menu, but a drift-corrected copy IS a
# file someone may want to analyse. It is therefore not skipped; its stem
# is reduced to the original's so that having both in one folder is
# reported as the same acquisition twice.
_SAME_ACQUISITION_PATTERNS: Tuple[Any, ...] = (re.compile(r"_aim(?![a-z0-9])"),)

# The channel-numbered ROI export written by ``MPS_explorer.save_roi``
# ("{stem}_ch{channel}_roi.csv"). It needs a pattern rather than a fixed
# suffix because of the channel number, and it cannot be shortened to
# "_roi": real files in this dataset are named "..._axon2_roi2.hdf5".
# Missing it meant a batch analysed every axon TWICE -- once as the
# Picasso HDF5 and once as its own ROI export -- and then reported the
# nesting statistics as if those were independent observations.
#
# Deliberately NOT anchored to a word boundary: these files get renamed
# by hand afterwards ("..._ch1_roi_filterby-123to57.csv" is in the real
# dataset), and "_" is a word character, so a \b here would let exactly
# those through. The marker identifies the file whatever follows it.
_DERIVED_PATTERNS: Tuple[Any, ...] = (
    re.compile(r"_ch\d+_roi"),
    # Written next to the input by the Picasso tools menu: a cluster subset,
    # a molecule map and binding events, none of them an axon's
    # localizations. Whole tokens only -- "__linked" files are the user's
    # own Picasso output and are real input.
    re.compile(r"_(?:clusters|molmap|link)(?![a-z0-9])"),
)


def is_derived_output(name: str) -> bool:
    """True for a file MPS Explorer wrote itself (see DERIVED_SUFFIXES)."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    if any(s in stem for s in DERIVED_SUFFIXES):
        return True
    return any(pattern.search(stem) for pattern in _DERIVED_PATTERNS)


# Columns that only the tables this program writes carry together: the
# axon table and its clusters and localizations (axon_id with
# analysis_id), the batch's log, and the dictionary of columns. Their
# names are the user's to choose -- "mps_axons.csv" by default, anything
# after -- so they are recognised by what they hold, not by their name.
# A batch over the April folder took the three tables of the manual test
# (2026-09-19) for axons.
_PROGRAM_TABLE_MARKS: Tuple[Tuple[str, ...], ...] = (
    ("axon_id", "analysis_id"),
    ("table", "position", "column", "unit", "meaning"),
)


def is_program_table(path: str) -> bool:
    """
    True for a CSV this program wrote as a table of results, whatever it
    is called: its first line holds the columns only those tables carry.
    The copy made for Excel (';' between fields) is recognised too.
    """
    if os.path.splitext(path)[1].lower() != ".csv":
        return False
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as handle:
            first = handle.readline()
    except OSError:
        return False
    names = {cell.strip().strip('"').lower()
             for cell in re.split(r"[,;\t]", first)}
    return any(all(mark in names for mark in marks)
               for marks in _PROGRAM_TABLE_MARKS)


def source_stem(name: str) -> str:
    """
    The part of a filename that identifies WHICH acquisition it came from.

    Two files that reduce to the same stem are the same axon under two
    names -- typically the Picasso HDF5 and an export made from it. Used
    to catch a derived suffix nobody has thought of yet, so that the next
    one shows up as a reported collision instead of as a silently doubled
    sample size.
    """
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    for suffix in DERIVED_SUFFIXES:
        stem = stem.replace(suffix, "")
    for pattern in _DERIVED_PATTERNS + _SAME_ACQUISITION_PATTERNS:
        stem = pattern.sub("", stem)
    return stem.strip("_")


def duplicate_sources(paths: List[str]) -> Dict[str, List[str]]:
    """
    Files among ``paths`` that appear to be the same acquisition twice.

    Returns {stem: [paths]} for every stem claimed by more than one file.
    Empty when every file is a distinct acquisition. A caller should
    report these rather than analyse them: counting one axon twice
    inflates every n, and makes a nesting or replication statistic
    describe the duplication instead of the biology.
    """
    groups: Dict[str, List[str]] = {}
    for path in paths:
        groups.setdefault(source_stem(path), []).append(path)
    return {
        stem: sorted(found)
        for stem, found in groups.items()
        if len(found) > 1
    }


def find_localization_files(
    root: str,
    pattern: str = "axon",
    extensions: Tuple[str, ...] = (".hdf5", ".h5", ".csv"),
    exclude_derived: bool = True,
) -> Tuple[list, list]:
    """
    Localization files under ``root``, sorted.

    Parameters
    ----------
    pattern : substring filter on the basename. "" takes every file, which
        in a Picasso working directory also picks up intermediate renders
        and unpicked files -- hence the default.
    exclude_derived : skip the outputs MPS Explorer writes itself: the
        derived files by their names, and the tables of results by their
        columns (``is_program_table``).

    Returns
    -------
    (files, skipped) -- ``skipped`` lists the derived outputs that were
    excluded, so a caller can report them instead of dropping them
    silently: a file vanishing from a batch with no explanation is as
    misleading as one being analysed wrongly.
    """
    found: list = []
    skipped: list = []
    needle = pattern.lower()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in extensions:
                continue
            if needle and needle not in name.lower():
                continue
            full = os.path.join(dirpath, name)
            if exclude_derived and (is_derived_output(name)
                                    or is_program_table(full)):
                skipped.append(full)
            else:
                found.append(full)
    return sorted(found), sorted(skipped)
