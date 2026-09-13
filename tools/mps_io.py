# -*- coding: utf-8 -*-
"""
Reading localization files without a GUI.

``MPS_explorer.import_file`` is a method on the Qt window, so every
headless path so far has re-implemented its own loader. They differ, and
the difference that matters is the pixel size: Picasso stores x and y in
CAMERA PIXELS and z in nanometres, so a wrong pixel size rescales every
lateral distance and squares into every cluster area without ever raising
an error.

Policy difference from the GUI, on purpose
------------------------------------------
When the Picasso YAML sidecar is missing, the GUI asks the user. A batch
run has nobody to ask, so this module REFUSES to load rather than falling
back to a default. The original software hardcoded 133 nm; silently
applying that to data acquired at 113 nm would rescale every distance by
18 % and every area by 39 %, and nothing downstream could detect it. Pass
``pixel_size_nm`` explicitly to override, and the provenance travels with
the data so the export can record which files were guessed.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Format codes match MPS_explorer's fileformat combo box.
FORMAT_PICASSO_HDF5 = 0
FORMAT_THUNDERSTORM_CSV = 1
FORMAT_CUSTOM_CSV = 2


@dataclass
class Localizations:
    """One file's localizations, in nanometres on every axis."""

    x_nm: NDArray[np.float64]
    y_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    path: str
    fileformat: int
    pixel_size_nm: Optional[float]
    # "yaml" | "override" | "not_applicable" -- never a guess, see above.
    pixel_size_source: str

    @property
    def n(self) -> int:
        return int(self.x_nm.size)


def read_pixel_size(hdf5_path: str) -> Optional[float]:
    """
    Pixel size in nm from the Picasso YAML sidecar, or None when there is
    no sidecar or no ``Pixelsize:`` line in it.

    Parsed line by line rather than with a YAML library so a sidecar whose
    other keys are malformed still yields the one value that matters.
    """
    yaml_path = os.path.splitext(hdf5_path)[0] + ".yaml"
    if not os.path.exists(yaml_path):
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("Pixelsize:"):
                    return float(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


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
    pixel_size_nm : overrides the YAML sidecar. Required for a Picasso
        HDF5 that has no sidecar, since there is nothing to ask here.

    Raises
    ------
    ValueError when a Picasso HDF5 has neither a sidecar nor an override,
    or when a CSV lacks usable coordinate columns.
    """
    fmt = detect_format(path) if fileformat is None else int(fileformat)

    if fmt == FORMAT_PICASSO_HDF5:
        import h5py as h5

        with h5.File(path, "r") as f:
            ds = f["locs"]
            x = np.asarray(ds["x"], dtype=float)
            y = np.asarray(ds["y"], dtype=float)
            z = np.asarray(ds["z"], dtype=float)

        if pixel_size_nm is not None:
            px, source = float(pixel_size_nm), "override"
        else:
            from_yaml = read_pixel_size(path)
            if from_yaml is None:
                raise ValueError(
                    f"{os.path.basename(path)} has no Picasso YAML sidecar, so "
                    f"its pixel size is unknown. Picasso stores x and y in "
                    f"camera pixels, so loading it without one would rescale "
                    f"every lateral distance and square the error into every "
                    f"cluster area. Pass pixel_size_nm explicitly if you know "
                    f"the value for this acquisition."
                )
            px, source = float(from_yaml), "yaml"

        return Localizations(
            x_nm=x * px, y_nm=y * px, z_nm=z, path=path, fileformat=fmt,
            pixel_size_nm=px, pixel_size_source=source)

    # --- CSV: already in nanometres, so no pixel size applies -----------
    import pandas as pd

    frame = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in frame.columns}
    named = [cols.get(f"{a} [nm]") for a in ("x", "y", "z")]
    if all(c is not None for c in named):
        x, y, z = (np.asarray(frame[c], dtype=float).ravel() for c in named)
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

    return Localizations(
        x_nm=x, y_nm=y, z_nm=z, path=path, fileformat=fmt,
        pixel_size_nm=None, pixel_size_source="not_applicable")


# Suffixes of the files MPS Explorer itself writes next to the data. They
# are DERIVED -- already clustered, filtered or reduced -- and some carry
# the same "x [nm],y [nm],z [nm]" header as a real ThunderSTORM export
# (``_all_clusters.csv`` is exactly that plus a cluster_id column). Content
# sniffing therefore cannot tell them apart from input data, and analysing
# one as raw localizations would silently re-process processed data into
# plausible-looking numbers. Excluding them by name is the only defence.
DERIVED_SUFFIXES: Tuple[str, ...] = (
    "_cluster_centers",
    "_all_clusters",
    "neighbor_distances",
    "_filtered_clusters_thunderstorm",
)


def is_derived_output(name: str) -> bool:
    """True for a file MPS Explorer wrote itself (see DERIVED_SUFFIXES)."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    return any(s in stem for s in DERIVED_SUFFIXES)


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
    exclude_derived : skip the outputs MPS Explorer writes itself.

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
            if exclude_derived and is_derived_output(name):
                skipped.append(full)
            else:
                found.append(full)
    return sorted(found), sorted(skipped)
