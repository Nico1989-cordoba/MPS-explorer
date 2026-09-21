# -*- coding: utf-8 -*-
"""
Persistence of the MPS analysis parameters between sessions.

The user loads axons in series from the same acquisition, so the DBSCAN
parameters should carry over from one file to the next AND from one run of
the application to the next -- retyping them every time is exactly how a
stale or wrong value slips into a batch unnoticed.

Within a single session the GUI's QLineEdit fields already keep their
value; this module adds the missing piece, which is surviving a restart.
Values are written to ``mps_analysis_settings.json`` next to the
application, in the same spirit as the existing ``config_loader`` (which
holds static defaults and is not written back to).

Nothing here silently changes a number: ``load_settings`` returns the
stored values and the caller decides whether to apply them, and the GUI
shows what was restored.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from tools.mps_identity import check_patterns

logger = logging.getLogger("MPS_explorer.mps_settings")

SETTINGS_FILENAME = "mps_analysis_settings.json"

# Paper defaults (Gazal et al. 2026). Used on a fresh install, and as the
# fallback whenever the stored file is missing or unreadable.
DEFAULT_EPS_NM = 25.0
DEFAULT_MIN_SAMPLES = 10
DEFAULT_SLAB_HALF_WIDTH_NM = 90.0

# DBCV is OFF by default (-1.0: no real score is ever below the metric's own
# floor, so the criterion never fires). Measured on real axon data, log10(area)
# vs. DBCV score correlates at -0.78 to -0.79: the score is not blind to
# cluster size, it is dominated by it. At the previous default of 0.0 this
# removed the largest cluster in an axon in every case checked -- exactly the
# clusters Gazal et al. (2026) interpret as spectrin oligomers and keep. Edge-
# touching remains the default automatic-curation criterion; DBCV is left
# available for the user to enable deliberately, understanding that it will
# also act as a de facto size filter.
DEFAULT_DBCV_THRESHOLD = -1.0

# The occupancy threshold of Gazal et al. (2026): a perimeter point counts
# as occupied within this Mahalanobis distance of a cluster's constrained
# Gaussian. It decides the occupancy outright -- 0.1 instead of 3 turned
# 46.5 % into 1.5 % on axon 7 -- so it is stored, exported, and never left
# to whatever a spin box happens to show.
DEFAULT_MAHALANOBIS_THRESHOLD = 3.0


@dataclass
class MPSSettings:
    """Analysis parameters that persist across sessions."""

    eps_nm: float = DEFAULT_EPS_NM
    min_samples: int = DEFAULT_MIN_SAMPLES
    # Channel 2's own, stored separately because they are not channel
    # 1's. Until 2026-09-20 both fields were seeded from the pair above
    # and only channel 1's was read back, so a value typed for channel 2
    # reverted at the next launch without saying so. 0 means "never set",
    # in which case channel 2 falls back to channel 1's -- an assumption
    # about the partner protein's density, which the two-channel panel
    # now measures and reports rather than making quietly.
    eps_nm_channel2: float = 0.0
    min_samples_channel2: int = 0
    slab_half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM
    dbcv_threshold: float = DEFAULT_DBCV_THRESHOLD
    mahalanobis_threshold: float = DEFAULT_MAHALANOBIS_THRESHOLD
    auto_analyze_on_cluster: bool = True
    last_export_dir: str = ""
    # Where the open dialogs start: the folder of the last file opened.
    last_open_dir: str = ""
    # Empty means "search the PATH and the usual install folders".
    picasso_path: str = ""
    # How the genotype, the protein, the slide, the ROI and the axon are
    # read off a path. The user writes them once for their own folders;
    # see tools.mps_identity, which never invents what they do not match.
    identity_patterns: Dict[str, str] = field(default_factory=dict)
    # The identity confirmed for the previous axon, and the folder it was
    # confirmed in, so the six or more axons of one measurement are not
    # retyped one by one. Carried over only within the same slide.
    identity_last: Dict[str, str] = field(default_factory=dict)
    identity_last_folder: str = ""
    # Pixel sizes a person had to supply, one per folder, so the six or
    # more axons of one acquisition are not asked about one at a time.
    # Keyed by tools.mps_pixel_size.folder_key. Only ever written when a
    # file carried no pixel size AND nothing near it recorded one; a
    # value here never overrides a file's own metadata.
    pixel_size_by_folder: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> "MPSSettings":
        """Clamp to physically meaningful ranges, falling back to the paper
        defaults for anything nonsensical. A corrupted settings file must
        never be able to inject an impossible parameter into an analysis."""
        if not (0 < self.eps_nm <= 1000):
            logger.warning(
                "Stored eps_nm=%r out of range; using %s", self.eps_nm,
                DEFAULT_EPS_NM)
            self.eps_nm = DEFAULT_EPS_NM
        if not (1 <= int(self.min_samples) <= 10000):
            logger.warning(
                "Stored min_samples=%r out of range; using %s",
                self.min_samples, DEFAULT_MIN_SAMPLES)
            self.min_samples = DEFAULT_MIN_SAMPLES
        self.min_samples = int(self.min_samples)
        if not (1 <= self.slab_half_width_nm <= 5000):
            logger.warning(
                "Stored slab_half_width_nm=%r out of range; using %s",
                self.slab_half_width_nm, DEFAULT_SLAB_HALF_WIDTH_NM)
            self.slab_half_width_nm = DEFAULT_SLAB_HALF_WIDTH_NM
        # The knob was retired on 2026-09-20. Measured over the 18 real
        # April axons (1143 clusters), the DBCV score correlates with
        # log10(cluster area) at -0.70, and in 10 of the 18 the largest
        # cluster is the lowest- or second-lowest-scoring one: any
        # threshold above off removes the big clusters first. A settings
        # file written before that is ignored rather than obeyed, because
        # a stored value would otherwise silently curate an analysis.
        if self.dbcv_threshold != DEFAULT_DBCV_THRESHOLD:
            logger.info(
                "Stored dbcv_threshold=%r ignored: the criterion was "
                "retired (it removes the largest clusters first); using %s",
                self.dbcv_threshold, DEFAULT_DBCV_THRESHOLD)
            self.dbcv_threshold = DEFAULT_DBCV_THRESHOLD
        if not (-1.0 <= self.dbcv_threshold <= 1.0):
            logger.warning(
                "Stored dbcv_threshold=%r out of range; using %s",
                self.dbcv_threshold, DEFAULT_DBCV_THRESHOLD)
            self.dbcv_threshold = DEFAULT_DBCV_THRESHOLD
        if not (0.1 <= self.mahalanobis_threshold <= 10.0):
            logger.warning(
                "Stored mahalanobis_threshold=%r out of range; using %s",
                self.mahalanobis_threshold, DEFAULT_MAHALANOBIS_THRESHOLD)
            self.mahalanobis_threshold = DEFAULT_MAHALANOBIS_THRESHOLD
        # 0 is the legal "never set"; anything else has to be a usable
        # parameter or it is dropped rather than clamped, because a
        # clamped clustering radius is a number nobody chose.
        try:
            eps2 = float(self.eps_nm_channel2)
        except (TypeError, ValueError):
            eps2 = 0.0
        if eps2 != 0.0 and not (0 < eps2 <= 1000):
            logger.warning(
                "Stored eps_nm_channel2=%r out of range; channel 2 will be "
                "asked for again", self.eps_nm_channel2)
            eps2 = 0.0
        self.eps_nm_channel2 = eps2
        try:
            min2 = int(float(self.min_samples_channel2))
        except (TypeError, ValueError):
            min2 = 0
        if min2 != 0 and not (1 <= min2 <= 10000):
            logger.warning(
                "Stored min_samples_channel2=%r out of range; channel 2 "
                "will be asked for again", self.min_samples_channel2)
            min2 = 0
        self.min_samples_channel2 = min2
        if not isinstance(self.picasso_path, str):
            self.picasso_path = ""
        for name in ("last_open_dir", "last_export_dir",
                     "identity_last_folder"):
            if not isinstance(getattr(self, name), str):
                setattr(self, name, "")
        for name in ("identity_patterns", "identity_last"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                setattr(self, name, {})
                continue
            setattr(self, name, {str(k): str(v) for k, v in value.items()
                                 if isinstance(v, str)})
        # A remembered pixel size is a number a person typed once; a
        # corrupted one would rescale every distance measured in that
        # folder without anything raising, so an entry that is not a
        # plausible pixel size is dropped rather than clamped.
        if not isinstance(self.pixel_size_by_folder, dict):
            self.pixel_size_by_folder = {}
        else:
            kept: Dict[str, float] = {}
            for folder, value in self.pixel_size_by_folder.items():
                try:
                    nm = float(value)
                except (TypeError, ValueError):
                    nm = 0.0
                if 1.0 <= nm <= 5000.0:
                    kept[str(folder)] = nm
                else:
                    logger.warning(
                        "Stored pixel size %r for %s is not a pixel size; "
                        "it will be asked for again", value, folder)
            self.pixel_size_by_folder = kept
        # A stored pattern that no longer compiles would raise at the next
        # export, in the middle of writing tables; drop it here instead and
        # fall back to the default for that field.
        broken = check_patterns(self.identity_patterns)
        for field_name, reason in broken.items():
            logger.warning(
                "Stored identity pattern for %s is not a valid regular "
                "expression (%s); using the default", field_name, reason)
            self.identity_patterns.pop(field_name, None)
        return self


def settings_path(directory: Optional[str] = None) -> Path:
    """Location of the settings file (next to the application by default)."""
    base = Path(directory) if directory else Path(__file__).resolve().parent.parent
    return base / SETTINGS_FILENAME


def load_settings(directory: Optional[str] = None) -> MPSSettings:
    """
    Read persisted settings, falling back to the paper defaults.

    Never raises: a missing, unreadable or corrupt file yields defaults and
    a log entry, because failing to start the GUI over a settings file
    would be worse than losing the stored preference.
    """
    path = settings_path(directory)
    if not path.exists():
        return MPSSettings()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s (%s); using defaults", path, exc)
        return MPSSettings()

    known = {f: data[f] for f in MPSSettings.__dataclass_fields__ if f in data}
    try:
        return MPSSettings(**known).validate()
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid settings in %s (%s); using defaults", path, exc)
        return MPSSettings()


def save_settings(settings: MPSSettings,
                  directory: Optional[str] = None) -> bool:
    """
    Persist settings. Returns True on success.

    Writes to a temporary file and replaces atomically so an interrupted
    write cannot leave a truncated file that would silently reset the
    user's parameters on the next launch.
    """
    path = settings_path(directory)
    tmp = path.with_suffix(".json.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(settings), f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.warning("Could not write %s (%s)", path, exc)
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False
