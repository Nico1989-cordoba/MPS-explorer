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
