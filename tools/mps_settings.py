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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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


@dataclass
class MPSSettings:
    """Analysis parameters that persist across sessions."""

    eps_nm: float = DEFAULT_EPS_NM
    min_samples: int = DEFAULT_MIN_SAMPLES
    slab_half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM
    dbcv_threshold: float = DEFAULT_DBCV_THRESHOLD
    auto_analyze_on_cluster: bool = True
    last_export_dir: str = ""

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
