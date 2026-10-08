# -*- coding: utf-8 -*-
"""
The viability explorer's work on one file, without Qt (so a worker process imports nothing of the GUI): the per-pair
criteria of both viability rules of one picked axon (``tools.mps_selection.geometry_for_file``: the batch's loader, the
pre-specified rings, z_quality, rule v2 on the lab coordinates, rule v2c), cached as JSON.

Geometry only (R8): no column statistic is computed here, so it may run on real axons.

The cache: ``<cache dir>/<key>.json`` with ``key`` = sha256 (12 hex) of the input file's sha256, the pixel size asked
for, ``tools.mps_selection.selection_source_sha()`` (the rules' sources and thresholds) and the sha256 of the
pre-specified column parameters (they build the rings). Any of them changes -> another key, a new computation.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

__all__ = ["CACHE_SCHEMA", "cache_dir_for", "cache_key", "geometry_job", "init_worker", "load_cached"]

CACHE_SCHEMA = "viability explorer cache v1"


def init_worker() -> None:
    """A worker process: one thread per numerical library (the pool runs several processes at once)."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def cache_dir_for(log_path: str) -> str:
    """``<log dir>/../viability_cache`` for the exploration log at ``log_path``."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(log_path))), "viability_cache")


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_STATIC: Dict[str, str] = {}


def _static_parts() -> str:
    """What every key shares in this process: the rules' sources and thresholds, the column parameters."""
    if "parts" not in _STATIC:
        from tools.mps_lumen import DEFAULT_COLUMNS_PARAMS_PATH
        from tools.mps_selection import selection_source_sha
        try:
            params = _file_sha256(DEFAULT_COLUMNS_PARAMS_PATH)
        except OSError:
            params = "no columns_params.yaml"
        _STATIC["parts"] = f"{selection_source_sha()}|{params}"
    return _STATIC["parts"]


def cache_key(path: str, pixel_size_nm: Optional[float]) -> Dict[str, str]:
    """{"key": 12 hex, "input_sha256": ..., "parts": ...} of one file and pixel size."""
    sha = _file_sha256(path)
    px = "metadata" if not pixel_size_nm else f"{float(pixel_size_nm):.6g}"
    parts = f"{CACHE_SCHEMA}|{sha}|{px}|{_static_parts()}"
    return {"key": hashlib.sha256(parts.encode("utf-8")).hexdigest()[:12], "input_sha256": sha, "parts": parts}


def load_cached(path: str, pixel_size_nm: Optional[float], cache_dir: str) -> Optional[Dict[str, Any]]:
    """The cached result of ``geometry_job`` for this file (None on a miss or an unreadable entry)."""
    k = cache_key(path, pixel_size_nm)
    p = os.path.join(cache_dir, f"{k['key']}.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict) or d.get("parts") != k["parts"]:
        return None
    d = dict(d, path=os.path.abspath(path), cached=True)
    return d


def geometry_job(path: str, pixel_size_nm: Optional[float], cache_dir: Optional[str]) -> Dict[str, Any]:
    """
    The criteria of one file as a JSON-able dict: {"path", "ok", "error", "criteria" (``AxonCriteria.to_json``),
    "input_sha256", "seconds", "cached"}. A failure is returned (``ok`` False), never raised (a worker job). Reads the
    cache first and writes it after a computation (atomically).
    """
    t0 = time.perf_counter()
    path = os.path.abspath(path)
    try:
        if cache_dir:
            hit = load_cached(path, pixel_size_nm, cache_dir)
            if hit is not None:
                hit["seconds"] = time.perf_counter() - t0
                return hit
        k = cache_key(path, pixel_size_nm)
        from tools.mps_selection import geometry_for_file
        crit = geometry_for_file(path, pixel_size_nm=pixel_size_nm or None)
        out: Dict[str, Any] = {"schema": CACHE_SCHEMA, "path": path, "ok": True, "error": "",
                               "criteria": crit.to_json(), "input_sha256": k["input_sha256"], "parts": k["parts"],
                               "key": k["key"], "computed_seconds": time.perf_counter() - t0}
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            target = os.path.join(cache_dir, f"{k['key']}.json")
            tmp = target + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            os.replace(tmp, target)
        out["cached"] = False
        out["seconds"] = time.perf_counter() - t0
        return out
    except Exception as exc:  # noqa: BLE001 - a worker job: reported, never raised
        return {"path": path, "ok": False, "error": f"{type(exc).__name__}: {exc}", "criteria": None,
                "cached": False, "seconds": time.perf_counter() - t0}
