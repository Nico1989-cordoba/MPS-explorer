"""
UI stage 2, step IMPL-0: the golden reference, captured BEFORE any stage-2 edit.

What it is for
--------------
Stage 2 of the UI redesign moves plots, merges controls and gives every
analysis parameter one home. None of that may change a number or an
exported byte, except for the documented behaviour changes B1-B13 of the
stage-2 design. This script drives the real application offscreen on
SIMULATED axons, over the design's grid of scenarios x entry points, and
writes down everything a later step must reproduce:

* every number the results windows show (the MPS analysis table, its
  provenance line and warnings; the rings window's header, segment and
  pair tables);
* every exported table, as written to disk: the axon / clusters /
  localizations tables of "Export axon" and the column dictionary beside
  them, both rings CSV files, the nearest-neighbour distances CSV (N = 1
  and N = 3) - header (column order) and contents;
* the main window's axial cut (zmin / zmax, ``_applied_slab``, the typed
  flag, the selection) and the provenance words (the analysis'
  ``slab_source``, the axoplasm panel's ``selection_z_source``, Two
  channels' slab and ``slab_mode``);
* the axoplasm panel's rows when the panel is open;
* the keyword arguments the GUI passes to ``analyze_axon``,
  ``analyze_all_segments``, ``compare_discard`` and ``build_tables``
  (recorded by wrappers that call the real function unchanged);
* the BatchSettings the batch window reads at Run, after its sync;
* the settings file the run leaves behind (after the main window closes).

R8: only simulated axons (``batch_columns.write_simulated_input``, the
``tools.mps_zquality_window.DEMO_CASES`` generator) and synthetic widefield
images are used. No real data is read and no column statistic is computed.

Isolation: every cell runs in its own Python process, with its working
directory, settings file (``tools.mps_settings.settings_path`` and
``MPS_explorer.load_settings / save_settings`` redirected), log folder,
parameter cache, review store and selection log inside the cell's own
temporary folder. The user's settings file is checked untouched (content
and mtime) at the end of the run.

Grid (DESIGN_S2.md 8.4)
-----------------------
Axons: ``A`` (DEMO_CASES "viable": circle contour, seed 39, axial leak 0.3)
and ``B`` (ellipse contour, seed 7, axial leak 0.3); each with three small
clusters placed inside the axon (not anchored to the membrane), so that the
axoplasm panel has something to discard. Channel 2 holds the other axon's
file, so that the Two channels panel can be opened.

Scenarios: S1 automatic; S2 eps 30 / min samples 8 in the results window;
S3 a mixture component chosen in the results window; S4 half-width 80;
S5 a typed Z range ("select Z range"); S6 Mahalanobis 2.5; S7 randomization
off; S8 a drawn contour guide; S9 the axoplasm panel with synthetic
widefield tubulin and spectrin images (shift measured) -> discard;
S10 Reset to paper defaults after S2 + S4 + S6; S11a / S11b the ROI dragged
after an analysis with discard, with the axoplasm panel visible / hidden;
S12a a drawn guide, then the same ROI re-applied; S12b the same, then a new
ROI.

Entry points: E1 the results window's re-run (its controls as they stand);
E2 "cluster Ch1"; E3 "MPS analysis"; E4 "Rings..." from the results window,
then the rings window at valley/0, valley/40, paper/40, paper/0 (each
exported); E5 "Batch" (its BatchSettings after sync).

Every cell records three snapshots: ``base`` (after the first "cluster
Ch1"), ``scenario`` (after the scenario's actions) and ``entry`` (after the
entry point), plus ``settings_file`` (after closing).

Normalisation (documented, nothing dropped silently)
---------------------------------------------------
* Absolute paths: the cell folder -> ``<CELL>``, the run's input folder ->
  ``<INPUTS>``, the repository -> ``<REPO>``; both slash styles, any case.
* The ``exported_at`` column (wall-clock time of the export) -> ``<TIME>``.
* Numeric arrays passed as keyword arguments are recorded as shape, dtype
  and SHA-256 of their bytes; analysis objects as the sequence number of
  the analysis that produced them (``analysis#k``).
* The localizations table (one row per localization, ~12,000 rows) is
  recorded as its header, its row count and one SHA-256 per column (of the
  column's text as written), so that a difference names the column; the
  other tables are recorded in full.
* Identical contents are stored once, under their SHA-256 (``blobs``).
* Program texts that quote a figure measured on unpublished data (today one
  sentence, "the April axons measure(d) 6-16 %", in a results-table note and
  the column dictionary) -> ``<REDACTED: ...>`` (``REDACTIONS``), so that
  no such figure is committed. Applied before any digest.
Nothing else is altered: every number is the text the program wrote or
showed. Log files, the parameter cache and the selection log are not
recorded (they hold times and session ids and are not outputs).

Usage (with the project's venv interpreter: the golden records the
interpreter and the numpy / scipy / scikit-learn versions, and results
differ in their last digits between scipy 1.17 and 1.18)
-----
    venv/Scripts/python.exe golden_ui_stage2.py --out DIR [--workers 6] [--axons A,B]
                              [--cells S1:E1,S2:E3,...]
    venv/Scripts/python.exe golden_ui_stage2_compare.py DIR1/golden.json DIR2/golden.json

``DIR`` must be a new or empty folder; the golden is ``DIR/golden.json``.
The committed copy is ``testdata/ui_stage2/parity_golden.json``.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
GOLDEN_VERSION = 1
PIXEL_NM = 130.0

AXONS: Dict[str, Tuple[str, int, float]] = {
    # contour, seed, axial-leak scale (batch_columns.write_simulated_input)
    "A": ("circle", 39, 0.3),     # tools.mps_zquality_window.DEMO_CASES["viable"]
    "B": ("ellipse", 7, 0.3),
}
SCENARIOS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
             "S11a", "S11b", "S12a", "S12b"]
ENTRIES = ["E1", "E2", "E3", "E4", "E5"]
SCENARIO_TEXT = {
    "S1": "automatic slab, nothing changed",
    "S2": "eps 30 and min samples 8 typed in the results window",
    "S3": "a mixture component chosen in the results window's axial peak box",
    "S4": "slab half-width 80 in the results window",
    "S5": "a Z range typed in the main window, then 'select Z range'",
    "S6": "Mahalanobis 2.5 in the results window",
    "S7": "randomization unticked in the results window",
    "S8": "a contour guide drawn (Draw contour..., Apply)",
    "S9": "axoplasm panel: synthetic widefield tubulin and spectrin, shift measured -> discard",
    "S10": "S2 + S4 + S6, then 'Reset to paper defaults'",
    "S11a": "S9, then the ROI dragged (+150 nm in x) with the axoplasm panel visible",
    "S11b": "S9, then the ROI dragged (+150 nm in x) with the axoplasm panel hidden",
    "S12a": "S8, then the same ROI re-applied ('select Z range')",
    "S12b": "S12a, then a new ROI (dragged +150 nm in x)",
}
ENTRY_TEXT = {
    "E1": "the results window re-runs with its controls as they stand",
    "E2": "'cluster Ch1' in the main window",
    "E3": "'MPS analysis' in the main window",
    "E4": "'Rings...' in the results window, then valley/0, valley/40, paper/40, paper/0",
    "E5": "'Batch': the BatchSettings read at Run, after the window's sync",
}
RINGS_STEPS = [("valley", 0.0), ("valley", 40.0), ("paper", 40.0), ("paper", 0.0)]
ROI_DRAG_NM = 150.0
# Program texts that quote a figure measured on unpublished data. They are
# replaced by a token so that the committed golden carries no such figure;
# a change of the text around them is still seen, and so is the token.
REDACTIONS = [
    (re.compile(r"the April axons measured? \d+(?:\.\d+)? ?(?:-|to) ?\d+(?:\.\d+)? ?%"),
     "<REDACTED: a figure measured on unpublished data>"),
]
LONG_LIST = 100          # longer keyword-argument lists: length + sha256
BLOB_MIN_BYTES = 1024    # larger JSON parts are stored once, under their sha256


# =========================================================================
#  Simulated inputs (deterministic)
# =========================================================================
def _interior_clusters(rng: np.random.Generator, cx: float, cy: float,
                       r_ring: float, z_levels: np.ndarray
                       ) -> Dict[str, np.ndarray]:
    """Three tight clusters well inside the ring: not anchored to it."""
    xs, ys, zs = [], [], []
    for k, (fr, ang) in enumerate(((0.30, 0.4), (0.45, 2.6), (0.25, 4.3))):
        n = 30
        x0 = cx + fr * r_ring * np.cos(ang)
        y0 = cy + fr * r_ring * np.sin(ang)
        z0 = float(z_levels[k % len(z_levels)])
        xs.append(x0 + rng.normal(0.0, 8.0, n))
        ys.append(y0 + rng.normal(0.0, 8.0, n))
        zs.append(z0 + rng.normal(0.0, 15.0, n))
    return {"x": np.concatenate(xs), "y": np.concatenate(ys),
            "z": np.concatenate(zs)}


def _write_picasso(path: str, x_nm, y_nm, z_nm, frame, lp_nm, lpz_nm,
                   n_frames: int) -> None:
    import h5py

    n = len(x_nm)
    rows = np.zeros(n, dtype=[
        ("frame", "u4"), ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("photons", "f4"), ("sx", "f4"), ("sy", "f4"), ("bg", "f4"),
        ("lpx", "f4"), ("lpy", "f4"), ("lpz", "f4")])
    rows["frame"] = np.asarray(frame, np.int64)
    rows["x"] = np.asarray(x_nm, float) / PIXEL_NM
    rows["y"] = np.asarray(y_nm, float) / PIXEL_NM
    rows["z"] = np.asarray(z_nm, float)
    rows["photons"] = 3000.0
    rows["sx"] = rows["sy"] = 1.0
    rows["bg"] = 50.0
    rows["lpx"] = rows["lpy"] = np.asarray(lp_nm, float) / PIXEL_NM
    rows["lpz"] = np.asarray(lpz_nm, float)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("locs", data=rows)
    with open(os.path.splitext(path)[0] + ".yaml", "w", encoding="utf-8") as f:
        f.write(f"Frames: {int(n_frames)}\nWidth: 64\nHeight: 64\n---\n"
                f"Generated by: Picasso Localize\nPixelsize: {PIXEL_NM:g}\n")


def _render(x_px: np.ndarray, y_px: np.ndarray, shape: Tuple[int, int],
            sigma: float) -> np.ndarray:
    from scipy import ndimage

    counts = np.zeros(shape, float)
    c = np.clip(np.floor(x_px).astype(int), 0, shape[1] - 1)
    r = np.clip(np.floor(y_px).astype(int), 0, shape[0] - 1)
    np.add.at(counts, (r, c), 1.0)
    return ndimage.gaussian_filter(counts, sigma)


def make_inputs(folder: str) -> Dict[str, Any]:
    """Write the simulated axons (Picasso HDF5 + YAML) and the synthetic
    widefield images. Returns their description (no absolute path)."""
    import tifffile

    import batch_columns as bc

    os.makedirs(folder, exist_ok=True)
    described: Dict[str, Any] = {}
    for key, (contour, seed, leak) in AXONS.items():
        npz = bc.write_simulated_input(
            os.path.join(folder, f"sim_{key}.npz"), contour, seed, leak)
        with np.load(npz) as d:
            x, y, z = (np.asarray(d[q], float) for q in ("x", "y", "z"))
            frame = np.asarray(d["frame"], np.int64)
            lp, lpz = np.asarray(d["lp"], float), np.asarray(d["lpz"], float)
            n_frames = int(np.asarray(d["n_frames"]).reshape(-1)[0])
        os.remove(npz)
        cx, cy = float(np.median(x)), float(np.median(y))
        r_ring = float(np.median(np.hypot(x - cx, y - cy)))
        rng = np.random.default_rng(1000 + seed)
        # The interior clusters sit at the axial levels of the densest z.
        hist, edges = np.histogram(z, bins=40)
        top = np.argsort(hist)[::-1][:3]
        levels = (edges[top] + edges[top + 1]) / 2.0
        extra = _interior_clusters(rng, cx, cy, r_ring, levels)
        m = extra["x"].size
        x = np.concatenate([x, extra["x"]])
        y = np.concatenate([y, extra["y"]])
        z = np.concatenate([z, extra["z"]])
        frame = np.concatenate([frame, rng.integers(0, n_frames, m)])
        lp = np.concatenate([lp, np.full(m, float(np.median(lp)))])
        lpz = np.concatenate([lpz, np.full(m, float(np.median(lpz)))])
        h5 = os.path.join(folder, f"sim_axon_{key}.hdf5")
        _write_picasso(h5, x, y, z, frame, lp, lpz, n_frames)
        # Synthetic widefield images on the movie's own pixel grid:
        # spectrin = the localizations blurred (a ring), tubulin = a
        # blurred disc inside the ring (the axoplasm).
        shape = (64, 64)
        spec = _render(x / PIXEL_NM, y / PIXEL_NM, shape, 1.3) * 40.0 + 100.0
        spec += np.random.default_rng(2000 + seed).normal(0.0, 2.0, shape)
        rr, cc = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
        d = np.hypot(cc + 0.5 - cx / PIXEL_NM, rr + 0.5 - cy / PIXEL_NM)
        radius_px = 0.85 * r_ring / PIXEL_NM
        tub = 100.0 + 900.0 * np.clip(0.5 + (radius_px - d), 0.0, 1.0)
        from scipy import ndimage
        tub = ndimage.gaussian_filter(tub, 1.0)
        tub += np.random.default_rng(3000 + seed).normal(0.0, 2.0, shape)
        tifffile.imwrite(os.path.join(folder, f"wf_spectrin_{key}.tif"),
                         np.clip(spec, 0, 65535).astype(np.uint16))
        tifffile.imwrite(os.path.join(folder, f"wf_tubulin_{key}.tif"),
                         np.clip(tub, 0, 65535).astype(np.uint16))
        described[key] = dict(
            generator="batch_columns.write_simulated_input",
            contour=contour, seed=seed, leak_scale=leak,
            n_localizations=int(x.size), n_interior_added=int(m),
            centre_nm=[round(cx, 3), round(cy, 3)],
            ring_radius_nm=round(r_ring, 3),
            hdf5=f"<INPUTS>/sim_axon_{key}.hdf5",
            sha256=_sha_file(h5))
    return described


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================================================================
#  Normalisation
# =========================================================================
class Normaliser:
    def __init__(self, replacements: List[Tuple[str, str]]):
        pairs: List[Tuple[str, str]] = []
        for path, token in replacements:
            for variant in {path, path.replace("\\", "/"),
                            path.replace("/", "\\")}:
                pairs.append((variant, token))
        # Longest first, so a cell folder inside the run folder wins.
        self.pairs = sorted(pairs, key=lambda p: -len(p[0]))

    def text(self, s: str) -> str:
        out = s
        for path, token in self.pairs:
            if path and path.lower() in out.lower():
                i = 0
                low_path = path.lower()
                parts = []
                low = out.lower()
                while True:
                    j = low.find(low_path, i)
                    if j < 0:
                        parts.append(out[i:])
                        break
                    parts.append(out[i:j])
                    parts.append(token)
                    i = j + len(path)
                out = "".join(parts)
        for pattern, token in REDACTIONS:
            out = pattern.sub(token, out)
        return out

    def value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.text(v)
        if isinstance(v, dict):
            return {str(self.text(str(k))): self.value(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self.value(x) for x in v]
        return v


_NORM: Optional["Normaliser"] = None


def describe_value(v: Any, analyses: Dict[int, int]) -> Any:
    """A JSON-able, exact description of a keyword argument. Strings are
    normalised (paths) BEFORE anything is hashed, so a digest never
    depends on where the run's folders are."""
    if isinstance(v, str):
        return _NORM.text(v) if _NORM is not None else v
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return {"float": repr(v)}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return {"float": repr(float(v)), "dtype": str(np.asarray(v).dtype)}
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        return {"array": list(a.shape), "dtype": str(a.dtype),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    if isinstance(v, (list, tuple)):
        items = [describe_value(x, analyses) for x in v]
        if len(items) > LONG_LIST:
            # Rows the export writes out in full anyway (the axoplasm
            # panel's per-localization rows): their length and a digest.
            text = json.dumps(items, sort_keys=True, default=str)
            return {type(v).__name__ + "_len": len(items),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        return {type(v).__name__: items}
    if isinstance(v, dict):
        return {"dict": {str(k): describe_value(x, analyses)
                         for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}}
    if id(v) in analyses:
        return f"analysis#{analyses[id(v)]}"
    name = type(v).__name__
    if name in ("AxonAnalysis",):
        return f"{name}(unrecorded)"
    try:
        import dataclasses
        if dataclasses.is_dataclass(v):
            return {name: {f.name: describe_value(getattr(v, f.name), analyses)
                           for f in dataclasses.fields(v)}}
    except Exception:  # noqa: BLE001
        pass
    return {"object": name, "repr": repr(v)[:300]}


# =========================================================================
#  The cell (runs in its own process)
# =========================================================================
def _csv_text_table(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    rows = list(csv.reader(io.StringIO(text)))
    return {"header": rows[0] if rows else [], "rows": rows[1:]}


class Cell:
    def __init__(self, axon: str, scenario: str, entry: str, cell_dir: str,
                 inputs_dir: str):
        self.axon, self.scenario, self.entry = axon, scenario, entry
        self.cell_dir = os.path.abspath(cell_dir)
        self.inputs_dir = os.path.abspath(inputs_dir)
        self.settings_dir = os.path.join(self.cell_dir, "settings")
        self.export_dir = os.path.join(self.cell_dir, "exports")
        for d in (self.settings_dir, self.export_dir):
            os.makedirs(d, exist_ok=True)
        self.norm = Normaliser([(self.cell_dir, "<CELL>"),
                                (self.inputs_dir, "<INPUTS>"),
                                (REPO, "<REPO>")])
        global _NORM
        _NORM = self.norm
        self.calls: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, str]] = []
        self.analyses: Dict[int, int] = {}
        self._keep: List[Any] = []
        self.save_queue: List[str] = []
        self.export_counter = 0
        self.snapshots: Dict[str, Any] = {}
        self.errors: List[str] = []

    # ------------------------------------------------------------ patches
    def patch(self) -> None:
        from PyQt5 import QtWidgets

        from tools import mps_settings

        original_path = mps_settings.settings_path
        settings_dir = self.settings_dir

        def settings_path(directory: Optional[str] = None):
            return original_path(directory or settings_dir)
        mps_settings.settings_path = settings_path

        import MPS_explorer
        MPS_explorer.load_settings = functools.partial(
            mps_settings.load_settings, directory=settings_dir)
        MPS_explorer.save_settings = functools.partial(
            mps_settings.save_settings, directory=settings_dir)

        # Modal message boxes: recorded, answered with OK.
        cell = self

        def box(kind: str):
            def show(*args: Any, **kwargs: Any) -> Any:
                title = str(args[1]) if len(args) > 1 else ""
                text = str(args[2]) if len(args) > 2 else ""
                cell.messages.append({"kind": kind, "title": title,
                                      "text": cell.norm.text(text)})
                return QtWidgets.QMessageBox.Ok
            return show
        for kind in ("information", "warning", "critical", "question"):
            setattr(QtWidgets.QMessageBox, kind, staticmethod(box(kind)))

        def save_name(*_a: Any, **_k: Any) -> Tuple[str, str]:
            if not cell.save_queue:
                cell.errors.append("an unexpected save dialog was answered with nothing")
                return "", ""
            return cell.save_queue.pop(0), "CSV Files (*.csv)"
        QtWidgets.QFileDialog.getSaveFileName = staticmethod(save_name)

        def no_open(*_a: Any, **_k: Any) -> Tuple[str, str]:
            cell.errors.append("an unexpected open dialog was answered with nothing")
            return "", ""
        QtWidgets.QFileDialog.getOpenFileName = staticmethod(no_open)

        # The export dialog: every table, to the path the harness queued.
        from tools import axon_export_ui

        class FakeExportDialog:
            def __init__(self, path: str, identity: Any, *, state: str = "",
                         **_kw: Any) -> None:
                self._identity = identity
                cell.last_export_state = cell.norm.text(str(state))
                cell.last_export_suggested = cell.norm.text(str(path))

            def exec_(self) -> int:
                return QtWidgets.QDialog.Accepted

            def path(self) -> str:
                return cell.save_queue.pop(0)

            def identity(self) -> Any:
                return self._identity

            def wants_clusters(self) -> bool:
                return True

            def wants_localizations(self) -> bool:
                return True
        axon_export_ui.ExportAxonDialog = FakeExportDialog  # type: ignore

        # Wrappers that record the keyword arguments and call the real code.
        from tools import axon_export, mps_analysis, mps_multisegment

        def wrap(module: Any, name: str, label: str, produces: bool) -> None:
            real = getattr(module, name)

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                entry = {"function": label,
                         "args": [describe_value(a, cell.analyses) for a in args],
                         "kwargs": {k: describe_value(v, cell.analyses)
                                    for k, v in sorted(kwargs.items())}}
                result = real(*args, **kwargs)
                if produces and result is not None:
                    for obj in ([result] if not hasattr(result, "all_clusters")
                                else [result, result.all_clusters,
                                      result.discard_applied]):
                        if id(obj) not in cell.analyses:
                            cell.analyses[id(obj)] = len(cell.analyses) + 1
                            cell._keep.append(obj)
                    entry["result"] = describe_value(result, cell.analyses) \
                        if not hasattr(result, "all_clusters") else {
                            "all_clusters": describe_value(result.all_clusters, cell.analyses),
                            "discard_applied": describe_value(result.discard_applied, cell.analyses)}
                cell.calls.append(cell.norm.value(entry))
                return result
            setattr(module, name, wrapper)

        wrap(MPS_explorer, "analyze_axon", "analyze_axon", True)
        wrap(mps_multisegment, "analyze_all_segments", "analyze_all_segments", False)
        wrap(mps_analysis, "compare_discard", "compare_discard", True)
        wrap(axon_export, "build_tables", "build_tables", False)

        # The contour editor: Apply, with a fixed path (a circle through the
        # centroids' mean radius, 72 points, counter-clockwise from +x).
        from tools import contour_editor

        def draw_contour(_parent: Any, centroids: Any, **_kw: Any):
            c = np.asarray(centroids, float)
            mx, my = c[:, 0].mean(), c[:, 1].mean()
            r = float(np.mean(np.hypot(c[:, 0] - mx, c[:, 1] - my)))
            t = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
            path = np.column_stack([mx + r * np.cos(t), my + r * np.sin(t)])
            cell.guide_drawn = describe_value(path, cell.analyses)
            return contour_editor.APPLY, path
        contour_editor.draw_contour = draw_contour

    # ------------------------------------------------------------ helpers
    def next_path(self, stem: str) -> str:
        self.export_counter += 1
        return os.path.join(self.export_dir, f"{self.export_counter:02d}_{stem}.csv")

    def pump(self, app: Any, seconds: float = 0.05) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)

    # ------------------------------------------------------------ records
    def record_main(self, mw: Any) -> Dict[str, Any]:
        roi = mw.roi_indices
        unf = mw.roi_indices_unfiltered
        out: Dict[str, Any] = {
            "zmin": None if mw.zmin is None else repr(float(mw.zmin)),
            "zmax": None if mw.zmax is None else repr(float(mw.zmax)),
            "field_zmin": mw.ui.lineEdit_zmin.text(),
            "field_zmax": mw.ui.lineEdit_zmax.text(),
            "z_range_user_edited": bool(mw._z_range_user_edited),
            "applied_slab": (None if mw._applied_slab is None
                             else [repr(float(v)) for v in mw._applied_slab]),
            "applied_roi_shape": (None if mw._applied_roi_shape is None
                                  else repr(mw._applied_roi_shape)),
            "roi_indices": describe_value(None if roi is None else np.asarray(roi), {}),
            "roi_indices_unfiltered": describe_value(None if unf is None else np.asarray(unf), {}),
            "field_eps": mw.ui.lineEdit_eps.text(),
            "field_min_samples": mw.ui.lineEdit_minsamples.text(),
            "settings_in_memory": self.norm.value(_settings_dict(mw.mps_settings)),
            "mps_analysis": (None if mw.mps_analysis is None
                             else describe_value(mw.mps_analysis, self.analyses)),
            "discard_held": (None if mw.mps_discard is None else {
                "base": describe_value(mw.mps_discard.base, self.analyses),
                "all_clusters": describe_value(mw.mps_discard.comparison.all_clusters, self.analyses),
                "discard_applied": describe_value(mw.mps_discard.comparison.discard_applied, self.analyses)}),
        }
        return out

    def record_provenance(self, mw: Any) -> Dict[str, Any]:
        a = mw.mps_analysis
        inputs = mw._axoplasm_inputs()
        out: Dict[str, Any] = {
            "analysis_slab_source": None if a is None else str(a.slab_source),
            "analysis_slab_nm": (None if a is None else
                                 [repr(float(a.slab_zmin_nm)), repr(float(a.slab_zmax_nm))]),
            "analysis_contour_guide": (None if a is None or a.contour_guide is None
                                       else describe_value(np.asarray(a.contour_guide), {})),
            "axoplasm_inputs_z_range_source": None if inputs is None else inputs.z_range_source,
            "axoplasm_inputs_z_range": (None if inputs is None or inputs.z_range is None
                                        else [repr(float(v)) for v in inputs.z_range]),
        }
        w = mw.axoplasm_window
        if w is not None:
            out["axoplasm_window_z_range_source"] = w.inputs.z_range_source
            out["axoplasm_window_shows_current_selection"] = \
                bool(w.inputs.selection_key is mw.roi_indices)
        t = mw.two_channel_window
        if t is not None:
            slab = t.inputs.slab
            out["two_channels_slab"] = (None if slab is None
                                        else [repr(float(v)) for v in slab])
            # The expression TwoChannelWindow.export_row writes (today).
            out["two_channels_slab_mode"] = "manual" if slab else "automatic"
            out["two_channels_slab_half_width_nm"] = repr(float(t.inputs.slab_half_width_nm))
            out["two_channels_kwargs_a"] = {
                k: describe_value(v, self.analyses)
                for k, v in sorted(t.inputs.kwargs_a.items()) if k != "source_name"}
        return self.norm.value(out)

    def record_results_window(self, mw: Any) -> Optional[Dict[str, Any]]:
        w = mw.mps_window
        if w is None:
            return None
        t = w.table
        cells = []
        for r in range(t.rowCount()):
            row = []
            for c in range(t.columnCount()):
                item = t.item(r, c)
                row.append("" if item is None else item.text())
            cells.append(row)
        headers = [t.horizontalHeaderItem(c).text() if t.horizontalHeaderItem(c) else ""
                   for c in range(t.columnCount())]
        hidden = [bool(t.isColumnHidden(c)) for c in range(t.columnCount())]
        warnings = [w.list_warnings.item(i).text() for i in range(w.list_warnings.count())]
        controls = {
            "peak_items": [w.combo_peak.itemText(i) for i in range(w.combo_peak.count())],
            "peak_index": w.combo_peak.currentIndex(),
            "half_width": repr(float(w.spin_half.value())),
            "eps": repr(float(w.spin_eps.value())),
            "min_samples": int(w.spin_min.value()),
            "mahalanobis": repr(float(w.spin_maha.value())),
            "randomization": bool(w.chk_random.isChecked()),
        }
        radios = {}
        for name in ("radio_measured", "radio_every", "radio_discard"):
            r = getattr(w, name, None)
            if r is not None:
                radios[name] = [r.text(), bool(r.isChecked()), bool(r.isVisibleTo(w))]
        return self.norm.value({
            "analysis": describe_value(w.analysis, self.analyses),
            "comparison": (None if w.comparison is None else {
                "all_clusters": describe_value(w.comparison.all_clusters, self.analyses),
                "discard_applied": describe_value(w.comparison.discard_applied, self.analyses)}),
            "provenance": w.lbl_provenance.text(),
            "headers": headers, "hidden_columns": hidden, "cells": cells,
            "warnings": warnings, "controls": controls, "radios": radios})

    def record_rings_window(self, rw: Any) -> Dict[str, Any]:
        def table(t: Any) -> Dict[str, Any]:
            return {"headers": [t.horizontalHeaderItem(c).text() if t.horizontalHeaderItem(c) else ""
                                for c in range(t.columnCount())],
                    "cells": [["" if t.item(r, c) is None else t.item(r, c).text()
                               for c in range(t.columnCount())] for r in range(t.rowCount())]}
        warnings = []
        lw = getattr(rw, "list_warnings", None)
        if lw is not None:
            warnings = [lw.item(i).text() for i in range(lw.count())]
        return self.norm.value({
            "mode_control": rw.combo_mode.currentData(),
            "guard_control": repr(float(rw.spin_guard.value())),
            "header": rw.lbl_header.text(),
            "segments": table(rw.table_seg), "pairs": table(rw.table_pair),
            "warnings": warnings})

    def export_axon(self, mw: Any) -> Dict[str, Any]:
        from tools import axon_export

        target = self.next_path("axon")
        self.save_queue = [target]
        n_msg = len(self.messages)
        self.last_export_state = None
        result = mw.export_axon()
        out: Dict[str, Any] = {"returned": None if result is None else self.norm.text(result),
                               "dialog_state": getattr(self, "last_export_state", None),
                               "messages": self.messages[n_msg:]}
        self.save_queue = []
        if result is None:
            return out
        paths = axon_export.table_paths(target)
        tables: Dict[str, Any] = {}
        for key, p in sorted(paths.items()):
            if os.path.exists(p):
                tables[key] = _csv_text_table(p)
        folder = os.path.dirname(target)
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            stem = os.path.splitext(os.path.basename(target))[0]
            if name.startswith(stem) and full not in paths.values() and name.endswith(".csv"):
                tables["other:" + name[len(stem):]] = _csv_text_table(full)
        out["tables"] = self.norm.value(tables)
        return out

    def distances(self, mw: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for n in (1, 3):
            n_msg = len(self.messages)
            mw.ui.lineEdit_Nneighbor.setText(str(n))
            mw.KNdist_hist()
            target = self.next_path(f"{n}neighbor_distances")
            self.save_queue = [target]
            mw.savedistdata()
            self.save_queue = []
            out[f"N{n}"] = {"messages": self.messages[n_msg:],
                            "table": (_csv_text_table(target) if os.path.exists(target) else None)}
        return out

    def axoplasm_rows(self, mw: Any) -> Optional[Dict[str, Any]]:
        w = mw.axoplasm_window
        if w is None:
            return None
        w.flush()
        if w.result is None:
            return {"result": None}
        rows: Dict[str, Any] = {"summary_row": {k: _plain(v) for k, v in w.summary_row().items()}}
        rows["cluster_rows"] = ([{k: _plain(v) for k, v in r.items()} for r in w.cluster_rows()]
                                if w.anchored is not None else None)
        loc = w.localization_rows()
        rows["localization_rows"] = {"n": len(loc),
                                     "sha256": hashlib.sha256(json.dumps(
                                         self.norm.value([{k: _plain(v) for k, v in r.items()} for r in loc]),
                                         sort_keys=True).encode("utf-8")).hexdigest()}
        rows["warnings"] = list(w.warnings())
        rows["anchored_n_discarded"] = (None if w.anchored is None else int(w.anchored.n_discarded))
        return self.norm.value(rows)

    def snapshot(self, mw: Any, label: str, full: bool = True) -> None:
        snap: Dict[str, Any] = {
            "main_cut": self.record_main(mw),
            "provenance": self.record_provenance(mw),
            "results_window": self.record_results_window(mw),
            "calls": self.calls, "messages": self.messages,
        }
        if mw.rings_window is not None:
            snap["rings_window"] = self.record_rings_window(mw.rings_window)
        if full:
            snap["axoplasm"] = self.axoplasm_rows(mw)
            snap["export_axon"] = self.export_axon(mw)
            snap["distances"] = self.distances(mw)
        snap["calls"] = self.calls
        snap["messages"] = self.messages
        self.snapshots[label] = json.loads(json.dumps(snap, default=_plain))
        self.calls = []
        self.messages = []

    # ------------------------------------------------------------ actions
    def place_roi(self, mw: Any, centre: Tuple[float, float], radius_nm: float) -> None:
        # The circle selects within ROI_DIAMETER_SCALE_FACTOR * size / 2 of
        # pos + size / 2 (update_ROI): choose the size that gives radius_nm.
        import MPS_explorer
        size = 2.0 * radius_nm / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((centre[0] - size / 2.0, centre[1] - size / 2.0), update=False, finish=False)
        mw.update_ROI()            # what sigRegionChangeFinished calls

    def drag_roi(self, mw: Any, dx: float) -> None:
        roi = mw.circular_roi
        p = roi.pos()
        roi.setPos((float(p.x()) + dx, float(p.y())), update=False, finish=False)
        mw.update_ROI()

    def window_edit(self, app: Any, spin: Any, value: Any) -> None:
        spin.setValue(value)
        spin.editingFinished.emit()
        self.pump(app, 0.02)

    def axoplasm_with_images(self, app: Any, mw: Any) -> None:
        mw.show_axoplasm_panel()
        w = mw.axoplasm_window
        assert w is not None, "axoplasm panel did not open"
        w.load_tubulin(os.path.join(self.inputs_dir, f"wf_tubulin_{self.axon}.tif"))
        w.load_reference(os.path.join(self.inputs_dir, f"wf_spectrin_{self.axon}.tif"))
        w.flush()
        w.measure()
        w.wait(300)
        self.pump(app, 0.2)
        w.flush()
        self.pump(app, 0.1)

    def run(self) -> Dict[str, Any]:
        os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(self.cell_dir, "selection_log")
        from PyQt5 import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["golden"])
        self.patch()
        import MPS_explorer
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir

        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(self.cell_dir, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        other = "B" if self.axon == "A" else "A"
        path1 = os.path.join(self.inputs_dir, f"sim_axon_{self.axon}.hdf5")
        path2 = os.path.join(self.inputs_dir, f"sim_axon_{other}.hdf5")
        with open(os.path.join(self.inputs_dir, "inputs.json"), encoding="utf-8") as f:
            described = json.load(f)["axons"][self.axon]
        centre = tuple(described["centre_nm"])
        radius = described["ring_radius_nm"] + 450.0

        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(path1, 0), "channel 1 did not load"
        assert mw.load_channel2(path2, 0), "channel 2 did not load"
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin",
                                   animal="sim", sample="golden", roi_name="roi",
                                   axon_name=f"axon{self.axon}")
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        self.place_roi(mw, centre, radius)
        mw.ui.pushButton_clusterch1.click()
        self.pump(app, 0.05)
        # Channel 2's own parameters typed, so that Two channels opens
        # without asking; it only shows the selection until Run.
        mw.ui.lineEdit_eps_2.setText("25")
        mw.ui.lineEdit_minsamples_2.setText("10")
        mw.show_two_channel_panel()
        self.pump(app, 0.05)
        self.snapshot(mw, "base")

        s = self.scenario
        w = mw.mps_window
        assert w is not None, "the results window did not open after cluster Ch1"
        if s == "S1":
            pass
        elif s == "S2":
            self.window_edit(app, w.spin_eps, 30.0)
            self.window_edit(app, w.spin_min, 8)
        elif s == "S3":
            a = w.analysis
            centre_now = (a.slab_zmin_nm + a.slab_zmax_nm) / 2.0
            means = list(map(float, a.z_result.means_nm))
            weights = list(map(float, a.z_result.weights))
            far = [i for i, m in enumerate(means) if abs(m - centre_now) > 100.0]
            pick = (max(far, key=lambda i: weights[i]) if far
                    else int(np.argmax(weights)))
            self.chosen_component = {"index": pick, "item": w.combo_peak.itemText(pick)}
            w.combo_peak.setCurrentIndex(pick)
            self.pump(app, 0.02)
        elif s == "S4":
            self.window_edit(app, w.spin_half, 80.0)
        elif s == "S5":
            zmin = float(mw.ui.lineEdit_zmin.text()) + 20.0
            zmax = float(mw.ui.lineEdit_zmax.text()) - 20.0
            for edit, v in ((mw.ui.lineEdit_zmin, zmin), (mw.ui.lineEdit_zmax, zmax)):
                text = f"{v:.1f}"
                edit.setText(text)
                edit.textEdited.emit(text)
            mw.ui.pushButton_zrange.click()
        elif s == "S6":
            self.window_edit(app, w.spin_maha, 2.5)
        elif s == "S7":
            w.chk_random.setChecked(False)
            self.pump(app, 0.02)
        elif s in ("S8", "S12a", "S12b"):
            w.btn_contour.click()
            self.pump(app, 0.02)
            if s in ("S12a", "S12b"):
                self.snapshot(mw, "guide_drawn", full=False)
                mw.ui.pushButton_zrange.click()          # the same ROI re-applied
                self.pump(app, 0.02)
            if s == "S12b":
                self.snapshot(mw, "same_roi_reapplied", full=False)
                self.drag_roi(mw, ROI_DRAG_NM)
                self.pump(app, 0.02)
        elif s in ("S9", "S11a", "S11b"):
            self.axoplasm_with_images(app, mw)
            if s in ("S11a", "S11b"):
                self.snapshot(mw, "discard_before_drag", full=False)
                if s == "S11b":
                    mw.axoplasm_window.hide()
                    self.pump(app, 0.02)
                self.drag_roi(mw, ROI_DRAG_NM)
                self.pump(app, 0.1)
                if mw.axoplasm_window is not None:
                    mw.axoplasm_window.flush()
                self.pump(app, 0.05)
        elif s == "S10":
            self.window_edit(app, w.spin_eps, 30.0)
            self.window_edit(app, w.spin_min, 8)
            self.window_edit(app, w.spin_half, 80.0)
            self.window_edit(app, w.spin_maha, 2.5)
            self.snapshot(mw, "before_reset", full=False)
            w.btn_reset.click()
            self.pump(app, 0.02)
        else:
            raise ValueError(s)
        self.snapshot(mw, "scenario")

        e = self.entry
        w = mw.mps_window
        if e == "E1":
            w.spin_eps.editingFinished.emit()
            self.pump(app, 0.02)
            self.snapshot(mw, "entry")
        elif e == "E2":
            mw.ui.pushButton_clusterch1.click()
            self.pump(app, 0.05)
            self.snapshot(mw, "entry")
        elif e == "E3":
            mw.ui.pushButton_remove_bad_cluster.click()
            self.pump(app, 0.05)
            self.snapshot(mw, "entry")
        elif e == "E4":
            rings: Dict[str, Any] = {}
            w.btn_rings.click()
            self.pump(app, 0.05)
            rw = mw.rings_window
            assert rw is not None, "the rings window did not open"
            for k, (mode, guard) in enumerate(RINGS_STEPS):
                if k > 0:
                    if rw.combo_mode.currentData() != mode:
                        rw.combo_mode.setCurrentIndex(rw.combo_mode.findData(mode))
                    if float(rw.spin_guard.value()) != guard:
                        rw.spin_guard.setValue(guard)
                    self.pump(app, 0.02)
                n_msg = len(self.messages)
                target = self.next_path(f"rings_{mode}_{int(guard)}")
                self.save_queue = [target]
                rw._on_export()
                self.save_queue = []
                stem, ext = os.path.splitext(target)
                pair_path = f"{stem}_pairs{ext}"
                rings[f"{mode}_{int(guard)}"] = self.norm.value({
                    "window": self.record_rings_window(rw),
                    "segments_csv": _csv_text_table(target) if os.path.exists(target) else None,
                    "pairs_csv": _csv_text_table(pair_path) if os.path.exists(pair_path) else None,
                    "export_messages": self.messages[n_msg:],
                    "calls": self.calls})
                self.calls = []
            self.snapshot(mw, "entry")
            self.snapshots["entry"]["rings"] = rings
        elif e == "E5":
            from tools import mps_batch as mb
            bw = mw.open_batch()
            self.pump(app, 0.05)
            if bw._sync is not None:
                bw._sync()
            settings = mb.BatchSettings.from_settings(bw.settings)
            import dataclasses
            batch = {k: repr(v) if isinstance(v, float) else v
                     for k, v in dataclasses.asdict(settings).items()}
            self.snapshot(mw, "entry")
            self.snapshots["entry"]["batch_settings"] = batch
            self.snapshots["entry"]["batch_settings_describe"] = settings.describe()
        else:
            raise ValueError(e)

        mw.close()
        self.pump(app, 0.05)
        settings_file = os.path.join(self.settings_dir, "mps_analysis_settings.json")
        if os.path.exists(settings_file):
            with open(settings_file, encoding="utf-8") as f:
                self.snapshots["settings_file"] = self.norm.value(json.load(f))
        else:
            self.snapshots["settings_file"] = None
        extra: Dict[str, Any] = {}
        if hasattr(self, "chosen_component"):
            extra["chosen_component"] = self.chosen_component
        if hasattr(self, "guide_drawn"):
            extra["guide_drawn"] = self.guide_drawn
        return {"axon": self.axon, "scenario": self.scenario, "entry": self.entry,
                "snapshots": self.snapshots, "harness": extra,
                "harness_errors": self.errors}


def _plain(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        return repr(float(v))
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return describe_value(v, {})
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, str)):
        return v
    return repr(v)


def _settings_dict(settings: Any) -> Dict[str, Any]:
    import dataclasses
    return {k: _plain(v) for k, v in dataclasses.asdict(settings).items()}


def cell_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)        # A:S1:E1
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args(argv)
    axon, scenario, entry = args.cell.split(":")
    cell = Cell(axon, scenario, entry, args.cell_dir, args.inputs)
    t0 = time.perf_counter()
    try:
        result = cell.run()
        result["ok"] = True
    except Exception:  # noqa: BLE001 - reported in the result
        result = {"axon": axon, "scenario": scenario, "entry": entry, "ok": False,
                  "error": cell.norm.text(traceback.format_exc()),
                  "snapshots": cell.snapshots}
    result["seconds"] = round(time.perf_counter() - t0, 1)
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True, default=_plain)
    return 0


# =========================================================================
#  The run (orchestrator)
# =========================================================================
def environment() -> Dict[str, Any]:
    """The interpreter and the numerical libraries the golden was made with.

    The last digits of some results (the axis least-squares fit, every
    digest) depend on the scipy / scikit-learn versions, so a golden is
    only comparable with a run made with the same ones: the project's
    venv. The interpreter path is recorded with the home folder as "~".
    """
    import importlib

    versions: Dict[str, Any] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "sklearn", "pyqtgraph", "h5py", "tifffile"):
        try:
            versions[name] = str(importlib.import_module(name).__version__)
        except Exception as error:  # noqa: BLE001 - recorded
            versions[name] = f"not importable: {error}"
    try:
        from PyQt5 import QtCore
        versions["PyQt5"] = QtCore.PYQT_VERSION_STR
        versions["Qt"] = QtCore.QT_VERSION_STR
    except Exception as error:  # noqa: BLE001 - recorded
        versions["PyQt5"] = f"not importable: {error}"
    exe = os.path.abspath(sys.executable)
    home = os.path.expanduser("~")
    if exe.lower().startswith(home.lower()):
        exe = "~" + exe[len(home):]
    return {"interpreter": exe, "versions": versions}


def _user_settings_state() -> Optional[Tuple[str, int]]:
    p = os.path.join(REPO, "mps_analysis_settings.json")
    if not os.path.exists(p):
        return None
    return _sha_file(p), os.stat(p).st_mtime_ns


def _blobify(store: Dict[str, Any], value: Any) -> Any:
    """Bottom-up: every dict or list whose JSON exceeds BLOB_MIN_BYTES is
    stored once in ``store`` and replaced by {"$blob": sha256[:24]}."""
    if isinstance(value, dict):
        value = {k: _blobify(store, v) for k, v in value.items()}
    elif isinstance(value, list):
        value = [_blobify(store, v) for v in value]
    else:
        return value
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    if len(text) < BLOB_MIN_BYTES:
        return value
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]
    store.setdefault(key, value)
    return {"$blob": key}


def expand(golden: Dict[str, Any], value: Any) -> Any:
    """The inverse of _blobify."""
    blobs = golden.get("blobs", {})
    if isinstance(value, dict):
        if set(value) == {"$blob"}:
            return expand(golden, blobs[value["$blob"]])
        return {k: expand(golden, v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(golden, v) for v in value]
    return value


def _compact_table(table: Optional[Dict[str, Any]], per_column: bool) -> Any:
    if table is None or not per_column:
        return table
    header = table["header"]
    rows = table["rows"]
    cols = {}
    for j, name in enumerate(header):
        h = hashlib.sha256()
        for r in rows:
            h.update((r[j] if j < len(r) else "\x00").encode("utf-8"))
            h.update(b"\x1f")
        cols[name] = h.hexdigest()[:24]
    return {"header": header, "n_rows": len(rows), "column_sha256": cols}


def _normalise_exported_at(table: Optional[Dict[str, Any]]) -> None:
    if not table or "exported_at" not in table.get("header", []):
        return
    j = table["header"].index("exported_at")
    for r in table["rows"]:
        if j < len(r) and r[j]:
            r[j] = "<TIME>"


def compact_cell(result: Dict[str, Any], blobs: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the table normalisation and store big parts as blobs."""
    out = json.loads(json.dumps(result))
    for label, snap in (out.get("snapshots") or {}).items():
        if label == "settings_file" or not isinstance(snap, dict):
            continue
        exp = snap.get("export_axon")
        if exp and exp.get("tables"):
            for key, table in exp["tables"].items():
                _normalise_exported_at(table)
                exp["tables"][key] = _compact_table(table, key == "localizations")
    out["snapshots"] = _blobify(blobs, out.get("snapshots") or {})
    return out


def run_all(out_dir: str, workers: int, axons: List[str], cells: Optional[List[str]]) -> int:
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        print(f"{out_dir} is not empty; give a new folder")
        return 2
    os.makedirs(out_dir, exist_ok=True)
    before = _user_settings_state()
    inputs_dir = os.path.join(out_dir, "inputs")
    t0 = time.perf_counter()
    described = make_inputs(inputs_dir)
    with open(os.path.join(inputs_dir, "inputs.json"), "w", encoding="utf-8") as f:
        json.dump({"axons": described}, f, indent=1, sort_keys=True)
    print(f"inputs written in {time.perf_counter() - t0:.0f} s")

    grid = [f"{a}:{s}:{e}" for a in axons for s in SCENARIOS for e in ENTRIES]
    if cells:
        grid = [g for g in grid if any(g.endswith(c) or g == c for c in cells)]
    env = dict(os.environ)
    env.update(QT_QPA_PLATFORM="offscreen", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="0")
    for k in [k for k in env if k.startswith("MPS_") and k != "MPS_SELECTION_LOG_DIR"]:
        env.pop(k)

    def one(cell: str) -> Dict[str, Any]:
        name = cell.replace(":", "_")
        cell_dir = os.path.join(out_dir, "cells", name)
        os.makedirs(cell_dir, exist_ok=True)
        result_path = os.path.join(cell_dir, "result.json")
        with open(os.path.join(cell_dir, "stdout.txt"), "w", encoding="utf-8") as log:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--run-cell", "--cell", cell,
                 "--cell-dir", cell_dir, "--inputs", inputs_dir, "--result", result_path],
                cwd=cell_dir, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        if not os.path.exists(result_path):
            return {"cell": cell, "ok": False, "error": f"no result (exit {proc.returncode})"}
        with open(result_path, encoding="utf-8") as f:
            res = json.load(f)
        res["cell"] = cell
        print(f"  {cell:14s} {'ok ' if res.get('ok') else 'FAIL'} {res.get('seconds', '?')} s",
              flush=True)
        return res

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as pool:
        results = list(pool.map(one, grid))

    blobs: Dict[str, Any] = {}
    golden_cells: Dict[str, Any] = {}
    timing = {}
    for res in results:
        cell = res.pop("cell")
        timing[cell] = res.pop("seconds", None)
        golden_cells[cell] = compact_cell(res, blobs)
    after = _user_settings_state()
    golden = {
        "golden_version": GOLDEN_VERSION,
        "what": "UI stage 2 IMPL-0 golden: simulated axons x scenarios x entry points, "
                "captured on main 501f2e9 before any stage-2 edit",
        "environment": environment(),
        "axons": described,
        "scenarios": SCENARIO_TEXT, "entries": ENTRY_TEXT,
        "rings_steps": [list(s) for s in RINGS_STEPS],
        "normalisation": [
            "absolute paths: cell folder -> <CELL>, input folder -> <INPUTS>, repository -> <REPO>",
            "exported_at column -> <TIME>",
            "arrays in keyword arguments -> shape, dtype, sha256 of bytes; analyses -> analysis#k",
            "localizations table -> header, row count, sha256 per column of the written text",
            "identical parts stored once under their sha256 (blobs)",
            "program texts quoting a figure of unpublished data -> <REDACTED: ...> (REDACTIONS)",
        ],
        "cells": golden_cells,
        "blobs": blobs,
    }
    with open(os.path.join(out_dir, "golden.json"), "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=1, sort_keys=True, ensure_ascii=False)
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump({"cells": timing, "total_seconds": round(time.perf_counter() - t0, 1)},
                  f, indent=1, sort_keys=True)
    failed = [c for c, r in golden_cells.items() if not r.get("ok")]
    harness = [c for c, r in golden_cells.items() if r.get("harness_errors")]
    print(f"{len(golden_cells)} cells, {len(failed)} failed, {len(harness)} with harness errors; "
          f"{time.perf_counter() - t0:.0f} s")
    if before != after:
        print("ERROR: the user's settings file changed during the run")
        return 3
    print("user settings file untouched" if before else "no user settings file existed; none created")
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--run-cell":
        return cell_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--axons", default="A,B")
    ap.add_argument("--cells", default="")
    args = ap.parse_args(argv)
    return run_all(args.out, args.workers, [a for a in args.axons.split(",") if a],
                   [c for c in args.cells.split(",") if c] or None)


if __name__ == "__main__":
    sys.exit(main())
