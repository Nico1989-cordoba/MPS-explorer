# -*- coding: utf-8 -*-
"""
Offscreen test of the clusters cut by the Picasso pick in the program (decision P-R59; docs/cut_clusters.md).

On simulated axon A (``golden_ui_stage2.make_inputs``; R8: no real data), with settings, logs and stores in a temp
folder. The simulated file plays the whole field; a per-axon file is cut out of it here with a polygon that runs
through the ring, written as Picasso v0.9.10's "Save picked localizations" writes it (a ``group`` column and the
Pick step in the YAML), next to the pick file "Save picks" writes.

  1. the per-axon file and its pick are found; the NeNA comes from the file or, measured on the whole-field file,
     from the folder's store; the analysis counts the cut clusters; the strip says "k of K (x %)";
  2. the map: the "Picasso pick boundary" group, its two rows ticked in the MPS view, the cut centres and the
     boundary drawn, the title names them; off with the images' colouring; screenshots;
  3. "Exclude cut clusters": off by default; ticked, the analysis leaves them out and the export says so;
     unticked, back to the same numbers;
  4. two picks that hold the file and nothing to tell them apart: not evaluated, the reason says "ambiguous", the
     dialog lists both, choosing one applies the rule;
  5. a pick file pointed to by hand (the dialog's button) for a file with none beside it;
  6. a file no pick holds, and a file with no pick file anywhere: "boundary unknown: not evaluated" in the export;
  7. the user's settings file is untouched.

Screenshots go to the folder in CUT_SCREENS_DIR when it is set.

Run:  venv\\Scripts\\python.exe test_cut_clusters_gui.py      (offscreen; about a minute)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv  # noqa: E402
import functools  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Callable, Dict, List, Optional  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

PASSED = 0
FAILED = 0
T0 = time.perf_counter()


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a test; it reports everything
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-6:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def write_picked(src_h5: str, out_h5: str, polygon_px: np.ndarray, *, group: Optional[int],
                 pick_step: Optional[Dict[str, Any]]) -> int:
    """The localizations of ``src_h5`` inside ``polygon_px`` (Picasso's rule), as Picasso writes a picked file."""
    import h5py
    import yaml

    from tools import mps_picks

    with h5py.File(src_h5, "r") as h:
        rows = h["locs"][:]
    keep = mps_picks.polygon_contains(rows["x"].astype(float), rows["y"].astype(float), polygon_px)
    rows = rows[keep]
    if group is not None:
        dtype = rows.dtype.descr + [("group", "<i4")]
        out = np.zeros(rows.size, dtype=dtype)
        for name in rows.dtype.names:
            out[name] = rows[name]
        out["group"] = group
        rows = out
    with h5py.File(out_h5, "w") as h:
        h.create_dataset("locs", data=rows)
    with open(os.path.splitext(src_h5)[0] + ".yaml", encoding="utf-8") as f:
        sidecar = f.read()
    if pick_step is not None:
        sidecar = sidecar.rstrip("\n") + "\n---\n" + yaml.dump(pick_step)
    with open(os.path.splitext(out_h5)[0] + ".yaml", "w", encoding="utf-8") as f:
        f.write(sidecar)
    return int(keep.sum())


def main() -> int:
    print("=" * 100)
    print("CLUSTERS CUT BY THE PICASSO PICK (P-R59) OFFSCREEN TEST")
    print("=" * 100)
    import yaml

    work = tempfile.mkdtemp(prefix="test_cut_clusters_gui_")
    os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(work, "selection_log")
    screens = os.environ.get("CUT_SCREENS_DIR", "")
    st: Dict[str, Any] = {}
    msgs: List[Any] = []

    def pump(seconds: float = 0.1) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)

    def shot(widget: Any, name: str) -> None:
        if screens:
            os.makedirs(screens, exist_ok=True)
            pump(0.2)
            widget.grab().save(os.path.join(screens, name))

    def export_row() -> Dict[str, str]:
        st["n_export"] = st.get("n_export", 0) + 1
        target = os.path.join(work, "exports", f"{st['n_export']:02d}_axon.csv")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        st["save_queue"] = [target]
        assert st["mw"].export_axon() is not None, msgs[-3:]
        from tools import axon_export
        with open(axon_export.table_paths(target)["axon"], encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        return rows[0]

    def load_and_cluster(path: str) -> Any:
        import MPS_explorer
        mw = st["mw"]
        assert mw.load_channel1(path, 0), msgs[-3:]
        # the identity confirmed for this file: the export asks nothing
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        d = st["described"]["A"]
        size = 2.0 * (d["ring_radius_nm"] + 450.0) / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((d["centre_nm"][0] - size / 2.0, d["centre_nm"][1] - size / 2.0), update=False, finish=False)
        mw.update_ROI()
        mw.ui.pushButton_clusterch1.click()
        pump(0.2)
        w = mw.mps_window
        assert w is not None and w.analysis is not None
        return w

    def setup() -> str:
        import golden_ui_stage2 as gold
        inputs = os.path.join(work, "inputs")
        described = gold.make_inputs(inputs)
        os.chdir(work)
        from tools import mps_settings
        settings_dir = os.path.join(work, "settings")
        os.makedirs(settings_dir, exist_ok=True)
        original = mps_settings.settings_path
        user_settings = original()
        st["user_settings"] = (str(user_settings), os.path.getmtime(user_settings)
                               if os.path.exists(user_settings) else None)
        mps_settings.settings_path = lambda directory=None: original(directory or settings_dir)
        import MPS_explorer
        MPS_explorer.load_settings = functools.partial(mps_settings.load_settings, directory=settings_dir)
        MPS_explorer.save_settings = functools.partial(mps_settings.save_settings, directory=settings_dir)
        for kind in ("information", "warning", "critical", "question"):
            setattr(QtWidgets.QMessageBox, kind,
                    staticmethod(lambda *a, _k=kind, **k: (msgs.append((_k, a[2] if len(a) > 2 else "")),
                                                           QtWidgets.QMessageBox.Ok)[1]))
        from tools import axon_export_ui

        class FakeExportDialog:
            def __init__(self, path: str, identity: Any, **_kw: Any) -> None:
                self._identity = identity

            def exec_(self) -> int:
                return QtWidgets.QDialog.Accepted

            def path(self) -> str:
                return st["save_queue"].pop(0)

            def identity(self) -> Any:
                return self._identity

            def wants_clusters(self) -> bool:
                return False

            def wants_localizations(self) -> bool:
                return False
        axon_export_ui.ExportAxonDialog = FakeExportDialog  # type: ignore[misc,assignment]
        from tools import export_ui
        export_ui.resolve_duplicates = lambda *_a, **_k: "append"  # type: ignore[assignment]
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir

        # the pick: drawn snugly ON the ring over its right third (the ring's median radius), with room
        # elsewhere -- the hand-drawn pick that cuts clusters
        px = gold.PIXEL_NM
        d = described["A"]
        cx, cy, r = d["centre_nm"][0], d["centre_nm"][1], d["ring_radius_nm"]
        theta = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        rad = np.where(np.abs(theta) < np.pi / 3, r, r + 600.0)
        poly_nm = np.column_stack([cx + rad * np.cos(theta), cy + rad * np.sin(theta)])
        poly_px = poly_nm / px
        picked_dir = os.path.join(inputs, "picked")
        os.makedirs(picked_dir)
        area = float(abs(np.sum(poly_px[:, 0] * np.roll(poly_px[:, 1], -1)
                                - np.roll(poly_px[:, 0], -1) * poly_px[:, 1])) / 2 * (px * 1e-3) ** 2)
        step = {"Generated by": "Picasso v0.9.10 Render : Pick", "Pick Shape": "Polygon",
                "Pick Areas (um^2)": [area], "Area (um^2)": area, "Number of picks": 1}
        whole = os.path.join(inputs, "sim_axon_A.hdf5")
        n = write_picked(whole, os.path.join(picked_dir, "sim_axon_A_picked_axon1.hdf5"), poly_px, group=0,
                         pick_step=step)
        verts = poly_px.tolist()
        with open(os.path.join(picked_dir, "sim_axon_A_picks.yaml"), "w", encoding="utf-8") as f:
            yaml.dump({"Shape": "Polygon", "Vertices": [verts + [verts[0]]]}, f)
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(work, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        mw.radioButton_circROI.setChecked(True)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="cut",
                                   roi_name="roi", axon_name="axonA")
        st.update(mw=mw, described=described, inputs=inputs, picked_dir=picked_dir, whole=whole, poly_px=poly_px,
                  poly_nm=poly_nm, settings_dir=settings_dir, step=step)
        return f"per-axon file of {n:,} localizations cut out of the simulated field with a polygon"

    def found_and_counted() -> str:
        mw = st["mw"]
        w = load_and_cluster(os.path.join(st["picked_dir"], "sim_axon_A_picked_axon1.hdf5"))
        mw.identity_path = mw._identity_source()
        session = mw.cut_session_now()
        assert session is not None and session.match.status == "matched", session.match.message
        assert "group agrees" in session.match.message and "area agrees" in session.match.message
        own = w.analysis.cut
        assert own is not None, "a pick file was found: the analysis is told"
        first = own.status
        if not own.evaluated:
            assert own.status == "NeNA unknown: not evaluated", own.status
        # the whole-field file, measured from the dialog's button
        dlg = mw.open_pick_boundary()
        dlg.ask_file = lambda *_a: st["whole"]
        dlg._on_nena()
        pump(0.2)
        cut = w.analysis.cut
        assert cut is not None and cut.evaluated, (cut.status, cut.reason)
        assert cut.nena_source.startswith("stored for this folder: whole-field file sim_axon_A.hdf5"), cut.nena_source
        assert cut.k is not None and cut.k >= 1, cut.count_words()
        assert w.cut_strip.label.text().startswith(cut.count_words()), w.cut_strip.label.text()
        assert not w.cut_strip.chk_exclude.isChecked()
        entry = mw._cut_store_entry(session.path)
        assert abs(entry["nena_nm"] - cut.nena_nm) < 1e-9 and entry["nena_source_file"] == "sim_axon_A.hdf5"
        # cut clusters are at the right of the ring, by the edge
        assert all(c[0] > st["described"]["A"]["centre_nm"][0] + 0.5 * st["described"]["A"]["ring_radius_nm"]
                   for c in cut.cut_centroids), cut.cut_centroids
        row = export_row()
        assert row["table_version"] == "axon tables v9" and row["cut_status"] == "evaluated"
        assert row["n_clusters_cut"] == str(cut.k) and row["cut_excluded"] == "False"
        assert row["cut_boundary_shape"] == "Polygon" and "sim_axon_A_picks.yaml" in row["cut_boundary_source"]
        rows = {r[0]: r for r in w.analysis.summary_rows()}
        assert rows["Clusters cut by the pick"][1] == cut.count_words()
        st["w"], st["cut"] = w, cut
        shot(dlg, "pick_boundary_dialog.png")
        dlg.close()
        return f"first '{first}', then {cut.count_words()} at NeNA {cut.nena_nm:.1f} nm from the whole field"

    def the_map() -> str:
        w, cut = st["w"], st["cut"]
        m = w.axon_map
        layers = m._layers
        assert "c_cut" in layers and "pick_boundary" in layers
        assert layers["c_cut"].enabled and layers["c_cut"].count == cut.k
        assert layers["pick_boundary"].enabled and len(layers["pick_boundary"].data["x"]) == 73
        state = m.layers.state()
        assert state.get("c_cut") and state.get("pick_boundary"), state
        assert "clusters cut by the pick" in m.lbl_line1.text() + m.lbl_line2.text() + str(
            m.plot_item().titleLabel.text), "the title names the layer"
        w.resize(1366, 728)
        w.show()
        shot(w, "cut_layer_results_window.png")
        shot(m.plot, "cut_layer_map.png")
        # off with the images' colouring
        from tools import mps_axon_map_layers as L
        inp = m.inputs()
        inp.colour_by = "images"
        groups = {g.key: g for g in L.build_map(inp)}
        assert all(not ly.enabled for ly in groups["pick"].layers)
        inp.colour_by = "status"
        return f"{cut.k} hexagons and the boundary drawn and ticked; disabled with the images' colouring"

    def exclusion() -> str:
        mw, w, cut = st["mw"], st["w"], st["cut"]
        kept = w.analysis.n_clusters_kept
        perimeter = w.analysis.perimeter_um
        w.cut_strip.chk_exclude.setChecked(True)
        pump(0.2)
        a = w.analysis
        assert mw.exclude_cut_clusters and a.cut.excluded and a.n_clusters_kept == kept - cut.k
        assert w.cut_strip.label.text().endswith("cut, excluded"), w.cut_strip.label.text()
        row = export_row()
        assert row["cut_excluded"] == "True" and int(row["n_clusters_kept"]) == kept - cut.k
        shot(w, "cut_layer_excluded.png")
        w.cut_strip.chk_exclude.setChecked(False)
        pump(0.2)
        b = w.analysis
        assert not b.cut.excluded and b.n_clusters_kept == kept and b.perimeter_um == perimeter
        return f"ticked: {a.n_clusters_kept} of {kept} clusters, row says so; unticked: back to {kept}"

    def ambiguous() -> str:
        mw = st["mw"]
        amb = os.path.join(st["inputs"], "ambiguous")
        os.makedirs(amb)
        write_picked(st["whole"], os.path.join(amb, "sim_axon_A_picked_2.hdf5"), st["poly_px"], group=None,
                     pick_step=None)
        verts = st["poly_px"].tolist()
        c = st["poly_px"].mean(axis=0)
        bigger = (c + 1.5 * (st["poly_px"] - c)).tolist()
        with open(os.path.join(amb, "picks_1.yaml"), "w", encoding="utf-8") as f:
            yaml.dump({"Shape": "Polygon", "Vertices": [verts + [verts[0]], bigger + [bigger[0]]]}, f)
        w = load_and_cluster(os.path.join(amb, "sim_axon_A_picked_2.hdf5"))
        cut = w.analysis.cut
        assert cut is not None and not cut.evaluated and "ambiguous" in cut.reason, (cut.status, cut.reason)
        assert w.cut_strip.label.text().startswith("more than one pick fits: choose one"), w.cut_strip.label.text()
        row = export_row()
        assert row["cut_status"].startswith("boundary unknown: not evaluated (pick ambiguous")
        assert row["n_clusters_cut"] == ""
        dlg = mw.open_pick_boundary()
        assert dlg.list_candidates.count() == 2
        shot(dlg, "pick_boundary_dialog_ambiguous.png")
        dlg.list_candidates.setCurrentRow(0)
        dlg._on_use()
        pump(0.2)
        cut2 = w.analysis.cut
        assert cut2.evaluated and "chosen by hand" in cut2.boundary_source, cut2.boundary_source
        dlg.close()
        return f"two picks: '{cut.reason[:60]}...'; chosen: {cut2.count_words()}"

    def by_hand() -> str:
        mw = st["mw"]
        lonely = os.path.join(work, "elsewhere", "deeper")
        os.makedirs(lonely)
        path = os.path.join(lonely, "axon_from_A.hdf5")
        write_picked(st["whole"], path, st["poly_px"], group=0, pick_step=st["step"])
        w = load_and_cluster(path)
        assert w.analysis.cut is None, "no pick file anywhere near: nothing told"
        row = export_row()
        assert row["cut_status"] == "boundary unknown: not evaluated (no Picasso pick boundary was given for this " \
                                    "file)", row["cut_status"]
        dlg = mw.open_pick_boundary()
        dlg.ask_file = lambda *_a: os.path.join(st["picked_dir"], "sim_axon_A_picks.yaml")
        dlg._on_file()
        pump(0.2)
        cut = w.analysis.cut
        assert cut is not None and cut.status in ("evaluated", "NeNA unknown: not evaluated"), cut.status
        assert "chosen by hand" in (cut.boundary_source or ""), cut.boundary_source
        entry = mw._cut_store_entry(path)
        assert entry.get("pick_files") == [os.path.join(st["picked_dir"], "sim_axon_A_picks.yaml")]
        dlg.close()
        return f"nothing beside it -> no rule; pointed to by hand -> {cut.status}; remembered for the folder"

    def unknown_boundary() -> str:
        # B's whole field in the folder of A's pick file: the pick does not hold it
        other = os.path.join(st["picked_dir"], "sim_axon_B_copy.hdf5")
        shutil.copy(os.path.join(st["inputs"], "sim_axon_B.hdf5"), other)
        shutil.copy(os.path.join(st["inputs"], "sim_axon_B.yaml"), os.path.splitext(other)[0] + ".yaml")
        w = load_and_cluster(other)
        cut = w.analysis.cut
        assert cut is not None and cut.status == "boundary unknown: not evaluated", cut.status
        assert "no pick holds every localization" in cut.reason, cut.reason
        row = export_row()
        assert "no pick holds every localization" in row["cut_status"] and row["n_clusters_cut"] == ""
        return f"'{row['cut_status'][:90]}...'"

    def user_settings_untouched() -> str:
        path, mtime = st["user_settings"]
        now = os.path.getmtime(path) if os.path.exists(path) else None
        assert now == mtime, "the user's settings file was written"
        return "the user's settings file untouched"

    check("setup (main window, a per-axon file cut out of simulated axon A, its pick file)", setup)
    check("the pick found, the NeNA from the whole field, k of K counted and exported", found_and_counted)
    check("the map: the pick group, its rows, the title, off with the images", the_map)
    check("Exclude cut clusters: off by default, on and off", exclusion)
    check("two picks hold the file: not evaluated, asked, then chosen", ambiguous)
    check("a pick file pointed to by hand", by_hand)
    check("a file no pick holds: boundary unknown, with the reason", unknown_boundary)
    check("the user's settings file", user_settings_untouched)

    app.processEvents()
    print("\n" + "=" * 100)
    print(f"total {time.perf_counter() - T0:.1f} s")
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 100)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
