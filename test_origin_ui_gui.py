# -*- coding: utf-8 -*-
"""
Offscreen test of the origin widgets (UI stage 2, design 12.3, 12.4, 12.8; ``tools/mps_origin_ui.py``).

Section 1 (IMPL-B) checks the widgets on their own; IMPL-F appends the windows that carry them.

  1. ``ParamField`` around an existing editor: the editor keeps its object name, limits and value; the badge reads
     the registry (paper -> user -> paper as a value departs and comes back, the default named in the tooltip); a
     value outside the documented range is flagged and NOT clamped; ``OriginBadge`` and ``RangeHint`` alone; the
     "ad hoc" badge in the warning text colour, "user" bold; ``MoreLine`` collapsed by default with the registry's
     line; ``DetailsPanel`` read-only, selectable, collapsible, filled from plain tuples.
  2. (IMPL-F) The windows on simulated axon A: the Measurement panel (each channel's pixel size with today's source
     words, the z calibration blank with its inputs disabled, a 2D file, a file's own record shown and not read; it
     stores nothing and writes no file); the KS p cell's tooltip with its text unchanged; the details panels without
     the p; the 1NN reference off and the CDF crossing outside the paper role; every registry key whose home is a
     stage-2 window carries a badge there; the axoplasm and rings badges follow their controls.

Nothing is computed on real data (R8): section 1 shows registry values and numbers handed to the widgets; section 2 runs on the simulated axon of the stage-2 golden.

Run:  venv\\Scripts\\python.exe test_origin_ui_gui.py      (offscreen; a few seconds)

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from typing import Callable, Optional  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
        for line in traceback.format_exc().strip().splitlines()[-4:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def main() -> int:
    print("=" * 100)
    print("ORIGIN WIDGETS (UI stage 2, design 12.3-12.8) OFFSCREEN TEST")
    print("=" * 100)
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    from tools import mps_origin_ui as oui
    from tools import mps_param_registry as reg
    from tools.mps_plot_style import verdict

    print("\n1. The widgets on their own")

    def param_field_eps() -> str:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setObjectName("spin_eps")
        spin.setRange(0.1, 1000.0)
        spin.setValue(25.0)
        spin.setToolTip("DBSCAN eps.")
        field = oui.ParamField("dbscan.eps_nm", spin)
        assert field.editor is spin and spin.objectName() == "spin_eps"
        assert (spin.minimum(), spin.maximum()) == (0.1, 1000.0), "the editor's limits must stay today's"
        assert field.badge.text() == "paper" and "font-weight: bold" not in field.badge.styleSheet()
        assert field.hint.text() == ""          # B6 documents no range for the analysis eps
        spin.setValue(30.0)
        assert field.badge.text() == "user" and "font-weight: bold" in field.badge.styleSheet()
        assert "you set 30 nm; default 25 nm (paper)" in field.badge.toolTip()
        assert spin.toolTip().startswith("DBSCAN eps.") and "default 25 nm" in spin.toolTip()
        spin.setValue(25.0)
        assert field.badge.text() == "paper" and "you set" not in spin.toolTip()
        return "badge paper -> user (bold, default in the tip) -> paper; editor name and limits untouched"

    def param_field_out_of_range() -> str:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setObjectName("spin_half")
        spin.setRange(1.0, 5000.0)
        spin.setValue(90.0)
        field = oui.ParamField("slab.half_width_nm", spin)
        assert field.hint.text() == "40-200" and not field.outside
        spin.setValue(250.0)
        assert spin.value() == 250.0, "an out-of-range value must not be clamped"
        assert field.hint.text() == "outside 40-200" and field.outside
        assert verdict("warn", dark=False) in field.hint.styleSheet()
        assert field.badge.text() == "user" and "only flagged" in field.hint.toolTip()
        spin.setValue(90.0)
        assert field.hint.text() == "40-200" and not field.outside and field.badge.text() == "paper"
        int_spin = QtWidgets.QSpinBox()
        int_spin.setRange(1, 10000)
        int_spin.setValue(10)
        f2 = oui.ParamField("dbscan.min_samples", int_spin)
        int_spin.setValue(8)
        assert f2.badge.text() == "user" and f2.value() == 8
        return "250 nm kept by the spin, hint 'outside 40-200' in the warning colour, back to '40-200' at 90"

    def badges_alone() -> str:
        b = oui.OriginBadge("axoplasm.margin_nm")
        assert b.text() == "ad hoc" and verdict("warn", dark=False) in b.styleSheet() and "later version" in b.toolTip()
        blank = oui.OriginBadge("measurement.z_calibration")
        assert blank.text() == "blank" and blank.objectName() == "badge_measurement_z_calibration"
        dark = oui.OriginBadge("dbscan.eps_nm", 30.0, dark=True)
        assert dark.text() == "user" and verdict("warn", dark=True) not in dark.styleSheet()
        hint = oui.RangeHint("slab.typed_range")
        assert hint.text() == "" and hint.isHidden()
        try:
            oui.OriginBadge("no.such.key")
        except KeyError:
            pass
        else:
            raise AssertionError("an unknown key must fail at once")
        combo = QtWidgets.QComboBox()
        combo.addItem("valley - cut at the density minimum", "valley")
        combo.addItem("paper", "paper")
        f = oui.ParamField("rings.mode", combo, show_range=False)
        assert f.badge.text() == "derived" and f.hint.isHidden()
        combo.setCurrentIndex(1)
        assert f.badge.text() == "user"
        return "ad hoc in the warning colour; blank; dark windows; an unknown key refused; combos follow their data"

    def more_line() -> str:
        m = oui.MoreLine("randomization")
        assert not m.is_expanded() and m.label.isHidden(), "a More line starts collapsed"
        assert m.label.text() == reg.more_line("randomization")
        m.button.setChecked(True)
        assert m.is_expanded() and not m.label.isHidden()
        assert "later version" in m.toolTip() and "Documented range" in m.toolTip()
        assert "SCI-" not in m.toolTip() and "record " not in m.toolTip(), m.toolTip()
        return "collapsed by default; the registry's line; every item's tooltip"

    def details_panel() -> str:
        p = oui.DetailsPanel("Details", object_name="details_axial")
        p.set_rows([("components (BIC)", "2"), ("density peak", "412.0 nm", "the automatic slab centre", "derived"),
                    ("mean Delta-Z", "- (one component)", "")])
        t = p.table
        assert t.rowCount() == 3 and t.item(1, 1).text() == "412.0 nm  [derived]"
        assert t.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        assert p.text().splitlines()[1] == "density peak: 412.0 nm [derived] (the automatic slab centre)"
        assert not p.is_collapsed()
        p.button.setChecked(False)
        assert p.is_collapsed() and t.isHidden()
        q = oui.DetailsPanel(collapsed=True)
        assert q.is_collapsed()
        return "read-only rows from plain tuples, badge beside the value, collapsible"

    check("ParamField: badge follows the editor (paper -> user -> paper), editor untouched", param_field_eps)
    check("ParamField: out of the documented range is flagged, never clamped", param_field_out_of_range)
    check("OriginBadge / RangeHint alone, combos, unknown keys", badges_alone)
    check("MoreLine", more_line)
    check("DetailsPanel", details_panel)

    # ------------------------------------------------------------------ 2. the windows (IMPL-F)
    print("\n2. The windows that carry them (simulated axon A; settings, logs and stores in a temp folder)")
    import functools
    import json
    import tempfile
    from types import SimpleNamespace
    work = tempfile.mkdtemp(prefix="test_origin_ui_gui_")
    os.environ["MPS_SELECTION_LOG_DIR"] = os.path.join(work, "selection_log")
    st: dict = {}
    msgs: list = []

    def files_under(folder: str) -> set:
        out = set()
        for root, _dirs, names in os.walk(folder):
            for n in names:
                out.add(os.path.relpath(os.path.join(root, n), folder))
        return out

    def pump(seconds: float = 0.1) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.005)

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
                    staticmethod(lambda *a, _k=kind, **k: (msgs.append((_k, a[1] if len(a) > 1 else "")),
                                                           QtWidgets.QMessageBox.Ok)[1]))
        from tools.mps_identity import AxonIdentity
        from tools.mps_selection_ui import set_app_store_dir
        mw = MPS_explorer.MPS_explorer()
        mw.columns_review_store_dir = os.path.join(work, "review_store")
        set_app_store_dir(mw.columns_review_store_dir)
        assert mw.measurement_panel.text() == "No file loaded."
        mw.radioButton_circROI.setChecked(True)
        assert mw.load_channel1(os.path.join(inputs, "sim_axon_A.hdf5"), 0)
        assert mw.load_channel2(os.path.join(inputs, "sim_axon_B.hdf5"), 0)
        mw.identity = AxonIdentity(genotype="SIM", protein="betaII-spectrin", animal="sim", sample="origin",
                                   roi_name="roi", axon_name="axonA")
        mw.identity_path = mw._identity_source()
        mw.scatterplot()
        d = described["A"]
        size = 2.0 * (d["ring_radius_nm"] + 450.0) / MPS_explorer.ROI_DIAMETER_SCALE_FACTOR
        roi = mw.circular_roi
        roi.setSize((size, size), update=False, finish=False)
        roi.setPos((d["centre_nm"][0] - size / 2.0, d["centre_nm"][1] - size / 2.0), update=False, finish=False)
        mw.update_ROI()
        mw.ui.pushButton_clusterch1.click()
        pump(0.2)
        w = mw.mps_window
        assert w is not None and w.analysis is not None
        st.update(mw=mw, w=w, inputs=inputs, settings_dir=settings_dir)
        return f"{w.analysis.n_clusters_kept} clusters"

    def measurement_panel() -> str:
        mw = st["mw"]
        p = mw.measurement_panel
        lines = p.text().splitlines()
        src = reg.pixel_source_words(mw.locs1.pixel_size_source)
        assert lines[0] == f"Channel 1: pixel size {mw.locs1.pixel_size_nm:g} nm ({src})", lines[0]
        assert lines[1].startswith("Channel 2: pixel size "), lines[1]
        assert lines[2] == "z calibration: - (none applied) [blank] - z as fitted by the localization software"
        assert p.badge_z.text() == "blank" and p.badge_bead.text() == "blank"
        assert not p.btn_calibration.isEnabled() and not p.btn_bead.isEnabled()
        assert "later version" in p.btn_calibration.toolTip() and "later version" in p.btn_bead.toolTip()
        assert "SCI-" not in p.btn_calibration.toolTip() + p.lbl_records.toolTip()
        assert set(p.keys_shown()) == {"measurement.pixel_size_nm", "measurement.z_calibration",
                                       "measurement.bead_stack"}
        # the same words as the MPS analysis window's provenance line (one dictionary)
        assert f"({src})" in st["w"].lbl_provenance.text()
        # a 2D file, and one with its own record: shown, never read
        from tools.mps_measurement_panel import ChannelFile
        flat = SimpleNamespace(pixel_size_nm=122.0, pixel_size_source="remembered", z_calibration=None, is_3d=False)
        rec = SimpleNamespace(pixel_size_nm=130.0, pixel_size_source="yaml", z_calibration={"X Coefficients": [1]},
                              is_3d=True)
        before_files = files_under(work)
        with open(os.path.join(st["settings_dir"], "mps_analysis_settings.json"), encoding="utf-8") as f:
            before_settings = f.read()
        p.set_files([ChannelFile.from_localizations(1, "a.hdf5", flat), ChannelFile.from_localizations(2, "b", rec)])
        text = p.text()
        assert "Channel 1: 2D file: no z" in text and "typed by hand for this folder earlier" in text
        assert "carries a Picasso 'Z Calibration' record (Data quality reads its calibrated range; z is not " \
               "corrected with it)" in text
        # 2D files only: no z rows (m14)
        p.set_files([ChannelFile.from_localizations(1, "a.hdf5", flat)])
        assert "z calibration" not in p.text() and not p.z_row.isVisibleTo(p) and not p.bead_row.isVisibleTo(p)
        with open(os.path.join(st["settings_dir"], "mps_analysis_settings.json"), encoding="utf-8") as f:
            assert f.read() == before_settings, "the Measurement panel must store nothing"
        assert files_under(work) == before_files, "the Measurement panel must write no file"
        mw._refresh_measurement_panel()
        assert p.text().splitlines()[0] == lines[0]
        return "pixel size with today's source words; z calibration blank, inputs disabled; 2D / own record; " \
               "nothing stored or written"

    def results_window() -> str:
        w = st["w"]
        from tools import mps_axon_map_layers as L
        # the KS p cell: the tooltip, its text untouched
        rows = [w.table.item(r, 0).text() for r in range(w.table.rowCount())]
        ks = rows.index("Randomization KS test")
        cell = w.table.item(ks, 1)
        assert "four times" in cell.toolTip() and "Monte Carlo" in cell.toolTip()
        assert "P-R" not in cell.toolTip() and "SCI-" not in cell.toolTip(), cell.toolTip()
        shown = w._shown()
        assert cell.text() == next(m for n, m, _p, _note in shown.summary_rows() if n == rows[ks])
        # details panels: existing numbers only, never the KS p
        cdf = " ".join(f"{r[0]} {r[1]}" for r in L.cdf_details(shown)).lower()
        assert "ks statistic d" in cdf and "p-value" not in cdf and "ks p" not in cdf
        area = w.details_area.rows()
        assert [r[0] for r in area] == ["median area", "median effective radius", "reference area",
                                        "reference effective radius"] and area[2][3] == "paper"
        assert w.details_area.is_collapsed() and w.axial.details.is_collapsed()
        assert w.nn.details_nn.rows() and w.nn.details_cdf.rows()
        # the 1NN reference starts off; the CDF crossing is never in the paper role
        assert "nn_reference" not in L.NN_ON
        crossing = [lay for lay in L.cdf_layers(shown) if lay.key == "cdf_crossing"]
        assert all(lay.style.role != "paper" for lay in crossing)
        return f"KS p tooltip, text unchanged; area details {len(area)} rows; crossing in '" + \
            (crossing[0].style.role if crossing else "-") + "'"

    def badges_cover_registry() -> str:
        mw, w = st["mw"], st["w"]
        mw.show_axoplasm_panel()
        aw = mw.axoplasm_window
        aw.load_tubulin(os.path.join(st["inputs"], "wf_tubulin_A.tif"))
        aw.flush()
        pump(0.3)
        w.btn_rings.click()
        pump(0.3)
        rw = mw.rings_window
        assert rw is not None
        shown = set()
        for window in (mw, w, aw, rw):
            for badge in window.findChildren(oui.OriginBadge):
                shown.add(badge.key)
        shown.update(mw.measurement_panel.keys_shown())
        for window in (w,):
            for more in window.findChildren(oui.MoreLine):
                shown.update(k for k, *_r in reg.more_items(more.group))
        homes = ("strip", "rings", "axoplasm", "measurement")
        missing = [e.key for e in reg.entries() if e.home in homes and e.key not in shown]
        assert not missing, f"registry keys of a stage-2 window with no badge: {missing}"
        unknown = [k for k in shown if k not in reg.keys()]
        assert not unknown, unknown
        # the axoplasm badges follow the panel: Otsu derived, a hand-set threshold user, the margin ad hoc
        assert aw.badge_threshold.text() == "derived" and aw.badge_margin.text() == "ad hoc"
        assert aw.field_smooth.badge.text() == "derived"
        aw.spin_threshold.setValue(aw.spin_threshold.value() + 1.0)
        assert aw.badge_threshold.text() == "user"
        aw.btn_otsu.click()
        pump(0.2)
        assert aw.badge_threshold.text() == "derived"
        aw.spin_margin.setValue(700.0)
        assert aw.badge_margin.text() == "user" and aw.hint_margin.text() == "outside 100-500"
        assert aw.spin_margin.value() == 700.0, "never clamped"
        aw.spin_margin.setValue(250.0)
        # the rings window: every value of its line with its badge
        text = rw.lbl_params.text()
        for words in ("eps 25 nm [paper]", "min samples 10 [paper]", "mode valley [derived]", "guard 0 nm [derived]"):
            assert words in text, (words, text)
        assert rw.field_mode.badge.text() == "derived" and rw.field_guard.badge.text() == "derived"
        return f"{len([k for k in shown if reg.info(k).home in homes])} keys shown; axoplasm and rings badges follow"

    def user_settings_untouched() -> str:
        path, mtime = st["user_settings"]
        now = os.path.getmtime(path) if os.path.exists(path) else None
        assert now == mtime, "the user's settings file was written"
        return "the user's settings file untouched"

    check("setup (main window, simulated axon A, MPS analysis)", setup)
    check("Measurement panel: pixel size and source, z calibration blank, inputs disabled, stores nothing",
          measurement_panel)
    check("MPS analysis window: KS p tooltip, details panels without the p, 1NN reference off, crossing neutral",
          results_window)
    check("every registry key of a stage-2 window carries a badge; axoplasm and rings badges", badges_cover_registry)
    check("the user's settings file", user_settings_untouched)
    _ = json

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
