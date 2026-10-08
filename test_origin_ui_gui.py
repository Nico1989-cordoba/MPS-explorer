# -*- coding: utf-8 -*-
"""
Offscreen test of the origin widgets (UI stage 2, design 12.3, 12.4, 12.8; ``tools/mps_origin_ui.py``).

Section 1 (IMPL-B) checks the widgets on their own; IMPL-F appends the windows that carry them.

  1. ``ParamField`` around an existing editor: the editor keeps its object name, limits and value; the badge reads
     the registry (paper -> user -> paper as a value departs and comes back, the default named in the tooltip); a
     value outside the documented range is flagged and NOT clamped; ``OriginBadge`` and ``RangeHint`` alone; the
     "ad hoc" badge in the warning text colour, "user" bold; ``MoreLine`` collapsed by default with the registry's
     line; ``DetailsPanel`` read-only, selectable, collapsible, filled from plain tuples.

Nothing is computed on data (R8): the widgets show registry values and numbers handed to them.

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
        assert field.hint.text() == "pending B6"
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
        assert b.text() == "ad hoc" and verdict("warn", dark=False) in b.styleSheet() and "SCI-7" in b.toolTip()
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
        assert "SCI-3" in m.toolTip() and "Documented range" in m.toolTip()
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
