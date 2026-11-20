"""Right-button drag pans a pyqtgraph view (manual test of 2026-10-23: "it would be nice to drag the map with the
right mouse button").

pyqtgraph's own ViewBox zooms on a right-button drag and opens its context menu on a right-button click. Here a
right-button drag moves the view exactly as a left-button drag does (the data under the cursor follows it); the
wheel still zooms; a right-button click with no drag still opens the menu, because pyqtgraph's scene only sends a
click when the mouse moved less than its drag distance between press and release (``GraphicsScene``:
``QApplication.startDragDistance()``, a few pixels by the platform's default) and sends a drag otherwise. The left
button is untouched (pan, or the ROI and selection tools that already handle it).
"""

from __future__ import annotations

from typing import Any

from PyQt5 import QtCore

RIGHT_DRAG_HELP = "Mouse: wheel zooms; drag with the left or the right button moves the map; right-click: menu."


def _pan(vb: Any, ev: Any) -> None:
    """Translate ``vb`` by the drag step of ``ev``, as ViewBox.mouseDragEvent does for the left button."""
    import numpy as np
    import pyqtgraph.functions as fn
    from pyqtgraph import Point

    ev.accept()
    dif = (ev.pos() - ev.lastPos()) * -1
    mask = np.array(vb.state["mouseEnabled"], dtype=np.float64)
    tr = fn.invertQTransform(vb.childGroup.transform())
    step = tr.map(dif * mask) - tr.map(Point(0, 0))
    x = step.x() if mask[0] == 1 else None
    y = step.y() if mask[1] == 1 else None
    vb._resetTarget()
    if x is not None or y is not None:
        vb.translateBy(x=x, y=y)
    vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"])


def install_right_drag_pan(vb: Any) -> Any:
    """Make a right-button drag pan ``vb`` (a pyqtgraph ViewBox); every other button and the wheel keep pyqtgraph's
    behaviour. Idempotent. Returns ``vb``."""
    if getattr(vb, "_right_drag_pan", False):
        return vb
    original = vb.mouseDragEvent

    def mouse_drag(ev: Any, axis: Any = None) -> None:
        if axis is None and ev.button() == QtCore.Qt.MouseButton.RightButton:
            _pan(vb, ev)
            return
        original(ev, axis=axis)

    vb.mouseDragEvent = mouse_drag
    vb._right_drag_pan = True
    return vb
