# -*- coding: utf-8 -*-
"""
The stale banner (UI stage 2, design 3.4 rules 4-5, 4.5): what a window says when it shows the analysis of a
previous selection, or rings computed for another selection or with other parameters.

Its text is built from the state (the pure functions below), so it never claims more than is true: which selection
the analysis is of and which one is current; whether the comparison without the discarded clusters was dropped when
the Axoplasm panel moved to the new selection; and what "Export axon" does meanwhile. Its two actions are
"Show the current selection" (the window draws the current selection with no analysis, as the Axoplasm panel's image
used to follow an ROI drag; "Show the analysed selection" goes back) and "Run the MPS analysis" (the main window's
"MPS analysis"). Nothing is recomputed by the banner itself.

A plain QWidget with its texts here, so stage 3 can move it into the one window and stage 4 translate it.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from tools.mps_plot_style import verdict

__all__ = ["StaleBanner", "analysis_stale_lines", "showing_current_lines", "rings_stale_lines",
           "SHOW_CURRENT", "SHOW_ANALYSED", "RUN_ANALYSIS"]

SHOW_CURRENT = "Show the current selection"
SHOW_ANALYSED = "Show the analysed selection"
RUN_ANALYSIS = "Run the MPS analysis"
SHOW_CURRENT_TIP = ("Draw the current selection with no MPS analysis: its localizations (or the Axoplasm panel's "
                    "classes and image, with the mask of the new ROI), its z with the fit and the cut. The "
                    "analysis of the previous selection is kept; nothing is recomputed.")
SHOW_ANALYSED_TIP = "Draw the analysis of the previous selection again."
RUN_TIP = "The main window's \"MPS analysis\": analyse the current selection."


def analysis_stale_lines(*, analysed_words: str, current_words: str, discard_dropped: bool) -> List[str]:
    """The banner over an analysis of a previous selection (3.4 rule 5)."""
    lines = [f"This analysis is of the previous selection ({analysed_words}); the current selection is "
             f"{current_words}."]
    if discard_dropped:
        lines.append("Its comparison without the discarded clusters was dropped when the Axoplasm panel moved to "
                     "the new selection.")
    lines.append("Export axon writes this analysis, as before; it refuses while the Axoplasm panel shows another "
                 "selection.")
    return lines


def showing_current_lines(*, current_words: str) -> List[str]:
    """The banner while the window shows the current selection with no analysis."""
    return [f"Showing the current selection ({current_words}) with no MPS analysis. The analysis of the previous "
            "selection is kept: \"Show the analysed selection\" draws it again; Export axon still writes it."]


def rings_stale_lines(*, selection_reason: str = "", params_reason: str = "") -> List[str]:
    """The rings window's banner (3.4 rule 4): the selection, the parameters, or both."""
    lines: List[str] = []
    for reason in (selection_reason, params_reason):
        if reason:
            text = reason[0].upper() + reason[1:]
            lines.append(text if text.endswith(".") else text + ".")
    return lines


class StaleBanner(QtWidgets.QFrame):
    """A line of text with up to two actions; hidden while there is nothing to say."""

    show_current_toggled = QtCore.pyqtSignal(bool)
    run_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, object_name: str = "stale_banner") -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        warn = verdict("warn", dark=False)
        self.setStyleSheet(f"#{object_name} {{ border: 1px solid {warn}; border-radius: 3px; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        self.label = QtWidgets.QLabel("")
        self.label.setObjectName(object_name + "_text")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setStyleSheet(f"color: {warn};")
        lay.addWidget(self.label, 1)
        self.btn_show = QtWidgets.QPushButton(SHOW_CURRENT)
        self.btn_show.setObjectName(object_name + "_show")
        self.btn_show.setToolTip(SHOW_CURRENT_TIP)
        self.btn_show.clicked.connect(self._on_show)
        lay.addWidget(self.btn_show)
        self.btn_run = QtWidgets.QPushButton(RUN_ANALYSIS)
        self.btn_run.setObjectName(object_name + "_run")
        self.btn_run.setToolTip(RUN_TIP)
        self.btn_run.clicked.connect(self.run_requested.emit)
        lay.addWidget(self.btn_run)
        self._showing_current = False
        self.setVisible(False)

    def set_state(self, lines: Sequence[str], *, offer_show: bool = False, showing_current: bool = False,
                  offer_run: bool = False) -> None:
        """Show ``lines`` (hidden when empty) with the actions offered."""
        self._showing_current = bool(showing_current)
        self.label.setText("\n".join(lines))
        self.btn_show.setText(SHOW_ANALYSED if showing_current else SHOW_CURRENT)
        self.btn_show.setToolTip(SHOW_ANALYSED_TIP if showing_current else SHOW_CURRENT_TIP)
        self.btn_show.setVisible(bool(offer_show))
        self.btn_run.setVisible(bool(offer_run))
        self.setVisible(bool(lines))

    def text(self) -> str:
        return str(self.label.text())

    def _on_show(self) -> None:
        self.show_current_toggled.emit(not self._showing_current)
