# -*- coding: utf-8 -*-
"""
The dialog behind "Export image...": which plot, where, how wide, and on
what background.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from PyQt5 import QtWidgets

from tools.mps_plot_style import verdict
from tools.figure_export import (
    COLUMN_WIDTHS_MM, DEFAULT_DPI, DEFAULT_WIDTH_MM, DPI_CHOICES,
    FigureRequest, pixels_for,
)

# Read on the application's white: the darker of the two dim greys.
HINT_STYLE = f"color: {verdict('dim', dark=False)}; font-size: 11px;"


class ExportFigureDialog(QtWidgets.QDialog):
    """What to write, and how."""

    def __init__(self, plots: Sequence[str], suggested: str,
                 parent: Optional[QtWidgets.QWidget] = None,
                 offer_white: bool = True):
        super().__init__(parent)
        self.setWindowTitle("Export image")
        self._suggested = suggested

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Writes the plot itself, not a screenshot of the window: at the "
            "width a journal asks for, and on the background it will be "
            "printed on.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        self.combo_plot = QtWidgets.QComboBox()
        self.combo_plot.addItems(list(plots))
        self.combo_plot.currentIndexChanged.connect(self._plot_changed)
        form.addRow("Plot:", self.combo_plot)

        where = QtWidgets.QHBoxLayout()
        self.edit_path = QtWidgets.QLineEdit(suggested)
        self.edit_path.textChanged.connect(self._refresh)
        where.addWidget(self.edit_path, 1)
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        where.addWidget(browse)
        form.addRow("File:", where)

        self.combo_width = QtWidgets.QComboBox()
        for label, mm in COLUMN_WIDTHS_MM:
            self.combo_width.addItem(label, mm)
        self.combo_width.setCurrentIndex(0)
        self.combo_width.currentIndexChanged.connect(self._refresh)
        form.addRow("Width:", self.combo_width)

        self.combo_dpi = QtWidgets.QComboBox()
        for dpi in DPI_CHOICES:
            self.combo_dpi.addItem(f"{dpi} dpi", dpi)
        self.combo_dpi.setCurrentIndex(list(DPI_CHOICES).index(DEFAULT_DPI))
        self.combo_dpi.setToolTip(
            "600 for a figure of lines and markers, 300 when it carries a "
            "widefield image. Below 300 no journal takes it.")
        self.combo_dpi.currentIndexChanged.connect(self._refresh)
        form.addRow("Resolution:", self.combo_dpi)
        layout.addLayout(form)

        self.chk_white = QtWidgets.QCheckBox(
            "Draw it on white, for print")
        self.chk_white.setChecked(True)
        self.chk_white.setToolTip(
            "The panels draw on black because that is what a screen is read "
            "on. Ticked, the plot is drawn again on white with the same "
            "colours before it is saved, which is what a journal expects.")
        self.chk_white.setVisible(offer_white)
        layout.addWidget(self.chk_white)

        self.label_size = QtWidgets.QLabel("")
        self.label_size.setStyleSheet(HINT_STYLE)
        self.label_size.setWordWrap(True)
        layout.addWidget(self.label_size)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            self)
        self._ok = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        self._ok.setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    # -- state -----------------------------------------------------------

    def request(self) -> FigureRequest:
        return FigureRequest(
            plot=self.combo_plot.currentText(),
            path=self.edit_path.text().strip(),
            width_mm=float(self.combo_width.currentData()),
            dpi=int(self.combo_dpi.currentData()),
            white=self.chk_white.isChecked() and self.chk_white.isVisible())

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export image", self.edit_path.text(),
            "PNG image (*.png);;SVG drawing (*.svg)")
        if path:
            self.edit_path.setText(path)

    def _plot_changed(self) -> None:
        # The proposed name follows the plot chosen, unless the user has
        # typed a path of their own.
        current = self.edit_path.text().strip()
        if current == self._suggested or not current:
            base, ext = os.path.splitext(self._suggested)
            safe = "".join(c if c.isalnum() else "_"
                           for c in self.combo_plot.currentText())
            stem = base.rsplit("_", 1)[0] if "_" in base else base
            self._suggested = f"{stem}_{safe.strip('_').lower()}{ext}"
            self.edit_path.setText(self._suggested)
        self._refresh()

    def _refresh(self) -> None:
        request = self.request()
        if request.is_svg:
            self.label_size.setText(
                "SVG: curves, not pixels. The resolution does not apply; the "
                "width is what the journal scales it to.")
            self.combo_dpi.setEnabled(False)
        else:
            self.combo_dpi.setEnabled(True)
            pixels = pixels_for(request.width_mm, request.dpi)
            self.label_size.setText(
                f"{pixels} pixels wide, from {request.width_mm:.0f} mm at "
                f"{request.dpi} dpi.")
        if self._ok is not None:
            self._ok.setEnabled(bool(request.path))


def ask(plots: Sequence[str], suggested: str,
        parent: Optional[QtWidgets.QWidget] = None,
        offer_white: bool = True) -> Optional[FigureRequest]:
    """Show the dialog; None when the user cancels."""
    dialog = ExportFigureDialog(plots, suggested, parent=parent,
                                offer_white=offer_white)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.request()
