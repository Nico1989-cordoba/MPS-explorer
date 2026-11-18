# -*- coding: utf-8 -*-
"""
The dialog behind "Export image...": which plot, where, how wide, and on
what background.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import os
from typing import AbstractSet, Dict, Optional, Sequence, Union

from PyQt5 import QtWidgets

from tools.mps_plot_style import verdict
from tools.figure_export import (
    COLUMN_WIDTHS_MM, DEFAULT_DPI, DEFAULT_WIDTH_MM, DPI_CHOICES,
    KNOWN_EXTENSIONS, TITLE_LINE_JOIN, FigureRequest, free_path, pixels_for,
)

# Read on the application's white: the darker of the two dim greys.
HINT_STYLE = f"color: {verdict('dim', dark=False)}; font-size: 11px;"


# Whether a white redraw is offered: for every plot (True), for none
# (False), or for the plots named in a set (UI stage 2, design 3.6: not for
# the axon map while it draws a widefield photograph).
WhiteOffer = Union[bool, AbstractSet[str]]


def offers_white(offer: WhiteOffer, plot: str) -> bool:
    """Whether ``offer`` offers the white redraw for ``plot``."""
    if isinstance(offer, bool):
        return offer
    return plot in offer


class ExportFigureDialog(QtWidgets.QDialog):
    """What to write, and how."""

    def __init__(self, plots: Sequence[str], suggested: str,
                 parent: Optional[QtWidgets.QWidget] = None,
                 offer_white: WhiteOffer = True,
                 titles: Optional[Dict[str, str]] = None,
                 legend: WhiteOffer = False):
        super().__init__(parent)
        self.setWindowTitle("Export image")
        # F6: the proposed file never overwrites one (a _2, _3... is added); a path chosen with Browse..., where
        # the system asked to confirm an overwrite, is kept as it is.
        suggested = free_path(suggested)
        self._suggested = suggested
        self._offer_white = offer_white
        self._titles = dict(titles or {})
        self._legend = legend
        self._confirmed = ""
        self._title_typed = False

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
        # F6: an editable title (the plot's own, its lines joined by " | "), and the legend of the rows drawn
        self.edit_title = QtWidgets.QLineEdit(self._titles.get(self.combo_plot.currentText(), ""))
        self.edit_title.setObjectName("edit_figure_title")
        self.edit_title.setToolTip("The figure's title, as the plot shows it; edit it for the paper. Lines are "
                                   f"separated by '{TITLE_LINE_JOIN.strip()}'. Empty: no title. The window keeps "
                                   "its own title.")
        self.edit_title.textEdited.connect(self._on_title_typed)
        form.addRow("Title:", self.edit_title)
        layout.addLayout(form)
        self.chk_legend = QtWidgets.QCheckBox("Legend of the rows drawn")
        self.chk_legend.setObjectName("chk_figure_legend")
        self.chk_legend.setChecked(True)
        self.chk_legend.setToolTip("Put a legend in the figure: every row ticked in the plot's layer panel (the "
                                   "methods on the map, the localizations, the contour...) with its mark and line, "
                                   "as drawn. The panel itself is not in the figure.")
        self.chk_legend.setVisible(offers_white(legend, self.combo_plot.currentText()))
        layout.addWidget(self.chk_legend)

        self.chk_white = QtWidgets.QCheckBox(
            "Draw it on white, for print")
        self.chk_white.setChecked(True)
        self.chk_white.setToolTip(
            "The panels draw on black because that is what a screen is read "
            "on. Ticked, the plot is drawn again on white with the same "
            "colours before it is saved, which is what a journal expects.")
        self.chk_white.setVisible(
            offers_white(offer_white, self.combo_plot.currentText()))
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
        path = self.edit_path.text().strip()
        if path and os.path.normcase(os.path.abspath(path)) != self._confirmed:
            path = free_path(path)                     # F6: never over a file the user did not confirm
        plot = self.combo_plot.currentText()
        return FigureRequest(
            plot=plot,
            path=path,
            width_mm=float(self.combo_width.currentData()),
            dpi=int(self.combo_dpi.currentData()),
            white=(self.chk_white.isChecked()
                   and offers_white(self._offer_white, plot)),
            title=(self.edit_title.text() if (self._title_typed or plot in self._titles) else None),
            legend=bool(self.chk_legend.isChecked() and offers_white(self._legend, plot)))

    def _on_title_typed(self, _text: str) -> None:
        self._title_typed = True

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export image", self.edit_path.text(),
            "PNG image (*.png);;SVG drawing (*.svg)")
        if path:
            # the system's dialog asked before choosing an existing file: that one may be overwritten
            self._confirmed = os.path.normcase(os.path.abspath(path))
            self.edit_path.setText(path)

    def _plot_changed(self) -> None:
        self.chk_white.setVisible(
            offers_white(self._offer_white, self.combo_plot.currentText()))
        self.chk_legend.setVisible(offers_white(self._legend, self.combo_plot.currentText()))
        if not self._title_typed:
            self.edit_title.setText(self._titles.get(self.combo_plot.currentText(), ""))
        # The proposed name follows the plot chosen, unless the user has
        # typed a path of their own.
        current = self.edit_path.text().strip()
        if current == self._suggested or not current:
            base, ext = os.path.splitext(self._suggested)
            safe = "".join(c if c.isalnum() else "_"
                           for c in self.combo_plot.currentText())
            stem = base.rsplit("_", 1)[0] if "_" in base else base
            self._suggested = free_path(f"{stem}_{safe.strip('_').lower()}{ext}")
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
            extra = ""
            if request.path and os.path.splitext(request.path)[1].lower() not in KNOWN_EXTENSIONS:
                extra = " The name has no .png or .svg: '.png' is added."
            if request.path and request.path != self.edit_path.text().strip():
                extra += (f" That file exists: it is written as {os.path.basename(request.path)} (nothing is "
                          f"overwritten).")
            self.label_size.setText(
                f"{pixels} pixels wide, from {request.width_mm:.0f} mm at "
                f"{request.dpi} dpi.{extra}")
        if self._ok is not None:
            self._ok.setEnabled(bool(request.path))


def ask(plots: Sequence[str], suggested: str,
        parent: Optional[QtWidgets.QWidget] = None,
        offer_white: WhiteOffer = True,
        titles: Optional[Dict[str, str]] = None,
        legend: WhiteOffer = False) -> Optional[FigureRequest]:
    """Show the dialog; None when the user cancels. ``offer_white``: True,
    False, or the set of plot names a white redraw is offered for. F6:
    ``titles`` the plots' titles to edit, ``legend`` the plots a legend of
    the rows drawn is offered for (as ``offer_white``)."""
    dialog = ExportFigureDialog(plots, suggested, parent=parent,
                                offer_white=offer_white, titles=titles, legend=legend)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return dialog.request()
