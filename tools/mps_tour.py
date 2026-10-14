# -*- coding: utf-8 -*-
"""
Beginner mode: a guided tour over a window, and "Explain this".

What the user sees
------------------
A card beside one control at a time. The rest of the window is a blurred, dimmed snapshot of itself; the control the
card is about stays sharp and live in a "hole", framed in the accent colour. The card has four fixed parts - what it
is, what it is for, how to read it, an example from the simulated example axon - and a "More" button that unfolds the
long explanation and, for a parameter the registry documents, the registry's own text. Its footer has Previous /
Next / Skip, "Step n of N", "See the full method" when the step names a page, and on the first card of a tour that
opened by itself a "Do not show again" box.

Keyboard: Right / Enter next, Left previous, Esc closes.

"Explain this" (Help menu, Shift+F1): the cursor changes and the next click on any control shows that control's card
alone - or a polite card saying there is none yet, with the control's tooltip - and the click never reaches the
control: no button runs, no field changes.

The pieces
----------
``TourController``  the state (on/off, which windows' tours were seen, the language; kept in
                    ``MPSSettings.beginner_tour`` through the callbacks it is given), the windows it watches, the
                    overlay on screen and the "Explain this" mode.
``TourOverlay``     the child widget of a top-level window that draws the veil and holds the card. It follows the
                    window's size (a new snapshot on every resize) and the target's place (polled), and places the
                    card where it fits, beside the target, flipping sides when it does not.
``TourCard``        the card.
``attach_help_button``  the small "?" in the corner of a panel that replays that panel's steps.

The texts are in ``tools/mps_tour_steps.py``.

When it does NOT open by itself
-------------------------------
A tour opens by itself only the first time its window is shown, only while beginner mode is on, and never on an
offscreen or minimal Qt platform - the golden harness and every GUI test run offscreen - unless the environment
variable ``MPS_TOUR_AUTOSTART`` is "1" ("0" turns it off everywhere). Started explicitly (menu, "?", a test), it opens
on any platform.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from tools import mps_plot_style as plot_style
from tools import mps_tour_steps as data

__all__ = [
    "Step", "steps_for", "find_target", "autostart_allowed", "method_url", "TourController", "TourOverlay",
    "TourCard", "attach_help_button", "backdrop", "ENV_AUTOSTART", "TOUR_PROPERTY", "OVERLAY_NAME", "CARD_NAME",
]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITHUB_BLOB = "https://github.com/Nico1989-cordoba/MPS-explorer/blob/main/"
ENV_AUTOSTART = "MPS_TOUR_AUTOSTART"
# The dynamic property that names a window's tour (``STEPS`` key).
TOUR_PROPERTY = "mps_tour_id"
OVERLAY_NAME = "mpsTourOverlay"
CARD_NAME = "mpsTourCard"
SECTIONS: Tuple[str, ...] = ("what", "why", "read", "example")
# Room between the target's frame and the card, and between the card and the window's edge.
_GAP = 14
_EDGE = 10
_HOLE_PAD = 6
_CARD_WIDTH = 440
# The smallest free side a card is put in (narrower or shorter: the card goes over the target instead).
_MIN_CARD_W = 240
_MIN_CARD_H = 120


# --------------------------------------------------------------------------- the steps
@dataclass(frozen=True)
class Step:
    """One card: the widget it is about (``target``, an objectName; "" for none) and its texts per language."""

    key: str
    target: str
    panel: str
    registry: Tuple[str, ...]
    badges: bool
    method: str
    texts: Dict[str, Dict[str, str]]

    def text(self, lang: str, field: str) -> str:
        """The text of ``field`` in ``lang``, its placeholders filled (demo facts, registry values)."""
        raw = self.texts.get(lang) or self.texts[data.DEFAULT_LANGUAGE]
        return fill(raw.get(field, ""), lang)


def fill(text: str, lang: str) -> str:
    values: Dict[str, str] = dict(data.facts(lang))
    values.update(data.registry_values())
    return text.format_map(values)


def steps_for(tour_id: str) -> List[Step]:
    """The steps of a window's tour ([] for a window with no tour yet)."""
    out: List[Step] = []
    for d in data.STEPS.get(tour_id, []):
        out.append(Step(key=str(d["key"]), target=str(d.get("target", "")), panel=str(d.get("panel", "")),
                        registry=tuple(d.get("registry", ())), badges=bool(d.get("badges", False)),
                        method=str(d.get("method", "")),
                        texts={lang: dict(d[lang]) for lang in data.LANGUAGES if lang in d}))
    return out


def find_target(window: QtWidgets.QWidget, name: str) -> Optional[QtWidgets.QWidget]:
    """The widget called ``name`` inside ``window`` (or the window itself), or None."""
    if not name:
        return None
    if window.objectName() == name:
        return window
    found = window.findChild(QtWidgets.QWidget, name)
    return found if isinstance(found, QtWidgets.QWidget) else None


def autostart_allowed() -> bool:
    """Whether a tour may open by itself here: not on an offscreen or minimal platform (the golden harness, the
    tests) unless ``MPS_TOUR_AUTOSTART=1``; never with ``MPS_TOUR_AUTOSTART=0``."""
    flag = os.environ.get(ENV_AUTOSTART, "").strip()
    if flag == "1":
        return True
    if flag == "0":
        return False
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform.startswith(("offscreen", "minimal")):
        return False
    if QtGui.QGuiApplication.instance() is not None:
        if QtGui.QGuiApplication.platformName().lower() in ("offscreen", "minimal"):
            return False
    return True


def method_url(path: str) -> QtCore.QUrl:
    """A page of the repository: the local file when it is there, otherwise the same path on GitHub."""
    local = os.path.join(REPO, *path.split("/"))
    if os.path.exists(local):
        return QtCore.QUrl.fromLocalFile(local)
    return QtCore.QUrl(GITHUB_BLOB + path)


# --------------------------------------------------------------------------- looks
def is_dark(widget: QtWidgets.QWidget) -> bool:
    return bool(widget.palette().color(QtGui.QPalette.Window).lightness() < 128)


def theme(dark: bool) -> Dict[str, Any]:
    """The card's colours, from the program's own tokens (tools.mps_plot_style and the toolbar's frame)."""
    if dark:
        return {"card": plot_style.PANEL_BG, "border": plot_style.PANEL_BORDER, "text": plot_style.TITLE_FG,
                "dim": plot_style.TEXT_DIM, "accent": plot_style.OKABE_ITO["sky_blue"],
                "button": plot_style.PANEL_TAB_BG, "on_accent": "#000000", "veil": (0, 0, 0, 150)}
    return {"card": "#ffffff", "border": "#adadad", "text": plot_style.TITLE_FG_LIGHT,
            "dim": plot_style.TEXT_DIM_LIGHT, "accent": plot_style.OKABE_ITO["blue"], "button": "#fdfdfd",
            "on_accent": "#ffffff", "veil": (26, 26, 26, 120)}


def backdrop(pixmap: QtGui.QPixmap, veil: Tuple[int, int, int, int]) -> QtGui.QPixmap:
    """A blurred, dimmed copy of ``pixmap``: scaled down and up again with smoothing (a cheap blur), then veiled."""
    image = pixmap.toImage()
    w, h = image.width(), image.height()
    if w < 4 or h < 4:
        return pixmap
    smooth = QtCore.Qt.SmoothTransformation
    ignore = QtCore.Qt.IgnoreAspectRatio
    small = image.scaled(max(1, w // 12), max(1, h // 12), ignore, smooth)
    middle = small.scaled(max(1, w // 3), max(1, h // 3), ignore, smooth)
    out = QtGui.QPixmap.fromImage(middle.scaled(w, h, ignore, smooth))
    out.setDevicePixelRatio(pixmap.devicePixelRatio())
    painter = QtGui.QPainter(out)
    painter.fillRect(QtCore.QRect(0, 0, w, h), QtGui.QColor(*veil))
    painter.end()
    return out


def _paragraphs(text: str) -> str:
    return "<br><br>".join(html.escape(p).replace("\n", "<br>") for p in text.split("\n\n") if p.strip())


def _section(heading: str, text: str) -> str:
    return f"<b>{html.escape(heading)}</b><br>{_paragraphs(text)}"


def registry_html(step: Step, lang: str) -> str:
    """The registry's own text for the step's parameters (and the badge definitions), as the card's "More" shows it.
    Read from ``tools.mps_param_registry`` when the card is drawn, so it cannot drift from it."""
    if not step.registry and not step.badges:
        return ""
    from tools import mps_param_registry as reg

    ui = data.UI[lang]
    parts: List[str] = []
    if step.registry:
        parts.append(f"<b>{html.escape(ui['registry'])}</b>")
        for key in step.registry:
            parts.append(html.escape(reg.tooltip(key)).replace("\n", "<br>"))
    if step.badges:
        parts.append(f"<b>{html.escape(ui['badges'])}</b>")
        parts.append("<br>".join(f"<i>{html.escape(b)}</i>: {html.escape(reg.BADGE_MEANING[b])}"
                                 for b in reg.BADGES))
    return "<br><br>".join(parts)


# --------------------------------------------------------------------------- the card
class TourCard(QtWidgets.QFrame):
    """Title, the four parts, "More", and the footer. Filled by the overlay."""

    def __init__(self, parent: QtWidgets.QWidget, colours: Dict[str, Any]) -> None:
        super().__init__(parent)
        self.setObjectName(CARD_NAME)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAutoFillBackground(False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 10)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel(self)
        self.title.setObjectName("tourTitle")
        self.title.setWordWrap(True)
        font = self.title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 2.0)
        self.title.setFont(font)
        header.addWidget(self.title, 1)
        self.btn_language = QtWidgets.QToolButton(self)
        self.btn_language.setObjectName("tourLanguage")
        self.btn_language.setCursor(QtCore.Qt.PointingHandCursor)
        header.addWidget(self.btn_language, 0, QtCore.Qt.AlignTop)
        layout.addLayout(header)

        self.scroll = QtWidgets.QScrollArea(self)
        self.scroll.setObjectName("tourScroll")
        self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.body = QtWidgets.QWidget()
        self.body.setObjectName("tourBody")
        body = QtWidgets.QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 4, 0)
        body.setSpacing(8)
        self.sections: Dict[str, QtWidgets.QLabel] = {}
        for name in SECTIONS:
            label = QtWidgets.QLabel(self.body)
            label.setObjectName(f"tourSection_{name}")
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.RichText)
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            body.addWidget(label)
            self.sections[name] = label
        self.note = QtWidgets.QLabel(self.body)
        self.note.setObjectName("tourNote")
        self.note.setWordWrap(True)
        body.addWidget(self.note)
        self.btn_more = QtWidgets.QPushButton(self.body)
        self.btn_more.setObjectName("tourMore")
        self.btn_more.setCheckable(True)
        self.btn_more.setCursor(QtCore.Qt.PointingHandCursor)
        body.addWidget(self.btn_more, 0, QtCore.Qt.AlignLeft)
        self.more = QtWidgets.QLabel(self.body)
        self.more.setObjectName("tourMoreText")
        self.more.setWordWrap(True)
        self.more.setTextFormat(QtCore.Qt.RichText)
        self.more.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.more.setVisible(False)
        body.addWidget(self.more)
        self.scroll.setWidget(self.body)
        self.scroll.viewport().setAutoFillBackground(False)
        self.body.setAutoFillBackground(False)
        layout.addWidget(self.scroll)

        row = QtWidgets.QHBoxLayout()
        self.btn_method = QtWidgets.QPushButton(self)
        self.btn_method.setObjectName("tourMethod")
        row.addWidget(self.btn_method)
        row.addStretch(1)
        self.counter = QtWidgets.QLabel(self)
        self.counter.setObjectName("tourCounter")
        row.addWidget(self.counter)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        self.chk_dont_show = QtWidgets.QCheckBox(self)
        self.chk_dont_show.setObjectName("tourDontShow")
        row2.addWidget(self.chk_dont_show)
        row2.addStretch(1)
        self.btn_skip = QtWidgets.QPushButton(self)
        self.btn_skip.setObjectName("tourSkip")
        self.btn_prev = QtWidgets.QPushButton(self)
        self.btn_prev.setObjectName("tourPrev")
        self.btn_next = QtWidgets.QPushButton(self)
        self.btn_next.setObjectName("tourNext")
        self.btn_next.setDefault(True)
        for button in (self.btn_skip, self.btn_prev, self.btn_next):
            button.setCursor(QtCore.Qt.PointingHandCursor)
            row2.addWidget(button)
        layout.addLayout(row2)
        self.apply_colours(colours)

    def apply_colours(self, c: Dict[str, Any]) -> None:
        self.setStyleSheet(
            f"QFrame#{CARD_NAME} {{ background: {c['card']}; border: 2px solid {c['accent']}; border-radius: 10px; }}"
            f"QFrame#{CARD_NAME} QLabel {{ color: {c['text']}; background: transparent; border: none; }}"
            f"QFrame#{CARD_NAME} QLabel#tourCounter, QFrame#{CARD_NAME} QLabel#tourNote {{ color: {c['dim']}; }}"
            f"QFrame#{CARD_NAME} QCheckBox {{ color: {c['dim']}; background: transparent; }}"
            f"QScrollArea#tourScroll, QWidget#tourBody {{ background: transparent; border: none; }}"
            f"QFrame#{CARD_NAME} QPushButton {{ background: {c['button']}; color: {c['text']};"
            f" border: 1px solid {c['border']}; border-radius: 4px; padding: 4px 12px; }}"
            f"QFrame#{CARD_NAME} QPushButton:hover {{ border-color: {c['accent']}; }}"
            f"QFrame#{CARD_NAME} QPushButton#tourNext {{ background: {c['accent']}; color: {c['on_accent']};"
            f" border-color: {c['accent']}; font-weight: bold; }}"
            f"QFrame#{CARD_NAME} QPushButton#tourMore {{ background: transparent; border: none;"
            f" color: {c['accent']}; padding: 0px; font-weight: bold; }}"
            f"QFrame#{CARD_NAME} QPushButton#tourMethod {{ color: {c['accent']}; }}"
            f"QToolButton#tourLanguage {{ background: transparent; color: {c['accent']}; border: 1px solid"
            f" {c['border']}; border-radius: 4px; padding: 1px 6px; font-weight: bold; }}"
        )

    def fit(self, width: int, max_height: int) -> None:
        """Width ``width``; as tall as its text needs, never taller than ``max_height`` (the body then scrolls)."""
        # Polished first: a label's height for a width depends on the font the style sheet gives it.
        self.ensurePolished()
        for child in self.findChildren(QtWidgets.QWidget):
            child.ensurePolished()
        self.setFixedWidth(width)
        margins = self.layout().contentsMargins()
        inner = max(80, width - margins.left() - margins.right())
        body_h = self.body.heightForWidth(inner - 4)
        if body_h <= 0:
            body_h = self.body.sizeHint().height()
        body_h += 4
        self.scroll.setFixedHeight(body_h)
        total = self.heightForWidth(width)
        if total <= 0:
            total = self.sizeHint().height()
        # The scroll bar only when the text does not fit: left "as needed", it flashed on while the card settled.
        policy = QtCore.Qt.ScrollBarAlwaysOff
        if total > max_height:
            self.scroll.setFixedHeight(max(48, body_h - (total - max_height)))
            total = max_height
            policy = QtCore.Qt.ScrollBarAsNeeded
        self.scroll.setVerticalScrollBarPolicy(policy)
        self.setFixedHeight(max(60, total))


# --------------------------------------------------------------------------- the overlay
class TourOverlay(QtWidgets.QWidget):
    """The veil and the card over one top-level window.

    ``mode``: "tour" (Previous / Next / Skip and the step counter), "single" (one step alone: Close), "none" (the
    polite card of "Explain this" for a control without a step: Close).
    """

    finished = QtCore.pyqtSignal(str)

    def __init__(self, window: QtWidgets.QWidget, steps: List[Step], controller: "TourController", *,
                 start: int = 0, mode: str = "tour", show_dont_show: bool = False, tooltip: str = "",
                 widget: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(window)
        self.setObjectName(OVERLAY_NAME)
        self._win = window
        self._steps = steps
        self._ctl = controller
        self._index = max(0, min(start, len(steps) - 1)) if steps else 0
        self._mode = mode if steps else "none"
        self._show_dont_show = show_dont_show
        self._tooltip = tooltip
        # The control a polite card is about (no step names it): framed all the same.
        self._widget = widget
        self._colours = theme(is_dark(window))
        self._background: Optional[QtGui.QPixmap] = None
        self._hole = QtCore.QRect()
        self._target: Optional[QtWidgets.QWidget] = None
        self._target_missing = False
        self._closed = False
        self._regrab_pending = False
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.card = TourCard(self, self._colours)
        self.card.btn_next.clicked.connect(self.next)
        self.card.btn_prev.clicked.connect(self.previous)
        self.card.btn_skip.clicked.connect(lambda _checked=False: self.close_tour("skipped"))
        self.card.btn_more.toggled.connect(self._on_more)
        self.card.btn_method.clicked.connect(self._open_method)
        self.card.btn_language.clicked.connect(lambda _checked=False: self._ctl.toggle_language())
        self.card.chk_dont_show.toggled.connect(self._on_dont_show)
        self._shortcuts: List[QtWidgets.QShortcut] = []
        for key, slot in ((QtCore.Qt.Key_Right, self.next), (QtCore.Qt.Key_Return, self.next),
                          (QtCore.Qt.Key_Enter, self.next), (QtCore.Qt.Key_Left, self.previous)):
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)
        escape = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        escape.setContext(QtCore.Qt.WindowShortcut)
        escape.activated.connect(lambda: self.close_tour("escaped"))
        self._shortcuts.append(escape)
        self._follow_timer = QtCore.QTimer(self)
        self._follow_timer.setInterval(250)
        self._follow_timer.timeout.connect(self._follow)
        self._ctl.language_changed.connect(self._on_language)
        window.installEventFilter(self)
        self.setGeometry(window.rect())

    # ------------------------------------------------------------ public
    def index(self) -> int:
        return self._index

    def step(self) -> Optional[Step]:
        return self._steps[self._index] if self._steps and self._mode != "none" else None

    def hole(self) -> QtCore.QRect:
        return QtCore.QRect(self._hole)

    def is_open(self) -> bool:
        return not self._closed

    def start(self) -> None:
        self.setGeometry(self._win.rect())
        self._grab()
        self.show()
        self.raise_()
        self._go(self._index)
        self.setFocus(QtCore.Qt.OtherFocusReason)
        self._follow_timer.start()
        # Once more after the first paint, when every label has its final font and size.
        QtCore.QTimer.singleShot(0, self._relayout)

    def next(self) -> None:
        if self._closed:
            return
        if self._mode != "tour" or self._index >= len(self._steps) - 1:
            self.close_tour("finished")
            return
        self._go(self._index + 1)

    def previous(self) -> None:
        if self._closed or self._mode != "tour" or self._index == 0:
            return
        self._go(self._index - 1)

    def expand(self, on: bool = True) -> None:
        """Unfold (or fold) the long text, as the "More" button does."""
        self.card.btn_more.setChecked(on)

    def close_tour(self, reason: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        self._follow_timer.stop()
        try:
            self._ctl.language_changed.disconnect(self._on_language)
        except TypeError:
            pass
        self._win.removeEventFilter(self)
        self.hide()
        self.finished.emit(reason)
        self.deleteLater()

    # ------------------------------------------------------------ steps
    def _go(self, index: int) -> None:
        self._index = index
        self._target = None
        self._target_missing = False
        step = self.step()
        if step is not None and step.target:
            target = find_target(self._win, step.target)
            if target is not None and target.isVisible():
                self._reveal(target)
                self._target = target
            else:
                self._target_missing = True
        elif step is None and self._widget is not None and self._widget is not self._win:
            try:
                if self._widget.isVisible():
                    self._target = self._widget
            except RuntimeError:
                self._widget = None
        self.card.btn_more.blockSignals(True)
        self.card.btn_more.setChecked(False)
        self.card.btn_more.blockSignals(False)
        self.card.more.setVisible(False)
        self._render()
        self._grab()
        self._relayout()

    def _reveal(self, target: QtWidgets.QWidget) -> None:
        """Scroll the target into view when it sits in a scroll area."""
        parent = target.parentWidget()
        while parent is not None and parent is not self._win:
            if isinstance(parent, QtWidgets.QScrollArea) and parent.widget() is not None \
                    and parent.widget().isAncestorOf(target):
                parent.ensureWidgetVisible(target, 40, 40)
            parent = parent.parentWidget()

    def _render(self) -> None:
        lang = self._ctl.language()
        ui = data.UI[lang]
        card = self.card
        step = self.step()
        if step is None:
            card.title.setText(ui["no_step_title"])
            card.sections["what"].setText(_paragraphs(ui["no_step_text"]))
            tip = self._tooltip.strip()
            card.sections["why"].setText(_section(ui["its_tooltip"], tip) if tip else "")
            for name in ("read", "example"):
                card.sections[name].setText("")
            card.more.setText("")
        else:
            card.title.setText(step.text(lang, "title"))
            for name in SECTIONS:
                card.sections[name].setText(_section(ui[name], step.text(lang, name)))
            more = _paragraphs(step.text(lang, "more"))
            extra = registry_html(step, lang)
            card.more.setText(more + ("<br><br>" + extra if extra else ""))
        for name, label in card.sections.items():
            label.setVisible(bool(label.text()))
        card.note.setText(ui["not_visible"] if self._target_missing else "")
        card.note.setVisible(self._target_missing)
        has_more = bool(card.more.text())
        card.btn_more.setVisible(has_more)
        card.btn_more.setText(("▴ " + ui["less"]) if card.btn_more.isChecked() else ("▾ " + ui["more"]))
        card.btn_method.setText(ui["method"])
        card.btn_method.setVisible(step is not None and bool(step.method))
        tour = self._mode == "tour"
        card.counter.setText(ui["step"].format(n=self._index + 1, total=len(self._steps)) if tour else "")
        card.counter.setVisible(tour)
        card.btn_prev.setText(ui["prev"])
        card.btn_prev.setVisible(tour)
        card.btn_prev.setEnabled(tour and self._index > 0)
        card.btn_skip.setText(ui["skip"])
        card.btn_skip.setVisible(tour and self._index < len(self._steps) - 1)
        last = self._index >= len(self._steps) - 1
        card.btn_next.setText(ui["next"] if tour and not last else (ui["finish"] if tour else ui["close"]))
        show_box = self._show_dont_show and tour and self._index == 0
        card.chk_dont_show.setText(ui["dont_show"])
        card.chk_dont_show.blockSignals(True)
        card.chk_dont_show.setChecked(not self._ctl.enabled())
        card.chk_dont_show.blockSignals(False)
        card.chk_dont_show.setVisible(show_box)
        other = "en" if lang == "es" else "es"
        card.btn_language.setText(other.upper())
        card.btn_language.setToolTip(f"{ui['language']}: {data.LANGUAGE_NAMES[other]}")

    def _on_language(self, _lang: str) -> None:
        if not self._closed:
            self._render()
            self._relayout()

    def _on_more(self, on: bool) -> None:
        self.card.more.setVisible(on)
        self._render()
        self._relayout()

    def _on_dont_show(self, checked: bool) -> None:
        self._ctl.set_enabled(not checked)

    def _open_method(self) -> None:
        step = self.step()
        if step is not None and step.method:
            QtGui.QDesktopServices.openUrl(method_url(step.method))

    # ------------------------------------------------------------ geometry
    def _target_rect(self) -> QtCore.QRect:
        w = self._target
        if w is None:
            return QtCore.QRect()
        try:
            if not w.isVisible():
                return QtCore.QRect()
            rect = QtCore.QRect(w.mapTo(self._win, QtCore.QPoint(0, 0)), w.size())
            parent = w.parentWidget()
            while parent is not None and parent is not self._win:
                rect = rect.intersected(QtCore.QRect(parent.mapTo(self._win, QtCore.QPoint(0, 0)), parent.size()))
                parent = parent.parentWidget()
        except RuntimeError:          # the widget was deleted under us
            self._target = None
            return QtCore.QRect()
        if rect.width() < 4 or rect.height() < 4:
            return QtCore.QRect()
        pad = _HOLE_PAD
        return rect.adjusted(-pad, -pad, pad, pad).intersected(self.rect().adjusted(3, 3, -3, -3))

    def _relayout(self) -> None:
        if self._closed:
            return
        if self.geometry() != self._win.rect():
            self.setGeometry(self._win.rect())
        self._hole = self._target_rect()
        self._place_card()
        region = QtGui.QRegion(self.rect())
        if not self._hole.isNull():
            region = region.subtracted(QtGui.QRegion(self._hole))
        region = region.united(QtGui.QRegion(self.card.geometry()))
        self.setMask(region)
        self.card.show()
        self.card.raise_()
        self.update()

    def _place_card(self) -> None:
        """Beside the target, in the first free side (right, left, below, above) where the card fits at its full
        height; when none does, in the largest free side, its text scrolling. Always inside the window; over the
        target only when no side has room for a card at all (a very small window)."""
        width, height = self.width(), self.height()
        hole = self._hole
        card = self.card
        if not hole.isNull():
            P = QtCore.QPoint
            sides = (
                ("right", QtCore.QRect(P(hole.right() + _GAP, _EDGE), P(width - _EDGE, height - _EDGE))),
                ("left", QtCore.QRect(P(_EDGE, _EDGE), P(hole.left() - _GAP, height - _EDGE))),
                ("below", QtCore.QRect(P(_EDGE, hole.bottom() + _GAP), P(width - _EDGE, height - _EDGE))),
                ("above", QtCore.QRect(P(_EDGE, _EDGE), P(width - _EDGE, hole.top() - _GAP))),
            )
            best: Optional[Tuple[int, str, QtCore.QRect, int]] = None
            for side, free in sides:
                if free.width() < _MIN_CARD_W or free.height() < _MIN_CARD_H:
                    continue
                cw = min(_CARD_WIDTH, free.width())
                card.fit(cw, 10 ** 6)
                if card.height() <= free.height():
                    self._put(side, free, cw)
                    return
                area = cw * free.height()
                if best is None or area > best[0]:
                    best = (area, side, free, cw)
            if best is not None:
                self._put(best[1], best[2], best[3])
                return
        cw = max(160, min(_CARD_WIDTH, width - 2 * _EDGE))
        card.fit(cw, max(120, height - 2 * _EDGE))
        ch = card.height()
        card.move(max(_EDGE, (width - cw) // 2), max(_EDGE, (height - ch) // 2))

    def _put(self, side: str, free: QtCore.QRect, cw: int) -> None:
        """The card in the free rectangle ``free`` on ``side`` of the target, as close to the target as it goes."""
        card = self.card
        card.fit(cw, free.height())
        ch = card.height()
        centre = self._hole.center()

        def along(lo: int, hi: int, size: int, middle: int) -> int:
            return max(lo, min(middle - size // 2, hi - size + 1))

        if side == "right":
            x, y = free.left(), along(free.top(), free.bottom(), ch, centre.y())
        elif side == "left":
            x, y = free.right() - cw + 1, along(free.top(), free.bottom(), ch, centre.y())
        elif side == "below":
            x, y = along(free.left(), free.right(), cw, centre.x()), free.top()
        else:
            x, y = along(free.left(), free.right(), cw, centre.x()), free.bottom() - ch + 1
        card.move(x, y)

    def _grab(self) -> None:
        """A new snapshot of the window, taken with the overlay out of it."""
        focus = QtWidgets.QApplication.focusWidget()
        had_focus = focus is not None and (focus is self or self.isAncestorOf(focus))
        visible = self.isVisible()
        if visible:
            self.setVisible(False)
        try:
            pixmap = self._win.grab()
        finally:
            if visible:
                self.setVisible(True)
                self.raise_()
        self._background = backdrop(pixmap, self._colours["veil"])
        if had_focus and focus is not None:
            focus.setFocus(QtCore.Qt.OtherFocusReason)

    def _follow(self) -> None:
        if self._closed:
            return
        if self._target_rect() != self._hole:
            self._grab()
            self._relayout()

    def _regrab_later(self) -> None:
        if not self._regrab_pending:
            self._regrab_pending = True
            QtCore.QTimer.singleShot(60, self._regrab_now)

    def _regrab_now(self) -> None:
        self._regrab_pending = False
        if self._closed:
            return
        self._grab()
        self._relayout()

    # ------------------------------------------------------------ Qt
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._win and not self._closed and event.type() == QtCore.QEvent.Resize:
            self.setGeometry(self._win.rect())
            self._regrab_later()
        return False

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        if self._background is not None:
            painter.drawPixmap(self.rect(), self._background)
        else:
            painter.fillRect(self.rect(), QtGui.QColor(*self._colours["veil"]))
        if not self._hole.isNull():
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            pen = QtGui.QPen(QtGui.QColor(self._colours["accent"]))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(QtCore.QRectF(self._hole).adjusted(-1.5, -1.5, 1.5, 1.5), 8.0, 8.0)
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        # A click on the veil does nothing (it never reaches the window under it).
        event.accept()


# --------------------------------------------------------------------------- "Explain this"
class _ExplainFilter(QtCore.QObject):
    """Application-wide while "Explain this" is on: the next mouse press on a widget is swallowed (and its release),
    and the controller is told which widget it was."""

    def __init__(self, controller: "TourController") -> None:
        super().__init__(controller)
        self._ctl = controller
        self.active = True
        self._swallow_release = False

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        kind = event.type()
        if self.active and kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonDblClick):
            if not isinstance(obj, QtWidgets.QWidget):
                return False
            self.active = False
            self._swallow_release = True
            widget = obj
            self._ctl._explain_ended()
            QtCore.QTimer.singleShot(0, lambda: self._ctl._explain_clicked(widget))
            return True
        if self._swallow_release and kind == QtCore.QEvent.MouseButtonRelease:
            if not isinstance(obj, QtWidgets.QWidget):
                return False
            self._swallow_release = False
            self._ctl._filter_done(self)
            return True
        if self.active and kind == QtCore.QEvent.KeyPress and isinstance(event, QtGui.QKeyEvent) \
                and event.key() == QtCore.Qt.Key_Escape:
            self.active = False
            self._ctl._explain_ended()
            self._ctl._filter_done(self)
            return True
        return False


# --------------------------------------------------------------------------- the controller
class TourController(QtCore.QObject):
    """Beginner mode for one application: its state, its windows, the overlay on screen and "Explain this".

    ``store`` returns the state dict (``MPSSettings.beginner_tour``, changed in place) and ``save`` writes it.
    """

    language_changed = QtCore.pyqtSignal(str)
    enabled_changed = QtCore.pyqtSignal(bool)
    explain_changed = QtCore.pyqtSignal(bool)

    def __init__(self, store: Callable[[], Dict[str, Any]], save: Callable[[], None],
                 parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._store = store
        self._save = save
        self._overlay: Optional[TourOverlay] = None
        self._filter: Optional[_ExplainFilter] = None
        self._explaining = False
        self._autostarted: Set[str] = set()

    # ------------------------------------------------------------ state
    def language(self) -> str:
        value = self._store().get("language")
        return value if value in data.LANGUAGES else data.DEFAULT_LANGUAGE

    def set_language(self, lang: str) -> None:
        if lang not in data.LANGUAGES or lang == self.language():
            return
        self._store()["language"] = lang
        self._save()
        self.language_changed.emit(lang)

    def toggle_language(self) -> None:
        self.set_language("en" if self.language() == "es" else "es")

    def ui(self, key: str) -> str:
        return data.UI[self.language()][key]

    def enabled(self) -> bool:
        value = self._store().get("enabled")
        return True if value is None else bool(value)

    def set_enabled(self, on: bool, *, forget_seen: bool = False) -> None:
        store = self._store()
        changed = self.enabled() != on
        store["enabled"] = bool(on)
        if forget_seen:
            store["seen"] = {}
            self._autostarted.clear()
        self._save()
        if changed:
            self.enabled_changed.emit(bool(on))

    def seen(self, tour_id: str) -> bool:
        seen = self._store().get("seen")
        return isinstance(seen, dict) and seen.get(tour_id) is True

    def mark_seen(self, tour_id: str) -> None:
        store = self._store()
        seen = store.get("seen")
        if not isinstance(seen, dict):
            seen = {}
            store["seen"] = seen
        if seen.get(tour_id) is not True:
            seen[tour_id] = True
            self._save()

    # ------------------------------------------------------------ windows
    def register_window(self, window: QtWidgets.QWidget, tour_id: str) -> None:
        """Name ``window``'s tour; its tour opens by itself the first time it is shown (beginner mode on)."""
        window.setProperty(TOUR_PROPERTY, tour_id)
        window.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Show and isinstance(obj, QtWidgets.QWidget) and obj.isWindow():
            tour_id = obj.property(TOUR_PROPERTY)
            if tour_id and self.enabled() and not self.seen(str(tour_id)) and autostart_allowed():
                window = obj
                QtCore.QTimer.singleShot(400, lambda: self._autostart(window, str(tour_id)))
        return False

    def _autostart(self, window: QtWidgets.QWidget, tour_id: str) -> None:
        try:
            if window.isVisible():
                self.maybe_autostart(window, tour_id)
        except RuntimeError:          # closed and deleted in the meantime
            pass

    def maybe_autostart(self, window: QtWidgets.QWidget, tour_id: str, *, force: bool = False) -> bool:
        """Open ``tour_id`` on ``window`` as if the window had just opened for the first time. ``force`` skips the
        platform check (a test). Returns whether it opened."""
        if not self.enabled() or self.seen(tour_id) or tour_id in self._autostarted:
            return False
        if not force and not autostart_allowed():
            return False
        if not steps_for(tour_id):
            return False
        self._autostarted.add(tour_id)
        return self.start(window, tour_id, auto=True) is not None

    def start(self, window: QtWidgets.QWidget, tour_id: str, *, panel: Optional[str] = None,
              auto: bool = False) -> Optional[TourOverlay]:
        """Open ``tour_id`` on ``window`` (only the steps of ``panel`` when given). A whole tour that closes, however
        it closes, marks the window's tour as seen."""
        steps = steps_for(tour_id)
        if panel is not None:
            steps = [s for s in steps if s.panel == panel]
        if not steps:
            return None
        self.set_explain(False)
        self.close()
        overlay = TourOverlay(window, steps, self, mode="tour", show_dont_show=auto)
        whole = panel is None
        overlay.finished.connect(lambda _reason, t=tour_id, w=whole: self._finished(t, w))
        self._overlay = overlay
        overlay.start()
        return overlay

    def show_step(self, window: QtWidgets.QWidget, step: Optional[Step], tooltip: str = "",
                  widget: Optional[QtWidgets.QWidget] = None) -> TourOverlay:
        """One card alone: ``step``, or the polite "no explanation yet" card with the control's tooltip."""
        self.close()
        overlay = TourOverlay(window, [step] if step is not None else [], self,
                              mode="single" if step is not None else "none", tooltip=tooltip, widget=widget)
        overlay.finished.connect(lambda _reason: self._finished("", False))
        self._overlay = overlay
        overlay.start()
        return overlay

    def overlay(self) -> Optional[TourOverlay]:
        overlay = self._overlay
        try:
            return overlay if overlay is not None and overlay.is_open() else None
        except RuntimeError:
            return None

    def close(self) -> None:
        overlay, self._overlay = self._overlay, None
        if overlay is not None:
            try:
                overlay.close_tour("replaced")
            except RuntimeError:
                pass

    def _finished(self, tour_id: str, whole: bool) -> None:
        if whole and tour_id:
            self.mark_seen(tour_id)
        overlay = self._overlay
        try:
            if overlay is not None and not overlay.is_open():
                self._overlay = None
        except RuntimeError:
            self._overlay = None

    # ------------------------------------------------------------ "Explain this"
    def explaining(self) -> bool:
        return self._explaining

    def set_explain(self, on: bool) -> None:
        if on == self._explaining:
            return
        app = QtWidgets.QApplication.instance()
        if on:
            if app is None:
                return
            self.close()
            if self._filter is not None:
                app.removeEventFilter(self._filter)
            self._filter = _ExplainFilter(self)
            app.installEventFilter(self._filter)
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WhatsThisCursor))
            self._explaining = True
            self.explain_changed.emit(True)
        else:
            if self._filter is not None:
                self._filter.active = False
                self._filter_done(self._filter)
            self._explain_ended()

    def _explain_ended(self) -> None:
        if self._explaining:
            self._explaining = False
            QtWidgets.QApplication.restoreOverrideCursor()
            self.explain_changed.emit(False)

    def _filter_done(self, flt: _ExplainFilter) -> None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.removeEventFilter(flt)
        if self._filter is flt:
            self._filter = None

    def _explain_clicked(self, widget: QtWidgets.QWidget) -> None:
        try:
            self.explain_widget(widget)
        except RuntimeError:          # the widget went away before the card could open
            pass

    def explain_widget(self, widget: QtWidgets.QWidget) -> TourOverlay:
        """The card of the step whose target is ``widget`` or one of its parents, in ``widget``'s window; the polite
        card (with the nearest tooltip) when there is none."""
        window = widget.window()
        tour_id = window.property(TOUR_PROPERTY)
        by_target = {s.target: s for s in steps_for(str(tour_id))} if tour_id else {}
        tooltip = ""
        current: Optional[QtWidgets.QWidget] = widget
        while current is not None:
            name = current.objectName()
            if name and name in by_target:
                return self.show_step(window, by_target[name])
            if not tooltip and current.toolTip().strip():
                tooltip = current.toolTip()
            if current is window:
                break
            current = current.parentWidget()
        return self.show_step(window, None, tooltip=tooltip, widget=widget)


# --------------------------------------------------------------------------- the "?" of a panel
class _Corner(QtCore.QObject):
    """Keeps a button in the top-right corner of its panel."""

    def __init__(self, panel: QtWidgets.QWidget, button: QtWidgets.QAbstractButton) -> None:
        super().__init__(button)
        self._panel = panel
        self._button = button
        panel.installEventFilter(self)
        self.place()

    def place(self) -> None:
        self._button.move(max(0, self._panel.width() - self._button.width() - 6), 0)
        self._button.raise_()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._panel and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.place()
        return False


def attach_help_button(panel: QtWidgets.QWidget, name: str, tooltip: str,
                       callback: Callable[[], Any]) -> QtWidgets.QToolButton:
    """A small "?" in the top-right corner of ``panel`` that calls ``callback``."""
    button = QtWidgets.QToolButton(panel)
    button.setObjectName(name)
    button.setText("?")
    button.setFixedSize(18, 18)
    button.setToolTip(tooltip)
    button.setCursor(QtCore.Qt.PointingHandCursor)
    button.setStyleSheet(
        "QToolButton { border: 1px solid #adadad; border-radius: 9px; background: #fdfdfd;"
        f" color: {plot_style.OKABE_ITO['blue']}; font-weight: bold; padding: 0px; }}"
        "QToolButton:hover { border-color: #0078d7; background: #e5f1fb; }")
    button.clicked.connect(lambda _checked=False: callback())
    _Corner(panel, button)
    button.show()
    return button
