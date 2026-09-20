# -*- coding: utf-8 -*-
"""
Checks that the plots can be read by someone who is colour blind, and
printed.

"Colour-blind-safe palette" was already written in a comment above the
old colours, and two roles in the same plot were drawn in the same green
all the same: the cluster centres and the occupied stretches of the
perimeter. A comment cannot check itself, so this does.

What it does
------------
Every pair of roles that appears together in one plot is simulated under
normal vision and under the three dichromacies (Machado, Oliveira and
Fernandes 2009, severity 1.0), converted to CIE Lab, and the distance
between them measured. Close pairs are listed by name, so a palette
change that collapses two roles fails here rather than in a figure.

The simulation is a model, not an eye: it says which pairs are at risk,
not what any particular person sees. Where two roles are close on
purpose -- the same family of data at two lightnesses -- the check says
so and requires the drawing to separate them another way, by symbol or
by size, which is recorded here beside the pair.

A verdict has no symbol: it is a line of text. It carries a MARK
instead -- ok, !, x, - -- and the last check here requires the four to
be distinct and to be named in TOGETHER, so a mark cannot quietly stop
being drawn and leave the whole weight back on the colour.

Run:  python validate_plot_colours.py
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Dict, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_plot_style import (  # noqa: E402
    DARK_BG, LIGHT_BG, MARKS, OKABE_ITO, PANEL_BG, ROLES, neutral, role,
)

PASSED = 0
FAILED = 0

# How far apart two colours must stay, in CIE Lab (CIE76). 25 is well
# above the "just noticeable" 2.3 and about what two categories need in a
# scatter plot seen at a glance.
APART = 25.0
# A role has to stand off the background it is drawn on by at least this.
OFF_BACKGROUND = 30.0

# Machado, Oliveira and Fernandes (2009), severity 1.0, applied to linear
# RGB. The three dichromacies; normal vision is the identity.
SIMULATIONS: Dict[str, np.ndarray] = {
    "normal": np.eye(3),
    "protanopia": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900]]),
}


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


# ===========================================================================
# Colour arithmetic
# ===========================================================================

def to_rgb(colour: str) -> np.ndarray:
    """A hex colour, or pyqtgraph's 'k'/'w', as sRGB in 0..1."""
    if colour in ("k", "black"):
        return np.zeros(3)
    if colour in ("w", "white"):
        return np.ones(3)
    value = colour.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def linearise(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92,
                    ((srgb + 0.055) / 1.055) ** 2.4)


def delinearise(linear: np.ndarray) -> np.ndarray:
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.clip(linear, 0, None) ** (1 / 2.4) - 0.055)


def simulate(colour: str, kind: str) -> np.ndarray:
    """``colour`` as this kind of vision receives it, in sRGB 0..1."""
    linear = linearise(to_rgb(colour))
    seen = np.clip(delinearise(SIMULATIONS[kind] @ linear), 0.0, 1.0)
    return np.asarray(seen, dtype=float)


def to_lab(srgb: np.ndarray) -> np.ndarray:
    """CIE Lab (D65), from sRGB in 0..1."""
    linear = linearise(srgb)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = m @ linear
    white = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / white
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]),
                     200 * (f[1] - f[2])])


def distance(a: str, b: str, kind: str) -> float:
    """How far apart two colours are, for this kind of vision."""
    return float(np.linalg.norm(to_lab(simulate(a, kind))
                                - to_lab(simulate(b, kind))))


def blend(colour: str, alpha: int, background: str) -> str:
    """What a colour drawn at ``alpha`` over ``background`` comes out as."""
    front, back = to_rgb(colour), to_rgb(background)
    mixed = front * (alpha / 255.0) + back * (1 - alpha / 255.0)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in mixed)


# ===========================================================================
# What is drawn together, and how each is told apart
# ===========================================================================

# One entry per plot: the roles drawn in it, with the alpha each is drawn
# at and the symbol that carries it when the colour alone would not.
# These lists ARE the specification the drawing code follows.
TOGETHER: Dict[str, Sequence[Tuple[str, int, str]]] = {
    "the contour plot": (
        ("noise", 90, "dot"),
        ("curated", 160, "x"),
        ("discarded", 255, "diamond"),
        ("locs", 200, "dot"),
        ("centroid", 255, "circle"),
        ("occupied", 255, "thick line"),
        ("centre", 255, "plus"),
    ),
    "the axial histogram": (
        ("locs", 170, "bars"),
        ("fit", 255, "dashed line"),
        ("slab", 60, "band"),
    ),
    "a histogram with a median and the paper's value": (
        ("locs", 190, "bars"),
        ("summary", 255, "line"),
        ("paper", 255, "dashed line"),
    ),
    "the randomization CDF": (
        ("observed", 255, "line"),
        ("randomized", 255, "line"),
        ("paper", 255, "dashed line"),
    ),
    "two channels": (
        ("locs", 170, "dot"),
        ("channel_b", 170, "triangle"),
        ("centre", 255, "plus"),
    ),
    # The axoplasm panel draws on a widefield image, not on black: its
    # background is whatever grey the axon is at that point, so its
    # colours are measured against a mid grey as well.
    "the axoplasm image": (
        ("locs", 255, "dot"),
        ("discarded", 255, "dot"),
        # Both edges are drawn with a one-pixel dark casing, which is
        # what makes them legible over a photograph: their own hues sit
        # at about the lightness of a mid grey. Circles for one, squares
        # for the other.
        ("image_tubulin", 255, "cased line and circles"),
        ("image_spectrin", 255, "cased line and squares"),
        ("contour_all", 255, "dashed line"),
        ("contour_kept", 255, "line"),
    ),
    # --- the two panels that report on the acquisition -----------------
    # Findings are text, not marks, and they are read one under the other
    # on the panel's own dark grey. Each carries the mark for its kind, so
    # the two that a deuteranope cannot separate by colour are still two
    # things.
    "the findings list": (
        ("good", 255, "a leading ok"),
        ("warn", 255, "a leading !"),
        ("bad", 255, "a leading x"),
        ("dim", 255, "a leading -"),
    ),
    "the precision check": (
        ("locs", 255, "histogram outline"),
        ("fit", 255, "the fitted curve and the line at its value"),
    ),
    # sx against sy, which differ by design: that is how z is encoded.
    # The strongest pair in the palette, and 20 apart in lightness as
    # well, so the two survive a photocopy.
    "the fitting-box check": (
        ("locs", 255, "histogram outline"),
        ("paired", 255, "histogram outline"),
    ),
    # One row of bars per component, and the dashed line at each
    # component's mean carries the mark of the verdict the table gives
    # it -- which is what keeps the warn and bad rows apart.
    "the axial check": (
        ("locs", 255, "histogram outline"),
        ("good", 255, "a thick bar and a dashed line marked ok"),
        ("warn", 255, "a thick bar and a dashed line marked !"),
        ("bad", 255, "a thick bar and a dashed line marked x"),
    ),
    "the kinetics check": (
        ("locs", 255, "cumulative curve"),
        ("fit", 255, "dashed curve"),
    ),
}

# What each plot is drawn on. Anything not named here is on black.
BACKGROUNDS: Dict[str, str] = {
    "the axoplasm image": "#808080",
    # The quality and DNA-PAINT panels paint their own window, and the
    # findings are read on that rather than on a plot's black.
    "the findings list": PANEL_BG,
}

# Pairs that come close under one kind of vision and are told apart by
# something other than colour. Each says what. Anything close and NOT
# here is a failure, which is what makes this list a specification and
# not an excuse: the drawing code has to do what it says.
BY_SHAPE: Dict[Tuple[str, str], str] = {
    ("noise", "curated"):
        "a faint 2 px dot against a brighter x: both are grey on purpose, "
        "since both mean 'not in the analysis'",
    ("curated", "centroid"):
        "an x against a circle with a pale outline",
    ("curated", "discarded"):
        "an x against a diamond",
    ("discarded", "occupied"):
        "diamonds inside the ring against a thick line running along it: "
        "the closest pair in the palette for a deuteranope, and the only "
        "one where both marks matter",
    ("curated", "locs"):
        "an x against a dot",
    ("locs", "centroid"):
        "a cloud of 3 px dots against 7 px circles with a pale outline",
    ("locs", "centre"):
        "a cloud of 3 px dots against one 18 px plus",
    ("centroid", "centre"):
        "circles on the ring against one plus at its middle",
    ("summary", "paper"):
        "a solid median against a dashed reference line",
    ("occupied", "fit"):
        "a thick line on the contour against a dashed curve on a histogram",
    ("occupied", "summary"):
        "never in the same plot: one is the contour, one a histogram",
    ("paper", "discarded"):
        "a dashed reference line against a scatter of localizations",
    ("noise", "slab"):
        "a faint scatter against a band behind a histogram",
    ("noise", "randomized"):
        "the same neutral grey, never in one plot",
    ("locs", "channel_b"):
        "dots against triangles, and the panel draws them in two plots "
        "as well as one",
    # --- the axoplasm panel, on a grey image ---------------------------
    # Six roles on a mid grey leave less room than six on black, so this
    # plot leans on shape more than any other: two sizes of dot, two
    # shapes of ring, a solid line and a dashed one.
    ("locs", "image_tubulin"):
        "2 px dots against a mask edge and 10 px circles",
    ("locs", "image_spectrin"):
        "2 px dots against an image edge and 10 px squares",
    ("locs", "contour_kept"):
        "a cloud of dots against one closed line through the centres",
    ("discarded", "contour_all"):
        "filled discs with a dark rim against a dashed line",
    ("image_tubulin", "image_spectrin"):
        "circles against squares, and two edges that are different places "
        "in the image: the outside of the axon and the inside of the "
        "spectrin ring",
    ("image_tubulin", "contour_kept"):
        "the mask's own irregular edge against the polygon through the "
        "cluster centres; apart for everyone but a tritanope, who is "
        "about one person in ten thousand",
    ("image_spectrin", "contour_all"):
        "a solid image edge against a dashed contour",
    ("image_spectrin", "contour_kept"):
        "an image edge against the polygon through the cluster centres",
    ("discarded", "image_spectrin"):
        "filled discs against an edge and open squares",
    # --- the quality panels --------------------------------------------
    ("warn", "bad"):
        "a leading ! against a leading x, in the findings list and on the "
        "axial plot's component lines alike. Orange against vermillion is "
        "18 apart for a deuteranope and these two are read together, so "
        "the mark is what separates them",
    ("good", "dim"):
        "a leading ok against a leading -: what passed against what could "
        "not be checked",
    ("locs", "good"):
        "a histogram outline against a thick horizontal bar and a dashed "
        "vertical line; apart for everyone but a tritanope",
}


def pairs_of(plot: str):
    drawn = TOGETHER[plot]
    for i, (name_a, alpha_a, shape_a) in enumerate(drawn):
        for name_b, alpha_b, shape_b in drawn[i + 1:]:
            yield (name_a, alpha_a, shape_a), (name_b, alpha_b, shape_b)


# ===========================================================================
# The checks
# ===========================================================================

def test_palette() -> None:
    print("\n--- the palette itself ---")

    def every_role_is_okabe_ito_or_neutral():
        allowed = set(OKABE_ITO.values()) | {"#9a9a9a", "#7f7f7f",
                                             "#c8c8c8"}
        stray = {name: colour for name, colour in ROLES.items()
                 if colour not in allowed}
        assert not stray, stray
        return f"{len(ROLES)} roles from {len(set(ROLES.values()))} colours"

    def no_yellow_in_a_role():
        # The brightest of the eight on black and the weakest on white,
        # and these plots are read on both.
        assert OKABE_ITO["yellow"] not in ROLES.values()
        return "yellow is kept out"

    def the_eight_are_apart_from_each_other():
        worst = None
        for kind in SIMULATIONS:
            for i, a in enumerate(list(OKABE_ITO.values())):
                for b in list(OKABE_ITO.values())[i + 1:]:
                    d = distance(a, b, kind)
                    if worst is None or d < worst[0]:
                        worst = (d, a, b, kind)
        # Okabe and Ito's own set has close pairs under dichromacy; what
        # matters is that the ROLES do not use those pairs together, which
        # the checks below are about. This one records the worst pair.
        assert worst is not None
        return (f"closest of the eight: {worst[1]} vs {worst[2]} at "
                f"{worst[0]:.0f} ({worst[3]})")

    check("every role is one of the eight, or a neutral grey",
          every_role_is_okabe_ito_or_neutral)
    check("yellow is not a role", no_yellow_in_a_role)
    check("how close the eight themselves come",
          the_eight_are_apart_from_each_other)


def test_roles_in_one_plot() -> None:
    print("\n--- roles drawn together stay apart ---")

    for plot in TOGETHER:
        def one_plot(plot=plot):
            bg = BACKGROUNDS.get(plot, DARK_BG)
            close = []
            for (a, alpha_a, shape_a), (b, alpha_b, shape_b) in pairs_of(plot):
                key = (a, b) if (a, b) in BY_SHAPE else (b, a)
                for kind in SIMULATIONS:
                    d = distance(blend(role(a), alpha_a, bg),
                                 blend(role(b), alpha_b, bg), kind)
                    if d < APART and key not in BY_SHAPE:
                        close.append(f"{a} vs {b}: {d:.0f} ({kind})")
            assert not close, close
            # And each of them has to stand off what it is drawn on.
            # A band behind a histogram is meant to be faint: it says
            # where the slab is, and anything stronger would compete with
            # the bars it sits under. A cased mark carries its own
            # contrast in the dark rim drawn around it.
            faint = [f"{a}: {distance(blend(role(a), alpha_a, bg), bg, k):.0f}"
                     f" ({k})"
                     for a, alpha_a, shape in TOGETHER[plot]
                     for k in SIMULATIONS
                     if "band" not in shape and "cased" not in shape
                     and distance(blend(role(a), alpha_a, bg), bg, k) < 20]
            assert not faint, faint
            worst = min(
                distance(blend(role(a), alpha_a, bg),
                         blend(role(b), alpha_b, bg), kind)
                for (a, alpha_a, _sa), (b, alpha_b, _sb) in pairs_of(plot)
                for kind in SIMULATIONS)
            return (f"{len(list(pairs_of(plot)))} pairs on "
                    f"{'grey' if bg != DARK_BG else 'black'}, "
                    f"closest {worst:.0f}")

        check(f"{plot}", one_plot)


def test_backgrounds() -> None:
    print("\n--- every role stands off both backgrounds ---")

    def on_black():
        weak = []
        for name, colour in ROLES.items():
            for kind in SIMULATIONS:
                d = distance(colour, DARK_BG, kind)
                if d < OFF_BACKGROUND:
                    weak.append(f"{name}: {d:.0f} ({kind})")
        assert not weak, weak
        return f"{len(ROLES)} roles, all above {OFF_BACKGROUND:.0f}"

    def on_white():
        # A figure for a journal is drawn on white, so a role that only
        # works on black is a role that cannot be published.
        weak = []
        for name, colour in ROLES.items():
            for kind in SIMULATIONS:
                d = distance(colour, LIGHT_BG, kind)
                if d < OFF_BACKGROUND:
                    weak.append(f"{name}: {d:.0f} ({kind})")
        assert not weak, weak
        return f"{len(ROLES)} roles, all above {OFF_BACKGROUND:.0f}"

    def the_neutral_follows_the_background():
        assert distance(neutral(True), DARK_BG, "normal") > 60
        assert distance(neutral(False), LIGHT_BG, "normal") > 60
        return f"{neutral(True)} on black, {neutral(False)} on white"

    def faint_roles_are_still_visible():
        # What is drawn at low alpha still has to be seen: noise at 120
        # over black is the faintest thing in any plot.
        d = distance(blend(role("noise"), 120, DARK_BG), DARK_BG, "normal")
        assert d > 20, d
        return f"noise at alpha 120 over black: {d:.0f}"

    check("on black", on_black)
    check("on white", on_white)
    check("the neutral follows the background",
          the_neutral_follows_the_background)
    check("what is drawn faint is still visible", faint_roles_are_still_visible)


def test_greyscale() -> None:
    print("\n--- printed in grey ---")

    def the_marks_differ_in_lightness_or_shape():
        # A figure photocopied keeps only lightness. Where two roles in
        # one plot come out at the same lightness, something else has to
        # tell them apart, and BY_SHAPE says what.
        flat = []
        for plot in TOGETHER:
            for (a, aa, sa), (b, ab, sb) in pairs_of(plot):
                la = to_lab(to_rgb(blend(role(a), aa, DARK_BG)))[0]
                lb = to_lab(to_rgb(blend(role(b), ab, DARK_BG)))[0]
                if abs(la - lb) < 12 and sa == sb:
                    key = (a, b) if (a, b) in BY_SHAPE else (b, a)
                    if key not in BY_SHAPE:
                        flat.append(f"{plot}: {a} vs {b} "
                                    f"({la:.0f} vs {lb:.0f}, both {sa})")
        assert not flat, flat
        return "every pair differs in lightness or in symbol"

    def what_is_excluded_has_its_own_symbol():
        drawn = dict((name, shape)
                     for name, _alpha, shape in TOGETHER["the contour plot"])
        assert drawn["curated"] == "x", drawn
        assert drawn["centroid"] == "circle", drawn
        assert drawn["centre"] == "plus", drawn
        return "curated x, centroids circles, centre a plus"

    def every_finding_carries_its_own_mark():
        # A verdict is text: it has no symbol to be told apart by, so it
        # carries a mark instead. Two kinds sharing one would put the
        # whole weight back on the colour.
        assert len(set(MARKS.values())) == len(MARKS), MARKS
        named = dict((name, shape)
                     for name, _alpha, shape in TOGETHER["the findings list"])
        for kind, mark in MARKS.items():
            assert kind in named, kind
            assert mark.strip() in named[kind], (kind, mark, named[kind])
        return "  ".join(f"{k}: {v.strip()!r}" for k, v in MARKS.items())

    check("no two marks of one plot are the same grey and the same shape",
          the_marks_differ_in_lightness_or_shape)
    check("what the analysis excluded is drawn with its own symbol",
          what_is_excluded_has_its_own_symbol)
    check("every kind of finding carries its own mark",
          every_finding_carries_its_own_mark)


def main() -> int:
    print("=" * 72)
    print("PLOT COLOUR CHECKS")
    print("=" * 72)
    test_palette()
    test_roles_in_one_plot()
    test_backgrounds()
    test_greyscale()
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
