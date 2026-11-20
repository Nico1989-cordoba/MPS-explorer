# -*- coding: utf-8 -*-
"""
The presentation mode of the perimeter density window with the answers of the axon that is OPEN in the window.

``tools.mps_tour_steps.PERIMETER_DEMO_STEPS`` holds, for each card, the question it answers ("what"), the plain
explanation ("why"), how to read the window ("read") and an example with the simulated example axon's numbers
("example"). This module rewrites only the "example" of every card from a run's ``PerimeterResult``: the text starts
"En este axón:" / "On this axon:" and says what THIS run measured, with the attribute names of
``tools.mps_perimeter_types`` (z0, slices, contour, profiles, a, b, c, diagnostics, extras). When there is no result
(Run not pressed, or still running), or a part was not reached (contour rejected, analysis stopped earlier), the
card says so in plain words ("no se calculó porque ...") and keeps the general explanation.

Nothing here computes a new statistic: every number is read from the result. The one derived quantity is the
expected axial leak of each pair of C, recomputed exactly as the analysis does for its reading
(``mps_perimeter_bc.leak_expected_r_pairs`` with the axon's own slice counts and the axial width the run used).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from tools import mps_perimeter_bc as bc
from tools import mps_perimeter_types as pdt
from tools import mps_tour_steps as data
from tools.mps_perimeter_window import (ALPHA, Result, contour_kind_text, contour_unreliable_reasons,
                                        perimeter_demo_facts)

LIVE_HEADING: Dict[str, str] = {"es": "En este axón", "en": "On this axon"}
# Why there is no result at all (the window passes one of these keys).
NO_RESULT_NOT_RUN = "not_run"
NO_RESULT_RUNNING = "running"

_STATUS_WORDS: Dict[str, Dict[str, str]] = {
    "es": {
        pdt.Status.OK.value: "terminó bien",
        pdt.Status.NOT_RUN.value: "no se corrió",
        pdt.Status.NO_FOCUS.value: "no hay foco disponible para el criterio de z0 elegido",
        pdt.Status.CONTOUR_FAILED.value: "el contorno fue rechazado por su control de calidad",
        pdt.Status.TOO_FEW_LOCS.value: "hay demasiado pocas localizaciones",
        pdt.Status.EMPTY_PROFILE.value: "el perfil quedó vacío",
        pdt.Status.INVALID_INPUT.value: "los datos de entrada no se pueden analizar",
        pdt.Status.ERROR.value: "hubo un error interno",
    },
    "en": {
        pdt.Status.OK.value: "finished",
        pdt.Status.NOT_RUN.value: "it was not run",
        pdt.Status.NO_FOCUS.value: "no focus is available for the chosen z0 criterion",
        pdt.Status.CONTOUR_FAILED.value: "the contour was rejected by its quality check",
        pdt.Status.TOO_FEW_LOCS.value: "there are too few localizations",
        pdt.Status.EMPTY_PROFILE.value: "the profile came out empty",
        pdt.Status.INVALID_INPUT.value: "the input cannot be analysed",
        pdt.Status.ERROR.value: "an internal error happened",
    },
}

# The five labels of ``mps_perimeter_bc.slice_reading_pairs`` (the reading of C the window shows).
_C_LABELS: Dict[str, Dict[str, str]] = {
    "es": {
        "registry between slices": "registro entre rodajas (z−1 y z+1 se parecen más que por azar)",
        "leakage only": "sólo fuga axial (los pares vecinos no superan lo que explica la fuga)",
        "neighbours above leakage": "vecinas por encima de la fuga (sugestivo, no concluyente)",
        "no correlation between slices": "sin correlación entre rodajas",
        "not computable": "no calculable",
    },
    "en": {
        "registry between slices": "registry between slices (z-1 and z+1 look alike beyond chance)",
        "leakage only": "leakage only (the neighbouring pairs do not exceed what the leak explains)",
        "neighbours above leakage": "neighbours above leakage (suggestive, not conclusive)",
        "no correlation between slices": "no correlation between slices",
        "not computable": "not computable",
    },
}

# ``mps_perimeter_bc.vs_leak_kind`` -> words (r(z-1, z+1) against the leak expected for that pair).
_VS_LEAK: Dict[str, Dict[str, str]] = {
    "es": {
        "above_upper": "supera la fuga esperada (σz {main} nm, rango {lo}–{hi} nm) y también su cota superior "
                       "({upper} nm)",
        "above": "supera la fuga esperada (σz {main} nm, rango {lo}–{hi} nm); queda dentro de la cota superior "
                 "({upper} nm)",
        "range": "supera la fuga a σz {main} nm, pero no a {hi} nm (dentro del rango medido)",
        "limit": "en el límite: supera la fuga a σz {lo} nm, pero no a {main} nm",
        "not_above": "no supera la fuga esperada (ni siquiera a σz {lo} nm)",
    },
    "en": {
        "above_upper": "above the expected leak (sigma_z {main} nm, range {lo}-{hi} nm) and above its upper bound "
                       "({upper} nm)",
        "above": "above the expected leak (sigma_z {main} nm, range {lo}-{hi} nm); within the upper bound "
                 "({upper} nm)",
        "range": "above the leak at sigma_z {main} nm, not at {hi} nm (within the measured range)",
        "limit": "at the limit: above the leak at sigma_z {lo} nm, not at {main} nm",
        "not_above": "not above the expected leak (not even at sigma_z {lo} nm)",
    },
}

_PAIR_NAME: Dict[str, Dict[Tuple[int, int], str]] = {
    "es": {(-1, 0): "z−1 / z0", (0, 1): "z0 / z+1", (-1, 1): "z−1 / z+1"},
    "en": {(-1, 0): "z-1 / z0", (0, 1): "z0 / z+1", (-1, 1): "z-1 / z+1"},
}


# ----------------------------------------------------------------------------------------------------
# Numbers the way each language writes them
# ----------------------------------------------------------------------------------------------------
def _finite(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def num(v: Any, fmt: str, lang: str, unit: str = "") -> str:
    """``fmt`` applied to ``v`` (es: 12.137 and 0,25; en: 12,137 and 0.25); "s/d" / "n/a" when not finite."""
    if not _finite(v):
        return "s/d" if lang == "es" else "n/a"
    text = fmt.format(float(v) if not isinstance(v, int) or "f" in fmt else v)
    if lang == "es":
        text = text.translate(str.maketrans(".,", ",."))
    return text + unit


def _int(v: Any, lang: str) -> str:
    return num(int(v), "{:,d}", lang) if _finite(v) else num(None, "", lang)


def _stat(t: Optional[pdt.TestReadout], lang: str, value_fmt: str = "{:+.2f}") -> str:
    if t is None:
        return num(None, "", lang)
    sep = "; " if lang == "es" else ", "
    return (f"{num(t.value, value_fmt, lang)} (z = {num(t.z, '{:.2f}', lang)}{sep}"
            f"p = {num(t.p, '{:.3f}', lang)})")


def _escape(text: str) -> str:
    """The tour fills placeholders with ``str.format_map``: braces of live text must be literal."""
    return text.replace("{", "{{").replace("}", "}}")


# ----------------------------------------------------------------------------------------------------
# Why a part was not computed
# ----------------------------------------------------------------------------------------------------
def no_result_reason(reason: str, lang: str) -> str:
    if reason == NO_RESULT_RUNNING:
        return ("el cálculo todavía está corriendo; cuando termine, volvé a abrir el Modo presentación"
                if lang == "es" else "the run is still going; when it finishes, open the presentation mode again")
    return ("todavía no se apretó Run en esta ventana; apretá Run y volvé a abrir el Modo presentación"
            if lang == "es" else "Run has not been pressed in this window yet; press Run and open the presentation "
                                 "mode again")


def stopped_reason(result: Result, lang: str) -> str:
    """Why the analysis did not reach its end, in words ("" when it did)."""
    if result.status == pdt.Status.OK:
        return ""
    words = str(_STATUS_WORDS[lang].get(result.status.value, result.status.value))
    if result.status == pdt.Status.CONTOUR_FAILED:
        return words + (" (ver el paso del contorno)" if lang == "es" else " (see the contour step)")
    if result.failure:
        detail = "texto del programa" if lang == "es" else "the program's words"
        return f"{words} ({detail}: «{result.failure}»)" if lang == "es" else f"{words} ({detail}: \"{result.failure}\")"
    return words


def _not_computed(what: Dict[str, str], why: str, lang: str) -> str:
    return (f"{what['es']} no se calculó porque {why}." if lang == "es"
            else f"{what['en']} was not computed because {why}.")


def _unreliable_tail(result: Result, lang: str) -> str:
    """The warning every number carries when the contour was accepted but its quality gate failed."""
    if contour_unreliable_reasons(result) is None:
        return ""
    return (" Ojo: el contorno quedó marcado como NO CONFIABLE, así que este número es orientativo y el axón queda "
            "fuera de los análisis de grupo." if lang == "es" else
            " Careful: the contour was marked NOT RELIABLE, so this number is indicative only and the axon is left "
            "out of the pooled analyses.")


# ----------------------------------------------------------------------------------------------------
# The contour, in words
# ----------------------------------------------------------------------------------------------------
def _rejection_words(result: Result, lang: str) -> str:
    """The reasons of a rejected contour: the two the gate writes most often in plain words, the others quoted."""
    f = perimeter_demo_facts(result)
    parts: List[str] = []
    if "pd_len_nm" in f:
        parts.append(
            f"la línea medía {_int(f['pd_len_nm'], lang)} nm, {num(f['pd_len_ratio'], '{:.1f}', lang)} veces el "
            f"perímetro de la nube (límite {num(f['pd_len_limit'], '{:.1f}', lang)}): serpenteaba por adentro del "
            "anillo" if lang == "es" else
            f"the line was {_int(f['pd_len_nm'], lang)} nm long, {num(f['pd_len_ratio'], '{:.1f}', lang)} times the "
            f"perimeter of the cloud (limit {num(f['pd_len_limit'], '{:.1f}', lang)}): it snaked inside the ring")
    if "pd_polar_off_nm" in f:
        parts.append(
            f"el plan B (contorno polar) también falló: en un tramo quedaba a {_int(f['pd_polar_off_nm'], lang)} nm "
            f"de las localizaciones (límite {_int(f['pd_polar_limit_nm'], lang)} nm)" if lang == "es" else
            f"plan B (the polar contour) failed too: along a stretch it was {_int(f['pd_polar_off_nm'], lang)} nm "
            f"from the localizations (limit {_int(f['pd_polar_limit_nm'], lang)} nm)")
    if not parts:
        ct = result.contour
        why = "; ".join(ct.warnings) if ct is not None and ct.warnings else ""
        if why:
            parts.append(f"motivo que da el programa: «{why}»" if lang == "es" else f"the program's reason: \"{why}\"")
        else:
            parts.append("el programa no dio el motivo" if lang == "es" else "the program gave no reason")
    return "; ".join(parts)


def contour_state_words(result: Optional[Result], lang: str, reason: str = NO_RESULT_NOT_RUN) -> Tuple[str, str]:
    """(kind, sentence): kind is "none" | "stopped" | "rejected" | "unreliable" | "drawn" | "reliable"."""
    es = lang == "es"
    if result is None:
        return "none", (f"el contorno no se calculó porque {no_result_reason(reason, lang)}" if es else
                        f"the contour was not computed because {no_result_reason(reason, lang)}")
    ct = result.contour
    if ct is None:
        return "stopped", (f"no hay contorno: no se calculó porque el análisis se detuvo antes "
                           f"({stopped_reason(result, lang)})" if es else
                           f"there is no contour: it was not computed because the analysis stopped earlier "
                           f"({stopped_reason(result, lang)})")
    if ct.status != pdt.Status.OK or ct.path is None:
        return "rejected", ("el contorno fue RECHAZADO por su control de calidad: " if es else
                            "the contour was REJECTED by its quality check: ") + _rejection_words(result, lang)
    dg = result.diagnostics
    unsupported = float("nan") if dg is None else 100.0 * float(dg.unsupported_fraction)
    numbers = (f"mide {num(ct.length_nm, '{:,.0f}', lang)} nm, con un residuo de "
               f"{num(ct.residual_sd_nm, '{:.0f}', lang)} nm, {ct.n_self_crossings} cruces y el "
               f"{num(unsupported, '{:.0f}', lang)} % del anillo sin datos" if es else
               f"it is {num(ct.length_nm, '{:,.0f}', lang)} nm long, with a residual of "
               f"{num(ct.residual_sd_nm, '{:.0f}', lang)} nm, {ct.n_self_crossings} crossings and "
               f"{num(unsupported, '{:.0f}', lang)} % of the ring without data")
    reasons = contour_unreliable_reasons(result)
    if reasons is not None:
        return "unreliable", (f"el contorno se aceptó pero quedó marcado como NO CONFIABLE ({contour_kind_text(ct)}); "
                              f"motivo que da el programa: «{reasons}». Se calculó igual ({numbers}), pero este axón "
                              "queda fuera de los análisis de grupo" if es else
                              f"the contour was accepted but marked NOT RELIABLE ({contour_kind_text(ct)}); the "
                              f"program's reason: \"{reasons}\". It was used anyway ({numbers}), but this axon is "
                              "left out of the pooled analyses")
    if ct.method == "drawn":
        return "drawn", (f"se usó el trazado dibujado a mano (opción de comparación, no el contorno automático): "
                         f"{numbers}" if es else
                         f"the hand-drawn trace was used (a comparison option, not the automatic contour): {numbers}")
    return "reliable", (f"el contorno es CONFIABLE ({contour_kind_text(ct)}): {numbers}" if es else
                        f"the contour is RELIABLE ({contour_kind_text(ct)}): {numbers}")


# ----------------------------------------------------------------------------------------------------
# One text per card
# ----------------------------------------------------------------------------------------------------
def _why_abc_missing(result: Optional[Result], lang: str, reason: str) -> str:
    if result is None:
        return no_result_reason(reason, lang)
    if result.status != pdt.Status.OK:
        if result.status == pdt.Status.CONTOUR_FAILED:
            return ("el contorno fue rechazado (paso del contorno): sin línea de membrana no hay nada que medir"
                    if lang == "es" else
                    "the contour was rejected (contour step): without a membrane line there is nothing to measure")
        return stopped_reason(result, lang)
    return ""


def _welcome(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    if result is None:
        return (f"todavía no hay respuestas de este axón: no se calcularon porque {no_result_reason(reason, lang)}. "
                "Las tarjetas igual explican qué pregunta responde cada parte." if es else
                f"there are no answers for this axon yet: they were not computed because "
                f"{no_result_reason(reason, lang)}. The cards still explain what each part asks.")
    label = (result.trace.label if result.trace is not None and result.trace.label else result.axon_id) or "?"
    p = result.params
    crit = {"peak": ("el pico", "the peak"), "focus_nominal": ("el foco nominal", "the nominal focus"),
            "focus_empirical": ("el foco empírico", "the empirical focus")}.get(p.z0_criterion.value,
                                                                              (p.z0_criterion.value,) * 2)
    half = result.band.half_width_nm if result.band is not None else p.band_half_width_nm
    _kind, state = contour_state_words(result, lang)
    done = [name for name, ok in (("A", bool(result.a)), ("B", result.b is not None), ("C", bool(result.c))) if ok]
    if es:
        calc = ("se calcularon " + ", ".join(done)) if done else "no se calcularon A, B ni C"
        return (f"las respuestas que siguen son las del axón abierto en la ventana ({label}), del último Run: z0 en "
                f"{crit[0]}, banda de ±{num(half, '{:g}', lang)} nm. Estado: {state}. En este Run {calc}.")
    calc = ("computed " + ", ".join(done)) if done else "A, B and C were not computed"
    return (f"the answers below are those of the axon open in the window ({label}), from the last Run: z0 at "
            f"{crit[1]}, band +-{num(half, '{:g}', lang)} nm. State: {state}. In this Run: {calc}.")


def _slices(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    what = {"es": "z0", "en": "z0"}
    if result is None:
        return _not_computed(what, no_result_reason(reason, lang), lang)
    z = result.z0
    if z.z0_nm is None:
        return _not_computed(what, stopped_reason(result, lang) or z.status.value, lang)
    crit = {"peak": ("pico", "peak"), "focus_nominal": ("foco nominal", "nominal focus"),
            "focus_empirical": ("foco empírico", "empirical focus")}.get(z.criterion.value, (z.criterion.value,) * 2)
    head = f"z0 = {num(z.z0_nm, '{:.0f}', lang)} nm ({crit[0] if es else crit[1]})."
    sl = result.slices
    if sl is None:
        return head + " " + _not_computed({"es": "El corte en rodajas", "en": "The slicing"},
                                          stopped_reason(result, lang), lang)
    n = [_int(v, lang) for v in sl.n_locs]
    text = (f"{head} Las rodajas tienen {n[0]} (z−1), {n[1]} (central) y {n[2]} (z+1) localizaciones; las de los "
            f"costados son el {num(sl.pct_of_central[0], '{:.0f}', lang)} % y el "
            f"{num(sl.pct_of_central[1], '{:.0f}', lang)} % de la central." if es else
            f"{head} The slices hold {n[0]} (z-1), {n[1]} (central) and {n[2]} (z+1) localizations; the side ones are "
            f"{num(sl.pct_of_central[0], '{:.0f}', lang)} % and {num(sl.pct_of_central[1], '{:.0f}', lang)} % of "
            "the central one.")
    dg = result.diagnostics
    if dg is not None and dg.z_truncated_slices:
        names = ", ".join(f"{k:+d}" for k in dg.z_truncated_slices)
        text += (f" El rango en z del archivo corta la(s) rodaja(s) {names}." if es else
                 f" The file's z range truncates slice(s) {names}.")
    return text


def _contour(result: Optional[Result], lang: str, reason: str) -> str:
    _kind, state = contour_state_words(result, lang, reason)
    return state[:1].upper() + state[1:] + "."


def _unanchored(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    what = {"es": "El recorte de espectrina interior", "en": "The interior trim"}
    if result is None:
        return _not_computed(what, no_result_reason(reason, lang), lang)
    half = result.band.half_width_nm if result.band is not None else None
    band = ("" if half is None else
            (f" El perfil cuenta las localizaciones dentro de la banda de ±{num(half, '{:g}', lang)} nm." if es else
             f" The profile counts the localizations within the +-{num(half, '{:g}', lang)} nm band."))
    trim = result.extras.get("contour_trim_nm")
    if trim is None:
        return ("en este Run el recorte de espectrina interior estaba apagado (es el valor por defecto): no se dejó "
                "nada afuera." if es else
                "in this Run the trim of interior spectrin was off (the default): nothing was left out.") + band
    n_trim = result.extras.get("contour_n_locs_trimmed")
    status = result.extras.get("contour_trim_status", "")
    return (f"en este Run el recorte estaba prendido a {num(trim, '{:g}', lang)} nm: {_int(n_trim, lang)} "
            f"localizaciones quedaron fuera del AJUSTE del contorno (estado: «{status}»); la calidad se midió con "
            "todas." if es else
            f"in this Run the trim was on at {num(trim, '{:g}', lang)} nm: {_int(n_trim, lang)} localizations were "
            f"left out of the contour FIT (status: \"{status}\"); the quality was measured on all of them.") + band


def _profile(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    what = {"es": "El perfil", "en": "The profile"}
    if result is None or result.profiles is None:
        why = _why_abc_missing(result, lang, reason)
        return _not_computed(what, why or ("el análisis se detuvo antes" if es else "the analysis stopped earlier"),
                             lang)
    prof = result.profiles
    p0 = prof.by_slice.get(0)
    if p0 is None:
        return _not_computed(what, "falta la rodaja central" if es else "the central slice is missing", lang)
    dg = result.diagnostics
    unsupported = float("nan") if dg is None else 100.0 * float(dg.unsupported_fraction)
    text = (f"el anillo desenrollado mide {num(prof.length_nm, '{:,.0f}', lang)} nm y se dividió en {p0.n_bins} "
            f"tramos de {num(p0.bin_nm_effective, '{:.0f}', lang)} nm; la rodaja central aporta "
            f"{_int(p0.n_locs, lang)} localizaciones dentro de la banda y el {num(unsupported, '{:.0f}', lang)} % de "
            "sus tramos quedó sin datos." if es else
            f"the unrolled ring is {num(prof.length_nm, '{:,.0f}', lang)} nm long and was cut into {p0.n_bins} "
            f"stretches of {num(p0.bin_nm_effective, '{:.0f}', lang)} nm; the central slice brings "
            f"{_int(p0.n_locs, lang)} localizations within the band and {num(unsupported, '{:.0f}', lang)} % of its "
            "stretches have no data.")
    if p0.max_tie:
        text += (" Hay un empate en el tramo más denso." if es else " There is a tie for the densest stretch.")
    return text + _unreliable_tail(result, lang)


def _a(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    what = {"es": "A", "en": "A"}
    why = _why_abc_missing(result, lang, reason)
    if why:
        return _not_computed(what, why, lang)
    assert result is not None
    c, p = result.a.get("contrast"), result.a.get("period")
    dg = result.diagnostics
    if c is None or p is None or (dg is not None and not dg.a_usable):
        return _not_computed(what, "lo que queda fuera de la zona de exclusión es demasiado corto para compararlo"
                             if es else "what remains outside the exclusion zone is too short to compare", lang)
    period_nm = None if dg is None else dg.a_period_nm
    excl = None if dg is None else dg.a_exclusion_nm
    parts_es: List[str] = []
    parts_en: List[str] = []
    if _finite(c.p) and float(c.p) <= ALPHA and _finite(c.z):
        if float(c.z) > 0:
            parts_es.append("hay más densidad cerca del máximo que en el nulo (se atrae)")
            parts_en.append("more density near the maximum than the null (attraction)")
        else:
            parts_es.append("hay menos densidad cerca del máximo que en el nulo (se repele)")
            parts_en.append("less density near the maximum than the null (repulsion)")
    if _finite(p.p) and float(p.p) <= ALPHA:
        parts_es.append(f"el patrón se repite cada {num(period_nm, '{:.0f}', lang)} nm")
        parts_en.append(f"the pattern repeats every {num(period_nm, '{:.0f}', lang)} nm")
    edge = bool(dg is not None and dg.a_period_at_edge)
    if es:
        reading = ("; ".join(parts_es) if parts_es else "nada alrededor del máximo se sale de lo que da el azar")
        text = (f"contraste cerca / lejos = {_stat(c, lang, '{:.3g}')}; período = {num(period_nm, '{:.0f}', lang)} "
                f"nm (z = {num(p.z, '{:.2f}', lang)}; p = {num(p.p, '{:.3f}', lang)}); zona de exclusión "
                f"{num(excl, '{:.0f}', lang)} nm. Lectura: {reading} (descriptivo: la decisión de grupo sale del "
                "Batch).")
        if edge:
            text += " El período cae en el borde de la búsqueda: tomarlo con cuidado."
    else:
        reading = ("; ".join(parts_en) if parts_en else "nothing around the maximum leaves what chance gives")
        text = (f"contrast near / far = {_stat(c, lang, '{:.3g}')}; period = {num(period_nm, '{:.0f}', lang)} nm "
                f"(z = {num(p.z, '{:.2f}', lang)}, p = {num(p.p, '{:.3f}', lang)}); exclusion zone "
                f"{num(excl, '{:.0f}', lang)} nm. Reading: {reading} (descriptive: the group call comes from the "
                "Batch).")
        if edge:
            text += " The period sits at the edge of the search: handle it with care."
    return text + _unreliable_tail(result, lang)


def _b(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    what = {"es": "B", "en": "B"}
    why = _why_abc_missing(result, lang, reason)
    if why:
        return _not_computed(what, why, lang)
    assert result is not None
    b = result.b
    if b is None:
        dg = result.diagnostics
        n_bins = None
        if result.profiles is not None and 0 in result.profiles.by_slice:
            n_bins = result.profiles.by_slice[0].n_bins
        wb = 0 if dg is None else dg.window_bins
        detail = (f"el perfil tiene {n_bins} tramos y la ventana de curvatura necesita {wb}" if es else
                  f"the profile has {n_bins} stretches and the curvature window needs {wb}") \
            if n_bins is not None and wb else ("el perfil es demasiado corto para la ventana de curvatura" if es
                                               else "the profile is too short for the curvature window")
        return _not_computed(what, detail, lang)
    bt = b.test
    if not (_finite(bt.p) and _finite(bt.value)):
        reading_es, reading_en = "no calculable", "not computable"
    elif float(bt.p) > ALPHA:
        reading_es = "ningún vínculo entre curvatura y densidad se sale del nulo de rotaciones"
        reading_en = "no link between curvature and density stands out of the rotation null"
    elif float(bt.value) < 0:
        reading_es = "más densidad en los tramos más curvados de la membrana"
        reading_en = "denser on the more curved stretches of the membrane"
    else:
        reading_es = "más densidad en los tramos más rectos de la membrana"
        reading_en = "denser on the straighter stretches of the membrane"
    if es:
        text = (f"Spearman = {_stat(bt, lang)}, contra un nulo de {_int(bt.n_null, lang)} rotaciones. τ mínimo = "
                f"{num(b.tau_min, '{:.3f}', lang)} (la tortuosidad más baja entre ventanas: cuanto más lejos de 1, "
                f"más contraste de curvatura hay para comparar). Control tortuosidad contra desplazamiento radial: "
                f"r = {num(b.tau_vs_offset, '{:+.2f}', lang)} (cerca de 0 = bien; cerca de ±1 = el contorno, y no la "
                f"membrana, podría estar armando B). Lectura: {reading_es}.")
        if not b.resolved:
            text += " La curvatura no está resuelta en este axón (radio menor que 600 nm): leer B con cautela."
    else:
        text = (f"Spearman = {_stat(bt, lang)}, against a null of {_int(bt.n_null, lang)} rotations. Minimum tau = "
                f"{num(b.tau_min, '{:.3f}', lang)} (the lowest window tortuosity: the further from 1, the more "
                f"curvature contrast there is to compare). Control, tortuosity vs radial offset: "
                f"r = {num(b.tau_vs_offset, '{:+.2f}', lang)} (near 0 = fine; near +-1 = the contour, not the "
                f"membrane, may drive B). Reading: {reading_en}.")
        if not b.resolved:
            text += " The curvature is not resolved on this axon (radius below 600 nm): read B with care."
    return text + _unreliable_tail(result, lang)


def leak_widths(result: Result) -> bc.LeakWidths:
    """The axial widths the run read C against (the file's own when measured, else the pilot's)."""
    dg = result.diagnostics
    if dg is not None and all(_finite(v) for v in dg.sigma_z_widths):
        lo, main, hi = (float(v) for v in dg.sigma_z_widths)
        if dg.sigma_z_source == bc.SIGMA_Z_SOURCE_MEASURED:
            return bc.measured_widths(lo, main, hi)
        return bc.LeakWidths(lo, main, hi, bc.LEAK_SIGMA_Z_UPPER_NM, dg.sigma_z_source or bc.SIGMA_Z_SOURCE_PILOT)
    return bc.pilot_widths()


def leak_by_pair(result: Result, sigma_z_nm: float) -> Dict[Tuple[int, int], float]:
    """The leak each pair of C would show from axial leakage alone, as the analysis computes it for its reading."""
    nan = float("nan")
    if result.slices is None:
        return {(-1, 0): nan, (0, 1): nan, (-1, 1): nan}
    p = result.params
    return bc.leak_expected_r_pairs(result.slices.n_locs, float(sigma_z_nm), p.slice_thickness_nm, p.slice_guard_nm)


def c_reading_words(label: str, lang: str) -> str:
    """The reading of C the analysis writes (``diagnostics.c_reading``), in ``lang``."""
    words = _C_LABELS[lang]
    m = re.match(r"^(.*?) \(within the measured range of sigma_z (\d+)-(\d+) nm: (.*)\)$", label or "")
    if not m:
        return words.get(label, label)
    head, lo, hi, rest = m.groups()
    others = []
    for part in rest.split("; "):
        mm = re.match(r"^(.*) at (\d+) nm$", part)
        if mm:
            short = words.get(mm.group(1), mm.group(1)).split(" (")[0]
            others.append(f"{short} a {mm.group(2)} nm" if lang == "es" else f"{short} at {mm.group(2)} nm")
        else:
            others.append(part)
    tail = (f" (dentro del rango medido de σz {lo}–{hi} nm: {'; '.join(others)})" if lang == "es" else
            f" (within the measured range of sigma_z {lo}-{hi} nm: {'; '.join(others)})")
    return words.get(head, head) + tail


def _c(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    what = {"es": "C", "en": "C"}
    why = _why_abc_missing(result, lang, reason)
    if why:
        return _not_computed(what, why, lang)
    assert result is not None
    if not result.c:
        return _not_computed(what, "no hubo perfiles para correlacionar" if es else "there were no profiles to "
                                                                                    "correlate", lang)
    w = leak_widths(result)
    leaks = {sz: leak_by_pair(result, sz) for sz in {w.lo, w.main, w.hi, w.upper}}
    main_leak = leaks[w.main]
    names = _PAIR_NAME[lang]
    skip = result.c.get((-1, 1))
    fmt = {"lo": f"{w.lo:.0f}", "main": f"{w.main:.0f}", "hi": f"{w.hi:.0f}", "upper": f"{w.upper:.0f}"}
    if skip is not None:
        vs = bc.m1_p1_vs_leak(skip.value, {sz: leaks[sz][(-1, 1)] for sz in leaks}, widths=w)
        vs_words = _VS_LEAK[lang].get(bc.vs_leak_kind(vs), "").format(**fmt) if vs else ""
        sig = _finite(skip.p) and float(skip.p) < ALPHA and _finite(skip.value) and float(skip.value) > 0
        if es:
            verdict = ("significativa y positiva: hay registro entre rodajas" if sig
                       else "no significativa: no hay registro entre rodajas que se salga del azar")
            main_text = (f"LECTURA PRINCIPAL, {names[(-1, 1)]} (no se tocan, la fuga casi no llega): r = "
                         f"{_stat(skip, lang)}; fuga esperada para este par r = "
                         f"{num(main_leak[(-1, 1)], '{:.2f}', lang)}" + (f", {vs_words}" if vs_words else "")
                         + f"; {verdict}.")
        else:
            verdict = ("significant and positive: there is registry between slices" if sig
                       else "not significant: no registry between slices beyond chance")
            main_text = (f"MAIN READING, {names[(-1, 1)]} (they do not touch, the leak hardly reaches): r = "
                         f"{_stat(skip, lang)}; expected leak for this pair r = "
                         f"{num(main_leak[(-1, 1)], '{:.2f}', lang)}" + (f", {vs_words}" if vs_words else "")
                         + f"; {verdict}.")
    else:
        main_text = (f"{names[(-1, 1)]}: no calculado." if es else f"{names[(-1, 1)]}: not computed.")
    neigh: List[str] = []
    for pair in ((-1, 0), (0, 1)):
        t = result.c.get(pair)
        if t is None:
            neigh.append(f"{names[pair]}: " + ("no calculado" if es else "not computed"))
            continue
        above = _finite(t.value) and _finite(main_leak[pair]) and float(t.value) > float(main_leak[pair])
        cmp_es = "por encima de la fuga" if above else "no supera la fuga"
        cmp_en = "above the leak" if above else "not above the leak"
        neigh.append(f"{names[pair]}: r = {_stat(t, lang)}, " +
                     (f"fuga esperada {num(main_leak[pair], '{:.2f}', lang)} ({cmp_es})" if es else
                      f"expected leak {num(main_leak[pair], '{:.2f}', lang)} ({cmp_en})"))
    dg = result.diagnostics
    label = "" if dg is None else dg.c_reading
    source = ("medido en este archivo" if es else "measured in this file") \
        if w.source == bc.SIGMA_Z_SOURCE_MEASURED else ("valor del piloto" if es else "the pilot's value")
    if es:
        text = (f"{main_text} Pares vecinos (comparten la fuga axial; se comparan con su fuga esperada): "
                f"{'; '.join(neigh)}. Lectura del programa: "
                f"{c_reading_words(label, lang) if label else 'no calculada'}. Fuga calculada con σz = "
                f"{num(w.main, '{:.0f}', lang)} nm ({source}) y las fracciones de rodaja de este axón.")
    else:
        text = (f"{main_text} Neighbouring pairs (they share the axial leak; compared with their expected leak): "
                f"{'; '.join(neigh)}. The program's reading: "
                f"{c_reading_words(label, lang) if label else 'not computed'}. Leak computed with sigma_z = "
                f"{num(w.main, '{:.0f}', lang)} nm ({source}) and this axon's slice fractions.")
    return text + _unreliable_tail(result, lang)


def _validation(_result: Optional[Result], lang: str, _reason: str) -> str:
    f = data.facts(lang)
    if lang == "es":
        return (f"estas cifras no se miden en este axón: vienen de las simulaciones del informe de validación. "
                f"Falsas alarmas C {f['val_c_fp_lo']}–{f['val_c_fp_hi']} %, B {f['val_b_fp']} %, A ≤ {f['val_a_fp']} %. "
                f"Detección de un patrón real: C {f['val_c_power']}, B {f['val_b_power']} en formas asimétricas "
                f"({f['val_b_power_sym']} en una elipse simétrica), A atractor {f['val_a_power']} con 18 axones.")
    return (f"these figures are not measured on this axon: they come from the simulations of the validation report. "
            f"False alarms C {f['val_c_fp_lo']}-{f['val_c_fp_hi']} %, B {f['val_b_fp']} %, A <= {f['val_a_fp']} %. "
            f"Detection of a real pattern: C {f['val_c_power']}, B {f['val_b_power']} on asymmetric shapes "
            f"({f['val_b_power_sym']} on a symmetric ellipse), A attractor {f['val_a_power']} with 18 axons.")


def _missing(result: Optional[Result], lang: str, reason: str) -> str:
    es = lang == "es"
    if result is None:
        return (f"de este axón todavía no hay nada calculado ({no_result_reason(reason, lang)}). Aun calculado, un "
                "axón solo no compara animales ni genotipos: eso sale del Batch." if es else
                f"nothing has been computed for this axon yet ({no_result_reason(reason, lang)}). Even computed, one "
                "axon compares no animals or genotypes: that comes from the Batch.")
    kind, _state = contour_state_words(result, lang)
    kind_words = {"es": {"stopped": "el análisis se detuvo antes del contorno", "rejected": "el contorno fue rechazado",
                         "unreliable": "el contorno no es confiable (queda fuera del grupo)",
                         "drawn": "se usó el trazado a mano (sólo comparación)",
                         "reliable": "el contorno es confiable"},
                  "en": {"stopped": "the analysis stopped before the contour", "rejected": "the contour was rejected",
                         "unreliable": "the contour is not reliable (left out of the group)",
                         "drawn": "the hand-drawn trace was used (comparison only)",
                         "reliable": "the contour is reliable"}}[lang].get(kind, kind)
    return (f"para este axón, {kind_words}. Un axón solo no compara animales ni genotipos: la lectura de grupo sale "
            "del Batch, con muchos axones de contorno confiable (A necesita al menos 9)." if es else
            f"for this axon, {kind_words}. One axon compares no animals or genotypes: the group reading comes from "
            "the Batch, with many axons with a reliable contour (A needs at least 9).")


_BUILDERS = {"pdd_welcome": _welcome, "pdd_slices": _slices, "pdd_contour": _contour, "pdd_unanchored": _unanchored,
             "pdd_profile": _profile, "pdd_a": _a, "pdd_b": _b, "pdd_c": _c, "pdd_validation": _validation,
             "pdd_missing": _missing}


def live_example(key: str, result: Optional[Result], lang: str, reason: str = NO_RESULT_NOT_RUN) -> str:
    """The "example" of card ``key`` for this run: "En este axón: ..." / "On this axon: ..." (plain text, braces not
    escaped). Cards this module does not know keep "" (the caller keeps the static text)."""
    build = _BUILDERS.get(key)
    if build is None:
        return ""
    return f"{LIVE_HEADING[lang]}: {build(result, lang, reason)}"


def live_steps(steps: List[Any], result: Optional[Result], reason: str = NO_RESULT_NOT_RUN) -> List[Any]:
    """``steps`` (``mps_tour.Step`` of the presentation tour) with every known card's "example" replaced by the
    answer of THIS run, in every language the step has. ``result`` None: no run yet (``reason`` says why)."""
    out: List[Any] = []
    for step in steps:
        texts = {lang: dict(t) for lang, t in step.texts.items()}
        for lang, t in texts.items():
            words = live_example(step.key, result, lang, reason)
            if words:
                t["example"] = _escape(words)
        out.append(replace(step, texts=texts))
    return out


__all__ = ["LIVE_HEADING", "NO_RESULT_NOT_RUN", "NO_RESULT_RUNNING", "live_example", "live_steps",
           "contour_state_words", "c_reading_words", "leak_widths", "leak_by_pair", "num"]
