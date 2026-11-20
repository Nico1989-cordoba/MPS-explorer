# -*- coding: utf-8 -*-
"""
The texts of the beginner tour (``tools/mps_tour.py``), in Spanish and English.

One list of steps per window. Each step is anchored to a widget by its ``objectName`` (``target``; empty for a card
in the middle of the window) and belongs to a panel (``panel``), so that a panel's "?" button replays only its part.
A card has four fixed parts - what it is, what it is for, how to read it, an example - and a longer text behind
"More" (``more``).

Where the texts come from
-------------------------
* A step about a parameter the registry documents names its keys in ``registry``. The card's "More" then shows the
  registry's own text (``tools.mps_param_registry.tooltip``, English) under its own heading, and the short texts here
  quote the registry's default and source through the placeholders below - never a number typed twice. The test
  ``test_beginner_tour_gui.py`` checks that every key exists.
* Every example number comes from the SIMULATED example axon (``examples/demo_axon``, written by
  ``tools/make_demo_axon.py``) and is written ``{name}`` from ``DEMO_FACTS``; every example says "ejemplo simulado" /
  "simulated example". The test re-measures each fact on the program's own output for that file.
* ``method``: a page of the repository that says more (opened locally, or on GitHub when the file is not there).

Placeholders: ``DEMO_FACTS`` (formatted per language) and the registry values of ``registry_values()``.

The MPS analysis window's tour is not written yet: its rows change in the pending science PRs. ``STEPS["mps_analysis"]``
is the hook - an empty list, which the engine treats as "no tour" (no card opens on its own, and "Explain this" says
there is no explanation yet).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

LANGUAGES: Tuple[str, ...] = ("es", "en")
DEFAULT_LANGUAGE = "es"
FIELDS: Tuple[str, ...] = ("title", "what", "why", "read", "example", "more")
# The word every example carries, per language (checked by the test).
SIMULATED_WORD: Dict[str, str] = {"es": "simulad", "en": "simulated"}

# What the program shows for the simulated example axon with the default parameters (eps 25 nm, min samples 10,
# half-width 90 nm), as measured by test_beginner_tour_gui.py on the program's own output. Nothing here is a
# measurement of a real axon.
DEMO_FACTS: Dict[str, Any] = {
    "n_locs": 12137,          # localizations in the file
    "n_frames": 60000,        # frames in its YAML
    "pixel_nm": 130,          # pixel size in its YAML
    "z_lo": -283,             # z range of the file, nm (rounded)
    "z_hi": 274,
    "peak": 173.2,            # centre of the automatic slab, nm
    "zmin": 83.2,             # the slab, nm (as the Z min / Z max fields show it)
    "zmax": 263.2,
    "n_slab": 4290,           # localizations in the ROI and the slab
    "n_clusters": 56,         # DBSCAN clusters of the slab
    "n_kept": 56,             # after the automatic curation
    "n_noise": 1418,          # localizations of the slab left as noise
    "nena": 9.4,              # Data quality: NeNA, nm (maximum-likelihood fit, fix/nena-fit)
    "lp_reported": 8.9,       # Data quality: reported lateral precision (median), nm
    "nena_ratio": 1.05,       # NeNA / reported
}


def _number(value: Any, lang: str) -> str:
    if isinstance(value, int):
        text = f"{value:,}"
        return text.replace(",", ".") if lang == "es" else text
    if isinstance(value, float):
        decimals = 2 if abs(value) < 1 else 1
        text = f"{value:.{decimals}f}"
        return text.replace(".", ",") if lang == "es" else text
    return str(value)


# What the perimeter density window shows for the same simulated example axon with its default bar (z0 from the peak,
# band +-90 nm, automatic contour), measured by ``tools.mps_perimeter_window.perimeter_demo_facts`` on the window's
# own result and checked by test_perimeter_demo_tour_gui.py. On this axon the automatic contour is REJECTED by its
# own quality gate, so A, B and C are not computed: the presentation tour says so instead of inventing numbers.
PERIMETER_DEMO_FACTS: Dict[str, Any] = {
    "pd_z0": 4,                # z0, nm, as the window writes it (peak of the z density curve)
    "pd_n_minus": 3534,        # localizations of the slices z-1, z0, z+1
    "pd_n_central": 4307,
    "pd_n_plus": 4242,
    "pd_pct_minus": 82,        # % of the central slice (rounded as the window writes it)
    "pd_pct_plus": 98,
    "pd_len_nm": 34540,        # length of the rejected automatic contour, nm
    "pd_len_ratio": 2.4,       # ... times the perimeter of the cloud's convex hull
    "pd_len_limit": 1.5,       # the gate's limit
    "pd_polar_off_nm": 90,     # the polar fallback: distance to the nearest band along a stretch, nm
    "pd_polar_limit_nm": 75,   # ... and its limit
}

# Numbers of the validation report of the perimeter analysis (simulation with the true contour; P11).
PERIMETER_VALIDATION: Dict[str, Any] = {
    "val_c_fp_lo": 4.6, "val_c_fp_hi": 5.6, "val_b_fp": 3.6, "val_a_fp": 2.6,
    "val_c_power": 0.98, "val_b_power": 0.90, "val_b_power_sym": 0.23, "val_a_power": 0.62,
}


def facts(lang: str) -> Dict[str, str]:
    """``DEMO_FACTS`` (and the perimeter window's demo facts) written the way ``lang`` writes numbers (es: 12.137 and
    83,2; en: 12,137 and 83.2)."""
    out = {k: _number(v, lang) for k, v in DEMO_FACTS.items()}
    out.update({k: _number(v, lang) for k, v in PERIMETER_DEMO_FACTS.items()})
    out.update({k: _number(v, lang) for k, v in PERIMETER_VALIDATION.items()})
    return out


def registry_values() -> Dict[str, str]:
    """The registry's values the texts quote, read from the registry (and through it from the code) now."""
    from tools import mps_param_registry as reg

    half = float(reg.default("slab.half_width_nm"))
    return {
        "eps_default": reg.format_value("dbscan.eps_nm"),
        "min_default": reg.format_value("dbscan.min_samples"),
        "min_range": reg.info("dbscan.min_samples").range_text,
        "half_width": reg.format_value("slab.half_width_nm"),
        "slab_total": f"{2.0 * half:g} nm",
        "paper": reg.PAPER,
    }


# --------------------------------------------------------------------------- the words of the engine
UI: Dict[str, Dict[str, str]] = {
    "es": {
        "help": "Ayuda",
        "beginner": "Modo principiante",
        "tour_window": "Recorrer esta ventana",
        "explain": "Explicame esto",
        "demo": "Axón simulado para demostrar las capacidades de análisis del MPS Explorer",
        "language": "Idioma de las explicaciones",
        "what": "Qué es",
        "why": "Para qué sirve",
        "read": "Cómo se lee",
        "example": "Ejemplo",
        "more": "Más",
        "less": "Menos",
        "method": "Ver método completo",
        "prev": "Anterior",
        "next": "Siguiente",
        "finish": "Terminar",
        "skip": "Saltar",
        "close": "Cerrar",
        "step": "Paso {n} de {total}",
        "dont_show": "No mostrar de nuevo",
        "no_step_title": "Todavía no hay explicación para este control",
        "no_step_text": "Este control no tiene una tarjeta en el recorrido todavía.",
        "its_tooltip": "Lo que dice su ayuda emergente:",
        "registry": "Del registro de parámetros (texto original, en inglés)",
        "badges": "Las etiquetas de origen, según el registro (en inglés)",
        "not_visible": "Este control no está a la vista ahora (por ejemplo, la ventana es muy chica o el control está "
                       "oculto): la tarjeta lo describe igual.",
        "panel_help": "Explicar este panel",
        "explain_hint": "Explicame esto: hacé clic en un control (Esc cancela)",
        "demo_missing": "No se encontró el axón de ejemplo en {path}. Se vuelve a generar con:\n"
                        "venv\\Scripts\\python.exe tools\\make_demo_axon.py",
        "demo_title": "Axón simulado de demostración",
    },
    "en": {
        "help": "Help",
        "beginner": "Beginner mode",
        "tour_window": "Tour this window",
        "explain": "Explain this",
        "demo": "Simulated axon to demonstrate the analysis capabilities of MPS Explorer",
        "language": "Language of the explanations",
        "what": "What it is",
        "why": "What it is for",
        "read": "How to read it",
        "example": "Example",
        "more": "More",
        "less": "Less",
        "method": "See the full method",
        "prev": "Previous",
        "next": "Next",
        "finish": "Finish",
        "skip": "Skip",
        "close": "Close",
        "step": "Step {n} of {total}",
        "dont_show": "Do not show again",
        "no_step_title": "There is no explanation for this control yet",
        "no_step_text": "This control has no card in the tour yet.",
        "its_tooltip": "What its tooltip says:",
        "registry": "From the parameter registry (original text)",
        "badges": "The origin badges, from the registry",
        "not_visible": "This control is not in view right now (the window may be too small, or the control hidden): "
                       "the card describes it anyway.",
        "panel_help": "Explain this panel",
        "explain_hint": "Explain this: click a control (Esc cancels)",
        "demo_missing": "The example axon was not found at {path}. It is written again with:\n"
                        "venv\\Scripts\\python.exe tools\\make_demo_axon.py",
        "demo_title": "Simulated demonstration axon",
    },
}

LANGUAGE_NAMES: Dict[str, str] = {"es": "Español", "en": "English"}

# The panels of the main window that have a "?" button: (panel id, objectName of the panel widget).
MAIN_PANELS: Tuple[Tuple[str, str], ...] = (
    ("files", "groupBox_files_3"),
    ("view", "groupBox_vis"),
    ("clustering", "groupBox"),
    ("measurement", "measurement_panel"),
)

_README = "README.md"
_GUIDE = "docs/beginner_mode.md"


def _step(key: str, target: str, panel: str, es: Dict[str, str], en: Dict[str, str], *,
          registry: Tuple[str, ...] = (), badges: bool = False, method: str = "",
          headings: bool = True) -> Dict[str, Any]:
    return {"key": key, "target": target, "panel": panel, "registry": registry, "badges": badges,
            "method": method, "headings": headings, "es": es, "en": en}


# --------------------------------------------------------------------------- the main window
MAIN_STEPS: List[Dict[str, Any]] = [
    _step("welcome", "", "", method=_GUIDE, es={
        "title": "Bienvenido/a a MPS Explorer",
        "what": "Un programa para medir el esqueleto periódico de membrana (MPS) de los axones a partir de "
                "localizaciones de microscopía de superresolución (SMLM) en 3D.",
        "why": "Este recorrido muestra, control por control, cómo se pasa de un archivo de localizaciones a los "
               "números de un axón y a una tabla exportada.",
        "read": "Siguiente (o →) avanza, Anterior (o ←) vuelve, Saltar o Esc cierra. El recuadro de color marca el "
                "control del que habla la tarjeta; el resto de la ventana queda desenfocado.",
        "example": "Podés seguirlo con el axón de ejemplo: Ayuda → «Axón simulado para demostrar las capacidades de análisis del MPS Explorer». Tiene "
                   "{n_locs} localizaciones simuladas (ejemplo simulado: no es una medición).",
        "more": "El modo principiante muestra el recorrido de cada ventana la primera vez que se abre; se apaga desde "
                "Ayuda o con la casilla «No mostrar de nuevo». «Explicame esto» (Shift+F1) cambia el cursor: el "
                "próximo clic sobre un control muestra su tarjeta en vez de ejecutarlo. El botón «?» de cada panel "
                "repite sólo la parte de ese panel.\n\nDe dónde salen los textos: cuando una tarjeta habla de un "
                "parámetro, sus valores por defecto y su origen se leen del registro de parámetros del programa "
                "(tools/mps_param_registry.py), el mismo que alimenta las etiquetas de origen, y «Más» muestra el "
                "texto original del registro. Todos los números de los ejemplos vienen del axón simulado que trae el "
                "programa (examples/demo_axon), y por eso dicen «ejemplo simulado».",
    }, en={
        "title": "Welcome to MPS Explorer",
        "what": "A program that measures the membrane-associated periodic skeleton (MPS) of axons from 3D "
                "super-resolution (SMLM) localizations.",
        "why": "This tour goes control by control from a localization file to the numbers of one axon and to an "
               "exported table.",
        "read": "Next (or →) moves on, Previous (or ←) goes back, Skip or Esc closes. The coloured frame marks the "
                "control the card is about; the rest of the window is blurred.",
        "example": "You can follow it with the example axon: Help → \"Simulated axon to demonstrate the analysis capabilities of MPS Explorer\". It holds "
                   "{n_locs} simulated localizations (simulated example: not a measurement).",
        "more": "Beginner mode shows each window's tour the first time that window opens; turn it off in Help or "
                "with the \"Do not show again\" box. \"Explain this\" (Shift+F1) changes the cursor: the next click "
                "on a control shows its card instead of running it. The \"?\" button of each panel replays only "
                "that panel's part.\n\nWhere the texts come from: when a card is about a parameter, its default and "
                "its source are read from the program's parameter registry (tools/mps_param_registry.py), the same "
                "one the origin badges come from, and \"More\" shows the registry's own text. Every example number "
                "comes from the simulated axon that ships with the program (examples/demo_axon), which is why each "
                "one says \"simulated example\".",
    }),
    # ------------------------------------------------------------------ loading
    _step("file_format", "comboBox_fileformat", "files", es={
        "title": "Formato del archivo",
        "what": "Cómo leer el archivo de localizaciones del canal 1: Picasso hdf5, CSV de ThunderSTORM o un CSV "
                "propio.",
        "why": "Cada formato guarda las columnas distinto; el programa las tiene que encontrar para saber dónde "
               "está cada localización.",
        "read": "Picasso hdf5 es lo habitual: el tamaño de píxel se lee del archivo YAML que está al lado del .hdf5. "
                "Con los CSV, el tamaño de píxel hay que escribirlo a mano.",
        "example": "El axón de ejemplo es un Picasso hdf5 con su YAML, que dice un píxel de {pixel_nm} nm (ejemplo "
                   "simulado).",
        "more": "Picasso guarda x e y en píxeles de cámara y z en nanómetros: por eso el formato importa, porque "
                "decide si hace falta un tamaño de píxel para convertir. Un archivo también se puede soltar "
                "arrastrándolo sobre la ventana; el formato se deduce de la extensión.",
    }, en={
        "title": "File format",
        "what": "How to read the channel-1 localization file: Picasso hdf5, ThunderSTORM csv or a custom csv.",
        "why": "Each format stores its columns differently; the program has to find them to know where each "
               "localization is.",
        "read": "Picasso hdf5 is the usual one: the pixel size is read from the YAML file beside the .hdf5. With "
                "the csv formats the pixel size has to be entered by hand.",
        "example": "The example axon is a Picasso hdf5 with its YAML, which says a pixel of {pixel_nm} nm "
                   "(simulated example).",
        "more": "Picasso stores x and y in camera pixels and z in nanometres: the format matters because it decides "
                "whether a pixel size is needed to convert. A file can also be dropped onto the window; the format "
                "is taken from its extension.",
    }),
    _step("load_channel1", "pushButton_browsefile", "files", es={
        "title": "Cargar el canal 1",
        "what": "Elige el archivo del canal 1: la proteína sobre la que se mide todo (por ejemplo, βII-espectrina).",
        "why": "Cada archivo suele ser un axón ya recortado («pickeado») en Picasso. Al cargarlo, el programa lo "
               "dibuja y pone un círculo que abarca todas sus localizaciones, ya aplicado: se puede clusterizar "
               "enseguida.",
        "read": "Si cargás un campo entero (con muchos axones), el círculo abarca todo el campo: movelo y achicalo "
                "hasta que contenga un solo axón, o usá un rectángulo o un polígono.",
        "example": "Al abrir el axón de ejemplo se cargan {n_locs} localizaciones y el círculo queda aplicado "
                   "(ejemplo simulado).",
        "more": "Un archivo nuevo borra todo lo calculado con el anterior - la selección, los clusters, el análisis "
                "y sus ventanas - para que nada del archivo viejo quede guardado con el nombre del nuevo. La carpeta "
                "del último archivo abierto se recuerda para el próximo diálogo.",
    }, en={
        "title": "Load channel 1",
        "what": "Choose the channel-1 file: the protein everything is measured on (for example betaII-spectrin).",
        "why": "Each file is usually one axon already picked in Picasso. When it loads, the program draws it and "
               "puts a circle around all its localizations, already applied: you can cluster at once.",
        "read": "If you load a whole field (many axons), the circle covers the whole field: move it and shrink it "
                "until it holds one axon, or use a rectangle or a polygon.",
        "example": "Opening the example axon loads {n_locs} localizations and applies the circle (simulated "
                   "example).",
        "more": "A new file clears everything computed from the previous one - the selection, the clusters, the "
                "analysis and its windows - so that nothing of the old file is saved under the new file's name. The "
                "folder of the last file opened is remembered for the next dialog.",
    }),
    _step("load_channel2", "pushButton_browsefile_2", "files", es={
        "title": "Cargar el canal 2 (opcional)",
        "what": "Una segunda proteína medida en el mismo campo (por ejemplo, aducina).",
        "why": "El análisis por axón se hace siempre sobre el canal 1. El canal 2 sirve para el panel Two "
               "channels, que compara las dos proteínas.",
        "read": "Se dibuja en bermellón sobre el canal 1 (celeste) y se recorta con la misma selección.",
        "example": "El axón de ejemplo trae un solo canal: el canal 2 queda vacío (ejemplo simulado).",
        "more": "El canal 2 tiene su propio eps y su propio min samples: nunca se copian del canal 1, porque las dos "
                "proteínas casi nunca están muestreadas igual (una puede dejar muchas más localizaciones por "
                "molécula que la otra).",
    }, en={
        "title": "Load channel 2 (optional)",
        "what": "A second protein imaged in the same field (for example adducin).",
        "why": "The per-axon analysis always runs on channel 1. Channel 2 is for the Two channels panel, which "
               "compares the two proteins.",
        "read": "It is drawn in vermillion over channel 1 (sky blue) and cut with the same selection.",
        "example": "The example axon has one channel only: channel 2 stays empty (simulated example).",
        "more": "Channel 2 has its own eps and its own min samples: they are never copied from channel 1, because "
                "the two proteins are almost never sampled alike (one may leave many more localizations per "
                "molecule than the other).",
    }),
    # ------------------------------------------------------------------ measurement
    _step("pixel_size", "measurement_pixel_sizes", "measurement", registry=("measurement.pixel_size_nm",),
          method=_README, es={
              "title": "Tamaño de píxel y unidades",
              "what": "El tamaño de píxel de cada canal cargado, y de dónde salió ese número.",
              "why": "Picasso guarda x e y en píxeles de cámara; para medir en nanómetros el programa multiplica por "
                     "el tamaño de píxel. Un valor equivocado reescala todas las distancias laterales, y las áreas "
                     "con el cuadrado.",
              "read": "Al lado del número está su origen: «from Picasso YAML» o «from the metadata inside the HDF5» "
                      "(leído del archivo); cualquier otro origen (escrito a mano, recordado para la carpeta, tomado "
                      "de un archivo vecino) se muestra en el color de advertencia.",
              "example": "Con el axón de ejemplo: «Channel 1: pixel size {pixel_nm} nm (from Picasso YAML)» (ejemplo "
                         "simulado).",
              "more": "z ya viene en nanómetros y no se convierte. Si un archivo no trae tamaño de píxel, el programa "
                      "lo pregunta una sola vez por carpeta y lo recuerda; lo recordado nunca pisa lo que dice un "
                      "archivo. Conviene comparar el valor con los datos de la adquisición (cámara y aumento).",
          }, en={
              "title": "Pixel size and units",
              "what": "The pixel size of each loaded channel, and where that number came from.",
              "why": "Picasso stores x and y in camera pixels; to measure in nanometres the program multiplies by "
                     "the pixel size. A wrong value rescales every lateral distance, and every area with its "
                     "square.",
              "read": "Beside the number is its source: \"from Picasso YAML\" or \"from the metadata inside the "
                      "HDF5\" (read from the file); any other source (typed by hand, remembered for the folder, "
                      "taken from a file beside it) is shown in the warning colour.",
              "example": "With the example axon: \"Channel 1: pixel size {pixel_nm} nm (from Picasso YAML)\" "
                         "(simulated example).",
              "more": "z is already in nanometres and is not converted. When a file carries no pixel size, the "
                      "program asks once per folder and remembers it; a remembered value never overrides what a "
                      "file says. Check the value against the acquisition (camera and magnification).",
          }),
    _step("z_calibration", "measurement_z_calibration", "measurement", registry=("measurement.z_calibration",),
          method=_README, es={
              "title": "Calibración en z",
              "what": "El registro de calibración axial (astigmatismo) con el que se calculó z en esta medición.",
              "why": "Dice con qué calibración se obtuvo z. Se lee y se muestra; nunca se usa para cambiar un z.",
              "read": "«- (none read)» quiere decir que no se encontró registro: z se usa tal como lo ajustó el "
                      "programa de localización. Con datos 3D, los botones de abajo permiten elegir un archivo de "
                      "calibración (para la carpeta o para un solo archivo) o una pila de perlas.",
              "example": "El axón de ejemplo no trae registro: el panel dice «z calibration: - (none read)» (ejemplo "
                         "simulado).",
              "more": "Picasso ya multiplicó cada z por el factor de magnificación del registro (Huang et al. 2008); "
                      "por eso el programa no lo vuelve a aplicar. Sólo el control de cobertura axial de Data "
                      "quality lee el rango calibrado del registro.",
          }, en={
              "title": "z calibration",
              "what": "The axial (astigmatism) calibration record z was computed with in this measurement.",
              "why": "It says which calibration z came from. It is read and shown; it is never used to change a z.",
              "read": "\"- (none read)\" means no record was found: z is used as the localization software fitted it. "
                      "With 3D data, the buttons under it choose a calibration file (for the folder or for one "
                      "file) or a bead stack.",
              "example": "The example axon carries no record: the panel says \"z calibration: - (none read)\" "
                         "(simulated example).",
              "more": "Picasso already multiplied every z by the record's magnification factor (Huang et al. 2008), "
                      "which is why the program does not apply it again. Only Data quality's axial coverage check "
                      "reads the record's calibrated range.",
          }),
    # ------------------------------------------------------------------ the view and the selection
    _step("overview", "verticalLayoutWidget", "view", es={
        "title": "La vista de arriba",
        "what": "Todas las localizaciones del canal 1 vistas desde arriba (x, y), con la forma de selección encima.",
        "why": "Es donde se elige qué axón analizar: la forma se mueve y se agranda o achica hasta contener un axón.",
        "read": "Cada punto es una localización: celeste el canal 1, bermellón el canal 2. En un axón cortado "
                "transversalmente, el esqueleto de espectrina se ve como un contorno cerrado.",
        "example": "En el ejemplo, las {n_locs} localizaciones forman un contorno cerrado y el círculo las abarca a "
                   "todas (ejemplo simulado).",
        "more": "Al soltar la forma después de arrastrarla, la selección se vuelve a aplicar sola. La rueda del mouse "
                "acerca y aleja; clic derecho → «View All» vuelve a mostrar todo.",
    }, en={
        "title": "The view from above",
        "what": "Every channel-1 localization seen from above (x, y), with the selection shape over them.",
        "why": "This is where you choose which axon to analyse: move the shape and resize it until it holds one "
               "axon.",
        "read": "Each point is a localization: sky blue for channel 1, vermillion for channel 2. In a "
                "cross-sectioned axon the spectrin skeleton shows as a closed contour.",
        "example": "In the example, the {n_locs} localizations make a closed contour and the circle holds them all "
                   "(simulated example).",
        "more": "When you release the shape after dragging it, the selection is applied again by itself. The mouse "
                "wheel zooms; right-click → \"View All\" shows everything again.",
    }),
    _step("roi_shape", "radioButton_circROI", "view", es={
        "title": "Forma de la selección",
        "what": "Círculo, rectángulo o polígono: la forma que decide qué localizaciones entran (en x, y).",
        "why": "El círculo es lo habitual para un axón cortado transversalmente; el polígono sirve para un axón que "
               "no es redondo o para dejar afuera a un vecino.",
        "read": "Círculo y rectángulo: se arrastra el medio para moverlos y el manejador para cambiar el tamaño. "
                "Polígono: se arrastra un vértice, un clic en un lado agrega uno, clic derecho sobre un vértice → "
                "«Remove handle» lo saca.",
        "example": "En el ejemplo, el círculo que se pone al cargar abarca las {n_locs} localizaciones (ejemplo "
                   "simulado).",
        "more": "Cambiar de forma vuelve a dibujar la vista con la forma nueva. El polígono arranca con cuatro lados "
                "y nunca baja de tres.",
    }, en={
        "title": "Selection shape",
        "what": "Circle, rectangle or polygon: the shape that decides which localizations are in (in x, y).",
        "why": "The circle is the usual choice for a cross-sectioned axon; the polygon is for an axon that is not "
               "round, or to leave a neighbour out.",
        "read": "Circle and rectangle: drag the middle to move them and the handle to resize. Polygon: drag a "
                "vertex, click a side to add one, right-click a vertex → \"Remove handle\" to take it out.",
        "example": "In the example, the circle put up at loading holds the {n_locs} localizations (simulated "
                   "example).",
        "more": "Changing the shape redraws the view with the new one. The polygon starts with four sides and never "
                "goes below three.",
    }),
    _step("z_histogram", "verticalLayoutWidget_2", "view", es={
        "title": "Distribución en z",
        "what": "El histograma de z (la profundidad) de todas las localizaciones del canal 1.",
        "why": "El análisis no usa todas: usa una franja de z («slab») alrededor del pico principal.",
        "read": "En un corte transversal, z corre a lo largo del axón: picos separados con regularidad suelen ser "
                "anillos sucesivos del esqueleto.",
        "example": "En el ejemplo, z va de {z_lo} a {z_hi} nm (ejemplo simulado).",
        "more": "El histograma de abajo es el del canal 2. La mezcla gaussiana ajustada a z y la franja elegida se "
                "ven en la pestaña Axial de la ventana de análisis MPS.",
    }, en={
        "title": "The z distribution",
        "what": "The histogram of z (depth) of every channel-1 localization.",
        "why": "The analysis does not use all of them: it uses an axial slab around the main peak.",
        "read": "In a cross-section, z runs along the axon: regularly spaced peaks are usually successive rings of "
                "the skeleton.",
        "example": "In the example, z runs from {z_lo} to {z_hi} nm (simulated example).",
        "more": "The histogram under it is channel 2's. The Gaussian mixture fitted to z and the chosen slab are on "
                "the Axial tab of the MPS analysis window.",
    }),
    _step("apply_roi", "pushButton_zrange", "view", registry=("slab.half_width_nm", "slab.centre"), es={
        "title": "Aplicar la selección y el corte en z",
        "what": "«Apply ROI» aplica la forma dibujada y el corte axial (Z min - Z max) a la selección.",
        "why": "El análisis usa sólo las localizaciones dentro de la forma y dentro de la franja de z. Por defecto "
               "la franja es el pico principal ± {half_width} ({slab_total} en total), la ventana publicada en "
               "{paper}.",
        "read": "Z min y Z max son de sólo lectura acá: muestran la franja elegida en la ventana de análisis "
                "(automática, un componente elegido o un rango tipeado allí).",
        "example": "En el ejemplo, la franja va de {zmin} a {zmax} nm, centrada en {peak} nm, y deja {n_slab} de las "
                   "{n_locs} localizaciones (ejemplo simulado).",
        "more": "La franja se centra en la moda de la mezcla gaussiana ajustada a z. Cuando dos picos tienen pesos "
                "parecidos la ventana de análisis lo avisa, porque otra franja estaría casi igual de justificada: "
                "en el ejemplo simulado aparece justamente ese aviso («Ambiguous main peak»).",
    }, en={
        "title": "Apply the selection and the axial cut",
        "what": "\"Apply ROI\" applies the drawn shape and the axial cut (Z min - Z max) to the selection.",
        "why": "The analysis uses only the localizations inside the shape and inside the axial slab. By default "
               "the slab is the main peak ± {half_width} ({slab_total} in all), the window published in {paper}.",
        "read": "Z min and Z max are read-only here: they show the slab chosen in the analysis window (automatic, a "
                "chosen component, or a range typed there).",
        "example": "In the example, the slab runs from {zmin} to {zmax} nm, centred on {peak} nm, and keeps {n_slab} "
                   "of the {n_locs} localizations (simulated example).",
        "more": "The slab is centred on the mode of the Gaussian mixture fitted to z. When two peaks have similar "
                "weights the analysis window says so, because another slab would be almost as well supported: the "
                "simulated example shows exactly that warning (\"Ambiguous main peak\").",
    }),
    # ------------------------------------------------------------------ clustering
    _step("algorithm", "comboBox_algorithm", "clustering", registry=("window.clustering",), es={
        "title": "Algoritmo de agrupamiento",
        "what": "Cómo agrupa esta ventana las localizaciones en clusters: DBSCAN (por defecto), HDBSCAN o Auto.",
        "why": "DBSCAN usa eps y min samples; HDBSCAN usa «Min Cluster Size» y encuentra clusters de densidades "
               "distintas.",
        "read": "Auto elige DBSCAN por debajo de 100.000 localizaciones y HDBSCAN por encima: es una decisión de "
                "velocidad, no sobre los datos.",
        "example": "El ejemplo tiene {n_locs} localizaciones: Auto elegiría DBSCAN (ejemplo simulado).",
        "more": "El análisis por axón (la ventana de análisis MPS) usa siempre DBSCAN con eps y min samples, elija lo "
                "que elija esta lista: la lista cambia sólo el agrupamiento que se dibuja en esta ventana. Su "
                "etiqueta en el registro es «ad hoc»: una versión futura quita el cambio automático a HDBSCAN.",
    }, en={
        "title": "Clustering algorithm",
        "what": "How this window groups the localizations into clusters: DBSCAN (the default), HDBSCAN or Auto.",
        "why": "DBSCAN uses eps and min samples; HDBSCAN uses \"Min Cluster Size\" and finds clusters of differing "
               "density.",
        "read": "Auto picks DBSCAN below 100,000 localizations and HDBSCAN above: a choice about speed, not about "
                "the data.",
        "example": "The example holds {n_locs} localizations: Auto would pick DBSCAN (simulated example).",
        "more": "The per-axon analysis (the MPS analysis window) always uses DBSCAN with eps and min samples, "
                "whatever this list says: the list only changes the clustering drawn in this window. Its registry "
                "badge is \"ad hoc\": a later version removes the automatic switch to HDBSCAN.",
    }),
    _step("dbscan_eps", "lineEdit_eps", "clustering", registry=("dbscan.eps_nm",), method=_README, es={
        "title": "eps de DBSCAN",
        "what": "La distancia máxima, en nanómetros, para que dos localizaciones cuenten como vecinas.",
        "why": "DBSCAN junta localizaciones vecinas en clusters; eps fija qué quiere decir «cerca».",
        "read": "Más chico parte un cluster en varios; más grande junta clusters vecinos. Por defecto "
                "{eps_default}, el valor publicado en {paper}. Se edita acá o en la ventana de análisis: es un "
                "único valor.",
        "example": "Con eps = {eps_default} y min samples = {min_default}, la franja del ejemplo da {n_clusters} "
                   "clusters (ejemplo simulado).",
        "more": "En términos físicos, eps tiene que ser mayor que la precisión de localización, para que las varias "
                "detecciones de una misma molécula queden juntas, y menor que la distancia entre parches vecinos, "
                "para no fundirlos. La etiqueta de abajo dice de dónde viene el valor en uso. El canal 2 tiene su "
                "propio eps. HDBSCAN no lo usa.",
    }, en={
        "title": "DBSCAN eps",
        "what": "The largest distance, in nanometres, at which two localizations still count as neighbours.",
        "why": "DBSCAN joins neighbouring localizations into clusters; eps sets what \"near\" means.",
        "read": "Smaller breaks one cluster into several; larger merges neighbouring clusters. Default "
                "{eps_default}, the value published in {paper}. Edited here or in the analysis window: one value.",
        "example": "With eps = {eps_default} and min samples = {min_default}, the example's slab gives {n_clusters} "
                   "clusters (simulated example).",
        "more": "Physically, eps has to be larger than the localization precision, so that the several detections "
                "of one molecule stay together, and smaller than the distance between neighbouring patches, so as "
                "not to merge them. The badge under it says where the value in use comes from. Channel 2 has its "
                "own eps. HDBSCAN does not use it.",
    }),
    _step("dbscan_min_samples", "lineEdit_minsamples", "clustering", registry=("dbscan.min_samples",),
          method=_README, es={
              "title": "min samples de DBSCAN",
              "what": "Cuántas localizaciones tienen que estar a menos de eps de un punto para que ese punto inicie "
                      "un cluster.",
              "why": "Separa clusters de ruido: lo que no llega a ese número queda como ruido (gris, con una x).",
              "read": "Subirlo deja más localizaciones como ruido; bajarlo deja pasar grupos más chicos. Por defecto "
                      "{min_default} ({paper}); rango documentado {min_range}.",
              "example": "En el ejemplo, con {min_default}, se forman {n_clusters} clusters y {n_noise} "
                         "localizaciones de la franja quedan como ruido (ejemplo simulado).",
              "more": "Un fluoróforo se enciende y apaga varias veces, así que una misma molécula deja varias "
                      "localizaciones; según el registro, el valor publicado se eligió acorde a los ciclos de "
                      "encendido medios del fluoróforo (Dempsey et al. 2011). Como cuenta localizaciones, depende de "
                      "cuántas veces se detectó cada molécula: por eso el canal 2 lleva el suyo.",
          }, en={
              "title": "DBSCAN min samples",
              "what": "How many localizations have to lie within eps of a point for that point to start a cluster.",
              "why": "It separates clusters from noise: whatever does not reach that number is noise (grey, an x).",
              "read": "Raising it leaves more localizations as noise; lowering it lets smaller groups through. "
                      "Default {min_default} ({paper}); documented range {min_range}.",
              "example": "In the example, with {min_default}, {n_clusters} clusters form and {n_noise} "
                         "localizations of the slab are left as noise (simulated example).",
              "more": "A fluorophore switches on and off several times, so one molecule leaves several "
                      "localizations; according to the registry, the published value was matched to the mean "
                      "switching cycles of the fluorophore (Dempsey et al. 2011). Since it counts localizations, it "
                      "depends on how often each molecule was detected: that is why channel 2 has its own.",
          }),
    _step("origin_badge", "badge_dbscan_eps_nm", "clustering", registry=("dbscan.eps_nm",), badges=True,
          method=_README, es={
              "title": "Etiqueta de origen",
              "what": "La etiqueta debajo de cada parámetro: de dónde sale el valor que se está usando.",
              "why": "Para saber si un número viene de una publicación, de una fórmula escrita, de una simulación "
                     "propia, de datos piloto, o si lo cambiaste vos.",
              "read": "paper (se muestra como la publicación), derived, simulation, pilot suggestion, user (lo "
                      "cambiaste vos), blank (sin valor). «ad hoc» y «unreviewed» marcan valores que una versión "
                      "futura revisa. Su ayuda emergente da la fuente y el rango documentado.",
              "example": "Con el axón de ejemplo y los valores por defecto, eps muestra «Gazal et al. (2026)», la "
                         "forma corta de «paper»; si escribís otro valor pasa a «user» (ejemplo simulado).",
              "more": "Un valor fuera de su rango documentado se marca, nunca se corrige solo. Las definiciones "
                      "exactas de cada etiqueta están abajo, tal como las escribe el registro.",
          }, en={
              "title": "Origin badge",
              "what": "The badge under each parameter: where the value in use comes from.",
              "why": "To know whether a number comes from a publication, a written formula, this project's own "
                     "simulation, pilot data, or from you.",
              "read": "paper (shown as the publication), derived, simulation, pilot suggestion, user (you changed "
                      "it), blank (no value). \"ad hoc\" and \"unreviewed\" mark values a later version revises. Its "
                      "tooltip gives the source and the documented range.",
              "example": "With the example axon and the defaults, eps shows \"Gazal et al. (2026)\", the short form "
                         "of \"paper\"; type another value and it becomes \"user\" (simulated example).",
              "more": "A value outside its documented range is flagged, never corrected by itself. The exact "
                      "definition of each badge is below, as the registry writes it.",
          }),
    _step("cluster_ch1", "pushButton_clusterch1", "clustering", registry=("curation.edge_criterion",), es={
        "title": "Clusterizar el canal 1 y analizar",
        "what": "Agrupa la selección del canal 1 con DBSCAN y corre el análisis completo del axón.",
        "why": "De este botón sale todo lo demás: el contorno (perímetro), la ocupancia, la aleatorización y lo "
               "que se exporta.",
        "read": "Al terminar se abre la ventana de análisis MPS. El gráfico de abajo muestra los clusters del canal 1; "
                "el ruido va en gris.",
        "example": "En el ejemplo: {n_clusters} clusters, {n_kept} conservados después de la curación automática "
                   "(ejemplo simulado).",
        "more": "La curación automática saca los clusters que quedan a menos de eps del borde de la selección (los "
                "corta la forma dibujada); si eso sacara más de la mitad, no se aplica. Ese resguardo todavía no "
                "tiene una derivación escrita: su etiqueta es «ad hoc».",
    }, en={
        "title": "Cluster channel 1 and analyse",
        "what": "Clusters the channel-1 selection with DBSCAN and runs the whole per-axon analysis.",
        "why": "Everything else comes from this button: the contour (perimeter), the occupancy, the randomization "
               "and what is exported.",
        "read": "When it finishes, the MPS analysis window opens. The plot under it shows channel 1's clusters; "
                "noise is grey.",
        "example": "In the example: {n_clusters} clusters, {n_kept} kept after the automatic curation (simulated "
                   "example).",
        "more": "The automatic curation removes the clusters within eps of the selection's edge (the drawn shape "
                "cuts them); if that would remove more than half of them it is not applied. That safeguard has no "
                "written derivation yet: its badge is \"ad hoc\".",
    }),
    _step("channel2_parameters", "lineEdit_eps_2", "clustering",
          registry=("channel2.eps_nm", "channel2.min_samples"), es={
              "title": "Parámetros del canal 2",
              "what": "El eps y el min samples propios del canal 2.",
              "why": "Las dos proteínas casi nunca están muestreadas igual, así que los parámetros del canal 1 no "
                     "sirven para el canal 2.",
              "read": "«auto» quiere decir que nadie eligió todavía: «cluster Ch2» estima uno con los puntos del "
                      "canal 2, y el panel Two channels pregunta antes de comparar. Lo elegido se recuerda por "
                      "carpeta.",
              "example": "El axón de ejemplo no tiene canal 2: los dos campos siguen en «auto» (ejemplo simulado).",
              "more": "Los clusters de «cluster Ch2» son para mirar: el panel Two channels vuelve a agrupar el canal 2 "
                      "con lo que digan estos dos campos.",
          }, en={
              "title": "Channel 2 parameters",
              "what": "Channel 2's own eps and min samples.",
              "why": "The two proteins are almost never sampled alike, so channel 1's parameters do not fit "
                     "channel 2.",
              "read": "\"auto\" means nobody has chosen one yet: \"cluster Ch2\" estimates one from channel 2's own "
                      "points, and the Two channels panel asks before it compares. What you choose is remembered "
                      "per folder.",
              "example": "The example axon has no channel 2: both fields stay on \"auto\" (simulated example).",
              "more": "The clusters of \"cluster Ch2\" are for looking at: the Two channels panel clusters channel 2 "
                      "again with whatever these two fields say.",
          }),
    _step("mps_analysis", "pushButton_remove_bad_cluster", "clustering", es={
        "title": "La ventana de análisis MPS",
        "what": "Abre la ventana de análisis de la selección actual, y corre el análisis si todavía no hay uno para "
                "esta selección.",
        "why": "Es la entrada a los resultados del axón: mapa del axón, perímetro, ocupancia, vecinos más "
               "cercanos y distribución axial.",
        "read": "Si el análisis en pantalla ya es de esta selección, sólo abre la ventana; sin selección, la abre "
                "vacía.",
        "example": "Con el ejemplo, la ventana muestra los {n_kept} clusters sobre el contorno del axón (ejemplo "
                   "simulado).",
        "more": "La ventana de análisis va a tener su propio recorrido en una próxima versión: sus filas cambian con "
                "los cambios científicos pendientes. Mientras tanto, «Explicame esto» (Shift+F1) sobre un control de "
                "esa ventana muestra su ayuda emergente.",
    }, en={
        "title": "The MPS analysis window",
        "what": "Opens the analysis window of the current selection, and runs the analysis when there is none yet "
                "for this selection.",
        "why": "It is the way into the axon's results: the axon map, the perimeter, the occupancy, the nearest "
               "neighbours and the axial distribution.",
        "read": "When the analysis on screen is already of this selection it only opens the window; with no "
                "selection it opens it empty.",
        "example": "With the example, the window shows the {n_kept} clusters on the axon's contour (simulated "
                   "example).",
        "more": "The analysis window will get its own tour in a later version: its rows change with the pending "
                "science changes. Meanwhile, \"Explain this\" (Shift+F1) on a control of that window shows its "
                "tooltip.",
    }),
    # ------------------------------------------------------------------ the toolbar
    _step("export_axon", "toolButton_export_axon", "toolbar", es={
        "title": "Exportar el axón",
        "what": "Escribe este axón en las tablas: una fila con los parámetros medidos (y con el descarte aplicado) y, "
                "si lo pedís, una fila por cluster y una por localización.",
        "why": "Son las tablas que después se juntan entre axones y entre animales.",
        "read": "Antes de exportar pide la identidad del axón (especie, genotipo, proteína, animal, portaobjetos, "
                "ROI, axón), propuesta a partir de la ruta del archivo. Al lado de las tablas se escribe un "
                "diccionario de columnas.",
        "example": "Con el axón de ejemplo, la tabla del axón recibe una sola fila: un axón, una fila (ejemplo "
                   "simulado).",
        "more": "Todo se escribe del estado en pantalla y de una sola vez. Lo que la ruta no dice queda vacío hasta "
                "que lo escribas; nada se inventa. «Copy a table for Excel» hace una copia que Excel abre bien con "
                "coma decimal, sin tocar la tabla original.",
    }, en={
        "title": "Export the axon",
        "what": "Writes this axon to the tables: one row with the measured parameters (and with the discard "
                "applied) and, if you ask, one row per cluster and one per localization.",
        "why": "These are the tables that are later pooled across axons and animals.",
        "read": "Before exporting it asks for the axon's identity (species, genotype, protein, animal, slide, ROI, "
                "axon), proposed from the file's path. A dictionary of the columns is written beside the tables.",
        "example": "With the example axon, the axon table gets one row: one axon, one row (simulated example).",
        "more": "Everything is written from the state on screen, in one go. What the path does not say stays empty "
                "until you type it; nothing is made up. \"Copy a table for Excel\" writes a copy Excel opens "
                "correctly where the decimal mark is a comma, without touching the table itself.",
    }),
    _step("data_quality", "toolButton_quality", "toolbar", es={
        "title": "Calidad de los datos",
        "what": "Controles de la adquisición, no del axón: precisión NeNA, caja de ajuste, resolución en z y deriva "
                "residual.",
        "why": "Para saber si los datos pueden responder la pregunta: si lo que se ve es estructura o es el límite "
               "del microscopio.",
        "read": "Cada hallazgo lleva una marca: «ok» está bien, «!» hay que mirarlo, «x» falló. NeNA mide la "
                "precisión con los propios datos (la misma molécula vista en dos cuadros seguidos); si coincide con "
                "la que informó el ajuste, esa precisión es creíble.",
        "example": "En el ejemplo, NeNA = {nena} nm frente a {lp_reported} nm informados (cociente {nena_ratio}): "
                   "«reported precisions are consistent with NeNA» (ejemplo simulado).",
        "more": "NeNA es el método de Endesfelder et al. (2014). Estos números no cambian con los parámetros de "
                "agrupamiento, por eso están fuera de la ventana de resultados. El ejemplo simulado también muestra "
                "avisos esperables: no trae «Box Size» en los metadatos, así que la caja de ajuste no se puede "
                "comparar.",
    }, en={
        "title": "Data quality",
        "what": "Checks of the acquisition, not of the axon: NeNA precision, fitting box, axial resolvedness and "
                "residual drift.",
        "why": "To know whether the data can answer the question: whether what you see is structure or the "
               "microscope's limit.",
        "read": "Each finding carries a mark: \"ok\" passed, \"!\" needs a look, \"x\" failed. NeNA measures the "
                "precision from the data itself (the same molecule seen in two consecutive frames); when it agrees "
                "with the precision the fit reported, that precision can be believed.",
        "example": "In the example, NeNA = {nena} nm against {lp_reported} nm reported (ratio {nena_ratio}): "
                   "\"reported precisions are consistent with NeNA\" (simulated example).",
        "more": "NeNA is the method of Endesfelder et al. (2014). These numbers do not change with the clustering "
                "parameters, which is why they are outside the results window. The simulated example also shows "
                "expected warnings: it carries no \"Box Size\" in its metadata, so the fitting box cannot be "
                "checked.",
    }),
    _step("two_channels", "toolButton_two_channels", "toolbar", es={
        "title": "Dos canales",
        "what": "Registra el canal 2 sobre el canal 1 (con marcadores o una calibración) y después los compara: "
                "fase axial y disposición transversal.",
        "why": "Para la pregunta de co-distribución: ¿la segunda proteína está en los mismos lugares del esqueleto "
               "que la espectrina?",
        "read": "Usa los parámetros propios del canal 2; si están en «auto», pregunta antes de comparar.",
        "example": "El axón de ejemplo tiene un solo canal, así que acá no hay nada que comparar (ejemplo simulado).",
        "more": "Registrar quiere decir llevar los dos canales al mismo sistema de coordenadas antes de medir "
                "distancias entre ellos.",
    }, en={
        "title": "Two channels",
        "what": "Registers channel 2 onto channel 1 (markers or a calibration), then compares them: axial phase and "
                "transverse arrangement.",
        "why": "For the co-distribution question: is the second protein where the spectrin skeleton is?",
        "read": "It uses channel 2's own parameters; when they are \"auto\" it asks before it compares.",
        "example": "The example axon has one channel, so there is nothing to compare here (simulated example).",
        "more": "Registering means bringing the two channels into one coordinate system before measuring distances "
                "between them.",
    }),
    _step("axoplasm", "toolButton_axoplasm", "toolbar", es={
        "title": "Axoplasma",
        "what": "Clasifica las localizaciones de βII-espectrina seleccionadas en membrana o interior, con una imagen "
                "widefield de βIII-tubulina.",
        "why": "Para separar los clusters de la membrana de los que quedan dentro del axón, y ver cuánto cambian los "
               "resultados sin ellos.",
        "read": "Los clusters que las imágenes ponen adentro se pueden descartar; la exportación guarda la fila "
                "medida y la fila con el descarte.",
        "example": "El axón de ejemplo no trae imágenes widefield: para clasificar hace falta cargar una de tubulina "
                   "(ejemplo simulado).",
        "more": "La imagen de tubulina marca el interior del axón; una de espectrina en widefield sirve para medir el "
                "corrimiento entre la imagen y las localizaciones.",
    }, en={
        "title": "Axoplasm",
        "what": "Classifies the selected betaII-spectrin localizations as membrane or interior with a widefield "
                "betaIII-tubulin image.",
        "why": "To tell the membrane clusters from the ones inside the axon, and to see how much the results change "
               "without them.",
        "read": "The clusters the images put inside can be discarded; the export keeps the measured row and the row "
                "with the discard.",
        "example": "The example axon has no widefield images: classifying needs a tubulin one loaded (simulated "
                   "example).",
        "more": "The tubulin image marks the inside of the axon; a widefield spectrin image measures the shift "
                "between the image and the localizations.",
    }),
    _step("paint", "toolButton_paint", "toolbar", es={
        "title": "DNA-PAINT",
        "what": "Une localizaciones en eventos de unión, rechaza el pegado inespecífico y mide la cinética de unión; "
                "el conteo qPAINT es una pestaña.",
        "why": "En DNA-PAINT una misma molécula aparece muchas veces, y eso cambia lo que significan los demás "
               "números del programa.",
        "read": "Un conteo qPAINT sin calibrar nunca se informa como número de moléculas.",
        "example": "El axón de ejemplo no es un archivo de DNA-PAINT: este panel no aplica (ejemplo simulado).",
        "more": "La unión de localizaciones usa un radio derivado de la precisión de cada archivo.",
    }, en={
        "title": "DNA-PAINT",
        "what": "Links localizations into binding events, rejects non-specific sticking and measures binding "
                "kinetics; qPAINT counting is a tab.",
        "why": "In DNA-PAINT one molecule appears many times, which changes what the program's other numbers mean.",
        "read": "An uncalibrated qPAINT count is never reported as a number of molecules.",
        "example": "The example axon is not a DNA-PAINT file: this panel does not apply (simulated example).",
        "more": "Linking uses a radius derived from each file's precision.",
    }),
    _step("batch", "toolButton_batch", "toolbar", es={
        "title": "Lote (Batch)",
        "what": "Analiza de una vez todos los axones pickeados de una carpeta, con los parámetros de esta ventana, y "
                "los escribe en la misma tabla que «Save axon».",
        "why": "Para procesar una adquisición entera con los mismos parámetros sin repetir todo a mano.",
        "read": "Su segunda pestaña resume una columna por grupo, por axón y por animal, sin calcular un p-valor.",
        "example": "El axón de ejemplo está en la carpeta examples/demo_axon; un lote de esa carpeta tendría un solo "
                   "axón (ejemplo simulado).",
        "more": "Los parámetros del lote son los de esta ventana en el momento de correrlo; la tabla dice con qué "
                "parámetros se analizó cada axón.",
    }, en={
        "title": "Batch",
        "what": "Analyses every picked axon under a folder at once, with this window's settings, and writes them to "
                "the same table \"Save axon\" writes.",
        "why": "To process a whole acquisition with the same parameters without doing it all by hand.",
        "read": "Its second tab summarises one column per group, per axon and per animal, without computing a "
                "p-value.",
        "example": "The example axon is in the folder examples/demo_axon; a batch of that folder would hold one axon "
                   "(simulated example).",
        "more": "The batch uses this window's parameters at the moment it runs; the table says which parameters each "
                "axon was analysed with.",
    }),
    _step("help", "toolButton_help", "toolbar", method=_GUIDE, es={
        "title": "Ayuda",
        "what": "El menú de ayuda: modo principiante, este recorrido, «Explicame esto», el axón de ejemplo y el "
                "idioma de las explicaciones.",
        "why": "Para volver a cualquier explicación cuando la necesites.",
        "read": "Shift+F1 activa «Explicame esto»: el próximo clic sobre un control muestra su tarjeta y no lo "
                "ejecuta. Los botones «?» de cada panel repiten sólo su parte.",
        "example": "Con el axón de ejemplo (simulado) abierto, apretá Shift+F1 y hacé clic en el campo de eps: "
                   "aparece su tarjeta y no se cambia nada.",
        "more": "El recorrido de cada ventana se muestra solo la primera vez que se abre, mientras el modo "
                "principiante esté prendido. Prenderlo de nuevo desde este menú vuelve a mostrar los recorridos.",
    }, en={
        "title": "Help",
        "what": "The help menu: beginner mode, this tour, \"Explain this\", the example axon and the language of the "
                "explanations.",
        "why": "To come back to any explanation when you need it.",
        "read": "Shift+F1 turns on \"Explain this\": the next click on a control shows its card and does not run it. "
                "The \"?\" button of each panel replays only its part.",
        "example": "With the example axon (simulated) open, press Shift+F1 and click the eps field: its card "
                   "appears and nothing changes.",
        "more": "Each window's tour shows by itself the first time the window opens, while beginner mode is on. "
                "Turning it on again from this menu shows the tours again.",
    }),
]

# --------------------------------------------------------------------------- the perimeter density window (P12)
_PD_DOC = "docs/perimeter_density.md"
PERIMETER_STEPS: List[Dict[str, Any]] = [
    _step("pd_welcome", "", "", method=_PD_DOC, es={
        "title": "El análisis de densidad sobre el perímetro",
        "what": "Mide cómo se reparte la βII-espectrina a lo largo de la membrana, sin clusters: tres rodajas en z, "
                "un contorno automático y el perfil de densidad sobre ese contorno.",
        "why": "Responde tres preguntas: si la densidad se amontona o se aleja del máximo (A), si sigue la curvatura "
               "de la membrana (B) y si las rodajas vecinas comparten el patrón (C).",
        "read": "Run analiza el axón que muestra la ventana del análisis MPS; la barra muestra el paso y Cancel lo "
                "frena en el paso siguiente. Save... guarda las tablas con el nombre que elijas.",
        "example": "Con el axón de ejemplo (simulado) abierto, apretá Run: en pocos segundos se llenan las tres "
                   "pestañas.",
        "more": "Cada número tiene su origen en docs/perimeter_density.md. Un axón solo no permite comparar "
                "animales: la lectura de grupo sale del Batch (casilla Perimeter density).",
    }, en={
        "title": "The perimeter density analysis",
        "what": "Measures how βII-spectrin is laid out along the membrane, without clusters: three z slices, an "
                "automatic contour and the density profile along that contour.",
        "why": "It answers three questions: does density pile up near or keep away from its maximum (A), does it "
               "follow the curvature of the membrane (B), and do neighbouring slices share the pattern (C).",
        "read": "Run analyses the axon the MPS analysis window shows; the bar shows the step and Cancel stops it at "
                "the next one. Save... writes the tables under the name you choose.",
        "example": "With the example axon (simulated) open, press Run: the three tabs fill in a few seconds.",
        "more": "Where every number comes from: docs/perimeter_density.md. One axon cannot compare animals: the "
                "group reading comes from the Batch (box Perimeter density).",
    }),
    _step("pd_z0", "perimeter_centre", "", method=_PD_DOC, es={
        "title": "Dónde se centra la rodaja central (z0)",
        "what": "Por defecto, el pico de la curva de densidad en z; también se puede usar el foco nominal (el del "
                "registro de calibración) o el empírico (donde sx = sy).",
        "why": "Las tres rodajas de 170 nm se cortan alrededor de z0; elegir otro centro cambia qué localizaciones "
               "caen en cada una.",
        "read": "El gráfico de la curva de densidad marca el pico (línea llena), los dos focos (rayada y punteada) "
                "y las tres rodajas sombreadas. La tabla dice la distancia pico - foco.",
        "example": "En un archivo simulado sin registro de calibración, «Focus, nominal» se detiene con «no focus» "
                   "y lo dice: nunca vuelve al pico en silencio.",
        "more": "Los porcentajes de la tabla de rodajas son las localizaciones de cada rodaja lateral respecto de la "
                "central (100 %).",
    }, en={
        "title": "Where the central slice is centred (z0)",
        "what": "By default, the peak of the z density curve; the nominal focus (from the calibration record) or the "
                "empirical one (where sx = sy) can be used instead.",
        "why": "The three 170 nm slices are cut around z0; another centre changes which localizations fall in each.",
        "read": "The density curve plot marks the peak (solid line), both foci (dashed and dotted) and the three "
                "shaded slices. The table gives the peak - focus distance.",
        "example": "On a simulated file without a calibration record, \"Focus, nominal\" stops with \"no focus\" and "
                   "says so: it never falls back to the peak in silence.",
        "more": "The percentages of the slices table are each lateral slice's localizations relative to the central "
                "one (100 %).",
    }),
    _step("pd_contour", "perimeter_contour_state", "", method=_PD_DOC, es={
        "title": "El estado del contorno",
        "what": "Si el contorno automático es confiable o no, con el motivo que da su compuerta de calidad.",
        "why": "Todo el perfil se mide sobre ese contorno: un contorno no confiable queda en la tabla por axón con "
               "su motivo, pero fuera de los análisis agrupados.",
        "read": "Verde y «ok»: confiable. Rojo y «x»: rechazado o no confiable; en el gráfico el contorno se dibuja "
                "rojo y rayado. La línea de abajo da largo, residuo y arco sin datos.",
        "example": "En un axón simulado con una banda ancha de localizaciones, el contorno puede salir «not "
                   "reliable» con el motivo de la compuerta.",
        "more": "La ventana no juzga el contorno: muestra lo que dice el módulo del contorno automático.",
    }, en={
        "title": "The state of the contour",
        "what": "Whether the automatic contour is reliable, with the reason its quality gate gives.",
        "why": "The whole profile is measured on that contour: an unreliable one stays in the per-axon table with "
               "its reason, but out of the pooled analyses.",
        "read": "Green with \"ok\": reliable. Red with \"x\": rejected or not reliable; the plot then draws the "
                "contour red and dashed. The line below gives length, residual and arc without data.",
        "example": "On a simulated axon with a wide band of localizations the contour can come out \"not reliable\" "
                   "with the gate's reason.",
        "more": "The window does not judge the contour: it shows what the automatic contour module says.",
    }),
    _step("pd_questions", "perimeter_tabs", "", method=_PD_DOC, es={
        "title": "Las preguntas A, B y C",
        "what": "Cada pregunta tiene su tabla, una lectura en una línea y sus avisos debajo, en gris.",
        "why": "Los avisos dicen lo que el número no puede decir: A es exploratoria y su lectura de grupo sale del "
               "perfil agrupado; en C los pares vecinos comparten la fuga axial; en B el signo se lee al revés.",
        "read": "B: Spearman NEGATIVO = más densidad donde la membrana está más curvada. C: el par z-1 / z+1 es la "
                "lectura limpia. A: «attractor» sólo cuenta si resiste la zona de exclusión.",
        "example": "En un axón simulado sin estructura, las tres lecturas dicen que nada se destaca del nulo.",
        "more": "Ningún p se calcula entre axones, animales ni genotipos en esta ventana.",
    }, en={
        "title": "Questions A, B and C",
        "what": "Each question has its table, a one-line reading and its warnings below it, in grey.",
        "why": "The warnings say what the number cannot: A is exploratory and its group call comes from the pooled "
               "profile; in C the neighbouring pairs share the axial leak; in B the sign reads backwards.",
        "read": "B: a NEGATIVE Spearman = more density where the membrane is more curved. C: the z-1 / z+1 pair is "
                "the clean reading. A: \"attractor\" counts only if it survives the exclusion zone.",
        "example": "On a simulated axon with no structure, the three readings say nothing stands out of the null.",
        "more": "No p is computed between axons, animals or genotypes in this window.",
    }),
]


# --------------------------------------------------------------------------- the perimeter window, presentation mode
# "Modo presentación": what each part of the perimeter density window asks, in plain words, and what the SIMULATED
# example axon answers (PERIMETER_DEMO_FACTS). Every number is a demo fact or a number of the validation report
# (PERIMETER_VALIDATION); nothing else is quoted.
PERIMETER_DEMO_STEPS: List[Dict[str, Any]] = [
    _step("pdd_welcome", "", "", method=_PD_DOC, headings=False, es={
        "title": "Modo presentación: densidad sobre el perímetro",
        "what": "¿Qué pregunta responde?: ¿la espectrina se reparte parejo a lo largo del anillo, o se amontona "
                "siguiendo algún patrón?",
        "why": "Explicación sencilla: se mira la membrana del axón como un anillo y se cuenta cuánta espectrina hay "
               "en cada tramo. Tres preguntas: A, ¿se amontona cerca del tramo más denso? B, ¿hay más donde la "
               "membrana se curva más? C, ¿las rodajas vecinas en z repiten el mismo patrón?",
        "read": "Cómo se lee: cada tarjeta marca una parte de la ventana y cambia sola a la pestaña que corresponde. "
                "Siguiente (→) avanza, Anterior (←) vuelve, Esc cierra.",
        "example": "La respuesta en el axón simulado: es un axón inventado por el programa (un círculo con tres "
                   "anillos), no una medición. Sirve para mostrar el camino, no para sacar conclusiones.",
        "more": "Todo lo que sigue usa los valores por defecto de la barra: z0 en el pico, banda de ±90 nm y "
                "contorno automático. El detalle de cada número está en docs/perimeter_density.md.",
    }, en={
        "title": "Presentation mode: density along the perimeter",
        "what": "What question does it answer?: is spectrin spread evenly along the ring, or does it pile up "
                "following some pattern?",
        "why": "Plain explanation: the axon's membrane is seen as a ring and the spectrin in each stretch is counted. "
               "Three questions: A, does it pile up near the densest stretch? B, is there more where the membrane "
               "bends more? C, do neighbouring z slices repeat the same pattern?",
        "read": "How to read it: each card marks one part of the window and switches by itself to the right tab. "
                "Next (→) moves on, Previous (←) goes back, Esc closes.",
        "example": "The answer on the simulated axon: an axon made up by the program (a circle with three rings), "
                   "not a measurement. It shows the path, not a conclusion.",
        "more": "Everything below uses the bar's defaults: z0 at the peak, band +-90 nm and the automatic contour. "
                "Where each number comes from: docs/perimeter_density.md.",
    }),
    _step("pdd_slices", "perimeter_z_density", "", method=_PD_DOC, headings=False, es={
        "title": "Las tres rodajas en z",
        "what": "¿Qué pregunta responde?: ¿a qué altura del axón se mira, y cuántas localizaciones entran en cada "
                "rodaja?",
        "why": "Explicación sencilla: se cortan tres rodajas de 170 nm de alto, una arriba de la otra. 170 nm es "
               "aproximadamente la distancia entre dos anillos de espectrina, así que cada rodaja junta más o menos "
               "un anillo. z0 es el centro de la rodaja del medio: la altura donde hay más localizaciones.",
        "example": "La respuesta en el axón simulado: z0 = {pd_z0} nm. Las rodajas tienen {pd_n_minus} (z−1), "
                   "{pd_n_central} (central) y {pd_n_plus} (z+1) localizaciones; las de los costados son el "
                   "{pd_pct_minus} % y el {pd_pct_plus} % de la central.",
        "read": "Cómo se lee: en la curva de densidad en z, la línea llena es el pico (z0) y las tres franjas "
                "sombreadas son las rodajas. La tabla «Slices» da los límites y los conteos.",
        "more": "Si la ventana encuentra un registro de calibración, también puede centrar en el foco; este archivo "
                "simulado no tiene, y la ventana lo dice en vez de inventarlo.",
    }, en={
        "title": "The three z slices",
        "what": "What question does it answer?: at which height of the axon do we look, and how many localizations "
                "fall in each slice?",
        "why": "Plain explanation: three slices 170 nm tall are cut, one above the other. 170 nm is about the "
               "distance between two spectrin rings, so each slice holds roughly one ring. z0 is the centre of the "
               "middle slice: the height with the most localizations.",
        "example": "The answer on the simulated axon: z0 = {pd_z0} nm. The slices hold {pd_n_minus} (z-1), "
                   "{pd_n_central} (central) and {pd_n_plus} (z+1) localizations; the side ones are "
                   "{pd_pct_minus} % and {pd_pct_plus} % of the central one.",
        "read": "How to read it: on the z density curve the solid line is the peak (z0) and the three shaded bands "
                "are the slices. The \"Slices\" table gives the limits and the counts.",
        "more": "When the window finds a calibration record it can also centre on the focus; this simulated file has "
                "none, and the window says so instead of making one up.",
    }),
    _step("pdd_contour", "perimeter_contour_state", "", method=_PD_DOC, headings=False, es={
        "title": "El contorno automático y su control de calidad",
        "what": "¿Qué pregunta responde?: ¿por dónde pasa la membrana, y ese trazado es confiable?",
        "why": "Explicación sencilla: el programa dibuja solo la línea de la membrana sobre las localizaciones. Todo "
               "lo que sigue se mide sobre esa línea: si está mal, los números también. Por eso un control de "
               "calidad la revisa, y si no pasa, el programa no sigue y dice por qué.",
        "example": "La respuesta en el axón simulado: el contorno fue RECHAZADO. La línea medía {pd_len_nm} nm, "
                   "{pd_len_ratio} veces el perímetro de la nube (límite {pd_len_limit}): serpenteaba por adentro del "
                   "anillo. El plan B (contorno polar) también falló: en un tramo quedaba a {pd_polar_off_nm} nm de "
                   "las localizaciones (límite {pd_polar_limit_nm} nm).",
        "read": "Cómo se lee: verde y «ok» = confiable. Rojo y «x» = rechazado, con el motivo escrito al lado. Acá "
                "está en rojo.",
        "more": "Es el comportamiento buscado: antes que dar números sobre una línea mala, el programa se frena. En "
                "este axón simulado la espectrina está muy agrupada (con huecos entre grupos), y eso confunde al "
                "trazado automático. Es una limitación conocida, no un error de la presentación.",
    }, en={
        "title": "The automatic contour and its quality check",
        "what": "What question does it answer?: where does the membrane run, and is that trace reliable?",
        "why": "Plain explanation: the program draws the membrane line on the localizations by itself. Everything "
               "after is measured on that line: if it is wrong, so are the numbers. So a quality check reviews it, "
               "and when it fails the program stops and says why.",
        "example": "The answer on the simulated axon: the contour was REJECTED. The line was {pd_len_nm} nm long, "
                   "{pd_len_ratio} times the perimeter of the cloud (limit {pd_len_limit}): it snaked inside the "
                   "ring. Plan B (the polar contour) failed too: along a stretch it was {pd_polar_off_nm} nm from the "
                   "localizations (limit {pd_polar_limit_nm} nm).",
        "read": "How to read it: green with \"ok\" = reliable. Red with \"x\" = rejected, with the reason beside "
                "it. Here it is red.",
        "more": "This is the intended behaviour: rather than numbers on a bad line, the program stops. In this "
                "simulated axon spectrin is strongly clustered (with gaps between clusters), which confuses the "
                "automatic trace. A known limitation, not a fault of the presentation.",
    }),
    _step("pdd_unanchored", "perimeter_cloud", "", method=_PD_DOC, headings=False, es={
        "title": "Espectrina que no está en la membrana",
        "what": "¿Qué pregunta responde?: ¿qué se hace con la espectrina que aparece adentro del axón, lejos de la "
               "membrana?",
        "why": "Explicación sencilla: la pregunta es sobre el esqueleto de la membrana. Lo que está más de 400 nm "
               "hacia adentro del anillo no es membrana y no se cuenta. Ese umbral es provisorio: sale de un solo "
               "experimento piloto y falta confirmarlo marcando a mano.",
        "example": "La respuesta en el axón simulado: este axón simulado no tiene grupos adentro, y en esta ventana "
                   "el recorte está apagado por defecto; el perfil cuenta todas las localizaciones de la banda.",
        "read": "Cómo se lee: en el gráfico de la nube se ven las tres rodajas en colores, el contorno y la banda "
                "de ±90 nm a su alrededor. Sólo cuenta lo que cae dentro de la banda.",
        "more": "La ventana «Clusters no anclados» hace la revisión de los grupos interiores (más de 400 nm: se "
                "excluye; entre 250 y 400 nm: dudoso, se marca y se queda).",
    }, en={
        "title": "Spectrin that is not on the membrane",
        "what": "What question does it answer?: what happens to spectrin found inside the axon, far from the "
                "membrane?",
        "why": "Plain explanation: the question is about the membrane skeleton. What lies more than 400 nm inside "
               "the ring is not membrane and is not counted. That threshold is provisional: it comes from one pilot "
               "experiment and still has to be confirmed by hand labelling.",
        "example": "The answer on the simulated axon: this simulated axon has no clusters inside, and in this window "
                   "the trim is off by default; the profile counts every localization of the band.",
        "read": "How to read it: the cloud plot shows the three slices in colours, the contour and the +-90 nm band "
                "around it. Only what falls inside the band counts.",
        "more": "The \"Unanchored clusters\" window reviews the interior clusters (beyond 400 nm: excluded; between "
                "250 and 400 nm: doubtful, marked and kept).",
    }),
    _step("pdd_profile", "perimeter_profile", "", method=_PD_DOC, headings=False, es={
        "title": "El perfil de densidad a lo largo del anillo",
        "what": "¿Qué pregunta responde?: ¿cuánta espectrina hay en cada tramo del anillo?",
        "why": "Explicación sencilla: se «desenrolla» el anillo como una cinta y se cuentan las localizaciones en "
               "tramitos iguales. El tramo más denso vale 1 y queda en el centro; el resto se compara con él.",
        "example": "La respuesta en el axón simulado: no hay perfil, porque el contorno fue rechazado (paso "
                   "anterior). Sin línea de membrana no hay cinta que desenrollar.",
        "read": "Cómo se lee: eje horizontal = posición a lo largo del anillo (nm), con el máximo en 0. Eje "
                "vertical = densidad relativa (máximo = 1). Un perfil plano es reparto parejo.",
        "more": "Los tramos sin datos (un arco largo sin localizaciones) se marcan como no medidos en vez de contarse "
                "como cero.",
    }, en={
        "title": "The density profile along the ring",
        "what": "What question does it answer?: how much spectrin is there in each stretch of the ring?",
        "why": "Plain explanation: the ring is \"unrolled\" like a tape and the localizations are counted in equal "
               "small stretches. The densest stretch is 1 and sits in the middle; the rest is compared with it.",
        "example": "The answer on the simulated axon: there is no profile, because the contour was rejected "
                   "(previous step). Without a membrane line there is no tape to unroll.",
        "read": "How to read it: horizontal axis = position along the ring (nm), maximum at 0. Vertical axis = "
                "relative density (maximum = 1). A flat profile is an even spread.",
        "more": "Stretches without data (a long arc without localizations) are marked as not measured instead of "
                "being counted as zero.",
    }),
    _step("pdd_a", "perimeter_a", "", method=_PD_DOC, headings=False, es={
        "title": "Pregunta A: ¿se amontona alrededor del máximo?",
        "what": "¿Qué pregunta responde?: cerca del tramo más denso, ¿hay más espectrina de lo esperable por azar "
                "(se atrae), menos (se repele) o se repite cada tanto?",
        "why": "Explicación sencilla: se compara el perfil real con muchos perfiles mezclados al azar (el «nulo»). "
               "Si el real se sale de lo que da el azar, hay patrón.",
        "example": "La respuesta en el axón simulado: A no se calculó, porque el contorno fue rechazado. La tabla y "
                   "la lectura quedan vacías: el programa no completa números que no midió.",
        "read": "Cómo se lee: la tabla da el contraste y el período con su z y su p; la línea en negrita lo dice en "
                "palabras. «Atractor» sólo cuenta si resiste la zona de exclusión.",
        "more": "A es exploratoria en un axón solo: la lectura de grupo se hace sobre el perfil promedio de muchos "
                "axones (Batch).",
    }, en={
        "title": "Question A: does it pile up around the maximum?",
        "what": "What question does it answer?: near the densest stretch, is there more spectrin than chance would "
                "give (attraction), less (repulsion), or does it repeat every so often?",
        "why": "Plain explanation: the real profile is compared with many profiles shuffled at random (the "
               "\"null\"). If the real one leaves what chance gives, there is a pattern.",
        "example": "The answer on the simulated axon: A was not computed, because the contour was rejected. The "
                   "table and the reading stay empty: the program does not fill in numbers it did not measure.",
        "read": "How to read it: the table gives the contrast and the period with their z and p; the bold line says "
                "it in words. \"Attractor\" counts only if it survives the exclusion zone.",
        "more": "A is exploratory on one axon: the group reading is made on the mean profile of many axons (Batch).",
    }),
    _step("pdd_b", "perimeter_b", "", method=_PD_DOC, headings=False, es={
        "title": "Pregunta B: ¿sigue la curvatura de la membrana?",
        "what": "¿Qué pregunta responde?: ¿hay más espectrina donde la membrana se dobla más?",
        "why": "Explicación sencilla: en cada tramo se mide qué tan curvada está la membrana y cuánta espectrina "
               "hay, y se ve si suben juntas (correlación de Spearman). Se compara con el anillo girado al azar. Un "
               "control revisa que la curvatura no sea un error del contorno.",
        "example": "La respuesta en el axón simulado: B no se calculó, porque el contorno fue rechazado. Además, en "
                   "un círculo la curvatura es casi la misma en todos lados: aunque hubiera contorno, B tendría poco "
                   "para comparar.",
        "read": "Cómo se lee: Spearman NEGATIVO = más espectrina donde la membrana está más curvada. La línea "
                "«Control» debe dar un valor chico; si es alto, puede ser el contorno y no la membrana.",
        "more": "En simulación, B detecta bien el efecto en formas asimétricas, pero casi nada en una elipse "
                "simétrica (ver el paso de validación).",
    }, en={
        "title": "Question B: does it follow the membrane's curvature?",
        "what": "What question does it answer?: is there more spectrin where the membrane bends more?",
        "why": "Plain explanation: in each stretch the bending of the membrane and the amount of spectrin are "
               "measured, and we check whether they rise together (Spearman correlation), against the ring rotated "
               "at random. A control checks that the curvature is not a contour error.",
        "example": "The answer on the simulated axon: B was not computed, because the contour was rejected. Also, "
                   "on a circle the curvature is almost the same everywhere: even with a contour, B would have "
                   "little to compare.",
        "read": "How to read it: a NEGATIVE Spearman = more spectrin where the membrane is more curved. The "
                "\"Control\" line should be small; if it is large, the contour and not the membrane may drive B.",
        "more": "In simulation B detects the effect well on asymmetric shapes, but hardly at all on a symmetric "
                "ellipse (see the validation step).",
    }),
    _step("pdd_c", "perimeter_pairs", "", method=_PD_DOC, headings=False, es={
        "title": "Pregunta C: ¿las rodajas vecinas repiten el patrón?",
        "what": "¿Qué pregunta responde?: el patrón de una rodaja, ¿se repite en la de arriba y en la de abajo?",
        "why": "Explicación sencilla: se correlacionan los perfiles de dos rodajas. Ojo: el microscopio «corre» un "
               "poco cada punto en z (unos 47 nm, la fuga axial), así que dos rodajas pegadas comparten puntos y se "
               "parecen aunque no haya patrón. Por eso la lectura limpia es z−1 contra z+1, que no se tocan.",
        "example": "La respuesta en el axón simulado: C no se calculó, porque el contorno fue rechazado. No hay r "
                   "entre rodajas que mostrar.",
        "read": "Cómo se lee: la tabla da r, z y p por par. Mirar primero z−1 / z+1. En los pares vecinos, comparar "
                "r con la «fuga esperada» que la ventana escribe abajo.",
        "more": "Los 47 nm son el ancho axial medido en el piloto (36–51 nm); cuando el archivo permite medirlo, la "
                "ventana usa el del propio archivo y lo avisa.",
    }, en={
        "title": "Question C: do neighbouring slices repeat the pattern?",
        "what": "What question does it answer?: does one slice's pattern repeat in the slice above and the one "
                "below?",
        "why": "Plain explanation: the profiles of two slices are correlated. Careful: the microscope \"smears\" "
               "each point a little in z (about 47 nm, the axial leak), so two touching slices share points and "
               "look alike even without a pattern. That is why the clean reading is z-1 against z+1, which do not "
               "touch.",
        "example": "The answer on the simulated axon: C was not computed, because the contour was rejected. There "
                   "is no r between slices to show.",
        "read": "How to read it: the table gives r, z and p per pair. Look first at z-1 / z+1. For neighbouring "
                "pairs, compare r with the \"expected leak\" the window writes below.",
        "more": "The 47 nm is the axial width measured in the pilot (36-51 nm); when the file allows it, the window "
                "uses the file's own width and says so.",
    }),
    _step("pdd_validation", "", "", method=_PD_DOC, headings=False, es={
        "title": "Cómo se validó",
        "what": "¿Qué pregunta responde?: ¿estas pruebas inventan patrones donde no los hay, y encuentran los que "
                "sí hay?",
        "why": "Explicación sencilla: se simularon axones sin patrón y con patrón conocido, y se contó cuántas "
               "veces cada prueba acierta o se equivoca. Lo esperable para «falsas alarmas» es cerca del 5 %.",
        "example": "La respuesta en el axón simulado: no sale de este axón, sino de las simulaciones del informe de "
                   "validación. Falsas alarmas C "
                   "{val_c_fp_lo}–{val_c_fp_hi} %, B {val_b_fp} %, A ≤ {val_a_fp} %. Detección de un patrón real: "
                   "C {val_c_power}, B {val_b_power} en formas asimétricas ({val_b_power_sym} en una elipse "
                   "simétrica), A atractor {val_a_power} con 18 axones.",
        "read": "Cómo se lee: falsas alarmas bajas = la prueba no inventa. Detección cerca de 1 = la prueba "
                "encuentra lo que hay. A con {val_a_power} todavía se queda corta.",
        "more": "Esas cifras son con el contorno verdadero de la simulación. A es segura (no inventa) pero poco "
                "sensible; por eso su lectura es exploratoria.",
    }, en={
        "title": "How it was validated",
        "what": "What question does it answer?: do these tests invent patterns where there are none, and do they "
                "find the ones that exist?",
        "why": "Plain explanation: axons without a pattern and with a known pattern were simulated, and we counted "
               "how often each test is right or wrong. The expected rate of \"false alarms\" is about 5 %.",
        "example": "The answer on the simulated axon: not from this axon, but from the simulations of the validation "
                   "report. False alarms C "
                   "{val_c_fp_lo}-{val_c_fp_hi} %, B {val_b_fp} %, A <= {val_a_fp} %. Detection of a real pattern: "
                   "C {val_c_power}, B {val_b_power} on asymmetric shapes ({val_b_power_sym} on a symmetric "
                   "ellipse), A attractor {val_a_power} with 18 axons.",
        "read": "How to read it: low false alarms = the test does not invent. Detection near 1 = the test finds "
                "what is there. A at {val_a_power} still falls short.",
        "more": "Those figures use the simulation's true contour. A is safe (it does not invent) but not very "
                "sensitive; so its reading is exploratory.",
    }),
    _step("pdd_missing", "perimeter_note_a_2", "", method=_PD_DOC, headings=False, es={
        "title": "Qué falta",
        "what": "¿Qué pregunta responde?: ¿qué no se puede concluir todavía?",
        "why": "Explicación sencilla: un axón solo no compara animales ni genotipos. En el piloto real sólo 3 o 4 "
               "contornos salieron confiables, y A necesita al menos 9 axones para decir algo.",
        "example": "La respuesta en el axón simulado: este axón simulado mostró el camino completo y también un "
                   "límite real: con espectrina muy agrupada, el contorno automático puede rechazarse.",
        "read": "Cómo se lee: la línea gris «One axon: no comparison between animals is possible here» lo recuerda "
                "en la pantalla. La comparación de grupos sale del Batch.",
        "more": "Próximos pasos: más axones con contorno confiable, y el etiquetado a mano para fijar el umbral de "
                "espectrina interior.",
    }, en={
        "title": "What is missing",
        "what": "What question does it answer?: what can we not conclude yet?",
        "why": "Plain explanation: one axon compares no animals or genotypes. In the real pilot only 3 or 4 contours "
               "came out reliable, and A needs at least 9 axons to say anything.",
        "example": "The answer on the simulated axon: this simulated axon showed the whole path and also a real "
                   "limit: with strongly clustered spectrin the automatic contour can be rejected.",
        "read": "How to read it: the grey line \"One axon: no comparison between animals is possible here\" says it "
                "on screen. The group comparison comes from the Batch.",
        "more": "Next steps: more axons with a reliable contour, and the hand labelling that fixes the threshold for "
                "interior spectrin.",
    }),
]

# --------------------------------------------------------------------------- the unanchored clusters window (P15)
UNANCHORED_STEPS: List[Dict[str, Any]] = [
    _step("ua_welcome", "", "", es={
        "title": "Clusters no anclados",
        "what": "Busca los clusters de βII-espectrina que quedan por debajo de la membrana (adentro del axón) usando "
                "el anillo que dibuja la propia espectrina, sin tubulina.",
        "why": "Un cluster interior no es parte del esqueleto de la membrana y cambia el perímetro, la ocupancia y "
               "la aleatorización. El análisis se informa siempre con todos los clusters y, al lado, sin los "
               "excluidos.",
        "read": "Más de 400 nm por debajo del anillo: excluido (cruz roja). Entre 250 y 400 nm: dudoso, se marca y "
                "se queda (aro naranja). Los dos umbrales son provisorios hasta el etiquetado a mano.",
        "example": "En un axón simulado con dos grumos plantados en el centro, la vista previa dice «Would exclude 2 "
                   "of …» y los dos grumos aparecen con cruz.",
        "more": "Origen de los números: 400 nm son datos piloto (un experimento) y ad hoc; 250 nm es ad hoc. Si la "
                "nube no es estrellada (una membrana plegada) el anillo se rechaza y no se excluye nada.",
    }, en={
        "title": "Unanchored clusters",
        "what": "Finds the βII-spectrin clusters that lie under the membrane (inside the axon) with the ring the "
                "spectrin itself draws, without tubulin.",
        "why": "An interior cluster is not part of the membrane skeleton and changes the perimeter, the occupancy "
               "and the randomization. The analysis is always reported with every cluster and, beside it, without "
               "the excluded ones.",
        "read": "More than 400 nm under the ring: excluded (red cross). Between 250 and 400 nm: doubtful, marked and "
                "kept (orange ring). Both thresholds are provisional until the hand labelling.",
        "example": "On a simulated axon with two lumps planted in the middle, the preview says \"Would exclude 2 of "
                   "…\" and both lumps carry a cross.",
        "more": "Where the numbers come from: 400 nm is pilot data (one experiment) and ad hoc; 250 nm is ad hoc. "
                "When the cloud is not star-shaped (a folded membrane) the ring is rejected and nothing is excluded.",
    }),
    _step("ua_method", "unanchored_method", "", es={
        "title": "Método y anillo de referencia",
        "what": "Automático (el anillo propio), Widefield (las imágenes del panel Axoplasm, si las hay) u Off (sólo "
                "las decisiones a mano). El anillo: polar de las localizaciones (por defecto) o suave de los centros.",
        "why": "El polar usa miles de localizaciones y es el más estable; el suave se ofrece para comparar, y la "
               "tabla exportada dice cuántos clusters cambian de veredicto con el otro anillo.",
        "read": "Cambiar el anillo o un umbral vuelve a correr la regla (barra de progreso y Cancel); los números "
                "del análisis no cambian hasta Apply.",
        "example": "En el axón simulado con dos grumos, subir «Exclude beyond» a 3000 nm deja la vista previa en "
                   "«Would exclude 0».",
        "more": "La profundidad es la distancia del centro al anillo, positiva hacia adentro, igual para los dos "
                "anillos.",
    }, en={
        "title": "Method and reference ring",
        "what": "Automatic (the own ring), Widefield (the Axoplasm panel's images, when there are any) or Off (the "
                "decisions by hand only). The ring: polar ring of the localizations (default) or smooth ring of the "
                "centres.",
        "why": "The polar ring uses thousands of localizations and is the most stable; the smooth one is offered to "
               "compare, and the exported table says how many clusters change verdict with the other ring.",
        "read": "Changing the ring or a threshold runs the rule again (progress bar and Cancel); the analysis' "
                "numbers do not change until Apply.",
        "example": "On the simulated axon with two lumps, raising \"Exclude beyond\" to 3000 nm leaves the preview at "
                   "\"Would exclude 0\".",
        "more": "The depth is the distance from the centre to the ring, positive inward, the same for both rings.",
    }),
    _step("ua_map", "unanchored_map", "", es={
        "title": "Excluir o restaurar con un clic",
        "what": "Un clic sobre un centro lo excluye (pide el motivo) o, si la regla lo excluyó, lo restaura. Un clic "
                "sobre un cluster que ya tiene una decisión la deshace: vuelve a mandar la regla.",
        "why": "La regla es provisoria: vos decidís los casos que la regla no ve (un axón vecino, glía, un "
               "artefacto) y los que marca de más.",
        "read": "Cruz roja: excluido. Aro naranja: dudoso. Cuadrado vacío: restaurado a mano. Punto verde: el resto. "
                "La línea punteada es el anillo de referencia.",
        "example": "En el axón simulado, un clic sobre un centro de la membrana pide el motivo y el centro pasa a "
                   "tener cruz; otro clic lo devuelve.",
        "more": "Las decisiones se guardan por posición (centro del cluster, tolerancia 50 nm): siguen valiendo si "
                "cambian eps, min samples o el slab y los clusters se renumeran.",
    }, en={
        "title": "Exclude or restore with a click",
        "what": "A click on a centre excludes it (a reason is asked) or, when the rule excluded it, restores it. A "
                "click on a cluster that already has a decision takes it back: the rule decides again.",
        "why": "The rule is provisional: you decide the cases it does not see (a neighbouring axon, glia, an "
               "artefact) and the ones it marks too eagerly.",
        "read": "Red cross: excluded. Orange ring: doubtful. Open square: restored by hand. Green dot: the rest. The "
                "dashed line is the reference ring.",
        "example": "On the simulated axon, a click on a membrane centre asks the reason and the centre gets a cross; "
                   "another click gives it back.",
        "more": "Decisions are kept by position (the cluster's centre, 50 nm tolerance): they still hold when eps, "
                "min samples or the slab change and the clusters are renumbered.",
    }),
    _step("ua_apply", "unanchored_apply", "", es={
        "title": "Aplicar y guardar",
        "what": "Apply vuelve a analizar el axón sin los excluidos, en segundo plano; la ventana del análisis MPS "
                "muestra las dos columnas y los exportes llevan las dos. Save... guarda las decisiones a mano.",
        "why": "Guardadas junto al archivo de datos (nombre-del-archivo_unanchored.json) se leen solas la próxima vez, también "
               "en el Batch.",
        "read": "Si las guardás con otro nombre, la ventana avisa que sólo el nombre por defecto se lee solo; Load... "
                "lee uno con otro nombre.",
        "example": "En el axón simulado con dos grumos, Apply deja «discard applied» con dos clusters menos.",
        "more": "Al abrir cada análisis, el programa corre la regla y el análisis sin los excluidos por su cuenta "
                "(la barra aparece en la ventana del análisis MPS, con Cancel).",
    }, en={
        "title": "Apply and save",
        "what": "Apply analyses the axon again without the excluded clusters, in the background; the MPS analysis "
                "window shows both columns and the exports carry both. Save... keeps the decisions by hand.",
        "why": "Saved beside the data file (data-file-name_unanchored.json) they are read by themselves next time, in the "
               "Batch too.",
        "read": "Saved under another name, the window warns that only the default name is read by itself; Load... "
                "reads one under another name.",
        "example": "On the simulated axon with two lumps, Apply leaves \"discard applied\" with two clusters fewer.",
        "more": "When each analysis opens, the program runs the rule and the analysis without the excluded clusters "
                "by itself (the bar shows in the MPS analysis window, with Cancel).",
    }),
]


COLUMNS_STEPS: List[Dict[str, Any]] = [
    _step("cr_welcome", "", "", es={
        "title": "Revisión de columnas (lumen review)",
        "what": "Muestra los clusters de cada anillo, deja revisar los dudosos del lumen y corre los tests de "
                "columnas (eclipsamiento entre anillos consecutivos) sobre lo que queda.",
        "why": "Las decisiones sobre qué clusters sacar cambian los tests; acá quedan registradas y se exportan.",
        "read": "Arriba, el mapa; a la derecha, los clusters dudosos, la viabilidad de cada par de anillos, la "
                "tolerancia tau y los resultados. «Two rings (unrolled)...» está al pie de esa columna.",
        "example": "En el axón simulado de ejemplo, cada anillo aparece como una fila de clusters sobre el contorno.",
        "more": "Hay dos tau distintos: «Tolerance tau (re-runs the analysis)» re-corre todos los tests del axón; "
                "el «tau of this view only» de la ventana «Two rings» cambia sólo esa vista y su CSV.",
    }, en={
        "title": "Columns review (lumen review)",
        "what": "Shows each ring's clusters, lets you review the doubtful lumen ones and runs the column tests "
                "(eclipsing between consecutive rings) on what is left.",
        "why": "Which clusters you remove changes the tests; here the decisions are logged and exported.",
        "read": "Top: the map; right: the doubtful clusters, the viability of each ring pair, the tolerance tau and "
                "the results. \"Two rings (unrolled)...\" is at the foot of that column.",
        "example": "On the simulated example axon, each ring is a row of clusters along the contour.",
        "more": "There are two different taus: \"Tolerance tau (re-runs the analysis)\" re-runs all the tests of the "
                "axon; the \"tau of this view only\" of the \"Two rings\" window changes only that view and its CSV.",
    }),
    _step("cr_tau", "tau_combo", "", es={
        "title": "Tolerance tau (re-corre el análisis)",
        "what": "Qué tan cerca (en nm) tienen que estar dos clusters de anillos vecinos para contar como eclipsados.",
        "why": "Cambiarlo re-corre TODOS los tests de columnas del axón. tau_0 = 31 nm es el valor "
              "pre-especificado; los demás son exploratorios.",
        "read": "Las opciones son tau_0, 62 nm, 100 nm o «Free». Todo resultado a otro tau dice EXPLORATORY.",
        "example": "Ejemplo simulado: al pasar de 31 a 62 nm se emparejan más clusters, parte por azar; la evidencia "
                   "es el exceso sobre el nulo, no la cantidad.",
        "more": "tau_0 se deriva (2,77 veces el error de centro de un cluster típico, 11,3 nm), no se midió en "
                "datos. La ventana «Two rings» tiene su propio tau, que no re-corre nada.",
    }, en={
        "title": "Tolerance tau (re-runs the analysis)",
        "what": "How close (nm) two clusters of neighbouring rings must be to count as eclipsed.",
        "why": "Changing it re-runs ALL the axon's column tests. tau_0 = 31 nm is the pre-specified value; the "
               "others are exploratory.",
        "read": "The options are tau_0, 62 nm, 100 nm or \"Free\". Every result at another tau says EXPLORATORY.",
        "example": "Simulated example: going from 31 to 62 nm matches more clusters, partly by chance; the evidence "
                   "is the excess over the null, not the count.",
        "more": "tau_0 is derived (2.77 times the centre error of a typical cluster, 11.3 nm), not measured on "
                "data. The \"Two rings\" window has its own tau, which re-runs nothing.",
    }),
    _step("cr_export", "export_button", "", es={
        "title": "Export tables...",
        "what": "Exporta ESTA revisión: una fila por cluster con su clase, tu acción y las banderas, y una fila con "
                "los resultados de la última corrida.",
        "why": "Deja por escrito qué se sacó y con qué resultado.",
        "read": "Nunca escribe en la tabla de otro análisis. La vista «Two rings» se exporta aparte, con su "
                "propio «Export CSV...».",
        "example": "Ejemplo simulado: tras sacar un cluster dudoso y correr, la tabla trae su fila con la acción.",
        "more": "Si el axón ya tiene una fila, sólo se reemplaza si lo confirmás.",
    }, en={
        "title": "Export tables...",
        "what": "Exports THIS review: one row per cluster with its class, your action and the flags, and one row "
                "with the results of the last run.",
        "why": "Records in writing what was removed and with what result.",
        "read": "It never writes into another analysis's table. The \"Two rings\" view is exported separately, with "
                "its own \"Export CSV...\".",
        "example": "Simulated example: after removing a doubtful cluster and running, the table has its row with "
                   "the action.",
        "more": "If the axon already has a row, it is replaced only if you confirm.",
    }),
    _step("cr_two_rings", "two_rings_button", "", es={
        "title": "Two rings (unrolled)...",
        "what": "Dos anillos consecutivos desplegados a lo largo de la membrana, con un tau que fijás a mano.",
        "why": "Para ver qué hace la tolerancia sobre anillos reales antes de aceptar tau_0.",
        "read": "Los clusters que coinciden dentro de tau se pintan; la ventana sombreada de 2 tau se arrastra. "
                "No es un test.",
        "example": "Ejemplo simulado: con tau = 31 nm se emparejan los clusters de las dos filas; con 10 nm, casi "
                   "ninguno.",
        "more": "Su tau afecta sólo esa vista y su CSV («tau of this view only»); no es el «Tolerance tau» de la "
                "revisión, que re-corre el análisis. Cualquier valor distinto de tau_0 es EXPLORATORY.",
    }, en={
        "title": "Two rings (unrolled)...",
        "what": "Two consecutive rings laid flat along the membrane, with a tau you set by hand.",
        "why": "To see what the tolerance does on real rings before accepting tau_0.",
        "read": "Clusters that coincide within tau are painted; the shaded window of 2 tau can be dragged. Not a "
                "test.",
        "example": "Simulated example: with tau = 31 nm the clusters of the two rows pair up; with 10 nm, almost "
                   "none.",
        "more": "Its tau affects only that view and its CSV (\"tau of this view only\"); it is not the review's "
                "\"Tolerance tau\", which re-runs the analysis. Any value other than tau_0 is EXPLORATORY.",
    }),
]


STEPS: Dict[str, List[Dict[str, Any]]] = {
    "main": MAIN_STEPS,
    # TODO(tour): the MPS analysis window's tour (results table, KS, regularity, rings) waits for the pending science
    # PRs that change its rows. An empty list is "no tour yet".
    "mps_analysis": [],
    "perimeter": PERIMETER_STEPS,
    "perimeter_demo": PERIMETER_DEMO_STEPS,
    "unanchored": UNANCHORED_STEPS,
    "columns": COLUMNS_STEPS,
}
