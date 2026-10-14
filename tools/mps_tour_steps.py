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
    "nena": 7.6,              # Data quality: NeNA, nm
    "lp_reported": 8.9,       # Data quality: reported lateral precision (median), nm
    "nena_ratio": 0.85,       # NeNA / reported
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


def facts(lang: str) -> Dict[str, str]:
    """``DEMO_FACTS`` written the way ``lang`` writes numbers (es: 12.137 and 83,2; en: 12,137 and 83.2)."""
    return {k: _number(v, lang) for k, v in DEMO_FACTS.items()}


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
        "demo": "Abrir axón de ejemplo (simulado)",
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
        "demo_title": "Axón de ejemplo (simulado)",
    },
    "en": {
        "help": "Help",
        "beginner": "Beginner mode",
        "tour_window": "Tour this window",
        "explain": "Explain this",
        "demo": "Open the example axon (simulated)",
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
        "demo_title": "Example axon (simulated)",
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
          registry: Tuple[str, ...] = (), badges: bool = False, method: str = "") -> Dict[str, Any]:
    return {"key": key, "target": target, "panel": panel, "registry": registry, "badges": badges,
            "method": method, "es": es, "en": en}


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
        "example": "Podés seguirlo con el axón de ejemplo: Ayuda → «Abrir axón de ejemplo (simulado)». Tiene "
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
        "example": "You can follow it with the example axon: Help → \"Open the example axon (simulated)\". It holds "
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
                "los escribe en la misma tabla que «Export axon».",
        "why": "Para procesar una adquisición entera con los mismos parámetros sin repetir todo a mano.",
        "read": "Su segunda pestaña resume una columna por grupo, por axón y por animal, sin calcular un p-valor.",
        "example": "El axón de ejemplo está en la carpeta examples/demo_axon; un lote de esa carpeta tendría un solo "
                   "axón (ejemplo simulado).",
        "more": "Los parámetros del lote son los de esta ventana en el momento de correrlo; la tabla dice con qué "
                "parámetros se analizó cada axón.",
    }, en={
        "title": "Batch",
        "what": "Analyses every picked axon under a folder at once, with this window's settings, and writes them to "
                "the same table \"Export axon\" writes.",
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

STEPS: Dict[str, List[Dict[str, Any]]] = {
    "main": MAIN_STEPS,
    # TODO(tour): the MPS analysis window's tour (results table, KS, regularity, rings) waits for the pending science
    # PRs that change its rows. An empty list is "no tour yet".
    "mps_analysis": [],
}
