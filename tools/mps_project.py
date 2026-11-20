# -*- coding: utf-8 -*-
"""
The project file of the automatic launcher (``proyecto.yaml``): one per batch of measurements.

What it is for
--------------
The windowless launcher must not read the main window's live settings
(``mps_analysis_settings.json``): those change every time an axon is explored, so a batch run from them can carry a
parameter someone tried an hour earlier. User decision P1 (a), 2026-11-07: the parameters of a launcher run come ONLY
from this file; what the file leaves out takes the program's default; the window's settings are only copied to the
run's manifest as a reference. This module reads and validates the file; it never touches the window's settings.

What of the launcher this repository holds: this module (a commented example is ``docs/launcher_project_example.yaml``;
``batch_settings_from_project`` turns a project into the Batch's settings); the pre-checks, ``tools/mps_precheck.py``,
run on their own with ``python -m tools.launcher_check <proyecto.yaml>`` (nothing is analysed; an axon estimated
above 2 hours waits for a confirmation); and ``tools/make_synthetic_batch.py``, a simulated batch to try them on. The
launcher that runs the analysis itself (stage F4) is not part of this repository.

The rules
---------
* **Strict.** A key the schema does not know is an error, with the closest known key suggested. A typo such as
  ``eps_mn`` must not silently become the default (the same rule as ``tools.mps_columns.load_columns_params`` for
  ``config/columns_params.yaml``).
* **Defaults.** A key left out takes the program's default; ``Project.defaults_used`` lists them, for the manifest.
* **Types and ranges.** Every value is checked; all the problems are reported at once, not one per attempt.
* **EXPLORATORY.** Each analysis parameter has a pre-specified value (the frozen ones of the statistical plan, sec.
  3.1). A file that sets another value is accepted, and ``Project.exploratory_by`` names every parameter that moved,
  with the value used and the pre-specified one. This extends to a whole run the rules the program already applies
  per column: tau = 31 nm carries no mark and any other tau is EXPLORATORY (rule D4); the two linking radii other
  than 3 sqrt(2) and 3 NeNA are EXPLORATORY (``clustering_integration.radii_exploratory``, rule B1).
* **The knockout line.** ``identidad.linea_ko`` is required (user decision 2026-11-07, P3 (a)): ``adducin`` or
  ``4.1B``. Groups are only ever compared within one line, never pooling the WT of two lines.
* **Channel 2** is out of the launcher for now but enters the thesis later: the key ``canal_2`` is reserved and only
  ``null`` is accepted, so a project written for it today fails loudly instead of being run as a one-channel batch.

Messages for the user are in Spanish (the report language of the launcher, ``informe.idioma``); the code and its
comments are in English like the rest of the repository.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import difflib
import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tools.mps_occupancy import SIGMA_CAP_FRACTION
from tools.mps_randomization import DEFAULT_N_RANDOMIZATIONS
from tools.mps_settings import (
    C_LABEL_DEFAULT, C_LABEL_RANGE, DEFAULT_DBCV_THRESHOLD, DEFAULT_EPS_NM, DEFAULT_MAHALANOBIS_THRESHOLD,
    DEFAULT_MIN_SAMPLES, DEFAULT_SLAB_HALF_WIDTH_NM,
)

PROJECT_VERSION = 1

# The per-axon null's replicates. The batch window keeps its own copy (tools/batch_window.py NULL_REPLICATES = 199,
# a Qt module this one must not import); test_launcher_project.py checks the two agree. [code: batch_window.py;
# plan est. sec. 3.1]
NULL_REPLICATES_DEFAULT = 199

# User decision 2026-11-07 (P3 (a)): the knockout line is declared per batch, with these two values.
KO_LINES: Tuple[str, ...] = ("adducin", "4.1B")

# The genotype pattern of the launcher: KO / WT as a whole token, also between underscores. [ad hoc: plan sec. 2.2;
# the program's own default \b(KO|WT)\b misses "_KO_", fixed in the default by a separate change (P2 (b))]
GENOTYPE_PATTERN_DEFAULT = r"(?<![A-Za-z0-9])(KO|WT)(?![A-Za-z0-9])"
ROI_PATTERN_DEFAULT = r"(ROI[ _-]*\d+)"            # [code: tools/mps_identity.py DEFAULT_PATTERNS]
AXON_PATTERN_DEFAULT = r"([Aa]xon[ _-]*\d+)"       # [code: tools/mps_identity.py DEFAULT_PATTERNS]

# A batch name becomes part of a folder name. [ad hoc]
BATCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def _lpc_radius_default() -> float:
    from tools.clustering_integration import LPC_RULE_RADIUS_IN_NENA
    return float(LPC_RULE_RADIUS_IN_NENA)


def _events_radius_default() -> float:
    from tools.clustering_integration import EVENTS_RADIUS_IN_NENA
    return float(EVENTS_RADIUS_IN_NENA)


def _tau_default() -> float:
    from tools.mps_eclipse_tolerance import TAU_SCALAR_DERIVED_NM
    return float(TAU_SCALAR_DERIVED_NM)


def _method_names() -> Tuple[str, ...]:
    from tools.clustering_strategies import registry_names
    return tuple(registry_names(include_experimental=True, available_only=False))


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------
# Kinds of value. "number" accepts an int or a float (never a bool); "path" is a text resolved against the project
# file's folder; "*_or_null" also accepts null.
KINDS = ("text", "int", "number", "bool", "text_list", "path", "path_or_null", "number_or_null", "int_or_null",
         "text_or_null", "map_text_number", "null_only")


@dataclass(frozen=True)
class FieldSpec:
    """One key of the project file."""

    key: str                                   # dotted: "parametros.eps_nm"
    kind: str
    default: Any = None
    required: bool = False
    choices: Optional[Tuple[Any, ...]] = None
    low: Optional[float] = None                # inclusive
    high: Optional[float] = None               # inclusive
    # A pre-specified analysis value: any other value marks the run EXPLORATORY.
    frozen: bool = False
    # How far a frozen number may be from the default and still be the default (a typed 0.3333 for 1/3). 0: exact.
    tolerance: float = 0.0
    # Custom rule for "this value is exploratory" (the null closure: "ambos" is not, "k" is).
    exploratory_if: Optional[Callable[[Any], bool]] = None
    origin: str = ""                           # where the default comes from
    meaning: str = ""                          # one line, Spanish, for the report


def _closure_exploratory(value: Any) -> bool:
    # "ambos" adds a table and the row still carries the n_locs null [code: tools/mps_settings.py:214-218].
    return str(value) == "k"


def _schema() -> Tuple[FieldSpec, ...]:
    F = FieldSpec
    return (
        F("version_proyecto", "int", PROJECT_VERSION, choices=(PROJECT_VERSION,), origin="[ad hoc]",
          meaning="versión del formato del archivo de proyecto"),
        F("nombre_tanda", "text", required=True, origin="obligatorio",
          meaning="nombre corto de la tanda (va al nombre de la carpeta de resultados)"),
        F("carpetas.entrada", "path", required=True, origin="obligatorio",
          meaning="carpeta con un .hdf5 + .yaml de Picasso por axón"),
        F("carpetas.salida", "path", required=True, origin="obligatorio",
          meaning="carpeta donde se crean los resultados; nunca dentro de la entrada"),
        F("archivos.extensiones", "text_list", [".hdf5"], origin="[ad hoc: la entrada pactada es Picasso]",
          meaning="extensiones que son localizaciones"),
        F("archivos.el_nombre_contiene", "text", "picked",
          origin="[ad hoc: convención de los picks del piloto; deja afuera el campo completo]",
          meaning="filtro obligatorio del nombre de archivo"),
        F("archivos.ignorar_carpetas", "text_list", [], origin="[ad hoc]",
          meaning="carpetas que se saltean (borradores)"),
        F("identidad.especie", "text", "mouse", origin="[code: tools/mps_identity.py DEFAULT_SPECIES]",
          meaning="especie"),
        F("identidad.linea_ko", "text", required=True, choices=KO_LINES,
          origin="obligatorio [decisión del usuario 2026-11-07, P3 (a)]",
          meaning="qué knockout es la tanda: adducin o 4.1B"),
        F("identidad.proteina_marcada", "text", "βII-espectrina", origin="[pedido del usuario]",
          meaning="proteína marcada (sólo inventario y manifiesto)"),
        F("identidad.secundario", "text", "AF647", origin="[pedido del usuario]",
          meaning="anticuerpo secundario (sólo inventario y manifiesto)"),
        F("identidad.patrones.genotipo", "text", GENOTYPE_PATTERN_DEFAULT, origin="[ad hoc: plano §2.2]",
          meaning="expresión regular que lee KO/WT del nombre"),
        F("identidad.patrones.roi", "text", ROI_PATTERN_DEFAULT, origin="[code: tools/mps_identity.py]",
          meaning="expresión regular del ROI"),
        F("identidad.patrones.axon", "text", AXON_PATTERN_DEFAULT, origin="[code: tools/mps_identity.py]",
          meaning="expresión regular del axón"),
        F("identidad.patrones.muestra", "text", "", origin="[code: tools/mps_identity.py] vacío = carpeta sobre el ROI",
          meaning="expresión regular de la muestra (opcional)"),
        F("identidad.genotipos_validos", "text_list", ["KO", "WT"], origin="[ad hoc]",
          meaning="genotipos aceptados (se comparan en mayúsculas)"),
        F("identidad.animales", "path_or_null", None, origin="[ad hoc] null = animal desconocido (plano §4)",
          meaning="archivo animales.csv (se usa recién en F6)"),
        F("calibracion.pixel_esperado_nm", "number", required=True, low=1.0, high=5000.0,
          origin="obligatorio; rango [code: tools/mps_pixel_size.py PLAUSIBLE_NM]",
          meaning="el píxel que debería traer cada YAML"),
        F("calibracion.pixel_respaldo_nm", "number_or_null", None, low=1.0, high=5000.0,
          origin="[ad hoc] null = un archivo sin píxel no se analiza",
          meaning="píxel sólo para un archivo sin YAML, sin /metadata y sin vecino"),
        F("calibracion.tolerancia_pixel", "number", 0.005, low=0.0, high=0.5,
          origin="[ad hoc: separa 133 de 135 nm, 1,5 %]",
          meaning="diferencia relativa de píxel que se considera otro píxel"),
        F("calibracion.frames_esperados", "int_or_null", None, low=1, high=10_000_000,
          origin="[ad hoc] null = no se compara", meaning="largo esperado de la película (aviso)"),
        F("calibracion.buffer", "text_or_null", None, choices=(None, "MEA", "βME", "bME", "BME"),
          origin="[code: tools/batch_window.py F5c: sin defecto a propósito]",
          meaning="buffer de imagen; sólo hace falta para Emitters"),
        # -- analysis parameters: frozen ---------------------------------------------------------------------
        F("parametros.metodo", "text", "DBSCAN", frozen=True, origin="[code: tools/mps_settings.py cluster_method]",
          meaning="método de clusterizado"),
        F("parametros.eps_nm", "number", DEFAULT_EPS_NM, low=1.0, high=500.0, frozen=True,
          origin="[lit: Gazal et al. 2026; code: tools/mps_settings.py DEFAULT_EPS_NM]", meaning="eps de DBSCAN"),
        F("parametros.min_samples", "int", DEFAULT_MIN_SAMPLES, low=1, high=10_000, frozen=True,
          origin="[lit: Gazal et al. 2026; code: tools/mps_settings.py DEFAULT_MIN_SAMPLES]",
          meaning="min samples de DBSCAN"),
        F("parametros.ventana_z_semiancho_nm", "number", DEFAULT_SLAB_HALF_WIDTH_NM, low=10.0, high=1000.0,
          frozen=True, origin="[lit: Gazal et al. 2026; code: tools/mps_settings.py]",
          meaning="semiancho de la ventana axial"),
        F("parametros.dbcv_umbral", "number", DEFAULT_DBCV_THRESHOLD, low=-1.0, high=1.0, frozen=True,
          origin="[code: tools/mps_settings.py DEFAULT_DBCV_THRESHOLD: apagado]", meaning="umbral DBCV"),
        F("parametros.mahalanobis_umbral", "number", DEFAULT_MAHALANOBIS_THRESHOLD, low=0.1, high=20.0, frozen=True,
          origin="[lit: Gazal et al. 2026; code: tools/mps_settings.py]", meaning="umbral de la ocupancia"),
        F("parametros.ocupancia_tope_sigma", "number", SIGMA_CAP_FRACTION, low=0.01, high=1.0, frozen=True,
          tolerance=1e-3, origin="[code: tools/mps_occupancy.py SIGMA_CAP_FRACTION]",
          meaning="tope de la elipse (fracción de d_max)"),
        F("parametros.aleatorizaciones", "int", DEFAULT_N_RANDOMIZATIONS, low=1, high=1_000_000, frozen=True,
          origin="[code: tools/mps_randomization.py DEFAULT_N_RANDOMIZATIONS]",
          meaning="colocaciones aleatorias del control KS"),
        F("parametros.semilla", "int", 0, low=0, high=2**31 - 1, frozen=True,
          origin="[code: tools/mps_batch.py BatchSettings.random_seed]", meaning="semilla de la aleatorización"),
        F("parametros.c_label", "number", C_LABEL_DEFAULT, low=C_LABEL_RANGE[0], high=C_LABEL_RANGE[1], frozen=True,
          origin="[lit: Dempsey et al. 2011, provisional; code: tools/mps_settings.py C_LABEL_DEFAULT]",
          meaning="ciclos por marca"),
        F("parametros.radio_lpc_regla_nena", "number", _lpc_radius_default(), low=0.1, high=50.0, frozen=True,
          tolerance=1e-3, origin="[own sim: docs/dbscan_rule.md; decisión B1: 3·√2]",
          meaning="radio de localizaciones por ciclo de la regla (en NeNA)"),
        F("parametros.radio_eventos_nena", "number", _events_radius_default(), low=0.1, high=50.0, frozen=True,
          tolerance=1e-3, origin="[own sim: docs/dbscan_rule.md; decisión D5]",
          meaning="radio de eventos y fusiones (en NeNA)"),
        F("parametros.tau_nm", "number", _tau_default(), low=1.0, high=1000.0, frozen=True,
          origin="[derivación: 2,77 × 11,3 nm, docs/eclipse_tolerance.md; regla D4]",
          meaning="tolerancia de eclipsamiento τ"),
        # -- null ----------------------------------------------------------------------------------------------
        F("nulo.correr", "bool", True, origin="[ad hoc: plano §2.2; las lecturas B del plan necesitan el nulo]",
          meaning="correr el nulo por axón"),
        F("nulo.replicas", "int", NULL_REPLICATES_DEFAULT, low=1, high=100_000, frozen=True,
          origin="[code: tools/batch_window.py NULL_REPLICATES; plan est. §3.1]", meaning="réplicas B del nulo"),
        F("nulo.cierre", "text", "n_locs", choices=("n_locs", "k", "ambos"), frozen=True,
          exploratory_if=_closure_exploratory, origin="[code: tools/mps_settings.py; decisión C4]",
          meaning="cierre del nulo"),
        F("comparar_metodos.correr", "bool", False, origin="[code: tools/mps_batch.py: nunca agrega un método solo]",
          meaning="comparar métodos (tabla aparte)"),
        F("comparar_metodos.metodos", "text_list", [], origin="[code: tools/mps_batch.py]",
          meaning="métodos de la comparación"),
        # -- policies (plan sec. 5) -----------------------------------------------------------------------------
        # P7 as modified by the user (2026-11-07): always the complete analysis, so the perimeter is always built
        # from every start; no "one start above N minutes" alternative.
        F("politicas.perimetro", "text", "todos_los_arranques", choices=("todos_los_arranques",),
          origin="[decisión del usuario 2026-11-07, P7]", meaning="contorno siempre con todos los arranques"),
        F("politicas.metodos_lentos", "text", "rechazar", choices=("rechazar", "permitir"),
          origin="[plano §5; decisión A2]", meaning="qué hacer si el método guardado es lento"),
        F("politicas.ventana_z", "text", "automatica", choices=("automatica",), origin="[plano §5]",
          meaning="pico axial automático"),
        F("politicas.ventana_z_manual", "map_text_number", {}, origin="[plano §5]",
          meaning="pico axial fijado a mano por archivo (z en nm)"),
        F("politicas.borde", "text", "envolvente_convexa", choices=("envolvente_convexa",), origin="[plano §5]",
          meaning="borde de la curación"),
        F("politicas.archivo_picks", "path_or_null", None, origin="[plano §5]",
          meaning="archivo de picks de Picasso (clústeres cortados, P-R59)"),
        F("politicas.sin_genotipo", "text", "frenar", choices=("frenar", "analizar_igual"), origin="[plano §5]",
          meaning="qué hacer con un archivo sin genotipo legible"),
        F("politicas.duplicados", "text", "frenar", choices=("frenar", "quedarse_con_el_primero"),
          origin="[plano §5]", meaning="qué hacer con la misma adquisición dos veces"),
        F("politicas.pixel_distinto", "text", "frenar", choices=("frenar", "usar_el_del_archivo"),
          origin="[plano §5]", meaning="qué hacer con un píxel distinto del esperado"),
        F("politicas.ilegible", "text", "frenar", choices=("frenar", "saltear"), origin="[plano §5]",
          meaning="qué hacer con un archivo ilegible"),
        # -- machine --------------------------------------------------------------------------------------------
        # No per-axon time limit: user decision 2026-11-07 (P7) forbids cutting an axon's analysis. A hung axon is
        # detected by a progress heartbeat (stage F5), not by a total-time limit.
        F("maquina.tope_horas", "number", 24.0, low=0.5, high=24.0 * 14,
          origin="[pedido del usuario: corridas <= 24 h]",
          meaning="tope de una corrida; se corta ENTRE axones y se reanuda (F5)"),
        F("maquina.ram_minima_gb", "number", 1.5, low=0.1, high=512.0,
          origin="[reporte: REPORT_integration3, la máquina falló por debajo de 1,5 GB]",
          meaning="RAM libre mínima para arrancar"),
        F("maquina.disco_minimo_gb", "number", 2.0, low=0.1, high=100_000.0, origin="[ad hoc]",
          meaning="disco libre mínimo en la salida"),
        F("maquina.contar_clusteres_para_estimar", "bool", True,
          origin="[ad hoc: el perímetro crece con el cubo de los clústeres; sin contarlos la estimación es una "
                 "cota inferior]",
          meaning="clusterizar cada archivo en el chequeo previo para estimar el tiempo del perímetro"),
        F("aceptar_axones_largos", "text_list", [], origin="[decisión del usuario 2026-11-07, P7]",
          meaning="axones de más de 2 h estimadas ya aceptados (nombre o ruta relativa a la entrada)"),
        F("canal_2", "null_only", None, origin="[decisión del usuario 2026-11-07: fuera por ahora]",
          meaning="reservado para el canal 2 (sólo null)"),
        F("rol_de_los_datos", "text", "exploratorio", choices=("exploratorio", "confirmatorio"),
          origin="[code: tools/mps_columns.py DATASET_ROLES]", meaning="exploratorio o confirmatorio"),
        F("preregistro", "path_or_null", None, origin="[plan est. §3.1]",
          meaning="pre-registro (obligatorio si confirmatorio)"),
        F("informe.idioma", "text", "es", choices=("es",), origin="[plano §2.2]", meaning="idioma del informe"),
    )


SCHEMA: Tuple[FieldSpec, ...] = _schema()
SPEC: Dict[str, FieldSpec] = {s.key: s for s in SCHEMA}
# Every section (a dotted prefix of some key) and the keys directly under it.
SECTIONS: Dict[str, List[str]] = {}
for _spec in SCHEMA:
    _parts = _spec.key.split(".")
    for _i in range(len(_parts)):
        _parent = ".".join(_parts[:_i])
        _child = ".".join(_parts[:_i + 1])
        SECTIONS.setdefault(_parent, [])
        if _child not in SECTIONS[_parent]:
            SECTIONS[_parent].append(_child)

# Frozen parameters the windowless analysis cannot apply yet: the Batch takes the two radii and tau from the
# program's constants (tools/mps_batch.py analyse_file passes neither to clustering_integration.axon_columns, and
# tau is not part of the per-axon table). A non-default value would be recorded but not used, so the pre-checks
# refuse it until the launcher (F4) threads it through.
NOT_APPLIED_YET: Tuple[str, ...] = ("parametros.radio_lpc_regla_nena", "parametros.radio_eventos_nena",
                                    "parametros.tau_nm")


# ---------------------------------------------------------------------------
# The result of reading
# ---------------------------------------------------------------------------

@dataclass
class Deviation:
    """A frozen parameter set to another value: the run is EXPLORATORY."""

    key: str
    value: Any
    default: Any

    def words(self) -> str:
        return f"{self.key} = {self.value!r} (prefijado: {self.default!r})"


@dataclass
class Project:
    """A validated project: every key of the schema has a value."""

    path: str
    sha256: str
    values: Dict[str, Any]
    defaults_used: List[str] = field(default_factory=list)
    exploratory_by: List[Deviation] = field(default_factory=list)

    def get(self, key: str) -> Any:
        if key not in SPEC:
            raise KeyError(f"{key!r} is not a key of the project file")
        return self.values[key]

    @property
    def exploratory(self) -> bool:
        """The run is exploratory: by its declared role, or because a frozen parameter moved."""
        return self.values["rol_de_los_datos"] != "confirmatorio" or bool(self.exploratory_by)

    def label(self) -> str:
        """The first words of a report: EXPLORATORY and why, or CONFIRMATORY."""
        if not self.exploratory:
            return "CONFIRMATORIO"
        why = []
        if self.values["rol_de_los_datos"] != "confirmatorio":
            why.append("rol de los datos: exploratorio")
        if self.exploratory_by:
            why.append("parámetros cambiados: " + "; ".join(d.words() for d in self.exploratory_by))
        return "EXPLORATORY (" + "; ".join(why) + ")"


@dataclass
class ProjectLoad:
    """What reading a project file gave: the project, or the errors that stop it (all of them)."""

    project: Optional[Project]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.project is not None and not self.errors


class ProjectError(ValueError):
    """The project file cannot be used; the message lists every problem."""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _suggest(key: str, known: Sequence[str]) -> str:
    close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
    return f" (¿quisiste decir '{close[0]}'?)" if close else ""


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _resolve_path(value: str, base: str) -> str:
    text = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not os.path.isabs(text):
        text = os.path.join(base, text)
    return os.path.normpath(os.path.abspath(text))


def _check_value(spec: FieldSpec, value: Any, base: str, errors: List[str]) -> Any:
    """``value`` converted to its kind, or None after appending an error."""
    k = spec.key
    kind = spec.kind
    if value is None:
        if kind.endswith("_or_null") or kind == "null_only":
            return None
        if kind == "text_list":
            errors.append(f"{k}: falta el valor (una lista, por ejemplo [])")
            return None
        errors.append(f"{k}: falta el valor")
        return None
    if kind == "null_only":
        errors.append(f"{k}: todavía no está soportado por el lanzador; dejalo en null "
                      f"(el canal 2 queda fuera por ahora, decisión del usuario 2026-11-07)")
        return None
    base_kind = kind.replace("_or_null", "")
    out: Any
    if base_kind == "text":
        if not isinstance(value, str):
            errors.append(f"{k}: tiene que ser un texto, no {value!r}")
            return None
        out = value
    elif base_kind == "path":
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{k}: tiene que ser una ruta (texto), no {value!r}")
            return None
        out = _resolve_path(value, base)
    elif base_kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                errors.append(f"{k}: tiene que ser un número entero, no {value!r}")
                return None
        out = int(value)
    elif base_kind == "number":
        if not _is_number(value):
            errors.append(f"{k}: tiene que ser un número, no {value!r}")
            return None
        out = float(value)
    elif base_kind == "bool":
        if not isinstance(value, bool):
            errors.append(f"{k}: tiene que ser true o false, no {value!r}")
            return None
        out = value
    elif base_kind == "text_list":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            errors.append(f"{k}: tiene que ser una lista de textos, por ejemplo [\"a\", \"b\"], no {value!r}")
            return None
        out = list(value)
    elif base_kind == "map_text_number":
        if not isinstance(value, dict) or not all(isinstance(a, str) and _is_number(b) for a, b in value.items()):
            errors.append(f"{k}: tiene que ser un diccionario {{\"archivo.hdf5\": número}}, no {value!r}")
            return None
        out = {str(a): float(b) for a, b in value.items()}
    else:                                                   # pragma: no cover - a schema bug
        raise AssertionError(f"unknown kind {kind}")
    if spec.choices is not None and out not in spec.choices:
        shown = ", ".join(repr(c) for c in spec.choices if c is not None)
        errors.append(f"{k}: {out!r} no es un valor aceptado; los valores aceptados son {shown}")
        return None
    if base_kind in ("int", "number"):
        if spec.low is not None and out < spec.low or spec.high is not None and out > spec.high:
            lo = "-inf" if spec.low is None else f"{spec.low:g}"
            hi = "inf" if spec.high is None else f"{spec.high:g}"
            errors.append(f"{k}: {out!r} está fuera del rango aceptado [{lo}, {hi}]")
            return None
    return out


def _walk(data: Any, prefix: str, found: Dict[str, Any], errors: List[str]) -> None:
    """Flatten the YAML mapping into dotted keys, refusing unknown keys and sections that are not mappings."""
    known = SECTIONS.get(prefix, [])
    if not isinstance(data, dict):
        where = prefix or "el archivo"
        errors.append(f"{where}: tiene que ser una sección (claves con dos puntos), no {data!r}")
        return
    for raw_key, value in data.items():
        key = str(raw_key)
        dotted = f"{prefix}.{key}" if prefix else key
        if dotted in SPEC:
            found[dotted] = value
        elif dotted in SECTIONS:
            _walk(value, dotted, found, errors)
        else:
            errors.append(f"clave desconocida '{dotted}'{_suggest(dotted, known + list(SPEC))}")


def default_values() -> Dict[str, Any]:
    """The value of every key that has a default (required keys are absent)."""
    out: Dict[str, Any] = {}
    for spec in SCHEMA:
        if not spec.required:
            d = spec.default
            out[spec.key] = list(d) if isinstance(d, list) else (dict(d) if isinstance(d, dict) else d)
    return out


def _is_exploratory(spec: FieldSpec, value: Any) -> bool:
    if spec.exploratory_if is not None:
        return bool(spec.exploratory_if(value))
    if isinstance(spec.default, float) or isinstance(value, float):
        return not math.isclose(float(value), float(spec.default), rel_tol=0.0,
                                abs_tol=max(spec.tolerance, 1e-9))
    return bool(value != spec.default)


def exploratory_deviations(values: Dict[str, Any]) -> List[Deviation]:
    """The frozen parameters of ``values`` that are not their pre-specified value, in schema order."""
    out = []
    for spec in SCHEMA:
        if spec.frozen and spec.key in values and values[spec.key] is not None:
            if _is_exploratory(spec, values[spec.key]):
                out.append(Deviation(spec.key, values[spec.key], spec.default))
    return out


def parse_project(text: str, path: str = "proyecto.yaml") -> ProjectLoad:
    """Validate the text of a project file. ``path`` sets the folder relative paths are read from."""
    import yaml  # type: ignore[import-untyped, unused-ignore]

    errors: List[str] = []
    warnings: List[str] = []
    base = os.path.dirname(os.path.abspath(path))
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        return ProjectLoad(None, [f"el archivo no es YAML válido: {error}"])
    if data is None:
        return ProjectLoad(None, ["el archivo de proyecto está vacío"])
    found: Dict[str, Any] = {}
    _walk(data, "", found, errors)

    values: Dict[str, Any] = {}
    defaults_used: List[str] = []
    for spec in SCHEMA:
        if spec.key in found:
            values[spec.key] = _check_value(spec, found[spec.key], base, errors)
        elif spec.required:
            if spec.key == "identidad.linea_ko":
                errors.append("identidad.linea_ko: falta declarar la línea KO de la tanda (adducin o 4.1B); las "
                              "comparaciones se hacen siempre dentro de una línea [decisión del usuario 2026-11-07]")
            else:
                errors.append(f"{spec.key}: es obligatorio ({spec.meaning})")
            values[spec.key] = None
        else:
            d = spec.default
            values[spec.key] = list(d) if isinstance(d, list) else (dict(d) if isinstance(d, dict) else d)
            defaults_used.append(spec.key)

    _cross_checks(values, errors, warnings)
    if errors:
        return ProjectLoad(None, errors, warnings)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    project = Project(path=os.path.abspath(path), sha256=sha, values=values, defaults_used=defaults_used,
                      exploratory_by=exploratory_deviations(values))
    return ProjectLoad(project, [], warnings)


def _inside(child: str, parent: str) -> bool:
    c = os.path.normcase(os.path.abspath(child))
    p = os.path.normcase(os.path.abspath(parent))
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


def _cross_checks(values: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    """Checks that need more than one key, or the disk."""
    name = values.get("nombre_tanda")
    if isinstance(name, str) and not BATCH_NAME_RE.match(name):
        errors.append(f"nombre_tanda: {name!r} no sirve como nombre de carpeta; usá letras sin acento, números, "
                      f"'.', '_' o '-', sin espacios (hasta 80)")
    entrada = values.get("carpetas.entrada")
    salida = values.get("carpetas.salida")
    if isinstance(entrada, str) and not os.path.isdir(entrada):
        errors.append(f"carpetas.entrada: la carpeta no existe: {entrada}")
    if isinstance(salida, str):
        if isinstance(entrada, str) and _inside(salida, entrada):
            errors.append("carpetas.salida: no puede estar dentro de carpetas.entrada (los resultados se mezclarían "
                          "con las mediciones)")
        elif not os.path.isdir(salida) and not os.path.isdir(os.path.dirname(salida)):
            errors.append(f"carpetas.salida: ni la carpeta ni la carpeta que la contiene existen: {salida}")
    for key in ("identidad.patrones.genotipo", "identidad.patrones.roi", "identidad.patrones.axon",
                "identidad.patrones.muestra"):
        pattern = values.get(key)
        if isinstance(pattern, str) and pattern:
            try:
                re.compile(pattern)
            except re.error as error:
                errors.append(f"{key}: no es una expresión regular válida ({error})")
    if values.get("identidad.patrones.genotipo") == "":
        errors.append("identidad.patrones.genotipo: no puede estar vacío (el genotipo se lee del nombre)")
    if not values.get("archivos.extensiones"):
        errors.append("archivos.extensiones: la lista no puede estar vacía")
    else:
        for ext in values["archivos.extensiones"] or []:
            if ext.lower() not in (".hdf5", ".h5"):
                errors.append(f"archivos.extensiones: {ext!r} no es un archivo de Picasso (.hdf5 o .h5)")
    if values.get("archivos.el_nombre_contiene") == "":
        warnings.append("archivos.el_nombre_contiene está vacío: también se tomarían los archivos de campo completo")
    genos = values.get("identidad.genotipos_validos")
    if isinstance(genos, list) and not genos:
        errors.append("identidad.genotipos_validos: la lista no puede estar vacía")
    method = values.get("parametros.metodo")
    if isinstance(method, str):
        names = _method_names()
        if method not in names:
            errors.append(f"parametros.metodo: {method!r} no es un método del programa{_suggest(method, names)}")
    for m in values.get("comparar_metodos.metodos") or []:
        if m not in _method_names():
            errors.append(f"comparar_metodos.metodos: {m!r} no es un método del programa")
    if values.get("comparar_metodos.correr") and not values.get("comparar_metodos.metodos"):
        errors.append("comparar_metodos: correr es true pero la lista de métodos está vacía")
    for key in ("identidad.animales", "politicas.archivo_picks", "preregistro"):
        p = values.get(key)
        if isinstance(p, str) and not os.path.isfile(p):
            errors.append(f"{key}: el archivo no existe: {p}")
    if values.get("rol_de_los_datos") == "confirmatorio" and not values.get("preregistro"):
        errors.append("rol_de_los_datos: confirmatorio exige un pre-registro (clave preregistro)")


def load_project(path: str) -> ProjectLoad:
    """Read and validate ``path``. Never raises for a bad file: the errors are in the result."""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            text = handle.read()
    except OSError as error:
        return ProjectLoad(None, [f"no se pudo leer el archivo de proyecto: {error}"])
    except UnicodeDecodeError:
        return ProjectLoad(None, ["el archivo de proyecto no está en UTF-8 (guardalo como UTF-8)"])
    return parse_project(text, path)


def load_project_strict(path: str) -> Project:
    """Like :func:`load_project` but raises ProjectError with every problem."""
    result = load_project(path)
    if result.project is None:
        raise ProjectError("El archivo de proyecto tiene problemas:\n  - " + "\n  - ".join(result.errors))
    return result.project


# ---------------------------------------------------------------------------
# To the Batch's settings
# ---------------------------------------------------------------------------

def batch_settings_from_project(project: Project) -> Any:
    """The ``tools.mps_batch.BatchSettings`` a launcher run analyses with: from the project only (P1 (a)).

    With every key at its default this equals ``BatchSettings.from_settings(MPSSettings())`` (a fresh install);
    test_launcher_project.py checks it. The two radii and tau are not BatchSettings fields (NOT_APPLIED_YET).
    """
    from tools.mps_batch import BatchSettings
    from tools.mps_randomization import RandomizationParams

    v = project.values
    closure = {"ambos": "both"}.get(v["nulo.cierre"], v["nulo.cierre"])
    return BatchSettings(
        eps_nm=float(v["parametros.eps_nm"]),
        min_samples=int(v["parametros.min_samples"]),
        slab_half_width_nm=float(v["parametros.ventana_z_semiancho_nm"]),
        dbcv_threshold=float(v["parametros.dbcv_umbral"]),
        mahalanobis_threshold=float(v["parametros.mahalanobis_umbral"]),
        n_randomizations=int(v["parametros.aleatorizaciones"]),
        random_seed=int(v["parametros.semilla"]),
        randomization_params=RandomizationParams.from_dict({}),
        sigma_cap_fraction=float(v["parametros.ocupancia_tope_sigma"]),
        eps_choice={},
        cluster_method=str(v["parametros.metodo"]),
        cluster_params={},
        cluster_methods_compared=list(v["comparar_metodos.metodos"]) if v["comparar_metodos.correr"] else [],
        c_label=float(v["parametros.c_label"]),
        null_closure=closure,
    )


def describe_defaults() -> List[str]:
    """One line per key, for documentation: key, default and its origin."""
    lines = []
    for spec in SCHEMA:
        d = "obligatorio" if spec.required else repr(spec.default)
        mark = " [congelado]" if spec.frozen else ""
        lines.append(f"{spec.key}: {d}{mark} -- {spec.meaning} {spec.origin}")
    return lines
