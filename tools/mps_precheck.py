# -*- coding: utf-8 -*-
"""
The pre-checks of the automatic launcher: what can be verified in seconds or minutes, before the first hour of
analysis, so that a batch does not run all night on a wrong pixel size or a file without genotype.

No Qt, nothing analysed: the files' names, their YAML, the size and a sample of the localization table, the state of
the machine, and -- for the time estimate only -- each file clustered up to the perimeter (the same reading and
clustering the analysis does, stopped before the perimeter is built; nothing is kept).

Every check ends as one of:

* ERROR -- stops the run before it starts, with what to fix;
* AVISO -- the run may start; it goes to the report;
* CONFIRMAR -- the time rule of user decision P7 (2026-11-07): an axon estimated above 2 hours stops the launcher
  before it starts until the user accepts it (project key ``aceptar_axones_largos``, the command-line flag
  ``--aceptar-axones-largos``, or an answer typed in an interactive console);
* NOTA / OK / NO APLICA -- recorded, never blocking.

The analysis is never cut by a time limit (P7: always the complete analysis); a hung axon is detected by a progress
heartbeat in stage F5, not by a total-time limit. The run as a whole may still be split BETWEEN axons and resumed
(F5), which is why exceeding ``maquina.tope_horas`` is an AVISO and not an error.

Sources of the rules: plan PLAN_FLUJO_AUTOMATICO.md sec. 3 (C1-C21), user decisions of 2026-11-07. Messages are in
Spanish (the language of the launcher's reports); the code is commented in English.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import contextlib
import csv
import ctypes
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from tools.mps_project import Project, ProjectLoad, load_project, NOT_APPLIED_YET, SPEC

ERROR = "ERROR"
AVISO = "AVISO"
CONFIRMAR = "CONFIRMAR"
NOTA = "NOTA"
OK = "OK"
NA = "NO APLICA"
LEVELS = (ERROR, CONFIRMAR, AVISO, NOTA, OK, NA)

# User decision 2026-11-07 (P7): an axon estimated above 2 hours stops the launcher before it starts and asks.
LONG_AXON_SECONDS = 2.0 * 3600.0
# Below this much free RAM the report warns (the plan's "sometimes < 3 GB"). [plan est. sec. 4.2]
RAM_WARN_GB = 3.0
# Units: Picasso stores x, y in camera pixels and z in nm [code: tools/mps_io.py _load_picasso_hdf5]. A camera chip
# is at most a few thousand pixels wide, so a median x or y above this is taken as already in nm. [ad hoc]
MAX_CAMERA_PX = 4096.0
# A z spread (SD) below this many "nm" is taken as z in micrometres (a 3D pick spreads over hundreds of nm). [ad hoc]
MIN_Z_SD_NM = 5.0
# A localization precision above one camera pixel suggests lpx stored in nm. [ad hoc: 10 nm on a 113-130 nm pixel
# is about 0.1 px]
MAX_LPX_PX = 1.0
# The axon's lateral extent (1st-99th percentile) a pick of one axon should have. [ad hoc: the plan's 0.3-15 um]
EXTENT_UM = (0.3, 15.0)
# How many rows of a table are read for the unit checks. [ad hoc]
SAMPLE_ROWS = 200_000
# Windows file attributes of a OneDrive file that is only in the cloud (reading it downloads it).
_FILE_ATTRIBUTE_OFFLINE = 0x1000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000


@dataclass
class Check:
    """One line of the report."""

    code: str
    title: str
    level: str
    message: str = ""
    files: List[str] = field(default_factory=list)


@dataclass
class TimeEstimate:
    """The estimated seconds of one axon (both passes of P6 (b))."""

    analysis_s: float              # one analysis without null (clustering + pipeline + perimeter)
    null_s: float                  # the per-axon null (0 when it does not run)
    total_s: float                 # pass 1 (analysis) + pass 2 (analysis again + null)
    perimeter_s: Optional[float]   # one perimeter from every start; None when the clusters were not counted
    n_null_analyses: float
    lower_bound: bool              # the clusters were not counted: the perimeter is missing from the figure

    def words(self) -> str:
        from tools.method_costs import words_seconds
        text = words_seconds(self.total_s)
        return ("al menos " + text) if self.lower_bound else text


@dataclass
class FileInfo:
    """What the pre-checks learned about one file."""

    path: str
    rel: str
    size_bytes: int = 0
    sha256: str = ""
    readable: bool = True
    unreadable: str = ""
    n_locs: int = 0
    columns: Tuple[str, ...] = ()
    has_z: bool = False
    metadata_source: str = ""
    pixel_nm: Optional[float] = None
    pixel_source: str = ""          # yaml / hdf5 / yaml_scan / neighbour / manual
    frames: Optional[int] = None
    steps: List[str] = field(default_factory=list)
    genotype_raw: str = ""
    genotype: str = ""
    protein_read: str = ""
    sample: str = ""
    roi: str = ""
    axon: str = ""
    n_slab_estimate: Optional[int] = None
    n_clusters: Optional[int] = None
    cluster_count_note: str = ""
    estimate: Optional[TimeEstimate] = None
    usable: bool = True             # still a candidate for the run after the per-file checks
    long_accepted: str = ""         # how a long axon was accepted ("proyecto", "linea de comandos", "consola")


@dataclass
class PrecheckResult:
    """Everything the pre-checks found."""

    project_path: str
    load: ProjectLoad
    started: str
    checks: List[Check] = field(default_factory=list)
    files: List[FileInfo] = field(default_factory=list)
    left_out_by_filter: List[str] = field(default_factory=list)
    left_out_by_folder: List[str] = field(default_factory=list)
    skipped_derived: List[str] = field(default_factory=list)
    total_seconds: Optional[float] = None

    @property
    def project(self) -> Optional[Project]:
        return self.load.project

    def by_level(self, level: str) -> List[Check]:
        return [c for c in self.checks if c.level == level]

    @property
    def errors(self) -> List[Check]:
        return self.by_level(ERROR)

    @property
    def warnings(self) -> List[Check]:
        return self.by_level(AVISO)

    @property
    def long_axons(self) -> List[FileInfo]:
        return [f for f in self.files if f.usable and f.estimate is not None
                and f.estimate.total_s > LONG_AXON_SECONDS]

    @property
    def long_unaccepted(self) -> List[FileInfo]:
        return [f for f in self.long_axons if not f.long_accepted]

    @property
    def status(self) -> str:
        """"errores" (cannot start), "confirmar" (long axons not accepted) or "ok"."""
        if self.errors:
            return "errores"
        if self.long_unaccepted:
            return "confirmar"
        return "ok"

    @property
    def can_start(self) -> bool:
        return self.status == "ok"

    def get(self, code: str) -> List[Check]:
        return [c for c in self.checks if c.code == code]


# ---------------------------------------------------------------------------
# Probes of the machine (injectable for tests)
# ---------------------------------------------------------------------------

def free_ram_gb() -> Optional[float]:
    """Free physical memory in GB, or None when it cannot be measured (no psutil in this environment)."""
    if sys.platform.startswith("win"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            kernel32 = getattr(ctypes, "windll").kernel32
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.ullAvailPhys) / 1024 ** 3
        except Exception:                                   # noqa: BLE001
            return None
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024 ** 2
    except OSError:
        return None
    return None


def free_disk_gb(path: str) -> Optional[float]:
    """Free space on the disk of ``path`` (or of its nearest existing parent) in GB."""
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        probe = parent
    try:
        return float(shutil.disk_usage(probe).free) / 1024 ** 3
    except OSError:
        return None


def git_state(repo: str) -> Tuple[Optional[str], Optional[bool]]:
    """(commit, has uncommitted changes) of the program's repository, or (None, None) outside git."""
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=20)
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=20)
        return commit.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Probes:
    """The machine as the pre-checks see it; tests replace these."""

    ram_gb: Callable[[], Optional[float]] = free_ram_gb
    disk_gb: Callable[[str], Optional[float]] = free_disk_gb
    git: Callable[[str], Tuple[Optional[str], Optional[bool]]] = git_state


# ---------------------------------------------------------------------------
# The time estimate (P7 rule)
# ---------------------------------------------------------------------------

def estimate_axon_seconds(method: str, n_locs_slab: int, n_clusters: Optional[int], *, null_on: bool,
                          replicates: int, closure: str) -> TimeEstimate:
    """
    The seconds of one axon, from the program's own cost figures:

    * one analysis = ``tools.mps_pipeline_null.estimated_seconds_per_replicate`` (the cost table: clustering per
      1000 localizations [own sim] + the rest of the pipeline [ad hoc]) + the perimeter from every start
      (``tools.mps_perimeter_cost.estimate_perimeter_seconds``, A n^B [own sim, docs/perimeter_time.md]);
    * the null = ``tools.mps_batch.null_cost_per_file`` (the B replicates, the K pilots, stage 2 of the regularity as
      an upper bound) plus one perimeter for each pipeline analysis it runs, because every replicate rebuilds the
      contour from every start (``PipelineSpec.from_analysis``: all_starts unless "one start");
    * two passes (user decision P6 (b)): every axon without null first, then again with its null.

    ``n_clusters`` None (not counted): the perimeter is left out and the figure is a lower bound. The perimeter
    term may overlap the cost table's "rest of the pipeline"; the sum errs long, which is the safe side for a
    warning. The launcher (F4) re-estimates with the time measured on the first axon.
    """
    from tools import mps_batch as mb
    from tools import mps_perimeter_cost as cost
    from tools import mps_pipeline_null as pn

    closure_word = {"ambos": "both"}.get(closure, closure)
    per = float(pn.estimated_seconds_per_replicate(method, int(n_locs_slab)))
    perimeter = None if n_clusters is None else float(cost.estimate_perimeter_seconds(int(n_clusters)))
    analysis = per + (perimeter or 0.0)
    null_s = 0.0
    n_null = 0.0
    if null_on:
        base = float(mb.null_cost_per_file(method, int(n_locs_slab), int(replicates), closure_word))
        n_null = base / per if per > 0 else 0.0
        null_s = base + n_null * (perimeter or 0.0)
    total = analysis + ((analysis + null_s) if null_on else 0.0)
    return TimeEstimate(analysis_s=analysis, null_s=null_s, total_s=total, perimeter_s=perimeter,
                        n_null_analyses=n_null, lower_bound=n_clusters is None)


def slab_estimate(z_nm: np.ndarray, half_width_nm: float) -> int:
    """
    The localizations within +/- ``half_width_nm`` of the densest z, for the time estimate only.

    [ad hoc approximation] The analysis finds the slab's centre with a Gaussian mixture on z
    (tools.mps_periodicity); here the centre is the peak of a 20 nm histogram smoothed over five bins. The count
    only scales the cost estimate.
    """
    z = np.asarray(z_nm, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0
    lo, hi = float(np.min(z)), float(np.max(z))
    if hi - lo < 1e-9:
        return int(z.size)
    bins = max(1, int(math.ceil((hi - lo) / 20.0)))
    counts, edges = np.histogram(z, bins=bins, range=(lo, hi))
    smooth = np.convolve(counts, np.ones(5) / 5.0, mode="same")
    k = int(np.argmax(smooth))
    centre = 0.5 * (edges[k] + edges[k + 1])
    return int(np.count_nonzero(np.abs(z - centre) <= half_width_nm))


@contextlib.contextmanager
def _counting_clusters(seen: Dict[str, int]) -> Iterator[None]:
    """
    Make the analysis stop at its perimeter question for EVERY file, recording the kept clusters.

    The analysis asks ``tools.mps_perimeter_cost`` before the perimeter only above its warning threshold; for the
    estimate the count is needed below it too. Inside this block ``needs_warning`` answers yes and the gate records
    the number and cancels, so nothing past the clustering runs. Single-threaded by design: the pre-checks are one
    process and run nothing in parallel.
    """
    from unittest import mock

    from tools import mps_perimeter_cost as cost

    def gate(n: int) -> str:
        seen["n"] = int(n)
        return cost.CANCEL

    with mock.patch.object(cost, "needs_warning", lambda _n: True), cost.asking(gate):
        yield


def count_kept_clusters(info: FileInfo, settings: Any) -> Tuple[Optional[int], str]:
    """The clusters the analysis would keep and join into the perimeter of this file, or (None, why)."""
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity

    needs = info.pixel_source in ("neighbour", "manual")
    item = mb.BatchFile(path=info.path, identity=AxonIdentity(), own_pixel_size_nm=None if needs else info.pixel_nm,
                        needs_pixel_size=needs, pixel_size_nm=info.pixel_nm if needs else None,
                        pixel_size_source=info.pixel_source if needs else "")
    seen: Dict[str, int] = {}
    with _counting_clusters(seen):
        record = mb.analyse_file(item, settings, allow_slow=True)
    if "n" in seen:
        return seen["n"], ""
    return None, (record.error or "la cuenta de clústeres no llegó al perímetro")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalise_genotype(value: str) -> str:
    """Upper case, no surrounding spaces: "wt " and "WT" are one genotype (plan sec. 2.2, C10)."""
    return str(value or "").strip().upper()


def normalise_ko_line(value: str) -> str:
    """The knockout line a name states, in the project's spelling ("" when it is not one of the two)."""
    v = str(value or "").strip().lower()
    if v in ("aducina", "adducina", "adducin", "add", "aducin"):
        return "adducin"
    if v == "4.1b":
        return "4.1B"
    return ""


def _axon_key(text: str) -> str:
    return re.sub(r"[\s_-]+", "", str(text or "").lower())


def _cloud_only(path: str) -> bool:
    try:
        attrs = getattr(os.stat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & (_FILE_ATTRIBUTE_OFFLINE | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))


def _in_onedrive(path: str) -> bool:
    p = os.path.normcase(os.path.abspath(path))
    roots = [os.environ.get(k, "") for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
    if any(r and (p + os.sep).startswith(os.path.normcase(os.path.abspath(r)) + os.sep) for r in roots):
        return True
    return any(part.lower().startswith("onedrive") for part in p.split(os.sep))


def _accepted(info: FileInfo, accepted: Sequence[str]) -> bool:
    names = {a.strip().replace("\\", "/").lower() for a in accepted if a.strip()}
    return info.rel.lower() in names or os.path.basename(info.path).lower() in names


# ---------------------------------------------------------------------------
# Reading one file (no analysis)
# ---------------------------------------------------------------------------

def _read_file(info: FileInfo) -> Optional[np.ndarray]:
    """Fill the file's table facts; return a sample of its rows, or None (``info.unreadable`` says why)."""
    import h5py

    from tools import mps_metadata

    try:
        if not h5py.is_hdf5(info.path):
            info.readable, info.unreadable = False, "no es un archivo HDF5"
            return None
        with h5py.File(info.path, "r") as handle:
            if "locs" not in handle:
                info.readable, info.unreadable = False, "no tiene la tabla '/locs' (no es un archivo de localizaciones de Picasso)"
                return None
            ds = handle["locs"]
            names = tuple(ds.dtype.names or ())
            info.columns = names
            info.n_locs = int(ds.shape[0])
            missing = [c for c in ("x", "y") if c not in names]
            if missing:
                info.readable, info.unreadable = False, f"le faltan las columnas {', '.join(missing)}"
                return None
            step = max(1, int(math.ceil(info.n_locs / SAMPLE_ROWS)))
            sample = ds[::step]
    except Exception as error:                              # noqa: BLE001 - reported as unreadable
        info.readable, info.unreadable = False, f"no se pudo abrir: {error}"
        return None
    info.has_z = "z" in info.columns
    meta, source = mps_metadata.load_metadata(info.path)
    info.metadata_source = source
    info.pixel_nm = mps_metadata.pixel_size_nm(meta)
    if info.pixel_nm is not None:
        info.pixel_source = source
    frames = mps_metadata.get_value(meta, "Frames")
    try:
        info.frames = int(frames) if frames is not None else None
    except (TypeError, ValueError):
        info.frames = None
    info.steps = mps_metadata.steps(meta)
    return np.asarray(sample)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def run_prechecks(project_path: str, *, accept_long: bool = False, count_clusters: Optional[bool] = None,
                  probes: Optional[Probes] = None, on_file: Optional[Callable[[int, int, str], None]] = None
                  ) -> PrecheckResult:
    """
    Run every pre-check on the project at ``project_path``.

    ``accept_long``: the command-line flag that accepts every axon above 2 hours (P7). ``count_clusters`` overrides
    the project's ``maquina.contar_clusteres_para_estimar``. ``on_file(i, n, name)`` reports progress.
    """
    probes = probes or Probes()
    load = load_project(project_path)
    result = PrecheckResult(project_path=os.path.abspath(project_path), load=load,
                            started=datetime.now().isoformat(timespec="seconds"))
    add = result.checks.append

    if load.project is None:
        add(Check("C1", "Archivo de proyecto válido", ERROR,
                  f"{len(load.errors)} problema(s) en el archivo de proyecto:\n" + "\n".join(f"- {e}" for e in load.errors)))
        return result
    project = load.project
    v = project.values
    add(Check("C1", "Archivo de proyecto válido", OK,
              f"leído y validado (sha256 {project.sha256[:12]}); {len(project.defaults_used)} clave(s) por defecto"))
    for w in load.warnings:
        add(Check("C1", "Archivo de proyecto válido", AVISO, w))
    add(Check("C1b", "Parámetros congelados", AVISO if project.exploratory_by else OK,
              ("la corrida queda EXPLORATORY por: " + "; ".join(d.words() for d in project.exploratory_by))
              if project.exploratory_by else "todos en su valor prefijado"))
    not_applied = [d for d in project.exploratory_by if d.key in NOT_APPLIED_YET]
    if not_applied:
        add(Check("C1c", "Parámetros que el análisis sin ventana todavía no aplica", ERROR,
                  "estos valores se registrarían pero el Batch usaría el prefijado, así que no se aceptan hasta que "
                  "el lanzador (F4) los aplique; dejalos en su valor prefijado: "
                  + "; ".join(d.words() for d in not_applied)))
    if v["rol_de_los_datos"] == "confirmatorio":
        add(Check("C20", "Corrida confirmatoria", ERROR,
                  "una corrida confirmatoria necesita el animal de cada fila (F6) y comparar los parámetros con el "
                  "pre-registro; esta versión del lanzador todavía no lo hace. Usá rol_de_los_datos: exploratorio."))
    add(Check("C24", "Canal 2", NA, "fuera del lanzador por ahora (decisión del usuario 2026-11-07); clave canal_2 "
                                     "reservada"))
    if v["identidad.animales"]:
        add(Check("C25", "Animal", AVISO, "identidad.animales está declarado pero el lanzador lo usa recién en F6; "
                                          "por ahora no se calcula nada entre genotipos"))
    else:
        add(Check("C25", "Animal", NOTA, "sin animal: se describe cada grupo pero no habrá ningún p-valor entre "
                                         "genotipos (plano §4); el animal nunca se adivina"))

    settings = _settings(project)
    _method_checks(result, project)
    _inventory(result, project)
    _file_checks(result, project, settings, count_clusters, on_file)
    _set_checks(result, project)
    _time_checks(result, project, accept_long)
    _machine_checks(result, project, probes)
    return result


def _settings(project: Project) -> Any:
    from tools.mps_project import batch_settings_from_project
    return batch_settings_from_project(project)


def _method_checks(result: PrecheckResult, project: Project) -> None:
    from tools.method_costs import is_slow, method_label

    v = project.values
    methods = [v["parametros.metodo"]] + (list(v["comparar_metodos.metodos"]) if v["comparar_metodos.correr"] else [])
    slow = [m for m in methods if is_slow(m)]
    if slow and v["politicas.metodos_lentos"] == "rechazar":
        result.checks.append(Check("C15", "Método lento", ERROR,
                                   "métodos lentos con politicas.metodos_lentos: rechazar: "
                                   + ", ".join(method_label(m) for m in slow)
                                   + ". Cambiá el método o poné metodos_lentos: permitir."))
    elif slow:
        result.checks.append(Check("C15", "Método lento", AVISO, "se permiten métodos lentos: "
                                   + ", ".join(method_label(m) for m in slow)))
    else:
        result.checks.append(Check("C15", "Método lento", OK, "ninguno"))
    if "emitters_bagol_like" in methods and not v["calibracion.buffer"]:
        result.checks.append(Check("C15", "Buffer de Emitters", ERROR,
                                   "Emitters necesita calibracion.buffer (MEA o βME); no tiene valor por defecto"))


def _inventory(result: PrecheckResult, project: Project) -> None:
    from tools import mps_io

    v = project.values
    root = v["carpetas.entrada"]
    exts = tuple(e.lower() for e in v["archivos.extensiones"])
    needle = v["archivos.el_nombre_contiene"]
    found, skipped = mps_io.find_localization_files(root, pattern=needle, extensions=exts)
    every, _ = mps_io.find_localization_files(root, pattern="", extensions=exts, exclude_derived=False)
    ignored = {d.strip().lower() for d in v["archivos.ignorar_carpetas"] if d.strip()}

    def in_ignored(path: str) -> bool:
        parts = _rel(os.path.dirname(path), root).lower().split("/")
        return any(p in ignored for p in parts)

    result.left_out_by_folder = [p for p in found if in_ignored(p)]
    kept = [p for p in found if not in_ignored(p)]
    result.skipped_derived = list(skipped)
    found_set = set(found) | set(skipped)
    result.left_out_by_filter = [p for p in every if p not in found_set and not in_ignored(p)]
    result.files = [FileInfo(path=p, rel=_rel(p, root)) for p in kept]
    if not kept:
        result.checks.append(Check("C2", "Archivos a analizar", ERROR,
                                   f"ningún archivo {', '.join(exts)} de {root} contiene {needle!r} en el nombre"))
    else:
        msg = f"{len(kept)} archivo(s) pasan el filtro {needle!r}"
        if result.left_out_by_filter:
            msg += f"; {len(result.left_out_by_filter)} dejado(s) afuera por el filtro (campo completo u otros)"
        if result.left_out_by_folder:
            msg += f"; {len(result.left_out_by_folder)} en carpetas ignoradas"
        if skipped:
            msg += f"; {len(skipped)} escrito(s) por el propio programa"
        result.checks.append(Check("C2", "Archivos a analizar", OK, msg,
                                   [_rel(p, root) for p in result.left_out_by_filter]))
    cloud = [f.rel for f in result.files if _cloud_only(f.path)]
    if cloud:
        result.checks.append(Check("C27", "Archivos sólo en la nube (OneDrive)", AVISO,
                                   f"{len(cloud)} archivo(s) no están en el disco; leerlos los descarga (y el "
                                   f"chequeo los lee para la huella)", cloud))


def _file_checks(result: PrecheckResult, project: Project, settings: Any, count_clusters: Optional[bool],
                 on_file: Optional[Callable[[int, int, str], None]]) -> None:
    from tools import mps_pixel_size
    from tools.mps_identity import propose

    v = project.values
    add = result.checks.append
    patterns = {"genotype": v["identidad.patrones.genotipo"], "roi_name": v["identidad.patrones.roi"],
                "axon_name": v["identidad.patrones.axon"], "sample": v["identidad.patrones.muestra"]}
    valid = {normalise_genotype(g) for g in v["identidad.genotipos_validos"]}
    expected_px = float(v["calibracion.pixel_esperado_nm"])
    tol = float(v["calibracion.tolerancia_pixel"])
    fallback = v["calibracion.pixel_respaldo_nm"]
    counting = v["maquina.contar_clusteres_para_estimar"] if count_clusters is None else bool(count_clusters)
    frames_expected = v["calibracion.frames_esperados"]
    unreadable_level = ERROR if v["politicas.ilegible"] == "frenar" else AVISO
    pixel_level = ERROR if v["politicas.pixel_distinto"] == "frenar" else AVISO
    geno_level = ERROR if v["politicas.sin_genotipo"] == "frenar" else AVISO

    bad: Dict[str, List[str]] = {k: [] for k in ("C3", "C4", "C5w", "C5e", "C5n", "C5m", "C6", "C7xy", "C7z", "C7lp",
                                                 "C7ext", "C8", "C9", "C10", "C14", "C22", "C26")}
    detail: Dict[str, List[str]] = {k: [] for k in bad}
    for index, info in enumerate(result.files):
        if on_file is not None:
            on_file(index, len(result.files), info.rel)
        try:
            info.size_bytes = os.path.getsize(info.path)
            info.sha256 = _sha256(info.path)
        except OSError as error:
            info.readable, info.unreadable = False, f"no se pudo leer: {error}"
        sample = _read_file(info) if info.readable else None
        # identity: read from the path, never guessed
        prop = propose(info.path, patterns=patterns)
        ident = prop.identity
        info.genotype_raw = ident.genotype
        # propose() already stores KO/WT in one spelling ("wt" -> "WT"); the spelling the file name used is
        # in its detail line ("'wt' found in the path"), and C10 must still report it.
        _spelled = re.match(r"'(.*?)' found in the path", prop.detail.get("genotype", ""))
        if _spelled:
            info.genotype_raw = _spelled.group(1)
        info.genotype = normalise_genotype(ident.genotype)
        info.protein_read = ident.protein
        info.sample, info.roi, info.axon = ident.sample, ident.roi_name, ident.axon_name
        if not info.genotype or info.genotype not in valid:
            bad["C9"].append(info.rel)
            detail["C9"].append(f"{info.rel}: " + (f"genotipo {info.genotype_raw!r} no está entre "
                                                   f"{sorted(valid)}" if info.genotype else "no se leyó el genotipo"))
            if geno_level == ERROR:
                info.usable = False
        elif info.genotype != info.genotype_raw:
            bad["C10"].append(info.rel)
            detail["C10"].append(f"{info.rel}: {info.genotype_raw!r} se lee como {info.genotype!r}")
        line = normalise_ko_line(ident.protein)
        if ident.protein and line and line != v["identidad.linea_ko"]:
            bad["C22"].append(info.rel)
            detail["C22"].append(f"{info.rel}: el nombre dice {ident.protein!r} y el proyecto declara "
                                 f"linea_ko {v['identidad.linea_ko']!r}")
            info.usable = False
        if not info.roi or not info.axon:
            missing = [w for w, val in (("ROI", info.roi), ("axón", info.axon)) if not val]
            bad["C8"].append(info.rel)
            detail["C8"].append(f"{info.rel}: no se leyó {' ni '.join(missing)}")
        if not info.readable or sample is None:
            bad["C3"].append(info.rel)
            detail["C3"].append(f"{info.rel}: {info.unreadable}")
            # Never analysed; the policy only decides whether it stops the whole run (frenar) or is skipped.
            info.usable = False
            continue
        if not info.has_z:
            bad["C4"].append(info.rel)
            detail["C4"].append(f"{info.rel}: no tiene columna z (2D)")
            info.usable = False
        # pixel size: the file's own record, else a neighbour, else the fallback, else refused
        if info.pixel_nm is None:
            res, _notes = mps_pixel_size.resolve(
                info.path, {}, lambda cands, notes: (fallback if not cands else None))
            if res is not None and res.token == "neighbour":
                info.pixel_nm, info.pixel_source = res.nm, "neighbour"
                bad["C5n"].append(info.rel)
                detail["C5n"].append(f"{info.rel}: sin YAML ni /metadata; píxel {res.nm:g} nm de un vecino ({res.source})")
            elif res is not None:
                info.pixel_nm, info.pixel_source = res.nm, "manual"
                bad["C5m"].append(info.rel)
                detail["C5m"].append(f"{info.rel}: sin píxel propio ni vecino; se usa pixel_respaldo_nm = {res.nm:g} nm")
            else:
                bad["C5e"].append(info.rel)
                detail["C5e"].append(f"{info.rel}: sin YAML, sin /metadata y sin un vecino con un único píxel; "
                                     f"sin pixel_respaldo_nm no se puede leer")
                info.usable = False
        elif info.metadata_source != "yaml":
            bad["C5w"].append(info.rel)
            how = {"hdf5": "sin YAML; píxel del /metadata embebido",
                   "yaml_scan": "YAML dañado; píxel leído línea por línea"}.get(info.metadata_source,
                                                                                info.metadata_source)
            detail["C5w"].append(f"{info.rel}: {how}")
        if info.pixel_nm is not None and abs(info.pixel_nm - expected_px) > tol * expected_px:
            bad["C6"].append(info.rel)
            detail["C6"].append(f"{info.rel}: píxel {info.pixel_nm:g} nm ({info.pixel_source}), esperado "
                                f"{expected_px:g} nm")
            if pixel_level == ERROR:
                info.usable = False
        # units, on a sample of rows
        x = np.asarray(sample["x"], float)
        y = np.asarray(sample["y"], float)
        if x.size:
            if float(np.median(np.abs(x))) > MAX_CAMERA_PX or float(np.median(np.abs(y))) > MAX_CAMERA_PX:
                bad["C7xy"].append(info.rel)
                detail["C7xy"].append(f"{info.rel}: x,y con mediana {float(np.median(x)):.0f}, "
                                      f"{float(np.median(y)):.0f}: parecen estar en nm y no en píxeles de cámara")
                info.usable = False
            elif info.pixel_nm is not None:
                ext_nm = max(float(np.percentile(x, 99) - np.percentile(x, 1)),
                             float(np.percentile(y, 99) - np.percentile(y, 1))) * info.pixel_nm
                if not EXTENT_UM[0] <= ext_nm / 1000.0 <= EXTENT_UM[1]:
                    bad["C7ext"].append(info.rel)
                    hint = " (¿campo completo?)" if ext_nm / 1000.0 > EXTENT_UM[1] else ""
                    detail["C7ext"].append(f"{info.rel}: extensión lateral {ext_nm / 1000.0:.2f} µm, fuera de "
                                           f"{EXTENT_UM[0]:g}-{EXTENT_UM[1]:g} µm{hint}")
        if info.has_z:
            z = np.asarray(sample["z"], float)
            if z.size and float(np.std(z)) < MIN_Z_SD_NM:
                bad["C7z"].append(info.rel)
                detail["C7z"].append(f"{info.rel}: z con desvío {float(np.std(z)):.3g}: parece estar en µm o ser "
                                     f"constante (Picasso guarda z en nm)")
                info.usable = False
        if "lpx" in info.columns:
            lp = np.asarray(sample["lpx"], float)
            if lp.size and float(np.median(lp)) > MAX_LPX_PX:
                bad["C7lp"].append(info.rel)
                detail["C7lp"].append(f"{info.rel}: precisión lpx mediana {float(np.median(lp)):.3g}: parece "
                                      f"estar en nm (Picasso la guarda en píxeles)")
        if frames_expected is not None and info.frames is not None and info.frames != frames_expected:
            bad["C14"].append(info.rel)
            detail["C14"].append(f"{info.rel}: {info.frames} cuadros, esperado {frames_expected}")
        if info.steps and not any("pick" in s.lower() for s in info.steps):
            bad["C26"].append(info.rel)
            detail["C26"].append(f"{info.rel}: el historial del YAML no tiene un paso de pick")
        # the estimate's inputs
        if info.has_z and info.usable:
            info.n_slab_estimate = slab_estimate(np.asarray(sample["z"], float),
                                                 float(v["parametros.ventana_z_semiancho_nm"]))
            info.n_slab_estimate = int(round(info.n_slab_estimate * info.n_locs / max(1, x.size)))
        if counting and info.usable and info.pixel_nm is not None:
            n, why = count_kept_clusters(info, settings)
            info.n_clusters, info.cluster_count_note = n, why

    def emit(code: str, key: str, title: str, level: str) -> None:
        if bad[key]:
            add(Check(code, title, level, "\n".join(detail[key]), list(bad[key])))

    emit("C3", "C3", "Archivo ilegible", unreadable_level)
    emit("C4", "C4", "Archivo sin z (2D)", ERROR)
    emit("C5", "C5w", "YAML faltante o dañado", AVISO)
    emit("C5", "C5n", "Píxel tomado de un vecino", AVISO)
    emit("C5", "C5m", "Píxel de respaldo (a mano)", AVISO)
    emit("C5", "C5e", "Archivo sin píxel", ERROR)
    emit("C6", "C6", "Píxel distinto del esperado", pixel_level)
    emit("C7", "C7xy", "Unidades: x,y", ERROR)
    emit("C7", "C7z", "Unidades: z", ERROR)
    emit("C7", "C7lp", "Unidades: precisión", AVISO)
    emit("C7", "C7ext", "Tamaño del pick", AVISO)
    emit("C8", "C8", "ROI o axón no leído del nombre", AVISO)
    emit("C9", "C9", "Genotipo desconocido", geno_level)
    emit("C10", "C10", "Genotipo escrito de otra forma", AVISO)
    emit("C14", "C14", "Película de otro largo", AVISO)
    emit("C22", "C22", "Línea KO del nombre distinta de la declarada", ERROR)
    emit("C26", "C26", "¿Pick o campo completo?", AVISO)
    for code, keys in (("C3", ["C3"]), ("C4", ["C4"]), ("C5", ["C5w", "C5n", "C5m", "C5e"]), ("C6", ["C6"]),
                       ("C7", ["C7xy", "C7z", "C7lp", "C7ext"]), ("C8", ["C8"]), ("C9", ["C9"]), ("C10", ["C10"]),
                       ("C14", ["C14"]), ("C22", ["C22"]), ("C26", ["C26"])):
        if result.files and not any(bad[k] for k in keys):
            add(Check(code, _TITLES[code], OK, "sin problemas"))
    manual = v["politicas.ventana_z_manual"]
    if manual:
        names = {os.path.basename(f.path).lower() for f in result.files} | {f.rel.lower() for f in result.files}
        orphan = [k for k in manual if k.replace("\\", "/").lower() not in names]
        if orphan:
            add(Check("C28", "Pico axial fijado a mano", AVISO,
                      "politicas.ventana_z_manual nombra archivos que no están en la tanda: " + ", ".join(orphan)))


_TITLES = {"C3": "Archivo ilegible", "C4": "Archivo sin z (2D)", "C5": "YAML y píxel", "C6": "Píxel esperado",
           "C7": "Unidades plausibles", "C8": "ROI y axón legibles", "C9": "Genotipo", "C10": "Genotipo normalizado",
           "C14": "Largo de la película", "C22": "Línea KO coherente", "C26": "Pick o campo completo"}


def _set_checks(result: PrecheckResult, project: Project) -> None:
    """Checks across files: duplicates, same name in two folders, same content, identity collisions, pixels."""
    from tools import mps_io

    v = project.values
    add = result.checks.append
    files = result.files
    if not files:
        return
    by_path = {f.path: f for f in files}
    dup_level = ERROR if v["politicas.duplicados"] == "frenar" else AVISO
    # C11: one acquisition under two names in the SAME folder (mps_io.duplicate_sources); across folders the same
    # name is another axon (user decision 2026-11-07) and a different name with the same stem is only a warning.
    same_folder: List[str] = []
    cross: List[str] = []
    for stem, group in mps_io.duplicate_sources([f.path for f in files]).items():
        by_folder: Dict[str, List[str]] = {}
        for p in group:
            by_folder.setdefault(os.path.normcase(os.path.dirname(p)), []).append(p)
        for members in by_folder.values():
            if len(members) > 1:
                same_folder.append(" = ".join(by_path[p].rel for p in members))
                if dup_level == ERROR:
                    for p in members:
                        by_path[p].usable = False
                else:
                    for p in sorted(members)[1:]:
                        by_path[p].usable = False
        names = {os.path.basename(p).lower() for p in group}
        if len(by_folder) > 1 and len(names) > 1:
            cross.append(" / ".join(by_path[p].rel for p in group))
    if same_folder:
        add(Check("C11", "Misma adquisición con dos nombres", dup_level,
                  "\n".join(same_folder) + ("" if dup_level == ERROR else
                                            "\n(se analiza sólo el primero de cada grupo)")))
    else:
        add(Check("C11", "Misma adquisición con dos nombres", OK, "ninguna"))
    if cross:
        add(Check("C11", "Posible misma adquisición en dos carpetas", AVISO, "\n".join(cross)))
    # C12: the same file name in two folders: two axons (user decision 2026-11-07), but today's axon_id is built
    # from the file name alone (tools/mps_identity.axon_id), so both rows would carry the same axon_id.
    by_name: Dict[str, List[FileInfo]] = {}
    for f in files:
        by_name.setdefault(os.path.basename(f.path).lower(), []).append(f)
    same = [g for g in by_name.values() if len(g) > 1]
    if same:
        add(Check("C12", "Mismo nombre de archivo en carpetas distintas", AVISO,
                  "se tratan como axones distintos (decisión del usuario 2026-11-07); ojo: hoy el identificador "
                  "axon_id sale sólo del nombre del archivo, así que sus filas compartirían axon_id. El lanzador (F4) "
                  "tiene que resolverlo antes de escribir tablas:\n" + "\n".join(" / ".join(f.rel for f in g)
                                                                                  for g in same)))
    else:
        add(Check("C12", "Mismo nombre de archivo en carpetas distintas", OK, "ninguno"))
    # C13: the same bytes under two names: a copy.
    by_sha: Dict[str, List[FileInfo]] = {}
    for f in files:
        if f.sha256:
            by_sha.setdefault(f.sha256, []).append(f)
    copies = [g for g in by_sha.values() if len(g) > 1]
    if copies:
        for g in copies:
            for f in g:
                f.usable = False
        add(Check("C13", "Mismo contenido con dos nombres (copia)", ERROR,
                  "\n".join(" = ".join(f.rel for f in g) for g in copies)))
    else:
        add(Check("C13", "Mismo contenido con dos nombres (copia)", OK, "ninguno"))
    # C8 (collision): two files that end with the same identity.
    idents: Dict[Tuple[str, str, str, str], List[FileInfo]] = {}
    for f in files:
        if f.roi and f.axon:
            idents.setdefault((f.sample.lower(), f.roi.lower(), _axon_key(f.axon), f.genotype), []).append(f)
    collide = [g for g in idents.values() if len(g) > 1]
    if collide:
        add(Check("C8", "Dos archivos con la misma identidad", ERROR,
                  "misma muestra, ROI, axón y genotipo leídos del nombre:\n"
                  + "\n".join(" = ".join(f.rel for f in g) for g in collide)))
    # C6 (set): two pixel sizes in one batch.
    pixels = sorted({round(f.pixel_nm, 3) for f in files if f.pixel_nm is not None})
    if len(pixels) > 1:
        level = ERROR if v["politicas.pixel_distinto"] == "frenar" else AVISO
        add(Check("C6", "Píxeles distintos en la tanda", level,
                  "la tanda mezcla píxeles: " + ", ".join(f"{p:g} nm" for p in pixels)))


def _time_checks(result: PrecheckResult, project: Project, accept_long: bool) -> None:
    from tools.method_costs import words_seconds

    v = project.values
    add = result.checks.append
    method = v["parametros.metodo"]
    total = 0.0
    lower = False
    for f in result.files:
        if not f.usable:
            continue
        n = f.n_slab_estimate if f.n_slab_estimate is not None else f.n_locs
        f.estimate = estimate_axon_seconds(method, int(n), f.n_clusters, null_on=bool(v["nulo.correr"]),
                                           replicates=int(v["nulo.replicas"]), closure=v["nulo.cierre"])
        total += f.estimate.total_s
        lower = lower or f.estimate.lower_bound
    result.total_seconds = total
    usable = [f for f in result.files if f.usable]
    if not usable:
        add(Check("C18", "Tiempo estimado", NA, "ningún archivo quedó para estimar"))
        return
    uncounted = [f.rel + (f": {f.cluster_count_note}" if f.cluster_count_note else "") for f in usable
                 if f.n_clusters is None]
    words = ("al menos " if lower else "") + words_seconds(total)
    cap = float(v["maquina.tope_horas"]) * 3600.0
    level = AVISO if total > cap else (AVISO if total > 0.8 * cap else OK)
    msg = (f"{len(usable)} axón(es), total estimado {words} (las dos pasadas: sin nulo y con nulo"
           f"{'' if v['nulo.correr'] else '; nulo apagado'}). Estimación de costo, no una medición.")
    if total > cap:
        msg += (f" Supera el tope de {v['maquina.tope_horas']:g} h por corrida: el lanzador cortará ENTRE axones y "
                f"se reanuda en otra corrida (F5); ningún axón se corta a la mitad.")
    if uncounted:
        msg += ("\nSin contar los clústeres (la cifra no incluye el perímetro y es una cota inferior): "
                + "; ".join(uncounted))
    add(Check("C18", "Tiempo estimado", level, msg))
    # P7: long axons
    accepted_list = list(v["aceptar_axones_largos"])
    for f in result.long_axons:
        if _accepted(f, accepted_list):
            f.long_accepted = "proyecto (aceptar_axones_largos)"
        elif accept_long:
            f.long_accepted = "línea de comandos (--aceptar-axones-largos)"
    longs = result.long_axons
    if not longs:
        add(Check("C23", "Axones de más de 2 h", OK, "ninguno"))
        return
    lines = [f"{f.rel}: {f.estimate.words() if f.estimate else '?'}"
             + (f" ({f.n_clusters} clústeres)" if f.n_clusters is not None else "")
             + (f" -- aceptado por {f.long_accepted}" if f.long_accepted else "") for f in longs]
    long_total = sum(f.estimate.total_s for f in longs if f.estimate)
    text = (f"{len(longs)} axón(es) estimado(s) en más de 2 h (total de esos: {words_seconds(long_total)}; total de "
            f"la tanda: {words}). Siempre se hace el análisis completo, sin tope por axón:\n" + "\n".join(lines))
    if result.long_unaccepted:
        text += ("\nPara correrlos sin supervisión: agregalos a aceptar_axones_largos en el proyecto, o lanzá con "
                 "--aceptar-axones-largos.")
        add(Check("C23", "Axones de más de 2 h", CONFIRMAR, text, [f.rel for f in result.long_unaccepted]))
    else:
        add(Check("C23", "Axones de más de 2 h", AVISO, text, [f.rel for f in longs]))


def _machine_checks(result: PrecheckResult, project: Project, probes: Probes) -> None:
    v = project.values
    add = result.checks.append
    ram = probes.ram_gb()
    floor = float(v["maquina.ram_minima_gb"])
    if ram is None:
        add(Check("C16", "RAM libre", AVISO, "no se pudo medir"))
    elif ram < floor:
        add(Check("C16", "RAM libre", ERROR, f"{ram:.1f} GB libres, por debajo de ram_minima_gb = {floor:g} GB; "
                                             f"cerrá programas y volvé a lanzar"))
    elif ram < RAM_WARN_GB:
        add(Check("C16", "RAM libre", AVISO, f"{ram:.1f} GB libres (menos de {RAM_WARN_GB:g} GB)"))
    else:
        add(Check("C16", "RAM libre", OK, f"{ram:.1f} GB libres"))
    salida = v["carpetas.salida"]
    disk = probes.disk_gb(salida)
    dfloor = float(v["maquina.disco_minimo_gb"])
    if disk is None:
        add(Check("C17", "Disco libre", AVISO, "no se pudo medir"))
    elif disk < dfloor:
        add(Check("C17", "Disco libre", ERROR, f"{disk:.1f} GB libres en la salida, por debajo de "
                                               f"disco_minimo_gb = {dfloor:g} GB"))
    else:
        add(Check("C17", "Disco libre", OK, f"{disk:.1f} GB libres en la salida"))
    if _in_onedrive(salida):
        add(Check("C17", "Salida dentro de OneDrive", AVISO,
                  "la carpeta de salida está en OneDrive: sincronizaría miles de escrituras durante la noche"))
    add(Check("C19", "Reanudar una corrida anterior", NA, "la reanudación llega con F5"))
    commit, dirty = probes.git(REPO)
    if commit is None:
        add(Check("C21", "Versión del programa", AVISO, "no se pudo leer el commit (¿fuera de git?)"))
    elif dirty:
        add(Check("C21", "Versión del programa", AVISO, f"commit {commit} con cambios sin commitear"))
    else:
        add(Check("C21", "Versión del programa", OK, f"commit {commit}"))


# ---------------------------------------------------------------------------
# The long-axon confirmation (P7)
# ---------------------------------------------------------------------------

def long_axons_text(result: PrecheckResult) -> str:
    """The list a person reads before accepting: each axon, its estimate, and the totals."""
    from tools.method_costs import words_seconds

    longs = result.long_unaccepted
    lines = ["Axones estimados en más de 2 horas (análisis completo, sin tope por axón):", ""]
    for f in longs:
        est = f.estimate
        extra = f" ({f.n_clusters} clústeres, {f.n_locs} localizaciones)" if f.n_clusters is not None else ""
        lines.append(f"  {f.rel}: {est.words() if est else '?'}{extra}")
    lines.append("")
    lines.append(f"Total de estos axones: {words_seconds(sum(f.estimate.total_s for f in longs if f.estimate))}")
    if result.total_seconds is not None:
        lines.append(f"Total de la tanda: {words_seconds(result.total_seconds)}")
    lines.append("")
    lines.append("Para aceptarlos sin supervisión, agregá al proyecto:")
    lines.append("aceptar_axones_largos:")
    lines.extend(f"  - '{f.rel}'" for f in longs)
    lines.append("o lanzá con --aceptar-axones-largos.")
    return "\n".join(lines)


def confirm_long_axons(result: PrecheckResult, ask: Callable[[str], str]) -> bool:
    """Ask once (``ask(text) -> answer``); "si" accepts every listed axon for this run. Returns True if accepted."""
    if not result.long_unaccepted:
        return True
    answer = str(ask(long_axons_text(result) + "\n\n¿Los aceptás para esta corrida? Escribí SI para aceptar: "))
    if answer.strip().lower() in ("si", "sí", "s", "yes", "y"):
        for f in result.long_unaccepted:
            f.long_accepted = "consola (respuesta del usuario)"
        for c in result.checks:
            if c.code == "C23" and c.level == CONFIRMAR:
                c.level = AVISO
                c.message += "\nAceptados en la consola por el usuario."
        return True
    return False


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

STATUS_WORDS = {"ok": "SE PUEDE ARRANCAR", "errores": "NO SE PUEDE ARRANCAR: hay errores",
                "confirmar": "FALTA CONFIRMAR: hay axones de más de 2 h sin aceptar"}


def report_text(result: PrecheckResult) -> str:
    from tools.method_costs import words_seconds

    p = result.project
    out: List[str] = []
    out.append(f"CHEQUEOS PREVIOS - {STATUS_WORDS[result.status]}")
    if p is not None:
        out.append(p.label())
        out.append("SIN ANIMAL" if not p.values["identidad.animales"] else "CON animales.csv (sin usar hasta F6)")
    out.append("")
    out.append(f"Proyecto: {result.project_path}")
    out.append(f"Fecha: {result.started}")
    if p is not None:
        v = p.values
        out.append(f"Tanda: {v['nombre_tanda']}   línea KO: {v['identidad.linea_ko']}   especie: "
                   f"{v['identidad.especie']}   {v['identidad.proteina_marcada']} / {v['identidad.secundario']}")
        out.append(f"Entrada: {v['carpetas.entrada']}")
        out.append(f"Salida: {v['carpetas.salida']}")
    n_err, n_conf, n_av = len(result.errors), len(result.by_level(CONFIRMAR)), len(result.warnings)
    out.append(f"Resultado: {n_err} error(es), {n_conf} a confirmar, {n_av} aviso(s)")
    out.append("")
    for level, title in ((ERROR, "ERRORES (frenan la corrida)"), (CONFIRMAR, "A CONFIRMAR"),
                         (AVISO, "AVISOS (la corrida sigue; quedan en el informe)"), (NOTA, "NOTAS")):
        items = result.by_level(level)
        if not items:
            continue
        out.append(title)
        out.append("-" * len(title))
        for c in items:
            out.append(f"[{c.code}] {c.title}")
            for line in c.message.splitlines():
                out.append(f"    {line}")
        out.append("")
    ok = [c for c in result.checks if c.level in (OK, NA)]
    if ok:
        out.append("SIN PROBLEMAS / NO APLICA")
        out.append("-------------------------")
        for c in ok:
            out.append(f"[{c.code}] {c.title}: {c.level} - {c.message.splitlines()[0] if c.message else ''}")
        out.append("")
    if result.files:
        out.append("ARCHIVOS")
        out.append("--------")
        for f in result.files:
            est = f.estimate.words() if f.estimate else "-"
            px = f"{f.pixel_nm:g} nm ({f.pixel_source})" if f.pixel_nm is not None else "sin píxel"
            out.append(f"{'  ' if f.usable else 'X '}{f.rel}")
            out.append(f"    genotipo {f.genotype or '?'} | {f.roi or '?'} / {f.axon or '?'} | muestra "
                       f"{f.sample or '-'} | {f.n_locs} loc. | {px} | clústeres "
                       f"{f.n_clusters if f.n_clusters is not None else '?'} | tiempo {est}")
        out.append("(X = no se analizaría)")
        if result.total_seconds is not None:
            out.append(f"Tiempo total estimado: {words_seconds(result.total_seconds)}")
        out.append("")
    if result.left_out_by_filter:
        out.append("Dejados afuera por el filtro del nombre:")
        out.extend(f"    {_rel(p, result.project.values['carpetas.entrada']) if result.project else p}"
                   for p in result.left_out_by_filter)
        out.append("")
    out.append("Origen de las cifras de tiempo: tabla de costos del programa (tools/method_costs.py, own sim + ad hoc) "
               "y ajuste del perímetro (tools/mps_perimeter_cost.py, own sim). Son estimaciones; el lanzador las "
               "recalcula con el tiempo real del primer axón.")
    return "\n".join(out) + "\n"


CSV_COLUMNS = ("code", "title", "level", "message", "files")


def write_report(result: PrecheckResult, folder: str) -> Tuple[str, str]:
    """Write ``chequeos_previos.txt`` and ``chequeos_previos.csv`` in ``folder`` (created). Never overwrites."""
    os.makedirs(folder, exist_ok=True)
    txt = os.path.join(folder, "chequeos_previos.txt")
    csv_path = os.path.join(folder, "chequeos_previos.csv")
    for path in (txt, csv_path):
        if os.path.exists(path):
            raise FileExistsError(path)
    with open(txt, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report_text(result))
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for c in result.checks:
            writer.writerow([c.code, c.title, c.level, c.message, " | ".join(c.files)])
    return txt, csv_path


def report_folder(result: PrecheckResult, now: Optional[datetime] = None) -> str:
    """A new folder for this check's report: ``<salida>/chequeos/<tanda>_<date_time>`` (or beside the project file
    when the project could not be read). A suffix keeps an existing folder untouched."""
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    p = result.project
    if p is not None:
        base = os.path.join(p.values["carpetas.salida"], "chequeos", f"{p.values['nombre_tanda']}_{stamp}")
    else:
        stem = os.path.splitext(os.path.basename(result.project_path))[0]
        base = os.path.join(os.path.dirname(result.project_path), f"chequeos_{stem}_{stamp}")
    folder, k = base, 2
    while os.path.exists(folder):
        folder, k = f"{base}_{k}", k + 1
    return folder


__all__ = ["Check", "FileInfo", "PrecheckResult", "Probes", "TimeEstimate", "run_prechecks", "estimate_axon_seconds",
           "confirm_long_axons", "long_axons_text", "report_text", "write_report", "report_folder",
           "LONG_AXON_SECONDS", "ERROR", "AVISO", "CONFIRMAR", "NOTA", "OK", "NA", "SPEC"]
