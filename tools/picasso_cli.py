# -*- coding: utf-8 -*-
"""
Calling Picasso as an external program, for the things not worth porting.

Some of what Picasso does is small enough to reimplement and check against
its source -- linking, NeNA, the sticking filter, all of which live in
``tools.mps_paint`` and ``tools.mps_quality``. Three things are not:

  AIM        fiducial-free drift correction that also corrects z. Picasso's
             RCC undrifting is 2D and never touches the axial direction, so
             a file undrifted "by RCC" has had no axial correction at all.
             (Ma et al., Sci Adv 2024, DOI 10.1126/sciadv.adm7765)
  G5M        molecular mapping with each component's sigma bounded by the
             localization precision, which is the tool for deciding whether
             one broad axial component is one ring or two unresolved ones.
             (Kowalewski, Reinhardt et al., Nat Commun 2026,
             DOI 10.1038/s41467-026-70198-5)
  SMLM       clustering with separate lateral and axial radii in 3D.
  clusterer  MPS Explorer clusters in 2D inside a z slab and cannot do this.

Why a subprocess and not an import
----------------------------------
``picassosr`` pins ``pandas<3`` and ``scikit-learn<1.8``; this project runs
pandas 3.0.5 and scikit-learn 1.9.0. Installing it into the same
environment would force both down. A subprocess has no version coupling at
all: Picasso reads an HDF5 and writes an HDF5, and the only contract
between us is the file format.

THE UNIT BOUNDARY
-----------------
Picasso's command line speaks CAMERA PIXELS -- ``intersectdist``,
``roiradius``, ``radius`` and ``radius_z`` are all pixels. Everything in
MPS Explorer is nanometres. Every function here therefore takes
nanometres and converts, because a bridge that passed nanometres straight
through would be off by the pixel size (113x, or 122x, or 130x) with
nothing downstream able to notice. The pixel size comes from the file's
own metadata; when that is missing the call is refused rather than
guessed, exactly as in ``tools.mps_io``.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import glob
import locale
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tools import mps_metadata

# Where a Windows one-click installer puts it, plus the obvious ones. The
# PATH is tried first; these are the fallback so the feature works without
# the user having to configure anything.
_CANDIDATE_PATHS: Tuple[str, ...] = (
    r"C:\Picasso\picasso.exe",
    r"C:\Program Files\Picasso\picasso.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Picasso\picasso.exe"),
    "/Applications/Picasso.app/Contents/MacOS/picasso",
)

# The frozen one-click build takes several seconds just to start, before it
# does any work, so every timeout here is generous.
DEFAULT_TIMEOUT_S = 1800
PROBE_TIMEOUT_S = 180


class PicassoNotFound(RuntimeError):
    """No Picasso executable could be located."""


class PicassoFailed(RuntimeError):
    """Picasso ran but did not produce what was asked of it."""


class PicassoCancelled(PicassoFailed):
    """The run was stopped with ``cancel_running`` before it finished."""


# Without this, every call from a GUI started with pythonw flashes a
# console window.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The frozen Windows build writes to a pipe in the ANSI code page, and
# ignores PYTHONIOENCODING and PYTHONUTF8.
_CHILD_ENCODING = (
    "mbcs" if sys.platform == "win32"
    else locale.getpreferredencoding(False)
)

# Subcommands that never print the input path (in 0.9.10 link prints only
# inside a try), so a path the code page cannot encode is no problem there.
_PATH_NEVER_PRINTED = frozenset({"link"})

# Subcommands that take the input path literally. Every other one passes it
# through glob.glob, where "[" and "]" -- legal in Windows names -- are a
# pattern, so the literal file is missed or a different file is processed.
_NO_GLOB_COMMANDS = frozenset({"g5m"})

# Processes started by run_command and not yet finished, so that a GUI can
# stop them from another thread.
_RUNNING: Dict[int, "subprocess.Popen[str]"] = {}
_CANCELLED: set = set()
_LOCK = threading.Lock()


def _kill_tree(proc: "subprocess.Popen[str]") -> None:
    # G5M fits clusters in worker processes; killing only the parent would
    # leave them running.
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, creationflags=_NO_WINDOW,
        )
    if proc.poll() is None:
        proc.kill()


def cancel_running() -> int:
    """Stop every Picasso run in progress. Returns how many were stopped."""
    with _LOCK:
        procs = list(_RUNNING.values())
        _CANCELLED.update(proc.pid for proc in procs)
    for proc in procs:
        try:
            _kill_tree(proc)
        except OSError:
            pass
    return len(procs)


@dataclass
class PicassoInstall:
    """A located Picasso executable and what it can do."""

    executable: str
    commands: Tuple[str, ...]

    def has(self, command: str) -> bool:
        return command in self.commands


@dataclass
class PicassoRun:
    """One completed invocation."""

    command: str
    args: List[str]
    returncode: int
    stdout: str
    stderr: str
    outputs: Dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0


# ===================================================================
#  Finding Picasso
# ===================================================================
def find_picasso(explicit: Optional[str] = None) -> Optional[PicassoInstall]:
    """
    Locate a Picasso executable, or return None.

    With ``explicit``, only that path is tried: a configured path that is
    not Picasso is reported as such rather than silently replaced by some
    other install. Without it, the PATH and then the usual install
    locations are searched. Returns None rather than raising, so a caller
    can disable the feature quietly: Picasso is genuinely optional here
    and its absence is not an error.

    The command list is read from ``picasso --help`` rather than assumed,
    because which subcommands exist varies between versions and an
    optimistic call would fail deep inside a long run instead of at the
    start.
    """
    if explicit:
        candidates: List[str] = [explicit]
    else:
        candidates = [shutil.which("picasso") or "", *_CANDIDATE_PATHS]

    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        commands = _probe_commands(candidate)
        if commands:
            return PicassoInstall(executable=candidate, commands=commands)
    return None


def _probe_commands(executable: str) -> Tuple[str, ...]:
    """The subcommands ``executable`` advertises, or () if it is not Picasso."""
    try:
        done = subprocess.run(
            [executable, "--help"], stdin=subprocess.DEVNULL,
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=PROBE_TIMEOUT_S, creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    text = (done.stdout or "") + (done.stderr or "")
    if "picasso" not in text.lower():
        return ()
    # argparse prints the subcommands inside braces in the usage line.
    start = text.find("{")
    end = text.find("}", start + 1)
    if start < 0 or end < 0:
        return ()
    return tuple(
        part.strip() for part in text[start + 1:end].split(",") if part.strip()
    )


# ===================================================================
#  Running a command
# ===================================================================
def _pixel_size_for(path: str, override: Optional[float]) -> float:
    """
    The camera pixel size to convert with, or a refusal.

    Every distance handed to Picasso is in camera pixels. Guessing this
    number would rescale the clustering radius, the intersection distance
    and the drift bound all at once, and the run would complete and
    produce a plausible file.
    """
    if override is not None:
        return float(override)
    info, _source = mps_metadata.load_metadata(path)
    value = mps_metadata.pixel_size_nm(info)
    if value is None:
        raise PicassoFailed(
            f"{os.path.basename(path)} carries no pixel size, and every "
            f"distance Picasso takes is in camera pixels. Pass "
            f"pixel_size_nm explicitly if you know it for this acquisition."
        )
    return float(value)


def _expected(path: str, suffixes: Dict[str, str]) -> Dict[str, str]:
    base = os.path.splitext(path)[0]
    return {name: base + suffix for name, suffix in suffixes.items()}


def _tail(result: "PicassoRun") -> str:
    detail = (result.stderr or result.stdout or "").strip()
    return f"\n{detail[-1500:]}" if detail else ""


def hdf5_columns(path: str) -> Tuple[str, ...]:
    """Column names of a Picasso HDF5, or () if it cannot be read."""
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            return tuple(handle["locs"].dtype.names or ())
    except (OSError, KeyError, ValueError):
        return ()


def run_command(
    command: str,
    path: str,
    extra: Sequence[object],
    outputs: Dict[str, str],
    *,
    install: Optional[PicassoInstall] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> PicassoRun:
    """
    Run one Picasso subcommand on one file and collect its outputs.

    Parameters
    ----------
    command : the subcommand, e.g. "aim".
    path : the input .hdf5.
    extra : arguments after the file; each is passed as ``str(value)``.
    outputs : {name: suffix} of the files Picasso is expected to write,
        e.g. {"locs": "_aim.hdf5", "drift": "_aimdrift.txt"}.

    Returns
    -------
    PicassoRun, whose ``outputs`` maps each name to a path that was
    verified to exist AND to have been written by this run.

    That verification matters more than it looks. Picasso writes its
    output next to the input under a fixed name, so a previous run leaves
    a file there. Without checking the modification time, a failed run
    would hand back a stale file from an earlier attempt -- with different
    parameters -- and nothing downstream could tell.
    """
    if install is None:
        install = find_picasso()
    if install is None:
        raise PicassoNotFound(
            "No Picasso executable found. Install it from "
            "https://github.com/jungmannlab/picasso/releases, or point at "
            "one explicitly. Everything else in MPS Explorer works without "
            "it; only the Picasso-backed tools are affected."
        )
    if not install.has(command):
        raise PicassoFailed(
            f"This Picasso build has no '{command}' command. It offers: "
            f"{', '.join(install.commands)}."
        )
    if not os.path.exists(path):
        raise PicassoFailed(f"{path} does not exist.")
    try:
        if command not in _PATH_NEVER_PRINTED:
            path.encode(_CHILD_ENCODING)
    except UnicodeEncodeError:
        # Picasso prints the path before doing any work, and the print
        # fails on a character the code page lacks (a Greek letter, say).
        raise PicassoFailed(
            f"Picasso cannot handle this path, because it contains a "
            f"character outside the Windows code page ({_CHILD_ENCODING}):\n"
            f"{path}\nMove or rename the file so that its full path has no "
            f"such characters."
        )

    expected = _expected(path, outputs)
    target = os.path.normcase(os.path.abspath(path))
    for name, candidate in expected.items():
        if os.path.normcase(os.path.abspath(candidate)) == target:
            raise PicassoFailed(
                f"picasso {command} would write its '{name}' output over the "
                f"input file {path}."
            )
    started = time.time()
    # A file already sitting at an output path is from an earlier run.
    # Recording its timestamp is what lets a stale result be told apart
    # from a fresh one below.
    before = {
        name: (os.path.getmtime(p) if os.path.exists(p) else None)
        for name, p in expected.items()
    }

    literal = path if command in _NO_GLOB_COMMANDS else glob.escape(path)
    args = [install.executable, command, literal, *[str(a) for a in extra]]
    try:
        proc = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, encoding=_CHILD_ENCODING,
            errors="replace", creationflags=_NO_WINDOW,
        )
    except OSError as error:
        raise PicassoFailed(f"could not start {install.executable}: {error}")
    with _LOCK:
        _RUNNING[proc.pid] = proc
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        raise PicassoFailed(
            f"picasso {command} did not finish within {timeout_s} s. The "
            f"one-click build is slow to start and G5M in particular can "
            f"take a long time on large clusters; raise timeout_s if the "
            f"run was simply long."
        )
    finally:
        with _LOCK:
            _RUNNING.pop(proc.pid, None)
            cancelled = proc.pid in _CANCELLED
            _CANCELLED.discard(proc.pid)
    elapsed = time.time() - started
    if cancelled:
        raise PicassoCancelled(
            f"picasso {command} was stopped after {elapsed:.0f} s. Any "
            f"output it had started writing is incomplete."
        )

    produced: Dict[str, str] = {}
    for name, candidate in expected.items():
        if not os.path.exists(candidate):
            continue
        stamp = os.path.getmtime(candidate)
        if before[name] is not None and stamp <= before[name]:
            continue          # untouched: a leftover, not this run's output
        produced[name] = candidate

    result = PicassoRun(
        command=command, args=args[1:], returncode=proc.returncode,
        stdout=stdout or "", stderr=stderr or "",
        outputs=produced, seconds=elapsed,
    )
    # Every expected output, not just one: a run that wrote the localizations
    # but died before the drift file is a failed run.
    missing = sorted(set(expected) - set(produced))
    if result.ok and not produced:
        raise PicassoFailed(
            f"picasso {command} finished without writing anything, after "
            f"{elapsed:.0f} s. Picasso skips, silently, a file whose "
            f"metadata (.yaml) it cannot read."
            + _tail(result)
        )
    if not result.ok or missing:
        detail = (result.stderr or result.stdout or "").strip()
        raise PicassoFailed(
            f"picasso {command} failed (exit {proc.returncode}) after "
            f"{elapsed:.0f} s"
            + (f"; no fresh output for {missing}" if missing else "")
            + (f".\n{detail[-1500:]}" if detail else ".")
        )
    return result


# ===================================================================
#  The three tools worth bridging
# ===================================================================
def undrift_aim(
    path: str,
    *,
    segmentation: int = 100,
    intersect_distance_nm: Optional[float] = None,
    max_drift_nm: Optional[float] = None,
    pixel_size_nm: Optional[float] = None,
    install: Optional[PicassoInstall] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> PicassoRun:
    """
    Fiducial-free drift correction with AIM, which also corrects z.

    This is the reason the bridge exists. Picasso's RCC undrifting -- the
    ``_byRCC1000`` in this project's own filenames -- is two-dimensional
    and never touches z, so an axial measurement made on an RCC-undrifted
    file has had no axial drift correction whatsoever. AIM has a 3D
    branch.

    Parameters
    ----------
    segmentation : frames per temporal segment. Lower means finer drift
        tracking and a longer run; it also needs enough localizations per
        segment to find the intersections, so a sparse acquisition wants a
        larger value.
    intersect_distance_nm : how close two localizations in consecutive
        segments must be to count as the same molecule. Picasso's guidance
        is 3 x NeNA; the default here is its own default of 20/130 camera
        pixels converted through this file's pixel size.
    max_drift_nm : the largest drift expected between two consecutive
        segments. Too small and the algorithm diverges; up to three times
        the intersection distance keeps it fast.

    Returns
    -------
    PicassoRun with outputs "locs" (the undrifted .hdf5) and "drift" (the
    per-frame drift .txt).
    """
    pixel = _pixel_size_for(path, pixel_size_nm)
    # Picasso's own defaults, expressed in nm so they travel between
    # cameras: 20/130 px and 60/130 px at the 130 nm pixel they were
    # written for.
    intersect_nm = 20.0 if intersect_distance_nm is None \
        else float(intersect_distance_nm)
    drift_nm = 60.0 if max_drift_nm is None else float(max_drift_nm)

    return run_command(
        "aim", path,
        ["-s", int(segmentation),
         "-i", intersect_nm / pixel,
         "-r", drift_nm / pixel],
        {"locs": "_aim.hdf5", "drift": "_aimdrift.txt"},
        install=install, timeout_s=timeout_s,
    )


def smlm_cluster(
    path: str,
    *,
    radius_nm: float,
    radius_z_nm: Optional[float] = None,
    min_locs: int = 10,
    frame_analysis: bool = False,
    pixel_size_nm: Optional[float] = None,
    install: Optional[PicassoInstall] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> PicassoRun:
    """
    Picasso's SMLM clusterer, with separate lateral and axial radii in 3D.

    MPS Explorer clusters in 2D inside an axial slab, which is the right
    thing when the slab holds one ring, and the wrong thing when it does
    not. The anisotropy is not cosmetic: lateral precision here is ~8 nm
    and axial ~47 nm, so an isotropic radius is either far too tight in z
    or far too loose in xy.

    Both radii are given in NANOMETRES. ``radius_z_nm`` is required for a
    3D file and must be left out for a 2D one; Picasso decides which it is
    from the presence of a ``z`` column.

    ``frame_analysis`` is Picasso's sticking filter -- the same rule
    ``tools.mps_paint.frame_analysis`` implements.

    Picasso's command line has three traps, handled here. The three
    trailing arguments are required positionals despite their defaults.
    ``basic_fa`` is parsed with ``type=bool``, so the string "False" means
    True; only an empty string switches it off. And the pixel size must be
    an integer: z is scaled by that rounded value, so the axial radius is
    converted with it too.

    Returns
    -------
    PicassoRun with outputs "locs" (localizations with a ``group`` column)
    and "centers" (one row per cluster).
    """
    pixel = _pixel_size_for(path, pixel_size_nm)
    picasso_pixel = int(round(pixel))
    is_3d = "z" in hdf5_columns(path)
    if is_3d and radius_z_nm is None:
        raise ValueError(
            f"{os.path.basename(path)} is 3D: an axial radius is required.")
    if not is_3d and radius_z_nm is not None:
        raise ValueError(
            f"{os.path.basename(path)} has no z column, so an axial radius "
            f"would be ignored. Leave radius_z_nm out.")
    radius_px = float(radius_nm) / pixel
    # Unused for 2D data, but the argument is not optional.
    radius_z_px = (radius_px if radius_z_nm is None
                   else float(radius_z_nm) / picasso_pixel)
    return run_command(
        "smlm_cluster", path,
        [radius_px, int(min_locs), picasso_pixel,
         "1" if frame_analysis else "", radius_z_px],
        {"locs": "_clusters.hdf5", "centers": "_cluster_centers.hdf5"},
        install=install, timeout_s=timeout_s,
    )


def molecular_map(
    path: str,
    *,
    min_locs: int = 10,
    min_sigma: float = 0.8,
    max_sigma: float = 1.5,
    loc_prec_handle: str = "local",
    calibration_path: Optional[str] = None,
    postprocess: bool = True,
    pixel_size_nm: Optional[float] = None,
    install: Optional[PicassoInstall] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> PicassoRun:
    """
    Molecular mapping with G5M: a mixture whose component widths are
    bounded by the localization precision.

    This is the one that speaks to the open question in this project. A
    plain GMM on z can make a component as wide as it likes, and on this
    data they come out ~1.8x the axial precision -- which leaves "one
    thick ring" and "two unresolved rings" indistinguishable. G5M forbids
    a component wider than ``max_sigma`` times the local precision, so a
    cloud that cannot be one emitter has to be fitted as two.

    REQUIRES CLUSTERED INPUT. G5M reads the ``group`` column, so run
    ``smlm_cluster`` (or Picasso's dbscan) first. Picasso's own guidance
    is DBSCAN at 2x the localization precision in 2D, 3x in 3D.

    ``min_sigma`` / ``max_sigma`` are FACTORS of the local precision when
    ``loc_prec_handle`` is "local", and NANOMETRES when it is "abs"
    (converted here; Picasso takes camera pixels). The defaults are
    Picasso's own.

    ``calibration_path`` is the astigmatism calibration .yaml. It is
    required for 3D data: without it Picasso 0.9.10 fails with an
    unbound variable instead of a message.

    ``postprocess`` (Picasso's default) filters the map before writing
    it: see ``G5M_FILTERS`` and ``molmap_summary``. One of the three
    filters is a sticking test on the spread of frames, and on a short
    acquisition it can remove nearly everything -- 94 of 166 molecules
    failed that test alone on the 3,412-frame 15.07.26 sample.

    Returns
    -------
    PicassoRun with output "molmap": one row per molecule, carrying
    ``n_events``, ``p_val``, ``std_frame`` and the fitted sigmas.
    """
    if loc_prec_handle not in ("local", "abs"):
        raise ValueError("loc_prec_handle must be 'local' or 'abs'.")
    # Picasso names the output with a case-sensitive replace over the whole
    # path. For "a.HDF5" that replaces nothing and the map is written OVER
    # the input; for a folder named "x.hdf5" it lands in another folder.
    if path.replace(".hdf5", "_molmap.hdf5") != \
            os.path.splitext(path)[0] + "_molmap.hdf5":
        raise ValueError(
            f"G5M cannot be run on {path} as it is named: Picasso would "
            f"write the result over the input or into another folder. "
            f"Give the file a lower-case .hdf5 extension, in a folder whose "
            f"name does not contain '.hdf5'.")
    if os.path.exists(path):
        columns = hdf5_columns(path)
        if "group" not in columns:
            raise ValueError(
                f"{os.path.basename(path)} has no 'group' column. G5M maps "
                f"molecules cluster by cluster: cluster the file first.")
        if "z" in columns and not calibration_path:
            raise ValueError(
                f"{os.path.basename(path)} is 3D: G5M needs the astigmatism "
                f"calibration .yaml it was localized with.")
    low, high = float(min_sigma), float(max_sigma)
    if loc_prec_handle == "abs":
        pixel = _pixel_size_for(path, pixel_size_nm)
        low, high = low / pixel, high / pixel
    extra: List[Any] = [
        "-ml", int(min_locs),
        "-lph", loc_prec_handle,
        "--min-sigma", low,
        "--max-sigma", high,
    ]
    if calibration_path:
        extra += ["-c", calibration_path]
    if not postprocess:
        # The flag READS as "do not postprocess"; passing it disables the
        # removal of sticking events and low-quality fits.
        extra.append("-p")
    return run_command(
        "g5m", path, extra, {"molmap": "_molmap.hdf5"},
        install=install, timeout_s=timeout_s,
    )


# Picasso's G5M postprocessing (identical in 0.9.10 and 0.11.1). All three
# comparisons are strict: "more than 3 events" means at least 4.
G5M_FILTERS = {
    "std_frame_fraction": 0.1,   # spread of frames > 10% of the acquisition
    "p_val": 0.015,
    "n_events": 3,
}


@dataclass
class MolmapSummary:
    """What a G5M molecular map holds, and what its filters did."""

    n_molecules: int
    n_fitted: Optional[int]          # before filtering, from the metadata
    filtered: bool
    n_frames: Optional[int]
    # For an UNFILTERED map: how many molecules pass each filter alone, and
    # all three together. None for a filtered map, whose rejects are gone.
    passing: Optional[Dict[str, int]] = None

    def lines(self) -> List[str]:
        out = []
        noun = "molecule" if self.n_molecules == 1 else "molecules"
        if self.filtered:
            fitted = (f" of {self.n_fitted} fitted"
                      if self.n_fitted is not None else "")
            out.append(f"{self.n_molecules} {noun} kept{fitted} by "
                       f"Picasso's filters.")
        else:
            out.append(f"{self.n_molecules} {noun}, unfiltered.")
        limit = (f"{G5M_FILTERS['std_frame_fraction'] * self.n_frames:.0f} "
                 f"frames" if self.n_frames else "10% of the acquisition")
        rules = {
            "std_frame": f"spread of frames above {limit} (sticking test)",
            "p_val": f"p value above {G5M_FILTERS['p_val']}",
            "n_events": f"more than {G5M_FILTERS['n_events']} binding events",
        }
        if self.passing is None:
            out.append("Filters: " + "; ".join(rules.values()) + ".")
        else:
            for key, rule in rules.items():
                out.append(f"  {self.passing[key]} pass: {rule}")
            out.append(f"  {self.passing['all']} pass all three")
        return out


def molmap_summary(path: str) -> MolmapSummary:
    """Read a ``_molmap.hdf5`` and account for Picasso's filtering."""
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        mols = handle["locs"][:]
    info, _ = mps_metadata.load_metadata(path)
    last = info[-1] if info else {}
    first = info[0] if info else {}
    filtered = bool(last.get("Filtered", False))
    n_frames = first.get("Frames")
    fitted = last.get("Number of molecules")
    passing = None
    names = mols.dtype.names or ()
    if not filtered and n_frames and {"std_frame", "p_val",
                                      "n_events"} <= set(names):
        tests = {
            "std_frame": mols["std_frame"]
            > G5M_FILTERS["std_frame_fraction"] * float(n_frames),
            "p_val": mols["p_val"] > G5M_FILTERS["p_val"],
            "n_events": mols["n_events"] > G5M_FILTERS["n_events"],
        }
        passing = {k: int(np.count_nonzero(v)) for k, v in tests.items()}
        passing["all"] = int(np.count_nonzero(
            tests["std_frame"] & tests["p_val"] & tests["n_events"]))
    return MolmapSummary(
        n_molecules=len(mols),
        n_fitted=int(fitted) if fitted is not None else None,
        filtered=filtered,
        n_frames=int(n_frames) if n_frames else None,
        passing=passing,
    )


def link_localizations(
    path: str,
    *,
    distance_nm: float,
    max_dark_time: int = 1,
    pixel_size_nm: Optional[float] = None,
    install: Optional[PicassoInstall] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> PicassoRun:
    """
    Picasso's own linking, for checking ours against it.

    ``tools.mps_paint.link_localizations`` reimplements the same rule, and
    given the same input the events agree one for one.
    ``mps_paint.build_events`` then departs on purpose, as its docstring
    explains: it keeps the events Picasso loses to a zero precision, drops
    events still bound in the last frame, and drops events made only of
    failed fits. validate_picasso_cli.py checks that these are the only
    differences.

    Returns
    -------
    PicassoRun with output "locs" ({base}_link.hdf5).
    """
    pixel = _pixel_size_for(path, pixel_size_nm)
    return run_command(
        "link", path,
        ["-d", float(distance_nm) / pixel, "-t", int(max_dark_time)],
        {"locs": "_link.hdf5"},
        install=install, timeout_s=timeout_s,
    )


def describe(install: Optional[PicassoInstall] = None) -> str:
    """One line saying whether the bridge is usable, for a status bar."""
    if install is None:
        install = find_picasso()
    if install is None:
        return (
            "Picasso not found - AIM undrifting, 3D clustering and G5M are "
            "unavailable. Everything else works."
        )
    wanted = ("aim", "smlm_cluster", "g5m", "link")
    missing = [c for c in wanted if not install.has(c)]
    return (
        f"Picasso at {install.executable}"
        + (f" (missing: {', '.join(missing)})" if missing else "")
    )
