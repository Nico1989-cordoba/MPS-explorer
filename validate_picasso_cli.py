# -*- coding: utf-8 -*-
"""
Checks for the Picasso subprocess bridge.

Most of it runs against a FAKE picasso -- a small script that advertises
the same subcommands and writes files where the real one would. That is
deliberate: the interesting failures here are about the bridge's own
bookkeeping (did the unit conversion happen, was a stale output mistaken
for a fresh one, is a missing install reported rather than crashed), and
those must be testable on a machine that has no Picasso at all.

When a real Picasso IS installed, the last section runs `aim` and `link`
against it for real.

Run:  python validate_picasso_cli.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import picasso_cli  # noqa: E402
from tools.picasso_cli import (  # noqa: E402
    PicassoCancelled,
    PicassoFailed,
    PicassoInstall,
    PicassoNotFound,
    cancel_running,
    describe,
    find_picasso,
    link_localizations,
    molecular_map,
    run_command,
    smlm_cluster,
    undrift_aim,
)

PASSED = 0
FAILED = 0
_TEMP_DIRS: list = []


def new_tmp(prefix: str) -> str:
    path = tempfile.mkdtemp(prefix=prefix)
    _TEMP_DIRS.append(path)
    return path

REAL_DATA = os.environ.get(
    "MPS_PAINT_SAMPLE",
    r"C:\Users\nicol\OneDrive\Doctorado\15.07.26"
    r"\260713_DNAPAINT_NCtransversal_bIIspt_TIRF4_Roi2_2_1"
    r"\260713_DNAPAINT_NCtransversal_bIIspt_TIRF4_Roi2_2_1_MMStack.ome_locs.hdf5",
)


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


# ----------------------------------------------------------- fake picasso
_FAKE = textwrap.dedent(
    '''
    import json, os, sys, time
    ARGS = sys.argv[1:]
    COMMANDS = "localize,undrift,aim,link,dbscan,smlm_cluster,g5m,dark"
    if not ARGS or ARGS[0] in ("--help", "-h"):
        print("usage: picasso [-h] {" + COMMANDS + "} ...")
        sys.exit(0)
    command, rest = ARGS[0], ARGS[1:]
    # Like the real one: g5m takes the path literally, everything else
    # passes it through glob and quietly does nothing when it matches
    # nothing.
    import glob
    paths = [rest[0]] if command == "g5m" else glob.glob(rest[0])
    time.sleep(float(os.environ.get("FAKE_PICASSO_SLEEP", "0")))
    if os.environ.get("FAKE_PICASSO_FAIL") == "1":
        sys.stderr.write("simulated failure\\n")
        sys.exit(2)
    for path in paths:
        base = os.path.splitext(path)[0]
        # Record what we were called with, so a test can read it back.
        with open(base + ".called", "w") as handle:
            json.dump(rest[1:], handle)
        if os.environ.get("FAKE_PICASSO_NOWRITE") == "1":
            continue
        targets = {
            "aim": [base + "_aim.hdf5", base + "_aimdrift.txt"],
            "link": [base + "_link.hdf5"],
            "smlm_cluster": [base + "_clusters.hdf5",
                             base + "_cluster_centers.hdf5"],
            # The real naming, case-sensitive replace and all.
            "g5m": [path.replace(".hdf5", "_molmap.hdf5")],
        }.get(command, [])
        if os.environ.get("FAKE_PICASSO_PARTIAL") == "1":
            targets = targets[:1]
        for target in targets:
            with open(target, "w") as handle:
                handle.write("written by the fake\\n")
    sys.exit(0)
    '''
).strip()


def make_fake(directory: str) -> PicassoInstall:
    """A fake picasso executable and an install pointing at it."""
    script = os.path.join(directory, "fake_picasso.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(_FAKE)
    launcher = os.path.join(directory, "fake_picasso.cmd")
    with open(launcher, "w", encoding="utf-8") as handle:
        handle.write(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    install = find_picasso(explicit=launcher)
    if install is None:
        raise AssertionError("the fake was not recognised as a picasso")
    return install


def make_input(directory: str, name: str = "sample.hdf5",
               columns: tuple = ("frame", "x", "y", "lpx", "lpy")) -> str:
    """A small real HDF5 with the given columns and a 130 nm pixel."""
    import h5py

    path = os.path.join(directory, name)
    dtype = [(c, "u4" if c in ("frame", "group") else "f4") for c in columns]
    rows = np.zeros(5, dtype=dtype)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("locs", data=rows)
    with open(os.path.splitext(path)[0] + ".yaml", "w", encoding="utf-8") as h:
        h.write("Frames: 1000\nWidth: 256\nHeight: 256\n---\n")
        h.write("Generated by: Picasso v0.9.10 Localize\nPixelsize: 130\n")
    return path


def called_args(path: str) -> list:
    with open(os.path.splitext(path)[0] + ".called", encoding="utf-8") as h:
        return list(json.load(h))


def called_with(path: str) -> str:
    return " ".join(called_args(path))


class env:
    """Set environment variables for the fake inside a with-block."""

    def __init__(self, **values: str) -> None:
        self.values = values

    def __enter__(self) -> None:
        os.environ.update(self.values)

    def __exit__(self, *exc: object) -> None:
        for key in self.values:
            os.environ.pop(key, None)


# ================================================================ discovery
def test_discovery() -> None:
    print("\n1. FINDING PICASSO")
    tmp = new_tmp("pc_find_")

    def recognises_a_picasso():
        install = make_fake(tmp)
        assert "aim" in install.commands
        assert install.has("g5m") and not install.has("nonsense")
        return f"{len(install.commands)} commands parsed from --help"

    def rejects_something_else():
        other = os.path.join(tmp, "not_picasso.cmd")
        with open(other, "w", encoding="utf-8") as handle:
            handle.write("@echo off\r\necho hello world\r\n")
        assert find_picasso(explicit=other) is None
        return "an unrelated executable is not accepted"

    def missing_path_is_none():
        assert find_picasso(explicit=os.path.join(tmp, "nope.exe")) is None
        return None

    def describe_reads_either_way():
        full = describe(make_fake(tmp))
        assert "fake_picasso" in full and "missing" not in full, full
        partial = describe(PicassoInstall("x", ("aim",)))
        assert "missing: smlm_cluster, g5m, link" in partial, partial
        real_find = picasso_cli.find_picasso
        picasso_cli.find_picasso = lambda explicit=None: None
        try:
            absent = describe()
        finally:
            picasso_cli.find_picasso = real_find
        assert "not found" in absent.lower(), absent
        return "complete, partial and absent installs all described"

    check("a picasso is recognised by its --help", recognises_a_picasso)
    check("a non-picasso executable is rejected", rejects_something_else)
    check("a missing path gives None", missing_path_is_none)
    check("describe()", describe_reads_either_way)


# ============================================================ unit boundary
def test_units() -> None:
    print("\n2. THE UNIT BOUNDARY  (nm in, camera pixels out)")
    tmp = new_tmp("pc_units_")
    install = make_fake(tmp)

    def aim_converts():
        path = make_input(tmp, "aim.hdf5")
        undrift_aim(path, segmentation=250, intersect_distance_nm=26.0,
                    max_drift_nm=78.0, install=install)
        args = called_with(path)
        # 26 / 130 = 0.2 ; 78 / 130 = 0.6
        assert "-i 0.2" in args, args
        assert "-r 0.6" in args, args
        assert "-s 250" in args, args
        return "26 nm -> 0.2 px at a 130 nm pixel"

    xyz = ("frame", "x", "y", "z", "lpx", "lpy", "lpz")

    def cluster_converts_both_radii():
        path = make_input(tmp, "cl.hdf5", xyz)
        smlm_cluster(path, radius_nm=26.0, radius_z_nm=65.0, min_locs=12,
                     install=install)
        # positional: radius min_locs pixelsize basic_fa radius_z
        args = called_args(path)
        assert args == ["0.2", "12", "130", "", "0.5"], args
        return "lateral and axial radii both converted"

    def cluster_bool_and_integer_traps():
        path = make_input(tmp, "cl2.hdf5")
        smlm_cluster(path, radius_nm=26.0, frame_analysis=True,
                     install=install)
        args = called_args(path)
        # 2D: the axial slot is a placeholder, frame analysis is "1".
        assert args == ["0.2", "10", "130", "1", "0.2"], args
        path3 = make_input(tmp, "cl3.hdf5", xyz)
        smlm_cluster(path3, radius_nm=21.5, radius_z_nm=54.0,
                     pixel_size_nm=107.5, install=install)
        args = called_args(path3)
        # z is scaled by Picasso's rounded pixel (108), so the axial radius
        # is converted with it; the lateral one with the exact 107.5.
        assert args == ["0.2", "10", "108", "", "0.5"], args
        return 'False is sent as "", pixel size as an integer'

    def cluster_dimension_is_checked():
        flat = make_input(tmp, "flat.hdf5")
        deep = make_input(tmp, "deep.hdf5", xyz)
        for path, kwargs in ((flat, {"radius_z_nm": 50.0}), (deep, {})):
            try:
                smlm_cluster(path, radius_nm=20.0, install=install, **kwargs)
            except ValueError:
                continue
            raise AssertionError(f"{os.path.basename(path)} {kwargs} ran")
        return "axial radius required in 3D and refused in 2D"

    def g5m_bounds_and_inputs():
        grouped = make_input(tmp, "g.hdf5",
                             ("frame", "x", "y", "lpx", "lpy", "group"))
        molecular_map(grouped, install=install)
        args = called_with(grouped)
        assert "--min-sigma 0.8 --max-sigma 1.5" in args, args
        molecular_map(grouped, loc_prec_handle="abs", min_sigma=13.0,
                      max_sigma=26.0, install=install)
        args = called_with(grouped)
        assert "--min-sigma 0.1 --max-sigma 0.2" in args, args
        ungrouped = make_input(tmp, "ng.hdf5")
        deep = make_input(tmp, "g3.hdf5", xyz + ("group",))
        for path in (ungrouped, deep):
            try:
                molecular_map(path, install=install)
            except ValueError:
                continue
            raise AssertionError(f"{os.path.basename(path)} was accepted")
        return "abs bounds converted; unclustered and uncalibrated 3D refused"

    def link_converts():
        path = make_input(tmp, "lk.hdf5")
        link_localizations(path, distance_nm=65.0, max_dark_time=2,
                           install=install)
        args = called_with(path)
        assert "-d 0.5" in args and "-t 2" in args, args
        return None

    def override_beats_metadata():
        path = make_input(tmp, "ov.hdf5")     # sidecar says 130
        undrift_aim(path, intersect_distance_nm=26.0, pixel_size_nm=65.0,
                    install=install)
        assert "-i 0.4" in called_with(path)   # 26/65, not 26/130
        return "explicit pixel size wins"

    def refuses_without_a_pixel_size():
        path = os.path.join(tmp, "bare.hdf5")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
        try:
            undrift_aim(path, install=install)
        except PicassoFailed as error:
            assert "pixel size" in str(error).lower()
            return "refused, not guessed"
        raise AssertionError("ran without knowing the pixel size")

    check("aim converts nm to camera pixels", aim_converts)
    check("3D clustering converts both radii", cluster_converts_both_radii)
    check("smlm_cluster's bool and integer arguments",
          cluster_bool_and_integer_traps)
    check("smlm_cluster checks 2D against 3D", cluster_dimension_is_checked)
    check("G5M sigma bounds and required inputs", g5m_bounds_and_inputs)
    check("link converts the distance", link_converts)
    check("an explicit pixel size overrides the metadata",
          override_beats_metadata)
    check("no pixel size anywhere is refused", refuses_without_a_pixel_size)


# ========================================================== failure modes
def test_failures() -> None:
    print("\n3. FAILURE MODES")
    tmp = new_tmp("pc_fail_")
    install = make_fake(tmp)

    def no_install():
        path = make_input(tmp, "noinst.hdf5")
        real_find = picasso_cli.find_picasso
        picasso_cli.find_picasso = lambda explicit=None: None
        try:
            run_command("aim", path, [], {"locs": "_aim.hdf5"}, install=None)
        except PicassoNotFound as error:
            assert "no picasso executable" in str(error).lower(), error
            return "PicassoNotFound, not a crash"
        finally:
            picasso_cli.find_picasso = real_find
        raise AssertionError("ran without an install")

    def unknown_command():
        path = make_input(tmp, "unk.hdf5")
        try:
            run_command("teleport", path, [], {"o": "_o.hdf5"},
                        install=install)
        except PicassoFailed as error:
            assert "teleport" in str(error)
            return "the build's command list is checked first"
        raise AssertionError("ran an unsupported command")

    def missing_input():
        try:
            run_command("aim", os.path.join(tmp, "ghost.hdf5"), [],
                        {"locs": "_aim.hdf5"}, install=install)
        except PicassoFailed as error:
            assert "does not exist" in str(error)
            return None
        raise AssertionError("accepted a missing input")

    def nonzero_exit_is_raised():
        path = make_input(tmp, "boom.hdf5")
        os.environ["FAKE_PICASSO_FAIL"] = "1"
        try:
            undrift_aim(path, install=install)
        except PicassoFailed as error:
            assert "simulated failure" in str(error)
            return "stderr is carried into the exception"
        finally:
            os.environ.pop("FAKE_PICASSO_FAIL", None)
        raise AssertionError("a failing run was reported as success")

    def stale_output_is_not_returned():
        # THE safety property. Picasso writes to a fixed name next to the
        # input, so a previous run leaves a file there. A failed run must
        # not hand that back as though it were fresh -- it was made with
        # different parameters and nothing downstream could tell.
        # The case that matters is a run that exits 0 WITHOUT writing --
        # the real Picasso does that for a file whose metadata it cannot
        # read -- because a non-zero exit is rejected anyway.
        path = make_input(tmp, "stale.hdf5")
        undrift_aim(path, install=install)          # leaves both outputs
        base = os.path.splitext(path)[0]
        for suffix in ("_aim.hdf5", "_aimdrift.txt"):
            old = os.path.getmtime(base + suffix) - 100
            os.utime(base + suffix, (old, old))
        with env(FAKE_PICASSO_NOWRITE="1"):
            try:
                undrift_aim(path, install=install)
            except PicassoFailed as error:
                assert "without writing anything" in str(error), error
            else:
                raise AssertionError("a stale output was accepted")
        # One output rewritten, the other left over: still a failure.
        with env(FAKE_PICASSO_PARTIAL="1"):
            try:
                undrift_aim(path, install=install)
            except PicassoFailed as error:
                assert "drift" in str(error), error
            else:
                raise AssertionError("a stale drift file was accepted")
        return "exit 0 with nothing new, or half new, is a failure"

    def a_failed_run_is_a_failure():
        path = make_input(tmp, "failed.hdf5")
        undrift_aim(path, install=install)
        with env(FAKE_PICASSO_FAIL="1"):
            try:
                undrift_aim(path, install=install)
            except PicassoFailed:
                return None
        raise AssertionError("a failing run over old outputs was accepted")

    def brackets_are_literal():
        folder = os.path.join(tmp, "Datos [2026]")
        os.makedirs(folder)
        target = make_input(folder, "Roi[1].hdf5")
        decoy = make_input(folder, "Roi1.hdf5")
        run = link_localizations(target, distance_nm=65.0, install=install)
        assert run.outputs["locs"] == os.path.splitext(target)[0] + \
            "_link.hdf5", run.outputs
        assert not os.path.exists(os.path.splitext(decoy)[0] + "_link.hdf5")
        # Brackets only in the name: an unescaped glob would match Roi1.
        plain = os.path.join(tmp, "plain")
        os.makedirs(plain)
        target = make_input(plain, "Roi[1].hdf5")
        decoy = make_input(plain, "Roi1.hdf5")
        link_localizations(target, distance_nm=65.0, install=install)
        assert os.path.exists(os.path.join(plain, "Roi[1]_link.hdf5"))
        assert not os.path.exists(os.path.join(plain, "Roi1_link.hdf5"))
        # G5M takes its path literally: escaping it would break it.
        grouped = make_input(folder, "Roi[1]_clusters.hdf5",
                             ("frame", "x", "y", "lpx", "lpy", "group"))
        run = molecular_map(grouped, install=install)
        assert run.outputs["molmap"].endswith("Roi[1]_clusters_molmap.hdf5")
        return "Roi[1] processed, Roi1 untouched; G5M given the path as is"

    def g5m_never_writes_over_its_input():
        folder = os.path.join(tmp, "upper")
        os.makedirs(folder)
        columns = ("frame", "x", "y", "lpx", "lpy", "group")
        upper = make_input(folder, "G.HDF5", columns)
        size = os.path.getsize(upper)
        tricky = os.path.join(tmp, "run.hdf5")
        os.makedirs(tricky)
        inside = make_input(tricky, "a.hdf5", columns)
        for path in (upper, inside):
            try:
                molecular_map(path, install=install)
            except ValueError as error:
                assert "lower-case" in str(error), error
                continue
            raise AssertionError(f"G5M was run on {path}")
        assert os.path.getsize(upper) == size
        return "upper-case extension and a '.hdf5' folder refused"

    def unencodable_path_is_refused():
        if sys.platform != "win32":
            return "not Windows"
        folder = os.path.join(tmp, "\u03b2II espectrina")
        os.makedirs(folder)
        path = make_input(folder, "s.hdf5")
        try:
            undrift_aim(path, install=install)
        except PicassoFailed as error:
            assert "code page" in str(error), error
        else:
            raise AssertionError("ran on a path Picasso cannot print")
        accented = os.path.join(tmp, "Nicol\u00e1s")
        os.makedirs(accented)
        undrift_aim(make_input(accented, "s.hdf5"), install=install)
        # link never prints the path, so it can take the Greek letter.
        link_localizations(path, distance_nm=65.0, install=install)
        return "a Greek letter refused (except by link), an accent accepted"

    def a_partial_output_is_a_failure():
        path = make_input(tmp, "half.hdf5")
        with env(FAKE_PICASSO_PARTIAL="1"):
            try:
                undrift_aim(path, install=install)
            except PicassoFailed as error:
                assert "drift" in str(error), error
                return "a run missing one of its outputs is not a success"
        raise AssertionError("half an output was accepted")

    def a_run_can_be_cancelled():
        # The fake is a .cmd that starts python, so this also checks that
        # the whole process tree goes: an orphaned child would keep the
        # output pipes open and the join below would time out.
        path = make_input(tmp, "slow.hdf5")
        outcome: dict = {}

        def worker() -> None:
            try:
                undrift_aim(path, install=install)
                outcome["result"] = "finished"
            except PicassoCancelled:
                outcome["result"] = "cancelled"
            except Exception as error:  # noqa: BLE001 - reported below
                outcome["result"] = repr(error)

        started = time.time()
        with env(FAKE_PICASSO_SLEEP="60"):
            thread = threading.Thread(target=worker)
            thread.start()
            deadline = time.time() + 30
            while not picasso_cli._RUNNING and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(1.0)
            stopped = cancel_running()
            thread.join(30)
        elapsed = time.time() - started
        assert stopped == 1, stopped
        assert outcome.get("result") == "cancelled", outcome
        assert elapsed < 20, f"took {elapsed:.0f} s"
        assert not picasso_cli._RUNNING
        return f"a 60 s run stopped after {elapsed:.1f} s"

    def bad_loc_prec_handle():
        try:
            molecular_map(make_input(tmp, "g5.hdf5"), loc_prec_handle="sideways",
                          install=install)
        except ValueError:
            return None
        raise AssertionError("accepted an invalid loc_prec_handle")

    check("no usable install", no_install)
    check("a command this build lacks", unknown_command)
    check("a missing input file", missing_input)
    check("a non-zero exit is raised with its stderr", nonzero_exit_is_raised)
    check("a stale output is not returned as fresh", stale_output_is_not_returned)
    check("a failing run over old outputs is a failure",
          a_failed_run_is_a_failure)
    check("[ ] in a path are taken literally", brackets_are_literal)
    check("G5M never writes over its input", g5m_never_writes_over_its_input)
    check("a path outside the code page is refused",
          unencodable_path_is_refused)
    check("an invalid G5M option", bad_loc_prec_handle)
    check("a partial output is a failure", a_partial_output_is_a_failure)
    check("a run can be cancelled", a_run_can_be_cancelled)


# ====================================================== the real thing
def test_against_real_picasso() -> None:
    print("\n4. AGAINST THE INSTALLED PICASSO")
    install = find_picasso()
    if install is None:
        print("  skip  no Picasso on this machine")
        return
    print(f"        {install.executable}")
    if not os.path.exists(REAL_DATA):
        print(f"  skip  no sample file at {REAL_DATA}")
        return

    tmp = new_tmp("pc_real_")
    sample = os.path.join(tmp, "sample.hdf5")
    shutil.copy(REAL_DATA, sample)
    sidecar = os.path.splitext(REAL_DATA)[0] + ".yaml"
    if os.path.exists(sidecar):
        shutil.copy(sidecar, os.path.splitext(sample)[0] + ".yaml")

    def aim_runs():
        run = undrift_aim(sample, segmentation=500, install=install,
                          timeout_s=900)
        assert run.ok and "locs" in run.outputs and "drift" in run.outputs
        assert os.path.getsize(run.outputs["locs"]) > 0
        drift = np.loadtxt(run.outputs["drift"])
        assert drift.ndim == 2 and drift.shape[0] > 10
        return (f"{run.seconds:.0f} s, drift over {drift.shape[0]} frames, "
                f"{drift.shape[1]} axes")

    link_px = 1.5
    shared: dict = {}

    def linked_both_ways() -> dict:
        """One `picasso link` run, and Picasso's own pipeline around ours."""
        if shared:
            return shared
        import h5py
        from collections import Counter

        from tools.mps_io import load_localizations
        from tools.mps_metadata import get_value
        from tools.mps_paint import build_events
        from tools.mps_paint import link_localizations as our_link

        def as_counter(first, n, length) -> Counter:
            return Counter(zip(np.asarray(first, np.int64).tolist(),
                               np.asarray(n, np.int64).tolist(),
                               np.asarray(length, np.int64).tolist()))

        loc = load_localizations(sample)
        run = link_localizations(sample, distance_nm=link_px * loc.pixel_size_nm,
                                 max_dark_time=1, install=install,
                                 timeout_s=900)
        with h5py.File(run.outputs["locs"], "r") as handle:
            theirs = handle["locs"][:]
        with h5py.File(sample, "r") as handle:
            raw = handle["locs"][:]

        # Picasso's lib.ensure_sanity on load: finite everywhere, inside the
        # image, and >= 0 -- so a precision of exactly zero gets through.
        width = float(get_value(loc.info, "Width"))
        height = float(get_value(loc.info, "Height"))
        n_frames = int(get_value(loc.info, "Frames"))
        ok = np.all([np.isfinite(raw[c].astype(float))
                     for c in raw.dtype.names], axis=0)
        ok &= (raw["x"] < width) & (raw["y"] < height)
        for c in ("x", "y", "lpx", "lpy", "lpz", "photons", "ellipticity",
                  "sx", "sy"):
            if c in raw.dtype.names:
                ok &= raw[c] >= 0
        sel = np.nonzero(ok)[0]
        frame = raw["frame"][sel].astype(np.int64)
        group = our_link(frame, raw["x"][sel].astype(float),
                         raw["y"][sel].astype(float), link_px, 1)
        n_ev = int(group.max()) + 1
        first = np.full(n_ev, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(first, group, frame)
        last = np.full(n_ev, -1, dtype=np.int64)
        np.maximum.at(last, group, frame)
        count = np.bincount(group, minlength=n_ev)
        # An event holding a zero precision gets an infinite weight, so a
        # NaN position, and io.save_locs drops the whole event.
        zero_lp = (raw["lpx"][sel] == 0) | (raw["lpy"][sel] == 0)
        poisoned = np.zeros(n_ev, dtype=bool)
        poisoned[group[zero_lp]] = True
        # Picasso's end test, last < Frames, never fires for 0-based frames.
        valid = (first > 0) & (last < n_frames)
        kept = valid & ~poisoned
        lost = valid & poisoned

        # What build_events should give on the same links: our end test,
        # and no event whose every localization is a failed fit.
        assert np.array_equal(sel, np.arange(len(raw))), "rows were dropped"
        usable = np.isfinite(loc.lp_lateral_nm)[sel]
        has_position = np.bincount(group, weights=usable.astype(float),
                                   minlength=n_ev) > 0
        at_end = last == n_frames - 1
        expected = (first > 0) & ~at_end & has_position
        length = last - first + 1

        ours = build_events(
            loc.frame, loc.x_nm, loc.y_nm, photons=loc.photons,
            lp_nm=loc.lp_lateral_nm, radius_nm=link_px * loc.pixel_size_nm,
            max_dark_time=1, n_frames=loc.n_frames,
        )
        shared.update(
            theirs=as_counter(theirs["frame"], theirs["n"], theirs["len"]),
            mimic=as_counter(first[kept], count[kept], length[kept]),
            expected=as_counter(first[expected], count[expected],
                                length[expected]),
            ours=as_counter(ours.first_frame, ours.n_locs, ours.length),
            n_theirs=len(theirs),
            n_ours=ours.n,
            lost_events=int(np.count_nonzero(lost)),
            regained=int(np.count_nonzero(lost & expected)),
            end_trimmed=int(np.count_nonzero(kept & at_end)),
            no_position=int(np.count_nonzero(kept & ~at_end
                                              & ~has_position)),
        )
        return shared

    def link_is_picassos_given_its_input():
        both = linked_both_ways()
        theirs, mimic = both["theirs"], both["mimic"]
        only_theirs = sum((theirs - mimic).values())
        only_ours = sum((mimic - theirs).values())
        assert only_theirs == 0 and only_ours == 0, (only_theirs, only_ours)
        return (f"{both['n_theirs']:,} events, identical; Picasso lost "
                f"{both['lost_events']} to a zero precision")

    def build_events_departs_from_picasso_only_as_documented():
        both = linked_both_ways()
        ours, expected = both["ours"], both["expected"]
        only_ours = sum((ours - expected).values())
        only_expected = sum((expected - ours).values())
        assert only_ours == 0 and only_expected == 0, \
            (only_ours, only_expected)
        return (f"{both['n_ours']:,} events against Picasso's "
                f"{both['n_theirs']:,}: +{both['regained']} that Picasso "
                f"lost to a zero precision, -{both['end_trimmed']} still "
                f"bound in the last frame, -{both['no_position']} made only "
                f"of failed fits")

    def brackets_with_the_real_picasso():
        folder = os.path.join(tmp, "Datos [2026]")
        os.makedirs(folder)
        target = os.path.join(folder, "Roi[1].hdf5")
        decoy = os.path.join(folder, "Roi1.hdf5")
        for path in (target, decoy):
            shutil.copy(sample, path)
            shutil.copy(os.path.splitext(sample)[0] + ".yaml",
                        os.path.splitext(path)[0] + ".yaml")
        run = link_localizations(target, distance_nm=180.0, install=install,
                                 timeout_s=900)
        assert run.outputs["locs"].endswith("Roi[1]_link.hdf5"), run.outputs
        assert not os.path.exists(os.path.join(folder, "Roi1_link.hdf5"))
        return "picasso link wrote Roi[1]_link.hdf5 and left Roi1 alone"

    def short_aim_segments_scramble_the_sample():
        from tools.mps_io import load_localizations
        from tools.mps_picasso_tools import (
            UNDRIFT_SCRAMBLED_RATIO, default_segmentation)
        from tools.mps_quality import repeat_neighbour_fraction

        loc = load_localizations(sample)
        radius = 2 * float(np.nanmedian(loc.lp_lateral_nm))
        before = repeat_neighbour_fraction(loc.frame, loc.x_nm, loc.y_nm,
                                           radius)
        kept = {}
        for segmentation in (100, default_segmentation(loc.n, loc.n_frames)):
            run = undrift_aim(sample, segmentation=segmentation,
                              install=install, timeout_s=900)
            after = load_localizations(run.outputs["locs"])
            kept[segmentation] = repeat_neighbour_fraction(
                after.frame, after.x_nm, after.y_nm, radius) / before
        short, chosen = kept.values()
        assert short < 0.5, kept
        assert chosen >= UNDRIFT_SCRAMBLED_RATIO, kept
        return ", ".join(f"{seg} frames keep {share:.0%}"
                         for seg, share in kept.items())

    def cluster_flag_reaches_picasso():
        from tools.mps_metadata import get_value, load_metadata
        import h5py

        seen = {}
        n_clusters = 0
        for flag in (False, True):
            run = smlm_cluster(sample, radius_nm=25.0, min_locs=10,
                               frame_analysis=flag, install=install,
                               timeout_s=900)
            info, _ = load_metadata(run.outputs["locs"])
            seen[flag] = get_value(info, "Basic frame analysis")
            if not flag:
                with h5py.File(run.outputs["centers"], "r") as handle:
                    n_clusters = len(handle["locs"])
        assert seen == {False: False, True: True}, seen
        return (f"{n_clusters} clusters; Picasso recorded frame analysis "
                f"off and on as asked")

    def diverged_fits_are_dropped():
        from tools.mps_io import load_localizations

        loc = load_localizations(sample)
        lp = loc.lp_lateral_nm
        assert lp is not None
        marked = ~np.isfinite(lp)
        lpx, lpy = loc.lpx_nm, loc.lpy_nm
        zero = (lpx == 0) | (lpy == 0)
        rel = np.minimum(lpx / np.median(lpx[lpx > 0]),
                         lpy / np.median(lpy[lpy > 0]))
        collapsed = ~zero & (rel < 0.1)
        assert np.all(marked[zero]) and np.all(marked[collapsed])
        # Nothing that looks like a genuine fit is thrown away.
        assert not np.any(marked & (rel >= 0.2)), np.count_nonzero(
            marked & (rel >= 0.2))
        widths = np.minimum(np.asarray(loc.columns["sx"]),
                            np.asarray(loc.columns["sy"]))
        return (f"{int(zero.sum())} zero and {int(collapsed.sum())} "
                f"collapsed precisions marked (median width of the "
                f"collapsed {np.median(widths[collapsed]):.2f} px)")

    check("picasso aim runs and writes a drift file", aim_runs)
    check("picasso link, given the same input, is ours event for event",
          link_is_picassos_given_its_input)
    check("build_events departs from picasso only as documented",
          build_events_departs_from_picasso_only_as_documented)
    check("[ ] in a path, with the real picasso",
          brackets_with_the_real_picasso)
    check("failed fits are marked in lp_lateral_nm", diverged_fits_are_dropped)
    check("smlm_cluster's frame-analysis switch reaches picasso",
          cluster_flag_reaches_picasso)
    check("short AIM segments scramble this sample; the default does not",
          short_aim_segments_scramble_the_sample)


def main() -> int:
    print("=" * 72)
    print("PICASSO BRIDGE CHECKS")
    print("=" * 72)
    test_discovery()
    test_units()
    test_failures()
    test_against_real_picasso()
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
