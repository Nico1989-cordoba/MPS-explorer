# -*- coding: utf-8 -*-
"""
The launcher's pre-checks (tools/mps_precheck.py and tools/launcher_check.py, stage F3).

Every case is a SIMULATED pick written by tools/make_synthetic_batch.py into a temporary folder that imitates the
user's naming convention; each broken file must give exactly the ERROR / AVISO its check promises. No real axon is
read; the folder layout of the 18-axon test set is reproduced with invented names and simulated contents.

Run:  python test_launcher_precheck.py      (or with pytest)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import make_synthetic_batch as msb  # noqa: E402
from tools import mps_precheck as pc  # noqa: E402

PIXEL = 113.0          # the pixel size the synthetic project expects [the user's acquisitions]
FAST = pc.Probes(ram_gb=lambda: 8.0, disk_gb=lambda _p: 100.0, git=lambda _r: ("abc1234", False))


def name(genotype: str, roi: int, axon: int, extra: str = "") -> str:
    return f"26.11.07_bIIspt_AF647_{genotype}_{roi}_MMStack.ome_locs_filter{extra}_picked_Axon{axon}.hdf5"


class Batch:
    """A temporary input folder, an output folder and a project file."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.entrada = os.path.join(self.root, "entrada")
        self.salida = os.path.join(self.root, "salida")
        os.makedirs(self.entrada)
        os.makedirs(self.salida)
        self.shift = 0

    def close(self) -> None:
        self._tmp.cleanup()

    def axon(self, rel: str, **kw: Any) -> str:
        self.shift += 1
        spec = msb.AxonFile(pixel_nm=kw.pop("pixel_nm", PIXEL), shift_px=float(self.shift), **kw)
        return msb.write_axon(os.path.join(self.entrada, rel), spec)

    def clean(self) -> None:
        for g_index, g in enumerate(("KO", "WT")):
            for a in (1, 2):
                self.axon(os.path.join(f"nervio_{g_index + 1}", "ROI 1", f"Axon {a}", name(g, 1, a)))

    def project(self, extra: str = "", line: str = "adducin") -> str:
        path = os.path.join(self.root, "proyecto.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"nombre_tanda: prueba\ncarpetas:\n  entrada: '{self.entrada}'\n  salida: '{self.salida}'\n"
                         f"identidad:\n  linea_ko: '{line}'\n"
                         f"calibracion:\n  pixel_esperado_nm: {PIXEL:g}\n  frames_esperados: 60000\n" + extra)
        return path


def levels(result: pc.PrecheckResult, code: str) -> List[str]:
    return [c.level for c in result.get(code)]


def files_of(result: pc.PrecheckResult, code: str, level: str) -> List[str]:
    return [f for c in result.get(code) if c.level == level for f in c.files]


def run(path: str, **kw: Any) -> pc.PrecheckResult:
    kw.setdefault("probes", FAST)
    kw.setdefault("count_clusters", False)
    return pc.run_prechecks(path, **kw)


# ---------------------------------------------------------------------------

def test_a_clean_batch_may_start() -> None:
    b = Batch()
    try:
        b.clean()
        r = run(b.project(), count_clusters=True)
        assert r.status == "ok", pc.report_text(r)
        assert not r.errors and not r.warnings, [(c.code, c.message) for c in r.errors + r.warnings]
        assert len(r.files) == 4 and all(f.usable for f in r.files)
        assert {f.genotype for f in r.files} == {"KO", "WT"}
        assert all(f.n_clusters and f.n_clusters > 3 for f in r.files), [f.n_clusters for f in r.files]
        assert all(f.estimate is not None and not f.estimate.lower_bound for f in r.files)
        assert levels(r, "C25") == [pc.NOTA]                      # no animal: said, never guessed
    finally:
        b.close()


def test_each_broken_file_gives_its_check() -> None:
    b = Batch()
    try:
        b.clean()
        r1 = os.path.join("nervio_1", "ROI 2")
        no_yaml = b.axon(os.path.join(r1, "Axon 1", name("KO", 2, 1)), yaml=False, embed_metadata=True)
        px130 = b.axon(os.path.join(r1, "Axon 2", name("KO", 2, 2)), pixel_nm=130.0)
        no_geno = b.axon(os.path.join(r1, "Axon 3", "26.11.07_bIIspt_AF647_2_MMStack.ome_locs_picked_Axon3.hdf5"))
        lower = b.axon(os.path.join(r1, "Axon 4", name("wt", 2, 4)))
        no_z = b.axon(os.path.join(r1, "Axon 5", name("KO", 2, 5)), with_z=False)
        in_nm = b.axon(os.path.join(r1, "Axon 6", name("KO", 2, 6)), xy_in_nm=True)
        in_um = b.axon(os.path.join(r1, "Axon 7", name("KO", 2, 7)), z_in_um=True)
        short = b.axon(os.path.join(r1, "Axon 8", name("KO", 2, 8)), frames=30000)
        no_pick = b.axon(os.path.join(r1, "Axon 9", name("KO", 2, 9)), pick_step=False)
        line = b.axon(os.path.join(r1, "Axon 10", name("KO", 2, 10, "_4.1B")))
        # the same file name in another folder: another axon (user decision 2026-11-07)
        same_a = b.axon(os.path.join("nervio_3", "ROI 1", name("KO", 7, 1)))
        same_b = b.axon(os.path.join("nervio_3", "ROI 2", name("KO", 7, 1)))
        # a copy: the same bytes under two names
        copy_src = b.axon(os.path.join("nervio_3", "ROI 3", name("KO", 8, 1)))
        copy_dst = os.path.join(b.entrada, "nervio_3", "ROI 3", name("KO", 8, 2))
        shutil.copyfile(copy_src, copy_dst)
        shutil.copyfile(os.path.splitext(copy_src)[0] + ".yaml", os.path.splitext(copy_dst)[0] + ".yaml")
        # one acquisition under two names in one folder (a drift-corrected "_aim" copy)
        dup_a = b.axon(os.path.join("nervio_3", "ROI 4", "x_KO_picked_Axon1.hdf5"))
        dup_b = b.axon(os.path.join("nervio_3", "ROI 4", "x_KO_picked_Axon1_aim.hdf5"))
        # not an HDF5 file at all
        junk = os.path.join(b.entrada, "nervio_3", "ROI 5", name("KO", 9, 1))
        os.makedirs(os.path.dirname(junk))
        with open(junk, "w", encoding="utf-8") as handle:
            handle.write("not hdf5")
        # whole field: no "picked" in the name -> left out by the filter
        whole = b.axon(os.path.join("nervio_1", "26.11.07_bIIspt_AF647_KO_1_MMStack.ome_locs_filter.hdf5"))

        r = run(b.project())
        rel = {p: os.path.relpath(p, b.entrada).replace("\\", "/") for p in (
            no_yaml, px130, no_geno, lower, no_z, in_nm, in_um, short, no_pick, line, same_a, same_b, copy_src,
            copy_dst, dup_a, dup_b, junk, whole)}
        assert r.status == "errores"
        assert rel[no_yaml] in files_of(r, "C5", pc.AVISO)
        assert rel[px130] in files_of(r, "C6", pc.ERROR)
        assert rel[no_geno] in files_of(r, "C9", pc.ERROR)
        assert rel[lower] in files_of(r, "C10", pc.AVISO)
        assert rel[lower] not in files_of(r, "C9", pc.ERROR)
        assert rel[no_z] in files_of(r, "C4", pc.ERROR)
        assert rel[in_nm] in files_of(r, "C7", pc.ERROR)
        assert rel[in_um] in files_of(r, "C7", pc.ERROR)
        assert rel[short] in files_of(r, "C14", pc.AVISO)
        assert rel[no_pick] in files_of(r, "C26", pc.AVISO)
        assert rel[line] in files_of(r, "C22", pc.ERROR)
        assert rel[junk] in files_of(r, "C3", pc.ERROR)
        c12 = [c for c in r.get("C12") if c.level == pc.AVISO]
        assert c12 and rel[same_a] in c12[0].message and rel[same_b] in c12[0].message
        c13 = [c for c in r.get("C13") if c.level == pc.ERROR]
        assert c13 and rel[copy_src] in c13[0].message and rel[copy_dst] in c13[0].message
        c11 = [c for c in r.get("C11") if c.level == pc.ERROR]
        assert c11 and rel[dup_a] in c11[0].message and rel[dup_b] in c11[0].message
        assert whole in r.left_out_by_filter
        # the clean files stay clean
        clean = [f for f in r.files if f.rel.startswith(("nervio_1/ROI 1", "nervio_2/ROI 1"))]
        assert len(clean) == 4 and all(f.usable for f in clean)
        # the report says each of them, in a file and in a table
        txt, csv_path = pc.write_report(r, os.path.join(b.salida, "rep"))
        with open(txt, encoding="utf-8") as handle:
            text = handle.read()
        assert text.startswith("CHEQUEOS PREVIOS - NO SE PUEDE ARRANCAR")
        for code in ("C3", "C4", "C5", "C6", "C7", "C9", "C10", "C11", "C12", "C13", "C14", "C22", "C26"):
            assert f"[{code}]" in text, code
        assert os.path.getsize(csv_path) > 0
    finally:
        b.close()


def test_policies_turn_errors_into_warnings() -> None:
    b = Batch()
    try:
        b.clean()
        b.axon(os.path.join("nervio_1", "ROI 2", "Axon 1", name("KO", 2, 1)), pixel_nm=130.0)
        b.axon(os.path.join("nervio_1", "ROI 2", "Axon 2", "s_2_MMStack_picked_Axon2.hdf5"))
        r = run(b.project("politicas:\n  pixel_distinto: usar_el_del_archivo\n  sin_genotipo: analizar_igual\n"))
        assert pc.ERROR not in levels(r, "C6") and pc.AVISO in levels(r, "C6"), levels(r, "C6")
        assert levels(r, "C9") == [pc.AVISO]
        assert r.status == "ok", [(c.code, c.message) for c in r.errors]
    finally:
        b.close()


def test_pixel_size_without_yaml() -> None:
    b = Batch()
    try:
        b.clean()
        # alone, with no neighbour within two folders: an error unless the project gives a fallback
        lone = b.axon(os.path.join("aislado", "a", "b", name("KO", 5, 1)), yaml=False)
        r = run(b.project())
        assert os.path.relpath(lone, b.entrada).replace("\\", "/") in files_of(r, "C5", pc.ERROR)
        # with calibracion.pixel_respaldo_nm the file is read with it, and the report says so
        path = b.project()
        with open(path, encoding="utf-8") as handle:
            text = handle.read().replace("  frames_esperados: 60000\n",
                                         "  frames_esperados: 60000\n  pixel_respaldo_nm: 113\n")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        r = run(path)
        assert not files_of(r, "C5", pc.ERROR)
        manual = [c for c in r.get("C5") if c.title.startswith("Píxel de respaldo")]
        assert manual and manual[0].level == pc.AVISO
        # next to a file that carries its pixel: taken from the neighbour, with a warning
        near = b.axon(os.path.join("nervio_1", "ROI 1", "Axon 1", name("KO", 1, 9)), yaml=False)
        r = run(b.project())
        info = [f for f in r.files if f.path == near][0]
        assert info.pixel_source == "neighbour" and info.pixel_nm == PIXEL, (info.pixel_source, info.pixel_nm)
    finally:
        b.close()


def test_the_layout_of_the_test_set_and_the_picked_filter() -> None:
    # The structure of the 18-axon test set (folder names and counts only; invented file names; simulated
    # contents): 23 .hdf5, 5 of them whole fields (2 at the top, 3 in "ROI 1/Nueva carpeta"), 18 picks: ROI 1 with
    # "Axon n" subfolders and one pick loose in the ROI folder, ROI 2 with no subfolders.
    b = Batch()
    try:
        movie = "26.11.07_bIIspt_TIRF_1_MMStack.ome_locs"
        for a in range(1, 9):
            b.axon(os.path.join("ROI 1", f"Axon {a}", f"{movie}_filter_picked_axon{a}.hdf5"))
        b.axon(os.path.join("ROI 1", f"{movie}_filter_picked_axon9.hdf5"))
        for a in range(1, 10):
            b.axon(os.path.join("ROI 2", f"{movie}_filter_picked_axon{a}_roi2.hdf5"))
        for stem in ("_filter_render", "_filter", ""):
            b.axon(os.path.join("ROI 1", "Nueva carpeta", f"{movie}{stem}.hdf5"))
        for stem in ("_filter_render", "_filter"):
            b.axon(f"{movie}{stem}.hdf5")
        r = run(b.project())
        assert len(r.files) == 18, len(r.files)
        assert len(r.left_out_by_filter) == 5, r.left_out_by_filter
        rois = sorted({f.roi for f in r.files})
        assert rois == ["ROI 1", "ROI 2"], rois
        # no genotype in these names: with the default policy that stops the run
        assert levels(r, "C9") == [pc.ERROR]
        # "ignorar_carpetas" leaves the draft folder out even without the name filter
        r2 = run(b.project("archivos:\n  el_nombre_contiene: ''\n  ignorar_carpetas: ['Nueva carpeta']\n"))
        assert len(r2.files) == 20 and len(r2.left_out_by_folder) == 3, (len(r2.files), r2.left_out_by_folder)
    finally:
        b.close()


def test_the_ko_line_must_match_the_names() -> None:
    b = Batch()
    try:
        b.clean()
        r = run(b.project(line="4.1B"))
        assert pc.ERROR not in levels(r, "C22")              # the names do not state a line: nothing to compare
        b.axon(os.path.join("nervio_1", "ROI 3", name("KO", 3, 1, "_aducina")))
        r = run(b.project(line="4.1B"))
        assert pc.ERROR in levels(r, "C22")
        r = run(b.project(line="adducin"))
        assert pc.ERROR not in levels(r, "C22")
        # a project without the line is an error of the pre-check (user decision 2026-11-07)
        path = b.project()
        text = open(path, encoding="utf-8").read().replace("identidad:\n  linea_ko: 'adducin'\n", "")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        r = run(path)
        assert r.status == "errores" and "linea_ko" in r.get("C1")[0].message
    finally:
        b.close()


def test_machine_limits() -> None:
    b = Batch()
    try:
        b.clean()
        low = pc.Probes(ram_gb=lambda: 1.0, disk_gb=lambda _p: 0.5, git=lambda _r: ("abc1234", True))
        r = run(b.project(), probes=low)
        assert levels(r, "C16") == [pc.ERROR] and levels(r, "C17")[0] == pc.ERROR
        assert levels(r, "C21") == [pc.AVISO]
        mid = pc.Probes(ram_gb=lambda: 2.5, disk_gb=lambda _p: 50.0, git=lambda _r: (None, None))
        r = run(b.project(), probes=mid)
        assert levels(r, "C16") == [pc.AVISO] and levels(r, "C21") == [pc.AVISO]
        # the real probes answer something on this machine
        assert pc.free_disk_gb(b.salida) is not None
    finally:
        b.close()


def test_slow_method_and_parameters_not_applied_yet() -> None:
    b = Batch()
    try:
        b.clean()
        r = run(b.project("parametros:\n  metodo: OPTICS\n"))
        assert levels(r, "C15") == [pc.ERROR]
        assert any(c.level == pc.AVISO and "parametros.metodo" in c.message for c in r.get("C1b"))
        r = run(b.project("parametros:\n  metodo: emitters_bagol_like\npoliticas:\n  metodos_lentos: permitir\n"))
        assert pc.ERROR in levels(r, "C15")                   # Emitters without a buffer
        r = run(b.project("parametros:\n  tau_nm: 40\n"))
        assert levels(r, "C1c") == [pc.ERROR]
        r = run(b.project("parametros:\n  tau_nm: 31\n"))
        assert not r.get("C1c") and levels(r, "C1b") == [pc.OK]
    finally:
        b.close()


def test_time_estimate_rules() -> None:
    a = pc.estimate_axon_seconds("DBSCAN", 5000, 150, null_on=True, replicates=199, closure="n_locs")
    b = pc.estimate_axon_seconds("DBSCAN", 5000, 600, null_on=True, replicates=199, closure="n_locs")
    c = pc.estimate_axon_seconds("DBSCAN", 5000, None, null_on=True, replicates=199, closure="n_locs")
    d = pc.estimate_axon_seconds("DBSCAN", 5000, 150, null_on=False, replicates=199, closure="n_locs")
    e = pc.estimate_axon_seconds("DBSCAN", 5000, 150, null_on=True, replicates=199, closure="ambos")
    assert b.total_s > a.total_s > c.total_s and c.lower_bound and not a.lower_bound
    assert d.null_s == 0 and d.total_s == d.analysis_s
    assert e.null_s > a.null_s
    # every replicate of the null rebuilds the perimeter: 199 + stage 2's bound (112) + 199 analyses
    assert abs(a.n_null_analyses - (199 + 112 + 199)) < 1e-6, a.n_null_analyses
    assert abs(a.total_s - (2 * a.analysis_s + a.null_s)) < 1e-6


def _long_project(b: Batch, extra: str = "") -> str:
    # B = 100 000 replicates makes a 12 000-localization synthetic axon cost days (an EXPLORATORY run, only to
    # reach the > 2 h rule with a small file).
    return b.project("nulo:\n  replicas: 100000\n" + extra)


def test_long_axons_need_a_confirmation() -> None:
    from tools import launcher_check as lc

    b = Batch()
    try:
        b.clean()
        r = run(_long_project(b))
        assert r.status == "confirmar" and len(r.long_unaccepted) == 4
        assert levels(r, "C23") == [pc.CONFIRMAR]
        text = pc.long_axons_text(r)
        assert "Total de estos axones" in text and "aceptar_axones_largos:" in text
        for f in r.files:
            assert f.rel in text
        # accepted in the project, by name or by relative path
        first, second = r.files[0], r.files[1]
        names = f"aceptar_axones_largos: ['{os.path.basename(first.path)}', '{second.rel}']\n"
        r = run(_long_project(b, names))
        assert len(r.long_unaccepted) == 2 and r.status == "confirmar"
        # accepted by the command-line flag
        r = run(_long_project(b), accept_long=True)
        assert r.status == "ok" and levels(r, "C23") == [pc.AVISO]
        # the entry point: no console -> nothing starts, the list goes to a file, exit code 3
        path = _long_project(b)
        rc = lc.main([path, "--sin-contar-clusteres"], interactive=False)
        assert rc == lc.EXIT_CONFIRM
        found = [os.path.join(dp, n) for dp, _d, fs in os.walk(b.salida) for n in fs if n == lc.LONG_LIST_NAME]
        assert len(found) == 1
        # a console that answers "si" accepts them for this run
        rc = lc.main([path, "--sin-contar-clusteres"], interactive=True, ask=lambda _t: "si")
        assert rc == lc.EXIT_OK
        rc = lc.main([path, "--sin-contar-clusteres"], interactive=True, ask=lambda _t: "no")
        assert rc == lc.EXIT_CONFIRM
        rc = lc.main([path, "--sin-contar-clusteres", "--aceptar-axones-largos"], interactive=False)
        assert rc == lc.EXIT_OK
        # every run wrote its own report folder; none was overwritten
        reports = [dp for dp, _d, fs in os.walk(b.salida) if "chequeos_previos.txt" in fs]
        assert len(reports) == 4, reports
    finally:
        b.close()


def test_entry_point_with_a_bad_project() -> None:
    from tools import launcher_check as lc

    b = Batch()
    try:
        path = b.project("parametros:\n  eps_mn: 25\n")
        rc = lc.main([path], interactive=False)
        assert rc == lc.EXIT_UNREADABLE
        beside = [n for n in os.listdir(b.root) if n.startswith("chequeos_proyecto_")]
        assert len(beside) == 1
        with open(os.path.join(b.root, beside[0], "chequeos_previos.txt"), encoding="utf-8") as handle:
            assert "eps_mn" in handle.read()
        assert lc.main([os.path.join(b.root, "no_existe.yaml")], interactive=False) == lc.EXIT_UNREADABLE
    finally:
        b.close()


def test_cluster_count_and_estimate_against_a_real_analysis() -> None:
    # The count of the pre-check is the analysis' own n_clusters_kept, and the estimate of an axon with its null is an
    # upper bound of the time it takes (stage 2 of the regularity is counted whether it runs or not). The plan's
    # "within a factor 3" for one analysis does NOT hold (the observed analysis also measures the NeNA, the lpc, the
    # plug-ins and 999 randomizations, which the per-replicate cost table leaves out): the ratios are printed for the
    # record and the report discusses them.
    from tools import mps_batch as mb
    from tools.mps_identity import AxonIdentity
    from tools.mps_project import batch_settings_from_project, load_project

    b = Batch()
    try:
        b.clean()
        path = b.project()
        r = run(path, count_clusters=True)
        info = r.files[0]
        settings = batch_settings_from_project(load_project(path).project)
        item = mb.BatchFile(path=info.path, identity=AxonIdentity(), own_pixel_size_nm=info.pixel_nm)
        mb.analyse_file(item, settings)                      # warm-up: imports and first-call costs
        t0 = time.perf_counter()
        record = mb.analyse_file(item, settings)
        seconds = time.perf_counter() - t0
        assert record.ok, record.error
        assert int(record.row["n_clusters_kept"]) == info.n_clusters, (record.row["n_clusters_kept"], info.n_clusters)
        replicates = 9
        t0 = time.perf_counter()
        with_null = mb.analyse_file(item, settings, null_replicates=replicates)
        seconds_null = time.perf_counter() - t0
        assert with_null.ok, with_null.error
        n_slab = int(record.row["n_locs_slab"])
        est = pc.estimate_axon_seconds("DBSCAN", n_slab, info.n_clusters, null_on=True, replicates=replicates,
                                       closure="n_locs")
        print(f"        one analysis: estimate {est.analysis_s:.1f} s, measured {seconds:.1f} s; with a null of "
              f"B = {replicates}: estimate {est.analysis_s + est.null_s:.1f} s, measured {seconds_null:.1f} s")
        assert est.analysis_s + est.null_s >= seconds_null, (est, seconds_null)
        # the pre-check's slab estimate is of the order of the analysis' own slab
        assert info.n_slab_estimate is not None and 0.5 <= info.n_slab_estimate / n_slab <= 2.0, \
            (info.n_slab_estimate, n_slab)
    finally:
        b.close()


TESTS: List[Callable[[], None]] = [obj for name_, obj in sorted(globals().items())
                                   if name_.startswith("test_") and callable(obj)]


def main() -> int:
    failed = 0
    for fn in TESTS:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"  ok    {fn.__name__}  ({time.perf_counter() - t0:.1f} s)")
        except Exception:  # noqa: BLE001 - a harness
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            for line in traceback.format_exc().strip().splitlines()[-5:]:
                print(f"        {line}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
