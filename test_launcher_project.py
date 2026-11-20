# -*- coding: utf-8 -*-
"""
The launcher's project file (tools/mps_project.py, stage F2): strict reading, defaults, types and ranges, and the
EXPLORATORY mark for any frozen parameter moved from its pre-specified value.

Synthetic only: the projects are written here, the folders are temporary and empty. No data file is read.

Run:  python test_launcher_project.py      (or with pytest)
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from typing import Any, Callable, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools import mps_project as mp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "docs", "launcher_project_example.yaml")


class Folders:
    """An empty input folder and an output folder, both temporary."""

    def __enter__(self) -> "Folders":
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.entrada = os.path.join(self.root, "entrada")
        self.salida = os.path.join(self.root, "salida")
        os.makedirs(self.entrada)
        os.makedirs(self.salida)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._tmp.cleanup()

    def minimal(self, extra: str = "") -> str:
        return (f"nombre_tanda: t1\ncarpetas:\n  entrada: '{self.entrada}'\n  salida: '{self.salida}'\n"
                f"identidad:\n  linea_ko: adducin\ncalibracion:\n  pixel_esperado_nm: 113\n" + extra)

    def parse(self, text: str) -> mp.ProjectLoad:
        return mp.parse_project(text, os.path.join(self.root, "proyecto.yaml"))


def _with(text: str, section: str, line: str) -> str:
    """``text`` with ``line`` added under ``section:`` (the section is created when missing)."""
    if re.search(rf"^{section}:\s*$", text, re.M):
        return re.sub(rf"^({section}:\s*)$", rf"\1\n  {line}", text, count=1, flags=re.M)
    return text + f"{section}:\n  {line}\n"


def test_the_example_is_valid_and_not_exploratory() -> None:
    with Folders() as f, open(EXAMPLE, encoding="utf-8") as handle:
        text = handle.read()
        text = re.sub(r"entrada: '[^']*'", lambda _m: f"entrada: '{f.entrada}'", text)
        text = re.sub(r"salida:  '[^']*'", lambda _m: f"salida:  '{f.salida}'", text)
        load = f.parse(text)
        assert load.ok, load.errors
        p = load.project
        assert p is not None
        # 0.3333 for 1/3 and 4.243 for 3 sqrt(2) are the pre-specified values as typed, not deviations.
        assert p.exploratory_by == [], [d.words() for d in p.exploratory_by]
        assert p.get("identidad.linea_ko") == "adducin"
        assert p.get("calibracion.pixel_esperado_nm") == 113.0
        # exploratory only by the declared role of the data (no pre-registration)
        assert p.exploratory and p.label().startswith("EXPLORATORY")


def test_unknown_key_is_an_error_with_a_suggestion() -> None:
    with Folders() as f:
        load = f.parse(f.minimal("parametros:\n  eps_mn: 25\n"))
        assert not load.ok
        assert any("clave desconocida 'parametros.eps_mn'" in e and "parametros.eps_nm" in e for e in load.errors), \
            load.errors
        load = f.parse(f.minimal("parametro:\n  eps_nm: 25\n"))
        assert any("clave desconocida 'parametro'" in e for e in load.errors), load.errors


def test_types_are_checked() -> None:
    with Folders() as f:
        for extra, word in (("parametros:\n  eps_nm: '25'\n", "parametros.eps_nm"),
                            ("parametros:\n  min_samples: 10.5\n", "parametros.min_samples"),
                            ("parametros:\n  min_samples: true\n", "parametros.min_samples"),
                            ("nulo:\n  correr: 'si'\n", "nulo.correr"),
                            ("archivos:\n  ignorar_carpetas: 'borrador'\n", "archivos.ignorar_carpetas"),
                            ("politicas:\n  ventana_z_manual: {a.hdf5: alto}\n", "politicas.ventana_z_manual"),
                            ("carpetas: 3\n", "carpetas")):
            load = f.parse(f.minimal() + extra if not extra.startswith("carpetas") else
                           f.minimal().replace(f"carpetas:\n  entrada: '{f.entrada}'\n  salida: '{f.salida}'\n",
                                               extra))
            assert not load.ok and any(word in e for e in load.errors), (extra, load.errors)
        # an integer typed as 10.0 is still 10
        load = f.parse(f.minimal("parametros:\n  min_samples: 10.0\n"))
        assert load.ok and load.project is not None and load.project.get("parametros.min_samples") == 10


def test_ranges_and_choices() -> None:
    with Folders() as f:
        for extra, word in (("parametros:\n  eps_nm: 0\n", "fuera del rango"),
                            ("parametros:\n  c_label: 500\n", "fuera del rango"),
                            ("nulo:\n  cierre: todos\n", "no es un valor aceptado"),
                            ("politicas:\n  perimetro: un_arranque\n", "no es un valor aceptado"),
                            ("parametros:\n  metodo: DBSCAM\n", "no es un método del programa")):
            load = f.parse(f.minimal(extra))
            assert not load.ok and any(word in e for e in load.errors), (extra, load.errors)


def test_the_ko_line_is_required_and_one_of_two() -> None:
    with Folders() as f:
        load = f.parse(f.minimal().replace("identidad:\n  linea_ko: adducin\n", ""))
        assert not load.ok and any("identidad.linea_ko" in e and "línea KO" in e for e in load.errors), load.errors
        load = f.parse(f.minimal().replace("linea_ko: adducin", "linea_ko: aducina"))
        assert not load.ok and any("'adducin', '4.1B'" in e for e in load.errors), load.errors
        load = f.parse(f.minimal().replace("linea_ko: adducin", "linea_ko: '4.1B'"))
        assert load.ok, load.errors


def test_folders() -> None:
    with Folders() as f:
        load = f.parse(f.minimal().replace(f.entrada, os.path.join(f.root, "no_existe")))
        assert any("carpetas.entrada: la carpeta no existe" in e for e in load.errors), load.errors
        load = f.parse(f.minimal().replace(f"salida: '{f.salida}'",
                                           f"salida: '{os.path.join(f.entrada, 'res')}'"))
        assert any("no puede estar dentro de carpetas.entrada" in e for e in load.errors), load.errors
        # relative paths are read from the project file's folder
        load = f.parse(f.minimal().replace(f"entrada: '{f.entrada}'", "entrada: 'entrada'"))
        assert load.ok and load.project is not None
        assert os.path.normcase(load.project.get("carpetas.entrada")) == os.path.normcase(f.entrada)


def test_every_problem_is_reported_at_once() -> None:
    with Folders() as f:
        load = f.parse(f.minimal("parametros:\n  eps_nm: -3\n  min_samples: 'diez'\nnombre_tanda2: x\n"))
        assert len(load.errors) >= 3, load.errors


def test_defaults_are_the_programs() -> None:
    from tools.mps_batch import BatchSettings
    from tools.mps_settings import MPSSettings

    with Folders() as f:
        load = f.parse(f.minimal())
        assert load.ok and load.project is not None, load.errors
        p = load.project
        assert "parametros.eps_nm" in p.defaults_used and "nulo.replicas" in p.defaults_used
        assert "nombre_tanda" not in p.defaults_used
        mine = mp.batch_settings_from_project(p)
        fresh = BatchSettings.from_settings(MPSSettings())
        assert mine == fresh, (mine, fresh)
        assert p.get("identidad.patrones.muestra") == ""      # optional and empty by default
        assert p.get("canal_2") is None and p.get("aceptar_axones_largos") == []


def test_exploratory_marks() -> None:
    with Folders() as f:
        cases: Dict[str, List[str]] = {
            "parametros:\n  eps_nm: 26\n": ["parametros.eps_nm"],
            "parametros:\n  tau_nm: 31\n": [],
            "parametros:\n  tau_nm: 31.0\n": [],
            "parametros:\n  tau_nm: 32\n": ["parametros.tau_nm"],
            "parametros:\n  radio_eventos_nena: 3.5\n": ["parametros.radio_eventos_nena"],
            "parametros:\n  radio_lpc_regla_nena: 4.2426\n": [],
            "parametros:\n  ocupancia_tope_sigma: 0.25\n": ["parametros.ocupancia_tope_sigma"],
            "parametros:\n  c_label: 14\n": [],
            "parametros:\n  c_label: 26\n": ["parametros.c_label"],
            "parametros:\n  metodo: HDBSCAN\n": ["parametros.metodo"],
            "nulo:\n  cierre: ambos\n": [],
            "nulo:\n  cierre: k\n": ["nulo.cierre"],
            "nulo:\n  replicas: 99\n": ["nulo.replicas"],
            "nulo:\n  correr: false\n": [],
        }
        for extra, want in cases.items():
            load = f.parse(f.minimal(extra))
            assert load.ok and load.project is not None, (extra, load.errors)
            got = [d.key for d in load.project.exploratory_by]
            assert got == want, (extra, got)
            if want:
                assert "parámetros cambiados" in load.project.label(), load.project.label()


def test_channel_2_is_reserved() -> None:
    with Folders() as f:
        assert f.parse(f.minimal("canal_2: null\n")).ok
        load = f.parse(f.minimal("canal_2: {proteina: aducina}\n"))
        assert not load.ok and any("canal_2" in e and "todavía no está soportado" in e for e in load.errors)


def test_batch_name_and_confirmatory() -> None:
    with Folders() as f:
        load = f.parse(f.minimal().replace("nombre_tanda: t1", "nombre_tanda: 'tanda uno'"))
        assert any("nombre_tanda" in e for e in load.errors), load.errors
        load = f.parse(f.minimal("rol_de_los_datos: confirmatorio\n"))
        assert any("pre-registro" in e for e in load.errors), load.errors


def test_null_replicates_match_the_batch_window() -> None:
    # The batch window is a Qt module; its constant is read from the source instead of importing it.
    with open(os.path.join(HERE, "tools", "batch_window.py"), encoding="utf-8") as handle:
        m = re.search(r"^NULL_REPLICATES\s*=\s*(\d+)", handle.read(), re.M)
    assert m and int(m.group(1)) == mp.NULL_REPLICATES_DEFAULT, m


def test_load_project_reads_a_file_and_strict_raises() -> None:
    with Folders() as f:
        path = os.path.join(f.root, "proyecto.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f.minimal("parametros:\n  eps_mn: 3\n"))
        load = mp.load_project(path)
        assert not load.ok
        try:
            mp.load_project_strict(path)
        except mp.ProjectError as error:
            assert "eps_mn" in str(error)
        else:
            raise AssertionError("load_project_strict did not raise")
        assert mp.load_project(os.path.join(f.root, "no.yaml")).errors


TESTS: List[Callable[[], None]] = [obj for name, obj in sorted(globals().items())
                                   if name.startswith("test_") and callable(obj)]


def main() -> int:
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception:  # noqa: BLE001 - a harness
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            for line in traceback.format_exc().strip().splitlines()[-4:]:
                print(f"        {line}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
