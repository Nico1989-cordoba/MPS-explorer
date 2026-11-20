# -*- coding: utf-8 -*-
"""
Run ONLY the launcher's pre-checks on a project file and write the report. Nothing is analysed.

    venv/Scripts/python.exe -m tools.launcher_check <proyecto.yaml> [--aceptar-axones-largos]
                                                   [--sin-contar-clusteres] [--informe-en <carpeta>]

The full launcher (stage F4) is not part of this repository; this entry point is the part of it that decides whether
a batch may start. It does not depend on Claude: it runs from a terminal, or with
``pythonw`` from a double-click (then there is no console and nothing is asked).

What it writes: ``chequeos_previos.txt`` (to read) and ``chequeos_previos.csv`` (one row per check) in a new folder
``<salida>/chequeos/<nombre_tanda>_<date_time>/`` -- or beside the project file when the project cannot be read.
When axons are estimated above 2 hours and nobody accepted them (user decision P7, 2026-11-07): in an interactive
console the list is shown and the user is asked; otherwise nothing starts and the list is written to
``axones_largos_a_confirmar.txt`` in the same folder.

Exit codes: 0 the batch may start; 1 errors; 2 the project cannot be read or the arguments are wrong; 3 axons above
2 hours wait for a confirmation.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

EXIT_OK, EXIT_ERRORS, EXIT_UNREADABLE, EXIT_CONFIRM = 0, 1, 2, 3
LONG_LIST_NAME = "axones_largos_a_confirmar.txt"


def _interactive() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _say(text: str) -> None:
    if sys.stdout is not None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"))


def main(argv: Optional[List[str]] = None, *, interactive: Optional[bool] = None,
         ask: Optional[Callable[[str], str]] = None) -> int:
    from tools import mps_precheck as pc

    ap = argparse.ArgumentParser(prog="python -m tools.launcher_check",
                                 description="Pre-checks of the MPS Explorer launcher (nothing is analysed).")
    ap.add_argument("proyecto", help="the project file (proyecto.yaml)")
    ap.add_argument("--aceptar-axones-largos", action="store_true",
                    help="accept every axon estimated above 2 hours (unattended run)")
    ap.add_argument("--sin-contar-clusteres", action="store_true",
                    help="do not cluster the files for the estimate (faster; the time is then a lower bound)")
    ap.add_argument("--informe-en", default=None, help="folder for the report (default: <salida>/chequeos/...)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as error:
        return EXIT_UNREADABLE if error.code else EXIT_OK
    if not os.path.isfile(args.proyecto):
        _say(f"No existe el archivo de proyecto: {args.proyecto}")
        return EXIT_UNREADABLE

    def progress(i: int, n: int, name: str) -> None:
        _say(f"  [{i + 1}/{n}] {name}")

    _say(f"Chequeos previos de {args.proyecto} ...")
    result = pc.run_prechecks(args.proyecto, accept_long=args.aceptar_axones_largos,
                              count_clusters=False if args.sin_contar_clusteres else None, on_file=progress)
    interactive = _interactive() if interactive is None else interactive
    if result.status == "confirmar" and interactive:
        pc.confirm_long_axons(result, ask or input)
    folder = args.informe_en or pc.report_folder(result)
    if args.informe_en:
        os.makedirs(folder, exist_ok=True)
    txt, _csv = pc.write_report(result, folder)
    if result.status == "confirmar":
        with open(os.path.join(folder, LONG_LIST_NAME), "w", encoding="utf-8", newline="\n") as handle:
            handle.write(pc.long_axons_text(result) + "\n")
    _say("")
    _say(pc.STATUS_WORDS[result.status])
    _say(f"Informe: {txt}")
    if result.project is None:
        return EXIT_UNREADABLE
    if result.status == "errores":
        return EXIT_ERRORS
    if result.status == "confirmar":
        _say(f"Lista de axones largos: {os.path.join(folder, LONG_LIST_NAME)}")
        return EXIT_CONFIRM
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
