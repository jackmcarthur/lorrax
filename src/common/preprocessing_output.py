"""Shared production report for LORRAX's mean-field preprocessing drivers.

This module owns presentation only.  Dipole and ``kin_ion`` hand it facts
from their canonical WFN, symmetry, metadata and operator objects; it does
not reopen files or reconstruct any physical policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from common.scientific_output import (
    FLOAT_DIGITS,
    abs_path,
    architecture_lines,
    file_table_lines,
    numerical_environment_lines,
    symmetry_sampling_lines,
)


_WARNING_WORDS = (
    "WARNING", "[WARN]", "FATAL", "FAILURE", "FAILED", "UNPHYSICAL",
    "NOT TRUSTWORTHY", "DEPRECATED",
)


class ScientificProductionReport:
    """One rank-zero scientific stream for a LORRAX core driver."""

    def __init__(self, path: str, *, runtime, debug: bool, stdout,
                 driver_name: str, calculation_name: str) -> None:
        self.path = abs_path(path)
        self.runtime = runtime
        self.debug = bool(debug)
        self.stdout = stdout
        self.driver_name = str(driver_name)
        self.calculation_name = str(calculation_name)
        self.rank = int(getattr(runtime, "process_index", 0))
        self._warnings: list[str] = []
        self._warnings_emitted = False
        self._stream = None
        if self.rank == 0:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.path, "w", encoding="utf-8", buffering=1)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def emit(self, line: str = "") -> None:
        if self.rank != 0:
            return
        text = str(line)
        self.stdout(text)
        if self._stream is not None:
            self._stream.write(text + "\n")

    def progress(self, line: str) -> None:
        self.emit(line)

    def legacy_print(self, *args, sep: str = " ", **kwargs) -> None:
        """Retain selected cadence/warnings; hide component chatter."""
        text = sep.join(str(value) for value in args)
        if (text.startswith("Started ") or text.startswith("Finished ")
                or (text.startswith("[ ") and " | " in text
                    and " / " in text)):
            self.progress(text)
            return
        if self.debug:
            self.stdout(text, **kwargs)
            return
        upper = text.upper()
        if any(word in upper for word in _WARNING_WORDS):
            cleaned = " ".join(text.split())
            if cleaned and cleaned not in self._warnings:
                self._warnings.append(cleaned)

    def heading(self, title: str) -> None:
        self.emit()
        self.emit(title.upper())
        self.emit("-" * len(title))

    def begin(self, *, input_file: str) -> None:
        self.emit("=" * 78)
        self.emit(f"LORRAX {self.calculation_name.upper()}")
        self.emit("=" * 78)
        self.emit(f"Driver         : {self.driver_name}")
        self.emit(f"Input          : {abs_path(input_file)}")

    def architecture(self, *, mesh_role: str) -> None:
        self.heading("Processor architecture")
        for line in architecture_lines(self.runtime, mesh_role=mesh_role):
            self.emit(line)

    def environment(self, *, wfn, lines: Iterable[str] = ()) -> None:
        self.heading("Numerical environment")
        for line in numerical_environment_lines(self.runtime):
            self.emit(line)
        self.emit(f"Wavefunctions  : {getattr(wfn, 'backend', 'unknown')} reader")
        for line in lines:
            self.emit(str(line))
        self.emit("Debug stream   : " + (
            "ON via LORRAX_DEBUG_PRINT" if self.debug else
            "OFF (set LORRAX_DEBUG_PRINT=1 for component diagnostics)"))

    def pathways(self, lines: Iterable[str]) -> None:
        self.heading("Physical and numerical pathways")
        for line in lines:
            self.emit(str(line))

    def system(self, *, natoms: int, species: Iterable[str], fft_grid,
               lines: Iterable[str] = ()) -> None:
        grid = tuple(int(value) for value in fft_grid)
        self.heading("System and basis")
        self.emit(f"Atoms          : {int(natoms)}")
        self.emit("Species / UPF  : " + (", ".join(str(v) for v in species)
                                          or "none"))
        self.emit(f"FFT grid       : {grid[0]} x {grid[1]} x {grid[2]}")
        for line in lines:
            self.emit(str(line))

    def sampling(self, *, wfn, sym) -> None:
        self.heading("Crystal symmetry and Brillouin-zone sampling")
        for line in symmetry_sampling_lines(wfn, sym, digits=FLOAT_DIGITS):
            self.emit(line)

    def bands(self, lines: Iterable[str]) -> None:
        self.heading("Electronic and band spaces")
        for line in lines:
            self.emit(str(line))

    def timings(self, stages: Iterable[tuple[str, float]], *, wall: float) -> None:
        self.heading("Major-stage timing")
        self.emit("  stage                    wall (s)     fraction")
        for name, seconds in stages:
            seconds = float(seconds)
            fraction = 100.0 * seconds / wall if wall else 0.0
            self.emit(f"  {str(name):<22} {seconds:14.5f}  {fraction:9.5f}%")
        self.emit(f"  {'total run':<22} {float(wall):14.5f}  {100.0:9.5f}%")

    def warnings(self) -> None:
        if self._warnings_emitted:
            return
        self._warnings_emitted = True
        if self._warnings:
            self.heading("Warnings")
            for warning in self._warnings:
                self.emit(f"  {warning}")

    def files(self, rows: Iterable[tuple[str, str, str]]) -> None:
        self.heading("Output files and inputs")
        for line in file_table_lines(rows):
            self.emit(line)

    def finish(self, *, status: str = "completed") -> None:
        self.warnings()
        self.emit()
        self.emit(f"LORRAX {self.calculation_name} {status}.")
        self.emit(f"Report written to {self.path}")
        self.close()


PreprocessingProductionReport = ScientificProductionReport


def timing_total(records, *names: str) -> float:
    """Sum canonical timing records without exposing collector internals."""
    wanted = set(names)
    return sum(float(row["inclusive"]) for row in records
               if str(row["name"]) in wanted)


__all__ = [
    "PreprocessingProductionReport",
    "ScientificProductionReport",
    "timing_total",
]
