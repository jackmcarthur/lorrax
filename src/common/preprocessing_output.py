"""Shared production report for LORRAX's mean-field preprocessing drivers.

This module owns presentation only.  Dipole and ``kin_ion`` hand it facts
from their canonical WFN, symmetry, metadata and operator objects; it does
not reopen files or reconstruct any physical policy.  The stream, header,
Devices, timing, warnings, files and closing blocks are
:class:`common.production_report.ProductionReport`'s.
"""

from __future__ import annotations

from typing import Iterable

from common.production_report import ProductionReport
from common.report_blocks import Setting


class ScientificProductionReport(ProductionReport):
    """One rank-zero scientific stream for a LORRAX core driver."""

    def environment(self, *, wfn, lines: Iterable[str] = ()) -> None:
        self.heading("Numerical environment")
        self.emit(f"Wavefunctions  : {getattr(wfn, 'backend', 'unknown')} reader")
        for line in lines:
            self.emit(str(line))

    def pseudopotentials(self, lines: Iterable[str]) -> None:
        """One block per run naming every UPF read and its SR/FR nature —
        lines come from psp.pseudos.pseudo_summary_lines (the beta-channel
        verdict, not the header's)."""
        self.heading("Pseudopotentials")
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

    def bands(self, lines: Iterable[str]) -> None:
        self.heading("Electronic and band spaces")
        for line in lines:
            self.emit(str(line))


PreprocessingProductionReport = ScientificProductionReport


def soc_setting(resolved: str) -> Setting:
    """The SETTINGS row of the V_NL spin-orbit treatment, from the projector
    setup's own statement (``j-RESOLVED, from QE <spinorbit>``)."""
    text = " ".join(str(resolved).split())
    head, _, rest = text.partition(", ")
    value = head.split(" (")[0].strip().lower() or "none"
    return Setting("spin-orbit V_NL", value, "QE" if "QE" in text else "derived",
                   rest or text)


__all__ = [
    "PreprocessingProductionReport",
    "ScientificProductionReport",
    "soc_setting",
]
