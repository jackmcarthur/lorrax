"""
psp/operator_checks.py — Pre-flight checks for DFT operator construction.

Call `validate_operator_inputs(...)` before computing kin+ion, dipole
matrix elements, or any other quantity that depends on pseudopotentials
and the Coulomb truncation scheme.  The function raises immediately on
fatal problems (missing pseudopotentials, atom/PP mismatch) and returns
a validated `truncation_2d` flag derived from `sys_dim`.

Usage
-----
    from psp.operator_checks import validate_operator_inputs

    ctx = validate_operator_inputs(
        pseudos=pseudos,
        wfn=wfn,
        sys_dim=params.get("sys_dim", 3),
    )
    # ctx.truncation_2d  — bool, safe to pass to build_local_ionic_potential_on_G_total
    # ctx.pseudos        — the validated dict (same object, just confirms non-empty)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from file_io import WFNReader


@dataclass(frozen=True)
class OperatorContext:
    """Validated operator construction parameters."""

    truncation_2d: bool
    sys_dim: int
    pseudos: dict


def validate_operator_inputs(
    pseudos: dict,
    wfn: WFNReader,
    sys_dim: int = 3,
    *,
    caller: str = "",
) -> OperatorContext:
    """Validate inputs before any operator matrix-element computation.

    Parameters
    ----------
    pseudos : dict
        Element-symbol → parsed UPF object, from ``load_pseudopotentials``.
    wfn : WFNReader
        Loaded wavefunction file (needed for atom types).
    sys_dim : int
        System dimensionality: 0 (molecule), 2 (slab), 3 (bulk).
    caller : str, optional
        Label for error messages (e.g. ``"get_kin_ion_chunked"``).

    Returns
    -------
    OperatorContext with validated ``truncation_2d`` flag and references.

    Raises
    ------
    RuntimeError
        If pseudopotentials are missing or do not cover all atom species,
        or if ``sys_dim`` is not one of {0, 2, 3}.
    """
    tag = f"[{caller}] " if caller else ""

    # ---- pseudopotentials must be present ----
    if not pseudos:
        raise RuntimeError(
            f"{tag}No pseudopotentials loaded.  "
            "Ensure *.upf files are in the working directory or pass an "
            "explicit pseudo_dir.  Without pseudopotentials V_loc and V_NL "
            "are zero and the resulting matrix elements are meaningless."
        )

    # ---- every atom species must have a matching PP ----
    import numpy as np

    atom_types = np.asarray(wfn.atom_types, dtype=int)
    unique_z = set(int(z) for z in atom_types)

    from psp.pseudos import symbol_to_Z as _symbol_to_Z

    covered_z = set()
    for elem in pseudos:
        z = _symbol_to_Z(elem)
        if z is not None:
            covered_z.add(z)

    missing = unique_z - covered_z
    if missing:
        raise RuntimeError(
            f"{tag}Pseudopotentials missing for atomic numbers {sorted(missing)}.  "
            f"Loaded PPs cover Z = {sorted(covered_z)}, "
            f"but the structure contains Z = {sorted(unique_z)}."
        )

    # ---- sys_dim → truncation_2d ----
    if sys_dim not in (0, 2, 3):
        raise RuntimeError(
            f"{tag}sys_dim={sys_dim} is not supported (must be 0, 2, or 3)."
        )

    truncation_2d = sys_dim == 2

    return OperatorContext(
        truncation_2d=truncation_2d,
        sys_dim=sys_dim,
        pseudos=pseudos,
    )
