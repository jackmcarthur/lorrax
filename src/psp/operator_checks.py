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

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402


@dataclass(frozen=True)
class OperatorContext:
    """Validated operator construction parameters."""

    truncation_2d: bool
    sys_dim: int
    pseudos: dict


def validate_operator_inputs(
    pseudos: dict,
    wfn: WfnLoader,
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


# ---------------------------------------------------------------------------
# Degeneracy consistency: does the operator have the symmetry of the states?
# ---------------------------------------------------------------------------

def check_degeneracy_consistency(
    H, energies, *, el_tol_ry: float = 1e-9, split_tol_ry: float = 1e-5,
    max_report: int = 8, label: str = "kin_ion", print_fn=print,
) -> dict:
    """Flag operator blocks that split a manifold the eigenvalues do not.

    THE ARGUMENT.  A set of bands degenerate in ``el`` spans an invariant
    subspace of the Hamiltonian that produced them.  ``H`` here is a PIECE of
    that Hamiltonian (T + V_loc + V_NL).  If ``H`` restricted to such a
    manifold is NOT a multiple of the identity, then ``H`` distinguishes
    states the full Hamiltonian did not — it has strictly lower symmetry than
    the operator the wavefunctions diagonalise.  For a fully-relativistic
    pseudopotential the usual cause is that V_NL was built j-RESOLVED while
    the wavefunctions came from a j-AVERAGED (``lspinorb=.false.``) run: the
    j-resolved V_NL carries spin-orbit, so it splits exactly the manifolds
    spin-orbit is allowed to split, in exactly the pattern it would produce.

    This needs NO metadata — not ``<spinorbit>``, not a deck key, not the UPF.
    It compares the operator against the wavefunctions it is being applied to,
    which is the only comparison that is always available.

    Eigenvalues of the restricted block are used, not diagonal entries: within
    a degenerate manifold the band basis is an arbitrary unitary, so only the
    block SPECTRUM is gauge-invariant.  A diagonal-only test reports scatter
    where there is a clean split and misses the structure entirely.

    WHEN THIS RETURNS CLEAN (constructed, not assumed):
      * scalar (nspinor=1) runs — V_NL is spin-scalar, cannot split;
      * genuine ``lspinorb=.true.`` runs — ``el`` is ALREADY split into the
        double-group multiplets, so each manifold is a single irrep and the
        j-resolved V_NL is a multiple of the identity on it;
      * FR pseudo + j-AVERAGED V_NL — spin-scalar by construction.
    Verified against this repo's fixtures: the Si ``lspinorb=.true.`` archive
    run is clean at Γ, the ``lspinorb=.false.`` one is not.

    KNOWN FALSE-POSITIVE CHANNEL, stated so it is not mistaken for proof: a
    manifold that is an ACCIDENTAL degeneracy of two distinct irreps may be
    split by a perfectly correct operator.  A split here means "this operator
    has lower symmetry than ``el``", which is a fact worth surfacing; it is
    evidence of a mode mismatch, not a proof of one.

    Parameters
    ----------
    H : (nk, nb, nb) — operator matrix in the band basis, Ry.
    energies : (nk, nb) — the eigenvalues those bands carry, Ry.
    el_tol_ry : bands closer than this are treated as one manifold.
    split_tol_ry : a block spectrum spread above this is reported.

    Returns
    -------
    dict with ``n_manifolds``, ``n_split``, ``max_split_ry``, ``worst`` and
    ``clean``.
    """
    import numpy as np

    H = np.asarray(H)
    en = np.asarray(energies, dtype=np.float64)
    nk = min(H.shape[0], en.shape[0])
    nb = min(H.shape[1], en.shape[1])

    n_manifolds = 0
    findings = []
    max_split = 0.0

    for ik in range(nk):
        e = en[ik, :nb]
        i = 0
        while i < nb:
            j = i
            while j + 1 < nb and abs(e[j + 1] - e[i]) <= el_tol_ry:
                j += 1
            deg = j - i + 1
            if deg > 1:
                n_manifolds += 1
                blk = np.asarray(H[ik, i:j + 1, i:j + 1])
                w = np.linalg.eigvalsh((blk + blk.conj().T) / 2.0).real
                spread = float(w.max() - w.min())
                max_split = max(max_split, spread)
                if spread > split_tol_ry:
                    dev = w - w.mean()
                    # degeneracy pattern of the SPLIT spectrum: a 2+4 here is
                    # the Γ8/Γ7 signature, a scatter is something else.
                    pat, run = [], 1
                    srt = np.sort(dev)
                    for q in range(1, deg):
                        if abs(srt[q] - srt[q - 1]) <= max(split_tol_ry * 0.1,
                                                           1e-12):
                            run += 1
                        else:
                            pat.append(run)
                            run = 1
                    pat.append(run)
                    findings.append(dict(ik=ik, band0=i, deg=deg, el=float(e[i]),
                                         spread_ry=spread, pattern=pat,
                                         dev_ry=dev.tolist()))
            i = j + 1

    findings.sort(key=lambda d: -d["spread_ry"])
    clean = not findings
    RY2MEV = 13605.693122994

    if not clean:
        bar = "=" * 78
        print_fn(f"\n{bar}")
        print_fn(f"DEGENERACY MISMATCH: {label} splits manifolds that "
                 f"'el' does not.")
        print_fn(bar)
        print_fn(f"  {len(findings)} of {n_manifolds} degenerate manifolds "
                 f"(|Δel| ≤ {el_tol_ry:.1e} Ry) are split by more than "
                 f"{split_tol_ry * RY2MEV:.3f} meV.")
        print_fn(f"  worst split = {max_split * RY2MEV:.4f} meV")
        print_fn("")
        print_fn(f"  {'k':>5} {'band0':>6} {'deg':>4} {'el (Ry)':>14} "
                 f"{'split (meV)':>13}  pattern")
        for d in findings[:max_report]:
            print_fn(f"  {d['ik']:5d} {d['band0']:6d} {d['deg']:4d} "
                     f"{d['el']:14.9f} {d['spread_ry'] * RY2MEV:13.4f}  "
                     f"{d['pattern']}")
        if len(findings) > max_report:
            print_fn(f"  ... {len(findings) - max_report} more")
        print_fn("")
        print_fn("  This operator distinguishes states the wavefunctions'")
        print_fn("  Hamiltonian did not.  With a fully-relativistic")
        print_fn("  pseudopotential the usual cause is a j-RESOLVED V_NL built")
        print_fn("  against wavefunctions from noncolin=.true.,")
        print_fn("  lspinorb=.false. (QE ran average_pp; its eigenvalues carry")
        print_fn("  no spin-orbit).  A 2+4 pattern on a 6-fold at Γ is the")
        print_fn("  Γ8/Γ7 spin-orbit signature.  The V_NL builder resolves")
        print_fn("  j-resolved vs j-averaged automatically by this same")
        print_fn("  measurement (psp.vnl_ops.measure_soc_mode), so a split")
        print_fn("  HERE means the operator disagrees with the wavefunctions")
        print_fn("  anyway — check the WFN / pseudopotential pairing.")
        print_fn(bar + "\n")
    else:
        print_fn(f"  {label}: degeneracy-consistent — "
                 f"{n_manifolds} degenerate manifolds, "
                 f"max split {max_split * RY2MEV:.4f} meV "
                 f"(≤ {split_tol_ry * RY2MEV:.3f} meV).")

    return dict(n_manifolds=n_manifolds, n_split=len(findings),
                max_split_ry=max_split, worst=findings[:max_report],
                clean=clean)
