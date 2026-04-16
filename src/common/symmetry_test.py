"""Debug utility for validating WFN symmetry metadata and SymMaps unfolding.

Usage:
    uv run python -m common.symmetry_test /path/to/WFN.h5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .symmetry_maps import SymMaps
from .wfnreader import WFNReader


@dataclass
class SymmetryTestSummary:
    n_sym_ok: int
    n_sym_total: int
    nk_ok: int
    nk_total: int


def _uniform_kgrid_points(kgrid: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Return the uniform crystal-coordinate k-grid implied by WFN metadata."""
    kx = np.linspace(0.0, 1.0, int(kgrid[0]), endpoint=False) + shift[0] / kgrid[0]
    ky = np.linspace(0.0, 1.0, int(kgrid[1]), endpoint=False) + shift[1] / kgrid[1]
    kz = np.linspace(0.0, 1.0, int(kgrid[2]), endpoint=False) + shift[2] / kgrid[2]
    mesh = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.mod(np.stack([arr.ravel() for arr in mesh], axis=1), 1.0)


def _wrap_frac(delta: np.ndarray) -> np.ndarray:
    """Wrap fractional-coordinate differences to the nearest image."""
    return delta - np.round(delta)


def _match_atom(
    transformed: np.ndarray,
    atom_positions: np.ndarray,
    atom_types: np.ndarray,
    atom_type: int,
    tol: float,
) -> int | None:
    """Return the matching atom index for a transformed fractional position."""
    candidates = np.where(atom_types == atom_type)[0]
    if candidates.size == 0:
        return None

    diffs = _wrap_frac(atom_positions[candidates] - transformed[None, :])
    metric = np.max(np.abs(diffs), axis=1)
    best = int(np.argmin(metric))
    if metric[best] > tol:
        return None
    return int(candidates[best])


def validate_atomic_symmetries(wfn: WFNReader, tol: float) -> tuple[int, list[str]]:
    """Check that every stored spatial symmetry maps atoms onto equivalent atoms."""
    atom_crys = np.asarray(wfn.atom_crys, dtype=np.float64)
    atom_types = np.asarray(wfn.atom_types)
    failures: list[str] = []
    n_ok = 0

    for sym_idx in range(int(wfn.ntran)):
        # BGW stores the accepted symmetry in ``mtrx`` after inverting the
        # raw spglib rotation. For atomic positions in crystal coordinates, use
        # the inverse of the stored matrix to recover the direct space-group
        # action x -> R x + tau.
        rot = np.asarray(np.linalg.inv(wfn.sym_matrices[sym_idx]), dtype=np.int32)
        tau = np.asarray(wfn.translations[sym_idx], dtype=np.float64) / (2.0 * np.pi)

        used_targets: set[int] = set()
        ok = True
        for atom_idx, pos in enumerate(atom_crys):
            transformed = np.mod(rot @ pos + tau, 1.0)
            match = _match_atom(
                transformed,
                atom_crys,
                atom_types,
                atom_types[atom_idx],
                tol,
            )
            if match is None or match in used_targets:
                ok = False
                failures.append(
                    f"sym {sym_idx}: atom {atom_idx} ({atom_types[atom_idx]}) "
                    f"maps to {transformed.tolist()} with no unique same-species match"
                )
                break
            used_targets.add(match)

        if ok:
            n_ok += 1

    return n_ok, failures


def validate_kpoint_unfolding(wfn: WFNReader, sym: SymMaps, tol: float) -> tuple[int, list[str]]:
    """Check that the irreducible wedge unfolds to the full uniform grid."""
    failures: list[str] = []
    full_grid = _uniform_kgrid_points(np.asarray(wfn.kgrid), np.asarray(wfn.shift))
    if full_grid.shape != sym.unfolded_kpts.shape:
        failures.append(
            f"full-grid size mismatch: generated {full_grid.shape[0]} "
            f"points, SymMaps has {sym.unfolded_kpts.shape[0]}"
        )
        return 0, failures

    nk_ok = 0
    for ik, k_full in enumerate(full_grid):
        deltas = _wrap_frac(sym.unfolded_kpts - k_full[None, :])
        metric = np.max(np.abs(deltas), axis=1)
        best = int(np.argmin(metric))
        if metric[best] > tol:
            failures.append(
                f"uniform-grid point {ik} {k_full.tolist()} missing from SymMaps.unfolded_kpts"
            )
            continue

        ik_full = best
        ik_irr = int(sym.irk_to_k_map[ik_full])
        sym_idx = int(sym.irk_sym_map[ik_full])
        sym_krep = np.asarray(sym.sym_mats_k[sym_idx], dtype=np.int32)
        kg0 = sym._get_umklapp_vector(wfn, ik_full, sym_idx, ik_irr, sym_krep)
        mapped = sym_krep @ np.asarray(wfn.kpoints[ik_irr], dtype=np.float64) + kg0
        if np.max(np.abs(mapped - sym.unfolded_kpts[ik_full])) > tol:
            failures.append(
                f"ik_full={ik_full}: S*k_irr + kg0 does not reproduce full k-point "
                f"(mapped={mapped.tolist()}, target={sym.unfolded_kpts[ik_full].tolist()})"
            )
            continue

        nk_ok += 1

    return nk_ok, failures


def run_symmetry_test(wfn_path: str, tol: float = 1e-6) -> SymmetryTestSummary:
    """Run both symmetry audits and raise on failure."""
    wfn = WFNReader(wfn_path)
    sym = SymMaps(wfn)

    n_sym_ok, atom_failures = validate_atomic_symmetries(wfn, tol)
    nk_ok, k_failures = validate_kpoint_unfolding(wfn, sym, tol)

    if atom_failures or k_failures:
        lines = ["symmetry_test failed:"]
        lines.extend(f"  - {msg}" for msg in atom_failures[:10])
        lines.extend(f"  - {msg}" for msg in k_failures[:10])
        if len(atom_failures) + len(k_failures) > 10:
            lines.append("  - ... additional failures omitted ...")
        raise SystemExit("\n".join(lines))

    return SymmetryTestSummary(
        n_sym_ok=n_sym_ok,
        n_sym_total=int(wfn.ntran),
        nk_ok=nk_ok,
        nk_total=int(np.prod(np.asarray(wfn.kgrid, dtype=np.int64))),
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wfn", help="Path to WFN.h5")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for periodic coordinate matching",
    )
    args = parser.parse_args()

    summary = run_symmetry_test(args.wfn, tol=args.tol)
    print(
        "symmetry_test passed: "
        f"{summary.n_sym_ok}/{summary.n_sym_total} spatial symmetries map atoms correctly, "
        f"{summary.nk_ok}/{summary.nk_total} uniform-grid k-points unfold correctly."
    )


if __name__ == "__main__":
    main()
