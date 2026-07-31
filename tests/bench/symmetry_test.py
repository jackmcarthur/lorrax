"""Debug utility for validating WFN symmetry metadata and SymMaps unfolding."""

from __future__ import annotations

import argparse

from file_io import WfnLoader as WFNReader
from common.symmetry_maps import SymMaps


def _status_line(label: str, passed: bool, detail: str) -> str:
    tag = "PASS" if passed else "FAIL"
    return f"[{tag}] {label}: {detail}"


def run_symmetry_test(wfn_path: str, tol: float = 1e-6) -> int:
    """Run symmetry audits and print a compact PASS/FAIL report."""
    wfn = WFNReader(wfn_path)
    sym = SymMaps(wfn)

    atom_failures = sym.validate_atomic_symmetries(wfn, tol=tol)
    k_failures = sym.validate_kgrid_unfolding(wfn, tol=tol)

    n_sym_fail = len(atom_failures)
    n_k_fail = len(k_failures)
    n_sym_ok = int(wfn.ntran) - n_sym_fail
    nk_total = int(sym.unfolded_kpts.shape[0])
    nk_ok = nk_total - n_k_fail

    print("Symmetry Test")
    print(f"WFN: {wfn_path}")
    print(_status_line("Atomic symmetries", n_sym_fail == 0, f"{n_sym_ok}/{int(wfn.ntran)} valid"))
    print(_status_line("K-grid unfolding", n_k_fail == 0, f"{nk_ok}/{nk_total} valid"))

    examples = atom_failures[:3] + k_failures[:3]
    if examples:
        print("Examples:")
        for msg in examples:
            print(f"  - {msg}")
        if n_sym_fail + n_k_fail > len(examples):
            print("  - ... additional failures omitted ...")

    return 0 if (n_sym_fail == 0 and n_k_fail == 0) else 1


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("wfn", help="Path to WFN.h5")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for periodic coordinate matching",
    )
    args = parser.parse_args()

    raise SystemExit(run_symmetry_test(args.wfn, tol=args.tol))


if __name__ == "__main__":
    main()
