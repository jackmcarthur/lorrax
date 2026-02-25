"""Write BSE eigenvectors to HDF5 file in BerkeleyGW format.

This is a small utility for producing an `eigenvectors.h5` file compatible with
BerkeleyGW's format (see `context/eigenvectors.h5.spec`).

Notes:
- This file is intentionally kept separate from the main BSE solvers.
- For now this primarily supports the TDA-style eigenvectors (A amplitudes).
"""

from __future__ import annotations

import argparse
from typing import Optional

import h5py
import numpy as np


def generate_kpts_grid(nkx: int, nky: int, nkz: int) -> np.ndarray:
    """Generate a simple Monkhorst-Pack style k-point grid in crystal coordinates [0,1)."""
    kpts = []
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                kx = ix / nkx
                ky = iy / nky
                kz = iz / nkz if nkz > 0 else 0.0
                kpts.append([kx, ky, kz])
    return np.asarray(kpts, dtype=np.float64)


def write_eigenvectors_h5(
    output_file: str,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    kpts: np.ndarray,
    n_val: int,
    n_cond: int,
    nkx: int = 1,
    nky: int = 1,
    nkz: int = 1,
    version: int = 1,
    exciton_Q_shifts: Optional[np.ndarray] = None,
) -> None:
    """Write BSE eigenvectors to HDF5 (BerkeleyGW format).

    Args:
        output_file: Path to output HDF5 file.
        eigenvalues: (n_eig,) exciton energies in Ry.
        eigenvectors: (n_eig, nc, nv, nk) or (n_eig, ns, nc, nv, nk) complex A_cvk amplitudes.
        kpts: (nk, 3) k-point coordinates in crystal units.
        n_val: Number of valence bands.
        n_cond: Number of conduction bands.
        nkx, nky, nkz: k-grid dimensions (for metadata only).
        version: File version number.
        exciton_Q_shifts: (nQ, 3) Q-shift vectors. Default: single Q=0.
    """
    eigenvalues = np.asarray(eigenvalues)
    eigenvectors = np.asarray(eigenvectors)
    kpts = np.asarray(kpts)

    n_eig = int(eigenvalues.shape[0])

    # Accept both (n_eig, nc, nv, nk) and (n_eig, ns, nc, nv, nk).
    if eigenvectors.ndim == 4:
        eigenvectors = eigenvectors[:, np.newaxis, :, :, :]
    if eigenvectors.ndim != 5:
        raise ValueError(f"unexpected eigenvector rank {eigenvectors.ndim}; expected 4 or 5")

    n_eig_check, ns, nc, nv, nk = eigenvectors.shape
    if n_eig_check != n_eig:
        raise ValueError(f"Mismatch: {n_eig_check} vs {n_eig} eigenvalues")
    if nc != n_cond or nv != n_val:
        raise ValueError(f"Mismatch: eigenvectors (nc,nv)=({nc},{nv}) vs ({n_cond},{n_val})")

    if exciton_Q_shifts is None:
        exciton_Q_shifts = np.zeros((1, 3), dtype=np.float64)
    nQ = int(exciton_Q_shifts.shape[0])

    # BSE Hamiltonian size (TDA): ns * nk * nv * nc
    bse_hamiltonian_size = int(ns * nk * nv * nc)
    evec_sz = bse_hamiltonian_size

    # Determine flavor (1 = real, 2 = complex)
    is_complex = np.iscomplexobj(eigenvectors)
    flavor = 2 if is_complex else 1

    # Spinor calculation: spin_kernel = 3 (BerkeleyGW convention).
    spin_kernel = 3

    kpts_fortran = kpts.T.copy()  # (3, nk)

    # Spec says dims are: [2 if complex], ns, nv, nc, nk, nevecs, nQ.
    # We'll store as: (nQ, nevecs, nk, nc, nv, ns, 2) for complex.
    evecs_reordered = eigenvectors.transpose(0, 4, 2, 3, 1)  # (n_eig, nk, nc, nv, ns)
    evecs_with_Q = evecs_reordered[np.newaxis, ...]  # (1, n_eig, nk, nc, nv, ns)

    if is_complex:
        evecs_storage = np.stack([evecs_with_Q.real, evecs_with_Q.imag], axis=-1)
    else:
        evecs_storage = evecs_with_Q.real

    with h5py.File(output_file, "w") as f:
        f.create_group("mf_header")
        f.create_group("eps_header")
        f.create_group("bse_header")

        exciton_header = f.create_group("exciton_header")
        exciton_header.create_dataset("version", data=version)
        exciton_header.create_dataset("flavor", data=flavor)

        params = exciton_header.create_group("params")
        params.create_dataset("bse_hamiltonian_size", data=bse_hamiltonian_size)
        params.create_dataset("evec_sz", data=evec_sz)
        params.create_dataset("spin_kernel", data=spin_kernel)
        params.create_dataset("nevecs", data=n_eig)
        params.create_dataset("ns", data=ns)
        params.create_dataset("nc", data=nc)
        params.create_dataset("nv", data=nv)
        params.create_dataset("use_tda", data=1)

        kpoints = exciton_header.create_group("kpoints")
        kpoints.create_dataset("nk", data=nk)
        kpoints.create_dataset("kpts", data=kpts_fortran)
        kpoints.create_dataset("nQ", data=nQ)
        kpoints.create_dataset("exciton_Q_shifts", data=exciton_Q_shifts.T)

        exciton_data = f.create_group("exciton_data")
        exciton_data.create_dataset("eigenvalues", data=eigenvalues)
        exciton_data.create_dataset("eigenvectors", data=evecs_storage)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write BSE eigenvectors to HDF5 (BerkeleyGW format).")
    parser.add_argument("input_npz", help="Input .npz file with 'eigenvalues' and 'eigenvectors' arrays.")
    parser.add_argument("-o", "--output", default="eigenvectors.h5", help="Output HDF5 file.")
    parser.add_argument("--n-val", type=int, required=True, help="Number of valence bands.")
    parser.add_argument("--n-cond", type=int, required=True, help="Number of conduction bands.")
    parser.add_argument("--nkx", type=int, default=1, help="k-grid dimension x.")
    parser.add_argument("--nky", type=int, default=1, help="k-grid dimension y.")
    parser.add_argument("--nkz", type=int, default=1, help="k-grid dimension z.")
    args = parser.parse_args(argv)

    data = np.load(args.input_npz)
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]

    kpts = generate_kpts_grid(args.nkx, args.nky, args.nkz)
    write_eigenvectors_h5(
        args.output,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kpts=kpts,
        n_val=args.n_val,
        n_cond=args.n_cond,
        nkx=args.nkx,
        nky=args.nky,
        nkz=args.nkz,
    )


if __name__ == "__main__":
    main()

