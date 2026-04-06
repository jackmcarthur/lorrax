"""
psp/dft_operators.py — Unified PW DFT operator kernels.

Provides fused, JIT-compiled operator kernels for the plane-wave DFT
Hamiltonian: T + V_loc + V_NL [+ V_H].  All other modules that need
DFT matrix elements or matvecs should source core functionality here.

Representations
---------------
  sparse-G : (nvec, nspinor, nG) — coefficients at valid G-vectors only
  FFT box  : (nvec, nspinor, nx, ny, nz) — dense 3-D grid (used
             internally by V_loc for the real-space multiply)

Normalization
-------------
All operators use the convention:

    <m|O|n> = sum_{s,G} conj(psi_m[s,G]) * (O psi)_n[s,G]

with no volume prefactors.  See hamiltonian_matvec.py docstring for
the derivation showing scale * deltaV * fft_norm * sqrt(1/Omega) = 1.
"""
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import common.timing as timing
from file_io import WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox

from psp.get_DFT_mtxels import (
    read_cohsex_input,
    generate_gvectors_k,
    load_pseudopotentials,
    build_atom_pp_assignments,
    compute_valence_density,
    compute_hartree_potential_real,
)
from psp.build_projectors_qe import (
    build_local_ionic_potential_on_G_total,
    qe_real_sph_harmonics,
)
from psp.projector_pipeline import build_vnl_plan


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OperatorSetup:
    """K-point-independent operator data.  Built once, shared across k."""
    V_r: jax.Array                  # (nx, ny, nz) V_loc [+V_H] [Ry]
    vnl_plan: dict
    assignments: list
    species_payload: list
    bdot: np.ndarray                # (3,3)
    B: np.ndarray                   # crystal-to-Cartesian K = k+G
    cell_volume: float
    fft_grid: tuple[int, int, int]


@dataclass
class KPointOperators:
    """Per-k precomputed operator data."""
    T_diag: jax.Array               # (nG,) kinetic diagonal [Ry]
    Gx: jax.Array                   # (nG,) int32
    Gy: jax.Array                   # (nG,) int32
    Gz: jax.Array                   # (nG,) int32
    V_r: jax.Array                  # shared ref to OperatorSetup.V_r
    vnl_projectors: list[tuple[jax.Array, jax.Array]]  # [(Z, E), ...]
    nG: int
    fft_grid: tuple[int, int, int]


# ---------------------------------------------------------------------------
# Setup builders  (thin wrappers — the heavy lifting is in existing code)
# ---------------------------------------------------------------------------

def build_operator_setup(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    pseudos: dict,
    *,
    include_hartree: bool = False,
    global_psi_G: jax.Array | None = None,
    truncation_2d: bool = True,
) -> OperatorSetup:
    """Precompute k-independent data: V_loc, VNL plan, optional V_H."""
    atom_pos = jnp.asarray(wfn.atom_crys, dtype=jnp.float64)
    atom_types = jnp.asarray(wfn.atom_types, dtype=jnp.int32)
    assignments = build_atom_pp_assignments(atom_pos, atom_types, pseudos)

    tmp: dict[int, dict] = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        key = id(ap.pseudo)
        entry = tmp.setdefault(key, {"pseudo": ap.pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    species_payload = [
        (e["pseudo"],
         np.asarray(e["positions"], dtype=float)
         if e["positions"] else np.zeros((0, 3), dtype=float))
        for e in tmp.values()
    ]

    V_loc_r = build_local_ionic_potential_on_G_total(
        assignments=[
            {"pseudo": ap.pseudo,
             "position": np.asarray(ap.position, dtype=float)}
            for ap in assignments
        ],
        species_groups=[
            (sp[0],
             (np.asarray(sp[1], dtype=float)
              if np.asarray(sp[1]).size > 0
              else np.zeros((0, 3), dtype=float)))
            for sp in species_payload
        ],
        fft_grid=tuple(int(x) for x in meta.fft_grid),
        bdot=np.asarray(wfn.bdot, dtype=float),
        cell_volume=float(wfn.cell_volume),
        bvec=np.asarray(wfn.bvec, dtype=float),
        blat=float(wfn.blat),
        truncation_2d=truncation_2d,
    )
    V_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    if include_hartree:
        if global_psi_G is None:
            raise ValueError("global_psi_G required for include_hartree")
        rho_val = compute_valence_density(global_psi_G, sym, wfn)
        V_H_r = compute_hartree_potential_real(
            rho_val,
            jnp.asarray(wfn.bdot, dtype=jnp.float64),
            bvec=jnp.asarray(wfn.bvec, dtype=jnp.float64),
            blat=float(wfn.blat),
            truncation_2d=False,
        )
        V_r = V_r + jnp.asarray(V_H_r, dtype=jnp.float64)

    bvec_np = np.asarray(wfn.bvec, dtype=float).T
    B = float(wfn.blat) * bvec_np.T
    q_max = 0.0
    for ik in range(sym.nk_tot):
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
        K_cart = (np.asarray(Gk_crys, dtype=float) + kvec[None, :]) @ B
        qk = np.sqrt(np.sum(K_cart**2, axis=1))
        if qk.size:
            q_max = max(q_max, float(np.max(qk)))

    vnl_plan = build_vnl_plan(
        pseudos, assignments, float(wfn.cell_volume), float(q_max)
    )

    return OperatorSetup(
        V_r=V_r,
        vnl_plan=vnl_plan,
        assignments=assignments,
        species_payload=species_payload,
        bdot=np.asarray(wfn.bdot, dtype=float),
        B=B,
        cell_volume=float(wfn.cell_volume),
        fft_grid=tuple(int(x) for x in meta.fft_grid),
    )


def build_kpoint_operators(
    k_idx: int,
    setup: OperatorSetup,
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    nspinor: int | None = None,
) -> KPointOperators:
    """Build per-k data: kinetic diagonal and VNL projectors Z."""
    if nspinor is None:
        nspinor = int(meta.nspinor)

    Gk_crys, _ = generate_gvectors_k(k_idx, sym, wfn, meta)
    Gk_np = np.asarray(Gk_crys, dtype=int)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)
    nG = Gk_np.shape[0]

    Gx = jnp.asarray(Gk_np[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_np[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_np[:, 2], dtype=jnp.int32)

    G_float = np.asarray(Gk_np, dtype=float)
    K_crys = G_float + kvec[None, :]
    T_diag = jnp.asarray(
        np.einsum('gi,ij,gj->g', K_crys, setup.bdot, K_crys),
        dtype=jnp.float64,
    )

    K_cart = K_crys @ setup.B
    K_norm = np.sqrt(np.sum(K_cart**2, axis=1))
    K_crys_j = jnp.asarray(K_crys, dtype=jnp.float64)

    vnl_projectors: list[tuple[jax.Array, jax.Array]] = []
    Y_cache: dict[int, jax.Array] = {}

    for _key, sp in setup.vnl_plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=float)
        if tau.size == 0:
            continue
        if tau.ndim == 1:
            tau = tau.reshape(1, 3)
        natoms = tau.shape[0]
        pref = float(sp['prefactor'])
        splines = sp['splines']

        tau_j = jnp.asarray(tau, dtype=jnp.float64)
        phase = jnp.exp(
            -2j * jnp.pi * (K_crys_j @ tau_j.T)
        ).T

        for l_key, info in sp['l_channels'].items():
            l = int(l_key)
            E_np = info['E']
            if E_np is None:
                continue
            beta_ids = info['beta_ids']
            if not beta_ids:
                continue
            nbeta = len(beta_ids)
            msize = 2 * l + 1

            F_bG = np.stack(
                [splines[(l, int(bid))](K_norm) for bid in beta_ids],
                axis=0,
            )
            radial = jnp.asarray(
                pref * (1j) ** l * F_bG, dtype=jnp.complex128,
            )

            if l not in Y_cache:
                Y_cache[l] = jnp.asarray(
                    qe_real_sph_harmonics(l, K_cart), dtype=jnp.complex128,
                )
            Y = Y_cache[l]

            Z_bmg = radial[:, None, :] * Y[None, :, :]
            Z_atoms = phase[:, None, None, :] * Z_bmg[None, ...]
            R = nbeta * msize
            Z_flat = Z_atoms.reshape(natoms, R, nG)

            E_j = info.get('E_j')
            if E_j is None:
                E_j = jnp.asarray(E_np, dtype=jnp.complex128)
            E_j = E_j[:nspinor, :nspinor]

            vnl_projectors.append((Z_flat, E_j))

    return KPointOperators(
        T_diag=T_diag,
        Gx=Gx, Gy=Gy, Gz=Gz,
        V_r=setup.V_r,
        vnl_projectors=vnl_projectors,
        nG=nG,
        fft_grid=setup.fft_grid,
    )


# ---------------------------------------------------------------------------
# Core fused kernels
# ---------------------------------------------------------------------------

@jax.jit
def apply_H_k(psi_box, T_diag, V_r, Gx, Gy, Gz, vnl_ZE):
    """Fused H|psi>: FFT-box in, sparse-G out.  Single JIT dispatch.

    Parameters
    ----------
    psi_box : (nvec, nspinor, nx, ny, nz) — trial vectors in FFT box
    T_diag  : (nG,) — kinetic diagonal
    V_r     : (nx, ny, nz) — real-space local potential
    Gx,Gy,Gz : (nG,) int32 — G-vector FFT-box indices
    vnl_ZE  : tuple of (Z, E) per VNL channel

    Returns
    -------
    H_psi_G : (nvec, nspinor, nG) — sparse-G
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]
    H_G = T_diag[None, None, :] * psi_G
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    H_G = H_G + jnp.fft.fftn(
        psi_r * V_r, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz]
    for Z, E in vnl_ZE:
        proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
        d = jnp.einsum('strq,aqtv->arsv', E, proj, optimize=True)
        H_G = H_G + jnp.einsum('arG,arsv->vsG', Z, d, optimize=True)
    return H_G


@jax.jit
def build_matrix_k(psi_box, T_diag, V_r, Gx, Gy, Gz, vnl_ZE):
    """Fused H matrix elements: <m|H|n> for all bands at one k-point.

    Same physics as apply_H_k, but contracts to (nb, nb) instead of
    returning sparse-G.  Single JIT dispatch.

    Parameters
    ----------
    psi_box : (nb, nspinor, nx, ny, nz) — wavefunctions
    (other args: same as apply_H_k)

    Returns
    -------
    H_mn : (nb, nb) complex128
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]            # (nb, ns, nG)

    # T
    H_mn = jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G),
        T_diag[None, None, :] * psi_G, optimize=True,
    )

    # V_loc
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    Vpsi_G = jnp.fft.fftn(
        psi_r * V_r, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz]
    H_mn = H_mn + jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G), Vpsi_G, optimize=True,
    )

    # V_NL
    for Z, E in vnl_ZE:
        proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
        d = jnp.einsum('strq,aqtv->arsv', E, proj, optimize=True)
        vnl_G = jnp.einsum('arG,arsv->vsG', Z, d, optimize=True)
        H_mn = H_mn + jnp.einsum(
            'msG,nsG->mn', jnp.conj(psi_G), vnl_G, optimize=True,
        )

    return H_mn


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def apply(psi_box: jax.Array, kops: KPointOperators) -> jax.Array:
    """Apply H|psi> at one k-point.  FFT-box in, sparse-G out."""
    return apply_H_k(
        psi_box, kops.T_diag, kops.V_r,
        kops.Gx, kops.Gy, kops.Gz,
        tuple(kops.vnl_projectors),
    )


def matrix(psi_box: jax.Array, kops: KPointOperators) -> jax.Array:
    """Build H_mn at one k-point.  Returns (nb, nb)."""
    return build_matrix_k(
        psi_box, kops.T_diag, kops.V_r,
        kops.Gx, kops.Gy, kops.Gz,
        tuple(kops.vnl_projectors),
    )


# ---------------------------------------------------------------------------
# Batched kin_ion writer (replaces gw/kin_ion_io_chunked logic)
# ---------------------------------------------------------------------------

def compute_kin_ion_all(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    setup: OperatorSetup,
    nb: int | None = None,
) -> np.ndarray:
    """Compute kin+ion matrix for all k-points.  Returns (nk, nb, nb).

    Uses the fused ``build_matrix_k`` kernel — single JIT dispatch per
    k-point instead of 3 separate dispatches in the old code.
    """
    if nb is None:
        nb = int(meta.b_id_4)
    nk = sym.nk_tot
    nspinor = int(meta.nspinor)
    kin_ion = np.zeros((nk, nb, nb), dtype=np.complex128)

    for ik in range(nk):
        with timing.section(f"dft_operators.kin_ion_k{ik}"):
            kops = build_kpoint_operators(ik, setup, wfn, sym, meta,
                                          nspinor=nspinor)
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
            H_k = matrix(wfn_k, kops)
            kin_ion[ik] = np.asarray(H_k)
            del wfn_k

    return kin_ion


# ---------------------------------------------------------------------------
# CLI: validate + benchmark
# ---------------------------------------------------------------------------

def main(argv=None):
    argp = argparse.ArgumentParser(
        description="dft_operators — validate and benchmark",
    )
    argp.add_argument("-i", "--input", required=True, help="cohsex.in path")
    argp.add_argument("-n", "--nb", type=int, default=None)
    args = argp.parse_args(argv)

    timing.reset()
    input_dir = os.path.dirname(os.path.abspath(args.input))
    params = read_cohsex_input(args.input)
    wfn_path = params.get("wfn_file", "WFN.h5")
    if not os.path.isabs(wfn_path):
        wfn_path = os.path.join(input_dir, wfn_path)

    nband = int(params.get("nband", 80))
    nval = int(params.get("nval", 26))
    ncond = int(params.get("ncond", 54))
    bispinor = bool(params.get("bispinor", False))
    nb = int(args.nb) if args.nb else nband

    print("== dft_operators: validate & benchmark ==")
    with timing.section("load"):
        wfn = WFNReader(wfn_path)
        sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(wfn, sym, nval, ncond, nb, 0, bispinor)
    print(f"  k={sym.nk_tot}, bands={nb}, nspinor={meta.nspinor}, "
          f"grid={meta.fft_grid}, devices={jax.device_count()}")

    pseudos = load_pseudopotentials(input_dir)

    with timing.section("build_setup"):
        setup = build_operator_setup(wfn, sym, meta, pseudos)

    # -- validate build_matrix_k against old code ---------------------------
    from psp.get_DFT_mtxels import compute_kinetic_k, compute_local_V_k
    from psp.projector_pipeline import compute_V_NL_k_minimal

    print("\nValidating build_matrix_k against old code...")
    all_pass = True
    for ik in range(sym.nk_tot):
        kops = build_kpoint_operators(ik, setup, wfn, sym, meta)
        wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)

        # New fused path
        H_new = matrix(wfn_k, kops)

        # Old path
        Gk_crys, kpoint = generate_gvectors_k(ik, sym, wfn, meta)
        bdot = np.asarray(wfn.bdot, dtype=float)
        kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
        T_old = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot)
        V_old = compute_local_V_k(wfn_k, Gk_crys, setup.V_r, wfn.cell_volume)
        K_crys = np.asarray(Gk_crys, dtype=float) + kvec[None, :]
        K_cart = K_crys @ setup.B
        VNL_old = compute_V_NL_k_minimal(
            wfn_k, Gk_crys, K_crys, K_cart,
            setup.vnl_plan, float(wfn.cell_volume),
        )
        H_old = jnp.asarray(T_old + V_old + VNL_old)

        err = float(jnp.max(jnp.abs(H_new - H_old)))
        ok = err < 1e-8
        all_pass = all_pass and ok
        print(f"  k={ik}: {'PASS' if ok else 'FAIL'}  max|err|={err:.2e}")

    # -- benchmark: fused build_matrix_k ------------------------------------
    print("\nBenchmark: fused build_matrix_k...")
    kops = build_kpoint_operators(0, setup, wfn, sym, meta)
    wfn_k = load_kpoint_fftbox(wfn, sym, meta, 0, nb)
    _ = matrix(wfn_k, kops); jax.block_until_ready(_)

    import time
    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        H = matrix(wfn_k, kops); jax.block_until_ready(H)
    dt_fused = (time.perf_counter() - t0) / N

    # Old separate-dispatch path
    Gk_crys, kpoint = generate_gvectors_k(0, sym, wfn, meta)
    bdot = np.asarray(wfn.bdot, dtype=float)
    kvec = np.asarray(sym.unfolded_kpts[0], dtype=float)
    _ = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot); jax.block_until_ready(_)
    t0 = time.perf_counter()
    for _ in range(N):
        T = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot)
        V = compute_local_V_k(wfn_k, Gk_crys, setup.V_r, wfn.cell_volume)
        K_crys = np.asarray(Gk_crys, dtype=float) + kvec[None, :]
        K_cart = K_crys @ setup.B
        VNL = compute_V_NL_k_minimal(
            wfn_k, Gk_crys, K_crys, K_cart,
            setup.vnl_plan, float(wfn.cell_volume),
        )
        H = T + V + VNL
        jax.block_until_ready(H)
    dt_old = (time.perf_counter() - t0) / N

    print(f"  Fused build_matrix_k: {dt_fused*1e3:.2f} ms/k")
    print(f"  Old separate calls:   {dt_old*1e3:.2f} ms/k")
    print(f"  Speedup: {dt_old/dt_fused:.1f}x")

    # -- benchmark: full kin_ion computation --------------------------------
    print(f"\nBenchmark: compute_kin_ion_all ({sym.nk_tot} k-points)...")
    with timing.section("kin_ion_all"):
        kin_ion = compute_kin_ion_all(wfn, sym, meta, setup, nb)
    print(f"  Shape: {kin_ion.shape}")

    timing.report(title="\n--- Timing (seconds) ---")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
