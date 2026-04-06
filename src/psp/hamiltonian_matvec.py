"""
psp/hamiltonian_matvec.py — Plane-wave DFT Hamiltonian matvec H|psi>.

Applies H = T + V_loc + V_NL [+ V_H] to a batch of trial vectors in the
plane-wave FFT-box representation, *without* building the full (nb x nb)
matrix.  Designed for iterative diagonalisation (Davidson, LOBPCG, ...).

Normalization convention
------------------------
Wavefunctions are stored in the FFT-box representation psi~(G) such that
matrix elements are simple inner products over valid G-vectors:

    <m|O|n> = sum_{s,G} conj(psi~_m[s,G]) * (O psi~)_n[s,G]

with no extra volume prefactors.  This means:

  * Kinetic:  (T psi~)_G  = |k+G|^2  *  psi~_G          (diagonal in G)
  * Local V:  (V psi~)_G  = FFT(V_r * IFFT(psi~, ortho), ortho)_G
  * V_NL:     (V_NL psi~) = sum_a  Z_a  E  Z_a^dag  psi~  (KB projectors)

The local-potential identity can be verified by expanding the existing
full-matrix code in get_DFT_mtxels.py:  the product of its internal
scale, deltaV, fft_norm, and sqrt(1/Omega) factors is exactly 1.

Multi-GPU
---------
Shard trial vectors along axis 0 (nvec) over a 1-D JAX device mesh.
Each H|psi> is independent per vector; no inter-device communication.

Usage
-----
    setup = build_hamiltonian_setup(wfn, sym, meta, pseudos)
    kham  = build_kpoint_hamiltonian(k_idx, setup, wfn, sym, meta)
    Hpsi  = apply_H_k(psi, kham)       # psi: (nvec, nspinor, nx, ny, nz)

    # Multi-GPU:
    mesh = create_mesh()
    psi  = shard_trial_vectors(psi, mesh)
    Hpsi = apply_H_k(psi, kham)        # automatically parallelised
"""
from __future__ import annotations

import os
import argparse
from dataclasses import dataclass

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
    compute_kinetic_k,
    compute_local_V_k,
    compute_valence_density,
    compute_hartree_potential_real,
)
from psp.build_projectors_qe import (
    build_local_ionic_potential_on_G_total,
    qe_real_sph_harmonics,
)
from psp.projector_pipeline import build_vnl_plan, compute_V_NL_k_minimal


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HamiltonianSetup:
    """K-point-independent Hamiltonian data.  Built once, shared across k.

    Attributes
    ----------
    V_r : jax.Array (nx, ny, nz)
        Combined real-space potential V_loc [+ V_H] on WFN FFT grid [Ry].
    vnl_plan : dict
        Per-species VNL plan from ``build_vnl_plan``.
    assignments : list[AtomPP]
        Atom-to-pseudopotential assignments.
    species_payload : list[tuple]
        [(pseudo, positions_array), ...] grouped by species.
    bdot : ndarray (3,3)
        Reciprocal-space metric tensor.
    B : ndarray (3,3)
        Crystal-to-Cartesian transformation for K = k+G vectors.
    cell_volume : float
        Unit cell volume [bohr^3].
    fft_grid : tuple[int,int,int]
        WFN FFT grid dimensions (nx, ny, nz).
    """
    V_r: jax.Array
    vnl_plan: dict
    assignments: list
    species_payload: list
    bdot: np.ndarray
    B: np.ndarray
    cell_volume: float
    fft_grid: tuple[int, int, int]


@dataclass
class KPointHamiltonian:
    """Everything needed to apply H at one k-point.

    Attributes
    ----------
    T_diag : jax.Array (nG,)
        Kinetic energy diagonal: |k+G|^2 in Ry.
    Gx, Gy, Gz : jax.Array (nG,)  int32
        FFT-box indices for scatter/gather between box and sparse-G.
    g_mask : jax.Array (nG,) or None
        Padding mask (1.0 for real G, 0.0 for padding).  None when
        every entry is valid (the common per-k case).
    V_r : jax.Array (nx, ny, nz)
        Combined real-space local potential (shared ref from setup).
    vnl_projectors : list[tuple[jax.Array, jax.Array]]
        One (Z, E) pair per (species, l-channel).
        Z : (natoms, R, nG) complex128  — KB projector, R = nbeta*(2l+1)
        E : (nspinor, nspinor, R, R)    — D matrix, sliced to nspinor
    nG : int
        Number of valid G-vectors at this k-point.
    """
    T_diag: jax.Array
    Gx: jax.Array
    Gy: jax.Array
    Gz: jax.Array
    g_mask: jax.Array | None
    V_r: jax.Array
    vnl_projectors: list[tuple[jax.Array, jax.Array]]
    nG: int


# ---------------------------------------------------------------------------
# Setup functions
# ---------------------------------------------------------------------------

def build_hamiltonian_setup(
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    pseudos: dict,
    *,
    include_hartree: bool = False,
    global_psi_G: jax.Array | None = None,
    truncation_2d: bool = True,
) -> HamiltonianSetup:
    """Precompute k-point-independent Hamiltonian data.

    Reuses ``build_local_ionic_potential_on_G_total`` for V_loc and
    ``build_vnl_plan`` for the VNL plan — no duplication.

    Parameters
    ----------
    wfn, sym, meta : standard LORRAX reader/symmetry/meta objects.
    pseudos : dict from ``load_pseudopotentials``.
    include_hartree : add valence Hartree V_H to V_r.
    global_psi_G : required if *include_hartree* is True (for rho_val).
    truncation_2d : 2-D slab Coulomb truncation on V_loc (default True).
    """
    with timing.section("hamiltonian_matvec.build_setup"):
        # -- atom-pseudo assignments (same helper as get_kin_ion) -----------
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

        # -- V_loc on WFN FFT grid -----------------------------------------
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

        # -- optional Hartree -----------------------------------------------
        if include_hartree:
            if global_psi_G is None:
                raise ValueError(
                    "global_psi_G required when include_hartree=True"
                )
            rho_val = compute_valence_density(global_psi_G, sym, wfn)
            V_H_r = compute_hartree_potential_real(
                rho_val,
                jnp.asarray(wfn.bdot, dtype=jnp.float64),
                bvec=jnp.asarray(wfn.bvec, dtype=jnp.float64),
                blat=float(wfn.blat),
                truncation_2d=False,
            )
            V_r = V_r + jnp.asarray(V_H_r, dtype=jnp.float64)

        # -- VNL plan -------------------------------------------------------
        bvec_np = np.asarray(wfn.bvec, dtype=float).T
        B = float(wfn.blat) * bvec_np.T
        q_max = 0.0
        for ik in range(sym.nk_tot):
            Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
            kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
            K_cart = (np.asarray(Gk_crys, dtype=float) + kvec[None, :]) @ B
            qk = np.sqrt(np.sum(K_cart ** 2, axis=1))
            if qk.size:
                q_max = max(q_max, float(np.max(qk)))

        vnl_plan = build_vnl_plan(
            pseudos, assignments, float(wfn.cell_volume), float(q_max)
        )

    return HamiltonianSetup(
        V_r=V_r,
        vnl_plan=vnl_plan,
        assignments=assignments,
        species_payload=species_payload,
        bdot=np.asarray(wfn.bdot, dtype=float),
        B=B,
        cell_volume=float(wfn.cell_volume),
        fft_grid=tuple(int(x) for x in meta.fft_grid),
    )


def build_kpoint_hamiltonian(
    k_idx: int,
    setup: HamiltonianSetup,
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    nspinor: int | None = None,
) -> KPointHamiltonian:
    """Build per-k data: kinetic diagonal and VNL projectors Z.

    The VNL projectors Z_a(G) are precomputed for each (species, l)
    channel so that ``apply_H_k`` needs only project/unproject — no
    spline evaluation or spherical-harmonic computation at apply time.

    Parameters
    ----------
    k_idx : k-point index in sym.unfolded_kpts.
    setup : from ``build_hamiltonian_setup``.
    wfn, sym, meta : standard LORRAX objects.
    nspinor : override spinor count (default: meta.nspinor).
    """
    if nspinor is None:
        nspinor = int(meta.nspinor)

    Gk_crys, _ = generate_gvectors_k(k_idx, sym, wfn, meta)
    Gk_np = np.asarray(Gk_crys, dtype=int)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)
    nG = Gk_np.shape[0]

    # FFT-box scatter/gather indices
    Gx = jnp.asarray(Gk_np[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_np[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_np[:, 2], dtype=jnp.int32)

    # Kinetic diagonal: T(G) = (k+G) . bdot . (k+G)  [Ry]
    G_float = np.asarray(Gk_np, dtype=float)
    K_crys = G_float + kvec[None, :]
    T_diag = jnp.asarray(
        np.einsum('gi,ij,gj->g', K_crys, setup.bdot, K_crys),
        dtype=jnp.float64,
    )

    # K vectors for VNL
    K_cart = K_crys @ setup.B
    K_norm = np.sqrt(np.sum(K_cart ** 2, axis=1))
    K_crys_j = jnp.asarray(K_crys, dtype=jnp.float64)

    # -- precompute VNL projectors Z for each (species, l) -----------------
    vnl_projectors: list[tuple[jax.Array, jax.Array]] = []
    Y_cache: dict[int, jax.Array] = {}

    for _key, sp in setup.vnl_plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=float)
        if tau.size == 0:
            continue
        if tau.ndim == 1:
            tau = tau.reshape(1, 3)
        natoms = tau.shape[0]
        pref = float(sp['prefactor'])      # 4 pi / sqrt(Omega)
        splines = sp['splines']

        # Atomic structure factors (k-dep, psi-indep)
        tau_j = jnp.asarray(tau, dtype=jnp.float64)
        # phase[atom, G] = exp(-2 pi i  K_crys . tau)
        phase = jnp.exp(
            -2j * jnp.pi * (K_crys_j @ tau_j.T)
        ).T  # (natoms, nG)

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

            # Radial form factors via pre-built splines
            F_bG = np.stack(
                [splines[(l, int(bid))](K_norm) for bid in beta_ids],
                axis=0,
            )  # (nbeta, nG)
            radial = jnp.asarray(
                pref * (1j) ** l * F_bG,
                dtype=jnp.complex128,
            )  # (nbeta, nG)

            # Real spherical harmonics Y'_lm(K) — cached per l
            if l not in Y_cache:
                Y_cache[l] = jnp.asarray(
                    qe_real_sph_harmonics(l, K_cart),
                    dtype=jnp.complex128,
                )  # (msize, nG)
            Y = Y_cache[l]

            # Z_atoms = phase * (radial x Y)  -> (natoms, R, nG)
            # where R = nbeta * msize
            Z_bmg = radial[:, None, :] * Y[None, :, :]     # (nbeta, ms, nG)
            Z_atoms = (
                phase[:, None, None, :] * Z_bmg[None, ...]  # (na, nb, ms, nG)
            )
            R = nbeta * msize
            Z_flat = Z_atoms.reshape(natoms, R, nG)

            # E block — slice to actual nspinor
            E_j = info.get('E_j')
            if E_j is None:
                E_j = jnp.asarray(E_np, dtype=jnp.complex128)
            E_j = E_j[:nspinor, :nspinor]  # (nspinor, nspinor, R, R)

            vnl_projectors.append((Z_flat, E_j))

    return KPointHamiltonian(
        T_diag=T_diag,
        Gx=Gx, Gy=Gy, Gz=Gz,
        g_mask=None,
        V_r=setup.V_r,
        vnl_projectors=vnl_projectors,
        nG=nG,
    )


# ---------------------------------------------------------------------------
# JIT kernels
# ---------------------------------------------------------------------------

@jax.jit
def _apply_kinetic(psi_G: jax.Array, T_diag: jax.Array) -> jax.Array:
    """T|psi> in sparse-G: multiply by |k+G|^2.

    (nvec, nspinor, nG) -> (nvec, nspinor, nG)
    """
    return T_diag[None, None, :] * psi_G


@jax.jit
def _apply_local_V(
    psi_box: jax.Array,
    V_r: jax.Array,
    Gx: jax.Array,
    Gy: jax.Array,
    Gz: jax.Array,
) -> jax.Array:
    """V_loc|psi> via real-space multiplication.

    The normalisation-free identity holds because the four volume factors
    in the full-matrix code (scale, deltaV, fft_norm, sqrt(1/Omega))
    cancel to unity when combined:

        sqrt(N/Omega) * (Omega/N) * sqrt(N) * sqrt(1/Omega) = 1

    So the matvec is just ortho-normalised FFT of V(r)*IFFT(psi):

        (V psi~)_G = FFT(V_r * IFFT(psi~, ortho), ortho)_G

    Parameters
    ----------
    psi_box : (nvec, nspinor, nx, ny, nz)
    V_r     : (nx, ny, nz)
    Gx,Gy,Gz : (nG,) int32

    Returns
    -------
    (nvec, nspinor, nG)  at valid G positions
    """
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    phi_r = psi_r * V_r
    phi_box = jnp.fft.fftn(phi_r, axes=(-3, -2, -1), norm='ortho')
    return phi_box[..., Gx, Gy, Gz]


@jax.jit
def _apply_vnl_channel(
    psi_G: jax.Array,
    Z: jax.Array,
    E: jax.Array,
) -> jax.Array:
    """One (species, l) Kleinman-Bylander contribution.

    V_NL|psi> = sum_atoms  Z_a  E  Z_a^dag  |psi>

    The three einsums correspond to:
      1. Project:   p[a,q,t,v] = sum_G conj(Z[a,q,G]) psi[v,t,G]
      2. Apply D:   d[a,r,s,v] = sum_{t,q} E[s,t,r,q] p[a,q,t,v]
      3. Unproject:  result[v,s,G] = sum_{a,r} Z[a,r,G] d[a,r,s,v]

    Parameters
    ----------
    psi_G : (nvec, nspinor, nG)
    Z     : (natoms, R, nG)         R = nbeta * (2l+1)
    E     : (nspinor, nspinor, R, R)

    Returns
    -------
    (nvec, nspinor, nG)
    """
    proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
    d = jnp.einsum('strq,aqtv->arsv', E, proj, optimize=True)
    return jnp.einsum('arG,arsv->vsG', Z, d, optimize=True)


# ---------------------------------------------------------------------------
# Top-level matvec
# ---------------------------------------------------------------------------

def apply_H_k(
    psi: jax.Array,
    kham: KPointHamiltonian,
) -> jax.Array:
    """Apply H = T + V_loc + V_NL to trial vectors at one k-point.

    Parameters
    ----------
    psi : (nvec, nspinor, nx, ny, nz)
        Trial vectors in FFT-box representation.  For multi-GPU, shard
        along axis 0 over a 1-D mesh.
    kham : KPointHamiltonian
        From ``build_kpoint_hamiltonian``.

    Returns
    -------
    H_psi : (nvec, nspinor, nx, ny, nz)
        Result in FFT-box form (nonzero only at valid G-vectors).
    """
    Gx, Gy, Gz = kham.Gx, kham.Gy, kham.Gz
    mask = kham.g_mask

    # -- extract sparse-G coefficients -------------------------------------
    psi_G = psi[..., Gx, Gy, Gz]                    # (nvec, nspinor, nG)
    if mask is not None:
        psi_G = psi_G * mask[None, None, :]

    # -- kinetic (diagonal in G) -------------------------------------------
    H_G = _apply_kinetic(psi_G, kham.T_diag)

    # -- local potential (FFT-based) ---------------------------------------
    V_G = _apply_local_V(psi, kham.V_r, Gx, Gy, Gz)
    if mask is not None:
        V_G = V_G * mask[None, None, :]
    H_G = H_G + V_G

    # -- nonlocal KB potential (per species/l channel) ---------------------
    for Z, E in kham.vnl_projectors:
        H_G = H_G + _apply_vnl_channel(psi_G, Z, E)

    # -- scatter back to FFT box -------------------------------------------
    H_psi = jnp.zeros_like(psi)
    return H_psi.at[..., Gx, Gy, Gz].set(H_G)


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def create_mesh(axis_name: str = 'batch') -> Mesh:
    """1-D device mesh over all available GPUs."""
    return Mesh(np.array(jax.devices()), (axis_name,))


def shard_trial_vectors(
    psi: jax.Array,
    mesh: Mesh,
    axis_name: str = 'batch',
) -> jax.Array:
    """Shard trial vectors along nvec (axis 0) across the mesh."""
    spec = NamedSharding(mesh, P(axis_name, None, None, None, None))
    return jax.lax.with_sharding_constraint(psi, spec)


# ---------------------------------------------------------------------------
# Validation against full-matrix code
# ---------------------------------------------------------------------------

def validate_against_full_matrix(
    k_idx: int,
    kham: KPointHamiltonian,
    setup: HamiltonianSetup,
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    nb: int,
    *,
    atol: float = 1e-8,
    verbose: bool = True,
) -> tuple[bool, float, float]:
    """Compare matvec against existing full-matrix code.

    Uses the DFT wavefunctions as trial vectors and checks that the
    projected Hamiltonian  H_mn = <psi_m|H|psi_n>  from the matvec
    matches the full H_mn matrix from compute_kinetic_k / compute_local_V_k
    / compute_V_NL_k_minimal.

    Returns (passed, max_abs_error, max_rel_error).
    """
    wfn_k = load_kpoint_fftbox(wfn, sym, meta, k_idx, nb)

    Gk_crys, kpoint = generate_gvectors_k(k_idx, sym, wfn, meta)
    bdot = np.asarray(wfn.bdot, dtype=float)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)

    # -- full matrices via existing code ------------------------------------
    T_mat = compute_kinetic_k(wfn_k, Gk_crys, kpoint, bdot)

    V_mat = compute_local_V_k(
        wfn_k, Gk_crys, setup.V_r, wfn.cell_volume
    )

    Gk_np = np.asarray(Gk_crys, dtype=float)
    K_crys = Gk_np + kvec[None, :]
    K_cart = K_crys @ setup.B
    VNL_mat = compute_V_NL_k_minimal(
        wfn_k, Gk_crys, K_crys, K_cart,
        setup.vnl_plan, float(wfn.cell_volume),
    )

    H_mat = jnp.asarray(T_mat + V_mat + VNL_mat)

    # -- matvec path --------------------------------------------------------
    H_psi = apply_H_k(wfn_k, kham)

    # -- project back: H_check[m,n] = <psi_m|H|psi_n> ----------------------
    Gx, Gy, Gz = kham.Gx, kham.Gy, kham.Gz
    psi_G = wfn_k[:, :, Gx, Gy, Gz]
    Hpsi_G = H_psi[:, :, Gx, Gy, Gz]
    H_check = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), Hpsi_G)

    # -- compare ------------------------------------------------------------
    err = float(jnp.max(jnp.abs(H_check - H_mat)))
    scale = jnp.maximum(jnp.abs(H_mat), 1e-30)
    rel = float(jnp.max(jnp.abs(H_check - H_mat) / scale))
    passed = err < atol

    if verbose:
        # Per-component breakdown
        T_psi = jnp.zeros_like(wfn_k)
        T_coeffs = _apply_kinetic(psi_G, kham.T_diag)
        T_psi = T_psi.at[..., Gx, Gy, Gz].set(T_coeffs)
        T_check = jnp.einsum(
            'msG,nsG->mn', jnp.conj(psi_G),
            T_psi[:, :, Gx, Gy, Gz],
        )

        V_coeffs = _apply_local_V(wfn_k, kham.V_r, Gx, Gy, Gz)
        V_check = jnp.einsum(
            'msG,nsG->mn', jnp.conj(psi_G), V_coeffs,
        )

        VNL_coeffs = jnp.zeros_like(psi_G)
        for Z, E in kham.vnl_projectors:
            VNL_coeffs = VNL_coeffs + _apply_vnl_channel(psi_G, Z, E)
        VNL_check = jnp.einsum(
            'msG,nsG->mn', jnp.conj(psi_G), VNL_coeffs,
        )

        print(f"    T    max|err| = {float(jnp.max(jnp.abs(T_check - T_mat))):.2e}")
        print(f"    Vloc max|err| = {float(jnp.max(jnp.abs(V_check - V_mat))):.2e}")
        print(f"    VNL  max|err| = {float(jnp.max(jnp.abs(VNL_check - VNL_mat))):.2e}")

    return passed, err, rel


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """Build Hamiltonian, validate matvec, report timings."""
    argp = argparse.ArgumentParser(
        description="PW DFT Hamiltonian matvec — validate and benchmark",
    )
    argp.add_argument("-i", "--input", required=True, help="cohsex.in path")
    argp.add_argument(
        "-k", "--kidx", type=int, default=0, help="k-point index to test"
    )
    argp.add_argument("-n", "--nb", type=int, default=None, help="number of bands")
    argp.add_argument(
        "--all-k", action="store_true", help="validate all k-points"
    )
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

    print("== Hamiltonian matvec: validate & benchmark ==")
    print(f"Loading WFN: {os.path.basename(wfn_path)}")
    with timing.section("load_wfn"):
        wfn = WFNReader(wfn_path)
        sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(wfn, sym, nval, ncond, nb, 0, bispinor)
    print(f"  k={sym.nk_tot}, bands={nb}, nspinor={meta.nspinor}, "
          f"grid={meta.fft_grid}")
    print(f"  Devices: {jax.device_count()}")

    pseudos = load_pseudopotentials(input_dir)

    # -- build setup --------------------------------------------------------
    print("\nBuilding Hamiltonian setup...")
    with timing.section("build_setup"):
        setup = build_hamiltonian_setup(wfn, sym, meta, pseudos)
    print(f"  V_loc grid: {tuple(int(x) for x in setup.V_r.shape)}, "
          f"VNL species: {len(setup.vnl_plan)}")

    # -- validate -----------------------------------------------------------
    k_indices = range(sym.nk_tot) if args.all_k else [args.kidx]
    all_pass = True

    for k_idx in k_indices:
        print(f"\nk-point {k_idx}:")
        with timing.section(f"build_k{k_idx}"):
            kham = build_kpoint_hamiltonian(k_idx, setup, wfn, sym, meta)
        print(f"  nG={kham.nG}, VNL channels={len(kham.vnl_projectors)}")

        with timing.section(f"validate_k{k_idx}"):
            ok, err, rel = validate_against_full_matrix(
                k_idx, kham, setup, wfn, sym, meta, nb
            )
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  max|err|={err:.2e}  max|rel|={rel:.2e}")
        all_pass = all_pass and ok

    # -- benchmark ----------------------------------------------------------
    k_idx = args.kidx
    kham = build_kpoint_hamiltonian(k_idx, setup, wfn, sym, meta)
    wfn_k = load_kpoint_fftbox(wfn, sym, meta, k_idx, nb)

    # warm-up JIT
    _ = apply_H_k(wfn_k, kham)
    jax.block_until_ready(_)

    import time
    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        H_psi = apply_H_k(wfn_k, kham)
        jax.block_until_ready(H_psi)
    dt = (time.perf_counter() - t0) / n_iter
    print(f"\nBenchmark (k={k_idx}, {nb} vectors): "
          f"{dt * 1e3:.2f} ms/apply  ({n_iter} iters)")

    timing.report(title="\n--- Timing (seconds) ---")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
