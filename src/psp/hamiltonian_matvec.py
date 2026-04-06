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

Multi-GPU (1-D mesh)
--------------------
Shard trial vectors along axis 0 (nvec) over a 1-D JAX device mesh.
Each H|psi> is independent per vector; no inter-device communication.

Usage (single k-point)::

    setup = build_hamiltonian_setup(wfn, sym, meta, pseudos)
    kham  = build_kpoint_hamiltonian(k_idx, setup, wfn, sym, meta)
    Hpsi  = apply_H_k(psi, kham)       # psi: (nvec, nspinor, nx, ny, nz)

Multi-GPU (2-D mesh, all k-points)
-----------------------------------
Use a 2-D device mesh ('k', 'g'):

  * **k-axis** — batches over k-points (independent, no communication).
  * **g-axis** — shards G-vectors within each k-point.

Parallelism by operator:

  * T:     diagonal in G — trivially local on each g-device.
  * V_loc: each g-device rematerialises the full FFT (allgather psi along
           g, do IFFT * V_r * FFT, keep local g-shard of the result).
  * V_NL:  projection Z^dag psi sums over G → GSPMD inserts allreduce.
           E (D-matrix) is replicated.  Unprojection Z d is local.

Usage (batched)::

    setup = build_hamiltonian_setup(wfn, sym, meta, pseudos)
    mesh  = create_mesh_2d()          # e.g. (4, 4) on 16 GPUs
    bham  = build_batched_hamiltonian(setup, wfn, sym, meta, mesh)
    Hpsi  = apply_H_batched(psi_G, bham, mesh)
    # psi_G : (nk, nvec, nspinor, nG_max)  sharded P('k', None, None, 'g')
"""
from __future__ import annotations

import os
import argparse
import functools
from dataclasses import dataclass, field

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

@jax.jit
def _apply_H_k_fused(psi_box, T_diag, V_r, Gx, Gy, Gz, vnl_ZE):
    """Fused H|psi>: FFT-box in, sparse-G out.  Single JIT dispatch."""
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


def apply_H_k(
    psi: jax.Array,
    kham: KPointHamiltonian,
    *,
    output: str = 'sparse',
) -> jax.Array:
    """Apply H = T + V_loc + V_NL to trial vectors at one k-point.

    Parameters
    ----------
    psi : (nvec, nspinor, nx, ny, nz)
        Trial vectors in FFT-box representation.
    kham : KPointHamiltonian
        From ``build_kpoint_hamiltonian``.
    output : 'sparse' (default) or 'box'
        Return format.  'sparse' returns (nvec, nspinor, nG) in a
        single fused JIT call (~2.5 ms).  'box' scatters back to
        the FFT box (~40 ms due to allocation + scatter).

    Returns
    -------
    If output='sparse': (nvec, nspinor, nG) sparse-G coefficients.
    If output='box':    (nvec, nspinor, nx, ny, nz) FFT-box.
    """
    if output == 'sparse':
        return _apply_H_k_fused(
            psi, kham.T_diag, kham.V_r,
            kham.Gx, kham.Gy, kham.Gz,
            tuple(kham.vnl_projectors),
        )

    # Legacy box output path
    Gx, Gy, Gz = kham.Gx, kham.Gy, kham.Gz
    mask = kham.g_mask
    psi_G = psi[..., Gx, Gy, Gz]
    if mask is not None:
        psi_G = psi_G * mask[None, None, :]
    H_G = _apply_kinetic(psi_G, kham.T_diag)
    V_G = _apply_local_V(psi, kham.V_r, Gx, Gy, Gz)
    if mask is not None:
        V_G = V_G * mask[None, None, :]
    H_G = H_G + V_G
    for Z, E in kham.vnl_projectors:
        H_G = H_G + _apply_vnl_channel(psi_G, Z, E)
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


# ===========================================================================
# 2-D mesh:  batched over k-points, G-vectors sharded
# ===========================================================================

@dataclass
class BatchedKHamiltonian:
    """All k-points padded to nG_max and stacked for 2-D mesh execution.

    Intended shardings on a Mesh(devices, ('k', 'g')):

    ========== ============================  ========================
    field      shape                         PartitionSpec
    ========== ============================  ========================
    T_diag     (nk, nG_max)                  P('k', 'g')
    Gx/Gy/Gz  (nk, nG_max)  int32           P('k', 'g')
    g_mask     (nk, nG_max)                  P('k', 'g')
    V_r        (nx, ny, nz)                  replicated
    vnl_Z      list of (nk, na, R, nG_max)   P('k', None, None, 'g')
    vnl_E      list of (ns, ns, R, R)        replicated
    ========== ============================  ========================
    """
    T_diag: jax.Array
    Gx: jax.Array
    Gy: jax.Array
    Gz: jax.Array
    g_mask: jax.Array
    V_r: jax.Array
    vnl_Z: list[jax.Array]
    vnl_E: list[jax.Array]
    fft_grid: tuple[int, int, int]
    nk: int
    nG_max: int


# -- builder ---------------------------------------------------------------

def build_batched_hamiltonian(
    setup: HamiltonianSetup,
    wfn: WFNReader,
    sym: symmetry_maps.SymMaps,
    meta: Meta,
    mesh: Mesh | None = None,
    nspinor: int | None = None,
) -> BatchedKHamiltonian:
    """Stack all k-point Hamiltonians into padded batched arrays.

    Optionally applies sharding constraints according to a 2-D *mesh*
    with axes ``('k', 'g')``.
    """
    if nspinor is None:
        nspinor = int(meta.nspinor)
    nk = sym.nk_tot

    # -- build per-k data and determine nG_max ------------------------------
    khams = []
    for ik in range(nk):
        khams.append(build_kpoint_hamiltonian(ik, setup, wfn, sym, meta,
                                              nspinor=nspinor))
    nG_raw = max(kh.nG for kh in khams)

    # -- round nk and nG_max up for divisibility by mesh axes ---------------
    if mesh is not None:
        nk_dev = mesh.shape['k']
        ng_dev = mesh.shape['g']
    else:
        nk_dev = ng_dev = 1
    nk_padded = int(np.ceil(nk / nk_dev)) * nk_dev
    nG_max = int(np.ceil(nG_raw / ng_dev)) * ng_dev

    # -- pad and stack scalars / G-indices ----------------------------------
    T_np = np.zeros((nk_padded, nG_max), dtype=np.float64)
    Gx_np = np.zeros((nk_padded, nG_max), dtype=np.int32)
    Gy_np = np.zeros((nk_padded, nG_max), dtype=np.int32)
    Gz_np = np.zeros((nk_padded, nG_max), dtype=np.int32)
    mask_np = np.zeros((nk_padded, nG_max), dtype=np.float64)

    for ik, kh in enumerate(khams):
        n = kh.nG
        T_np[ik, :n] = np.asarray(kh.T_diag)
        Gx_np[ik, :n] = np.asarray(kh.Gx)
        Gy_np[ik, :n] = np.asarray(kh.Gy)
        Gz_np[ik, :n] = np.asarray(kh.Gz)
        mask_np[ik, :n] = 1.0

    # -- stack VNL Z across k-points ----------------------------------------
    # Channel structure is identical for every k (same species/l set),
    # only the numerical values of Z differ because K = k + G changes.
    n_channels = len(khams[0].vnl_projectors)
    vnl_Z_list: list[jax.Array] = []
    vnl_E_list: list[jax.Array] = []

    for c in range(n_channels):
        _, E_c = khams[0].vnl_projectors[c]
        vnl_E_list.append(E_c)
        # Stack Z for this channel across k-points (+ k-padding)
        Z_ref, _ = khams[0].vnl_projectors[c]
        na, R, _ = Z_ref.shape
        Z_per_k = []
        for ik in range(nk_padded):
            if ik < nk:
                Z_k, _ = khams[ik].vnl_projectors[c]
                nG_k = Z_k.shape[-1]
                if nG_k < nG_max:
                    Z_k = jnp.pad(Z_k, ((0, 0), (0, 0), (0, nG_max - nG_k)))
            else:
                Z_k = jnp.zeros((na, R, nG_max), dtype=jnp.complex128)
            Z_per_k.append(Z_k)
        vnl_Z_list.append(jnp.stack(Z_per_k, axis=0))

    # -- convert to JAX arrays, optionally shard ----------------------------
    T = jnp.asarray(T_np)
    Gx = jnp.asarray(Gx_np)
    Gy = jnp.asarray(Gy_np)
    Gz = jnp.asarray(Gz_np)
    g_mask = jnp.asarray(mask_np)

    if mesh is not None:
        kg = NamedSharding(mesh, P('k', 'g'))
        rep = NamedSharding(mesh, P(None, None, None))
        T = jax.device_put(T, kg)
        Gx = jax.device_put(Gx, kg)
        Gy = jax.device_put(Gy, kg)
        Gz = jax.device_put(Gz, kg)
        g_mask = jax.device_put(g_mask, kg)
        setup_V_r = jax.device_put(setup.V_r, rep)
        z_spec = NamedSharding(mesh, P('k', None, None, 'g'))
        vnl_Z_list = [jax.device_put(z, z_spec) for z in vnl_Z_list]
    else:
        setup_V_r = setup.V_r

    return BatchedKHamiltonian(
        T_diag=T,
        Gx=Gx, Gy=Gy, Gz=Gz,
        g_mask=g_mask,
        V_r=setup_V_r,
        vnl_Z=vnl_Z_list,
        vnl_E=vnl_E_list,
        fft_grid=setup.fft_grid,
        nk=nk,       # real k-count (before padding)
        nG_max=nG_max,
    )


# -- 2-D mesh helpers ------------------------------------------------------

def create_mesh_2d(
    nk_devices: int | None = None,
    ng_devices: int | None = None,
) -> Mesh:
    """Create a 2-D device mesh with axes ``('k', 'g')``.

    If neither dimension is specified, defaults to (n_devices, 1) — pure
    k-batching with no G-sharding — which is a safe fallback.
    """
    devices = np.array(jax.devices())
    n = len(devices)
    if nk_devices is not None and ng_devices is not None:
        if nk_devices * ng_devices != n:
            raise ValueError(
                f"nk*ng ({nk_devices}*{ng_devices}) != {n} devices"
            )
    elif nk_devices is not None:
        ng_devices = n // nk_devices
    elif ng_devices is not None:
        nk_devices = n // ng_devices
    else:
        nk_devices, ng_devices = n, 1
    return Mesh(devices.reshape(nk_devices, ng_devices), ('k', 'g'))


# -- per-k V_loc kernel (vmapped over k) -----------------------------------

def _vloc_single_k(
    psi_g: jax.Array,
    gx: jax.Array,
    gy: jax.Array,
    gz: jax.Array,
    mask: jax.Array,
    V_r: jax.Array,
) -> jax.Array:
    """V_loc|psi> for one k-point in sparse-G.

    psi_g : (nvec, nspinor, nG_max)
    gx,gy,gz : (nG_max,) int32
    mask : (nG_max,)
    V_r : (nx, ny, nz)

    Returns (nvec, nspinor, nG_max)
    """
    nvec, nspinor, _ = psi_g.shape
    nx, ny, nz = V_r.shape
    psi_masked = psi_g * mask[None, None, :]
    # Scatter into FFT box — use .add() not .set() because padded
    # G-indices (all zeros) would alias the real G=0 entry with .set().
    # Since padding values are zero (via mask), .add() is correct.
    psi_box = jnp.zeros((nvec, nspinor, nx, ny, nz), dtype=psi_g.dtype)
    psi_box = psi_box.at[:, :, gx, gy, gz].add(psi_masked)
    # IFFT -> V*psi -> FFT
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    phi_box = jnp.fft.fftn(psi_r * V_r, axes=(-3, -2, -1), norm='ortho')
    return phi_box[:, :, gx, gy, gz] * mask[None, None, :]


# -- batched apply ----------------------------------------------------------

def apply_H_batched(
    psi_G: jax.Array,
    bham: BatchedKHamiltonian,
    mesh: Mesh,
) -> jax.Array:
    """Apply H for all k-points on a 2-D (k, g) device mesh.

    Parameters
    ----------
    psi_G : (nk, nvec, nspinor, nG_max)
        Trial vectors in sparse-G, sharded as P('k', None, None, 'g').
    bham : BatchedKHamiltonian
    mesh : 2-D Mesh with axes ('k', 'g').

    Returns
    -------
    H_psi_G : (nk, nvec, nspinor, nG_max)
        Same sharding as input.
    """
    # Sharding specs — captured as constants inside the JIT closure.
    kg = NamedSharding(mesh, P('k', None, None, 'g'))
    k_only = NamedSharding(mesh, P('k', None, None, None))
    kg_2d = NamedSharding(mesh, P('k', 'g'))
    k_2d = NamedSharding(mesh, P('k', None))

    # Freeze the VNL channel list into a tuple so JIT treats each
    # element as a separate traced leaf (the loop is unrolled at trace).
    vnl_Z = tuple(bham.vnl_Z)
    vnl_E = tuple(bham.vnl_E)

    @functools.partial(jax.jit, static_argnames=('fft_grid',))
    def _core(psi_G, T_diag, Gx, Gy, Gz, g_mask, V_r, vnl_Z, vnl_E,
              fft_grid):
        mask = g_mask                                    # (nk, nG_max)
        psi = psi_G * mask[:, None, None, :]             # zero padding

        # ── kinetic: diagonal, G-local ────────────────────────────────
        H_G = T_diag[:, None, None, :] * psi

        # ── V_loc: rematerialise FFT on every g-device ────────────────
        #
        # Force psi and G-indices to be replicated along the g-axis so
        # that each g-device has the complete data needed for the FFT.
        # After the FFT the result is re-sharded along g.
        psi_rep = jax.lax.with_sharding_constraint(psi, k_only)
        Gx_rep = jax.lax.with_sharding_constraint(Gx, k_2d)
        Gy_rep = jax.lax.with_sharding_constraint(Gy, k_2d)
        Gz_rep = jax.lax.with_sharding_constraint(Gz, k_2d)
        mask_rep = jax.lax.with_sharding_constraint(mask, k_2d)

        V_G = jax.vmap(
            _vloc_single_k, in_axes=(0, 0, 0, 0, 0, None)
        )(psi_rep, Gx_rep, Gy_rep, Gz_rep, mask_rep, V_r)

        V_G = jax.lax.with_sharding_constraint(V_G, kg)
        H_G = H_G + V_G

        # ── V_NL: project (allreduce over g), unproject (g-local) ─────
        #
        # The einsum Z^dag psi sums over G.  With G sharded, GSPMD
        # automatically inserts an allreduce.  The E matrix (D block) is
        # replicated.  The unproject Z d is local to each g-shard.
        for Z_k, E_c in zip(vnl_Z, vnl_E):
            # Z_k: (nk, natoms, R, nG_max) — P('k', None, None, 'g')
            # psi:  (nk, nvec, nspinor, nG_max) — P('k', None, None, 'g')
            # Project
            proj = jnp.einsum(
                'kaqG,kvtG->kaqtv', jnp.conj(Z_k), psi, optimize=True,
            )  # (nk, na, R, ns, nv) — G contracted → replicated along g
            # Apply D
            d = jnp.einsum(
                'strq,kaqtv->karsv', E_c, proj, optimize=True,
            )  # replicated along g
            # Unproject
            vnl_G = jnp.einsum(
                'karG,karsv->kvsG', Z_k, d, optimize=True,
            )  # (nk, nv, ns, nG_max) — G-sharded again
            H_G = H_G + vnl_G

        return H_G * mask[:, None, None, :]

    return _core(
        psi_G, bham.T_diag, bham.Gx, bham.Gy, bham.Gz, bham.g_mask,
        bham.V_r, vnl_Z, vnl_E, bham.fft_grid,
    )


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

    # -- matvec path (sparse-G output) -------------------------------------
    Gx, Gy, Gz = kham.Gx, kham.Gy, kham.Gz
    Hpsi_G = apply_H_k(wfn_k, kham)  # (nb, nspinor, nG)

    # -- project back: H_check[m,n] = <psi_m|H|psi_n> ----------------------
    psi_G = wfn_k[:, :, Gx, Gy, Gz]
    H_check = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), Hpsi_G)

    # -- compare ------------------------------------------------------------
    err = float(jnp.max(jnp.abs(H_check - H_mat)))
    scale = jnp.maximum(jnp.abs(H_mat), 1e-30)
    rel = float(jnp.max(jnp.abs(H_check - H_mat) / scale))
    passed = err < atol

    if verbose:
        # Per-component breakdown
        T_coeffs = _apply_kinetic(psi_G, kham.T_diag)
        T_check = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), T_coeffs)

        V_coeffs = _apply_local_V(wfn_k, kham.V_r, Gx, Gy, Gz)
        V_check = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), V_coeffs)

        VNL_coeffs = jnp.zeros_like(psi_G)
        for Z, E in kham.vnl_projectors:
            VNL_coeffs = VNL_coeffs + _apply_vnl_channel(psi_G, Z, E)
        VNL_check = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), VNL_coeffs)

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
    argp.add_argument(
        "--batched", action="store_true",
        help="validate batched 2-D mesh path (all k at once)",
    )
    argp.add_argument(
        "--nk-devices", type=int, default=None,
        help="k-axis size for 2-D mesh (default: all devices on k)",
    )
    argp.add_argument(
        "--ng-devices", type=int, default=None,
        help="g-axis size for 2-D mesh (default: 1)",
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

    # -- batched 2-D mesh validation ------------------------------------------
    if args.batched:
        print("\n" + "=" * 60)
        print("Batched 2-D mesh validation")
        print("=" * 60)
        mesh2d = create_mesh_2d(args.nk_devices, args.ng_devices)
        print(f"  Mesh: {mesh2d.shape}  axes={mesh2d.axis_names}")

        with timing.section("build_batched"):
            bham = build_batched_hamiltonian(setup, wfn, sym, meta, mesh2d)
        print(f"  nk={bham.nk}, nG_max={bham.nG_max}, "
              f"VNL channels={len(bham.vnl_Z)}")

        # Build batched psi_G from individual k-point wavefunctions
        nk_padded = bham.T_diag.shape[0]  # includes k-padding
        psi_all = np.zeros(
            (nk_padded, nb, meta.nspinor, bham.nG_max),
            dtype=np.complex128,
        )
        for ik in range(sym.nk_tot):
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
            Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
            Gk_np = np.asarray(Gk_crys, dtype=int)
            gx = Gk_np[:, 0]; gy = Gk_np[:, 1]; gz = Gk_np[:, 2]
            psi_all[ik, :, :, :len(gx)] = np.asarray(
                wfn_k[:, :, gx, gy, gz]
            )
        psi_j = jax.device_put(
            jnp.asarray(psi_all),
            NamedSharding(mesh2d, P('k', None, None, 'g')),
        )

        # Apply batched
        with timing.section("apply_batched"):
            H_psi_batched = apply_H_batched(psi_j, bham, mesh2d)
            jax.block_until_ready(H_psi_batched)

        # Compare against per-k results
        batched_pass = True
        for ik in range(sym.nk_tot):
            kham = build_kpoint_hamiltonian(ik, setup, wfn, sym, meta)
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
            H_psi_ref = apply_H_k(wfn_k, kham)

            # Extract reference in sparse-G
            Gx_k, Gy_k, Gz_k = kham.Gx, kham.Gy, kham.Gz
            ref_G = H_psi_ref[:, :, Gx_k, Gy_k, Gz_k]

            # Extract batched result (first nG entries, rest is padding)
            nG = kham.nG
            bat_G = np.asarray(H_psi_batched[ik, :, :, :nG])
            ref_np = np.asarray(ref_G)

            err = float(np.max(np.abs(bat_G - ref_np)))
            ok = err < 1e-8
            batched_pass = batched_pass and ok
            status = "PASS" if ok else "FAIL"
            print(f"  k={ik}: {status}  max|err|={err:.2e}")

        all_pass = all_pass and batched_pass

        # Benchmark batched
        _ = apply_H_batched(psi_j, bham, mesh2d)
        jax.block_until_ready(_)
        import time as _time
        n_iter_b = 20
        t0 = _time.perf_counter()
        for _ in range(n_iter_b):
            H_b = apply_H_batched(psi_j, bham, mesh2d)
            jax.block_until_ready(H_b)
        dt_b = (_time.perf_counter() - t0) / n_iter_b
        print(f"\n  Batched apply (all {bham.nk} k, {nb} vecs): "
              f"{dt_b * 1e3:.2f} ms/call  ({n_iter_b} iters)")

    timing.report(title="\n--- Timing (seconds) ---")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
