"""
psp/vnl_ops.py — Fast VNL operator: build once, apply many times.

All channels × atoms × betas are concatenated into a single dense
projector matrix Z of shape (total_R, nG).  E is a block-diagonal
(nspinor, nspinor, total_R, total_R) matrix.  The VNL operator is
then a single set of einsums with no Python loops.

Radial form factors use table lookup on a uniform q-grid (linear
interpolation) instead of spline evaluation. This keeps the production
path in a single JAX/table architecture and is still accurate with a
50k-point table (spacing ~1e-4).

Solid harmonics (Cartesian polynomials) replace angular Y_lm for
autodiff-safe behaviour at K=0.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from psp.build_projectors_qe import (
    build_E_blocks_full,
)
from psp.radial_jax import (
    differentiate_uniform_table,
    make_projector_table,
    make_uniform_q_grid,
    radial_weights,
)
# Import solid harmonics without circular dependency
# (dft_operators imports vnl_ops, so we can't import from dft_operators here)
from psp.solid_harmonics import solid_harmonics_jax as _solid_harmonics_jax


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChannelMeta:
    """Metadata for one (species, l) VNL channel."""
    l: int
    nbeta: int
    msize: int                      # 2l+1
    R: int                          # nbeta * msize
    tau: np.ndarray                 # (natoms, 3) crystal positions
    E: np.ndarray                   # (2, 2, R, R) D-matrix (full spin)
    beta_table_start: int           # index into flattened radial table
    natoms: int


@dataclass
class VNLSetup:
    """K-independent VNL data.  Built once from pseudopotentials.

    Radial form factors are stored as tables on a uniform q-grid,
    enabling O(1) batched lookup instead of per-scalar B-spline
    evaluation.
    """
    channels: list[ChannelMeta]
    # Radial tables on uniform q-grid
    dq: float                       # grid spacing
    n_q: int                        # number of grid points
    q_max: float
    G_table: jax.Array              # (total_nbeta, n_q)  G_l(q) = F_l(q)/q^l
    Gp_table: jax.Array             # (total_nbeta, n_q)  G'_l(q)
    prefactor: float                # 4π/√Ω
    # Structure
    B: np.ndarray                   # (3,3) crystal→Cartesian
    cell_volume: float
    # Block sizes for E assembly
    total_R: int
    nspinor: int


@dataclass
class VNLKData:
    """Per-k-point VNL projector data.  Precomputed for fast apply."""
    Z: jax.Array                    # (total_R, nG) complex128
    E_super: jax.Array              # (nspinor, nspinor, total_R, total_R)
    nG: int
    total_R: int
    # Optional: Cartesian k-derivatives for velocity
    dZ: jax.Array | None            # (3, total_R, nG) or None


# ---------------------------------------------------------------------------
# Setup builder
# ---------------------------------------------------------------------------

def build_vnl_setup(
    wfn, sym=None, meta=None, pseudos=None,
    n_q: int = 4000,
    nspinor: int | None = None,
    q_max: float | None = None,
) -> VNLSetup:
    """Build k-independent VNL data: radial tables, channel metadata.

    Parameters
    ----------
    wfn : WFNReader or CrystalData (needs atom_crys, bvec, blat, cell_volume)
    sym : SymMaps, optional — used to determine q_max if not provided.
    meta : Meta, optional — used with sym for q_max scan.
    pseudos : dict — element → UPF
    q_max : float, optional — if provided, skip the k-point scan for q_max.
        Use this when SymMaps is not available (standalone Davidson path).
    """
    if nspinor is None:
        nspinor = int(meta.nspinor) if meta is not None else int(wfn.nspinor)

    from psp.get_DFT_mtxels import build_atom_pp_assignments

    atom_pos = jnp.asarray(wfn.atom_crys, dtype=jnp.float64)
    atom_types = jnp.asarray(wfn.atom_types, dtype=jnp.int32)
    assignments = build_atom_pp_assignments(atom_pos, atom_types, pseudos)

    bvec_np = np.asarray(wfn.bvec, dtype=float).T
    B = float(wfn.blat) * bvec_np.T
    cell_volume = float(wfn.cell_volume)
    prefactor = (4.0 * np.pi) / np.sqrt(cell_volume)

    # Determine q_max. For PW spheres this is already bounded by ecutwfc.
    if q_max is None:
        if hasattr(wfn, "ecutwfc"):
            q_max = float(np.sqrt(float(wfn.ecutwfc)))
        else:
            if sym is None:
                raise ValueError("Either q_max or (sym, meta) required for build_vnl_setup")
            q_max = 0.0
            for ik in range(sym.nk_tot):
                Gk, _ = generate_gvectors_k(ik, sym, wfn, meta)
                kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
                K_cart = (np.asarray(Gk, dtype=float) + kvec[None, :]) @ B
                qk = np.sqrt(np.sum(K_cart ** 2, axis=1))
                if qk.size:
                    q_max = max(q_max, float(np.max(qk)))
    q_max *= 1.01  # small margin

    dq = q_max / (n_q - 1)
    q_grid = make_uniform_q_grid(q_max, n_q)

    # Group atoms by pseudopotential
    species_map: dict[int, dict] = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        key = id(ap.pseudo)
        ent = species_map.setdefault(key, {
            'pseudo': ap.pseudo, 'positions': [],
        })
        ent['positions'].append(np.asarray(ap.position, dtype=float))

    # Build channels and radial tables
    channels: list[ChannelMeta] = []
    G_rows: list[np.ndarray] = []
    Gp_rows: list[np.ndarray] = []
    beta_idx = 0

    for key, ent in species_map.items():
        pseudo = ent['pseudo']
        tau = np.asarray(ent['positions'], dtype=float)
        if tau.ndim == 1:
            tau = tau.reshape(1, 3)
        natoms = tau.shape[0]

        E_blocks = build_E_blocks_full(pseudo)
        ppmesh = getattr(pseudo, 'pp_mesh', None)
        ppnl = getattr(pseudo, 'pp_nonlocal', None)
        if ppmesh is None or ppnl is None:
            continue
        r = np.asarray(ppmesh.pp_r.value, dtype=float)
        try:
            rab = np.asarray(ppmesh.pp_rab.value, dtype=float)
        except Exception:
            rab = None
        weights = radial_weights(r, rab, scheme="rab")

        # Enumerate l-channels
        betas = list(getattr(getattr(pseudo, 'pp_nonlocal', None), 'pp_beta', []))
        per_l: dict[int, list[int]] = {}
        for idx, beta in enumerate(betas, start=1):
            l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
            per_l.setdefault(l, []).append(idx)

        for l, beta_ids in per_l.items():
            E_np = E_blocks.get(l)
            if E_np is None:
                continue
            nbeta = len(beta_ids)
            msize = 2 * l + 1
            R = nbeta * msize

            # Evaluate F_l(q) splines on uniform grid → build G_l = F_l/q^l
            for bid in beta_ids:
                beta = betas[int(bid) - 1]
                raw_beta = np.asarray(beta.value, dtype=float)
                if r[0] == 0.0 and len(r) > 1:
                    beta_r = np.empty_like(r)
                    beta_r[1:] = raw_beta[1:] / r[1:]
                    beta_r[0] = beta_r[1]
                else:
                    beta_r = raw_beta / r
                F_vals = make_projector_table(l, r, beta_r, q_grid, weights).values
                if l == 0:
                    G_vals = F_vals.copy()
                else:
                    G_vals = np.empty(n_q, dtype=np.float64)
                    G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
                    G_vals[0] = F_vals[1] / q_grid[1] ** l  # limit

                Gp_vals = differentiate_uniform_table(G_vals, dq)

                G_rows.append(G_vals)
                Gp_rows.append(Gp_vals)

            channels.append(ChannelMeta(
                l=l, nbeta=nbeta, msize=msize, R=R,
                tau=tau, E=E_np, beta_table_start=beta_idx,
                natoms=natoms,
            ))
            beta_idx += nbeta

    if G_rows:
        G_table = jnp.asarray(np.stack(G_rows, axis=0), dtype=jnp.float64)
        Gp_table = jnp.asarray(np.stack(Gp_rows, axis=0), dtype=jnp.float64)
    else:
        G_table = jnp.zeros((0, n_q), dtype=jnp.float64)
        Gp_table = jnp.zeros((0, n_q), dtype=jnp.float64)

    total_R = sum(ch.R * ch.natoms for ch in channels)

    return VNLSetup(
        channels=channels,
        dq=dq, n_q=n_q, q_max=q_max,
        G_table=G_table, Gp_table=Gp_table,
        prefactor=prefactor,
        B=B, cell_volume=cell_volume,
        total_R=total_R, nspinor=nspinor,
    )


# ---------------------------------------------------------------------------
# Table-lookup radial evaluation
# ---------------------------------------------------------------------------

@jax.jit
def _table_interp(q, dq, table):
    """Linear interpolation on uniform grid.

    q     : (nG,) query points
    dq    : scalar grid spacing
    table : (nbeta, n_q) values

    Returns (nbeta, nG).
    """
    n_q = table.shape[1]
    idx = jnp.floor(q / dq).astype(jnp.int32)
    idx = jnp.clip(idx, 0, n_q - 2)
    t = (q - idx * dq) / dq
    t = jnp.clip(t, 0.0, 1.0)
    return (1.0 - t)[None, :] * table[:, idx] + t[None, :] * table[:, idx + 1]


# ---------------------------------------------------------------------------
# Per-k projector construction
# ---------------------------------------------------------------------------

def build_vnl_kdata(
    k_idx: int,
    setup: VNLSetup,
    wfn, sym, meta,
    *,
    compute_dZ: bool = False,
) -> VNLKData:
    """Build dense Z [and dZ] for one k-point (SymMaps path)."""
    from psp.get_DFT_mtxels import generate_gvectors_k

    Gk_crys, _ = generate_gvectors_k(k_idx, sym, wfn, meta)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)
    return _build_vnl_kdata_core(kvec, np.asarray(Gk_crys, dtype=int),
                                  setup, compute_dZ=compute_dZ)


def build_vnl_kdata_from_kvec(
    kvec: np.ndarray,
    Gk_int: np.ndarray,
    setup: VNLSetup,
    crystal=None,
    meta=None,
    *,
    compute_dZ: bool = False,
) -> VNLKData:
    """Build dense Z [and dZ] from explicit k-vector + G-list (no SymMaps)."""
    return _build_vnl_kdata_core(np.asarray(kvec, dtype=float),
                                  np.asarray(Gk_int, dtype=int),
                                  setup, compute_dZ=compute_dZ)


def _build_vnl_kdata_core(
    kvec: np.ndarray,
    Gk_np: np.ndarray,
    setup: VNLSetup,
    *,
    compute_dZ: bool = False,
) -> VNLKData:
    """Core: build dense VNL projectors from k-vector + G-vectors."""
    nspinor = setup.nspinor
    nG = Gk_np.shape[0]

    K_crys = jnp.asarray(Gk_np, dtype=jnp.float64) + jnp.asarray(kvec)[None, :]
    B_j = jnp.asarray(setup.B, dtype=jnp.float64)
    K_cart = K_crys @ B_j

    _EPS2 = 1e-60
    K_sq = jnp.sum(K_cart ** 2, axis=1)
    q = jnp.sqrt(K_sq + _EPS2)     # (nG,) — safe at K=0

    # Evaluate ALL radial form factors at once via table lookup
    G_all = _table_interp(q, setup.dq, setup.G_table)    # (total_nbeta, nG)
    Gp_all = _table_interp(q, setup.dq, setup.Gp_table)  # (total_nbeta, nG)

    # K/q for radial derivative direction (stable at K=0)
    K_over_q = K_cart / q[:, None]   # (nG, 3)

    # Build dense Z, E_super, and optionally dZ
    Z_blocks = []
    dZ_blocks = []  # list of (3, R_block, nG)
    E_blocks_diag = []

    for ch in setup.channels:
        l = ch.l
        msize = ch.msize
        nbeta = ch.nbeta
        R = ch.R
        tau_j = jnp.asarray(ch.tau, dtype=jnp.float64)
        natoms = ch.natoms
        c_il = setup.prefactor * (1j) ** l

        # Radial: (nbeta, nG)
        b0 = ch.beta_table_start
        G_bG = G_all[b0:b0 + nbeta]
        Gp_bG = Gp_all[b0:b0 + nbeta]

        # Solid harmonics: (msize, nG), polynomial
        S = _solid_harmonics_jax(l, K_cart)

        # Phases per atom: (natoms, nG)
        phase = jnp.exp(-2j * jnp.pi * (K_crys @ tau_j.T)).T

        # Z per atom: c_il * G_bG * S * phase → (natoms, nbeta, msize, nG)
        radS = G_bG[:, None, :] * S[None, :, :]          # (nbeta, msize, nG)
        Z_per_atom = c_il * phase[:, None, None, :] * radS[None, :]
        Z_flat = Z_per_atom.reshape(natoms * R, nG)       # (natoms*R, nG)
        Z_blocks.append(Z_flat)

        # E: replicate per atom → block-diagonal
        E_np = ch.E[:nspinor, :nspinor]                   # (ns, ns, R, R)
        # Build block-diag: (ns, ns, natoms*R, natoms*R)
        E_big = np.zeros((nspinor, nspinor, natoms * R, natoms * R),
                         dtype=np.complex128)
        for a in range(natoms):
            s = a * R
            E_big[:, :, s:s+R, s:s+R] = E_np
        E_blocks_diag.append(E_big)

        if compute_dZ:
            # dS/dK_cart: (3, msize, nG)
            dS_list = []
            for j in range(3):
                ej = jnp.zeros((nG, 3)).at[:, j].set(1.0)
                _, dS_j = jax.jvp(
                    lambda K: _solid_harmonics_jax(l, K), (K_cart,), (ej,)
                )
                dS_list.append(dS_j)
            dS = jnp.stack(dS_list, axis=0)               # (3, msize, nG)

            # dphase/dK_cart_j: -2πi (B^{-1})^T tau . e_j * phase
            Binv = jnp.linalg.inv(B_j)
            tau_cart_eff = tau_j @ Binv.T                  # (natoms, 3)
            dphase = -2j * jnp.pi * tau_cart_eff[:, :, None] * phase[:, None, :]
            # dphase: (natoms, 3, nG)

            # dZ/dK_j = c_il * [G'(q) K_j/q S + G dS_j] * phase + c_il * G S * dphase_j
            # Radial derivative: Gp_bG (nbeta, nG), K_over_q.T (3, nG)
            # drad: (3, nbeta, nG) = G'(q) * K_j/q
            drad = Gp_bG[None, :, :] * K_over_q.T[:, None, :]
            # Core: (3, nbeta, msize, nG) = drad*S + G*dS
            core = drad[:, :, None, :] * S[None, None, :, :]
            core = core + G_bG[None, :, None, :] * dS[:, None, :, :]
            # Apply c_il and phase: (natoms, 3, nbeta, msize, nG)
            dZ_core = c_il * phase[:, None, None, None, :] * core[None, :, :, :, :]
            # Phase derivative term: (natoms, 3, nbeta, msize, nG)
            dZ_phase = c_il * radS[None, None, :, :, :] * dphase[:, :, None, None, :]
            dZ_total = dZ_core + dZ_phase
            # Reshape: (3, natoms*R, nG)
            dZ_flat = dZ_total.transpose(1, 0, 2, 3, 4).reshape(3, natoms * R, nG)
            dZ_blocks.append(dZ_flat)

    if Z_blocks:
        Z = jnp.concatenate(Z_blocks, axis=0)                  # (total_R, nG)

        # Assemble block-diagonal E_super
        E_super = np.zeros((nspinor, nspinor, setup.total_R, setup.total_R),
                           dtype=np.complex128)
        offset = 0
        for E_big in E_blocks_diag:
            r = E_big.shape[2]
            E_super[:, :, offset:offset+r, offset:offset+r] = E_big
            offset += r
        E_super_j = jnp.asarray(E_super, dtype=jnp.complex128)

        dZ_j = None
        if compute_dZ and dZ_blocks:
            dZ_j = jnp.concatenate(dZ_blocks, axis=1)         # (3, total_R, nG)
    else:
        Z = jnp.zeros((0, nG), dtype=jnp.complex128)
        E_super_j = jnp.zeros((nspinor, nspinor, 0, 0), dtype=jnp.complex128)
        dZ_j = (jnp.zeros((3, 0, nG), dtype=jnp.complex128) if compute_dZ else None)

    return VNLKData(
        Z=Z, E_super=E_super_j, nG=nG,
        total_R=setup.total_R, dZ=dZ_j,
    )


# ---------------------------------------------------------------------------
# Core operators — single JIT, no loops
# ---------------------------------------------------------------------------

@jax.jit
def apply_vnl(psi_G, Z, E_super):
    """V_NL|psi> = Z E Z† |psi>.   (nvec, nspinor, nG) → same.

    Z       : (total_R, nG)
    E_super : (nspinor, nspinor, total_R, total_R)
    """
    P = jnp.einsum('RG,vsG->Rsv', jnp.conj(Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtv->Rsv', E_super, P, optimize=True)
    return jnp.einsum('RG,Rsv->vsG', Z, D, optimize=True)


@jax.jit
def vnl_matrix(psi_G, Z, E_super):
    """V_NL matrix elements <m|V_NL|n>.   Returns (nb, nb)."""
    P = jnp.einsum('RG,nsG->Rsn', jnp.conj(Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtn->Rsn', E_super, P, optimize=True)
    return jnp.einsum('Rsm,Rsn->mn', jnp.conj(P), D, optimize=True)


@jax.jit
def vnl_velocity_matrix(psi_G, Z, dZ, E_super):
    """dV_NL/dK_cart matrix elements.  Returns (3, nb, nb).

    dZ : (3, total_R, nG)
    """
    P = jnp.einsum('RG,nsG->Rsn', jnp.conj(Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtn->Rsn', E_super, P, optimize=True)
    # dP[j] = dZ[j]† ψ
    dP = jnp.einsum('jRG,nsG->jRsn', jnp.conj(dZ), psi_G, optimize=True)
    dD = jnp.einsum('stRQ,jQtn->jRsn', E_super, dP, optimize=True)
    # v[j] = conj(dP[j]) D + conj(P) dD[j]
    t1 = jnp.einsum('jRsm,Rsn->jmn', jnp.conj(dP), D, optimize=True)
    t2 = jnp.einsum('Rsm,jRsn->jmn', jnp.conj(P), dD, optimize=True)
    return t1 + t2
