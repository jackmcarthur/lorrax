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

from psp.radial.build_projectors_qe import build_E_blocks_full
from psp.radial.radial_jax import differentiate_uniform_table
from psp.radial.solid_harmonics import solid_harmonics_jax as _solid_harmonics_jax


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

    Radial form factors are stored as tables on a uniform q-grid.
    Per-row metadata (beta_idx, l, m, tau) is pre-flattened so the
    per-k Z assembly is a single vectorized operation — no loops.
    """
    channels: list[ChannelMeta]
    dq: float
    n_q: int
    q_max: float
    G_table: jax.Array              # (total_nbeta, n_q)
    Gp_table: jax.Array             # (total_nbeta, n_q)
    prefactor: float                # 4π/√Ω
    B: np.ndarray                   # (3,3) crystal→Cartesian
    cell_volume: float
    total_R: int
    nspinor: int
    E_super: jax.Array | None = None  # (nspinor, nspinor, total_R, total_R)
    l_max: int = 0
    # ── Pre-flattened per-row metadata for vectorized Z assembly ──
    # Each row r of Z(total_R, nG) is: c_il * G[beta_idx[r], q] * S[l[r], m[r], G] * phase[tau[r], G]
    row_beta_idx: jax.Array | None = None  # (total_R,) int — which G_table row
    row_l: jax.Array | None = None         # (total_R,) int — angular momentum
    row_m: jax.Array | None = None         # (total_R,) int — m index into S_all[l]
    row_tau: jax.Array | None = None       # (total_R, 3) float — atom position (crystal)


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
    """
    from psp.species import extract_species, build_atom_species_map
    from psp.radial_tables import build_all_tables

    if nspinor is None:
        nspinor = int(meta.nspinor) if meta is not None else int(wfn.nspinor)

    B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=float)
    cell_volume = float(wfn.cell_volume)
    prefactor = (4.0 * np.pi) / np.sqrt(cell_volume)

    # Determine q_max
    if q_max is None:
        if hasattr(wfn, "ecutwfc"):
            q_max = float(np.sqrt(float(wfn.ecutwfc)))
        else:
            from psp.dft_operators import generate_gvectors_k
            q_max = 0.0
            for ik in range(sym.nk_tot):
                Gk, _ = generate_gvectors_k(ik, sym, wfn, meta)
                kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
                K_cart = (np.asarray(Gk, dtype=float) + kvec[None, :]) @ B
                qk = np.sqrt(np.sum(K_cart ** 2, axis=1))
                if qk.size:
                    q_max = max(q_max, float(np.max(qk)))
    q_max *= 1.01

    # Extract species data and projector tables
    species_list = extract_species(pseudos, nspinor=nspinor)
    tables = build_all_tables(species_list, q_max, n_q)
    species_natoms, species_tau, _ = build_atom_species_map(wfn, species_list)
    q_grid = tables["q"]
    dq = tables["dq"]

    # Build channels and G_l/G'_l tables from species projectors
    channels: list[ChannelMeta] = []
    G_rows: list[np.ndarray] = []
    Gp_rows: list[np.ndarray] = []
    beta_idx = 0

    for isp, sp in enumerate(species_list):
        natoms = int(species_natoms[isp])
        if natoms == 0:
            continue
        tau = species_tau[isp, :natoms]

        E_blocks = build_E_blocks_full(pseudos[sp.element])

        # Group projectors by l
        per_l: dict[int, list[int]] = {}
        for ip in range(sp.n_proj):
            l = int(sp.proj_l[ip])
            per_l.setdefault(l, []).append(ip)

        for l, proj_ids in per_l.items():
            E_np = E_blocks.get(l)
            if E_np is None:
                continue
            nbeta = len(proj_ids)
            msize = 2 * l + 1
            R = nbeta * msize

            # F_l(q) from pre-built tables → G_l = F_l/q^l, then differentiate
            for ip in proj_ids:
                F_vals = tables["proj_tables"][isp][ip]
                if l == 0:
                    G_vals = F_vals.copy()
                else:
                    G_vals = np.empty(n_q, dtype=np.float64)
                    G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
                    G_vals[0] = F_vals[1] / q_grid[1] ** l
                G_rows.append(G_vals)
                Gp_rows.append(differentiate_uniform_table(G_vals, dq))

            channels.append(ChannelMeta(
                l=l, nbeta=nbeta, msize=msize, R=R,
                tau=tau, E=E_np, beta_table_start=beta_idx,
                natoms=natoms,
            ))
            beta_idx += nbeta

    G_table = jnp.asarray(np.stack(G_rows), dtype=jnp.float64) if G_rows else jnp.zeros((0, n_q), dtype=jnp.float64)
    Gp_table = jnp.asarray(np.stack(Gp_rows), dtype=jnp.float64) if Gp_rows else jnp.zeros((0, n_q), dtype=jnp.float64)
    total_R = sum(ch.R * ch.natoms for ch in channels)
    l_max = max((ch.l for ch in channels), default=0)

    # ── Pre-flatten per-row metadata for vectorized Z assembly ──
    # Each row of Z corresponds to one (atom, beta, m) combination.
    # Flatten ALL channels into (total_R,) index arrays.
    row_beta_idx = []
    row_l = []
    row_m = []
    row_tau = []

    for ch in channels:
        for a in range(ch.natoms):
            for ib in range(ch.nbeta):
                for im in range(ch.msize):
                    row_beta_idx.append(ch.beta_table_start + ib)
                    row_l.append(ch.l)
                    row_m.append(im)
                    row_tau.append(ch.tau[a])

    row_beta_idx_j = jnp.asarray(row_beta_idx, dtype=jnp.int32)
    row_l_j = jnp.asarray(row_l, dtype=jnp.int32)
    row_m_j = jnp.asarray(row_m, dtype=jnp.int32)
    row_tau_np = np.array(row_tau, dtype=np.float64).reshape(-1, 3) if row_tau else np.zeros((0, 3), dtype=np.float64)
    row_tau_j = jnp.asarray(row_tau_np, dtype=jnp.float64)

    # ── Pre-build E_super (k-independent block-diagonal D-matrix) ──
    E_super = np.zeros((nspinor, nspinor, total_R, total_R), dtype=np.complex128)
    offset = 0
    for ch in channels:
        E_np = ch.E[:nspinor, :nspinor]
        R = ch.R
        for a in range(ch.natoms):
            E_super[:, :, offset:offset+R, offset:offset+R] = E_np
            offset += R
    E_super_j = jnp.asarray(E_super, dtype=jnp.complex128)

    return VNLSetup(
        channels=channels,
        dq=dq, n_q=n_q, q_max=q_max,
        G_table=G_table, Gp_table=Gp_table,
        prefactor=prefactor,
        B=B, cell_volume=cell_volume,
        total_R=total_R, nspinor=nspinor,
        E_super=E_super_j, l_max=l_max,
        row_beta_idx=row_beta_idx_j,
        row_l=row_l_j, row_m=row_m_j, row_tau=row_tau_j,
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
    from psp.dft_operators import generate_gvectors_k

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
    """Build dense VNL projectors Z — fully vectorized, no Python loops.

    Z[r, G] = c_il * G_beta(q) * S_lm(K) * exp(-2πi K·τ)

    All per-row metadata (beta_idx, l, m, tau) was pre-flattened at setup time.
    """
    from psp.radial.solid_harmonics import all_solid_harmonics

    nG = Gk_np.shape[0]
    K_crys = jnp.asarray(Gk_np, dtype=jnp.float64) + jnp.asarray(kvec)[None, :]
    B_j = jnp.asarray(setup.B, dtype=jnp.float64)
    K_cart = K_crys @ B_j
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + 1e-60)

    # Radial form factors: evaluate all betas at all G-vectors
    G_all = _table_interp(q, setup.dq, setup.G_table)    # (total_nbeta, nG)

    # Solid harmonics: all l in one call
    S_all = all_solid_harmonics(K_cart, l_max=setup.l_max)  # (l_max+1, 2*l_max+1, nG)

    # ── Vectorized Z assembly — one operation, no loops ──
    # Each row r: Z[r] = c_il[r] * G[beta_idx[r], :] * S[l[r], m[r], :] * phase[r, :]
    G_r = G_all[setup.row_beta_idx]                        # (total_R, nG)
    S_r = S_all[setup.row_l, setup.row_m]                  # (total_R, nG)
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ setup.row_tau.T)).T  # (total_R, nG)
    c_il_r = setup.prefactor * (1j) ** setup.row_l         # (total_R,)

    Z = c_il_r[:, None] * G_r * S_r * phase_r             # (total_R, nG)

    # dZ for velocity (optional — still uses per-channel JVP, TODO: vectorize)
    dZ_j = None
    if compute_dZ:
        K_over_q = K_cart / q[:, None]
        Gp_all = _table_interp(q, setup.dq, setup.Gp_table)
        dZ_blocks = []
        for ch in setup.channels:
            l, nbeta, R = ch.l, ch.nbeta, ch.R
            tau_j = jnp.asarray(ch.tau, dtype=jnp.float64)
            c_il = setup.prefactor * (1j) ** l

            G_bG = G_all[ch.beta_table_start:ch.beta_table_start + nbeta]
            Gp_bG = Gp_all[ch.beta_table_start:ch.beta_table_start + nbeta]
            S = S_all[l, :2*l+1]
            phase = jnp.exp(-2j * jnp.pi * (K_crys @ tau_j.T)).T

            dS = jnp.stack([
                jax.jvp(lambda K: _solid_harmonics_jax(l, K),
                        (K_cart,), (jnp.zeros((nG, 3)).at[:, j].set(1.0),))[1]
                for j in range(3)
            ], axis=0)

            Binv = jnp.linalg.inv(B_j)
            dphase = -2j * jnp.pi * (tau_j @ Binv.T)[:, :, None] * phase[:, None, :]
            drad = Gp_bG[None, :, :] * K_over_q.T[:, None, :]
            core = drad[:, :, None, :] * S[None, None, :, :] + G_bG[None, :, None, :] * dS[:, None, :, :]
            dZ_core = c_il * phase[:, None, None, None, :] * core[None, :]
            radS = G_bG[:, None, :] * S[None, :, :]
            dZ_phase = c_il * radS[None, None, :, :, :] * dphase[:, :, None, None, :]
            dZ_blocks.append((dZ_core + dZ_phase).transpose(1, 0, 2, 3, 4).reshape(3, ch.natoms * R, nG))
        dZ_j = jnp.concatenate(dZ_blocks, axis=1) if dZ_blocks else None

    return VNLKData(
        Z=Z, E_super=setup.E_super, nG=nG,
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
