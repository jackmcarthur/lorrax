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

import functools
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp

from psp.radial.build_projectors_qe import (
    build_E_blocks_full, pseudo_has_j_channels, pseudo_soc_strength_ry,
)
from psp.radial.radial_jax import differentiate_uniform_table
from psp.radial.solid_harmonics import solid_harmonics_jax as _solid_harmonics_jax
from psp.radial_tables import projector_deriv_table as _projector_deriv_table


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
    # WHICH PROJECTORS THESE ARE.  True = j-resolved (spin-orbit in V_NL);
    # False = j-averaged scalar-relativistic (QE average_pp).  Resolved by
    # ``resolve_soc_mode`` and carried so a consumer can stamp it into an
    # artifact's provenance instead of guessing from ``nspinor``.
    soc: bool = True
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
    # Optional: Cartesian k-derivatives for velocity.  ``None`` IF AND ONLY
    # IF the builder was called with ``compute_dZ=False`` — a builder asked
    # for dZ always returns a ``(3, total_R, nG)`` array, including the
    # degenerate ``total_R == 0`` case (no channels), where it is the empty
    # array rather than ``None``.  Consumers may therefore branch on
    # ``compute_dZ``, never on ``dZ is None`` after asking for it.
    dZ: jax.Array | None            # (3, total_R, nG), or None iff not asked for
    # 1 on the k's physical G, 0 on the ngkmax pad — None when the G-list
    # was the k's own ragged sphere.  Set by ``build_vnl_kdata`` (the
    # SymMaps path), which ALSO zeroes Z/dZ on the pad columns, so this
    # field documents the layout rather than being a required argument to
    # anything: a kdata from that route is already inert on the pad.
    g_mask: np.ndarray | None = None   # (nG,) float64 or None


@dataclass(frozen=True)
class VNLProjectorCoefficientBlock:
    r"""Low-rank ``<beta|psi>`` coefficients and operator derivatives.

    ``c=(group,beta,spin,band)``.  When present, ``dc`` and ``d2c`` prepend
    one and two Cartesian derivative axes.  ``group`` is normally the atom
    axis for one channel; the flattened production projector uses one group.
    The band axis stays last so callers can persist or stream their existing
    band blocks without ever materialising a band-by-G derivative state.
    """

    c: jax.Array
    dc: jax.Array | None
    d2c: jax.Array | None
    E: jax.Array


@dataclass(frozen=True)
class VNLMatrixDerivatives:
    """Band matrices ``V_NL``, ``dV_NL/dK``, and ``d2V_NL/dK2``."""

    value: jax.Array
    gamma_cart: jax.Array
    lambda_cart: jax.Array


@dataclass(frozen=True)
class VNLKetDerivatives:
    """Explicit G-space VNL action, formed only when a consumer requests it."""

    value_ket: jax.Array
    gamma_cart_ket: jax.Array
    lambda_cart_ket: jax.Array


# ---------------------------------------------------------------------------
# Setup builder
# ---------------------------------------------------------------------------

SOC_BANNER = "=" * 78


def resolve_soc_mode(pseudos, wfn=None, *, soc=None, nspinor=1,
                     caller: str = "", print_fn=print) -> bool:
    """Decide j-RESOLVED vs j-AVERAGED projectors, and SAY SO.

    ``noncolin`` (nspinor=2) is NOT ``lspinorb``.  Nothing in a BerkeleyGW
    ``WFN.h5`` records which one the DFT run used — ``mf_header`` carries
    ``nspinor`` and nothing else — so for that input the question is not
    answerable from the file and the honest outcome is an ANNOUNCED
    assumption, never a silent one.

    Signals consulted, in order of authority:

    1. ``soc=`` passed by the caller (a deck key or CLI flag).  AUTHORITATIVE.
    2. ``wfn.spinorbit`` — QE's ``<spinorbit>`` from ``data-file-schema.xml``,
       present when the structure came from a QE ``.save`` (see
       ``file_io.qe_save_reader``).  AUTHORITATIVE.
    3. Nothing.  UNDETERMINED → keep the historical j-resolved default and
       print the banner below.

    Returns the resolved ``soc`` bool to hand to ``build_E_blocks_full``.

    WHEN THIS RETURNS QUIETLY (the FALSE branch, i.e. no announcement):
      * no pseudo resolves j (scalar-relativistic UPF, or ℓ=0 only) — there is
        no choice to make, both paths build the same operator;
      * nspinor == 1 and no pseudo resolves j — same;
      * an authoritative signal exists and it agrees with what will be built.
    Those are real cases that occur in this repo's own fixtures, so the check
    is not vacuously loud.
    """
    tag = f"[{caller}] " if caller else ""
    j_pseudos = {el: p for el, p in (pseudos or {}).items()
                 if pseudo_has_j_channels(p)}

    # ── nothing resolves j: the question does not arise ──
    if not j_pseudos:
        if soc:
            print_fn(f"{tag}soc=True requested but no pseudopotential resolves "
                     f"j = ℓ±1/2; V_NL is spin-scalar either way.")
        return False

    declared = soc
    source = "caller"
    if declared is None:
        declared = getattr(wfn, "spinorbit", None)
        source = "QE <spinorbit>"

    strengths = {el: pseudo_soc_strength_ry(p) for el, p in j_pseudos.items()}
    worst_ry = max(strengths.values(), default=0.0)
    worst_ev = worst_ry * 13.605693122994

    if declared is None:
        # ── UNDETERMINED.  Announce; do not decide silently. ──
        print_fn(f"\n{SOC_BANNER}")
        print_fn(f"{tag}SPIN-ORBIT MODE UNDETERMINED — assuming lspinorb=.TRUE.")
        print_fn(SOC_BANNER)
        print_fn(f"  Fully-relativistic pseudopotentials: {sorted(j_pseudos)}")
        print_fn(f"  Wavefunctions: nspinor={nspinor} "
                 f"(noncollinear — which does NOT imply spin-orbit)")
        print_fn(f"  max |D(ℓ,j=ℓ+1/2) − D(ℓ,j=ℓ−1/2)| = {worst_ry:.6f} Ry "
                 f"= {worst_ev:.4f} eV")
        print_fn("")
        print_fn("  V_NL will be built j-RESOLVED, i.e. WITH spin-orbit.  If the")
        print_fn("  DFT run that produced these wavefunctions used")
        print_fn("  noncolin=.true., lspinorb=.false., then QE ran average_pp")
        print_fn("  and its eigenvalues carry NO spin-orbit — this operator will")
        print_fn("  split degeneracies the wavefunctions do not have.")
        print_fn("")
        print_fn("  Nothing in a BerkeleyGW WFN.h5 records lspinorb.  Resolve it:")
        print_fn("    * pass soc=True/False explicitly, or")
        print_fn("    * build from a QE .save so <spinorbit> can be read.")
        print_fn(SOC_BANNER + "\n")
        return True

    declared = bool(declared)
    if declared:
        print_fn(f"{tag}V_NL: j-RESOLVED (spin-orbit ON), from {source}.  "
                 f"FR pseudos {sorted(j_pseudos)}, "
                 f"ΔD = {worst_ry:.6f} Ry = {worst_ev:.4f} eV.")
        if nspinor == 1:
            print_fn(f"\n{SOC_BANNER}")
            print_fn(f"{tag}MISMATCH: spin-orbit requested with nspinor=1.")
            print_fn(SOC_BANNER)
            print_fn("  A j-resolved V_NL has no representation on one-component")
            print_fn("  wavefunctions: only the E^{↑↑} block is retained, which is")
            print_fn("  m-dependent and is NOT the scalar-relativistic operator.")
            print_fn("  Pass soc=False for a scalar-relativistic run.")
            print_fn(SOC_BANNER + "\n")
        return True

    # declared False, and the pseudos DO resolve j → the averaged path
    print_fn(f"{tag}V_NL: j-AVERAGED (scalar-relativistic, QE average_pp), "
             f"from {source}.  FR pseudos {sorted(j_pseudos)}, "
             f"discarding ΔD = {worst_ry:.6f} Ry = {worst_ev:.4f} eV of "
             f"spin-orbit.")
    return False


def build_vnl_setup(
    wfn, sym=None, meta=None, pseudos=None,
    n_q: int = 4000,
    nspinor: int | None = None,
    q_max: float | None = None,
    soc: bool | None = None,
    print_fn=print,
) -> VNLSetup:
    """Build k-independent VNL data: radial tables, channel metadata.

    Parameters
    ----------
    wfn : WFNReader or CrystalData (needs atom_crys, bvec, blat, cell_volume)
    sym : SymMaps, optional — used to determine q_max if not provided.
    meta : Meta, optional — used with sym for q_max scan.
    pseudos : dict — element → UPF
    q_max : float, optional — if provided, skip the k-point scan for q_max.
    soc : bool, optional — j-RESOLVED (True) vs j-AVERAGED (False) projectors.
        ``None`` (default) means "not declared": ``resolve_soc_mode`` looks for
        a QE ``<spinorbit>`` flag and, failing that, keeps the historical
        j-resolved behaviour and ANNOUNCES the assumption.  See
        ``psp.radial.build_projectors_qe`` for why noncolin ≠ lspinorb.
    """
    from psp.species import extract_species, build_atom_species_map
    from psp.radial_tables import build_all_tables

    if nspinor is None:
        nspinor = int(meta.nspinor) if meta is not None else int(wfn.nspinor)

    soc_resolved = resolve_soc_mode(
        pseudos, wfn, soc=soc, nspinor=int(nspinor),
        caller="build_vnl_setup", print_fn=print_fn)

    B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=float)
    cell_volume = float(wfn.cell_volume)
    prefactor = (4.0 * np.pi) / np.sqrt(cell_volume)

    # Determine q_max
    if q_max is None:
        if hasattr(wfn, "ecutwfc"):
            q_max = float(np.sqrt(float(wfn.ecutwfc)))
        else:
            from psp.dft_operators import padded_gvectors
            tab = padded_gvectors(wfn, k="full_bz")
            q_max = 0.0
            for ik in range(sym.nk_tot):
                G_pad, g_mask = tab.at(ik)
                kvec = np.asarray(sym.unfolded_kpts[ik], dtype=float)
                K_cart = (G_pad.astype(float) + kvec[None, :]) @ B
                # Pad rows are G=(0,0,0), i.e. |k| — small, but not
                # provably below the physical maximum, so zero them
                # rather than argue about it.  q_max only ever grows the
                # radial table, so an over-estimate is harmless while an
                # under-estimate silently clips the form factors.
                qk = np.sqrt(np.sum(K_cart ** 2, axis=1)) * g_mask
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

        E_blocks = build_E_blocks_full(pseudos[sp.element], soc=soc_resolved)

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

            # F_l(q) from pre-built tables → G_l = F_l/q^l.
            # Gp_l = dG_l/dq is obtained ANALYTICALLY via the Bessel-recurrence
            # formula  dG_l/dq = -H_{l+1}(β; q)/q^l  (see radial_tables.py).
            # H_{l+1} is the *raw* (unscaled) deriv Hankel pre-computed by
            # build_all_tables on GPU; we apply the q^l division here.  This
            # replaces an earlier central-FD derivative which was O(dq²)
            # biased for curved G_l(q) and led to ~10% errors in velocity
            # matrix elements taken via jax.jvp through _table_interp.
            for ip in proj_ids:
                F_vals = tables["proj_tables"][isp][ip]
                H_vals = tables["deriv_tables"][isp][ip]
                if l == 0:
                    G_vals = F_vals.copy()
                    Gp_vals = -H_vals
                else:
                    G_vals = np.empty(n_q, dtype=np.float64)
                    G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
                    G_vals[0] = F_vals[1] / q_grid[1] ** l
                    Gp_vals = np.zeros(n_q, dtype=np.float64)
                    Gp_vals[1:] = -H_vals[1:] / q_grid[1:] ** l
                    # q=0 limit of dG_l/dq is 0 (j_{l+1}(qr) ~ q^{l+1}, so
                    # H_{l+1}/q^l → 0 as q → 0).
                G_rows.append(G_vals)
                Gp_rows.append(Gp_vals)

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
        E_super=E_super_j, l_max=l_max, soc=bool(soc_resolved),
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


@jax.custom_jvp
def _interp_with_deriv(q, dq, table, deriv_table):
    """Linear interp of ``table`` at ``q``, with a custom JVP that uses
    ``deriv_table`` as the q-tangent instead of the forward-slope of the
    linear interpolation.

    Drop-in replacement for ``_table_interp(q, dq, table)`` whenever callers
    want ``jax.jvp`` / ``jax.jacfwd`` through the form factor to produce the
    **physical** derivative (velocity operator matrix elements, k·p response,
    ∂χ/∂q in the Sternheimer solver).  The physical dG/dq is tabulated at
    setup time via the Bessel-recurrence identity (see
    ``radial_tables.projector_deriv_table``) and passed in as
    ``deriv_table`` — this avoids the O(dq²) bias that afflicts the
    forward-slope derivative of the linear interpolant.
    """
    return _table_interp(q, dq, table)


@_interp_with_deriv.defjvp
def _interp_with_deriv_jvp(primals, tangents):
    q, dq, table, deriv_table = primals
    q_dot, _, _, _ = tangents
    val = _table_interp(q, dq, table)
    slope = _table_interp(q, dq, deriv_table)
    return val, slope * q_dot


# ---------------------------------------------------------------------------
# Per-k projector construction
# ---------------------------------------------------------------------------

def build_vnl_kdata(
    k_idx: int,
    setup: VNLSetup,
    wfn, sym, meta,
    *,
    compute_dZ: bool = False,
    gvectors=None,
) -> VNLKData:
    """Build dense Z [and dZ] for one k-point (SymMaps path).

    The G-list is the loader's FIXED ``(ngkmax, 3)`` table, so
    ``_assemble_Z_jit`` — whose compile cache is keyed on ``nG`` — lowers
    ONCE for the whole k sweep instead of once per distinct ``ngk``.

    Z and dZ are returned already ZEROED on the pad columns, and the mask
    that did it is carried on :attr:`VNLKData.g_mask`.  That is what makes
    this route safe to drop into a caller that does not mask ψ: every
    contraction in ``vnl_matrix`` / ``apply_vnl`` / ``vnl_velocity_matrix``
    passes through Z or dZ at least once, so a zero there is enough.
    (Left unmasked, the pad columns would be finite — the core evaluates
    Z at K = kvec on those rows — and would contract against ψ(Γ), which
    the pad rows alias.)

    Pass ``gvectors`` (a :class:`psp.dft_operators.PaddedGVectors`) to
    reuse one table across a sweep.
    """
    from psp.dft_operators import padded_gvectors

    tab = padded_gvectors(wfn, k="full_bz") if gvectors is None else gvectors
    G_pad, g_mask = tab.at(k_idx)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)
    kdata = _build_vnl_kdata_core(kvec, np.asarray(G_pad, dtype=int),
                                  setup, compute_dZ=compute_dZ)
    mask_j = jnp.asarray(g_mask, dtype=kdata.Z.real.dtype)
    return VNLKData(
        Z=kdata.Z * mask_j[None, :],
        E_super=kdata.E_super,
        nG=kdata.nG,
        total_R=kdata.total_R,
        dZ=None if kdata.dZ is None else kdata.dZ * mask_j[None, None, :],
        g_mask=g_mask,
    )


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


def build_vnl_kdata_traced(kvec, Gk_int, setup: VNLSetup, *,
                           compute_dZ: bool = False) -> VNLKData:
    """:func:`build_vnl_kdata_from_kvec` for a TRACED ``kvec``/``Gk_int``.

    Same body, minus the ``np.asarray`` coercion of the two per-k
    operands — which is what forbids the eager entry point inside a
    trace: ``np.asarray`` on a ``lax.scan`` carry raises
    ``TracerArrayConversionError``.  The core is already pure jax in
    both branches (``_assemble_Z_jit`` for ``compute_dZ=False``, jnp +
    ``jax.jvp`` for the derivative), so nothing else has to change.

    The caller is ``common.mtxel_sweep``'s V_NL and dipole operators,
    which build Z (and dZ) for the scan's current k.  Everything the
    setup carries is k-independent and closes over as a constant; only
    ``kvec`` and the D10 fixed-shape G table vary per iteration, so the
    body lowers ONCE for the whole sweep.
    """
    return _build_vnl_kdata_core(kvec, Gk_int, setup, compute_dZ=compute_dZ)


@functools.partial(jax.jit, static_argnames=('l_max',))
def _assemble_Z_jit(
    kvec, Gk_int,
    B, dq, G_table, Gp_table, prefactor,
    row_beta_idx, row_l, row_m, row_tau,
    *, l_max,
):
    """JIT'd body of ``_build_vnl_kdata_core`` for ``compute_dZ=False``.

    Pulled to module scope so its compile cache is shared across every
    per-k call site (run_sternheimer, run_nscf Davidson, kpm_dos,
    get_dipole_mtxels).  Without this, each k re-traces the eager
    ``K_cart``/``q``/``S_all``/``G_all``/``Z`` arithmetic and emits a
    long tail of single-primitive XLA modules.

    All inputs are JAX arrays — ``setup`` is decomposed at the call site
    so the dataclass (a Python pytree, not abstractable positionally)
    doesn't enter the jit.  ``l_max`` is the only static arg (it
    controls the unrolled solid-harmonics block).
    """
    from psp.radial.solid_harmonics import all_solid_harmonics

    K_crys = Gk_int.astype(jnp.float64) + kvec[None, :]
    K_cart = K_crys @ B
    # Regularizer: avoids 1/q divergence in autodiff.  See
    # _build_vnl_kdata_core for the full physics rationale.
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + 1e-8)
    G_all = _interp_with_deriv(q, dq, G_table, Gp_table)        # (total_nbeta, nG)
    S_all = all_solid_harmonics(K_cart, l_max=l_max)            # (l_max+1, 2*l_max+1, nG)

    G_r = G_all[row_beta_idx]                                    # (total_R, nG)
    S_r = S_all[row_l, row_m]                                    # (total_R, nG)
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ row_tau.T)).T     # (total_R, nG)
    c_il_r = prefactor * (1j) ** row_l                           # (total_R,)
    return c_il_r[:, None] * G_r * S_r * phase_r                 # (total_R, nG)


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

    Tail-G handling: if ``Gk_np`` is pre-padded by the caller (e.g.
    ``setup_H_k_from_kvec`` padding to ``ngkmax``), Z is computed at
    those padded entries too — at K = kvec (no zero-G in the padded
    rows), which gives finite, in-table values for q.  The caller is
    responsible for masking Z at padded entries before any contraction.
    """
    nG = Gk_np.shape[0]

    if not compute_dZ:
        Z = _assemble_Z_jit(
            jnp.asarray(kvec, dtype=jnp.float64),
            jnp.asarray(Gk_np, dtype=jnp.int32),
            jnp.asarray(setup.B, dtype=jnp.float64),
            jnp.asarray(setup.dq, dtype=jnp.float64),
            setup.G_table, setup.Gp_table,
            jnp.asarray(setup.prefactor, dtype=jnp.float64),
            setup.row_beta_idx, setup.row_l, setup.row_m, setup.row_tau,
            l_max=int(setup.l_max),
        )
        return VNLKData(Z=Z, E_super=setup.E_super, nG=nG,
                        total_R=setup.total_R, dZ=None)

    # ── compute_dZ=True path: still eager (used by get_dipole_mtxels).
    #    TODO: jit this too once the per-channel for-loop is vectorised.
    from psp.radial.solid_harmonics import all_solid_harmonics

    K_crys = jnp.asarray(Gk_np, dtype=jnp.float64) + jnp.asarray(kvec)[None, :]
    B_j = jnp.asarray(setup.B, dtype=jnp.float64)
    K_cart = K_crys @ B_j
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + 1e-8)

    G_all = _interp_with_deriv(q, setup.dq, setup.G_table, setup.Gp_table)
    S_all = all_solid_harmonics(K_cart, l_max=setup.l_max)

    G_r = G_all[setup.row_beta_idx]
    S_r = S_all[setup.row_l, setup.row_m]
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ setup.row_tau.T)).T
    c_il_r = setup.prefactor * (1j) ** setup.row_l

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
        # ``compute_dZ=True`` PROMISES AN ARRAY, and the empty-channel case
        # used to break that promise silently.  A setup with no channels
        # (no pseudopotentials loaded, or none covering the structure)
        # produces ``dZ_blocks == []``; returning ``None`` there handed a
        # `NoneType` to ``apply_vnl_velocity_to_ket``, which conjugates its
        # ``dZ`` argument — so the failure surfaced ~30 s later as
        # ``TypeError: conjugate requires ndarray or scalar arguments`` six
        # frames inside a jitted einsum, naming neither the deck nor the
        # missing file.  ``Z`` already degrades gracefully in that case (it
        # is the ``(0, nG)`` empty projector matrix and every contraction
        # through it is zero); ``dZ`` now degrades the SAME way, at
        # ``(3, total_R, nG)`` with ``total_R == 0``.
        #
        # THIS IS THE SECOND LINE OF DEFENCE, NOT THE FIX.  A zero V_NL
        # velocity is the arithmetically correct answer for an empty
        # projector set and the wrong ANSWER for a real deck, which is why
        # the drivers must refuse the empty set up front —
        # ``psp.operator_checks.validate_operator_inputs``, now called by
        # ``get_dipole_mtxels`` as it already was by ``gw.kin_ion_io`` and
        # ``get_DFT_mtxels``.  What this branch buys is that the kernel's
        # documented contract is true for every caller, including the ones
        # that legitimately hold no projectors.
        dZ_j = (jnp.concatenate(dZ_blocks, axis=1) if dZ_blocks
                else jnp.zeros((3, int(setup.total_R), nG), dtype=Z.dtype))

    return VNLKData(
        Z=Z, E_super=setup.E_super, nG=nG,
        total_R=setup.total_R, dZ=dZ_j,
    )


# ---------------------------------------------------------------------------
# Core operators — single JIT, no loops
# ---------------------------------------------------------------------------

def contract_projector_coefficients(
    psi_G,
    Z,
    E_super,
    *,
    dZ=None,
    d2Z=None,
):
    r"""Contract a band block once with beta and its operator derivatives.

    ``Z`` may be ``(beta,G)`` or ``(group,beta,G)``; derivative arrays
    prepend Cartesian axes.  The returned carrier has no G axis.  This is
    the canonical coefficient-space entry point shared by flattened VNL
    applications and the channel-streamed gauge-vertex builder.
    """
    Z_grouped = Z[None, ...] if Z.ndim == 2 else Z
    if Z_grouped.ndim != 3:
        raise ValueError("Z must have shape (beta,G) or (group,beta,G)")
    c = jnp.einsum(
        'prG,nsG->prsn', jnp.conj(Z_grouped), psi_G, optimize=True)

    dc = None
    if dZ is not None:
        dZ_grouped = dZ[:, None, ...] if dZ.ndim == 3 else dZ
        if dZ_grouped.ndim != 4:
            raise ValueError(
                "dZ must have shape (cart,beta,G) or "
                "(cart,group,beta,G)")
        dc = jnp.einsum(
            'xprG,nsG->xprsn', jnp.conj(dZ_grouped), psi_G,
            optimize=True)

    d2c = None
    if d2Z is not None:
        d2Z_grouped = d2Z[:, :, None, ...] if d2Z.ndim == 4 else d2Z
        if d2Z_grouped.ndim != 5:
            raise ValueError(
                "d2Z must have shape (cart,cart,beta,G) or "
                "(cart,cart,group,beta,G)")
        d2c = jnp.einsum(
            'xyprG,nsG->xyprsn', jnp.conj(d2Z_grouped), psi_G,
            optimize=True)

    return VNLProjectorCoefficientBlock(
        c=c, dc=dc, d2c=d2c, E=E_super)


def vnl_value_and_gamma_from_projector_coefficients(block):
    r"""Close ``c^dag E c`` and its first operator derivative in beta space."""
    c, dc, E = block.c, block.dc, block.E
    Ec = jnp.einsum('strq,pqtn->prsn', E, c, optimize=True)
    value = jnp.einsum(
        'prsm,prsn->mn', jnp.conj(c), Ec, optimize=True)
    if dc is None:
        return value, None
    Edc = jnp.einsum('strq,xpqtn->xprsn', E, dc, optimize=True)
    gamma = (
        jnp.einsum(
            'xprsm,prsn->xmn', jnp.conj(dc), Ec, optimize=True)
        + jnp.einsum(
            'prsm,xprsn->xmn', jnp.conj(c), Edc, optimize=True)
    )
    return value, gamma


def vnl_matrix_derivatives_from_projector_coefficients(block):
    r"""Close the exact first and second VNL operator derivatives.

    ``E`` is k-independent in the canonical norm-conserving
    pseudopotential owner.  At fixed ket block, the Hessian is

    ``d2(c^dag E c) = d2c^dag E c + dc^dag E dc``
    ``                  + dc^dag E dc + c^dag E d2c``.

    Eigenstate/projector drift is deliberately absent; it belongs to the
    Sternheimer response, not to this Hamiltonian-derivative API.
    """
    value, gamma = vnl_value_and_gamma_from_projector_coefficients(block)
    if block.dc is None or block.d2c is None:
        raise ValueError("VNL contact requires both dZ and d2Z coefficients")
    c, dc, d2c, E = block.c, block.dc, block.d2c, block.E
    Ec = jnp.einsum('strq,pqtn->prsn', E, c, optimize=True)
    Edc = jnp.einsum('strq,xpqtn->xprsn', E, dc, optimize=True)
    Ed2c = jnp.einsum(
        'strq,xypqtn->xyprsn', E, d2c, optimize=True)
    contact = (
        jnp.einsum(
            'xyprsm,prsn->xymn', jnp.conj(d2c), Ec, optimize=True)
        + jnp.einsum(
            'xprsm,yprsn->xymn', jnp.conj(dc), Edc, optimize=True)
        + jnp.einsum(
            'yprsm,xprsn->xymn', jnp.conj(dc), Edc, optimize=True)
        + jnp.einsum(
            'prsm,xyprsn->xymn', jnp.conj(c), Ed2c, optimize=True)
    )
    return VNLMatrixDerivatives(
        value=value, gamma_cart=gamma, lambda_cart=contact)


def _apply_vnl_value_and_gamma_from_projector_coefficients(block, Z, dZ):
    """One coefficient-space owner for explicit value/velocity ket action."""
    Z_grouped = Z[None, ...] if Z.ndim == 2 else Z
    dZ_grouped = dZ[:, None, ...] if dZ.ndim == 3 else dZ
    c, dc, E = block.c, block.dc, block.E
    Ec = jnp.einsum('strq,pqtn->prsn', E, c, optimize=True)
    Edc = jnp.einsum('strq,xpqtn->xprsn', E, dc, optimize=True)
    value_ket = jnp.einsum(
        'prG,prsn->nsG', Z_grouped, Ec, optimize=True)
    gamma_ket = (
        jnp.einsum(
            'xprG,prsn->xnsG', dZ_grouped, Ec, optimize=True)
        + jnp.einsum(
            'prG,xprsn->xnsG', Z_grouped, Edc, optimize=True)
    )
    return value_ket, gamma_ket, Ec, Edc


def apply_vnl_derivatives_to_ket(psi_G, Z, dZ, d2Z, E_super):
    r"""Apply ``V_NL``, ``dV_NL``, and ``d2V_NL`` to a fixed ket block.

    This is the explicit G-space door for Sternheimer/operator-action
    consumers.  It first contracts the canonical low-rank coefficients and
    re-expands whole blocks only here.  With k-independent ``E``, the contact
    action is

    ``ddZ_ij E c + dZ_i E dc_j + dZ_j E dc_i + Z E d2c_ij``.

    ``Z`` may be flattened or retain a leading independent group/atom axis,
    matching :func:`contract_projector_coefficients`.
    """
    block = contract_projector_coefficients(
        psi_G, Z, E_super, dZ=dZ, d2Z=d2Z)
    Z_grouped = Z[None, ...] if Z.ndim == 2 else Z
    dZ_grouped = dZ[:, None, ...] if dZ.ndim == 3 else dZ
    d2Z_grouped = d2Z[:, :, None, ...] if d2Z.ndim == 4 else d2Z
    d2c = block.d2c
    value_ket, gamma_ket, Ec, Edc = (
        _apply_vnl_value_and_gamma_from_projector_coefficients(
            block, Z_grouped, dZ_grouped))
    Ed2c = jnp.einsum(
        'strq,xypqtn->xyprsn', E_super, d2c, optimize=True)
    lambda_ket = (
        jnp.einsum(
            'xyprG,prsn->xynsG', d2Z_grouped, Ec, optimize=True)
        + jnp.einsum(
            'xprG,yprsn->xynsG', dZ_grouped, Edc, optimize=True)
        + jnp.einsum(
            'yprG,xprsn->xynsG', dZ_grouped, Edc, optimize=True)
        + jnp.einsum(
            'prG,xyprsn->xynsG', Z_grouped, Ed2c, optimize=True)
    )
    return VNLKetDerivatives(
        value_ket=value_ket,
        gamma_cart_ket=gamma_ket,
        lambda_cart_ket=lambda_ket,
    )

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
    block = contract_projector_coefficients(psi_G, Z, E_super)
    value, _ = vnl_value_and_gamma_from_projector_coefficients(block)
    return value


@jax.jit
def apply_vnl_velocity_to_ket(psi_G, Z, dZ, E_super):
    """∂V_NL/∂K_cart^α applied to a ket on the G-sphere.

    Returns ``(3, nb, nspinor, nG)`` complex — the velocity-applied ket
    ``v_NL^α |n,k⟩``  in the SAME G-sphere layout as ``psi_G``.

    Math:  ``∂V_NL/∂k^α = (∂Z^α) E Z† + Z E (∂Z^α)†``  (k-dependence
    flows through the projectors Z(k); E_super is k-independent).  The
    apply form leaves the bra index free so callers can either contract
    against ``conj(psi_G)`` to get the q=0 matrix element (see
    :func:`vnl_velocity_matrix`) OR against a different bra at a
    different k for the finite-q matrix element used in the SOS chi
    head/wing pipeline.
    """
    block = contract_projector_coefficients(
        psi_G, Z, E_super, dZ=dZ)
    _, gamma_ket, _, _ = (
        _apply_vnl_value_and_gamma_from_projector_coefficients(
            block, Z, dZ))
    return gamma_ket


@jax.jit
def vnl_velocity_matrix(psi_G, Z, dZ, E_super):
    """⟨m | ∂V_NL/∂K_cart^α | n⟩ matrix elements at one k.  Returns (3, nb, nb).

    A same-k band matrix closes directly in beta-coefficient space.  The
    separate :func:`apply_vnl_velocity_to_ket` door remains authoritative
    when a Sternheimer solve or a different finite-q bra genuinely needs
    the G-space operator action.
    """
    block = contract_projector_coefficients(
        psi_G, Z, E_super, dZ=dZ)
    _, gamma = vnl_value_and_gamma_from_projector_coefficients(block)
    return gamma
