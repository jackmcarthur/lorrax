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
    # ``(start, stop, channel_index)`` for each independently coupled atom
    # block.  Low-memory kernels traverse these blocks without ever splitting
    # a canonical ``ChannelMeta.E`` matrix or inventing a second dense D.
    coupled_row_blocks: tuple[tuple[int, int, int], ...] = ()
    # Analytic second radial derivative of the reduced projector form factor.
    # Kept beside G/Gp so every VNL derivative consumes the same radial owner.
    Gpp_table: jax.Array | None = None     # (total_nbeta, n_q)
    # Analytic third radial derivative.  This is a separately priced
    # capability for the ICL q^2 transfer jet; uniform current/contact does
    # not allocate or compile it.
    Gppp_table: jax.Array | None = None    # (total_nbeta, n_q)
    # Content identity built from the host radial/projector/E data before
    # device transfer.  Empty only on hand-built test fixtures; production
    # uniform gauge transactions refuse an empty value.
    uniform_gauge_fingerprint: str = ""


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
class _VNLProjectorCoefficientBlock:
    r"""Private in-memory ``<beta|psi>`` operator-derivative carrier.

    It has no G axis and never crosses a public/persistence boundary. A
    durable carrier would require a canonical k/G/mask/PP/SOC/band-window
    fingerprint, which no shared artifact contract owns yet.
    """

    c: jax.Array                    # (R, spin, band)
    dc_cart: jax.Array              # (cart, R, spin, band)
    d2c_cart: jax.Array             # (cart, cart, R, spin, band)
    E: jax.Array
    d3c_cart: jax.Array | None = None


@dataclass(frozen=True)
class VNLGaugeKetDerivatives:
    """Canonical VNL current/contact action on a two-component ket block."""

    gamma_cart_ket: jax.Array
    lambda_cart_ket: jax.Array
    third_cart_ket: jax.Array | None = None


ICL_STRAIGHT_GAUGE_PATH = "icl_straight_segment_v1"


@dataclass(frozen=True)
class ICLVNLTransferJet:
    r"""Straight-segment VNL photon jet for ``bra k-q, ket k``.

    All arrays are unscaled Pauli-Hamiltonian derivatives.  With
    ``V_a = d V_NL(k) / d k_a`` and Cartesian transfer ``q``, the
    Ismail-Beigi--Chang--Louie straight Wilson segment gives

    ``Gamma_a(k,q) = V_a - q_b V_ab/2 + O(q^2)``.

    Thus the transfer gradient is not a second current construction: it is
    exactly minus one half of the incumbent uniform contact.  The uniform
    current and contact fields remain byte-for-byte the arrays returned by
    :class:`VNLGaugeKetDerivatives`.  ``d2gamma_dq2_cart_ket`` is present only
    for a setup carrying the separately priced physical third projector
    derivative; differentiating the linear table interpolant is never used as
    a substitute.
    """

    gamma0_cart_ket: jax.Array
    dgamma_dq_cart_ket: jax.Array
    lambda0_cart_ket: jax.Array
    d2gamma_dq2_cart_ket: jax.Array | None = None


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
    compute_contact: bool = False,
    compute_transfer_q2: bool = False,
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
    compute_contact : bool
        Opt in to the extra analytic ``G''`` radial table needed by the
        uniform nonlocal contact.  False by default so existing Hamiltonian,
        NSCF, and dipole setup pays no l+2 Bessel compilation/pass and no
        uniform-gauge content-fingerprint pass.
    compute_transfer_q2 : bool
        Opt in to the analytic ``G'''`` table and l+3 Bessel family needed by
        the ICL straight-segment second transfer derivative.  This implies
        ``compute_contact`` because the q2 jet includes the same q0 contact.
        False leaves the uniform/current compile family unchanged.
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
    compute_contact = bool(compute_contact or compute_transfer_q2)
    tables = build_all_tables(
        species_list, q_max, n_q,
        second_derivatives=compute_contact,
        third_derivatives=bool(compute_transfer_q2))
    species_natoms, species_tau, _ = build_atom_species_map(wfn, species_list)
    q_grid = tables["q"]
    dq = tables["dq"]

    # Build channels and G_l/G'_l tables from species projectors
    channels: list[ChannelMeta] = []
    G_rows: list[np.ndarray] = []
    Gp_rows: list[np.ndarray] = []
    Gpp_rows: list[np.ndarray] = []
    Gppp_rows: list[np.ndarray] = []
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
                Gpp_vals = (tables["second_deriv_tables"][isp][ip]
                            if compute_contact else None)
                Gppp_vals = (tables["third_deriv_tables"][isp][ip]
                             if compute_transfer_q2 else None)
                if l == 0:
                    G_vals = F_vals.copy()
                    Gp_vals = -H_vals
                else:
                    G_vals = np.empty(n_q, dtype=np.float64)
                    G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
                    G_vals[0] = tables["reduced_origins"][isp][ip]
                    Gp_vals = np.zeros(n_q, dtype=np.float64)
                    Gp_vals[1:] = -H_vals[1:] / q_grid[1:] ** l
                    # q=0 limit of dG_l/dq is 0 (j_{l+1}(qr) ~ q^{l+1}, so
                    # H_{l+1}/q^l → 0 as q → 0).
                G_rows.append(G_vals)
                Gp_rows.append(Gp_vals)
                if compute_contact:
                    Gpp_rows.append(Gpp_vals)
                if compute_transfer_q2:
                    Gppp_rows.append(Gppp_vals)

            channels.append(ChannelMeta(
                l=l, nbeta=nbeta, msize=msize, R=R,
                tau=tau, E=E_np, beta_table_start=beta_idx,
                natoms=natoms,
            ))
            beta_idx += nbeta

    G_table_np = (np.stack(G_rows).astype(np.float64, copy=False)
                  if G_rows else np.zeros((0, n_q), dtype=np.float64))
    Gp_table_np = (np.stack(Gp_rows).astype(np.float64, copy=False)
                   if Gp_rows else np.zeros((0, n_q), dtype=np.float64))
    Gpp_table_np = (
        (np.stack(Gpp_rows).astype(np.float64, copy=False)
         if Gpp_rows else np.zeros((0, n_q), dtype=np.float64))
        if compute_contact else None)
    G_table = jnp.asarray(G_table_np, dtype=jnp.float64)
    Gp_table = jnp.asarray(Gp_table_np, dtype=jnp.float64)
    Gpp_table = (None if Gpp_table_np is None
                 else jnp.asarray(Gpp_table_np, dtype=jnp.float64))
    Gppp_table_np = (
        (np.stack(Gppp_rows).astype(np.float64, copy=False)
         if Gppp_rows else np.zeros((0, n_q), dtype=np.float64))
        if compute_transfer_q2 else None)
    Gppp_table = (None if Gppp_table_np is None
                  else jnp.asarray(Gppp_table_np, dtype=jnp.float64))
    total_R = sum(ch.R * ch.natoms for ch in channels)
    l_max = max((ch.l for ch in channels), default=0)

    # ── Pre-flatten per-row metadata for vectorized Z assembly ──
    # Each row of Z corresponds to one (atom, beta, m) combination.
    # Flatten ALL channels into (total_R,) index arrays.
    row_beta_idx = []
    row_l = []
    row_m = []
    row_tau = []
    coupled_row_blocks = []
    row_offset = 0

    for channel_index, ch in enumerate(channels):
        for a in range(ch.natoms):
            block_start = row_offset
            for ib in range(ch.nbeta):
                for im in range(ch.msize):
                    row_beta_idx.append(ch.beta_table_start + ib)
                    row_l.append(ch.l)
                    row_m.append(im)
                    row_tau.append(ch.tau[a])
                    row_offset += 1
            coupled_row_blocks.append(
                (block_start, row_offset, channel_index))

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

    uniform_gauge_fingerprint = ""
    if compute_contact:
        # Authenticate the exact host values before device transfer.  This is
        # the single PP/SOC/radial identity consumed by the uniform current,
        # contact, and Hall transaction; ordinary VNL/dipole setup neither
        # has the G'' capability nor pays this O(table-size) pass.
        import hashlib
        from common.parallel_transport import fingerprint_update_value

        digest = hashlib.sha256()
        digest.update(b"lorrax.vnl_uniform_gauge/v1\0")

        for label, value in (
            ("B_cart", B),
            ("cell_volume", np.float64(cell_volume)),
            ("grid", np.asarray((dq, n_q, q_max), dtype=np.float64)),
            ("prefactor", np.float64(prefactor)),
            ("nspinor_soc", np.asarray(
                (int(nspinor), int(bool(soc_resolved))), dtype=np.int64)),
            ("G", G_table_np),
            ("Gp", Gp_table_np),
            ("Gpp", Gpp_table_np),
            ("row_beta", np.asarray(row_beta_idx, dtype=np.int32)),
            ("row_l", np.asarray(row_l, dtype=np.int32)),
            ("row_m", np.asarray(row_m, dtype=np.int32)),
            ("row_tau", row_tau_np),
            ("coupled_blocks", np.asarray(coupled_row_blocks, dtype=np.int64)),
            ("E_super", E_super),
        ):
            fingerprint_update_value(digest, label, value)
        if compute_transfer_q2:
            # Preserve the incumbent contact-only identity exactly.  The
            # additional radial capability joins the identity only when it
            # can affect the finite-transfer jet.
            fingerprint_update_value(digest, "Gppp", Gppp_table_np)
        uniform_gauge_fingerprint = "sha256:" + digest.hexdigest()

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
        coupled_row_blocks=tuple(coupled_row_blocks),
        Gpp_table=Gpp_table,
        Gppp_table=Gppp_table,
        uniform_gauge_fingerprint=uniform_gauge_fingerprint,
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


@jax.custom_jvp
def _interp_with_two_derivs(q, dq, table, deriv_table, second_deriv_table):
    """Table interpolation carrying two physical derivative levels."""
    return _table_interp(q, dq, table)


@_interp_with_two_derivs.defjvp
def _interp_with_two_derivs_jvp(primals, tangents):
    q, dq, table, deriv_table, second_deriv_table = primals
    q_dot, _, _, _, _ = tangents
    value = _table_interp(q, dq, table)
    slope = _interp_with_deriv(
        q, dq, deriv_table, second_deriv_table)
    return value, slope * q_dot


def _reduced_radial_values_on_cart(K_cart, dq, table):
    """Exact-origin reduced radial values shared by H and gauge vertices."""
    q2 = jnp.sum(K_cart * K_cart, axis=1)
    q_safe = jnp.sqrt(jnp.where(q2 > 0.0, q2, 1.0))
    q = jnp.where(q2 > 0.0, q_safe, 0.0)
    return _table_interp(q, dq, table)


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
    return _build_vnl_kdata_core(
        np.asarray(kvec, dtype=float), np.asarray(Gk_int, dtype=int),
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
    B, dq, G_table, prefactor,
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
    K_crys = Gk_int.astype(jnp.float64) + kvec[None, :]
    K_cart = K_crys @ B
    G_all = _reduced_radial_values_on_cart(
        K_cart, dq, G_table)                                  # (total_nbeta, nG)
    return _assemble_projector_rows(
        K_crys, K_cart, G_all, prefactor,
        row_beta_idx, row_l, row_m, row_tau, l_max=l_max)


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
            setup.G_table,
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
    q2 = jnp.sum(K_cart ** 2, axis=1)
    q_safe = jnp.sqrt(jnp.where(q2 > 0.0, q2, 1.0))
    q = jnp.where(q2 > 0.0, q_safe, 0.0)

    G_all = _reduced_radial_values_on_cart(
        K_cart, setup.dq, setup.G_table)
    S_all = all_solid_harmonics(K_cart, l_max=setup.l_max)

    G_r = G_all[setup.row_beta_idx]
    S_r = S_all[setup.row_l, setup.row_m]
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ setup.row_tau.T)).T
    c_il_r = setup.prefactor * (1j) ** setup.row_l

    Z = c_il_r[:, None] * G_r * S_r * phase_r             # (total_R, nG)

    # dZ for velocity (optional — still uses per-channel JVP, TODO: vectorize)
    dZ_j = None
    if compute_dZ:
        K_over_q = jnp.where(
            q2[:, None] > 0.0, K_cart / q_safe[:, None], 0.0)
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
# Uniform-gauge derivatives — canonical rows, bounded private coefficients
# ---------------------------------------------------------------------------

_FINITE_Q_GATE = "EM-VERTEX-FINITE-Q-WILSON"


def require_uniform_gauge_transfer(
    q_cart_bohr_inv=(0.0, 0.0, 0.0), *, caller: str,
) -> None:
    """One fail-closed boundary for the still-unbound finite-q VNL path."""
    q = np.asarray(q_cart_bohr_inv, dtype=np.float64)
    if q.shape != (3,):
        raise ValueError(
            f"{caller}: q_cart_bohr_inv must have shape (3,), got {q.shape}")
    if not np.array_equal(q, np.zeros(3, dtype=np.float64)):
        raise NotImplementedError(
            f"GATE {_FINITE_Q_GATE}: got q_cart_bohr_inv={q.tolist()}; "
            "only exact uniform q=0 is bound. A finite-q nonlocal "
            "pseudopotential vertex requires the repository-selected "
            "Wilson-line/path prescription.")


@jax.custom_jvp
def _interp_reduced_on_cart(
    K_cart, dq, table, dtable, d2table, d3table,
):
    """Exact-origin reduced radial table with physical Cartesian JVPs."""
    return _reduced_radial_values_on_cart(K_cart, dq, table)


@_interp_reduced_on_cart.defjvp
def _interp_reduced_on_cart_jvp(primals, tangents):
    K_cart, dq, table, dtable, d2table, d3table = primals
    dK_cart, _, _, _, _, _ = tangents
    q2 = jnp.sum(K_cart * K_cart, axis=1)
    q_safe = jnp.sqrt(jnp.where(q2 > 0.0, q2, 1.0))
    q = jnp.where(q2 > 0.0, q_safe, 0.0)
    value = _reduced_radial_values_on_cart(K_cart, dq, table)
    radial_prime = _interp_with_two_derivs(
        q, dq, dtable, d2table, d3table)
    # The primal is G''.  Its custom JVP is the physical G''' table when a
    # third Cartesian derivative is requested; contact-only traces consume
    # only the primal and therefore do not create a third-derivative family.
    radial_second = _interp_with_deriv(q, dq, d2table, d3table)
    radial_prime_over_q = jnp.where(
        q2[None, :] > 0.0,
        radial_prime / q_safe[None, :],
        radial_second)
    tangent = radial_prime_over_q * jnp.sum(
        K_cart * dK_cart, axis=1)[None, :]
    return value, tangent


def _assemble_projector_rows(
    K_crys, K_cart, G_all, prefactor,
    row_beta_idx, row_l, row_m, row_tau, *, l_max,
):
    """The single flattened production projector-row spelling."""
    from psp.radial.solid_harmonics import all_solid_harmonics

    S_all = all_solid_harmonics(K_cart, l_max=l_max)
    G_r = G_all[row_beta_idx]
    S_r = S_all[row_l, row_m]
    phase_r = jnp.exp(-2j * jnp.pi * (K_crys @ row_tau.T)).T
    c_il_r = prefactor * (1j) ** row_l
    return c_il_r[:, None] * G_r * S_r * phase_r


def _assemble_uniform_projector_rows(
    k_crys, G_int, setup: VNLSetup,
    row_beta_idx, row_l, row_m, row_tau,
):
    """Gauge-differentiable view of the canonical flattened row owner."""
    B = jnp.asarray(setup.B, dtype=jnp.float64)
    K_crys = G_int.astype(jnp.float64) + k_crys[None, :]
    K_cart = K_crys @ B
    G_all = _interp_reduced_on_cart(
        K_cart, jnp.asarray(setup.dq), setup.G_table, setup.Gp_table,
        setup.Gpp_table,
        (setup.Gppp_table if setup.Gppp_table is not None
         else jnp.zeros_like(setup.Gpp_table)))
    return _assemble_projector_rows(
        K_crys, K_cart, G_all, jnp.asarray(setup.prefactor),
        row_beta_idx, row_l, row_m, row_tau, l_max=int(setup.l_max))


def _projector_derivatives_cartesian_rows(
    k_crys, G_chunk, setup: VNLSetup,
    row_beta_idx, row_l, row_m, row_tau, g_mask, row_mask,
    *, derivative_order: int = 2,
):
    """Z derivatives through order two or three for one row/G tile."""
    if int(derivative_order) not in (2, 3):
        raise ValueError("derivative_order must be 2 or 3")
    if int(derivative_order) == 3 and setup.Gppp_table is None:
        raise ValueError(
            "GATE EM-VERTEX-VNL-GPPP-MISSING: rebuild VNLSetup with "
            "compute_transfer_q2=True")
    B = jnp.asarray(setup.B, dtype=jnp.float64)
    Binv = jnp.linalg.inv(B)

    def z_at_cart_shift(delta_cart):
        shifted_k = k_crys + delta_cart @ Binv
        return _assemble_uniform_projector_rows(
            shifted_k, G_chunk, setup,
            row_beta_idx, row_l, row_m, row_tau)

    zero = jnp.zeros((3,), dtype=jnp.float64)
    Z = z_at_cart_shift(zero)
    dZ = jnp.moveaxis(jax.jacfwd(z_at_cart_shift)(zero), -1, 0)
    d2_raw = jax.jacfwd(jax.jacfwd(z_at_cart_shift))(zero)
    d2Z = jnp.moveaxis(d2_raw, (-2, -1), (0, 1))
    mask = (
        row_mask[:, None].astype(Z.real.dtype)
        * g_mask[None, :].astype(Z.real.dtype))
    through_second = (
        Z * mask,
        dZ * mask[None, :, :],
        d2Z * mask[None, None, :, :],
    )
    if int(derivative_order) == 2:
        return through_second
    d3_raw = jax.jacfwd(jax.jacfwd(jax.jacfwd(z_at_cart_shift)))(zero)
    d3Z = jnp.moveaxis(d3_raw, (-3, -2, -1), (0, 1, 2))
    return through_second + (
        d3Z * mask[None, None, None, :, :],
    )


def _contract_projector_coefficients(psi_G, Z, dZ, d2Z, E, d3Z=None):
    """Contract one G tile into the private low-rank coefficient carrier."""
    return _VNLProjectorCoefficientBlock(
        c=jnp.einsum("RG,nsG->Rsn", jnp.conj(Z), psi_G, optimize=True),
        dc_cart=jnp.einsum(
            "aRG,nsG->aRsn", jnp.conj(dZ), psi_G, optimize=True),
        d2c_cart=jnp.einsum(
            "abRG,nsG->abRsn", jnp.conj(d2Z), psi_G, optimize=True),
        E=E,
        d3c_cart=(None if d3Z is None else jnp.einsum(
            "abcRG,nsG->abcRsn", jnp.conj(d3Z), psi_G,
            optimize=True)),
    )


def _coupled_projector_coefficients(block):
    """Apply the canonical PP/SOC E block once to c/dc/d2c."""
    E = block.E
    Ec = jnp.einsum("stRQ,Qtn->Rsn", E, block.c, optimize=True)
    Edc = jnp.einsum(
        "stRQ,aQtn->aRsn", E, block.dc_cart, optimize=True)
    Ed2c = jnp.einsum(
        "stRQ,abQtn->abRsn", E, block.d2c_cart, optimize=True)
    return Ec, Edc, Ed2c


def _apply_vnl_gauge_from_coefficients(block, Z, dZ, d2Z):
    """Re-expand current/contact only where a G-space action is requested."""
    Ec, Edc, Ed2c = _coupled_projector_coefficients(block)
    gamma = (
        jnp.einsum("aRG,Rsn->ansG", dZ, Ec, optimize=True)
        + jnp.einsum("RG,aRsn->ansG", Z, Edc, optimize=True))
    contact = (
        jnp.einsum("abRG,Rsn->abnsG", d2Z, Ec, optimize=True)
        + jnp.einsum("aRG,bRsn->abnsG", dZ, Edc, optimize=True)
        + jnp.einsum("bRG,aRsn->abnsG", dZ, Edc, optimize=True)
        + jnp.einsum("RG,abRsn->abnsG", Z, Ed2c, optimize=True))
    return VNLGaugeKetDerivatives(
        gamma_cart_ket=gamma, lambda_cart_ket=contact)


def _apply_vnl_third_from_coefficients(block, Z, dZ, d2Z, d3Z):
    r"""Apply ``d3 V_NL / dk_a dk_b dk_c`` from the same coefficients."""
    if block.d3c_cart is None:
        raise ValueError("third VNL action requires d3 projector coefficients")
    Ec, Edc, Ed2c = _coupled_projector_coefficients(block)
    Ed3c = jnp.einsum(
        "stRQ,abcQtn->abcRsn", block.E, block.d3c_cart, optimize=True)
    return (
        jnp.einsum("abcRG,Rsn->abcnsG", d3Z, Ec, optimize=True)
        + jnp.einsum("abRG,cRsn->abcnsG", d2Z, Edc, optimize=True)
        + jnp.einsum("acRG,bRsn->abcnsG", d2Z, Edc, optimize=True)
        + jnp.einsum("bcRG,aRsn->abcnsG", d2Z, Edc, optimize=True)
        + jnp.einsum("aRG,bcRsn->abcnsG", dZ, Ed2c, optimize=True)
        + jnp.einsum("bRG,acRsn->abcnsG", dZ, Ed2c, optimize=True)
        + jnp.einsum("cRG,abRsn->abcnsG", dZ, Ed2c, optimize=True)
        + jnp.einsum("RG,abcRsn->abcnsG", Z, Ed3c, optimize=True)
    )


def _coupled_projector_row_blocks(setup: VNLSetup, max_rows: int):
    """Validate and return each canonical coupled block exactly once."""
    if int(max_rows) <= 0:
        raise ValueError("projector_row_chunk must be positive")
    blocks = tuple(setup.coupled_row_blocks)
    if int(setup.total_R) and not blocks:
        raise ValueError(
            "GATE EM-VERTEX-VNL-ROW-PROVENANCE: rebuild VNLSetup with "
            "canonical coupled_row_blocks")
    expected = 0
    normalized = []
    for raw_block in blocks:
        if len(raw_block) != 3:
            raise ValueError(
                "GATE EM-VERTEX-VNL-ROW-PROVENANCE: coupled blocks must "
                "carry (start,stop,channel_index)")
        block_start, block_stop, channel_index = raw_block
        start, stop = int(block_start), int(block_stop)
        ich = int(channel_index)
        if start != expected or stop <= start:
            raise ValueError(
                "GATE EM-VERTEX-VNL-ROW-COVERAGE: coupled blocks must "
                f"cover [0,total_R) once; expected {expected}, got "
                f"({start},{stop})")
        if ich < 0 or ich >= len(setup.channels):
            raise ValueError(
                "GATE EM-VERTEX-VNL-E-PROVENANCE: invalid channel index "
                f"{ich} for {len(setup.channels)} channels")
        expected_width = int(setup.channels[ich].R)
        if stop - start != expected_width:
            raise ValueError(
                "GATE EM-VERTEX-VNL-E-PROVENANCE: coupled row width "
                f"{stop - start} does not match ChannelMeta.R="
                f"{expected_width} for channel {ich}")
        if stop - start > int(max_rows):
            raise ValueError(
                "GATE EM-VERTEX-VNL-ROW-CHUNK: a coupled projector block "
                f"has {stop - start} rows, exceeding projector_row_chunk="
                f"{int(max_rows)}; splitting its PP/SOC E block is forbidden")
        normalized.append((start, stop, ich))
        expected = stop
    if expected != int(setup.total_R):
        raise ValueError(
            "GATE EM-VERTEX-VNL-ROW-COVERAGE: coupled blocks end at "
            f"{expected}, total_R={int(setup.total_R)}")

    cursor = 0
    for start, stop, _ in normalized:
        if start != cursor or stop <= start:
            raise AssertionError(
                "internal VNL row packer produced a gap/overlap/duplicate")
        cursor = stop
    if cursor != int(setup.total_R):
        raise AssertionError("internal VNL row packer lost projector rows")
    return tuple(normalized)


def _compact_channel_couplings(setup: VNLSetup, row_width: int):
    """Stack canonical ``ChannelMeta.E`` blocks at one bounded scan shape.

    There is one compact entry per channel, not one ``total_R x total_R``
    matrix and not one duplicate per atom.  ``E_super`` remains the ordinary
    Hamiltonian owner's derived dense representation; this gauge action never
    pads or copies it.
    """
    blocks = []
    for ich, channel in enumerate(setup.channels):
        E = jnp.asarray(channel.E[:2, :2], dtype=jnp.complex128)
        R = int(channel.R)
        if E.shape != (2, 2, R, R):
            raise ValueError(
                "GATE EM-VERTEX-VNL-E-PROVENANCE: ChannelMeta.E for "
                f"channel {ich} has shape {E.shape}, expected (2,2,{R},{R})")
        blocks.append(jnp.pad(
            E, ((0, 0), (0, 0), (0, row_width - R),
                (0, row_width - R))))
    return jnp.stack(blocks, axis=0)


def apply_uniform_vnl_derivatives_to_ket(
    psi_G,
    G_int,
    k_crys,
    setup: VNLSetup,
    g_mask,
    *,
    q_cart_bohr_inv=(0.0, 0.0, 0.0),
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
    compute_third: bool = False,
) -> VNLGaugeKetDerivatives:
    r"""Apply uniform VNL Gamma/Lambda with bounded row and G carriers.

    ``psi_G`` must be the explicit two-component large-component block
    ``Psi_L=(band,2,G)``. Four-component input refuses; the named bispinor
    owner must slice once before entering this Pauli pseudopotential API.

    One fixed-shape outer ``lax.scan`` traverses packed complete E blocks.
    Two inner G scans first accumulate private ``c/dc/d2c`` and then
    re-expand the action. No full-G Z/dZ/d2Z or band-square matrix exists.
    ``compute_third=True`` extends those same scans with ``d3c/d3Z`` and a
    third Hamiltonian action for the ICL q2 jet.  It requires the separately
    priced setup capability and does not alter the default compile family.
    """
    require_uniform_gauge_transfer(
        q_cart_bohr_inv, caller="apply_uniform_vnl_derivatives_to_ket")
    psi = jnp.asarray(psi_G)
    G = jnp.asarray(G_int, dtype=jnp.int32)
    mask = jnp.asarray(g_mask, dtype=jnp.float64)
    if psi.ndim != 3 or int(psi.shape[1]) != 2:
        raise ValueError(
            "GATE EM-VERTEX-LARGE-COMPONENTS: expected explicit Psi_L "
            f"with shape (band,2,G), got {tuple(psi.shape)}")
    if int(setup.nspinor) != 2:
        raise ValueError(
            "GATE EM-VERTEX-PAULI-VNL: VNLSetup.nspinor must be 2, got "
            f"{int(setup.nspinor)}")
    if setup.Gpp_table is None:
        raise ValueError(
            "GATE EM-VERTEX-VNL-GPP-MISSING: rebuild VNLSetup with "
            "compute_contact=True")
    if bool(compute_third) and setup.Gppp_table is None:
        raise ValueError(
            "GATE EM-VERTEX-VNL-GPPP-MISSING: rebuild VNLSetup with "
            "compute_transfer_q2=True")
    if (setup.row_beta_idx is None or setup.row_l is None
            or setup.row_m is None
            or setup.row_tau is None):
        raise ValueError(
            "GATE EM-VERTEX-VNL-SETUP: incomplete canonical VNLSetup")
    if G.shape != (psi.shape[-1], 3) or mask.shape != (psi.shape[-1],):
        raise ValueError(
            "paired G/mask/Psi_L mismatch: got "
            f"G={G.shape}, mask={mask.shape}, psi={psi.shape}")
    if int(g_chunk) <= 0:
        raise ValueError("g_chunk must be positive")

    nband, nG = int(psi.shape[0]), int(psi.shape[-1])
    gstep = int(g_chunk)
    ncarrier = ((nG + gstep - 1) // gstep) * gstep
    gpad = ncarrier - nG
    psi_pad = jnp.pad(psi, ((0, 0), (0, 0), (0, gpad)))
    G_pad = jnp.pad(G, ((0, gpad), (0, 0)))
    mask_pad = jnp.pad(mask, (0, gpad))
    nchunk = ncarrier // gstep
    psi_chunks = jnp.moveaxis(
        psi_pad.reshape(nband, 2, nchunk, gstep), 2, 0)
    G_chunks = G_pad.reshape(nchunk, gstep, 3)
    mask_chunks = mask_pad.reshape(nchunk, gstep)

    gamma_zero = jnp.zeros((3, nband, 2, ncarrier), dtype=psi.dtype)
    contact_zero = jnp.zeros(
        (3, 3, nband, 2, ncarrier), dtype=psi.dtype)
    third_zero = (jnp.zeros(
        (3, 3, 3, nband, 2, ncarrier), dtype=psi.dtype)
        if bool(compute_third) else None)
    row_blocks = _coupled_projector_row_blocks(
        setup, int(projector_row_chunk))
    if not row_blocks:
        return VNLGaugeKetDerivatives(
            gamma_cart_ket=gamma_zero[..., :nG],
            lambda_cart_ket=contact_zero[..., :nG],
            third_cart_ket=(None if third_zero is None
                            else third_zero[..., :nG]))

    row_width = max(stop - start for start, stop, _ in row_blocks)
    row_starts = jnp.asarray(
        [start for start, _, _ in row_blocks], jnp.int32)
    row_lengths = jnp.asarray(
        [stop - start for start, stop, _ in row_blocks], jnp.int32)
    row_channels = jnp.asarray(
        [channel for _, _, channel in row_blocks], jnp.int32)
    row_beta_padded = jnp.pad(setup.row_beta_idx, (0, row_width))
    row_l_padded = jnp.pad(setup.row_l, (0, row_width))
    row_m_padded = jnp.pad(setup.row_m, (0, row_width))
    row_tau_padded = jnp.pad(setup.row_tau, ((0, row_width), (0, 0)))
    channel_E = _compact_channel_couplings(setup, row_width)

    def row_pass(total, row_spec):
        row_start, row_length, row_channel = row_spec
        row_beta = jax.lax.dynamic_slice_in_dim(
            row_beta_padded, row_start, row_width, axis=0)
        row_l = jax.lax.dynamic_slice_in_dim(
            row_l_padded, row_start, row_width, axis=0)
        row_m = jax.lax.dynamic_slice_in_dim(
            row_m_padded, row_start, row_width, axis=0)
        row_tau = jax.lax.dynamic_slice_in_dim(
            row_tau_padded, row_start, row_width, axis=0)
        row_mask = jnp.arange(row_width, dtype=jnp.int32) < row_length
        E_block = channel_E[row_channel]
        E_block = E_block * (
            row_mask[None, None, :, None]
            * row_mask[None, None, None, :]).astype(E_block.dtype)
        coeff_zero = (
            jnp.zeros((row_width, 2, nband), dtype=psi.dtype),
            jnp.zeros((3, row_width, 2, nband), dtype=psi.dtype),
            jnp.zeros((3, 3, row_width, 2, nband), dtype=psi.dtype),
        )
        if bool(compute_third):
            coeff_zero = coeff_zero + (jnp.zeros(
                (3, 3, 3, row_width, 2, nband), dtype=psi.dtype),)

        def coefficient_pass(carry, xs):
            psi_part, G_part, mask_part = xs
            derivatives = _projector_derivatives_cartesian_rows(
                k_crys, G_part, setup, row_beta, row_l, row_m, row_tau,
                mask_part, row_mask,
                derivative_order=(3 if bool(compute_third) else 2))
            Z, dZ, d2Z = derivatives[:3]
            d3Z = derivatives[3] if bool(compute_third) else None
            part = _contract_projector_coefficients(
                psi_part, Z, dZ, d2Z, E_block, d3Z=d3Z)
            updated = (
                carry[0] + part.c,
                carry[1] + part.dc_cart,
                carry[2] + part.d2c_cart,
            )
            if bool(compute_third):
                updated = updated + (carry[3] + part.d3c_cart,)
            return updated, None

        coefficient_arrays, _ = jax.lax.scan(
            coefficient_pass, coeff_zero,
            (psi_chunks, G_chunks, mask_chunks), unroll=1)
        coefficients = _VNLProjectorCoefficientBlock(
            c=coefficient_arrays[0], dc_cart=coefficient_arrays[1],
            d2c_cart=coefficient_arrays[2], E=E_block,
            d3c_cart=(coefficient_arrays[3]
                      if bool(compute_third) else None))

        def expansion_pass(carry, xs):
            G_part, mask_part = xs
            derivatives = _projector_derivatives_cartesian_rows(
                k_crys, G_part, setup, row_beta, row_l, row_m, row_tau,
                mask_part, row_mask,
                derivative_order=(3 if bool(compute_third) else 2))
            Z, dZ, d2Z = derivatives[:3]
            out = _apply_vnl_gauge_from_coefficients(
                coefficients, Z, dZ, d2Z)
            outputs = (out.gamma_cart_ket, out.lambda_cart_ket)
            if bool(compute_third):
                outputs = outputs + (_apply_vnl_third_from_coefficients(
                    coefficients, Z, dZ, d2Z, derivatives[3]),)
            return carry, outputs

        _, expanded_chunks = jax.lax.scan(
            expansion_pass, None, (G_chunks, mask_chunks), unroll=1)
        gamma_chunks, contact_chunks = expanded_chunks[:2]
        gamma_block = jnp.transpose(
            gamma_chunks, (1, 2, 3, 0, 4)).reshape(
                3, nband, 2, ncarrier)
        contact_block = jnp.transpose(
            contact_chunks, (1, 2, 3, 4, 0, 5)).reshape(
                3, 3, nband, 2, ncarrier)
        updated_total = (
            total[0] + gamma_block,
            total[1] + contact_block,
        )
        if bool(compute_third):
            third_chunks = expanded_chunks[2]
            third_block = jnp.transpose(
                third_chunks, (1, 2, 3, 4, 5, 0, 6)).reshape(
                    3, 3, 3, nband, 2, ncarrier)
            updated_total = updated_total + (total[2] + third_block,)
        return updated_total, None

    initial = (gamma_zero, contact_zero)
    if bool(compute_third):
        initial = initial + (third_zero,)
    totals, _ = jax.lax.scan(
        row_pass, initial,
        (row_starts, row_lengths, row_channels), unroll=1)
    gamma, contact = totals[:2]
    return VNLGaugeKetDerivatives(
        gamma_cart_ket=gamma[..., :nG],
        lambda_cart_ket=contact[..., :nG],
        third_cart_ket=(totals[2][..., :nG]
                        if bool(compute_third) else None))


def apply_icl_vnl_transfer_jet_to_ket(
    psi_G,
    G_int,
    k_crys,
    setup: VNLSetup,
    g_mask,
    *,
    projector_row_chunk: int = 64,
    g_chunk: int = 1024,
    include_q2: bool = False,
) -> ICLVNLTransferJet:
    r"""Apply the canonical ICL straight-segment VNL jet at ``q=0``.

    ``k_crys`` is the ket momentum in the repository's transition
    orientation ``<bra,k-q|Gamma(k,q)|ket,k>``.  In the positive raw
    Hamiltonian-vertex convention used by :mod:`common.mtxel_sweep`,

    .. math::

       \Gamma_a^{\rm NL}(k,q)
       = \int_0^1 d\lambda\,\partial_a V_{\rm NL}(k-\lambda q)
       = V_{,a}(k)-\frac{q_b}{2}V_{,ab}(k)
         +\frac{q_bq_c}{6}V_{,abc}(k)+O(q^3).

    The sole bounded projector-coefficient scan already produces both
    derivatives.  This wrapper only binds their finite-transfer meaning; it
    performs no second projector contraction and creates no persistent
    coefficient, G-space, or band-square carrier.
    """
    uniform = apply_uniform_vnl_derivatives_to_ket(
        psi_G,
        G_int,
        k_crys,
        setup,
        g_mask,
        projector_row_chunk=projector_row_chunk,
        g_chunk=g_chunk,
        compute_third=bool(include_q2),
    )
    return ICLVNLTransferJet(
        gamma0_cart_ket=uniform.gamma_cart_ket,
        dgamma_dq_cart_ket=-0.5 * uniform.lambda_cart_ket,
        lambda0_cart_ket=uniform.lambda_cart_ket,
        d2gamma_dq2_cart_ket=(
            uniform.third_cart_ket / 3.0
            if bool(include_q2) else None),
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
    P  = jnp.einsum('RG,nsG->Rsn', jnp.conj(Z),  psi_G, optimize=True)   # ⟨Z|n⟩
    D  = jnp.einsum('stRQ,Qtn->Rsn', E_super, P, optimize=True)
    dP = jnp.einsum('jRG,nsG->jRsn', jnp.conj(dZ), psi_G, optimize=True)
    dD = jnp.einsum('stRQ,jQtn->jRsn', E_super, dP, optimize=True)
    # (∂Z^j) D — first piece in the symmetrized derivative
    t1 = jnp.einsum('jRG,Rsn->jnsG', dZ, D, optimize=True)
    # Z dD — second piece
    t2 = jnp.einsum('RG,jRsn->jnsG', Z,  dD, optimize=True)
    return t1 + t2


@jax.jit
def vnl_velocity_matrix(psi_G, Z, dZ, E_super):
    """⟨m | ∂V_NL/∂K_cart^α | n⟩ matrix elements at one k.  Returns (3, nb, nb).

    Thin bra-contraction wrapper around :func:`apply_vnl_velocity_to_ket`
    so the q=0 dipole path and the finite-q SOS path share the same
    underlying velocity application — no duplicated logic.
    """
    v_ket = apply_vnl_velocity_to_ket(psi_G, Z, dZ, E_super)            # (3, nb, ns, nG)
    return jnp.einsum('msG,jnsG->jmn', jnp.conj(psi_G), v_ket,
                       optimize=True)
