#!/usr/bin/env python3
"""kin+ion computation: T + V_loc + V_NL (+ V_H) for all k → kin_ion.h5.

**By default this writes TWO datasets** and never mixes them:

``kin_ion``   (nk, nb, nb) — ``T + V_loc + V_NL``, **pristine**.
``v_hartree`` (nk, nb, nb) — the exact FFT-grid ⟨mk|V_H|nk⟩, Ry.

Keeping them apart is what makes one file serve every consumer: the same
``kin_ion.h5`` feeds a run that wants the exact V_H (``hartree_source=
stored``) and one that wants the ISDF quadrature (``isdf``), the VH
column of ``sigma_diag.dat`` stops reading 0.000 by construction, and a
QSGW band rotation gets the **full matrix** rather than a diagonal it
cannot transform.  At 12×12 / 80 bands the extra array is ~15 MB.

``--no-hartree`` skips V_H entirely (ionic-only file, ISDF route
mandatory).  ``--fold-hartree`` reproduces the legacy pre-``v_hartree``
format that added V_H *into* ``kin_ion`` and stamped ``has_hartree=True``;
it exists only to regenerate old artifacts bit-for-bit.

Which V_H source to use — read this before trusting an eqp number
-----------------------------------------------------------------
``H₀ = ⟨T+V_ion+V_NL⟩ + ⟨V_H⟩`` is a catastrophic cancellation: for MoS₂
the two terms are −502 eV and +461 eV and their sum is −42 eV, so V_H's
*relative* accuracy is H₀'s accuracy in absolute eV.

* **stored / gspace (exact FFT-grid).**  ⟨mk|V_H|nk⟩ through the same
  local-potential normalisation V_loc takes: analytically exact and
  centroid-count independent.  Pinned to QE's ``kih.dat`` at rms
  1e-4 eV.  Built by ONE distributed kernel
  (:func:`compute_hartree_matrix`) shared by this CLI, the driver's
  ``gspace`` route and the QSGW loop: ρ is partitioned over
  (k, band-chunk) and reduced with a single 1.4 MB psum, the Poisson
  solve is replicated on purpose, and ⟨mk|V_H|nk⟩ is one k-scan with
  that k's bands sharded over every process (``common.mtxel_sweep``),
  so it keeps scaling to P = nb where the previous k-partitioned route
  stopped at P = nk.
* **isdf (V_q[0] tile).**  Costs nothing extra, inherits the GW run's
  full P-way distribution, recomputable in-loop for QSGW.  Measured vs
  the exact route (MoS₂, scorecard §S.5): ~1 % on the occupied manifold
  at 6 centroids per ζ-fit band, 0.11–0.20 % from 9 c/band upward —
  where it **plateaus** (2.5× more centroids, and a 1e-12 rank cutoff
  instead of 1e-8, each buy <5 % relative).

The plateau is why the exact sources exist.  A 0.5 % V_H error is 2 eV
of H₀, and the VBM and CBM errors do not cancel: on the MoS₂ 12×12 at
606 centroids the ISDF route is 0.54 % rms over the occupied manifold
and that is +1.91 eV on the VBM and −2.25 eV on the CBM at K — **4.15 eV
on the band gap**.  50 meV QP energies need V_H to ~1e-4 relative, two
orders past the plateau.

Every physical convention is inherited from the run's own input deck
(``sys_dim`` → Coulomb truncation, ``nval``/``ncond``/``nband`` → band
window, ``bispinor``, the WFN's FFT grid), and the resolved values are
stamped into the output so the generator and the GW run cannot silently
disagree.  ``--sys_dim`` may only *confirm* the deck, never contradict it.

Usage:
  python -m gw.kin_ion_io -i cohsex.in -o kin_ion.h5 [-n NB] [--no-hartree]
"""

import os

# ---- join the distributed world BEFORE anything touches XLA ------------
# THE startup call (runtime module docstring): env defaults, fail-fast
# hook, jax.distributed, CPU fallback, the run's clique-warmed ('x','y')
# mesh, compile cache, rank-0 report.  ``jax.distributed.initialize()``
# refuses to run once the XLA backend is up, and the import graph below
# (``psp.*``) reaches jax; ``runtime`` itself imports no jax, and the call
# is idempotent — which is what makes this safe under
# ``gw.sigma_dispatch``'s LAZY import of this module from inside an
# already-started driver: there it returns the existing stack.
from runtime import initialize_communicator_stack             # noqa: E402
RUNTIME = initialize_communicator_stack()

import argparse                                               # noqa: E402

import numpy as np
import jax.numpy as jnp
import h5py

from common import symmetry_maps, Meta
from common.gvec_fft_box import refuse_padded_gvecs_without_mask
from common.collectives import (barrier, device_count,
                                local_share, process_rank_world,
                                psum_replicate, resolve_mesh)
from common.wfn_transforms import load_kpoint_fftbox_local
import common.timing as timing
from file_io import WfnLoader as WFNReader
from file_io.kin_ion import HARTREE_DATASET
from gw.gw_config import read_lorrax_input as read_cohsex_input
from psp.pseudos import load_pseudopotentials, build_atom_pp_assignments
from psp.dft_operators import padded_gvectors, vnl_matrix_from_kdata
from psp.radial.build_projectors_qe import build_local_ionic_potential_on_G_total
from psp.get_DFT_mtxels import (
    compute_kinetic_k,
    compute_local_V_k,
    build_hartree_potential,
    spin_degeneracy_factor,
    valence_density_from_kpoint,
)
from psp.operator_checks import validate_operator_inputs
import psp.vnl_ops as vnl_ops


def _resolve_against(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


# ===========================================================================
# THE IRREDUCIBLE k-SET, AND WHAT THE UNFOLD ACTUALLY IS
# ===========================================================================
# ``kin_ion`` = T + V_loc + V_NL (and ``v_hartree`` = V_H) are SCALAR
# operators built from the crystal's own potentials, so each commutes with
# every operation of the space group AND with time reversal.  The WFN file
# stores ψ on the IBZ only, and ψ(Sk) is *defined* — by
# ``WfnLoader.load(k='full_bz')``, which is the only producer — as the
# symmetry image of ψ(k_irr).  So the full-BZ table holds only ``nrk``
# distinct matrices and computing it k by k is redundant work.
#
# But the symmetry table is TRS-AUGMENTED, and the two halves do NOT give
# the same rule.  ``sym_mats_k`` is ``concat([S, -S])``: rows below
# ``ntran`` are ordinary spatial operations, rows at or past it carry a
# factor of time reversal.
#
#   UNITARY ROW (sym_idx < ntran).  ψ_n(Sk) = R ψ_n(k_irr) with R unitary
#   (τ phase, umklapp and the spinor SU(2) all inside R), so
#
#       ⟨m,Sk|O|n,Sk⟩ = ⟨m,k_irr|R† O R|n,k_irr⟩ = ⟨m,k_irr|O|n,k_irr⟩
#
#   as a MATRIX, not merely isospectral: the same R acts on bra and ket
#   and O commutes with it, so no rotation, no phase and no
#   degenerate-subspace unitary is left over.  A pure copy.
#
#   ANTIUNITARY ROW (sym_idx >= ntran).  The image is Θ ψ with Θ = (spatial
#   part)∘(time reversal), and Θ is antiunitary: ⟨Θa|Θb⟩ = conj⟨a|b⟩.
#   With [O, Θ] = 0,
#
#       ⟨Θm|O|Θn⟩ = ⟨Θm|Θ(O n)⟩ = conj⟨m|O|n⟩
#
#   — the ELEMENT-WISE conjugate of the parent's matrix, not a copy of it.
#   (For an exactly Hermitian O that equals the transpose, but the
#   conjugate is what the derivation gives and what survives the operand
#   being Hermitian only to round-off.)
#
# Hence the unfold is a gather PLUS a conjugation on the time-reversed
# rows, which is exactly what ``common.symmetry_maps.star_broadcast``
# does; this module calls it rather than re-deriving the rule.
#
# A CAVEAT THAT COSTS NOTHING TO STATE.  The predicate below is
# ``sym_idx_k >= ntran``, i.e. "is this row time-reversed", but the
# quantity that actually governs the conjugation is whether the row and
# its REFERENCE differ in TRS-ness.  The two coincide here because the
# reference is the file's own IBZ slab, read verbatim with no symmetry
# operation applied at all — the representative is untransformed by
# construction.  It would stop coinciding the moment a representative
# were itself a time-reversed row.
#
# AND A WARNING ABOUT VALIDATING THIS.  A cell with inversion symmetry
# needs no time-reversal rows to cover its mesh, so its ``sym_idx_k``
# contains none and it cannot exercise the antiunitary branch at all.
# Agreement measured on such a system says nothing whatever about the
# conjugation; it has to be checked on a deck that actually has TRS rows.
#
# Nor does a within-star spread test say anything: this routine writes the
# star members as exact copies (up to the conjugation), so the spread is
# identically zero whether the unfold is right or wrong — a broadcast that
# wrote ONE matrix everywhere would score just as perfectly.  The checks
# that CAN fail are a regenerated table diffed element-by-element against
# one the full-BZ path produced, and a count of distinct rows.
#
# One more trap for whoever validates this next.  ``star_spread`` is the
# obvious tool to reach for and it does NOT work on a TRS deck: it
# compares each star member against the FIRST ROW of its star, and its
# conjugation predicate is the member's own TRS flag, so whenever that
# first row is itself time-reversed the comparison is off by a conjugate
# and it reports a huge spread on a table that is exactly right.  The
# predicate it wants is ``trs(member) XOR trs(ref)``.  That is not the
# bug fixed here — the broadcast above is safe because ITS reference is
# the untransformed IBZ slab, never a TRS row — but the two look alike
# enough to be worth telling apart.


def broadcast_ibz_to_full_bz(A_irr, sym):
    """``(nrk, …) → (nk_tot, …)`` through the star map, conjugating on TRS.

    Thin adapter over :func:`common.symmetry_maps.star_broadcast` so the
    time-reversal rule has ONE implementation in the tree.  ``star_broadcast``
    orders ``A_irr`` by ``star_select``'s first-occurrence rows; the sweeps
    here run on ``k='ibz'``, whose rows are raw IBZ indices, so the labels
    passed are the identity — which makes its gather ``A[irr_idx_k]``, the
    parent map, with ``conj`` applied on the time-reversed rows.

    ``n_sym_spatial`` is derived from ``sym.sym_mats_k`` (always
    ``2·ntran`` long, both SymMaps branches) rather than from the WFN
    header, because that is the same derivation ``unfold_psi`` uses to
    decide which rows get conjugated when it BUILDS ψ(Sk).  Reading it
    from the header instead would let the producer and the consumer of
    that convention drift apart.

    ``None`` in, ``None`` out: the callers below gather with
    ``owner_only=True``, so the peers hold no table to broadcast.
    """
    if A_irr is None:
        return None
    A = np.asarray(A_irr)
    parent = np.asarray(sym.irr_idx_k, dtype=int)
    if int(parent.max(initial=-1)) >= A.shape[0]:
        raise ValueError(
            f"broadcast_ibz_to_full_bz: irr_idx_k reaches IBZ row "
            f"{int(parent.max())} but the computed table has only "
            f"{A.shape[0]} — the sweep did not run on the IBZ k-set.")
    n_sym_spatial = int(np.asarray(sym.sym_mats_k).shape[0]) // 2
    return symmetry_maps.star_broadcast(
        A, sym.irr_idx_k, sym.sym_idx_k, n_sym_spatial,
        irr_labels=np.arange(A.shape[0], dtype=np.int32))


# ---- artifact provenance ---------------------------------------------------
# A kin_ion.h5 with no stamp of WHAT it was made from is how a stale
# committed fixture survived a month of green tests.  The two stamps below
# are deliberately BOUNDED (see each docstring) and each says its own scope
# in a companion attr, so no reader can over-trust them.

_WFN_CHECKSUM_SCOPE = (
    "md5 over the WFN's /mf_header group ONLY (not the psi coefficients): "
    "every dataset in sorted-path order contributing 'path|dtype|shape' "
    "then its C-order raw bytes")


def _wfn_checksum(wfn_path: str) -> str:
    """Content hash of the WFN's ``mf_header`` — ``'md5:<hex>'``.

    SCOPE IS THE HEADER ALONE, and that is a bound, not an oversight:
    hashing the coefficients means a second full read of the WFN (9 MB on
    the Si fixture, hundreds of GB in production) for a provenance stamp.
    The header pins the lattice, the atoms, the FFT grid, the k-set,
    ngk/ngkmax, the whole G-sphere and every DFT eigenvalue, so every way
    a consumer's WFN can be a *different calculation* shows up here.  What
    it does NOT catch is the same calculation rerun to a different ψ
    gauge.  ``wfn_checksum_scope`` is written beside it so a consumer
    comparing hashes knows exactly which of those two questions it just
    answered.  Never raises: a stamp that can abort a 3-hour generator is
    worse than no stamp.
    """
    import hashlib
    try:
        h = hashlib.md5()
        with h5py.File(wfn_path, "r") as f:
            grp = f["mf_header"]
            names = []
            grp.visit(names.append)
            for name in sorted(names):
                obj = grp.get(name)
                if not isinstance(obj, h5py.Dataset):
                    continue
                arr = np.ascontiguousarray(obj[()])
                h.update(f"{name}|{arr.dtype.str}|{arr.shape}".encode())
                h.update(arr.tobytes())
        return "md5:" + h.hexdigest()
    except Exception as exc:                                  # noqa: BLE001
        return f"unknown:{type(exc).__name__}"


def _generator_commit() -> str:
    """Commit of the SOURCE TREE THIS MODULE RAN FROM — not the cwd's.

    The CLI is normally invoked from a scratch work directory that is not
    a checkout (or is a *different* one), so ``git rev-parse`` in the cwd
    names the wrong tree or nothing at all; ``git -C <src>`` anchors it to
    the file that is actually executing.  Falls back to
    ``'unknown:<reason>'`` rather than an empty string or a fake hash — a
    named reason is information, a blank attr is the failure mode this
    stamp exists to remove.

    PRICED, because a provenance stamp that costs real wall is a stamp
    someone will delete.  MEASURED from inside the shifter container with
    the device stack up (the tree on Lustre), against 0.03 s for the same
    commands from a login shell:

        bare fork of this process        0.181 s   (page tables of a live
                                                    JAX runtime)
        rev-parse --short HEAD           0.519 s
        status --porcelain -uno          0.625 s
        describe --always --dirty        3.583 s   <- rejected

    so this pair is ~1.1 s, once, on rank 0 — three orders below the
    generator run it stamps at any production shape, and it is charged to
    the ``write_h5`` timing section rather than hidden in ``(untimed)``.
    ``describe --dirty`` would have done it in one fork and costs 3× more:
    it walks the tag graph on top of the same worktree refresh.

    The ``-dirty`` suffix is not decoration.  A stamp that reads clean on
    an edited tree is worse than no stamp, because it is the exact claim
    the reader wanted to check.
    """
    import subprocess
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        rev = subprocess.run(
            ["git", "-C", src, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=60)
        if rev.returncode != 0:
            return "unknown:not-a-git-checkout"
        out = rev.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", src, "status", "--porcelain",
             "--untracked-files=no"],
            capture_output=True, text=True, timeout=60)
        if dirty.returncode == 0 and dirty.stdout.strip():
            out += "-dirty"
        return out
    except Exception as exc:                                  # noqa: BLE001
        return f"unknown:{type(exc).__name__}"


def get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, vnl_setup, wfn, g_mask=None,
                  V_H_r=None):
    """Compute T + V_loc + V_NL (+ V_H) for a single k-point.

    Parameters
    ----------
    wfn_k : (nb, nspinor, nx, ny, nz) — wavefunctions in FFT box
    Gk_crys : (nG, 3) int — G-vector indices for this k.  May be the
        k's own ``ngk`` rows or the ``ngkmax``-padded table, in which
        case ``g_mask`` is REQUIRED (pad rows are the FFT-box pad
        sentinel — a valid box index, so an absent mask double-counts
        that component rather than crashing).  Enforced by
        :func:`common.gvec_fft_box.refuse_padded_gvecs_without_mask`,
        which lives beside the routine that BUILDS the pad so the
        detector and the producer share one invariant.
    kvec : (3,) float — k-point in crystal coords
    V_loc_r : (nx, ny, nz) — local ionic potential on FFT grid
    vnl_setup : VNLSetup from vnl_ops.build_vnl_setup (or None to skip V_NL)
    wfn : WFNReader (for bdot, bvec, blat, cell_volume)
    g_mask : (nG,) float or None — 1 on physical G, 0 on pad columns.
    V_H_r : (nx, ny, nz) or None — mean-field Hartree potential on the
        SAME FFT grid as ``V_loc_r``.  Folded in through the identical
        local-potential route, so H₀'s ~500 eV cancellation closes inside
        one exact routine instead of across two numerical schemes.
    """
    Gk_np = np.asarray(Gk_crys, dtype=int)
    if g_mask is None:
        refuse_padded_gvecs_without_mask(
            Gk_np, getattr(wfn, "fft_grid", None),
            where="get_kin_ion_k: Gk_crys")
    bdot_np = np.asarray(wfn.bdot, dtype=float)
    T_k = compute_kinetic_k(wfn_k, Gk_crys, kvec, bdot_np, g_mask=g_mask)
    V_loc_k = compute_local_V_k(
        wfn_k, Gk_crys, V_loc_r, wfn.cell_volume, g_mask=g_mask
    )

    V_NL_k = 0.0
    if vnl_setup is not None:
        # Z is built ON THE SAME (padded) G-list, which is the contract
        # ``_build_vnl_kdata_core`` documents: Z at a pad row is finite
        # (it is evaluated at K = kvec) and the caller must mask before
        # contracting.  Masking ψ_G is sufficient and is what
        # ``vnl_matrix_from_kdata(mask=…)`` does — every contraction in
        # ``vnl_ops.vnl_matrix`` runs through ψ_G at least once.
        kdata = vnl_ops.build_vnl_kdata_from_kvec(
            np.asarray(kvec, dtype=float), Gk_np, vnl_setup)
        V_NL_k = vnl_matrix_from_kdata(wfn_k, Gk_crys, kdata, mask=g_mask)

    H_k = T_k + V_loc_k + V_NL_k
    if V_H_r is not None:
        H_k = H_k + compute_local_V_k(
            wfn_k, Gk_crys, V_H_r, wfn.cell_volume, g_mask=g_mask
        )
    return H_k


# ===========================================================================
# THE EXACT V_H, distributed
# ===========================================================================
# Everything below exists so that ONE implementation of the exact V_H serves
# all three consumers — the ``kin_ion.h5`` generator CLI, the driver's
# ``hartree_source=gspace`` route, and the QSGW loop that will rebuild V_H
# from an updated ρ every SC iteration.  It is mesh-aware from 1×1
# (single-node CLI) up to a production mesh.
#
# The plumbing itself — mesh, work partition, psum, gather, the pipelined
# per-k sweep — is ``common.collectives``; nothing here reaches under it.
#
# COMMUNICATION CONTRACT (this is the whole design, in four lines):
#
#   ρ(r)              partitioned over (k, band-chunk); per-rank partials
#                     combined by **exactly one psum of nx·ny·nz f64**
#                     (1.4 MB at 12×12).  Nothing else is reduced.
#   V_H(r) = Poisson  **REPLICATED BY DESIGN** — zero collectives; see the
#                     step-2 comment inside ``compute_hartree_matrix``.
#   ⟨mk|V_H|nk⟩       ONE k-scan (``common.mtxel_sweep``): k is a trip
#                     count, and THAT k's bands are sharded over every
#                     process.  One reshard per k along 'x' (9.4 MB at
#                     b600/P=64); the output stays sharded
#                     ``P(None,'x','y')`` and is gathered only at the
#                     boundary, by name.
#   ψ                 for ρ: loaded per rank for the (k, band) windows that
#                     rank owns — process-local, no collective implied.
#                     For the matrix elements: the G-SPHERE for all k,
#                     band-sharded and resident (≈19 MB/rank at b600/P=64).
#
# Why the partitions differ between the two sweeps: ρ is a sum over
# (k, band) and both axes are free, so k alone suffices until P > nk and a
# band chunking is layered on.  ⟨mk|V_H|nk⟩ contracts the FULL band window
# against itself at fixed k, so splitting bands means either a reduction or
# a two-sided split — and the two-sided split is what ``mtxel_sweep``
# does: shard the OUTPUT ``H[m_X, n_Y]`` and replicate the contraction
# axis.  The alternative (shard G, psum the partials) needs every rank to
# hold a full (nb, nb) to reduce into, which is the wall being removed.


def rho_work_items(nk: int, nocc: int, world: int) -> list[tuple[int, int, int]]:
    """The ρ sweep's work list: ``(ik, b_lo, b_hi)`` items, k-major.

    One item per k while ``world <= nk`` — i.e. the band axis is left
    whole, which makes the P=1 sweep the *identical* sequence of
    operations the serial implementation performed (one k at a time,
    all occupied bands, in k order) and therefore bit-for-bit its
    result.  Past ``world > nk`` the occupied manifold is cut into
    ``ceil(world/nk)`` contiguous band chunks so ranks beyond the k
    count still get work; ``valence_density_from_kpoint`` sums whatever
    bands it is handed, so no band-index bookkeeping leaks out of here.
    """
    nk = int(nk)
    nocc = int(nocc)
    n_bchunk = max(1, -(-int(world) // max(nk, 1)))
    n_bchunk = max(1, min(n_bchunk, nocc))
    bounds = [(nocc * i // n_bchunk, nocc * (i + 1) // n_bchunk)
              for i in range(n_bchunk)]
    return [(ik, lo, hi) for ik in range(nk) for lo, hi in bounds if hi > lo]


def _load_rotated_occ_fftbox(wfn, meta, ik: int, U_k):
    """The ``nocc`` CURRENT-basis occupied orbitals at k, in the FFT box.

    ``U_k`` is ``(nmix, nocc)``: column ``n`` gives the current
    (QP / mixed) occupied orbital ``n`` as a combination of the first
    ``nmix`` DFT orbitals from the WFN file,
    ``ψ^cur_n = Σ_m U[m, n] ψ^DFT_m``.

    The rotation is applied on the **G-flat** coefficients, before the
    scatter into the FFT box: ``nmix·nocc·ns·ngkmax`` flops instead of
    ``nmix·nocc·ns·N_r`` (a 20× saving at MoS₂ 12×12), and the box is
    only ever materialised at ``nocc`` bands rather than ``nmix``.
    """
    from common.collectives import single_device_mesh
    from common.wfn_transforms import to_box
    nmix = int(np.shape(U_k)[0])
    # Same reasoning as the unrotated leg in build_valence_density_distributed:
    # ask the loader for the extent meta declares, so the small components are
    # lifted rather than zero-filled below.
    psi_g = wfn.load_process_local(bands=(0, nmix), k=[int(ik)],
                                   bispinor=(int(meta.nspinor) == 4))
    ns_have = int(psi_g.shape[2])
    if int(meta.nspinor) > ns_have:
        from common.wfn_transforms import _refuse_spinor_zero_fill
        _refuse_spinor_zero_fill(int(meta.nspinor), ns_have,
                                 origin="kin_ion_io._load_rotated_occ_fftbox")
    U = jnp.asarray(U_k, dtype=psi_g.dtype)
    psi_g = jnp.einsum('mn,kmsg->knsg', U, psi_g, optimize=True)
    box = to_box(psi_g, wfn.box_index(k=[int(ik)]),
                 tuple(int(s) for s in meta.fft_grid),
                 mesh=single_device_mesh())
    return box[0]


def build_valence_density_distributed(wfn, sym, meta, nocc: int, *,
                                      nk: int | None = None,
                                      mesh=None,
                                      psi_rotation=None,
                                      print_fn=print) -> np.ndarray:
    """ρ_v(r) on the ψ FFT box grid — k/band-partitioned, ONE psum.

    Replaces the serial k loop.  Each rank sweeps only the
    ``(k, band-chunk)`` items :func:`rho_work_items` hands it, loading
    ψ process-locally for exactly those windows, and accumulates into
    its own full-grid partial ρ.  ρ is *small* — ``nx·ny·nz`` f64,
    1.4 MB for the MoS₂ 12×12 — so replicating the accumulator costs
    nothing and the combination is a single all-reduce of that size.
    Sharding ρ itself would buy 1.4 MB of memory and cost an
    all-to-all; it is deliberately not done.

    The FFT flops (nk·nocc 3-D FFTs, 1.1e11 at 12×12) divide by P
    exactly, and so does the ψ read.  The pad-band contract is
    irrelevant here because the items carry real band bounds, not
    mesh-rounded ones.

    Mirrors :func:`psp.get_DFT_mtxels.compute_valence_density` — SAME
    per-k quadrature helper, no second copy of the density math — and
    never holds more than one ``(k, band-chunk)`` of ψ, which is what
    keeps the 144-k / 400-band decks inside a node.  The unfolded full
    BZ carries uniform weights ``1/nk_tot`` by construction
    (``SymMaps`` expands the IBZ to the full mesh), so no ``kweights``
    lookup is needed.

    THE QSGW SEAM — ``psi_rotation``
    --------------------------------
    ``None`` (default, and the only thing a one-shot run needs) builds ρ
    from the WFN file's DFT orbitals.

    Pass ``(nk, nmix, nocc)`` and ρ is built instead from the CURRENT
    occupied orbitals ``ψ^cur_n = Σ_m U[k, m, n] ψ^DFT_m`` — i.e. from
    whatever mixed/rotated wavefunctions the SC loop is holding, which
    is what makes an updated-density QSGW iteration possible rather
    than a fixed-mean-field one.  This is the *density-side* twin of
    ``sigma_dispatch``'s ``hartree_basis_rotation``, and the two are
    orthogonal by construction: this one changes WHICH density V_H is
    generated by; that one changes which BASIS the resulting operator
    is expressed in.  The matrix-element sweep still uses the file's
    DFT orbitals, so the kernel keeps returning a DFT-basis operator
    and the existing ``U†·O_DFT·U`` seam still applies unchanged.

    With a rotation supplied the band axis is NOT chunked (the mixing
    couples all ``nmix`` bands, so a band split would need its own
    reduction); the sweep stays k-partitioned, which is full-rate for
    every P ≤ nk.

    Returns the summed ρ(r) as a host array identical on every rank.
    """
    mesh = resolve_mesh(mesh)
    _, world = process_rank_world()
    nk = int(sym.nk_tot if nk is None else nk)
    nx, ny, nz = (int(s) for s in meta.fft_grid)
    rotated = psi_rotation is not None
    items = (rho_work_items(nk, int(nocc), 1) if rotated
             else rho_work_items(nk, int(nocc), world))
    mine = local_share(items)
    f_spin = spin_degeneracy_factor(wfn)
    wk = 1.0 / float(nk)
    print_fn(f"    rho sweep: {len(items)} (k, band-chunk) items over "
             f"P={world} ranks; this rank has {len(mine)}"
             f"{'  [rotated ψ: k-partition only]' if rotated else ''}")
    rho_local = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    for n_done, (ik, b_lo, b_hi) in enumerate(mine):
        if n_done % 32 == 0:
            print_fn(f"    rho: item {n_done + 1}/{len(mine)} "
                     f"(k={ik + 1}/{nk}, bands [{b_lo},{b_hi}))...")
        if rotated:
            psi_k = _load_rotated_occ_fftbox(
                wfn, meta, ik, np.asarray(psi_rotation)[ik])
        else:
            # bispinor from meta, not defaulted: meta.nspinor == 4 IS the
            # bispinor flag (common/meta.py), and omitting it here made the
            # loader return 2 components that the callee then zero-filled to
            # 4 — silently building rho and V_H from large components only.
            # The zero fill now refuses; this is the call that keeps it
            # unreachable, by LIFTING the small components instead.
            psi_k = load_kpoint_fftbox_local(
                wfn, meta, ik, b_hi, b_lo=b_lo,
                bispinor=(int(meta.nspinor) == 4))
        rho_local = rho_local + valence_density_from_kpoint(
            psi_k, nocc=None, weight=wk,
            cell_volume=float(wfn.cell_volume), spin_degeneracy=f_spin,
        )
        del psi_k
    with timing.section("vh_rho_psum"):
        return psum_replicate(np.asarray(rho_local), mesh)


def compute_hartree_matrix(wfn, sym, meta, *, truncation_2d: bool,
                           nb: int, mesh=None,
                           psi_rotation=None,
                           print_fn=print,
                           owner_only: bool = False):
    """The exact FFT-grid ⟨mk|V_H|nk⟩ for all k — **(nk, nb, nb) Ry**.

    SINGLE SOURCE for every exact-V_H consumer: the CLI in this module
    (which stores the result as ``kin_ion.h5``'s ``v_hartree``
    dataset), the driver's ``hartree_source=gspace`` route, and the
    QSGW loop that rebuilds V_H from an updated ρ.  Both stored and
    gspace must produce the same numbers or the two sources would
    disagree, so there is exactly one implementation — and, since this
    revision, exactly one *distributed* implementation.

    Distribution: see the contract block at the head of this section.
    ρ is partitioned over (k, band-chunk) and reduced with one psum;
    the Poisson solve is replicated; ⟨mk|V_H|nk⟩ is ONE k-scan with that
    k's bands sharded over every process (``common.mtxel_sweep``).
    ``mesh`` is the collectives' device mesh — pass the run's own (the
    driver does) or leave it None and one is derived, 1×1 on a single
    device.  P=1 is bit-for-bit the serial result.

    Needs no pseudopotentials: ρ comes from ψ, V_H from the Poisson
    solve, and the matrix element from the same normalisation chain
    (``psp.get_DFT_mtxels.local_potential_scalars``) V_loc takes — the
    two plans call it, so they agree by construction and differ only in
    the reassociation the sharding forces on the G sum.
    ``truncation_2d`` MUST be the run's own convention (deck
    ``sys_dim``); mixing it with V_loc's is a large systematic error
    inside a ~500 eV cancellation.

    Returns the FULL matrix as a host ``numpy`` array replicated on
    every rank, not the diagonal: a QSGW band rotation needs
    ⟨m|V_H|n⟩ off-diagonals to transform H₀ into the QP basis.  A caller
    that wants the SHARDED block instead calls
    :func:`common.mtxel_sweep.sweep_matrix_elements` directly, which is
    what ``gw.sc_iteration.rebuild_hartree_dft_basis`` does.

    ``owner_only=True`` (this module's CLI, whose only consumer is the
    rank-0 h5 write) assembles on rank 0 alone and returns ``None`` on
    every other rank — see :func:`common.mtxel_sweep.blocks_to_host`.
    Replicated consumers (gspace route) keep the default.
    """
    mesh = resolve_mesh(mesh)
    _, world = process_rank_world()
    nocc = int(wfn.nelec)
    nk = int(sym.nk_tot)
    if nocc > nb:
        raise ValueError(
            f"V_H needs the {nocc} occupied bands but only {nb} were requested")

    # ---- 1. ρ(r): (k, band-chunk)-partitioned, one psum ----------------
    # Bootstrap the ρ all-reduce BEFORE the sweep, on a zero array of the
    # exact same shape.  MEASURED on Frontera/Gloo at P=4: the first call
    # to this reduction costs 11.7 s (XLA lowering of the shard_map module
    # + Gloo's communicator handshake for that replica group) and every
    # later call costs milliseconds — ``runtime.nccl_warmup``'s generic
    # psums do NOT cover it, because they lower a different module.  Left
    # inside the sweep it is a P-independent constant that masquerades as
    # 70 % of the ρ phase and destroys the strong-scaling reading.  Doing
    # it here also means a QSGW loop pays it once, on iteration 0.
    if world > 1:
        with timing.section("vh_collective_bootstrap"):
            psum_replicate(
                np.zeros(tuple(int(s) for s in meta.fft_grid),
                         dtype=np.float64), mesh)

    print_fn(f"\nBuilding valence density from {nocc} occupied bands "
             f"(P={world}, {nk} k-points)...")
    f_spin = spin_degeneracy_factor(wfn)
    with timing.section("vh_rho"):
        rho_np = build_valence_density_distributed(
            wfn, sym, meta, nocc, nk=nk, mesh=mesh,
            psi_rotation=psi_rotation, print_fn=print_fn)

    # ---- 2. Poisson: REPLICATED BY DESIGN ------------------------------
    # Two 3-D FFTs on a 1.4 MB array: 3.1e7 flop against the sweep's
    # 1.2e12, i.e. 3e-5 of the work.  Sharding it would replace a free
    # duplicated computation with an all-to-all (a distributed 3-D FFT
    # is two transposes) and buy back 1.4 MB of memory per rank.  Every
    # rank therefore solves the same Poisson equation from the same
    # replicated ρ and gets bit-identical V_H(r) — which is also what
    # makes the k-partitioned matrix-element sweep below trivially
    # rank-invariant.  Revisit only above N_r ≈ 1e8.
    V_H_r = build_hartree_potential(
        jnp.asarray(rho_np), wfn,
        truncation_2d=bool(truncation_2d),
        expected_electrons=f_spin * float(nocc),
        print_fn=print_fn,
    )
    # V_H(r) is replicated (step 2), so it is closed over by the operator
    # as a constant; nothing about it is per-k.
    V_H_r = jnp.asarray(np.asarray(V_H_r, dtype=np.float64),
                        dtype=jnp.float64)
    del rho_np

    # ---- 3. ⟨mk|V_H|nk⟩: ONE k-scan over the IRREDUCIBLE k ----
    #
    # THE k-SET IS THE IBZ, and the full-BZ table is the star broadcast of
    # it — see "THE IRREDUCIBLE k-SET" at the head of this module for the
    # derivation, including the conjugation the time-reversed rows need.
    # V_H is a local scalar potential, so it commutes with the space group
    # and with time reversal like any other term here.
    #
    # ρ ABOVE IS UNTOUCHED and stays a full-BZ sum: it is a sum over the
    # zone, not a per-k operator, and rebuilding it from the IBZ with
    # weights is a different (unsymmetrised) quadrature.  V_H is then a
    # local scalar potential on the FFT grid like V_loc, so it takes the
    # same symmetry argument as the kin+ion sweep and no other.
    #
    # ``tests/multi_device/mtxel_callsite_gate.py`` check 5 compares this
    # function against a full-BZ per-k local plan at 1e-12 relative and is
    # the one gate in the tree that would notice if it did not.
    #
    # Replaces the k-partitioned ``gather_k_blocks`` sweep.  That sweep
    # built each k's FULL-BAND FFT box on one rank (1.77 GB at b600
    # bispinor) and could not use more than ``nk`` ranks at all — each
    # rank took a whole k, so its wall was one full-band k however large
    # P was.  ``sweep_matrix_elements`` scans k one at a time with that
    # k's bands sharded over every process, so ``nk`` is a trip count and
    # parallel efficiency is ``nb_logical/nb_padded``.  Measured
    # b600-class at P=64, worst rank: 4.975 s / 10.83 GiB before,
    # 2.162 s / 8.21 GiB after (jobs 7888877, 7888907).  At P = nk it is
    # ~1.45× SLOWER — the per-k reshard is pure overhead once the old
    # plan already fills the machine — which is expected and is the
    # documented crossover, not a regression.
    #
    # Fixed-shape G (owner decision D10, 2026-07-30): ``padded_gvectors``
    # hands over the loader's OWN ``(nk, ngkmax, 3)`` table, so the scan
    # body lowers ONCE for the whole k range.  The pad columns are made
    # inert by ``g_mask`` — mandatory, not tidy: pad rows hold ``(0,0,0)``,
    # a valid box index that ALIASES physical Γ, so a forgotten mask is
    # silently wrong inside H₀'s ~500 eV cancellation rather than loud.
    #
    # ψ arrives on the G-SPHERE, band-sharded, resident for all k: 1.2 GB
    # globally at b600 but ≈19 MB/rank at P=64, against the 1.77 GB ONE
    # box the route above materialised per k.  ``band_sphere_spec`` is the
    # single definition of that layout, shared with the loader, so no
    # reshard is inserted between the read and the scan.
    from common.mtxel_sweep import (SweepGeometry, band_sphere_spec,
                                    blocks_to_host,
                                    local_potential_operator,
                                    sweep_matrix_elements)
    nk_irr = int(wfn.nkpts)
    gtab = padded_gvectors(wfn, k="ibz")
    psi_G = wfn.load(bands=(0, nb), k="ibz", sharding=band_sphere_spec())
    geom = SweepGeometry(mesh=mesh, fft_grid=meta.fft_grid,
                         ngkmax=int(psi_G.shape[3]), nb=nb,
                         ns=int(psi_G.shape[2]), nk=nk_irr,
                         cell_volume=float(wfn.cell_volume))
    print_fn(f"\n⟨mk|V_H|nk⟩: one k-scan over {nk_irr} IRREDUCIBLE k-points "
             f"(broadcast to {nk} full-BZ k), "
             f"{geom.nb} bands sharded over P={world}...")
    with timing.section("vh_matrix"):
        H_vh = sweep_matrix_elements(
            psi_G, operator=local_potential_operator(geom, V_H_r), geom=geom,
            gvecs=gtab.gvecs, gmask=gtab.mask,
            box_index=wfn.box_index(k="ibz"),
            # ``wfn.kpoints``, NOT ``sym.unfolded_kpts``: ψ and the G-list
            # are the raw IBZ slab, and the only k that belongs with them
            # is the file's own — the pairing is self-consistent by
            # definition, whatever wrapping convention the unfold uses.
            kvecs=np.asarray(wfn.kpoints, dtype=np.float64))
        # THE BOUNDARY, stated rather than implied.  Both consumers of this
        # function want a host ``(nk_tot, nb, nb)`` — the FULL BZ, which is
        # why the star broadcast happens here and not at either call site:
        # the CLI writes it with serial h5py on rank 0
        # (``owner_only=True``), and ``sigma_dispatch``'s gspace route
        # hands it to ``replicate_to_mesh`` as a replicated global operand.
        # Neither can take a sharded block or an IBZ-shaped one, so both
        # the sharding and the k-set are undone HERE and named.  The QSGW
        # consumer that CAN stay sharded
        # (``sc_iteration.rebuild_hartree_dft_basis``) calls
        # ``sweep_matrix_elements`` directly and never reaches this line.
        return broadcast_ibz_to_full_bz(
            blocks_to_host(H_vh, nb=nb, owner_only=owner_only), sym)


def build_argparser() -> argparse.ArgumentParser:
    """The CLI's argument parser.

    Split out of :func:`main` so the V_H defaults are pinnable by a unit
    test without running the generator.
    """
    argp = argparse.ArgumentParser(description="Chunked kin+ion computation")
    argp.add_argument("-i", "--input", required=True, help="cohsex / GW input file")
    argp.add_argument("-o", "--output", default=None, help="output HDF5 (default: kin_ion.h5)")
    argp.add_argument("-n", "--nb", type=int, default=None, help="number of bands")
    argp.add_argument("--sys_dim", type=int, default=None,
                      help="system dimensionality: 0, 2, or 3.  Must AGREE with "
                           "the input file when the file specifies it.")
    argp.add_argument("--pseudo_dir", default=None,
                      help="directory containing *.upf files (default: input file dir)")
    argp.add_argument("--hartree", dest="hartree", action="store_true", default=True,
                      help="DEFAULT: also compute the exact FFT-grid ⟨mk|V_H|nk⟩ and "
                           "store it as the SEPARATE 'v_hartree' array (kin_ion "
                           "itself stays pristine T+V_loc+V_NL)")
    argp.add_argument("--no-hartree", dest="hartree", action="store_false",
                      help="skip V_H entirely: the file carries T+V_loc+V_NL only and "
                           "the GW run must supply V_H from the ISDF V_q[0] tile")
    argp.add_argument("--fold-hartree", dest="fold_hartree", action="store_true",
                      default=False,
                      help="LEGACY/compat: add V_H INTO the kin_ion values and stamp "
                           "has_hartree=True (the pre-'v_hartree' format).  Only for "
                           "reproducing old artifacts — the stored-array default is "
                           "strictly better (kin_ion stays reusable, QSGW gets the "
                           "full matrix, and the VH column stops reading 0.000).")
    return argp


def main(argv=None):
    import time as _time
    _t_main = _time.perf_counter()
    args = build_argparser().parse_args(argv)

    timing.reset()

    # (the distributed init happens at module import — see the header)
    rank, world = process_rank_world()
    print0 = print if rank == 0 else (lambda *a, **k: None)
    print0(f"== kin_ion_io ==  (P={world} process"
           f"{'es' if world != 1 else ''}, {device_count()} devices)")

    # Persistent compile cache — same call, same position (after the
    # distributed init, before any jit) as gw.gw_jax:145.  Without it this
    # CLI re-lowers every kernel on every invocation, which at nb=256 is
    # most of the wall: 40 s of recorded sections inside a 124 s run.  The
    # per-k kernels here recompile once per DISTINCT ngk (see the COMPILE
    # NOTE in compute_hartree_matrix), so the miss count is the IBZ k-count,
    # not one.
    from common.jax_compile_cache import ensure_jax_compile_cache
    ensure_jax_compile_cache()

    # ---- the multi-rank contract: DISTRIBUTE, then write once -----------
    # This used to refuse ``srun -n P`` outright, because every rank redid
    # the whole calculation on device 0 and then overwrote the same
    # ``kin_ion.h5`` at rc=0.  Both halves of that are now fixed: the k
    # sweeps below are distributed (ρ over (k, band-chunk); the matrix
    # elements over bands, one k at a time, ``common.mtxel_sweep``), and
    # the write is coordinated — rank 0 alone opens the file, after the
    # boundary gather that has already given it every k.
    #
    # What is still fatal is the *other* multi-rank failure mode: a
    # launcher that starts P tasks while ``jax.distributed`` sees one
    # process each.  Then there is no world to partition over, every rank
    # computes the full result, and every rank believes it is rank 0 — the
    # original clobber, with none of the safety.  Detect it by comparing
    # what the launcher advertises against what JAX joined.
    _nproc_env = int(os.environ.get(
        "JAX_PROCESS_COUNT",
        os.environ.get("JAX_NUM_PROCESSES",
                       os.environ.get("SLURM_NTASKS", "1"))))
    if _nproc_env > 1 and world <= 1:
        raise SystemExit(
            f"the launcher advertises {_nproc_env} tasks (SLURM_NTASKS / "
            f"JAX_PROCESS_COUNT) but jax.distributed joined a world of "
            f"{world}.  Every task would redo the whole calculation and "
            f"overwrite the same output file.  Fix the distributed launch "
            f"(JAX_COORDINATOR_ADDRESS must be reachable from every "
            f"task) or run `-n 1`.")

    input_dir = os.path.dirname(os.path.abspath(args.input))

    # ---- parse input: the deck is the single source of truth ----
    # Everything physical (Coulomb truncation, band window, spinor
    # treatment, FFT grid) is inherited from the same file the GW run
    # reads.  A CLI flag may only confirm the deck, never silently
    # override it, so the generator and the run cannot disagree.
    params = read_cohsex_input(args.input)
    wfn_path = _resolve_against(params.get("wfn_file", "WFN.h5"), input_dir)

    sys_dim_file = params.get("sys_dim")
    if args.sys_dim is not None and sys_dim_file is not None and (
        int(args.sys_dim) != int(sys_dim_file)
    ):
        raise SystemExit(
            f"--sys_dim {args.sys_dim} contradicts sys_dim={int(sys_dim_file)} in "
            f"{os.path.basename(args.input)}.  kin_ion.h5 carries the Coulomb "
            "truncation convention for the whole run — fix the deck instead."
        )
    sys_dim = int(args.sys_dim if args.sys_dim is not None
                  else (sys_dim_file if sys_dim_file is not None else 3))

    print0(f"Loading WFN: {os.path.basename(wfn_path)}")
    # The module-top ``initialize_communicator_stack()`` already built and
    # clique-warmed the run's mesh; handing it to the loader is what lets
    # ``backend=auto`` pick the collective phdf5 read at P>1 instead of
    # the per-rank eager h5py read (scorecard BD.2 — htransform already
    # did this, dipole/kin-ion/kmeans did not).
    mesh_xy = RUNTIME.mesh
    with timing.section("load_wfn"):
        wfn = WFNReader(wfn_path)
        wfn.adopt_mesh(mesh_xy)
        sym = symmetry_maps.SymMaps(wfn)

    nval = int(params.get("nval", 5))
    ncond = int(params.get("ncond", 5))
    nband = int(params.get("nband", 100))
    bispinor = bool(params.get("bispinor", False))

    # Band window the GW run will actually ask for: ``load_kin_ion_submatrix``
    # reads [b_id_0, b_id_3) = [0, nelec + ncond).  Sizing the file below
    # that silently truncates the run's window, so it is a hard floor;
    # ``nband`` (the polarizability window) is the natural default.
    nb_window = int(wfn.nelec) + ncond
    nb_req = int(args.nb) if args.nb is not None else max(int(nband), nb_window)
    if nb_req < nb_window:
        raise SystemExit(
            f"Requested {nb_req} bands but the deck's sigma window needs "
            f"nelec+ncond = {int(wfn.nelec)}+{ncond} = {nb_window}."
        )
    nb_eff = max(1, min(int(wfn.nbands), nb_req))
    if nb_eff < nb_window:
        raise SystemExit(
            f"{os.path.basename(wfn_path)} only has {int(wfn.nbands)} bands but "
            f"the deck's sigma window needs {nb_window}."
        )
    meta = Meta.from_system(wfn, sym, nval, ncond, nb_eff, 0, bispinor)
    nx, ny, nz = meta.fft_grid
    # ρ (and hence V_H) lives on the ψ FFT box, which for a BGW WFN is
    # already the ecutrho grid — do NOT let a stale ``grid_rho`` attribute
    # push the density onto a different mesh than ``compute_local_V_k``.
    if getattr(wfn, 'grid_rho', None) is not None and (
        tuple(int(x) for x in wfn.grid_rho) != tuple(int(x) for x in meta.fft_grid)
    ):
        raise SystemExit(
            f"wfn.grid_rho={tuple(wfn.grid_rho)} != FFT box {tuple(meta.fft_grid)}"
        )
    print0(f"Bands: {nb_eff} (deck nband={nband}, sigma window needs {nb_window}), "
          f"FFT grid: {meta.fft_grid}, k-points: {sym.nk_tot}")
    print0(f"sys_dim: {sys_dim}   bispinor: {bispinor}   "
          f"nspin/nspinor: {int(getattr(wfn, 'nspin', 1))}/{int(wfn.nspinor)}")
    print0(f"nval={nval} ncond={ncond} nelec(bands)={int(wfn.nelec)}")
    print0(f"Hartree folded in: {args.hartree}")

    # ---- load pseudopotentials ----
    pseudo_dir = args.pseudo_dir or input_dir
    pseudos = load_pseudopotentials(pseudo_dir)
    if not pseudos:
        # Also try the QE subdirectory (common sandbox layout)
        for fallback in [os.path.join(input_dir, '..', 'qe', 'scf'),
                         os.path.join(input_dir, '..', 'qe', 'nscf')]:
            pseudos = load_pseudopotentials(fallback)
            if pseudos:
                print0(f"Found pseudopotentials in {fallback}")
                break

    # ---- validate (will raise if pseudos missing or sys_dim invalid) ----
    ctx = validate_operator_inputs(
        pseudos=pseudos, wfn=wfn, sys_dim=sys_dim,
        caller="kin_ion_io",
    )
    print0(f"Pseudopotentials: {list(ctx.pseudos.keys())}")
    print0(f"Coulomb truncation: {'2D slab' if ctx.truncation_2d else '3D bulk'}")

    # ---- build structure data ----
    atom_positions = np.asarray(wfn.atom_crys, dtype=float)
    atom_types = np.asarray(wfn.atom_types, dtype=int)
    assignments = build_atom_pp_assignments(
        jnp.asarray(atom_positions), jnp.asarray(atom_types), pseudos
    )
    species_tmp = {}
    for ap in assignments:
        if ap.pseudo is None:
            continue
        key = id(ap.pseudo)
        entry = species_tmp.setdefault(key, {"pseudo": ap.pseudo, "positions": []})
        entry["positions"].append(np.asarray(ap.position, dtype=float))
    species_payload = [
        (e["pseudo"], np.asarray(e["positions"], dtype=float)
         if e["positions"] else np.zeros((0, 3), dtype=float))
        for e in species_tmp.values()
    ]

    # ---- build V_loc on the FFT grid (k-independent) ----
    print0("Building V_loc...")
    with timing.section("build_V_loc"):
        V_loc_r = build_local_ionic_potential_on_G_total(
            assignments=[
                {"pseudo": ap.pseudo, "position": np.asarray(ap.position, dtype=float)}
                for ap in assignments
            ],
            species_groups=species_payload,
            fft_grid=(nx, ny, nz),
            bdot=np.asarray(wfn.bdot, dtype=float),
            cell_volume=float(wfn.cell_volume),
            bvec=np.asarray(wfn.bvec, dtype=float),
            blat=float(wfn.blat),
            truncation_2d=ctx.truncation_2d,
        )
        V_loc_r = jnp.asarray(V_loc_r, dtype=jnp.float64)

    vnl_setup = None
    if pseudos:
        print0("Building unified V_NL setup...")
        vnl_setup = vnl_ops.build_vnl_setup(
            wfn,
            sym,
            meta,
            pseudos,
            nspinor=int(wfn.nspinor),
        )

    # ---- build the mean-field V_H on the same FFT grid (k-independent) ----
    # SAME Coulomb convention as V_loc above (``ctx.truncation_2d``, i.e.
    # the deck's sys_dim) — which is also QE's, since the DFT run that
    # produced E_DFT/vxc.dat used ``assume_isolated='2D'`` for a slab.
    # ``mesh_xy`` was taken from ``RUNTIME.mesh`` above (built and
    # clique-warmed by the module-top ``initialize_communicator_stack()``),
    # so the communicator-bootstrap cost is paid once, up front, instead of
    # inside the ρ psum.  Measured on Frontera/Gloo at P=4: the FIRST
    # collective of a run costs ~12 s of topology exchange and the second
    # costs microseconds, so without it the 1.4 MB all-reduce looks like
    # 70 % of the kernel and the strong-scaling numbers are measuring the
    # transport's handshake.
    #
    # ``owner_only``: this CLI's only V_H consumers are the rank-0 h5
    # write and the rank-0 diagnostic print, so no peer needs the
    # replicated (nk, nb, nb) table (BD.4).  The driver's gspace route
    # (``sigma_dispatch``) keeps the replicated default.
    v_h_all = None
    if args.hartree:
        with timing.section("build_V_H"):
            v_h_all = compute_hartree_matrix(
                wfn, sym, meta, truncation_2d=ctx.truncation_2d, nb=nb_eff,
                mesh=mesh_xy, print_fn=print0, owner_only=True)

    # ---- compute kin+ion: ONE k-scan, bands sharded over every rank -----
    # ``kin_ion`` stays PRISTINE (T + V_loc + V_NL) unless --fold-hartree
    # is given: V_H rides along as its own dataset so the same file can
    # feed a run that wants the exact V_H and one that wants the ISDF
    # quadrature, and so a QSGW rotation has the full ⟨m|V_H|n⟩ matrix.
    #
    # Same replacement, same reasons as ⟨mk|V_H|nk⟩ above: the k-partitioned
    # route boxed a whole k's bands on one rank and stopped scaling at
    # P = nk.  Here the three terms are summed ON THE KET
    # (``sum_operators``) so ⟨m|T+V_loc+V_NL|n⟩ is ONE sweep with one
    # reshard and one einsum, not three of each.
    #
    #   T       |k+G|² ψ            diagonal in G, no FFT
    #   V_loc   F[V(r) F⁻¹ψ]        the only real-space excursion
    #   V_NL    Z E Z† ψ            projector sum, G-local, no FFT
    #
    # V_NL DID fit the operator protocol: its G sum is over the replicated
    # G axis with the band index free, so it needs no collective and forms
    # no (nb, nb).  ``get_kin_ion_k`` is left in place — it is the per-k
    # local-plan kernel the sweep is gated against.
    out_path = args.output or os.path.join(input_dir, "kin_ion.h5")
    from common.mtxel_sweep import (SweepGeometry, band_sphere_spec,
                                    blocks_to_host, kinetic_operator,
                                    local_potential_operator, sum_operators,
                                    sweep_matrix_elements, vnl_operator)
    #
    # THE k-SET IS THE IBZ, and the written table is the star broadcast of
    # it — see "THE IRREDUCIBLE k-SET" at the head of this module for the
    # derivation, including the conjugation the time-reversed rows need.
    # T, V_loc and V_NL are built from the lattice and the atomic
    # positions, so they are exactly symmetric by construction and this is
    # the sweep the argument fits most cleanly.  The FILE is unchanged:
    # still (nk_tot, nb, nb), still in full-BZ order, so no consumer of
    # ``kin_ion.h5`` sees the k-set at all.
    nk_irr = int(wfn.nkpts)
    gtab = padded_gvectors(wfn, k="ibz")
    psi_G = wfn.load(bands=(0, nb_eff), k="ibz",
                     sharding=band_sphere_spec())
    geom = SweepGeometry(mesh=mesh_xy, fft_grid=meta.fft_grid,
                         ngkmax=int(psi_G.shape[3]), nb=nb_eff,
                         ns=int(psi_G.shape[2]), nk=nk_irr,
                         cell_volume=float(wfn.cell_volume))
    terms = [kinetic_operator(geom, np.asarray(wfn.bdot, dtype=float)),
             local_potential_operator(geom, V_loc_r)]
    if vnl_setup is not None:
        terms.append(vnl_operator(geom, vnl_setup))
    print0(f"\n⟨mk|T+V_loc+V_NL|nk⟩: one k-scan over {nk_irr} IRREDUCIBLE "
           f"k-points (broadcast to {sym.nk_tot} full-BZ k), "
           f"{geom.nb} bands sharded over P={world}...")
    # ONE ``kin_ion`` timing section around the WHOLE sweep, count 1 — not
    # one per k.  A per-k section would time the dispatch and attribute the
    # compute to whoever happened to block next.
    with timing.section("kin_ion"):
        H_kin_ion = sweep_matrix_elements(
            psi_G, operator=sum_operators(*terms), geom=geom,
            gvecs=gtab.gvecs, gmask=gtab.mask,
            box_index=wfn.box_index(k="ibz"),
            # the file's own IBZ k, for the same reason as the V_H sweep:
            # ψ and the G-list here are the raw IBZ slab.
            kvecs=np.asarray(wfn.kpoints, dtype=np.float64))
        # THE BOUNDARY: the sink is a serial h5py write on rank 0, which
        # cannot take a sharded operand, so the block is gathered to the
        # owner here and nowhere else.  ``owner_only`` keeps the peers'
        # transient at one chunk instead of the whole (nrk, nb, nb).  The
        # star broadcast follows immediately, so what leaves this block is
        # the full-BZ table the writer has always been handed.
        kin_ion_all = broadcast_ibz_to_full_bz(
            blocks_to_host(H_kin_ion, nb=nb_eff, owner_only=True), sym)
    del H_kin_ion, psi_G

    # ``owner_only``: from here on ``kin_ion_all``/``v_h_all`` exist on
    # rank 0 ONLY (None on the peers) — every consumer below is rank-0.
    # ``folded`` is derived from the args, not from ``v_h_all``, so the
    # provenance flag stays rank-invariant.
    folded = bool(args.fold_hartree and args.hartree)
    if folded and rank == 0:
        kin_ion_all = kin_ion_all + v_h_all

    # ---- write output: COORDINATED, rank 0 only -------------------------
    # Rank 0 alone holds the gathered arrays (owner_only gather), and the
    # file needs exactly one writer.  This is what the old multi-rank
    # refusal becomes: not "you may not run multi-rank" but "multi-rank
    # writes through one rank, after the gather".  The barrier below keeps
    # the peers alive until the file is closed — an early exit would have
    # srun tear the step down mid-write.
    print0(f"\nWriting to {out_path}...")
    desc = ("T + V_loc + V_NL + V_H matrix elements (H_DFT - V_xc)"
            if folded else "T + V_loc + V_NL matrix elements")
    with timing.section("write_h5"):
        if rank == 0:
            with h5py.File(out_path, "w") as f:
                ds = f.create_dataset("kin_ion", data=kin_ion_all, dtype=np.complex128)
                ds.attrs["description"] = desc
                ds.attrs["nk"] = sym.nk_tot
                ds.attrs["nb"] = nb_eff
                ds.attrs["sys_dim"] = sys_dim
                ds.attrs["truncation_2d"] = ctx.truncation_2d
                ds.attrs["pseudopotentials"] = str(list(pseudos.keys()))
                # ---- provenance: everything a consumer must agree with ----
                # ``has_hartree`` means the LEGACY fold-in: V_H is inside the
                # kin_ion VALUES and no consumer may add another.  It stays
                # False for the stored-array default, which is what makes the
                # new format safe for old readers (they see pristine kin_ion
                # and correctly supply their own ISDF V_H).
                ds.attrs["has_hartree"] = folded
                ds.attrs["hartree_truncation_2d"] = bool(
                    ctx.truncation_2d) if folded else False
                ds.attrs["input_file"] = os.path.basename(args.input)
                ds.attrs["wfn_file"] = os.path.basename(wfn_path)
                ds.attrs["nval"] = nval
                ds.attrs["ncond"] = ncond
                ds.attrs["nband_input"] = nband
                ds.attrs["nelec_bands"] = int(wfn.nelec)
                ds.attrs["bispinor"] = bool(bispinor)
                ds.attrs["nspinor"] = int(wfn.nspinor)
                ds.attrs["fft_grid"] = np.asarray(meta.fft_grid, dtype=np.int32)
                # ---- WHAT THIS FILE WAS MADE FROM -------------------------
                # ``wfn_file`` above is a BASENAME: every WFN in the project
                # is called WFN.h5, so it identifies nothing.  A kin_ion.h5
                # that carries no content hash of its WFN and no commit of
                # the code that wrote it cannot be told from a stale one by
                # any test — which is how a broken committed fixture (star
                # spread 2.7e+01 meV against this generator's 6.6e-11 meV)
                # went a month without being noticed.  ``ngkmax`` joins them
                # because it is the ONE shape in the sweep that comes from
                # the WFN rather than the deck, so a mismatch localises a
                # wrong-WFN diagnosis immediately.
                ds.attrs["ngkmax"] = int(wfn.ngkmax)
                ds.attrs["wfn_checksum"] = _wfn_checksum(wfn_path)
                ds.attrs["wfn_checksum_scope"] = _WFN_CHECKSUM_SCOPE
                ds.attrs["generator_commit"] = _generator_commit()
                # The k-set actually COMPUTED.  The dataset is and remains
                # full-BZ ``(nk, nb, nb)``; these say that ``nk - nrk`` of
                # those rows are symmetry copies, not independent evaluations.
                ds.attrs["nrk"] = int(wfn.nkpts)
                ds.attrs["k_set_computed"] = "ibz"
                if v_h_all is not None and not folded:
                    vh = f.create_dataset(HARTREE_DATASET, data=v_h_all,
                                          dtype=np.complex128)
                    vh.attrs["description"] = (
                        "<mk|V_H|nk> (Ry), exact FFT-grid mean-field Hartree; "
                        "NOT included in the 'kin_ion' dataset")
                    vh.attrs["truncation_2d"] = bool(ctx.truncation_2d)
                    vh.attrs["nocc"] = int(wfn.nelec)
                    vh.attrs["fft_grid"] = np.asarray(meta.fft_grid, dtype=np.int32)

    barrier("kin_ion_written")

    # Diagnostics read the gathered tables, which only rank 0 holds.
    _src = ("FOLDED into kin_ion (legacy)" if folded else
            (f"stored as '{HARTREE_DATASET}'" if args.hartree
             else "absent (ISDF route required)"))
    if rank == 0:
        print0(f"Wrote {os.path.basename(out_path)}: kin_ion "
               f"{kin_ion_all.shape}, sys_dim={sys_dim}, V_H {_src}")
        if v_h_all is not None:
            d0 = np.real(np.diagonal(kin_ion_all[0])) * 13.605693122994
            v0 = np.real(np.diagonal(v_h_all[0])) * 13.605693122994
            print0("  kin_ion diag (eV), k=0, first 8: "
                  + "  ".join(f"{v:.4f}" for v in d0[:8]))
            print0("  V_H     diag (eV), k=0, first 8: "
                  + "  ".join(f"{v:.4f}" for v in v0[:8]))
    if rank == 0:
        timing.report(title="--- Timing (seconds) ---",
                      wall=_time.perf_counter() - _t_main)
    return 0


# ---------------------------------------------------------------------------
# Forwarding shims — NOT used by anything in this file.
# ---------------------------------------------------------------------------
# These four names used to be defined here; they are generic k-partition
# plumbing and now live in ``common.collectives``.  Two call sites still
# import them from this module (``gw.sigma_dispatch``,
# ``tests/test_sanity_gates_jax.py``), which are outside this workstream's
# file ownership — see requests R3/R4.  Delete this block once both move.
from common.collectives import (                       # noqa: E402,F401
    gather_indexed_blocks as _gather_indexed_blocks,
    psum_replicate as _psum_replicate,
    replicate_to_mesh,
    sweep_local_k,
)


if __name__ == "__main__":
    raise SystemExit(main())
