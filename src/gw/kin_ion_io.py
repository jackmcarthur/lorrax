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
  ``compute_local_V_k`` route V_loc takes: analytically exact and
  centroid-count independent.  Pinned to QE's ``kih.dat`` at rms
  1e-4 eV.  Built by ONE distributed kernel
  (:func:`compute_hartree_matrix`) shared by this CLI, the driver's
  ``gspace`` route and the QSGW loop: ρ is partitioned over
  (k, band-chunk) and reduced with a single 1.4 MB psum, the Poisson
  solve is replicated on purpose, and ⟨mk|V_H|nk⟩ is partitioned over
  k with no reduction at all.  It strong-scales; see scorecard §X.
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
# ``jax.distributed.initialize()`` refuses to run once the XLA backend is
# up, and the import graph below (``psp.*``) reaches jax.  ``runtime``
# itself imports no jax, and every piece of ``bootstrap()`` is idempotent
# through an env sentinel — which is what makes this safe under
# ``gw.sigma_dispatch``'s LAZY import of this module from inside an
# already-bootstrapped driver: there the calls are all no-ops.
from runtime import bootstrap                                 # noqa: E402
bootstrap()          # env defaults + jax.distributed + CPU fallback + failfast

import argparse                                               # noqa: E402

import numpy as np
import jax.numpy as jnp
import h5py

from common import symmetry_maps, Meta
from common.collectives import (barrier, device_count, gather_k_blocks,
                                local_share, prepare_mesh, process_rank_world,
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


def get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, vnl_setup, wfn, g_mask=None,
                  V_H_r=None):
    """Compute T + V_loc + V_NL (+ V_H) for a single k-point.

    Parameters
    ----------
    wfn_k : (nb, nspinor, nx, ny, nz) — wavefunctions in FFT box
    Gk_crys : (nG, 3) int — G-vector indices for this k.  May be the
        k's own ``ngk`` rows or the ``ngkmax``-padded table, in which
        case ``g_mask`` is REQUIRED (pad rows hold ``(0,0,0)``, a valid
        box index that aliases Γ, so an absent mask double-counts ψ(G=0)
        rather than crashing).
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
    if g_mask is None and int(np.count_nonzero(~Gk_np.any(axis=1))) > 1:
        # A physical G-sphere contains (0,0,0) at most ONCE, and
        # ``WfnLoader.gvecs()`` pads with exactly that row — so more than
        # one all-zero row means a padded list arrived without its mask.
        # Refuse instead of returning a plausible number: the pad rows are
        # a VALID FFT-box index, so the unmasked answer is silently wrong
        # by (ngkmax - ngk) extra copies of the Γ component, inside a
        # ~500 eV cancellation.  Cheap: one host reduction over (nG, 3)
        # int per k.
        raise ValueError(
            f"get_kin_ion_k: Gk_crys has "
            f"{int(np.count_nonzero(~Gk_np.any(axis=1)))} all-zero rows, so "
            f"it is ngkmax-padded, but g_mask is None.  Pass the mask from "
            f"psp.dft_operators.padded_gvectors(...).at(ik).")
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
#   ⟨mk|V_H|nk⟩       partitioned over k, each k computed rank-locally with
#                     NO cross-rank reduction (a k block is independent);
#                     combined by one ~15-35 MB all-gather.
#   ψ                 loaded per rank for the (k, band) windows that rank
#                     owns — process-local, never a global sharded array,
#                     so the I/O scales with P too and no collective is
#                     implied by the load.
#
# Why the partitions differ between the two sweeps: ρ is a sum over
# (k, band) and both axes are free, whereas ⟨mk|V_H|nk⟩ contracts the FULL
# band window against itself at fixed k, so bands cannot be split without
# introducing a reduction.  k is the axis both share, and it is the one the
# dominant sweep (matrix elements, 1.05e12 of the 1.2e12 flops at 12×12)
# needs.  Band chunking is layered on top of the ρ sweep only when P > nk,
# which is the only regime where k alone stops filling the machine.


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
    psi_g = wfn.load_process_local(bands=(0, nmix), k=[int(ik)])
    ns_have = int(psi_g.shape[2])
    if int(meta.nspinor) > ns_have:
        psi_g = jnp.pad(psi_g, ((0, 0), (0, 0),
                                (0, int(meta.nspinor) - ns_have), (0, 0)))
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
            psi_k = load_kpoint_fftbox_local(wfn, meta, ik, b_hi, b_lo=b_lo)
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
                           print_fn=print):
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
    the Poisson solve is replicated; ⟨mk|V_H|nk⟩ is partitioned over k
    with no reduction at all and gathered once.  ``mesh`` is the
    collectives' device mesh — pass the run's own (the driver does) or
    leave it None and one is derived, 1×1 on a single device.  P=1 is
    bit-for-bit the serial result.

    Needs no pseudopotentials: ρ comes from ψ, V_H from the Poisson
    solve, and the matrix element from the same ``compute_local_V_k``
    route V_loc takes.  ``truncation_2d`` MUST be the run's own
    convention (deck ``sys_dim``); mixing it with V_loc's is a large
    systematic error inside a ~500 eV cancellation.

    Returns the FULL matrix as a host ``numpy`` array replicated on
    every rank, not the diagonal: a QSGW band rotation needs
    ⟨m|V_H|n⟩ off-diagonals to transform H₀ into the QP basis.
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
    # Force it back to a process-local host array so the k sweep below
    # runs entirely in single-device land (no global operands ⇒ XLA can
    # emit no collective there even in principle).
    V_H_r = jnp.asarray(np.asarray(V_H_r, dtype=np.float64),
                        dtype=jnp.float64)
    del rho_np

    # ---- 3. ⟨mk|V_H|nk⟩: k-partitioned, no reduction, one gather ------
    #
    # Fixed-shape G (owner decision D10, 2026-07-30).  ``padded_gvectors``
    # hands over the loader's OWN ``(nk, ngkmax, 3)`` table instead of
    # slicing it back to each k's ``ngk``, so every k presents identical
    # operand shapes: ``_compute_local_V_k_jit`` lowers ONCE instead of
    # once per distinct ngk, and the loop becomes a sequence of dispatches
    # of the same executable that ``collectives.sweep_local_k`` can
    # pipeline behind a single readback.  The pad columns are made inert by
    # ``g_mask`` — see ``PaddedGVectors`` for why the mask is mandatory
    # rather than merely tidy (pad rows are ``(0,0,0)``, i.e. Γ).
    #
    # This supersedes the pre-D10 COMPILE NOTE, whose stated objection was
    # that pad columns "change the summation order" of a matrix element
    # inside H₀'s ~500 eV cancellation.  Appending exact zeros does not
    # change a sum; what a shape change moves is XLA's reduction
    # BLOCKING, which means the ragged route was itself carrying a
    # per-ngk-dependent association and this route makes it uniform.
    # Neither is a reference, so the two are compared on the merits at
    # 1e-12 (tests/test_kin_ion_padded_gvectors.py).
    gtab = padded_gvectors(wfn, k="full_bz")

    def _vh_block(ik):
        wfn_k = load_kpoint_fftbox_local(wfn, meta, ik, nb)
        G_pad, g_mask = gtab.at(ik)
        return compute_local_V_k(
            wfn_k, G_pad, V_H_r, wfn.cell_volume, g_mask=g_mask)

    return gather_k_blocks(nk, _vh_block, item_shape=(nb, nb),
                           label="vh_matrix", print_fn=print_fn)


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
    # loops below are partitioned over ranks (``collectives.gather_k_blocks``),
    # and the write is coordinated — rank 0 alone opens the file, after a
    # gather that has already given it every k.
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
    with timing.section("load_wfn"):
        wfn = WFNReader(wfn_path)
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
    # ``prepare_mesh`` also pays the communicator-bootstrap cost once, up
    # front, instead of inside the ρ psum.  Measured on Frontera/Gloo at
    # P=4: the FIRST collective of a run costs ~12 s of topology exchange
    # and the second costs microseconds, so without it the 1.4 MB
    # all-reduce looks like 70 % of the kernel and the strong-scaling
    # numbers are measuring the transport's handshake.
    mesh_xy = prepare_mesh(print_fn=print0)
    v_h_all = None
    if args.hartree:
        with timing.section("build_V_H"):
            v_h_all = compute_hartree_matrix(
                wfn, sym, meta, truncation_2d=ctx.truncation_2d, nb=nb_eff,
                mesh=mesh_xy, print_fn=print0)

    # ---- compute kin+ion per k-point (k-partitioned, one gather) --------
    # ``kin_ion`` stays PRISTINE (T + V_loc + V_NL) unless --fold-hartree
    # is given: V_H rides along as its own dataset so the same file can
    # feed a run that wants the exact V_H and one that wants the ISDF
    # quadrature, and so a QSGW rotation has the full ⟨m|V_H|n⟩ matrix.
    #
    # Same partition as the V_H matrix sweep and for the same reason: a
    # k block of ⟨mk|T+V|nk⟩ is independent of every other, so this is
    # embarrassingly parallel with one indexed gather at the end and no
    # reduction anywhere.  (The two sweeps are kept separate rather than
    # fused so that ``compute_hartree_matrix`` stays the one shared kernel
    # all three consumers call; the price is one extra ψ read per k, which
    # is itself now P-way parallel.)
    out_path = args.output or os.path.join(input_dir, "kin_ion.h5")
    gtab = padded_gvectors(wfn, k="full_bz")

    def _kin_ion_block(ik):
        wfn_k = load_kpoint_fftbox_local(wfn, meta, ik, nb_eff)
        G_pad, g_mask = gtab.at(ik)
        return get_kin_ion_k(wfn_k, G_pad, sym.unfolded_kpts[ik],
                             V_loc_r, vnl_setup, wfn, g_mask=g_mask)

    # ``gather_k_blocks`` records ONE ``kin_ion_k`` section around the WHOLE
    # sweep, count 1 — not one per k.  The per-k timer this replaces only
    # worked because ``np.asarray(H_k)`` synchronised inside it; with the
    # readback deferred a per-k section would time the dispatch and
    # attribute the compute to whoever happened to block next.
    kin_ion_all = gather_k_blocks(sym.nk_tot, _kin_ion_block,
                                  item_shape=(nb_eff, nb_eff),
                                  label="kin_ion", print_fn=print0)

    folded = bool(args.fold_hartree and v_h_all is not None)
    if folded:
        kin_ion_all = kin_ion_all + v_h_all

    # ---- write output: COORDINATED, rank 0 only -------------------------
    # Every rank holds the identical gathered arrays, so the file needs
    # exactly one writer.  This is what the old multi-rank refusal
    # becomes: not "you may not run multi-rank" but "multi-rank writes
    # through one rank, after the gather".  The barrier below keeps the
    # peers alive until the file is closed — an early exit would have
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

    _src = ("FOLDED into kin_ion (legacy)" if folded else
            (f"stored as '{HARTREE_DATASET}'" if v_h_all is not None
             else "absent (ISDF route required)"))
    print0(f"Wrote {os.path.basename(out_path)}: kin_ion {kin_ion_all.shape}, "
          f"sys_dim={sys_dim}, V_H {_src}")
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
