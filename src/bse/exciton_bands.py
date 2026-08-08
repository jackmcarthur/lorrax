"""Exciton bandstructure E_S(Q) along a high-symmetry Q path (TDA).

Finite-momentum TDA BSE ``H_Q = D_Q + V_Q − W`` in the pair basis
``|v k, c k+Q⟩`` (LORRAX convention: electron leg shifted by +Q), solved at
every Q of a user-supplied high-symmetry path with ONE compiled engine:

  * **Q path** — read from the SAME ``K_POINTS crystal_b`` block (QE-style,
    labels via ``#``) that the htransform bandstructure driver consumes;
    parsing / node labels / path-distance accumulation reuse
    ``bandstructure.htransform`` (``generate_kpath_from_qe_segments`` /
    ``initialize_kpath``).  One path format in the package.
  * **ψ_c(k+Q, r_μ), ε_c(k+Q)** — htransform (``bse_setup.compute_wfns_fi``
    with an explicit q-list = {k + Q}): mutually consistent eigenpairs of
    one interpolated fH_{k+Q}; no shifted NSCF, no Sternheimer
    (arbitrary_q_bse.md §1/§2a-b).  htransform returns cell-periodic u at
    wrapped labels — the same torus convention as the stored grid ψ.  The fH
    is built over the FULL loaded band window (all valence + all conduction of
    the input, e.g. nband=40 = 26v+14c), EXACTLY as the standard SP
    bandstructure driver builds it; the BSE conduction bands are a sub-window
    returned interior to it, guarded above by the extra conduction bands so no
    selection boundary cuts a near-degenerate pair.  Running on a sliver
    conduction window instead makes the off-grid caches ring 100-1000 meV
    (spurious exciton dips); use a full-band input.
  * **Direct kernel W(k−k')** — UNCHANGED coarse tiles + coarse-k FFT
    convolution (all k-differences stay on-grid when every conduction leg
    shifts by the same Q; §2c per-element verification).  The matvec is the
    ONE production trial-stack matvec (``bse_stack_matvec``) with the
    conduction slots holding the Q-shifted caches — no parallel kernel.
  * **Exchange V_Q** — from ``bse.vq_interp``:
      ``--vq-mode=interp``  F-scheme + b26p interpolation (fast; ONE jitted
                            evaluator, per-Q dispatch-only), or
      ``--vq-mode=refit``   per-Q ζ refit (compute-don't-interpolate — the
                            off-grid GROUND TRUTH; expensive), or
      ``--vq-mode=both``    interp on the full path + refit on
                            ``--refit-points`` spot checks, solved in the
                            SAME compiled scan (extra scan rows) so the
                            interp-vs-refit ΔE_S table is apples-to-apples.
    Momentum labeling: the pair density conj(ψ_c[k+Q]) ψ_v[k] pairs with
    the tile at TILE momentum q = wrap(−Q) (reference-metric convention,
    ``vq_interp`` docstrings); on-grid this is exactly the stored
    ``V_qmunu[wrap(−Q)]`` slot.
  * **Γ endpoint** — at exactly Γ the production q=0 tile is used (stored
    head-body convention: compute_vcoul zeroes G=0, the mini-BZ-averaged
    head is the loader's rank-1 injection).  At every finite Q the G=0 term
    of v(Q+G) is KEPT (BGW ``energy_loss`` convention; finite_q_bse.md) —
    the physical nonanalytic exchange branch, so E(Q→0) need not equal
    E(Γ) on the longitudinal states.  Both facts are written into the .dat
    header.
  * **THE SCAN** — the per-Q solve (hoisted pair amplitudes + block-Lanczos
    over the stack matvec) is one jitted function ``lax.scan``-ned over the
    whole Q list: ONE compile serves every Q (verified by the
    JAX_LOG_COMPILES census; per-q-recompile lesson, PHASE2_LOG).
    Interpolated V_Q are Hermitized (0.5(V+V^H)) — commutes with the pair
    contraction; the anti-Hermitian residue is stencil noise.

Outputs: ``<prefix>.dat`` (path distance, Q, lowest n_eig in eV, labeled
header; refit rows flagged) and ``<prefix>.png`` (bands along the path,
high-symmetry ticks; refit spot checks overlaid as markers).

Band-pad guard: mesh padding adds ψ=0 conduction/valence slots; with ε=0
they would alias into the LOW spectrum (ε_c,pad − ε_v ≈ −ε_v).  Pad
energies are pushed out of the window (ε_c,pad=+1e3 Ry, ε_v,pad=−1e3 Ry) —
pad transitions then sit at +2e3 Ry, harmless to the lowest n_eig.  (No-op
at 1×1 mesh: there are no pads.)

Run (never on a login node; module-free srun+shifter runner):
    python -m bse.exciton_bands -i cohsex.in --n-val 4 --n-cond 4 \
        --n-eig 6 --vq-mode both --refit-points 0,8,15,22,29
The input file must carry a K_POINTS crystal_b block (the Q path).
"""
from __future__ import annotations

import os
import time
from functools import partial

import numpy as np

# THE startup call (runtime module docstring): env defaults, SLURM-aware
# ``jax.distributed.initialize``, CPU fallback, the run's clique-warmed
# square ('x','y') mesh, compile cache, rank-0 report.  Must run
# BEFORE this module's own ``import jax`` and any ``jax.devices()`` /
# mesh creation so a multi-node srun yields the full global device set.
# This driver names ``--px/--py``; ``_create_mesh_xy(px, py)`` in main()
# reuses the startup mesh when the requested shape matches it and builds
# (and warms) the requested one when it does not.
from runtime import initialize_communicator_stack
RUNTIME = initialize_communicator_stack()

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from solvers.lanczos import block_lanczos_eig_jit
from common.fft_helpers import make_sharded_ifftn_3d
from .bse_io import (_find_restart_file, load_bse_data_from_restart_sharded,
                     decimate_W_q_to_subgrid, make_w_densifier,
                     PAD_EPS_GUARD_RY)
from .bse_ring_comm import make_bse_shardings
from .bse_serial import compute_pair_amplitude
from .bse_stack_matvec import build_bse_stack_matvec
from .bse_w_exact import _create_mesh_xy
from . import vq_interp

RY2EV = 13.6056980659
# PAD_EPS_GUARD_RY now lives in bse_io — the module that OWNS the band pad —
# and is imported above.  It was defined here because this driver was the
# only one that knew the loader's zero ε pad was a wrong number and repaired
# it locally; the loader is correct now, so the constant belongs at the seam
# and every BSE driver inherits the guard instead of one of nine.


def _gather_host(x):
    """Gather a ``jax.Array`` to a full host numpy array, identical on every
    process, whether it is REPLICATED or PROCESS-SPANNING.

    ONE line of delegation to :func:`common.collectives.gather_to_host`, and
    the reasoning stays HERE because this driver is where it was learned:

    ``jax.device_get`` raises on an array whose shards live on OTHER processes
    (the diagnostic gate's Q-shifted ψ_c is μ-sharded ``P(...,'x')``, so on a
    16-process / 4-node run no single process holds it all) — those need
    ``multihost_utils.process_allgather``, which stitches the full logical
    array (gathering only the sharded axes) on every process.  But a
    FULLY-ADDRESSABLE array (replicated, or single-process) must NOT go through
    ``process_allgather(tiled=True)`` — that concatenates each process's full
    copy and DUPLICATES the leading axis (e.g. eps_c (144,·) → (16·144,·)).  So
    the service branches on ``is_fully_addressable``: ``device_get`` when the
    whole array is local, ``process_allgather`` only when shards are remote.
    The branch is a global property (identical on every process), so the
    collective stays in lockstep.  On one process everything is fully
    addressable ⇒ plain device_get.

    WHAT THIS DRIVER ACTUALLY FEEDS IT, NOW.  Only ``evs_dev`` — the
    block-Lanczos Ritz values, ``P()``-sharded.  The driver's own log line
    records that such an array reports ``fully_addressable=False`` at P=64
    (job 7882507), so it takes the service's ``is_fully_replicated`` arm: a
    LOCAL buffer read, no collective.  The one caller that used to hand this
    function a μ-sharded, genuinely process-spanning array — the on-grid
    diagnostic gate — no longer does; it contracts μ on device instead (see
    :func:`_gate_stats_on_device` for the three warm-up attempts that failed
    to make that gather work under impl=mpi).  So this driver now issues no
    cross-process host gather at all.

    The branch still matters and is still gated in
    ``tests/test_bse_gather_and_mesh.py``, because at 64 ranks the wrong arm
    silently returns an array 64× too long on the leading axis, and because
    the service is shared.
    """
    from common.collectives import gather_to_host
    return gather_to_host(x)


# ===========================================================================
# the single-compile path solver: scan(per-Q block-Lanczos) over the Q list
# ===========================================================================
def build_path_solver(mesh_xy: Mesh, nkx: int, nky: int, nkz: int,
                      nc_pad: int, nv_pad: int, *, n_eig: int,
                      block_size: int, max_iter: int, n_reorth: int | None = None):
    """One jitted ``solve_path`` for a whole Q list.

        solve_path(psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q,
                   psi_v_X, psi_v_Y, eps_v, W_R) -> evs (nQ, n_eig)

    The scan body per Q: hoist the exchange pair amplitudes M_X/M_Y from
    the Q-shifted conduction ψ (audit-P3 contract — matvec args, computed
    once per solve, reused across all Lanczos iterations), then run the
    fixed-iteration block Lanczos with the ONE production stack matvec.
    Everything Q-dependent arrives as scan xs; Q-independent operands
    (valence ψ/ε, W_R) are loop constants.  ONE XLA compile serves every
    Q — and every later call with the same nQ (compile census deliverable).
    """
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz
    n_flat = nc_pad * nv_pad * nk
    # ``krep`` (LORRAX_BSE_MATVEC_OPT): pin the FLAT Krylov vectors to
    # replicated.  Without a constraint the (block, n_flat) axis inherits the
    # tiling of ``sh.X`` through the reshape, so the Lanczos algebra — every
    # reorthogonalisation dot product, the QR, the Ritz eigh — runs on a
    # sharded 1024-long axis and each of those becomes a collective.  With it,
    # exactly one all-gather per matvec and the rest is local.  See the dial's
    # documentation in bse_stack_matvec for the residency it costs and why
    # that makes it wrong at large pair dimension.
    from .bse_stack_matvec import matvec_opts as _mv_opts
    krylov_rep = (NamedSharding(mesh_xy, P()) if "krep" in _mv_opts() else None)
    if n_reorth is None:
        # FULL reorthogonalisation by default: exciton windows are small
        # (n_flat = nc·nv·nk ~ 10²-10⁴), the Krylov space often saturates
        # (solvers.lanczos clamps at floor(n/bs)), and partial reorth at
        # saturation breeds ghost duplicates.  Full reorth costs
        # O(M·n·bs²) — negligible next to the matvec at every BSE size.
        n_reorth = max_iter
    matvec = build_bse_stack_matvec(mesh_xy, nkx, nky, nkz, kernel="bse")

    @jax.jit
    def solve_path(psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q,
                   psi_v_X, psi_v_Y, eps_v, W_R):
        def body(carry, xs):
            psi_c_X, psi_c_Y, eps_c, V = xs
            psi_c_X = lax.with_sharding_constraint(psi_c_X, sh.psi_x)
            psi_c_Y = lax.with_sharding_constraint(psi_c_Y, sh.psi_y)
            V = lax.with_sharding_constraint(V, sh.V)
            M_X = lax.with_sharding_constraint(
                compute_pair_amplitude(psi_c_X, psi_v_X), sh.psi_x)
            M_Y = lax.with_sharding_constraint(
                compute_pair_amplitude(psi_c_Y, psi_v_Y), sh.psi_y)

            def matvec_block(Vb):
                if krylov_rep is not None:
                    Vb = lax.with_sharding_constraint(Vb, krylov_rep)
                X = Vb.reshape(block_size, nc_pad, nv_pad, nk)
                X = lax.with_sharding_constraint(X, sh.X)
                HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                            eps_c, eps_v, W_R, V, M_X, M_Y)
                HX = HX.reshape(block_size, -1)
                if krylov_rep is not None:
                    HX = lax.with_sharding_constraint(HX, krylov_rep)
                return HX

            evs, _ = block_lanczos_eig_jit(
                matvec_block, n_flat, n_eig=n_eig, block_size=block_size,
                max_iter=max_iter, n_reorth=n_reorth)
            return carry, evs[:n_eig].real

        _, evs_all = lax.scan(body, None, (psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q))
        return evs_all

    return solve_path


# ===========================================================================
# stacks: htransform conduction caches + V_Q tiles for the whole path
# ===========================================================================
def build_conduction_stacks(bundle, nQ, nk, n_cond, n_cond_pad, n_rmu,
                            n_rmu_pad, mesh_xy):
    """Reshape the htransform bundle over the concatenated {k+Q} list into
    per-Q conduction caches, padded to the loader's mesh extents — one
    jitted reshape+pad+reshard, no host round-trip (the bundle stays on
    device; at 40 path points the old device_get→np.pad→2×device_put moved
    ~1.7 GB through the host).

    Returns (psi_cQ_X, psi_cQ_Y, eps_cQ):
        psi_cQ_[XY]: (nQ, nk, nc_pad, ns, n_rmu_pad), μ on x / y
        eps_cQ:      (nQ, nk, nc_pad) — pad bands at +PAD_EPS_GUARD_RY
    """
    x5 = NamedSharding(mesh_xy, P(None, None, None, None, "x"))
    y5 = NamedSharding(mesh_xy, P(None, None, None, None, "y"))
    rep = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=(x5, y5, rep))
    def _stacks(psi, eps):
        ns = psi.shape[2]
        psi = psi.reshape(nQ, nk, n_cond, ns, n_rmu)
        eps = eps.reshape(nQ, nk, n_cond)
        psi = jnp.pad(psi, ((0, 0), (0, 0), (0, n_cond_pad - n_cond),
                            (0, 0), (0, n_rmu_pad - n_rmu)))
        eps = jnp.pad(eps, ((0, 0), (0, 0), (0, n_cond_pad - n_cond)),
                      constant_values=PAD_EPS_GUARD_RY)
        return psi, psi, eps

    return _stacks(bundle.psi_rmu_Y, bundle.enk_full)


def _gate_stats_on_device(eps_ht, eps_st, psi_ht, psi_st, nc, mesh_xy):
    """Both gate numbers computed ON DEVICE; only replicated results read back.

    Returns ``(max|Δε_c|, Gram(nk, nc, nc))`` as REPLICATED arrays, read with
    ``addressable_data(0)`` — a purely local buffer read that issues no
    collective at all.

    WHY NOT ``gather_to_host``.  Three separate attempts to make a host gather
    work here failed on hardware at P=16 under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi``, always in the same frame and
    always with "Communicator requested from a thread that is not the one MPI
    was initialized from": the mesh-clique warm-up did not cover it (job
    7882523), a ``process_allgather`` warm-up on a host operand did not (job
    7882531), and a path-exact warm-up on a genuinely sharded operand did not
    either (job 7882555/7882561).  ``multihost_utils.process_allgather``
    identity-jits to ``P()``, and at this point in the program that jit
    acquires a communicator from an intra-op pool worker; the warm-up
    mechanism that fixes ``shard_map(psum)`` cliques does not fix it.

    So stop trying to gather.  The μ contraction rides a mesh-AXIS psum, which
    IS warmed and works; everything else here is already replicated, so the
    host side needs no collective whatsoever.  This is also strictly cheaper:
    the old code pulled ``nk·nc·ns·n_μ`` complex numbers to every process
    inside a diagnostic, on the one axis the whole BSE is sharded over.

    Pad slots are exact zeros on both ψ legs, so the padded μ extent is
    carried through rather than sliced — slicing a sharded axis would be a
    reshard, and adding zeros changes neither a norm nor an inner product.
    """
    rep = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=(rep, rep, rep))
    def _f(e_ht, e_st, A, B):
        # ORDER.  htransform returns eigenvalues ASCENDING (they come out of an
        # eigensolve); the stored restart holds them in DFT-BAND-INDEX order.
        # QP corrections reorder bands, so those two orders differ BY
        # CONSTRUCTION, not by error.  Differencing them index-by-index — which
        # is what this gate did — measures the permutation, not the
        # interpolation.
        #
        # Measured on run_EXB_c785 (2026-07-31): the index-wise number is
        # 80.694 meV and is reproducible from ``eqp1.dat`` ALONE, with no
        # htransform, no Galerkin basis and no interpolation in the arithmetic:
        #   max_k max_{b in [26,34)} |sort(EQP)[b] - EQP[b]| = 80.6935 meV
        #   at k=(0.25,-0.5,0), band 30, where the QP shifts push band 30 above
        #   31 and 32.
        # The order-matched number on the same run is 6.5e-11 meV.  The 80.694
        # was read as this path's accuracy limit for a whole campaign.  The
        # Perlmutter campaign hit the identical trap at a fixed 57.902 meV.
        #
        # ``d`` is therefore the SET comparison — did htransform recover the
        # same conduction manifold — which is the question this gate asks.
        # ``d_idx`` is kept and reported alongside so the permutation stays
        # VISIBLE rather than silently folded into the accuracy number: a large
        # d_idx with a tiny d is band reordering and is expected; a large d is
        # the interpolation basis failing.
        d = jnp.max(jnp.abs(jnp.sort(e_ht[:, :nc], axis=-1)
                            - jnp.sort(e_st[:, :nc], axis=-1)))
        d_idx = jnp.max(jnp.abs(e_ht[:, :nc] - e_st[:, :nc]))
        A = A[:, :nc]
        B = B[:, :nc]
        nA = jnp.sqrt(jnp.sum(jnp.abs(A) ** 2, axis=(2, 3)))[:, :, None, None]
        nB = jnp.sqrt(jnp.sum(jnp.abs(B) ** 2, axis=(2, 3)))[:, :, None, None]
        G = jnp.einsum("kasm,kbsm->kab", jnp.conj(A / nA), B / nB)
        return d, d_idx, G

    return _f(eps_ht, eps_st, psi_ht, psi_st)


def _local(x):
    """This process's own buffer of a REPLICATED device array — no collective.

    ``addressable_data(0)`` is a local read.  For a ``P()``-sharded array
    every process holds the whole thing, so this IS the full array; unlike
    ``gather_to_host`` it cannot issue (or refuse) a communicator request.
    """
    return np.asarray(jax.device_get(x.addressable_data(0))
                      if hasattr(x, "addressable_data") else x)


def gate_htransform_vs_stored(psi_cQ_gamma, eps_cQ_gamma, data,
                              mesh_xy, log=print):
    """On-grid consistency gate at a Γ path point: htransform conduction
    ε vs the stored restart ε (max |Δ|), and the gauge-free per-k subspace
    overlap min singular value of ⟨ψ_ht|ψ_stored⟩ (bands × bands, spin+μ
    contracted).  Values printed; hard-fails only on gross breakage (>0.5
    overlap loss / >0.1 Ry ε drift) — htransform accuracy is
    centroid-count-governed and reported, not silently trusted.

    ψ IS NOT GATHERED.  This used to pull both the htransform and the stored
    μ-sharded ψ to host on every process and do the contraction in numpy.
    Two things were wrong with that.  (1) Scaling: it is an ``O(N_μ)``
    all-gather per process inside a DIAGNOSTIC, on the one axis the whole BSE
    is sharded over.  (2) It did not work: at P=16 under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` the gather refuses with
    "Communicator requested from a thread that is not the one MPI was
    initialized from" — reproduced 6/6 and 4/4 cells in jobs 7882523 and
    7882531, in this exact frame, and NOT cured by either of two
    ``process_allgather`` warm-up attempts.  Contracting μ on device (where
    the psum rides an already-warmed mesh-axis clique) removes the operand
    that could not be gathered instead of trying harder to gather it, and
    leaves a ``nk·nc²`` object that every process holds.

    ε goes the same way: it is folded into the same jit and returned as a
    replicated scalar, so the host side of this gate issues NO collective at
    all — see ``_gate_stats_on_device`` for the three warm-up attempts that
    did not make a host gather work here.
    """
    nc = int(data["n_cond"])
    d_dev, d_idx_dev, G_dev = _gate_stats_on_device(
        eps_cQ_gamma, data["eps_c"], psi_cQ_gamma, data["psi_c_X"],
        nc, mesh_xy)
    d_eps = float(_local(d_dev))
    d_eps_idx = float(_local(d_idx_dev))
    G = _local(G_dev)
    smin = 1.0
    for k in range(G.shape[0]):
        sv = np.linalg.svd(G[k], compute_uv=False)
        smin = min(smin, float(sv.min()))
    d_meV = d_eps * RY2EV * 1e3
    d_meV_idx = d_eps_idx * RY2EV * 1e3
    log(f"  [gate] htransform@Γ vs stored: max|Δε_c| = {d_meV:.6f} meV "
        f"(order-matched; this is the accuracy number), "
        f"conduction-subspace overlap min-sval = {smin:.4f}")
    log(f"  [gate] band-index-wise |Δε_c| = {d_meV_idx:.3f} meV — this is "
        f"NOT an accuracy figure.  htransform returns energies ascending and "
        f"the restart stores them in DFT-band order, so QP band reordering "
        f"alone produces a large value here.  Compare the order-matched "
        f"number above.")
    # The min-sval (subspace overlap) does NOT see energy corruption: an
    # over-packed interp window keeps the ψ_c SPAN (min-sval healthy) while the
    # recovered conduction ENERGIES drift by ~eV (640c/nband=80: min-sval 0.86
    # but on-grid |Δε_c| ~955 meV — 10_lorrax_exciton_bands_80interp_8v8c).  So
    # the ENERGY gate is the authoritative interp-basis check, tightened here.
    if d_meV > 20.0:
        log(f"  [warn] on-grid conduction ENERGY error {d_meV:.1f} meV ≫ the "
            f"~1-2 meV htransform floor — the interp basis is over-packed: "
            f"640-scale centroids cannot orthonormalize high oscillatory bands, "
            f"so their non-orthonormal Galerkin coeffs pollute fH=Σf(ε)ccᴴ.  "
            f"Reduce nband toward the BSE window (a few guard bands).")
    assert d_eps < 0.05 and smin > 0.5, (
        f"htransform conduction cache grossly inconsistent with the stored grid "
        f"(max|Δε_c|={d_meV:.1f} meV, min-sval={smin:.3f}) — interp basis broken. "
        f"The htransform fH energy recovery needs orthonormal Galerkin coeffs; "
        f"640-scale centroids cannot carry a full 80-band window (per-band "
        f"capacity finding: runs/MoS2/04_mos2_12x12_bands_2026-07-18/"
        f"10_lorrax_exciton_bands_80interp_8v8c).")
    return d_eps, smin


# ===========================================================================
# driver
# ===========================================================================
_PROBE_TICK = None      # instrument: set by an out-of-tree profiling probe


def rerun_check_enabled(args) -> bool:
    """Is the diagnostic warm re-run of the solve scan switched on?

    The re-run is a pure reproducibility assert: it re-solves the ENTIRE Q
    scan a second time and compares the two tables.  It buys no physics, and
    it was measured at **37.7 % of driver wall** at P=4 / 41 Q (and 38.1 % at
    P=64 / 91 Q, job 7882533) — the single largest row in the stage table.
    That is why the default flipped OFF on 2026-08-08 and the check now has to
    be asked for, with ``--rerun-check``.

    ``--skip-rerun-check`` predates the flip and is still accepted, because
    the campaign harnesses and the archived launch recipes pass it.  It now
    *names* the default instead of changing it, and it still wins if both
    flags are given: a flag whose name says "skip" must never switch the
    re-run on.

    This predicate is the ONE place the decision is taken — the driver calls
    it rather than inlining the boolean, so that
    ``tests/test_exciton_bands_rerun_default.py`` gates the expression the
    driver actually evaluates rather than a copy of it.
    """
    if getattr(args, "skip_rerun_check", False):
        return False
    return bool(getattr(args, "rerun_check", False))


def build_parser():
    """The driver's argparse parser, built where a test can reach it.

    This used to be inline in ``main``, which meant the only way to observe a
    flag's default was to run a full solve.  It is a plain extraction: same
    arguments, same order, same defaults.
    """
    import argparse
    # ``eigh_backend`` + ``use_low_mem_eigh`` are ONE axis with ONE
    # resolver, and the CLI vocabulary is the resolver's own list rather
    # than a hand-copied tuple that drifts (it had: this flag accepted
    # auto|off|cusolvermp|slate while distrib_la had grown ``distributed``
    # and ``scalapack``, so the CPU distributed eigh was unreachable from
    # here).  Function-local because gw_config parses decks and this
    # module is imported by things that do not.
    from gw.gw_config import eigh_backend_choices

    ap = argparse.ArgumentParser(allow_abbrev=False,
        description="Exciton bandstructure E_S(Q) along a K_POINTS crystal_b path")
    ap.add_argument("-i", "--input", required=True,
                    help="cohsex.in with a K_POINTS crystal_b Q-path block")
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-cond", type=int, default=4)
    ap.add_argument("--n-eig", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=40,
                    help="block-Lanczos iterations (Krylov = block·iter)")
    ap.add_argument("--vq-mode", choices=("interp", "refit", "both", "ongrid"),
                    default="interp")
    ap.add_argument("--eqp", type=str, default=None,
                    help="BGW-format eqp1.dat (the one LORRAX's GW writes): "
                         "run the whole BSE on QUASIPARTICLE energies instead "
                         "of DFT.  Both legs are corrected — the stored "
                         "valence/conduction eps from the restart AND the "
                         "htransform's enk_sigma, so the interpolated "
                         "eps_c(k+Q) is a QP band and the on-grid gate "
                         "compares QP against QP.")
    ap.add_argument("--refit-points", type=str, default=None,
                    help="comma list of path indices to refit "
                         "(vq-mode=both; default ~5 evenly spaced)")
    ap.add_argument("--refit-r-chunk", type=int, default=2048,
                    help="r-grid chunk of the refit Z build (vq_interp."
                         "refit_prepare); the per-chunk pair-density temp "
                         "is (nk, nb, nb, r_chunk) c128 — shrink on dense "
                         "k-grids (e.g. 512 at 12x12) to fit device memory")
    ap.add_argument("--a-band", type=int, default=None,
                    help="htransform window-RELATIVE band index whose "
                         "bandwidth sets the f-transform width a = 4*BW "
                         "(bandstructure.htransform._f_params_from_energies). "
                         "Default (None) matches the standard SP driver: a "
                         "from the top band of the full window.  Set this to a "
                         "low-bandwidth conduction band only if the selected "
                         "conduction caches land in the f'->0 compression zone "
                         "(a large default a from a dispersive top guard band "
                         "can collapse off-grid eps_c(k+Q) by eV).")
    ap.add_argument("--alpha", type=float, default=vq_interp.ALPHA)
    ap.add_argument("--eps-tik", type=float, default=vq_interp.EPS_TIK)
    ap.add_argument("--head-minibz-average", action="store_true", default=None,
                    help="Per-Q mini-BZ Coulomb head cell-averaging: replace "
                         "the finite-Q exchange head POINT value v(Q+G*) with "
                         "the mini-BZ cell average <v_LR(Q+G*)>_mBZ (fixes the "
                         "4-13%% near-Γ/zone-boundary head error, "
                         "arbitrary_q_bse.md §16.4).  Overrides the cohsex.in "
                         "``head_minibz_average`` key; default (unset) uses it.")
    ap.add_argument("--eigh-backend", default=None,
                    choices=eigh_backend_choices(),
                    help="OVERRIDES the input-file ``eigh_backend`` key "
                         "(default: use the key, which defaults to auto).  "
                         "Hermitian eigensolver for BOTH distributed-eigh "
                         "sites: the coarse exchange tiles C_q (vq_interp) and "
                         "the htransform fH_q (bse_setup).  auto|off = the "
                         "q-BATCHED native path (every device solves its own "
                         "q-shard).  cusolvermp|slate route ONE tile at a time "
                         "through the distributed-linalg FFI — the regime "
                         "where a single matrix no longer fits on one device "
                         "(a WIDE fH band window), at the cost of nq "
                         "sequential solves.  Needs a square mesh and one JAX "
                         "process per device.")
    ap.add_argument("--px", type=int, default=1)
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--rerun-check", action="store_true",
                    help="RUN the diagnostic warm re-run of the solve scan "
                         "(reproducibility assert + dispatch-only per-Q "
                         "timing).  OFF by default since 2026-08-08: the "
                         "re-run is a full second solve pass and measured "
                         "37.7%% of driver wall at P=4/41 Q.  Ask for it when "
                         "a configuration is new or under suspicion.")
    ap.add_argument("--skip-rerun-check", action="store_true",
                    help="skip the diagnostic warm re-run.  This is now the "
                         "DEFAULT and the flag is a no-op kept so existing "
                         "harnesses and launch recipes keep parsing; it wins "
                         "over --rerun-check if both are passed.")
    ap.add_argument("--extra-q", type=str, default=None,
                    help="';'-separated extra fractional Q (each 'x,y,z') "
                         "appended to the path and solved in the SAME scan "
                         "(one compile, no extra cost per point beyond the "
                         "scan row).  They are NOT plotted; they are written "
                         "to the .dat with mode 'extra'.  The intended use is "
                         "the reference-free SYMMETRY test: pass the "
                         "point-group images of a few off-grid path Q — "
                         "symmetry-equivalent Q must give identical E_S, so "
                         "any disagreement is off-grid interpolation error, "
                         "and its size localises the leg.")
    ap.add_argument("--out-prefix", type=str, default="exciton_bands")
    ap.add_argument("--w-coarse-grid", type=str, default=None,
                    help="NX,NY,NZ — sample the screened W on this COARSE BZ "
                         "sub-grid (a divisor of the WFN/BSE grid), then "
                         "zero-pad W_R back to the fine grid (exact trig "
                         "interpolation) for the direct term.  Enables cheap "
                         "coarse-W + fine exciton sampling.  Default (unset) "
                         "keeps the native fine W byte-identical.")
    return ap


def main(argv=None):
    # ``resolve_eigh_backend`` stays a function-local import for the reason
    # given in ``build_parser``: gw_config parses decks, and this module is
    # imported by things that do not.
    from gw.gw_config import resolve_eigh_backend

    ap = build_parser()
    args = ap.parse_args(argv)

    # ---- Stage timing -----------------------------------------------------
    # DELIBERATELY a driver-local two-column table (``name  seconds``) and NOT
    # ``common.timing.report``: eight live campaign harnesses parse this table
    # with ``grep htransform_psi_cQ | awk '{print $2}'``, and the collector's
    # table puts COUNT in column 2.  Switching formats would leave every one of
    # them reading a small integer as a wall time and reporting it as green —
    # the void-instrument failure mode this campaign has already paid for nine
    # times.  What IS fixed here is completeness: every phase between
    # ``t_wall`` and the report now has a row, and the table closes with an
    # explicit ``(untimed)`` residual so it always sums to TOTAL.  A reader can
    # therefore tell "this accounting is complete" from "43% of the wall is
    # somewhere else" without doing arithmetic — which is exactly what went
    # wrong when job 7882533's 4633 s read as two phases and ~2000 s of
    # mystery (the mystery was ``solve_scan_cold``, a fully-executed pass that
    # WAS in the table; the reader summed the wrong two rows).
    t_wall = time.time()
    timers: dict[str, float] = {}

    def tick(name, t0):
        timers[name] = timers.get(name, 0.0) + (time.time() - t0)
        if _PROBE_TICK is not None:                       # instrument:
            _PROBE_TICK(name, timers[name])               # instrument:

    # Work done BEFORE main(): this module's ``initialize_communicator_stack()`` and every import
    # under it.  75.0 s to first output on a cold Frontera node vs 2.1 s warm
    # (job 7881949), and previously outside the table's clock entirely, so the
    # printed TOTAL could be a minute short of the job's own wall.
    from common.timing import process_elapsed_s as _proc_elapsed
    _pre_main = _proc_elapsed()
    if _pre_main is not None:
        timers["imports_and_runtime"] = _pre_main
        t_wall -= _pre_main
    t_prologue = time.time()

    # Rank-0 I/O guard.  In a multi-node run (one process per GPU) the file
    # writes (.dat / .png) and progress prints must run on process 0 ONLY —
    # otherwise 16 processes race on the same paths.  Non-I/O host numpy (Q
    # path / k-roll construction, the per-Q mini-BZ head QMC) runs redundantly
    # on every process; it is deterministic, so all processes agree and the
    # sharded solve consumes identical operands.  ``log`` is the rank-0 print;
    # it is also threaded into the heavy helpers as ``log_fn`` so their
    # progress is not emitted 16×.
    _rank0 = jax.process_index() == 0

    def log(*a, **k):
        if _rank0:
            print(*a, **k)

    mesh_xy = _create_mesh_xy(args.px, args.py)
    log(f"[dist] jax.device_count()={jax.device_count()} "
        f"process_count()={jax.process_count()} "
        f"local_device_count()={jax.local_device_count()}; "
        f"mesh_xy.shape={dict(mesh_xy.shape)} (px={args.px}, py={args.py})")

    # ── Q path from the ONE K_POINTS crystal_b machinery ─────────────────
    from gw.gw_config import read_lorrax_input
    from bandstructure import htransform as ht
    from bandstructure.bse_setup import compute_wfns_fi

    params = read_lorrax_input(args.input)
    # Input file is the source of truth; the CLI flag is an override — and
    # BOTH go through the one resolver, which is where ``use_low_mem_eigh``
    # folds in.  This driver used to inline the precedence and never call
    # ``resolve_eigh_backend`` at all, so a deck saying
    # ``use_low_mem_eigh = true`` got the native q-batched path anyway.
    # (Both distributed-eigh sites below — compute_wfns_fi and
    # build_vq_evaluator — read this resolved value.)
    args.eigh_backend = resolve_eigh_backend(
        params, override=args.eigh_backend)
    # The INTENT travels to compute_wfns_fi as well: its refusal contract
    # (refuse at resolve time, never fall back to the whole-matrix native
    # path) is armed by this flag, not by the resolved library name.
    _use_low_mem_eigh = bool(params.get("use_low_mem_eigh", False))
    if not params.get("kpoints_crystal_b"):
        raise ValueError(f"{args.input} has no K_POINTS crystal_b block — "
                         "the exciton Q path comes from it (same format as "
                         "the htransform bandstructure driver)")

    # Everything above — the startup call, jax.distributed, mesh creation and its
    # MPI clique warm-up, the input parse — is the driver prologue.  Named
    # rather than left in the residual: at P=16 ``jax.distributed`` init alone
    # measured 43.8 s, and on a cold node the software stack boots off Lustre
    # for 75 s (job 7881949).  Neither is physics and neither had a row.
    tick("prologue", t_prologue)

    # ── load the Q-independent BSE data (production loader) ──────────────
    t0 = time.time()
    restart_file = _find_restart_file(args.input)
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=args.n_val, n_cond=args.n_cond,
        mesh_xy=mesh_xy, input_file=args.input, inject_head=True,
        load_v_full=(args.vq_mode == "ongrid"))
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_val, n_cond = int(data["n_val"]), int(data["n_cond"])
    nv_pad, nc_pad = int(data["n_val_pad"]), int(data["n_cond_pad"])
    n_rmu, n_rmu_pad = int(data["n_rmu"]), int(data["n_rmu_pad"])
    # pad-ε guard on the valence side (loader zero-pads; see module doc).
    # Done ON DEVICE (jnp.where) to preserve the loader's sharding: a host
    # device_get→jnp.asarray round-trip would fail on a process-spanning shard
    # in a multi-node run AND drop the sharding.
    # ── QP energies (--eqp) ───────────────────────────────────────────────
    # BOTH legs of the pair basis have to move together or the diagonal
    # D_Q = eps_c(k+Q) - eps_v(k) mixes QP conduction with DFT valence.  The
    # stored leg is re-sliced here (n_occ is RE-resolved on the corrected
    # energies, so a QP-driven gap change cannot mis-slice); the interpolated
    # leg is corrected below, right after ``initialize_wfns``.
    #
    # ``input_file=None`` on purpose: with it, apply_eqp_corrections asserts the
    # eqp file is IBZ-sized (nk_ibz == sym.nk_red).  LORRAX's own GW writes
    # eqp1.dat on the FULL BZ (one block per k of the WFN, same order), so the
    # energy-matching branch is the correct one and it maps 1:1 here.
    enk_qp_full = None
    if args.eqp:
        from .bse_io import (apply_eqp_and_reslice_bands, apply_eqp_corrections,
                             resolve_n_occ)
        import h5py as _h5py
        with _h5py.File(restart_file, "r") as _f:
            _enk_dft_full = np.asarray(_f["enk_full"][:])
        # n_occ has to come from the WFN's ``ifmax`` (via ``input_file``), but
        # ``input_file`` cannot be handed to apply_eqp_and_reslice_bands — it
        # would reach apply_eqp_corrections' IBZ branch and assert.  Resolve it
        # here and pass it explicitly instead.
        n_occ_in = resolve_n_occ(_enk_dft_full, input_file=args.input)
        data["eps_v"], data["eps_c"], n_occ_qp = apply_eqp_and_reslice_bands(
            restart_file, args.eqp, None, n_val, n_cond, n_occ_in,
            mesh_xy.devices.shape[0], mesh_xy.devices.shape[1])
        enk_qp_full = apply_eqp_corrections(_enk_dft_full, args.eqp)
        _shift_ev = (enk_qp_full - _enk_dft_full) * RY2EV
        log(f"  [eqp] {os.path.basename(args.eqp)}: n_occ={n_occ_qp}, "
            f"QP shifts min/max = {_shift_ev.min():+.4f} / {_shift_ev.max():+.4f} eV; "
            f"BSE runs on QUASIPARTICLE energies")
    if nv_pad > n_val:
        # The loader (and apply_eqp_and_reslice_bands, above) now write the
        # signed guard themselves, so this is no longer a repair — it is the
        # CHECK that they did.  Kept because this driver is where the wrong
        # number was first noticed; a silent regression to a zero ε pad puts
        # spurious transitions BELOW the exciton onset on every BSE driver,
        # not just this one, and this is the cheapest place that would see it.
        _pad_eps_v = jnp.asarray(data["eps_v"])[:, n_val:]
        _worst = float(jnp.max(_pad_eps_v.real))
        if _worst > -0.5 * PAD_EPS_GUARD_RY:
            raise ValueError(
                f"exciton_bands: loader returned an unguarded valence pad — "
                f"max eps_v over the {nv_pad - n_val} pad bands is {_worst:.3e} "
                f"Ry, expected <= {-0.5 * PAD_EPS_GUARD_RY:.3e}. A zero pad "
                f"here makes DeltaE = eps_c - 0 a spurious transition BELOW "
                f"every physical one. See bse_io.PAD_EPS_GUARD_RY.")
    tick("load_bse", t0)

    # ── htransform setup + Q path ────────────────────────────────────────
    t0 = time.time()
    (wfn, sym, meta, _mesh, _S, ctilde, B_at_mu,
     enk_sigma) = ht.initialize_wfns(args.input, params, log,
                                     mesh_xy=mesh_xy)
    if enk_qp_full is not None:
        # The interpolated leg.  ``initialize_wfns(eqp_file=...)`` is NOT used:
        # its ``htransform.read_eqp_energies`` expects the "n=… EQP=…" text
        # form, not the columnar eqp1.dat LORRAX's GW writes, and it swallows
        # the parse failure with a log line — i.e. it would silently leave DFT
        # energies in place.  One parser (``bse_io.read_bgw_eqp``) for both legs.
        _b0 = int(wfn.nelec) - int(params["nval"])
        _b1 = int(wfn.nelec) + int(params["ncond"])
        enk_sigma = jnp.asarray(enk_qp_full[:, _b0:_b1].T)      # (nb, nk) Ry
        log(f"  [eqp] htransform enk_sigma <- QP bands [{_b0},{_b1})")
    kpath_frac, x_path, node_idx, node_labels, _gp = ht.initialize_kpath(
        wfn, params)
    Qpath = np.asarray(kpath_frac, dtype=np.float64)
    nQ_path = Qpath.shape[0]
    # --extra-q rows ride the SAME scan (extra xs rows, one compile).  They are
    # appended to Qpath so every Q-dependent stage — htransform caches, V_Q,
    # the solve — treats them exactly like path points; only the plot and the
    # path-distance axis stop at nQ_path.
    Q_extra = np.zeros((0, 3))
    if args.extra_q:
        Q_extra = np.array([[float(v) for v in seg.split(",")]
                            for seg in args.extra_q.split(";") if seg.strip()],
                           dtype=np.float64)
        Qpath = np.concatenate([Qpath, Q_extra], axis=0)
        log(f"  +{Q_extra.shape[0]} --extra-q point(s) appended to the scan")
    nQ = Qpath.shape[0]
    log(f"Q path: {nQ_path} path points (+{nQ - nQ_path} extra), nodes at "
        f"{list(map(int, node_idx))} labels {node_labels}")
    tick("htransform_setup", t0)

    # ── conduction caches ψ_c(k+Q), ε_c(k+Q) for the whole path ──────────
    # FULL-BAND htransform basis — the single lever that removes the off-grid
    # window-cache ringing.  compute_wfns_fi builds fH from ALL bands in
    # ``ctilde`` (the entire loaded window = input nval+ncond) and only
    # RETURNS the sub-window [b_min, b_max); so a full-band ctilde gives a
    # full-band fH regardless of how few conduction bands the BSE keeps.  With
    # the standard driver's window (nband=40 = 26v+14c) the BSE conduction
    # bands [b_min, b_max) sit strictly INTERIOR, guarded above by the extra
    # conduction bands — every selection boundary stays off any near-
    # degenerate (Kramers) pair.  A SLIVER conduction window whose top
    # boundary cuts a near-degenerate pair instead rings 100-1000 meV off-grid
    # (05_htransform_spbands/gap_scan; Si degeneracy root-cause 73e58f79).
    #
    # But the interp window is TWO-SIDED: too small rings off-grid (above),
    # too LARGE corrupts on-grid.  fH = Σ_n f(ε_n) c_n c_nᴴ recovers energies
    # via eigvals=f(ε_n) ONLY if the Galerkin coeffs c_n are orthonormal; a
    # fixed 640-scale centroid set cannot orthonormalize high oscillatory bands
    # (Gram error → 40% for the top bands), so packing a full 80-band window
    # pollutes eps_c(k+Q) — on-grid |Δε_c| cliffs from ~1 meV (nband≤48) to
    # ~955 meV (nband=80) for MoS2/640c, invisible to the subspace min-sval but
    # caught by the tightened on-grid ENERGY gate.  So keep nband MODEST: a few
    # guard bands above the BSE window (nband≈40-48 for an 8v8c MoS2 run), not
    # maximal.  Reconciliation: 10_lorrax_exciton_bands_80interp_8v8c.
    t0 = time.time()
    nb_window = int(ctilde.shape[1])    # bands in the htransform fH (= input nval+ncond)
    nval_in = int(params["nval"])       # window-relative CBM index (VBM = nval_in-1)
    b_min, b_max = nval_in, nval_in + n_cond
    n_guard = nb_window - b_max         # conduction bands ABOVE the BSE selection
    if b_max > nb_window:
        raise ValueError(
            f"BSE conduction window [{b_min},{b_max}) exceeds the htransform "
            f"fH window ({nb_window} bands): raise nband in {args.input} to "
            f">= {b_max}, or drop --n-cond to <= {nb_window - nval_in}")
    if n_guard < 4:
        log(f"  [warn] only {n_guard} conduction guard band(s) above the BSE "
            f"selection — a selection boundary near a Kramers pair can ring "
            f"off-grid; widen the input's ncond/nband (>= {b_max + 4} bands).")
    if n_guard > 16:
        log(f"  [warn] htransform fH spans {nb_window} bands with {n_guard} "
              f"conduction guards above the BSE window — a LARGE interp window "
              f"does NOT improve (and past a system-dependent cliff WRECKS) the "
              f"returned conduction ENERGIES.  fH=Σf(ε)ccᴴ needs orthonormal "
              f"Galerkin coeffs; 640-scale centroids cannot orthonormalize high "
              f"oscillatory bands (Gram error →40%% for the top bands), so they "
              f"pollute eps_c(k+Q).  MoS2/640c on-grid |Δε_c|: ~1 meV at "
              f"nband≤48, 7 meV at 64, ~955 meV at 80.  min-sval does not see "
              f"this; the on-grid gate does.  Keep nband just above the BSE "
              f"window unless the on-grid gate stays <~20 meV.")
    log(f"  full-band htransform: fH over {nb_window} bands "
        f"({nval_in}v + {nb_window - nval_in}c); BSE conduction "
        f"[{b_min},{b_max}) = {n_cond} band(s) + {n_guard} guard(s)")
    k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                                  np.arange(nkz) / nkz, indexing="ij"),
                      axis=-1).reshape(-1, 3)
    q_list = (Qpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)
    # kgrid_co is the COARSE grid that ``ctilde`` lives on (= the WFN/restart
    # grid, from ``meta``), NOT ``(nkx,nky,nkz)`` — those come from ``data`` and
    # are the FINE grid after a ``bse_k_grid`` coarse→fine densification, which
    # ``k_frac`` above correctly uses for the fine BSE k-sum.  ``build_fH_R``
    # ifft-reshapes ctilde's k-axis into ``kgrid_co``, so it MUST equal
    # ``prod(coarse)``; passing the fine grid crashes (36-k ctilde ≠ 12×12).
    # No-op when bse_k_grid is unset (data grid == meta grid).
    kgrid_co_ct = (int(meta.nkx), int(meta.nky), int(meta.nkz))
    bundle = compute_wfns_fi(
        ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
        kgrid_co=kgrid_co_ct, band_window_fi=(b_min, b_max),
        mesh_xy=mesh_xy, q_list=q_list, a_band_index=args.a_band,
        # ``q_list`` is nQ RIGID SHIFTS of one Γ-centred grid — the line that
        # builds it two statements up says so, and nothing downstream can
        # rediscover it (a block boundary is not a coordinate wrap).  Declaring
        # it lets ``compute_wfns_fi`` evaluate fH_q as one FFT per Q instead of
        # nQ·nk Fourier scans.  The declaration is VERIFIED against ``q_list``
        # there before any route changes, so a future edit to ``Qpath`` or
        # ``k_frac`` that breaks the structure demotes to the scan and says so
        # rather than returning wrong bands.
        q_structure=((nkx, nky, nkz), Qpath),
        eigh_backend=args.eigh_backend,
        use_low_mem_eigh=_use_low_mem_eigh, log_fn=log)
    psi_cQ_X, psi_cQ_Y, eps_cQ = build_conduction_stacks(
        bundle, nQ, nk, n_cond, nc_pad, n_rmu, n_rmu_pad, mesh_xy)
    # Everything the htransform produced is now copied into the conduction
    # stacks and nothing below reads it again.  Drop it before the V_Q model
    # build: ``vq_interp.build_cq`` now returns C_q as a (μ, ν)-face SHARDED
    # device array (no per-proc host gather), but the coarse ζ / P_R
    # intermediates it builds still want the htransform leftovers gone.
    # ``bundle`` alone is ψ at every (Q, k) in both shardings.
    del bundle, ctilde, B_at_mu, enk_sigma
    tick("htransform_psi_cQ", t0)

    # on-grid gate at the first Γ node (path convention: starts at Γ)
    iGamma = [i for i in range(nQ)
              if np.linalg.norm(Qpath[i] - np.round(Qpath[i])) < 1e-9]
    if iGamma:
        # ψ stays ON DEVICE (μ-sharded): the gate contracts μ there and only
        # the (nk, nc, nc) Gram comes to host.  ε is replicated, so it is
        # gathered cheaply inside the gate.  See gate_htransform_vs_stored.
        # TIMED because it is a DIAGNOSTIC on the critical path: it runs one
        # host ``svd`` per k.  A diagnostic is allowed to cost something; it
        # is not allowed to cost something invisibly.
        t0 = time.time()
        gate_htransform_vs_stored(
            psi_cQ_X[iGamma[0]], eps_cQ[iGamma[0]], data, mesh_xy, log=log)
        tick("gamma_gate", t0)

    # ── V_Q tiles ─ ONE shared arbitrary-Q model build.  The bse_k_grid
    #    coarse→fine general init (bse_io) calls the SAME
    #    ``vq_interp.build_vq_evaluator`` — there is a single exchange-interp
    #    orchestration, not one here and one there. ─────────────────────────
    t0 = time.time()
    # Q ON the coarse BZ grid needs NO exchange model at all: the production
    # tile V_qmunu[wrap(−Q)] IS the answer, and the driver already uses exactly
    # that at the one on-grid point it always has (Γ, below).  ``--vq-mode
    # ongrid`` extends that existing special case to every on-grid Q — no
    # interpolation error, no b26p stencil, no ζ.  It also makes the exciton
    # bandstructure runnable on a restart whose ζ is stored IBZ-only (the
    # D3h-orbit-closure cascade), which ``vq_interp`` refuses.  Cost: the full
    # (μ, ν, nkx, nky, nkz) exchange tensor alongside W_q.
    ongrid = (args.vq_mode == "ongrid")
    kgrid_bse = np.array([nkx, nky, nkz], dtype=np.int64)
    if ongrid:
        frac = Qpath * kgrid_bse[None, :]
        off = np.max(np.abs(frac - np.round(frac)))
        if off > 1e-6:
            raise SystemExit(
                f"--vq-mode=ongrid needs every Q on the {nkx}x{nky}x{nkz} BSE "
                f"grid; the path is off by {off:.3e} grid units.  Use a "
                f"K_POINTS block whose segments land on the grid, or "
                f"--vq-mode=interp (which needs FULL-BZ ζ storage).")
        log(f"  exchange: EXACT on-grid tiles V_qmunu[wrap(-Q)] "
            f"({nQ} Q, all on the {nkx}x{nky}x{nkz} grid) — no interpolation")
        head_mbz = False
        zx = prep = eval_vq = pinvF = coeffs_packed = None
    else:
        # Per-Q mini-BZ head cell-averaging: CLI --head-minibz-average overrides
        # the cohsex.in ``head_minibz_average`` key (default off = point value).
        head_mbz = (bool(args.head_minibz_average)
                    if args.head_minibz_average is not None
                    else bool(params.get("head_minibz_average", False)))
        log(f"  arbitrary-Q head: {'mini-BZ cell average' if head_mbz else 'point value'} "
            f"(head_minibz_average={head_mbz})")
        vqm = vq_interp.build_vq_evaluator(
            restart_file, mesh_xy, n_rmu_pad, alpha=args.alpha,
            eps_tik=args.eps_tik, eigh_backend=args.eigh_backend,
            head_minibz_average=head_mbz, log_fn=log)
        zx, prep = vqm.zx, vqm.prep
        eval_vq, pinvF, coeffs_packed = vqm.eval_vq, vqm.pinvF, vqm.coeffs_packed
    tick("vq_prepare", t0)

    grid_xy = NamedSharding(mesh_xy, P("x", "y"))

    def _hermitize(V):
        return 0.5 * (V + jnp.conj(V).T)

    t0 = time.time()
    V_rows = []
    v_gamma = jax.device_put(data["V_q0"], grid_xy)
    n_eval_calls = 0
    for iQ in range(nQ):
        Qw = Qpath[iQ] - np.round(Qpath[iQ])
        if np.linalg.norm(Qw) < 1e-9:
            V_rows.append(_hermitize(v_gamma))       # production q=0 tile
            continue
        q_tile = -Qpath[iQ]                          # tile momentum = wrap(−Q)
        q_tile_np = q_tile - np.round(q_tile)
        if ongrid:
            ix, iy, iz = (np.round(q_tile_np * kgrid_bse).astype(int)
                          % kgrid_bse)
            V_rows.append(_hermitize(
                jax.device_put(data["V_q_full"][:, :, ix, iy, iz], grid_xy)))
            n_eval_calls += 1
            continue
        q_tile = jnp.asarray(q_tile_np)
        if head_mbz:
            gstar, head_val = vq_interp.minibz_head_vlr(
                zx, prep, q_tile_np, alpha=args.alpha)
            V_rows.append(_hermitize(eval_vq(
                q_tile, prep["V_SRc"], pinvF, coeffs_packed,
                jnp.asarray(head_val, dtype=jnp.float64),
                jnp.asarray(gstar, dtype=jnp.int32))))
        else:
            V_rows.append(_hermitize(eval_vq(q_tile, prep["V_SRc"], pinvF,
                                             coeffs_packed)))
        n_eval_calls += 1
    if args.vq_mode == "refit":
        raise SystemExit("--vq-mode=refit alone is not wired; use "
                         "--vq-mode=both (interp path + refit spot checks)")
    refit_idx = []
    if args.vq_mode == "both":
        if args.refit_points:
            refit_idx = sorted({int(s) for s in args.refit_points.split(",")
                                if s.strip() != ""})
        else:
            refit_idx = sorted({int(i) for i in
                                np.linspace(1, nQ - 2, 5).round()})
        rst = vq_interp.refit_prepare(args.input, mesh_xy, zx,
                                      r_chunk=args.refit_r_chunk)
        for iQ in refit_idx:
            q_tile = -Qpath[iQ]
            V_np = vq_interp.refit_vq(zx, rst, q_tile, mesh_xy)
            V_pad = np.zeros((n_rmu_pad, n_rmu_pad), dtype=np.complex128)
            V_pad[:n_rmu, :n_rmu] = 0.5 * (V_np + V_np.conj().T)
            # Process-local (AA.1): V_pad is host numpy, identical on every
            # rank; plain device_put would fire the hidden assert_equal
            # all-gather.  LORRAX_CHECK_REPLICA=1 re-arms it.
            from common.collectives import device_put_process_local
            V_rows.append(device_put_process_local(V_pad, grid_xy))
    n_solve = nQ + len(refit_idx)
    V_stack = jax.device_put(jnp.stack(V_rows),
                             NamedSharding(mesh_xy, P(None, "x", "y")))
    tick("vq_eval", t0)

    # refit rows reuse their Q's conduction caches: extend the scan xs
    if refit_idx:
        sel = jnp.asarray(list(range(nQ)) + refit_idx)
        psi_cQ_X = psi_cQ_X[sel]
        psi_cQ_Y = psi_cQ_Y[sel]
        eps_cQ = eps_cQ[sel]

    # ── W_R once (the ONE sharded-FFT helper), then the single-compile scan ──
    t0 = time.time()
    sh = make_bse_shardings(mesh_xy)
    _ifftn = make_sharded_ifftn_3d(mesh_xy, sh.W.spec, sh.W.spec,
                                   axes=(2, 3, 4), norm="ortho")
    if args.w_coarse_grid is None:
        W_R = _ifftn(data["W_q"])                 # fast path: native fine W (byte-identical)
    else:
        cg = tuple(int(s) for s in args.w_coarse_grid.split(","))
        if len(cg) != 3:
            raise ValueError("--w-coarse-grid expects NX,NY,NZ")
        if cg == (nkx, nky, nkz):
            W_R = _ifftn(data["W_q"])             # equal grids → no-op, byte-identical
        else:
            # Coarse-W → fine direct term.  Sub-sample the fine W_q onto the
            # coarse BZ sub-grid (same ISDF μ-basis; q=0 head-tile preserved),
            # then the ONE sharded densifier (bse_io.make_w_densifier: shard_map
            # ifft to the coarse R-lattice + jitted R zero-pad = exact trig
            # interpolation, (μ,ν) sharding preserved throughout — no eager pad
            # + device_put re-shard).  The convolution then runs on the fine
            # grid with the fine (nkx,nky,nkz) solver — cheap coarse W, fine
            # excitons.
            W_q_coarse = decimate_W_q_to_subgrid(data["W_q"], cg)
            densify_W = make_w_densifier(mesh_xy, sh.W.spec, (nkx, nky, nkz),
                                         output="R")
            W_R = densify_W(W_q_coarse)
            log(f"[coarse-W] W sampled on {cg[0]}x{cg[1]}x{cg[2]} sub-grid of "
                f"{nkx}x{nky}x{nkz}, zero-padded in R (trig-interp to fine grid)")
    solver = build_path_solver(
        mesh_xy, nkx, nky, nkz, nc_pad, nv_pad, n_eig=args.n_eig,
        block_size=args.block_size, max_iter=args.max_iter)
    tick("w_r_and_build", t0)
    t_c0 = time.time()
    evs_dev = solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                     data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R)
    # The block-Lanczos Ritz values come from a small replicated eigh(T) — the
    # scan output is fully addressable on every process, so the rank-0
    # ``device_get`` below reconstructs the whole (n_solve, n_eig) table
    # (no cross-process gather needed).  Logged so the run proves it.
    log(f"[dist] evs sharding={evs_dev.sharding}, "
        f"fully_addressable={evs_dev.is_fully_addressable}")
    evs_all = _gather_host(evs_dev)                   # (n_solve, n_eig) Ry
    t_first = time.time() - t_c0
    tick("solve_scan_cold", t_c0)
    if rerun_check_enabled(args):
        # warm re-run: census-clean per-Q cost + reproducibility assert.
        # Pure diagnostic — it re-executes the ENTIRE Q scan a second time.
        # Measured share of the wall when it is on: 1767.17 s of a 4633.36 s
        # run = 38.1% at P=64/91 Q (job 7882533), 47.74 s of 126.46 s = 37.7%
        # at P=4/41 Q — the single largest row in the table either way.  Which
        # is why it is OFF unless ``--rerun-check`` asks for it; see
        # ``rerun_check_enabled`` for the whole decision.
        t_w0 = time.time()
        evs2 = _gather_host(
            solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                   data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R))
        t_warm = time.time() - t_w0
        tick("solve_scan_warm", t_w0)
        assert np.allclose(evs2, evs_all, atol=1e-10), \
            "scan re-run not reproducible"
        log(f"solve_path: cold {t_first:.2f}s (incl. ONE compile), warm "
            f"{t_warm:.2f}s = {t_warm/n_solve*1e3:.1f} ms/Q over {n_solve} Q")
        log(f"  [diagnostic cost] the warm re-run is a REPRODUCIBILITY CHECK, "
            f"not physics: it re-solves all {n_solve} Q and is "
            f"{100.0*t_warm/max(time.time()-t_wall, 1e-9):.0f}% of the wall so "
            f"far.  Drop --rerun-check once this configuration is trusted.")
    else:
        log(f"solve_path: cold {t_first:.2f}s (incl. ONE compile) over "
            f"{n_solve} Q; warm re-run check SKIPPED (the default since "
            f"2026-08-08 — pass --rerun-check to run it)")
    t0 = time.time()
    mem = solver.lower(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                       data["psi_v_X"], data["psi_v_Y"], data["eps_v"],
                       W_R).compile().memory_analysis()
    log(f"solve_path memory_analysis: temp={mem.temp_size_in_bytes/2**20:.1f} MiB "
        f"args={mem.argument_size_in_bytes/2**20:.1f} MiB "
        f"out={mem.output_size_in_bytes/2**20:.1f} MiB")
    # TIMED: this is an AOT ``lower().compile()`` of the SAME program that was
    # just executed, for a memory report.  Whether it hits XLA's in-process
    # executable cache or recompiles from scratch is an XLA implementation
    # detail, and the difference is the whole compile (~157 s at P=64, 91 Q).
    # A diagnostic that can silently cost a compile gets its own row.
    tick("mem_analysis", t0)
    t_out0 = time.time()

    evs_path = evs_all[:nQ]
    evs_refit = {iQ: evs_all[nQ + j] for j, iQ in enumerate(refit_idx)}

    # ── outputs (rank 0 ONLY — the .dat / .png writes and the plot must not
    #    race across the 16 processes; evs_all is fully addressable on every
    #    process, so rank 0 holds the complete table) ──────────────────────
    if _rank0:
        labels = [(lbl or "") for lbl in node_labels]
        dat = args.out_prefix + ".dat"
        with open(dat, "w", encoding="utf8") as fh:
            fh.write("# Exciton bandstructure E_S(Q), TDA, LORRAX\n")
            fh.write(f"# input: {os.path.abspath(args.input)}\n")
            fh.write(f"# window: n_val={n_val} n_cond={n_cond}; n_eig={args.n_eig}; "
                     f"kgrid {nkx}x{nky}x{nkz}; vq_mode={args.vq_mode}\n")
            fh.write("# conventions: |v k, c k+Q>; exchange tile at wrap(-Q) "
                     "keeps G=0 at finite Q (energy_loss); Gamma uses the "
                     "production q=0 head-body tile; energies in eV\n")
            fh.write(f"# nodes: {' '.join(f'{int(i)}:{l}' for i, l in zip(node_idx, labels))}\n")
            fh.write("# iQ  s_path  Qx  Qy  Qz  mode  E_1..E_neig (eV)\n")
            for iQ in range(nQ):
                row = " ".join(f"{e*RY2EV:.6f}" for e in evs_path[iQ])
                # extras have no path distance: reuse the last path x so the
                # column stays numeric, and flag them in the mode column.
                sx = x_path[iQ] if iQ < nQ_path else x_path[nQ_path - 1]
                mode = "interp" if iQ < nQ_path else "extra "
                fh.write(f"{iQ:4d} {sx:.6f} "
                         f"{Qpath[iQ][0]: .6f} {Qpath[iQ][1]: .6f} "
                         f"{Qpath[iQ][2]: .6f} {mode} {row}\n")
            for iQ in refit_idx:
                row = " ".join(f"{e*RY2EV:.6f}" for e in evs_refit[iQ])
                fh.write(f"{iQ:4d} {x_path[iQ]:.6f} "
                         f"{Qpath[iQ][0]: .6f} {Qpath[iQ][1]: .6f} "
                         f"{Qpath[iQ][2]: .6f} refit  {row}\n")
        print(f"Wrote {dat}")

        if refit_idx:
            print("\ninterp vs refit (ground truth) at spot-check Q:")
            hdr = f"{'iQ':>4} {'s':>8} " + " ".join(
                f"{'dE'+str(j+1)+'(meV)':>10}" for j in range(args.n_eig))
            print(hdr)
            for iQ in refit_idx:
                d = (evs_path[iQ] - evs_refit[iQ]) * RY2EV * 1e3
                print(f"{iQ:4d} {x_path[iQ]:8.4f} "
                      + " ".join(f"{v:10.3f}" for v in d))

        # plot (Agg; no display)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for b in range(args.n_eig):
            ax.plot(x_path, evs_path[:nQ_path, b] * RY2EV, lw=1.2, color="C0",
                    alpha=0.9, label="interp" if b == 0 else None)
        for iQ in refit_idx:
            ax.scatter(np.full(args.n_eig, x_path[iQ]), evs_refit[iQ] * RY2EV,
                       s=22, facecolors="none", edgecolors="C3", zorder=5,
                       label="refit (ground truth)" if iQ == refit_idx[0] else None)
        ticks = x_path[np.asarray(node_idx, dtype=int)]
        for xpos in ticks:
            ax.axvline(xpos, color="k", lw=0.6, alpha=0.3)
        ax.set_xticks(ticks, labels)
        ax.set_xlim(x_path[0], x_path[-1])
        ax.set_ylabel("$E_S(Q)$ (eV)")
        ax.set_title("Exciton bandstructure (TDA)")
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        png = args.out_prefix + ".png"
        fig.savefig(png, dpi=180)
        print(f"Wrote {png}")

        timers["outputs"] = time.time() - t_out0
        total = time.time() - t_wall
        untimed = total - sum(timers.values())
        print("\n--- timings (s) ---")
        for k, v in timers.items():
            print(f"  {k:<22s} {v:9.2f}   {100.0*v/max(total,1e-9):5.1f}%")
        # The residual, ALWAYS printed.  A small number here is the table's own
        # evidence that it is complete; a large one names the gap instead of
        # leaving a reader to discover it by subtraction.
        print(f"  {'(untimed)':<22s} {untimed:9.2f}   "
              f"{100.0*untimed/max(total,1e-9):5.1f}%")
        print(f"  {'TOTAL':<22s} {total:9.2f}   100.0%")

    # LEAVE TOGETHER.  Everything above this line inside ``if _rank0`` — the
    # .dat write, the interp-vs-refit table, matplotlib — is rank-0 only and
    # takes seconds.  Without this barrier ranks 1..P-1 return from main(),
    # exit, and tear their MPI process down while rank 0 is still in
    # matplotlib; MPI aborts the surviving rank and the job exits 134 (SIGABRT)
    # AFTER having written a completely correct .dat and .png.  Measured: job
    # 7882507 cell exb64s, rc=134 with exb_smoke_p64.dat/.png both present and
    # correct.  That is worse than a plain failure, because a harness cannot
    # tell it from one — every multi-rank exciton_bands run scored FAIL on a
    # successful calculation.  ``barrier`` is a no-op at process_count()<=1 and
    # is not on any solver path: one sync, once, after all the physics.
    from common.collectives import barrier
    barrier("exciton_bands.outputs_written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
