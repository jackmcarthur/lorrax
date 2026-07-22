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

# LORRAX distributed bootstrap — SINGLE-SOURCED in ``runtime/`` (the same three
# calls gw.gw_jax / psp.run_nscf use; runtime.__init__ owns the SLURM-aware
# ``jax.distributed.initialize`` pattern).  ``set_default_env()`` must run
# BEFORE ``import jax`` (JAX reads its env at import); ``init_jax_distributed()``
# must run BEFORE any ``jax.devices()`` / mesh creation so a multi-node srun
# (one process per GPU, CUDA_VISIBLE_DEVICES=$SLURM_LOCALID) yields the full
# global device set.  Both are idempotent and no-ops in single-process runs
# (sentinel guard + proc_count<=1), so the 1-GPU path is byte-unchanged.
from runtime import set_default_env
set_default_env()

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()

from solvers.lanczos import block_lanczos_eig_jit
from common.fft_helpers import make_sharded_ifftn_3d
from .bse_io import (_find_restart_file, load_bse_data_from_restart_sharded,
                     decimate_W_q_to_subgrid, pad_W_R_to_grid)
from .bse_ring_comm import make_bse_shardings
from .bse_serial import compute_pair_amplitude
from .bse_stack_matvec import build_bse_stack_matvec
from .bse_w_exact import _create_mesh_xy
from . import vq_interp

RY2EV = 13.6056980659
PAD_EPS_GUARD_RY = 1.0e3


def _gather_host(x):
    """Gather a ``jax.Array`` to a full host numpy array, identical on every
    process, whether it is REPLICATED or PROCESS-SPANNING.

    ``jax.device_get`` raises on an array whose shards live on OTHER processes
    (the diagnostic gate's Q-shifted ψ_c is μ-sharded ``P(...,'x')``, so on a
    16-process / 4-node run no single process holds it all) — those need
    ``multihost_utils.process_allgather``, which stitches the full logical
    array (gathering only the sharded axes) on every process.  But a
    FULLY-ADDRESSABLE array (replicated, or single-process) must NOT go through
    ``process_allgather(tiled=True)`` — that concatenates each process's full
    copy and DUPLICATES the leading axis (e.g. eps_c (144,·) → (16·144,·)).  So
    branch on ``is_fully_addressable``: ``device_get`` when the whole array is
    local, ``process_allgather`` only when shards are remote.  The branch is a
    global property (identical on every process), so the collective stays in
    lockstep.  On one process everything is fully addressable ⇒ plain device_get.
    """
    if getattr(x, "is_fully_addressable", True):
        return np.asarray(jax.device_get(x))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


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
                X = Vb.reshape(block_size, nc_pad, nv_pad, nk)
                X = lax.with_sharding_constraint(X, sh.X)
                HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                            eps_c, eps_v, W_R, V, M_X, M_Y)
                return HX.reshape(block_size, -1)

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


def gate_htransform_vs_stored(psi_cQ_gamma, eps_cQ_gamma, data, log=print):
    """On-grid consistency gate at a Γ path point: htransform conduction
    ε vs the stored restart ε (max |Δ|), and the gauge-free per-k subspace
    overlap min singular value of ⟨ψ_ht|ψ_stored⟩ (bands × bands, spin+μ
    contracted).  Values printed; hard-fails only on gross breakage (>0.5
    overlap loss / >0.1 Ry ε drift) — htransform accuracy is
    centroid-count-governed and reported, not silently trusted."""
    eps_st = _gather_host(data["eps_c"])                  # (nk, nc_pad)
    psi_st = _gather_host(data["psi_c_X"])                # (nk, nc_pad, ns, μ_pad)
    nc = int(data["n_cond"])
    nmu = int(data["n_rmu"])
    eps_ht = np.asarray(eps_cQ_gamma)[:, :nc]
    d_eps = float(np.max(np.abs(eps_ht - eps_st[:, :nc])))
    psi_ht = np.asarray(psi_cQ_gamma)[:, :nc, :, :nmu]
    smin = 1.0
    for k in range(psi_st.shape[0]):
        A = psi_ht[k].reshape(nc, -1)
        B = psi_st[k, :nc, :, :nmu].reshape(nc, -1)
        # normalize rows (htransform ψ normalized in the α metric, stored ψ
        # in the full-r metric — only the SPAN is gauge-free at centroids)
        A = A / np.linalg.norm(A, axis=1, keepdims=True)
        B = B / np.linalg.norm(B, axis=1, keepdims=True)
        s = np.linalg.svd(A.conj() @ B.T, compute_uv=False)
        smin = min(smin, float(s.min()))
    d_meV = d_eps * RY2EV * 1e3
    log(f"  [gate] htransform@Γ vs stored: max|Δε_c| = {d_meV:.3f} meV, "
        f"conduction-subspace overlap min-sval = {smin:.4f}")
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
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
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
    ap.add_argument("--eigh-backend", default="auto",
                    choices=("auto", "off", "cusolvermp", "slate"),
                    help="Hermitian eigensolver for BOTH distributed-eigh "
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
    ap.add_argument("--skip-rerun-check", action="store_true",
                    help="skip the diagnostic warm re-run of the solve scan "
                         "(reproducibility assert + dispatch-only timing); "
                         "the re-run costs a full second solve pass")
    ap.add_argument("--out-prefix", type=str, default="exciton_bands")
    ap.add_argument("--w-coarse-grid", type=str, default=None,
                    help="NX,NY,NZ — sample the screened W on this COARSE BZ "
                         "sub-grid (a divisor of the WFN/BSE grid), then "
                         "zero-pad W_R back to the fine grid (exact trig "
                         "interpolation) for the direct term.  Enables cheap "
                         "coarse-W + fine exciton sampling.  Default (unset) "
                         "keeps the native fine W byte-identical.")
    args = ap.parse_args(argv)

    t_wall = time.time()
    timers: dict[str, float] = {}

    def tick(name, t0):
        timers[name] = timers.get(name, 0.0) + (time.time() - t0)

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
    if not params.get("kpoints_crystal_b"):
        raise ValueError(f"{args.input} has no K_POINTS crystal_b block — "
                         "the exciton Q path comes from it (same format as "
                         "the htransform bandstructure driver)")

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
        _band_ax = jnp.arange(nv_pad)
        data["eps_v"] = jnp.where(_band_ax[None, :] >= n_val,
                                  -PAD_EPS_GUARD_RY, data["eps_v"])
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
    nQ = Qpath.shape[0]
    log(f"Q path: {nQ} points, nodes at {list(map(int, node_idx))} "
        f"labels {node_labels}")
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
        eigh_backend=args.eigh_backend, log_fn=log)
    psi_cQ_X, psi_cQ_Y, eps_cQ = build_conduction_stacks(
        bundle, nQ, nk, n_cond, nc_pad, n_rmu, n_rmu_pad, mesh_xy)
    tick("htransform_psi_cQ", t0)

    # on-grid gate at the first Γ node (path convention: starts at Γ)
    iGamma = [i for i in range(nQ)
              if np.linalg.norm(Qpath[i] - np.round(Qpath[i])) < 1e-9]
    if iGamma:
        gate_htransform_vs_stored(
            _gather_host(psi_cQ_X[iGamma[0]]),
            _gather_host(eps_cQ[iGamma[0]]), data, log=log)

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
            V_rows.append(jax.device_put(jnp.asarray(V_pad), grid_xy))
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
            # ifft to the coarse R-lattice, then ZERO-PAD R back to the fine
            # grid = exact trig interpolation of W(k) that passes through the
            # coarse samples.  The convolution then runs on the fine grid with
            # the fine (nkx,nky,nkz) solver — cheap coarse W, fine excitons.
            W_q_coarse = decimate_W_q_to_subgrid(data["W_q"], cg)
            W_R_coarse = _ifftn(W_q_coarse)
            W_R = pad_W_R_to_grid(W_R_coarse, (nkx, nky, nkz))
            W_R = jax.device_put(W_R, sh.W)       # restore μ/ν sharding on the padded R-tensor
            log(f"[coarse-W] W sampled on {cg[0]}x{cg[1]}x{cg[2]} sub-grid of "
                f"{nkx}x{nky}x{nkz}, zero-padded in R (trig-interp to fine grid)")
    solver = build_path_solver(
        mesh_xy, nkx, nky, nkz, nc_pad, nv_pad, n_eig=args.n_eig,
        block_size=args.block_size, max_iter=args.max_iter)
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
    if not args.skip_rerun_check:
        # warm re-run: census-clean per-Q cost + reproducibility assert.
        # Pure diagnostic — 42% of a 40-pt 12×12 production wall (183 s);
        # skip it with --skip-rerun-check once a configuration is trusted.
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
    else:
        log(f"solve_path: cold {t_first:.2f}s (incl. ONE compile) over "
            f"{n_solve} Q; warm re-run check SKIPPED")
    mem = solver.lower(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                       data["psi_v_X"], data["psi_v_Y"], data["eps_v"],
                       W_R).compile().memory_analysis()
    log(f"solve_path memory_analysis: temp={mem.temp_size_in_bytes/2**20:.1f} MiB "
        f"args={mem.argument_size_in_bytes/2**20:.1f} MiB "
        f"out={mem.output_size_in_bytes/2**20:.1f} MiB")

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
                fh.write(f"{iQ:4d} {x_path[iQ]:.6f} "
                         f"{Qpath[iQ][0]: .6f} {Qpath[iQ][1]: .6f} "
                         f"{Qpath[iQ][2]: .6f} interp {row}\n")
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
            ax.plot(x_path, evs_path[:, b] * RY2EV, lw=1.2, color="C0",
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

        print("\n--- timings (s) ---")
        for k, v in timers.items():
            print(f"  {k:<22s} {v:9.2f}")
        print(f"  {'TOTAL':<22s} {time.time()-t_wall:9.2f}")
    return 0


if __name__ == "__main__":
    main()
