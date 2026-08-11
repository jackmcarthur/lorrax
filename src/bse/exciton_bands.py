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

import h5py
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

from solvers.lanczos import (alpha_herm_sink, block_lanczos_eig_jit,
                             report_alpha_herm, split_alpha_sink)
from common.band_degeneracy import (DEFAULT_MODE, DEGENERACY_TOL_RY, MODES,
                                    check_band_window)
from common.fft_helpers import make_sharded_ifftn_3d
from .bse_io import (_find_restart_file, load_bse_data_from_restart_sharded,
                     decimate_W_q_to_subgrid, make_w_densifier,
                     build_w_head_channel, resolve_w_head_densify,
                     _resolve_head_params, PAD_EPS_GUARD_RY)
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
                      block_size: int, max_iter: int, n_reorth: int | None = None,
                      head_tensor: bool = False):
    """One jitted ``solve_path`` for a whole Q list.

        solve_path(psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q,
                   psi_v_X, psi_v_Y, eps_v, W_R)              (default)
        solve_path(..., W_R, D_head, M_Q)                     (head_tensor)
            -> (evs (nQ, n_eig), alpha (per-Q α-Hermiticity scalars))

    ``head_tensor`` adds the cell-averaged nonanalytic exchange head as a
    rank-three-over-transitions term (``bse_stack_matvec``'s ``head_tensor``).
    ``D_head`` is ``conj(d)`` on the BSE window, ``(nk, nc_pad, nv_pad, 3)``,
    Q-INDEPENDENT — it is the q→0 coefficient of the pair amplitude, a
    property of the k-point transition — while ``M_Q`` is the per-Q cell
    moment ``(nQ, 3, 3)`` and therefore rides the scan xs beside ``V_Q``.
    Default False traces the pre-existing program unchanged.

    The second output is the α-Hermiticity report as DATA: it is what keeps
    this program free of host callbacks and therefore persistable in JAX's
    compilation cache.  Feed it to :func:`_report_alpha_over_path` together
    with ``solve_path.alpha_labels`` (filled at trace time) to run the gate on
    the host.  Nothing about the invariant changes — only where it is emitted.

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
    matvec = build_bse_stack_matvec(mesh_xy, nkx, nky, nkz, kernel="bse",
                                    head_tensor=head_tensor)
    # Filled at TRACE time by the sink below (see the module note on the
    # persistable-scan fix); the static half of the α-Hermiticity report.
    alpha_labels: list = []

    @jax.jit
    def solve_path(psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q,
                   psi_v_X, psi_v_Y, eps_v, W_R, D_head=None, M_Q=None):
        def body(carry, xs):
            if head_tensor:
                psi_c_X, psi_c_Y, eps_c, V, M_head = xs
            else:
                psi_c_X, psi_c_Y, eps_c, V = xs
                M_head = None
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
                extra = (D_head, M_head) if head_tensor else ()
                HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                            eps_c, eps_v, W_R, V, M_X, M_Y, *extra)
                HX = HX.reshape(block_size, -1)
                return HX

            # The α-Hermiticity report leaves the trace as DATA, not as a host
            # callback.  jax._src.compiler._cache_write refuses to persist any
            # module carrying a host callback, and this jit wraps a lax.scan
            # over the whole Q path — the single most expensive compile in the
            # driver.  Un-sunk, it was rebuilt on every warm run.  The sink is
            # opened INSIDE the scan body because that is the trace the solver
            # is traced in; the three scalars per Q ride out as scan ys.
            with alpha_herm_sink() as _sink:
                evs, _ = block_lanczos_eig_jit(
                    matvec_block, n_flat, n_eig=n_eig, block_size=block_size,
                    max_iter=max_iter, n_reorth=n_reorth)
            _labels, _payload = split_alpha_sink(_sink)
            alpha_labels[:] = _labels
            return carry, (evs[:n_eig].real, _payload)

        xs = ((psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q, M_Q) if head_tensor
              else (psi_cQ_X, psi_cQ_Y, eps_cQ, V_Q))
        _, (evs_all, alpha_all) = lax.scan(body, None, xs)
        return evs_all, alpha_all

    solve_path.alpha_labels = alpha_labels
    return solve_path


def _report_alpha_over_path(labels, alpha_all, log=print):
    """Run the α-Hermiticity gate on the per-Q scalars the scan returned.

    ``alpha_all`` is the scan-stacked payload: one ``(dev, scale, worst)``
    triple per label, each of shape ``(nQ,)``.  The gate's verdict is a
    per-solve ratio ``dev/scale`` against ``ALPHA_HERM_RTOL``, so reporting the
    Q that MAXIMISES that ratio reports the worst violator on the whole path —
    a stricter statement than any single Q, in one line instead of nQ lines.
    The Q index is named so a failure is locatable.

    Returns True only if the worst Q on every label passes (and therefore, the
    ratio being maximised, only if every Q passes).
    """
    if not labels:
        return True
    ok = True
    for (name, form), (dev, scale, worst) in zip(labels, alpha_all):
        dev = np.asarray(dev)
        scale = np.asarray(scale)
        worst = np.asarray(worst)
        rel = dev / np.maximum(scale, np.finfo(dev.dtype).tiny)
        iq = int(np.argmax(rel))
        log(f"[alpha-herm] worst of {dev.size} Q is Q#{iq} "
            f"(dev/scale = {rel[iq]:.3e})")
        # ``form`` is a LOOKUP KEY into solvers.lanczos._ALPHA_FORMS, not free
        # text; the per-path context goes in ``name``, which is.
        ok = report_alpha_herm(
            ((f"{name} (worst of {dev.size} Q, at Q#{iq})", form),),
            ((dev[iq], scale[iq], worst[iq]),)) and ok
    return ok


# ===========================================================================
# stacks: htransform conduction caches + V_Q tiles for the whole path
# ===========================================================================
def build_head_dipole_operand(args, nk, nc_pad, nv_pad, n_val, n_cond,
                              *, log=print):
    """``conj(d)`` on the BSE window, ``(nk, nc_pad, nv_pad, 3)`` complex.

    The cell-averaged exchange head is
    ``K^head_{t,t'} = (1/N_k) conj(d_a(t)) M_ab d_b(t')``, and ``d`` is the
    dipole LORRAX already computes and ships in ``dipole.h5``: the head's
    q-linear coefficient is exactly ``∂_q M_cv(k,q,0)|_0 = -i d``, so the
    ``∂_q ζ̃`` the μ-basis route would have needed never appears
    (``LT_HEAD_PROBLEM.md`` §6).

    Returned pre-conjugated and in ``M_X``'s ``(k, c, v, ·)`` layout, with a
    Cartesian axis of length 3 where μ was, so the matvec's head term reads
    exactly as its exchange term (``bse_stack_matvec``, ``head_tensor``).
    The ``(c, v)`` window and its padding match the loader's:
    ``val_indices = arange(n_occ-n_val, n_occ)`` lowest-first, which is the
    same slice ``slice_dipole_to_bse_window`` takes.

    REPRESENTATION CONSISTENCY, STATED RATHER THAN GLOSSED.  The exchange
    BODY lives in the ISDF centroid basis and this head does not — the
    dipoles are exact Cartesian matrix elements on the G-sphere.  That is a
    deliberate, controlled inconsistency (``LT_HEAD_PROBLEM.md`` §6.4): the
    μ-basis reconstruction of ``lim_{q→0} M_cv`` is precisely the quantity
    whose failure to vanish leaves the spurious Γ residual, and the exact
    dipole has no such residual, so the trade favours the dipole.  Two
    consequences follow and are real: head and body no longer cancel each
    other's ISDF error, and no run with the head on is bit-comparable to one
    without it.  That is why the feature is opt-in and why the off arm is
    gated bit-identical.

    The dipoles also carry the velocity-sign convention of the FILE they came
    from.  Since the head is quadratic in ``d`` it inherits that convention
    at full strength.  As of 2026-08-09 the tree's default arm is ``+1`` and
    the tracked fixtures were re-cut on it, so the provenance stamp is read
    and logged rather than assumed.
    """
    from .absorption_common import (build_dipole_vector_bse, load_dipole_h5,
                                    slice_dipole_to_bse_window)
    from .bse_io import resolve_n_occ

    path = args.dipole if getattr(args, "dipole", None) else \
        os.path.join(os.path.dirname(os.path.abspath(args.input)), "dipole.h5")
    dipole_cart, deltaE, attrs = load_dipole_h5(path)
    with h5py.File(path, "r") as f:
        sign = f.attrs.get("prov_vnl_velocity_sign", None)
        skip_vnl = f.attrs.get("skip_vnl", None)
    if int(attrs["nk"]) != int(nk):
        raise SystemExit(
            f"{path} has nk={attrs['nk']} but the BSE grid has nk={nk}.  The "
            f"head tensor contracts the dipole against the BSE transition "
            f"index, so the two k-lists must be the same list in the same "
            f"order.  Rebuild the dipole from this deck's WFN.")
    n_occ = resolve_n_occ(None, input_file=args.input)
    d_alpha, _ = slice_dipole_to_bse_window(dipole_cart, deltaE,
                                            n_occ, n_val, n_cond)
    d_vec = build_dipole_vector_bse(d_alpha, n_cond_pad=nc_pad,
                                    n_val_pad=nv_pad)      # (3, c, v, k)
    log(f"  [head-tensor] dipoles from {os.path.basename(path)} "
        f"(nk={attrs['nk']}, nbands={attrs['nbands']}, n_occ={n_occ}, "
        f"velocity_sign={sign!r}, skip_vnl={skip_vnl!r}); "
        f"max|d|^2 = {float(np.max(np.abs(d_vec)**2)):.4g} bohr^2")
    return np.conj(np.transpose(d_vec, (3, 1, 2, 0)))      # (k, c, v, 3)


def resolve_isdf_basis(restart_file, params, input_file, *, n_rmu_bundle,
                       log=print):
    """``(centroid_table, keep_idx | None)`` — WHICH ISDF basis to fit ψ in.

    On a natively-fitted bundle this is the deck's ``centroids_file`` and
    nothing else happens: the restart's μ basis IS the deck's centroid set,
    and ``keep_idx`` is ``None``.  That arm is byte-for-byte the program this
    driver has always run.

    ON A DOWNFOLDED BUNDLE THE TWO ARE DIFFERENT OBJECTS, and conflating them
    is the second of the three ways this driver failed on one (measured
    2026-08-10, PIPELINE_HEALTH.md).  The small bundle's basis is a SUBSET of
    the parent's centroid points — ``downfold.md``: "the new
    wavefunction-at-centroid coefficients are a literal column slice of the
    ones already on disk" — and the htransform leg does not read the stored
    coefficients, it REFITS ψ against the centroid table it is handed.  That
    fit needs a Galerkin basis spanning ``nk·nb``, which is a completely
    different sizing criterion from the retained window's PAIR-DENSITY rank
    that μ_S was chosen against, and is much larger: 1280 against μ_S = 189
    on the walk's silicon deck, where the fit failed the ``build_fH_R``
    orthonormality gate rather than merely losing accuracy.

    So the basis to FIT in is the parent's, and the answer is then sliced to
    the kept rows — which lands exactly on the columns the bundle stores,
    because that slice is the downfold's own definition of them.  μ_S buys
    the (μ, μ) tensors and the BSE matvec; it does not and cannot buy the
    interpolation fit, and this function is where that is stated.

    The parent's table comes off the bundle's own ``downfold_provenance``,
    not off the deck: a deck copied beside the small bundle names a file that
    is not there (the walk's first failure — a bare ``FileNotFoundError`` out
    of ``np.loadtxt``, with nothing to say it was a downfold consequence).
    The deck is still honoured when it names a readable table of the right
    size, so a user who has arranged their own is not overruled.
    """
    from file_io import read_downfold_provenance

    input_dir = os.path.dirname(os.path.abspath(input_file))

    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(input_dir, p)

    deck_path = _resolve(str(params.get("centroids_file",
                                        "centroids_frac.txt")))
    prov = read_downfold_provenance(restart_file)

    if prov is None:
        if not os.path.isfile(deck_path):
            raise SystemExit(
                f"exciton_bands: the deck's centroids_file resolves to "
                f"{deck_path}, which does not exist.  This driver refits "
                f"psi(k+Q) at the centroid points rather than reading the "
                f"stored coefficients, so it needs the coordinate table the "
                f"restart's ISDF basis was built on.  Point centroids_file "
                f"at it, or copy it beside {os.path.basename(input_file)}.  "
                f"(bse.bse_jax reads this bundle without a centroid table at "
                f"all, which is why the two drivers disagree about whether "
                f"this file is needed.)")
        return deck_path, None

    keep_idx = prov.get("keep_idx")
    mu_parent = int(prov.get("parent_mu", 0))
    if keep_idx is None:
        raise SystemExit(
            f"exciton_bands: {restart_file} carries a downfold_provenance "
            f"group with no keep_idx dataset, so there is no way to map its "
            f"mu={n_rmu_bundle} basis back to the parent's centroid points.  "
            f"The bundle predates the stamp; re-run gw.downfold_cli on the "
            f"parent to produce one this driver can open.")
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    if keep_idx.shape[0] != int(n_rmu_bundle):
        raise SystemExit(
            f"exciton_bands: {restart_file} stores mu={n_rmu_bundle} but its "
            f"downfold_provenance keeps {keep_idx.shape[0]} parent centroid "
            f"rows.  The bundle and its provenance describe different bases; "
            f"re-run the downfold rather than trusting either.")

    # The parent table: the bundle's own record first, the deck second.
    cands = []
    stamped = prov.get("parent_centroids_file")
    if stamped:
        cands.append((str(stamped), "downfold_provenance"))
    cands.append((deck_path, "the deck's centroids_file"))
    for path, whence in cands:
        if not os.path.isfile(path):
            continue
        n_rows = int(np.loadtxt(path, ndmin=2).shape[0])
        if mu_parent and n_rows != mu_parent:
            log(f"  [downfold] {path} ({whence}) has {n_rows} rows, not the "
                f"parent's {mu_parent} — not the table this bundle was "
                f"downfolded from; skipped.")
            continue
        log(f"  [downfold] this bundle is a DOWNFOLD of {prov.get('parent_file', '?')} "
            f"(mu {mu_parent} -> {n_rmu_bundle}).  The htransform leg refits "
            f"psi in the PARENT basis from {path} ({whence}) and the result "
            f"is sliced to the {keep_idx.shape[0]} kept centroid rows — the "
            f"same column slice that defines the bundle's own psi_full_y.  "
            f"mu_S sizes the (mu,mu) tensors and the BSE matvec; the "
            f"interpolation fit is sized by nk*nb and cannot use it.")
        return path, keep_idx

    raise SystemExit(
        f"exciton_bands: {restart_file} is a downfolded bundle (mu "
        f"{mu_parent} -> {n_rmu_bundle}) and this driver needs the PARENT "
        f"run's centroid table to refit psi(k+Q); none of "
        f"{[c[0] for c in cands]} is a readable {mu_parent}-row table.\n"
        f"  Why the parent's and not the small one: the htransform Galerkin "
        f"fit needs a basis spanning nk*nb, which is a different (and much "
        f"larger) criterion than the retained window's pair-density rank "
        f"that mu_S was chosen against.  The fit runs at the parent's mu and "
        f"its output is then sliced to this bundle's kept rows.\n"
        f"  Fix: re-run gw.downfold_cli with parent_centroids_file set to "
        f"the parent deck's centroids_file (the driver then records its path "
        f"on the bundle and this resolves itself), or point this deck's "
        f"centroids_file at that table directly.")


def apply_q_per_segment(params, q_per_segment, log=print) -> int:
    """Raise every ``K_POINTS crystal_b`` segment to at least N points.

    IT IS A BANDSTRUCTURE, SO SAMPLE IT LIKE ONE.  The deck format carries a
    per-segment point count, and the decks in this campaign carried counts of
    one and two — which draws Γ–X–W–L–Γ–Σ with eight diagonalisations in
    total and produces a plot of straight lines between corners.  The owner's
    reading of that plot ("it looks like shit ... i thought that was
    obvious") is the reason this floor exists and is on by default.

    A FLOOR, not an override, and the distinction is the whole gate:

      * a segment the deck leaves sparser than ``q_per_segment`` is raised to
        it, so the DEFAULT (16) draws at least 16 diagonalisations per
        segment on every deck that ever ran, without editing any deck;
      * a segment the deck already asks for MORE points on keeps its own
        count, so a hand-tuned dense deck is not silently coarsened;
      * ``--q-per-segment 1`` is the identity — every deck count is already
        >= 1 — so the sparse behaviour is still reachable, by asking for it
        explicitly and only that way.

    Only ``bse.exciton_bands`` calls this.  The mutation is on the parsed
    ``params`` dict this driver owns for the length of one run, so the
    single-particle bandstructure driver reading the same deck is untouched.

    Returns the number of path points the raised path will carry.
    """
    seginfo = params.get("kpoints_crystal_b") or {}
    segs = seginfo.get("segments", [])
    if len(segs) < 2:
        return 0
    floor = int(q_per_segment)
    if floor < 1:
        raise SystemExit(
            f"exciton_bands: --q-per-segment={floor} is not a point count.  "
            f"1 is the identity (every deck count is already >= 1) and 16 is "
            f"the default.")
    # segs[0] is the path's first corner; segs[i>0] carries the count for the
    # segment that ENDS at it (generate_kpath_from_qe_segments).
    deck_counts = [max(1, int(s.get("n", 1))) for s in segs[1:]]
    for seg, n_deck in zip(segs[1:], deck_counts):
        seg["n"] = max(n_deck, floor)
    raised = [(i, a, max(a, floor))
              for i, a in enumerate(deck_counts, start=1) if a < floor]
    n_pts = 1 + sum(max(a, floor) for a in deck_counts)
    if raised:
        log(f"  [q-per-segment] floor {floor}: raised {len(raised)} of "
            f"{len(deck_counts)} segment(s) "
            f"({', '.join(f'#{i}: {a}->{b}' for i, a, b in raised)}); "
            f"the path now carries {n_pts} Q "
            f"(deck alone: {1 + sum(deck_counts)}).  It is a bandstructure — "
            f"pass --q-per-segment 1 for the deck's own counts.")
    else:
        log(f"  [q-per-segment] floor {floor}: every deck segment already "
            f"asks for at least that many; path unchanged at {n_pts} Q.")
    return n_pts


def require_zeta_for_interp(restart_file, vq_mode, kgrid) -> str:
    """The ζ file the off-grid exchange needs — or a refusal that explains it.

    ANNOUNCE OR REFUSE, BEFORE h5py GETS TO SAY IT ITS WAY.
    ``vq_interp.build_vq_evaluator`` looks for ``zeta_q.h5`` beside the
    restart, and a missing one is not an exotic condition: it is what EVERY
    downfolded bundle written before 2026-08-10 looks like, and what one
    whose parent's ζ was stored on the q-IBZ wedge still looks like.  The
    bare ``OSError`` from that path names a file and nothing else — no hint
    that the absence is a downfold consequence, and no hint that
    ``--vq-mode=ongrid`` needs no ζ at all.  Third of the three ways this
    driver failed on a downfolded bundle (PIPELINE_HEALTH.md, 2026-08-10).

    A separate function rather than an inline block so the refusal can be
    gated without a solve — same reason ``rerun_check_enabled`` is one.
    """
    if vq_mode == "ongrid":
        return ""
    path = os.path.join(os.path.dirname(os.path.abspath(restart_file)),
                        "zeta_q.h5")
    if os.path.isfile(path):
        return path
    prov = None
    try:
        from file_io import read_downfold_provenance
        prov = read_downfold_provenance(restart_file)
    except Exception:                                          # noqa: BLE001
        pass
    nx, ny, nz = (int(v) for v in kgrid)
    raise SystemExit(
        f"exciton_bands: --vq-mode={vq_mode} builds the exchange tile at "
        f"arbitrary Q by interpolating the stored zeta, and there is no "
        f"{path}.\n"
        + ("  This bundle is a DOWNFOLD.  gw.downfold_cli transports zeta "
           "into the small basis (zeta_S = conj(T[q]) zeta_L) and writes it "
           "beside the bundle, so a bundle without one was either written "
           "before that existed or has a PARENT whose zeta is stored on the "
           "q-IBZ wedge — which vq_interp refuses on the parent too, so "
           "--vq-mode=interp was never available on this lineage.  Re-fit "
           "the parent with LORRAX_FORCE_FULL_BZ=1 and re-run the "
           "downfold.\n"
           if prov is not None else
           "  The GW run that wrote this bundle left no zeta_q.h5 beside it "
           "(a moved tmp/, or a restart written without the fit).\n")
        + f"  --vq-mode=ongrid needs no zeta at all and is EXACT at every Q "
          f"that lands on the {nx}x{ny}x{nz} BSE grid.")


def build_conduction_stacks(bundle, nQ, nk, n_cond, n_cond_pad, n_rmu,
                            n_rmu_pad, mesh_xy, *, keep_idx=None):
    """Reshape the htransform bundle over the concatenated {k+Q} list into
    per-Q conduction caches, padded to the loader's mesh extents — one
    jitted reshape+pad+reshard, no host round-trip (the bundle stays on
    device; at 40 path points the old device_get→np.pad→2×device_put moved
    ~1.7 GB through the host).

    ``keep_idx`` (downfolded bundles only) is the column slice from the
    parent ISDF basis the htransform fitted in to the basis the RESTART
    stores — see :func:`resolve_isdf_basis` for why those differ.  It runs
    inside the same jit, before the pad, so the parent-width ψ never lands
    in a second host buffer and the μ axis reaches its sharding constraint
    already at the small extent.

    THE RESTART'S μ IS THE AUTHORITY, and the assertion below says so: the
    htransform's basis is an input to this function, ``n_rmu`` is what every
    other operand in the solve was sized by, and a mismatch is a wrong-basis
    contraction that every shape check downstream would pass once the pad
    absorbed it.  (Same shape of reasoning as the post-snap window authority
    at 7449ece0: one place resolves the extent, everything else asserts
    against it rather than re-deriving it.)

    Returns (psi_cQ_X, psi_cQ_Y, eps_cQ):
        psi_cQ_[XY]: (nQ, nk, nc_pad, ns, n_rmu_pad), μ on x / y
        eps_cQ:      (nQ, nk, nc_pad) — pad bands at +PAD_EPS_GUARD_RY
    """
    x5 = NamedSharding(mesh_xy, P(None, None, None, None, "x"))
    y5 = NamedSharding(mesh_xy, P(None, None, None, None, "y"))
    rep = NamedSharding(mesh_xy, P())

    # HOST numpy, closed over — NOT a ``jnp.asarray`` outside the jit.  An
    # eager ``device_put`` of a host array is the AA.1 hidden-all-gather site
    # this driver already documents at its refit rows; a numpy array closed
    # over by the trace becomes an HLO constant instead, which is process-local
    # by construction and costs nothing at any P.
    n_rmu_src = int(bundle.psi_rmu_Y.shape[-1])
    keep = None if keep_idx is None else np.asarray(keep_idx, dtype=np.int32)
    n_after = n_rmu_src if keep is None else int(keep.shape[0])
    if n_after != int(n_rmu):
        raise ValueError(
            f"exciton_bands: the htransform fitted psi at {n_rmu_src} "
            f"centroid point(s)"
            + ("" if keep is None else
               f", of which {n_after} are kept")
            + f", but the restart bundle stores mu={n_rmu}.  The conduction "
              f"caches would be contracted against a W and V in a different "
              f"basis.  The restart's mu is the authority here: point the "
              f"deck's centroids_file at the table this bundle's basis came "
              f"from (a downfolded bundle records the parent's path in its "
              f"downfold_provenance group).")

    @partial(jax.jit, out_shardings=(x5, y5, rep))
    def _stacks(psi, eps):
        ns = psi.shape[2]
        psi = psi.reshape(nQ, nk, n_cond, ns, n_rmu_src)
        if keep is not None:
            psi = jnp.take(psi, keep, axis=4)
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
    k_worst = -1
    sv_worst = None
    for k in range(G.shape[0]):
        sv = np.linalg.svd(G[k], compute_uv=False)
        if float(sv.min()) < smin:
            smin, k_worst, sv_worst = float(sv.min()), k, sv
    d_meV = d_eps * RY2EV * 1e3
    d_meV_idx = d_eps_idx * RY2EV * 1e3
    log(f"  [gate] htransform@Γ vs stored: max|Δε_c| = {d_meV:.6f} meV "
        f"(order-matched; this is the accuracy number), "
        f"conduction-subspace overlap min-sval = {smin:.4f}")
    if sv_worst is not None:
        # The WHOLE spectrum at the worst k, because the min alone is not
        # diagnostic: svals all ≈ s < 1 is a normalisation/weight error, while
        # svals ≈ (1, …, 1, s) is one genuinely missing subspace direction —
        # the window boundary cutting a degenerate multiplet.  Different bugs,
        # different fixes, and the gate used to print a number that could be
        # either.
        log(f"  [gate] overlap svals at the worst k (k={k_worst}): "
            f"{np.array2string(np.asarray(sv_worst), precision=4)}")
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
    ap.add_argument("--band-degeneracy", choices=MODES, default=DEFAULT_MODE,
                    help="what to do when --n-val/--n-cond cut a degenerate "
                         "multiplet (Kramers pair under SOC+TRS, any irrep "
                         "of dimension > 1): 'strict' (the default since "
                         "2026-08-10) refuses, naming the counts that would "
                         "work; 'snap' widens the window OUTWARD to the "
                         "multiplet boundary and says so loudly; 'off' "
                         "proceeds on the cut multiplet.  Half a multiplet "
                         "breaks the exciton multiplets by an amount with no "
                         "convergence parameter (Si 12x12 SOC: eV-scale "
                         "off-grid), and a widened window is a different "
                         "calculation, so widening is something you ask for.")
    ap.add_argument("--degeneracy-tol-ry", type=float,
                    default=DEGENERACY_TOL_RY, dest="degeneracy_tol_ry",
                    help=f"'same multiplet' tolerance in Ry (default "
                         f"{DEGENERACY_TOL_RY:.4e} = 1 meV; the smallest "
                         f"between-pair boundary gap measured on the Si "
                         f"12x12 SOC deck was 5.9 meV)")
    ap.add_argument("--q-per-segment", type=int, default=16,
                    dest="q_per_segment",
                    help="MINIMUM number of Q the driver puts on each "
                         "K_POINTS crystal_b segment (default 16).  It is a "
                         "FLOOR on the deck's own per-segment counts, not an "
                         "override: a segment the deck asks for MORE points "
                         "on keeps its own count, and '--q-per-segment 1' is "
                         "the identity, which is the only way to get the "
                         "deck's counts verbatim.  The default exists "
                         "because E_S(Q) along a high-symmetry path is a "
                         "BANDSTRUCTURE: the decks in tree carried counts of "
                         "1 and 2, which drew Γ-X-W-L-Γ-Σ with eight "
                         "diagonalisations and straight lines between them.  "
                         "Each extra Q is one more row of the SAME compiled "
                         "solve scan (no extra compile), so density is "
                         "cheap; it is the htransform ψ_c(k+Q) cache that "
                         "grows linearly, and that is what to watch on a "
                         "densified (bse_k_grid) bundle.")
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
    ap.add_argument("--dipole", default=None,
                    help="dipole.h5 for the mini-BZ head TENSOR (only read "
                         "when head_minibz_average is ON).  Default: "
                         "dipole.h5 beside the input file.  The head's "
                         "q-linear coefficient IS this dipole, which is why "
                         "the cell-averaged head needs no d(zeta)/dq.")
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
    ap.add_argument("--w-head-densify", type=str, default=None,
                    choices=("c1", "legacy"),
                    help="How W's Γ head crosses the coarse→fine densifier. "
                         "'c1' (default) splits it off first and re-attaches "
                         "it analytically per fine q — the head is never "
                         "interpolated, which is what BerkeleyGW's kernel.x / "
                         "intkernel split exists to guarantee.  'legacy' lets "
                         "the Kronecker-delta head ride through the "
                         "trigonometric interpolant: that is the documented "
                         "defect and this arm exists ONLY as the A/B control "
                         "that prices it.  No effect without "
                         "--w-coarse-grid.")
    ap.add_argument("--w-head-gamma-cell", type=str, default="fine",
                    choices=("fine", "coarse"),
                    help="Which mini-BZ cell the re-attached Γ head is "
                         "averaged over.  'fine' (default) is correct.  "
                         "'coarse' is the design's RED TWIN — invisible when "
                         "the grids are equal, and it breaks the head sum "
                         "rule everywhere else.  For gate work only.")
    return ap


def _gamma_only_grid(kgrid, value):
    """A head-scalar grid that is ``value`` at Γ and zero everywhere else.

    The array shape :func:`gw.head_densify.attach_head_channel` consumes, used
    here to SUBTRACT the head the loader already injected (pass a negative
    value) so the densifier's operand is the body alone.  Spelled through the
    same attach helper rather than as a bespoke ``.at[...,0,0,0].add()`` so the
    detach and the re-attach cannot disagree about which slice, which
    projector, or which 1/Ω the head lives on — a mismatched pair here would
    leave a residual rank-one term with no shape error to catch it.
    """
    S = np.zeros(tuple(int(s) for s in kgrid), dtype=np.float64)
    S[0, 0, 0] = float(value)
    return S


def _resolve_native_w_head(restart_file, input_file, wfn, *, log=print):
    """The ``whead`` the loader injected into this NATIVE-grid restart's W.

    ``--w-coarse-grid`` decimates a natively fine W, so the head sitting in
    ``data["W_q"][:,:,0,0,0]`` is the FINE cell's average — the number C1 has
    to remove before sub-sampling and put back after densifying.  Returns
    ``None`` (with a reason logged) when there is no head to move, in which
    case the caller runs the plain densifier and nothing is lost.
    """
    try:
        import h5py
        with h5py.File(restart_file, "r") as f:
            whead_restart = (np.asarray(f["whead"][:], dtype=np.complex128)
                             if "whead" in f else None)
            w0_ready = bool(f["W0_qmunu"].attrs.get("W0_ready", False)
                            if "W0_qmunu" in f else False)
    except Exception as exc:
        log(f"[coarse-W] cannot read the restart's head ({exc}); "
            f"falling back to w_head_densify=legacy for this run")
        return None
    if not w0_ready:
        log("[coarse-W] the restart carries no ready screened W0, so the "
            "loader injected no whead and there is no head to split off; "
            "the densifier operand is already head-free")
        return None
    _vhead, whead, cell_volume, _src = _resolve_head_params(
        input_file, None, whead_restart, float(wfn.cell_volume))
    if whead is None or cell_volume is None:
        log("[coarse-W] no whead resolved (no deck override, no restart "
            "scalar); the densifier operand is already head-free")
        return None
    return {"whead": float(complex(whead[0]).real),
            "cell_volume": float(cell_volume)}


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
    # Sampling density BEFORE the path is generated: a floor on the deck's
    # per-segment counts, default 16, so the default output is a
    # bandstructure rather than a line drawing between corners.  See
    # apply_q_per_segment for why this is a floor and not an override.
    apply_q_per_segment(params, args.q_per_segment, log=log)

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
        load_v_full=(args.vq_mode == "ongrid"),
        degeneracy_mode=args.band_degeneracy,
        degeneracy_tol_ry=args.degeneracy_tol_ry)
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
            mesh_xy.devices.shape[0], mesh_xy.devices.shape[1],
            degeneracy_mode=args.band_degeneracy,
            degeneracy_tol_ry=args.degeneracy_tol_ry)
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

    # ── which ISDF basis the htransform fits in ──────────────────────────
    # BEFORE initialize_wfns, because that call is the one that reads the
    # centroid table.  On a plain bundle this returns the deck's own path
    # and ``None``, and the program below is unchanged; on a downfolded one
    # it returns the PARENT's table and the column slice back to this
    # bundle's basis.  See resolve_isdf_basis for why those are different
    # questions and why the deck alone cannot answer the second.
    centroids_path, keep_idx = resolve_isdf_basis(
        restart_file, params, args.input, n_rmu_bundle=n_rmu, log=log)
    params["centroids_file"] = centroids_path

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
    # The htransform window is the SECOND place a band boundary is cut in this
    # driver, and it is cut in a different index space (window-relative, not
    # absolute), so the loader's snap does not automatically make it safe.
    # ``enk_sigma`` is (nb, nk) in the SAME window-relative indexing as
    # b_min/b_max, so the check is exact here.  Report-only: b_max is pinned to
    # the already-resolved n_cond, and widening it here would desynchronise the
    # conduction caches from the BSE window the loader sized.  The prose above
    # has warned about exactly this failure since the Si root-cause; this makes
    # it a measurement instead of a warning about a possibility.
    check_band_window(
        np.asarray(enk_sigma).T, b_min, b_max,
        tol_ry=args.degeneracy_tol_ry, mode=args.band_degeneracy,
        where="exciton_bands htransform conduction window", log=log)
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
        eigh_backend=args.eigh_backend,
        use_low_mem_eigh=_use_low_mem_eigh, log_fn=log)
    psi_cQ_X, psi_cQ_Y, eps_cQ = build_conduction_stacks(
        bundle, nQ, nk, n_cond, nc_pad, n_rmu, n_rmu_pad, mesh_xy,
        keep_idx=keep_idx)
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
    # ``refit`` used to refuse ("not wired") and to be unusable anyway: its
    # kernel was the slab one.  Since 2026-08-10 it is the exact arbitrary-Q
    # exchange on a bulk deck — the ONLY one, because the interpolation model
    # is slab-only — and it is what a dense Q path on a 3-D crystal runs.
    pure_refit = (args.vq_mode == "refit")
    kgrid_bse = np.array([nkx, nky, nkz], dtype=np.int64)
    # WHICH GRID INDEXES THE EXCHANGE TILES.  Normally the bundle's own.  After
    # a ``bse_k_grid`` coarse→fine densification the bundle's grid is the FINE
    # one but the stored tiles are still the COARSE ones — exact where they
    # exist (V_{μν}(q) is a ζ/G-sphere object and never sees the k-grid) and
    # simply ABSENT at a fine q that is not a coarse q.  So the on-grid Q set
    # is the coarse grid on a densified bundle, and the refusal below says so
    # instead of letting a ``None`` subscript speak for it.
    V_ongrid = data.get("V_q_full")
    kgrid_vq = kgrid_bse
    vq_src = "the bundle grid"
    if V_ongrid is None and data.get("V_q_coarse") is not None:
        V_ongrid = data["V_q_coarse"]
        kgrid_vq = np.asarray(data["V_q_coarse_grid"], dtype=np.int64)
        vq_src = "the COARSE restart grid (bse_k_grid densification)"
    if ongrid:
        if V_ongrid is None:
            raise SystemExit(
                "--vq-mode=ongrid needs the stored exchange tensor V_qmunu and "
                "this bundle carries none.")
        frac = Qpath * kgrid_vq[None, :]
        off = np.max(np.abs(frac - np.round(frac)))
        if off > 1e-6:
            raise SystemExit(
                f"--vq-mode=ongrid needs every Q on the "
                f"{kgrid_vq[0]}x{kgrid_vq[1]}x{kgrid_vq[2]} EXCHANGE-TILE grid "
                f"({vq_src}); the path is off by {off:.3e} grid units.  On a "
                f"bse_k_grid-densified bundle that grid is the COARSE restart "
                f"grid, not the fine BSE grid "
                f"({nkx}x{nky}x{nkz}): densifying moves the k-SUM, it does not "
                f"create exchange tiles at new q.  Use a K_POINTS block whose "
                f"segments land on the coarse grid, or --vq-mode=interp (which "
                f"needs FULL-BZ ζ storage and an exchange model valid for this "
                f"cell).\n"
                f"  CHECK --q-per-segment FIRST if this deck used to run: it "
                f"floors every segment at {args.q_per_segment} points by "
                f"default (2026-08-10 — E_S(Q) on a path is a bandstructure), "
                f"and a floor above ~2 puts most Q off any coarse grid.  "
                f"--q-per-segment 1 restores the deck's own counts exactly; "
                f"the dense path is what needs the full-BZ ζ.")
        log(f"  exchange: EXACT on-grid tiles V_qmunu[wrap(-Q)] "
            f"({nQ} Q, all on the "
            f"{kgrid_vq[0]}x{kgrid_vq[1]}x{kgrid_vq[2]} grid = {vq_src}) "
            f"— no interpolation")
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
        zeta_path = require_zeta_for_interp(restart_file, args.vq_mode,
                                            (nkx, nky, nkz))
        if pure_refit:
            # NO interpolation model is built at all.  The b26p long-range fit
            # is the slab-only half of vq_interp; the refit path fits ζ at the
            # target Q from the htransform ψ and contracts it with the
            # producer's own Coulomb door.  So this branch loads ζ and stops.
            zx = vq_interp.load_zeta_coarse(restart_file, zeta_path,
                                            mesh=mesh_xy, log_fn=log,
                                            require_slab=False)
            prep = eval_vq = pinvF = coeffs_packed = None
            if head_mbz:
                raise SystemExit(
                    "exciton_bands: --head-minibz-average needs the "
                    "interpolation model's long-range pieces (minibz_head_vlr "
                    "reads prep), which --vq-mode=refit does not build.  The "
                    "refit keeps the full v(Q+G) including G=0 at every Q, "
                    "which is the pointwise head this flag would replace.")
        else:
            vqm = vq_interp.build_vq_evaluator(
                restart_file, mesh_xy, n_rmu_pad, alpha=args.alpha,
                eps_tik=args.eps_tik, eigh_backend=args.eigh_backend,
                head_minibz_average=head_mbz, log_fn=log)
            zx, prep = vqm.zx, vqm.prep
            eval_vq, pinvF, coeffs_packed = (vqm.eval_vq, vqm.pinvF,
                                             vqm.coeffs_packed)
    tick("vq_prepare", t0)

    # ── the per-Q refit state + ITS GATE, before any tile is used ─────────
    rst = None
    if pure_refit:
        t0 = time.time()
        rst = vq_interp.refit_prepare(args.input, mesh_xy, zx,
                                      r_chunk=args.refit_r_chunk, log_fn=log,
                                      policy=zx.get("policy"),
                                      keep_idx=keep_idx)
        if V_ongrid is None:
            raise SystemExit(
                "exciton_bands: --vq-mode=refit certifies itself against the "
                "stored V_qmunu at on-grid q and this bundle carries none, so "
                "there is nothing to check the off-grid tiles against.  Load a "
                "bundle with the exchange tensor, or use --vq-mode=ongrid.")
        vq_interp.refit_ongrid_null(zx, rst, V_ongrid, kgrid_vq, mesh_xy,
                                    log_fn=log)
        tick("refit_prepare_and_null", t0)

    grid_xy = NamedSharding(mesh_xy, P("x", "y"))

    def _hermitize(V):
        return 0.5 * (V + jnp.conj(V).T)

    t0 = time.time()
    V_rows = []
    # Per-Q cell moment for the tensor head.  Zero for Γ and for every Q on
    # the OFF arm, which makes the head term an exact no-op there — the Γ
    # endpoint keeps the production q=0 tile and its rank-one head, as its
    # own docstring promises.
    M_rows = np.zeros((nQ, 3, 3), dtype=np.float64)
    head_scalars: list = []
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
            ix, iy, iz = (np.round(q_tile_np * kgrid_vq).astype(int)
                          % kgrid_vq)
            V_rows.append(_hermitize(
                jax.device_put(V_ongrid[:, :, ix, iy, iz], grid_xy)))
            n_eval_calls += 1
            continue
        if pure_refit:
            V_np = vq_interp.refit_vq(zx, rst, q_tile_np, mesh_xy, log_fn=log)
            V_pad = np.zeros((n_rmu_pad, n_rmu_pad), dtype=np.complex128)
            V_pad[:n_rmu, :n_rmu] = 0.5 * (V_np[:n_rmu, :n_rmu]
                                           + V_np[:n_rmu, :n_rmu].conj().T)
            from common.collectives import device_put_process_local
            V_rows.append(device_put_process_local(V_pad, grid_xy))
            n_eval_calls += 1
            continue
        q_tile = jnp.asarray(q_tile_np)
        if head_mbz:
            gstar, head_val, M_ab = vq_interp.minibz_head_vlr(
                zx, prep, q_tile_np, alpha=args.alpha, moment=True)
            # The head channel leaves the mu basis entirely: v[gstar] -> 0
            # removes the LR G* column from the tile, and the cell-averaged
            # head comes back as the rank-three tensor term in the matvec.
            # Injecting BOTH would double-count the head.
            M_rows[iQ] = M_ab
            V_rows.append(_hermitize(eval_vq(
                q_tile, prep["V_SRc"], pinvF, coeffs_packed,
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(gstar, dtype=jnp.int32))))
            head_scalars.append((iQ, float(head_val), float(np.trace(M_ab))))
        else:
            V_rows.append(_hermitize(eval_vq(q_tile, prep["V_SRc"], pinvF,
                                             coeffs_packed)))
        n_eval_calls += 1
    if pure_refit:
        log(f"  exchange: per-Q ζ REFIT at all {n_eval_calls} finite Q "
            f"(compute-don't-interpolate; Γ keeps the production q=0 tile), "
            f"certified against the stored V_qmunu by the on-grid null above")
    refit_idx = []
    if args.vq_mode == "both":
        if args.refit_points:
            refit_idx = sorted({int(s) for s in args.refit_points.split(",")
                                if s.strip() != ""})
        else:
            refit_idx = sorted({int(i) for i in
                                np.linspace(1, nQ - 2, 5).round()})
        rst = vq_interp.refit_prepare(args.input, mesh_xy, zx,
                                      r_chunk=args.refit_r_chunk,
                                      policy=zx.get("policy"),
                                      keep_idx=keep_idx)
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
    if head_mbz:
        # refit rows carry no head tensor (they are the ground-truth
        # point-value comparison); zero is an exact no-op in the term.
        M_stack = np.concatenate(
            [M_rows, np.zeros((len(refit_idx), 3, 3))], axis=0)
        for iQ, hv, trM in head_scalars[:4]:
            log(f"  [head-tensor] Q#{iQ}: <v_LR>_mBZ = {hv:.6f}, "
                f"tr M_ab = {trM:.6e} Ry/bohr^2")
        if len(head_scalars) > 4:
            log(f"  [head-tensor] ... {len(head_scalars)} Q in total")
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
    # W_q is DONATED on the two SAME-SHAPE paths, and the caller-side
    # reference dropped, copied from ``bse_lanczos``'s W_R build.  XLA grants
    # the alias only when the output shape matches the input, so W_R becomes
    # W_q's buffer rather than a second live tile for the whole band scan:
    # 56.25 MiB/rank on the Si 4x4x4 deck, 404 MB/rank at mu=10015 / P=64,
    # 4.1 GB/rank at mu=32k.  In-jit peak is unchanged; the win is entirely
    # caller-side (FFT_DONATION_AUDIT.md 2.3).  Value-identical.
    # The ``--w-coarse-grid`` path below is DELIBERATELY not donated: it
    # sub-samples W_q to a coarse sub-grid first, so the densifier's operand
    # is a different array of a different shape and donation is declined
    # anyway.  It also still needs ``data["W_q"]`` to build that sub-grid,
    # which is why the reference is dropped per-branch and not once up here.
    _ifftn_donated = jax.jit(_ifftn, donate_argnums=(0,))
    if args.w_coarse_grid is None:
        W_R = _ifftn_donated(data["W_q"])         # fast path: native fine W (byte-identical)
        data["W_q"] = None                        # release the caller-side reference
    else:
        cg = tuple(int(s) for s in args.w_coarse_grid.split(","))
        if len(cg) != 3:
            raise ValueError("--w-coarse-grid expects NX,NY,NZ")
        if cg == (nkx, nky, nkz):
            W_R = _ifftn_donated(data["W_q"])     # equal grids → no-op, byte-identical
            data["W_q"] = None                    # release the caller-side reference
        else:
            # Coarse-W → fine direct term.  Sub-sample the fine W_q onto the
            # coarse BZ sub-grid (same ISDF μ-basis; q=0 head-tile preserved),
            # then the ONE sharded densifier (bse_io.make_w_densifier: shard_map
            # ifft to the coarse R-lattice + jitted R zero-pad = exact trig
            # interpolation, (μ,ν) sharding preserved throughout — no eager pad
            # + device_put re-shard).  The convolution then runs on the fine
            # grid with the fine (nkx,nky,nkz) solver — cheap coarse W, fine
            # excitons.
            #
            # C1 (default, ``--w-head-densify c1``).  The Γ head is SPLIT OFF
            # before the sub-sampling, so what gets trig-interpolated is the
            # body alone, and the head is re-attached analytically at every
            # fine q inside the coarse Γ cell (gw.head_densify).  Note the
            # asymmetry with the ``bse_k_grid`` path: there the loader can
            # DEFER the injection because it knows a densification is coming,
            # whereas here the restart is natively fine and the loader has
            # already injected — so the head is subtracted back off, using the
            # same rank-one object with the same scalar.  That subtraction is
            # float-inexact at the 1e-16 level and is inherent to the harness,
            # not to C1; the legacy arm carries the identical rounding because
            # it decimates the very same injected tile.
            # Imported HERE, not at module scope.  ``gw.head_densify``
            # bootstraps the service search path at import time
            # (``ffi._services.ensure_on_path()``), and doing that from a
            # driver's import block reorders sys.path for every run of the
            # driver, including the ones that never densify.  Measured: it
            # moved the X-point exciton cluster of the default path by 3e-5 eV
            # — reproducibly, on a near-degenerate sextet, with no array in
            # the code changed.  A feature that is off by default must not be
            # able to do that, so the import lives inside the branch that
            # uses it.
            from gw.head_densify import attach_head_channel
            w_head_mode = resolve_w_head_densify(args.w_head_densify, params)
            W_q_fine_in = data["W_q"]
            head_ch = None
            if w_head_mode == "c1":
                head_ch = _resolve_native_w_head(
                    restart_file, args.input, wfn, log=log)
            if head_ch is not None:
                W_q_fine_in = attach_head_channel(
                    W_q_fine_in, data["g0_X"], data["g0_Y"],
                    _gamma_only_grid((nkx, nky, nkz), -head_ch["whead"]),
                    head_ch["cell_volume"])
            W_q_coarse = decimate_W_q_to_subgrid(W_q_fine_in, cg)
            densify_W = make_w_densifier(mesh_xy, sh.W.spec, (nkx, nky, nkz),
                                         output="k" if head_ch else "R")
            W_dense = densify_W(W_q_coarse)
            if head_ch is not None:
                S_fine = build_w_head_channel(
                    wfn, sym, meta, params, coarse_grid=cg,
                    fine_grid=(nkx, nky, nkz), whead=head_ch["whead"],
                    # THE REFERENCE GRID IS THE FINE ONE HERE.  The restart is
                    # natively fine, so its whead is the fine cell's average —
                    # which makes C1's Γ entry reproduce it EXACTLY, i.e. the
                    # arm that simulates a coarse W puts back the head the
                    # fine run really had.  Passing the coarse grid instead
                    # would silently rescale the head by ~m².
                    ref_grid=(nkx, nky, nkz),
                    input_file=args.input, restart_file=restart_file,
                    gamma_cell=args.w_head_gamma_cell, log_fn=log)
                W_dense = attach_head_channel(
                    W_dense, data["g0_X"], data["g0_Y"], S_fine,
                    head_ch["cell_volume"])
                W_R = _ifftn(W_dense)
            else:
                W_R = W_dense
            log(f"[coarse-W] W sampled on {cg[0]}x{cg[1]}x{cg[2]} sub-grid of "
                f"{nkx}x{nky}x{nkz}, zero-padded in R (trig-interp to fine "
                f"grid) [w_head_densify={w_head_mode}"
                + (f", gamma_cell={args.w_head_gamma_cell}]"
                   if head_ch else "]"))
    # ── the tensor head's transition-side operand: conj(d) on the BSE window ──
    # Q-INDEPENDENT.  d is the q→0 coefficient of the pair amplitude
    # (LT_HEAD_PROBLEM.md §6.1), so it belongs to the k-point transition, and
    # only the cell moment M_ab carries Q.
    head_args = ()
    if head_mbz:
        D_head = build_head_dipole_operand(
            args, nk, nc_pad, nv_pad, n_val, n_cond, log=log)
        head_args = (jnp.asarray(D_head),
                     jnp.asarray(M_stack, dtype=jnp.float64))
    solver = build_path_solver(
        mesh_xy, nkx, nky, nkz, nc_pad, nv_pad, n_eig=args.n_eig,
        block_size=args.block_size, max_iter=args.max_iter,
        head_tensor=head_mbz)
    tick("w_r_and_build", t0)
    t_c0 = time.time()
    evs_dev, alpha_dev = solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                                data["psi_v_X"], data["psi_v_Y"],
                                data["eps_v"], W_R, *head_args)
    # The block-Lanczos Ritz values come from a small replicated eigh(T) — the
    # scan output is fully addressable on every process, so the rank-0
    # ``device_get`` below reconstructs the whole (n_solve, n_eig) table
    # (no cross-process gather needed).  Logged so the run proves it.
    log(f"[dist] evs sharding={evs_dev.sharding}, "
        f"fully_addressable={evs_dev.is_fully_addressable}")
    evs_all = _gather_host(evs_dev)                   # (n_solve, n_eig) Ry
    # The α-Hermiticity invariant, replayed on the host from scalars the scan
    # returned.  Same tolerance, same message, same LORRAX_SANITY=strict raise
    # as the in-jit callback it replaces — see _report_alpha_over_path.
    _report_alpha_over_path(solver.alpha_labels, jax.device_get(alpha_dev),
                            log=log)
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
                   data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R,
                   *head_args)[0])
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
                       W_R, *head_args).compile().memory_analysis()
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

    # CLOSE THE LOADER HERE, EXPLICITLY, WHILE THE RANKS ARE STILL IN STEP.
    # ``initialize_wfns`` hands back a MESH-AWARE ``WfnLoader``
    # (``htransform.setup_wfn_and_sym`` -> ``WfnLoader(wfn_file, mesh=mesh_xy)``),
    # so at P>1 it picks the phdf5 backend and owns a ``SlabIO`` whose
    # ``close()`` runs an UNCONDITIONAL COLLECTIVE barrier
    # (``file_io/_slab_io_ffi.py``, ``_barrier("slab_io_ffi_close_attrs")``).
    # Left to ``WfnLoader.__del__``, that collective fires whenever the
    # garbage collector happens to drop the object during interpreter
    # shutdown — a moment no two ranks agree on.  Measured on this tree
    # (jobs 56550230 steps .9/.10, 2026-08-09): three ranks parked in
    # ``__del__`` -> ``SlabIO.close`` -> ``sync_global_devices`` (NCCL,
    # spinning in ``cuStreamSynchronize``) while the fourth had already
    # reached ``ffi.io._atexit_close_all`` -> ``phdf5_close`` -> ``H5Fclose``
    # -> ``MPI_File_close`` -> ``MPI_Barrier`` (spinning in ``sched_yield``).
    # Two disjoint collective domains, neither ever satisfied: the payload
    # was complete and every output written, and the step then held its four
    # GPUs at 4x100% CPU until the allocation died.
    #
    # AFTER the barrier above, not before: that barrier is what guarantees
    # rank 0's matplotlib/.dat block has finished, so every rank enters this
    # close at the same point.  ``close()`` is idempotent and nulls the
    # handles, so the later ``__del__`` becomes a no-op on every rank and
    # nothing collective is left to interpreter shutdown.  A no-op at P=1,
    # where the SlabIO barrier is already a no-op.
    try:
        wfn.close()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [exciton_bands] WfnLoader.close() failed "
              f"({type(exc).__name__}: {exc}); continuing to exit")
    return 0


if __name__ == "__main__":
    # ``runtime.finalize_process``, not a bare ``SystemExit``: the explicit
    # ``wfn.close()`` above removes the collective from ``__del__``, and this
    # removes the SECOND unordered collective — ``ffi.io._atexit_close_all``,
    # whose ``H5Fclose`` on the restart/zeta contexts is collective too and
    # otherwise runs at whatever point each rank's interpreter teardown
    # reaches it.  ``finalize_process`` runs the effects barrier, the
    # distributed shutdown and the atexit hooks in ONE stated order on every
    # rank and then ends with ``os._exit``.  Same adopter pattern as
    # ``gw.gw_jax``, which is the sibling driver that has never hung.
    from runtime import finalize_process
    finalize_process(main())
