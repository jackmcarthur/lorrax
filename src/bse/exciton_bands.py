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
    wrapped labels — the same torus convention as the stored grid ψ.
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

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from solvers.lanczos import block_lanczos_eig_jit
from common.fft_helpers import make_sharded_ifftn_3d
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import make_bse_shardings
from .bse_serial import compute_pair_amplitude
from .bse_stack_matvec import build_bse_stack_matvec
from .bse_w_exact import _create_mesh_xy
from . import vq_interp

RY2EV = 13.6056980659
PAD_EPS_GUARD_RY = 1.0e3


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
    per-Q conduction caches, padded to the loader's mesh extents.

    Returns (psi_cQ_X, psi_cQ_Y, eps_cQ):
        psi_cQ_[XY]: (nQ, nk, nc_pad, ns, n_rmu_pad), μ on x / y
        eps_cQ:      (nQ, nk, nc_pad) — pad bands at +PAD_EPS_GUARD_RY
    """
    psi = np.asarray(jax.device_get(bundle.psi_rmu_Y))    # (nQ·nk, nc, ns, n_μ)
    eps = np.asarray(jax.device_get(bundle.enk_full))     # (nQ·nk, nc)
    ns = psi.shape[2]
    psi = psi.reshape(nQ, nk, n_cond, ns, n_rmu)
    eps = eps.reshape(nQ, nk, n_cond)
    if n_rmu_pad > n_rmu:
        psi = np.pad(psi, ((0, 0),) * 4 + ((0, n_rmu_pad - n_rmu),))
    if n_cond_pad > n_cond:
        psi = np.pad(psi, ((0, 0), (0, 0), (0, n_cond_pad - n_cond),
                           (0, 0), (0, 0)))
        eps = np.pad(eps, ((0, 0), (0, 0), (0, n_cond_pad - n_cond)),
                     constant_values=PAD_EPS_GUARD_RY)
    x5 = NamedSharding(mesh_xy, P(None, None, None, None, "x"))
    y5 = NamedSharding(mesh_xy, P(None, None, None, None, "y"))
    rep = NamedSharding(mesh_xy, P())
    psi_j = jnp.asarray(psi)
    return (jax.device_put(psi_j, x5), jax.device_put(psi_j, y5),
            jax.device_put(jnp.asarray(eps), rep))


def gate_htransform_vs_stored(psi_cQ_gamma, eps_cQ_gamma, data, log=print):
    """On-grid consistency gate at a Γ path point: htransform conduction
    ε vs the stored restart ε (max |Δ|), and the gauge-free per-k subspace
    overlap min singular value of ⟨ψ_ht|ψ_stored⟩ (bands × bands, spin+μ
    contracted).  Values printed; hard-fails only on gross breakage (>0.5
    overlap loss / >0.1 Ry ε drift) — htransform accuracy is
    centroid-count-governed and reported, not silently trusted."""
    eps_st = np.asarray(jax.device_get(data["eps_c"]))    # (nk, nc_pad)
    psi_st = np.asarray(jax.device_get(data["psi_c_X"]))  # (nk, nc_pad, ns, μ_pad)
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
    log(f"  [gate] htransform@Γ vs stored: max|Δε_c| = {d_eps*RY2EV*1e3:.3f} meV, "
        f"conduction-subspace overlap min-sval = {smin:.4f}")
    assert d_eps < 0.1 and smin > 0.5, \
        "htransform conduction cache grossly inconsistent with the stored grid"
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
    ap.add_argument("--vq-mode", choices=("interp", "refit", "both"),
                    default="interp")
    ap.add_argument("--refit-points", type=str, default=None,
                    help="comma list of path indices to refit "
                         "(vq-mode=both; default ~5 evenly spaced)")
    ap.add_argument("--alpha", type=float, default=vq_interp.ALPHA)
    ap.add_argument("--eps-tik", type=float, default=vq_interp.EPS_TIK)
    ap.add_argument("--eigh-backend", default="auto",
                    choices=("auto", "off", "cusolvermp", "slate"))
    ap.add_argument("--px", type=int, default=1)
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--out-prefix", type=str, default="exciton_bands")
    args = ap.parse_args(argv)

    t_wall = time.time()
    timers: dict[str, float] = {}

    def tick(name, t0):
        timers[name] = timers.get(name, 0.0) + (time.time() - t0)

    mesh_xy = _create_mesh_xy(args.px, args.py)

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
        mesh_xy=mesh_xy, input_file=args.input, inject_head=True)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_val, n_cond = int(data["n_val"]), int(data["n_cond"])
    nv_pad, nc_pad = int(data["n_val_pad"]), int(data["n_cond_pad"])
    n_rmu, n_rmu_pad = int(data["n_rmu"]), int(data["n_rmu_pad"])
    # pad-ε guard on the valence side (loader zero-pads; see module doc)
    if nv_pad > n_val:
        eps_v = np.asarray(jax.device_get(data["eps_v"]))
        eps_v[:, n_val:] = -PAD_EPS_GUARD_RY
        data["eps_v"] = jnp.asarray(eps_v)
    tick("load_bse", t0)

    # ── htransform setup + Q path ────────────────────────────────────────
    t0 = time.time()
    (wfn, sym, meta, _mesh, _S, ctilde, B_at_mu,
     enk_sigma) = ht.initialize_wfns(args.input, params, print,
                                     mesh_xy=mesh_xy)
    kpath_frac, x_path, node_idx, node_labels, _gp = ht.initialize_kpath(
        wfn, params)
    Qpath = np.asarray(kpath_frac, dtype=np.float64)
    nQ = Qpath.shape[0]
    print(f"Q path: {nQ} points, nodes at {list(map(int, node_idx))} "
          f"labels {node_labels}")
    tick("htransform_setup", t0)

    # ── conduction caches ψ_c(k+Q), ε_c(k+Q) for the whole path ──────────
    t0 = time.time()
    nval_in = int(params["nval"])       # window offset: conduction starts here
    b_min, b_max = nval_in, nval_in + n_cond
    k_frac = np.stack(np.meshgrid(np.arange(nkx) / nkx, np.arange(nky) / nky,
                                  np.arange(nkz) / nkz, indexing="ij"),
                      axis=-1).reshape(-1, 3)
    q_list = (Qpath[:, None, :] + k_frac[None, :, :]).reshape(-1, 3)
    bundle = compute_wfns_fi(
        ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
        kgrid_co=(nkx, nky, nkz), band_window_fi=(b_min, b_max),
        mesh_xy=mesh_xy, q_list=q_list, log_fn=print)
    psi_cQ_X, psi_cQ_Y, eps_cQ = build_conduction_stacks(
        bundle, nQ, nk, n_cond, nc_pad, n_rmu, n_rmu_pad, mesh_xy)
    tick("htransform_psi_cQ", t0)

    # on-grid gate at the first Γ node (path convention: starts at Γ)
    iGamma = [i for i in range(nQ)
              if np.linalg.norm(Qpath[i] - np.round(Qpath[i])) < 1e-9]
    if iGamma:
        gate_htransform_vs_stored(
            np.asarray(jax.device_get(psi_cQ_X))[iGamma[0]],
            np.asarray(jax.device_get(eps_cQ))[iGamma[0]], data)

    # ── V_Q tiles ────────────────────────────────────────────────────────
    t0 = time.time()
    zeta_file = os.path.join(os.path.dirname(restart_file), "zeta_q.h5")
    zx = vq_interp.load_zeta_coarse(restart_file, zeta_file)
    C_q = vq_interp.build_cq(zx)
    vq_interp.run_gates(zx, C_q)
    prep = vq_interp.prepare_coarse(zx, C_q, mesh_xy, alpha=args.alpha,
                                    eps_tik=args.eps_tik,
                                    eigh_backend=args.eigh_backend)
    des = vq_interp.lr_design_blocks(zx, prep)
    coeffs = vq_interp.fit_lr_model(des)
    vq_interp.run_nulls(zx, prep, des, coeffs)
    eval_vq = vq_interp.make_eval_vq(zx, prep, des, mesh_xy, n_rmu_pad)
    pinvF = jnp.asarray(vq_interp.stencil_pinv(
        zx["qfr"], vq_interp.stencil_r7(zx)))
    coeffs_packed = vq_interp.pack_coeffs(des, coeffs)
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
        q_tile = jnp.asarray(q_tile - np.round(q_tile))
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
        rst = vq_interp.refit_prepare(args.input, mesh_xy, zx)
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
    W_R = _ifftn(data["W_q"])
    solver = build_path_solver(
        mesh_xy, nkx, nky, nkz, nc_pad, nv_pad, n_eig=args.n_eig,
        block_size=args.block_size, max_iter=args.max_iter)
    t_c0 = time.time()
    evs_all = solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                     data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R)
    evs_all = np.asarray(jax.device_get(evs_all))     # (n_solve, n_eig) Ry
    t_first = time.time() - t_c0
    tick("solve_scan_cold", t_c0)
    # warm re-run for the census-clean per-Q cost (dispatch-only)
    t_w0 = time.time()
    evs2 = np.asarray(jax.device_get(
        solver(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
               data["psi_v_X"], data["psi_v_Y"], data["eps_v"], W_R)))
    t_warm = time.time() - t_w0
    tick("solve_scan_warm", t_w0)
    assert np.allclose(evs2, evs_all, atol=1e-10), "scan re-run not reproducible"
    mem = solver.lower(psi_cQ_X, psi_cQ_Y, eps_cQ, V_stack,
                       data["psi_v_X"], data["psi_v_Y"], data["eps_v"],
                       W_R).compile().memory_analysis()
    print(f"solve_path: cold {t_first:.2f}s (incl. ONE compile), warm "
          f"{t_warm:.2f}s = {t_warm/n_solve*1e3:.1f} ms/Q over {n_solve} Q")
    print(f"solve_path memory_analysis: temp={mem.temp_size_in_bytes/2**20:.1f} MiB "
          f"args={mem.argument_size_in_bytes/2**20:.1f} MiB "
          f"out={mem.output_size_in_bytes/2**20:.1f} MiB")

    evs_path = evs_all[:nQ]
    evs_refit = {iQ: evs_all[nQ + j] for j, iQ in enumerate(refit_idx)}

    # ── outputs ──────────────────────────────────────────────────────────
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
