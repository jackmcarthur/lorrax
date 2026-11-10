#!/usr/bin/env python3
"""Exact plane-wave Σ^B and Σ_x of a bispinor run on the device mesh: the GPU port of
``tools/sigma_b_pw_reference.py``.

    srun -n P src/ffi/cpp/select_gpu.sh python -u tools/sigma_b_pw_device.py \
        --wfn WFN.h5 --nsigma 144 --out pw.npz [--ecut RY] [--sys-dim 2] \
        [--alpha-scale 1] [--kleft 0,3] [--box alias_free|wfn] [--no-x] [--nchunk 16]

The same Σ^B and Σ_x as the CPU reference, for the bands [0, nsigma): the bands
a run's ``sigTT`` column reports, [0, nelec + ncond).  The formulas, the
normalized RKB lift, the kernel v(K) P^T(K̂) and every convention are the
reference's (its docstring); this file imports its WFN header, lift, Coulomb
and projector helpers, so the two cannot drift.  P ranks of the run's square
mesh share the work:

* rank r takes the right rows k' = jP + r (j = 0, 1, ...); slots past N_k are
  zero-weight copies, so P need not divide N_k;
* for every occupied v, the pair fields Ψ_n† γ̃^I Ψ_v of every Σ band are formed
  on the box, one FFT per (n, I), and the sphere 0 < |K|² ≤ E_c is gathered:
  M^I_nv(K);
* T^c_nv(K) = √v(K) Σ_i P^T_ci(K̂) M^i_nv(K).  P^T is a real symmetric
  projector and R^J_vm(K) = conj(M^J_mv(K)) (certified to round-off by the CPU
  reference's two-FFT bookkeeping), so

      Σ^B_nm = (1/N_k) Σ_{k',v,K,c} T^c_nv conj(T^c_mv),
      Σ^x_nm = −(1/N_k) Σ_{k',v,K} v(K) M^0_nv conj(M^0_mv):

  Grams of row slabs, each formed by ``distrib_la.replicated_gram`` with the
  ranks' partials added in rank order, so the result does not depend on which
  device holds which rank, and both are Hermitian PSD (−Σ^x) by construction.

``--box alias_free`` (the default) runs every FFT on the smallest box whose
extents are 2·3·5·7-smooth and exceed m^pair_j + m^Σ_j per crystal axis (m: the
Miller half-widths of the pair spectrum, twice the WFN's, and of the sphere).
No alias of a pair spectrum then lands on the sphere, so M there equals the
WFN-box M of the reference exactly (CrI3 75·75·250 → 60·60·189 points).
``--box wfn`` uses the WFN's own box, as the CPU reference does.

Output (npz): the CPU reference's keys, ``sigma_b_ry``, ``sigma_x_ry``
``(n_rows, nsigma, nsigma)`` in Ry at ``k_rows`` (WFN order), ``kpts``,
``enk_ev``, and a JSON ``receipt``.  ``sigma_x_ry`` is zero under ``--no-x``,
which drops the γ̃^0 channel (a quarter of the FFTs).

Cost per (k, k') pair: nocc·nsigma·(3 or 4) box FFTs and nocc Grams of
(3 n_K) × nsigma; memory per rank: the Σ bands and the occupied bands of one
k' on the box, one FFT chunk, M and T.  First use: the CrI3 6×6 row-9 ladder
(2026-11-09), proven against the CPU reference on MoS2 3×3 SOC first.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# LORRAX's runtime before NumPy (GATE blas_spin) and before JAX (initialize_communicator_stack).
from runtime import initialize_communicator_stack, rank0_print  # noqa: E402

import numpy as np  # noqa: E402

RY_EV = 13.605693122994


def _reference_module():
    """``tools/sigma_b_pw_reference.py`` as a module (h5py and NumPy only)."""
    path = Path(__file__).with_name("sigma_b_pw_reference.py")
    spec = importlib.util.spec_from_file_location("sigma_b_pw_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = _reference_module()


def smooth_at_least(n: int) -> int:
    """The smallest 2·3·5·7-smooth integer ≥ n."""
    m = max(1, int(n))
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def read_header(path):
    """The WFN's k list, occupations, energies, cell and G lists (no coefficients)."""
    import h5py
    with h5py.File(path, "r") as f:
        h = f["mf_header"]
        kp, cr = h["kpoints"], h["crystal"]
        if int(kp["nspin"][()]) != 1 or int(kp["nspinor"][()]) != 2:
            raise ValueError("need a noncollinear spinor WFN (nspin 1, nspinor 2)")
        out = dict(
            rk=np.asarray(kp["rk"][()], np.float64),
            ngk=np.asarray(kp["ngk"][()], np.int64),
            kgrid=tuple(int(x) for x in kp["kgrid"][()]),
            occ=np.asarray(kp["occ"][()], np.float64)[0],
            el=np.asarray(kp["el"][()], np.float64)[0],
            ecutwfc=float(kp["ecutwfc"][()]),
            avec=np.asarray(cr["avec"][()], np.float64) * float(cr["alat"][()]),
            bvec=np.asarray(cr["bvec"][()], np.float64) * float(cr["blat"][()]),
            celvol=float(cr["celvol"][()]),
            fft=tuple(int(x) for x in h["gspace/FFTgrid"][()]),
            nbands=int(f["wfns/coeffs"].shape[0]),
        )
        gv = np.asarray(f["wfns/gvecs"][()], np.int64)
    off = np.concatenate([[0], np.cumsum(out["ngk"])])
    out["off"] = off
    out["gk"] = [gv[off[i]:off[i + 1]] for i in range(len(out["ngk"]))]
    out["m_wf"] = np.max(np.abs(gv), axis=0)
    return out


def read_bands(path, hdr, ik, nb):
    """ψ_L(G) of bands [0, nb) at stored row ``ik``: (nb, 2, ngk) complex128."""
    import h5py
    a, b = int(hdr["off"][ik]), int(hdr["off"][ik + 1])
    with h5py.File(path, "r") as f:
        raw = f["wfns/coeffs"][:nb, :, a:b, :]
    return raw[..., 0] + 1j * raw[..., 1]


def flat_index(G, box):
    """C-order flat index of Miller vectors ``G (n, 3)`` in ``box``."""
    g = [np.mod(G[:, j], box[j]) for j in range(3)]
    return ((g[0] * box[1] + g[1]) * box[2] + g[2]).astype(np.int32)


def _kernel(mesh, *, nsig, nocc, ngk, nK, box, nch, with_x):
    """One (k, k') slot per rank: ``(Σ^B, Σ^x)`` partials summed over the ranks, replicated."""
    import jax
    import jax.numpy as jnp
    from jax import lax, shard_map
    from jax.sharding import PartitionSpec as P
    import distrib_la
    from common.fft_helpers import local_ifftn3

    flat = tuple(mesh.axis_names)
    nr = int(np.prod(box))
    nI = 4 if with_x else 3
    n_chunks = -(-nsig // nch)
    pauli = REF.PAULI

    def place(coef, idx):
        """u(r) = Σ_G c(G) e^{iG·r} on the box (the reference's ifftn · N_r)."""
        z = jnp.zeros(coef.shape[:2] + (nr,), coef.dtype).at[:, :, idx].add(coef)
        return local_ifftn3(z.reshape(coef.shape[:2] + box)) * nr

    def vertices(uv):
        """γ̃^I uv, I = 0..3 (or 1..3): (nI, 4, *box); α^i = [[0, σ^i], [σ^i, 0]]."""
        up, lo = uv[0:2], uv[2:4]

        def sigma(s, x):
            return jnp.stack([sum(s[a, b] * x[b] for b in range(2) if s[a, b] != 0)
                              for a in range(2)])
        out = [uv] if with_x else []
        for s in pauli:
            out.append(jnp.concatenate([sigma(s, lo), sigma(s, up)]))
        return jnp.stack(out)

    def body(psik, idxk, psip, idxp, kidx, sw, swP):
        u_k = place(psik, idxk)                                   # (nsig, 4, *box)
        u_p = place(psip[0], idxp[0])                             # (nocc, 4, *box)
        kidx, sw, swP = kidx[0], sw[0], swP[0]

        def per_v(v, acc):
            HB, HX = acc
            av = vertices(lax.dynamic_index_in_dim(u_p, v, 0, keepdims=False))

            def per_chunk(c, M):
                s = jnp.minimum(c * nch, nsig - nch)
                cu = jnp.conj(lax.dynamic_slice_in_dim(u_k, s, nch, axis=0))
                f = sum(cu[:, a][:, None] * av[None, :, a] for a in range(4))
                F = local_ifftn3(f).reshape(nch, nI, nr)
                return lax.dynamic_update_slice_in_dim(M, jnp.take(F, kidx, axis=2), s, axis=0)

            M = lax.fori_loop(0, n_chunks, per_chunk, jnp.zeros((nsig, nI, nK), jnp.complex128))
            Mi = M[:, nI - 3:]
            T = jnp.stack([sum(swP[None, :, c, i] * Mi[:, i] for i in range(3)) for c in range(3)],
                          axis=1).reshape(nsig, 3 * nK)
            HB = HB + distrib_la.replicated_gram(jnp.conj(T).T, axes=flat)
            if with_x:
                X = sw[None, :] * M[:, 0]
                HX = HX - distrib_la.replicated_gram(jnp.conj(X).T, axes=flat)
            return HB, HX

        z = jnp.zeros((nsig, nsig), jnp.complex128)
        return lax.fori_loop(0, nocc, per_v, (z, z))

    rep, per = P(), P(flat)
    return jax.jit(shard_map(body, mesh=mesh, in_specs=(rep, rep) + (per,) * 5,
                             out_specs=(rep, rep), check_vma=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--nsigma", type=int, required=True,
                    help="Σ bands [0, nsigma) (the run's nelec + ncond)")
    ap.add_argument("--ecut", type=float, default=None,
                    help="bare-Coulomb sphere |K|² ≤ E_c in Ry (default: WFN ecutwfc)")
    ap.add_argument("--sys-dim", type=int, default=2)
    ap.add_argument("--alpha-scale", type=float, default=1.0)
    ap.add_argument("--kleft", default=None, help="comma list of k rows (default all)")
    ap.add_argument("--box", choices=("alias_free", "wfn"), default="alias_free")
    ap.add_argument("--no-x", action="store_true", help="skip Σ_x (the γ̃^0 channel)")
    ap.add_argument("--nchunk", type=int, default=16, help="Σ bands per FFT batch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    initialize_communicator_stack(print_fn=rank0_print)
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import prepare_mesh

    t0 = time.time()
    mesh = prepare_mesh(axis_names=("x", "y"))
    flat = tuple(mesh.axis_names)
    n_rank = int(mesh.size)
    w = read_header(a.wfn)
    rk, kgrid, fft_wfn = w["rk"], w["kgrid"], w["fft"]
    nk = len(rk)
    if nk != int(np.prod(kgrid)):
        raise ValueError(f"WFN holds {nk} k of the {kgrid} grid; the reference needs the full grid")
    occ = w["occ"]
    nocc_k = np.sum(occ > 0.5, axis=1)
    if np.any(np.abs(occ - np.round(occ)) > 1e-8) or len(set(nocc_k.tolist())) != 1:
        raise ValueError("insulating 0/1 occupations with one occupied count are required")
    nocc = int(nocc_k[0])
    nsig = int(a.nsigma)
    if not nocc <= nsig <= w["nbands"]:
        raise ValueError(f"nsigma must lie in [nocc={nocc}, nbands={w['nbands']}]")
    ecut = w["ecutwfc"] if a.ecut is None else float(a.ecut)
    bvec, avec, om = w["bvec"], w["avec"], w["celvol"]
    if np.max(np.abs(avec @ bvec.T - 2 * np.pi * np.eye(3))) > 1e-8:
        raise ValueError("avec·bvec ≠ 2π")
    zc = None
    if a.sys_dim == 2:
        if np.max(np.abs(avec[2, :2])) > 1e-10 or np.max(np.abs(avec[:2, 2])) > 1e-10:
            raise ValueError("sys_dim 2 needs a3 ∥ z ⟂ a1, a2")
        zc = 0.5 * abs(avec[2, 2])
    h = REF.HALFALPHA * a.alpha_scale
    rows = list(range(nk)) if a.kleft is None else [int(x) for x in a.kleft.split(",")]

    # ---- the sphere candidates (a superset of every pair's K = k − k' + G), the box
    box_g = REF.miller_box(fft_wfn)
    qmax = max(float(np.linalg.norm((rk[i] - rk[j]) @ bvec)) for i in rows for j in range(nk))
    cand = box_g[np.linalg.norm(box_g @ bvec, axis=1) <= np.sqrt(ecut) + qmax + 1e-9]
    m_t = np.max(np.abs(cand), axis=0)
    m_pair = 2 * w["m_wf"]
    if a.box == "wfn":
        box = tuple(fft_wfn)
    else:
        box = tuple(min(smooth_at_least(int(m_pair[j] + m_t[j] + 1)), fft_wfn[j]) for j in range(3))
    if any(2 * int(w["m_wf"][j]) >= box[j] or 2 * int(m_t[j]) >= box[j] for j in range(3)):
        raise ValueError(f"box {box} cannot hold the WFN spheres (m_wf {w['m_wf']}) or the Σ sphere {m_t}")
    alias_free = all(box[j] > m_pair[j] + m_t[j] for j in range(3))

    def sphere(ik, jk):
        """(G, K, flat box index) of 0 < |K|² ≤ E_c, K = k_ik − k_jk + G."""
        K = (rk[ik] - rk[jk])[None] + cand
        K = K @ bvec
        K2 = np.sum(K * K, axis=1)
        sel = (K2 <= ecut) & (K2 > 1e-12)
        return cand[sel], K[sel], flat_index(cand[sel], box)

    nK = max(len(sphere(i, j)[0]) for i in rows for j in range(nk))
    nK = -(-nK // 256) * 256
    ngk = int(np.max(w["ngk"]))
    n_slot = -(-nk // n_rank)
    receipt = dict(wfn=os.path.realpath(a.wfn), nk=nk, kgrid=kgrid, fft_wfn=fft_wfn, box=box,
                   box_kind=a.box, alias_free=bool(alias_free), m_wf=w["m_wf"].tolist(),
                   m_sigma=m_t.tolist(), nocc=nocc, nsigma=nsig, ecut_ry=ecut, sys_dim=a.sys_dim,
                   zc_bohr=zc, alpha_fs=2 * h, celvol=om, k_rows=rows, ranks=n_rank,
                   mesh=dict(mesh.shape), slots_per_rank=n_slot, nK_pad=nK, ngk_pad=ngk,
                   with_x=not a.no_x, nchunk=int(a.nchunk))
    rank0_print(f"[sigma_b_pw_device] {json.dumps(receipt)}")

    # ---- per-rank right rows: lifted occupied bands at k' = jP + r, read once
    rep = NamedSharding(mesh, P())
    per = NamedSharding(mesh, P(flat))
    my_ranks = sorted({int(i) for i in np.flatnonzero(
        [d.process_index == jax.process_index() for d in mesh.devices.reshape(-1)])})
    lift_dev = 0.0

    def lifted(ik, nb):
        psi = read_bands(a.wfn, w, ik, nb)
        K = (rk[ik][None] + w["gk"][ik]) @ bvec
        Psi = REF.rkb_lift(psi, K, h)
        dev = float(np.max(np.abs(np.sum(np.abs(Psi) ** 2, axis=(1, 2))
                                  - np.sum(np.abs(psi) ** 2, axis=(1, 2)))))
        coef = np.zeros((nb, 4, ngk), np.complex128)
        coef[:, :, :Psi.shape[2]] = Psi
        idx = np.zeros((ngk,), np.int32)
        idx[:Psi.shape[2]] = flat_index(w["gk"][ik], box)
        return coef, idx, dev

    right = {}
    for r in my_ranks:
        for j in range(n_slot):
            jk = j * n_rank + r
            if jk < nk:
                coef, idx, dev = lifted(jk, nocc)
                lift_dev = max(lift_dev, dev)
                right[(r, j)] = (coef, idx)

    def per_rank(fn, shape, dtype):
        def cb(index):
            r = index[0].start or 0
            return np.asarray(fn(r), dtype)[None]
        return jax.make_array_from_callback((n_rank,) + shape, per, cb)

    def replicated(x):
        return jax.make_array_from_callback(x.shape, rep, lambda index: x[index])

    kernel = _kernel(mesh, nsig=nsig, nocc=nocc, ngk=ngk, nK=nK, box=box,
                     nch=min(int(a.nchunk), nsig), with_x=not a.no_x)
    sig_b = np.zeros((len(rows), nsig, nsig), np.complex128)
    sig_x = np.zeros_like(sig_b)
    gmax_frac, nK_tot, n_pairs = 0.0, 0, 0
    for ir, ik in enumerate(rows):
        coef_k, idx_k, dev = lifted(ik, nsig)
        lift_dev = max(lift_dev, dev)
        psik, idxk = replicated(coef_k), replicated(idx_k)
        for j in range(n_slot):
            def slot(r):
                jk = j * n_rank + r
                return jk if jk < nk else None

            def sph(r):
                jk = slot(r)
                kidx = np.zeros((nK,), np.int32)
                sw = np.zeros((nK,), np.float64)
                swP = np.zeros((nK, 3, 3), np.float64)
                if jk is not None:
                    G, K, fi = sphere(ik, jk)
                    v = REF.coulomb(K, celvol=om, sys_dim=a.sys_dim, zc=zc)
                    kidx[:len(G)] = fi
                    sw[:len(G)] = np.sqrt(v)
                    swP[:len(G)] = np.sqrt(v)[:, None, None] * REF.transverse_projector(K)
                return kidx, sw, swP

            cache = {r: sph(r) for r in my_ranks}
            empty = (np.zeros((nocc, 4, ngk), np.complex128), np.zeros((ngk,), np.int32))
            args = (psik, idxk,
                    per_rank(lambda r: right.get((r, j), empty)[0], (nocc, 4, ngk), np.complex128),
                    per_rank(lambda r: right.get((r, j), empty)[1], (ngk,), np.int32),
                    per_rank(lambda r: cache[r][0], (nK,), np.int32),
                    per_rank(lambda r: cache[r][1], (nK,), np.float64),
                    per_rank(lambda r: cache[r][2], (nK, 3, 3), np.float64))
            HB, HX = kernel(*args)
            sig_b[ir] += np.asarray(HB.addressable_shards[0].data)
            sig_x[ir] += np.asarray(HX.addressable_shards[0].data)
            for r in range(n_rank):
                jk = j * n_rank + r
                if jk < nk:
                    G, _, _ = sphere(ik, jk)
                    gmax_frac = max(gmax_frac, float(np.max(np.abs(G) / (np.array(fft_wfn) // 2))))
                    nK_tot += len(G)
                    n_pairs += 1
        sig_b[ir] /= nk
        sig_x[ir] /= nk
        rank0_print(f"[sigma_b_pw_device] k row {ik} done {time.time() - t0:.1f}s  diag Σ^B (meV) "
                    f"{np.round(np.real(np.diag(sig_b[ir]))[:8] * RY_EV * 1e3, 4).tolist()}")

    receipt["lift_isometry_dev"] = lift_dev
    receipt["sphere_max_miller_over_half_box"] = gmax_frac
    receipt["n_K_per_kpair_mean"] = nK_tot / max(n_pairs, 1)
    for name, S in (("sigma_b", sig_b), ("sigma_x", sig_x)):
        receipt[f"{name}_hermiticity_maxdev_ry"] = float(np.max(np.abs(S - np.conj(np.swapaxes(S, 1, 2)))))
        receipt[f"{name}_max_abs_ry"] = float(np.max(np.abs(S)))
    receipt["sigma_b_min_eig_ry"] = float(min(np.linalg.eigvalsh(0.5 * (S + S.conj().T)).min()
                                             for S in sig_b))
    receipt["wall_s"] = time.time() - t0
    if jax.process_index() == 0:
        np.savez(a.out, sigma_b_ry=sig_b, sigma_x_ry=sig_x, k_rows=np.array(rows), kpts=rk[rows],
                 enk_ev=w["el"][rows, :nsig] * RY_EV, receipt=json.dumps(receipt))
    rank0_print("[sigma_b_pw_device] receipt " + json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
