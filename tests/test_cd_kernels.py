"""CPU pins of the CD q-parallel kernels (gw.cd.sigma.build_contraction_kernels) on an emulated
four-device 2x2 mesh against a dense per-q sum of the pinned laws (gw.cd.laws).

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; an emulated CPU mesh
checks device-count logic).  What is pinned: the owner-rank real-axis branches equal the W_{-q}^T
partner form, the q padding (nk = 6 on P = 4, garbage in the pad rows) contributes nothing, and
accumulation over two calls is acc + psum(increment) (not P x acc).
"""
import os
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent(r'''
    import numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.cd import laws
    from gw.cd.sigma import build_contraction_kernels, contract, carrier_band_blocks, pair_psi_slabs
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(7)
    nk, n, L, m, s, nE, ndev = 6, 8, 5, 3, 2, 4, 4
    nq_pad, nq_loc = 8, 2
    eta, h = 0.25 / 13.6, 0.0625 / 13.6
    c = lambda *sh: rng.normal(size=sh) + 1j * rng.normal(size=sh)
    psi = c(nk, L, s, n) / 3
    enk = np.sort(rng.uniform(-0.3, 0.3, (nk, L)), axis=1)
    f = np.where(enk < 0, 1.0, 0.0); f[1, 2] = 0.4
    omega = rng.uniform(-0.25, 0.25, nE)
    kw = np.arange(nk)
    eps_own = enk[:, :m]
    KMQ = np.array([[(k - q) % nk for k in range(nk)] for q in range(nk)])
    qneg = np.array([(-q) % nk for q in range(nk)])
    KMQ_pad = np.concatenate([KMQ, KMQ[:2] * 0]); qneg_pad = np.concatenate([qneg, [0, 0]])
    qvalid = np.r_[np.ones(nk), np.zeros(nq_pad - nk)]
    u_all, w_all = np.array([0.01, 0.03, 0.2]), np.array([0.02, 0.05, 0.3])
    rep = NamedSharding(mesh, P())
    put = lambda a: jax.device_put(np.asarray(a), rep)
    psi_q, psi_nq = pair_psi_slabs(carrier_band_blocks(put(psi), mesh), mesh, rows_q=KMQ_pad[:, kw],
                                   rows_nq=KMQ_pad[qneg_pad][:, kw], l0=0, L=L, L_neg=L, block_bands=2)
    T = dict(psi_q=psi_q, psi_nq=psi_nq, psi_M=put(psi[:, :m]), KMQ=put(KMQ_pad.astype(np.int32)), kw=put(kw.astype(np.int32)),
             enk=put(enk), f=put(f), omega=put(omega), eps_own=put(eps_own), qneg=put(qneg_pad.astype(np.int32)),
             qvalid=put(qvalid), u_all=put(u_all), w_all=put(w_all))
    beta = tuple(float(b) for b in laws.BETA_RY)
    K = build_contraction_kernels(mesh, nk=nk, nq_loc=nq_loc, eta=eta, h_ry=h, beta=beta)
    nz = 4
    W = c(nz, nq_pad, n, n); dW = c(nz, nq_pad, n, n)            # pad rows 6, 7 are garbage on purpose
    qpar4 = NamedSharding(mesh, P(None, ("x", "y"))); qpar3 = NamedSharding(mesh, P(("x", "y")))
    Wd, dWd = jax.device_put(W, qpar4), jax.device_put(dW, qpar4)
    zeros = lambda *sh: put(np.zeros(sh, complex))
    def rows(q):
        r = KMQ[q][kw]
        rho = np.einsum("klsu,kmsu->kmlu", psi[r].conj(), psi[:, :m])
        return rho, omega[None, :, None] - enk[r][:, None, :], eps_own[:, :, None] - enk[r][:, None, :], f[r]
    con = lambda rho, op: np.asarray(contract(jnp.asarray(rho), jnp.asarray(op), nk))
    # real axis: two calls of two knots each (accumulation), knots 3..6; full band tables (every l on
    # both branches) are the dense contraction -- tests/test_cd_banded_residues.py pins the banded tables
    z = (np.arange(3, 3 + nz) * h) + 1j * eta
    acc, own = zeros(nE, nk, m, m), zeros(nk, m)
    full = put(np.broadcast_to(np.arange(L, dtype=np.int32), (nq_pad, nk, L)).copy())
    for a0 in (0, 2):
        acc, own = K.real(jax.device_put(W[a0:a0 + 2], qpar4), jax.device_put(dW[a0:a0 + 2], qpar4),
                          jnp.asarray(z[a0:a0 + 2]), jnp.int32(3 + a0), acc, own, T, (full, full), (False, True))
    ref, ref_o = 0, 0
    for q in range(nk):
        rho, x, xo, fq = rows(q)
        for iz in range(nz):
            a, d = con(rho, W[iz, q]), con(rho, dW[iz, q])
            at, dt = con(rho, W[iz, qneg[q]].T), con(rho, dW[iz, qneg[q]].T)
            ref = ref + np.asarray(laws.real_knot(3 + iz, z[iz], a, d, at, dt, x, fq, h))
            ref_o = ref_o + np.asarray(laws.real_knot(3 + iz, z[iz], a, d, at, dt, xo, fq, h, apply_fn=laws.own_apply))
    def err(mine, theirs):
        mine, theirs = np.asarray(mine), np.asarray(theirs)
        return float(np.abs(mine - theirs).max() / max(np.abs(theirs).max(), 1e-300))
    out = dict(real=err(acc, ref), real_own=err(own, ref_o))
    # imaginary axis: two calls
    u, w = np.array([0.01, 0.05, 0.1, 0.4]), np.array([0.02, 0.04, 0.1, 0.2])
    acc, own = zeros(nE, nk, m, m), zeros(nk, m)
    for a0 in (0, 2):
        acc, own = K.imag(jax.device_put(W[a0:a0 + 2], qpar4), put(u[a0:a0 + 2]), put(w[a0:a0 + 2]), acc, own, T)
    ref, ref_o = 0, 0
    for q in range(nk):
        rho, x, xo, fq = rows(q)
        for iz in range(nz):
            a = con(rho, W[iz, q])
            ref = ref + np.asarray(laws.imag_raw_node(a, u[iz], w[iz], x, fq, eta))
            ref_o = ref_o + np.asarray(laws.imag_raw_node(a, u[iz], w[iz], xo, fq, eta, apply_fn=laws.own_apply))
    out.update(imag=err(acc, ref), imag_own=err(own, ref_o))
    # anchor (knot 0) and Sigma_x
    acc, own = K.anchor(jax.device_put(W[0], qpar3), jax.device_put(dW[0], qpar3), zeros(nE, nk, m, m), zeros(nk, m), T)
    ref, ref_o = 0, 0
    for q in range(nk):
        rho, x, xo, fq = rows(q)
        a0, d0 = con(rho, W[0, q]), con(rho, dW[0, q])
        tot, cp, cm = laws.anchor_part(a0, d0, x, fq, eta, beta)
        ref = ref + np.asarray(tot - laws.anchor_subtraction(cp, cm, laws.imag_anchor_sums(u_all, w_all, x, fq, eta, beta)))
        tot, _, _ = laws.anchor_part(a0, d0, xo, fq, eta, beta, apply_fn=laws.own_apply)
        ref_o = ref_o + np.asarray(tot - laws.anchor_subtraction(cp, cm, laws.imag_anchor_sums(u_all, w_all, xo, fq, eta, beta), apply_fn=laws.own_apply))
    out.update(anchor=err(acc, ref), anchor_own=err(own, ref_o))
    sx = K.sigma_x(jax.device_put(W[1], qpar3), zeros(nk, m, m), T)
    sx = K.sigma_x(jax.device_put(W[1], qpar3), sx, T)               # twice: 2 x the one-call value
    ref = sum(np.einsum("kmnl,kl->kmn", con(rows(q)[0], W[1, q]), rows(q)[3]) for q in range(nk))
    out.update(sigma_x_twice=err(sx, 2 * ref))
    print("CDKERNELS", out)
    assert max(out.values()) < 1e-12, out
''')


def test_cd_kernels_match_dense_reference_on_four_cpu_devices():
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip())
    run = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0 and "CDKERNELS" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
