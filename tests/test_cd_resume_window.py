"""CPU pins of the CD own-energy rows (``cd_own_energy_band_min``) and the per-chunk checkpoint/resume.

Unit cell: exempt from the four-GPU rule (AGENT_PREAMBLE: unit and CPU cells; an emulated four-device CPU
mesh checks the replicated-accumulator IO).  What is pinned:

1. Own-energy rows.  With ``own0 = 1`` (the key at band 2) and ``eps_own`` cut to rows >= 1, the four
   kernels give the omega-grid cube of the whole window and own-energy values equal to the whole-window
   run's rows >= 1 at the same knots (J held fixed), on dense and on banded tables; the reach rule
   (``residue_reach_need``) drops when a deep own row leaves; ``own_energy_window`` and the deck gates
   refuse out-of-range values.
2. Checkpoint/resume.  ``stream_cd_chunks`` with ``CdCheckpoint`` (``file_io.sigma_checkpoint``,
   ``CD_SCHEMA``) killed at a chunk and resumed reproduces the uninterrupted accumulators BITWISE, at knot
   chunks below, at and above the 2*GH ghost tail, from either axis, twice in a row, and after a
   ``LORRAX_CD_STOP_AFTER`` stop.  chi0 is called on the planned windows of ``gw.cd.chi0_rules.cd_windows``
   only, and the toy chi0 depends on its whole z batch, so only a resume that re-runs the uninterrupted
   window calls (``real_replay_start``) rebuilds the ghost tail: the red twin (the naive resume that
   recomputes the ghosts in a fresh call) must differ.  A foreign identity is refused and left in place.
3. Partial chi0 rules (a ``--windows`` census build): ``bank_chi0`` loads one only with ``allow_partial``
   (the driver passes it under ``LORRAX_CD_STOP_AFTER`` alone), the built windows equal the full build,
   and a point of an unbuilt window refuses (GATE cd_chi0_rule_unbuilt) instead of a silent zero.
"""
import os
import subprocess
import sys
import textwrap

ENV = dict(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")

KERNELS = textwrap.dedent(r'''
    import numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.cd import laws
    from gw.cd.sigma import (build_contraction_kernels, residue_band_tables, residue_reach_need,
                             own_energy_window, carrier_band_blocks, pair_psi_slabs)
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(5)
    nk, n, L, m, s, nE = 6, 8, 12, 3, 2, 4
    nq_pad, nq_loc, o0 = 8, 2, 1
    eta, h = 0.25 / 13.6, 0.0625 / 13.6
    c = lambda *sh: rng.normal(size=sh) + 1j * rng.normal(size=sh)
    psi = c(nk, L, s, n) / 3
    enk = np.sort(np.concatenate([rng.uniform(-0.6, -0.55, (nk, 1)), rng.uniform(-0.05, -0.01, (nk, 4)),
                                  rng.uniform(0.01, 0.08, (nk, L - 5))], axis=1), axis=1)
    f = np.where(enk < 0, 1.0, 0.0)
    omega = np.array([-0.03, 0.0, 0.02, 0.05])
    kw = np.arange(nk)
    KMQ = np.array([[(k - q) % nk for k in range(nk)] for q in range(nk)])
    qneg = np.array([(-q) % nk for q in range(nk)])
    KMQ_pad = np.concatenate([KMQ, KMQ[:2] * 0]); qneg_pad = np.concatenate([qneg, [0, 0]])
    qvalid = np.r_[np.ones(nk), np.zeros(nq_pad - nk)]
    u_all, w_all = np.array([0.01, 0.03, 0.2]), np.array([0.02, 0.05, 0.3])
    rep = NamedSharding(mesh, P())
    put = lambda a: jax.device_put(np.asarray(a), rep)
    eps_full = enk[:, :m]                         # row 0 is the deep (semicore-like) band at ~ -0.58 Ry
    eps_cut = eps_full[:, o0:]
    # reach: the deep own row dominates the whole-window need; the cut need is the omega grid's
    need_full = residue_reach_need(omega, eps_full, enk, f)
    need_cut = residue_reach_need(omega, eps_cut, enk, f)
    assert need_cut < need_full, (need_cut, need_full)
    J = int(np.ceil(need_full / h)) + 2           # J held FIXED at the whole-window reach for both runs
    psi_q, psi_nq = pair_psi_slabs(carrier_band_blocks(put(psi), mesh), mesh, rows_q=KMQ_pad[:, kw],
                                   rows_nq=KMQ_pad[qneg_pad][:, kw], l0=0, L=L, L_neg=L, block_bands=2)
    def tables(eps):
        return dict(psi_q=psi_q, psi_nq=psi_nq, psi_M=put(psi[:, :m]), KMQ=put(KMQ_pad.astype(np.int32)),
                    kw=put(kw.astype(np.int32)), enk=put(enk), f=put(f), omega=put(omega), eps_own=put(eps),
                    qneg=put(qneg_pad.astype(np.int32)), qvalid=put(qvalid), u_all=put(u_all), w_all=put(w_all))
    beta = tuple(float(b) for b in laws.BETA_RY)
    kern = lambda **kw_: build_contraction_kernels(mesh, nk=nk, nq_loc=nq_loc, eta=eta, h_ry=h, beta=beta, **kw_)
    Tf, Tc = tables(eps_full), tables(eps_cut)
    qpar4 = NamedSharding(mesh, P(None, ("x", "y"))); qpar3 = NamedSharding(mesh, P(("x", "y")))
    zeros = lambda *sh: put(np.zeros(sh, complex))
    out, bit = {}, {}
    def cmp(name, acc_f, own_f, acc_c, own_c):
        a_f, a_c, o_f, o_c = (np.asarray(v) for v in (acc_f, acc_c, own_f, own_c))
        assert o_c.shape == (nk, m - o0) and a_c.shape == a_f.shape, (o_c.shape, a_c.shape)
        scale = max(np.abs(o_f).max(), 1e-300)
        out[name + "_own"] = float(np.abs(o_c - o_f[:, o0:]).max() / scale)
        out[name + "_cube"] = float(np.abs(a_c - a_f).max() / max(np.abs(a_f).max(), 1e-300))
        bit[name] = bool(np.array_equal(o_c, o_f[:, o0:]) and np.array_equal(a_c, a_f))
    # untiled, then 5-band tiles (gw.cd.sigma.tile_sum): the own rows are the whole window's tail in both
    def check(sfx, Kf, Kc):
        nz = 4
        # imaginary axis
        W = c(nz, nq_pad, n, n); dW = c(nz, nq_pad, n, n)
        u, w = np.array([0.01, 0.05, 0.1, 0.4]), np.array([0.02, 0.04, 0.1, 0.2])
        res = []
        for K, T, mo in ((Kf, Tf, m), (Kc, Tc, m - o0)):
            acc, own = zeros(nE, nk, m, m), zeros(nk, mo)
            acc, own = K.imag(jax.device_put(W, qpar4), put(u), put(w), acc, own, T)
            res += [acc, own]
        cmp("imag" + sfx, res[0], res[1], res[2], res[3])
        # anchor
        res = []
        for K, T, mo in ((Kf, Tf, m), (Kc, Tc, m - o0)):
            res += list(K.anchor(jax.device_put(W[0], qpar3), jax.device_put(dW[0], qpar3),
                                 zeros(nE, nk, m, m), zeros(nk, mo), T))
        cmp("anchor" + sfx, *res)
        # real axis: dense tables, then the banded tables each run would build, over the same J
        full = put(np.broadcast_to(np.arange(L, dtype=np.int32), (nq_pad, nk, L)).copy())
        z = (np.arange(0, nz) * h) + 1j * eta
        res = []
        for K, T, mo in ((Kf, Tf, m), (Kc, Tc, m - o0)):
            res += list(K.real(jax.device_put(W, qpar4), jax.device_put(dW, qpar4), jnp.asarray(z), jnp.int32(0),
                               zeros(nE, nk, m, m), zeros(nk, mo), T, (full, full), (False, True)))
        cmp("real_dense" + sfx, *res)
        chunk = nz
        n_chunks = -(-J // chunk)
        res = []
        for K, T, eps, mo in ((Kf, Tf, eps_full, m), (Kc, Tc, eps_cut, m - o0)):
            tabs_fn, _ = residue_band_tables(enk, f, KMQ_pad, qneg_pad, kw, omega, eps, h, chunk, n_chunks)
            acc, own = zeros(nE, nk, m, m), zeros(nk, mo)
            for ic in range(n_chunks):
                bands, negative = tabs_fn(ic)
                if not bands:
                    continue
                Wc, dWc = W, dW                                     # same W at every chunk (a rule-free check)
                zc = (np.arange(ic * chunk, (ic + 1) * chunk) * h) + 1j * eta
                acc, own = K.real(jax.device_put(Wc, qpar4), jax.device_put(dWc, qpar4), jnp.asarray(zc),
                                  jnp.int32(ic * chunk), acc, own, T, tuple(put(b) for b in bands), negative)
            res += [acc, own]
        cmp("real_banded" + sfx, *res)
    check("", kern(), kern(own0=o0))
    check("_tile5", kern(band_tile=5), kern(band_tile=5, own0=o0))
    # helpers and gates
    assert own_energy_window(0, 48, 1) == 0 and own_energy_window(0, 48, 9) == 8
    for bad in (49, 100):
        try:
            own_energy_window(0, 48, bad)
        except ValueError as exc:
            assert "GATE cd_own_energy_band_min" in str(exc), str(exc)
        else:
            raise AssertionError(bad)
    print("CDWINDOW", out, "bitwise", bit, "need", need_full, need_cut, "J", J)
    assert max(out.values()) < 1e-13, out
''')

RESUME = textwrap.dedent(r'''
    import os, sys, tempfile, numpy as np, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import gw.cd.sigma as cs
    from gw.cd.sigma import stream_cd_chunks, CdCheckpoint, parse_stop_after, real_replay_start, GH
    assert jax.device_count() == 4, jax.devices()
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    rep = NamedSharding(mesh, P())
    put = lambda a: jax.device_put(np.asarray(a), rep)
    host = lambda a: np.asarray(a)
    h, eta = 0.013, 0.021
    w4, w3 = np.arange(1, 5)[None, :, None], np.arange(1, 4)[None, None, :]

    def make_chi(chunk, first_at=(0,)):
        """The planned windows of gw.cd.chi0_rules.cd_windows; ``first_at`` chunks get the ghost-extended form
        (only chunk 0 in a correct run: a resume that recomputed its ghosts in a fresh call is the red twin)."""
        def chi_real(ic):
            j0 = ic * chunk
            lo, hi = (j0 - GH, j0 + chunk + GH) if ic in first_at else (j0 + GH, j0 + chunk + GH)
            zz = h * np.arange(lo, hi) + 1j * eta
            # depends on the WHOLE batch (mean): a call with another z batch changes every value
            return put(np.sin(3 * zz)[:, None, None] * w4 + np.cos(zz)[:, None, None] * w3 + 1e-3 * zz.mean())
        return chi_real
    join = jax.jit(lambda a, b: jnp.concatenate([a, b], axis=0))
    tail_of = jax.jit(lambda a: a[a.shape[0] - 2 * GH:] * 1.0)

    class Kill(Exception):
        pass

    def make(kill=None, seen=None):
        def imag_step(ic, c0, c1, accs):
            if kill == ("imag", ic):
                raise Kill
            g = put(np.full((4, 3), 0.1 * (ic + 1) + 0.01j * c0))
            return dict(accs, acc_imag=accs["acc_imag"] * 0.999 + g, own_imag=accs["own_imag"] * 0.99 + g[0])
        def real_step(ic, j0, j1, box, accs):
            if kill == ("real", ic):
                raise Kill
            chi_e = box.pop()
            if seen is not None:
                seen[ic] = host(chi_e).tobytes()
            core = chi_e[GH:chi_e.shape[0] - GH]
            slope = (chi_e[2 * GH:] - chi_e[:chi_e.shape[0] - 2 * GH]) / (2 * GH * h)
            inc = jnp.sum(core * slope, axis=0)
            out = dict(accs, acc_real=accs["acc_real"] * 0.999 + inc, own_real=accs["own_real"] * 0.99 + inc[1])
            if j0 == 0:
                out.update(acc_anc=accs["acc_anc"] + chi_e[GH], own_anc=accs["own_anc"] + chi_e[GH, 0])
            return out
        return imag_step, real_step

    def zeros():
        return dict(acc_imag=put(np.zeros((4, 3), complex)), own_imag=put(np.zeros(3, complex)),
                    acc_real=put(np.zeros((4, 3), complex)), own_real=put(np.zeros(3, complex)),
                    acc_anc=put(np.zeros((4, 3), complex)), own_anc=put(np.zeros(3, complex)))

    def run(chunk, *, kill=None, ckpt=None, resume=None, stop_after=None, seen=None, first_at=(0,)):
        imag_chunks = [(c0, c0 + 2) for c0 in range(0, 10, 2)]
        real_chunks = [(j0, j0 + chunk) for j0 in range(0, 7 * chunk, chunk)]
        imag_step, real_step = make(kill, seen)
        on_chunk = None if ckpt is None else (lambda axis, ic, accs: ckpt.write(axis, ic, accs, timings={"x": 1.0}))
        return stream_cd_chunks(imag_chunks=imag_chunks, real_chunks=real_chunks, chi_real=make_chi(chunk, first_at), join=join,
                                tail_of=tail_of, imag_step=imag_step, real_step=real_step, accs=zeros(),
                                resume=resume, on_chunk=on_chunk, stop_after=stop_after)

    def same(a, b):
        return all(np.array_equal(host(a[k]), host(b[k])) for k in a)

    tmp = tempfile.mkdtemp()
    identity = dict(deck_sha256="d" * 64, tree="t", J=7, chunk_real=None)
    report = []
    for chunk in (2, 3, 8, 9):                         # 2, 3 < 2*GH = 8 <= 8, 9
        ref_seen = {}
        ref, info = run(chunk, seen=ref_seen)
        assert info["real_run"] == 7 and info["imag_run"] == 5, info
        ident = dict(identity, chunk_real=chunk)
        for kills in ([("imag", 0)], [("imag", 3)], [("real", 0)], [("real", 1)], [("real", 4)], [("real", 6)],
                      [("real", 2), ("real", 5)], [("imag", 2), ("real", 3)]):
            path = os.path.join(tmp, f"c{chunk}_{len(report)}", "tmp", "cd_checkpoint.h5")
            ck = CdCheckpoint(path, ident, mesh, print_fn=lambda *a: None)
            resume, seen = None, {}
            for kill in kills:
                try:
                    run(chunk, kill=kill, ckpt=ck, resume=resume, seen=seen)
                    raise AssertionError(f"no kill at {kill}")
                except Kill:
                    pass
                resume = ck.read()
            final, info = run(chunk, ckpt=ck, resume=resume, seen=seen)
            ok = same(final, ref) and all(seen[i] == ref_seen[i] for i in seen)
            report.append((chunk, kills, ok, info["replayed"]))
            assert ok, (chunk, kills, info)
        # stop-after, then continue (the probe and the production stop-and-report)
        path = os.path.join(tmp, f"c{chunk}_stop", "tmp", "cd_checkpoint.h5")
        ck = CdCheckpoint(path, ident, mesh, print_fn=lambda *a: None)
        _, info = run(chunk, ckpt=ck, stop_after=parse_stop_after("real:2"))
        assert info["stopped"] and info["real_run"] == 2, info
        final, info = run(chunk, ckpt=ck, resume=ck.read())
        assert same(final, ref) and info["resumed_from"] == ["real", 1], info
        # real_from (top-window probe): the last chunk's ghost-extended chi0 equals the uninterrupted one
        seen = {}
        _, info = run(chunk, seen=seen, stop_after=parse_stop_after("imag:0,real:1,real_from:-1"))
        assert info["imag_truncated"] and info["real_run"] == 1 and list(seen) == [6], info
        assert seen[6] == ref_seen[6] and info["replayed"] == min(6, -(-2 * GH // chunk)), info
        assert real_replay_start(6, chunk) == max(0, 6 - (-(-2 * GH // chunk))), chunk
        # red twin: the naive resume (no replay; the resumed chunk recomputes its ghosts in a fresh call)
        path = os.path.join(tmp, f"c{chunk}_red", "tmp", "cd_checkpoint.h5")
        ck = CdCheckpoint(path, ident, mesh, print_fn=lambda *a: None)
        try:
            run(chunk, kill=("real", 4), ckpt=ck)
        except Kill:
            pass
        keep = cs.real_replay_start
        cs.real_replay_start = lambda start, chunk, gh=GH: start
        try:
            red, _ = run(chunk, ckpt=ck, resume=ck.read(), first_at=(0, 4))
        finally:
            cs.real_replay_start = keep
        assert not same(red, ref), f"red twin passed at chunk {chunk}: the toy chi0 is not batch-sensitive"
    # a foreign identity is refused and the file left in place
    ck = CdCheckpoint(path, dict(ident, J=8), mesh, print_fn=lambda *a: None)
    assert ck.read() is None and os.path.exists(path)
    for bad in ("imag", "imag:x", "real:-1", "foo:1"):
        try:
            parse_stop_after(bad)
        except ValueError as exc:
            assert "GATE cd_stop_after" in str(exc)
        else:
            raise AssertionError(bad)
    assert parse_stop_after(" imag:1, real:2 ,real_from:-1") == dict(imag=1, real=2, real_from=-1)
    print("CDRESUME", len(report), "cases bitwise;", [(r[0], r[3]) for r in report][:6])
''')

CONFIG = textwrap.dedent(r'''
    from gw import gw_config as gc
    base = dict(gc._DEFAULTS)
    base.update(sigma_frequency_integration="contour_deformation", compute_mode="mpa", head_correction="off",
                linalg="local", sigma_regularization_ev=0.25, use_band_extrapolation=False)
    def check(**kv):
        p = dict(base, **kv)
        p[gc._DECK_NAMED_KEYS] = set(kv) | {"sigma_frequency_integration", "use_band_extrapolation"}
        gc._resolve_cd_inputs(p)
        return p
    p = check(cd_own_energy_band_min=9, cd_checkpoint="Resume")
    assert p["cd_own_energy_band_min"] == 9 and p["cd_checkpoint"] == "resume", p
    for kv, gate in ((dict(cd_own_energy_band_min=0), "cd_own_energy_band_min"),
                     (dict(cd_checkpoint="auto"), "cd_checkpoint")):
        try:
            check(**kv)
        except ValueError as exc:
            assert f"GATE {gate}" in str(exc), str(exc)
        else:
            raise AssertionError(kv)
    p = dict(gc._DEFAULTS); p[gc._DECK_NAMED_KEYS] = {"cd_checkpoint"}; p["cd_checkpoint"] = "write"
    try:
        gc._resolve_cd_inputs(p)
    except ValueError as exc:
        assert "GATE cd_applicability" in str(exc), str(exc)
    else:
        raise AssertionError("cd_checkpoint admitted on the pole route")
    print("CDCONFIG ok")
''')

PARTIAL = textwrap.dedent(r"""
    import os, tempfile
    from types import SimpleNamespace
    import numpy as np, jax
    jax.config.update("jax_enable_x64", True)
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.wfn_layout import PSI_NMU_SPEC, PSI_MUN_SPEC
    from gw.cd import chi0_rules as cr
    from gw.cd.sigma import bank_chi0
    assert jax.device_count() == 4
    mesh = Mesh(np.array(jax.devices()).reshape(2, 2), ("x", "y"))
    put = lambda a, sp: jax.device_put(a, NamedSharding(mesh, sp))
    quiet = lambda *a, **k: None
    rng = np.random.default_rng(11)
    grid, nb, nbl, s, n, nocc = (2, 2, 1), 12, 10, 1, 8, 4             # the fixture of tests/test_cd_bank_chi0.py
    nk = int(np.prod(grid))
    psi = (rng.standard_normal((nk, nb, s, n)) + 1j * rng.standard_normal((nk, nb, s, n))) / 3
    e = np.sort(np.concatenate([rng.uniform(-1.0, -0.2, (nk, nocc)), rng.uniform(0.2, 1.5, (nk, nb - nocc))], axis=1), axis=1)
    f = np.zeros((nk, nb)); f[:, :nocc] = 1.0
    q_ids = np.array([0, 1, 3])
    meta = SimpleNamespace(nk_tot=nk, nkx=grid[0], nky=grid[1], nkz=grid[2], kgrid=grid, nspinor=s, nspin=1,
                           b_id_4_chi_user=nbl, mu_basis=SimpleNamespace(n_packed=n, n_logical=n))
    fx = SimpleNamespace(enk=put(e, P(None, None)), occ=put(f, P(None, None)), layout="face", green_parent=None,
                         psi_mun=put(psi.transpose(0, 2, 3, 1), PSI_MUN_SPEC), psi_nmu=put(psi, PSI_NMU_SPEC),
                         slices=SimpleNamespace(b0=0, b4_logical=nbl, nb_full=nb))
    knots = np.array([-0.1, -0.05, 0.0, 0.45, 0.9, 1.6])
    imag = [1j * np.array([0.04, 0.3, 2.0])]
    real = [knots[:2] + 0.05j, knots[2:4] + 0.05j, knots[4:] + 0.05j]
    points, windows = cr.window_members(imag, real)
    tmp = tempfile.mkdtemp()
    full, _ = bank_chi0(fx, meta, mesh, points, q_ids=q_ids, mu_ry=0.0, rule_tolerance=1e-8, slots=4, windows=windows,
                        rule_capacity=192, save_path=os.path.join(tmp, "full.pkl"), print_fn=quiet)
    payload, _ = cr.build_rules(fx, meta, points, windows, mu_ry=0.0, tolerance=1e-8, slots=4, capacity=192,
                                serial=True, build=[0, 2], print_fn=quiet)            # imag:0 and real:1 only
    assert payload["partial"], payload["built"]
    path = os.path.join(tmp, "partial.pkl")
    cr.save_rules(path, payload)
    try:
        bank_chi0(fx, meta, mesh, points, q_ids=q_ids, mu_ry=0.0, rule_tolerance=1e-8, slots=4, windows=windows,
                  rule_capacity=192, rules_file=path, print_fn=quiet)
        raise AssertionError("a partial rule file loaded outside the instrument")
    except ValueError as exc:
        assert "GATE cd_chi0_rules_file" in str(exc), str(exc)
    part, note = bank_chi0(fx, meta, mesh, points, q_ids=q_ids, mu_ry=0.0, rule_tolerance=1e-8, slots=4,
                           windows=windows, rule_capacity=192, rules_file=path, allow_partial=True, print_fn=quiet)
    assert note["partial_rules"], note
    built = np.concatenate([imag[0], real[1]])
    a, b = np.asarray(part(built)), np.asarray(full(built))
    dev = float(np.abs(a - b).max() / np.abs(b).max())
    try:
        part(real[0])
        raise AssertionError("an unbuilt window evaluated")
    except ValueError as exc:
        assert "GATE cd_chi0_rule_unbuilt" in str(exc), str(exc)
    print("CDPARTIAL built windows vs full build: max rel", dev, "bitwise", bool(np.array_equal(a, b)))
    assert dev < 1e-14, dev
""")


def _run(script, devices=None, timeout=900):
    env = dict(os.environ, **ENV)
    if devices:
        env["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={devices}").strip()
    run = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=timeout)
    return run


def test_cd_own_energy_rows_equal_the_whole_window_at_fixed_J():
    run = _run(KERNELS, devices=4)
    print(run.stdout[-1500:])
    assert run.returncode == 0 and "CDWINDOW" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]


def test_cd_resume_reproduces_the_uninterrupted_accumulators_bitwise():
    run = _run(RESUME, devices=4)
    print(run.stdout[-1500:])
    assert run.returncode == 0 and "CDRESUME" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]


def test_cd_own_energy_and_checkpoint_deck_gates():
    run = _run(CONFIG)
    print(run.stdout[-1500:])
    assert run.returncode == 0 and "CDCONFIG ok" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]


def test_cd_partial_rules_only_under_the_instrument():
    run = _run(PARTIAL, devices=4)
    print(run.stdout[-1500:])
    assert run.returncode == 0 and "CDPARTIAL" in run.stdout, run.stdout[-3000:] + run.stderr[-3000:]
