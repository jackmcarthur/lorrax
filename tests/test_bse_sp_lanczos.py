"""Gates for the matrix-free structure-preserving non-TDA solver.

Two tiers, deliberately.

**Tier 1 — the algebra, deck-free.**  A synthetic definite Bethe-Salpeter
Hamiltonian built in numpy (``A`` exactly Hermitian, ``B`` exactly complex
symmetric) and its exact dense spectrum.  These run on CPU in seconds and they
are where the SOLVER is gated: the recurrence, the two-coefficient
reorthogonalisation, the real restart rotation, the X/Y lift and its closed-form
normalisation.  Every one has a red twin that breaks exactly one of those and
asserts the gate goes red -- because a solver gate that cannot fail is not a
gate.  The single-coefficient-set twin is the important one: it returns pairs
that are GENUINE eigenpairs of H with a residual at round-off and an invariant
(a) that looks perfect, and they are the wrong eigenvalues (each one doubled,
half the spectrum missing).  Nothing but invariant (b) catches it.

**Tier 2 — the kernel, on the deck fixture.**  The fusion identity against the
ring half-appliers on ``bse_dense_state`` (MoS2 3x3x1, 2v2c, N=36), and the
statement that ``pair(X, s=0)`` IS the shipped TDA stack matvec.  These gate the
PORT rather than the solver, and they are cheap enough to run every time.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from solvers import bse_sp_lanczos as spl  # noqa: E402


# ===========================================================================
# Tier 1 — synthetic definite BSE
# ===========================================================================

def _synthetic_bse(N=120, seed=0, coupling=0.30):
    """``A`` Hermitian positive definite, ``B`` complex SYMMETRIC.

    Positive definiteness of ``K = [[A,B],[B*,A*]]`` is the method's
    precondition, so the diagonal shift is chosen to guarantee it and the test
    asserts it rather than hoping.
    """
    rng = np.random.default_rng(seed)
    Ar = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A = Ar + Ar.conj().T
    A = A + (np.linalg.norm(A, 2) * 1.5) * np.eye(N)
    Br = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    B = coupling * (Br + Br.T) / 2.0
    return A, B


def _appliers(A, B):
    Aj = jnp.asarray(A)
    Bj = jnp.asarray(B)

    def F(Z):                      # Z: (b, N)
        return Z @ Aj.T + jnp.conj(Z) @ Bj.T

    def G(Z):
        return Z @ Aj.T - jnp.conj(Z) @ Bj.T
    return F, G


def _dense_positive_spectrum(A, B, n):
    H = np.block([[A, B], [-B.conj(), -A.conj()]])
    lam = np.linalg.eigvals(H)
    assert np.abs(lam.imag).max() < 1e-8 * np.abs(lam.real).max(), \
        "the synthetic operator must have a real spectrum"
    return np.sort(lam.real[lam.real > 0])[:n], H


@pytest.fixture(scope="module")
def synth():
    N = 120
    A, B = _synthetic_bse(N)
    K = np.block([[A, B], [B.conj(), A.conj()]])
    assert np.linalg.eigvalsh(0.5 * (K + K.conj().T)).min() > 0, \
        "K must be positive definite -- otherwise the method does not apply"
    ref, H = _dense_positive_spectrum(A, B, 8)
    return dict(N=N, A=A, B=B, H=H, ref=ref)


def _solve(synth, **kw):
    F, G = _appliers(synth["A"], synth["B"])
    kw.setdefault("n_eig", 8)
    kw.setdefault("m_max", 60)
    kw.setdefault("n_keep", 20)
    kw.setdefault("n_restarts", 6)
    om, X, Y, dg = spl.sdy_lanczos_eig(F, G, (synth["N"],), **kw)
    return (np.asarray(om), np.asarray(X), np.asarray(Y),
            {k: np.asarray(v) for k, v in dg.items()})


def test_sdy_reproduces_the_dense_nonhermitian_spectrum(synth):
    """The whole method, end to end, against ``eigvals([[A,B],[-B*,-A*]])``."""
    om, X, Y, dg = _solve(synth)
    err = np.abs(om - synth["ref"]).max()
    assert err < 1e-9, f"worst |dlambda| = {err:.3e} (dense {synth['ref']})"
    # invariants (a) and (b), on the returned Ritz block
    assert dg["orth_err"] < 1e-10, f"Re(U^H V) != I: {dg['orth_err']:.3e}"
    assert dg["im_uu"] < 1e-10, f"Im(U^H U) != 0: {dg['im_uu']:.3e}"
    assert dg["im_vv"] < 1e-8, f"Im(V^H V) != 0: {dg['im_vv']:.3e}"
    # the definiteness certificates
    assert dg["kappa_start"] > 0
    assert dg["beta_sq_min"] > 0
    # the metric-symmetry certificate, which is what alpha_im is NOT
    assert dg["metric_sym_err"] < 1e-12, \
        f"metric asymmetry {dg['metric_sym_err']:.3e}"


def test_xy_normalisation_is_exact_and_the_pairs_solve_H(synth):
    """``X^H X - Y^H Y = +1`` by closed form, and ``[X;Y]`` really is an
    eigenvector of the FULL non-Hermitian operator -- not merely of ``T``."""
    om, X, Y, _dg = _solve(synth)
    snorm = (np.sum(np.abs(X) ** 2, axis=1) - np.sum(np.abs(Y) ** 2, axis=1))
    assert np.abs(snorm - 1.0).max() < 1e-10, \
        f"X^HX - Y^HY: max |.-1| = {np.abs(snorm - 1).max():.3e}"
    H = synth["H"]
    for j in range(len(om)):
        z = np.concatenate([X[j], Y[j]])
        r = np.linalg.norm(H @ z - om[j] * z) / om[j]
        assert r < 1e-10, f"state {j}: ||Hz - wz||/w = {r:.3e}"


def test_red_twin_one_coefficient_set_is_caught(synth):
    """RED TWIN.  Enforce invariant (a) only -- the failure the derivation's
    §4.2 warns about, and the one that does NOT announce itself.

    The assertions are ordered to make the point: the run's residual and its
    invariant (a) stay at round-off, so a residual-based or orthogonality-based
    convergence test passes.  The EIGENVALUES are wrong (each returned twice,
    half the spectrum missing) and only invariant (b) sees it coming.
    """
    om_ok, _X, _Y, dg_ok = _solve(synth)
    om, X, Y, dg = _solve(synth, single_coeff_set=True)
    # what still looks fine
    assert dg["orth_err"] < 1e-10, \
        "invariant (a) is expected to survive -- that is the whole problem"
    for j in range(len(om)):
        z = np.concatenate([X[j], Y[j]])
        assert np.linalg.norm(synth["H"] @ z - om[j] * z) / om[j] < 1e-8
    # what is actually broken
    assert dg["im_vv"] > 1e-4, \
        f"invariant (b) should be destroyed, got im_vv = {dg['im_vv']:.3e}"
    # The free per-step detector, scored as a RATIO against the control rather
    # than against an absolute threshold: its absolute size scales with the
    # coupling strength (5e-7 on this synthetic fixture, 2.5 on the Si 4x4x4
    # deck), while the separation from a healthy run is ~9 orders either way.
    # An absolute threshold would be measuring the fixture's coupling.
    assert dg_ok["imag_drift"] < 1e-12, "the good run must NOT drift"
    assert dg["imag_drift"] > 1e6 * max(float(dg_ok["imag_drift"]), 1e-18), \
        (f"the free per-step detector should fire: twin "
         f"{dg['imag_drift']:.3e} vs control {dg_ok['imag_drift']:.3e}")
    err = np.abs(om - synth["ref"]).max()
    assert err > 1e-3, f"the spectrum should be wrong, got {err:.3e}"
    # the signature failure: duplicated levels
    assert len(np.unique(np.round(om, 8))) < len(om), \
        "the single-set run returns each eigenvalue twice"
    assert np.abs(om_ok - synth["ref"]).max() < 1e-9, "control must be right"


def test_red_twin_complex_restart_rotation_is_caught(synth):
    """RED TWIN.  ``F(U Q) = V Q`` holds because ``Q`` is REAL; make it complex
    and the companion basis stops being ``F`` of the first one."""
    om, _X, _Y, dg = _solve(synth, q_phase=0.3)
    assert np.abs(om - synth["ref"]).max() > 1e-4
    assert dg["orth_err"] > 1e-6, \
        f"a complex Q must break Re(U^H V) = I, got {dg['orth_err']:.3e}"


def test_red_twin_a_broken_operator_is_caught_by_metric_symmetry(synth):
    """The operator-integrity gate, and the correction it encodes.

    ``metric_sym_err`` fires on a non-Hermitian ``A`` and on a non-symmetric
    ``B``.  ``alpha_im_rel`` -- which NONTDA_MATRIXFREE_DERIVATION.md §4.6
    proposes as the detector, with a 1e-4 threshold -- does NOT distinguish
    them, because it is nonzero for a perfectly correct operator too: the piece
    it measures is ``Im(x̄^T B x̄)``, a complex-SYMMETRIC quadratic form with no
    reality theorem.  This test pins both halves of that correction.
    """
    A, B = synth["A"], synth["B"]
    rng = np.random.default_rng(31)
    # a correct operator: metric symmetric, alpha_im NOT small
    _om, _X, _Y, dg_ok = _solve(synth, m_max=30, n_keep=10, n_restarts=1)
    assert dg_ok["metric_sym_err"] < 1e-12
    assert dg_ok["alpha_im_rel"] > 1e-6, (
        "alpha_im is NOT an error indicator with a coupling block present; if "
        "this ever becomes small the fixture lost its coupling, not the code "
        "its bug")
    N = synth["N"]
    for label, Ab, Bb in (
            ("A non-Hermitian",
             A + 1e-3 * np.linalg.norm(A) * rng.standard_normal((N, N)), B),
            ("B non-symmetric", A,
             B + 1e-3 * np.linalg.norm(B) * rng.standard_normal((N, N)))):
        F, G = _appliers(Ab, Bb)
        _o, _x, _y, dg = spl.sdy_lanczos_eig(
            F, G, (N,), n_eig=4, m_max=30, n_keep=10, n_restarts=1)
        assert float(dg["metric_sym_err"]) > 1e-6, \
            f"{label}: metric_sym_err = {float(dg['metric_sym_err']):.3e}"


def test_b_to_zero_is_the_hermitian_limit(synth):
    """With ``B = 0`` the method must return ``eigvalsh(A)`` -- and only then
    does ``alpha_im_rel`` collapse to round-off."""
    A = synth["A"]
    Z = np.zeros_like(A)
    F, G = _appliers(A, Z)
    ref = np.sort(np.linalg.eigvalsh(A))[:8]
    om, _X, _Y, dg = spl.sdy_lanczos_eig(
        F, G, (synth["N"],), n_eig=8, m_max=60, n_keep=20, n_restarts=6)
    assert np.abs(np.asarray(om) - ref).max() < 1e-9
    assert float(dg["alpha_im_rel"]) < 1e-12, \
        "in the TDA limit alpha_im IS the Hermitian detector"


def test_fixed_shape_trace_counts_do_not_depend_on_n_restarts(synth):
    """THE fixed-shape claim, stated so it can fail.

    Three traced bodies exist -- the step (twice: cold cycle and restart cycle),
    the two ``_build_T`` branches, and the restart -- and the counts are the
    same for 2 restart cycles and for 20.  ``compile_cache_stats`` cannot make
    this statement: with the persistent cache warm it reads zero compiles no
    matter how many distinct programs the loop dispatches.
    """
    counts = {}
    for nr in (2, 20):
        spl.reset_trace_counts()
        _solve(synth, n_eig=4, m_max=30, n_keep=10, n_restarts=nr)
        counts[nr] = dict(spl.TRACE_COUNTS)
    assert counts[2] == counts[20], counts
    # Every body is traced a CONSTANT number of times.  The step body has at
    # most two traces -- the cold cycle's fori_loop and the restart cycle's --
    # and jax is free to share one jaxpr between them when the avals match, so
    # the claim is "bounded and independent of n_restarts", not a fixed 2.
    # Pinning it to 2 would fail on a jax that got better at caching.
    assert 1 <= counts[2]["sdy_step"] <= 2, counts[2]
    assert counts[2]["build_T_first"] == 1
    assert counts[2]["build_T_restart"] == 1
    assert counts[2]["restart"] == 2


def test_pair_application_count_is_stated_not_sampled():
    """The cost model is arithmetic on the loop shape, so it cannot drift."""
    assert spl.sdy_steps(140, 40, 3) == 140 + 3 * 100
    assert spl.sdy_pair_applications(140, 40, 3) == 2 + 2 * 440
    # two per step: one G, one F.  Plus the start normalisation and the
    # metric-symmetry certificate, each one application, each paid once.
    assert spl.sdy_pair_applications(10, 5, 0) == 2 + 20


def test_refuses_a_malformed_shape_request(synth):
    with pytest.raises(ValueError, match="n_eig"):
        _solve(synth, n_eig=30, n_keep=20, m_max=60)


# ===========================================================================
# Tier 2 — the kernel port, on the deck fixture
# ===========================================================================

def _pair_ctx(data):
    """1x1 mesh + the argument tuple both stack matvecs take."""
    from jax.sharding import Mesh
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)
    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_y)
        W_R = jnp.fft.ifftn(jnp.array(data["W_q"]), axes=(2, 3, 4), norm="ortho")
        W_R = jax.lax.with_sharding_constraint(W_R, sh.W)
        V_q0 = jax.lax.with_sharding_constraint(data["V_q0"], sh.V)
        M_X = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(psi_c_X, psi_v_X), sh.psi_x)
        M_Y = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(psi_c_Y, psi_v_Y), sh.psi_y)
    args = (psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
            jnp.asarray(data["eps_c"]), jnp.asarray(data["eps_v"]),
            W_R, V_q0, M_X, M_Y)
    return mesh, sh, args


def _random_X(nb, nc, nv, nk, seed=1234):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((nb, nc, nv, nk))
                       + 1j * rng.standard_normal((nb, nc, nv, nk)))


@pytest.mark.gpu
def test_fused_pair_matches_the_ring_half_appliers(bse_dense_state):
    """GATE 1 of the derivation, at the tightest tolerance it allows.

    ``decode(conv(encode_A(x) + s*encode_B(x̄)))`` against the ring path's two
    independent chains.  This is a contraction REASSOCIATION of a sum, not a
    bit-exact rearrangement, so 1e-12 relative is the standard (cf. the
    ``contract_bands_block_reshard`` note in ``bse_ring_comm.py``).

    RED TWIN: the same comparison with the coupling sign flipped, which must be
    O(1) -- otherwise the gate is measuring nothing, because ``B`` could be
    zero and everything above would still pass.
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_ring_comm import build_bse_ring_matvec_full
    from bse.bse_stack_matvec import build_bse_stack_pair_matvec
    data = bse_dense_state
    mesh, sh, args = _pair_ctx(data)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["psi_c"].shape[1])
    nv = int(data["psi_v"].shape[1])
    with mesh:
        _mv, ap_A, ap_B = build_bse_ring_matvec_full(
            mesh, nkx, nky, nkz, include_W=True, screening=False,
            return_half_appliers=True)
        pair = build_bse_stack_pair_matvec(mesh, nkx, nky, nkz)
        X = jax.lax.with_sharding_constraint(_random_X(3, nc, nv, nk), sh.X)
        a = np.asarray(ap_A(X, args[0], args[1], args[2], args[3], args[4],
                            args[5], args[6], args[7], args[8]))
        b = np.asarray(ap_B(jnp.conj(X), args[0], args[1], args[2], args[3],
                            args[6], args[7], args[8]))
        for s in (1.0, -1.0):
            got = np.asarray(pair(X, jnp.asarray(s), *args))
            want = a + s * b
            rel = np.linalg.norm(got - want) / np.linalg.norm(want)
            assert rel < 1e-12, f"s={s:+.0f}: fusion identity rel {rel:.3e}"
        # RED TWIN — the sign of the coupling must matter
        got_p = np.asarray(pair(X, jnp.asarray(1.0), *args))
        wrong = a - b
        red = np.linalg.norm(got_p - wrong) / np.linalg.norm(wrong)
        assert red > 1e-8, (
            f"flipping the coupling sign changed nothing (rel {red:.3e}) -- "
            f"the B block is not contributing and this gate is void")


@pytest.mark.gpu
def test_pair_at_s_zero_is_the_shipped_tda_matvec(bse_dense_state):
    """The B->0 limit of the port, at the kernel level.

    ``pair(X, 0)`` and ``build_bse_stack_matvec`` must agree to round-off: the
    A-block path through the ported code is the SAME encode / conv / decode the
    shipped TDA matvec runs, factored out rather than copied.  A tolerance of
    1e-13 rather than bit-equality because ``s * T_B`` is still summed into
    ``T_A`` at ``s = 0``, which is an extra add the shipped path does not do.
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_stack_matvec import (build_bse_stack_matvec,
                                      build_bse_stack_pair_matvec)
    data = bse_dense_state
    mesh, sh, args = _pair_ctx(data)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["psi_c"].shape[1])
    nv = int(data["psi_v"].shape[1])
    with mesh:
        tda = build_bse_stack_matvec(mesh, nkx, nky, nkz)
        pair = build_bse_stack_pair_matvec(mesh, nkx, nky, nkz)
        X = jax.lax.with_sharding_constraint(_random_X(3, nc, nv, nk), sh.X)
        want = np.asarray(tda(X, *args))
        got = np.asarray(pair(X, jnp.asarray(0.0), *args))
        rel = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert rel < 1e-13, f"pair(X, 0) vs TDA stack matvec: rel {rel:.3e}"
        # and the coupling must actually be doing something at s = 1
        got1 = np.asarray(pair(X, jnp.asarray(1.0), *args))
        assert (np.linalg.norm(got1 - want) / np.linalg.norm(want)) > 1e-10


@pytest.mark.gpu
def test_unfused_twin_agrees_with_the_fused_pair(bse_dense_state):
    """The twin that prices the fusion must be VALUE-identical to it, or the
    speedup it measures is a speedup over a different calculation."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_stack_matvec import build_bse_stack_pair_matvec
    data = bse_dense_state
    mesh, sh, args = _pair_ctx(data)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["psi_c"].shape[1])
    nv = int(data["psi_v"].shape[1])
    with mesh:
        fused = build_bse_stack_pair_matvec(mesh, nkx, nky, nkz)
        unf = build_bse_stack_pair_matvec(mesh, nkx, nky, nkz, fuse=False)
        X = jax.lax.with_sharding_constraint(_random_X(3, nc, nv, nk), sh.X)
        for s in (1.0, -1.0):
            a = np.asarray(fused(X, jnp.asarray(s), *args))
            b = np.asarray(unf(X, jnp.asarray(s), *args))
            rel = np.linalg.norm(a - b) / np.linalg.norm(b)
            assert rel < 1e-12, f"s={s:+.0f}: fused vs unfused rel {rel:.3e}"


@pytest.mark.gpu
def test_sdy_solver_on_the_deck_fixture_matches_dense(bse_dense_state):
    """The whole ladder, end to end, on a real (tiny) BSE operator: the
    matrix-free solve against ``eigvals([[A,B],[-B*,-A*]])`` assembled from the
    ring half-appliers on the same operator instance in the same process."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_ring_comm import build_bse_ring_matvec_full
    from bse.bse_stack_matvec import build_bse_stack_pair_matvec
    data = bse_dense_state
    mesh, sh, args = _pair_ctx(data)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["psi_c"].shape[1])
    nv = int(data["psi_v"].shape[1])
    N = nc * nv * nk
    with mesh:
        _mv, ap_A, ap_B = build_bse_ring_matvec_full(
            mesh, nkx, nky, nkz, include_W=True, screening=False,
            return_half_appliers=True)
        pair = build_bse_stack_pair_matvec(mesh, nkx, nky, nkz)
        eye = jnp.asarray(np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk))
        A = np.asarray(ap_A(eye, args[0], args[1], args[2], args[3], args[4],
                            args[5], args[6], args[7], args[8])
                       ).reshape(N, N).T
        B = np.asarray(ap_B(eye, args[0], args[1], args[2], args[3],
                            args[6], args[7], args[8])).reshape(N, N).T
        H = np.block([[A, B], [-B.conj(), -A.conj()]])
        lam = np.linalg.eigvals(H).real
        ref = np.sort(lam[lam > 1e-9])[:4]

        def F(Z):
            return pair(Z, jnp.asarray(1.0), *args)

        def G(Z):
            return pair(Z, jnp.asarray(-1.0), *args)
        om, X, Y, dg = spl.sdy_lanczos_eig(
            F, G, (nc, nv, nk), n_eig=4, m_max=N - 2,
            n_keep=min(20, N - 4), n_restarts=2, sharding=None)
    om = np.asarray(om)
    assert float(dg["kappa_start"]) > 0, "K must be positive definite"
    assert float(dg["beta_sq_min"]) > 0
    assert np.abs(om - ref).max() < 1e-6, f"matrix-free {om} vs dense {ref}"
    Xn = np.asarray(X).reshape(4, -1)
    Yn = np.asarray(Y).reshape(4, -1)
    snorm = np.sum(np.abs(Xn) ** 2, 1) - np.sum(np.abs(Yn) ** 2, 1)
    assert np.abs(snorm - 1.0).max() < 1e-8


@pytest.mark.gpu
def test_driver_matrixfree_route_matches_the_dense_route(bse_dense_state):
    """The two routes through ``solve_bse_nontda_sharded`` must be
    indistinguishable from their outputs.

    Same entry point, same data, same convention -- ``solver='dense'`` assembles
    and diagonalises, ``solver='matrixfree'`` never forms a matrix.  If they
    disagree the wiring is wrong, and the failure shows up as an eigenvalue
    difference rather than as a shape error, which is the only kind of
    disagreement that matters here.
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_nontda import solve_bse_nontda_sharded
    from tests.test_bse_dense_reference import _nontda_data_from_subset
    data = bse_dense_state
    d1, mesh = _nontda_data_from_subset(data)
    om_d, ev_d, _ = solve_bse_nontda_sharded(d1, mesh, n_eig=4, include_W=True,
                                             solver="dense")
    d2, mesh2 = _nontda_data_from_subset(data)
    om_m, ev_m, _ = solve_bse_nontda_sharded(
        d2, mesh2, n_eig=4, include_W=True, solver="matrixfree",
        mf_m_max=30, mf_n_keep=12, mf_n_restarts=3)
    om_d = np.sort(np.asarray(jax.device_get(om_d)))
    om_m = np.sort(np.asarray(jax.device_get(om_m)))
    assert np.abs(om_d - om_m).max() < 1e-8, f"dense {om_d} vs mf {om_m}"
    ev_m = np.asarray(jax.device_get(ev_m))
    for i in range(4):
        X = ev_m[i, 0].reshape(-1)
        Y = ev_m[i, 1].reshape(-1)
        s = float(np.real(np.conj(X) @ X - np.conj(Y) @ Y))
        assert abs(s - 1.0) < 1e-8, f"state {i}: X^HX-Y^HY = {s:.6f}"


def test_driver_refuses_an_unknown_solver_name():
    from bse.bse_nontda import solve_bse_nontda_sharded
    with pytest.raises(ValueError, match="matrixfree"):
        solve_bse_nontda_sharded({}, None, solver="magic")
