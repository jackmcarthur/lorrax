"""Occupation weights in the BSE operator and its Davidson solves (payload key ``occupation_weights``).

With occupations ``f`` per spin-orbital the transition pairs are those with ``f_v > f_c`` and

    A = dE + sqrt(df) K_A sqrt(df),     B = sqrt(df) K_B sqrt(df'),     A' = dE' + sqrt(df') K' sqrt(df'),

``df = f_v(k) - f_c(k+Q)`` on the resonant pairs and ``df' = f_v(k+Q) - f_c(k)`` on the antiresonant
ones; a pair with ``df <= 0`` is masked (decoupled, ``MASKED_PAIR_RY`` on its diagonal).  The payload
is the synthetic one of ``tests/test_bse_full_finite_q_cpu.py`` (128 transitions) with occupations in
which two "conduction" bands are partly filled and two "valence" bands partly emptied, so that some
pairs are masked; the band sets play the role of overlapping windows.  Reference: the NumPy ``K`` of
that file's definition with the weights applied to its kernel part, and ``numpy.linalg.eig``.

1. The production blocks at a finite pair momentum equal ``diag(D) + diag(w) K_kernel diag(w)`` with
   the masked diagonal, on one device and the 2x2 mesh; at ``Q = 0`` the paired blocks do, and ``B``
   is complex symmetric; the Tamm-Dancoff matvec gives the ``A`` block.  Red twins: weights of one
   are the unweighted operator; a missing output-side weight misses the reference.
2. The solves return the lowest roots of ``eig`` on the weighted ``K``: the dense and Davidson full
   routes at finite Q and at ``Q = 0``, and the Tamm-Dancoff Davidson; the returned vectors are
   exactly zero on the masked pairs and no root sits at a masked pair's ``dE``.
3. Refusals: a partner without weights (``GATE nontda_occupation_weights``), the Lanczos solvers
   (``GATE bse_occupation_weights``), the matrix-free Lanczos.

``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest
tests/test_bse_occupations_cpu.py -s``.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from bse import bse_lanczos, bse_nontda  # noqa: E402
from bse.bse_preconditioner import MASKED_PAIR_RY, occupation_weight_cv_k  # noqa: E402
from bse.bse_ring_comm import make_bse_shardings  # noqa: E402
from bse.bse_window import PAD_EPS_GUARD_RY  # noqa: E402
from tests import bse_full_oracle as orc  # noqa: E402
from tests import test_bse_full_finite_q_cpu as fq  # noqa: E402

N, NK, NC, NV = fq.N, fq.NK, fq.NC, fq.NV
RY_EV = 13.6056980659


def _occupations(seed=17):
    """``(f_c_plus, f_v, f_c, f_v_plus)`` (nk, n): occupations of c at k+Q, v at k, c at k, v at k+Q."""
    rng = np.random.default_rng(seed)

    def cond():
        f = np.zeros((NK, NC))
        f[:, :2] = 0.1 + 0.6 * rng.random((NK, 2))            # two partly filled bands
        return f

    def val():
        f = np.ones((NK, NV))
        f[:, 2:] = 0.3 + 0.6 * rng.random((NK, 2))            # two partly emptied bands
        return f
    return cond(), val(), cond(), val()


def _weights(q0=False):
    """Host ``(w, wb)`` (n_c, n_v, nk): resonant and antiresonant pairs."""
    fcp, fv, fc, fvp = _occupations()
    w = np.asarray(occupation_weight_cv_k(fcp, fv))
    wb = w if q0 else np.asarray(occupation_weight_cv_k(fc, fvp))
    assert 0 < np.sum(w == 0) < 0.3 * w.size and 0 < np.sum(wb == 0) < 0.3 * wb.size   # some pairs masked
    return w, wb


def _reference(ops, w, wb, *, head_tensor=False):
    """``K`` with occupation weights from the unweighted definition: the diagonal energies stay,
    the kernel part is scaled on both sides, masked pairs get ``MASKED_PAIR_RY``."""
    K0 = fq.reference_K(ops, head_tensor=head_tensor)
    d_res = ((ops["ec_plus"] - fq.HEAD_RY)[:, :, None] - ops["ev"][:, None, :]).transpose(1, 2, 0).reshape(-1)
    d_anti = ((ops["ec"] - fq.HEAD_RY)[:, :, None] - ops["ev_plus"][:, None, :]).transpose(1, 2, 0).reshape(-1)
    d = np.concatenate([d_res, d_anti])
    W = np.concatenate([w.reshape(-1), wb.reshape(-1)])
    return np.diag(np.where(W > 0, d, MASKED_PAIR_RY)) + W[:, None] * (K0 - np.diag(d)) * W[None, :], W


def _payload(mesh, ops, w, wb, *, partner=True, **kw):
    from lxkit import device_put_process_local
    wsh = NamedSharding(mesh, P("x", "y", None))
    data = fq._payload(mesh, ops, partner=partner, **kw)
    data["occupation_weights"] = device_put_process_local(w, wsh)
    if partner:
        data["partner"]["occupation_weights"] = device_put_process_local(wb, wsh)
    return data


def _blocks(mesh, data):
    sh = make_bse_shardings(mesh)
    pair, up, low = bse_nontda._operator(data, mesh, sh, include_W=True)
    if low is None:
        return bse_nontda._materialize_A_B(pair, up, sh, NC, NV, NK)
    A, B, B_low, A_low = bse_nontda._materialize_K(pair, up, low, sh, NC, NV, NK)
    return np.block([[A, B], [B_low, A_low]])


def test_the_masked_diagonal_is_the_pad_guard():
    assert MASKED_PAIR_RY == PAD_EPS_GUARD_RY


# ---------------------------------------------------------------------------
# 1. the operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", [1, 2])
@pytest.mark.parametrize("head_tensor", [False, True])
def test_weighted_blocks_at_finite_q(side, head_tensor):
    ops = fq._operands()
    w, wb = _weights()
    ref, _ = _reference(ops, w, wb, head_tensor=head_tensor)
    assert fq._rel(ref, ref.conj().T) <= 1e-13
    got = _blocks(fq._mesh(side), _payload(fq._mesh(side), ops, w, wb, head_tensor=head_tensor))
    # relative to the largest unmasked entry (the masked diagonal, 1e3 Ry, would hide an error)
    err = float(np.max(np.abs(got - ref)) / np.max(np.abs(np.where(np.abs(ref) < 1e2, ref, 0.0))))
    assert err <= 1e-12, err
    # red twins: weights of one are the unweighted operator; one side of the weight alone is not the reference
    ones = np.ones_like(w)
    plain = _blocks(fq._mesh(side), _payload(fq._mesh(side), ops, ones, ones, head_tensor=head_tensor))
    assert fq._rel(plain, fq.reference_K(ops, head_tensor=head_tensor)) <= 1e-12
    assert fq._rel(plain, ref) > 1e-3
    print(f"\nMEASURED weighted finite-Q K, {side}x{side}, head tensor {head_tensor}: {err:.2e} from the "
          f"definition; {int(np.sum(w == 0))} + {int(np.sum(wb == 0))} masked pairs of {2 * N}")


def test_weighted_blocks_at_q0_and_tamm_dancoff():
    from lxkit import device_put_process_local
    from bse.bse_stack_matvec import build_bse_stack_matvec
    ops = fq._operands(q0=True)
    w, _ = _weights(q0=True)
    ref, _ = _reference(ops, w, w)
    mesh = fq._mesh(2)
    A, B = _blocks(mesh, _payload(mesh, ops, w, w, partner=False))
    assert fq._rel(A, ref[:N, :N]) <= 1e-12
    assert np.max(np.abs(B - ref[:N, N:])) <= 1e-12 * np.max(np.abs(ref))
    orc.assert_bse_structure(A, B, tol=1e-12)
    # the Tamm-Dancoff matvec on identity columns is the same A
    data = _payload(mesh, ops, w, w, partner=False)
    sh = make_bse_shardings(mesh)
    W_R = bse_nontda.payload_w_r(data, mesh, sh, include_W=True)
    mv = build_bse_stack_matvec(mesh, *fq.KGRID, kernel="bse", occupations=True)
    from bse.bse_head import head_shifted_eps_c
    eye = device_put_process_local(np.eye(N, dtype=complex).reshape(N, NC, NV, NK), sh.X)
    out = mv(eye, data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
             head_shifted_eps_c(data["eps_c"], data), data["eps_v"], W_R, data["V_q0"], data["M"],
             data["occupation_weights"])
    A_tda = np.asarray(out).reshape(N, N).T
    assert fq._rel(A_tda, ref[:N, :N]) <= 1e-12


# ---------------------------------------------------------------------------
# 2. the solves
# ---------------------------------------------------------------------------

def _host(evecs, n):
    Z = np.asarray(evecs).reshape(n, -1, N)
    return [Z[:, i].T for i in range(Z.shape[1])]


@pytest.mark.parametrize("side", [1, 2])
def test_weighted_full_solves_match_eig(side):
    mesh = fq._mesh(side)
    n_eig = 6
    for q0 in (False, True):
        ops = fq._operands(q0=q0)
        w, wb = _weights(q0=q0)
        K, W = _reference(ops, w, wb)
        ref = orc.solve_general_pencil(K)
        assert ref["definite"] and ref["omega"][n_eig] < 0.5 * MASKED_PAIR_RY
        masked_x, masked_y = (w.reshape(-1) == 0), (wb.reshape(-1) == 0)
        for solver, tol in (("dense", 1e-10), ("davidson", 1e-9)):
            om, ev, _ = bse_nontda.solve_bse_nontda_sharded(
                _payload(mesh, ops, w, wb, partner=not q0), mesh, n_eig=n_eig, solver=solver)
            err = float(np.max(np.abs(np.asarray(om) - ref["omega"][:n_eig])))
            assert err <= tol, (q0, solver, err)
            X, Y = _host(ev, n_eig)
            if solver == "davidson":
                assert bse_nontda.LAST_DAVIDSON["info"]["paired"] is q0
                assert np.max(np.abs(X[masked_x])) == 0.0 and np.max(np.abs(Y[masked_y])) == 0.0
            else:
                assert np.max(np.abs(X[masked_x])) <= 1e-10 and np.max(np.abs(Y[masked_y])) <= 1e-10
            assert np.max(np.abs(orc.sigma_gram(X, Y) - np.eye(n_eig))) <= 1e-8
            print(f"\nMEASURED weighted {'Q = 0' if q0 else 'finite Q'} {solver} {side}x{side}: "
                  f"{err * RY_EV:.1e} eV from eig"
                  + (f", {bse_nontda.LAST_DAVIDSON['info']['pair_calls']} pair calls"
                     if solver == "davidson" else ""))


def test_weighted_tamm_dancoff_davidson_matches_eigvalsh():
    mesh = fq._mesh(1)                    # the planned Davidson's CPU thread gate allows one device here
    ops = fq._operands(q0=True)
    w, _ = _weights(q0=True)
    K, _ = _reference(ops, w, w)
    exact = np.linalg.eigvalsh(K[:N, :N])
    n_eig = 6
    om, ev, _ = bse_lanczos.solve_bse_sharded(_payload(mesh, ops, w, w, partner=False), mesh, n_eig=n_eig)
    err = float(np.max(np.abs(np.asarray(om) - exact[:n_eig])))
    assert err <= 1e-9, err
    X = np.asarray(ev).reshape(n_eig, N).T
    assert np.max(np.abs(X[w.reshape(-1) == 0])) == 0.0
    # the unweighted operator's lowest roots are different numbers (the weights do something)
    plain = np.linalg.eigvalsh(fq.reference_K(ops)[:N, :N])
    assert np.max(np.abs(plain[:n_eig] - exact[:n_eig])) > 1e-5
    print(f"\nMEASURED weighted Tamm-Dancoff Davidson: {err * RY_EV:.1e} eV from eigvalsh")


# ---------------------------------------------------------------------------
# 3. refusals
# ---------------------------------------------------------------------------

def test_occupation_refusals():
    mesh = fq._mesh(1)
    ops = fq._operands()
    w, wb = _weights()
    data = _payload(mesh, ops, w, wb)
    del data["partner"]["occupation_weights"]
    with pytest.raises(ValueError, match="GATE nontda_occupation_weights"):
        bse_nontda.solve_bse_nontda_sharded(data, mesh, n_eig=4)
    q0 = fq._operands(q0=True)
    with pytest.raises(ValueError, match="GATE bse_occupation_weights"):
        bse_lanczos.solve_bse_sharded(_payload(mesh, q0, w, w, partner=False), mesh, n_eig=4,
                                      solver_kind="lanczos")
    with pytest.raises(ValueError, match="GATE nontda_route_option"):
        bse_nontda.solve_bse_nontda_sharded(_payload(mesh, q0, w, w, partner=False), mesh, n_eig=4,
                                            solver="matrixfree")
