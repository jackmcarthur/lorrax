"""Dense oracle for the LADDER screening operator (``screening_diagrams = w_bse``).

An INDEPENDENT dense build of the symplectic ladder Hamiltonian — explicit
numpy einsums from psi, eps, V_q and W_q, no call into any production kernel —
dense-solved and compared against the sharded resolvent
(``bse.w_ladder.build_ladder_resolvent`` +
``bse_w_exact.apply_screening_resolvent_block``) at omega = 0 and one imaginary
omega, at q = 0 AND one finite symmetry-reduced q.

The operator under test (derivation: ``bse/w_ladder.py`` module docstring):

    A = diag(D) + Kx - Kd,   B = Kx - Kd_B
    H = [[   A                 ,  B              ],
         [ -(Kx - conj(Kd_B))  , -(diag(D) + Kx - conj(Kd)) ]]

with ``Kx`` the ring/Hartree kernel ``M v M^dag`` (SAME in all four blocks and
never conjugated — the Hartree rung is the outer product of the probe injection
and the density readout, so it is branch-blind), ``Kd`` the optical direct term
and ``Kd_B`` its c'<->v'-swapped partner.  The anti-resonant row is a HYBRID:
the RING part un-conjugated (that is what keeps ``W - v = v chi v`` exact) and
the DIRECT parts conjugated (they are an ordinary two-particle kernel, so
``K^AA = conj(K^RR)``, ``K^AR = conj(K^RA)``).

RED TWINS (QUALITY_PATTERNS 1 — a check that passes under the bug is worse than
no check): flipping the B-block W sign, and using the naive un-conjugated
``[-B, -A]`` row, each make the SAME comparison fail.  The second twin is not
hypothetical: it is the row this feature was first written with, and it left
``W_ladder(q=0, 0)`` non-Hermitian at 2.1e-05 (probe leg, JID 57052808).

Fixture: the session-scoped ``gnppm_session`` GW run (MoS2 3x3x1, nk=9), loaded
head-LESS through the production sharded loader at a 2v2c window with
``load_v_full=True`` — the payload shape the ladder facade itself consumes.
N = nc*nv*nk is a few tens, so the dense build and the 2N x 2N solve are trivial.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                          # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P                # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import bse_io                                           # noqa: E402
from bse.bse_w_exact import (                                    # noqa: E402
    apply_screening_resolvent_block, build_finite_q_data,
    _symmetry_reduced_q_list,
)
from bse.w_ladder import build_ladder_resolvent                  # noqa: E402


# ---------------------------------------------------------------------------
# Payload (module-scoped: one restart read serves every cell below).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ladder_payload(gnppm_session):
    """Head-less 2v2c BSE payload + 1x1 mesh + input path, ``load_v_full=True``.

    Head-LESS (``inject_head=False``) because the resolvent carries no q=0 head
    by construction; ``load_v_full`` because the finite-q cell needs the
    ``V_qmunu[q]`` tiles ``build_finite_q_data`` swaps in.
    """
    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=2, n_cond=2, mesh_xy=mesh, input_file=input_path,
        inject_head=False, load_v_full=True)
    return data, mesh, input_path


def _host(x):
    return np.asarray(jax.device_get(x))


# ---------------------------------------------------------------------------
# The independent dense build.  Explicit einsums only.
# ---------------------------------------------------------------------------
def _q_flat(k, kp, grid):
    """Flat index of q = (k - k') mod grid, C-order over (kx, ky, kz)."""
    ck = np.array(np.unravel_index(k, grid))
    ckp = np.array(np.unravel_index(kp, grid))
    return int(np.ravel_multi_index(tuple((ck - ckp) % np.array(grid)), grid))


def _dense_ladder_blocks(payload, *, with_row=False):
    """(A, B, Mmat, N) for the ladder operator, built from psi/eps/V_q/W_q only.

    ``with_row=True`` also returns the derived anti-resonant row
    ``(-(Kx - conj(Kd_B)), -(diag(D) + Kx - conj(Kd)))``.

    Flat pair index ``I = (c, v, k)`` in C order — the layout the sharded
    ``(b, c, v, k)`` trial vector reshapes to.
    """
    psi_c = _host(payload["psi_c_X"])            # (k, c, s, mu)
    psi_v = _host(payload["psi_v_X"])            # (k, v, s, mu)
    eps_c = _host(payload["eps_c"])              # (k, c)
    eps_v = _host(payload["eps_v"])              # (k, v)
    V_q0 = _host(payload["V_q0"])                # (mu, nu)
    W_q = _host(payload["W_q"])                  # (mu, nu, nkx, nky, nkz)
    grid = (int(payload["nkx"]), int(payload["nky"]), int(payload["nkz"]))
    nk = grid[0] * grid[1] * grid[2]
    nc, nv, nmu = psi_c.shape[1], psi_v.shape[1], psi_c.shape[3]
    N = nc * nv * nk

    # Pair amplitude M[k, c, v, mu] = sum_s conj(psi_c) psi_v.
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)

    # D — transition energies, layout (c, v, k).
    D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))

    # Ring / Hartree kernel Kx = M V M^dag / Nk, DENSE in (k, k').
    lhs = np.einsum("kcvM,MN->kcvN", M, V_q0)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)).reshape(N, N) / nk

    # Direct terms.  Kd keeps the (c, v) role assignment on both sides; Kd_B
    # swaps the KET's electron/hole roles (the de-excitation branch).
    Wflat = W_q.reshape(nmu, nmu, nk)
    Kd = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
    Kd_B = np.zeros_like(Kd)
    for k in range(nk):
        for kp in range(nk):
            Wq = Wflat[:, :, _q_flat(k, kp, grid)]
            Pc = np.einsum("ctm,Ctm->cCm", np.conj(psi_c[k]), psi_c[kp])
            Pv = np.einsum("vsn,Vsn->vVn", psi_v[k], np.conj(psi_v[kp]))
            Kd[:, :, k, :, :, kp] = np.einsum("cCm,mn,vVn->cvCV", Pc, Wq, Pv) / nk
            PcB = np.einsum("ctm,Vtm->cVm", np.conj(psi_c[k]), psi_v[kp])
            PvB = np.einsum("vsn,Csn->vCn", psi_v[k], np.conj(psi_c[kp]))
            Kd_B[:, :, k, :, :, kp] = np.einsum(
                "cVm,mn,vCn->cvCV", PcB, Wq, PvB) / nk

    Kd2, KdB2 = Kd.reshape(N, N), Kd_B.reshape(N, N)
    Dm = np.diag(D.reshape(-1).astype(np.complex128))
    A = Dm + Kx - Kd2
    B = Kx - KdB2
    Mmat = np.transpose(M, (1, 2, 0, 3)).reshape(N, nmu)
    if with_row:
        # Ring un-conjugated, direct conjugated (derivation step 4).
        row = (-(Kx - np.conj(KdB2)), -(Dm + Kx - np.conj(Kd2)))
        return A, B, Mmat, N, row
    return A, B, Mmat, N


def _dense_wc_columns(payload, A, B, Mmat, N, cols, z, *, bottom):
    """``W(z) - v`` columns from the dense 2N solve.

    Seed ``[f; -f]`` with ``f = M (V g) / sqrt(Nk)``; readout
    ``w = V M^dag (X + Y) / sqrt(Nk)`` — the generator/snapshot pair written out.
    ``bottom`` is the anti-resonant block row and is REQUIRED: it is the half of
    the operator this feature had to derive, so a default would be exactly the
    assumption the gate exists to test.
    """
    V_q0 = _host(payload["V_q0"])
    nk = int(payload["nkx"]) * int(payload["nky"]) * int(payload["nkz"])
    snk = np.sqrt(float(nk))
    bot_left, bot_right = bottom
    H = np.block([[A, B], [bot_left, bot_right]])
    out = np.zeros((V_q0.shape[0], len(cols)), dtype=np.complex128)
    lhs = z * np.eye(2 * N, dtype=np.complex128) - H
    for i, nu0 in enumerate(cols):
        g = np.zeros(V_q0.shape[0], dtype=np.float64)
        g[int(nu0)] = 1.0
        f = (Mmat @ (V_q0 @ g)) / snk
        x = np.linalg.solve(lhs, np.concatenate([f, -f]))
        s = x[:N] + x[N:]
        out[:, i] = (V_q0 @ (np.conj(Mmat).T @ s)) / snk
    return out


def _sharded_wc_columns(payload, mesh, cols, z, *, include_w=True):
    """The production route: ladder resolvent + the operator-agnostic engine.

    Payload convention is carried by the payload itself: a
    ``build_finite_q_data`` product supplies the rung's physical operand
    slots; a raw payload falls back to its density arrays (which are
    physical there).  See ``ladder_matvec_operands``."""
    from bse.bse_feast import ladder_matvec_operands, matvec_operands
    matvec, diag_h, gen, snapshot, sh = build_ladder_resolvent(
        mesh, payload, include_w=include_w)
    n_rmu = int(payload["V_q0"].shape[0])
    py = mesh.devices.shape[1]
    n_pad = int(np.ceil(len(cols) / py) * py)
    G = np.zeros((n_pad, n_rmu), dtype=np.float64)
    for i, nu0 in enumerate(cols):
        G[i, int(nu0)] = 1.0
    W_tile, resids = apply_screening_resolvent_block(
        G, z, payload, matvec, diag_h, gen, snapshot, sh,
        max_iter=400, tol=1e-11,
        operands_fn=(ladder_matvec_operands if include_w else matvec_operands))
    return _host(W_tile)[:, :len(cols)], _host(resids)[:len(cols)], W_tile


def _relerr(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / max(np.linalg.norm(np.asarray(b)), 1e-300))


def _probe_cols(payload, n=3):
    """Probe the largest-norm columns of the q=0 (W0 - V) body, plus one more."""
    nlog = int(payload["n_rmu"])
    T = _host(payload["W_q"][:, :, 0, 0, 0]) - _host(payload["V_q0"])
    order = np.argsort(-np.linalg.norm(T[:nlog, :nlog], axis=0))
    return np.concatenate([order[:n - 1], [nlog // 2]]).astype(int)


# ---------------------------------------------------------------------------
# Gates.
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("omega", [0.0, 0.25j])
def test_w_ladder_matches_dense_oracle_q0(ladder_payload, omega):
    """Sharded ladder resolvent == dense (z - H)^-1 at q = 0, static and on iR."""
    harness.skip_unless_gpu(pytest)
    data, mesh, _ = ladder_payload
    cols = _probe_cols(data)
    A, B, Mmat, N, row = _dense_ladder_blocks(data, with_row=True)
    ref = _dense_wc_columns(data, A, B, Mmat, N, cols, complex(omega), bottom=row)
    got, resids, W_tile = _sharded_wc_columns(data, mesh, cols, complex(omega))
    assert W_tile.sharding.spec == P("x", "y"), (
        f"ladder tile carries {W_tile.sharding.spec}, expected P('x','y')")
    assert resids.max() < 1e-8, (
        f"GMRES not converged on the ladder operator: max resid "
        f"{resids.max():.2e} (the operator is stiffer than the RPA one; a "
        f"truncated solve would make the comparison below meaningless)")
    nlog = int(data["n_rmu"])
    rel = _relerr(got[:nlog], ref[:nlog])
    assert rel < 1e-8, f"omega={omega}: ladder vs dense oracle rel_err {rel:.3e}"


def _synthetic_payload(mesh, *, nkx=2, nky=2, nkz=1, nc=2, nv=2, nmu=8, seed=7):
    """A restart-FREE BSE payload in the production key contract.

    Why this exists rather than a second on-disk fixture: the four-GPU rule
    wants this operator checked on a 2x2 mesh, and the ``gnppm_session`` route
    cannot supply one IN-PROCESS — the GW driver's SlabIO FFI refuses
    ``mesh 2x2=4 != jax.process_count()=1`` (measured 2026-08-15), so a cell
    that needs both a restart and four in-process devices is a MULTI-PROCESS
    gate and belongs in ``tests/multi_device/``.  Random psi/eps/V/W in the same
    dict contract exercise the identical kernels with no I/O at all.

    The tensors are physical where it matters: ``V_q0`` real symmetric (a q=0
    Coulomb tile), ``W_q`` Hermitian in (mu, nu) per q AND obeying
    ``W(-q) = conj(W(q))`` — the identity that makes the direct rung Hermitian
    and its swapped partner complex-symmetric.  Energies are well separated so
    the shifted solve at z = 0 is well conditioned.
    """
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude

    nk = nkx * nky * nkz
    rng = np.random.default_rng(seed)
    sh = make_bse_shardings(mesh)

    def _c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / 8.0

    psi_c = _c(nk, nc, 1, nmu)
    psi_v = _c(nk, nv, 1, nmu)
    eps_c = 1.0 + 0.3 * rng.standard_normal((nk, nc))
    eps_v = -1.0 + 0.3 * rng.standard_normal((nk, nv))
    Vq0 = rng.standard_normal((nmu, nmu)) * 0.1
    Vq0 = Vq0 + Vq0.T                                   # real symmetric
    Wq = _c(nmu, nmu, nk) * 0.1
    Wq = 0.5 * (Wq + np.conj(np.transpose(Wq, (1, 0, 2))))     # Hermitian per q
    idx = np.ravel_multi_index(
        tuple((-np.array(np.unravel_index(np.arange(nk), (nkx, nky, nkz))))
              % np.array([[nkx], [nky], [nkz]])), (nkx, nky, nkz))
    Wq = 0.5 * (Wq + np.conj(Wq[:, :, idx]))            # W(-q) = conj(W(q))
    Wq = Wq.reshape(nmu, nmu, nkx, nky, nkz)

    with mesh:
        d = {
            "psi_c_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_x),
            "psi_c_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_y),
            "psi_v_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_x),
            "psi_v_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_y),
            "eps_c": jnp.asarray(eps_c), "eps_v": jnp.asarray(eps_v),
            "W_q": jax.lax.with_sharding_constraint(jnp.asarray(Wq), sh.W),
            "V_q0": jax.lax.with_sharding_constraint(jnp.asarray(Vq0), sh.V),
            "nkx": nkx, "nky": nky, "nkz": nkz,
            "n_cond_pad": nc, "n_val_pad": nv, "n_rmu": nmu,
        }
        d["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_X"], d["psi_v_X"]), sh.psi_x)
        d["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_Y"], d["psi_v_Y"]), sh.psi_y)
    return d


def _trs_synthetic_payload(mesh, *, nkx=3, nky=3, nkz=1, nc=2, nv=2, nmu=8,
                           seed=13, ns=1):
    """`_synthetic_payload` variant in the exact TRS pair gauge, with V_q_full.

    ``psi(-k) = Theta psi(k)`` by construction (``Theta = K`` scalar,
    ``i sigma_y K`` spinor; real psi / Kramers pairs at TRIM points), eps
    symmetric under k -> -k, V_q_full Hermitian per q with V(-q) = conj(V(q)).
    This is the payload on which the ladder's conj-pattern anti-resonant
    channel is EXACT, so production reciprocity must hold to solver tolerance
    — and scrambling the gauge (a legal per-band phase at the -k slots) must
    break it.  Grid default 3x3x1: only Gamma is TRIM, so the k-pair overwrite
    path is actually exercised (a 2x2 grid is ALL TRIM points)."""
    from bse.bse_serial import compute_pair_amplitude
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_w_exact import _spin_rotation, _theta
    nk = nkx * nky * nkz
    grid = (nkx, nky, nkz)
    rng = np.random.default_rng(seed)
    coords = np.stack(np.unravel_index(np.arange(nk), grid), axis=1)
    neg = np.ravel_multi_index(tuple(((-coords) % np.array(grid)).T), grid)
    R = _spin_rotation(ns)

    def _c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / 8.0

    def _trs_bands(nb):
        assert ns == 1 or nb % 2 == 0, "spinor windows need Kramers-even nb"
        psi = _c(nk, nb, ns, nmu)
        eps = np.sort(rng.standard_normal((nk, nb)), axis=1)
        for k in range(nk):
            kn = int(neg[k])
            if k < kn:
                psi[kn] = _theta(psi[k], R)
                eps[kn] = eps[k]
            elif k == kn:
                if ns == 1:
                    psi[k] = psi[k].real
                else:
                    # Kramers pairs at TRIM: phi_{2j+1} = Theta phi_{2j},
                    # pairwise-degenerate energies.
                    for j in range(0, nb, 2):
                        psi[k, j + 1] = _theta(psi[k, j:j + 1], R)[0]
                        eps[k, j + 1] = eps[k, j]
        return psi, eps

    psi_c, eps_c = _trs_bands(nc)
    psi_v, eps_v = _trs_bands(nv)
    eps_c += 1.0
    eps_v -= 1.0

    def _recip_tiles(scale):
        T = _c(nmu, nmu, nk) * scale
        T = 0.5 * (T + np.conj(np.transpose(T, (1, 0, 2))))    # Hermitian per q
        T = 0.5 * (T + np.conj(T[:, :, neg]))                  # T(-q) = conj(T(q))
        return T

    Wq = _recip_tiles(0.1)
    Vq = _recip_tiles(0.1)
    Vq[:, :, 0] = Vq[:, :, 0].real                             # q=0 real symmetric
    sh = make_bse_shardings(mesh)
    with mesh:
        d = {
            "psi_c_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_x),
            "psi_c_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_y),
            "psi_v_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_x),
            "psi_v_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_y),
            "eps_c": jnp.asarray(eps_c), "eps_v": jnp.asarray(eps_v),
            "W_q": jax.lax.with_sharding_constraint(
                jnp.asarray(Wq.reshape(nmu, nmu, nkx, nky, nkz)), sh.W),
            "V_q0": jax.lax.with_sharding_constraint(
                jnp.asarray(Vq[:, :, 0]), sh.V),
            "V_q_full": jnp.asarray(Vq.reshape(nmu, nmu, nkx, nky, nkz)),
            "nkx": nkx, "nky": nky, "nkz": nkz,
            "n_cond_pad": nc, "n_val_pad": nv, "n_rmu": nmu,
        }
        d["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_X"], d["psi_v_X"]), sh.psi_x)
        d["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_Y"], d["psi_v_Y"]), sh.psi_y)
    return d, neg


def _reciprocity_of_production(data, mesh, q, cols):
    """max|W(-q) - conj(W(q))|/|W(q)| through the production resolvent."""
    grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    tiles = {}
    for qq in (q, _neg_q(q, grid)):
        dq = build_finite_q_data(data, qq, mesh)
        got, resids, _ = _sharded_wc_columns(dq, mesh, cols, 0.0 + 0.0j)
        assert resids.max() < 1e-8, f"q={qq}: GMRES resid {resids.max():.2e}"
        tiles[qq] = got
    a, b = tiles[q], tiles[_neg_q(q, grid)]
    return float(np.abs(b - np.conj(a)).max() / max(np.abs(a).max(), 1e-300))


@pytest.mark.gpu
@pytest.mark.parametrize("ns", [1, 2])
def test_w_ladder_trs_gauge_mechanism_on_a_synthetic_payload(ns):
    """Reciprocity holds in the TRS/Kramers gauge, BREAKS under a legal gauge
    change, and `enforce_trs_pair_gauge` cures it — the whole mechanism,
    restart-free, scalar AND spinor.

    This is the coverage for the finite-q anti-resonant channel (w_ladder
    derivation step 8).  The ns=2 arm is the spinor acceptance: the dense
    FIRST-PRINCIPLES operator obeys W(-q) = conj(W(q)) on a Kramers-gauge
    spinor payload, production matches it at both ±q, and the cure works
    through Theta = i*sigma_y*K.  NOTE the deliberate absence of a
    "drop the sigma_y" red twin: every contraction this operator performs is
    a spin-singlet s-summed bilinear in which the real-orthogonal sigma_y
    cancels IDENTICALLY, so that twin is a no-op by the same theorem that
    makes the spinor extension valid (enforce_trs_pair_gauge docstring); the
    discriminating twin is the gauge scramble below.
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_ring_comm import create_mesh_xy
    from bse.bse_serial import compute_pair_amplitude
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_w_exact import enforce_trs_pair_gauge
    mesh = create_mesh_xy(1, 1)
    data, neg = _trs_synthetic_payload(mesh, ns=ns)
    q = (1, 0, 0)
    grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))
    cols = np.array([0, 3, 5], dtype=int)

    # Dense first-principles arm: the derived operator itself is reciprocal
    # in this gauge, and production realizes it at both ±q.
    refs = {}
    for qq in (q, _neg_q(q, grid)):
        fake, A, B, Mmat, N, row = _dense_ladder_blocks_finite_q(data, qq)
        refs[qq] = _dense_wc_columns(fake, A, B, Mmat, N, cols, 0.0 + 0.0j,
                                     bottom=row)
    rel_dense = _relerr(refs[_neg_q(q, grid)], np.conj(refs[q]))
    print(f"[trs-mechanism ns={ns}] dense reciprocity {rel_dense:.3e}")
    assert rel_dense < 1e-10, (
        f"ns={ns}: the dense first-principles ladder operator violates "
        f"W(-q)=conj(W(q)) at {rel_dense:.3e} IN the TRS/Kramers gauge")

    rel_good = _reciprocity_of_production(data, mesh, q, cols)
    assert rel_good < 1e-8, (
        f"ns={ns}: reciprocity fails at {rel_good:.3e} ON the TRS-gauge "
        f"payload — the conj-pattern channel (or the rung operand slots) is "
        f"wrong even in its own gauge")
    for qq in (q, _neg_q(q, grid)):
        dq = build_finite_q_data(data, qq, mesh)
        got, resids, _ = _sharded_wc_columns(dq, mesh, cols, 0.0 + 0.0j)
        assert resids.max() < 1e-8
        rel = _relerr(got, refs[qq])
        print(f"[trs-mechanism ns={ns}] q={qq}: production vs dense {rel:.3e}")
        assert rel < 1e-8, (
            f"ns={ns}, q={qq}: production vs first-principles dense "
            f"rel_err {rel:.3e}")

    # A LEGAL gauge change: random per-band phases on every non-TRIM -k slot.
    rng = np.random.default_rng(3)
    sh = make_bse_shardings(mesh)
    scr = dict(data)
    for name in ("psi_c", "psi_v"):
        psi = _host(scr[f"{name}_X"]).copy()
        nb = psi.shape[1]
        for k in range(psi.shape[0]):
            if int(neg[k]) < k:          # the -k member of each proper pair
                psi[k] *= np.exp(2j * np.pi * rng.random(nb))[:, None, None]
        scr[f"{name}_X"] = jax.device_put(psi, sh.psi_x)
        scr[f"{name}_Y"] = jax.device_put(psi, sh.psi_y)
    with mesh:
        scr["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(scr["psi_c_X"], scr["psi_v_X"]), sh.psi_x)
        scr["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(scr["psi_c_Y"], scr["psi_v_Y"]), sh.psi_y)

    rel_bad = _reciprocity_of_production(scr, mesh, q, cols)
    print(f"[trs-mechanism] TRS gauge {rel_good:.3e}   scrambled {rel_bad:.3e}")
    assert rel_bad > 1e-5, (
        f"a random per-band gauge at the -k slots left reciprocity at "
        f"{rel_bad:.3e} — then the operator is gauge-INsensitive and the "
        f"whole step-8 mechanism (and this cell) is misconceived")

    cured = enforce_trs_pair_gauge(scr, mesh)
    rel_cured = _reciprocity_of_production(cured, mesh, q, cols)
    print(f"[trs-mechanism] cured {rel_cured:.3e}")
    assert rel_cured < 1e-8, (
        f"enforce_trs_pair_gauge left reciprocity at {rel_cured:.3e} on the "
        f"scrambled payload it exists to repair")


@pytest.mark.gpu
@pytest.mark.parametrize("ns", [1, 2])
def test_trs_gauge_is_deterministic_and_jitter_stable(ns):
    """Acceptance for the registered gauge-nondeterminism defect.

    (a) identical input -> BIT-identical canonical gauge (two consecutive
    calls); (b) a re-mixed realization — random unitaries within degenerate
    blocks and random per-band phases on every slot independently, the class
    of variation upstream GPU nondeterminism produces run-to-run — maps to
    the SAME canonical gauge to numerical tolerance.  The canonicalization is
    span-anchored (fixed-order probes + largest-element phases, no eigh/SVD),
    so the output depends on the WFN's subspaces, not on how a given run
    happened to realize their bases.  Kramers TRIM blocks (ns=2) ride the
    same discipline.
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_ring_comm import create_mesh_xy
    from bse.bse_w_exact import enforce_trs_pair_gauge
    mesh = create_mesh_xy(1, 1)
    data, neg = _trs_synthetic_payload(mesh, ns=ns)
    # Force a degenerate pair at a proper +-k SOURCE slot so the
    # subspace canonicalizer is actually exercised (the payload's random
    # energies are otherwise non-degenerate away from Kramers TRIM pairs).
    k_src = next(k for k in range(int(neg.size)) if k < int(neg[k]))
    for nm in ("eps_c", "eps_v"):
        e = _host(data[nm]).copy()
        e[k_src, 1] = e[k_src, 0]
        e[int(neg[k_src])] = e[k_src]
        data[nm] = jnp.asarray(e)

    g1 = enforce_trs_pair_gauge(data, mesh)
    g2 = enforce_trs_pair_gauge(data, mesh)
    for nm in ("psi_c_X", "psi_v_X", "eps_c", "eps_v"):
        assert np.array_equal(_host(g1[nm]), _host(g2[nm])), (
            f"{nm}: two consecutive enforce_trs_pair_gauge calls on identical "
            f"input are not bit-identical — the canonicalization is "
            f"nondeterministic and the registered defect is not fixed")

    # The jitter model: every slot independently re-phased and every
    # degenerate block independently re-mixed (TRS pairing broken, as a
    # fresh upstream run breaks it).
    rng = np.random.default_rng(29)
    jit_payload = dict(data)
    for nm in ("psi_c", "psi_v"):
        psi = _host(data[f"{nm}_X"]).copy()
        e = _host(data["eps_c" if nm == "psi_c" else "eps_v"])
        for k in range(psi.shape[0]):
            b0 = 0
            while b0 < e.shape[1]:
                b1 = b0 + 1
                while b1 < e.shape[1] and e[k, b1] - e[k, b1 - 1] < 1e-8:
                    b1 += 1
                m = b1 - b0
                Q = np.linalg.qr(rng.standard_normal((m, m))
                                 + 1j * rng.standard_normal((m, m)))[0]
                blk = psi[k, b0:b1].reshape(m, -1)
                psi[k, b0:b1] = (Q @ blk).reshape(psi[k, b0:b1].shape)
                b0 = b1
        from bse.bse_ring_comm import make_bse_shardings
        sh = make_bse_shardings(mesh)
        jit_payload[f"{nm}_X"] = jax.device_put(psi, sh.psi_x)
        jit_payload[f"{nm}_Y"] = jax.device_put(psi, sh.psi_y)
    g3 = enforce_trs_pair_gauge(jit_payload, mesh)
    for nm in ("psi_c_X", "psi_v_X"):
        d = float(np.abs(_host(g1[nm]) - _host(g3[nm])).max())
        assert d < 1e-10, (
            f"{nm}: a re-mixed degenerate realization canonicalized to a "
            f"DIFFERENT gauge (max|diff| = {d:.3e}) — the canonical basis is "
            f"anchored to the run-varying input basis, not to the subspace")


@pytest.mark.gpu
def test_w_ladder_gauge_sensitivity_is_real_bounded_and_reciprocity_blind():
    """TWO different VALID TRS gauges: reciprocity holds in BOTH, and the W
    tiles DIFFER at the ~1e-4 scale — the v1 ladder operator is intrinsically
    gauge-SENSITIVE, and this cell characterizes it instead of wishing it away.

    First written asserting invariance; the measurement refuted the
    assertion (rel = 1.870e-04 with arm-A reciprocity 5.1e-17, truefinal
    leg 2026-08-16) and became the finding.  Mechanism: the conj-pattern
    anti-resonant channel makes a complex in-block band rotation NOT a
    similarity transform of the 2N operator — the Y-channel kernels
    transform in the CONJUGATE representation while the shared seed
    ``[f; -f]`` and readout ``M (X + Y)`` transform un-conjugated — so W
    depends on the Bloch gauge at the rung scale even with exact pairing
    and closure.  Reciprocity is BLIND to it (any paired gauge is
    reciprocal), which is why the finite-q gates could not see it.

    Consequences this cell pins: (a) the canonicalizer
    (``enforce_trs_pair_gauge``) is LOAD-BEARING for reproducibility — same
    WFN, same gauge, same W — and its scalar-Si QP move (claim 240,
    6.53 meV) is this freedom being re-realized, not a numerics defect;
    (b) the gauge spread is the v1 systematic to quote alongside w_bse QP
    deltas; (c) the deep fix is the honestly Theta-built backward channel
    (conjugate-covariant Y seed/readout), a named deferral — if it lands,
    the sensitivity assert below goes to zero and THIS CELL must be
    flipped back to an invariance gate.

    Gauge B: a random unitary within a degenerate block and random band
    phases at the SOURCE slot, propagated Theta-consistently to the
    partner — a valid TRS pairing, deliberately not canonical.
    """
    harness.skip_unless_gpu(pytest)
    from bse.bse_ring_comm import create_mesh_xy, make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude
    from bse.bse_w_exact import _theta, _spin_rotation
    mesh = create_mesh_xy(1, 1)
    data, neg = _trs_synthetic_payload(mesh)
    # a degenerate pair at a source slot, so the gauge freedom is non-trivial
    k_src = next(k for k in range(int(neg.size)) if k < int(neg[k]))
    for nm in ("eps_c", "eps_v"):
        e = _host(data[nm]).copy()
        e[k_src, 1] = e[k_src, 0]
        e[int(neg[k_src])] = e[k_src]
        data[nm] = jnp.asarray(e)
    q = (1, 0, 0)
    cols = np.array([0, 3, 5], dtype=int)
    rel_a = _reciprocity_of_production(data, mesh, q, cols)

    rng = np.random.default_rng(41)
    R = _spin_rotation(1)
    sh = make_bse_shardings(mesh)
    gaugeB = dict(data)
    for nm in ("psi_c", "psi_v"):
        psi = _host(gaugeB[f"{nm}_X"]).copy()
        Q = np.linalg.qr(rng.standard_normal((2, 2))
                         + 1j * rng.standard_normal((2, 2)))[0]
        blk = psi[k_src, 0:2].reshape(2, -1)
        psi[k_src, 0:2] = (Q @ blk).reshape(psi[k_src, 0:2].shape)
        ph = np.exp(2j * np.pi * rng.random(psi.shape[1]))
        psi[k_src] *= ph[:, None, None]
        psi[int(neg[k_src])] = _theta(psi[k_src], R)   # Theta-consistent partner
        gaugeB[f"{nm}_X"] = jax.device_put(psi, sh.psi_x)
        gaugeB[f"{nm}_Y"] = jax.device_put(psi, sh.psi_y)
    with mesh:
        gaugeB["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(gaugeB["psi_c_X"], gaugeB["psi_v_X"]), sh.psi_x)
        gaugeB["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(gaugeB["psi_c_Y"], gaugeB["psi_v_Y"]), sh.psi_y)

    tiles = {}
    for tag, payload in (("A", data), ("B", gaugeB)):
        dq = build_finite_q_data(payload, q, mesh)
        got, resids, _ = _sharded_wc_columns(dq, mesh, cols, 0.0 + 0.0j)
        assert resids.max() < 1e-8
        tiles[tag] = got
    rel_b = _reciprocity_of_production(gaugeB, mesh, q, cols)
    rel = _relerr(tiles["B"], tiles["A"])
    print(f"[gauge-sensitivity] W(gaugeB) vs W(gaugeA) rel = {rel:.3e}   "
          f"(reciprocity: arm A {rel_a:.3e}, arm B {rel_b:.3e})")
    assert max(rel_a, rel_b) < 1e-8, (
        f"reciprocity failed on a VALID TRS gauge (A {rel_a:.3e}, "
        f"B {rel_b:.3e}) — pairing itself is broken, which is a different "
        f"defect than the gauge sensitivity this cell characterizes")
    assert rel > 1e-8, (
        f"two valid TRS gauges now agree to {rel:.3e} — the operator has "
        f"become gauge-invariant (honest backward channel landed?).  That is "
        f"GOOD news with obligations: flip this cell to an invariance gate "
        f"and retire the canonicalizer's claim-240 QP systematic note")
    assert rel < 1e-2, (
        f"gauge spread {rel:.3e} is an order beyond the characterized ~1e-4 "
        f"scale — the sensitivity has grown; find out why before quoting any "
        f"w_bse QP number")


@pytest.mark.gpu
def test_trs_gauge_refuses_a_non_trs_kgrid():
    """A deck whose eps(-k) != eps(k) has NO exact gauge for the conj-pattern
    channel — the helper must refuse by name, not symmetrize a lie."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_ring_comm import create_mesh_xy
    from bse.bse_w_exact import enforce_trs_pair_gauge
    mesh = create_mesh_xy(1, 1)
    data, _ = _trs_synthetic_payload(mesh)
    broken = dict(data)
    e = _host(broken["eps_c"]).copy()
    e[1] += 1.0e-3                       # (1,0,0): a proper +-k pair member
    broken["eps_c"] = jnp.asarray(e)
    with pytest.raises(ValueError, match="trs_gauge_energies_disagree"):
        enforce_trs_pair_gauge(broken, mesh)


@pytest.mark.gpu
@pytest.mark.parametrize("px,py", [(1, 1), (2, 2)])
def test_w_ladder_matches_dense_oracle_on_a_mesh(px, py):
    """The dense identity on a 1x1 AND a 2x2 mesh — the FOUR-GPU rule.

    A P=1-only verification is never sufficient (QUALITY_PATTERNS, the four-GPU
    banner): the ladder path adds a second ``encode_T`` ring/all-gather chain
    (the c'<->v' swapped one) and a CONJUGATED re-application of BOTH direct
    appliers in the anti-resonant row, all inside ``shard_map`` bodies whose
    ring permutations, dynamic slices and ``psum_scatter``s are no-ops at
    ``px = py = 1``.  The 2x2 leg is the one that exercises them; the 1x1 leg is
    its control, so a disagreement is attributable to the mesh and not to the
    synthetic payload.  Skips below 4 devices rather than passing vacuously.

    UNDER THE ORDINARY SUITE THE 2x2 LEG SKIPS, and that is not a silent gap:
    ``conftest.pytest_configure`` pins every pytest process (xdist worker or
    not) to ONE GPU, so ``jax.device_count() == 1``.  It needs four devices
    visible to one process:

        LORRAX_MESH_CELL=1 lx run -G 4 -n 1 -- \
            python3 -m pytest -q tests/test_bse_w_ladder_dense.py -k on_a_mesh

    (``LORRAX_MESH_CELL=1`` is the suite's own "this process IS the widened
    one" switch; it makes ``pytest_configure`` skip the pin.)  MEASURED green
    that way on 2026-08-15, JID 57052808 / nid001120.  The ``mesh(4)`` marker is
    NOT usable for the restart-based cells in this file: its child sets
    ``LORRAX_FFT_FFI=0`` and the GW driver the ``gnppm_session`` fixture runs
    refuses without the cuFFT backend, and the driver's SlabIO also refuses
    ``mesh 2x2=4 != process_count()=1``.
    """
    harness.skip_unless_gpu(pytest)
    if px * py > jax.device_count():
        pytest.skip(f"needs {px * py} devices; have {jax.device_count()}")
    from bse.bse_ring_comm import create_mesh_xy
    mesh = create_mesh_xy(px, py)
    data = _synthetic_payload(mesh)
    cols = np.array([0, 3, 5], dtype=int)
    A, B, Mmat, N, row = _dense_ladder_blocks(data, with_row=True)
    ref = _dense_wc_columns(data, A, B, Mmat, N, cols, 0.0 + 0.0j, bottom=row)
    got, resids, W_tile = _sharded_wc_columns(data, mesh, cols, 0.0 + 0.0j)
    assert W_tile.sharding.spec == P("x", "y")
    assert resids.max() < 1e-8, (
        f"{px}x{py}: GMRES not converged ({resids.max():.2e})")
    rel = _relerr(got[:data["n_rmu"]], ref[:data["n_rmu"]])
    assert rel < 1e-9, f"{px}x{py} mesh: ladder vs dense oracle rel_err {rel:.3e}"
    # The static tile is Hermitian on this payload for the same reason it is on
    # the TRS deck: K^d is a Hermitian kernel matrix under the derived row.
    sub = got[cols, :]
    asym = float(np.abs(sub - sub.conj().T).max() / max(np.abs(sub).max(), 1e-300))
    assert asym < 1e-9, f"{px}x{py}: W_ladder(0) not Hermitian ({asym:.3e})"


def _roll_host(arr, q, grid):
    """``out[k] = arr[k - q]`` on the C-order flat k-axis (axis 0) — the same
    convention as ``bse_w_exact._roll_k_index``, re-derived here so the
    reference shares no code with the machinery it checks."""
    a = arr.reshape(grid + arr.shape[1:])
    a = np.roll(a, shift=(int(q[0]), int(q[1]), int(q[2])), axis=(0, 1, 2))
    return a.reshape((-1,) + arr.shape[1:])


def _dense_ladder_blocks_finite_q(payload, q, *, with_row=True):
    """FIRST-PRINCIPLES dense ladder operator at momentum q.

    Built from the RAW payload arrays only — no ``build_finite_q_data``, so
    nothing here shares the production path's finite-q conventions; each one is
    applied explicitly and separately:

      * pair basis ``|v k, c k-q⟩``: c-legs rolled so slot k holds ``k - q``
        (matches the stored ``W0_qmunu[q]`` convention);
      * the four DENSITY vertices carry the GW ``chi_0(q)`` flip — written here
        as ``M_flip = conj(M_q)`` on the vertex itself, NOT by conjugating psi
        (that wholesale conjugation is exactly what the production payload
        does, and what the rung must not inherit);
      * the direct rungs are built from the PHYSICAL rolled psi, un-flipped;
      * the anti-resonant row is the derived hybrid (step 4) on those physical
        rungs.

    The prior version of this reference consumed the ``build_finite_q_data``
    payload and therefore certified the operator's conventions WITH the
    payload's — the shared assumption under which the resonant/anti-resonant
    rungs sat swapped at finite q, green here and red on reciprocity
    (claim 0215)."""
    grid = (int(payload["nkx"]), int(payload["nky"]), int(payload["nkz"]))
    nk = grid[0] * grid[1] * grid[2]
    psi_c = _roll_host(_host(payload["psi_c_X"]), q, grid)   # (k, c, s, mu), c at k-q
    psi_v = _host(payload["psi_v_X"])                        # (k, v, s, mu)
    eps_c = _roll_host(_host(payload["eps_c"]), q, grid)     # (k, c)
    eps_v = _host(payload["eps_v"])
    V_q0 = _host(payload["V_q_full"])[:, :, int(q[0]), int(q[1]), int(q[2])]
    W_q = _host(payload["W_q"])
    nc, nv, nmu = psi_c.shape[1], psi_v.shape[1], psi_c.shape[3]
    N = nc * nv * nk

    # Density vertex, flipped ON THE VERTEX: M_flip = conj(sum_s conj(psi_c) psi_v).
    M = np.conj(np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v))
    D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    lhs = np.einsum("kcvM,MN->kcvN", M, V_q0)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)).reshape(N, N) / nk

    # Direct rungs from the PHYSICAL rolled psi (no flip): W's momentum is the
    # internal k - k' (the q-shifts of the two c-legs cancel).
    Wflat = W_q.reshape(nmu, nmu, nk)
    Kd = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
    Kd_B = np.zeros_like(Kd)
    for k in range(nk):
        for kp in range(nk):
            Wq = Wflat[:, :, _q_flat(k, kp, grid)]
            Pc = np.einsum("ctm,Ctm->cCm", np.conj(psi_c[k]), psi_c[kp])
            Pv = np.einsum("vsn,Vsn->vVn", psi_v[k], np.conj(psi_v[kp]))
            Kd[:, :, k, :, :, kp] = np.einsum("cCm,mn,vVn->cvCV", Pc, Wq, Pv) / nk
            PcB = np.einsum("ctm,Vtm->cVm", np.conj(psi_c[k]), psi_v[kp])
            PvB = np.einsum("vsn,Csn->vCn", psi_v[k], np.conj(psi_c[kp]))
            Kd_B[:, :, k, :, :, kp] = np.einsum(
                "cVm,mn,vCn->cvCV", PcB, Wq, PvB) / nk
    Kd2, KdB2 = Kd.reshape(N, N), Kd_B.reshape(N, N)
    Dm = np.diag(D.reshape(-1).astype(np.complex128))
    A = Dm + Kx - Kd2
    B = Kx - KdB2
    Mmat = np.transpose(M, (1, 2, 0, 3)).reshape(N, nmu)
    row = (-(Kx - np.conj(KdB2)), -(Dm + Kx - np.conj(Kd2)))
    fake = dict(payload)
    fake["V_q0"] = V_q0                    # for _dense_wc_columns's seed/readout
    if with_row:
        return fake, A, B, Mmat, N, row
    return fake, A, B, Mmat, N


def _neg_q(q, grid):
    return tuple(int((-int(qi)) % int(g)) for qi, g in zip(q, grid))


@pytest.mark.gpu
def test_w_ladder_matches_dense_oracle_finite_q(ladder_payload):
    """Production ladder resolvent == FIRST-PRINCIPLES dense operator at ±q,
    and the dense operator itself obeys ``W(-q) = conj(W(q))``.

    Exercises the whole finite-q convention stack independently: the roll, the
    density-vertex flip, the UN-flipped rung, and the flipped-payload
    physical rung operand slots (``ladder_rung_slots``) that keep the resonant and
    anti-resonant rungs on their own rows.  The prior form of this cell built
    its reference FROM the ``build_finite_q_data`` payload, sharing exactly the
    convention under test — it stayed green while the rungs sat swapped
    (claim 0215's 3.6e-4 reciprocity break).  Reciprocity of the dense
    reference is asserted here too, so the derivation is checked against
    itself before production is checked against it.
    """
    harness.skip_unless_gpu(pytest)
    data, mesh, input_path = ladder_payload
    grid = (int(data["nkx"]), int(data["nky"]), int(data["nkz"]))

    # Precondition of the whole reciprocity statement: the stored W obeys
    # W(-q) = conj(W(q)), equivalently W_R elementwise-real.  If the INPUT
    # breaks it, nothing downstream can hold it.
    W = _host(data["W_q"])
    recip_in = 0.0
    for kf in range(grid[0] * grid[1] * grid[2]):
        c = np.unravel_index(kf, grid)
        cneg = tuple((-np.array(c)) % np.array(grid))
        recip_in = max(recip_in, float(np.abs(
            W[:, :, cneg[0], cneg[1], cneg[2]] - np.conj(W[:, :, c[0], c[1], c[2]])
        ).max()))
    recip_in /= max(float(np.abs(W).max()), 1e-300)
    assert recip_in < 1e-9, (
        f"input W breaks W(-q)=conj(W(q)) at {recip_in:.3e}; the ladder "
        f"cannot be gated on a property its kernel does not have")

    q_list = _symmetry_reduced_q_list(input_path)
    nz = q_list[np.any(q_list != 0, axis=1)]
    assert len(nz), "no nonzero symmetry-reduced q on this fixture"
    q = tuple(int(v) for v in nz[int(np.argmin((nz.astype(np.int64) ** 2).sum(axis=1)))])
    qm = _neg_q(q, grid)
    cols = _probe_cols(data)
    nlog = int(data["n_rmu"])

    # (0) NEGATIVE CONTROL — the raw diagonalizer gauge: the dense operator
    # must VIOLATE reciprocity there, or the gauge enforcement below is
    # curing nothing.  Measured 1.043e-03 on this fixture, 2026-08-16.
    refs_raw = {}
    for qq in (q, qm):
        fake, A, B, Mmat, N, row = _dense_ladder_blocks_finite_q(data, qq)
        refs_raw[qq] = _dense_wc_columns(fake, A, B, Mmat, N, cols, 0.0 + 0.0j,
                                         bottom=row)
    rel_raw = _relerr(refs_raw[qm][:nlog], np.conj(refs_raw[q][:nlog]))
    print(f"[dense-finite-q] raw-gauge dense reciprocity: {rel_raw:.3e}")
    assert rel_raw > 1e-5, (
        f"the raw-gauge dense operator obeys reciprocity at {rel_raw:.3e} — "
        f"then the TRS-gauge mechanism this cell certifies is not present on "
        f"this fixture and the enforcement below is untested")

    from bse.bse_w_exact import enforce_trs_pair_gauge
    data_g = enforce_trs_pair_gauge(data, mesh)

    refs, gots = {}, {}
    for qq in (q, qm):
        fake, A, B, Mmat, N, row = _dense_ladder_blocks_finite_q(data_g, qq)
        refs[qq] = _dense_wc_columns(fake, A, B, Mmat, N, cols, 0.0 + 0.0j,
                                     bottom=row)
        dq = build_finite_q_data(data_g, qq, mesh)
        got, resids, _ = _sharded_wc_columns(dq, mesh, cols, 0.0 + 0.0j)
        assert resids.max() < 1e-8, (
            f"q={qq}: GMRES not converged (max resid {resids.max():.2e})")
        gots[qq] = got

    # (a) The derivation is self-consistent IN THE TRS GAUGE: dense reciprocity.
    rel_dense = _relerr(refs[qm][:nlog], np.conj(refs[q][:nlog]))
    print(f"[dense-finite-q] TRS-gauge dense reciprocity: {rel_dense:.3e}")
    assert rel_dense < 1e-8, (
        f"the FIRST-PRINCIPLES dense ladder operator violates "
        f"W(-q) = conj(W(q)) at {rel_dense:.3e} IN THE TRS PAIR GAUGE — the "
        f"derivation, not the implementation, would be wrong")
    # (b) Production (gauge + physical rung slots) matches it at ±q separately.
    for qq in (q, qm):
        rel = _relerr(gots[qq][:nlog], refs[qq][:nlog])
        print(f"[dense-finite-q] q={qq}: production vs dense {rel:.3e}")
        assert rel < 1e-8, (
            f"q={qq}: ladder vs first-principles dense oracle rel_err {rel:.3e}")


@pytest.mark.gpu
def test_w_ladder_red_twin_uncompensated_flip_fails(ladder_payload):
    """RED TWIN: the rung fed the FLIPPED arrays must fail.

    Overriding the rung's operand slots with the FLIPPED density arrays is
    exactly the pre-2026-08-16 operator — the rung running on conjugated
    psi.  It must MISMATCH the first-principles dense reference at finite q;
    if it ever matches, the green cell above is not discriminating the rung
    operand convention and the reciprocity defect class (claim 0215) has
    gone unwatched.
    """
    harness.skip_unless_gpu(pytest)
    data, mesh, input_path = ladder_payload
    from bse.bse_w_exact import enforce_trs_pair_gauge
    data = enforce_trs_pair_gauge(data, mesh)   # isolate the FLIP defect alone
    q_list = _symmetry_reduced_q_list(input_path)
    nz = q_list[np.any(q_list != 0, axis=1)]
    q = tuple(int(v) for v in nz[int(np.argmin((nz.astype(np.int64) ** 2).sum(axis=1)))])
    cols = _probe_cols(data)
    nlog = int(data["n_rmu"])
    fake, A, B, Mmat, N, row = _dense_ladder_blocks_finite_q(data, q)
    ref = _dense_wc_columns(fake, A, B, Mmat, N, cols, 0.0 + 0.0j, bottom=row)
    dq = build_finite_q_data(data, q, mesh)
    dq_bad = dict(dq)
    for slot, src in (("psi_c_W_X", "psi_c_X"), ("psi_c_W_Y", "psi_c_Y"),
                      ("psi_v_W_X", "psi_v_X"), ("psi_v_W_Y", "psi_v_Y")):
        dq_bad[slot] = dq[src]                     # flipped arrays into the rung
    got_defective, resids, _ = _sharded_wc_columns(dq_bad, mesh, cols, 0.0 + 0.0j)
    assert resids.max() < 1e-8, f"GMRES not converged ({resids.max():.2e})"
    rel = _relerr(got_defective[:nlog], ref[:nlog])
    assert rel > 1e-6, (
        f"the UNCOMPENSATED flipped-payload operator differs from the "
        f"first-principles reference by only {rel:.3e} at q={q} — the finite-q "
        f"cell cannot see the rung-swap defect it exists to watch")


@pytest.mark.gpu
def test_w_ladder_red_twin_flipped_B_block_w_sign_fails(ladder_payload):
    """RED TWIN: B = Kx + Kd_B (W sign flipped in the coupling) must NOT match.

    This is the discriminating check for the derivation's step 4 — a
    branch-independent or wrong-signed coupling direct term changes
    ``A + B``, hence the static tile ``-2 v M^dag (A+B)^-1 M v``.  If this cell
    ever goes green, the green cell above is passing on something other than the
    operator it claims to test.
    """
    harness.skip_unless_gpu(pytest)
    data, mesh, _ = ladder_payload
    cols = _probe_cols(data)
    A, B, Mmat, N, row = _dense_ladder_blocks(data, with_row=True)
    # B = Kx - Kd_B, so Kd_B = Kx - B and the sign-flipped twin is Kx + Kd_B,
    # with the anti-resonant row flipped to match.
    Kx_only = _ring_kernel_only(data)
    Kd_B = Kx_only - B
    B_twin = Kx_only + Kd_B
    row_twin = (-(Kx_only + np.conj(Kd_B)), row[1])
    ref_good = _dense_wc_columns(data, A, B, Mmat, N, cols, 0.0 + 0.0j, bottom=row)
    ref_twin = _dense_wc_columns(data, A, B_twin, Mmat, N, cols, 0.0 + 0.0j,
                                 bottom=row_twin)
    got, _, _ = _sharded_wc_columns(data, mesh, cols, 0.0 + 0.0j)
    nlog = int(data["n_rmu"])
    rel_good = _relerr(got[:nlog], ref_good[:nlog])
    rel_twin = _relerr(got[:nlog], ref_twin[:nlog])
    assert rel_good < 1e-8, f"control leg regressed: rel_err {rel_good:.3e}"
    assert rel_twin > 1e-3, (
        f"flipping the B-block W sign changed the dense reference by only "
        f"{rel_twin:.3e} — the comparison does not discriminate the coupling "
        f"direct term, so the green cell proves nothing about it")


@pytest.mark.gpu
@pytest.mark.parametrize("row_kind", ["naive_symplectic", "dropped_w"])
def test_w_ladder_red_twin_wrong_antiresonant_row_fails(ladder_payload, row_kind):
    """RED TWIN: two wrong anti-resonant rows must NOT match the production one.

    ``naive_symplectic`` is ``[-B, -A]`` — the un-conjugated symplectic row this
    feature was FIRST written with.  It is not a strawman: it survives the RPA
    limit exactly and differs from the derived row by only ~1e-3 relative here,
    which is precisely why it needs a twin.  What exposed it was hermiticity
    (2.1e-05 vs 6.9e-15, probe leg JID 57052808) — the same class as the
    historical optical ``[-B, -A]`` bug in ``_antiresonant_row``'s docstring.

    ``dropped_w`` removes the direct term from the bottom-right block entirely.
    """
    harness.skip_unless_gpu(pytest)
    data, mesh, _ = ladder_payload
    cols = _probe_cols(data)
    A, B, Mmat, N, row = _dense_ladder_blocks(data, with_row=True)
    twin_row = ((-B, -A) if row_kind == "naive_symplectic"
                else (row[0], -_resonant_block_without_w(data)))
    ref_twin = _dense_wc_columns(data, A, B, Mmat, N, cols, 0.0 + 0.0j,
                                 bottom=twin_row)
    got, _, _ = _sharded_wc_columns(data, mesh, cols, 0.0 + 0.0j)
    nlog = int(data["n_rmu"])
    rel_twin = _relerr(got[:nlog], ref_twin[:nlog])
    assert rel_twin > 1e-5, (
        f"anti-resonant row twin {row_kind!r} changed the dense reference by "
        f"only {rel_twin:.3e} — the comparison does not discriminate that row")


# ---------------------------------------------------------------------------
# Small independent pieces the twins need (same einsums, no production kernel).
# ---------------------------------------------------------------------------
def _ring_kernel_only(payload):
    """``Kx = M V M^dag / Nk`` alone, as an (N, N) matrix."""
    psi_c = _host(payload["psi_c_X"])
    psi_v = _host(payload["psi_v_X"])
    V_q0 = _host(payload["V_q0"])
    nk = int(payload["nkx"]) * int(payload["nky"]) * int(payload["nkz"])
    N = psi_c.shape[1] * psi_v.shape[1] * nk
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)
    lhs = np.einsum("kcvM,MN->kcvN", M, V_q0)
    return np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)).reshape(N, N) / nk


def _resonant_block_without_w(payload):
    """``diag(D) + Kx`` — the A block with the direct term dropped."""
    eps_c = _host(payload["eps_c"])
    eps_v = _host(payload["eps_v"])
    D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    return (np.diag(D.reshape(-1).astype(np.complex128))
            + _ring_kernel_only(payload))
