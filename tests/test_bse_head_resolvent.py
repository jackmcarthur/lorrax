"""Gates for the ladder head resolvent (``bse.head_resolvent``).

WHAT IS AND IS NOT INDEPENDENT HERE, stated up front because "it converged" is
not a result:

* **Test 1 (guide section 8) is the gate, and it is independent.**  The
  reference is ``common.chi_from_dipole.compute_S_omega`` — the tree's OTHER
  implementation of the independent-particle dipole head, consumed by
  ``gw.head_correction`` and ``vcoul.q0_average``, sharing no code with
  ``bse/head_resolvent.py``.  It is what pins the
  ``pref = 4/(Omega N_k n_spin n_spinor)`` factor, the factor of two, and every
  sign.  ``K^d = K^x = 0`` is reached by ZEROING THE OPERANDS ``W_q`` / ``V_q0``,
  so the production ladder matvec, preconditioner and block-GMRES engine are all
  still the ones under test — not a smaller stand-in.

* **The conditioning and dense-solve cells are NOT independent** of the matvec:
  they densify the production operator by applying it to a 2N basis, so they
  check the SOLVE (Krylov convergence, the readout, the seed) against an exact
  solve of the same operator.  Operator-vs-first-principles independence is
  ``tests/test_bse_w_ladder_dense.py``'s job and is not restated here.

Fixtures: the restart-free synthetic payload contract from
``test_bse_w_ladder_dense._synthetic_payload``, rebuilt locally so this file has
no cross-test-module import, at two sizes.  ``SMALL`` (2x2x1, 2v2c, N = 16) for
the convention gates.  ``BIG`` (4x4x1, 4v4c, nmu = 16, N = 256, 2N = 512) for
the conditioning cell ONLY, because at N = 16 a full GMRES cycle spans the whole
space and every frequency "converges in five iterations" — a conditioning table
measured there would be an artefact of the fixture, not of the operator.
Every cell is seconds on one device.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest

import harness  # noqa: F401  (path bootstrap)

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                          # noqa: E402
from jax.sharding import Mesh                                    # noqa: E402

jax.config.update("jax_enable_x64", True)

from bse import head_resolvent as hr                             # noqa: E402
from common.chi_from_dipole import compute_S_omega               # noqa: E402


class Dims(NamedTuple):
    nkx: int
    nky: int
    nkz: int
    nc: int
    nv: int
    nmu: int

    @property
    def nk(self):
        return self.nkx * self.nky * self.nkz

    @property
    def npair(self):
        return self.nc * self.nv * self.nk


SMALL = Dims(2, 2, 1, 2, 2, 8)
BIG = Dims(4, 4, 1, 4, 4, 16)

#: Arbitrary but FIXED: Test 1 compares two formulas that must carry the same
#: prefactor, so the only requirement is that both sides see these numbers.
CELL_VOLUME, NSPIN, NSPINOR = 137.0, 1, 1


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def _neg_k_index(D: Dims):
    """Flat index of ``-k`` on the C-order (nkx, nky, nkz) grid."""
    grid = (D.nkx, D.nky, D.nkz)
    c = np.array(np.unravel_index(np.arange(D.nk), grid))
    return np.ravel_multi_index(
        tuple((-c) % np.array([[D.nkx], [D.nky], [D.nkz]])), grid)


def _synthetic_payload(mesh, D: Dims = SMALL, *, seed=20260817, trs=False,
                       kernel_scale=1.0):
    """Restart-FREE BSE payload in the production key contract.

    Same construction as ``test_bse_w_ladder_dense._synthetic_payload``: ``V_q0``
    real symmetric, ``W_q`` Hermitian per q and obeying ``W(-q) = conj(W(q))`` —
    the two identities that make the direct rung Hermitian and its swapped
    partner complex-symmetric, i.e. the preconditions of the operator's
    eta-pseudo-Hermiticity.  Energies well separated so a shifted solve at small
    ``z`` is well conditioned.

    ``trs=True`` additionally symmetrizes ``eps`` under ``k -> -k``.  That is the
    SAMPLING half of the symmetry ruling (see the module docstring of
    ``bse.head_resolvent``): it is what lets the transition sum pair terms, and
    it is deliberately NOT the default, so the generic cells measure the honest
    non-symmetric case.

    ``kernel_scale`` multiplies ``V_q0`` and ``W_q``, i.e. ``K^x`` and ``K^d``
    together.  At 1.0 the operator is strongly diagonally dominant (measured
    off-diagonal Frobenius fraction 0.001 at ``BIG``), which makes the Jacobi
    preconditioner nearly exact and every shifted solve converge in ~5
    iterations — a number about the FIXTURE, not the operator.  The conditioning
    cell therefore reports a weak and a strong arm and says which is which.
    """
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude

    rng = np.random.default_rng(seed)
    sh = make_bse_shardings(mesh)
    neg = _neg_k_index(D)

    def _c(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / 8.0

    psi_c = _c(D.nk, D.nc, 1, D.nmu)
    psi_v = _c(D.nk, D.nv, 1, D.nmu)
    eps_c = 2.0 + 0.3 * rng.standard_normal((D.nk, D.nc))
    eps_v = -2.0 + 0.3 * rng.standard_normal((D.nk, D.nv))
    if trs:
        eps_c = 0.5 * (eps_c + eps_c[neg])
        eps_v = 0.5 * (eps_v + eps_v[neg])
    Vq0 = rng.standard_normal((D.nmu, D.nmu)) * 0.1 * kernel_scale
    Vq0 = Vq0 + Vq0.T
    Wq = _c(D.nmu, D.nmu, D.nk) * 0.1 * kernel_scale
    Wq = 0.5 * (Wq + np.conj(np.transpose(Wq, (1, 0, 2))))
    Wq = 0.5 * (Wq + np.conj(Wq[:, :, neg]))
    Wq = Wq.reshape(D.nmu, D.nmu, D.nkx, D.nky, D.nkz)

    with mesh:
        d = {
            "psi_c_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_x),
            "psi_c_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_y),
            "psi_v_X": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_x),
            "psi_v_Y": jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_y),
            "eps_c": jnp.asarray(eps_c), "eps_v": jnp.asarray(eps_v),
            "W_q": jax.lax.with_sharding_constraint(jnp.asarray(Wq), sh.W),
            "V_q0": jax.lax.with_sharding_constraint(jnp.asarray(Vq0), sh.V),
            "nkx": D.nkx, "nky": D.nky, "nkz": D.nkz,
            "n_cond_pad": D.nc, "n_val_pad": D.nv, "n_rmu": D.nmu,
        }
        d["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_X"], d["psi_v_X"]), sh.psi_x)
        d["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_Y"], d["psi_v_Y"]), sh.psi_y)
    return d


@pytest.fixture(scope="module")
def mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))


@pytest.fixture(scope="module")
def payload(mesh):
    return _synthetic_payload(mesh, SMALL)


def _velocity(D: Dims, seed, *, trs=False):
    """A random Cartesian velocity block ``<c k| v_i |v k>``, ``(3, nk, nc, nv)``.

    ``trs=True`` imposes ``v(-k) = -conj(v(k))`` — the relation the antiunitary
    map gives between a Kramers-partner pair's velocity matrix elements.  It
    makes ``sum_k conj(v_i) v_j`` REAL term-pair by term-pair, which is the ONLY
    thing that makes the head tensor symmetric (nothing about the engine does).
    """
    rng = np.random.default_rng(seed)
    shape = (3, D.nk, D.nc, D.nv)
    v = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    if trs:
        v = 0.5 * (v - np.conj(v[:, _neg_k_index(D)]))
    return v


def _delta_from_payload(data):
    """``(nk, nc, nv)`` transition energies of the payload — the operator's own
    diagonal, in the ``v_cvk`` axis order."""
    return np.transpose(hr.payload_delta(data), (2, 0, 1))


def _reference_S(D: Dims, v_cvk, delta_cvk, z_values):
    """``compute_S_omega`` on the same window — the INDEPENDENT Test 1 oracle.

    Its API is the full ``(3, nk, nb, nb)`` dipole / ``(nk, nb, nb)`` deltaE /
    ``(nk, nb)`` occupation triple, and it slices ``c = nelec..nb``,
    ``v = 0..nelec`` out of them.  So the (nc, nv) block is embedded at
    ``[.., nv + c, v]`` with ``nelec = nv``; nothing outside that block is read.
    """
    nb = D.nv + D.nc
    dip = np.zeros((3, D.nk, nb, nb), dtype=np.complex128)
    dE = np.zeros((D.nk, nb, nb), dtype=np.float64)
    dip[:, :, D.nv:, :D.nv] = v_cvk
    dE[:, D.nv:, :D.nv] = delta_cvk
    f_nk = np.zeros((D.nk, nb), dtype=np.float64)
    f_nk[:, :D.nv] = 1.0
    return np.asarray(compute_S_omega(
        jnp.asarray(dip), jnp.asarray(dE), jnp.asarray(f_nk),
        CELL_VOLUME, D.nk, NSPIN, NSPINOR,
        jnp.asarray(np.atleast_1d(z_values), dtype=jnp.complex128), eta=0.0))


def _pref(D: Dims):
    return hr.head_prefactor(CELL_VOLUME, D.nk, NSPIN, NSPINOR)


def _dense_operator(D: Dims, op, data):
    """The 2N x 2N dense matrix of the production matvec, by applying it to the
    2N-column identity in ONE call (the probe axis is a batch axis).

    NOT an independent build — that is ``test_bse_w_ladder_dense``'s job.  This
    exists so the conditioning cell can name the eigenvalues it is walking
    toward and can solve exactly at the same ``z``.
    """
    n = D.npair
    eye = np.zeros((2, 2 * n, D.nc, D.nv, D.nk), dtype=np.complex128)
    for m in range(2 * n):
        b, t = divmod(m, n)
        c, v, k = np.unravel_index(t, (D.nc, D.nv, D.nk))
        eye[b, m, c, v, k] = 1.0
    x = jax.lax.with_sharding_constraint(jnp.asarray(eye), op.sh.X_full)
    out = np.asarray(jax.device_get(op.matvec(x, *op.operands_fn(data))))
    # out[:, m] is H applied to basis vector m -> column m of the dense H.
    return out.transpose(1, 0, 2, 3, 4).reshape(2 * n, 2 * n).T


def _dense_head(D: Dims, H, d_pair, z, pref):
    """Exact ``Xi(z)`` from a dense solve of the SAME operator: the [d; -d] seed
    and the X + Y readout, with ``numpy.linalg.solve`` in place of GMRES."""
    n = D.npair
    d = np.asarray(d_pair).reshape(3, n)
    b = np.concatenate([d, -d], axis=1).T                  # (2N, 3)
    x = np.linalg.solve(z * np.eye(2 * n) - H, b)          # (2N, 3)
    s = x[:n] + x[n:]                                      # (N, 3)
    return pref * (np.conj(d) @ s)


def _dipole(D: Dims, data, seed, *, trs=False):
    delta = _delta_from_payload(data)
    v = _velocity(D, seed, trs=trs)
    return v, delta, hr.build_head_dipole(v, delta, n_cond_pad=D.nc,
                                          n_val_pad=D.nv)


# ---------------------------------------------------------------------------
# Test 1 — the gate.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("z", [0.0, 1.0, 2.5 + 0.1j, 0.7j])
def test_1_zero_kernel_head_reproduces_compute_S_omega(payload, mesh, z):
    """K^d = K^x = 0 => Xi must BE the independent-particle dipole head.

    SCOPE: q = 0, one synthetic 2x2x1 / 2v2c payload, four frequencies (static,
    real sub-gap, complex, imaginary), single device.  What it pins is the
    convention — prefactor, factor of two, seed sign, readout sign, index order
    — which is exactly what a convention gate is for and is grid-independent.
    It says nothing about the ladder kernel; that is the cell below.
    """
    data = hr.zero_kernel_payload(payload)
    op = hr.build_head_operator(mesh, data, include_w=True)
    v_cvk, delta, d_pair = _dipole(SMALL, payload, 11)

    got = hr.solve_head_tensor(op, data, d_pair, [z], pref=_pref(SMALL),
                               max_iter=64, tol=1e-13, arm="K=0",
                               delta_cvk=delta)
    ref = _reference_S(SMALL, v_cvk, delta, [z])

    assert got.delta_mismatch < 1e-14, (
        f"the dipole and the operator disagree on Delta by "
        f"{got.delta_mismatch:.3e}; Test 1 compares two objects built from "
        f"different transition energies and would be meaningless")
    assert got.resids.max() < 1e-11, f"GMRES not converged: {got.resids}"
    rel = np.linalg.norm(got.xi[0] - ref[0]) / np.linalg.norm(ref[0])
    print(f"\n[TEST 1] z={z!r:>12}  rel_err vs compute_S_omega = {rel:.3e}  "
          f"max_true_resid = {got.resids.max():.2e}  iters = "
          f"{got.iters.max()}")
    assert rel < 1e-12, (
        f"z={z}: Xi vs compute_S_omega rel_err {rel:.3e}\n"
        f"Xi  =\n{got.xi[0]}\nref =\n{ref[0]}")


def test_1_red_twin_a_missing_factor_of_two_fails(payload, mesh):
    """RED TWIN (QUALITY_PATTERNS 1): the gate above must be able to SEE the
    prefactor it claims to pin.  Halving it has to break the comparison."""
    data = hr.zero_kernel_payload(payload)
    op = hr.build_head_operator(mesh, data, include_w=True)
    v_cvk, delta, d_pair = _dipole(SMALL, payload, 11)
    bad = hr.solve_head_tensor(op, data, d_pair, [1.0], pref=0.5 * _pref(SMALL),
                               max_iter=64, tol=1e-13)
    ref = _reference_S(SMALL, v_cvk, delta, [1.0])
    rel = np.linalg.norm(bad.xi[0] - ref[0]) / np.linalg.norm(ref[0])
    assert rel > 0.4, f"a 2x prefactor error scored only {rel:.3e}"


def test_1_red_twin_a_resonant_only_seed_fails(payload, mesh):
    """RED TWIN: the ``[d; -d]`` antiresonant seed is what produces the factor
    of two, so a ``[d; 0]`` (resonant-only) seed must fail Test 1."""
    data = hr.zero_kernel_payload(payload)
    op = hr.build_head_operator(mesh, data, include_w=True)
    v_cvk, delta, d_pair = _dipole(SMALL, payload, 11)
    rhs = np.array(jax.device_get(hr.build_head_rhs(d_pair, op.sh)))
    rhs[1] = 0.0
    bad = hr.solve_head_tensor(op, data, d_pair, [1.0], pref=_pref(SMALL),
                               max_iter=64, tol=1e-13, rhs=jnp.asarray(rhs))
    ref = _reference_S(SMALL, v_cvk, delta, [1.0])
    rel = np.linalg.norm(bad.xi[0] - ref[0]) / np.linalg.norm(ref[0])
    assert rel > 0.1, f"dropping the antiresonant seed scored only {rel:.3e}"


def test_head_rhs_is_the_probe_block_antiresonant_structure(payload, mesh):
    """``build_head_rhs`` emits exactly ``build_probe_rhs``'s ``[f; -f]`` shape
    with ``f -> d`` — the structural claim the convention ruling rests on."""
    op = hr.build_head_operator(mesh, payload, include_w=True)
    _, _, d_pair = _dipole(SMALL, payload, 3)
    rhs = np.asarray(jax.device_get(hr.build_head_rhs(d_pair, op.sh)))
    assert rhs.shape == (2, 3, SMALL.nc, SMALL.nv, SMALL.nk)
    np.testing.assert_allclose(rhs[0], d_pair, atol=0, rtol=0)
    np.testing.assert_allclose(rhs[1], -np.asarray(d_pair), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# The interacting operator: correctness, conditioning, symmetry.
# ---------------------------------------------------------------------------
def test_ladder_head_matches_a_dense_solve_of_the_same_operator(payload, mesh):
    """With the full ladder kernel on, the GMRES head equals the exact solve.

    SCOPE: this validates the SEED, the READOUT and the Krylov solve against a
    dense LU of the same operator — not the operator itself (see the module
    docstring).
    """
    op = hr.build_head_operator(mesh, payload, include_w=True)
    H = _dense_operator(SMALL, op, payload)
    _, _, d_pair = _dipole(SMALL, payload, 5)
    z = 0.3 + 0.05j
    got = hr.solve_head_tensor(op, payload, d_pair, [z], pref=_pref(SMALL),
                               max_iter=200, tol=1e-13, arm="ladder")
    ref = _dense_head(SMALL, H, d_pair, z, _pref(SMALL))
    assert got.resids.max() < 1e-11, f"GMRES not converged: {got.resids}"
    rel = np.linalg.norm(got.xi[0] - ref) / np.linalg.norm(ref)
    assert rel < 1e-10, f"ladder head vs dense solve rel_err {rel:.3e}"


def test_operator_is_eta_pseudo_hermitian(payload, mesh):
    """``eta H = (eta H)^dag``, ``eta = diag(I, -I)`` — the structural
    certificate the Hermiticity half of the symmetry ruling is derived from
    (``w_ladder`` module docstring step 5)."""
    op = hr.build_head_operator(mesh, payload, include_w=True)
    H = _dense_operator(SMALL, op, payload)
    eta = np.diag(np.concatenate([np.ones(SMALL.npair), -np.ones(SMALL.npair)]))
    A = eta @ H
    rel = np.linalg.norm(A - A.conj().T) / np.linalg.norm(A)
    print(f"\n[eta-pseudo-Hermiticity] |etaH - (etaH)^dag|/|etaH| = {rel:.3e}")
    assert rel < 1e-12, f"eta-pseudo-Hermiticity violated at {rel:.3e}"


def test_xi_is_hermitian_at_real_z_but_not_symmetric(payload, mesh):
    """THE SYMMETRY RULING, measured on both halves.

    * Hermitian at real ``z``: STRUCTURAL, from eta-pseudo-Hermiticity.  Holds
      with the full ladder kernel and a generic complex dipole.
    * Symmetric: NOT structural.  It needs the transition SUM to pair
      ``k`` with ``-k`` (``v(-k) = -conj(v(k))``, ``Delta(-k) = Delta(k)``).
      Without that pairing Xi is measurably non-symmetric — asserted as a red
      twin, because a reader that assumes symmetry silently keeps only
      ``(Xi + Xi^T)/2`` (``gw.head_densify.head_scalar_pointwise`` contracts
      with REAL mini-BZ draws) and cannot detect what it dropped.
    """
    op = hr.build_head_operator(mesh, payload, include_w=True)
    _, _, d_pair = _dipole(SMALL, payload, 7)
    got = hr.solve_head_tensor(op, payload, d_pair, [0.4], pref=_pref(SMALL),
                               max_iter=200, tol=1e-13)
    print(f"\n[symmetry, no TRS pairing] |Xi-Xi^T|/|Xi| = {got.asym[0]:.3e}  "
          f"|Xi-Xi^dag|/|Xi| = {got.anti_herm[0]:.3e}")
    assert got.resids.max() < 1e-11
    assert got.anti_herm[0] < 1e-9, (
        f"Xi is not Hermitian at real z: |Xi - Xi^dag|/|Xi| = "
        f"{got.anti_herm[0]:.3e}.  That contradicts eta-pseudo-Hermiticity of "
        f"the operator, so it is an operator or readout defect, not a "
        f"tolerance.")
    assert got.asym[0] > 1e-3, (
        f"Xi came out symmetric ({got.asym[0]:.3e}) on a payload with no "
        f"k -> -k pairing.  Either the engine is symmetrizing (it must not) or "
        f"this fixture accidentally acquired TRS.")


def test_xi_becomes_real_symmetric_under_k_to_minus_k_pairing(mesh):
    """The other half of the ruling: with TRS-paired velocities and energies the
    SAME engine returns a real symmetric Xi.  Sampling, not structure.

    SCOPE: K = 0, so the pairing statement is about the transition SUM alone —
    imposing TRS on the random ladder kernel as well is a separate construction
    (``test_bse_w_ladder_dense._trs_synthetic_payload``) and is not what this
    cell is about.
    """
    data = hr.zero_kernel_payload(
        _synthetic_payload(mesh, SMALL, seed=99, trs=True))
    op = hr.build_head_operator(mesh, data, include_w=True)
    _, _, d_pair = _dipole(SMALL, data, 13, trs=True)
    got = hr.solve_head_tensor(op, data, d_pair, [0.4], pref=_pref(SMALL),
                               max_iter=64, tol=1e-13)
    imag = np.max(np.abs(got.xi[0].imag)) / np.max(np.abs(got.xi[0]))
    print(f"\n[symmetry, TRS-paired] |Xi-Xi^T|/|Xi| = {got.asym[0]:.3e}  "
          f"max|Im Xi|/max|Xi| = {imag:.3e}")
    assert got.asym[0] < 1e-12, f"TRS-paired Xi asym {got.asym[0]:.3e}"
    assert imag < 1e-12, f"TRS-paired Xi has |Im|/|Xi| = {imag:.3e}"


ETAS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def _conditioning_walk(D, mesh, data, *, label, max_iter=400, tol=1e-11,
                       seed=17):
    """Walk ``z = lambda_min(+) + i*eta`` toward an eigenvalue and print the
    table.  Returns ``(lam, rows)`` with ``rows = (eta, iters, true_resid,
    rel_err_vs_dense, |Xi|)``."""
    op = hr.build_head_operator(mesh, data, include_w=True)
    H = _dense_operator(D, op, data)
    evals = np.linalg.eigvals(H)
    real_pos = np.sort(np.real(evals[(np.real(evals) > 0.0)
                                     & (np.abs(np.imag(evals)) < 1e-8)]))
    assert real_pos.size, "no real positive eigenvalue to approach"
    lam = float(real_pos[0])
    offdiag = (np.linalg.norm(H - np.diag(np.diag(H)))
               / np.linalg.norm(H))

    _, _, d_pair = _dipole(D, data, seed)
    z_list = [lam + 1j * e for e in ETAS]
    got = hr.solve_head_tensor(op, data, d_pair, z_list, pref=_pref(D),
                               max_iter=max_iter, tol=tol, arm=label)
    rows = []
    for i, e in enumerate(ETAS):
        ref = _dense_head(D, H, d_pair, z_list[i], _pref(D))
        rel = np.linalg.norm(got.xi[i] - ref) / np.linalg.norm(ref)
        rows.append((e, int(got.iters[i].max()), float(got.resids[i].max()),
                     rel, float(np.linalg.norm(got.xi[i]))))
    gap = float(np.min(np.abs(real_pos[1:] - lam))) if real_pos.size > 1 else 0.0
    print(f"\n[conditioning: {label}] lambda_min(+) = {lam:.6f} Ry, nearest "
          f"other real eigenvalue {gap:.3e} away; N = {D.npair}, "
          f"2N = {2*D.npair}, off-diagonal Frobenius fraction {offdiag:.4f}, "
          f"cap = {max_iter}, tol = {tol:.0e}")
    print("   eta        iters   true_resid    err_vs_dense   |Xi|")
    for e, it, res, rel, nrm in rows:
        print(f"  {e:.0e}   {it:5d}   {res:.3e}    {rel:.3e}    {nrm:.4e}")
    return lam, rows


@pytest.mark.parametrize("kernel_scale,label", [(1.0, "weak K"),
                                                (100.0, "strong K")])
def test_conditioning_as_z_approaches_an_eigenvalue(mesh, kernel_scale, label):
    """GMRES conditioning as ``z -> lambda(H_bar)`` — production WILL sample it.

    Reports iterations, the TRUE relative residual and the error against an
    exact solve at the same ``z``.  Two claims, and only the first is a pass/fail
    assertion:

    1. The solve never returns a truncated answer silently — either it converges
       within the cap, or ``iters`` reaches the cap and says so.  (The
       cap-reporting half is pinned separately by
       :func:`test_a_truncated_solve_is_reported_not_hidden`.)
    2. REPORTED, not asserted: ``|Xi|`` grows as ``1/eta`` while the relative
       GMRES tolerance is held fixed, so the ABSOLUTE accuracy of the head
       degrades in lockstep with the pole enhancement.  That is the production
       hazard a frequency grid near an exciton walks into, and it is a property
       of the object, not of this fixture.

    SCOPE: ``BIG`` (N = 256, 2N = 512), synthetic operator, single device.  The
    two ``kernel_scale`` arms exist because the natural synthetic kernel is
    strongly diagonally dominant (off-diagonal Frobenius fraction ~0.001), which
    makes the Jacobi preconditioner nearly exact; the iteration counts on that
    arm are a FLOOR and must not be quoted as production numbers.  A real deck's
    ladder operator is stiffer — ``test_bse_w_ladder_dense`` says so in as many
    words — and the real-payload cell below is the measurement on one.
    """
    D = BIG
    data = _synthetic_payload(mesh, D, seed=31337, kernel_scale=kernel_scale)
    max_iter = 400
    _, rows = _conditioning_walk(D, mesh, data, label=label, max_iter=max_iter)
    for e, it, res, rel, _ in rows:
        assert it <= max_iter
        if it < max_iter:
            assert res < 1e-9, (
                f"eta={e:.0e}: GMRES exited at {it} iterations below the cap "
                f"but the TRUE residual is {res:.3e} — the early-exit test and "
                f"the recomputed residual disagree, which is the "
                f"resid_relative_to defect class")
            assert rel < 1e-6, (
                f"eta={e:.0e}: converged (resid {res:.3e}) but Xi is "
                f"{rel:.3e} from the exact solve")


def test_a_truncated_solve_is_reported_not_hidden(mesh):
    """FAILURE SIGNATURE, not a success marker: at a cap the solve cannot meet,
    ``iters`` must reach the cap and the true residual must be large.

    This is the whole reason :class:`HeadTensorResult` carries ``iters``: a head
    that quietly returned a 2-iteration Krylov guess would still look like a
    3x3 tensor, and no downstream consumer could tell.
    """
    D = BIG
    data = _synthetic_payload(mesh, D, seed=31337, kernel_scale=100.0)
    op = hr.build_head_operator(mesh, data, include_w=True)
    _, _, d_pair = _dipole(D, data, 17)
    got = hr.solve_head_tensor(op, data, d_pair, [2.0 + 1e-4j], pref=_pref(D),
                               max_iter=2, tol=1e-11)
    assert int(got.iters.max()) == 2, f"iters did not report the cap: {got.iters}"
    assert got.resids.max() > 1e-6, (
        f"a 2-iteration solve of this operator reported a converged residual "
        f"{got.resids.max():.2e}; the residual is not measuring the solve")


def _dims_from_payload(data) -> Dims:
    return Dims(int(data["nkx"]), int(data["nky"]), int(data["nkz"]),
                int(data["n_cond_pad"]), int(data["n_val_pad"]),
                int(data["V_q0"].shape[0]))


def test_conditioning_on_the_real_gnppm_ladder_payload(gnppm_session, mesh):
    """The same walk on a REAL restart payload — the production operator.

    SCOPE: the gnppm fixture (MoS2 3x3x1, nk = 9) at a 2v2c window,
    ``inject_head=False`` (the resolvent carries no q=0 head by construction),
    q = 0, one device.  N is only 36, so this is a stiffness measurement on a
    physically-scaled kernel, NOT a large-Krylov study; the dipole is random
    because conditioning is a property of the operator and the pole, not of the
    seed.  Quoted alongside the synthetic arms so the two can be compared rather
    than conflated.
    """
    from bse import bse_io

    input_path = str(gnppm_session.run_dir / gnppm_session.input_name)
    restart = bse_io._find_restart_file(input_path)
    data = bse_io.load_bse_data_from_restart_sharded(
        restart, n_val=2, n_cond=2, mesh_xy=mesh, input_file=input_path,
        inject_head=False)
    D = _dims_from_payload(data)
    max_iter = 400
    _, rows = _conditioning_walk(D, mesh, data, label="gnppm 2v2c ladder",
                                 max_iter=max_iter, seed=23)
    for e, it, res, rel, _ in rows:
        assert it <= max_iter
        if it < max_iter:
            assert res < 1e-9, (
                f"eta={e:.0e}: exited at {it} < cap with true residual "
                f"{res:.3e}")


# ---------------------------------------------------------------------------
# Operator identity.
# ---------------------------------------------------------------------------
def test_operator_identity_is_cached_and_distinct(mesh):
    """``id(matvec)`` is the block-GMRES cache key, so identity is correctness.

    Three claims: same structural key => the SAME object (no leaked compile per
    call); different key (``include_w``) => a DIFFERENT object; and
    ``assert_distinct_operators`` refuses the collision that would make one arm
    of an A/B re-run the other.
    """
    # OWN payloads: ``build_ladder_resolvent`` mutates the dict it is handed
    # (``ensure_W_R``), and the ``include_w=False`` arm installs W_R = W_q where
    # the ``include_w=True`` arm installs its inverse-FFT.  Running both against
    # a shared fixture would leave whichever ran last in place for every other
    # cell — an order-dependent corruption, invisible under xdist.
    own = _synthetic_payload(mesh, SMALL, seed=4242)
    other = _synthetic_payload(mesh, SMALL, seed=4242)
    a = hr.build_head_operator(mesh, own, include_w=True)
    a2 = hr.build_head_operator(mesh, own, include_w=True)
    b = hr.build_head_operator(mesh, other, include_w=False)
    assert a.matvec_id == a2.matvec_id, "structural key did not hit the cache"
    assert a.key != b.key and a.matvec_id != b.matvec_id
    hr.assert_distinct_operators(a, a2, b)

    forged = hr.HeadOperator(matvec=a.matvec, diag_h=a.diag_h, sh=a.sh,
                             operands_fn=a.operands_fn, key=b.key)
    with pytest.raises(AssertionError, match="share matvec id"):
        hr.assert_distinct_operators(a, forged)


def test_zero_kernel_payload_shares_the_operator_with_the_full_one(payload, mesh):
    """Test 1's ``K = 0`` arm is an OPERAND change, so it must ride the same
    compiled program as the production arm — the runtime-argument design in
    ``bse_w_exact._get_block_gmres_solver``.  If this ever stops holding, Test 1
    has quietly moved to testing a different operator than production runs."""
    full = hr.build_head_operator(mesh, payload, include_w=True)
    zero = hr.build_head_operator(mesh, hr.zero_kernel_payload(payload),
                                  include_w=True)
    assert zero.key == full.key and zero.matvec_id == full.matvec_id


def test_zero_kernel_payload_gets_its_own_preconditioner(payload, mesh):
    """...and the shared program must NOT come with a shared ``diag_h``.

    ``diag_h`` is ``diag(H)``, a function of the payload's ``eps``/``W_q``/
    ``V_q0``.  Caching it alongside the matvec is the bug this cell pins: it
    would precondition the zeroed arm with the full kernel's diagonal (or the
    reverse, depending on call order) and Test 1 would be measuring a
    mis-preconditioned solve.
    """
    full = hr.build_head_operator(mesh, payload, include_w=True)
    zero = hr.build_head_operator(mesh, hr.zero_kernel_payload(payload),
                                  include_w=True)
    dh_full = np.asarray(jax.device_get(full.diag_h))
    dh_zero = np.asarray(jax.device_get(zero.diag_h))
    delta = np.transpose(hr.payload_delta(payload), (0, 1, 2))
    # K = 0 => diag(H) is exactly [Delta, -Delta].
    np.testing.assert_allclose(dh_zero[0, 0], delta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dh_zero[1, 0], -delta, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(dh_full - dh_zero)) > 1e-6, (
        "the full and zeroed arms share a preconditioner diagonal")


def test_build_head_dipole_refuses_a_non_positive_delta(payload):
    v = _velocity(SMALL, 2)
    delta = _delta_from_payload(payload).copy()
    delta[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        hr.build_head_dipole(v, delta, n_cond_pad=SMALL.nc, n_val_pad=SMALL.nv)
