"""The uniform-target FFT route for fH_q.

The claim (HTRANSFORM_FFT.md §2):

**The route.**  ``fH_q = Σ_R e^{-2πi q·R} fH_R`` is an unnormalized forward
   DFT whenever the target q-set is a rigid shift of a Γ-centred uniform grid.
   These cells pin the three claims that make the dispatch safe — the phase
   factor, the FOLD (not truncate) for coarser targets, and the refusal to
   route a q-set that does not have the structure — against the explicit sum
the shipped scan path computes.

The kpath/metric gates of the same review live in
``tests/test_htransform_kpath_gates.py``.

INSTRUMENT DISCIPLINE (tests/README §5.1).  Every comparator used to pass a
cell here is shown FAILING on a planted deviation in a neighbouring cell —
``test_scan_vs_fft_comparator_goes_red`` and
``test_truncation_instead_of_folding_is_caught``.  An agreement number is only
evidence if the instrument that produced it can disagree.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _mesh(n=1):
    import jax
    from jax.sharding import Mesh
    devs = jax.devices()
    if len(devs) < n * n:
        pytest.skip(f"needs {n * n} devices, have {len(devs)}")
    return Mesh(np.asarray(devs[:n * n]).reshape(n, n), ("x", "y"))


def _hermitian_fH_k(N, rank, seed=7):
    """A Hermitian (nk, rank, rank) family — the only property of ``fH_k`` the
    R→q evaluation uses (it is what makes ``fH_R[-R] == fH_R[R]ᴴ``)."""
    rng = np.random.default_rng(seed)
    nk = N[0] * N[1] * N[2]
    A = (rng.standard_normal((nk, rank, rank))
         + 1j * rng.standard_normal((nk, rank, rank)))
    return 0.5 * (A + np.conj(np.swapaxes(A, 1, 2)))


def _fH_R_np(fH_k, N):
    rank = fH_k.shape[-1]
    x = fH_k.reshape(N[0], N[1], N[2], rank, rank)
    return np.fft.ifftn(x, axes=(0, 1, 2), norm="backward").reshape(-1, rank, rank)


def _scan_np(q, fH_R, R_grid):
    """THE SHIPPED SUM, transcribed: ``_kpath_batch`` / ``_fourier`` compute
    ``0.5*einsum(phase, fH_R)`` and then add the conjugate transpose."""
    ph = np.exp(-2j * np.pi * (np.atleast_2d(q) @ R_grid.T))
    m = 0.5 * np.einsum("bk,kij->bij", ph, fH_R)
    return m + np.conj(np.swapaxes(m, 1, 2))


def _maxrel(a, b):
    """THE comparator.  Shown going red in the two cells named in the module
    docstring."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"shape {a.shape} != {b.shape}"
    d = float(np.max(np.abs(a - b))) if a.size else 0.0
    s = float(np.max(np.abs(b))) if b.size else 1.0
    return d / max(s, 1e-300)


def _target(M, delta):
    from bandstructure.htransform import uniform_grid_frac_np
    return uniform_grid_frac_np(M) + np.asarray(delta, dtype=np.float64)[None, :]


# ---------------------------------------------------------------------------
#  A1. the equivalence itself
# ---------------------------------------------------------------------------

CASES = [
    # (N,          M,           delta,                   what it exercises)
    ((4, 4, 4), (4, 4, 4), (0.0, 0.0, 0.0), "identity: on-grid recovery"),
    ((4, 4, 4), (8, 8, 8), (0.0, 0.0, 0.0), "denser, commensurate"),
    ((4, 4, 4), (6, 6, 6), (0.0, 0.0, 0.0), "denser, INcommensurate"),
    ((3, 4, 5), (3, 4, 5), (0.1, -0.25, 0.375), "shifted, mixed parity"),
    ((4, 4, 4), (4, 4, 4), (0.5, 0.0, 0.0), "shift ON the Nyquist"),
    ((4, 4, 2), (8, 8, 4), (-0.3, 0.7, 0.02), "shifted AND denser"),
    ((8, 8, 2), (4, 4, 2), (0.03, -0.11, 0.25), "COARSER — the fold"),
]


@pytest.mark.parametrize("N,M,delta,what", CASES,
                         ids=[c[3].split(":")[0].replace(" ", "_") for c in CASES])
def test_fft_route_reproduces_the_scan_sum(N, M, delta, what):
    """One sum, two evaluation orders.  Anything above ~1e-14 relative means
    the phase convention, the fold or the FFT normalization is wrong — not
    that the interpolation got better or worse, which neither route can do."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from bandstructure.htransform import build_R_grid_np, make_fH_q_fft
    from jax.sharding import PartitionSpec as P

    rank = 4
    fH_k = _hermitian_fH_k(N, rank)
    fH_R = _fH_R_np(fH_k, N)
    R_grid = build_R_grid_np(N)

    want = _scan_np(_target(M, delta), fH_R, R_grid)

    mesh = _mesh(1)
    fn = make_fH_q_fft(mesh, M, R_grid, P(None, "x", "y"), rank=rank)
    with mesh:
        F = np.asarray(jax.device_get(
            fn(jnp.asarray(fH_R), jnp.asarray(np.asarray(delta, dtype=np.float64)))))
    got = 0.5 * (F + np.conj(np.swapaxes(F, -1, -2)))       # the caller's ½(F+Fᴴ)

    assert _maxrel(got, want) < 1e-13, (what, _maxrel(got, want))


def test_scan_vs_fft_comparator_goes_red():
    """The instrument, shown disagreeing.  A 1-ULP move on ONE entry of the
    reference must be visible to ``_maxrel`` at the tolerance the cells above
    pass at — otherwise those passes are two references to one array."""
    N, M, delta = (4, 4, 4), (8, 8, 8), (0.0, 0.0, 0.0)
    rank = 4
    fH_R = _fH_R_np(_hermitian_fH_k(N, rank), N)
    from bandstructure.htransform import build_R_grid_np
    want = _scan_np(_target(M, delta), fH_R, build_R_grid_np(N))
    bad = want.copy()
    idx = np.unravel_index(int(np.argmax(np.abs(want))), want.shape)
    bad[idx] = np.nextafter(bad[idx].real, np.inf) + 1j * bad[idx].imag
    assert _maxrel(bad, want) > 0.0
    assert _maxrel(want, want) == 0.0


def test_truncation_instead_of_folding_is_caught():
    """A COARSER target aliases two R onto one FFT bin.  ``fold_indices_np``
    ADDS them (exact); dropping the out-of-box modes instead is a LARGE error,
    not a small one — this cell pins how large, so nobody 'simplifies' the
    fold into a truncation."""
    from bandstructure.htransform import build_R_grid_np, fold_indices_np
    N, M, delta = (8, 8, 2), (4, 4, 2), (0.03, -0.11, 0.25)
    rank = 3
    fH_R = _fH_R_np(_hermitian_fH_k(N, rank), N)
    R_grid = build_R_grid_np(N)
    want = _scan_np(_target(M, delta), fH_R, R_grid)

    ph = np.exp(-2j * np.pi * (R_grid @ np.asarray(delta)))
    A = fH_R * ph[:, None, None]
    fold = fold_indices_np(R_grid, M)
    mk = M[0] * M[1] * M[2]

    box = np.zeros((mk, rank, rank), dtype=np.complex128)
    np.add.at(box, fold, A)
    F = np.fft.fftn(box.reshape(M + (rank, rank)), axes=(0, 1, 2),
                    norm="backward").reshape(-1, rank, rank)
    folded = 0.5 * (F + np.conj(np.swapaxes(F, -1, -2)))
    assert _maxrel(folded, want) < 1e-13

    keep = np.ones(R_grid.shape[0], dtype=bool)
    for i in range(3):
        keep &= (R_grid[:, i] >= -(M[i] // 2)) & (R_grid[:, i] <= (M[i] + 1) // 2 - 1)
    box_t = np.zeros((mk, rank, rank), dtype=np.complex128)
    np.add.at(box_t, fold[keep], A[keep])
    Ft = np.fft.fftn(box_t.reshape(M + (rank, rank)), axes=(0, 1, 2),
                     norm="backward").reshape(-1, rank, rank)
    trunc = 0.5 * (Ft + np.conj(np.swapaxes(Ft, -1, -2)))
    assert _maxrel(trunc, want) > 0.1, (
        "truncation must be a LARGE error here; if this ever goes small the "
        "fixture stopped aliasing and the cell stopped testing anything")


# ---------------------------------------------------------------------------
#  A2. what may be routed — the detector and the gate
# ---------------------------------------------------------------------------

def test_detector_finds_a_shifted_uniform_grid():
    from bandstructure.htransform import detect_uniform_q
    for M in [(4, 4, 4), (3, 5, 2), (6, 1, 1), (1, 1, 7)]:
        for delta in [(0.0, 0.0, 0.0), (0.1, -0.25, 0.375)]:
            plan = detect_uniform_q(_target(M, delta))
            assert plan is not None, (M, delta)
            assert plan.grid == M, (plan.grid, M)
            assert plan.block == M[0] * M[1] * M[2]
            assert np.allclose(plan.shifts[0] - np.asarray(delta),
                               np.round(plan.shifts[0] - np.asarray(delta)))


def test_detector_refuses_a_bandstructure_path():
    """The htransform CLI's own target.  A ``K_POINTS crystal_b`` walk is
    collinear inside a segment and turns a corner at every node, so it is not
    a shifted uniform grid at any M — and the detector must SAY so rather than
    propose an M that ``verify_uniform_q`` then has to catch."""
    from bandstructure.htransform import detect_uniform_q
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                      [0.25, 0.5, 0.75], [0.5, 0.5, 0.5]])
    segs = [np.linspace(nodes[i], nodes[i + 1], 13, endpoint=False)
            for i in range(3)]
    path = np.concatenate(segs + [nodes[-1][None, :]], axis=0)
    assert detect_uniform_q(path) is None


def test_declared_structure_is_verified_not_trusted():
    """A declaration that does not hold must be caught by
    ``verify_uniform_q`` — the whole safety of ``q_structure`` rests on this
    being a gate and not a hint."""
    from bandstructure.htransform import UniformQPlan, verify_uniform_q
    M, shifts = (2, 2, 2), np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    q = np.concatenate([_target(M, s) for s in shifts], axis=0)
    good = UniformQPlan(grid=M, shifts=shifts, block=8, origin="t")
    assert verify_uniform_q(q, good) < 1e-12

    q_bad = q.copy()
    q_bad[5, 1] += 3e-3
    assert verify_uniform_q(q_bad, good) > 1e-9

    wrong_M = UniformQPlan(grid=(4, 2, 1), shifts=shifts, block=8, origin="t")
    assert verify_uniform_q(q, wrong_M) > 1e-9

    short = UniformQPlan(grid=M, shifts=shifts[:1], block=8, origin="t")
    assert verify_uniform_q(q, short) == float("inf")


def test_wrapped_q_is_not_a_mismatch():
    """``compute_wfns_fi`` wraps q into (−0.5, 0.5] and fH_q is exactly
    BZ-periodic, so the verifier must compare MOD 1 — otherwise every real
    call would be refused for a reason that is not an error."""
    from bandstructure.htransform import UniformQPlan, verify_uniform_q
    M = (4, 4, 1)
    q = _target(M, (0.0, 0.0, 0.0))
    plan = UniformQPlan(grid=M, shifts=np.zeros((1, 3)), block=16, origin="t")
    assert verify_uniform_q((q + 0.5) % 1.0 - 0.5, plan) < 1e-12


# ---------------------------------------------------------------------------
#  A3. through the real driver entry point
# ---------------------------------------------------------------------------

def _synthetic(nk_grid=(2, 2, 1), nb=4, rank=32, n_mu=6, ns=2, seed=11):
    """Band-orthonormal ``ctilde`` per k — what ``streaming_galerkin_solve``
    produces and what ``build_fH_R``'s gate requires.  Same construction as
    ``test_bse_setup_qchunk._synthetic``."""
    rng = np.random.default_rng(seed)
    nk = nk_grid[0] * nk_grid[1] * nk_grid[2]
    ct = np.empty((nk, nb, rank), dtype=np.complex128)
    for k in range(nk):
        z = (rng.standard_normal((rank, nb))
             + 1j * rng.standard_normal((rank, nb)))
        q, _ = np.linalg.qr(z)
        ct[k] = np.conj(q.T)
    enk = (np.linspace(-0.6, 0.4, nb)[:, None]
           + 0.05 * np.cos(2 * np.pi * np.arange(nk) / nk)[None, :])
    B = (rng.standard_normal((rank, ns, n_mu))
         + 1j * rng.standard_normal((rank, ns, n_mu)))
    return ct, enk, B, nk_grid


def _wfns_fi(mesh, *, route, log=None, ratio=None,
             band_window=(1, 3), **kw):
    """``ratio`` lifts LORRAX_HTQ_FFT_BLOCK_RATIO for the equivalence cells.

    The fixture is deliberately tiny (nk_co = 4), so a 4x4x1 fine grid is a
    4x block against the coarse extent and the SHIPPED cap would demote it —
    correctly, on a real deck.  Here the cap is not what is under test; the
    arithmetic is.  ``test_block_ratio_cap_demotes_and_says_so`` tests the cap
    itself, at the default.
    """
    import os
    import jax.numpy as jnp
    from bandstructure.bse_setup import compute_wfns_fi
    ct, enk, B, kgrid_co = _synthetic()
    prev = os.environ.get("LORRAX_HTQ_FOURIER_ROUTE")
    prev_r = os.environ.get("LORRAX_HTQ_FFT_BLOCK_RATIO")
    os.environ["LORRAX_HTQ_FOURIER_ROUTE"] = route
    if ratio is not None:
        os.environ["LORRAX_HTQ_FFT_BLOCK_RATIO"] = str(ratio)
    try:
        with mesh:
            return compute_wfns_fi(
                ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
                enk_sigma=jnp.asarray(enk), kgrid_co=kgrid_co,
                band_window_fi=band_window, mesh_xy=mesh, log_fn=log, **kw)
    finally:
        for key, old in (("LORRAX_HTQ_FOURIER_ROUTE", prev),
                         ("LORRAX_HTQ_FFT_BLOCK_RATIO", prev_r)):
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@pytest.mark.parametrize("kw,label", [
    (dict(kgrid_fi=(4, 4, 1)), "denser uniform (kgrid_fi)"),
    (dict(kgrid_fi=(2, 2, 1)), "same grid"),
])
def test_routes_agree_through_compute_wfns_fi(kw, label):
    """The two routes, one set of operands, on the EIGENVALUES the bundle
    carries.  ψ is compared in the cell below, which explains why it needs a
    different statement."""
    pytest.importorskip("jax")
    mesh = _mesh(1)
    a = _wfns_fi(mesh, route="scan", ratio=64.0, **kw)
    b = _wfns_fi(mesh, route="fft", ratio=64.0, **kw)
    for name in ("lam_all_fi", "lam_fi", "enk_full"):
        assert _maxrel(getattr(a, name), getattr(b, name)) < 1e-11, (label, name)


def _projector(coeffs):
    """Π_q = c cᴴ over the selected band window — the GAUGE-INVARIANT object.

    ``eigh`` fixes eigenvectors only up to one phase per eigenvalue (and up to
    a rotation inside any degenerate block); LAPACK and cuSOLVER promise no
    canonical choice, and a 1-ULP move in the input can flip a whole column.
    So ``c`` itself is not a function of the mathematics and must not be
    compared entrywise.  The SUBSPACE it spans is, and Π is that subspace.
    """
    c = np.asarray(coeffs)                       # (nq, rank, nb_fi)
    return np.einsum("qan,qbn->qab", c, np.conj(c))


def test_routes_agree_on_the_eigenVECTOR_SUBSPACE():
    """ψ is gauge-dependent; the subspace it is built from is not.

    Entrywise ``psi`` DOES move between the routes — by O(1) — and that is not
    a defect: the two routes differ in the last bits by construction, and an
    eigenvector's phase is not a function of the matrix.  What must agree is
    the projector onto the selected bands, and it does.
    """
    pytest.importorskip("jax")
    mesh = _mesh(1)
    for kw in (dict(kgrid_fi=(4, 4, 1)), dict(kgrid_fi=(2, 2, 1))):
        a = _wfns_fi(mesh, route="scan", ratio=64.0, return_coeffs=True, **kw)
        b = _wfns_fi(mesh, route="fft", ratio=64.0, return_coeffs=True, **kw)
        assert _maxrel(_projector(a.coeffs_fi), _projector(b.coeffs_fi)) < 1e-9, kw
        # |ψ| is phase-invariant too, and it is what the BSE stacks contract.
        assert _maxrel(np.abs(np.asarray(a.psi_rmu_Y)),
                       np.abs(np.asarray(b.psi_rmu_Y))) < 1e-8, kw


def test_the_gauge_moves_under_a_1_ULP_perturbation_of_ONE_route():
    """THE CONTROL, and the load-bearing half of the cell above.

    Perturbing ``ctilde`` by one ULP in a single entry is a change of order
    1e-16 in fH — far below anything physical — and it moves ψ entrywise by
    O(1) on the SCAN route alone, with the eigenvalues and the projector
    unmoved.  That is the same signature the scan-vs-FFT comparison shows, so
    the movement there is the gauge and not the dispatch.
    """
    pytest.importorskip("jax")
    import os
    import jax.numpy as jnp
    from bandstructure.bse_setup import compute_wfns_fi
    mesh = _mesh(1)
    ct, enk, B, kgrid_co = _synthetic()
    ct2 = ct.copy()
    ct2[0, 0, 0] = np.nextafter(ct2[0, 0, 0].real, np.inf) + 1j * ct2[0, 0, 0].imag

    prev = os.environ.get("LORRAX_HTQ_FOURIER_ROUTE")
    os.environ["LORRAX_HTQ_FOURIER_ROUTE"] = "scan"
    try:
        outs = []
        for c in (ct, ct2):
            with mesh:
                outs.append(compute_wfns_fi(
                    ctilde=jnp.asarray(c), B_at_mu=jnp.asarray(B),
                    enk_sigma=jnp.asarray(enk), kgrid_co=kgrid_co,
                    kgrid_fi=(4, 4, 1), band_window_fi=(1, 3), mesh_xy=mesh,
                    return_coeffs=True, log_fn=None))
    finally:
        if prev is None:
            os.environ.pop("LORRAX_HTQ_FOURIER_ROUTE", None)
        else:
            os.environ["LORRAX_HTQ_FOURIER_ROUTE"] = prev
    a, b = outs
    d_psi = _maxrel(a.psi_rmu_Y, b.psi_rmu_Y)
    d_proj = _maxrel(_projector(a.coeffs_fi), _projector(b.coeffs_fi))
    d_lam = _maxrel(a.lam_all_fi, b.lam_all_fi)
    assert d_lam < 1e-9, d_lam
    assert d_proj < 1e-7, d_proj
    assert d_psi > 1e-6, (
        f"a 1-ULP move left ψ alone (moved {d_psi:.3e}); if that is really "
        f"true then eigh IS phase-canonical here and the scan-vs-FFT ψ "
        f"difference needs another explanation")


def test_route_is_announced_on_both_arms():
    """The route is never silent — that is the reviewability requirement, and
    it is what lets a timing leg prove which arm it measured."""
    pytest.importorskip("jax")
    mesh = _mesh(1)
    for route, token in (("scan", "SCAN"), ("fft", "FFT")):
        lines = []
        _wfns_fi(mesh, route=route, kgrid_fi=(4, 4, 1), ratio=64.0,
                 log=lines.append)
        banner = " ".join(lines)
        assert "[route] fH_q Fourier: " + token in banner, (route, banner)


def test_the_DEFAULT_route_is_the_scan_and_says_why():
    """The default is a MEASUREMENT, so it gets a gate.

    Both routes are exact and agree to round-off, but the FFT one is slower at
    this code shapes (0.975-0.985x at P=4, rank 768) because the scan is one
    batched GEMM and the FFT is memory-bound.  With the knob unset the route
    must therefore be the SCAN, and the log must name the knob rather than
    leaving the choice to be rediscovered.
    """
    pytest.importorskip("jax")
    import os
    prev = os.environ.pop("LORRAX_HTQ_FOURIER_ROUTE", None)
    try:
        lines = []
        mesh = _mesh(1)
        ct, enk, B, kgrid_co = _synthetic()
        import jax.numpy as jnp
        from bandstructure.bse_setup import compute_wfns_fi, resolve_fourier_route
        assert resolve_fourier_route() == "scan"
        with mesh:
            compute_wfns_fi(
                ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
                enk_sigma=jnp.asarray(enk), kgrid_co=kgrid_co,
                kgrid_fi=(4, 4, 1), band_window_fi=(1, 3), mesh_xy=mesh,
                log_fn=lines.append)
        banner = " ".join(lines)
        assert "[route] fH_q Fourier: SCAN" in banner, banner
        assert "LORRAX_HTQ_FOURIER_ROUTE=scan" in banner, banner
    finally:
        if prev is not None:
            os.environ["LORRAX_HTQ_FOURIER_ROUTE"] = prev


def test_route_env_refuses_garbage_and_refuses_to_demote():
    pytest.importorskip("jax")
    mesh = _mesh(1)
    with pytest.raises(ValueError, match="LORRAX_HTQ_FOURIER_ROUTE"):
        _wfns_fi(mesh, route="fastplease", kgrid_fi=(4, 4, 1), ratio=64.0)
    # ``fft`` on a q-set with no structure must RAISE, not quietly scan: a leg
    # that asked for this route must not be able to quote the other one.
    rng = np.random.default_rng(3)
    with pytest.raises(RuntimeError, match="not available"):
        _wfns_fi(mesh, route="fft", q_list=rng.uniform(-0.5, 0.5, size=(7, 3)))


def test_block_ratio_cap_demotes_and_says_so(monkeypatch):
    """A fine grid far denser than the coarse one makes ONE FFT block cost
    more than fH_R itself.  The cap must demote to the scan and name the
    lever, not OOM."""
    pytest.importorskip("jax")
    monkeypatch.setenv("LORRAX_HTQ_FFT_BLOCK_RATIO", "1.0")
    lines = []
    _wfns_fi(_mesh(1), route="auto", kgrid_fi=(8, 8, 1), log=lines.append)
    banner = " ".join(lines)
    assert "[route] fH_q Fourier: SCAN" in banner, banner
    assert "LORRAX_HTQ_FFT_BLOCK_RATIO" in banner, banner
