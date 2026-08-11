"""Gates for the in-XLA eigensolver and the fused fit path.

CPU only, ``JAX_PLATFORMS=cpu``; nothing here touches a cluster.

WHAT IS BEING CERTIFIED, AND WHY IT IS NOT A BIT-IDENTITY CLAIM.
``gw.mpa.small_eig`` replaces the vendor eigensolver inside the fit with
a fixed-count Hessenberg-QR iteration that stays in XLA.  A different
root-finder returns different roots in the last digits -- that is not a
defect to be tuned away, it is what changing the algebra means -- so the
equivalence these cells assert is the campaign's own norm, W REBUILT AT
THE SAMPLES, and not pole-for-pole identity.  The things that ARE
asserted bit-exactly are the two that must be: that the fused path
reproduces itself run to run on the same device, and that jit and eager
agree for a FIXED backend.

The suite:

* agreement with LAPACK on the real Loewner geometry, across the pole
  schedule and past it;
* the RED TWIN for the sweep count -- too few sweeps must visibly fail,
  because a fixed iteration count that nobody has seen fail is a
  constant somebody guessed;
* jit/vmap cleanliness, bit-exact, which is the property the whole
  module was designed around;
* the fused fit against the unfused one: W-rebuild norm, guard fire
  masks element by element, and run-to-run reproducibility;
* the two refusals.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from gw.mpa import diagnostics, pade_fit, sampling, small_eig  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _grid(n_p=8, omega_m=4.0, **kw):
    return np.asarray(sampling.double_parallel_grid(n_p, omega_m, **kw),
                      dtype=np.complex128)


def _planted_tile(n_elements, n_p, *, n_true=None, seed=0):
    """A tile of synthesized ``W_c`` rows from planted poles.

    ``n_true`` below ``n_p`` is the honest production situation -- the
    schedule fixes the pole count and the data supports whatever it
    supports -- and it is what makes the prune guards fire, which is the
    case the fused path has to reproduce.
    """

    rng = np.random.default_rng(seed)
    n_true = n_p if n_true is None else int(n_true)
    z = _grid(n_p)
    a = rng.uniform(0.3, 2.8, size=(n_elements, n_true))
    g = rng.uniform(0.02, 0.25, size=(n_elements, n_true))
    Om = a - 1j * g
    B = 0.1 * (rng.normal(size=(n_elements, n_true))
               + 1j * rng.normal(size=(n_elements, n_true)))
    denom = z[None, :, None] ** 2 - Om[:, None, :] ** 2
    W = np.sum(2.0 * Om[:, None, :] * B[:, None, :] / denom, axis=2)
    return W.astype(np.complex128), z


def _loewner_matrices(W, z, n_p):
    """The matrix whose eigenvalues the Loewner solve actually takes."""

    x, x_max = pade_fit._x_normalisation(jnp.asarray(z))
    x_hat = x / x_max.astype(jnp.complex128)

    def build(w):
        L, sL = pade_fit._loewner_pencil(w, x_hat, n_p)
        u, s, vh = jnp.linalg.svd(L, full_matrices=False)
        s_inv = jnp.where(s > 1e-13 * s[0],
                          1.0 / jnp.where(s > 0, s, 1.0), 0.0)
        L_pinv = vh.conj().T @ (s_inv.astype(L.dtype)[:, None] * u.conj().T)
        return L_pinv @ sL

    return jax.jit(jax.vmap(build))(jnp.asarray(W))


def _sorted(v):
    v = np.asarray(v)
    order = np.lexsort((v.imag, v.real), axis=-1)
    return np.take_along_axis(v, order, axis=-1)


def _rel_eig_error(got, ref):
    ref = _sorted(ref)
    scale = np.max(np.abs(ref), axis=-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return np.max(np.abs(_sorted(got) - ref) / scale)


def _rebuild_at_samples(Omega, B, z, valid):
    """``W`` as the fitted pole set reproduces it on the sample grid.

    ``eval_mpa_model`` evaluates ONE element's pole set against a vector
    of frequencies, so a block is a vmap of it -- the same relationship
    the fit kernel has to its batched form.
    """

    zz = jnp.asarray(z)
    per_element = jax.vmap(
        lambda om, b, v: pade_fit.eval_mpa_model(om, b, zz, valid=v))
    return np.asarray(per_element(
        jnp.asarray(Omega), jnp.asarray(B), jnp.asarray(valid)))


def _w_rebuild_norm(a, b, W):
    """THE CAMPAIGN'S EQUIVALENCE NORM, stated once and used everywhere.

    Relative Frobenius distance between two rebuilds of the same block,
    normalised by the sampled ``W`` those rebuilds are of -- so it reads
    as "the two pole sets describe the same screening to this many
    digits", which is the claim, rather than as "these two eigenvalue
    lists differ", which is not.
    """

    return float(np.linalg.norm(a - b) / max(np.linalg.norm(W), 1e-300))


# ---------------------------------------------------------------------------
# the eigensolver on its own
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_p", [2, 4, 8, 10, 12])
def test_small_eig_matches_lapack_on_the_production_geometry(n_p):
    """The matrices this will actually see: MORE POLES THAN THE DATA HAS.

    That is the production situation -- the schedule fixes ``n_p`` and
    the screening supports whatever it supports, which is why the null
    guard exists at all -- and it is the well-conditioned one for the
    eigensolver: a truncated pseudo-inverse hands the iteration a large
    exact null space, and an eigenvalue at zero needs no sweeps.  The
    tolerance here is the level at which agreeing with LAPACK stops
    being a distinction, since LAPACK and the vendor GPU routine
    themselves differ by ~2e-12 on these matrices.
    """

    W, z = _planted_tile(256, n_p, n_true=min(3, n_p), seed=5)
    X = _loewner_matrices(W, z, n_p)
    ref = np.linalg.eigvals(np.asarray(X))
    got = jax.jit(jax.vmap(small_eig.eigvals))(X)
    err = _rel_eig_error(got, ref)
    assert err < 1e-9, f"n_p={n_p}: worst relative eigenvalue error {err:.3e}"
    print(f"[small_eig n_p={n_p}, rank-deficient] worst rel eig err {err:.3e}")


@pytest.mark.parametrize("n_p", [8, 10, 12])
def test_agreement_with_lapack_is_floored_by_the_pencils_conditioning(n_p):
    """On FULL-RANK data the two eigensolvers stop agreeing at ~1e-9, and
    that is a property of the problem rather than of either of them.

    This cell exists because the number looks alarming next to the
    1e-13 above and would otherwise read as a defect in the iteration.
    It is not: the Loewner matrix on full-rank data runs ``cond`` of
    1e5-1e8, so its eigenvalues carry a sensitivity of about
    ``eps * cond``, and no algorithm delivers digits the problem does
    not have.  THE ASSERTION THAT SHOWS IT IS A FLOOR is the second one
    -- doubling the sweep count does not lower the disagreement, which
    non-convergence always would.

    It is also the reason the fused path's equivalence gate is stated in
    the W-rebuild norm: the poles of an ill-conditioned pencil are not
    the reproducible object, and the screening they rebuild is.
    """

    W, z = _planted_tile(256, n_p, seed=5)
    X = _loewner_matrices(W, z, n_p)
    ref = np.linalg.eigvals(np.asarray(X))

    shipped = jax.jit(jax.vmap(small_eig.eigvals))(X)
    doubled = jax.jit(jax.vmap(lambda A: small_eig.eigvals(
        A, n_sweeps=2 * small_eig.DEFAULT_SWEEPS)))(X)
    err_shipped = _rel_eig_error(shipped, ref)
    err_doubled = _rel_eig_error(doubled, ref)

    assert err_shipped < 1e-6, (
        f"n_p={n_p}: {err_shipped:.3e} is too large to be conditioning")
    assert err_doubled > 0.5 * err_shipped, (
        f"n_p={n_p}: doubling the sweeps moved the disagreement "
        f"{err_shipped:.3e} -> {err_doubled:.3e}, so it was not a floor "
        "and DEFAULT_SWEEPS is too low")
    print(f"[conditioning floor n_p={n_p}] sweeps={small_eig.DEFAULT_SWEEPS} "
          f"{err_shipped:.3e}, doubled {err_doubled:.3e}")


def test_red_twin_too_few_sweeps_does_not_converge():
    """RED TWIN for the iteration count.

    A fixed sweep count nobody has watched fail is a number somebody
    guessed.  Two sweeps must leave a visibly unconverged eigenvalue --
    if it did not, the shipped eight would be indefensible padding and
    this cell would be the one to say so.
    """

    n_p = 10
    W, z = _planted_tile(256, n_p, seed=7)
    X = _loewner_matrices(W, z, n_p)
    ref = np.linalg.eigvals(np.asarray(X))

    starved = jax.jit(jax.vmap(
        lambda A: small_eig.eigvals(A, n_sweeps=2)))(X)
    shipped = jax.jit(jax.vmap(
        lambda A: small_eig.eigvals(A, n_sweeps=small_eig.DEFAULT_SWEEPS)))(X)

    err_starved = _rel_eig_error(starved, ref)
    err_shipped = _rel_eig_error(shipped, ref)
    assert err_starved > 1e-3, (
        "two sweeps converged, so the shipped sweep count is not "
        f"buying anything: starved error {err_starved:.3e}")
    # 1e-6 and not 1e-9: this cell uses full-rank data, whose eigenvalue
    # conditioning floors the agreement at ~1e-9 no matter how many
    # sweeps are spent.  The claim being made is that the shipped count
    # REACHES that floor and two sweeps is nowhere near it.
    assert err_shipped < 1e-6
    assert err_starved > 1e3 * err_shipped
    print(f"[sweep red twin] starved {err_starved:.3e} -> "
          f"shipped {err_shipped:.3e}")


def test_small_eig_is_jit_and_vmap_clean():
    """Bit-identical under jit and under vmap.

    This is the property the module is built around: static shapes, a
    fixed iteration count and no data-dependent branch mean the compiler
    has no freedom to reassociate anything the eager path did not.
    """

    n_p = 8
    W, z = _planted_tile(32, n_p, seed=9)
    X = _loewner_matrices(W, z, n_p)

    eager = np.asarray(jax.vmap(small_eig.eigvals)(X))
    jitted = np.asarray(jax.jit(jax.vmap(small_eig.eigvals))(X))
    assert np.array_equal(eager, jitted), "jit moved the eigenvalues"

    loop = np.stack([np.asarray(small_eig.eigvals(X[i]))
                     for i in range(X.shape[0])])
    assert np.array_equal(eager, loop), "vmap is not the loop"


def test_gate_small_eig_square():
    with pytest.raises(ValueError, match="GATE small_eig_square"):
        small_eig.eigvals(jnp.zeros((3, 4), dtype=jnp.complex128))


def test_gate_eig_mode_known():
    """A typo must be refused, not read as the default.

    Same reasoning as ``GATE fit_solve_mode_known``: the two backends do
    not agree bit for bit, so silently substituting one is how a store
    acquires poles it cannot account for.
    """

    W, z = _planted_tile(1, 8, seed=1)
    with pytest.raises(ValueError, match="GATE eig_mode_known"):
        pade_fit.fit_mpa_poles(W[0], z, 8, eig="jacobi")


# ---------------------------------------------------------------------------
# the fused fit against the unfused one
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_p,n_true,tol", [
    # Rank-deficient -- the production situation, and the one where the
    # two root-finders are for practical purposes the same routine.
    (10, 3, 1e-12),
    (8, 2, 1e-12),
    # Full-rank -- where the pencil's own conditioning separates them.
    (8, 8, 1e-6),
    (10, 10, 1e-6),
    (12, 12, 1e-6),
])
def test_fused_fit_rebuilds_W_at_the_samples(n_p, n_true, tol):
    """GATE (a) at unit scale: the two paths describe the same screening.

    THE TOLERANCE IS NOT ONE NUMBER AND THAT IS THE FINDING.  Where the
    solve is rank-deficient -- more poles fitted than the data supports,
    which is production -- the two paths rebuild ``W`` to 1e-14, because
    the eigenvalues they disagree about are the null space's and the null
    guard drops those before a residue is ever fitted to them.  Where the
    pencil is full rank its eigenvalues carry ``eps * cond ~ 1e-9`` of
    intrinsic sensitivity, the two root-finders land at opposite ends of
    it, and the rebuild separates by ~1e-7.

    What does NOT change between the two arms is how well either fits the
    data: the fidelity numbers below agree to four significant figures.
    The choice of root-finder is not what limits this fit.
    """

    W, z = _planted_tile(512, n_p, n_true=n_true, seed=13)
    Om_u, B_u, d_u = pade_fit.fit_mpa_poles_batched(W, z, n_p)
    Om_f, B_f, d_f = jax.jit(
        lambda t: pade_fit.fit_mpa_poles_batched(
            t, z, n_p, eig="jax_qr"))(jnp.asarray(W))

    reb_u = _rebuild_at_samples(Om_u, B_u, z, d_u["valid"])
    reb_f = _rebuild_at_samples(Om_f, B_f, z, d_f["valid"])

    between = _w_rebuild_norm(reb_f, reb_u, W)
    fid_u = _w_rebuild_norm(reb_u, W, W)
    fid_f = _w_rebuild_norm(reb_f, W, W)

    print(f"[W-rebuild n_p={n_p} n_true={n_true}] fused-vs-unfused "
          f"{between:.3e}   fidelity unfused {fid_u:.3e} fused {fid_f:.3e}")
    assert between < tol
    # Neither arm may fit the samples appreciably worse than the other.
    # The floor keeps a comparison of two machine-precision numbers from
    # being read as a ratio.
    assert fid_f <= max(2.0 * fid_u, 1e-12)


_PRUNE_GUARDS = ("n_pruned_coincident", "n_pruned_out_of_range",
                 "n_pruned_null")
_REPAIR_GUARDS = ("n_reflected", "n_time_order_flipped")


@pytest.mark.parametrize("n_p,n_true", [(10, 3), (8, 2), (8, 8), (10, 10)])
def test_which_poles_survive_the_guards_is_identical(n_p, n_true):
    """GATE (c), stated on the quantity that is actually well defined.

    THE SURVIVING-POLE MASK IS BIT-IDENTICAL in every configuration
    measured, and so are all three PRUNING guards, element by element.
    That is the guard outcome anything downstream can observe: which
    poles reach the residue fit and the store.
    """

    W, z = _planted_tile(512, n_p, n_true=n_true, seed=17)
    _, _, d_u = pade_fit.fit_mpa_poles_batched(W, z, n_p)
    _, _, d_f = jax.jit(
        lambda t: pade_fit.fit_mpa_poles_batched(
            t, z, n_p, eig="jax_qr"))(jnp.asarray(W))

    assert np.array_equal(np.asarray(d_u["valid"]),
                          np.asarray(d_f["valid"])), "survivors differ"
    for k in _PRUNE_GUARDS:
        u, f = np.asarray(d_u[k]), np.asarray(d_f[k])
        assert np.array_equal(u, f), (
            f"{k}: {int(np.sum(u != f))} of {u.size} elements disagree")


def test_red_twin_the_repair_guards_cannot_be_gated_on_null_poles():
    """THE HONEST LIMIT OF GATE (c), asserted rather than glossed.

    ``reflection`` fires on ``Re b < 0`` and ``time_order`` on
    ``Im b > 0``.  Both are SIGN TESTS, and on a rank-deficient solve the
    poles they are applied to include the ones the truncated
    pseudo-inverse returned at numerical zero -- where the sign is a coin
    flip between any two root-finders, and would be between two runs of
    the same one on different hardware.  So their per-element fire counts
    do NOT agree across a change of eigensolver, and a gate that demanded
    they did would be a gate on rounding noise.

    This cell pins both halves of that: the repair counts DISAGREE on
    rank-deficient data, and it does not matter, because the poles in
    question are pruned by the null guard and the rebuilt ``W`` agrees to
    1e-14 anyway.  On full-rank data, where there is no null space, the
    same counts agree exactly.
    """

    # Rank-deficient: the repair guards disagree, the answer does not.
    W, z = _planted_tile(512, 10, n_true=3, seed=17)
    Ou, Bu, du = pade_fit.fit_mpa_poles_batched(W, z, 10)
    Of, Bf, df = jax.jit(lambda t: pade_fit.fit_mpa_poles_batched(
        t, z, 10, eig="jax_qr"))(jnp.asarray(W))
    disagree = max(
        int(np.sum(np.asarray(du[k]) != np.asarray(df[k])))
        for k in _REPAIR_GUARDS)
    assert disagree > 0, (
        "the repair guards agreed on rank-deficient data, so this cell "
        "is no longer exhibiting the thing it documents")
    rebuild = _w_rebuild_norm(
        _rebuild_at_samples(Of, Bf, z, df["valid"]),
        _rebuild_at_samples(Ou, Bu, z, du["valid"]), W)
    assert rebuild < 1e-12, (
        f"the repair guards disagreed AND the screening moved ({rebuild:.3e})"
        " -- that would be a real defect, not a sign test on a null pole")
    print(f"[repair guards] rank-deficient: {disagree} elements disagree, "
          f"W-rebuild {rebuild:.3e}")

    # Full-rank: no null space, so they agree exactly.
    W, z = _planted_tile(512, 10, n_true=10, seed=17)
    _, _, du = pade_fit.fit_mpa_poles_batched(W, z, 10)
    _, _, df = jax.jit(lambda t: pade_fit.fit_mpa_poles_batched(
        t, z, 10, eig="jax_qr"))(jnp.asarray(W))
    for k in _REPAIR_GUARDS:
        assert np.array_equal(np.asarray(du[k]), np.asarray(df[k])), k


def test_the_fused_path_reproduces_itself_run_to_run():
    """GATE (b): bit-exact across two runs of the same binary/device.

    The equivalence gate against the unfused path is a norm, not an
    identity -- which makes THIS the cell that keeps the fused path
    honest.  A result that is merely close to the other path AND
    irreproducible against itself is not a result.
    """

    n_p = 10
    W, z = _planted_tile(256, n_p, n_true=4, seed=19)
    fn = jax.jit(lambda t: pade_fit.fit_mpa_poles_batched(
        t, z, n_p, eig="jax_qr"))

    Om1, B1, d1 = fn(jnp.asarray(W))
    Om2, B2, d2 = fn(jnp.asarray(W))
    assert np.array_equal(np.asarray(Om1), np.asarray(Om2))
    assert np.array_equal(np.asarray(B1), np.asarray(B2))
    assert np.array_equal(np.asarray(d1["valid"]), np.asarray(d2["valid"]))


def test_every_fit_and_diagnostics_entry_point_takes_one_mode_dict():
    """A caller holds ONE dict describing how the fit is to be done and
    forwards it to the fit AND to the fit's diagnostics.

    That is how every harness in this campaign is written, and it is why
    a keyword present on ``fit_mpa_poles`` and absent from
    ``solve_conditioning`` is not a cosmetic asymmetry: it raises
    TypeError on the second call, and the workaround -- filtering the
    dict at the call site -- is how a probe ends up measuring the
    default while believing it measured the mode it asked for.

    The cell walks every public entry point that performs a fit, so the
    next keyword to be added has one place that fails if it is added in
    only half of them.
    """

    mode = dict(rcond=1.0e-13, solve="loewner", affine=True, eig="jax_qr")
    n_p = 8
    W, z = _planted_tile(8, n_p, seed=29)
    err = 1.0e-9 * np.ones(2 * n_p, dtype=np.complex128)

    pade_fit.fit_mpa_poles(W[0], z, n_p, **mode)
    pade_fit.fit_mpa_poles_batched(W, z, n_p, **mode)
    diagnostics.solve_conditioning(W[0], z, n_p, **mode)
    diagnostics.holdout_residual(W[0], z, n_p, **mode)
    diagnostics.perturbation_refit(W[0], z, n_p, err, **mode)
    diagnostics.diagnostics_batched(
        diagnostics.solve_conditioning, W, z, n_p, **mode)


def test_the_conditioning_door_reports_the_backend_it_was_asked_for():
    """PLUMBED, not swallowed -- asserted rather than trusted.

    Four of this door's eight fields are downstream of the root-finding,
    so a version that accepted ``eig`` and ignored it would report one
    backend's residuals beside the other's poles.  The check is exact
    equality with the fit that WAS run in that mode, in both modes; the
    second assertion confirms the first can actually detect a swallow,
    by showing the two modes do not return the same number here.
    """

    n_p = 8
    W, z = _planted_tile(8, n_p, seed=31)
    seen = {}
    for eig in pade_fit.EIG_MODES:
        _, _, diag = pade_fit.fit_mpa_poles(W[0], z, n_p, eig=eig)
        door = diagnostics.solve_conditioning(W[0], z, n_p, eig=eig)
        for door_key, diag_key in diagnostics._CONDITIONING_FROM_DIAG.items():
            assert np.array_equal(
                np.asarray(door[door_key]), np.asarray(diag[diag_key])), (
                f"eig={eig}: {door_key} is not the fit's {diag_key}")
        seen[eig] = float(np.asarray(door["forward_residual"]))

    assert seen["lapack"] != seen["jax_qr"], (
        "the two backends returned an identical forward residual on this "
        "element, so the equality above could not have caught a swallowed "
        "eig= and this cell is not testing what it claims")


def test_fused_and_unfused_are_bit_identical_for_a_fixed_backend():
    """Fusion alone moves nothing; only the root-finder does.

    Separating the two is what makes the W-rebuild tolerance above
    attributable: if jit by itself already perturbed the answer, the
    norm could not be read as a statement about the eigensolver.
    """

    n_p = 8
    W, z = _planted_tile(64, n_p, seed=23)
    for eig in pade_fit.EIG_MODES:
        eager = pade_fit.fit_mpa_poles_batched(W, z, n_p, eig=eig)
        fused = jax.jit(lambda t: pade_fit.fit_mpa_poles_batched(
            t, z, n_p, eig=eig))(jnp.asarray(W))
        assert np.array_equal(np.asarray(eager[0]), np.asarray(fused[0])), eig
        assert np.array_equal(np.asarray(eager[1]), np.asarray(fused[1])), eig
