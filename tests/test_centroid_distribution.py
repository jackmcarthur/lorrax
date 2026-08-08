"""Gates for the centroid package's use of the device layer.

``common.collectives`` is THE device layer; ``centroid.distribution``
holds only centroid-specific policy.  Four things are pinned here, each
with a negative control in the same test so a vacuously-green assertion is
detectable:

1. :func:`common.collectives.single_device_mesh` holds a device THIS
   process can address, and ``centroid.distribution`` does not define a
   second mesh constructor.  The idiom both replaced,
   ``Mesh(jax.devices()[:1])``, holds process 0's device on every rank.
2. The sharded pivoted-Cholesky select reproduces the single-device
   reference on a 1x1 mesh, pivot for pivot.
3. The orbit-aware Lloyd step reduces to the plain one when the symmetry
   group is trivial — the invariant that keeps the two branches of
   ``kmeans_isdf.make_lloyd_loop`` honest.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────
# 1. Meshes
# ─────────────────────────────────────────────────────────────────────────

def test_process_local_mesh_is_addressable():
    """Every device in the process-local mesh belongs to THIS process.

    The mesh comes from ``common.collectives.single_device_mesh`` — the one
    device layer.  ``centroid.distribution`` deliberately does not define a
    second one; this test pins the property the centroid code depends on.
    """
    from src.common.collectives import single_device_mesh

    mesh = single_device_mesh()
    local = set(jax.local_devices())
    assert mesh.devices.size == 1
    assert set(mesh.devices.flat) <= local, (
        f"process_local_mesh holds {list(mesh.devices.flat)}, which this "
        f"process ({jax.process_index()}) cannot address; local devices are "
        f"{jax.local_devices()}")

    # Negative control: the SAME membership test applied to the idiom this
    # function replaced must be able to come out false.  At P=1 the global
    # and local device lists coincide, so the control can only be shown to
    # discriminate past P=1 -- say so rather than claim a pass.
    global_first = jax.devices()[:1]
    if jax.process_count() == 1:
        assert set(global_first) <= local          # P=1: they agree, as expected
        pytest.skip("P=1: jax.devices()[0] IS this process's device, so the "
                    "negative control cannot fire here. Run under P>=2 "
                    "(wk_REL/wave1_kmeans/kmmesh.sbatch) to exercise it.")
    else:
        assert not (set(global_first) <= local) or jax.process_index() == 0


def test_process_local_mesh_is_cached():
    """Same object every call — the to_box/to_rmu jit caches key on it."""
    from src.common.collectives import single_device_mesh
    assert single_device_mesh() is single_device_mesh()


def test_centroid_layer_defines_no_second_mesh_constructor():
    """``centroid.distribution`` must delegate, not reimplement.

    The orchestrator's 2026-07-30 ruling: ``common.collectives`` is THE
    device layer.  This test fails if the centroid layer ever grows its own
    ``single_device_mesh`` / ``process_local_mesh`` again — the exact name
    collision that made one spelling mean 'this rank's device' and the
    other 'global device 0, refuse at P>1'.
    """
    from src.centroid import distribution as dist
    for name in ("single_device_mesh", "process_local_mesh"):
        owner = getattr(getattr(dist, name, None), "__module__", None)
        assert owner in (None, "common.collectives", "src.common.collectives"), (
            f"centroid.distribution.{name} is defined in {owner}; it must be "
            f"common.collectives' or absent")


def test_build_mesh_single_device_is_2d():
    """A one-device run still gets a 2-D ('x','y') mesh, so the downstream
    pipeline has exactly one code path."""
    from src.centroid import distribution as dist
    if jax.process_count() > 1:
        pytest.skip("--no-shard is refused past one process")
    mesh = dist.build_mesh(10 ** 6, shard=False)
    assert tuple(mesh.axis_names) == ("x", "y")
    assert mesh.devices.shape == (1, 1)


# ─────────────────────────────────────────────────────────────────────────
# 2. Sharded pivoted-Cholesky select == single-device reference
# ─────────────────────────────────────────────────────────────────────────

def _psd_gram(M, rank, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((M, rank)) + 1j * rng.standard_normal((M, rank)))
    G = A @ A.conj().T
    return jnp.asarray(0.5 * (G + G.conj().T), dtype=jnp.complex128)


def test_sharded_select_matches_single_device_reference():
    from src.centroid import distribution as dist
    from src.centroid.pivoted_cholesky import (
        pivoted_cholesky_select, make_sharded_pivoted_cholesky_select)

    if jax.process_count() > 1:
        pytest.skip("reference comparison is a one-process gate")

    M, k_keep = 32, 12
    G = _psd_gram(M, rank=M, seed=3)
    mesh = dist.build_mesh(10 ** 6, shard=False)

    piv_ref, _, rank_ref, _, d_ref, _, _ = pivoted_cholesky_select(G, k_keep)
    step = make_sharded_pivoted_cholesky_select(
        mesh, M, k_keep, mesh_axis=dist.MESH_AXES)
    piv_sh, _, rank_sh, _, d_sh, _, _ = step(G)

    np.testing.assert_array_equal(np.asarray(piv_ref), np.asarray(piv_sh))
    assert int(rank_ref) == int(rank_sh)
    np.testing.assert_allclose(np.asarray(d_ref), np.asarray(d_sh),
                               rtol=1e-10, atol=0.0)

    # Negative control: the comparison must be able to FAIL.  A Gram with a
    # different pivot order has to produce different pivots; if it does not,
    # the assertion above proved nothing.
    piv_other, *_ = pivoted_cholesky_select(_psd_gram(M, rank=M, seed=4),
                                            k_keep)
    assert not np.array_equal(np.asarray(piv_ref), np.asarray(piv_other)), (
        "negative control did not fire: two different Grams gave the same "
        "pivot sequence, so the equality test above is vacuous")


# ─────────────────────────────────────────────────────────────────────────
# 2b. R1 — the kernel REFUSES instead of diverging
# ─────────────────────────────────────────────────────────────────────────
#
# Three cells, one per failure mode ``CENTROID_GEN_ASSESSMENT.md`` §4.1-§4.3
# measured, each written so its before-behaviour is stated and each with a
# constructible-FALSE twin.  The refusals themselves live in
# ``refuse_unless_select_certified``, which is a free function precisely so
# these can hand it real kernel output instead of standing up a WFN.


def _maxdiag(G):
    return float(np.real(np.asarray(jnp.diag(G))).max())


def _floor_of(G):
    return float(np.sqrt(np.finfo(np.float64).eps)) * _maxdiag(G)


def test_select_stops_at_the_rank_floor_instead_of_diverging():
    """Past the numerical rank: finite L, -1 sentinels, and a refusal.

    BEFORE (measured on this box, Gram of true rank 10 with k_keep=40):
    ``pivot_val`` clamped to ``eps`` so ``denom = sqrt(eps) ~ 1.5e-8`` and
    the column norms went 3.2e-07, ..., 5.2e+04, 2.2e+16, 1.1e+39, 1.4e+85,
    inf, inf, nan — first non-finite column j=22, twelve iterations past the
    true rank.  ``argmax`` over NaN then returned the first unpicked indices
    IN ARRAY ORDER (``0 1 3 4 5 7 8 9 10 11 13 ...``), which are real
    candidate indices, so nothing refused and ``trR_over_trG`` going NaN was
    the only visible trace.
    """
    from src.centroid.pivoted_cholesky import (pivoted_cholesky_select,
                                               refuse_unless_select_certified)

    M, k_keep, true_rank = 64, 40, 10
    G = _psd_gram(M, rank=true_rank, seed=1)
    piv, L, rank, _, d_taken, trR, psd = pivoted_cholesky_select(
        G, k_keep)

    assert int(rank) == true_rank, (
        f"stopping rule certified {int(rank)} directions on a Gram of true "
        f"rank {true_rank}")
    assert np.all(np.isfinite(np.asarray(L))), (
        "L went non-finite: the divisor clamp is not holding the recurrence")
    assert np.all(np.isfinite(np.asarray(trR))), "trR_over_trG went non-finite"
    piv_np = np.asarray(piv)
    assert np.all(piv_np[true_rank:] == -1), (
        f"pivots past the rank are {piv_np[true_rank:][:8]}, not the -1 "
        f"sentinel — the old index-ordered NaN pivots are back")
    assert len(np.unique(piv_np[:true_rank])) == true_rank

    # THE REFUSAL, and it names the rank-deficiency cause rather than the
    # exhaustion one: 64 points were available, only 10 were independent.
    with pytest.raises(RuntimeError, match="RANK-DEFICIENT"):
        refuse_unless_select_certified(
            piv, int(rank), psd, n_keep=k_keep, M=M, orbit_id=None,
            d0max=float(np.real(np.asarray(jnp.diag(G))).max()))

    # CONSTRUCTIBLE-FALSE TWIN: the same call on a full-rank Gram must
    # return, or the refusal above proves only that the function raises.
    Gok = _psd_gram(M, rank=M, seed=1)
    p2, _, r2, _, _, _, psd2 = pivoted_cholesky_select(Gok, k_keep)
    assert int(r2) == k_keep
    refuse_unless_select_certified(
        p2, int(r2), psd2, n_keep=k_keep, M=M, orbit_id=None,
        d0max=float(np.real(np.asarray(jnp.diag(Gok))).max()))


def test_select_refuses_when_the_orbits_run_out():
    """Orbit exhaustion stops; it does not repeat index 0.

    BEFORE (measured, M=96 in 12 orbits of 8, k_keep=20): once every orbit
    was inactive ``masked_d`` was uniformly -inf, ``pivot_val`` clamped to
    ``eps`` rather than NaN and ``argmax`` over a uniform array returned 0,
    so the pivot list came back ``[54 40 30 86 72 6 32 62 22 88 8 64 0 0 0 0
    0 0 0 0]`` — twelve genuine pivots then index 0 eight times, arithmetic
    finite throughout.  The unfold is ``np.isin(orbit_id, picked)``, a union,
    so the delivered set was 96 points against a nominal 20x8 = 160.
    """
    from src.centroid.pivoted_cholesky import (pivoted_cholesky_select,
                                               refuse_unless_select_certified)

    n_orb, orb_size, k_keep = 12, 8, 20
    oid = np.repeat(np.arange(n_orb, dtype=np.int32), orb_size)
    rng = np.random.default_rng(7)
    B = (rng.standard_normal((n_orb, 40))
         + 1j * rng.standard_normal((n_orb, 40)))
    A = B[oid]                       # orbit members share a row -> rank 12
    G = jnp.asarray(A @ A.conj().T, dtype=jnp.complex128)
    G = 0.5 * (G + G.conj().T)
    M = int(G.shape[0])

    piv, _, rank, _, _, _, psd = pivoted_cholesky_select(
        G, k_keep, jnp.asarray(oid))
    piv_np = np.asarray(piv)
    assert int(rank) == n_orb
    assert np.all(piv_np[n_orb:] == -1), (
        f"pivots past orbit exhaustion are {piv_np[n_orb:][:8]}, not the -1 "
        f"sentinel — the index-0 repetition is back")
    assert len(np.unique(piv_np[:n_orb])) == n_orb, (
        "a real orbit was picked twice")

    with pytest.raises(RuntimeError, match="nothing left to pick"):
        refuse_unless_select_certified(
            piv, int(rank), psd, n_keep=k_keep, M=M, orbit_id=oid,
            d0max=float(np.real(np.asarray(jnp.diag(G))).max()))

    # CONSTRUCTIBLE-FALSE TWIN: ask for FEWER orbits than exist and the same
    # kernel + the same contract must go through untouched.
    piv_ok, _, rank_ok, _, _, _, psd_ok = pivoted_cholesky_select(
        G, n_orb - 2, jnp.asarray(oid))
    assert int(rank_ok) == n_orb - 2
    refuse_unless_select_certified(
        piv_ok, int(rank_ok), psd_ok, n_keep=n_orb - 2, M=M,
        orbit_id=oid, d0max=float(np.real(np.asarray(jnp.diag(G))).max()))


def test_select_detects_an_indefinite_gram():
    """The PSD detector survives the clamp — it reads the value before it.

    BEFORE (measured, lambda_min = -4.95e-01 against lambda_max = 4.95e+02,
    i.e. indefinite by a part in a thousand): reported rank 23 of 24, all 24
    pivots distinct, L entirely finite, tr(R)/tr(G) at the end 0.0 and
    ``min(d_final)`` exactly 0.0.  There was NO signal anywhere in the
    return tuple that the input was not positive semidefinite, because the
    ``jnp.maximum(..., 0.0)`` on the Schur update destroyed the classic
    detector before it could be observed.  LAPACK's ``pstrf`` returns
    ``INFO > 0`` for exactly this case.
    """
    from src.centroid.pivoted_cholesky import (pivoted_cholesky_select,
                                               refuse_unless_select_certified)

    n = 24
    rng = np.random.default_rng(11)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    lam = np.linspace(1.0, 495.0, n)
    lam[0] = -0.495                                   # one negative eigenvalue
    Gbad = Q @ np.diag(lam) @ Q.conj().T
    Gbad = jnp.asarray(0.5 * (Gbad + Gbad.conj().T), dtype=jnp.complex128)
    assert np.linalg.eigvalsh(np.asarray(Gbad))[0] < 0

    piv, _, rank, _, _, _, psd = pivoted_cholesky_select(Gbad, n)
    d_min_raw, at_row, at_step = (float(psd[0]), int(psd[1]), int(psd[2]))
    floor = _floor_of(Gbad)
    assert d_min_raw < -floor, (
        f"the residual diagonal never went measurably negative "
        f"(d_min_raw={d_min_raw:.3e}, floor={-floor:.3e}) — the "
        f"detector is reading a clamped value again")
    # pstrf's INFO names WHERE.  Both indices must be real, not the -1
    # 'never happened' sentinel, or the refusal below cannot name a pivot.
    assert 0 <= at_row < n, f"at_row={at_row} is not a candidate row"
    assert 0 <= at_step < n, f"at_step={at_step} is not an iteration"

    with pytest.raises(RuntimeError, match="not positive semidefinite"):
        refuse_unless_select_certified(
            piv, int(rank), psd, n_keep=int(rank), M=n, orbit_id=None,
            d0max=float(np.real(np.asarray(jnp.diag(Gbad))).max()))

    # CONSTRUCTIBLE-FALSE TWIN: the SAME spectrum with the sign flipped back
    # is PSD, and must not trip the detector.  Same Q, same conditioning, so
    # the only thing that changed is definiteness.
    lam[0] = +0.495
    Gok = Q @ np.diag(lam) @ Q.conj().T
    Gok = jnp.asarray(0.5 * (Gok + Gok.conj().T), dtype=jnp.complex128)
    p2, _, r2, _, _, _, psd2 = pivoted_cholesky_select(Gok, n)
    assert float(psd2[0]) >= -_floor_of(Gok), (
        f"a PSD Gram tripped the not-PSD detector "
        f"(d_min_raw={float(psd2[0]):.3e} against floor "
        f"{-_floor_of(Gok):.3e}) — the threshold is too tight")
    refuse_unless_select_certified(
        p2, int(r2), psd2, n_keep=int(r2), M=n, orbit_id=None,
        d0max=float(np.real(np.asarray(jnp.diag(Gok))).max()))


def test_select_tolerance_is_a_documented_knob_not_a_constant():
    """``tol_rel`` moves the stopping point, and the default is sqrt(eps).

    The floor is ``tol_rel · max(diag G)`` — relative to the largest INITIAL
    diagonal, which is what makes the answer invariant to how G is scaled.
    ``sqrt(eps)`` is the default because it is the number this kernel has
    always computed for its ``rank`` report, so adopting it as the stopping
    rule leaves every existing deck's reported rank where it was.  LAPACK
    ``?pstrf`` uses ``n·eps`` instead; a caller that wants that policy has
    to be able to ask for it, and this is where that is pinned.
    """
    from src.centroid.pivoted_cholesky import pivoted_cholesky_select

    M, k_keep = 64, 40
    G = _psd_gram(M, rank=M, seed=17)

    # Scale invariance: the SAME Gram times 1e6 must stop in the same place.
    r_1 = int(pivoted_cholesky_select(G, k_keep)[2])
    r_scaled = int(pivoted_cholesky_select(1.0e6 * G, k_keep)[2])
    assert r_1 == r_scaled == k_keep, (
        f"the floor is not scale-relative: rank {r_1} vs {r_scaled} on the "
        f"same Gram scaled by 1e6")

    # A brutally loose tolerance must stop EARLY — that is the knob working.
    # MEASURED on this Gram (max diag 168.06): tol_rel = 1e-2, 1e-1 all still
    # run the full 40, because the residual diagonals of a full-rank random
    # Gram are genuinely still above 1.7 and 16.8 at the 40th pivot.  0.3
    # (floor 50.4) is the first that bites, at 37/40.  That is the honest
    # scale here and the number is quoted rather than guessed.
    r_loose = int(pivoted_cholesky_select(G, k_keep, tol_rel=0.3)[2])
    assert r_loose < k_keep, (
        f"tol_rel=0.3 (floor {0.3 * _maxdiag(G):.1f}) "
        f"still certified {r_loose}/{k_keep} directions — the override is "
        f"not reaching the stopping rule")

    # LAPACK's own policy (n·eps) is looser than sqrt(eps), so on a
    # rank-deficient Gram it must certify at least as many directions.
    Gd = _psd_gram(M, rank=10, seed=1)
    r_sqrt = int(pivoted_cholesky_select(Gd, k_keep)[2])
    r_lapack = int(pivoted_cholesky_select(
        Gd, k_keep, tol_rel=M * float(np.finfo(np.float64).eps))[2])
    assert r_lapack >= r_sqrt, (
        f"n·eps ({r_lapack}) certified fewer than sqrt(eps) ({r_sqrt}), "
        f"but n·eps = {M * np.finfo(np.float64).eps:.3e} is the LOOSER "
        f"floor of the two")


def test_select_refuses_a_pivot_outside_the_candidate_range():
    """The pad/sentinel guard is UNCONDITIONAL and can be made to fire.

    Assessment §4.5: the existing pad control could not be falsified — a
    padded Gram run with ``active_init=None`` also produced no pad pivot, so
    the guard was belt-and-braces whose braces had never been observed to
    hold anything up.  This drives the contract directly, which is the only
    way to observe it firing.  It also pins the 2026-08-07 change that took
    the guard out from under ``if n_pad``: an unpadded problem is exactly
    where a -1 sentinel would otherwise index from the END of the candidate
    list and hand back a real-looking centroid.
    """
    from src.centroid.pivoted_cholesky import refuse_unless_select_certified

    M = 60
    ok = np.arange(4, dtype=np.int32)
    kw = dict(n_keep=4, M=M, M_pad=64, orbit_id=None, d0max=1.0)

    refuse_unless_select_certified(ok, 4, (0.0, -1, -1), **kw)   # twin

    for label, piv in (("pad row", np.array([0, 1, 2, 61], dtype=np.int32)),
                       ("sentinel", np.array([0, 1, 2, -1], dtype=np.int32))):
        with pytest.raises(RuntimeError, match="outside the candidate range"):
            refuse_unless_select_certified(piv, 4, (0.0, -1, -1),
                                           **kw), label


# ─────────────────────────────────────────────────────────────────────────
# 3. Orbit-aware Lloyd reduces to plain Lloyd on the trivial group
# ─────────────────────────────────────────────────────────────────────────

def _bumpy_density(N=10, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, N, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    rho = 1e-3 * np.ones((N, N, N))
    for c in rng.random((3, 3)):
        d = np.stack([X - c[0], Y - c[1], Z - c[2]], axis=-1)
        d -= np.round(d)
        rho += np.exp(-np.sum(d ** 2, axis=-1) / 0.02)
    return rho


def test_orbit_path_with_trivial_group_matches_plain_path():
    """With the identity as the only symmetry op, orbit representatives ARE
    literal points and the two branches of ``make_lloyd_loop`` must agree."""
    from src.centroid import distribution as dist
    from src.centroid.kmeans_isdf import weighted_kmeans_jax

    if jax.process_count() > 1:
        pytest.skip("branch-equality gate is a one-process test")

    avec = np.diag([4.0, 4.5, 5.0])
    rho = _bumpy_density()
    mesh = dist.build_mesh(rho.size, shard=False)
    eye = np.eye(3, dtype=np.int32)[None]
    zero = np.zeros((1, 3), dtype=np.float64)
    kw = dict(N_c=8, max_steps=40, tolerance=1e-4, seed=0, mesh=mesh,
              init_method="kpp")

    _, c_plain, s_plain, _ = weighted_kmeans_jax(avec, rho, **kw)
    _, c_orbit, s_orbit, _ = weighted_kmeans_jax(
        avec, rho, R=eye, Rinv=eye, tau=zero, **kw)

    assert s_plain == s_orbit, (
        f"trivial-group orbit path took {s_orbit} Lloyd steps vs "
        f"{s_plain} for the plain path")
    np.testing.assert_allclose(np.asarray(c_plain), np.asarray(c_orbit),
                               rtol=0, atol=1e-12)

    # Negative control: a NON-trivial group must move the answer, otherwise
    # the agreement above would be explained by the orbit machinery being
    # inert rather than by it being correct.
    inv = -np.eye(3, dtype=np.int32)
    grp = np.stack([np.eye(3, dtype=np.int32), inv])
    _, c_inv, _, _ = weighted_kmeans_jax(
        avec, rho, R=grp, Rinv=grp, tau=np.zeros((2, 3)), **kw)
    assert not np.allclose(np.asarray(c_plain), np.asarray(c_inv),
                           atol=1e-8), (
        "negative control did not fire: adding inversion to the group left "
        "the centroids unchanged, so the trivial-group agreement is vacuous")


# ─────────────────────────────────────────────────────────────────────────
# 4. R4 — the sharded select on MORE THAN ONE SHARD
# ─────────────────────────────────────────────────────────────────────────
#
# THE HOLE THIS CLOSES.  The
# ``test_sharded_select_matches_single_device_reference`` cell above builds
# its mesh with ``dist.build_mesh(10**6, shard=False)``, which
# returns a 1x1 mesh.  Both sides of that comparison therefore run with one
# shard, ``M_slab == M``, and EVERY collective in
# ``make_sharded_pivoted_cholesky_select`` is satisfied vacuously: the
# ``pmax`` tie-break has nothing to tie against, the ``psum`` of ``L[p, :]``
# broadcasts from the only shard to itself, the orbit-id broadcast is the
# identity, and the ``pmax`` on the noise floor sees one value.  Assessment
# §4.0 measured that there was no test anywhere in the repository running
# this kernel on more than one shard — ``tests/multi_device/`` has sixteen
# gates and none of them touch centroids — and called it the single clearest
# test gap in the package.  The kernel PASSES when you actually test it; the
# hole is a risk carried, not a live bug, but it means every change to that
# kernel is unguarded.  This is the guard.
#
# HOW FOUR DEVICES GET HERE.  ``--xla_force_host_platform_device_count`` is
# read by XLA at the FIRST jax import, and the monorepo suite has already
# imported jax by the time this file is collected, so the flag cannot be set
# in-process.  The worker below therefore runs in a SUBPROCESS with the flag
# in its environment — the same idiom
# ``tests/test_transverse_lu_resolve.py`` uses, and the reason this cell
# runs unconditionally instead of skipping the way an in-process
# ``require_devices`` would.

_R4_WORKER = r'''
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh
assert jax.device_count() == 4, f"want 4 devices, got {jax.device_count()}"

from centroid.pivoted_cholesky import (pivoted_cholesky_select,
                                       make_sharded_pivoted_cholesky_select)

mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
AX = ("x", "y")

def gram(M, rank, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, rank)) + 1j * rng.standard_normal((M, rank))
    G = A @ A.conj().T
    return jnp.asarray(0.5 * (G + G.conj().T), dtype=jnp.complex128)

def real_gram(M, rank, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, rank))
    return jnp.asarray(A @ A.T, dtype=jnp.float64)

def check(name, G, k, oid=None):
    oj = None if oid is None else jnp.asarray(oid)
    ref = pivoted_cholesky_select(G, k, oj)
    step = make_sharded_pivoted_cholesky_select(
        mesh, int(G.shape[0]), k, mesh_axis=AX)
    sh = step(G) if oid is None else step(G, jnp.asarray(oid))
    pr, ps = np.asarray(ref[0]), np.asarray(sh[0])
    assert np.array_equal(pr, ps), f"{name}: piv {pr} vs {ps}"
    assert int(ref[2]) == int(sh[2]), (
        f"{name}: rank {int(ref[2])} vs {int(sh[2])}")
    np.testing.assert_allclose(np.asarray(ref[4]), np.asarray(sh[4]),
                               rtol=1e-12, atol=0.0, err_msg=name)
    np.testing.assert_allclose(float(ref[6][0]), float(sh[6][0]), rtol=1e-10,
                               atol=1e-300, err_msg=name + " d_min_raw")
    assert int(ref[6][1]) == int(sh[6][1]), f"{name}: psd at_row"
    print(f"  OK {name}: rank {int(ref[2])}/{k}, "
          f"M_slab={int(G.shape[0])//4}, piv[0]={pr[0]}")
    return pr

# (a) full rank, and it must really be sharded four ways.
p_full = check("full rank M=64 k=20", gram(64, 64, 3), 20)

# (b) RANK-DEFICIENT: the stopping rule has to agree across shard counts,
#     which is the property that needs the floor to be a pmax.
check("rank 12 of 64, k=40", gram(64, 12, 5), 40)

# (c) ORBIT mode with k_keep ABOVE the orbit count -> both stop at 16.
oid = np.repeat(np.arange(16, dtype=np.int32), 4)
rng = np.random.default_rng(21)
B = rng.standard_normal((16, 30)) + 1j * rng.standard_normal((16, 30))
A = B[oid]
Go = jnp.asarray(A @ A.conj().T, dtype=jnp.complex128)
check("orbit 16x4, k=24 (over)", 0.5 * (Go + Go.conj().T), 24, oid)

# (d) a REAL float64 Gram — nothing in the suite tested one.
check("real float64 M=64 k=20", real_gram(64, 64, 8), 20)

# THE FALSIFICATION (assessment R4's own bar).  Flipping the tie-break from
# lowest to highest global index must BREAK the reference comparison; if it
# does not, every cell above is vacuous.  Build a Gram with a deliberate
# exact tie across shards so the tie-break is load-bearing.
M = 64
blk = np.asarray(gram(16, 16, 33))
Gt = jnp.asarray(np.kron(np.eye(4), blk), dtype=jnp.complex128)
piv_lo = np.asarray(pivoted_cholesky_select(Gt, 8)[0])
step = make_sharded_pivoted_cholesky_select(mesh, M, 8, mesh_axis=AX)
assert np.array_equal(piv_lo, np.asarray(step(Gt)[0])), "tied Gram disagrees"

import types
import centroid.pivoted_cholesky as pc
from jax import lax as _lax
_real_pmax = _lax.pmax
def _flip(x, axis_name):
    # Invert ONLY the tie-break pmax (the one on a negated int32 index).
    # The kernel computes global_p = -pmax(-winner_p), i.e. the LOWEST real
    # index among the shards holding the max.  Here x IS -winner_p; map the
    # "not me" sentinel (+2**30) out of the way and return -max(winner_p),
    # so the caller's negation yields the HIGHEST real index instead.  A
    # genuine, valid, different tie-break -- not a broken kernel.
    if getattr(x, "dtype", None) == jnp.int32:
        w = jnp.where(-x == jnp.int32(2**30), jnp.int32(-1), -x)
        return -_real_pmax(w, axis_name)
    return _real_pmax(x, axis_name)
# SimpleNamespace, not a class: functions in a class body become bound
# methods and every lax call would receive the namespace as its first arg.
_fake = types.SimpleNamespace(**{k: getattr(_lax, k) for k in dir(_lax)
                                 if not k.startswith("_")})
_fake.pmax = _flip
pc.lax = _fake
step_hi = pc.make_sharded_pivoted_cholesky_select(mesh, M, 8, mesh_axis=AX)
piv_hi = np.asarray(step_hi(Gt)[0])
pc.lax = _lax
assert not np.array_equal(piv_lo, piv_hi), (
    f"NEGATIVE CONTROL DID NOT FIRE: flipping the tie-break left the pivots "
    f"at {piv_lo} — the multi-shard comparison is still vacuous")
print(f"  OK negative control: lowest-index {piv_lo} vs highest {piv_hi}")

# ---- collectives PER ITERATION, counted in the lowered HLO ----------------
# NCCL latency is unmeasurable here (emulated CPU devices), so the count is
# the gate.  XLA's metadata op_name carries `while/body` for exactly the ops
# that run once per iteration, which is the number that matters: the select
# is latency-bound (assessment 4.2 -- 900 iterations x 3 collectives is 2700
# round trips at 20-40 us each, "most of the 0.129 s").
import re as _re
def per_iter_collectives(M_, k_, oid_):
    step_ = make_sharded_pivoted_cholesky_select(mesh, M_, k_, mesh_axis=AX)
    G_ = gram(M_, M_, 3)
    txt = (step_.lower(G_).compile().as_text() if oid_ is None
           else step_.lower(G_, jnp.asarray(oid_)).compile().as_text())
    return [l for l in txt.split("\n")
            if ("all-reduce(" in l or "all-gather(" in l)
            and "while/body" in l]

pt = per_iter_collectives(64, 20, None)
ob = per_iter_collectives(64, 12, np.repeat(np.arange(16, dtype=np.int32), 4))
print(f"  per-iteration collectives: point {len(pt)}, orbit {len(ob)}")
assert len(pt) == 3, f"point mode: {len(pt)} per-iteration collectives {pt}"
assert len(ob) == 3, f"orbit mode: {len(ob)} per-iteration collectives {ob}"
# The orbit-id broadcast must RIDE the L[p,:] psum, not be a fourth trip:
# at k_keep=12 the fused psum is c128[13], not c128[12] plus an s32[].
assert any("c128[13]" in l for l in ob), (
    f"the orbit-id broadcast is not fused into the L[p,:] psum: {ob}")
assert not any("s32[] all-reduce" in l for l in ob if "pmax" not in l), (
    f"a separate integer all-reduce survived in orbit mode: {ob}")
print("R4_MULTISHARD_OK")
'''


def test_sharded_select_on_four_real_shards():
    """The select kernel, on a real 2x2 mesh, against the reference.

    Four cells the 1x1 gate above cannot reach — full rank, rank-deficient
    (which exercises the ``pmax``'d noise floor and the shared stopping
    rule), orbit mode with ``k_keep`` past the orbit count, and a real
    ``float64`` Gram — plus the falsification assessment R4 asked for: with
    the tie-break flipped from lowest to highest global index the reference
    comparison MUST break, or the whole cell is vacuous.
    """
    import os
    import subprocess
    import sys

    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src")
    env = os.environ.copy()
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "")
                        + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", _R4_WORKER], env=env,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, (
        f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}")
    assert "R4_MULTISHARD_OK" in out.stdout, out.stdout


# ─────────────────────────────────────────────────────────────────────────
# 5. R2 — rank reported at POINT granularity, not orbit granularity
# ─────────────────────────────────────────────────────────────────────────

def test_point_granularity_rank_separates_orbits_from_points():
    """The number that would have caught D3 before it spent 7 GiB.

    In orbit mode the select deflates the Schur complement by ONE direction
    per orbit while removing all n_sym members from contention, so the rank
    it reports counts ORBITS.  D3's gate passed at "42 of 42 directions
    certified" and blessed a file of 1908 POINTS whose ζ Gram then
    truncated to 1440-1455 modes per q — 23.7-24.5 %, logged eight times a
    leg and read by nobody.  The two numbers were never comparable, and
    nothing in the pipeline computed the second one.

    Constructed here so the gap is exact and known in advance: 12 orbits of
    8 built from 12 independent feature rows, so the delivered 96-point set
    has exactly 12 independent directions.  A gate reading the orbit rank
    sees 12/12 and passes; the point-granularity number says 12 of 96.
    """
    from src.centroid.pivoted_cholesky import (pivoted_cholesky_select,
                                               point_granularity_rank)

    n_orb, orb_size = 12, 8
    oid = np.repeat(np.arange(n_orb, dtype=np.int32), orb_size)
    rng = np.random.default_rng(5)
    B = (rng.standard_normal((n_orb, 40))
         + 1j * rng.standard_normal((n_orb, 40)))
    A = B[oid]
    G = jnp.asarray(A @ A.conj().T, dtype=jnp.complex128)
    G = 0.5 * (G + G.conj().T)

    piv, _, rank, *_ = pivoted_cholesky_select(G, n_orb, jnp.asarray(oid))
    keep = np.isin(oid, oid[np.asarray(piv)[np.asarray(piv) >= 0]])
    assert int(rank) == n_orb, "orbit-granularity rank"
    assert keep.sum() == n_orb * orb_size, "all orbits unfolded"

    pt_rank, n_pts, why = point_granularity_rank(G, keep)
    assert why == "", why
    assert n_pts == n_orb * orb_size
    assert pt_rank == n_orb, (
        f"the delivered {n_pts}-point set was built from {n_orb} "
        f"independent features, so its point-granularity rank must be "
        f"{n_orb}; got {pt_rank}")
    # THE POINT: the orbit rank and the point count are not comparable, and
    # that gap is what a passing gate used to hide.
    assert int(rank) == pt_rank < n_pts

    # CONSTRUCTIBLE-FALSE TWIN: a genuinely full-rank point set must report
    # its rank EQUAL to its point count, or the number above is measuring
    # something other than independence.
    n = 48
    rng2 = np.random.default_rng(6)
    A2 = (rng2.standard_normal((n, 96)) + 1j * rng2.standard_normal((n, 96)))
    G2 = jnp.asarray(A2 @ A2.conj().T, dtype=jnp.complex128)
    G2 = 0.5 * (G2 + G2.conj().T)
    full_rank, n2, why2 = point_granularity_rank(G2, np.ones(n, dtype=bool))
    assert why2 == "", why2
    assert full_rank == n2 == n, (
        f"a full-rank {n}-point Gram reported {full_rank} independent "
        f"directions; the twin does not discriminate")

    # The cap REPORTS rather than silently skipping: "no number" and "the
    # number is fine" must not look alike in a log.
    capped, n3, why3 = point_granularity_rank(G, keep, cap=8)
    assert capped is None and n3 == n_pts and "exceeds the O(n^3) cap" in why3
