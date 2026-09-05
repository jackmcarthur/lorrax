"""Independent numerical oracle for the parent-k → full-k operator transport.

WHAT IS UNDER TEST.  A parent-k ζ fit never forms the Green-oriented pair
projector at every full-zone k.  It forms it at the raw parents only and
TRANSPORTS it with the symmetry service.  For an orbit-closed centroid set
``{r_μ}`` and an orbit-closed real-grid set ``{r_j}``::

    D_k[a, μ, b, j] = Σ_n w_n ψ_{n k a}(r_μ) conj(ψ_{n k b}(r_j))

with real weights ``w_n``.  The claim is that for a child ``k`` with raw
parent ``kbar = sym.irr_idx_k[k]`` and typed operation row
``g = sym.sym_idx_k[k]``::

    D_k        = U_k T_k U_k†                     (over the spin indices)
    T_k[μ, j]  = phase_k[μ] conj(phase_k[j]) D_kbar[α_g(μ), α_g(j)]
    phase_k[x] = exp(2πi kbar · L_{g,x})

with ``T_k`` conjugated elementwise on the antiunitary rows
(``g ≥ ntran``) and ``U_k = sym.spinor_action(sym.sym_idx_k, nspinor=2)[k]``.
The relation is band-diagonal, so it holds for ANY real ``w_n``; both a
uniform and a graded weight are measured below and the difference between
them is what makes ``U_k`` observable at all on this deck (see DECK FACT 2).

WHY THIS IS AN ORACLE AND NOT A TAUTOLOGY.  The reference ``T_k`` is built
by the production service call ``symmetry_maps.unfold_isdf_operator`` on a
rectangular (centroid × grid) endpoint pair — a centroid-axis double gather
plus an umklapp phase, entirely in the ISDF basis.  The ground truth is
built by a completely different route: the loader's certified per-G ψ unfold
(``WfnLoader.load(k='full_bz')`` — τ phases, G-list relabel, TR conjugation)
sampled at the two point sets through ``common.wfn_transforms.gflat_to_rmu``,
then contracted band by band.  Agreement means the operator-level transport
and the wavefunction-level unfold are the same physics.

DISCRIMINATION.  A test that only ever passes measures nothing, so the same
harness is run with deliberate breakages — the antiunitary conjugation
switched off, the umklapp phase built on ``-kbar``, the spinor rotation
replaced by the identity — and their errors are asserted to be O(1).

DECK FACT 1 — THIS DECK USES NO ANTIUNITARY ROW.
``tests/regression/si_cohsex_debug/WFN.h5`` reaches all 64 full-zone k from
its 8 stored parents with SPATIAL rows only: ``max(sym.sym_idx_k) = 23 <
ntran = 48``, measured here and independently recorded as ``n_trs = 0`` in
``tests/test_star_wedge_measured_values.py``.  Switching the conjugation off
therefore cannot move the loader-assigned transport at all, and reporting
~0 for that switch would be a vacuous "pass".  The antiunitary branch is
instead exercised on a SECOND, equally valid row assignment: every one of
the 64 k also has a time-reversal-composed row reaching the same parent
(measured, 64/64), and ``D_k`` at uniform weight is the kernel of a
projector, hence independent of which group element reaches ``k``.  Dropping
the conjugation on THAT assignment is O(1) wrong, which is the measurement
the vacuous arm was supposed to make.

DECK FACT 2 — AT UNIFORM WEIGHT THIS DECK CANNOT SEE ``U_k``.
``si_cohsex_debug`` is a scalar-relativistic calculation stored in
two-component form with exact spin pairing (the same property
``tests/test_scalar_psp.py`` relies on).  Every band is purely up or purely
down and the partners share a spatial part, so at ``w_n = 1`` over complete
Kramers pairs ``D_k`` is numerically a multiple of the 2x2 identity in spin
— and ``U D U† = D`` for every unitary ``U``.  Replacing ``U_k`` by the
identity is then NOT a detectable breakage, which the ``spin_scalar_``
``residual`` diagnostic quantifies.  A graded weight ``w_n = n + 1`` splits
the Kramers partners, ``D_k`` stops being spin-scalar, and the same
substitution becomes O(1) wrong.  The graded arm is used only where the
relation is exact band by band (the loader's own rows); it is deliberately
NOT used on the alternative-row arms, where a little-group element may mix
degenerate partners and only the projector is invariant.

DECK FACT 3 — THE ANTIUNITARY ARM'S FLOOR IS THE FILE'S, NOT THE CODE'S.
The alternative-row arm does not close to the 1e-15 of the spatial arm.
Its residual is measured to be the committed file's own antiunitary
self-inconsistency: ``parent_trs_selfconsistency`` transports each parent's
own operator by an antiunitary element of that parent's little group and
compares it with itself, using nothing but the same service call, and lands
at the same order.  The loader's independent two-component DFT-reference
check reports a residual of 6.1e-08 on this file for the same reason.
``TOL_TRS_FILE`` is set from that measurement and is a statement about the
fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.wfn_transforms import gflat_to_rmu
from symmetry_maps import centroid_source_map_and_wrap, unfold_isdf_operator
from wfn_loader import WfnLoader

#: The in-tree deck: Si 4x4x4, 8 stored parents -> 64 k, nspinor = 2,
#: 48 operations of which 36 are non-symmorphic.
DECK_WFN = (Path(__file__).resolve().parent
            / "regression" / "si_cohsex_debug" / "WFN.h5")

#: Bands carried into the projector.  8 = the full Si valence manifold at
#: nspinor = 2, so the uniform-weight band cut is a gapped, Kramers-complete
#: subspace and the projector argument of DECK FACT 1 applies.
DECK_NB = 8

#: Agreement floor for a transport that is exact band by band.  Both routes
#: are complex128 end to end; the residual is FFT + gather round-off on an
#: O(1)-normalised operator.
TOL_REL = 1e-10

#: Floor for an arm that leans on the FILE's antiunitary self-consistency
#: rather than on an exact band-diagonal identity.  See DECK FACT 3.
TOL_TRS_FILE = 1e-6

#: A broken transport must be wrong by a lot, not by a little.
TOL_BROKEN = 1e-2


def mesh_1x1() -> Mesh:
    """The single-device (x, y) mesh.  Same code path as production; the
    all-to-all endpoint redistribution degenerates to a local gather and no
    endpoint-extent divisibility applies."""
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))


def orbit_closed_points(sym, fft_grid, *, seed, n_seeds):
    """A random, orbit-closed set of FFT-grid points.

    The forward BGW real-space action is ``r' = mtrx⁻¹·r + τ``; the union of
    the seeds' orbits under it is closed by construction (the group is closed
    under composition), which is what
    :func:`symmetry_maps.centroid_source_map_and_wrap` demands of an endpoint
    table.  Closure is not assumed here — the table build validates it.
    """
    grid = np.asarray(fft_grid, dtype=np.int64).reshape(3)
    S = np.asarray(sym.sym_matrices, dtype=np.float64)
    rinv = np.rint(np.linalg.inv(S)).astype(np.int64)
    tau = np.asarray(sym.translations, dtype=np.float64) / (2.0 * np.pi)
    tau_int = np.rint(tau * grid[None, :]).astype(np.int64)
    if np.max(np.abs(tau * grid[None, :] - tau_int)) > 1e-8:
        raise AssertionError(
            "fractional translations are not commensurate with the FFT grid; "
            "no point set on this grid can be orbit closed")
    rng = np.random.default_rng(seed)
    pts: set[tuple[int, int, int]] = set()
    for _ in range(int(n_seeds)):
        r = np.asarray([rng.integers(0, int(g)) for g in grid], dtype=np.int64)
        for s in range(rinv.shape[0]):
            img = (rinv[s] @ r + tau_int[s]) % grid
            pts.add(tuple(int(v) for v in img))
    return np.asarray(sorted(pts), dtype=np.int32)


def frac_to_fft_idx(frac, fft_grid):
    """Fractional coordinates -> FFT-grid indices (round, then wrap)."""
    grid = np.asarray(fft_grid, dtype=np.int64).reshape(3)
    idx = np.rint(np.asarray(frac, dtype=np.float64) * grid[None, :])
    return (idx.astype(np.int64) % grid).astype(np.int32)


def endpoint_tables(sym, points, fft_grid):
    """``(α, L)`` for one endpoint, TRS-augmented, closure validated."""
    return centroid_source_map_and_wrap(
        np.asarray(points, dtype=np.int32),
        np.asarray(sym.sym_matrices),
        np.asarray(sym.translations),
        tuple(int(v) for v in fft_grid),
        validate=True, extend_trs=True)


def sample_psi(loader, psi, kspec, points, mesh, *, chunk_size):
    """ψ(G-flat) -> ψ at the given FFT-grid points, Bloch phase included."""
    return np.asarray(gflat_to_rmu(
        psi,
        loader.box_index_dev(k=kspec, mesh=mesh),
        np.asarray(points, dtype=np.int32),
        mesh=mesh,
        fft_grid=tuple(int(v) for v in loader.fft_grid),
        kvecs_frac=loader.kvecs(k=kspec),
        norm="ortho",
        chunk_size=chunk_size,
    ))


def pair_projector(psi_left, psi_right, weights):
    """``D[k, a, μ, b, j] = Σ_n w_n ψ_left[k,n,a,μ] conj(ψ_right[k,n,b,j])``.

    Band-pad rows are zero-filled by the loader contract, so they add
    nothing whatever ``weights`` says about them.
    """
    w = np.asarray(weights, dtype=np.float64)
    return np.einsum('n,knam,knbj->kambj', w, psi_left, np.conj(psi_right),
                     optimize=True)


def spin_scalar_residual(D):
    """How far ``D`` is from ``δ_ab`` times a scalar, relative to its size.

    The reason ``U_k`` is or is not observable in a given weighting.
    """
    trace_half = 0.5 * (D[:, 0, :, 0, :] + D[:, 1, :, 1, :])
    dev = D.copy()
    dev[:, 0, :, 0, :] -= trace_half
    dev[:, 1, :, 1, :] -= trace_half
    return float(np.max(np.abs(dev))) / float(np.max(np.abs(D)))


def transport(D_parent, *, irr_idx, sym_idx, tables_left, tables_right,
              q_irr_frac, mesh, n_sym_spatial, spin_action):
    """Transport ``D`` from the raw parents to every full-zone k.

    Spatial/phase/antiunitary transport is the SERVICE call
    ``unfold_isdf_operator``, one scalar (c, d) spin block at a time.  The
    2x2 spin block action ``D[a,b] = Σ_cd U[a,c] conj(U[b,d]) T[c,d]`` is
    applied here in numpy so that this file never re-implements any part of
    the endpoint transport it is checking.  ``spin_action=None`` is the
    ``U_k = 1`` breakage.
    """
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    left_perm, left_L = tables_left
    right_perm, right_L = tables_right
    n_spin = int(D_parent.shape[1])
    n_full = int(np.asarray(irr_idx).shape[0])
    n_left = int(D_parent.shape[2])
    n_right = int(D_parent.shape[4])
    T = np.empty((n_full, n_spin, n_spin, n_left, n_right),
                 dtype=np.complex128)
    for c in range(n_spin):
        for d in range(n_spin):
            block = jax.device_put(
                np.ascontiguousarray(D_parent[:, c, :, d, :]), sharding)
            T[:, c, d] = np.asarray(unfold_isdf_operator(
                block,
                irr_idx=np.asarray(irr_idx, dtype=np.int32),
                sym_idx=np.asarray(sym_idx, dtype=np.int32),
                sym_perm=left_perm, L_table=left_L,
                right_sym_perm=right_perm, right_L_table=right_L,
                q_irr_frac=np.asarray(q_irr_frac, dtype=np.float64),
                mesh_xy=mesh,
                n_sym_spatial=int(n_sym_spatial),
                trs_rule="conj"))
    if spin_action is None:
        return np.moveaxis(T, 2, 3)
    U = np.asarray(spin_action, dtype=np.complex128)
    return np.einsum('kac,kbd,kcdmj->kambj', U, np.conj(U), T, optimize=True)


def rel_err(candidate, reference):
    """max |Δ| / max |reference| — one number per arm."""
    scale = float(np.max(np.abs(reference)))
    if scale == 0.0:
        raise AssertionError("reference operator is identically zero")
    return float(np.max(np.abs(candidate - reference))) / scale


def _k_images(sym, stored):
    """``images[kbar, s]`` = ``sym_mats_k[s] · kbar`` wrapped to [0, 1)."""
    images = np.mod(np.einsum(
        'sij,kj->ksi', np.asarray(sym.sym_mats_k, dtype=np.int64),
        np.asarray(stored, dtype=np.float64)), 1.0)
    images[images > 0.99999] = 0.0
    return images


def antiunitary_row_assignment(sym, loader):
    """A second, equally valid row assignment preferring antiunitary rows.

    For each full-zone k, every row ``g`` with ``sym_mats_k[g] · kbar ≡ k``
    (mod a reciprocal lattice vector) reaches the same parent; the lowest
    such row with ``g ≥ ntran`` is chosen when one exists.  Returns
    ``(sym_idx_alt, n_antiunitary)``.
    """
    ntran = int(np.asarray(sym.sym_matrices).shape[0])
    images = _k_images(sym, loader.kpoints)
    target = np.mod(np.asarray(sym.unfolded_kpts, dtype=np.float64), 1.0)
    target[target > 0.99999] = 0.0
    irr = np.asarray(sym.irr_idx_k, dtype=np.int32)
    out = np.asarray(sym.sym_idx_k, dtype=np.int32).copy()
    n_anti = 0
    for kf in range(target.shape[0]):
        hits = np.where(np.all(
            np.abs(images[int(irr[kf])] - target[kf]) < 1e-6, axis=1))[0]
        anti = hits[hits >= ntran]
        if anti.size:
            out[kf] = int(anti[0])
            n_anti += 1
    return out, n_anti


def antiunitary_little_group_rows(sym, loader):
    """One antiunitary row fixing each stored parent, where one exists.

    ``sym_mats_k[h] · kbar ≡ kbar`` with ``h ≥ ntran``.  Transporting a
    parent's own operator by ``h`` and comparing it with itself measures the
    FILE's antiunitary self-consistency on exactly the objects this oracle
    transports.  Returns ``(rows, mask)``.
    """
    ntran = int(np.asarray(sym.sym_matrices).shape[0])
    stored = np.mod(np.asarray(loader.kpoints, dtype=np.float64), 1.0)
    stored[stored > 0.99999] = 0.0
    images = _k_images(sym, loader.kpoints)
    rows = np.zeros(stored.shape[0], dtype=np.int32)
    mask = np.zeros(stored.shape[0], dtype=bool)
    for kb in range(stored.shape[0]):
        hits = np.where(np.all(
            np.abs(images[kb] - stored[kb]) < 1e-6, axis=1))[0]
        anti = hits[hits >= ntran]
        if anti.size:
            rows[kb] = int(anti[0])
            mask[kb] = True
    return rows, mask


def run_oracle(wfn_path, *, nb, centroid_points=None, centroid_seed=0,
               grid_seed=1, n_seeds=2, chunk_size=64, print_fn=print):
    """Build both routes and measure every arm.  Returns a dict of numbers.

    ``centroid_points`` overrides the random centroid endpoint with an
    explicit orbit-closed FFT-index table (the production centroid file).
    """
    mesh = mesh_1x1()
    loader = WfnLoader(str(wfn_path), backend="eager")
    try:
        sym = loader.symmetry()
        fft_grid = tuple(int(v) for v in loader.fft_grid)
        ntran = int(np.asarray(sym.sym_matrices).shape[0])

        if centroid_points is None:
            centroid_points = orbit_closed_points(
                sym, fft_grid, seed=centroid_seed, n_seeds=n_seeds)
        centroid_points = np.asarray(centroid_points, dtype=np.int32)
        grid_points = orbit_closed_points(
            sym, fft_grid, seed=grid_seed, n_seeds=n_seeds)

        tau = np.asarray(sym.translations, dtype=np.float64) / (2.0 * np.pi)
        n_nonsymmorphic = int(np.sum(
            np.any(np.abs(tau - np.rint(tau)) > 1e-8, axis=1)))
        print_fn(f"[oracle] wfn                 {wfn_path}")
        print_fn(f"[oracle] fft_grid            {fft_grid}")
        print_fn(f"[oracle] parents/full k      "
                 f"{int(loader.nkpts)} -> {int(sym.nk_tot)}")
        print_fn(f"[oracle] ntran / nonsymm     {ntran} / {n_nonsymmorphic}")
        print_fn(f"[oracle] nspinor / nbands    {int(loader.nspinor)} / {nb}")
        print_fn(f"[oracle] centroid endpoint   "
                 f"{centroid_points.shape[0]} points")
        print_fn(f"[oracle] grid endpoint       "
                 f"{grid_points.shape[0]} points")

        # Endpoint tables.  ``validate=True`` is the orbit-closure verdict.
        tables_left = endpoint_tables(sym, centroid_points, fft_grid)
        tables_right = endpoint_tables(sym, grid_points, fft_grid)

        irr_idx = np.asarray(sym.irr_idx_k, dtype=np.int32)
        sym_idx = np.asarray(sym.sym_idx_k, dtype=np.int32)
        print_fn(f"[oracle] sym_idx_k range     "
                 f"[{int(sym_idx.min())}, {int(sym_idx.max())}]  "
                 f"antiunitary rows used: {int(np.sum(sym_idx >= ntran))}"
                 f"/{sym_idx.size}")

        # ---- ground truth: the certified per-G psi unfold, sampled ----
        psi_full = loader.load(bands=(0, nb), k="full_bz")
        psi_parent = loader.load(bands=(0, nb), k="ibz")
        full_mu = sample_psi(loader, psi_full, "full_bz", centroid_points,
                             mesh, chunk_size=chunk_size)
        full_j = sample_psi(loader, psi_full, "full_bz", grid_points,
                            mesh, chunk_size=chunk_size)
        par_mu = sample_psi(loader, psi_parent, "ibz", centroid_points,
                            mesh, chunk_size=chunk_size)
        par_j = sample_psi(loader, psi_parent, "ibz", grid_points,
                           mesh, chunk_size=chunk_size)
        del psi_full, psi_parent

        nb_stored = int(full_mu.shape[1])
        w_uniform = np.ones(nb_stored, dtype=np.float64)
        w_graded = 1.0 + np.arange(nb_stored, dtype=np.float64)

        D_full = pair_projector(full_mu, full_j, w_uniform)
        D_parent = pair_projector(par_mu, par_j, w_uniform)
        D_full_g = pair_projector(full_mu, full_j, w_graded)
        D_parent_g = pair_projector(par_mu, par_j, w_graded)
        del full_mu, full_j, par_mu, par_j

        q_irr = loader.kvecs(k="ibz")
        U = np.asarray(sym.spinor_action(sym_idx, nspinor=2))
        common = dict(irr_idx=irr_idx, tables_left=tables_left,
                      tables_right=tables_right, mesh=mesh)

        results: dict[str, float] = {}
        results["_spin_scalar_residual_uniform_w"] = spin_scalar_residual(
            D_full)
        results["_spin_scalar_residual_graded_w"] = spin_scalar_residual(
            D_full_g)

        results["baseline"] = rel_err(transport(
            D_parent, sym_idx=sym_idx, q_irr_frac=q_irr,
            n_sym_spatial=ntran, spin_action=U, **common), D_full)
        results["baseline_graded_w"] = rel_err(transport(
            D_parent_g, sym_idx=sym_idx, q_irr_frac=q_irr,
            n_sym_spatial=ntran, spin_action=U, **common), D_full_g)

        # (i) antiunitary conjugation dropped.  Telling the service that the
        # spatial half is 2*ntran rows long makes ``trs_used`` False, so no
        # row is conjugated; nothing else changes.  Inert here — DECK FACT 1.
        results["no_antiunitary_conj"] = rel_err(transport(
            D_parent, sym_idx=sym_idx, q_irr_frac=q_irr,
            n_sym_spatial=2 * ntran, spin_action=U, **common), D_full)

        # (ii) umklapp phase built on -kbar.
        results["negated_parent_k_phase"] = rel_err(transport(
            D_parent, sym_idx=sym_idx, q_irr_frac=-q_irr,
            n_sym_spatial=ntran, spin_action=U, **common), D_full)

        # (iii) spinor rotation replaced by the identity.  Blind at uniform
        # weight, O(1) at graded weight — DECK FACT 2.
        results["identity_spinor"] = rel_err(transport(
            D_parent, sym_idx=sym_idx, q_irr_frac=q_irr,
            n_sym_spatial=ntran, spin_action=None, **common), D_full)
        results["identity_spinor_graded_w"] = rel_err(transport(
            D_parent_g, sym_idx=sym_idx, q_irr_frac=q_irr,
            n_sym_spatial=ntran, spin_action=None, **common), D_full_g)

        # The antiunitary branch, on a second valid row assignment.  Uniform
        # weight only: this arm leans on projector invariance, which a graded
        # weight would forfeit.
        sym_idx_alt, n_anti_alt = antiunitary_row_assignment(sym, loader)
        results["_n_antiunitary_forced"] = float(n_anti_alt)
        if n_anti_alt:
            U_alt = np.asarray(sym.spinor_action(sym_idx_alt, nspinor=2))
            results["antiunitary_rows"] = rel_err(transport(
                D_parent, sym_idx=sym_idx_alt, q_irr_frac=q_irr,
                n_sym_spatial=ntran, spin_action=U_alt, **common), D_full)
            results["antiunitary_rows_no_conj"] = rel_err(transport(
                D_parent, sym_idx=sym_idx_alt, q_irr_frac=q_irr,
                n_sym_spatial=2 * ntran, spin_action=U_alt, **common), D_full)

        # The file's own antiunitary self-consistency, on these very
        # operators: transport each parent by an antiunitary element of its
        # OWN little group and compare with itself.  DECK FACT 3.
        lg_rows, lg_mask = antiunitary_little_group_rows(sym, loader)
        results["_n_parents_with_antiunitary_little_group"] = float(
            np.sum(lg_mask))
        if np.all(lg_mask):
            n_parent = int(D_parent.shape[0])
            U_lg = np.asarray(sym.spinor_action(lg_rows, nspinor=2))
            results["parent_trs_selfconsistency"] = rel_err(transport(
                D_parent,
                irr_idx=np.arange(n_parent, dtype=np.int32),
                sym_idx=lg_rows, tables_left=tables_left,
                tables_right=tables_right, mesh=mesh,
                q_irr_frac=q_irr, n_sym_spatial=ntran,
                spin_action=U_lg), D_parent)

        width = max(len(k) for k in results)
        for name, value in results.items():
            if name.startswith("_n_"):
                print_fn(f"[oracle] {name:<{width}}  {int(value)}")
            elif name.startswith("_"):
                print_fn(f"[oracle] {name:<{width}}  {value:.6e}")
            else:
                print_fn(f"[oracle] {name:<{width}}  max rel err {value:.6e}")
        return results
    finally:
        loader.close()


@pytest.fixture(scope="module")
def oracle_results():
    if not DECK_WFN.is_file():
        pytest.skip(f"{DECK_WFN} is not staged in this checkout")
    return run_oracle(DECK_WFN, nb=DECK_NB, centroid_seed=20260905,
                      grid_seed=915260202, n_seeds=2)


@pytest.mark.parametrize("arm", ["baseline", "baseline_graded_w"])
def test_parent_transport_reproduces_the_direct_full_bz_projector(
        oracle_results, arm):
    """The service transport equals the directly computed full-zone D_k."""
    assert oracle_results[arm] < TOL_REL, (
        "the parent-k operator transport disagrees with the projector built "
        f"directly from the loader's full-BZ psi ({arm}): max rel err "
        f"{oracle_results[arm]:.6e}")


def test_the_antiunitary_row_assignment_transports_to_the_file_floor(
        oracle_results):
    """Reaching every k by a time-reversal-composed row reproduces D_k.

    Not to the 1e-15 of the spatial arm: the ceiling is the committed file's
    own antiunitary self-consistency, which the neighbouring
    ``parent_trs_selfconsistency`` measures with the same service call and no
    reference to a full-zone k at all.  DECK FACT 3.
    """
    if "antiunitary_rows" not in oracle_results:
        pytest.skip("no full-zone k on this deck has an antiunitary row")
    assert oracle_results["antiunitary_rows"] < TOL_TRS_FILE, (
        "transport on time-reversal-composed rows disagrees beyond the "
        "file's own antiunitary floor: max rel err "
        f"{oracle_results['antiunitary_rows']:.6e}")
    if "parent_trs_selfconsistency" in oracle_results:
        assert oracle_results["antiunitary_rows"] < 100.0 * max(
            oracle_results["parent_trs_selfconsistency"], TOL_REL), (
            "the antiunitary arm is far above the file's measured "
            "antiunitary self-consistency "
            f"({oracle_results['parent_trs_selfconsistency']:.6e}); its "
            "residual is then NOT attributable to the fixture")


@pytest.mark.parametrize("arm", [
    "negated_parent_k_phase",
    "identity_spinor_graded_w",
    "antiunitary_rows_no_conj",
])
def test_the_oracle_discriminates_a_broken_transport(oracle_results, arm):
    """Each deliberate breakage is O(1) wrong, so a pass is informative."""
    if arm not in oracle_results:
        pytest.skip(f"arm {arm} is not defined on this deck")
    assert oracle_results[arm] > TOL_BROKEN, (
        f"arm {arm} was expected to be O(1) wrong but measured "
        f"{oracle_results[arm]:.6e}; the oracle is not discriminating")


def test_the_two_blind_arms_are_blind_for_the_recorded_reasons(
        oracle_results):
    """Recorded properties of THIS deck, not of the transport rule.

    Both are the reason the discriminating arms above had to be built
    differently, and both would change on a deck with antiunitary rows in
    ``sym_idx_k`` or with genuine spinor mixing.
    """
    # DECK FACT 1: no antiunitary row is assigned, so the conj switch is a
    # no-op on the loader's own assignment.
    assert oracle_results["_n_antiunitary_forced"] > 0
    assert oracle_results["no_antiunitary_conj"] == pytest.approx(
        oracle_results["baseline"], rel=1e-12, abs=1e-15)
    # DECK FACT 2: at uniform weight the operator is spin-scalar, so U_k is
    # unobservable; the graded weight is what breaks that.
    assert oracle_results["_spin_scalar_residual_uniform_w"] < 1e-6
    assert oracle_results["_spin_scalar_residual_graded_w"] > 1e-2
    assert oracle_results["identity_spinor"] < 1e-6
