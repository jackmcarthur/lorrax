"""MPA scalar head coupled to each map's current shared-pole body W.

Only Gamma is evaluated, one frequency at a time. The common head owner
folds its wings and the common MPA owner fits the resulting scalar samples.

The head is the charge head on scalar (N_spinor = 1) and two-component
(N_spinor = 2) stores alike: the Coulomb kernel couples the charge density
only, so every vertex is the spin-traced pair density
``rho_ij(mu) = sum_s conj(psi_is(mu)) psi_js(mu)`` (traced inside
``gw.qsgw_head.head_wings_sharded``), the velocity is the spinor-traced matrix
element, the capacity is ``2/(n_spin n_spinor)`` states per band, and the body
is the spin-traced charge operator on ``n_mu x n_mu`` (factor spin axis 1).
Nothing crosses the body, so the fold ``S + Y W Z / Omega`` is one contraction
on both stores. Certified by ``tests/test_shared_pole_head_two_component.py``
(a spin-doubled two-component store reproduces the scalar head; a global SU(2)
rotation leaves it invariant).

An ordered store (measured broken time reversal, ``scalar-ordered-ph``)
carries the DIRECT frequency-dependent head only, ``head_correction =
no_local_fields``: the Cartesian tensor ``S_ab(z)`` of
``gw.qsgw_head.head_s_tensor_sharded`` needs no time-reversal assumption.
Its two Lehmann terms for one unordered pair combine at the same ``k``,

    (f_j - f_i)/D^2 [conj(T_ab)/(z - D) - T_ab/(z + D)]
        = 2 (f_j - f_i) [Re T_ab - i z Im T_ab / D] / (D (z^2 - D^2)),
    T_ab = conj(v^a_ij) v^b_ij,  D = e_i - e_j > 0,

so the signed occupation difference, the energy-ordered pair sum, the
conjugation and the denominator ``D [(z + i eta)^2 - D^2]`` all survive;
what time reversal removed was only the antisymmetric ``Im T_ab`` part
(nonzero on a magnet, odd in ``z``), which the mini-BZ quadratic form
``q.S.q`` annihilates whatever weight it carries.  The k sum runs over the
full zone unfolded by the measured magnetic group.  The wing/body fold of a
signed store is NOT implemented (owner scope 2026-09-21), so
``head_correction = full`` refuses on an ordered store by name; the
delivered ordered head is direct-term-only and on the one-shot and
``sc_head_update = off`` routes carry no intraband Drude/Thomas-Fermi
piece. The direct-only SC route may instead select ``dft_velocity``: it
rotates the authenticated DFT dipole into each map's QP basis and uses that
map's Fermi-surface weights in the common Drude/Thomas-Fermi kernels. The
ordered interband tensor is checked in ``tests/test_head_direct_ordered.py``.
"""
from types import SimpleNamespace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=None)
def _gamma_body(mesh):
    """Raw latent body; current factors are operands of one retained callable."""
    from distrib_la import matmul
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit(out_shardings=face)
    def evaluate(s, b, poles, counts, v):
        active = jnp.arange(b.shape[-1])[None, :] < counts[:, None]
        weights = jnp.where(active, 1/(s-poles), 0)
        return v+matmul(b*weights[:, None, :], b, mesh=mesh,
                        backend="distributed", transb="C")
    return evaluate


@lru_cache(maxsize=None)
def _realized_gamma_body(mesh, realize):
    """One admitted all-P executable for V + Pi_G Wc at complex z².

    Only residue endpoints transform under antiunitary operations. Taking
    the conjugate of a completed W(z²) would conjugate its causal scalar
    coefficient as well; its same-time transpose is the required partner.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import transpose_xy
    face = NamedSharding(mesh, P(None, "x", "y"))
    raw = _gamma_body(mesh)

    @jax.jit(out_shardings=face)
    def evaluate(s, b, poles, counts, v):
        wc = raw(s, b, poles, counts, jnp.zeros_like(v))
        partner = jax.lax.with_sharding_constraint(transpose_xy(wc, mesh), face)
        wc, _ = realize(wc, partner)
        return v + wc
    return evaluate


def refuse_unsupported_shared_pole_head(config, *, trs_allowed, nspinor):
    """Refuse, at input resolution, a full shared-pole head the Gamma body cannot evaluate.

    Called once from ``gw_jax._load_system_inputs`` right after the WFN
    symmetry is final and before any basis, bank or constructor exists.  The
    same refusal used to fire inside :func:`build_shared_pole_head`, after
    the bank, moments and constructor had run and ``model.h5`` was committed,
    so a rerun in that directory then refused on the committed model (Q0HEAD
    2026-09-16).  ``trs_allowed`` is the final ``SymMaps.trs_allowed`` the
    store's representation follows; ``nspinor`` is the source WFN's. The
    bare-transverse shared-pole route stores its charge operator on the
    four-component carrier, so its preflight must use that store extent.
    """
    from .gw_config import HeadCorrection, uses_bare_transverse_shared_pole
    if (config.sigma.w_model != "shared_pole"
            or config.head.correction is not HeadCorrection.FULL):
        return
    store_nspinor = (4 if (bool(getattr(config, "bispinor", False))
                          and uses_bare_transverse_shared_pole(config)) else nspinor)
    _refuse_head_representation(trs_allowed=trs_allowed, nspinor=store_nspinor)


def _refuse_head_representation(*, trs_allowed, nspinor):
    """The one owner of what the full Gamma head evaluates: the TRS-even charge body.

    ``nspinor`` 1 or 2 is the charge representation (module docstring): the
    two-component store holds the same spin-traced charge operator and its
    head vertices trace the spinor index, so both evaluate identically. The
    four-component charge carrier has no certified scalar wing/body fold.
    The full photon layout has a separate Gamma completion.
    """
    remedy = ("On this store head_correction = no_local_fields delivers the direct "
              "frequency-dependent head (exact without time reversal; the k sum runs over "
              "the full zone under the measured magnetic group), which is the ordered head "
              "the owner asked for (2026-09-21); head_correction = off is the headless "
              "brute-grid development mode (owner policy 2026-09-18).")
    if not bool(trs_allowed):
        raise ValueError(
            "GATE shared_pole_head_ordered: got time-reversal-broken symmetry (an ordered "
            "store, representation scalar-ordered-ph) with head_correction = full; want "
            "scalar-trs-even-s for full; why: full folds the wings through the Gamma body, "
            "and the body evaluator is the time-reversal-even form b (s - Lambda)^-1 b^dagger, "
            "not the signed particle-hole model an ordered store declares; the signed "
            "wing/body fold is not implemented (owner scope 2026-09-21, wings and body "
            "fold deferred). " + remedy)
    if int(nspinor) not in (1, 2):
        raise ValueError(
            f"GATE shared_pole_head_nspinor: got N_spinor = {int(nspinor)}; want 1 or 2 (the "
            "certified full-head representations); why: the four-component charge "
            "carrier has no certified scalar wing/body fold. Use head_correction = "
            "no_local_fields for its direct Gamma head, or head_correction = off "
            "for a headless development deck.")


def shared_pole_head_plan(config, recipe, *, material_class):
    """Use the MPA sample-plan owner on the current logical energy span."""
    from .mpa.model import make_mpa_plan
    return make_mpa_plan(config, SimpleNamespace(x_max=recipe["census"]["energy_span_ry"]),
                         material_class=material_class)


def build_shared_pole_head(handle, header, V_q, wfns, meta, config, *,
                           mesh_xy, wfn, response, head_resolver, plan,
                           material_class, occupation_state):
    """Return a small MPA head fit and current-map scalar head samples.

    The total Gamma body is V + Pi_G[b(z²-Lambda)^-1 b†], in Ry. Its factor
    matrix and every mu² result are sharded over both processor axes.
    The fit uses unbroadened complex sample coordinates; the existing MPA
    Sigma-head consumer owns its causal evaluation convention.
    """
    from file_io.slab_io import SlabIO
    from file_io.shared_pole_store import read_shared_pole_matrix
    from .gw_config import HeadCorrection
    from .mpa.model import fit_head_samples
    from .mpa.sample_plan import plan_z
    from .qsgw_head import IterationHeadSamples, finalize_iteration_head_sample
    from .response_bank import _reserve
    from .qgrid_symmetry import shared_pole_operator_realizer
    import distrib_la

    if config.head.uses_bgw_metal_q0shift:
        raise ValueError("GATE shared_pole_head: shifted finite-q BGW head is not a Gamma head")
    if plan is None:
        plan = shared_pole_head_plan(config, meta.shared_pole_recipe,
                                      material_class=material_class)
    z = np.asarray(plan_z(plan), np.complex128)
    points = tuple(map(complex, z)) if response is None else tuple(response.omegas)
    if response is not None and not np.array_equal(np.asarray(points[:len(z)]), z):
        raise ValueError("GATE shared_pole_head: response and MPA scalar sample grids differ")
    full = config.head.correction is HeadCorrection.FULL
    if full and response is None:
        raise ValueError("GATE shared_pole_head: full head requires the direct response and wings")
    ledger = meta.shared_pole_capacity
    ambient = ledger.live_stages
    samples = []
    if full:
        if header is None:
            from file_io.shared_pole_store import validate_shared_pole_model
            header = validate_shared_pole_model(handle["path"], expected_identity=handle["identity"],
                mesh_xy=mesh_xy, capacity=ledger)
        # Decks are refused at input resolution (refuse_unsupported_shared_pole_head);
        # this re-reads the same owner on the store actually opened.
        _refuse_head_representation(
            trs_allowed=str(header.get("representation", "")) == "scalar-trs-even-s",
            nspinor=int(header["nspinor"]))
        parents = np.flatnonzero(np.asarray(header["q_irr_full_idx"]) == 0)
        if len(parents) != 1:
            raise ValueError("GATE shared_pole_head: expected one Gamma parent")
        # The hardware ledger can admit a scaling WARN above 3U. That
        # does not waive the individual matrix bound for the new projector.
        # Refuse before reading factors or compiling its numerical work.
        # The ledger's unit is U = 16 Q (N_spinor N_mu)^2 / P, so the store's
        # (Q, N_spinor, N_mu) must reproduce it; the body matrix the
        # projector bounds is the spin-traced n_mu x n_mu charge operator on
        # every admitted store, hence the spin-free logical bound.
        logical = (16*int(header["n_q_full"])*int(header["n_mu_logical"])**2
                   / mesh_xy.size)
        unit = int(header["nspinor"])**2 * logical
        projection_bytes = 16*meta.mu_basis.n_packed**2 // mesh_xy.size
        if ledger.U_bytes_per_rank != unit:
            raise ValueError("GATE shared_pole_head_capacity: store/current-map geometry mismatch")
        if projection_bytes > logical:
            raise ValueError(
                "GATE shared_pole_head_capacity: Gamma projection exceeds the all-P "
                f"logical matrix bound ({projection_bytes} > {logical} bytes per rank)")
        iq = int(parents[0])
        realize = shared_pole_operator_realizer(meta, header,
            q_full_idx=np.asarray([0]), mesh_xy=mesh_xy)
        with SlabIO(handle["path"], mode="r", mesh=mesh_xy) as io:
            b, poles, counts = read_shared_pole_matrix(io, (iq, iq+1), meta=meta, header=header)
        # The linalg service owns the distributed rectangular products and
        # workspace estimate. No q-local whole-matrix copy is introduced.
        algebra = distrib_la.plan("solve_lu", mesh_xy, backend="distributed", n=b.shape[1])
        evaluate = _realized_gamma_body(mesh_xy, realize)
        from symmetry_maps import QirrOperator
        # q = 0 is its own orbit: its wedge row is the full-zone row.
        args = (b, poles, counts, QirrOperator.of(V_q).representative_row(0)[None])
        stats = evaluate.lower(jnp.asarray(1j, jnp.complex128), *args).compile().memory_analysis()
        if stats is None:
            raise ValueError("GATE shared_pole_head: matrix evaluation memory unavailable")
        resident = int(b.size*b.dtype.itemsize//mesh_xy.size + poles.size*8)
        native = distrib_la.workspace_bytes_per_rank(algebra, "gemm",
            (b.shape, (b.shape[0], b.shape[2], b.shape[1])), np.complex128)
        name, _ = _reserve(meta, "head", resident,
            3*stats.output_size_in_bytes+stats.temp_size_in_bytes+native)
        ledger.live_stages = ambient+(name,)
    for index, point in enumerate(points):
        total = None
        if full:
            value = evaluate(jnp.asarray(point*point, jnp.complex128), *args)
            total = value[0]
        samples.append(head_resolver.at(point) if response is None else
            finalize_iteration_head_sample(response, index, total,
                wfn=wfn, meta=meta, config=config, mesh=mesh_xy))
        del total
        if full:
            del value
    if full:
        del args, b, poles, counts
    ledger.live_stages = ambient
    head = fit_head_samples(samples[:len(z)], z, int(config.mpa.n_poles),
        model="qsgw_schur_"+config.mpa.pole_solver if full else "dft_direct_"+config.mpa.pole_solver,
        solve=config.mpa.pole_solver, occupation_state=occupation_state)
    head.update(identity=handle["identity"], body_digest=handle["digest"], completion=True)
    iteration = IterationHeadSamples(omegas=points, samples=tuple(samples),
        sigma_energies_ry=np.asarray(wfns.enk[:, wfns.slices.sigma]),
        sigma_occupations=np.asarray(wfns.occ[:, wfns.slices.sigma]) if occupation_state is None else
            np.asarray(occupation_state.f_kn[:, wfns.slices.sigma]),
        efermi_ry=float(meta.shared_pole_recipe["census"]["mu_ry"]))
    return head, iteration
