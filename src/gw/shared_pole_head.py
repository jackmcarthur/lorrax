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
takes the same fold with its signed forms.  Its Gamma body is the store's
normalization at ``-q = q``,

    Wc(z) = b D+(z) b^H - [b D-(z) b^H]^T,  D+-(z) = 1/(2 W (z -+ W)),  W = sqrt(Lambda),

evaluated at ``z`` (not ``z^2``; real ``b`` gives back ``b (z^2 - Lambda)^-1
b^H``), and its wings take the Lehmann weight of every ordered pair
(``gw.qsgw_head._head_wing_interband_weight``), which keeps the odd-in-z
part time reversal removes.  The Cartesian tensor ``S_ab(z)`` of
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
full zone unfolded by the measured magnetic group.  The scalar head stays
even in ``z``: the Lehmann form gives ``chi_ab(-z) = chi_ba(z)`` for the
tensor, the wings (``Y(-z) = Z(z)^T``) and the body (``W(-z) = W(z)^T``), so
``S_eff(-z) = S_eff(z)^T`` and ``q.S_eff.q`` with real ``q`` is even; the
``s = z^2`` MPA head fit applies unchanged.  An ordered METAL folds the same
way: its wing pairs split by the S kernel's Fermi-surface share, with the
intraband wing taking it (``qsgw_head._head_wing_kernel_face``), and the
plan's shifted origin is the static Thomas-Fermi row
(``qsgw_head.metal_static_rows``).  The signed fold is checked in
``tests/test_shared_pole_head_ordered_fold.py``, the ordered metal in
``tests/test_shared_pole_head_ordered_metal.py``.
"""
from types import SimpleNamespace
from functools import lru_cache

import sys

import jax
import jax.numpy as jnp
import dataclasses

import numpy as np
from common import timing


def _gamma_gemm_route(mesh, b_shape, budget_bytes):
    """The Gamma body GEMM's ``(backend, batched_route)`` for ``distrib_la.matmul``.

    The staged batch route (each rank's ``ceil(Q/P)`` whole matrices, one local
    GEMM) where it fits beside the ledger's live stages (``distrib_la.matmul.staged_fits``,
    the test ``matmul(batched_route='auto', budget_bytes=)`` makes); otherwise the
    distributed provider on the faces. Each priced by its program compiled (never run).
    Q is one Gamma parent, so the staged route puts whole N_mu x N_mu matrices
    on every rank (zero pads past the first): 66 GB/rank at N_mu 30k.
    ``budget_bytes`` is the same on every rank, so is the route; the provider
    is ``panel_matmul``, an XLA program that opens no vendor context."""
    import distrib_la
    from lxkit import memory
    from distrib_la import route_bytes, staged_fits
    q, n, k = (int(v) for v in b_shape)
    ranks, shape = int(mesh.size), (q, n, k)
    fits = staged_fits(mesh, shape, shape, "complex128", "N", "C", int(budget_bytes))
    staged = dataclasses.replace(memory.priced(route_bytes(mesh, shape, shape, "complex128", "N", "C"),
                                               "staged GEMM program (compiled)"), whole=min(n, k))
    faces = memory.priced(0, "face GEMM program (compiled)", ranks=ranks, spread=ranks * route_bytes(
        mesh, shape, shape, "complex128", "N", "C", route="auto", backend="auto"))
    if memory.admit(f"shared-pole head Gamma GEMM {q} x {n} x {k}",
                    (("staged", staged, int(budget_bytes) if fits else 0),
                     ("faces", faces)), int(budget_bytes)) == "staged":
        return "off", distrib_la.ROUTE_BATCH_RESHARD
    return "auto", "auto"


@lru_cache(maxsize=None)
def _gamma_body(mesh, route, ordered=False):
    """Raw latent body at complex ``z``; current factors are operands of one retained callable.

    ``route`` is the GEMM's ``(backend, batched_route)`` (:func:`_gamma_gemm_route`).
    TRS-even store: ``V + b (z^2 - Lambda)^-1 b^H``.  Ordered store (module
    docstring): ``V + b D+ b^H - [b D- b^H]^T``, the second GEMM transposed in
    its faces as ``mpa.sigma.shared_pole_static_wc`` does at ``-q = q``."""
    from distrib_la import matmul
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import transpose_xy
    face = NamedSharding(mesh, P(None, "x", "y"))
    backend, batched_route = route

    @jax.jit(out_shardings=face)
    def evaluate(z, b, poles, counts, v):
        active = jnp.arange(b.shape[-1])[None, :] < counts[:, None]
        wc = lambda weights: matmul(b*jnp.where(active, weights, 0)[:, None, :], b, mesh=mesh,
                                    backend=backend, batched_route=batched_route, transb="C")
        if not ordered:
            return v + wc(1/(z*z-poles))
        w = jnp.sqrt(poles)
        minus = jax.lax.with_sharding_constraint(transpose_xy(wc(1/(2*w*(z+w))), mesh), face)
        return v + wc(1/(2*w*(z-w))) - minus
    return evaluate


@lru_cache(maxsize=None)
def _realized_gamma_body(mesh, realize, route, ordered=False):
    """One admitted all-P executable for V + Pi_G Wc at complex z.

    Only residue endpoints transform under antiunitary operations. Taking
    the conjugate of a completed W(z²) would conjugate its causal scalar
    coefficient as well; its same-time transpose is the required partner.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import transpose_xy
    face = NamedSharding(mesh, P(None, "x", "y"))
    raw = _gamma_body(mesh, route, ordered)

    @jax.jit(out_shardings=face)
    def evaluate(z, b, poles, counts, v):
        wc = raw(z, b, poles, counts, jnp.zeros_like(v))
        partner = jax.lax.with_sharding_constraint(transpose_xy(wc, mesh), face)
        wc, _ = realize(wc, partner)
        return v + wc
    return evaluate


def refuse_unsupported_shared_pole_head(config, *, nspinor):
    """Refuse, at input resolution, a full shared-pole head the Gamma body cannot evaluate.

    Called once from ``gw_jax._load_system_inputs`` right after the WFN
    symmetry is final and before any basis, bank or constructor exists.  The
    same refusal used to fire inside :func:`build_shared_pole_head`, after
    the bank, moments and constructor had run and ``model.h5`` was committed,
    so a rerun in that directory then refused on the committed model (Q0HEAD
    2026-09-16).  ``nspinor`` is the source WFN's. The
    bare-transverse shared-pole route stores its charge operator on the
    four-component carrier, so its preflight must use that store extent.
    """
    from .gw_config import HeadCorrection, uses_bare_transverse_shared_pole
    if (config.sigma.w_model != "shared_pole"
            or config.head.correction is not HeadCorrection.FULL):
        return
    store_nspinor = (4 if (bool(getattr(config, "bispinor", False))
                          and uses_bare_transverse_shared_pole(config)) else nspinor)
    _refuse_head_representation(nspinor=store_nspinor)


def _refuse_head_representation(*, nspinor):
    """The one owner of what the full Gamma head evaluates: the scalar charge body.

    ``nspinor`` 1 or 2 is the charge representation (module docstring): the
    two-component store holds the same spin-traced charge operator and its
    head vertices trace the spinor index, so both evaluate identically. The
    four-component charge carrier has no certified scalar wing/body fold.
    The full photon layout has a separate Gamma completion.  Either store
    representation folds, insulator or metal (module docstring).
    """
    if int(nspinor) not in (1, 2):
        raise ValueError(
            f"GATE shared_pole_head_nspinor: got N_spinor = {int(nspinor)}; want 1 or 2 (the "
            "certified full-head representations); why: the four-component charge "
            "carrier has no certified scalar wing/body fold. Use head_correction = "
            "no_local_fields for its direct Gamma head, or head_correction = off "
            "for a headless development deck.")


def shared_pole_head_plan(config, recipe, *, material_class):
    """Use the MPA sample-plan owner on the current logical energy span.

    The span is snapped up onto the build grid (``snap_outward``), so the
    head sample nodes are a function of a grid cell, not of the exact span.
    The head Loewner fit runs at condition ~1e11, and an exact span let one
    ulp of the SC input move every node by one ulp and the fitted head poles
    by ~1e-5 relative: CrI3 8x8 SC map-1 conduction eqp moved ~0.1 meV.
    """
    from .mpa.model import make_mpa_plan
    from .sigma_box_plan import snap_outward
    span = snap_outward(float(recipe["census"]["energy_span_ry"]), 1., +1)
    return make_mpa_plan(config, SimpleNamespace(x_max=span),
                         material_class=material_class)


def build_shared_pole_head(handle, header, V_q, wfns, meta, config, *,
                           mesh_xy, wfn, response, head_resolver, plan,
                           material_class, occupation_state):
    """Return a small MPA head fit and current-map scalar head samples.

    The total Gamma body is V + Pi_G[Wc(z)], in Ry (module docstring). Its factor
    matrix and every mu² result are sharded over both processor axes.
    The fit uses unbroadened complex sample coordinates; the existing MPA
    Sigma-head consumer owns its causal evaluation convention.
    """
    from file_io.shared_pole_store import open_shared_pole_model, read_shared_pole_matrix
    from .gw_config import HeadCorrection
    from .mpa.model import fit_head_samples, head_moment_report
    from .mpa.sample_plan import plan_z
    from .qsgw_head import (IterationHeadSamples, finalize_iteration_head_rows,
                            head_moment_m1, iteration_head_sample_terms,
                            metal_response_nodes)
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
    # A metal's static row takes the body at z = 0, as its head operands do.
    nodes = metal_response_nodes(points, config, metal=material_class == "metal")
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
        ordered = str(header.get("representation", "")) != "scalar-trs-even-s"
        _refuse_head_representation(nspinor=int(header["nspinor"]))
        parents = np.flatnonzero(np.asarray(header["q_irr_full_idx"]) == 0)
        if len(parents) != 1:
            raise ValueError("GATE shared_pole_head: expected one Gamma parent")
        # The hardware ledger can admit a scaling WARN above 3U; the new
        # projector's individual matrix bound warns the same way (owner
        # 2026-10-01: no refusal on a memory figure).  The ledger's unit is U = 16 Q (N_spinor N_mu)^2 / P, so the store's
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
            from lxkit import memory
            memory.admit("shared-pole head Gamma projection (all-P logical bound)",
                         memory.priced(projection_bytes, "closed form"), logical, device=False)
        iq = int(parents[0])
        realize = shared_pole_operator_realizer(meta, header,
            q_full_idx=np.asarray([0]), mesh_xy=mesh_xy)
        with open_shared_pole_model(handle["path"], mesh_xy=mesh_xy) as io:
            b, poles, counts = read_shared_pole_matrix(io, (iq, iq+1), meta=meta, header=header)
        # The linalg service owns the distributed rectangular products and
        # workspace estimate. The GEMM route is a capacity decision against
        # the ledger's room (_gamma_gemm_route), and the reservation prices
        # that route's workspace.
        resident = int(b.size*b.dtype.itemsize//mesh_xy.size + poles.size*8)
        route = _gamma_gemm_route(mesh_xy, b.shape,
                                  ledger.room_bytes_per_rank(ambient) - resident)
        evaluate = _realized_gamma_body(mesh_xy, realize, route, ordered)
        from symmetry_maps import QirrOperator
        # q = 0 is its own orbit: its wedge row is the full-zone row.
        args = (b, poles, counts, QirrOperator.of(V_q).representative_row(0)[None])
        stats = evaluate.lower(jnp.asarray(1j, jnp.complex128), *args).compile().memory_analysis()
        if stats is None:
            raise ValueError("GATE shared_pole_head: matrix evaluation memory unavailable")
        native = distrib_la.matmul_workspace_bytes_per_rank(
            mesh_xy, (b.shape, (b.shape[0], b.shape[2], b.shape[1])), np.complex128,
            backend=route[0], batched_route=route[1])
        name, _ = _reserve(meta, "head", resident,
            3*stats.output_size_in_bytes+stats.temp_size_in_bytes+native)
        ledger.live_stages = ambient+(name,)
    # Each point's Gamma body is resident only while its 3x3 fold is formed;
    # the mini-BZ cell averages of all points then share one draw
    # (finalize_iteration_head_rows), bit-identical to one call per point.
    terms = []
    with timing.section("head.sample_terms", announce=True,
                        label="shared-pole head sample terms"):
        for index, point in enumerate(points):
            total = None
            if full:
                value = evaluate(jnp.asarray(nodes[index], jnp.complex128), *args)
                total = value[0]
            if response is None:
                samples.append(head_resolver.at(point))
            else:
                terms.append(iteration_head_sample_terms(response, index, total,
                    meta=meta, config=config, mesh=mesh_xy))
            del total
            if full:
                del value
        if full:
            del args, b, poles, counts
    if terms:
        rows = [t for t in terms if isinstance(t, dict)]
        with timing.section("head.cell_average", announce=True,
                        label="shared-pole head mini-BZ average"):
            averaged = iter(finalize_iteration_head_rows(
                response, rows, wfn=wfn, meta=meta, config=config))
            samples = [next(averaged) if isinstance(t, dict) else t for t in terms]
    ledger.live_stages = ambient
    if jax.process_index() == 0 and samples:
        origin = min(range(len(points)), key=lambda i: abs(points[i]))
        s0 = samples[origin]
        print(f"  shared-pole head: origin sample z={complex(points[origin]):.6g} Ry, "
              f"W^c={complex(s0.wcoul0) - complex(s0.vc0):.9g} Ry bohr^3 "
              f"(<v>={complex(s0.vc0).real:.9g})", file=sys.stderr, flush=True)
        if getattr(s0, "S_cart", None) is not None:
            _S = np.real(np.asarray(s0.S_cart))
            print("  shared-pole head: origin interband 8 pi qhat.S.qhat principal = "
                  + "/".join(f"{x:.6g}" for x in 8.0 * np.pi * np.linalg.eigvalsh(0.5 * (_S + _S.T))),
                  file=sys.stderr, flush=True)
    with timing.section("head.mpa_fit", announce=True,
                        label="shared-pole head pole fit"):
        # The rows past the fit nodes include the imaginary-axis moment rule,
        # folded through this same exact Gamma body (qsgw_head.head_moment_m1).
        head = fit_head_samples(samples[:len(z)], z, int(config.mpa.n_poles),
            model="qsgw_schur_"+config.mpa.pole_solver if full else "dft_direct_"+config.mpa.pole_solver,
            solve=config.mpa.pole_solver, occupation_state=occupation_state,
            moment_m1=head_moment_m1(samples, points, metal=material_class == "metal"))
    if jax.process_index() == 0:
        print(head_moment_report(head), file=sys.stderr, flush=True)
    head.update(identity=handle["identity"], body_digest=handle["digest"], completion=True)
    iteration = IterationHeadSamples(omegas=points, samples=tuple(samples),
        sigma_energies_ry=np.asarray(wfns.enk[:, wfns.slices.sigma]),
        sigma_occupations=np.asarray(wfns.occ[:, wfns.slices.sigma]) if occupation_state is None else
            np.asarray(occupation_state.f_kn[:, wfns.slices.sigma]),
        efermi_ry=float(meta.shared_pole_recipe["census"]["mu_ry"]))
    return head, iteration
