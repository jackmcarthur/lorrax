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

An ordered store (measured broken time reversal, representation
``scalar-ordered-ph``) declares the signed particle-hole model of the
shared-pole page (SP 2); at Gamma, where ``q = -q``, its body is

    W_c(z) = sum_j [ H_j / (2 Omega_j (z - Omega_j)) - H_j^T / (2 Omega_j (z + Omega_j)) ],
    H_j = b_j b_j^dagger,

i.e. ``P(z) + Q(z)^T`` with ``P = b diag(1/(2 Omega (z - Omega))) b^dagger`` and
``Q = b diag(-1/(2 Omega (z + Omega))) b^dagger``.  It satisfies
``W_c(-z) = W_c(z)^T``; with real residues (``H_j = H_j^T``, the
magnetisation-to-zero limit) it is the even body ``b (z^2 - Lambda)^-1
b^dagger``, and for complex residues the even body misweights the
anti-Hermitian channel ``i Im H_j`` by ``1`` where ``z/Omega_j`` belongs.
The wings follow the same rule (``gw.qsgw_head.head_wings_sharded``,
``trs_allowed=False``): the complete signed Lehmann sum, with
``Y(-z) = -Z(z)^T``.  The direct head ``S`` needs no time-reversal
assumption, and the folded scalar ``q.S_eff(z).q`` is even in ``z`` on an
ordered store because ``S_eff(-z) = S_eff(z)^T``, so the head fit and its
Sigma consumer are unchanged.  Certified by
``tests/test_shared_pole_head_ordered.py``.
"""
from types import SimpleNamespace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


def _factor_product(b, weights, mesh):
    """``b diag(weights) b^dagger`` through the distributed rectangular GEMM."""
    from distrib_la import matmul
    return matmul(b*weights[:, None, :], b, mesh=mesh, backend="distributed", transb="C")


def _ordered_weights(z, poles, counts):
    """Particle and hole weights of the signed Gamma body at complex ``z``.

    ``poles`` are ``Omega_j^2`` (Ry^2) with inert sentinels beyond ``counts``.
    Returns ``p_j = 1/(2 Omega_j (z - Omega_j))`` and
    ``q_j = -1/(2 Omega_j (z + Omega_j))`` on active slots, zero elsewhere.
    """
    active = jnp.arange(poles.shape[-1])[None, :] < counts[:, None]
    omega = jnp.sqrt(jnp.where(active, poles, 1.0))
    particle = jnp.where(active, 1/(2*omega*(z-omega)), 0)
    hole = jnp.where(active, -1/(2*omega*(z+omega)), 0)
    return particle, hole


@lru_cache(maxsize=None)
def _gamma_body(mesh):
    """Raw latent body; current factors are operands of one retained callable."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit(out_shardings=face)
    def evaluate(s, b, poles, counts, v):
        active = jnp.arange(b.shape[-1])[None, :] < counts[:, None]
        weights = jnp.where(active, 1/(s-poles), 0)
        return v+_factor_product(b, weights, mesh)
    return evaluate


@lru_cache(maxsize=None)
def _ordered_gamma_halves(mesh):
    """Raw signed particle and hole products ``P(z)``, ``Q(z)`` of an ordered store."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh, P(None, "x", "y"))

    @jax.jit(out_shardings=(face, face))
    def evaluate(z, b, poles, counts):
        particle, hole = _ordered_weights(z, poles, counts)
        return _factor_product(b, particle, mesh), _factor_product(b, hole, mesh)
    return evaluate


@lru_cache(maxsize=None)
def _ordered_gamma_body(mesh):
    """Raw signed Gamma body ``V + P(z) + Q(z)^T`` at complex ``z`` (module docstring)."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh, P(None, "x", "y"))
    halves = _ordered_gamma_halves(mesh)

    @jax.jit(out_shardings=face)
    def evaluate(z, b, poles, counts, v):
        particle, hole = halves(z, b, poles, counts)
        return v+particle+jax.lax.with_sharding_constraint(jnp.swapaxes(hole, -2, -1), face)
    return evaluate


@lru_cache(maxsize=None)
def _realized_gamma_body(mesh, realize):
    """One admitted all-P executable for V + Pi_G Wc at complex z².

    Only residue endpoints transform under antiunitary operations. Taking
    the conjugate of a completed W(z²) would conjugate its causal scalar
    coefficient as well; its same-time transpose is the required partner.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh, P(None, "x", "y"))
    raw = _gamma_body(mesh)

    @jax.jit(out_shardings=face)
    def evaluate(s, b, poles, counts, v):
        wc = raw(s, b, poles, counts, jnp.zeros_like(v))
        partner = jax.lax.with_sharding_constraint(jnp.swapaxes(wc, -2, -1), face)
        wc, _ = realize(wc, partner)
        return v + wc
    return evaluate


@lru_cache(maxsize=None)
def _realized_ordered_gamma_body(mesh, realize):
    """One admitted all-P executable for V + Pi_G [P(z) + Q(z)^T] at complex z.

    The antiunitary partner of the signed body is again its same-time
    transpose, ``P(z)^T + Q(z)``: every residue endpoint transposes and no
    scalar coefficient is conjugated.  It is assembled from the two raw
    halves so that each half is transposed once.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    face = NamedSharding(mesh, P(None, "x", "y"))
    halves = _ordered_gamma_halves(mesh)

    @jax.jit(out_shardings=face)
    def evaluate(z, b, poles, counts, v):
        particle, hole = halves(z, b, poles, counts)
        transpose = lambda m: jax.lax.with_sharding_constraint(jnp.swapaxes(m, -2, -1), face)
        wc, _ = realize(particle + transpose(hole), hole + transpose(particle))
        return v + wc
    return evaluate


def refuse_unsupported_shared_pole_head(config, *, nspinor):
    """Refuse, at input resolution, a full shared-pole head the Gamma body cannot evaluate.

    Called once from ``gw_jax._load_system_inputs`` right after the WFN
    symmetry is final and before any basis, bank or constructor exists.  The
    same refusal used to fire inside :func:`build_shared_pole_head`, after
    the bank, moments and constructor had run and ``model.h5`` was committed,
    so a rerun in that directory then refused on the committed model (Q0HEAD
    2026-09-16).  ``nspinor`` is the WFN's, 1 or 2 for the charge
    representation this head evaluates.  Both time-reversal verdicts are
    admitted: an even store evaluates ``b (s - Lambda)^-1 b^dagger`` and an
    ordered store the signed body of the module docstring.
    """
    from .gw_config import HeadCorrection
    if (config.sigma.w_model != "shared_pole"
            or config.head.correction is not HeadCorrection.FULL):
        return
    _refuse_head_representation(representation="scalar-trs-even-s", nspinor=nspinor)


_HEAD_REPRESENTATIONS = ("scalar-trs-even-s", "scalar-ordered-ph")


def _refuse_head_representation(*, representation, nspinor):
    """The one owner of what the full Gamma head evaluates: the even or ordered charge body.

    ``nspinor`` 1 or 2 is the charge representation (module docstring): the
    two-component store holds the same spin-traced charge operator and its
    head vertices trace the spinor index, so both evaluate identically.  An
    N_spinor = 4 store is the kinetic-balance bispinor lift with a photon
    layout; its Gamma completion is the packed photon head, not this scalar
    charge head.  ``representation`` selects the even or the signed
    evaluator; anything else is not a scalar charge model.
    """
    if str(representation) not in _HEAD_REPRESENTATIONS:
        raise ValueError(
            f"GATE shared_pole_head_representation: got {representation!r}; want one of "
            f"{_HEAD_REPRESENTATIONS}; why: the scalar charge head evaluates the "
            "time-reversal-even body or the signed particle-hole body of a scalar charge "
            "store and nothing else.")
    if int(nspinor) not in (1, 2):
        raise ValueError(
            f"GATE shared_pole_head_nspinor: got N_spinor = {int(nspinor)}; want 1 or 2 (the "
            "charge representation); why: this scalar charge head evaluates the spin-traced "
            "charge body, and an N_spinor = 4 store is the bispinor lift whose Gamma "
            "completion is the packed photon head (gw.photon_sigma), not this one. A one-shot "
            "deck runs with head_correction = off.")


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
        representation = str(header.get("representation", ""))
        _refuse_head_representation(
            representation=representation, nspinor=int(header["nspinor"]))
        ordered = representation == "scalar-ordered-ph"
        if ordered and bool(getattr(response, "trs_allowed", True)):
            raise ValueError(
                "GATE shared_pole_head_wings: got an ordered store (scalar-ordered-ph) with "
                "time-reversal-even head wings; want wings built under the same measured "
                "trs_allowed = False verdict (head_wings_sharded, ordered Lehmann sum); why: "
                "the even pair sum cancels the magnetisation-odd wing channel that the signed "
                "Gamma body carries, so the fold would mix two time-reversal models.")
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
        # The even body is a function of s = z^2; the signed body of an
        # ordered store keeps the sign of z (module docstring).
        evaluate = (_realized_ordered_gamma_body if ordered else _realized_gamma_body)(
            mesh_xy, realize)
        coordinate = (lambda z: z) if ordered else (lambda z: z*z)
        args = (b, poles, counts, V_q[:1])
        stats = evaluate.lower(jnp.asarray(coordinate(1j), jnp.complex128), *args).compile().memory_analysis()
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
            value = evaluate(jnp.asarray(coordinate(point), jnp.complex128), *args)
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
