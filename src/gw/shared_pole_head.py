"""MPA scalar head coupled to each map's current shared-pole body W.

Only Gamma is evaluated, one frequency at a time. The common head owner
folds its wings and the common MPA owner fits the resulting scalar samples.
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
    face = NamedSharding(mesh, P(None, "x", "y"))
    raw = _gamma_body(mesh)

    @jax.jit(out_shardings=face)
    def evaluate(s, b, poles, counts, v):
        wc = raw(s, b, poles, counts, jnp.zeros_like(v))
        partner = jax.lax.with_sharding_constraint(jnp.swapaxes(wc, -2, -1), face)
        wc, _ = realize(wc, partner)
        return v + wc
    return evaluate


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
        parents = np.flatnonzero(np.asarray(header["q_irr_full_idx"]) == 0)
        if len(parents) != 1:
            raise ValueError("GATE shared_pole_head: expected one Gamma parent")
        # The hardware ledger can admit a scaling WARN above 3U. That
        # does not waive the individual matrix bound for the new projector.
        # Refuse before reading factors or compiling its numerical work.
        unit = (16*int(header["n_q_full"])*int(header["n_mu_logical"])**2
                / mesh_xy.size)
        projection_bytes = 16*meta.mu_basis.n_packed**2 // mesh_xy.size
        if int(header["nspinor"]) != 1 or ledger.U_bytes_per_rank != unit:
            raise ValueError("GATE shared_pole_head_capacity: store/current-map geometry mismatch")
        if projection_bytes > unit:
            raise ValueError(
                "GATE shared_pole_head_capacity: Gamma projection exceeds the all-P "
                f"logical matrix bound ({projection_bytes} > {unit} bytes per rank)")
        iq = int(parents[0])
        realize = shared_pole_operator_realizer(meta, header,
            q_full_idx=np.asarray([0]), mesh_xy=mesh_xy)
        with SlabIO(handle["path"], mode="r", mesh=mesh_xy) as io:
            b, poles, counts = read_shared_pole_matrix(io, (iq, iq+1), meta=meta, header=header)
        # The linalg service owns the distributed rectangular products and
        # workspace estimate. No q-local whole-matrix copy is introduced.
        algebra = distrib_la.plan("solve_lu", mesh_xy, backend="distributed", n=b.shape[1])
        evaluate = _realized_gamma_body(mesh_xy, realize)
        args = (b, poles, counts, V_q[:1])
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
