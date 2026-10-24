"""Physical non-TDA ladder density response, W_BSE(z)-v = C (z-H)^-1 B.

At stored +q the two branches are R=(c[k+q],v[k]) and
A=(v[k+q],c[k]), with distinct vertices F_R and F_A. The injection
is [F_R; -F_A]v and the readout is v[F_R^dag,F_A^dag]. Their outer
product supplies all four Hartree blocks. Each statically screened direct
block uses the same physical encode/W-convolution/decode contraction.
No TRS sewing, real-orbital assumption or projection of W(z) is used.

The historical folded RPA representation remains the unchanged default
when include_w=False. A physical payload can also use a zero direct rung
to test the explicit-branch RPA/Dyson identity. Both representations use the
same GMRES, FFT, seed and readout owners and keep pair/centroid arrays
sharded over all processors.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Optional

import numpy as np

# Canonical JAX GPU/CPU bootstrap — single-sourced in runtime.bootstrap().
# MUST run before this module's own `import jax` (same rule as bse_w_exact).
from runtime import bootstrap
bootstrap()

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .bse_feast import (ensure_W_R, build_preconditioner_diagonal_sharded,
                        ladder_matvec_operands, matvec_operands,
                        _get_deflation_arnoldi)
from .bse_io import load_bse_data_from_restart_sharded
from .bse_ring_comm import (
    build_bse_ring_matvec_full,
    build_realspace_random_transition_generator,
    build_density_snapshot_operator,
    make_bse_shardings,
)
from .bse_w_exact import (
    _build_rpa_resolvent,
    apply_screening_resolvent_block,
    build_probe_rhs,
    build_finite_q_data,
    _symmetry_tables,
    _get_block_moment_actions,
    _project_screening_actions,
    _get_block_gmres_solver,
    _get_probe_rhs_program,
    _get_branch_projection_program,
)
from common.collectives import gather_to_host

LADDER_RESPONSE_CONVENTION = "physical-ra-exact-static-head-v2"
LADDER_INTERNAL_HEAD_POLICY = "screened-transfer-gamma-exact-diagonal-v1"


def ladder_static_head_identity(restart_path, input_file, *, cell_volume):
    """Bind the exact internal rung head, separate from the dense W0 body.

    Uses the loader's scalar/override and optical-diagonal owners. Head
    values are Ry bohr^3, volume is bohr^3, and the applied shift is Ry.
    Only the small G=0 vector is hashed on the host; no W/psi is loaded.
    """
    import hashlib
    from file_io.restart_bundle import read_head_metadata
    from .bse_head import _resolve_head_params, q0_head_diagonal

    header = read_head_metadata(restart_path)
    _, whead, volume, _ = _resolve_head_params(
        input_file, header["bare_head"], header["screened_head"], cell_volume)
    if whead is None or volume is None or header["head_vector"] is None:
        raise ValueError("GATE ladder_static_head_identity: missing screened head, volume or G=0 vector")
    value = complex(np.asarray(whead).reshape(-1)[0])
    volume = float(volume)
    grid = tuple(map(int, np.asarray(header["grid"]).reshape(-1)))
    if (not np.isfinite((value.real, value.imag, volume)).all()
            or volume <= 0 or len(grid) != 3 or min(grid) <= 0):
        raise ValueError("GATE ladder_static_head_identity: invalid head, volume or k grid")
    g0 = np.ascontiguousarray(header["head_vector"], dtype=np.complex128)
    return dict(policy=LADDER_INTERNAL_HEAD_POLICY, q0_head="diagonal",
                whead_ry_bohr3=[value.real, value.imag], cell_volume_bohr3=volume,
                kgrid=list(grid), q0_head_ry=q0_head_diagonal(value, volume, np.prod(grid)),
                g0_shape=list(g0.shape), g0_sha256=hashlib.sha256(g0.tobytes()).hexdigest())

jax.config.update("jax_enable_x64", True)


def build_ladder_resolvent(mesh_xy: Mesh, data: dict, *, include_w: bool = True,
                           fuse_ladder_rung: bool = True):
    """Build the physical R/A operator and its shared seed/solve/readout stack.

    The bare RPA default retains its original representation; explicit
    physical payloads also permit include_w=False for the RPA identity gate.
    W_R depends on k-k' only and is prepared once, outside the q sweep.
    """
    if not include_w and not data.get("physical_branches", False):
        return _build_rpa_resolvent(mesh_xy, data)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    if not data.get("physical_branches", False):
        data.update(build_finite_q_data(data, (0, 0, 0), mesh_xy, physical_branches=True))
    ensure_W_R(data, include_W=include_w, mesh_xy=mesh_xy)
    # ``fuse_ladder_rung`` is a COST switch, not a physics one: it sums the two
    # rung T intermediates of each block row before the (mu, nu, s, s, k) FFT
    # chain instead of after, so the matvec runs that chain twice instead of
    # four times (bse_ring_comm._impl_core_fused).  Same operator to rounding;
    # False is the A/B arm.
    matvec = build_bse_ring_matvec_full(
        mesh_xy, nkx, nky, nkz, include_W=include_w, screening=True,
        physical_branches=True, fuse_ladder_rung=fuse_ladder_rung)
    diag_h = build_preconditioner_diagonal_sharded(
        data, mesh_xy, include_W=include_w, use_tda=False)
    gen = build_realspace_random_transition_generator(
        mesh_xy, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"]))
    snapshot = build_density_snapshot_operator(
        mesh_xy, nkx, nky, nkz, scatter_nu_on_y=True)
    return matvec, diag_h, gen, snapshot, make_bse_shardings(mesh_xy)


def refuse_chain_path(include_w: bool) -> None:
    """Refuse the ``w_omega_chain`` z^2 reduction for the ladder operator.

    Wired-but-refused in the ``head_channel.refuse_if_unimplemented`` style: the
    path is named, the reason is the algebra, and the generalization is a named
    deferral rather than a half-build.
    """
    if not include_w:
        return
    raise NotImplementedError(
        "w_omega_chain is refused for the ladder operator (include_w=True). "
        "Its z^2 reduction assumes A - B = D EXACTLY (w_omega_chain.py:39-46 — "
        "S = D^{1/2}(A+B)D^{1/2}, seed and readout scaled by D^{1/2}), which "
        "holds for the RPA screening operator (A = D + V, B = V) and is broken "
        "by the ladder: A - B = D - W_d + W_d^B. Using the chain here would "
        "silently evaluate a DIFFERENT operator at every frequency, not an "
        "approximation of this one. Physical finite-q R/A states also have "
        "distinct vertices and diagonals; they cannot be identified in a "
        "single N-dimensional real-linear action. The "
        "correct structural certificate is full-space eta-pseudo-Hermiticity "
        "(eta H = (eta H)^dag, eta = diag(I,-I)), not the N-dimensional SDY "
        "kappa metric. A reusable frequency basis must retain the full 2N "
        "operator. The direct path evaluates each "
        "frequency with its own shifted block-GMRES solve "
        "(bse_w_exact.apply_screening_resolvent_block).")


@dataclass(frozen=True)
class WLadderWedge:
    """The ladder ``W(z) - v`` bodies on the irreducible q-wedge.

    ``wc``
        complex128 ``(nz, nq_irr, mu_pad, mu_pad)`` on the PADDED centroid
        extent, sharded ``NamedSharding(mesh_xy, P(None, None, 'x', 'y'))`` —
        the same class as the existing full-BZ ``W_q``, so no rank ever holds a
        whole mu^2 tile.  ``wc[iz, iq, mu, nu]`` is ``W_q(z) - v_q`` for
        ``z = z_list_ry[iz]``, ``q = q_irr_kgrid_int[iq]``.  Head-LESS bodies
        (the resolvent carries no q=0 head by construction; the loader runs
        ``inject_head=False``), in the GW ``chi_0(q)`` vertex convention.  The
        pad rows/columns are ZERO: the probe block is the identity in the padded
        basis and the loader's psi pad is zero, so a pad column solves to zero
        rather than to noise.

        PADDED, NOT PAD-STRIPPED — a deviation from DESIGN_2026-08-15 section 4,
        stated loudly because that document says "pad stripped".  The two halves
        of the design's own sentence are incompatible: ``n_rmu`` is a physical
        centroid count with no divisibility (399 on the gnppm fixture) and
        ``P(None, None, 'x', 'y')`` requires the last two axes to divide the
        mesh, so a stripped tile CANNOT carry the documented sharding on any
        mesh wider than 1x1.  MEASURED 2026-08-15 (JID 57064957, the
        wiring-closure gate's first 2x2 run): ``ValueError: ... global size of
        its dimension 2 should be divisible by 2, but it is equal to 399``.
        Padding wins over sharding-by-slice because the alternative — dropping
        the constraint and letting XLA pick a layout for a 399-wide axis — is a
        replicated mu^2 object per rank, the one thing the scaling doctrine
        forbids.  Nothing downstream loses: the GW consumer's ``_pad_mu``
        re-pads to exactly this extent, ``V_q`` and the unfold tables already
        live there, and ``n_rmu`` below carries the logical count.
    ``n_rmu``
        the LOGICAL centroid count, i.e. where the pad begins.  Carried so a
        consumer that needs the physical sub-tile can slice it on the host side
        of a gather instead of inferring it from a shape.
    ``z_list_ry``
        the complex frequencies in Rydberg, as passed.
    ``q_irr_kgrid_int``
        ``(nq_irr, 3)`` integer kgrid steps, row 0 = Gamma — the SAME canonical
        ``SymMaps`` table the GW producer uses.
    ``irr_idx_q`` / ``sym_idx_q`` / ``q_irr_full_idx``
        the full-BZ unfold tables from ``SymMaps``, carried so the caller can
        hand them straight to ``symmetry_maps.unfold_isdf_operator`` without a
        second symmetry pass.
    ``gmres_resid`` / ``gmres_iters``
        ``(nz, nq_irr, n_probe)`` host arrays — per-COLUMN relative residual and
        while-loop exit index.  The caller gates on these, never on a return
        code: ``iters == gmres_max_iter`` means the column was TRUNCATED at the
        cap, not converged (the rc=0 class, QUALITY_PATTERNS 7).
    ``include_w``
        which operator produced the tiles; stamp it on any artifact these enter
        (QUALITY_PATTERNS 10 — provenance).
    """

    wc: jax.Array
    n_rmu: int
    z_list_ry: np.ndarray
    q_irr_kgrid_int: np.ndarray
    irr_idx_q: np.ndarray
    sym_idx_q: np.ndarray
    q_irr_full_idx: np.ndarray
    gmres_resid: np.ndarray
    gmres_iters: np.ndarray
    include_w: bool
    #: Optional q->0 tensor from the SAME transition manifold and operator.
    #: It is micro-reducible (finite-G Hartree/local fields are already in
    #: the resolvent), so consumers must restore only the omitted macroscopic
    #: v head and must not apply a head/body Schur fold again.
    head_result: object | None = None
    #: Optional materialized exact response fields (None when streamed).
    #: Slopes/reflected values have the wc shape; M0..M3 have only a q axis.
    response_fields: dict | None = None


def _identity_probe_blocks(n_logical: int, n_pad: int, chunk: int):
    """Yield unit-column probes with only one ``(chunk, n_pad)`` host block.

    Rows above the logical centroid count remain zero in the final block.
    Constructing the full identity, or retaining all chunks, would each cost
    ``8 * n_pad**2`` host bytes on every rank even with a small probe chunk.
    """
    for c0 in range(0, n_logical, chunk):
        n_real = min(chunk, n_logical - c0)
        rows = np.arange(n_real)
        probes = np.zeros((chunk, n_pad), dtype=np.float64)
        probes[rows, c0 + rows] = 1.0
        yield c0, n_real, probes


def _probe_scaling_bytes(data, mesh):
    """Shape-only proposal per probe; the executed programs admit the choice."""
    nc, nv = int(data["n_cond_pad"]), int(data["n_val_pad"])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    n, ns = int(data["V_q0"].shape[0]), int(data["psi_c_X"].shape[2])
    px, py, p = int(mesh.shape["x"]), int(mesh.shape["y"]), int(mesh.size)
    pair = nc * nv * nk / p
    # Seed decode and snapshot ring panels scale in the probe axis, whereas
    # GMRES/Arnoldi themselves scan one probe and keep one Krylov workspace.
    return max(8*n*nk/py + 32*nc*nv*nk/py + 32*pair,
               32*nc*n*nk*(ns+1)/p + 32*n*nk/px + 96*pair)


def _plan_probe_chunk(context, requested, *, n_pad, carry_bytes, stack_bytes):
    """Compile the real seed/solve/project/harvest at a bounded proposed width.

    ``carry_bytes`` reserves the facade's not-yet-built response bank. The
    current payload is already resident. Positive requests are priced and
    honored; automatic proposals use the common tile bound, then at most a
    minimum-width and one adjusted-width check. All planning operands are abstract.
    """
    from lxkit import memory
    from runtime.aot_memory import agreed_chunk, check_chunk, compiled_new_bytes
    from runtime.tiles import tile_units
    dq, sh, mesh = context["data"], context["sh"], context["mesh"]
    py, p = int(mesh.shape["y"]), int(mesh.size)
    nc, nv = int(dq["n_cond_pad"]), int(dq["n_val_pad"])
    nk = int(dq["nkx"] * dq["nky"] * dq["nkz"])
    rank = int(context["deflation_rank"])
    slope, moments = bool(context["with_slope"]), bool(context["with_moments"])
    abstract = lambda a: (jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=a.sharding)
                          if isinstance(a, jax.Array) else a)
    args = jax.tree.map(abstract, context["operands"])
    diag = abstract(context["diag"])
    scalar = jax.ShapeDtypeStruct((), jnp.complex128)
    pair_shape = (nc, nv, nk)
    one = jax.ShapeDtypeStruct((2, 1, *pair_shape), jnp.complex128, sharding=sh.X_full)
    physical = bool(dq.get("physical_branches", False))
    seed_program = _get_probe_rhs_program(context["gen"], sh, nk, physical_branches=physical)
    solver = _get_block_gmres_solver(context["matvec"], sh,
        context["max_iter"], context["tol"], jnp.complex128, with_slope=slope,
        retain_branches=physical)
    psi = lambda key, layout: jax.ShapeDtypeStruct(
        dq[key].shape, dq[key].dtype, sharding=layout)
    v = abstract(dq["V_q0"] if context["snapshot_v"] is None else context["snapshot_v"])
    deflation, deflation_bytes, harvest_bytes = None, 0, 0
    if rank:
        u = jax.ShapeDtypeStruct((rank, *one.shape), jnp.complex128,
            sharding=NamedSharding(mesh, P(None, *sh.X_full.spec)))
        c = jax.ShapeDtypeStruct((rank, rank), jnp.complex128,
                                sharding=NamedSharding(mesh, P()))
        deflation = (u, c)
        deflation_bytes = 16*(2*rank*nc*nv*nk//p + rank*rank)
        arnoldi = max(LADDER_DEFLATION_ARNOLDI, 2*rank)
        ex = _get_deflation_arnoldi(context["matvec"], one, n_arnoldi=arnoldi).lower(
            one, diag, scalar, args).compile()
        # Include the subsequent V[:m] -> U contraction and uploaded host
        # factors beside the returned Arnoldi basis, not just its workspace.
        harvest_bytes = compiled_new_bytes(ex) + deflation_bytes + 16*arnoldi*rank
    ndefl = 1 if context["z_outer"] else context["nz"]
    deflation_bytes *= ndefl
    # Eager seed/project callers temporarily request ring-layout psi views.
    # Price their additional carriers; the solve itself takes resting 2-D psi.
    ring_bytes = sum(int(a.size*a.dtype.itemsize) //
        int(mesh.shape["x" if key.endswith("_X") else "y"])
        for key in _PSI_KEYS if isinstance((a := dq.get(key)), jax.Array))
    figures = {}

    def build(units):
        chunk = int(units)*py
        rhs = jax.ShapeDtypeStruct((2, chunk, *pair_shape), jnp.complex128,
                                   sharding=sh.X_full)
        readout = (rhs if physical else jax.ShapeDtypeStruct(
            (chunk, *pair_shape), jnp.complex128, sharding=sh.X))
        probes = jax.ShapeDtypeStruct((chunk, n_pad), jnp.float64, sharding=sh.S_k0)
        seed_args = ((psi("psi_c_A_X", sh.psi_x), psi("psi_v_A_X", sh.psi_x))
                     if physical else ())
        seed = seed_program.lower(probes, psi("psi_c_X", sh.psi_x),
                                  psi("psi_v_X", sh.psi_x), abstract(dq["V_q0"]),
                                  *seed_args).compile()
        solve = solver.lower(rhs, diag, scalar, args, deflation).compile()
        project = (_get_branch_projection_program(context["snapshot"], sh)
                   if physical else context["snapshot"])
        project_args = ((psi("psi_c_A_Y", sh.psi_y), psi("psi_v_A_Y", sh.psi_y))
                        if physical else ())
        snap = project.lower(readout, psi("psi_c_Y", sh.psi_y),
                             psi("psi_v_Y", sh.psi_y), v, *project_args).compile()
        rhs_bytes = 32*chunk*nc*nv*nk//p
        readout_bytes, tile_bytes = rhs_bytes if physical else rhs_bytes//2, 16*n_pad*chunk//p
        snap_new = compiled_new_bytes(snap)
        nread = 3 if slope else 1
        moment_carry = 4*tile_bytes if moments else 0
        costs = [8*chunk*n_pad//py + compiled_new_bytes(seed) + ring_bytes,
                 rhs_bytes + moment_carry + compiled_new_bytes(solve),
                 rhs_bytes + moment_carry + nread*readout_bytes +
                 (nread-1)*tile_bytes + snap_new + ring_bytes]
        if moments:
            moment = context["moment_actions"].lower(rhs, args).compile()
            costs.extend([rhs_bytes + compiled_new_bytes(moment),
                          rhs_bytes + 4*readout_bytes + 3*tile_bytes + snap_new + ring_bytes])
        if rank:
            costs.append(rhs_bytes + moment_carry + one.size*16//p + harvest_bytes)
        # All response matrices remain face-sharded. The facade also holds
        # future completed rows and one non-donated column-placement output.
        figures[int(units)] = max(int(stack_bytes),
            int(carry_bytes + deflation_bytes + max(costs)))
        return solve

    per_probe = _probe_scaling_bytes(dq, mesh)
    top = n_pad//py
    units = min(top, int(requested)//py) if requested else tile_units(per_probe*py, top)
    room = memory.room() if memory.run() else (1 << 30) + int(carry_bytes)
    compiled = build(units)
    extra = lambda count, ex: max(0, figures[count] - compiled_new_bytes(ex))
    if requested:
        price = -agreed_chunk(-figures[units])
        memory.admit("BSE requested probe block", memory.priced(price,
            "compiled seed/GMRES/snapshot/moments/Arnoldi and live carries"), room)
    else:
        price = -agreed_chunk(-figures[units])
        fixed = max(int(stack_bytes), int(carry_bytes)+deflation_bytes)
        per_unit = py*per_probe
        if price > room and units > 1:
            # A single wide-block price cannot separate its fixed Krylov
            # workspace from its scaling outputs. Read the executable at the
            # irreducible width only when needed, then fit that affine floor;
            # otherwise check_chunk would wrongly scale the fixed workspace
            # and could choose another over-budget width when a smaller fits.
            minimum_compiled = build(1)
            floor = -agreed_chunk(-figures[1])
            if floor > room:
                units, compiled, price = 1, minimum_compiled, floor
            else:
                measured_slope = max(0., (price-floor)/(units-1))
                fixed, per_unit = max(fixed, floor-measured_slope), max(per_unit, measured_slope)
                proposed = max(1, min(units-1, int((room-fixed)//max(per_unit, 1.))))
                proposed_compiled = minimum_compiled if proposed == 1 else build(proposed)
                proposed_price = -agreed_chunk(-figures[proposed])
                if proposed_price <= room:
                    units, compiled, price = proposed, proposed_compiled, proposed_price
                else:
                    # XLA can change schedules between widths: its temporary
                    # size need not be affine. Never execute an over-budget
                    # interpolation when an already-compiled width fits.
                    memory.record_route("BSE automatic probe block", "mesh-y minimum",
                        f"compiled width {proposed*py} exceeds room; measured minimum fits")
                    units, compiled, price = 1, minimum_compiled, floor
        choice = check_chunk(units, build=build, compiled=compiled,
            fixed=fixed, per_unit=per_unit, room=room, extra=extra,
            minimum=1, stage="BSE automatic probe block")
        units, price = choice.chunk, choice.price
    memory.book(memory.priced(price, "compiled ladder block and response carries", increment=True),
                stage="bse.probe_block")
    print(f"[w_ladder] {'requested' if requested else 'automatic'} probe chunk {units*py}/{n_pad}; "
          f"compiled block+carry {price/1e9:.3f} GB/rank, room {room/1e9:.3f} GB/rank", flush=True)
    return units*py


def sweep_q_wedge(
    data: dict,
    mesh_xy: Mesh,
    q_list,
    z_list_ry,
    *,
    include_w: bool,
    probe_blocks_for_q: Callable[[int, tuple], Iterable],
    gmres_tol: float,
    gmres_max_iter: int,
    on_result: Callable,
    build_hook: Optional[Callable] = None,
    route: str = "plain",
    deflation_rank: Optional[int] = None,
    z_outer: bool = False,
    on_z_done: Optional[Callable] = None,
    with_slope: bool = False,
    with_moments: bool = False,
    probe_chunk_plan: Optional[Callable] = None,
):
    """Walk the q-wedge x probe-chunk x z-list with ONE compiled engine.

    This is the single per-q loop of the screening resolvent — the one
    ``bse_w_exact.main --compare-wq`` used to carry inline and now calls.  There
    is no second loop: ``--compare-wq`` differs only in
    ``probe_blocks_for_q`` (it probes the largest-``|W0-V|`` columns of each q's
    own disk tile, one block) and in ``on_result`` (it prints a comparison row
    instead of placing a chunk into a tile).

    The engine (``matvec``, ``gen``, ``snapshot``) depends on mesh and k-grid
    ONLY, so it is built ONCE here, outside every loop; the q-dependent operands
    (rolled ``psi_c``/``eps_c``, the ``V_q`` tile, the hoisted ``M``s, the
    preconditioner diagonal) and ``z`` flow into the cached block-GMRES engine
    as runtime arguments.  The first (q, chunk, z) point carries the compile;
    every later one is dispatch.  Rebuilding the engine per q — or per chunk —
    silently pays one full compile each time, because the block-GMRES cache is
    keyed on ``id(matvec)``.

    ``probe_blocks_for_q(iq, q)`` returns an iterable of ``(c0, n_real, G)``:
    ``G`` is the ``(n_probe, n_rmu)`` probe block (``n_probe % py == 0``),
    ``n_real`` how many of its rows are live (the rest are zero pad), and ``c0``
    the caller's own offset, passed back untouched.
    ``on_result(iq, q, iz, z, c0, n_real, W_tile, resids, iters)`` gets the
    DEVICE tile still sharded ``P('x','y')``; ``build_hook(iq, q, dq)`` is
    called once per q after the q-shifted payload exists.

    ``route='lift'`` runs route A — the Hartree ring removed from the Krylov
    problem by Woodbury — and then the tile ``on_result`` receives is
    ``T = Pi v``, NOT ``W - v``: the caller must close it with
    :func:`dyson_close_tile` (``compute_wc_qwedge`` does).  The
    sweep does not close it itself because the close is once per (q, z) over
    the whole assembled tile, i.e. after the probe chunks have been placed.

    ``deflation_rank`` — recycled eigenvectors per ``(q, z)`` (None = the
    measured default for this operator, :data:`LADDER_DEFLATION_RANK` on the
    ladder and 0 on the RPA operator; 0 = the diagonal-only engine).  See
    :func:`_deflation_for`.

    ``z_outer`` walks z OUTSIDE the q wedge (one z at a time: every q of z[0],
    then every q of z[1], ...) and calls ``on_z_done(iz, z)`` after the last q
    of each z, so a caller can hand that z's wedge off and free it before the
    next z starts.  The engine is still built once; the q-shifted payload, the
    preconditioner and the seed are rebuilt per (z, q), which is small next to
    the solve.  Every (q, z) solve sees the same operands either way, so the
    tiles do not depend on the order.  ``False`` is the historical order.

    PSI RESTS 2-D: every psi array is held with the band axis on one mesh
    axis and mu on the other (1/P per rank, :func:`_psi_at_rest_2d`), never
    band-whole and replicated over a sqrt(P) axis.  The jitted GMRES engine
    takes the 2-D operands and its matvec reshards to the ring layout inside
    the program, so the band-whole panels exist only transiently per matvec;
    the eager seed, preconditioner and snapshot get a transient ring view
    (:func:`_psi_ring_view`).  MEASURED against the replicated layout on Si 4^3
    SOC (480 mu, 8v x 20c, 8 q, 64 columns, P4, z = 0): wall +5.0 % warm /
    +1.0 % cold, W bitwise identical (claim 2861).  Chunked band-panel gathers
    inside the ring steps (transient 2/sqrt(P) of the band axis) are the
    matvec owner's follow-on.
    """
    # ``build_ladder_resolvent`` -> ``ensure_W_R(data, ...)`` populates
    # ``data['W_R']`` ONCE, here, OUTSIDE the q loop.  That is a correctness-
    # neutral, cost-material distinction:
    #
    #   * ``W_R = ifft_k(W_q)`` is a function of the INTERNAL momentum k - k'
    #     and of nothing else.  ``build_finite_q_data`` returns ``dict(data)``
    #     with only the conduction slots and ``V_q0`` swapped — it never
    #     touches ``W_q`` — so every q's ``W_R`` is the SAME array, and the
    #     shallow copy already carries it.
    #   * calling ``ensure_W_R`` per q was not merely recomputing it: it goes
    #     through ``bse_io.make_w_densifier``, which builds a FRESH ``@jax.jit``
    #     each call (``bse_densify.py:195``), so the tracing cache missed on
    #     "never seen function: _densify" at every q and the wedge paid n_q + 1
    #     trace+compiles.  MEASURED 2026-08-16 by the HLO/dispatch audit
    #     (``jax_explain_cache_misses``, evidence/hlo_audit/FINDINGS.md F1):
    #     4 jit__densify modules for a 2-q sweep, and exactly one cache miss,
    #     at the second q.
    #
    # This is what makes the module docstring's "the whole q x z sweep
    # dispatches ONE compiled program" true rather than aspirational.  The
    # PRECONDITIONER stays inside the loop: it reads the rolled ``eps_c`` and
    # the finite-q ``V_q0``, so it genuinely is per-q.
    if route not in ("plain", "lift"):
        raise ValueError(f"route must be 'plain' or 'lift', got {route!r}")
    # Explicit physical R/A states require no TRS gauge construction or
    # host gathering. The old folded RPA path remains the control.
    data = _psi_ring_view(data)
    matvec, _, gen, snapshot, sh = build_ladder_resolvent(
        mesh_xy, data, include_w=include_w)
    # ROUTE A (the Woodbury ring lift), as two constant tiles and nothing else.
    # The Hartree ring is the rank-N_mu dyad ``sigma v tau``; removing it from
    # the Krylov problem EXACTLY is (i) zero the ``V_q0`` the MATVEC reads —
    # only ``apply_V_ring_only`` reads it — and (ii) project with the IDENTITY
    # so the tile is ``T = Pi v`` and not ``v Pi v``.  The caller closes with
    # ``W - v = v T (I - T)^{-1}``.  MEASURED (opt_precond 2026-08-16, and
    # reproduced by opt_integration): 14.5 -> 11.1 GMRES iterations at z=0 and
    # a 60x SMALLER true residual at the same nominal tolerance, because with
    # the ring gone the projected residual GMRES exits on is honest.  The
    # per-iteration cost is unchanged (the ring is ~4 % of the ladder matvec).
    lift = (route == "lift")
    if lift:
        n_pad = int(data["V_q0"].shape[0])
        zero_v = jax.lax.with_sharding_constraint(
            jnp.zeros((n_pad, n_pad), dtype=data["V_q0"].dtype), sh.V)
        eye_v = jax.lax.with_sharding_constraint(
            jnp.eye(n_pad, dtype=data["V_q0"].dtype), sh.V)
    if "W_R" not in data:                              # pragma: no cover
        raise RuntimeError(
            "sweep_q_wedge: build_ladder_resolvent did not populate "
            "data['W_R'].  The per-q loop below relies on the shallow copy "
            "build_finite_q_data returns carrying it; re-adding an in-loop "
            "ensure_W_R would restore the per-q densifier retrace instead.")
    z_list_ry = np.asarray(z_list_ry, dtype=np.complex128)
    if deflation_rank is None:
        deflation_rank = LADDER_DEFLATION_RANK if include_w else 0
    physical = bool(data.get("physical_branches", False))
    op_fn = partial(ladder_matvec_operands if physical else matvec_operands,
                    include_W=include_w)
    moment_actions = (_get_block_moment_actions(matvec, sh, jnp.complex128,
                                               retain_branches=physical)
                      if with_moments else None)
    data = _psi_at_rest_2d(data, mesh_xy)
    probe_planned = False

    def _q_payload(iq, qv):
        q = (int(qv[0]), int(qv[1]), int(qv[2]))
        dq = _psi_at_rest_2d(
            build_finite_q_data(_psi_ring_view(data), q, mesh_xy,
                                physical_branches=physical), mesh_xy)
        # Each physical branch has its own exact diagonal. The lifted
        # solve removes only the Hartree tile from that same operator.
        solve_dq = dict(dq)
        if lift:
            solve_dq["V_q0"] = zero_v
        diag_hq = build_preconditioner_diagonal_sharded(
            _psi_ring_view(solve_dq), mesh_xy,
            include_W=include_w, use_tda=False)
        if build_hook is not None:
            build_hook(iq, q, dq)
        return q, dq, solve_dq, diag_hq

    def _solve_blocks(iq, q, dq, solve_dq, diag_hq, z_items):
        nonlocal probe_planned
        if probe_chunk_plan is not None and not probe_planned:
            probe_chunk_plan(dict(data=dq, sh=sh, mesh=mesh_xy, matvec=matvec,
                gen=gen, snapshot=snapshot, diag=diag_hq, operands=op_fn(solve_dq),
                snapshot_v=(eye_v if lift else None), deflation_rank=deflation_rank,
                max_iter=gmres_max_iter, tol=gmres_tol, with_slope=with_slope,
                with_moments=with_moments, moment_actions=moment_actions,
                z_outer=z_outer, nz=len(z_list_ry)))
            probe_planned = True
        # One deflation space per (q, z), harvested on the first probe block
        # and reused by every later block: the operator does not depend on
        # the probe, so the eigenvectors every column pays for are found once.
        defl_by_z: dict = {}
        for c0, n_real, G in probe_blocks_for_q(iq, q):
            # NOT cast to float64 here: ``build_probe_rhs`` REFUSES a complex
            # probe block by name, and a cast on this line would discard its
            # imaginary part first (numpy's complex -> float is a warning, not
            # an error).  The seed is real-valued by construction — every
            # probe in the tree is a unit column or the identity — and that is
            # now an assumption the code states rather than one it enforces by
            # silently losing half the operand.
            # SEED HOIST: stage 1 reads the probe block and the q-shifted
            # payload and NOTHING z-dependent, so it is built once per
            # (q, probe block) rather than once per (q, block, z).  Three
            # dispatches per (q, z) on the RPA arm — the seed rebuilds M_X
            # locally and costs ~27 RPA-matvec-equivalents there (opt_shifts,
            # 2026-08-16) — and free on the ladder arm, where the matvec
            # dwarfs it.
            rhs = build_probe_rhs(G, _psi_ring_view(dq), gen, sh)
            moments = {}
            if with_moments and any(iz == 0 for iz, _z in z_items):
                moments = _project_screening_actions(
                    moment_actions(rhs, op_fn(dq)), dq, snapshot, view=_psi_ring_view)
            for iz, z in z_items:
                if deflation_rank > 0 and iz not in defl_by_z:
                    defl_by_z[iz] = _deflation_for(
                        matvec, diag_hq, complex(z),
                        op_fn(solve_dq if lift else dq), rhs,
                        rank=int(deflation_rank))
                result = apply_screening_resolvent_block(
                    G, complex(z), dq, matvec, diag_hq, gen, snapshot, sh,
                    max_iter=gmres_max_iter, tol=gmres_tol, return_iters=True,
                    rhs=rhs,
                    solve_data=(solve_dq if lift else None),
                    snapshot_v=(eye_v if lift else None),
                    operands_fn=op_fn,
                    deflation=defl_by_z.get(iz),
                    view=_psi_ring_view, with_slope=with_slope)
                W_tile, resids, iters = result[:3]
                if with_slope or with_moments:
                    fields = dict(result[3]) if with_slope else {}
                    if iz == 0:
                        fields.update(moments)
                    on_result(iq, q, iz, complex(z), c0, n_real,
                              W_tile, resids, iters, fields)
                else:
                    on_result(iq, q, iz, complex(z), c0, n_real,
                              W_tile, resids, iters)

    if z_outer:
        for iz, z in enumerate(z_list_ry):
            for iq, qv in enumerate(q_list):
                q, dq, solve_dq, diag_hq = _q_payload(iq, qv)
                _solve_blocks(iq, q, dq, solve_dq, diag_hq, [(iz, z)])
                del dq, solve_dq, diag_hq
            if on_z_done is not None:
                on_z_done(iz, complex(z))
        return
    for iq, qv in enumerate(q_list):
        q, dq, solve_dq, diag_hq = _q_payload(iq, qv)
        _solve_blocks(iq, q, dq, solve_dq, diag_hq, list(enumerate(z_list_ry)))


#: The psi carriers of the ring matvec payload: band axis 1, mu last.
_PSI_KEYS = ("psi_c_X", "psi_c_Y", "psi_v_X", "psi_v_Y",
             "psi_c_W_X", "psi_c_W_Y", "psi_v_W_X", "psi_v_W_Y",
             "psi_c_A_X", "psi_c_A_Y", "psi_v_A_X", "psi_v_A_Y")


def _psi_at_rest_2d(d: dict, mesh_xy: Mesh) -> dict:
    """Shallow copy of ``d`` with every psi carrier at 1/P per rank.

    The ring layout keeps the band axis whole and mu on one mesh axis
    (``*_X``: mu on x; ``*_Y``: mu on y), i.e. replicated over the other
    sqrt(P) axis.  At rest the band axis takes that other axis instead:
    ``*_X`` -> ``P(None, 'y', None, 'x')``, ``*_Y`` -> ``P(None, 'x', None,
    'y')``.  The band extent must divide that axis; the loader pads it to the
    mesh (``pad_bands``), and an extent that does not divide is refused by
    ``runtime.padding.authenticate_padded_axis`` rather than replicated.
    """
    from runtime.padding import authenticate_padded_axis
    out = dict(d)
    for key in _PSI_KEYS:
        a = d.get(key)
        if a is None:
            continue
        mu_axis = "x" if key.endswith("_X") else "y"
        band_axis = "y" if mu_axis == "x" else "x"
        # The loader pads the band window to the mesh (pad_bands); the
        # padding owner refuses a band carrier that is not already padded.
        n_band = int(a.shape[1])
        authenticate_padded_axis(n_band, n_band, int(mesh_xy.shape[band_axis]),
                                 name=f"ladder psi {key} band carrier")
        out[key] = _reshard(
            a, NamedSharding(mesh_xy, P(None, band_axis, None, mu_axis)))
    return out


def _psi_ring_view(d: dict) -> dict:
    """Transient ring-layout view of a 2-D payload, for the eager stages.

    The seed generator, preconditioner and snapshot are jits with explicit
    ring ``in_shardings``, which refuse a committed argument of another
    layout.  The view lives only for the one eager call it feeds.
    """
    out = dict(d)
    for key in _PSI_KEYS:
        a = d.get(key)
        if a is None:
            continue
        mu_axis = a.sharding.spec[3] if len(a.sharding.spec) > 3 else None
        out[key] = _reshard(
            a, NamedSharding(a.sharding.mesh, P(None, None, None, mu_axis)))
    return out


_RESHARD_CACHE: dict = {}


def _reshard(a: jax.Array, sharding: NamedSharding) -> jax.Array:
    """On-device relayout of a sharded array (a jitted identity)."""
    if a.sharding == sharding:
        return a
    fn = _RESHARD_CACHE.get(sharding)
    if fn is None:
        fn = jax.jit(lambda t: t, out_shardings=sharding)
        _RESHARD_CACHE[sharding] = fn
    return fn(a)


#: Recycled eigenvectors per (q, z) on the LADDER operator, and the Arnoldi
#: length that finds them.  MEASURED on Si 4x4x4 SOC (480 centroids, 8v x 20c,
#: P4; ``runs/Si/106_bsew_20260926``): the ladder's diagonally preconditioned
#: resolvent has eigenvalues far from 1 at finite q -- on BOTH sides of zero
#: ([-0.78, 4.22] at the third wedge q) -- and every probe column paid for
#: them separately: 9 / 22 / 98 GMRES iterations at the first three wedge q,
#: against 7-9 on the RPA operator of the same payload, whose preconditioned
#: spectrum sits in [1, 2.15].  Through this engine, iterations per column at
#: (rank, Arnoldi) = 0, (32, 64), (48, 96), (64, 128): 22.4 / 11.5 / 9.3 / 8.3
#: at q1 and 97.8 / 53.8 / 41.3 / 23.1 at q2; one warm harvest at (64, 128)
#: costs 1.44 s = 128 matvecs, once per (q, z).  A harvest reused at a
#: neighbouring q is WORSE than none (111 iterations at q2), so it is per q.
#: The RPA operator gets no deflation.
LADDER_DEFLATION_RANK = 64
LADDER_DEFLATION_ARNOLDI = 128


def _deflation_for(matvec, diag_h, z, operands, rhs, *, rank: int):
    """Harvest the ``(U, C)`` spectral deflation for one ``(q, z)`` operator.

    Seeded with the sum of the block's right-hand sides (every probe column's
    content, deterministic and rank-consistent).  ``LADDER_DEFLATION_ARNOLDI``
    matvecs, once per ``(q, z)``; the solve reuses the result for every probe
    chunk.
    """
    from .bse_feast import harvest_spectral_deflation
    seed = jnp.sum(rhs, axis=1, keepdims=True)
    defl, _ritz = harvest_spectral_deflation(
        matvec, diag_h, z, operands, seed,
        n_arnoldi=max(LADDER_DEFLATION_ARNOLDI, 2 * int(rank)),
        rank=int(rank))
    return defl


def prepare_ladder_head(restart_path, mesh_xy, *, input_file, head_dipole_path,
                        head_n_occ, include_w=True, data=None):
    """Prepare the one Gamma operator and dipole for a streamed head solve.

    Returns the existing operator, runtime payload, dipole and its transition
    energies. A caller refining frequency supports retains these locally,
    instead of loading/rebuilding the same wavefunctions and W_R per batch.
    No payload is placed in a module cache.
    """
    if data is None:
        data = load_bse_data_from_restart_sharded(
            restart_path, n_val=10**9, n_cond=10**9, mesh_xy=mesh_xy,
            input_file=input_file, inject_head=False, inject_w_head=include_w,
            q0_head="diagonal", load_v_full=False)
        # This Gamma-only payload is owned here. Supplied data is borrowed,
        # including a body's already prepared W_R, and is never donated here.
        if include_w:
            ensure_W_R(data, True, mesh_xy, consume_W_q=True)
    from .absorption_common import (slice_dipole_to_bse_window)
    from file_io.restart_bundle import (load_dipole_h5)
    from .head_resolvent import build_head_dipole, build_head_operator

    from .bse_window import _parse_wfn_path
    dipole_cart, delta_e, _ = load_dipole_h5(
        head_dipole_path, wfn_path=_parse_wfn_path(input_file))
    n_val = int(data["n_val"])
    n_cond = int(data["n_cond"])
    d_alpha, delta_cv = slice_dipole_to_bse_window(
        dipole_cart, delta_e, int(head_n_occ), n_val, n_cond)
    velocity_cv = np.asarray(d_alpha) * np.asarray(delta_cv)[None]
    d_pair = build_head_dipole(
        velocity_cv, delta_cv,
        n_cond_pad=int(data["n_cond_pad"]),
        n_val_pad=int(data["n_val_pad"]))
    head_data = build_finite_q_data(_psi_ring_view(data), (0, 0, 0), mesh_xy,
                                    physical_branches=True)
    head_op = build_head_operator(mesh_xy, head_data, include_w=include_w)
    return head_op, head_data, d_pair, delta_cv


def compute_ladder_head(
    restart_path, z_list_ry, mesh_xy, *, input_file, head_dipole_path,
    head_n_occ, head_pref, gmres_tol, gmres_max_iter, include_w=True, data=None,
    with_slope=False, with_moments=False,
):
    """Gamma-only micro-reducible tensor on the body's exact H backend.

    The same full transition-window loader supplies the operator; a prepared
    head can instead be retained across refinements by the head-fit owner.
    """
    from .head_resolvent import solve_head_tensor
    head_op, head_data, d_pair, delta_cv = prepare_ladder_head(
        restart_path,mesh_xy,input_file=input_file,head_dipole_path=head_dipole_path,
        head_n_occ=head_n_occ,include_w=include_w,data=data)
    return solve_head_tensor(
        head_op, head_data, d_pair, z_list_ry, pref=float(head_pref),
        max_iter=int(gmres_max_iter), tol=float(gmres_tol),
        arm=("micro_reducible_bse" if include_w
             else "micro_reducible_rpa_resolvent"),
        delta_cvk=delta_cv, with_slope=with_slope, with_moments=with_moments)


def compute_wc_qwedge(
    restart_path: str,
    z_list_ry,
    mesh_xy: Mesh,
    *,
    include_w: bool = True,
    gmres_tol: float,
    gmres_max_iter: int,
    probe_chunk: Optional[int] = None,
    input_file: Optional[str] = None,
    route: str = "auto",
    allow_replicated_dyson: bool = False,
    head_dipole_path: Optional[str] = None,
    head_n_occ: Optional[int] = None,
    head_pref: Optional[float] = None,
    deflation_rank: Optional[int] = None,
    on_z: Optional[Callable] = None,
    with_slope: bool = False,
    with_moments: bool = False,
) -> WLadderWedge:
    """Ladder ``W(z) - v`` bodies for every irreducible q and every requested z.

    ``on_z`` (optional) switches the facade to ONE z AT A TIME: the sweep walks
    z outside the q wedge, and after the last q of each z it calls
    ``on_z(iz, z, wc_z, resid_z, iters_z)`` with that z's
    ``(nq_irr, mu_pad, mu_pad)`` wedge at ``P(None, 'x', 'y')`` and its
    ``(nq_irr, n_pad)`` host residuals and iteration counts, then drops the
    tile.  The resident set is one z's wedge instead of the
    ``(nz, nq, mu, mu)`` stack, and the returned wedge carries ``wc=None``.
    psi rests 2-D (:func:`sweep_q_wedge`).

    ``with_slope`` supplies exact ``dWc_ds`` (Ry^-1, s=z_Ry^2) and
    ``Wc_minus_z`` (Ry) for an independent evenness check. ``with_moments``
    computes M0..M3 once per q through the same full H (units Ry^(j+2));
    M0/M2 certify vanishing odd-in-z coefficients and M1/M3 feed the model.
    The caller must check these raw fields; none is Hermitized or projected.
    With either flag, ``on_z`` receives a sixth dictionary of face-sharded
    ``(nq_irr, mu_pad, mu_pad)`` fields. M0..M3 are supplied at the first z
    only and then released. Without streaming the
    returned ``response_fields`` carries all requested values (slopes and
    reflected values have a leading z axis). Streaming retains no response
    matrices in the return. Residual/iteration outputs cover every shifted solve.
    These actions require the full operator, so ``route='auto'`` chooses
    ``plain`` and an explicitly requested ring lift refuses.

    The facade the GW stage helper calls.  ``z_list_ry`` are complex Rydberg
    frequencies (cohsex: ``{0}``; gn_ppm: ``{0, i w_p}``; mpa: its sample plan);
    the result is a :class:`WLadderWedge`.

    ``probe_chunk`` is the memory knob: columns of the mu^2 tile solved per
    block. None chooses a bounded width from the common tile policy, then
    checks the executed programs and carries against the run's memory room.
    A positive request is honored and priced. The carrier must be a
    multiple of ``py`` because the reduce-scatter snapshot tiles the probe axis
    over ``y``. Chunking trades stage dispatches for memory; the operator and
    solver tolerance are unchanged. The compiled engine is shared, since ``n_probe`` is the only shape that
    changes and a chunked sweep uses ONE ``n_probe`` for all chunks except a
    final short one (which is zero-padded up, not re-shaped).

    CONTRACT NOTE (deviation, stated loudly): the frozen facade in
    DESIGN_2026-08-15 section 4 takes ``restart_path`` alone.  The symmetry
    wedge cannot be derived from the restart — ``SymMaps`` needs the WFN, which
    is named by the INPUT DECK — and the sharded loader needs the same deck to
    resolve the k-grid for flat-q restart layouts.  ``input_file`` is therefore
    an ADDITIVE keyword; every call in the frozen contract's shape still type-
    checks, and omitting it raises naming exactly what is missing rather than
    guessing a deck path next to the restart.

    ``route`` — which resolvent the solve runs.

    * ``"lift"`` — route A: the Hartree ring is removed from the Krylov
      problem by Woodbury and put back with ONE dense ``N_mu`` Dyson,
      ``W - v = v T (I - T)^{-1}``.  Fewer iterations (14.5 -> 11.1 at z=0,
      measured) at unchanged per-iteration cost, and a strictly SMALLER true
      residual at the same nominal tolerance.
    * ``"plain"`` — the full operator with the in-tree diagonal
      preconditioner: the historical path, kept selectable for A/B and used
      wherever the closing Dyson cannot run.
    * ``"auto"`` (default) — ``"lift"`` on a single device, ``"plain"``
      above it, because the closing Dyson is ``jnp.linalg.solve`` and
      materialising the ``N_mu x N_mu`` tile per rank is the replication the
      scaling doctrine forbids.  ``allow_replicated_dyson=True`` takes
      ``"lift"`` anyway and accepts exactly the per-rank memory class the RPA
      Dyson's own default (``linalg = local``, ``gw.w_isdf.solve_w``)
      already has; routing the close through ``distrib_la`` — the twin of that
      solver's ``distributed`` plan — is the registered follow-on, named here
      rather than half-built.
    """
    if input_file is None:
        raise ValueError(
            "compute_wc_qwedge needs input_file= (the COHSEX/GW input deck) in "
            "addition to restart_path: the irreducible q-wedge comes from "
            "SymMaps, which is built from the WFN named by the deck, and the "
            "sharded BSE loader resolves the k-grid from it for flat-q restart "
            "layouts. Pass the same deck the GW driver was run with.")
    px, py = mesh_xy.devices.shape
    z_list_ry = np.asarray(z_list_ry, dtype=np.complex128)
    if z_list_ry.ndim != 1 or z_list_ry.size == 0:
        raise ValueError(
            f"z_list_ry must be a non-empty 1-D list of complex Rydberg "
            f"frequencies; got shape {z_list_ry.shape}.")
    # Argument checks BEFORE the restart read: a caller that got probe_chunk
    # wrong should learn it in milliseconds, not after a multi-GB sharded load.
    if probe_chunk is not None:
        if int(probe_chunk) <= 0:
            raise ValueError(f"probe_chunk must be positive; got {probe_chunk}")
        from runtime.padding import padded_axis
        probe_chunk = padded_axis(
            int(probe_chunk), mesh_xy, name="BSE ladder probe chunk",
            spec=P("y"), axis=0).carrier

    # FULL chi0 band window on both legs — the pair basis must span exactly the
    # window the GW chi0 consumed, or the ladder resolvent is solving a smaller
    # problem than the W_R kernel it is fed (band-window parity, bse_w_exact.main).
    data = load_bse_data_from_restart_sharded(
        restart_path, n_val=10**9, n_cond=10**9, mesh_xy=mesh_xy,
        input_file=input_file, inject_head=False, inject_w_head=include_w,
        q0_head="diagonal", load_v_full=True)
    # The public facade owns this restart read; no caller can retain W_q.
    # Per-q diagonals and the optional later head share the small saved q=0
    # slice and this one W_R instead of keeping both all-q representations.
    if include_w:
        ensure_W_R(data, True, mesh_xy, consume_W_q=True)
    data = _psi_at_rest_2d(data, mesh_xy)

    head_args = (head_dipole_path, head_n_occ, head_pref)
    if any(value is not None for value in head_args) and not all(
            value is not None for value in head_args):
        raise ValueError(
            "ladder head requires head_dipole_path, head_n_occ, and "
            "head_pref together; a partial request cannot identify the "
            "transition manifold or normalization.")

    sym = _symmetry_tables(input_file)
    q_list = np.asarray(sym.q_irr_kgrid_int, dtype=int)
    n_pad = int(data["V_q0"].shape[0])
    nlog = int(data["n_rmu"])
    chunk = n_pad if probe_chunk is None else min(int(probe_chunk), n_pad)

    # Probe block ROWS are the probe columns (apply_screening_resolvent_block
    # takes (n_probe, n_rmu) with row i the density-space probe vector), so the
    # full basis enters as the identity and a chunk as a row slice of it.  A
    # short final chunk is zero-PADDED up to `chunk` rather than reshaped, so
    # every chunk hits the same compiled engine.
    #
    # THE WALK STOPS AT ``nlog``, NOT AT ``n_pad``, and that is a correctness
    # requirement rather than a saved solve.  A probe on a PAD centroid has an
    # identically zero right-hand side (the loader zero-pads psi, so
    # ``f = M^dag (v g_pad) = 0``), and the shifted GMRES normalises by
    # ``||r0||``: 0/0 makes the whole column NaN, which then survives every
    # later step because ``0 * NaN`` is NaN, not 0.  MEASURED 2026-08-15 (JID
    # 57064957, the wiring-closure gate at 2x2): with the identity taken over
    # all 400 padded columns the assembled tile carried ``nan+nanj`` at every
    # pad entry while all 399 logical columns were correct.  The pad columns of
    # ``W - v`` are ZERO by construction — there is no such centroid — and the
    # accumulator is allocated zero, so not solving them IS the right value.
    # The same reasoning covers the zero ROWS of a short final chunk: their
    # columns are NaN inside ``W_tile`` and ``_accumulate_columns`` takes only
    # the first ``n_real``.
    nz, nq = int(z_list_ry.size), int(len(q_list))
    tiles = [[None] * nq for _ in range(nz)]
    resid = np.zeros((nz, nq, n_pad), dtype=np.float64)
    iters = np.zeros((nz, nq, n_pad), dtype=np.int32)
    response_tiles = ({name: [[None] * nq for _ in range(nz)]
                       for name in ("dWc_ds", "Wc_minus_z")} if with_slope else {})
    moment_tiles = ({name: [None] * nq for name in ("M0", "M1", "M2", "M3")}
                    if with_moments else {})
    moment_fields = {}

    if route not in ("auto", "plain", "lift"):
        raise ValueError(
            f"route must be 'auto', 'plain' or 'lift', got {route!r}")
    n_dev = int(np.prod(mesh_xy.devices.shape))
    if route == "auto":
        route = ("lift" if not (with_slope or with_moments)
                 and (n_dev == 1 or allow_replicated_dyson) else "plain")
    if route == "lift" and (with_slope or with_moments):
        raise ValueError("Response slopes/moments require route='plain': they are "
                         "actions of the full H, including its Hartree ring.")
    if route == "lift" and n_dev > 1 and not allow_replicated_dyson:
        raise NotImplementedError(
            f"route='lift' closes with jnp.linalg.solve on the N_mu x N_mu "
            f"tile, a single-device dense solver, but this mesh is "
            f"{mesh_xy.devices.shape}.  Above one device the close must go "
            f"through the distributed solver (services/distrib_la), the twin "
            f"of gw.w_isdf.solve_w's 'distributed' plan; that routing is the "
            f"registered follow-on.  Pass allow_replicated_dyson=True to "
            f"accept the per-rank N_mu^2 tile the RPA Dyson's own DEFAULT "
            f"('local') plan already holds, or route='plain'/'auto'.")

    def plan_probes(context):
        nonlocal chunk
        # Reserve the full future result even though first-q planning happens
        # before any of its rows exist. Stack/placement copies are separate
        # lifetimes; do not sum their peaks with a live GMRES workspace.
        face = 16*n_pad*n_pad//n_dev
        bank_fields = nq*((1+2*int(with_slope))*(1 if on_z is not None else nz)
                          + 4*int(with_moments) + int(route == "lift"))
        chunk = _plan_probe_chunk(context, probe_chunk, n_pad=n_pad,
            carry_bytes=(bank_fields+1)*face,
            stack_bytes=((bank_fields+nq) if on_z is not None else 3*bank_fields)*face)

    def _on_result(iq, q, iz, z, c0, n_real, W_tile, resids, its, fields=None):
        tiles[iz][iq] = _accumulate_columns(
            tiles[iz][iq], W_tile, c0, n_real, n_pad, mesh_xy)
        resid[iz, iq, c0:c0 + n_real] = np.asarray(gather_to_host(resids))[:n_real]
        iters[iz, iq, c0:c0 + n_real] = np.asarray(gather_to_host(its))[:n_real]
        for name, value in (fields or {}).items():
            if name in moment_tiles:
                moment_tiles[name][iq] = _accumulate_columns(
                    moment_tiles[name][iq], value, c0, n_real, n_pad, mesh_xy)
            else:
                response_tiles[name][iz][iq] = _accumulate_columns(
                    response_tiles[name][iz][iq], value, c0, n_real, n_pad, mesh_xy)

    # The lift's closing Dyson needs each q's OWN ``v`` tile, which only the
    # q-shifted payload carries; ``build_hook`` is the sweep's existing seam
    # for exactly that and costs nothing on the plain route.
    v_by_q: dict = {}

    wedge_sh = NamedSharding(mesh_xy, P(None, None, "x", "y"))

    def _stack_rows(rows):
        value = jax.lax.with_sharding_constraint(
            jnp.stack(rows, axis=0).astype(jnp.complex128),
            NamedSharding(mesh_xy, P(None, "x", "y")))
        _assert_pad_block_is_zero(value, nlog)
        return value

    def _on_z_done(iz, z):
        row = tiles[iz]
        if route == "lift":
            row = [dyson_close_tile(row[iq], v_by_q[iq],
                                    allow_replicated_solve=allow_replicated_dyson)
                   for iq in range(nq)]
        wc_z = jax.lax.with_sharding_constraint(
            jnp.stack(row, axis=0).astype(jnp.complex128),
            NamedSharding(mesh_xy, P(None, "x", "y")))
        tiles[iz] = None
        del row
        _assert_pad_block_is_zero(wc_z, nlog)
        if with_slope or with_moments:
            fields = {}
            for name, by_z in response_tiles.items():
                fields[name] = _stack_rows(by_z[iz])
                by_z[iz] = None
            if iz == 0:
                for name in tuple(moment_tiles):
                    rows = moment_tiles.pop(name)
                    moment_fields[name] = _stack_rows(rows)
                    del rows
            fields.update(moment_fields)
            on_z(iz, z, wc_z, resid[iz], iters[iz], fields)
            if iz == 0:
                moment_fields.clear()
        else:
            on_z(iz, z, wc_z, resid[iz], iters[iz])

    sweep_q_wedge(
        data, mesh_xy, q_list, z_list_ry, include_w=include_w,
        probe_blocks_for_q=lambda _iq, _q: _identity_probe_blocks(nlog, n_pad, chunk),
        gmres_tol=gmres_tol, gmres_max_iter=gmres_max_iter,
        on_result=_on_result, route=route, deflation_rank=deflation_rank,
        build_hook=((lambda iq, q, dq: v_by_q.__setitem__(iq, dq["V_q0"]))
                    if route == "lift" else None),
        z_outer=on_z is not None,
        on_z_done=(_on_z_done if on_z is not None else None),
        with_slope=with_slope, with_moments=with_moments,
        probe_chunk_plan=plan_probes)

    if on_z is not None:
        wc = None
    elif route == "lift":
        # Every accumulated tile is ``T = Pi v``; the ring goes back in here,
        # ONCE per (q, z), through the closing Dyson.
        tiles = [[dyson_close_tile(tiles[iz][iq], v_by_q[iq],
                                   allow_replicated_solve=allow_replicated_dyson)
                  for iq in range(nq)] for iz in range(nz)]

    # Stack (nz, nq, mu_pad, nu_pad) and KEEP the pad — see WLadderWedge.wc.
    # The pad rows/columns are already zero (identity probe on a zero-psi pad
    # centroid), and stripping them here would make the tile's last two axes
    # n_rmu-wide, which the documented P(None, None, 'x', 'y') sharding cannot
    # express for an n_rmu that does not divide the mesh (399 on a 2x2 —
    # measured, JID 57064957).  The logical extent travels as `n_rmu`.
    if on_z is None:
        stacked = jnp.stack([jnp.stack(row, axis=0) for row in tiles], axis=0)
        wc = jax.lax.with_sharding_constraint(
            stacked.astype(jnp.complex128), wedge_sh)
        _assert_pad_block_is_zero(wc, nlog)
    response_fields = None
    if (with_slope or with_moments) and on_z is None:
        response_fields = {name: _stack_rows(rows) for name, rows in moment_tiles.items()}
        for name, by_z in response_tiles.items():
            response_fields[name] = jax.lax.with_sharding_constraint(
                jnp.stack([_stack_rows(rows) for rows in by_z], axis=0), wedge_sh)
    head_result = None
    if head_dipole_path is not None:
        head_result = compute_ladder_head(
            restart_path, z_list_ry, mesh_xy, input_file=input_file,
            head_dipole_path=head_dipole_path, head_n_occ=head_n_occ,
            head_pref=head_pref, gmres_tol=gmres_tol,
            gmres_max_iter=gmres_max_iter, include_w=include_w, data=data)
    return WLadderWedge(
        wc=wc,
        n_rmu=nlog,
        z_list_ry=z_list_ry,
        q_irr_kgrid_int=q_list,
        irr_idx_q=np.asarray(sym.irr_idx_q),
        sym_idx_q=np.asarray(sym.sym_idx_q),
        q_irr_full_idx=np.asarray(sym.q_irr_full_idx),
        gmres_resid=resid,
        gmres_iters=iters,
        include_w=bool(include_w),
        head_result=head_result,
        response_fields=response_fields,
    )


#: ``(shape, dtype, mesh) -> jitted Dyson close``.  Cached for the reason every
#: other jit in this family is: ``jax.jit`` keys on function identity.
_DYSON_CACHE: dict = {}


def dyson_close_tile(T: jax.Array, V_q0: jax.Array, *,
                     allow_replicated_solve: bool = False) -> jax.Array:
    """``W(z) - v = v T (I - T)^{-1}`` — the lift's closing algebra.

    ``T = Pi v`` on the padded centroid extent.  The pad rows and columns of
    ``T`` are identically zero (the probe walk stops at ``n_rmu`` and the
    identity projector kills pad centroids), so ``I - T`` is block
    ``[[I - T_ll, 0], [0, I]]`` and the inverse never touches the pad — which is
    why the identity, and not the zero tile, is the right projector.

    Solved as ``(I - T)^T Y^T = T^T`` and then ``v Y``; a dense ``N_mu^3``.  This
    is the SAME dense Dyson the RPA production path performs per (q, z)
    (``gw.mpa.model._solve_wc``), not a new class of object, but ``jnp.linalg``
    is a single-device solver: above one device this must route through
    ``distrib_la``, and the caller is refused rather than silently gathering an
    ``N_mu^2`` tile per rank (the F3 hazard, in its dense-algebra form).

    ``allow_replicated_solve=True`` takes that gather DELIBERATELY, and the
    scope of the concession is exactly the RPA path's own: ``gw.w_isdf.solve_w``
    has TWO plans and its DEFAULT (``linalg = local``) is a per-q dense
    LU in which "each rank holds whole (mu, mu) tiles for its q's".  So this
    flag makes the ladder's close the same memory class as the production RPA
    close, no better and no worse, and it is opt-in so that the choice is in
    the caller's log rather than in this function's silence.  The
    ``distrib_la`` routing — the twin of that solver's ``distributed`` plan —
    remains the registered follow-on, and is what the P -> 1e3 envelope needs.
    """
    mesh = V_q0.sharding.mesh if hasattr(V_q0.sharding, "mesh") else None
    n_dev = int(np.prod(mesh.devices.shape)) if mesh is not None else 1
    if n_dev > 1 and not allow_replicated_solve:
        raise NotImplementedError(
            "dyson_close_tile uses jnp.linalg.solve, a single-device dense "
            f"solver, but this tile lives on a {mesh.devices.shape} mesh.  The "
            "N_mu x N_mu Dyson close of the ring lift is the same dense solve "
            "the RPA path already does per (q, z) and must go through the "
            "distributed solver (services/distrib_la) at P>1; gathering the "
            "tile per rank here would be exactly the mu^2-replication the "
            "scaling doctrine forbids.  Named deferral, not a half-build.  "
            "allow_replicated_solve=True accepts the gather explicitly (the "
            "memory class of gw.w_isdf.solve_w's DEFAULT 'local' plan).")
    key = (tuple(T.shape), str(T.dtype), str(V_q0.dtype), T.sharding)
    fn = _DYSON_CACHE.get(key)
    if fn is None:
        out_sh = T.sharding

        def _close(T_, V_):
            n = T_.shape[-1]
            A = jnp.eye(n, dtype=T_.dtype) - T_
            # Y = T (I - T)^{-1}  <=>  (I - T)^T Y^T = T^T
            from distrib_la import solve_local   # 2^31-byte span route (ROCM)
            Y = solve_local(A.T, T_.T).T
            # Put the product back on the wedge tile's own sharding: at P>1
            # the dense solve above is replicated by construction, and letting
            # GSPMD pick the OUTPUT layout would leave the wedge stack's
            # with_sharding_constraint to resolve it with a second collective.
            return jax.lax.with_sharding_constraint(
                V_.astype(T_.dtype) @ Y, out_sh)

        fn = jax.jit(_close)
        _DYSON_CACHE[key] = fn
    return fn(T, V_q0)


#: ``(shape, dtype, sharding, nlog) -> jitted worst-pad reduction``.  Cached
#: for the reason every other jit in this file is: ``jax.jit`` keys on function
#: identity, so building one per call re-traces every time (the F1 defect, in
#: miniature).
_PAD_CHECK_CACHE: dict = {}


def _assert_pad_block_is_zero(wc: jax.Array, nlog: int) -> None:
    """The pad rows and columns of the wedge are EXACTLY zero, checked here.

    The producer owns this guarantee (design amendment 2026-08-15): a
    consumer cannot tell "the pad is zero because there is no such centroid"
    from "the pad is zero because nothing was written there", and every
    downstream contraction multiplies the pad by a zero ζ row, where a NaN or
    an inf would NOT be inert (``0 * NaN`` is NaN).

    A MASKED REDUCTION UNDER ``jax.jit``, NOT A SLICE, and the difference is
    the whole μ²-class question.  The obvious spelling —
    ``max(|wc[..., nlog:, :]|), max(|wc[..., :, nlog:]|)`` evaluated eagerly —
    slices a ``P(None, None, 'x', 'y')`` array on the axes it is sharded on,
    and each eager slice ALL-GATHERS that axis: measured
    ``c128[nz, nq, N_mu, N_mu/p]`` per rank, ~4.0 GB/rank at N_mu = 1e4,
    p = 32, nz = 8, nq = 10, on the return path of every wedge whose n_rmu is
    not a multiple of p_x·p_y — i.e. essentially every real deck (HLO audit
    2026-08-16, FINDINGS.md F3; it is the same hazard as the
    ``stacked[:, :, :nlog, :nlog]`` return this file already lost once).
    Masking is elementwise, so GSPMD keeps the operand exactly where it is and
    the only collectives are the volume-preserving ones the reduction itself
    needs; the two scalars come back replicated.
    """
    n_pad = int(wc.shape[-1])
    if n_pad == nlog:
        return
    key = (tuple(wc.shape), str(wc.dtype), wc.sharding, int(nlog))
    fn = _PAD_CHECK_CACHE.get(key)
    if fn is None:
        sharding, n_logical = wc.sharding, int(nlog)

        def _worst(a):
            a = jax.lax.with_sharding_constraint(a, sharding)
            is_pad = jnp.arange(a.shape[-1]) >= n_logical
            mag = jnp.abs(a)
            return jnp.max(jnp.where(is_pad[:, None] | is_pad[None, :],
                                     mag, jnp.zeros((), mag.dtype)))

        fn = jax.jit(_worst)
        _PAD_CHECK_CACHE[key] = fn
    worst = float(jax.device_get(fn(wc)))
    if not (worst == 0.0):
        raise RuntimeError(
            f"w_ladder: the padded block of the returned wedge is not zero "
            f"(max |.| = {worst!r} over rows/cols {nlog}:{n_pad}).  Pad "
            f"centroids do not exist, so their W - v must be identically "
            f"zero; a NaN here is the zero-rhs GMRES signature (0/0 in the "
            f"Arnoldi normalisation) and a finite value means a pad column "
            f"was solved as if it were physical.")


#: ``(devices, n_pad, n_real, tile shape/dtype) -> jitted placement``.
_ACCUMULATE_CACHE: dict = {}


def _accumulate_columns(acc, W_tile, c0: int, n_real: int, n_pad: int,
                        mesh_xy: Mesh):
    """Place a probe chunk's ``(mu_X, probe_Y)`` tile into the (mu, nu) tile.

    ``acc`` stays sharded ``P('x','y')`` throughout and ``None`` on the first
    chunk allocates.

    THE PLACEMENT RUNS UNDER ``jax.jit`` WITH EXPLICIT SHARDINGS, and that is
    not decoration.  Eagerly, ``dynamic_update_slice`` on the nu axis of a
    ``P('x','y')`` accumulator crosses a dispatch boundary with no sharding
    context, and XLA satisfies it by ALL-GATHERING the accumulator's nu axis:
    measured ``c128[N_mu/p_x, N_mu]`` per rank per placement, which divides by
    sqrt(P) rather than by P (~50 MB/rank at N_mu = 1e4, p_x = 32) — so the
    ``probe_chunk`` knob documented as trading dispatches for MEMORY was, at
    every setting but the default single chunk, spending memory instead (HLO
    audit 2026-08-16, FINDINGS.md F2).  The identical body under ``jit`` with
    in/out shardings emits one shard-sized gather.  The single-chunk arm was
    already clean and stays bit-identical; it simply no longer has to be the
    only safe setting.

    ``c0`` IS BAKED IN AS A CONSTANT, one tiny compiled placement per chunk
    offset, and that is the trade the measurement forces.  Chunk offsets are
    shard-aligned by construction (``chunk % py == 0``), so a CONSTANT index
    lets GSPMD see the alignment and move only shard-sized pieces; a TRACED
    index hides it and XLA falls back to all-gathering the accumulator's whole
    nu axis — measured, on this same probe: ``c128[N_mu/p_x, N_mu]`` with a
    traced index against ``c128[N_mu/p_x, N_mu/p_y]`` with a constant one.  A
    program that moves nothing but data costs milliseconds to compile and is
    compiled once per offset per process; a per-placement mu-row gather costs
    bytes on every chunk of every q of every z.  ``n_real`` is static for the
    ordinary reason — it sets the update's shape.
    """
    tile_sh = NamedSharding(mesh_xy, P("x", "y"))
    if acc is None:
        acc = jax.lax.with_sharding_constraint(
            jnp.zeros((n_pad, n_pad), dtype=jnp.complex128), tile_sh)
    key = (tile_sh, int(n_pad), int(n_real), int(c0),
           tuple(W_tile.shape), str(W_tile.dtype))
    fn = _ACCUMULATE_CACHE.get(key)
    if fn is None:
        width, start = int(n_real), int(c0)

        def _place(acc_, tile_):
            # acc and the incoming chunk both carry the tile sharding (the
            # chunk's probe axis is `chunk` wide and `chunk % py == 0`).  The
            # UPDATE deliberately carries no constraint: a short final chunk's
            # width need not divide the mesh, and 15 columns on a 2x2 is a
            # legal intermediate and an illegal NamedSharding.
            acc_ = jax.lax.with_sharding_constraint(acc_, tile_sh)
            tile_ = jax.lax.with_sharding_constraint(tile_, tile_sh)
            upd = tile_[:, :width].astype(jnp.complex128)
            return jax.lax.with_sharding_constraint(
                jax.lax.dynamic_update_slice(acc_, upd, (0, start)), tile_sh)

        # No in_/out_shardings: this is called from inside another jit by the
        # HLO probe, and a nested jit carrying explicit shardings is not a
        # supported shape.  The constraints above express the same thing.
        fn = jax.jit(_place)
        _ACCUMULATE_CACHE[key] = fn
    return fn(acc, W_tile)
