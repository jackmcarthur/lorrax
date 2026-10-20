"""Prepared full-centroid static TD-aGW control, not a collision-off HF alias.

The accepted SC anchor is retained once. Only its density increment is
Hartree[Vbar] minus screened exchange[W(0)]. No zeta-file export, new kernel,
GW collision, equilibrium subtraction or spatial rank approximation is added.
This campaign adapter has not been admitted on the material/GPU trajectory.
"""


def static_control_from_body(reference, W0, head, *, packed_G0, cell_volume):
    """Reuse native Hartree/Fock owners with a declared pre-head static body.

    ``reference`` is build_sc_td_reference's authenticated plain bundle.
    ``W0`` is the matching full-zone QirrOperator, Ry, XY-sharded. Its head
    is separate, as returned by build_static_screened_w. ``head.wcoul0`` is
    FULL SCREENED W, NOT Wc: never add vc0 to it. ``packed_G0`` is small host
    metadata in the source mu_basis packing. Large arrays stay distributed.
    Equal electronic/bosonic grids only; no implicit coarse-W interpolation.
    """
    import math
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from symmetry_maps import QirrOperator
    from tdagw.snapshot_runtime import _native_hf_apply
    from tdagw.self_energy import SelfEnergyModel
    from tdagw.dynsig import make_dynsig_kernels, _w_R, WR_SPEC
    from bse.bse_head import _inject_q0_head

    prepared, bare = reference["prepared"], reference["model"]
    grid, mesh = tuple(prepared["kgrid"]), prepared["mesh"]
    if tuple(prepared.get("qgrid", grid)) != grid:
        raise ValueError("GATE static_agw_grid: initial control requires qgrid=kgrid")
    if getattr(bare.apply, "func", None) is not _native_hf_apply:
        raise ValueError("GATE static_agw_reference: authenticated native HF-delta operand owner required")
    op = QirrOperator.of(W0)
    if not op.is_whole_zone() or op.n_full != int(prepared["nk"]):
        raise ValueError("GATE static_agw_W: whole electronic-grid static W required")
    if not np.array_equal(np.asarray(op.full_rows), np.arange(int(prepared["nk"]))):
        raise ValueError("GATE static_agw_W_rows: canonical full-q rows required before FFT")
    if tuple(op.values.shape) != tuple(prepared["V_q"].shape):
        raise ValueError("GATE static_agw_W: static W and bare V centroid carriers differ")
    g0 = np.asarray(packed_G0)
    if g0.shape != (op.values.shape[-1],) or not np.isfinite(g0).all():
        raise ValueError("GATE static_agw_G0: exact source-packed finite G0 vector required")
    volume, screened_head = float(cell_volume), complex(head.wcoul0)
    if (not math.isfinite(volume) or volume <= 0
            or not np.isfinite([screened_head.real, screened_head.imag]).all()
            or abs(screened_head.imag) > 1.e-10*max(1., abs(screened_head.real))):
        raise ValueError("GATE static_agw_head: real static screened head and positive volume required")
    gx = device_put_process_local(g0, NamedSharding(mesh, P("x")))
    gy = device_put_process_local(g0, NamedSharding(mesh, P("y")))
    wgrid = jax.jit(lambda w: jnp.moveaxis(w, 0, -1).reshape(w.shape[1], w.shape[2], *grid),
        out_shardings=NamedSharding(mesh, WR_SPEC))(op.values)
    # Keep G=0-excluded Hartree V0 unchanged. The injector touches W only:
    # wcoul0/volume once, with NO extra vc0 and NO scalar exchange head later.
    v0 = bare.params[3]
    _, wgrid, head_log = _inject_q0_head(v0, wgrid, gx, gy, None,
        jnp.asarray([screened_head]), volume, w0_ready=True)
    flat = jax.jit(lambda w: jnp.moveaxis(w.reshape(w.shape[0], w.shape[1], -1), -1, 0),
        out_shardings=NamedSharding(mesh, P(None, "x", "y")))(wgrid)
    kernel = make_dynsig_kernels(mesh, grid)
    wr = jax.jit(lambda w: _w_R(w, kernel),
        out_shardings=NamedSharding(mesh, WR_SPEC))(flat)
    px, py, _, vh, rho_eq, _ = bare.params
    model = SelfEnergyModel(apply=bare.apply, params=(px, py, wr, vh, rho_eq, 0.),
        linear=True, name="static TD-aGW: bare Hartree plus screened W0 exchange; NO GW collision")
    return dict(model=model, receipt=dict(
        arm="A_STATIC_SEX_HARTREE_NO_COLLISION", head=head_log,
        W_head_scope="HeadSample.wcoul0 is full screened W; injected once, scalar head=0",
        Hartree="original G0-excluded bare V0 unchanged", spatial_projection=False,
        anchor="accepted SC H once; delta_Hartree[Vbar]-delta_SEX[W0]",
        collision="off; not frozen-W MGKBA", material_admission="NOT_RUN"))


def build_static_control(source, reference):
    """Synthesize current accepted-state W0 through its existing native owner."""
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.static_screening import build_static_screened_w
    mesh, meta = source["mesh"], source["meta"]
    if tuple(meta.qgrid) != tuple(meta.kgrid):
        raise ValueError("GATE static_agw_grid: initial control requires qgrid=kgrid")
    receipt, frame = reference["receipts"], source["frame"]
    if any(receipt.get(key) != frame.get(key) for key in
           ("frame", "input_h_sha256", "wfn_fingerprint")):
        raise ValueError("GATE static_agw_reference: W source and accepted TD reference identities differ")
    wm = fresh_shared_pole_capacity(meta, mesh)
    wm.shared_pole_capacity.at_phase()
    W0, head = build_static_screened_w(None, source["V_q"], config=source["config"],
        meta=wm, mesh_xy=mesh, shared_pole=source["W_by_role"]["shared_pole"],
        head_resolver=source["W_by_role"]["iteration_head"])
    return static_control_from_body(reference, W0, head,
        packed_G0=meta.mu_basis.pack_host(source["G0_canonical"]),
        cell_volume=float(meta.cell_volume))
