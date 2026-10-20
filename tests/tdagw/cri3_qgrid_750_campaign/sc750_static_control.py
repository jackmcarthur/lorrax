"""Prepared full-centroid static TD-aGW control, not a collision-off HF alias.

The accepted SC anchor is retained once. Only its density increment is
Hartree[Vbar] minus screened exchange[W(0)]. No zeta-file export, new kernel,
GW collision, equilibrium subtraction or spatial rank approximation is added.
Static-A material calibration trajectories completed in steps 60056663.10/.11;
the midpoint record is evidence/STATIC_A_MIDPOINT_60056663.json. This is not
MGKBA, updated-W, positivity or three-arm driven-comparison admission.
"""


def static_control_from_body(reference, W0, head, *, packed_G0, cell_volume,
                             capacity=None):
    """Reuse native Hartree/Fock owners with a declared pre-head static body.

    ``reference`` is build_sc_td_reference's authenticated plain bundle.
    ``W0`` is the matching full-zone QirrOperator, Ry, XY-sharded. Its head
    is separate, as returned by build_static_screened_w. ``head.wcoul0`` is
    FULL SCREENED W, NOT Wc: never add vc0 to it. ``packed_G0`` is small host
    metadata in the source mu_basis packing. Large arrays stay distributed.
    A nested coarse bosonic grid uses ordinary coarse-q quadrature, not W
    interpolation. Its enlarged fine-grid Fourier carrier is admitted using
    ``capacity`` before dispatch. Equal grids retain their existing path.
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
    qgrid = tuple(prepared.get("qgrid", grid))
    comb = None
    if qgrid != grid:
        from common.fft_helpers import make_bosonic_q_comb
        comb = make_bosonic_q_comb(mesh, grid, qgrid)
        if capacity is None:
            raise ValueError("GATE static_agw_capacity: coarse q-comb requires its live capacity ledger")
    if getattr(bare.apply, "func", None) is not _native_hf_apply:
        raise ValueError("GATE static_agw_reference: authenticated native HF-delta operand owner required")
    op = QirrOperator.of(W0)
    nq = int(np.prod(qgrid))
    if not op.is_whole_zone() or op.n_full != nq:
        raise ValueError("GATE static_agw_W: whole bosonic-grid static W required")
    if not np.array_equal(np.asarray(op.full_rows), np.arange(nq)):
        raise ValueError("GATE static_agw_W_rows: canonical full-q rows required before FFT")
    if tuple(op.values.shape[1:]) != tuple(prepared["V_q"].shape[1:]):
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
    wgrid = jax.jit(lambda w: jnp.moveaxis(w, 0, -1).reshape(w.shape[1], w.shape[2], *qgrid),
        out_shardings=NamedSharding(mesh, WR_SPEC))(op.values)
    # Keep G=0-excluded Hartree V0 unchanged. The injector touches W only:
    # wcoul0/volume once, with NO extra vc0 and NO scalar exchange head later.
    v0 = bare.params[3]
    _, wgrid, head_log = _inject_q0_head(v0, wgrid, gx, gy, None,
        jnp.asarray([screened_head]), volume, w0_ready=True)
    flat = jax.jit(lambda w: jnp.moveaxis(w.reshape(w.shape[0], w.shape[1], -1), -1, 0),
        out_shardings=NamedSharding(mesh, P(None, "x", "y")))(wgrid)
    kernel = make_dynsig_kernels(mesh, grid)
    comb_receipt = None
    if comb is None:
        wr = jax.jit(lambda w: _w_R(w, kernel),
            out_shardings=NamedSharding(mesh, WR_SPEC))(flat)
    else:
        # Head is already in the COARSE q0 cell. The same Nk/Nq weight applies
        # to it and the body; Hartree remains the original fine-grid bare V0.
        from runtime.aot_memory import compiled_new_bytes
        program = jax.jit(lambda w: _w_R(comb(w), kernel),
            out_shardings=NamedSharding(mesh, WR_SPEC))
        compiled = program.lower(flat).compile()
        stats = compiled.memory_analysis()
        if stats is None:
            raise ValueError("GATE static_agw_capacity: compiled q-comb memory price required")
        new_bytes = compiled_new_bytes(compiled)
        resident = max(0, int(stats.output_size_in_bytes)-int(stats.alias_size_in_bytes))
        capacity.at_phase()
        comb_receipt = capacity.reserve("static_agw.coarse_q_comb",
            resident_bytes_per_rank=resident,
            workspace_bytes_per_rank=max(0, new_bytes-resident))
        wr = compiled(flat)
    px, py, _, vh, rho_eq, _ = bare.params
    model = SelfEnergyModel(apply=bare.apply, params=(px, py, wr, vh, rho_eq, 0.),
        linear=True, name="static TD-aGW: bare Hartree plus screened W0 exchange; NO GW collision")
    return dict(model=model, receipt=dict(
        arm="A_STATIC_SEX_HARTREE_NO_COLLISION", head=head_log,
        W_head_scope="HeadSample.wcoul0 is full screened W; injected once, scalar head=0",
        Hartree="original G0-excluded bare V0 unchanged", spatial_projection=False,
        anchor="accepted SC H once; delta_Hartree[Vbar]-delta_SEX[W0]",
        kgrid=list(grid), qgrid=list(qgrid), q_comb_weight=float(prepared["nk"]/nq),
        q_quadrature="ordinary coarse full-BZ q sum; NOT W interpolation",
        q_comb_capacity=comb_receipt,
        collision="off; not frozen-W MGKBA", material_admission="NOT_RUN"))


def build_static_control(source, reference):
    """Synthesize current accepted-state W0 through its existing native owner."""
    from gw.ppm_sigma import fresh_shared_pole_capacity
    from gw.static_screening import build_static_screened_w
    mesh, meta = source["mesh"], source["meta"]
    if tuple(meta.qgrid) != tuple(meta.kgrid):
        from common.fft_helpers import make_bosonic_q_comb
        make_bosonic_q_comb(mesh, tuple(meta.kgrid), tuple(meta.qgrid))
    receipt, frame = reference["receipts"], source["frame"]
    if any(receipt.get(key) != frame.get(key) for key in
           ("frame", "input_h_sha256", "wfn_fingerprint")):
        raise ValueError("GATE static_agw_reference: W source and accepted TD reference identities differ")
    wm = fresh_shared_pole_capacity(meta, mesh)
    wm.shared_pole_capacity.at_phase()
    interaction = source["V_q"] if tuple(meta.qgrid) == tuple(meta.kgrid) else source["V_screen"]
    W0, head = build_static_screened_w(None, interaction, config=source["config"],
        meta=wm, mesh_xy=mesh, shared_pole=source["W_by_role"]["shared_pole"],
        head_resolver=source["W_by_role"]["iteration_head"])
    return static_control_from_body(reference, W0, head,
        packed_G0=meta.mu_basis.pack_host(source["G0_canonical"]),
        cell_volume=float(meta.cell_volume), capacity=wm.shared_pole_capacity)
