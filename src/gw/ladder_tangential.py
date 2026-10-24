"""Ladder response samples for the common tangential constructor.

The density resolvent owns Wc, its derivative and its moments. Direction
selection, bank storage, pole construction and Sigma remain their existing
owners. In particular, RPA Dyson derivatives and moments never enter here.
"""
from functools import lru_cache
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


@lru_cache(maxsize=None)
def _relative_difference(mesh):
    return jax.jit(lambda a, b: jnp.max(jnp.abs(a-b)) /
                   jnp.maximum(jnp.max(jnp.abs(a)), jnp.finfo(jnp.float64).tiny),
                   out_shardings=NamedSharding(mesh, P()))


@lru_cache(maxsize=None)
def _relative_norm(mesh):
    def ratio(a, b):
        numerator, scale = jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))
        return jnp.where(scale > 0, numerator/jnp.maximum(scale, jnp.finfo(jnp.float64).tiny),
                         jnp.where(numerator == 0, 0., jnp.inf))
    return jax.jit(ratio, out_shardings=NamedSharding(mesh, P()))


def _solver_accuracy(recipe):
    """True-residual budget below the bank and unchanged Gram certificates.

    One decade of headroom is an input-solve allocation, not a bound on
    Hermite-Gram error amplification. The constructor still certifies its
    actual matrix without PSD repair. No production GN-PPM control changes.
    """
    from .screening_bse import _GMRES_MAX_ITER
    from .tangential_recipe import shared_real_pole_gates_v1_r3b
    bank = float(recipe["bank_rule_tolerance"])
    gram = abs(float(shared_real_pole_gates_v1_r3b["normalized_gram_validity"]["threshold"]))
    if not np.isfinite((bank, gram)).all() or min(bank, gram) <= 0:
        raise ValueError("GATE ladder_tangential_accuracy: positive finite bank and Gram budgets required")
    return dict(gmres_tol=0.1*min(bank, gram), gmres_max_iter=int(_GMRES_MAX_ITER),
                policy="bank-relative-true-residual-v1")


def _gate_solve(residuals, iterations, *, accuracy, where):
    worst = float(np.max(residuals))
    hottest = int(np.max(iterations))
    tol, cap = accuracy["gmres_tol"], accuracy["gmres_max_iter"]
    if not np.isfinite(worst) or worst > tol or hottest >= cap:
        raise ValueError(f"GATE ladder_tangential_solve: {where}: residual {worst:.3e}, "
                         f"requested {tol:.3e}, iterations {hottest}/{cap}; "
                         "no inaccurate or truncated response may be fitted")


def make_ladder_tangential_source(wfns, V_q, *, quad, e_ref, sym,
                                  centroid_indices, config, meta, mesh_xy,
                                  tensors_filename, head_resolver,
                                  head_channel=None, print_fn=print):
    """Current one-shot source, with a static-kernel identity and lazy head.

    All body arrays are packed complex128 faces [q, mu_X, mu_Y]. The backend
    returns canonical centroid order; pack through the same basis the RPA
    bank uses before selecting directions or writing. One frequency's wedge
    is consumed at a time. Only small head tensors and receipts are retained.
    """
    from .screening_bse import (_assert_wedge_matches_run,
                                prepare_ladder_restart)
    state = {}

    def prepare():
        if "identity" not in state:
            state["solver"] = _solver_accuracy(meta.tangential_recipe)
            from bse.w_ladder import (LADDER_RESPONSE_CONVENTION,
                                      ladder_static_head_identity)
            from .response_bank import operator_digest
            static = prepare_ladder_restart(
                wfns, V_q, quad=quad, e_ref=e_ref, sym=sym,
                centroid_indices=centroid_indices, config=config, meta=meta,
                mesh_xy=mesh_xy, tensors_filename=tensors_filename,
                head_resolver=head_resolver, head_channel=head_channel,
                print_fn=print_fn)
            state["identity"] = dict(screening_diagrams="w_bse",
                ladder_response_convention=LADDER_RESPONSE_CONVENTION,
                ladder_static_w_sha256=operator_digest(static, mesh_xy, cache=False),
                ladder_static_head=ladder_static_head_identity(
                    tensors_filename, config.input_file, cell_volume=meta.cell_volume),
                ladder_head_policy=str(getattr(config.head.correction, "value", config.head.correction)),
                ladder_gmres_tol=state["solver"]["gmres_tol"],
                ladder_gmres_max_iter=state["solver"]["gmres_max_iter"],
                ladder_solve_accuracy_policy=state["solver"]["policy"])
            del static
        return dict(state["identity"])

    def produce(bank_io):
        from bse.w_ladder import compute_wc_qwedge
        from file_io.tangential_store import (_conversion_bytes, tangential_bank_writer,
                                               validate_tangential_bank)
        from .gw_config import linalg_resolution
        from .response_bank import (_bank_context, _receipt, _finish_receipt,
                                    _reserve, bank_points)
        from .tangential_directions import charge_line_selection
        from .tangential_execution import line_selection_execution, line_selection_room
        prepare()
        header, qids, census = _bank_context(wfns, meta, sym, bank_io, mesh_xy)
        if bool(header.get("ordered")) or not bool(sym.trs_allowed):
            raise ValueError("GATE ladder_tangential_representation: the ladder requires measured time reversal")
        points = bank_points(meta.tangential_recipe)
        sample_receipt = _receipt("samples", census, bank_io)
        moment_receipt = _receipt("moments", census, bank_io)
        for receipt in (sample_receipt, moment_receipt):
            receipt.update(screening_diagrams="w_bse", response="density ladder resolvent",
                           static_kernel=dict(state["identity"]))
        started = time.monotonic()
        ledger, basis = meta.tangential_capacity, meta.mu_basis
        ambient = ledger.live_stages
        p0, p1 = map(int, header["line_panels"]["sample_span"])
        n, nq = int(basis.n_packed), len(qids)
        face_bytes = 16*n*n//int(mesh_xy.size)
        selection = None
        if p1 > p0:
            backend = linalg_resolution({"linalg": config.backend.linalg}).eigh_backend
            fields = 7 if p0 == 0 else 3
            line_carry = fields*nq*(16*int(basis.n_canonical)**2//int(mesh_xy.size)
                                   + (face_bytes if not basis.is_identity else 0))
            execution, resident, workspace = line_selection_execution(
                [n], mesh=mesh_xy, ledger=ledger, nq=nq,
                carry=line_carry, eigh_backend=backend)
            room = line_selection_room(ledger, resident, carry=line_carry) if execution == "face" else None
            selection = charge_line_selection(meta, mesh_xy=mesh_xy, ordered=False,
                execution=execution, nq=nq, eigh_backend=backend, room=room)
            sample_receipt["line_selection"] = dict(execution=execution,
                samples=[p0, p1], resident_bytes_per_rank=resident,
                workspace_bytes_per_rank=workspace)
        moments_written = False

        def accept(iz, z, value, residuals, iterations, fields):
            nonlocal moments_written
            _gate_solve(residuals, iterations, accuracy=state["solver"], where=f"Wc/slope z={z}")
            # Count the backend's canonical outputs and their packed copies
            # together; the basis conversion can allocate at a new extent.
            names = ("Wc", "dWc_ds", "Wc_minus_z") + (() if moments_written else
                     ("M0", "M1", "M2", "M3"))
            operands = {name: array for name, array in dict(Wc=value, **fields).items() if name in names}
            canonical_bytes = sum(int(a.size*a.dtype.itemsize)//int(mesh_xy.size)
                                  for a in operands.values())
            packed_bytes = len(operands)*nq*face_bytes if not basis.is_identity else 0
            packing_temporary = max(_conversion_bytes(basis, a.shape, a.sharding.spec,
                unpack=False, operator=True)[2] for a in operands.values())
            ledger.live_stages = ambient
            stage, _ = _reserve(meta, "ladder_sample", canonical_bytes+packed_bytes,
                                packing_temporary, book=False)
            ledger.live_stages = ambient+(stage,)
            packed = {}
            for name, array in operands.items():
                packed[name] = basis.pack_operator(array)
                # One conversion's temporary, rather than seven independent
                # asynchronous pack programs live beside the canonical bank.
                packed[name].block_until_ready()
            del operands
            even = float(_relative_difference(mesh_xy)(packed["Wc"], packed["Wc_minus_z"]))
            if not np.isfinite(even) or even > 2e-6:
                raise ValueError(f"GATE ladder_tangential_evenness: Wc(z) versus Wc(-z) {even:.3e}; "
                                 "the even-s pole model cannot represent this response")
            sample_receipt.setdefault("evenness", []).append(dict(sample=int(iz), relative=even))
            del packed["Wc_minus_z"]
            with tangential_bank_writer(bank_io["path"], meta=meta,
                    expected_identity=bank_io["identity"], mesh_xy=mesh_xy) as (_, current, write):
                if not moments_written:
                    # M0/M2 are forbidden odd powers on this measured-TRS
                    # even-s route. The backend returns them for this gate.
                    ratios = [float(_relative_norm(mesh_xy)(packed[a], packed[b]))
                              for a, b in (("M0", "M1"), ("M2", "M3"))]
                    if not np.isfinite(ratios).all() or max(ratios) > 2e-6:
                        raise ValueError(f"GATE ladder_tangential_odd_moments: ratios {ratios}; "
                                         "do not replace the ladder moments with RPA moments")
                    write(q_span=(0,nq), M1=packed["M1"], M3=packed["M3"])
                    moment_receipt["odd_moment_ratios"] = ratios
                    moments_written = True
                value, slope = packed.pop("Wc"), packed.pop("dWc_ds")
                del packed
                if p0 <= iz < p1:
                    from lxkit import memory
                    live = ledger.live_stages
                    stage, _ = _reserve(meta, "ladder_line_selection", resident,
                                        workspace, book=False)
                    ledger.live_stages = live+(stage,)
                    memory.book(memory.priced(resident+selection.eigh_bytes(),
                        "ladder line selection: admitted cut blocks and compiled eigh", increment=True),
                        stage="bank.line_select")
                    pair = [value, slope]
                    del value, slope
                    lines = selection.select(iz, pair)
                    from runtime.aot_memory import compiled_new_bytes
                    from .response_bank import _compiled
                    memory.book(memory.priced(sum(compiled_new_bytes(_compiled(selection.to_face, args))
                        for _, _, args in selection.panel_fields(lines)),
                        "ladder line panels: compiled at their fields", increment=True),
                        stage="bank.line_panels")
                    write(q_span=(0,nq), line=selection.panels(iz, lines))
                    del lines
                    ledger.live_stages = live
                else:
                    write(q_span=(0,nq), sample_span=(iz,iz+1),
                          Wc=value[:,None], dWc_ds=slope[:,None])
                sample_receipt["batches"].append(dict(sample=int(iz),
                    max_residual=float(np.max(residuals)), max_iterations=int(np.max(iterations))))
            ledger.live_stages = ambient

        from runtime.padding import padded_axis
        requested = int(config.screening.ladder_probe_chunk)
        chunk = (padded_axis(requested, mesh_xy, name="ladder probe chunk",
                             spec=P("y"), axis=0).carrier if requested else None)
        wedge = compute_wc_qwedge(str(tensors_filename), points, mesh_xy,
            input_file=config.input_file, include_w=True,
            gmres_tol=state["solver"]["gmres_tol"], gmres_max_iter=state["solver"]["gmres_max_iter"],
            probe_chunk=chunk, with_slope=True, with_moments=True, on_z=accept)
        _assert_wedge_matches_run(wedge, sym)
        current = validate_tangential_bank(bank_io["path"], expected_identity=bank_io["identity"],
                                           mesh_xy=mesh_xy, require_complete=True)
        for receipt in (sample_receipt, moment_receipt):
            receipt["completion"] = True
            _finish_receipt(receipt, meta, current, started)
        return sample_receipt, moment_receipt

    def head():
        from .ladder_head_fit import build_ladder_head_fit
        prepare()
        if "head" not in state:
            state["head"] = build_ladder_head_fit(tensors_filename,config,meta,mesh_xy,
                head_resolver,identity=state["identity"])
        return state["head"]

    return dict(identity=prepare, bank=produce, head=head)
