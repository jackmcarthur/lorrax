"""Compile the executed ladder stages; verify width admission and padded tails."""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace


def backend_namespace():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    ns = dict(jax=jax, jnp=jnp, np=np, NamedSharding=NamedSharding, P=P,
              _BLOCK_GMRES_CACHE={}, _PROBE_RHS_CACHE={}, _ARNOLDI_CACHE={},
              _PSI_KEYS=("psi_c_X", "psi_c_Y", "psi_v_X", "psi_v_Y"),
              LADDER_DEFLATION_ARNOLDI=6)
    selected = {
        "bse_feast.py": {"_apply_shifted_matvec", "_gmres_solve_core",
                         "apply_spectral_deflation", "_get_deflation_arnoldi",
                         "harvest_spectral_deflation"},
        "bse_w_exact.py": {"_resolvent_response_actions", "_get_block_gmres_solver",
            "_bind_tol", "_get_probe_rhs_program", "_moment_response_actions",
            "_get_block_moment_actions"},
        "w_ladder.py": {"_identity_probe_blocks", "_probe_scaling_bytes", "_plan_probe_chunk"},
    }
    for name, names in selected.items():
        path = Path(__file__).parents[1] / "src/bse" / name
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), ns)
    return ns


def fixture(mesh, ns, *, band_factor=2):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    nc, nv, nk, nlog = band_factor*px, band_factor*py, 1, 17
    n = ((nlog + px*py-1)//(px*py))*(px*py)
    pair = nc*nv*nk
    sh = SimpleNamespace(X=NamedSharding(mesh, P(None, "x", "y", None)),
        X_full=NamedSharding(mesh, P(None, None, "x", "y", None)),
        S=NamedSharding(mesh, P(None, "y", None)),
        S_k0=NamedSharding(mesh, P(None, "y")), V=NamedSharding(mesh, P("x", "y")),
        psi_x=NamedSharding(mesh, P(None, None, None, "x")),
        psi_y=NamedSharding(mesh, P(None, None, None, "y")))
    put = lambda value, layout: device_put_process_local(np.asarray(value), layout)
    rng = np.random.default_rng(47)
    vertex = rng.normal(size=(pair, n))*0.015
    vertex[:, nlog:] = 0
    v = np.diag(np.r_[np.linspace(0.4, 0.8, nlog), np.zeros(n-nlog)])
    ring = vertex @ v @ vertex.T
    a = np.diag(np.linspace(1.4, 2.8, pair)) + ring
    h = np.block([[a, ring], [-ring, -a]]).astype(np.complex128)
    psi_c, psi_v = np.zeros((nk, nc, 1, n), np.complex128), np.zeros((nk, nv, 1, n), np.complex128)
    data = dict(n_cond_pad=nc, n_val_pad=nv, nkx=1, nky=1, nkz=1,
        psi_c_X=put(psi_c, sh.psi_x), psi_v_X=put(psi_v, sh.psi_x),
        psi_c_Y=put(psi_c, sh.psi_y), psi_v_Y=put(psi_v, sh.psi_y), V_q0=put(v, sh.V))

    @jax.jit
    def gen(g, _pc, _pv, bare):
        return (g[:, :, 0] @ bare.T @ jnp.asarray(vertex).T).reshape(g.shape[0], nc, nv, nk)

    @jax.jit
    def snapshot(value, _pc, _pv, bare):
        result = bare @ jnp.asarray(vertex).T @ value.reshape(value.shape[0], pair).T
        return jax.lax.with_sharding_constraint(result, sh.V)

    @jax.jit
    def matvec(value, matrix):
        b = value.shape[1]
        columns = jnp.moveaxis(value, 1, 0).reshape(b, 2*pair).T
        answer = (matrix @ columns).T.reshape(b, 2, nc, nv, nk)
        return jax.lax.with_sharding_constraint(jnp.moveaxis(answer, 0, 1), sh.X_full)

    context = dict(data=data, sh=sh, mesh=mesh, gen=gen, snapshot=snapshot,
        matvec=matvec, operands=(put(h, NamedSharding(mesh, P())),),
        diag=put(np.diag(h).reshape(2, 1, nc, nv, nk), sh.X_full),
        snapshot_v=None, deflation_rank=0, max_iter=48, tol=1e-11,
        with_slope=True, with_moments=True, moment_actions=None, z_outer=True, nz=2)
    context["moment_actions"] = ns["_get_block_moment_actions"](matvec, sh, jnp.complex128)
    return context, nlog, n, h, vertex, v


def evaluate(context, ns, nlog, n, chunk):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from common.collectives import device_put_process_local, gather_to_host
    data, sh = context["data"], context["sh"]
    seed = ns["_get_probe_rhs_program"](context["gen"], sh, 1)
    solve = ns["_get_block_gmres_solver"](context["matvec"], sh, 48, 1e-11,
                                         jnp.complex128, with_slope=context["with_slope"])
    result = np.zeros((n, n), np.complex128)
    slope = result.copy() if context["with_slope"] else None
    for start, real, probes in ns["_identity_probe_blocks"](nlog, n, chunk):
        rhs = seed(device_put_process_local(probes, sh.S_k0),
                   data["psi_c_X"], data["psi_v_X"], data["V_q0"])
        solved = solve(rhs, context["diag"], jnp.asarray(0.25j), context["operands"])
        value, residual, _iterations = solved[:3]
        assert float(np.max(gather_to_host(residual))) < 2e-10
        actions = [(result, value)]
        if slope is not None:
            actions.append((slope, solved[3]["dWc_ds"]))
        for target, action in actions:
            tile = context["snapshot"](action, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
            target[:, start:start+real] = np.asarray(gather_to_host(tile))[:, :real]
    return result, slope


def run_capacity_gate(mesh, *, with_slope=True, band_factor=2):
    import jax
    import numpy as np
    from lxkit import memory
    from unittest.mock import patch
    jax.config.update("jax_enable_x64", True)
    ns = backend_namespace()
    context, nlog, n, h, vertex, v = fixture(mesh, ns, band_factor=band_factor)
    context["with_slope"] = context["with_moments"] = bool(with_slope)
    plan = ns["_plan_probe_chunk"]
    prices = []
    with patch.object(memory, "book", lambda price, **kw: prices.append(price.bytes())):
        memory.set_budget(1 << 30)
        full = plan(context, None, n_pad=n, carry_bytes=10000, stack_bytes=12000)
        assert full == n
        full_price = prices[-1]
        floor = plan(context, int(mesh.shape["y"]), n_pad=n, carry_bytes=10000, stack_bytes=12000)
        floor_price = prices[-1]
        room = int((full_price+floor_price)//2)
        used = int(memory.run()["budget"]-memory.room(bucketed=False))
        memory.set_budget(used+room)
        room_reads, read_room = [], memory.room
        def metered_room(**kwargs):
            value = read_room(**kwargs)
            room_reads.append(value)
            return value
        with patch.object(memory, "room", metered_room):
            chosen = plan(context, None, n_pad=n, carry_bytes=10000, stack_bytes=12000)
        actual_room = room_reads[-1]
        assert int(mesh.shape["y"]) <= chosen < n
        assert chosen % int(mesh.shape["y"]) == 0
        assert prices[-1] <= actual_room
        reduced_price = prices[-1]
        assert plan(context, n, n_pad=n, carry_bytes=10000, stack_bytes=12000) == n
        assert prices[-1] > actual_room  # explicit requests are honored, not silently clipped
    value, derivative = evaluate(context, ns, nlog, n, chosen)
    full_value, full_derivative = evaluate(context, ns, nlog, n, full)
    seed = np.vstack([vertex @ v, -vertex @ v])
    readout = np.hstack([v @ vertex.T, v @ vertex.T])
    r = np.linalg.inv(0.25j*np.eye(h.shape[0])-h)
    np.testing.assert_allclose(value, readout @ r @ seed, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(value, full_value, atol=2e-12, rtol=2e-10)
    if derivative is not None:
        np.testing.assert_allclose(derivative, -readout @ r @ r @ seed/(0.5j), atol=2e-12, rtol=2e-10)
        np.testing.assert_allclose(derivative, full_derivative, atol=2e-12, rtol=2e-10)
    assert np.all(value[nlog:] == 0) and np.all(value[:, nlog:] == 0)
    return dict(mesh=list(mesh.devices.shape), logical=nlog, padded=n, chunk=chosen,
                full_price=full_price, reduced_price=reduced_price, room=actual_room,
                max_value_error=float(np.max(np.abs(value-readout @ r @ seed))))


def test_compiled_probe_budget_and_tail_parity():
    import jax
    import numpy as np
    from jax.sharding import Mesh
    for side in (1, 2, 3):
        if len(jax.devices("cpu")) >= side*side:
            result = run_capacity_gate(Mesh(np.asarray(jax.devices("cpu")[:side*side]).reshape(side, side), ("x", "y")))
            assert result["reduced_price"] < result["full_price"]


def test_large_default_proposal_is_bounded_without_constructing_large_arrays():
    from runtime.tiles import tile_units
    ns = backend_namespace()
    data = dict(n_cond_pad=20000, n_val_pad=100, nkx=10, nky=10, nkz=5,
        V_q0=SimpleNamespace(shape=(10000, 10000)),
        psi_c_X=SimpleNamespace(shape=(500, 20000, 2, 10000)))
    for side in (1, 2, 3, 8):
        mesh = SimpleNamespace(shape={"x": side, "y": side}, size=side*side)
        per_probe = ns["_probe_scaling_bytes"](data, mesh)
        units = tile_units(per_probe*side, (10000+side-1)//side)
        assert units*side < 10000


def test_value_only_probe_budget_and_math():
    import jax
    import numpy as np
    from jax.sharding import Mesh
    side = 2 if len(jax.devices("cpu")) >= 4 else 1
    mesh = Mesh(np.asarray(jax.devices("cpu")[:side*side]).reshape(side, side), ("x", "y"))
    assert run_capacity_gate(mesh, with_slope=False, band_factor=4)["max_value_error"] < 2e-12


def test_shared_arnoldi_harvest_ritz_and_deflated_solve():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from common.collectives import gather_to_host
    jax.config.update("jax_enable_x64", True)
    ns = backend_namespace()
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    context, _nlog, n, h, _vertex, _v = fixture(mesh, ns)
    probes = np.zeros((1, n))
    probes[0, 0] = 1
    data, sh = context["data"], context["sh"]
    from common.collectives import device_put_process_local
    rhs = ns["_get_probe_rhs_program"](context["gen"], sh, 1)(
        device_put_process_local(probes, sh.S_k0), data["psi_c_X"], data["psi_v_X"], data["V_q0"])
    args = (context["matvec"], context["diag"], 0.25j, context["operands"], rhs)
    deflation, ritz = ns["harvest_spectral_deflation"](*args, n_arnoldi=6, rank=2)
    program = ns["_get_deflation_arnoldi"](context["matvec"], rhs, n_arnoldi=6)
    assert program is ns["_get_deflation_arnoldi"](context["matvec"], rhs, n_arnoldi=6)
    vectors, hessenberg = program(rhs, context["diag"], jnp.asarray(0.25j), context["operands"])
    hm = np.asarray(gather_to_host(hessenberg))[:6, :6]
    np.testing.assert_allclose(np.sort_complex(ritz), np.sort_complex(np.linalg.eigvals(hm)), atol=1e-14)
    solve = ns["_get_block_gmres_solver"](context["matvec"], sh, 48, 1e-11, jnp.complex128)
    result, residual, _ = solve(rhs, context["diag"], jnp.asarray(0.25j), context["operands"], deflation)
    dense = np.linalg.solve(0.25j*np.eye(len(h))-h, np.asarray(gather_to_host(rhs)).reshape(len(h), 1))
    expected = dense.reshape(2, 1, *context["diag"].shape[-3:]).sum(axis=0)
    np.testing.assert_allclose(gather_to_host(result), expected, atol=2e-12, rtol=2e-10)
    assert np.max(gather_to_host(residual)) < 2e-10
    from lxkit import memory
    memory.set_budget(1 << 30)
    context["deflation_rank"] = 2
    assert ns["_plan_probe_chunk"](context, 1, n_pad=n, carry_bytes=10000, stack_bytes=12000) == 1


if __name__ == "__main__":
    import json
    import runtime
    stack = runtime.initialize_communicator_stack()
    from common.gpu_utils import resolve_device_budget_gb
    resolve_device_budget_gb()
    import jax
    summary = run_capacity_gate(stack.mesh)
    print(f"r{jax.process_index()} LADDER_PROBE_CAPACITY PASS " + json.dumps(summary), flush=True)
    runtime.finalize_process(0)
